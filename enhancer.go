package main

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

// The prompt enhancer (prompt_enhancer in config.yaml) is an LLM that the
// manager runs as the instance prompt-enhancer. Enhance, next to an image
// model's prompt in the chat, sends the short prompt to it and puts the
// detailed description it writes back in the composer, to read and edit
// before sending.

// pupPromptsJSON is BFL's system_prompts.json for FLUX.3 klein's prompt
// upsampler (black-forest-labs/flux-3-image-klein-pup at a57899549f8f,
// Apache-2.0), unchanged. The upsampler was tuned on these prompts; other
// capable LLMs can follow them too.
//
//go:embed pup_system_prompts.json
var pupPromptsJSON []byte

var enhancerPrompts = func() map[string]string {
	var m map[string]string
	if err := json.Unmarshal(pupPromptsJSON, &m); err != nil || m[enhanceT2I] == "" || m[enhanceI2I] == "" {
		panic(fmt.Sprintf("pup_system_prompts.json lacks the t2i and i2i prompts: %v", err))
	}
	return m
}()

const (
	enhancerName         = "prompt-enhancer"
	rolePromptEnhancer   = "prompt_enhancer"
	defaultEnhancerPort  = 9120
	defaultEnhancerModel = "flux3-klein-pup"
	defaultEnhancerCtx   = 16384
	defaultEnhancerWait  = 2 * time.Minute
	enhancerMaxTokens    = 3584 // BFL's max_new_tokens
	enhancerParallel     = 2
	enhancerMaxInFlight  = 8
	maxEnhanceInput      = 8 << 10
	maxEnhancedBytes     = 32 << 10

	enhanceT2I = "t2i" // a caption for a new image
	enhanceI2I = "i2i" // an instruction for an edit of the attached images
)

// enhancerModels are BFL's prompt upsampler for FLUX.3 klein (a Qwen3.5-9B
// fine-tune) as AtomicChat's GGUF files; each comes with its F16 vision
// projector, which edits need. Any other chat model works too: an HF
// repo:quant or a local .gguf.
var enhancerModels = map[string]string{
	"flux3-klein-pup":    "AtomicChat/FLUX-3-Image-Klein-pup-GGUF:AD-IQ4_XS", // 5.5 GB + 0.9 GB projector
	"flux3-klein-pup-q5": "AtomicChat/FLUX-3-Image-Klein-pup-GGUF:AD-Q5_K_M", // 6.6 GB
	"flux3-klein-pup-q8": "AtomicChat/FLUX-3-Image-Klein-pup-GGUF:Q8_0",      // 9.5 GB
}

// PromptEnhancerConf runs the prompt enhancer as its own instance,
// prompt-enhancer. It picks a model and port for one more exec'd
// llama-server, so like image_recognition it is set only in config.yaml.
type PromptEnhancerConf struct {
	Enabled       bool     `yaml:"enabled" json:"enabled"`
	Model         string   `yaml:"model,omitempty" json:"model,omitempty"`
	Port          int      `yaml:"port,omitempty" json:"port,omitempty"`
	GPUIDs        []int    `yaml:"gpu_ids,omitempty" json:"gpu_ids,omitempty"`
	ContextLength int      `yaml:"context_length,omitempty" json:"context_length,omitempty"`
	ExtraArgs     []string `yaml:"extra_args,omitempty" json:"extra_args,omitempty"`
	Timeout       duration `yaml:"timeout,omitempty" json:"timeout,omitempty"`
}

func (c PromptEnhancerConf) model() string {
	m := strings.TrimSpace(c.Model)
	if m == "" {
		m = defaultEnhancerModel
	}
	if ref, ok := enhancerModels[strings.ToLower(m)]; ok {
		return ref
	}
	return m
}

func (c PromptEnhancerConf) port() int {
	if c.Port > 0 {
		return c.Port
	}
	return defaultEnhancerPort
}

func (c PromptEnhancerConf) timeout() time.Duration {
	if c.Timeout.Duration > 0 {
		return c.Timeout.Duration
	}
	return defaultEnhancerWait
}

// instanceConf is the prompt-enhancer instance. Its extra_args are its own,
// so flags meant for the chat models (a draft model, -np 1) don't reach it.
func (c PromptEnhancerConf) instanceConf() InstanceConf {
	gpus := append([]int(nil), c.GPUIDs...)
	if len(gpus) == 0 {
		gpus = []int{0}
	}
	ctx := c.ContextLength
	if ctx == 0 {
		ctx = defaultEnhancerCtx
	}
	jinja := true
	extra := append([]string{}, c.ExtraArgs...)
	return InstanceConf{Name: enhancerName, Model: c.model(), Port: c.port(), GPUIDs: gpus, AutoStart: true,
		ContextLength: &ctx, Jinja: &jinja, ExtraArgs: &extra, role: rolePromptEnhancer}
}

func (c PromptEnhancerConf) check() error {
	var errs []error
	if c.Port != 0 && !validPort(c.Port) {
		errs = append(errs, fmt.Errorf("prompt_enhancer.port must be 1-65535"))
	}
	if err := validateModelRef(c.model()); err != nil {
		errs = append(errs, fmt.Errorf("prompt_enhancer.model: %w (use flux3-klein-pup, flux3-klein-pup-q5, flux3-klein-pup-q8 or owner/repo:QUANT)", err))
	}
	for _, id := range c.GPUIDs {
		if id < 0 {
			errs = append(errs, fmt.Errorf("prompt_enhancer.gpu_ids: gpu id %d must be >= 0", id))
		}
	}
	if c.ContextLength < 0 {
		errs = append(errs, fmt.Errorf("prompt_enhancer.context_length must be > 0"))
	}
	if c.Timeout.Duration < 0 {
		errs = append(errs, fmt.Errorf("prompt_enhancer.timeout must be >= 0"))
	}
	return errors.Join(errs...)
}

// clash reports an instance that would collide with the prompt enhancer.
func (c PromptEnhancerConf) clash(ic InstanceConf) error {
	if !c.Enabled {
		return nil
	}
	if ic.Name == enhancerName {
		return fmt.Errorf("the name %q belongs to the prompt enhancer (prompt_enhancer in config.yaml); pick another name", enhancerName)
	}
	if ic.Port == c.port() {
		return fmt.Errorf("port %d is used by the prompt enhancer (prompt_enhancer in config.yaml); pick another port", ic.Port)
	}
	return nil
}

// checkAgainst reports the other model sections' instances that the prompt
// enhancer would collide with.
func (c PromptEnhancerConf) checkAgainst(cfg *Config) []error {
	if !c.Enabled {
		return nil
	}
	var errs []error
	p := c.port()
	if p == cfg.ManagerPort {
		errs = append(errs, fmt.Errorf("prompt_enhancer.port %d is the manager's own port", p))
	}
	if r := cfg.ImageRecognition; r.Enabled && r.port() == p {
		errs = append(errs, fmt.Errorf("prompt_enhancer.port %d is the image reader's port", p))
	}
	var others []InstanceConf
	if cfg.ImageGeneration.Enabled {
		for _, e := range cfg.ImageGeneration.entries() {
			others = append(others, e.instanceConf())
		}
	}
	if cfg.TextToSpeech.Enabled {
		for _, e := range cfg.TextToSpeech.entries() {
			others = append(others, e.instanceConf())
		}
	}
	for _, o := range others {
		if o.Port == p {
			errs = append(errs, fmt.Errorf("prompt_enhancer.port %d is %s's port", p, o.Name))
		}
		if o.Name == enhancerName {
			errs = append(errs, fmt.Errorf("%s: the name belongs to the prompt enhancer", o.Name))
		}
	}
	return errs
}

type promptEnhancer struct {
	mgr      *Manager
	model    string
	timeout  time.Duration
	http     *http.Client
	slots    chan struct{}
	inFlight atomic.Int32
}

func newPromptEnhancer(c PromptEnhancerConf, mgr *Manager) *promptEnhancer {
	if !c.Enabled {
		return nil
	}
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.Proxy = nil // our own child process
	return &promptEnhancer{
		mgr:     mgr,
		model:   c.model(),
		timeout: c.timeout(),
		http: &http.Client{Transport: tr, CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		}},
		slots: make(chan struct{}, enhancerParallel),
	}
}

// enhancerStatus is what /api/status says about the prompt enhancer.
type enhancerStatus struct {
	Instance string        `json:"instance"`
	Model    string        `json:"model"`
	State    InstanceState `json:"state"`
	Error    string        `json:"error,omitempty"`
}

func (p *promptEnhancer) Status() enhancerStatus {
	st := enhancerStatus{Instance: enhancerName, Model: p.model, State: StateStopped}
	if inst := p.mgr.Get(enhancerName); inst != nil {
		s := inst.Status()
		st.State, st.Error = s.State, s.LastError
	}
	return st
}

type enhanceImage struct {
	Mime string
	Data []byte
}

// enhanceReq is one prompt to enhance. With images it is an edit of them;
// Canvas ("16:9") is the shape of a new image.
type enhanceReq struct {
	Prompt string
	Images []enhanceImage
	Canvas string
}

// enhanced is what /api/enhance returns.
type enhanced struct {
	Text    string `json:"text"`
	Mode    string `json:"mode"`
	Model   string `json:"model"`
	Ms      int64  `json:"ms"`
	Cut     bool   `json:"truncated,omitempty"`
	in, out int64
}

func (p *promptEnhancer) Enhance(ctx context.Context, req enhanceReq) (enhanced, error) {
	st := statsOf(p.mgr)
	start := time.Now()
	done := st.begin()
	res, err := p.enhance(ctx, req)
	done()
	if ctx.Err() == nil {
		rec := statRec{At: start, Source: srcChat, Route: "enhance", Model: enhancerName, DurMs: time.Since(start).Milliseconds(), TTFTMs: -1, In: res.in, Out: res.out}
		var xe *extractError
		switch {
		case errors.As(err, &xe) && xe.Status == http.StatusServiceUnavailable:
			rec.Outcome = outcomeRejected
		case err != nil:
			rec.Outcome, rec.ErrMsg = outcomeError, err.Error()
		}
		st.record(rec)
	}
	return res, err
}

// enhanceBody is the chat request BFL's model card gives: the mode's system
// prompt, thinking off, greedy decoding, presence penalty 1.5 for a new image
// and 0 for an edit. An edit's message is the references, then three lines.
func enhanceBody(req enhanceReq) (mode string, body []byte) {
	mode = enhanceT2I
	var user any
	presence := 1.5
	if len(req.Images) > 0 {
		mode, presence = enhanceI2I, 0
		parts := make([]contentPart, 0, len(req.Images)+1)
		for _, img := range req.Images {
			parts = append(parts, contentPart{Type: "image_url", ImageURL: map[string]string{"url": "data:" + img.Mime + ";base64," + base64.StdEncoding.EncodeToString(img.Data)}})
		}
		parts = append(parts, contentPart{Type: "text", Text: "Reference images follow in order (the first image, the second image, ...).\nUser request: " + req.Prompt + "\nOutput the caption."})
		user = parts
	} else if req.Canvas != "" {
		user = "TARGET CANVAS: " + req.Canvas + "\n" + req.Prompt
	} else {
		user = req.Prompt
	}
	body, _ = json.Marshal(map[string]any{
		"messages": []map[string]any{
			{"role": "system", "content": enhancerPrompts[mode]},
			{"role": "user", "content": user},
		},
		"temperature":          0,
		"top_p":                0.95,
		"top_k":                20,
		"presence_penalty":     presence,
		"max_tokens":           enhancerMaxTokens,
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
	})
	return mode, body
}

func (p *promptEnhancer) enhance(ctx context.Context, req enhanceReq) (enhanced, error) {
	inst, err := helperReady(p.mgr, enhancerName)
	if err != nil {
		return enhanced{}, err
	}
	if p.inFlight.Add(1) > enhancerMaxInFlight {
		p.inFlight.Add(-1)
		return enhanced{}, &extractError{http.StatusServiceUnavailable, "busy", enhancerName + " is busy with other prompts. Try again in a moment."}
	}
	defer p.inFlight.Add(-1)
	select {
	case p.slots <- struct{}{}:
	case <-ctx.Done():
		return enhanced{}, ctx.Err()
	}
	defer func() { <-p.slots }()

	start := time.Now()
	mode, body := enhanceBody(req)
	rctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()
	hreq, err := http.NewRequestWithContext(rctx, http.MethodPost, inst.baseURL()+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return enhanced{}, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	timedOut := func() error {
		return &extractError{http.StatusGatewayTimeout, "timeout", fmt.Sprintf("%s didn't finish within %s.", enhancerName, p.timeout)}
	}
	resp, err := p.http.Do(hreq)
	if err != nil {
		switch {
		case ctx.Err() != nil:
			return enhanced{}, ctx.Err()
		case errors.Is(rctx.Err(), context.DeadlineExceeded):
			return enhanced{}, timedOut()
		}
		var ue *url.Error
		if errors.As(err, &ue) {
			err = ue.Err
		}
		return enhanced{}, &extractError{http.StatusBadGateway, "unreachable", fmt.Sprintf("%s didn't answer: %v.", enhancerName, err)}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg := fmt.Sprintf("%s couldn't enhance the prompt: %s", enhancerName, upstreamError(resp))
		if mode == enhanceI2I {
			msg += " (it reads attached images only with its vision projector, the mmproj file)"
		}
		return enhanced{}, &extractError{http.StatusBadGateway, "unreadable", msg}
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int64 `json:"prompt_tokens"`
			CompletionTokens int64 `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&out); err != nil {
		switch {
		case ctx.Err() != nil:
			return enhanced{}, ctx.Err()
		case errors.Is(rctx.Err(), context.DeadlineExceeded):
			return enhanced{}, timedOut()
		}
		return enhanced{}, &extractError{http.StatusBadGateway, "unreadable", enhancerName + " sent an answer llama-manager can't read."}
	}
	if len(out.Choices) == 0 {
		return enhanced{}, &extractError{http.StatusBadGateway, "unreadable", enhancerName + " sent no prompt."}
	}
	c := out.Choices[0]
	text, cut := cleanEnhanced(c.Message.Content)
	if text == "" {
		return enhanced{}, &extractError{http.StatusBadGateway, "unreadable", enhancerName + " sent an empty prompt."}
	}
	return enhanced{Text: text, Mode: mode, Model: p.model, Ms: time.Since(start).Milliseconds(), Cut: cut || c.FinishReason == "length",
		in: out.Usage.PromptTokens, out: out.Usage.CompletionTokens}, nil
}

// cleanEnhanced drops any reasoning a thinking model wrote first and quotes
// around the whole answer, and keeps at most maxEnhancedBytes.
func cleanEnhanced(s string) (string, bool) {
	if i := strings.LastIndex(s, "</think>"); i >= 0 {
		s = s[i+len("</think>"):]
	} else if strings.HasPrefix(strings.TrimSpace(s), "<think>") {
		return "", false
	}
	s = strings.TrimSpace(s)
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' && strings.Count(s, `"`) == 2 {
		s = strings.TrimSpace(s[1 : len(s)-1])
	}
	if len(s) <= maxEnhancedBytes {
		return s, false
	}
	cut := maxEnhancedBytes
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return strings.TrimRight(s[:cut], " \t\n"), true
}

// canvasRatio is a size such as 1344x768 as the shape the enhancer composes
// for: 7:4.
func canvasRatio(size string) string {
	w, h := parseImageSize(size, 16)
	a, b := w, h
	for b != 0 {
		a, b = b, a%b
	}
	return fmt.Sprintf("%d:%d", w/a, h/a)
}

// handleEnhance expands a short image prompt: POST {"prompt", "model" (the
// image model it is for), "size", "images": [{"mime", "data" (base64)}]}.
// Images make it an edit, and only go along when the model edits.
func (ws *WebServer) handleEnhance(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if ws.enhancer == nil {
		writeExtractErr(w, http.StatusServiceUnavailable, "off", "The prompt enhancer isn't turned on in config.yaml.")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxChatSendBody)
	var in struct {
		Prompt string `json:"prompt"`
		Model  string `json:"model"`
		Size   string `json:"size"`
		Images []struct {
			Mime string `json:"mime"`
			Data []byte `json:"data"`
		} `json:"images"`
	}
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeExtractErr(w, http.StatusRequestEntityTooLarge, "too_large", "The attached images are too large to send.")
			return
		}
		writeExtractErr(w, http.StatusBadRequest, "unreadable", "invalid json: "+err.Error())
		return
	}
	prompt := strings.TrimSpace(in.Prompt)
	switch {
	case prompt == "":
		writeExtractErr(w, http.StatusBadRequest, "empty", "Write a short prompt first; Enhance adds the details.")
		return
	case len(prompt) > maxEnhanceInput || !utf8.ValidString(prompt):
		writeExtractErr(w, http.StatusRequestEntityTooLarge, "too_large", fmt.Sprintf("Enhance takes a prompt of up to %d KB of UTF-8 text.", maxEnhanceInput>>10))
		return
	case len(in.Images) > maxAttachments:
		writeExtractErr(w, http.StatusBadRequest, "too_many", fmt.Sprintf("Enhance takes at most %d images.", maxAttachments))
		return
	}
	req := enhanceReq{Prompt: prompt}
	edits := true
	if ws.imageGen != nil {
		if e, ok := ws.imageGen.model(in.Model); ok && e.Preset.Edit == editNone {
			edits = false
		}
	}
	for _, img := range in.Images {
		if !edits {
			break
		}
		if !imageMimes[img.Mime] || len(img.Data) == 0 || len(img.Data) > maxImageBytes {
			writeExtractErr(w, http.StatusBadRequest, "unreadable", fmt.Sprintf("Images must be PNG, JPEG, GIF or WebP, up to %d MB.", maxImageBytes>>20))
			return
		}
		req.Images = append(req.Images, enhanceImage{Mime: img.Mime, Data: img.Data})
	}
	if len(req.Images) == 0 {
		req.Canvas = canvasRatio(in.Size)
	}
	res, err := ws.enhancer.Enhance(r.Context(), req)
	var ee *extractError
	switch {
	case errors.As(err, &ee):
		writeExtractErr(w, ee.Status, ee.Reason, ee.Msg)
	case err != nil:
		writeExtractErr(w, http.StatusBadGateway, "unreachable", err.Error())
	default:
		writeJSON(w, http.StatusOK, res)
	}
}

// helperReady returns a helper instance (the image reader, the prompt
// enhancer) when it can take a request.
func helperReady(mgr *Manager, name string) (*Instance, error) {
	inst := mgr.Get(name)
	if inst == nil {
		return nil, &extractError{http.StatusServiceUnavailable, "off", name + " isn't running on this server."}
	}
	switch s := inst.Status(); s.State {
	case StateRunning, StateUnhealthy:
		return inst, nil
	case StateStarting, StateRestarting:
		return nil, &extractError{http.StatusServiceUnavailable, "starting", name + " is still starting (downloading or loading its model). Try again in a moment."}
	case StateCrashed:
		msg := name + " has crashed"
		if s.LastError != "" {
			msg += " (" + s.LastError + ")"
		}
		return nil, &extractError{http.StatusServiceUnavailable, "stopped", msg + ". Check its log on the Instances page."}
	default:
		return nil, &extractError{http.StatusServiceUnavailable, "stopped", name + " is stopped. Start it on the Instances page."}
	}
}
