package main

import (
	"bytes"
	"context"
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

// The image reader is a small vision model (image_recognition in config.yaml)
// that the manager runs as the instance image-reader. Images attached in the
// chat are sent to it, and its description goes to chat models that can't
// see images. Models with a vision projector still get the image itself.

const extractReader = "reader" // attachments.extract for a described image

// readerPrompt was tested on English invoices, charts, handwriting, photos and
// screenshots with Qwen3.5 2B and 4B; it asks for the text first because
// that is what the chat model most often needs and can't guess.
const readerPrompt = "Describe this image for an assistant that cannot see it. First write out all visible text exactly as it " +
	"appears, with tables as Markdown tables. Then describe what the image shows: what kind of image it is, the " +
	"main objects, layout and colours, and for charts the values shown. Be accurate; say when something is unclear " +
	"instead of guessing."

const readerCutNote = "\n\n[The description stops here: the image reader reached its length limit.]"

type imageReader struct {
	mgr      *Manager
	model    string // what config.yaml asks for, resolved to an HF reference
	timeout  time.Duration
	http     *http.Client
	slots    chan struct{}
	inFlight atomic.Int32
}

func newImageReader(c ImageRecConf, mgr *Manager) *imageReader {
	if !c.Enabled {
		return nil
	}
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.Proxy = nil // the reader is our own child process
	return &imageReader{
		mgr:     mgr,
		model:   c.model(),
		timeout: c.timeout(),
		http: &http.Client{Transport: tr, CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		}},
		slots: make(chan struct{}, readerParallel),
	}
}

// readerStatus is what /api/status says about the image reader.
type readerStatus struct {
	Instance string        `json:"instance"`
	Model    string        `json:"model"`
	State    InstanceState `json:"state"`
	Error    string        `json:"error,omitempty"`
}

func (r *imageReader) Status() readerStatus {
	st := readerStatus{Instance: readerName, Model: r.model, State: StateStopped}
	if inst := r.mgr.Get(readerName); inst != nil {
		s := inst.Status()
		st.State, st.Error = s.State, s.LastError
	}
	return st
}

// described is one image's description, as /api/describe returns it.
type described struct {
	Text    string `json:"text"`
	Model   string `json:"model"`
	Ms      int64  `json:"ms"`
	Cut     bool   `json:"truncated,omitempty"`
	in, out int64  // tokens, for the dashboard
}

// Describe asks the reader what the image shows. ctx ends when the browser
// or the reply gives up; the configured timeout caps one image.
func (r *imageReader) Describe(ctx context.Context, data []byte, mime string) (described, error) {
	st := statsOf(r.mgr)
	start := time.Now()
	done := st.begin()
	d, err := r.describe(ctx, data, mime)
	done()
	if ctx.Err() == nil {
		rec := statRec{At: start, Source: srcChat, Route: "describe", Model: readerName, DurMs: time.Since(start).Milliseconds(), TTFTMs: -1, In: d.in, Out: d.out}
		var xe *extractError
		switch {
		case errors.As(err, &xe) && xe.Status == http.StatusServiceUnavailable:
			rec.Outcome = outcomeRejected
		case err != nil:
			rec.Outcome, rec.ErrMsg = outcomeError, err.Error()
		}
		st.record(rec)
	}
	return d, err
}

func (r *imageReader) describe(ctx context.Context, data []byte, mime string) (described, error) {
	inst, err := helperReady(r.mgr, readerName)
	if err != nil {
		return described{}, err
	}
	if r.inFlight.Add(1) > readerMaxInFlight {
		r.inFlight.Add(-1)
		return described{}, &extractError{http.StatusServiceUnavailable, "busy", readerName + " is busy with other images. Try again in a moment."}
	}
	defer r.inFlight.Add(-1)
	select {
	case r.slots <- struct{}{}:
	case <-ctx.Done():
		return described{}, ctx.Err()
	}
	defer func() { <-r.slots }()

	start := time.Now()
	body, _ := json.Marshal(map[string]any{
		"messages": []map[string]any{{"role": "user", "content": []contentPart{
			{Type: "image_url", ImageURL: map[string]string{"url": "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data)}},
			{Type: "text", Text: readerPrompt},
		}}},
		"temperature":          0,
		"max_tokens":           readerMaxTokens,
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
	})
	rctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodPost, inst.baseURL()+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return described{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := r.http.Do(req)
	if err != nil {
		switch {
		case ctx.Err() != nil:
			return described{}, ctx.Err()
		case errors.Is(rctx.Err(), context.DeadlineExceeded):
			return described{}, &extractError{http.StatusGatewayTimeout, "timeout", fmt.Sprintf("%s didn't finish within %s.", readerName, r.timeout)}
		}
		var ue *url.Error
		if errors.As(err, &ue) {
			err = ue.Err
		}
		return described{}, &extractError{http.StatusBadGateway, "unreachable", fmt.Sprintf("%s didn't answer: %v.", readerName, err)}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg := fmt.Sprintf("%s couldn't read the image: %s", readerName, upstreamError(resp))
		if mime == "image/webp" {
			msg += " (llama-server can't decode WebP; attach PNG or JPEG)"
		}
		return described{}, &extractError{http.StatusBadGateway, "unreadable", msg}
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
		if ctx.Err() != nil {
			return described{}, ctx.Err()
		}
		if errors.Is(rctx.Err(), context.DeadlineExceeded) {
			return described{}, &extractError{http.StatusGatewayTimeout, "timeout", fmt.Sprintf("%s didn't finish within %s.", readerName, r.timeout)}
		}
		return described{}, &extractError{http.StatusBadGateway, "unreadable", readerName + " sent an answer llama-manager can't read."}
	}
	if len(out.Choices) == 0 {
		return described{}, &extractError{http.StatusBadGateway, "unreadable", readerName + " sent no description."}
	}
	c := out.Choices[0]
	text, cut := cleanDescription(c.Message.Content)
	if text == "" {
		return described{}, &extractError{http.StatusBadGateway, "unreadable", readerName + " sent an empty description."}
	}
	if c.FinishReason == "length" {
		text, cut = text+readerCutNote, true
	}
	return described{Text: text, Model: r.model, Ms: time.Since(start).Milliseconds(), Cut: cut, in: out.Usage.PromptTokens, out: out.Usage.CompletionTokens}, nil
}

// cleanDescription drops any reasoning a thinking model wrote first, tidies
// the text like a document's and keeps at most maxReaderTextBytes.
func cleanDescription(s string) (string, bool) {
	if i := strings.LastIndex(s, "</think>"); i >= 0 {
		s = s[i+len("</think>"):]
	} else if strings.HasPrefix(strings.TrimSpace(s), "<think>") {
		return "", false // all reasoning, no answer
	}
	text, _ := cleanDocText(s, false)
	if len(text) <= maxReaderTextBytes {
		return text, false
	}
	cut := maxReaderTextBytes
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	return strings.TrimRight(text[:cut], " \t\n"), true
}

// handleDescribe sends one image to the reader: POST the image as the body,
// its type as Content-Type.
func (ws *WebServer) handleDescribe(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if ws.reader == nil {
		writeExtractErr(w, http.StatusServiceUnavailable, "off", "Image recognition isn't turned on in config.yaml.")
		return
	}
	mime := mediaType(r.Header.Get("Content-Type"))
	if !imageMimes[mime] {
		writeExtractErr(w, http.StatusUnsupportedMediaType, "unreadable", "Images must be PNG, JPEG, GIF or WebP.")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxImageBytes)
	data, err := io.ReadAll(r.Body)
	var tooBig *http.MaxBytesError
	switch {
	case errors.As(err, &tooBig):
		writeExtractErr(w, http.StatusRequestEntityTooLarge, "too_large", fmt.Sprintf("The image is over %d MB.", maxImageBytes>>20))
		return
	case err != nil:
		writeExtractErr(w, http.StatusBadRequest, "unreadable", "reading the upload: "+err.Error())
		return
	case len(data) == 0:
		writeExtractErr(w, http.StatusBadRequest, "unreadable", "The image is empty.")
		return
	}
	res, err := ws.reader.Describe(r.Context(), data, mime)
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

func hasImages(hist []historyTurn) bool {
	for _, h := range hist {
		for _, a := range h.Attachments {
			if a.Kind == "image" {
				return true
			}
		}
	}
	return false
}

func hasDescribedImages(hist []historyTurn) bool {
	for _, h := range hist {
		for _, a := range h.Attachments {
			if a.Kind == "image" && a.Text != "" {
				return true
			}
		}
	}
	return false
}

// describeHistory has the reader describe every image in hist that has no
// description yet (sent before image_recognition was on, or while a model
// that sees images was picked), saves each and fills it in. model is the chat
// instance, for the error message.
func (cs *ChatService) describeHistory(ctx context.Context, g *generation, model string, hist []historyTurn) error {
	var todo []*Attachment
	for i := range hist {
		for j := range hist[i].Attachments {
			if a := &hist[i].Attachments[j]; a.Kind == "image" && a.Text == "" {
				todo = append(todo, a)
			}
		}
	}
	if len(todo) == 0 {
		return nil
	}
	what := todo[0].Name
	if len(todo) > 1 {
		what = fmt.Sprintf("%d images", len(todo))
	}
	g.setStatus(fmt.Sprintf("%s is reading %s", readerName, what))
	defer g.setStatus("")

	errs := make([]error, len(todo))
	done := make(chan struct{})
	var left atomic.Int32
	left.Store(int32(len(todo)))
	for i, a := range todo {
		go func() {
			defer func() {
				if left.Add(-1) == 0 {
					close(done)
				}
			}()
			res, err := cs.reader.Describe(ctx, a.Data, a.Mime)
			if err != nil {
				errs[i] = err
				return
			}
			a.Text, a.Extract = res.Text, extractReader
			if err := cs.store.SetImageText(a.ID, res.Text); err != nil {
				errs[i] = fmt.Errorf("saving the description: %w", err)
			}
		}()
	}
	<-done
	for i, err := range errs {
		if err == nil {
			continue
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		var ee *extractError
		if errors.As(err, &ee) {
			err = errors.New(strings.TrimSuffix(ee.Msg, "."))
		}
		return fmt.Errorf("%s can't see images, and the image reader couldn't describe %s: %v", model, todo[i].Name, err)
	}
	return nil
}
