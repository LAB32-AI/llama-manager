package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// The system prompts are BFL's file byte for byte; an edit would no longer be
// what the model was tuned on.
func TestPupSystemPromptsAreBFLs(t *testing.T) {
	sum := sha256.Sum256(pupPromptsJSON)
	if got := hex.EncodeToString(sum[:]); got != "b5da3ed8bc04ab59dbdb747930f7dbff847e868903fed8028f85ee1a4ac82fb4" {
		t.Errorf("pup_system_prompts.json changed: sha256 %s", got)
	}
	for _, k := range []string{enhanceT2I, enhanceI2I} {
		if !strings.Contains(enhancerPrompts[k], "prompt upsampler") {
			t.Errorf("%s prompt: %.80q", k, enhancerPrompts[k])
		}
	}
}

func TestPromptEnhancerConfigRules(t *testing.T) {
	inst := "\ninstances:\n  - {name: chat, model: /m.gguf, port: %d, gpu_ids: [0]}\n"
	withPort := func(p int) string { return strings.Replace(inst, "%d", strconv.Itoa(p), 1) }
	for in, ok := range map[string]bool{
		"prompt_enhancer: {enabled: true}": true,
		"prompt_enhancer: {enabled: true, model: flux3-klein-pup-q5, gpu_ids: [4], port: 9200}":                                 true,
		"prompt_enhancer: {enabled: true, model: unsloth/Qwen3.5-9B-GGUF:Q4_K_M}":                                               true,
		"prompt_enhancer: {enabled: true, model: /models/pup.gguf, extra_args: [--mmproj, x]}":                                  true,
		"prompt_enhancer: {enabled: true, model: 'not a repo'}":                                                                 false,
		"prompt_enhancer: {enabled: true, model: 'a/b:Q4;rm'}":                                                                  false,
		"prompt_enhancer: {enabled: true, port: 70000}":                                                                         false,
		"prompt_enhancer: {enabled: true, port: 8080}":                                                                          false, // the manager's port
		"prompt_enhancer: {enabled: true, gpu_ids: [-1]}":                                                                       false,
		"prompt_enhancer: {enabled: true, timeout: -1s}":                                                                        false,
		"prompt_enhancer: {enabled: true, context_length: -4}":                                                                  false,
		"prompt_enhancer: {enabled: true, gpu: 1}":                                                                              false, // typo'd key
		"prompt_enhancer: {enabled: true}" + withPort(9120):                                                                     false, // port taken
		"prompt_enhancer: {enabled: false}" + withPort(9120):                                                                    true,
		"prompt_enhancer: {enabled: true, port: 9300}" + withPort(9120):                                                         true,
		"prompt_enhancer: {enabled: true, port: 9095}\nimage_recognition: {enabled: true}":                                      false,
		"prompt_enhancer: {enabled: true, port: 9096}\nimage_generation: {enabled: true}":                                       false,
		"prompt_enhancer: {enabled: true, port: 9110}\ntext_to_speech: {enabled: true}":                                         false,
		"prompt_enhancer: {enabled: true}\nimage_recognition: {enabled: true}":                                                  true,
		"prompt_enhancer: {enabled: true}\nimage_generation: {enabled: true, name: prompt-enhancer}":                            false,
		"prompt_enhancer: {enabled: true}\ninstances:\n  - {name: prompt-enhancer, model: /m.gguf, port: 9100, gpu_ids: [0]}\n": false,
	} {
		cfg, err := decodeConfig([]byte("server_bin: /bin/true\nmanager_port: 8080\n" + in))
		if err == nil {
			err = cfg.validate()
		}
		if (err == nil) != ok {
			t.Errorf("%s: err = %v, want ok %v", in, err, ok)
		}
	}

	def := PromptEnhancerConf{Enabled: true}.instanceConf()
	ctx, jinja := defaultEnhancerCtx, true
	want := InstanceConf{Name: "prompt-enhancer", Model: "AtomicChat/FLUX-3-Image-Klein-pup-GGUF:AD-IQ4_XS", Port: 9120, GPUIDs: []int{0}, AutoStart: true,
		ContextLength: &ctx, Jinja: &jinja, ExtraArgs: &[]string{}, role: rolePromptEnhancer}
	if !reflect.DeepEqual(def, want) {
		t.Errorf("default enhancer instance:\n got %+v\nwant %+v", def, want)
	}
	for in, ref := range map[string]string{
		"Flux3-Klein-Pup-Q8":             "AtomicChat/FLUX-3-Image-Klein-pup-GGUF:Q8_0",
		"flux3-klein-pup-q5":             "AtomicChat/FLUX-3-Image-Klein-pup-GGUF:AD-Q5_K_M",
		"unsloth/Qwen3.5-9B-GGUF:Q4_K_M": "unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
		"  ":                             "AtomicChat/FLUX-3-Image-Klein-pup-GGUF:AD-IQ4_XS",
	} {
		if m := (PromptEnhancerConf{Model: in}).model(); m != ref {
			t.Errorf("model %q: %q, want %q", in, m, ref)
		}
	}

	// A save keeps the section as written and doesn't add one to configs without it.
	dir := t.TempDir()
	for name, in := range map[string]string{
		"with":    "server_bin: /bin/true\nprompt_enhancer:\n  enabled: true\n  model: flux3-klein-pup-q5\n  gpu_ids: [4]\n",
		"without": "server_bin: /bin/true\n",
	} {
		path := filepath.Join(dir, name+".yaml")
		os.WriteFile(path, []byte(in), 0o600)
		cfg, err := loadConfig(path)
		if err != nil {
			t.Fatal(err)
		}
		cfg.mu.Lock()
		cfg.NGL = 12
		err = cfg.saveLocked()
		cfg.mu.Unlock()
		if err != nil {
			t.Fatal(err)
		}
		out, _ := os.ReadFile(path)
		again, err := loadConfig(path)
		if err != nil {
			t.Fatal(err)
		}
		if name == "with" && !reflect.DeepEqual(again.PromptEnhancer, PromptEnhancerConf{Enabled: true, Model: "flux3-klein-pup-q5", GPUIDs: []int{4}}) {
			t.Errorf("prompt_enhancer lost on save:\n%s", out)
		}
		if name == "without" && strings.Contains(string(out), "prompt_enhancer") {
			t.Errorf("a config without the enhancer gained the section:\n%s", out)
		}
	}
}

func TestPromptEnhancerRunsAsItsOwnInstance(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ExtraArgs = []string{"--spec-type", "draft-mtp"}
	cfg.ImageRecognition = ImageRecConf{Enabled: true, GPUIDs: []int{4}}
	cfg.PromptEnhancer = PromptEnhancerConf{Enabled: true, GPUIDs: []int{4}, ExtraArgs: []string{"-fa", "on"}}
	cfg.Instances = []InstanceConf{{Name: "chat", Model: "/m.gguf", Port: 9090, GPUIDs: []int{0}}}
	mgr := NewManager(cfg)
	names := []string{}
	for _, in := range mgr.Instances() {
		names = append(names, in.conf.Name)
	}
	if !slices.Equal(names, []string{"chat", "image-reader", "prompt-enhancer"}) {
		t.Fatalf("managed instances: %v", names)
	}
	p := mgr.Get(enhancerName)
	if p.Status().Role != rolePromptEnhancer {
		t.Errorf("role: %q", p.Status().Role)
	}
	spec, err := buildLaunch(p.conf, cfg)
	if err != nil {
		t.Fatal(err)
	}
	line := strings.Join(spec.args, " ")
	for _, want := range []string{"-hf AtomicChat/FLUX-3-Image-Klein-pup-GGUF:AD-IQ4_XS", "--port 9120", "-c 16384", "--jinja", "-fa on"} {
		if !strings.Contains(line, want) {
			t.Errorf("enhancer args lack %q: %s", want, line)
		}
	}
	if strings.Contains(line, "draft-mtp") || strings.Contains(line, "--no-mmproj") {
		t.Errorf("enhancer args: %s", line)
	}
	if !slices.Contains(spec.env, "GGML_VK_VISIBLE_DEVICES=4") {
		t.Errorf("enhancer GPU: %v", spec.env)
	}

	if err := cfg.AddInstance(InstanceConf{Name: "x", Model: "/m.gguf", Port: 9120, GPUIDs: []int{0}}); err == nil {
		t.Error("an instance may not take the enhancer's port")
	}
	if err := cfg.AddInstance(InstanceConf{Name: "prompt-enhancer", Model: "/m.gguf", Port: 9121, GPUIDs: []int{0}}); err == nil {
		t.Error("an instance may not take the enhancer's name")
	}
	if err := cfg.UpdateInstance("chat", InstanceConf{Name: "chat", Model: "/m.gguf", Port: 9120, GPUIDs: []int{0}}); err == nil {
		t.Error("an edit may not move an instance onto the enhancer's port")
	}
}

func TestPromptEnhancerCannotBeChangedOverTheAPI(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.PromptEnhancer = PromptEnhancerConf{Enabled: true, GPUIDs: []int{4}}
	same := "server_bin: /bin/true\nallowed_hosts: [example.com]\nprompt_enhancer:\n  enabled: true\n  gpu_ids: [4]\n"
	os.WriteFile(cfg.path, []byte(same), 0o640)
	ws := NewWebServer(NewManager(cfg), cfg, NewDownloadManager())
	serve := func(method, path, body string) *httptest.ResponseRecorder {
		rr := httptest.NewRecorder()
		ws.ServeHTTP(rr, httptest.NewRequest(method, path, strings.NewReader(body)))
		return rr
	}
	for _, m := range []string{http.MethodPut, http.MethodDelete} {
		if rr := serve(m, "/api/config/instances/prompt-enhancer", `{"name":"prompt-enhancer","model":"evil/x","port":9120,"gpu_ids":[0]}`); rr.Code != http.StatusConflict {
			t.Errorf("%s prompt-enhancer: %d %s", m, rr.Code, rr.Body)
		}
	}
	serve(http.MethodPut, "/api/settings", `{"ngl":5,"prompt_enhancer":{"enabled":true,"model":"evil/x"}}`)
	if cfg.PromptEnhancer.Model != "" {
		t.Errorf("settings API changed the enhancer: %+v", cfg.PromptEnhancer)
	}

	if rr := postImport(t, ws, same+"instances:\n  - {name: a, model: /m.gguf, port: 9090, gpu_ids: [0]}\n"); rr.Code != http.StatusOK {
		t.Fatalf("import with the same section: %d %s", rr.Code, rr.Body)
	}
	if ws.mgr.Get(enhancerName) == nil || ws.mgr.Get("a") == nil {
		t.Error("an import must keep the enhancer and add the new instance")
	}
	for name, in := range map[string]string{
		"changed": same + "  model: evil/x\n",
		"dropped": "server_bin: /bin/true\nallowed_hosts: [example.com]\n",
	} {
		if rr := postImport(t, ws, in); rr.Code != http.StatusForbidden {
			t.Errorf("%s section: %d %s", name, rr.Code, rr.Body)
		}
	}
	if rr := postImport(t, ws, same+"instances:\n  - {name: a, model: /m.gguf, port: 9120, gpu_ids: [0]}\n"); rr.Code != http.StatusBadRequest {
		t.Errorf("an import may not put an instance on the enhancer's port: %d %s", rr.Code, rr.Body)
	}

	rr := serve(http.MethodGet, "/api/status", "")
	var st ServerStatus
	json.Unmarshal(rr.Body.Bytes(), &st)
	if st.PromptEnhancer == nil || st.PromptEnhancer.Instance != enhancerName || st.PromptEnhancer.Model != "AtomicChat/FLUX-3-Image-Klein-pup-GGUF:AD-IQ4_XS" {
		t.Errorf("status: %+v", st.PromptEnhancer)
	}
}

// enhancerRig is a chat rig with a running prompt-enhancer backed by answer,
// and two image models: z-image (makes images only) and klein (edits).
type enhancerRig struct {
	*chatAPIRig
	mu    sync.Mutex
	calls []map[string]any
}

func newEnhancerRig(t *testing.T, answer func(w http.ResponseWriter, r *http.Request)) *enhancerRig {
	t.Helper()
	er := &enhancerRig{chatAPIRig: newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) { titleReply(w, "T") })}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		er.mu.Lock()
		er.calls = append(er.calls, body)
		er.mu.Unlock()
		answer(w, r)
	}))
	t.Cleanup(up.Close)
	conf := PromptEnhancerConf{Enabled: true, Port: portOf(t, up.URL)}
	ws := er.ws
	ws.cfg.mu.Lock()
	ws.cfg.PromptEnhancer = conf
	ws.cfg.mu.Unlock()
	if err := ws.mgr.AddInstance(conf.instanceConf()); err != nil {
		t.Fatal(err)
	}
	setState(ws.mgr.Get(enhancerName), StateRunning)
	ws.enhancer = newPromptEnhancer(conf, ws.mgr)
	ws.imageGen = newImageGen(ImageGenConf{Enabled: true, Models: []ImageModelConf{
		{Name: "z-image", Model: "z-image-turbo", Port: 1}, {Name: "klein", Model: "flux3-klein", Port: 2},
	}}, ws.mgr)
	return er
}

func (er *enhancerRig) lastCall(t *testing.T) map[string]any {
	t.Helper()
	er.mu.Lock()
	defer er.mu.Unlock()
	if len(er.calls) == 0 {
		t.Fatal("the enhancer got no request")
	}
	return er.calls[len(er.calls)-1]
}

func enhancerReply(w http.ResponseWriter, content, finish string) {
	writeJSON(w, 200, map[string]any{
		"choices": []map[string]any{{"message": map[string]string{"content": content}, "finish_reason": finish}},
		"usage":   map[string]int{"prompt_tokens": 900, "completion_tokens": 250},
	})
}

func TestEnhanceSendsBFLsRequest(t *testing.T) {
	er := newEnhancerRig(t, func(w http.ResponseWriter, r *http.Request) {
		enhancerReply(w, "\n\"Full-color outdoor photograph of a calm alpine lake at dawn, mist over the water.\"\n", "stop")
	})
	resp, b := er.do(t, "POST", "/api/enhance", map[string]any{"prompt": "  generate lake ", "model": "klein", "size": "1344x768"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("enhance: %d %s", resp.StatusCode, b)
	}
	var res struct {
		Text, Mode, Model string
		Truncated         bool
	}
	json.Unmarshal(b, &res)
	if res.Text != "Full-color outdoor photograph of a calm alpine lake at dawn, mist over the water." || res.Mode != "t2i" || res.Truncated {
		t.Errorf("result: %+v", res)
	}
	if res.Model != "AtomicChat/FLUX-3-Image-Klein-pup-GGUF:AD-IQ4_XS" {
		t.Errorf("model: %q", res.Model)
	}
	body := er.lastCall(t)
	msgs := body["messages"].([]any)
	sys, user := msgs[0].(map[string]any), msgs[1].(map[string]any)
	if sys["role"] != "system" || sys["content"] != enhancerPrompts["t2i"] {
		t.Errorf("system message: %.100v", sys)
	}
	if user["content"] != "TARGET CANVAS: 7:4\ngenerate lake" {
		t.Errorf("user message: %q", user["content"])
	}
	for k, v := range map[string]any{"temperature": 0.0, "top_p": 0.95, "top_k": 20.0, "presence_penalty": 1.5, "max_tokens": 3584.0} {
		if body[k] != v {
			t.Errorf("%s = %v, want %v", k, body[k], v)
		}
	}
	if kw, _ := body["chat_template_kwargs"].(map[string]any); kw["enable_thinking"] != false {
		t.Errorf("thinking must be off: %v", body["chat_template_kwargs"])
	}

	// With images for a model that edits, it writes an edit instruction
	// around them.
	png := []byte("\x89PNG dog")
	img := map[string]any{"mime": "image/png", "data": base64.StdEncoding.EncodeToString(png)}
	if resp, b := er.do(t, "POST", "/api/enhance", map[string]any{"prompt": "put a red bandana on the dog", "model": "klein", "images": []any{img, img}}); resp.StatusCode != http.StatusOK {
		t.Fatalf("edit: %d %s", resp.StatusCode, b)
	} else if json.Unmarshal(b, &res); res.Mode != "i2i" {
		t.Errorf("edit mode: %+v", res)
	}
	body = er.lastCall(t)
	msgs = body["messages"].([]any)
	if msgs[0].(map[string]any)["content"] != enhancerPrompts["i2i"] || body["presence_penalty"] != 0.0 {
		t.Errorf("edit request: system %.60v, presence %v", msgs[0], body["presence_penalty"])
	}
	parts := msgs[1].(map[string]any)["content"].([]any)
	if len(parts) != 3 {
		t.Fatalf("edit parts: %v", parts)
	}
	if u := parts[0].(map[string]any)["image_url"].(map[string]any)["url"]; u != "data:image/png;base64,"+base64.StdEncoding.EncodeToString(png) {
		t.Errorf("image part: %v", u)
	}
	if txt := parts[2].(map[string]any)["text"]; txt != "Reference images follow in order (the first image, the second image, ...).\nUser request: put a red bandana on the dog\nOutput the caption." {
		t.Errorf("edit text: %q", txt)
	}

	// A model that can't edit gets a new image's caption, images or not.
	if resp, b := er.do(t, "POST", "/api/enhance", map[string]any{"prompt": "a dog", "model": "z-image", "images": []any{img}}); resp.StatusCode != http.StatusOK {
		t.Fatalf("z-image: %d %s", resp.StatusCode, b)
	}
	if u := er.lastCall(t)["messages"].([]any)[1].(map[string]any)["content"]; u != "TARGET CANVAS: 1:1\na dog" {
		t.Errorf("z-image user message: %v", u)
	}

	var got statAgg
	statsOf(er.ws.mgr).allTotals(func(k statKey, a *statAgg) {
		if k.Route == "enhance" && k.Model == enhancerName && k.Source == srcChat {
			got = *a
		}
	})
	if got.Requests != 3 || got.Errors != 0 {
		t.Errorf("dashboard counts: %+v", got)
	}
}

func TestEnhanceErrors(t *testing.T) {
	var mu sync.Mutex
	reply := func(w http.ResponseWriter) { enhancerReply(w, "ok", "stop") }
	er := newEnhancerRig(t, func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		f := reply
		mu.Unlock()
		f(w)
	})
	set := func(f func(w http.ResponseWriter)) {
		mu.Lock()
		reply = f
		mu.Unlock()
	}
	errOf := func(b []byte) (string, string) {
		var e struct{ Error, Reason string }
		json.Unmarshal(b, &e)
		return e.Error, e.Reason
	}
	for name, c := range map[string]struct {
		body   map[string]any
		status int
		reason string
	}{
		"empty":     {map[string]any{"prompt": "  "}, 400, "empty"},
		"long":      {map[string]any{"prompt": strings.Repeat("x", maxEnhanceInput+1)}, 413, "too_large"},
		"bad image": {map[string]any{"prompt": "x", "images": []any{map[string]any{"mime": "image/svg+xml", "data": "PHN2Zz4="}}}, 400, "unreadable"},
		"many":      {map[string]any{"prompt": "x", "images": make([]any, maxAttachments+1)}, 400, "too_many"},
	} {
		if resp, b := er.do(t, "POST", "/api/enhance", c.body); resp.StatusCode != c.status {
			t.Errorf("%s: %d %s", name, resp.StatusCode, b)
		} else if _, r := errOf(b); r != c.reason {
			t.Errorf("%s: reason %q", name, r)
		}
	}
	if resp, _ := er.do(t, "GET", "/api/enhance", nil); resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("GET: %d", resp.StatusCode)
	}

	set(func(w http.ResponseWriter) { enhancerReply(w, "<think>planning</think> A long caption", "length") })
	resp, b := er.do(t, "POST", "/api/enhance", map[string]any{"prompt": "lake"})
	var res enhancedJSON
	json.Unmarshal(b, &res)
	if resp.StatusCode != 200 || res.Text != "A long caption" || !res.Truncated {
		t.Errorf("cut answer: %d %s", resp.StatusCode, b)
	}

	set(func(w http.ResponseWriter) {
		writeJSON(w, 500, map[string]any{"error": map[string]any{"message": "image input is not supported"}})
	})
	img := map[string]any{"mime": "image/png", "data": base64.StdEncoding.EncodeToString([]byte("\x89PNG"))}
	resp, b = er.do(t, "POST", "/api/enhance", map[string]any{"prompt": "edit it", "model": "klein", "images": []any{img}})
	if msg, _ := errOf(b); resp.StatusCode != http.StatusBadGateway || !strings.Contains(msg, "image input is not supported") || !strings.Contains(msg, "mmproj") {
		t.Errorf("upstream error: %d %s", resp.StatusCode, b)
	}

	set(func(w http.ResponseWriter) { enhancerReply(w, "<think>only thinking", "length") })
	if resp, b := er.do(t, "POST", "/api/enhance", map[string]any{"prompt": "lake"}); resp.StatusCode != http.StatusBadGateway {
		t.Errorf("empty answer: %d %s", resp.StatusCode, b)
	}

	for state, reason := range map[InstanceState]string{StateStarting: "starting", StateStopped: "stopped", StateCrashed: "stopped"} {
		setState(er.ws.mgr.Get(enhancerName), state)
		resp, b := er.do(t, "POST", "/api/enhance", map[string]any{"prompt": "lake"})
		if msg, r := errOf(b); resp.StatusCode != http.StatusServiceUnavailable || r != reason || !strings.Contains(msg, enhancerName) {
			t.Errorf("%s: %d %s", state, resp.StatusCode, b)
		}
	}
	setState(er.ws.mgr.Get(enhancerName), StateRunning)

	er.ws.enhancer.timeout = 50 * time.Millisecond
	set(func(w http.ResponseWriter) { time.Sleep(300 * time.Millisecond); enhancerReply(w, "late", "stop") })
	resp, b = er.do(t, "POST", "/api/enhance", map[string]any{"prompt": "lake"})
	if _, r := errOf(b); resp.StatusCode != http.StatusGatewayTimeout || r != "timeout" {
		t.Errorf("timeout: %d %s", resp.StatusCode, b)
	}

	er.ws.enhancer = nil
	resp, b = er.do(t, "POST", "/api/enhance", map[string]any{"prompt": "lake"})
	if _, r := errOf(b); resp.StatusCode != http.StatusServiceUnavailable || r != "off" {
		t.Errorf("off: %d %s", resp.StatusCode, b)
	}
}

type enhancedJSON struct {
	Text      string `json:"text"`
	Truncated bool   `json:"truncated"`
}

func TestCanvasRatioAndCleanEnhanced(t *testing.T) {
	for in, want := range map[string]string{"1024x1024": "1:1", "1344x768": "7:4", "768x1344": "4:7", "1152x896": "9:7", "": "1:1", "bogus": "1:1"} {
		if got := canvasRatio(in); got != want {
			t.Errorf("canvasRatio(%q) = %q, want %q", in, got, want)
		}
	}
	for in, want := range map[string]string{
		"  plain  ":                        "plain",
		`"quoted all over"`:                "quoted all over",
		`A sign reads "OPEN" in red.`:      `A sign reads "OPEN" in red.`,
		`"Begins" and ends with "quotes"`:  `"Begins" and ends with "quotes"`,
		"<think>x</think>\n\nThe caption.": "The caption.",
		"<think>never closed":              "",
	} {
		if got, _ := cleanEnhanced(in); got != want {
			t.Errorf("cleanEnhanced(%q) = %q, want %q", in, got, want)
		}
	}
	long := strings.Repeat("é", maxEnhancedBytes)
	got, cut := cleanEnhanced(long)
	if !cut || len(got) > maxEnhancedBytes || !strings.HasPrefix(long, got) {
		t.Errorf("long answer: cut %v, %d bytes", cut, len(got))
	}
}
