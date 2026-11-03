package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net"
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
	"unicode/utf8"
)

func TestImageRecognitionConfigRules(t *testing.T) {
	inst := "\ninstances:\n  - {name: chat, model: /m.gguf, port: %d, gpu_ids: [0]}\n"
	withPort := func(p int) string { return strings.Replace(inst, "%d", strconv.Itoa(p), 1) }
	for in, ok := range map[string]bool{
		"image_recognition: {enabled: true}":                                                                                   true,
		"image_recognition: {enabled: true, model: qwen3.5-4b, gpu_ids: [4], port: 9200}":                                      true,
		"image_recognition: {enabled: true, model: ggml-org/gemma-4-E4B-it-GGUF:Q4_0}":                                         true,
		"image_recognition: {enabled: true, model: /models/r.gguf, extra_args: [--mmproj, x]}":                                 true,
		"image_recognition: {enabled: true, model: 'not a repo'}":                                                              false,
		"image_recognition: {enabled: true, model: 'a/b:Q4;rm'}":                                                               false,
		"image_recognition: {enabled: true, port: 70000}":                                                                      false,
		"image_recognition: {enabled: true, port: 8080}":                                                                       false, // the manager's port
		"image_recognition: {enabled: true, gpu_ids: [-1]}":                                                                    false,
		"image_recognition: {enabled: true, timeout: -1s}":                                                                     false,
		"image_recognition: {enabled: true, context_length: -4}":                                                               false,
		"image_recognition: {enabled: true, gpu: 1}":                                                                           false, // typo'd key
		"image_recognition: {enabled: true}" + withPort(9095):                                                                  false, // port taken
		"image_recognition: {enabled: false}" + withPort(9095):                                                                 true,
		"image_recognition: {enabled: true, port: 9300}" + withPort(9095):                                                      true,
		"image_recognition: {enabled: true}\ninstances:\n  - {name: image-reader, model: /m.gguf, port: 9100, gpu_ids: [0]}\n": false,
	} {
		cfg, err := decodeConfig([]byte("server_bin: /bin/true\nmanager_port: 8080\n" + in))
		if err == nil {
			err = cfg.validate()
		}
		if (err == nil) != ok {
			t.Errorf("%s: err = %v, want ok %v", in, err, ok)
		}
	}

	def := ImageRecConf{Enabled: true}.instanceConf()
	ctx, jinja := defaultReaderCtx, true
	want := InstanceConf{Name: "image-reader", Model: "unsloth/Qwen3.5-2B-GGUF:Q4_K_M", Port: 9095, GPUIDs: []int{0}, AutoStart: true,
		ContextLength: &ctx, Jinja: &jinja, ExtraArgs: &[]string{"--image-min-tokens", "256"}, role: roleReader}
	if !reflect.DeepEqual(def, want) {
		t.Errorf("default reader instance:\n got %+v\nwant %+v", def, want)
	}
	if m := (ImageRecConf{Model: "Qwen3.5-4B"}).model(); m != "unsloth/Qwen3.5-4B-GGUF:Q4_K_M" {
		t.Errorf("4b preset: %q", m)
	}
	if own := (ImageRecConf{ExtraArgs: []string{"--image-min-tokens", "1024"}}).instanceConf(); !reflect.DeepEqual(*own.ExtraArgs, []string{"--image-min-tokens", "1024"}) {
		t.Errorf("an image-min-tokens in extra_args must replace the default: %v", *own.ExtraArgs)
	}

	// A save keeps the section as written and doesn't add one to configs without it.
	dir := t.TempDir()
	for name, in := range map[string]string{
		"with":    "server_bin: /bin/true\nimage_recognition:\n  enabled: true\n  model: qwen3.5-4b\n  gpu_ids: [4]\n",
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
		if name == "with" && !reflect.DeepEqual(again.ImageRecognition, ImageRecConf{Enabled: true, Model: "qwen3.5-4b", GPUIDs: []int{4}}) {
			t.Errorf("image_recognition lost on save:\n%s", out)
		}
		if name == "without" && strings.Contains(string(out), "image_recognition") {
			t.Errorf("a config without the reader gained the section:\n%s", out)
		}
	}
}

// The reader is one more managed instance with its own flags: the global
// extra_args (a draft model, -np 1) are for the chat models.
func TestReaderRunsAsItsOwnInstance(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ExtraArgs = []string{"--spec-type", "draft-mtp"}
	cfg.ImageRecognition = ImageRecConf{Enabled: true, GPUIDs: []int{4}, ExtraArgs: []string{"-fa", "on"}}
	cfg.Instances = []InstanceConf{{Name: "chat", Model: "/m.gguf", Port: 9090, GPUIDs: []int{0}}}
	mgr := NewManager(cfg)
	names := []string{}
	for _, in := range mgr.Instances() {
		names = append(names, in.conf.Name)
	}
	if !slices.Equal(names, []string{"chat", "image-reader"}) {
		t.Fatalf("managed instances: %v", names)
	}
	r := mgr.Get("image-reader")
	if r.Status().Role != roleReader || mgr.Get("chat").Status().Role != "" {
		t.Errorf("roles: %q %q", r.Status().Role, mgr.Get("chat").Status().Role)
	}
	spec, err := buildLaunch(r.conf, cfg)
	if err != nil {
		t.Fatal(err)
	}
	line := strings.Join(spec.args, " ")
	for _, want := range []string{"-hf unsloth/Qwen3.5-2B-GGUF:Q4_K_M", "--port 9095", "-c 16384", "--jinja", "--image-min-tokens 256 -fa on"} {
		if !strings.Contains(line, want) {
			t.Errorf("reader args lack %q: %s", want, line)
		}
	}
	if strings.Contains(line, "draft-mtp") {
		t.Errorf("global extra_args reached the reader: %s", line)
	}
	if !slices.Contains(spec.env, "GGML_VK_VISIBLE_DEVICES=4") {
		t.Errorf("reader GPU: %v", spec.env)
	}

	if err := cfg.AddInstance(InstanceConf{Name: "x", Model: "/m.gguf", Port: 9095, GPUIDs: []int{0}}); err == nil {
		t.Error("an instance may not take the reader's port")
	}
	if err := cfg.AddInstance(InstanceConf{Name: "image-reader", Model: "/m.gguf", Port: 9096, GPUIDs: []int{0}}); err == nil {
		t.Error("an instance may not take the reader's name")
	}
	if err := cfg.UpdateInstance("chat", InstanceConf{Name: "chat", Model: "/m.gguf", Port: 9095, GPUIDs: []int{0}}); err == nil {
		t.Error("an edit may not move an instance onto the reader's port")
	}
}

func TestReaderCannotBeChangedOverTheAPI(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ImageRecognition = ImageRecConf{Enabled: true, Port: 9095}
	os.WriteFile(cfg.path, []byte("server_bin: /bin/true\nallowed_hosts: [example.com]\nimage_recognition:\n  enabled: true\n  port: 9095\n"), 0o640)
	ws := NewWebServer(NewManager(cfg), cfg, NewDownloadManager())
	serve := func(method, path, body string) *httptest.ResponseRecorder {
		rr := httptest.NewRecorder()
		ws.ServeHTTP(rr, httptest.NewRequest(method, path, strings.NewReader(body)))
		return rr
	}
	for _, m := range []string{http.MethodPut, http.MethodDelete} {
		if rr := serve(m, "/api/config/instances/image-reader", `{"name":"image-reader","model":"evil/x","port":9095,"gpu_ids":[0]}`); rr.Code != http.StatusConflict {
			t.Errorf("%s image-reader: %d %s", m, rr.Code, rr.Body)
		}
	}
	serve(http.MethodPut, "/api/settings", `{"ngl":5,"image_recognition":{"enabled":true,"model":"evil/x"}}`)
	if cfg.ImageRecognition.Model != "" {
		t.Errorf("settings API changed the reader: %+v", cfg.ImageRecognition)
	}

	same := "server_bin: /bin/true\nallowed_hosts: [example.com]\nimage_recognition:\n  enabled: true\n  port: 9095\n"
	if rr := postImport(t, ws, same+"instances:\n  - {name: a, model: /m.gguf, port: 9090, gpu_ids: [0]}\n"); rr.Code != http.StatusOK {
		t.Fatalf("import with the same section: %d %s", rr.Code, rr.Body)
	}
	if ws.mgr.Get("image-reader") == nil || ws.mgr.Get("a") == nil {
		t.Error("an import must keep the reader and add the new instance")
	}
	for name, in := range map[string]string{
		"changed": "server_bin: /bin/true\nallowed_hosts: [example.com]\nimage_recognition:\n  enabled: true\n  port: 9095\n  model: evil/x\n",
		"dropped": "server_bin: /bin/true\nallowed_hosts: [example.com]\n",
	} {
		if rr := postImport(t, ws, in); rr.Code != http.StatusForbidden {
			t.Errorf("%s section: %d %s", name, rr.Code, rr.Body)
		}
	}
	if rr := postImport(t, ws, same+"instances:\n  - {name: a, model: /m.gguf, port: 9095, gpu_ids: [0]}\n"); rr.Code != http.StatusBadRequest {
		t.Errorf("an import may not put an instance on the reader's port: %d %s", rr.Code, rr.Body)
	}
}

// readerRig is a chat rig whose "c" model can't see, plus a running
// image-reader backed by describe and a model "v" that can see.
type readerRig struct {
	*chatAPIRig
	mu    sync.Mutex
	calls []map[string]any
}

func newReaderRig(t *testing.T, describe func(w http.ResponseWriter, r *http.Request)) *readerRig {
	t.Helper()
	chatModel := func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if r.URL.Path == "/props" {
			writeJSON(w, 200, map[string]any{"modalities": map[string]bool{"vision": false}})
			return
		}
		if body["stream"] != true {
			titleReply(w, "T")
			return
		}
		sse(w, delta("content", "ok"), finalChunk, "[DONE]")
	}
	rr := &readerRig{chatAPIRig: newChatAPIRig(t, chatModel)}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		rr.mu.Lock()
		rr.calls = append(rr.calls, body)
		rr.mu.Unlock()
		describe(w, r)
	}))
	t.Cleanup(up.Close)
	conf := ImageRecConf{Enabled: true, Port: portOf(t, up.URL)}
	ws := rr.ws
	ws.cfg.mu.Lock()
	ws.cfg.ImageRecognition = conf
	ws.cfg.mu.Unlock()
	if err := ws.mgr.AddInstance(conf.instanceConf()); err != nil {
		t.Fatal(err)
	}
	setState(ws.mgr.Get(readerName), StateRunning)
	ws.reader = newImageReader(conf, ws.mgr)
	ws.chats.reader = ws.reader

	seeing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if r.URL.Path == "/props" {
			writeJSON(w, 200, map[string]any{"modalities": map[string]bool{"vision": true}})
			return
		}
		rr.chatAPIRig.mu.Lock()
		rr.chatAPIRig.bodies = append(rr.chatAPIRig.bodies, body)
		rr.chatAPIRig.mu.Unlock()
		if body["stream"] != true {
			titleReply(w, "T")
			return
		}
		sse(w, delta("content", "seen"), finalChunk, "[DONE]")
	}))
	t.Cleanup(seeing.Close)
	if err := ws.mgr.AddInstance(InstanceConf{Name: "v", Model: "/v.gguf", Port: portOf(t, seeing.URL), GPUIDs: []int{1}}); err != nil {
		t.Fatal(err)
	}
	setState(ws.mgr.Get("v"), StateRunning)
	return rr
}

func portOf(t *testing.T, u string) int {
	t.Helper()
	_, p, err := net.SplitHostPort(strings.TrimPrefix(u, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	n, _ := strconv.Atoi(p)
	return n
}

func setState(inst *Instance, s InstanceState) {
	inst.mu.Lock()
	inst.state = s
	inst.mu.Unlock()
}

func (rr *readerRig) readerCalls() int {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	return len(rr.calls)
}

// lastPrompt is the user message of the last reply request a chat model got.
func (rr *readerRig) lastPrompt(t *testing.T) any {
	t.Helper()
	rr.chatAPIRig.mu.Lock()
	defer rr.chatAPIRig.mu.Unlock()
	for i := len(rr.bodies) - 1; i >= 0; i-- {
		if rr.bodies[i]["stream"] == true {
			msgs := rr.bodies[i]["messages"].([]any)
			return msgs[len(msgs)-1].(map[string]any)["content"]
		}
	}
	t.Fatal("no reply request was sent")
	return nil
}

func readerReply(w http.ResponseWriter, content, finish string) {
	writeJSON(w, 200, map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": content}, "finish_reason": finish}}})
}

func imgAtt(name string, data []byte) map[string]any {
	return map[string]any{"kind": "image", "name": name, "mime": "image/png", "data": base64.StdEncoding.EncodeToString(data)}
}

func TestChatDescribesImagesForModelsThatCantSee(t *testing.T) {
	release := make(chan struct{})
	rr := newReaderRig(t, func(w http.ResponseWriter, r *http.Request) {
		<-release
		readerReply(w, "<think>hmm</think>Text: Total due 1,250.00 EUR\n\nA scanned invoice.", "stop")
	})
	c := rr.newChat(t, `{}`)
	png := []byte("\x89PNG invoice")
	if resp, b := rr.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "What is the total?", "attachments": []map[string]any{imgAtt("scan.png", png)}}); resp.StatusCode != http.StatusCreated {
		t.Fatalf("send: %d %s", resp.StatusCode, b)
	}

	// While the reader works, followers see what the reply waits on.
	resp, err := http.Get(rr.srv.URL + "/api/chats/" + c.ID + "/events")
	if err != nil {
		t.Fatal(err)
	}
	var snap genSnapshot
	for sc := newSSEScanner(resp); sc.Scan(); {
		if sc.event == "snapshot" {
			json.Unmarshal([]byte(sc.data), &snap)
			break
		}
	}
	resp.Body.Close()
	if snap.Status != "image-reader is reading scan.png" {
		t.Errorf("snapshot status: %q", snap.Status)
	}
	close(release)
	if v := rr.waitReply(t, c.ID); v.Status != variantDone {
		t.Fatalf("reply: %+v", v)
	}

	rr.mu.Lock()
	call := rr.calls[0]
	rr.mu.Unlock()
	parts := call["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if u := parts[0].(map[string]any)["image_url"].(map[string]any)["url"]; u != "data:image/png;base64,"+base64.StdEncoding.EncodeToString(png) {
		t.Errorf("reader got image %v", u)
	}
	if parts[1].(map[string]any)["text"] != readerPrompt || call["temperature"] != float64(0) || call["chat_template_kwargs"].(map[string]any)["enable_thinking"] != false {
		t.Errorf("reader request: %v", call)
	}
	prompt, ok := rr.lastPrompt(t).(string)
	if !ok {
		t.Fatalf("a model that can't see must get text only, got %v", rr.lastPrompt(t))
	}
	if !strings.Contains(prompt, "Attached image scan.png, described by an image-reading model because you can't see images") ||
		!strings.Contains(prompt, "```\nText: Total due 1,250.00 EUR\n\nA scanned invoice.\n```") || strings.Contains(prompt, "hmm") ||
		!strings.HasSuffix(prompt, "What is the total?") {
		t.Errorf("prompt: %q", prompt)
	}

	// The description is saved with the image and made only once.
	d := rr.detail(t, c.ID)
	a := d.Messages[0].Attachments[0]
	if a.Extract != extractReader || a.TextSize == 0 {
		t.Fatalf("attachment meta: %+v", a)
	}
	if r, b := rr.do(t, "GET", "/api/attachments/"+a.ID+"?as=text", nil); !strings.HasPrefix(string(b), "Text: Total due") || !strings.HasPrefix(r.Header.Get("Content-Type"), "text/plain") {
		t.Errorf("description view: %v %q", r.Header, b)
	}
	if r, b := rr.do(t, "GET", "/api/attachments/"+a.ID, nil); !bytes.Equal(b, png) || r.Header.Get("Content-Type") != "image/png" {
		t.Errorf("the image itself: %v %q", r.Header, b)
	}
	if resp, b := rr.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "And the VAT?"}); resp.StatusCode != http.StatusCreated {
		t.Fatalf("second send: %d %s", resp.StatusCode, b)
	}
	rr.waitReply(t, c.ID)
	if n := rr.readerCalls(); n != 1 {
		t.Errorf("reader called %d times, want once", n)
	}

	// A model that can see gets the image itself, not the description.
	last := rr.detail(t, c.ID).Messages[3]
	if resp, b := rr.do(t, "POST", "/api/chats/"+c.ID+"/messages/"+last.ID+"/regenerate", map[string]any{"instance": "v"}); resp.StatusCode != http.StatusCreated {
		t.Fatalf("regenerate: %d %s", resp.StatusCode, b)
	}
	rr.waitReply(t, c.ID)
	hist := rr.chatAPIRig.bodies[len(rr.chatAPIRig.bodies)-1]["messages"].([]any)
	first := hist[0].(map[string]any)["content"].([]any)
	if first[0].(map[string]any)["text"] != "What is the total?" || first[1].(map[string]any)["type"] != "image_url" ||
		first[1].(map[string]any)["image_url"].(map[string]any)["url"] != "data:image/png;base64,"+base64.StdEncoding.EncodeToString(png) {
		t.Errorf("seeing model's first turn: %v", first)
	}
}

// sseReader reads a server-sent event stream one event at a time.
func newSSEScanner(resp *http.Response) *sseReader { return &sseReader{resp: resp} }

type sseReader struct {
	resp        *http.Response
	buf         []byte
	event, data string
}

func (s *sseReader) Scan() bool {
	s.event, s.data = "", ""
	tmp := make([]byte, 4096)
	for {
		if i := bytes.Index(s.buf, []byte("\n\n")); i >= 0 {
			block := string(s.buf[:i])
			s.buf = s.buf[i+2:]
			for _, ln := range strings.Split(block, "\n") {
				if v, ok := strings.CutPrefix(ln, "event: "); ok {
					s.event = v
				}
				if v, ok := strings.CutPrefix(ln, "data: "); ok {
					s.data = v
				}
			}
			if s.event != "" {
				return true
			}
			continue
		}
		n, err := s.resp.Body.Read(tmp)
		s.buf = append(s.buf, tmp[:n]...)
		if err != nil && n == 0 {
			return false
		}
	}
}

func TestChatUsesTheBrowsersDescription(t *testing.T) {
	rr := newReaderRig(t, func(w http.ResponseWriter, r *http.Request) { readerReply(w, "from the server", "stop") })
	c := rr.newChat(t, `{}`)
	att := imgAtt("cat.png", []byte("\x89PNG cat"))
	att["text"], att["extract"] = "A grey cat on a sofa.\x00", "reader"
	if resp, b := rr.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "", "attachments": []map[string]any{att}}); resp.StatusCode != http.StatusCreated {
		t.Fatalf("send: %d %s", resp.StatusCode, b)
	}
	rr.waitReply(t, c.ID)
	if rr.readerCalls() != 0 {
		t.Error("an image described in the browser must not be described again")
	}
	if p, _ := rr.lastPrompt(t).(string); !strings.Contains(p, "```\nA grey cat on a sofa.\n```") {
		t.Errorf("prompt: %q", p)
	}
}

func TestChatFailsClearlyWhenTheReaderIsDown(t *testing.T) {
	rr := newReaderRig(t, func(w http.ResponseWriter, r *http.Request) { readerReply(w, "unused", "stop") })
	setState(rr.ws.mgr.Get(readerName), StateStopped)
	c := rr.newChat(t, `{}`)
	rr.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "what is this", "attachments": []map[string]any{imgAtt("x.png", []byte("png"))}})
	v := rr.waitReply(t, c.ID)
	if v.Status != variantError || !strings.Contains(v.Error, "c can't see images, and the image reader couldn't describe x.png: image-reader is stopped") {
		t.Errorf("reply: %+v", v)
	}
	if rr.readerCalls() != 0 {
		t.Error("a stopped reader must not be called")
	}
}

func TestDescribeAPI(t *testing.T) {
	var reply func(w http.ResponseWriter, r *http.Request)
	rr := newReaderRig(t, func(w http.ResponseWriter, r *http.Request) { reply(w, r) })
	describe := func(ctype string, body []byte) (int, map[string]any) {
		req, _ := http.NewRequest("POST", rr.srv.URL+"/api/describe", bytes.NewReader(body))
		req.Header.Set("Content-Type", ctype)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out map[string]any
		json.NewDecoder(resp.Body).Decode(&out)
		return resp.StatusCode, out
	}

	reply = func(w http.ResponseWriter, r *http.Request) { readerReply(w, "A red parrot on a branch.", "stop") }
	if code, out := describe("image/jpeg", []byte("jpeg")); code != 200 || out["text"] != "A red parrot on a branch." || out["model"] != "unsloth/Qwen3.5-2B-GGUF:Q4_K_M" {
		t.Errorf("describe: %d %v", code, out)
	}
	reply = func(w http.ResponseWriter, r *http.Request) { readerReply(w, "Row 1\nRow 2", "length") }
	if code, out := describe("image/png", []byte("png")); code != 200 || out["truncated"] != true || !strings.HasSuffix(out["text"].(string), readerCutNote) {
		t.Errorf("cut description: %d %v", code, out)
	}
	reply = func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 400, map[string]any{"error": map[string]string{"message": "failed to load image"}})
	}
	if code, out := describe("image/webp", []byte("webp")); code != http.StatusBadGateway || out["reason"] != "unreadable" || !strings.Contains(out["error"].(string), "failed to load image") || !strings.Contains(out["error"].(string), "can't decode WebP") {
		t.Errorf("reader error: %d %v", code, out)
	}
	if code, out := describe("image/svg+xml", []byte("<svg/>")); code != http.StatusUnsupportedMediaType {
		t.Errorf("svg: %d %v", code, out)
	}
	if code, _ := describe("image/png", bytes.Repeat([]byte("x"), maxImageBytes+1)); code != http.StatusRequestEntityTooLarge {
		t.Errorf("too large: %d", code)
	}
	for st, reason := range map[InstanceState]string{StateStarting: "starting", StateStopped: "stopped", StateCrashed: "stopped"} {
		setState(rr.ws.mgr.Get(readerName), st)
		if code, out := describe("image/png", []byte("png")); code != http.StatusServiceUnavailable || out["reason"] != reason {
			t.Errorf("reader %s: %d %v", st, code, out)
		}
	}
	setState(rr.ws.mgr.Get(readerName), StateRunning)

	// The status tells the browser whether to describe images.
	_, b := rr.do(t, "GET", "/api/status", nil)
	var st ServerStatus
	json.Unmarshal(b, &st)
	if st.Documents.Images == nil || st.Documents.Images.Instance != "image-reader" || st.Documents.Images.State != StateRunning {
		t.Errorf("status: %s", b)
	}

	rr.ws.reader = nil
	if code, out := describe("image/png", []byte("png")); code != http.StatusServiceUnavailable || out["reason"] != "off" {
		t.Errorf("reader off: %d %v", code, out)
	}
}

func TestDescribeTimesOut(t *testing.T) {
	rr := newReaderRig(t, func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	})
	rr.ws.reader.timeout = 100 * time.Millisecond
	_, err := rr.ws.reader.Describe(t.Context(), []byte("png"), "image/png")
	var ee *extractError
	if !errors.As(err, &ee) || ee.Reason != "timeout" {
		t.Errorf("err = %v", err)
	}
}

func TestDecodeAttachmentsImageDescriptions(t *testing.T) {
	enc := base64.StdEncoding.EncodeToString([]byte("png"))
	text := func(s string) *string { return &s }
	for name, a := range map[string]sendAttachment{
		"wrong source": {Kind: "image", Name: "a.png", Mime: "image/png", Data: enc, Text: text("x"), Extract: "browser"},
		"too long":     {Kind: "image", Name: "a.png", Mime: "image/png", Data: enc, Text: text(strings.Repeat("x", maxReaderTextBytes+8<<10)), Extract: "reader"},
		"not utf-8":    {Kind: "image", Name: "a.png", Mime: "image/png", Data: enc, Text: text("\xff"), Extract: "reader"},
	} {
		if _, err := decodeAttachments([]sendAttachment{a}); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	got, err := decodeAttachments([]sendAttachment{{Kind: "image", Name: "a.png", Mime: "image/png", Data: enc, Text: text("  \n "), Extract: "reader"}})
	if err != nil || got[0].Extract != "" || got[0].Text != "" {
		t.Errorf("a blank description is dropped, not stored: %+v %v", got, err)
	}
}

func TestCleanDescription(t *testing.T) {
	for in, want := range map[string]string{
		"<think>\nlet me look\n</think>\n\nA dog.": "A dog.",
		"<think>still thinking":                    "",
		"Plain\r\n\r\n\r\ntext​":                   "Plain\n\ntext",
	} {
		if got, _ := cleanDescription(in); got != want {
			t.Errorf("cleanDescription(%q) = %q, want %q", in, got, want)
		}
	}
	long, cut := cleanDescription(strings.Repeat("é", maxReaderTextBytes))
	if !cut || len(long) > maxReaderTextBytes || !utf8.ValidString(long) {
		t.Errorf("long description: cut %v, %d bytes", cut, len(long))
	}
}
