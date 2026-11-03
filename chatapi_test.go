package main

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// chatAPIRig serves a WebServer with chat history and one running instance
// "c" backed by upstream, which sees each request's decoded JSON body.
type chatAPIRig struct {
	ws  *WebServer
	srv *httptest.Server // the manager, over real HTTP so SSE streams

	mu     sync.Mutex
	bodies []map[string]any
}

func newChatAPIRig(t *testing.T, upstream func(w http.ResponseWriter, r *http.Request, body map[string]any)) *chatAPIRig {
	t.Helper()
	rig := &chatAPIRig{}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		rig.mu.Lock()
		rig.bodies = append(rig.bodies, body)
		rig.mu.Unlock()
		upstream(w, r, body)
	}))
	t.Cleanup(up.Close)
	_, portStr, _ := net.SplitHostPort(strings.TrimPrefix(up.URL, "http://"))
	port, _ := strconv.Atoi(portStr)

	ws, cfg := newTestServer(t)
	cfg.AllowedHosts = append(cfg.AllowedHosts, "127.0.0.1")
	if err := ws.mgr.AddInstance(InstanceConf{Name: "c", Model: "/m.gguf", Port: port, GPUIDs: []int{0}}); err != nil {
		t.Fatal(err)
	}
	inst := ws.mgr.Get("c")
	inst.mu.Lock()
	inst.state = StateRunning
	inst.mu.Unlock()

	store, err := OpenChatStore(filepath.Join(t.TempDir(), "chats.db"))
	if err != nil {
		t.Fatal(err)
	}
	ws.chats = NewChatService(store, ws.mgr)
	rig.ws = ws
	rig.srv = httptest.NewServer(ws)
	t.Cleanup(func() {
		rig.srv.Close()
		ws.chats.Shutdown(5 * time.Second)
		store.Close()
	})
	return rig
}

func (rig *chatAPIRig) do(t *testing.T, method, path string, body any) (*http.Response, []byte) {
	t.Helper()
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = strings.NewReader(string(b))
	}
	req, _ := http.NewRequest(method, rig.srv.URL+path, rd)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp, b
}

func (rig *chatAPIRig) newChat(t *testing.T, params string) Chat {
	t.Helper()
	resp, b := rig.do(t, "POST", "/api/chats", map[string]any{"instance": "c", "params": json.RawMessage(params)})
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("create chat: %d %s", resp.StatusCode, b)
	}
	var c Chat
	json.Unmarshal(b, &c)
	return c
}

func (rig *chatAPIRig) detail(t *testing.T, id string) chatDetail {
	t.Helper()
	_, b := rig.do(t, "GET", "/api/chats/"+id, nil)
	var d chatDetail
	if err := json.Unmarshal(b, &d); err != nil {
		t.Fatalf("detail: %v: %s", err, b)
	}
	return d
}

// waitReply waits until the chat's last reply has left the streaming state.
func (rig *chatAPIRig) waitReply(t *testing.T, id string) Variant {
	t.Helper()
	var v Variant
	waitFor(t, 5*time.Second, "reply to finish", func() bool {
		d := rig.detail(t, id)
		last := d.Messages[len(d.Messages)-1]
		v = last.Variants[last.VI]
		return v.Status != variantStreaming && !rig.ws.chats.Active(id)
	})
	return v
}

func sse(w http.ResponseWriter, chunks ...string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	for _, c := range chunks {
		fmt.Fprintf(w, "data: %s\n\n", c)
		w.(http.Flusher).Flush()
	}
}

func delta(field, text string) string {
	b, _ := json.Marshal(map[string]any{"choices": []map[string]any{{"delta": map[string]string{field: text}}}})
	return string(b)
}

const finalChunk = `{"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"prompt_n":12,"prompt_per_second":480.5,"predicted_n":3,"predicted_per_second":61.25}}`

func titleReply(w http.ResponseWriter, title string) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": title}}}})
}

func TestChatSendStreamsSavesAndTitles(t *testing.T) {
	rig := newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if body["stream"] != true {
			titleReply(w, `"KV Cache Sizing"`)
			return
		}
		sse(w, delta("reasoning_content", "Think first. "), delta("content", "Hello "), delta("content", "there"), finalChunk, "[DONE]")
	})
	c := rig.newChat(t, `{"temp":0.3,"sys":"Be brief.","think":false,"max":"","top_k":"20"}`)
	img := base64.StdEncoding.EncodeToString([]byte("\x89PNG fake"))
	resp, b := rig.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{
		"content": "size my kv cache",
		"attachments": []map[string]string{
			{"kind": "file", "name": "notes.txt", "data": base64.StdEncoding.EncodeToString([]byte("ctx=65536"))},
			{"kind": "image", "name": "a.png", "mime": "image/png", "data": img},
		},
	})
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("send: %d %s", resp.StatusCode, b)
	}
	v := rig.waitReply(t, c.ID)
	if v.Status != variantDone || v.Content != "Hello there" || v.Reasoning != "Think first. " {
		t.Fatalf("saved reply: %+v", v)
	}
	var st replyStats
	json.Unmarshal(v.Stats, &st)
	if st.TPS != 61.25 || st.Tokens != 3 || st.PromptTok != 12 || st.Finish != "stop" {
		t.Errorf("stats: %+v", st)
	}

	rig.mu.Lock()
	sent := rig.bodies[0]
	rig.mu.Unlock()
	msgs := sent["messages"].([]any)
	if sys := msgs[0].(map[string]any); sys["role"] != "system" || sys["content"] != "Be brief." {
		t.Errorf("system prompt not sent first: %v", msgs[0])
	}
	user := msgs[1].(map[string]any)["content"].([]any)
	text := user[0].(map[string]any)["text"].(string)
	if !strings.Contains(text, "size my kv cache") || !strings.Contains(text, "Attached file notes.txt") || !strings.Contains(text, "ctx=65536") {
		t.Errorf("text file not inlined: %q", text)
	}
	if url := user[1].(map[string]any)["image_url"].(map[string]any)["url"].(string); url != "data:image/png;base64,"+img {
		t.Errorf("image part: %q", url)
	}
	if sent["temperature"] != 0.3 || sent["top_k"] != float64(20) {
		t.Errorf("sampling not forwarded: %v", sent)
	}
	if _, ok := sent["max_tokens"]; ok {
		t.Error("empty max must not be sent")
	}
	if kw, _ := sent["chat_template_kwargs"].(map[string]any); kw["enable_thinking"] != false {
		t.Errorf("thinking off should be forwarded: %v", sent["chat_template_kwargs"])
	}

	waitFor(t, 5*time.Second, "model-written title", func() bool { return rig.detail(t, c.ID).Chat.Title == "KV Cache Sizing" })
}

func readEvents(t *testing.T, resp *http.Response, until string) []string {
	t.Helper()
	var events []string
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		if name, ok := strings.CutPrefix(sc.Text(), "event: "); ok {
			events = append(events, name)
			if name == until {
				break
			}
		}
	}
	return events
}

func TestChatEventsFollowAndOutliveTheBrowser(t *testing.T) {
	release := make(chan struct{})
	rig := newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if body["stream"] != true {
			http.Error(w, "no title today", 500)
			return
		}
		sse(w, delta("content", "part one, "))
		<-release
		fmt.Fprintf(w, "data: %s\n\n", delta("content", "part two"))
		fmt.Fprintf(w, "data: %s\n\n", finalChunk)
	})
	c := rig.newChat(t, `{}`)
	if resp, b := rig.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "hi"}); resp.StatusCode != http.StatusCreated {
		t.Fatalf("send: %d %s", resp.StatusCode, b)
	}

	// A second send while the first is streaming is refused.
	if resp, _ := rig.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "again"}); resp.StatusCode != http.StatusConflict {
		t.Errorf("concurrent send: %d, want 409", resp.StatusCode)
	}

	// A browser that follows and then goes away doesn't stop the reply.
	resp, err := http.Get(rig.srv.URL + "/api/chats/" + c.ID + "/events")
	if err != nil {
		t.Fatal(err)
	}
	if ev := readEvents(t, resp, "snapshot"); len(ev) == 0 || ev[len(ev)-1] != "snapshot" {
		t.Fatalf("events: %v", ev)
	}
	resp.Body.Close()
	list := []Chat{}
	_, b := rig.do(t, "GET", "/api/chats", nil)
	json.Unmarshal(b, &list)
	if len(list) != 1 || !list[0].Streaming {
		t.Errorf("list should flag the streaming chat: %+v", list)
	}
	close(release)
	if v := rig.waitReply(t, c.ID); v.Status != variantDone || v.Content != "part one, part two" {
		t.Fatalf("reply should finish after the browser left: %+v", v)
	}

	// With nothing running, events answer idle at once.
	resp, err = http.Get(rig.srv.URL + "/api/chats/" + c.ID + "/events")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ev := readEvents(t, resp, "idle"); len(ev) != 1 || ev[0] != "idle" {
		t.Errorf("idle chat events: %v", ev)
	}
}

func TestChatStopKeepsPartialReply(t *testing.T) {
	rig := newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		sse(w, delta("content", "the start"))
		<-r.Context().Done()
	})
	c := rig.newChat(t, `{}`)
	rig.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "write a lot"})
	waitFor(t, 5*time.Second, "first token", func() bool {
		d := rig.detail(t, c.ID)
		_, snapOK := func() (genSnapshot, bool) {
			s, _, unsub, ok := rig.ws.chats.Subscribe(c.ID)
			if ok {
				unsub()
			}
			return s, ok && s.Content != ""
		}()
		return len(d.Messages) == 2 && snapOK
	})
	if resp, _ := rig.do(t, "POST", "/api/chats/"+c.ID+"/stop", nil); resp.StatusCode != http.StatusOK {
		t.Fatalf("stop: %d", resp.StatusCode)
	}
	if v := rig.waitReply(t, c.ID); v.Status != variantStopped || v.Content != "the start" {
		t.Errorf("stopped reply: %+v", v)
	}
}

func TestChatRegenerateEditAndSelect(t *testing.T) {
	n := 0
	var mu sync.Mutex
	rig := newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if body["stream"] != true {
			titleReply(w, "T")
			return
		}
		mu.Lock()
		n++
		k := n
		mu.Unlock()
		sse(w, delta("content", fmt.Sprintf("answer %d", k)), finalChunk)
	})
	c := rig.newChat(t, `{}`)
	rig.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "q"})
	rig.waitReply(t, c.ID)
	d := rig.detail(t, c.ID)
	reply := d.Messages[1]

	if resp, b := rig.do(t, "POST", "/api/chats/"+c.ID+"/messages/"+reply.ID+"/regenerate", nil); resp.StatusCode != http.StatusCreated {
		t.Fatalf("regenerate: %d %s", resp.StatusCode, b)
	}
	rig.waitReply(t, c.ID)
	d = rig.detail(t, c.ID)
	if r := d.Messages[1]; len(r.Variants) != 2 || r.VI != 1 || r.Variants[1].Content != "answer 2" {
		t.Fatalf("after regenerate: %+v", r)
	}
	if resp, _ := rig.do(t, "PATCH", "/api/chats/"+c.ID+"/messages/"+reply.ID, map[string]int{"vi": 0}); resp.StatusCode != http.StatusOK {
		t.Errorf("select version: %d", resp.StatusCode)
	}
	if rig.detail(t, c.ID).Messages[1].VI != 0 {
		t.Error("version selection not saved")
	}

	user := d.Messages[0]
	if resp, b := rig.do(t, "PUT", "/api/chats/"+c.ID+"/messages/"+user.ID, map[string]string{"content": "better q"}); resp.StatusCode != http.StatusCreated {
		t.Fatalf("edit: %d %s", resp.StatusCode, b)
	}
	rig.waitReply(t, c.ID)
	d = rig.detail(t, c.ID)
	if len(d.Messages) != 2 || d.Messages[0].Content != "better q" || d.Messages[1].Variants[0].Content != "answer 3" {
		t.Fatalf("after edit: %+v", d.Messages)
	}
}

func TestChatRejectsStoppedInstanceAndBadAttachments(t *testing.T) {
	rig := newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) { sse(w, finalChunk) })
	c := rig.newChat(t, `{}`)
	for name, body := range map[string]map[string]any{
		"svg":    {"content": "x", "attachments": []map[string]string{{"kind": "image", "name": "a.svg", "mime": "image/svg+xml", "data": "PHN2Zz4="}}},
		"binary": {"content": "x", "attachments": []map[string]string{{"kind": "file", "name": "a.bin", "data": base64.StdEncoding.EncodeToString([]byte{0xff, 0xfe, 0x00})}}},
		"empty":  {"content": "  "},
	} {
		if resp, _ := rig.do(t, "POST", "/api/chats/"+c.ID+"/messages", body); resp.StatusCode != http.StatusBadRequest {
			t.Errorf("%s: %d, want 400", name, resp.StatusCode)
		}
	}
	inst := rig.ws.mgr.Get("c")
	inst.mu.Lock()
	inst.state = StateStopped
	inst.mu.Unlock()
	resp, b := rig.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "hello"})
	if resp.StatusCode != http.StatusConflict || !strings.Contains(string(b), "start it") {
		t.Errorf("stopped instance: %d %s", resp.StatusCode, b)
	}
	if d := rig.detail(t, c.ID); len(d.Messages) != 0 {
		t.Errorf("a refused message must not be stored: %+v", d.Messages)
	}
}

func TestAttachmentsCannotRunScripts(t *testing.T) {
	rig := newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) { sse(w, finalChunk) })
	c := rig.newChat(t, `{}`)
	rig.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "x", "attachments": []map[string]string{
		{"kind": "image", "name": "a.png", "mime": "image/png", "data": base64.StdEncoding.EncodeToString([]byte("png"))},
		{"kind": "file", "name": "page.html", "data": base64.StdEncoding.EncodeToString([]byte("<script>alert(1)</script>"))},
	}})
	rig.waitReply(t, c.ID)
	atts := rig.detail(t, c.ID).Messages[0].Attachments
	for _, a := range atts {
		resp, _ := rig.do(t, "GET", "/api/attachments/"+a.ID, nil)
		if resp.Header.Get("X-Content-Type-Options") != "nosniff" || !strings.Contains(resp.Header.Get("Content-Security-Policy"), "sandbox") {
			t.Errorf("%s: missing nosniff/sandbox headers: %v", a.Name, resp.Header)
		}
		ct := resp.Header.Get("Content-Type")
		if a.Kind == "image" && ct != "image/png" {
			t.Errorf("image served as %q", ct)
		}
		if a.Kind == "file" && (!strings.HasPrefix(ct, "text/plain") || !strings.HasPrefix(resp.Header.Get("Content-Disposition"), "attachment")) {
			t.Errorf("text file must download as plain text, got %q %q", ct, resp.Header.Get("Content-Disposition"))
		}
	}
}

func TestChatEndpointsKeepOriginCheck(t *testing.T) {
	rig := newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) { sse(w, finalChunk) })
	c := rig.newChat(t, `{}`)
	for _, tc := range []struct{ method, path string }{
		{"POST", "/api/chats"}, {"POST", "/api/chats/" + c.ID + "/messages"}, {"DELETE", "/api/chats/" + c.ID}, {"PATCH", "/api/chats/" + c.ID},
	} {
		req, _ := http.NewRequest(tc.method, rig.srv.URL+tc.path, strings.NewReader(`{"content":"x"}`))
		req.Header.Set("Origin", "http://evil.example")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusForbidden {
			t.Errorf("%s %s from a foreign origin: %d, want 403", tc.method, tc.path, resp.StatusCode)
		}
	}
	if _, err := rig.ws.chats.store.GetChat(c.ID); err != nil {
		t.Error("the cross-site delete must not have happened")
	}
}

func TestChatExportAndUnavailableStore(t *testing.T) {
	rig := newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if body["stream"] != true {
			titleReply(w, "Backoff in Go")
			return
		}
		sse(w, delta("content", "Use `time.After`."), finalChunk)
	})
	c := rig.newChat(t, `{}`)
	rig.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "how to back off?"})
	rig.waitReply(t, c.ID)
	waitFor(t, 5*time.Second, "title", func() bool { return rig.detail(t, c.ID).Chat.Title == "Backoff in Go" })
	resp, b := rig.do(t, "GET", "/api/chats/"+c.ID+"/export", nil)
	md := string(b)
	if !strings.Contains(resp.Header.Get("Content-Disposition"), `filename="backoff-in-go.md"`) || !strings.HasPrefix(md, "# Backoff in Go") ||
		!strings.Contains(md, "**You**\n\nhow to back off?") || !strings.Contains(md, "**c**\n\nUse `time.After`.") {
		t.Errorf("export: %q %q", resp.Header.Get("Content-Disposition"), md)
	}

	ws, _ := newTestServer(t)
	rec := httptest.NewRecorder()
	ws.ServeHTTP(rec, httptest.NewRequest("GET", "/api/chats", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("without a database the chat API should answer 503, got %d", rec.Code)
	}
}

func TestStatusAndStaticAssets(t *testing.T) {
	ws, _ := newTestServer(t)
	rec := httptest.NewRecorder()
	ws.ServeHTTP(rec, httptest.NewRequest("GET", "/api/status", nil))
	var st ServerStatus
	if err := json.Unmarshal(rec.Body.Bytes(), &st); err != nil || st.Addresses == nil || st.ChatHistory {
		t.Errorf("status: %v %s", err, rec.Body.String())
	}

	rec = httptest.NewRecorder()
	ws.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if !strings.Contains(rec.Body.String(), "/static/app.js?v="+assetVersion) {
		t.Error("index should reference the versioned app.js")
	}
	rec = httptest.NewRecorder()
	ws.ServeHTTP(rec, httptest.NewRequest("GET", "/static/app.js?v="+assetVersion, nil))
	if rec.Code != 200 || !strings.Contains(rec.Header().Get("Cache-Control"), "immutable") {
		t.Errorf("versioned asset: %d %q", rec.Code, rec.Header().Get("Cache-Control"))
	}
	rec = httptest.NewRecorder()
	ws.ServeHTTP(rec, httptest.NewRequest("GET", "/static/app.js", nil))
	if rec.Header().Get("Cache-Control") != "no-cache" {
		t.Errorf("unversioned asset must revalidate, got %q", rec.Header().Get("Cache-Control"))
	}
}

func TestAddressKind(t *testing.T) {
	for _, tc := range []struct{ iface, ip, want string }{
		{"en0", "192.168.1.50", "lan"}, {"eth0", "10.0.0.4", "lan"}, {"utun3", "10.8.0.1", "vpn"},
		{"tailscale0", "100.101.1.2", "vpn"}, {"en0", "100.64.3.3", "vpn"}, {"en0", "8.8.8.8", "other"},
	} {
		if got := addressKind(tc.iface, net.ParseIP(tc.ip)); got != tc.want {
			t.Errorf("%s %s: %s, want %s", tc.iface, tc.ip, got, tc.want)
		}
	}
	if !templateThinks("{% if enable_thinking %}") || templateThinks("{{ messages }}") {
		t.Error("templateThinks")
	}
}
