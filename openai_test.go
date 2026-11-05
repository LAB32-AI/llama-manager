package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// v1Rig is a WebServer with instance "c" running behind a fake llama-server
// that records each request and answers per path.
type v1Rig struct {
	*chatAPIRig
	mu    sync.Mutex
	calls []v1Call
}

type v1Call struct {
	path string
	body map[string]any
	auth string
}

func newV1Rig(t *testing.T, answer func(w http.ResponseWriter, r *http.Request, body map[string]any)) *v1Rig {
	t.Helper()
	vr := &v1Rig{}
	vr.chatAPIRig = newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		vr.mu.Lock()
		vr.calls = append(vr.calls, v1Call{r.URL.Path, body, r.Header.Get("Authorization")})
		vr.mu.Unlock()
		if answer != nil {
			answer(w, r, body)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"path": r.URL.Path, "model": body["model"]})
	})
	return vr
}

func (vr *v1Rig) lastCall(t *testing.T) v1Call {
	t.Helper()
	vr.mu.Lock()
	defer vr.mu.Unlock()
	if len(vr.calls) == 0 {
		t.Fatal("llama-server got no request")
	}
	return vr.calls[len(vr.calls)-1]
}

func (vr *v1Rig) nCalls() int {
	vr.mu.Lock()
	defer vr.mu.Unlock()
	return len(vr.calls)
}

// req sends a request with headers to the rig's manager.
func (vr *v1Rig) req(t *testing.T, method, path, body string, headers map[string]string) (*http.Response, []byte) {
	t.Helper()
	var rd io.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	req, _ := http.NewRequest(method, vr.srv.URL+path, rd)
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	for k, v := range headers {
		if k == "Host" {
			req.Host = v
			continue
		}
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp, b
}

type openAIErr struct {
	Error struct {
		Message string  `json:"message"`
		Type    string  `json:"type"`
		Param   *string `json:"param"`
		Code    *string `json:"code"`
	} `json:"error"`
}

// wantAPIError checks an OpenAI-style JSON error and returns it.
func wantAPIError(t *testing.T, what string, resp *http.Response, b []byte, status int, code string) openAIErr {
	t.Helper()
	var e openAIErr
	if resp.StatusCode != status {
		t.Errorf("%s: status %d, want %d: %s", what, resp.StatusCode, status, b)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("%s: Content-Type %q, want application/json: %s", what, ct, b)
	}
	if err := json.Unmarshal(b, &e); err != nil || e.Error.Message == "" || e.Error.Type == "" {
		t.Errorf("%s: not an OpenAI error: %s", what, b)
		return e
	}
	got := ""
	if e.Error.Code != nil {
		got = *e.Error.Code
	}
	if got != code {
		t.Errorf("%s: code %q, want %q: %s", what, got, code, b)
	}
	return e
}

func addInstance(t *testing.T, ws *WebServer, name string, state InstanceState) {
	t.Helper()
	if err := ws.mgr.AddInstance(InstanceConf{Name: name, Model: "/m.gguf", Port: freePort(t), GPUIDs: []int{0}}); err != nil {
		t.Fatal(err)
	}
	setState(ws.mgr.Get(name), state)
}

func TestV1ModelsListsOnlyRunning(t *testing.T) {
	vr := newV1Rig(t, nil)
	addInstance(t, vr.ws, "stopped-one", StateStopped)
	addInstance(t, vr.ws, "loading-one", StateStarting)
	inst := vr.ws.mgr.Get("c")
	inst.mu.Lock()
	inst.startedAt = time.Unix(1700000000, 0)
	inst.mu.Unlock()

	resp, b := vr.req(t, "GET", "/v1/models", "", nil)
	var list struct {
		Object string    `json:"object"`
		Data   []v1Model `json:"data"`
	}
	if resp.StatusCode != http.StatusOK || json.Unmarshal(b, &list) != nil || list.Object != "list" {
		t.Fatalf("models: %d %s", resp.StatusCode, b)
	}
	if len(list.Data) != 1 || list.Data[0] != (v1Model{ID: "c", Object: "model", Created: 1700000000, OwnedBy: "llama-manager"}) {
		t.Errorf("only the running instance should be listed: %s", b)
	}
	if resp, b := vr.req(t, "GET", "/v1/models/c", "", nil); resp.StatusCode != http.StatusOK || !strings.Contains(string(b), `"id":"c"`) {
		t.Errorf("retrieve c: %d %s", resp.StatusCode, b)
	}
	resp, b = vr.req(t, "GET", "/v1/models/stopped-one", "", nil)
	wantAPIError(t, "retrieve a stopped model", resp, b, http.StatusNotFound, "model_not_found")
	if resp, b := vr.req(t, "GET", "/models", "", nil); resp.StatusCode != http.StatusOK || !strings.Contains(string(b), `"id":"c"`) {
		t.Errorf("/models without /v1: %d %s", resp.StatusCode, b)
	}
}

func TestV1ForwardsLlamaServerRoutes(t *testing.T) {
	vr := newV1Rig(t, nil)
	for _, p := range llamaRoutes {
		resp, b := vr.req(t, "POST", p, `{"model":"c","input":"x"}`, nil)
		if resp.StatusCode != http.StatusOK || !strings.Contains(string(b), `"path":"`+p+`"`) {
			t.Errorf("%s: %d %s", p, resp.StatusCode, b)
		}
		if c := vr.lastCall(t); c.path != p || c.body["model"] != "c" {
			t.Errorf("%s reached llama-server as %s %v", p, c.path, c.body)
		}
	}
	// Apps given the base URL without /v1.
	if resp, b := vr.req(t, "POST", "/chat/completions", `{"model":"c","messages":[]}`, nil); resp.StatusCode != http.StatusOK || vr.lastCall(t).path != "/v1/chat/completions" {
		t.Errorf("/chat/completions: %d %s", resp.StatusCode, b)
	}
	if resp, _ := vr.req(t, "GET", "/models/", "", nil); resp.StatusCode != http.StatusNotFound {
		t.Errorf("/models/ is a model id lookup: %d", resp.StatusCode)
	}
	if resp, _ := vr.req(t, "GET", "/static/app.css", "", nil); resp.StatusCode != http.StatusOK {
		t.Errorf("the dashboard's own paths must not be taken as /v1: %d", resp.StatusCode)
	}
}

func TestV1StreamsResponses(t *testing.T) {
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "event: response.output_text.delta\ndata: {\"delta\":\"Hi\"}\n\n")
		w.(http.Flusher).Flush()
		fmt.Fprint(w, "event: response.completed\ndata: {}\n\n")
	})
	resp, b := vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"x","stream":true}`, nil)
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "text/event-stream" ||
		!strings.Contains(string(b), "response.output_text.delta") || !strings.Contains(string(b), "response.completed") {
		t.Fatalf("stream: %d %q %s", resp.StatusCode, resp.Header.Get("Content-Type"), b)
	}
	if resp.Header.Get("X-Request-Id") == "" {
		t.Error("no X-Request-Id")
	}
}

func TestV1ErrorsAreOpenAIJSON(t *testing.T) {
	vr := newV1Rig(t, nil)
	addInstance(t, vr.ws, "off", StateStopped)
	addInstance(t, vr.ws, "warming", StateStarting)

	resp, b := vr.req(t, "POST", "/v1/chat/completions", `{"model":"gpt-4o","messages":[]}`, nil)
	e := wantAPIError(t, "unknown model", resp, b, http.StatusNotFound, "model_not_found")
	if !strings.Contains(e.Error.Message, "Running models: c") || e.Error.Type != "invalid_request_error" || e.Error.Param == nil || *e.Error.Param != "model" {
		t.Errorf("unknown model: %s", b)
	}
	resp, b = vr.req(t, "POST", "/v1/chat/completions", `{"messages":[]}`, nil)
	wantAPIError(t, "no model", resp, b, http.StatusBadRequest, "")
	resp, b = vr.req(t, "POST", "/v1/chat/completions", `{bad`, nil)
	wantAPIError(t, "bad json", resp, b, http.StatusBadRequest, "")
	resp, b = vr.req(t, "POST", "/v1/chat/completions", `{"model":"off"}`, nil)
	if e := wantAPIError(t, "stopped", resp, b, http.StatusNotFound, "model_not_found"); !strings.Contains(e.Error.Message, "stopped") {
		t.Errorf("stopped: %s", b)
	}
	resp, b = vr.req(t, "POST", "/v1/embeddings", `{"model":"warming"}`, nil)
	wantAPIError(t, "starting", resp, b, http.StatusServiceUnavailable, "model_not_ready")
	resp, b = vr.req(t, "PUT", "/v1/embeddings", "{}", nil)
	wantAPIError(t, "PUT embeddings", resp, b, http.StatusMethodNotAllowed, "method_not_allowed")
	if resp.Header.Get("Allow") != "POST" {
		t.Errorf("405 without Allow: %v", resp.Header)
	}
	resp, b = vr.req(t, "POST", "/v1/nothing/here", "{}", nil)
	if e := wantAPIError(t, "unknown path", resp, b, http.StatusNotFound, ""); e.Error.Message != "Invalid URL (POST /v1/nothing/here)" {
		t.Errorf("unknown path: %s", b)
	}
	for _, p := range []string{"/v1/files", "/v1/batches", "/v1/moderations", "/v1/vector_stores/vs_1/files", "/v1/assistants",
		"/v1/agents/sessions", "/v1/skills", "/v1/vaults/v1", "/v1/webhook_endpoints", "/v1/live/sessions", "/v1/realtime/calls/c1/accept",
		"/v1/audio/voices", "/v1/audio/voice_consents/x", "/v1/decisions", "/v1/projects/p/roles", "/v1/realtime/translations/client_secrets"} {
		resp, b = vr.req(t, "POST", p, "{}", nil)
		wantAPIError(t, p, resp, b, http.StatusNotFound, "unsupported_endpoint")
	}
	resp, b = vr.req(t, "POST", "/v1/images/generations", `{"prompt":"x"}`, nil)
	wantAPIError(t, "images off", resp, b, http.StatusNotFound, "model_not_found")
	resp, b = vr.req(t, "POST", "/v1/audio/speech", `{"input":"x"}`, nil)
	wantAPIError(t, "speech off", resp, b, http.StatusNotFound, "model_not_found")

	// An upstream that can't be reached: forwardLlama's plain-text error
	// comes out as JSON.
	inst := vr.ws.mgr.Get("off")
	setState(inst, StateRunning)
	resp, b = vr.req(t, "POST", "/v1/chat/completions", `{"model":"off"}`, nil)
	wantAPIError(t, "unreachable upstream", resp, b, http.StatusBadGateway, "")

	old := maxAPIBody
	maxAPIBody = 1 << 20
	t.Cleanup(func() { maxAPIBody = old })
	resp, b = vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","x":"`+strings.Repeat("a", 2<<20)+`"}`, nil)
	wantAPIError(t, "too big", resp, b, http.StatusRequestEntityTooLarge, "request_too_large")
}

func TestV1TakesLargeVisionRequests(t *testing.T) {
	vr := newV1Rig(t, nil)
	img := strings.Repeat("A", 6<<20) // a phone photo as base64 is over the old 4 MiB cap
	body := `{"model":"c","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,` + img + `"}}]}]}`
	if resp, b := vr.req(t, "POST", "/v1/chat/completions", body, nil); resp.StatusCode != http.StatusOK {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
}

func TestV1FallbackModel(t *testing.T) {
	vr := newV1Rig(t, nil)
	vr.ws.cfg.mu.Lock()
	vr.ws.cfg.DefaultModel = "c"
	vr.ws.cfg.mu.Unlock()
	for _, body := range []string{`{"model":"gpt-4o-mini","messages":[]}`, `{"messages":[]}`} {
		if resp, b := vr.req(t, "POST", "/v1/chat/completions", body, nil); resp.StatusCode != http.StatusOK {
			t.Errorf("%s: %d %s", body, resp.StatusCode, b)
		}
	}
	if resp, _ := vr.req(t, "POST", "/v1/chat/completions", `{"model":"C","messages":[]}`, nil); resp.StatusCode != http.StatusOK {
		t.Errorf("model names should match ignoring case: %d", resp.StatusCode)
	}
	addInstance(t, vr.ws, "named-but-stopped", StateStopped)
	resp, b := vr.req(t, "POST", "/v1/chat/completions", `{"model":"named-but-stopped"}`, nil)
	wantAPIError(t, "a named model doesn't fall back", resp, b, http.StatusNotFound, "model_not_found")
}

func setKeys(ws *WebServer, keys ...string) {
	ws.cfg.mu.Lock()
	ws.cfg.APIKeys = keys
	ws.cfg.mu.Unlock()
}

func TestV1APIKeys(t *testing.T) {
	vr := newV1Rig(t, nil)
	chat := `{"model":"c","messages":[]}`
	for name, h := range map[string]map[string]string{"no key": nil, "any key": {"Authorization": "Bearer sk-whatever"}} {
		if resp, b := vr.req(t, "POST", "/v1/chat/completions", chat, h); resp.StatusCode != http.StatusOK {
			t.Errorf("without api_keys, %s must work: %d %s", name, resp.StatusCode, b)
		}
	}
	setKeys(vr.ws, "sk-one", "sk-two")
	for name, h := range map[string]map[string]string{
		"none":         nil,
		"wrong":        {"Authorization": "Bearer sk-three"},
		"not bearer":   {"Authorization": "Basic sk-one"},
		"key prefixed": {"Authorization": "Bearer sk-one2"},
	} {
		resp, b := vr.req(t, "POST", "/v1/chat/completions", chat, h)
		wantAPIError(t, name, resp, b, http.StatusUnauthorized, "invalid_api_key")
	}
	for name, h := range map[string]map[string]string{
		"bearer":     {"Authorization": "Bearer sk-two"},
		"bearer lc":  {"Authorization": "bearer sk-one"},
		"x-api-key":  {"X-Api-Key": "sk-one"},
		"azure-like": {"Api-Key": "sk-two"},
	} {
		if resp, b := vr.req(t, "POST", "/v1/chat/completions", chat, h); resp.StatusCode != http.StatusOK {
			t.Errorf("%s: %d %s", name, resp.StatusCode, b)
		}
	}
	if c := vr.lastCall(t); c.auth != "" {
		t.Errorf("the key must not be passed to llama-server: %q", c.auth)
	}
	resp, b := vr.req(t, "GET", "/v1/models", "", nil)
	wantAPIError(t, "models without key", resp, b, http.StatusUnauthorized, "invalid_api_key")
	if resp, _ := vr.req(t, "GET", "/v1/images/files/img_nope.png", "", nil); resp.StatusCode != http.StatusNotFound {
		t.Errorf("image links need no key (they are unguessable): %d", resp.StatusCode)
	}
	if resp, _ := vr.req(t, "GET", "/api/status", "", nil); resp.StatusCode != http.StatusOK {
		t.Errorf("the dashboard API doesn't take keys: %d", resp.StatusCode)
	}
	if resp, _ := vr.req(t, "OPTIONS", "/v1/chat/completions", "", nil); resp.StatusCode != http.StatusNoContent {
		t.Errorf("a preflight carries no key: %d", resp.StatusCode)
	}
}

func TestV1HostCheck(t *testing.T) {
	vr := newV1Rig(t, nil)
	chat := `{"model":"c","messages":[]}`
	resp, b := vr.req(t, "POST", "/v1/chat/completions", chat, map[string]string{"Host": "rebound.example.net"})
	wantAPIError(t, "unknown host", resp, b, http.StatusForbidden, "host_not_allowed")
	if resp, b := vr.req(t, "POST", "/v1/chat/completions", chat, map[string]string{"Host": "host.docker.internal:8080"}); resp.StatusCode != http.StatusOK {
		t.Errorf("apps in Docker reach the host as host.docker.internal: %d %s", resp.StatusCode, b)
	}
	setKeys(vr.ws, "sk-one")
	if resp, b := vr.req(t, "POST", "/v1/chat/completions", chat, map[string]string{"Host": "gpu.example.net", "Authorization": "Bearer sk-one"}); resp.StatusCode != http.StatusOK {
		t.Errorf("a valid key stands in for allowed_hosts: %d %s", resp.StatusCode, b)
	}
	resp, b = vr.req(t, "POST", "/v1/chat/completions", chat, map[string]string{"Host": "gpu.example.net", "Authorization": "Bearer nope"})
	wantAPIError(t, "unknown host, bad key", resp, b, http.StatusForbidden, "host_not_allowed")
	if resp, _ := vr.req(t, "GET", "/v1/images/files/img_nope.png", "", map[string]string{"Host": "gpu.example.net"}); resp.StatusCode != http.StatusNotFound {
		t.Errorf("picture links are fetched without a key, from any host name: %d", resp.StatusCode)
	}
}

func TestV1CORS(t *testing.T) {
	vr := newV1Rig(t, nil)
	chat := `{"model":"c","messages":[]}`
	page := "http://app.lan:3000"
	preflight := map[string]string{"Origin": page, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization, content-type, x-stainless-os"}

	resp, b := vr.req(t, "OPTIONS", "/v1/chat/completions", "", preflight)
	wantAPIError(t, "preflight without cors_origins", resp, b, http.StatusForbidden, "origin_not_allowed")
	resp, b = vr.req(t, "POST", "/v1/chat/completions", chat, map[string]string{"Origin": page})
	wantAPIError(t, "cross-site POST without cors_origins", resp, b, http.StatusForbidden, "origin_not_allowed")
	if resp, _ := vr.req(t, "POST", "/v1/chat/completions", chat, map[string]string{"Origin": strings.TrimSuffix(vr.srv.URL, "/")}); resp.StatusCode != http.StatusOK {
		t.Errorf("same-origin POST: %d", resp.StatusCode)
	}

	vr.ws.cfg.mu.Lock()
	vr.ws.cfg.CORSOrigins = []string{page}
	vr.ws.cfg.mu.Unlock()
	resp, _ = vr.req(t, "OPTIONS", "/v1/chat/completions", "", preflight)
	if resp.StatusCode != http.StatusNoContent || resp.Header.Get("Access-Control-Allow-Origin") != page ||
		resp.Header.Get("Access-Control-Allow-Headers") != preflight["Access-Control-Request-Headers"] ||
		!strings.Contains(resp.Header.Get("Access-Control-Allow-Methods"), "POST") {
		t.Errorf("preflight: %d %v", resp.StatusCode, resp.Header)
	}
	resp, b = vr.req(t, "POST", "/v1/chat/completions", chat, map[string]string{"Origin": page})
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Access-Control-Allow-Origin") != page {
		t.Errorf("allowed page: %d %v %s", resp.StatusCode, resp.Header, b)
	}
	resp, b = vr.req(t, "POST", "/v1/chat/completions", chat, map[string]string{"Origin": "http://evil.example"})
	wantAPIError(t, "another page", resp, b, http.StatusForbidden, "origin_not_allowed")
	if resp, _ := vr.req(t, "POST", "/api/instances/c/start", "", map[string]string{"Origin": page}); resp.StatusCode != http.StatusForbidden {
		t.Errorf("cors_origins must not open the dashboard API: %d", resp.StatusCode)
	}

	vr.ws.cfg.mu.Lock()
	vr.ws.cfg.CORSOrigins = []string{"*"}
	vr.ws.cfg.mu.Unlock()
	resp, _ = vr.req(t, "POST", "/v1/chat/completions", chat, map[string]string{"Origin": "http://evil.example"})
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Access-Control-Allow-Origin") != "http://evil.example" {
		t.Errorf("* allows every page: %d %v", resp.StatusCode, resp.Header)
	}
}

func TestSettingsAPIKeysAndFallbackModel(t *testing.T) {
	ws, cfg := newTestServer(t)
	if err := ws.mgr.AddInstance(InstanceConf{Name: "qwen", Model: "/m.gguf", Port: 9301, GPUIDs: []int{0}}); err != nil {
		t.Fatal(err)
	}
	cfg.mu.Lock()
	cfg.Instances = append(cfg.Instances, InstanceConf{Name: "qwen", Model: "/m.gguf", Port: 9301, GPUIDs: []int{0}})
	cfg.mu.Unlock()
	put := func(body string) *httptest.ResponseRecorder {
		rr := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(body))
		ws.ServeHTTP(rr, req)
		return rr
	}
	if rr := put(`{"api_keys":["  sk-a ","","sk-b","sk-a"],"default_model":"qwen"}`); rr.Code != http.StatusOK {
		t.Fatalf("save: %d %s", rr.Code, rr.Body)
	}
	s := cfg.GetSettings()
	if !slices.Equal(*s.APIKeys, []string{"sk-a", "sk-b"}) || *s.DefaultModel != "qwen" || s.CORSOrigins == nil {
		t.Errorf("settings: %+v %q %v", *s.APIKeys, *s.DefaultModel, s.CORSOrigins)
	}
	data, _ := os.ReadFile(cfg.path)
	if !strings.Contains(string(data), "api_keys:") || !strings.Contains(string(data), "default_model: qwen") {
		t.Errorf("not saved:\n%s", data)
	}
	for name, body := range map[string]string{
		"key with a space":  `{"api_keys":["sk a"]}`,
		"unknown model":     `{"default_model":"nope"}`,
		"cors over the API": `{"cors_origins":["*"]}`,
	} {
		if rr := put(body); rr.Code != http.StatusBadRequest {
			t.Errorf("%s: %d %s", name, rr.Code, rr.Body)
		}
	}
	if rr := put(`{"ngl":7}`); rr.Code != http.StatusOK || len(*cfg.GetSettings().APIKeys) != 2 {
		t.Errorf("another setting must keep the keys: %d %v", rr.Code, *cfg.GetSettings().APIKeys)
	}
	if rr := put(`{"api_keys":[],"default_model":"","cors_origins":[]}`); rr.Code != http.StatusOK {
		t.Fatalf("clear: %d %s", rr.Code, rr.Body)
	}
	if s := cfg.GetSettings(); len(*s.APIKeys) != 0 || *s.DefaultModel != "" {
		t.Errorf("not cleared: %+v", s)
	}
}

func TestCORSOriginsCannotBeChangedOverTheAPI(t *testing.T) {
	ws, cfg := newTestServer(t)
	cfg.mu.Lock()
	cfg.CORSOrigins = []string{"http://app.lan:3000"}
	cfg.mu.Unlock()
	if rr := postImport(t, ws, "server_bin: /bin/true\ncors_origins: ['*']\n"); rr.Code != http.StatusForbidden {
		t.Errorf("import widening cors_origins: %d %s", rr.Code, rr.Body)
	}
	if rr := postImport(t, ws, "server_bin: /bin/true\n"); rr.Code != http.StatusForbidden {
		t.Errorf("import dropping cors_origins: %d %s", rr.Code, rr.Body)
	}
	if rr := postImport(t, ws, "server_bin: /bin/true\ncors_origins: ['HTTP://App.lan:3000/']\napi_keys: [sk-x]\n"); rr.Code != http.StatusOK {
		t.Errorf("import with the same origins (and new keys): %d %s", rr.Code, rr.Body)
	}
	if !slices.Equal(cfg.APIKeys, []string{"sk-x"}) {
		t.Errorf("api_keys can be imported: %v", cfg.APIKeys)
	}
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"cors_origins":["http://app.lan:3000","http://x.lan"]}`)))
	if rr.Code == http.StatusOK || len(cfg.CORSOrigins) != 1 {
		t.Errorf("settings changed cors_origins: %d %v", rr.Code, cfg.CORSOrigins)
	}
}

func TestConfigChecksAPIKeysAndOrigins(t *testing.T) {
	for name, y := range map[string]string{
		"bad key":    "server_bin: /bin/true\napi_keys: ['a b']\n",
		"bad origin": "server_bin: /bin/true\ncors_origins: ['app.lan']\n",
		"path":       "server_bin: /bin/true\ncors_origins: ['http://app.lan/x']\n",
		"bad model":  "server_bin: /bin/true\ndefault_model: 'a/b'\n",
	} {
		cfg, err := decodeConfig([]byte(y))
		if err != nil {
			t.Fatal(err)
		}
		if cfg.validate() == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	cfg, _ := decodeConfig([]byte("server_bin: /bin/true\ncors_origins: ['HTTP://App.Lan:3000/', '*']\napi_keys: [' k1 ', k1]\n"))
	if err := cfg.validate(); err != nil || !slices.Equal(cfg.CORSOrigins, []string{"http://app.lan:3000", "*"}) || !slices.Equal(cfg.APIKeys, []string{"k1"}) {
		t.Errorf("normalized: %v %v %v", err, cfg.CORSOrigins, cfg.APIKeys)
	}
}

func TestBuildLaunchSetsAlias(t *testing.T) {
	cfg := launchCfg("metal", 0)
	spec, err := buildLaunch(InstanceConf{Name: "qwen", Model: "/m/a.gguf", Port: 9001, GPUIDs: []int{0}}, cfg)
	if err != nil || !strings.Contains(joined(spec.args), " --alias qwen") {
		t.Errorf("args %q %v", joined(spec.args), err)
	}
	own := []string{"-a", "mine"}
	spec, _ = buildLaunch(InstanceConf{Name: "qwen", Model: "/m/a.gguf", Port: 9001, GPUIDs: []int{0}, ExtraArgs: &own}, cfg)
	if strings.Contains(joined(spec.args), "--alias") {
		t.Errorf("extra_args' own alias must win: %q", joined(spec.args))
	}
}

// TestV1RealtimeProxiesWebSocket: the WebSocket reaches llama-server's
// /v1/realtime with the key taken out, and bytes flow both ways.
func TestV1RealtimeProxiesWebSocket(t *testing.T) {
	upstream := make(chan http.Header, 1)
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if r.URL.Path != "/v1/realtime" || !isWebSocket(r) {
			http.NotFound(w, r)
			return
		}
		upstream <- r.Header.Clone()
		conn, brw, err := http.NewResponseController(w).Hijack()
		if err != nil {
			return
		}
		defer conn.Close()
		// Picks no subprotocol: the manager must answer with one the client
		// offered, or a browser drops the connection.
		brw.WriteString("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
		brw.Flush()
		line, _ := brw.ReadString('\n')
		brw.WriteString("echo " + line)
		brw.Flush()
	})
	setKeys(vr.ws, "sk-rt")
	var respHeaders string
	dial := func(origin, protocols string) (*bufio.Reader, net.Conn, string) {
		conn, err := net.Dial("tcp", strings.TrimPrefix(vr.srv.URL, "http://"))
		if err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(conn, "GET /v1/realtime?model=c HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Protocol: %s\r\n", strings.TrimPrefix(vr.srv.URL, "http://"), protocols)
		if origin != "" {
			fmt.Fprintf(conn, "Origin: %s\r\n", origin)
		}
		fmt.Fprint(conn, "\r\n")
		br := bufio.NewReader(conn)
		status, _ := br.ReadString('\n')
		respHeaders = ""
		for {
			l, err := br.ReadString('\n')
			if err != nil || l == "\r\n" {
				break
			}
			respHeaders += strings.ToLower(l)
		}
		return br, conn, status
	}
	br, conn, status := dial("", "realtime, openai-insecure-api-key.sk-rt")
	defer conn.Close()
	if !strings.Contains(status, "101") {
		t.Fatalf("handshake: %q", status)
	}
	if !strings.Contains(respHeaders, "sec-websocket-protocol: realtime\r\n") {
		t.Errorf("the client must get back a subprotocol it offered (not the key): %q", respHeaders)
	}
	h := <-upstream
	if h.Get("Sec-WebSocket-Protocol") != "realtime" || h.Get("Authorization") != "" {
		t.Errorf("the key reached llama-server: %v", h)
	}
	fmt.Fprint(conn, "hello\n")
	conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	if got, _ := br.ReadString('\n'); got != "echo hello\n" {
		t.Errorf("relay: %q", got)
	}
	_, c2, status := dial("", "realtime, openai-insecure-api-key.wrong")
	c2.Close()
	if !strings.Contains(status, "401") {
		t.Errorf("wrong key: %q", status)
	}
	_, c3, status := dial("http://evil.example", "realtime, openai-insecure-api-key.sk-rt")
	c3.Close()
	if !strings.Contains(status, "403") {
		t.Errorf("a WebSocket from another site: %q", status)
	}
}

func TestBlobCacheExpiresAndBounds(t *testing.T) {
	c := newBlobCache(10, time.Hour)
	c.put("a", []byte("12345"), "x")
	c.put("b", []byte("12345"), "x")
	c.put("c", []byte("1"), "x")
	if _, ok := c.get("a"); ok {
		t.Error("over the size cap the oldest goes")
	}
	if _, ok := c.get("c"); !ok {
		t.Error("newest kept")
	}
	c = newBlobCache(100, time.Millisecond)
	c.put("a", []byte("1"), "x")
	time.Sleep(5 * time.Millisecond)
	if _, ok := c.get("a"); ok {
		t.Error("expired item served")
	}
}

func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(content), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

// fakeFFmpeg replaces ffmpeg with a script that records its arguments and
// writes out (or fails with an empty out).
func fakeFFmpeg(t *testing.T, out string) string {
	t.Helper()
	dir := t.TempDir()
	args := filepath.Join(dir, "args")
	script := "#!/bin/sh\necho \"$@\" > " + args + "\n"
	if out == "" {
		script += "echo broken >&2\nexit 1\n"
	} else {
		script += "printf '%s' '" + out + "'\n"
	}
	bin := writeTemp(t, "ffmpeg", script)
	old := ffmpegPath
	ffmpegPath = func() string { return bin }
	t.Cleanup(func() { ffmpegPath = old })
	return args
}

func noFFmpeg(t *testing.T) {
	old := ffmpegPath
	ffmpegPath = func() string { return "" }
	t.Cleanup(func() { ffmpegPath = old })
}

func readFile(t *testing.T, p string) string {
	t.Helper()
	b, _ := os.ReadFile(p)
	return string(b)
}

var _ = bytes.NewReader

func TestV1Health(t *testing.T) {
	vr := newV1Rig(t, nil)
	addInstance(t, vr.ws, "loading", StateStarting)
	addInstance(t, vr.ws, "off", StateStopped)
	setKeys(vr.ws, "sk-one")
	for _, p := range []string{"/health", "/v1/health", "/health?model=c", "/v1/health?model=C"} {
		resp, b := vr.req(t, "GET", p, "", map[string]string{"Host": "monitor.example.net"})
		if resp.StatusCode != http.StatusOK || strings.TrimSpace(string(b)) != `{"status":"ok"}` {
			t.Errorf("%s (public: no key, any host): %d %s", p, resp.StatusCode, b)
		}
	}
	if resp, _ := vr.req(t, "HEAD", "/health", "", nil); resp.StatusCode != http.StatusOK {
		t.Errorf("HEAD: %d", resp.StatusCode)
	}
	resp, b := vr.req(t, "GET", "/health?model=loading", "", nil)
	wantAPIError(t, "loading model", resp, b, http.StatusServiceUnavailable, "model_not_ready")
	resp, b = vr.req(t, "GET", "/v1/health?model=off", "", nil)
	wantAPIError(t, "stopped model", resp, b, http.StatusNotFound, "model_not_found")
	resp, b = vr.req(t, "GET", "/v1/health?model=nope", "", nil)
	wantAPIError(t, "unknown model", resp, b, http.StatusNotFound, "model_not_found")
	resp, b = vr.req(t, "POST", "/health", "{}", map[string]string{"Authorization": "Bearer sk-one"})
	wantAPIError(t, "POST", resp, b, http.StatusMethodNotAllowed, "method_not_allowed")
	resp, b = vr.req(t, "GET", "/v1/models", "", nil)
	wantAPIError(t, "the rest of /v1 still needs the key", resp, b, http.StatusUnauthorized, "invalid_api_key")
}

func TestUpstreamTimeout(t *testing.T) {
	cfg := &Config{ExtraArgs: []string{"--timeout", "1800"}}
	for _, c := range []struct {
		extra *[]string
		want  time.Duration
	}{
		{nil, 30 * time.Minute}, // the global extra_args
		{&[]string{"-to", "3600"}, time.Hour},
		{&[]string{"--timeout=900"}, 15 * time.Minute},
		{&[]string{"--timeout", "60"}, chatTimeout}, // never below the default
		{&[]string{"-np", "2"}, chatTimeout},
	} {
		inst := NewInstance(InstanceConf{Name: "x", ExtraArgs: c.extra}, cfg)
		if got := inst.upstreamTimeout(); got != c.want {
			t.Errorf("%v: %s, want %s", c.extra, got, c.want)
		}
	}
	if clientWithTimeout(chatTimeout) != chatClient || clientWithTimeout(time.Hour) != clientWithTimeout(time.Hour) || clientWithTimeout(time.Hour) == chatClient {
		t.Error("clients are reused per timeout")
	}
}

// TestV1TimeoutIsAJSON504: a llama-server that never starts answering gets a
// 504 that says how to wait longer.
func TestV1TimeoutIsAJSON504(t *testing.T) {
	old := chatClient
	chatClient = &http.Client{Transport: newChatTransport(150 * time.Millisecond)}
	defer func() { chatClient = old }()
	release := make(chan struct{})
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	})
	t.Cleanup(func() { close(release) })
	resp, b := vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[]}`, nil)
	if e := wantAPIError(t, "silent upstream", resp, b, http.StatusGatewayTimeout, ""); !strings.Contains(e.Error.Message, "--timeout") {
		t.Errorf("should say how to wait longer: %s", b)
	}
}
