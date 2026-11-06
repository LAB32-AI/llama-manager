package main

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

// fromInternet are the headers tailscaled adds to a request that came in
// through Funnel.
func fromInternet(ip string, more map[string]string) map[string]string {
	h := map[string]string{funnelHeader: "?1", "X-Forwarded-For": ip, "Host": fakeTailnetHost}
	for k, v := range more {
		h[k] = v
	}
	return h
}

func TestPublicRequestsReachOnlyV1(t *testing.T) {
	vr := newV1Rig(t, nil)
	vr.ws.cfg.mu.Lock()
	vr.ws.cfg.AllowedHosts = []string{"*"}
	vr.ws.cfg.mu.Unlock()
	pub := fromInternet("203.0.113.7", nil)
	for _, p := range []string{"/", "/api/status", "/api/settings", "/chat/completions", "/v1/../api/status", "/v1/./models", "/static/app.js"} {
		resp, b := vr.req(t, "GET", p, "", pub)
		wantAPIError(t, "internet "+p, resp, b, http.StatusNotFound, "")
	}
	if resp, b := vr.req(t, "GET", "/v1/health", "", pub); resp.StatusCode != http.StatusOK {
		t.Errorf("/v1/health stays open: %d %s", resp.StatusCode, b)
	}
	chat := `{"model":"c","messages":[]}`
	// No keys: this network may use the API freely (allowed_hosts "*"), the
	// internet never.
	if resp, b := vr.req(t, "POST", "/v1/chat/completions", chat, map[string]string{"Host": fakeTailnetHost}); resp.StatusCode != http.StatusOK {
		t.Errorf("LAN without keys: %d %s", resp.StatusCode, b)
	}
	resp, b := vr.req(t, "POST", "/v1/chat/completions", chat, pub)
	e := wantAPIError(t, "internet without keys", resp, b, http.StatusUnauthorized, "invalid_api_key")
	if !strings.Contains(e.Error.Message, "none yet") {
		t.Errorf("message: %q", e.Error.Message)
	}

	setKeys(vr.ws, "sk-one")
	vr.ws.cfg.mu.Lock()
	vr.ws.cfg.AllowedHosts = nil
	vr.ws.cfg.mu.Unlock()
	resp, b = vr.req(t, "POST", "/v1/chat/completions", chat, fromInternet("203.0.113.7", map[string]string{"Authorization": "Bearer nope"}))
	wantAPIError(t, "internet, wrong key", resp, b, http.StatusUnauthorized, "invalid_api_key")
	if resp, b := vr.req(t, "POST", "/v1/chat/completions", chat, fromInternet("203.0.113.7", map[string]string{"Authorization": "Bearer sk-one"})); resp.StatusCode != http.StatusOK {
		t.Errorf("internet with a key, a Host not in allowed_hosts: %d %s", resp.StatusCode, b)
	}
	// Without the header the tailnet name is just an unknown Host.
	resp, b = vr.req(t, "POST", "/v1/chat/completions", chat, map[string]string{"Host": fakeTailnetHost})
	wantAPIError(t, "unknown host, no key", resp, b, http.StatusUnauthorized, "invalid_api_key")
	resp, b = vr.req(t, "GET", "/api/status", "", map[string]string{"Host": fakeTailnetHost})
	if resp.StatusCode != http.StatusForbidden {
		t.Errorf("dashboard on an unknown Host: %d %s", resp.StatusCode, b)
	}
}

func TestPublicWrongKeysAreLimitedPerAddress(t *testing.T) {
	vr := newV1Rig(t, nil)
	setKeys(vr.ws, "sk-right")
	chat := `{"model":"c","messages":[]}`
	bad := fromInternet("203.0.113.9", map[string]string{"Authorization": "Bearer sk-guess"})
	for i := 0; i < badKeyLimit; i++ {
		resp, b := vr.req(t, "POST", "/v1/chat/completions", chat, bad)
		wantAPIError(t, fmt.Sprintf("guess %d", i), resp, b, http.StatusUnauthorized, "invalid_api_key")
	}
	resp, b := vr.req(t, "POST", "/v1/chat/completions", chat, fromInternet("203.0.113.9", map[string]string{"Authorization": "Bearer sk-right"}))
	wantAPIError(t, "after too many guesses, even the right key waits", resp, b, http.StatusTooManyRequests, "rate_limit_exceeded")
	if ra, _ := strconv.Atoi(resp.Header.Get("Retry-After")); ra < 1 || ra > 60 {
		t.Errorf("Retry-After = %q", resp.Header.Get("Retry-After"))
	}
	if resp, b := vr.req(t, "POST", "/v1/chat/completions", chat, fromInternet("198.51.100.4", map[string]string{"Authorization": "Bearer sk-right"})); resp.StatusCode != http.StatusOK {
		t.Errorf("another address isn't blocked: %d %s", resp.StatusCode, b)
	}
	if resp, b := vr.req(t, "POST", "/v1/chat/completions", chat, map[string]string{"Authorization": "Bearer sk-right"}); resp.StatusCode != http.StatusOK {
		t.Errorf("the LAN isn't blocked: %d %s", resp.StatusCode, b)
	}
}

func TestClientIPTrustsForwardedOnlyFromFunnel(t *testing.T) {
	for _, c := range []struct {
		remote string
		funnel bool
		want   string
	}{
		{"127.0.0.1:5000", true, "203.0.113.5"},
		{"[::1]:5000", true, "203.0.113.5"},
		{"127.0.0.1:5000", false, "127.0.0.1"},
		{"192.168.1.20:5000", true, "192.168.1.20"},
	} {
		r := httptest.NewRequest("GET", "/v1/models", nil)
		r.RemoteAddr = c.remote
		r.Header.Set("X-Forwarded-For", "10.9.9.9, 203.0.113.5")
		if c.funnel {
			r.Header.Set(funnelHeader, "?1")
		}
		if got := clientIP(r); got != c.want {
			t.Errorf("%s funnel=%v: %s, want %s", c.remote, c.funnel, got, c.want)
		}
	}
}

func TestLimiterRateConcurrencyAndPublicCap(t *testing.T) {
	l := newAPILimiter()
	now := time.Unix(1_800_000_000, 0)
	k := APIKey{Name: "alice", Key: "sk-a", RPM: 2}
	for i := 0; i < 2; i++ {
		rel, e := l.admit(k, false, 0, now)
		if e != nil {
			t.Fatalf("request %d: %s", i, e.msg)
		}
		rel()
	}
	_, e := l.admit(k, false, 0, now)
	if e == nil || e.retry < 29*time.Second || e.retry > 31*time.Second || !strings.Contains(e.msg, "alice") {
		t.Fatalf("third request in a minute at 2 rpm: %+v", e)
	}
	if rel, e := l.admit(k, false, 0, now.Add(30*time.Second)); e != nil {
		t.Errorf("half a minute later one request is back: %s", e.msg)
	} else {
		rel()
	}

	c := APIKey{Key: "sk-c", Concurrent: 1}
	rel, e := l.admit(c, false, 0, now)
	if e != nil {
		t.Fatal(e.msg)
	}
	if _, e := l.admit(c, false, 0, now); e == nil {
		t.Error("a second request at once over concurrent: 1")
	}
	rel()
	rel() // freeing twice must not free someone else's room
	rel2, e := l.admit(c, false, 0, now)
	if e != nil {
		t.Errorf("after the first finished: %s", e.msg)
	}
	if _, e := l.admit(c, false, 0, now); e == nil {
		t.Error("a double release freed an extra slot")
	}
	rel2()

	free := APIKey{Key: "sk-free"}
	p1, e := l.admit(free, true, 1, now)
	if e != nil {
		t.Fatal(e.msg)
	}
	if _, e := l.admit(APIKey{Key: "sk-other"}, true, 1, now); e == nil {
		t.Error("the internet's cap counts every key")
	}
	if rel, e := l.admit(APIKey{Key: "sk-other"}, false, 1, now); e != nil {
		t.Errorf("the LAN isn't under the internet's cap: %s", e.msg)
	} else {
		rel()
	}
	p1()
}

func TestKeyLimitsOverHTTP(t *testing.T) {
	release := make(chan struct{})
	got := make(chan struct{}, 4)
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if body["slow"] == true {
			got <- struct{}{}
			<-release
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true})
	})
	vr.ws.cfg.mu.Lock()
	vr.ws.cfg.APIKeys = []APIKey{{Name: "bob", Key: "sk-bob", RPM: 1}, {Name: "carol", Key: "sk-carol", Concurrent: 1}}
	vr.ws.cfg.mu.Unlock()
	bob := map[string]string{"Authorization": "Bearer sk-bob"}
	if resp, b := vr.req(t, "POST", "/v1/chat/completions", `{"model":"c"}`, bob); resp.StatusCode != http.StatusOK {
		t.Fatalf("first: %d %s", resp.StatusCode, b)
	}
	resp, b := vr.req(t, "POST", "/v1/chat/completions", `{"model":"c"}`, bob)
	e := wantAPIError(t, "second at 1 rpm", resp, b, http.StatusTooManyRequests, "rate_limit_exceeded")
	if e.Error.Type != "rate_limit_error" || resp.Header.Get("Retry-After") == "" {
		t.Errorf("type %q, Retry-After %q", e.Error.Type, resp.Header.Get("Retry-After"))
	}

	carol := map[string]string{"Authorization": "Bearer sk-carol"}
	done := make(chan int, 1)
	go func() {
		resp, _ := vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","slow":true}`, carol)
		done <- resp.StatusCode
	}()
	<-got
	resp, b = vr.req(t, "POST", "/v1/chat/completions", `{"model":"c"}`, carol)
	wantAPIError(t, "carol's second request at once", resp, b, http.StatusTooManyRequests, "rate_limit_exceeded")
	close(release)
	if code := <-done; code != http.StatusOK {
		t.Errorf("the slow request: %d", code)
	}
	if resp, b := vr.req(t, "POST", "/v1/chat/completions", `{"model":"c"}`, carol); resp.StatusCode != http.StatusOK {
		t.Errorf("after it finished: %d %s", resp.StatusCode, b)
	}
}

func TestAPIKeyConfigForms(t *testing.T) {
	cfg, err := decodeConfig([]byte(`server_bin: /bin/true
api_keys:
  - sk-plain
  - name: alice
    key: sk-alice
    rpm: 30
    concurrent: 2
`))
	if err != nil || cfg.validate() != nil {
		t.Fatalf("%v %v", err, cfg.validate())
	}
	want := []APIKey{{Key: "sk-plain"}, {Name: "alice", Key: "sk-alice", RPM: 30, Concurrent: 2}}
	if fmt.Sprint(cfg.APIKeys) != fmt.Sprint(want) {
		t.Errorf("keys: %+v", cfg.APIKeys)
	}
	out, _ := yaml.Marshal(cfg)
	if !strings.Contains(string(out), "    - sk-plain\n") || !strings.Contains(string(out), "name: alice") {
		t.Errorf("a plain key stays a plain string, a named one a mapping:\n%s", out)
	}
	back, err := decodeConfig(out)
	if err != nil || fmt.Sprint(back.APIKeys) != fmt.Sprint(want) {
		t.Errorf("round trip: %v %+v", err, back.APIKeys)
	}
	if k := (APIKey{Key: "sk-0123456789"}); k.label() != "…6789" {
		t.Errorf("label of an unnamed key: %q", k.label())
	}

	for name, y := range map[string]string{
		"typo":       "api_keys:\n  - {key: sk-a, rmp: 30}\n",
		"same name":  "api_keys:\n  - {name: Al, key: sk-a}\n  - {name: al, key: sk-b}\n",
		"negative":   "api_keys:\n  - {key: sk-a, rpm: -1}\n",
		"too many":   "api_keys:\n  - {key: sk-a, concurrent: 5000}\n",
		"two lines":  "api_keys:\n  - {name: \"a\\nb\", key: sk-a}\n",
		"space":      "api_keys:\n  - {key: 'sk a'}\n",
		"https port": "public_api:\n  tailscale_funnel: true\n  https_port: 8080\n",
		"cap":        "public_api:\n  max_concurrent: -1\n",
	} {
		cfg, err := decodeConfig([]byte("server_bin: /bin/true\n" + y))
		if err == nil {
			err = cfg.validate()
		}
		if err == nil {
			t.Errorf("%s: accepted", name)
		}
	}

	ws, c := newTestServer(t)
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"api_keys":["sk-old",{"name":"dave","key":" sk-dave ","rpm":60,"concurrent":2}]}`)))
	if rr.Code != http.StatusOK {
		t.Fatalf("settings: %d %s", rr.Code, rr.Body)
	}
	if fmt.Sprint(c.APIKeys) != fmt.Sprint([]APIKey{{Key: "sk-old"}, {Name: "dave", Key: "sk-dave", RPM: 60, Concurrent: 2}}) {
		t.Errorf("keys from settings: %+v", c.APIKeys)
	}
	var s Settings
	_ = json.Unmarshal(rr.Body.Bytes(), &s)
	if s.APIUsage == nil || len(s.APIUsage) != 2 {
		t.Errorf("settings report each key's usage: %s", rr.Body)
	}
}

func TestAPIUsageIsCountedPerKey(t *testing.T) {
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
			fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":5,\"total_tokens\":16}}\n\ndata: [DONE]\n\n")
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": `say "prompt_tokens": 999`}}},
			"usage": map[string]any{"prompt_tokens": 7, "completion_tokens": 3, "prompt_tokens_details": map[string]any{"cached_tokens": 1}}})
	})
	vr.ws.cfg.mu.Lock()
	vr.ws.cfg.APIKeys = []APIKey{{Name: "alice", Key: "sk-alice"}, {Name: "bob", Key: "sk-bob"}}
	vr.ws.cfg.mu.Unlock()
	alice := map[string]string{"Authorization": "Bearer sk-alice"}
	for i := 0; i < 2; i++ {
		if resp, b := vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[]}`, alice); resp.StatusCode != http.StatusOK {
			t.Fatalf("%d %s", resp.StatusCode, b)
		}
	}
	if resp, b := vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","stream":true,"messages":[]}`, alice); resp.StatusCode != http.StatusOK {
		t.Fatalf("stream: %d %s", resp.StatusCode, b)
	}
	vr.req(t, "POST", "/v1/chat/completions", `{"model":"c"}`, map[string]string{"Authorization": "Bearer sk-wrong"})
	vr.req(t, "POST", "/v1/responses/input_tokens", `{"model":"c","input":"x"}`, map[string]string{"Authorization": "Bearer sk-bob"})

	check := func(when string) {
		t.Helper()
		_, b := vr.do(t, "GET", "/api/settings", nil)
		var s Settings
		if err := json.Unmarshal(b, &s); err != nil || len(s.APIUsage) != 2 {
			t.Fatalf("%s: %v %s", when, err, b)
		}
		a := s.APIUsage[0]
		if a.Today != (usageCount{Requests: 3, InputTokens: 7 + 7 + 11, OutputTokens: 3 + 3 + 5}) || a.Month != a.Today || a.LastUsed == 0 {
			t.Errorf("%s: alice %+v", when, a)
		}
		if b := s.APIUsage[1]; b.Today.Requests != 1 || b.Today.InputTokens != 0 || b.Today.OutputTokens != 0 {
			t.Errorf("%s: bob only counted tokens, which aren't used: %+v", when, b)
		}
	}
	check("in memory")
	vr.ws.usage.flush()
	check("written")
	var rows int
	_ = vr.ws.chats.store.db.QueryRow(`SELECT COUNT(*) FROM api_usage WHERE key_id = ?`, keyID("sk-alice")).Scan(&rows)
	if rows != 1 {
		t.Errorf("rows for alice: %d", rows)
	}
	alice2 := map[string]string{"Authorization": "Bearer sk-alice"}
	vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[]}`, alice2)
	vr.ws.usage.flush()
	_, b := vr.do(t, "GET", "/api/settings", nil)
	var s Settings
	_ = json.Unmarshal(b, &s)
	if s.APIUsage[0].Today.Requests != 4 {
		t.Errorf("a second flush adds to the day's row: %+v", s.APIUsage[0])
	}
}

func TestMeterFindsCountsSplitAcrossWrites(t *testing.T) {
	rr := httptest.NewRecorder()
	m := &meterWriter{ResponseWriter: rr}
	m.Header().Set("Content-Type", "application/json")
	for _, part := range []string{`{"content":"\"output_tokens\": 5","usage":{"input_tok`, `ens":1`, `23,"output_tokens":4`, `5}}`} {
		m.Write([]byte(part))
	}
	if m.in != 123 || m.out != 45 || m.status != 200 {
		t.Errorf("in %d out %d status %d", m.in, m.out, m.status)
	}
	m2 := &meterWriter{ResponseWriter: httptest.NewRecorder()}
	m2.Header().Set("Content-Type", "audio/wav")
	m2.Write([]byte(`"prompt_tokens": 5`))
	if m2.in != 0 {
		t.Error("audio isn't read for counts")
	}
}

func TestClientSecretWorksOnlyWhileItsKeyIsSet(t *testing.T) {
	vr := newV1Rig(t, nil)
	setKeys(vr.ws, "sk-maker", "sk-other")
	resp, b := vr.req(t, "POST", "/v1/realtime/client_secrets", `{}`, map[string]string{"Authorization": "Bearer sk-maker"})
	var out struct {
		Value string `json:"value"`
	}
	if resp.StatusCode != http.StatusOK || json.Unmarshal(b, &out) != nil || !strings.HasPrefix(out.Value, "ek_") {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	ek := map[string]string{"Authorization": "Bearer " + out.Value}
	resp, b = vr.req(t, "GET", "/v1/realtime", "", ek)
	wantAPIError(t, "the secret gets past the key check", resp, b, http.StatusBadRequest, "")
	setKeys(vr.ws, "sk-other")
	resp, b = vr.req(t, "GET", "/v1/realtime", "", ek)
	wantAPIError(t, "its key was removed", resp, b, http.StatusUnauthorized, "invalid_api_key")
}

func TestPublicAPICannotBeChangedOverTheAPI(t *testing.T) {
	ws, cfg := newTestServer(t)
	cfg.mu.Lock()
	cfg.PublicAPI = PublicAPIConf{MaxConcurrent: 2}
	cfg.mu.Unlock()
	for name, y := range map[string]string{
		"turn on":  "server_bin: /bin/true\npublic_api:\n  tailscale_funnel: true\n  max_concurrent: 2\n",
		"raise":    "server_bin: /bin/true\npublic_api:\n  max_concurrent: 50\n",
		"drop":     "server_bin: /bin/true\n",
		"swap bin": "server_bin: /bin/true\npublic_api:\n  max_concurrent: 2\n  tailscale_bin: /tmp/evil\n",
	} {
		if rr := postImport(t, ws, y); rr.Code != http.StatusForbidden {
			t.Errorf("import %s: %d %s", name, rr.Code, rr.Body)
		}
	}
	if rr := postImport(t, ws, "server_bin: /bin/true\npublic_api:\n  max_concurrent: 2\napi_keys: [sk-x]\n"); rr.Code != http.StatusOK {
		t.Errorf("import with the same section: %d %s", rr.Code, rr.Body)
	}
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"public_api":{"tailscale_funnel":true,"tailscale_bin":"/tmp/evil"}}`)))
	if cfg.PublicAPI != (PublicAPIConf{MaxConcurrent: 2}) {
		t.Errorf("settings changed public_api: %d %+v", rr.Code, cfg.PublicAPI)
	}
}

// funnelRig runs the funnel supervisor against the fake tailscale.
type funnelRig struct {
	f    *funnel
	dir  string
	keys atomic.Bool
}

func newFunnelRig(t *testing.T, mode string, port int) *funnelRig {
	t.Helper()
	for _, v := range []struct {
		p *time.Duration
		d time.Duration
	}{{&funnelKeyPoll, 30 * time.Millisecond}, {&funnelConfirmAfter, 300 * time.Millisecond}, {&funnelRetryFirst, 50 * time.Millisecond}, {&funnelRetryMax, 100 * time.Millisecond}, {&funnelStopWait, 2 * time.Second}} {
		old := *v.p
		*v.p = v.d
		t.Cleanup(func() { *v.p = old })
	}
	rig := &funnelRig{dir: t.TempDir()}
	t.Setenv(fakeTailscaleMode, mode)
	t.Setenv(fakeTailscaleDir, rig.dir)
	rig.keys.Store(true)
	rig.f = newFunnel(PublicAPIConf{TailscaleFunnel: true, TailscaleBin: os.Args[0], HTTPSPort: port}, 8099, rig.keys.Load)
	t.Cleanup(rig.f.Close)
	return rig
}

func (rig *funnelRig) runs() []string {
	b, _ := os.ReadFile(filepath.Join(rig.dir, "args"))
	return strings.Fields(strings.ReplaceAll(strings.TrimSpace(string(b)), " ", "_"))
}

func (rig *funnelRig) stopped() bool {
	_, err := os.Stat(filepath.Join(rig.dir, "stopped"))
	return err == nil
}

func (rig *funnelRig) waitState(t *testing.T, state string) *funnelStatus {
	t.Helper()
	var st *funnelStatus
	waitFor(t, 5*time.Second, "funnel state "+state, func() bool { st = rig.f.Status(); return st.State == state })
	return st
}

func TestFunnelRunsAndStopsTheCLI(t *testing.T) {
	rig := newFunnelRig(t, "ok", 0)
	rig.f.Start()
	st := rig.waitState(t, funnelOn)
	if st.URL != "https://"+fakeTailnetHost+"/v1" || st.Port != 443 {
		t.Errorf("status: %+v", st)
	}
	if runs := rig.runs(); len(runs) != 1 || runs[0] != "funnel_--https=443_--set-path_/v1_http://127.0.0.1:8099/v1" {
		t.Errorf("command: %q", runs)
	}
	rig.f.Close()
	if !rig.stopped() {
		t.Error("closing must stop the CLI (that takes /v1 off the internet)")
	}
}

func TestFunnelWaitsForAKey(t *testing.T) {
	rig := newFunnelRig(t, "ok", 8443)
	rig.keys.Store(false)
	rig.f.Start()
	rig.waitState(t, funnelNoKey)
	time.Sleep(100 * time.Millisecond)
	if len(rig.runs()) != 0 {
		t.Fatal("tailscale ran without a key")
	}
	rig.keys.Store(true)
	st := rig.waitState(t, funnelOn)
	if !strings.HasSuffix(rig.runs()[0], "--https=8443_--set-path_/v1_http://127.0.0.1:8099/v1") {
		t.Errorf("command: %q", rig.runs())
	}
	_ = st
	rig.keys.Store(false)
	rig.waitState(t, funnelNoKey)
	waitFor(t, 3*time.Second, "CLI stopped", rig.stopped)
}

func TestFunnelReportsWhatTheCLISays(t *testing.T) {
	t.Run("approval", func(t *testing.T) {
		rig := newFunnelRig(t, "approval", 0)
		rig.f.Start()
		if st := rig.waitState(t, funnelApproval); st.ApproveURL != "https://login.tailscale.com/f/funnel?node=fakeNode" {
			t.Errorf("%+v", st)
		}
	})
	t.Run("denied", func(t *testing.T) {
		rig := newFunnelRig(t, "denied", 0)
		rig.f.Start()
		st := rig.waitState(t, funnelError)
		if !strings.Contains(st.Error, "Access denied") || !strings.Contains(st.Error, "sudo tailscale set --operator=") {
			t.Errorf("%+v", st)
		}
	})
	t.Run("quiet", func(t *testing.T) {
		rig := newFunnelRig(t, "quiet", 10000)
		rig.f.Start()
		if st := rig.waitState(t, funnelOn); st.URL != "https://"+fakeTailnetHost+":10000/v1" {
			t.Errorf("%+v", st)
		}
	})
	t.Run("killed", func(t *testing.T) {
		rig := newFunnelRig(t, "ok", 0)
		rig.f.Start()
		rig.waitState(t, funnelOn)
		b, err := os.ReadFile(filepath.Join(rig.dir, "pid"))
		pid, _ := strconv.Atoi(strings.TrimSpace(string(b)))
		if err != nil || pid <= 0 {
			t.Fatalf("pid file: %v %q", err, b)
		}
		_ = syscall.Kill(pid, syscall.SIGKILL)
		st := rig.waitState(t, funnelError)
		if !strings.Contains(st.Error, "killed") || strings.Contains(st.Error, "Ctrl+C") {
			t.Errorf("why it stopped: %+v", st)
		}
		rig.waitState(t, funnelOn)
	})
	t.Run("crash", func(t *testing.T) {
		rig := newFunnelRig(t, "crash", 0)
		rig.f.Start()
		if st := rig.waitState(t, funnelError); !strings.Contains(st.Error, "boom") {
			t.Errorf("%+v", st)
		}
		waitFor(t, 3*time.Second, "a retry", func() bool { return len(rig.runs()) >= 2 })
	})
	t.Run("missing", func(t *testing.T) {
		rig := newFunnelRig(t, "ok", 0)
		rig.f.bin = filepath.Join(t.TempDir(), "no-tailscale")
		rig.f.Start()
		if st := rig.waitState(t, funnelError); !strings.Contains(st.Error, "no-tailscale") {
			t.Errorf("%+v", st)
		}
	})
}

// TestRunOpensFunnelAndClosesIt drives the real entrypoint: Funnel starts
// once the manager serves, shows in /api/status, and stops on shutdown.
func TestRunOpensFunnelAndClosesIt(t *testing.T) {
	rig := newFunnelRig(t, "ok", 0)
	path, _ := writeRunConfig(t, freePort(t))
	b, _ := os.ReadFile(path)
	b = append(b, []byte(fmt.Sprintf("api_keys: [sk-run]\npublic_api:\n  tailscale_funnel: true\n  tailscale_bin: %s\n", os.Args[0]))...)
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	stop := make(chan os.Signal, 1)
	addrCh := make(chan string, 1)
	done := make(chan error, 1)
	go func() { done <- run(path, stop, func(a string) { addrCh <- a }) }()
	var addr string
	select {
	case addr = <-addrCh:
	case err := <-done:
		t.Fatalf("run exited early: %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("run never became ready")
	}
	_, port, _ := net.SplitHostPort(addr)
	var st ServerStatus
	waitFor(t, 5*time.Second, "public_api on in /api/status", func() bool {
		resp, err := http.Get("http://127.0.0.1:" + port + "/api/status")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		st = ServerStatus{}
		_ = json.NewDecoder(resp.Body).Decode(&st)
		return st.PublicAPI != nil && st.PublicAPI.State == funnelOn
	})
	if st.PublicAPI.URL != "https://"+fakeTailnetHost+"/v1" {
		t.Errorf("%+v", st.PublicAPI)
	}
	if runs := rig.runs(); len(runs) != 1 || !strings.HasSuffix(runs[0], "http://127.0.0.1:"+port+"/v1") {
		t.Errorf("funnel points at the manager's port: %q", runs)
	}
	stop <- syscall.SIGTERM
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("run: %v", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("run did not return after the signal")
	}
	if !rig.stopped() {
		t.Error("shutdown must stop tailscale funnel")
	}
}
