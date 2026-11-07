package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func fp(v float64) *float64 { return &v }
func ip(v int) *int         { return &v }

func TestChatParamsFollowTheGeneralSettings(t *testing.T) {
	gen := ChatConf{SystemPrompt: "Answer in Russian.", Temperature: fp(0.4), TopK: ip(20), MaxTokens: 512}

	p := parseChatParams(json.RawMessage(`{"sys":"  Be brief.  ","think":true}`), gen)
	if p.System != "Answer in Russian.\n\nBe brief." {
		t.Errorf("general prompt first, then the chat's: %q", p.System)
	}
	if !reflect.DeepEqual(p.opts, map[string]any{"temperature": 0.4, "top_k": 20, "max_tokens": 512}) {
		t.Errorf("unset values come from the general settings, the rest isn't sent: %v", p.opts)
	}

	p = parseChatParams(json.RawMessage(`{"sys_general":false,"temp":0.9,"top_k":"5","max":0,"top_p":0.8}`), gen)
	if p.System != "" {
		t.Errorf("a chat can leave the general prompt out: %q", p.System)
	}
	if !reflect.DeepEqual(p.opts, map[string]any{"temperature": 0.9, "top_k": 5, "top_p": 0.8}) {
		t.Errorf("the chat's own values win; its max 0 means no limit: %v", p.opts)
	}

	p = parseChatParams(json.RawMessage(`{"sys":"Only mine."}`), ChatConf{})
	if p.System != "Only mine." || len(p.opts) != 0 {
		t.Errorf("no general settings: %q %v", p.System, p.opts)
	}
	if p := parseChatParams(nil, gen); p.System != "Answer in Russian." || !p.Think {
		t.Errorf("a chat with no params: %+v", p)
	}
}

func TestChatConfRules(t *testing.T) {
	for _, c := range []ChatConf{
		{Temperature: fp(5.5)}, {TopP: fp(1.2)}, {MinP: fp(-0.1)}, {RepeatPenalty: fp(9)}, {TopK: ip(-1)},
		{MaxTokens: -1}, {SystemPrompt: strings.Repeat("x", maxSystemPrompt+1)}, {Voice: "a\nb"}, {VoiceLang: "english"},
	} {
		if _, err := c.clean(); err == nil {
			t.Errorf("accepted %+v", c)
		}
	}
	c, err := ChatConf{SystemPrompt: "  hi \n", Temperature: fp(0), TopK: ip(0), VoiceLang: "AUTO", Voice: " tts|me "}.clean()
	if err != nil || c.SystemPrompt != "hi" || c.VoiceLang != "" || c.Voice != "tts|me" || *c.Temperature != 0 {
		t.Errorf("clean: %+v %v", c, err)
	}
}

// The general chat settings are saved in config.yaml through Settings, and a
// rejected value changes nothing.
func TestGeneralChatSettingsOverTheAPI(t *testing.T) {
	ws, cfg := newTestServer(t)
	put := func(body string) *httptest.ResponseRecorder {
		rr := httptest.NewRecorder()
		ws.ServeHTTP(rr, httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(body)))
		return rr
	}
	if rr := put(`{"chat":{"system_prompt":" Be kind. ","temperature":0.6,"top_k":30,"max_tokens":2048,"voice":"ming|f1","voice_lang":"ru"}}`); rr.Code != http.StatusOK {
		t.Fatalf("PUT: %d %s", rr.Code, rr.Body.String())
	}
	want := ChatConf{SystemPrompt: "Be kind.", Temperature: fp(0.6), TopK: ip(30), MaxTokens: 2048, Voice: "ming|f1", VoiceLang: "ru"}
	if !reflect.DeepEqual(cfg.ChatDefaults(), want) {
		t.Fatalf("saved %+v", cfg.ChatDefaults())
	}
	raw, err := os.ReadFile(cfg.path)
	if err != nil {
		t.Fatal(err)
	}
	var onDisk struct {
		Chat ChatConf `yaml:"chat"`
	}
	if err := yaml.Unmarshal(raw, &onDisk); err != nil || !reflect.DeepEqual(onDisk.Chat, want) {
		t.Fatalf("config.yaml chat section: %+v %v\n%s", onDisk.Chat, err, raw)
	}

	if rr := put(`{"chat":{"temperature":7}}`); rr.Code != http.StatusBadRequest {
		t.Errorf("temperature 7: %d", rr.Code)
	}
	if rr := put(`{"ngl":5}`); rr.Code != http.StatusOK || !reflect.DeepEqual(cfg.ChatDefaults(), want) {
		t.Errorf("a save without chat must keep it: %d %+v", rr.Code, cfg.ChatDefaults())
	}

	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/api/settings", nil))
	var got struct {
		Chat ChatConf `json:"chat"`
	}
	if json.Unmarshal(rr.Body.Bytes(), &got) != nil || !reflect.DeepEqual(got.Chat, want) {
		t.Errorf("GET: %s", rr.Body.String())
	}
}

func TestChatRepliesUseTheGeneralChatSettings(t *testing.T) {
	rig := newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if body["stream"] != true {
			titleReply(w, "T")
			return
		}
		sse(w, delta("content", "ok"), finalChunk, "[DONE]")
	})
	rig.ws.cfg.mu.Lock()
	rig.ws.cfg.Chat = ChatConf{SystemPrompt: "Answer in Russian.", Temperature: fp(0.4), MaxTokens: 300}
	rig.ws.cfg.mu.Unlock()

	sent := func(params string) map[string]any {
		t.Helper()
		c := rig.newChat(t, params)
		if resp, b := rig.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "hi"}); resp.StatusCode != http.StatusCreated {
			t.Fatalf("send: %d %s", resp.StatusCode, b)
		}
		rig.waitReply(t, c.ID)
		rig.mu.Lock()
		defer rig.mu.Unlock()
		for i := len(rig.bodies) - 1; i >= 0; i-- {
			if rig.bodies[i]["stream"] == true {
				return rig.bodies[i]
			}
		}
		t.Fatal("no reply request")
		return nil
	}
	system := func(b map[string]any) string {
		m := b["messages"].([]any)[0].(map[string]any)
		if m["role"] != "system" {
			return ""
		}
		return m["content"].(string)
	}

	b := sent(`{"sys":"Be brief."}`)
	if system(b) != "Answer in Russian.\n\nBe brief." || b["temperature"] != 0.4 || b["max_tokens"] != 300.0 {
		t.Errorf("general settings not applied: system %q, temperature %v, max_tokens %v", system(b), b["temperature"], b["max_tokens"])
	}
	if _, ok := b["top_p"]; ok {
		t.Error("top_p is set nowhere, so it must be left to the instance")
	}
	b = sent(`{"sys_general":false,"temp":0.2}`)
	if system(b) != "" || b["temperature"] != 0.2 {
		t.Errorf("the chat's own settings: system %q, temperature %v", system(b), b["temperature"])
	}
}

// Chats from before v7 hold every default the UI used to save; the upgrade
// drops those (and only those), so the chats follow the general settings.
func TestChatStoreV7DropsOldDefaultParams(t *testing.T) {
	path := filepath.Join(t.TempDir(), "chats.db")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range []string{chatSchema, `PRAGMA user_version = 6`,
		`INSERT INTO chats (id, title, params, created_at, updated_at) VALUES ('a', 'a', '{"temp":0.7,"top_p":0.95,"min_p":0.05,"top_k":40,"rep":1,"max":"","seed":"","think":true,"sys":"","size":"1024x1024"}', 1, 1)`,
		`INSERT INTO chats (id, title, params, created_at, updated_at) VALUES ('b', 'b', '{"temp":0.3,"top_p":0.95,"max":0,"seed":1234567890123456789,"sys":"Be brief.","loras":{"img":[{"id":"x","w":0.5}]}}', 1, 1)`,
		`INSERT INTO chats (id, title, params, created_at, updated_at) VALUES ('c', 'c', 'not json', 1, 1)`} {
		if _, err := db.Exec(q); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
	}
	db.Close()
	s, err := OpenChatStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for id, want := range map[string]string{
		"a": `{"size":"1024x1024","think":true}`,
		"b": `{"loras":{"img":[{"id":"x","w":0.5}]},"seed":1234567890123456789,"sys":"Be brief.","temp":0.3}`,
		"c": `not json`,
	} {
		var got string
		s.db.QueryRow(`SELECT params FROM chats WHERE id = ?`, id).Scan(&got)
		if got != want {
			t.Errorf("chat %s params = %s, want %s", id, got, want)
		}
	}
	var v int
	s.db.QueryRow(`PRAGMA user_version`).Scan(&v)
	if v != chatSchemaVersion {
		t.Errorf("user_version = %d", v)
	}
}

func TestPropsReportTheInstanceSampling(t *testing.T) {
	rig := newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		writeJSON(w, http.StatusOK, map[string]any{"default_generation_settings": map[string]any{"n_ctx": 4096,
			"params": map[string]any{"temperature": 0.6, "top_p": 0.95, "min_p": 0.0, "top_k": 20, "repeat_penalty": 1.0, "seed": 42}}})
	})
	p, err := rig.ws.mgr.Get("c").FetchProps(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if want := map[string]float64{"temperature": 0.6, "top_p": 0.95, "min_p": 0, "top_k": 20, "repeat_penalty": 1}; p.NCtx != 4096 || !reflect.DeepEqual(p.Sampling, want) {
		t.Errorf("props: %+v", p)
	}
}
