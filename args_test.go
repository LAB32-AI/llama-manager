package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestSplitArgs(t *testing.T) {
	cases := map[string][]string{
		``:                 {},
		`-np 1   -ub 2048`: {"-np", "1", "-ub", "2048"},
		`--chat-template-kwargs '{"enable_thinking": false}'`: {"--chat-template-kwargs", `{"enable_thinking": false}`},
		`--override-kv "tokenizer.chat_template=str:a b"`:     {"--override-kv", "tokenizer.chat_template=str:a b"},
		`a\ b "q\"uote" 'it'\''s' ''`:                         {"a b", `q"uote`, "it's", ""},
		"tab\tsep\nnewline":                                   {"tab", "sep", "newline"},
	}
	for in, want := range cases {
		got, err := splitArgs(in)
		if err != nil {
			t.Errorf("splitArgs(%q): %v", in, err)
			continue
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("splitArgs(%q) = %q, want %q", in, got, want)
		}
	}
	for _, bad := range []string{`'open`, `"open`, `trailing\`} {
		if _, err := splitArgs(bad); err == nil {
			t.Errorf("splitArgs(%q) accepted unbalanced input", bad)
		}
	}
}

func TestJoinArgsRoundTrips(t *testing.T) {
	for _, args := range [][]string{
		{"-np", "1", "--cache-reuse", "256"},
		{"--chat-template-kwargs", `{"enable_thinking": false}`},
		{"it's", `back\slash`, `"dq"`, "", "  lead", "tab\there", "ünï"},
		{"--reasoning-budget", "-1"},
	} {
		line := joinArgs(args)
		got, err := splitArgs(line)
		if err != nil {
			t.Fatalf("splitArgs(joinArgs(%q)) = %v (line %q)", args, err, line)
		}
		if !reflect.DeepEqual(got, args) {
			t.Errorf("round trip %q -> %q -> %q", args, line, got)
		}
	}
	if got := joinArgs([]string{"-np", "1", "--cache-reuse", "256"}); got != "-np 1 --cache-reuse 256" {
		t.Errorf("plain args got quoted: %q", got)
	}
}

// TestSettingsSaveKeepsArgsWithSpaces is the P1 #14 regression: the UI showed
// extra_args joined by spaces and saved them split on whitespace, so an
// unrelated settings save turned '{"enable_thinking": false}' into two broken
// arguments.
func TestSettingsSaveKeepsArgsWithSpaces(t *testing.T) {
	ws, cfg := newTestServer(t)
	cfg.ExtraArgs = []string{"--chat-template-kwargs", `{"enable_thinking": false}`, "-np", "1"}

	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/api/settings", nil))
	var s map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &s); err != nil {
		t.Fatal(err)
	}
	line, _ := s["extra_args_line"].(string)

	// What the UI sends back after the user changes only ngl.
	body, _ := json.Marshal(map[string]any{"ngl": 50, "context_length": 2048, "extra_args_line": line})
	rr = httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(string(body))))
	if rr.Code != http.StatusOK {
		t.Fatalf("PUT: %d %s", rr.Code, rr.Body.String())
	}
	if want := []string{"--chat-template-kwargs", `{"enable_thinking": false}`, "-np", "1"}; !reflect.DeepEqual(cfg.ExtraArgs, want) {
		t.Fatalf("extra_args after save = %q, want %q", cfg.ExtraArgs, want)
	}
}

func TestSettingsRejectsUnbalancedQuotes(t *testing.T) {
	ws, cfg := newTestServer(t)
	cfg.ExtraArgs = []string{"-np", "1"}
	body := `{"ngl":50,"context_length":2048,"extra_args_line":"--foo 'unterminated"}`
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(body)))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("got %d, want 400", rr.Code)
	}
	if !reflect.DeepEqual(cfg.ExtraArgs, []string{"-np", "1"}) {
		t.Fatalf("extra_args changed by a rejected save: %q", cfg.ExtraArgs)
	}
}

func TestSettingsStillAcceptsExtraArgsArray(t *testing.T) {
	ws, cfg := newTestServer(t)
	body := `{"ngl":50,"context_length":2048,"extra_args":["--a b","c"]}`
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(body)))
	if rr.Code != http.StatusOK || !reflect.DeepEqual(cfg.ExtraArgs, []string{"--a b", "c"}) {
		t.Fatalf("got %d, extra_args %q", rr.Code, cfg.ExtraArgs)
	}
}
