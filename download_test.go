package main

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestValidateRepo(t *testing.T) {
	cases := []struct {
		repo string
		ok   bool
	}{
		{"owner/name", true},
		{"bartowski/Some-Model_GGUF", true},
		{"a.b/c.d", true},
		{"", false},
		{"no-slash", false},
		{"too/many/slashes", false},
		{"owner/name; rm -rf /", false},
		{"owner/name with space", false},
		{"owner/$evil", false},
		{"../../etc/passwd", false},
		{"owner/" + strings.Repeat("a", 200), false},
	}
	for _, c := range cases {
		err := validateRepo(c.repo)
		if c.ok && err != nil {
			t.Errorf("validateRepo(%q) = %v, want nil", c.repo, err)
		}
		if !c.ok && err == nil {
			t.Errorf("validateRepo(%q) = nil, want error", c.repo)
		}
	}
}

func TestValidateQuant(t *testing.T) {
	cases := []struct {
		quant string
		ok    bool
	}{
		{"", true},
		{"IQ4_XS", true},
		{"Q8_0", true},
		{"Q4_K_M", true},
		{"with space", false},
		{"semi;colon", false},
		{"-leading-dash", false},
		{strings.Repeat("a", 33), false},
	}
	for _, c := range cases {
		err := validateQuant(c.quant)
		if c.ok && err != nil {
			t.Errorf("validateQuant(%q) = %v, want nil", c.quant, err)
		}
		if !c.ok && err == nil {
			t.Errorf("validateQuant(%q) = nil, want error", c.quant)
		}
	}
}

func TestFetchQuantsParsesSiblings(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/owner/repo") {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"siblings":[
			{"rfilename":"model-Q4_K_M.gguf"},
			{"rfilename":"model-Q8_0.gguf"},
			{"rfilename":"model-Q4_K_M.gguf"},
			{"rfilename":"README.md"},
			{"rfilename":"model-IQ4_XS.gguf"}
		]}`))
	}))
	defer srv.Close()

	got, err := fetchQuants("owner/repo", srv.URL)
	if err != nil {
		t.Fatalf("fetchQuants: %v", err)
	}
	want := []string{"IQ4_XS", "Q4_K_M", "Q8_0"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("fetchQuants = %v, want %v", got, want)
	}
}

func TestFetchQuantsRejectsBadRepo(t *testing.T) {
	if _, err := fetchQuants("../../etc/passwd", "https://example.invalid"); err == nil {
		t.Fatal("expected error for traversal repo, got nil")
	}
}

func TestDownloadStartValidatesInput(t *testing.T) {
	dm := NewDownloadManager("/nonexistent/binary")
	if err := dm.Start("not-a-repo", ""); err == nil {
		t.Error("expected error for invalid repo")
	}
	if err := dm.Start("owner/name", "bad quant"); err == nil {
		t.Error("expected error for invalid quant")
	}
}
