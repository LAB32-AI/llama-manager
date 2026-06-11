package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
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

// TestFetchQuantsSplitGGUF guards the regression where multi-part GGUFs made the
// picker list shard counts ("00003") instead of quant names. Mirrors how unsloth
// publishes large models: every quant in its own folder, split across shards,
// with the quant repeated in the filename.
func TestFetchQuantsSplitGGUF(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"siblings":[
			{"rfilename":"UD-Q4_K_XL/Model-Name-120B-UD-Q4_K_XL-00001-of-00003.gguf"},
			{"rfilename":"UD-Q4_K_XL/Model-Name-120B-UD-Q4_K_XL-00002-of-00003.gguf"},
			{"rfilename":"UD-Q4_K_XL/Model-Name-120B-UD-Q4_K_XL-00003-of-00003.gguf"},
			{"rfilename":"Q8_0/Model-Name-120B-Q8_0-00001-of-00004.gguf"},
			{"rfilename":"MXFP4_MOE/Model-Name-120B-MXFP4_MOE-00001-of-00003.gguf"},
			{"rfilename":"Model-Name-120B-IQ4_XS.gguf"},
			{"rfilename":"README.md"}
		]}`))
	}))
	defer srv.Close()

	got, err := fetchQuants("owner/repo", srv.URL)
	if err != nil {
		t.Fatalf("fetchQuants: %v", err)
	}
	want := []string{"IQ4_XS", "MXFP4_MOE", "Q4_K_XL", "Q8_0"}
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
	dm := NewDownloadManager()
	if err := dm.Start("not-a-repo", ""); err == nil {
		t.Error("expected error for invalid repo")
	}
	if err := dm.Start("owner/name", "bad quant"); err == nil {
		t.Error("expected error for invalid quant")
	}
}

const siblingsJSON = `{"siblings":[
	{"rfilename":"README.md"},
	{"rfilename":"Model-Q4_K_M.gguf"},
	{"rfilename":"Model-Q5_K_M.gguf"},
	{"rfilename":"Big-Q8_0-00001-of-00002.gguf"},
	{"rfilename":"Big-Q8_0-00002-of-00002.gguf"}
]}`

func TestResolveQuantFiles(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(siblingsJSON))
	}))
	defer srv.Close()

	got, err := resolveQuantFiles("owner/name", "Q4_K_M", srv.URL)
	if err != nil {
		t.Fatalf("Q4_K_M: %v", err)
	}
	if !reflect.DeepEqual(got, []string{"Model-Q4_K_M.gguf"}) {
		t.Errorf("Q4_K_M = %v", got)
	}

	// Split GGUF: both parts must be returned for the matched quant.
	got, err = resolveQuantFiles("owner/name", "Q8_0", srv.URL)
	if err != nil {
		t.Fatalf("Q8_0: %v", err)
	}
	want := []string{"Big-Q8_0-00001-of-00002.gguf", "Big-Q8_0-00002-of-00002.gguf"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Q8_0 = %v, want %v", got, want)
	}

	if _, err := resolveQuantFiles("owner/name", "NOPE", srv.URL); err == nil {
		t.Error("expected error for missing quant")
	}
}

// TestDownloadEndToEnd drives the full HTTP path: API resolve -> file download
// into the HF hub cache, with no llama.cpp involved.
func TestDownloadEndToEnd(t *testing.T) {
	const body = "GGUF-FAKE-CONTENT"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/api/models/") {
			w.Write([]byte(`{"siblings":[{"rfilename":"Model-Q4_K_M.gguf"}]}`))
			return
		}
		if strings.HasSuffix(r.URL.Path, "/resolve/main/Model-Q4_K_M.gguf") {
			w.Write([]byte(body))
			return
		}
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()

	t.Setenv("HF_HOME", t.TempDir())
	oldAPI, oldRes := hfAPIBase, hfResolveBase
	hfAPIBase, hfResolveBase = srv.URL+"/api/models", srv.URL
	defer func() { hfAPIBase, hfResolveBase = oldAPI, oldRes }()

	dm := NewDownloadManager()
	if err := dm.Start("owner/name", "Q4_K_M"); err != nil {
		t.Fatalf("Start: %v", err)
	}

	deadline := time.Now().Add(5 * time.Second)
	var st DownloadStatus
	for time.Now().Before(deadline) {
		st = dm.GetStatus()
		if st.Status == "done" || st.Status == "failed" {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if st.Status != "done" {
		t.Fatalf("status = %q, logs=%v", st.Status, st.Logs)
	}

	want := filepath.Join(huggingfaceHubDir(), "models--owner--name", "snapshots", "main", "Model-Q4_K_M.gguf")
	data, err := os.ReadFile(want)
	if err != nil {
		t.Fatalf("downloaded file missing: %v", err)
	}
	if string(data) != body {
		t.Errorf("content = %q, want %q", data, body)
	}
}
