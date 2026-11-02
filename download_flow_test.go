package main

import (
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// hfFake serves the HF API (siblings) and resolve endpoints from a map of
// rfilename -> handler.
type hfFake struct {
	siblings []string
	files    map[string]http.HandlerFunc
	api      http.HandlerFunc // optional override for the API call
}

func (f *hfFake) start(t *testing.T) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/models/") {
			if f.api != nil {
				f.api(w, r)
				return
			}
			var b strings.Builder
			b.WriteString(`{"siblings":[`)
			for i, s := range f.siblings {
				if i > 0 {
					b.WriteString(",")
				}
				b.WriteString(`{"rfilename":"` + s + `"}`)
			}
			b.WriteString("]}")
			w.Write([]byte(b.String()))
			return
		}
		if _, rest, ok := strings.Cut(r.URL.Path, "/resolve/main/"); ok {
			if h := f.files[rest]; h != nil {
				h(w, r)
				return
			}
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)
	hermeticCaches(t)
	oldAPI, oldRes := hfAPIBase, hfResolveBase
	hfAPIBase, hfResolveBase = srv.URL+"/api/models", srv.URL
	t.Cleanup(func() { hfAPIBase, hfResolveBase = oldAPI, oldRes })
}

func serveBody(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(body)) }
}

func waitDownload(t *testing.T, dm *DownloadManager) DownloadStatus {
	t.Helper()
	var st DownloadStatus
	waitFor(t, 5*time.Second, "download to finish", func() bool {
		st = dm.GetStatus()
		return st.Status != "" && st.Status != "downloading"
	})
	return st
}

// TestDownloadStartDoesNotBlockStatusDuringResolve (P2): Start held dm.mu over
// the HuggingFace API call, so every status poll (and Stop) hung for as long
// as the API took, up to 20s.
func TestDownloadStartDoesNotBlockStatusDuringResolve(t *testing.T) {
	release := make(chan struct{})
	f := &hfFake{api: func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}}
	f.start(t)
	t.Cleanup(func() { close(release) })

	dm := NewDownloadManager()
	startErr := make(chan error, 1)
	go func() { startErr <- dm.Start("owner/name", "Q4_K_M") }()
	waitFor(t, time.Second, "job published", func() bool { return dm.GetStatus().Active })

	done := make(chan DownloadStatus, 1)
	go func() { done <- dm.GetStatus() }()
	select {
	case st := <-done:
		if st.Status != "downloading" {
			t.Fatalf("status during resolve = %q", st.Status)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("GetStatus blocked while Start waited on the HuggingFace API")
	}

	dm.Stop()
	select {
	case err := <-startErr:
		if err == nil {
			t.Fatal("Start returned nil after being stopped during resolve")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Stop did not abort the HuggingFace API call")
	}
	if st := dm.GetStatus(); st.Status != "stopped" {
		t.Fatalf("status = %q, want stopped", st.Status)
	}
}

func hubSnapshot(repo string) string {
	return filepath.Join(huggingfaceHubDir(), "models--"+strings.ReplaceAll(repo, "/", "--"), "snapshots", "main")
}

func listFiles(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			out = append(out, strings.TrimPrefix(p, root+string(os.PathSeparator)))
		}
		return nil
	})
	return out
}

// TestFailedSplitDownloadLeavesNothingBehind (P2): each part was renamed to its
// final name as soon as it finished, so when a later part failed the earlier
// ones stayed and the Models list offered a model with parts missing.
func TestFailedSplitDownloadLeavesNothingBehind(t *testing.T) {
	f := &hfFake{
		siblings: []string{"Big-Q8_0-00001-of-00002.gguf", "Big-Q8_0-00002-of-00002.gguf"},
		files: map[string]http.HandlerFunc{
			"Big-Q8_0-00001-of-00002.gguf": serveBody("part1"),
			"Big-Q8_0-00002-of-00002.gguf": func(w http.ResponseWriter, r *http.Request) { http.Error(w, "boom", 500) },
		},
	}
	f.start(t)
	dm := NewDownloadManager()
	if err := dm.Start("owner/name", "Q8_0"); err != nil {
		t.Fatal(err)
	}
	if st := waitDownload(t, dm); st.Status != "failed" {
		t.Fatalf("status = %q, want failed", st.Status)
	}
	if left := listFiles(t, hubSnapshot("owner/name")); len(left) != 0 {
		t.Fatalf("files left after a failed split download: %v", left)
	}
	if models := scanHuggingfaceCache(huggingfaceHubDir()); len(models) != 0 {
		t.Fatalf("incomplete model listed: %+v", models)
	}
}

func TestStoppedSplitDownloadLeavesNothingBehind(t *testing.T) {
	f := &hfFake{
		siblings: []string{"Big-Q8_0-00001-of-00002.gguf", "Big-Q8_0-00002-of-00002.gguf"},
		files: map[string]http.HandlerFunc{
			"Big-Q8_0-00001-of-00002.gguf": serveBody("part1"),
			"Big-Q8_0-00002-of-00002.gguf": func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Length", "1000")
				w.Write([]byte("partial"))
				w.(http.Flusher).Flush()
				<-r.Context().Done()
			},
		},
	}
	f.start(t)
	dm := NewDownloadManager()
	if err := dm.Start("owner/name", "Q8_0"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 3*time.Second, "second part in flight", func() bool {
		for _, l := range dm.GetStatus().Logs {
			if strings.Contains(l, "downloading Big-Q8_0-00002") {
				return true
			}
		}
		return false
	})
	dm.Stop()
	if st := waitDownload(t, dm); st.Status != "stopped" {
		t.Fatalf("status = %q, want stopped", st.Status)
	}
	if left := listFiles(t, hubSnapshot("owner/name")); len(left) != 0 {
		t.Fatalf("files left after stopping: %v", left)
	}
}

// TestDownloadSkipsAlreadyCompleteParts (P2): every run re-downloaded every
// file, even ones already on disk.
func TestDownloadSkipsAlreadyCompleteParts(t *testing.T) {
	var part1Hits atomic.Int32
	f := &hfFake{
		siblings: []string{"Big-Q8_0-00001-of-00002.gguf", "Big-Q8_0-00002-of-00002.gguf"},
		files: map[string]http.HandlerFunc{
			"Big-Q8_0-00001-of-00002.gguf": func(w http.ResponseWriter, r *http.Request) { part1Hits.Add(1); w.Write([]byte("new1")) },
			"Big-Q8_0-00002-of-00002.gguf": serveBody("part2"),
		},
	}
	f.start(t)
	mustWriteFile(t, filepath.Join(hubSnapshot("owner/name"), "Big-Q8_0-00001-of-00002.gguf"), 4)

	dm := NewDownloadManager()
	if err := dm.Start("owner/name", "Q8_0"); err != nil {
		t.Fatal(err)
	}
	if st := waitDownload(t, dm); st.Status != "done" {
		t.Fatalf("status = %q, logs %v", st.Status, st.Logs)
	}
	if n := part1Hits.Load(); n != 0 {
		t.Errorf("complete part re-downloaded %d time(s)", n)
	}
	if data, _ := os.ReadFile(filepath.Join(hubSnapshot("owner/name"), "Big-Q8_0-00002-of-00002.gguf")); string(data) != "part2" {
		t.Errorf("part 2 = %q", data)
	}
}

// TestSweepIncompleteRemovesOnlyOurLeftovers (P2): a crash mid-download left a
// multi-GB .incomplete file nobody could see or delete. huggingface_hub's own
// in-progress files in blobs/ must survive the sweep.
func TestSweepIncompleteRemovesOnlyOurLeftovers(t *testing.T) {
	hermeticCaches(t)
	repo := filepath.Join(huggingfaceHubDir(), "models--o--n")
	ours := filepath.Join(repo, "snapshots", "main", "Q4", "M-Q4.gguf.incomplete")
	theirs := filepath.Join(repo, "blobs", "abc123.incomplete")
	model := filepath.Join(repo, "snapshots", "main", "M-Q8_0.gguf")
	for _, p := range []string{ours, theirs, model} {
		mustWriteFile(t, p, 1)
	}
	SweepIncomplete()
	if _, err := os.Stat(ours); !os.IsNotExist(err) {
		t.Error("leftover download temp not removed")
	}
	for _, p := range []string{theirs, model} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("%s removed by the sweep", p)
		}
	}
}

// TestGatedRepoDownloadSendsToken (P2): switching from llama-server -hf to our
// own HTTP client dropped HF_TOKEN, so gated repos failed with 401.
func TestGatedRepoDownloadSendsToken(t *testing.T) {
	t.Setenv("HF_TOKEN", "hf_secret")
	authed := func(h http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Authorization") != "Bearer hf_secret" {
				http.Error(w, "gated", http.StatusUnauthorized)
				return
			}
			h(w, r)
		}
	}
	f := &hfFake{
		siblings: []string{"M-Q4_K_M.gguf"},
		files:    map[string]http.HandlerFunc{"M-Q4_K_M.gguf": authed(serveBody("gguf"))},
	}
	f.api = authed(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"siblings":[{"rfilename":"M-Q4_K_M.gguf"}]}`))
	})
	f.start(t)

	if q, err := FetchQuants("owner/gated"); err != nil || len(q) != 1 {
		t.Fatalf("FetchQuants on a gated repo: %v %v", q, err)
	}
	dm := NewDownloadManager()
	if err := dm.Start("owner/gated", "Q4_K_M"); err != nil {
		t.Fatal(err)
	}
	if st := waitDownload(t, dm); st.Status != "done" {
		t.Fatalf("status = %q, logs %v", st.Status, st.Logs)
	}
}

// TestTokenNotForwardedToCDN: HF answers resolve/ with a redirect to a CDN on
// another domain; the token must not follow it there.
func TestTokenNotForwardedToCDN(t *testing.T) {
	t.Setenv("HF_TOKEN", "hf_secret")
	gotAuth := make(chan string, 1)
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth <- r.Header.Get("Authorization")
		w.Write([]byte("gguf"))
	}))
	defer cdn.Close()
	_, cdnPort, _ := strings.Cut(strings.TrimPrefix(cdn.URL, "http://"), ":")

	f := &hfFake{
		siblings: []string{"M-Q4_K_M.gguf"},
		files: map[string]http.HandlerFunc{"M-Q4_K_M.gguf": func(w http.ResponseWriter, r *http.Request) {
			// Same machine, different domain (localhost vs 127.0.0.1).
			http.Redirect(w, r, "http://localhost:"+cdnPort+"/blob", http.StatusFound)
		}},
	}
	f.start(t)
	dm := NewDownloadManager()
	if err := dm.Start("owner/name", "Q4_K_M"); err != nil {
		t.Fatal(err)
	}
	if st := waitDownload(t, dm); st.Status != "done" {
		t.Fatalf("status = %q, logs %v", st.Status, st.Logs)
	}
	if auth := <-gotAuth; auth != "" {
		t.Fatalf("token forwarded to the CDN: %q", auth)
	}
}

func TestNoAuthorizationWithoutToken(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "")
	req, err := hfRequest(t.Context(), "https://huggingface.co/api/models/a/b")
	if err != nil {
		t.Fatal(err)
	}
	if h := req.Header.Get("Authorization"); h != "" {
		t.Fatalf("Authorization = %q without a token", h)
	}
}

// TestStalledDownloadFails (P2): a peer that kept the connection open but
// stopped sending left the download "downloading" forever.
func TestStalledDownloadFails(t *testing.T) {
	old := downloadIdleTimeout
	downloadIdleTimeout = 200 * time.Millisecond
	defer func() { downloadIdleTimeout = old }()

	f := &hfFake{
		siblings: []string{"M-Q4_K_M.gguf"},
		files: map[string]http.HandlerFunc{"M-Q4_K_M.gguf": func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Length", "1000000")
			for i := 0; i < 3; i++ { // keeps arriving for a while: must not trip the watchdog
				w.Write([]byte(strings.Repeat("x", 100)))
				w.(http.Flusher).Flush()
				time.Sleep(100 * time.Millisecond)
			}
			<-r.Context().Done() // then the peer goes silent
		}},
	}
	f.start(t)
	dm := NewDownloadManager()
	if err := dm.Start("owner/name", "Q4_K_M"); err != nil {
		t.Fatal(err)
	}
	st := waitDownload(t, dm)
	if st.Status != "failed" || !strings.Contains(strings.Join(st.Logs, "\n"), "stalled") {
		t.Fatalf("status = %q, logs %v; want failed with a stall reason", st.Status, st.Logs)
	}
	if left := listFiles(t, hubSnapshot("owner/name")); len(left) != 0 {
		t.Fatalf("files left: %v", left)
	}
}

// TestElapsedStopsWhenDownloadEnds (nit): the status kept counting from the
// start time forever, so a finished download's "elapsed" grew without bound.
func TestElapsedStopsWhenDownloadEnds(t *testing.T) {
	dm := NewDownloadManager()
	job := &DownloadJob{Repo: "o/n", Status: "downloading", Started: time.Now().Add(-10 * time.Minute)}
	dm.active = job
	if got := dm.GetStatus().Elapsed; got != "10m" {
		t.Fatalf("running elapsed = %q, want 10m", got)
	}
	job.finish("done", "ok")
	job.mu.Lock()
	job.ended = job.Started.Add(2 * time.Minute) // finished 8 minutes ago
	job.mu.Unlock()
	if got := dm.GetStatus().Elapsed; got != "2m" {
		t.Fatalf("finished elapsed = %q, want the 2m it ran", got)
	}
}
