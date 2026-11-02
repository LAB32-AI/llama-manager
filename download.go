package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// HuggingFace's own repo-id rules: each part starts and ends with [A-Za-z0-9_],
// may contain . _ - in between, and never contains ".." or "--". Quants are
// tags like Q4_K_M or UD-Q4_K_XL (no leading dash, so never a flag).
var (
	repoRe  = regexp.MustCompile(`^[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?/[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?$`)
	quantRe = regexp.MustCompile(`^[A-Za-z0-9_][A-Za-z0-9_-]{0,31}$`)
)

// downloadIdleTimeout aborts a download whose connection stops delivering
// data. Var so tests can shorten it.
var downloadIdleTimeout = 60 * time.Second

var errDownloadStalled = errors.New("download stalled")

// HuggingFace endpoints. Vars (not consts) so tests can point them at httptest.
var (
	hfAPIBase     = "https://huggingface.co/api/models"
	hfResolveBase = "https://huggingface.co"
)

func validateRepo(repo string) error {
	if !repoRe.MatchString(repo) || strings.Contains(repo, "..") || strings.Contains(repo, "--") {
		return fmt.Errorf("invalid repo %q: must be owner/name using [A-Za-z0-9._-]", repo)
	}
	return nil
}

func validateQuant(quant string) error {
	if quant == "" {
		return nil
	}
	if !quantRe.MatchString(quant) {
		return fmt.Errorf("invalid quant %q: must be 1-32 of [A-Za-z0-9_-], not starting with -", quant)
	}
	return nil
}

// isLocalModel reports whether an instance's model is a file path (passed to
// llama-server with -m) rather than a HuggingFace reference (passed with -hf).
func isLocalModel(m string) bool {
	return strings.HasPrefix(m, "/") || strings.HasSuffix(m, ".gguf")
}

// validateModelRef checks an instance's model. HuggingFace references
// ("owner/repo" or "owner/repo:QUANT") go through validateRepo/validateQuant
// like every other HF input before they reach exec.Command.
func validateModelRef(m string) error {
	if isLocalModel(m) {
		return nil
	}
	repo, quant, _ := strings.Cut(m, ":")
	if err := validateRepo(repo); err != nil {
		return fmt.Errorf("model must be a .gguf path or owner/repo[:QUANT]: %w", err)
	}
	if err := validateQuant(quant); err != nil {
		return fmt.Errorf("model must be a .gguf path or owner/repo[:QUANT]: %w", err)
	}
	return nil
}

type DownloadManager struct {
	mu     sync.Mutex
	active *DownloadJob
}

type DownloadJob struct {
	Repo    string    `json:"repo"`
	Quant   string    `json:"quant"`
	Status  string    `json:"status"` // "downloading", "done", "failed", "stopped"
	Logs    []string  `json:"logs"`
	Started time.Time `json:"started"`
	ended   time.Time // set when the job reaches a final status
	cancel  context.CancelFunc
	mu      sync.Mutex
}

type DownloadStatus struct {
	Active  bool     `json:"active"`
	Repo    string   `json:"repo,omitempty"`
	Quant   string   `json:"quant,omitempty"`
	Status  string   `json:"status,omitempty"`
	Logs    []string `json:"logs,omitempty"`
	Elapsed string   `json:"elapsed,omitempty"`
}

func NewDownloadManager() *DownloadManager { return &DownloadManager{} }

// Start resolves the .gguf file(s) for repo:quant on HuggingFace and downloads
// them over HTTP into the HF hub cache (models--<org>--<repo>/snapshots/main/),
// where the Models tab scanner already looks. Downloading ourselves — instead
// of shelling to `llama-server -hf` — means it works even when llama.cpp was
// built without TLS support.
func (dm *DownloadManager) Start(repo, quant string) error {
	if err := validateRepo(repo); err != nil {
		return err
	}
	if err := validateQuant(quant); err != nil {
		return err
	}

	// Publish the job before talking to HuggingFace and don't hold dm.mu over
	// the network call: status polling and Stop must stay responsive.
	dm.mu.Lock()
	if job := dm.active; job != nil && job.status() == "downloading" {
		dm.mu.Unlock()
		return fmt.Errorf("download already in progress: %s:%s", job.Repo, job.Quant)
	}
	ctx, cancel := context.WithCancel(context.Background())
	job := &DownloadJob{
		Repo:    repo,
		Quant:   quant,
		Status:  "downloading",
		Started: time.Now(),
		cancel:  cancel,
	}
	dm.active = job
	dm.mu.Unlock()

	model := repo
	if quant != "" {
		model += ":" + quant
	}
	job.log("resolving " + model)
	files, err := resolveQuantFiles(ctx, repo, quant, hfAPIBase)
	if err != nil {
		if ctx.Err() != nil {
			job.finish("stopped", "download stopped by user")
			return fmt.Errorf("download stopped")
		}
		job.finish("failed", err.Error())
		return err
	}
	log.Printf("[download] started: %s (%d file(s))", model, len(files))
	job.log(fmt.Sprintf("%s → %d file(s)", model, len(files)))

	go dm.run(ctx, job, repo, files)
	return nil
}

func (dm *DownloadManager) run(ctx context.Context, job *DownloadJob, repo string, files []string) {
	destDir := filepath.Join(huggingfaceHubDir(), "models--"+strings.ReplaceAll(repo, "/", "--"), "snapshots", "main")
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		job.finish("failed", "creating cache dir: "+err.Error())
		return
	}

	type part struct{ name, dest, tmp string }
	var todo []part
	for _, f := range files {
		dest := filepath.Join(destDir, filepath.FromSlash(f))
		if !strings.HasPrefix(filepath.Clean(dest), filepath.Clean(destDir)+string(os.PathSeparator)) {
			job.finish("failed", "unsafe file path from repo: "+f)
			return
		}
		// Final names only ever appear once a whole model is complete (see
		// below), so an existing file is a finished one.
		if _, err := os.Stat(dest); err == nil {
			job.log("already downloaded: " + f)
			continue
		}
		todo = append(todo, part{name: f, dest: dest, tmp: dest + ".incomplete"})
	}

	var fetched []part
	fail := func(status, msg string) {
		for _, p := range fetched {
			os.Remove(p.tmp)
		}
		job.finish(status, msg)
	}
	for _, p := range todo {
		if ctx.Err() != nil {
			fail("stopped", "download stopped by user")
			return
		}
		if err := os.MkdirAll(filepath.Dir(p.dest), 0o755); err != nil {
			fail("failed", "creating dir: "+err.Error())
			return
		}
		urlStr := hfResolveBase + "/" + repo + "/resolve/main/" + fileURLPath(p.name)
		job.log("downloading " + p.name)
		if err := downloadToFile(ctx, urlStr, p.tmp, job); err != nil {
			os.Remove(p.tmp)
			if ctx.Err() != nil {
				fail("stopped", "download stopped by user")
			} else {
				fail("failed", err.Error())
			}
			return
		}
		fetched = append(fetched, p)
		job.log("fetched " + p.name)
	}

	// Rename only after every part arrived: a split model must not show up
	// in the Models list (and get picked for an instance) with parts missing.
	for i, p := range fetched {
		if err := os.Rename(p.tmp, p.dest); err != nil {
			for _, done := range fetched[:i] {
				os.Remove(done.dest)
			}
			fail("failed", "saving "+p.name+": "+err.Error())
			return
		}
	}
	job.finish("done", "download complete")
}

// SweepIncomplete removes temp files left in the hub cache by downloads that
// were interrupted by a crash or restart; they are invisible in the UI and
// can't be deleted from it. Only snapshots/ is swept: that is where this
// downloader writes, while huggingface_hub keeps its own in-progress files in
// blobs/. Call it before any download can be running.
func SweepIncomplete() {
	repos, _ := filepath.Glob(filepath.Join(huggingfaceHubDir(), "models--*", "snapshots"))
	for _, snaps := range repos {
		_ = filepath.WalkDir(snaps, func(p string, d fs.DirEntry, err error) error {
			if err == nil && !d.IsDir() && strings.HasSuffix(d.Name(), ".incomplete") {
				if os.Remove(p) == nil {
					log.Printf("[download] removed leftover %s", p)
				}
			}
			return nil
		})
	}
}

func (dm *DownloadManager) Stop() {
	dm.mu.Lock()
	job := dm.active
	dm.mu.Unlock()
	if job == nil {
		return
	}
	job.mu.Lock()
	cancel := job.cancel
	job.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	log.Printf("[download] stop requested")
}

func (dm *DownloadManager) GetStatus() DownloadStatus {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	if dm.active == nil {
		return DownloadStatus{Active: false}
	}

	dm.active.mu.Lock()
	defer dm.active.mu.Unlock()

	logs := make([]string, len(dm.active.Logs))
	copy(logs, dm.active.Logs)

	return DownloadStatus{
		Active:  dm.active.Status == "downloading",
		Repo:    dm.active.Repo,
		Quant:   dm.active.Quant,
		Status:  dm.active.Status,
		Logs:    logs,
		Elapsed: formatDuration(dm.active.elapsedLocked()),
	}
}

// resolveQuantFiles returns the .gguf rfilenames in repo whose quant token (the
// trailing -SEGMENT before .gguf, ignoring any -NNNNN-of-NNNNN split suffix)
// equals quant — multiple are returned for split GGUFs. If quant is empty and
// the repo has exactly one .gguf, that file is returned.
func resolveQuantFiles(ctx context.Context, repo, quant, apiBase string) ([]string, error) {
	if err := validateRepo(repo); err != nil {
		return nil, err
	}
	owner, name, _ := strings.Cut(repo, "/")
	endpoint := fmt.Sprintf("%s/%s/%s", apiBase, url.PathEscape(owner), url.PathEscape(name))
	req, err := hfRequest(ctx, endpoint)
	if err != nil {
		return nil, err
	}
	client := &http.Client{Timeout: 20 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching repo info: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HuggingFace API returned %d", resp.StatusCode)
	}

	var result struct {
		Siblings []struct {
			RFilename string `json:"rfilename"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	// Match on the same token the quant picker shows, so what's listed and
	// what's downloaded can't disagree.
	var all, matched []string
	for _, s := range result.Siblings {
		f := s.RFilename
		if !strings.HasSuffix(f, ".gguf") || isProjector(f) {
			continue
		}
		all = append(all, f)
		bare := strings.TrimSuffix(modelPartRe.ReplaceAllString(path.Base(f), ""), ".gguf")
		if quant != "" && (quantToken(f) == quant || bare == quant) {
			matched = append(matched, f)
		}
	}

	if quant == "" {
		if len(all) == 1 {
			return all, nil
		}
		return nil, fmt.Errorf("repo has %d gguf files; specify a quant", len(all))
	}
	if len(matched) == 0 {
		return nil, fmt.Errorf("no .gguf file for quant %q in %s", quant, repo)
	}
	sort.Strings(matched)
	return matched, nil
}

// downloadToFile streams urlStr into path, honoring ctx cancellation and
// logging progress to the job. The caller decides when the file becomes
// visible. Go's net/http handles the HTTPS/CDN redirect itself, so no TLS
// support in llama.cpp is needed.
func downloadToFile(ctx context.Context, urlStr, path string, job *DownloadJob) error {
	// No overall timeout (a model can take hours), but abort if the
	// connection stops delivering data, instead of hanging forever.
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	idle := time.AfterFunc(downloadIdleTimeout, func() { cancel(errDownloadStalled) })
	defer idle.Stop()
	stalled := func(err error) error {
		if errors.Is(context.Cause(ctx), errDownloadStalled) {
			return fmt.Errorf("%w: no data for %s", errDownloadStalled, downloadIdleTimeout)
		}
		return err
	}

	req, err := hfRequest(ctx, urlStr)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return stalled(err)
	}
	defer resp.Body.Close()
	name := strings.TrimSuffix(filepath.Base(path), ".incomplete")
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d fetching %s", resp.StatusCode, name)
	}

	out, err := os.Create(path)
	if err != nil {
		return err
	}
	pw := &progressWriter{job: job, total: resp.ContentLength, name: name, onData: func() { idle.Reset(downloadIdleTimeout) }}
	_, err = io.Copy(out, io.TeeReader(resp.Body, pw))
	err = stalled(err)
	if err == nil {
		err = out.Sync() // durable before the rename that makes it visible
	}
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	return err
}

// progressWriter logs a download progress line at most every 2s.
type progressWriter struct {
	job    *DownloadJob
	total  int64
	name   string
	done   int64
	last   time.Time
	onData func()
}

func (p *progressWriter) Write(b []byte) (int, error) {
	if p.onData != nil {
		p.onData()
	}
	p.done += int64(len(b))
	if time.Since(p.last) >= 2*time.Second {
		p.last = time.Now()
		if p.total > 0 {
			p.job.log(fmt.Sprintf("%s: %d/%d MB (%d%%)", p.name, p.done>>20, p.total>>20, 100*p.done/p.total))
		} else {
			p.job.log(fmt.Sprintf("%s: %d MB", p.name, p.done>>20))
		}
	}
	return len(b), nil
}

// hfRequest builds a GET to HuggingFace carrying the user's token, if any, so
// gated and private repos work (llama-server -hf read HF_TOKEN the same way).
// net/http drops Authorization when HF redirects to its CDN on another
// domain, so the token never leaves huggingface.co.
func hfRequest(ctx context.Context, urlStr string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, urlStr, nil)
	if err != nil {
		return nil, err
	}
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		token = os.Getenv("HUGGING_FACE_HUB_TOKEN")
	}
	if token = strings.TrimSpace(token); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return req, nil
}

func fileURLPath(f string) string {
	parts := strings.Split(f, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
}

// elapsedLocked is the job's run time, frozen once it has finished.
func (job *DownloadJob) elapsedLocked() time.Duration {
	if !job.ended.IsZero() {
		return job.ended.Sub(job.Started)
	}
	return time.Since(job.Started)
}

func (job *DownloadJob) status() string {
	job.mu.Lock()
	defer job.mu.Unlock()
	return job.Status
}

func (job *DownloadJob) log(line string) {
	job.mu.Lock()
	job.addLog(line)
	job.mu.Unlock()
}

// finish sets a terminal status (unless one is already set) and logs msg.
func (job *DownloadJob) finish(status, msg string) {
	job.mu.Lock()
	if job.Status == "downloading" || status == "stopped" {
		job.Status = status
		job.addLog(msg)
		if job.ended.IsZero() {
			job.ended = time.Now()
		}
	}
	job.mu.Unlock()
	log.Printf("[download] %s: %s:%s — %s", status, job.Repo, job.Quant, msg)
}

func (job *DownloadJob) addLog(line string) {
	job.Logs = append(job.Logs, line)
	if len(job.Logs) > 500 {
		job.Logs = job.Logs[len(job.Logs)-500:]
	}
}

// quantFileRe takes the quant tag at the end of a GGUF name, after a "-" or
// "." separator (Model-Q4_K_M.gguf, llama-2-7b.Q4_K_M.gguf), keeping unsloth's
// "UD-" prefix so UD-Q4_K_XL and Q4_K_XL stay distinct choices.
var quantFileRe = regexp.MustCompile(`[-.]((?:UD-)?[A-Za-z0-9_]+)\.gguf$`)

// quantToken extracts the downloadable quant label from a GGUF rfilename, e.g.
//
//	UD-Q4_K_XL/Model-UD-Q4_K_XL-00001-of-00003.gguf -> "UD-Q4_K_XL"
//	Model-Q8_0.gguf                                  -> "Q8_0"
//	llama-2-7b.Q4_K_M.gguf                           -> "Q4_K_M"
//
// It strips any -NNNNN-of-NNNNN split suffix first so a sharded quant yields
// the same token as a single-file one (otherwise every split model would list
// its shard count, "00003", as a quant). Returns "" when there is no
// recognizable quant, and for vision projectors, which aren't models.
func quantToken(rfilename string) string {
	if isProjector(rfilename) {
		return ""
	}
	base := strings.TrimSuffix(modelPartRe.ReplaceAllString(rfilename, ""), ".gguf")
	m := quantFileRe.FindStringSubmatch(base + ".gguf")
	if len(m) < 2 {
		return ""
	}
	return m[1]
}

// isProjector reports whether a GGUF is a multimodal projector (mmproj-*.gguf),
// which llama-server loads with --mmproj, never as the model itself.
func isProjector(rfilename string) bool {
	return strings.HasPrefix(strings.ToLower(path.Base(rfilename)), "mmproj")
}

func FetchQuants(repo string) ([]string, error) {
	return fetchQuants(repo, hfAPIBase)
}

func fetchQuants(repo, base string) ([]string, error) {
	if err := validateRepo(repo); err != nil {
		return nil, err
	}
	owner, name, _ := strings.Cut(repo, "/")
	endpoint := fmt.Sprintf("%s/%s/%s", base, url.PathEscape(owner), url.PathEscape(name))
	req, err := hfRequest(context.Background(), endpoint)
	if err != nil {
		return nil, err
	}
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching repo info: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HuggingFace API returned %d", resp.StatusCode)
	}

	var result struct {
		Siblings []struct {
			RFilename string `json:"rfilename"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	quants := []string{}
	seen := make(map[string]bool)

	for _, s := range result.Siblings {
		if !strings.HasSuffix(s.RFilename, ".gguf") {
			continue
		}
		q := quantToken(s.RFilename)
		if q == "" {
			continue
		}
		if !seen[q] {
			seen[q] = true
			quants = append(quants, q)
		}
	}

	sort.Strings(quants)
	return quants, nil
}
