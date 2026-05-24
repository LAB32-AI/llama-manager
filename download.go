package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

var (
	repoRe  = regexp.MustCompile(`^[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}$`)
	quantRe = regexp.MustCompile(`^[A-Za-z0-9_]{1,32}$`)
)

// HuggingFace endpoints. Vars (not consts) so tests can point them at httptest.
var (
	hfAPIBase     = "https://huggingface.co/api/models"
	hfResolveBase = "https://huggingface.co"
)

func validateRepo(repo string) error {
	if !repoRe.MatchString(repo) {
		return fmt.Errorf("invalid repo: must match owner/name with [A-Za-z0-9._-]")
	}
	return nil
}

func validateQuant(quant string) error {
	if quant == "" {
		return nil
	}
	if !quantRe.MatchString(quant) {
		return fmt.Errorf("invalid quant: must match [A-Za-z0-9_]{1,32}")
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

	dm.mu.Lock()
	defer dm.mu.Unlock()

	if dm.active != nil && dm.active.Status == "downloading" {
		return fmt.Errorf("download already in progress: %s:%s", dm.active.Repo, dm.active.Quant)
	}

	files, err := resolveQuantFiles(repo, quant, hfAPIBase)
	if err != nil {
		return err
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

	model := repo
	if quant != "" {
		model += ":" + quant
	}
	log.Printf("[download] started: %s (%d file(s))", model, len(files))
	job.log(fmt.Sprintf("resolving %s → %d file(s)", model, len(files)))

	go dm.run(ctx, job, repo, files)
	return nil
}

func (dm *DownloadManager) run(ctx context.Context, job *DownloadJob, repo string, files []string) {
	destDir := filepath.Join(huggingfaceHubDir(), "models--"+strings.ReplaceAll(repo, "/", "--"), "snapshots", "main")
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		job.finish("failed", "creating cache dir: "+err.Error())
		return
	}

	for _, f := range files {
		if ctx.Err() != nil {
			job.finish("stopped", "download stopped by user")
			return
		}
		dest := filepath.Join(destDir, filepath.FromSlash(f))
		if !strings.HasPrefix(filepath.Clean(dest), filepath.Clean(destDir)+string(os.PathSeparator)) {
			job.finish("failed", "unsafe file path from repo: "+f)
			return
		}
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			job.finish("failed", "creating dir: "+err.Error())
			return
		}
		urlStr := hfResolveBase + "/" + repo + "/resolve/main/" + fileURLPath(f)
		job.log("downloading " + f)
		if err := downloadToFile(ctx, urlStr, dest, job); err != nil {
			if ctx.Err() != nil {
				job.finish("stopped", "download stopped by user")
			} else {
				job.finish("failed", err.Error())
			}
			return
		}
		job.log("saved " + f)
	}
	job.finish("done", "download complete")
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
		Elapsed: formatDuration(time.Since(dm.active.Started)),
	}
}

// resolveQuantFiles returns the .gguf rfilenames in repo whose quant token (the
// trailing -SEGMENT before .gguf, ignoring any -NNNNN-of-NNNNN split suffix)
// equals quant — multiple are returned for split GGUFs. If quant is empty and
// the repo has exactly one .gguf, that file is returned.
func resolveQuantFiles(repo, quant, apiBase string) ([]string, error) {
	if err := validateRepo(repo); err != nil {
		return nil, err
	}
	owner, name, _ := strings.Cut(repo, "/")
	endpoint := fmt.Sprintf("%s/%s/%s", apiBase, url.PathEscape(owner), url.PathEscape(name))
	client := &http.Client{Timeout: 20 * time.Second}
	resp, err := client.Get(endpoint)
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

	var all, matched []string
	for _, s := range result.Siblings {
		f := s.RFilename
		if !strings.HasSuffix(f, ".gguf") {
			continue
		}
		all = append(all, f)
		token := strings.TrimSuffix(modelPartRe.ReplaceAllString(f, ""), ".gguf")
		if quant != "" && (strings.HasSuffix(token, "-"+quant) || token == quant) {
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

// downloadToFile streams urlStr into dest (via a .incomplete temp + rename),
// honoring ctx cancellation and logging progress to the job. Go's net/http
// handles the HTTPS/CDN redirect itself, so no TLS support in llama.cpp is
// needed.
func downloadToFile(ctx context.Context, urlStr, dest string, job *DownloadJob) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, urlStr, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d fetching %s", resp.StatusCode, filepath.Base(dest))
	}

	tmp := dest + ".incomplete"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	pw := &progressWriter{job: job, total: resp.ContentLength, name: filepath.Base(dest)}
	_, copyErr := io.Copy(out, io.TeeReader(resp.Body, pw))
	closeErr := out.Close()
	if copyErr != nil {
		os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		os.Remove(tmp)
		return closeErr
	}
	return os.Rename(tmp, dest)
}

// progressWriter logs a download progress line at most every 2s.
type progressWriter struct {
	job   *DownloadJob
	total int64
	name  string
	done  int64
	last  time.Time
}

func (p *progressWriter) Write(b []byte) (int, error) {
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

func fileURLPath(f string) string {
	parts := strings.Split(f, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
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

var quantFileRe = regexp.MustCompile(`-([A-Za-z0-9_]+)\.gguf$`)

func FetchQuants(repo string) ([]string, error) {
	return fetchQuants(repo, hfAPIBase)
}

func fetchQuants(repo, base string) ([]string, error) {
	if err := validateRepo(repo); err != nil {
		return nil, err
	}
	owner, name, _ := strings.Cut(repo, "/")
	endpoint := fmt.Sprintf("%s/%s/%s", base, url.PathEscape(owner), url.PathEscape(name))
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Get(endpoint)
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
		matches := quantFileRe.FindStringSubmatch(s.RFilename)
		if len(matches) < 2 {
			continue
		}
		q := matches[1]
		if !seen[q] {
			seen[q] = true
			quants = append(quants, q)
		}
	}

	sort.Strings(quants)
	return quants, nil
}
