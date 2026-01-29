package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os/exec"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

type DownloadManager struct {
	serverBin string
	mu        sync.Mutex
	active    *DownloadJob
}

type DownloadJob struct {
	Repo    string `json:"repo"`
	Quant   string `json:"quant"`
	Status  string `json:"status"` // "downloading", "done", "failed", "stopped"
	Logs    []string `json:"logs"`
	Started time.Time `json:"started"`
	cmd     *exec.Cmd
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

func NewDownloadManager(serverBin string) *DownloadManager {
	return &DownloadManager{serverBin: serverBin}
}

func (dm *DownloadManager) Start(repo, quant string) error {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	if dm.active != nil && dm.active.Status == "downloading" {
		return fmt.Errorf("download already in progress: %s:%s", dm.active.Repo, dm.active.Quant)
	}

	model := repo
	if quant != "" {
		model = repo + ":" + quant
	}

	cmd := exec.Command(dm.serverBin, "-hf", model, "--port", "0")

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		stdout.Close()
		return fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		stdout.Close()
		stderr.Close()
		return fmt.Errorf("starting download: %w", err)
	}

	job := &DownloadJob{
		Repo:    repo,
		Quant:   quant,
		Status:  "downloading",
		Started: time.Now(),
		cmd:     cmd,
	}
	dm.active = job

	log.Printf("[download] started: %s", model)

	go job.captureOutput(stdout)
	go job.captureOutput(stderr)

	go func() {
		err := cmd.Wait()
		job.mu.Lock()
		defer job.mu.Unlock()
		if job.Status == "stopped" {
			return
		}
		if err != nil {
			job.Status = "failed"
			job.addLog("process exited: " + err.Error())
			log.Printf("[download] failed: %s - %v", model, err)
		} else {
			job.Status = "done"
			job.addLog("download complete")
			log.Printf("[download] completed: %s", model)
		}
	}()

	return nil
}

func (dm *DownloadManager) Stop() {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	if dm.active == nil || dm.active.cmd == nil || dm.active.cmd.Process == nil {
		return
	}

	dm.active.mu.Lock()
	dm.active.Status = "stopped"
	dm.active.addLog("download stopped by user")
	dm.active.mu.Unlock()

	dm.active.cmd.Process.Kill()
	log.Printf("[download] stopped by user")
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

func (job *DownloadJob) captureOutput(r io.Reader) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 64*1024)
	for scanner.Scan() {
		line := scanner.Text()
		job.mu.Lock()
		job.addLog(line)
		if strings.Contains(line, "listening on") || strings.Contains(line, "all slots are idle") {
			if job.cmd != nil && job.cmd.Process != nil {
				job.Status = "done"
				job.addLog("model downloaded, stopping server")
				go job.cmd.Process.Kill()
			}
		}
		job.mu.Unlock()
	}
}

func (job *DownloadJob) addLog(line string) {
	job.Logs = append(job.Logs, line)
	if len(job.Logs) > 500 {
		job.Logs = job.Logs[len(job.Logs)-500:]
	}
}

func FetchQuants(repo string) ([]string, error) {
	url := fmt.Sprintf("https://huggingface.co/api/models/%s", repo)
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Get(url)
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

	quantRe := regexp.MustCompile(`-([A-Za-z0-9_]+)\.gguf$`)
	quants := []string{}
	seen := make(map[string]bool)

	for _, s := range result.Siblings {
		if !strings.HasSuffix(s.RFilename, ".gguf") {
			continue
		}
		matches := quantRe.FindStringSubmatch(s.RFilename)
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
