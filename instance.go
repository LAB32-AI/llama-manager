package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

type InstanceState string

const (
	StateStopped    InstanceState = "stopped"
	StateStarting   InstanceState = "starting"
	StateRunning    InstanceState = "running"
	StateCrashed    InstanceState = "crashed"
	StateRestarting InstanceState = "restarting"
)

const logBufferSize = 200

type Instance struct {
	conf InstanceConf
	cfg  *Config

	mu           sync.Mutex
	state        InstanceState
	cmd          *exec.Cmd
	startedAt    time.Time
	restartCount int
	lastError    string
	logs         *ringBuffer

	stopCh chan struct{}
}

func NewInstance(conf InstanceConf, cfg *Config) *Instance {
	return &Instance{
		conf:  conf,
		cfg:   cfg,
		state: StateStopped,
		logs:  newRingBuffer(logBufferSize),
	}
}

type InstanceStatus struct {
	Name         string        `json:"name"`
	Model        string        `json:"model"`
	Port         int           `json:"port"`
	GPUIDs       []int         `json:"gpu_ids"`
	State        InstanceState `json:"state"`
	Uptime       string        `json:"uptime"`
	UptimeSec    float64       `json:"uptime_sec"`
	RestartCount int           `json:"restart_count"`
	LastError    string        `json:"last_error,omitempty"`
}

func (inst *Instance) Status() InstanceStatus {
	inst.mu.Lock()
	defer inst.mu.Unlock()

	s := InstanceStatus{
		Name:         inst.conf.Name,
		Model:        inst.conf.Model,
		Port:         inst.conf.Port,
		GPUIDs:       inst.conf.GPUIDs,
		State:        inst.state,
		RestartCount: inst.restartCount,
		LastError:    inst.lastError,
	}

	if inst.state == StateRunning || inst.state == StateStarting {
		d := time.Since(inst.startedAt)
		s.UptimeSec = d.Seconds()
		s.Uptime = formatDuration(d)
	}

	return s
}

func (inst *Instance) State() InstanceState {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	return inst.state
}

func (inst *Instance) Logs() []string {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	return inst.logs.Lines()
}

func (inst *Instance) Start() (<-chan struct{}, error) {
	inst.mu.Lock()
	defer inst.mu.Unlock()

	if inst.state == StateRunning || inst.state == StateStarting {
		return nil, fmt.Errorf("instance %q is already %s", inst.conf.Name, inst.state)
	}

	inst.cfg.mu.RLock()
	serverBin := inst.cfg.ServerBin
	host := inst.cfg.Host
	ngl := inst.cfg.NGL
	mainGPU := inst.cfg.MainGPU
	ctxLen := inst.cfg.ContextLength
	cacheK := inst.cfg.CacheTypeK
	cacheV := inst.cfg.CacheTypeV
	gpuEnv := inst.cfg.GPUEnvVar()
	inst.cfg.mu.RUnlock()

	if inst.conf.NGL != nil {
		ngl = *inst.conf.NGL
	}
	if inst.conf.ContextLength != nil {
		ctxLen = *inst.conf.ContextLength
	}
	if inst.conf.CacheTypeK != nil {
		cacheK = *inst.conf.CacheTypeK
	}
	if inst.conf.CacheTypeV != nil {
		cacheV = *inst.conf.CacheTypeV
	}

	args := []string{}
	if strings.HasPrefix(inst.conf.Model, "/") || strings.HasSuffix(inst.conf.Model, ".gguf") {
		args = append(args, "-m", inst.conf.Model)
	} else {
		args = append(args, "-hf", inst.conf.Model)
	}
	args = append(args,
		"--port", strconv.Itoa(inst.conf.Port),
		"--host", host,
		"-ngl", strconv.Itoa(ngl),
		"-c", strconv.Itoa(ctxLen),
	)

	if gpuEnv != "" {
		if len(inst.conf.GPUIDs) > 1 {
			args = append(args, "-mg", "0")
			ratio := fmt.Sprintf("%.2f", 1.0/float64(len(inst.conf.GPUIDs)))
			parts := make([]string, len(inst.conf.GPUIDs))
			for i := range parts {
				parts[i] = ratio
			}
			args = append(args, "--tensor-split", strings.Join(parts, ","))
		} else {
			args = append(args, "-mg", strconv.Itoa(mainGPU))
		}
	}

	if cacheK != "" {
		args = append(args, "-ctk", cacheK)
	}
	if cacheV != "" {
		args = append(args, "-ctv", cacheV)
	}
	args = append(args, "--metrics")

	cmd := exec.Command(serverBin, args...)
	if gpuEnv != "" {
		gpuList := intsToStrings(inst.conf.GPUIDs)
		cmd.Env = append(cmd.Environ(), fmt.Sprintf("%s=%s", gpuEnv, strings.Join(gpuList, ",")))
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		stdout.Close()
		return nil, fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		stdout.Close()
		stderr.Close()
		return nil, fmt.Errorf("starting process: %w", err)
	}

	inst.cmd = cmd
	inst.state = StateStarting
	inst.startedAt = time.Now()
	inst.lastError = ""
	inst.stopCh = make(chan struct{})

	if gpuEnv != "" {
		log.Printf("[%s] process started (pid %d) on port %d, gpus %v (%s=%s)",
			inst.conf.Name, cmd.Process.Pid, inst.conf.Port, inst.conf.GPUIDs, gpuEnv, strings.Join(intsToStrings(inst.conf.GPUIDs), ","))
	} else {
		log.Printf("[%s] process started (pid %d) on port %d (metal)",
			inst.conf.Name, cmd.Process.Pid, inst.conf.Port)
	}

	go inst.captureOutput(stdout)
	go inst.captureOutput(stderr)

	exitCh := make(chan struct{})
	go func() {
		err := cmd.Wait()
		inst.mu.Lock()
		if inst.state != StateStopped {
			inst.state = StateCrashed
			if err != nil {
				inst.lastError = err.Error()
			} else {
				inst.lastError = "process exited unexpectedly"
			}
			log.Printf("[%s] process exited: %s", inst.conf.Name, inst.lastError)
			if inst.stopCh != nil {
				close(inst.stopCh)
				inst.stopCh = nil
			}
		}
		inst.cmd = nil
		inst.mu.Unlock()
		close(exitCh)
	}()

	return exitCh, nil
}

func (inst *Instance) Stop() error {
	inst.mu.Lock()
	defer inst.mu.Unlock()

	if inst.state == StateStopped {
		return nil
	}

	inst.state = StateStopped
	if inst.stopCh != nil {
		close(inst.stopCh)
		inst.stopCh = nil
	}

	if inst.cmd == nil || inst.cmd.Process == nil {
		return nil
	}

	log.Printf("[%s] stopping process (pid %d)", inst.conf.Name, inst.cmd.Process.Pid)
	return inst.cmd.Process.Kill()
}

func (inst *Instance) SetState(s InstanceState) {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	inst.state = s
}

func (inst *Instance) IncrementRestarts() {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	inst.restartCount++
}

func (inst *Instance) RestartCount() int {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	return inst.restartCount
}

func (inst *Instance) ResetRestarts() {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	inst.restartCount = 0
}

func (inst *Instance) captureOutput(r io.Reader) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		inst.mu.Lock()
		inst.logs.Add(line)
		inst.mu.Unlock()
	}
}

func (inst *Instance) CheckHealth() bool {
	inst.cfg.mu.RLock()
	host := inst.cfg.Host
	inst.cfg.mu.RUnlock()
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	url := fmt.Sprintf("http://%s:%d/health", host, inst.conf.Port)
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

type InstanceMetrics struct {
	PromptTokensSec    float64 `json:"prompt_tokens_sec"`
	PredictedTokensSec float64 `json:"predicted_tokens_sec"`
	PromptTokensTotal  float64 `json:"prompt_tokens_total"`
	PredictedTotal     float64 `json:"predicted_total"`
	KVCacheUsage       float64 `json:"kv_cache_usage"`
	RequestsProcessing float64 `json:"requests_processing"`
	RequestsDeferred   float64 `json:"requests_deferred"`
}

func (inst *Instance) FetchMetrics() *InstanceMetrics {
	if inst.State() != StateRunning {
		return nil
	}
	inst.cfg.mu.RLock()
	host := inst.cfg.Host
	inst.cfg.mu.RUnlock()
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	url := fmt.Sprintf("http://%s:%d/metrics", host, inst.conf.Port)
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	m := &InstanceMetrics{}
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		val, err := strconv.ParseFloat(parts[len(parts)-1], 64)
		if err != nil {
			continue
		}
		key := parts[0]
		if idx := strings.IndexByte(key, '{'); idx >= 0 {
			key = key[:idx]
		}
		switch key {
		case "llamacpp:prompt_tokens_seconds":
			m.PromptTokensSec = val
		case "llamacpp:predicted_tokens_seconds":
			m.PredictedTokensSec = val
		case "llamacpp:prompt_tokens_total":
			m.PromptTokensTotal = val
		case "llamacpp:tokens_predicted_total":
			m.PredictedTotal = val
		case "llamacpp:kv_cache_usage_ratio":
			m.KVCacheUsage = val
		case "llamacpp:requests_processing":
			m.RequestsProcessing = val
		case "llamacpp:requests_deferred":
			m.RequestsDeferred = val
		}
	}
	return m
}

type ringBuffer struct {
	lines []string
	size  int
	pos   int
	full  bool
}

func newRingBuffer(size int) *ringBuffer {
	return &ringBuffer{
		lines: make([]string, size),
		size:  size,
	}
}

func (rb *ringBuffer) Add(line string) {
	rb.lines[rb.pos] = line
	rb.pos++
	if rb.pos >= rb.size {
		rb.pos = 0
		rb.full = true
	}
}

func (rb *ringBuffer) Lines() []string {
	if !rb.full {
		result := make([]string, rb.pos)
		copy(result, rb.lines[:rb.pos])
		return result
	}
	result := make([]string, rb.size)
	copy(result, rb.lines[rb.pos:])
	copy(result[rb.size-rb.pos:], rb.lines[:rb.pos])
	return result
}

func intsToStrings(ids []int) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = strconv.Itoa(id)
	}
	return out
}

func formatDuration(d time.Duration) string {
	d = d.Round(time.Second)
	total := int(d.Seconds())
	days := total / 86400
	hours := (total % 86400) / 3600
	mins := (total % 3600) / 60
	if days > 0 {
		return fmt.Sprintf("%dd %dh %dm", days, hours, mins)
	}
	if hours > 0 {
		return fmt.Sprintf("%dh %dm", hours, mins)
	}
	return fmt.Sprintf("%dm", mins)
}
