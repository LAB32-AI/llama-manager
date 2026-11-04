package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
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
	StateUnhealthy  InstanceState = "unhealthy"
	StateCrashed    InstanceState = "crashed"
	StateRestarting InstanceState = "restarting"
)

const logBufferSize = 2000

// outputWaitDelay bounds how long Wait keeps reading output after the process
// has exited, in case a grandchild still holds the pipes open.
var outputWaitDelay = 5 * time.Second

// stopGrace is how long Stop waits after SIGTERM before escalating to SIGKILL;
// stopWait bounds the wait for the process to be reaped after SIGKILL. Vars so
// tests can shorten them.
var (
	stopGrace = 10 * time.Second
	stopWait  = 10 * time.Second
)

// testHookAfterWait, when set, runs in the exit goroutine after the process has
// been reaped but before the instance state is updated. Tests use it to widen
// the teardown window that a large model's slow exit opens in production.
var testHookAfterWait func()

// proc is one run of the child process. Every goroutine tied to a run (exit
// watcher, health loop) holds its own *proc and only mutates instance state
// while inst.proc still points at it, so a previous run can never clobber the
// next one.
type proc struct {
	cmd        *exec.Cmd
	host       string // address the manager uses to reach this run's server
	startedAt  time.Time
	done       chan struct{} // closed when the run ends: stopped or exited
	exited     chan struct{} // closed once the process has been reaped
	doneClosed bool          // guarded by Instance.mu
	termSent   bool          // guarded by Instance.mu
}

type Instance struct {
	conf InstanceConf
	cfg  *Config

	// opMu serializes lifecycle operations (start/stop/restart/remove) coming
	// from callers. Supervisor goroutines never take it, so a caller may hold
	// it while waiting for a supervisor to exit.
	opMu sync.Mutex

	mu           sync.Mutex
	state        InstanceState
	proc         *proc
	startedAt    time.Time
	restartCount int
	lastError    string
	logs         *ringBuffer
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
	// Role is "image_reader" for the instance image_recognition runs and
	// "image_gen" for image_generation's.
	Role string `json:"role,omitempty"`
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
		Role:         inst.conf.role,
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

func (inst *Instance) Start() (*proc, error) {
	inst.mu.Lock()
	defer inst.mu.Unlock()

	if inst.proc != nil {
		return nil, fmt.Errorf("instance %q: previous process (pid %d) has not exited yet", inst.conf.Name, inst.proc.cmd.Process.Pid)
	}
	if inst.state == StateRunning || inst.state == StateStarting {
		return nil, fmt.Errorf("instance %q is already %s", inst.conf.Name, inst.state)
	}

	spec, err := buildLaunch(inst.conf, inst.cfg)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(spec.bin, spec.args...)
	if len(spec.env) > 0 {
		cmd.Env = append(cmd.Environ(), spec.env...)
	}

	// One writer for both streams: os/exec then gives the child a single pipe,
	// so the log keeps the order the server wrote in. Wait waits for the copy
	// to finish, so the last lines (usually the crash reason) are kept, and
	// WaitDelay stops a grandchild that inherited the pipe from stalling exit
	// detection after the server itself is gone.
	output := &lineWriter{add: inst.addLog}
	cmd.Stdout = output
	cmd.Stderr = output
	cmd.WaitDelay = outputWaitDelay
	setProcAttrs(cmd)

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting process: %w", err)
	}

	p := &proc{cmd: cmd, host: connectHost(spec.host), startedAt: time.Now(), done: make(chan struct{}), exited: make(chan struct{})}
	inst.proc = p
	inst.state = StateStarting
	inst.startedAt = p.startedAt
	inst.lastError = ""

	if len(spec.env) > 0 {
		log.Printf("[%s] process started (pid %d) on port %d, gpus %v (%s)",
			inst.conf.Name, cmd.Process.Pid, inst.conf.Port, inst.conf.GPUIDs, strings.Join(spec.env, " "))
	} else {
		log.Printf("[%s] process started (pid %d) on port %d (metal)",
			inst.conf.Name, cmd.Process.Pid, inst.conf.Port)
	}

	go func() {
		err := cmd.Wait()
		// Reap anything the server left behind in its process group (children
		// of a wrapper script) so nothing keeps holding the port or VRAM.
		killGroup(cmd.Process)
		output.Flush()
		if testHookAfterWait != nil {
			testHookAfterWait()
		}
		inst.mu.Lock()
		if inst.proc == p {
			if inst.state != StateStopped {
				inst.state = StateCrashed
				if inst.lastError == "" {
					if err != nil {
						inst.lastError = err.Error()
					} else {
						inst.lastError = "process exited unexpectedly"
					}
				}
				log.Printf("[%s] process exited: %s", inst.conf.Name, inst.lastError)
			}
			inst.endRunLocked(p)
			inst.proc = nil
		}
		inst.mu.Unlock()
		close(p.exited)
	}()

	return p, nil
}

// markStartFailed records a failed start attempt. It leaves the state alone if
// a process still exists (the previous run is still exiting).
func (inst *Instance) markStartFailed(err error) {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	if inst.proc != nil {
		return
	}
	inst.state = StateCrashed
	inst.lastError = "start failed: " + err.Error()
}

func (inst *Instance) endRunLocked(p *proc) {
	if !p.doneClosed {
		p.doneClosed = true
		close(p.done)
	}
}

// Stop marks the instance stopped and shuts the current process down: SIGTERM
// to its process group, then SIGKILL after stopGrace. It returns only once the
// process has been reaped, so a Start issued after Stop returns can never
// overlap the old process.
func (inst *Instance) Stop() error {
	inst.mu.Lock()
	inst.state = StateStopped
	p := inst.proc
	if p != nil {
		inst.endRunLocked(p)
		if !p.termSent {
			p.termSent = true
			log.Printf("[%s] stopping process (pid %d)", inst.conf.Name, p.cmd.Process.Pid)
			terminateGroup(p.cmd.Process)
		}
	}
	inst.mu.Unlock()

	if p == nil {
		return nil
	}
	select {
	case <-p.exited:
		return nil
	case <-time.After(stopGrace):
	}
	log.Printf("[%s] pid %d still running %s after SIGTERM, sending SIGKILL", inst.conf.Name, p.cmd.Process.Pid, stopGrace)
	killGroup(p.cmd.Process)
	select {
	case <-p.exited:
		return nil
	case <-time.After(stopWait):
		return fmt.Errorf("instance %q: process (pid %d) did not exit within %s of SIGKILL", inst.conf.Name, p.cmd.Process.Pid, stopWait)
	}
}

// transition moves the instance from one state to another only if it is still
// in `from` and p is still its current run (nil meaning "no process"). It
// reports whether the change happened, so callers acting on an earlier reading
// can never overwrite a state set concurrently by Stop or the exit watcher.
func (inst *Instance) transition(p *proc, from, to InstanceState) bool {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	if inst.proc != p || inst.state != from {
		return false
	}
	inst.state = to
	return true
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

func (inst *Instance) addLog(line string) {
	inst.mu.Lock()
	inst.logs.Add(line)
	inst.mu.Unlock()
}

// maxLogLine caps one captured log line. The rest of an over-long line is
// dropped, so a single huge line (a template or tool-call dump) can neither
// stall capture nor balloon memory.
const maxLogLine = 8 * 1024

const truncatedMarker = " ...[truncated]"

// lineWriter splits a byte stream into log lines. It is written by a single
// os/exec copy goroutine, so it needs no locking of its own.
type lineWriter struct {
	add  func(string)
	buf  []byte
	over bool
}

func (w *lineWriter) Write(b []byte) (int, error) {
	n := len(b)
	for len(b) > 0 {
		i := bytes.IndexByte(b, '\n')
		chunk := b
		if i >= 0 {
			chunk = b[:i]
		}
		if !w.over {
			if room := maxLogLine - len(w.buf); len(chunk) > room {
				w.buf = append(w.buf, chunk[:room]...)
				w.over = true
			} else {
				w.buf = append(w.buf, chunk...)
			}
		}
		if i < 0 {
			break
		}
		w.emit()
		b = b[i+1:]
	}
	return n, nil
}

func (w *lineWriter) emit() {
	line := strings.TrimSuffix(string(w.buf), "\r")
	if w.over {
		line += truncatedMarker
	}
	w.add(line)
	w.buf = w.buf[:0]
	w.over = false
}

// Flush emits a final line that ended without a newline.
func (w *lineWriter) Flush() {
	if len(w.buf) > 0 || w.over {
		w.emit()
	}
}

// connectHost turns a --host bind address into one the manager can dial.
func connectHost(bind string) string {
	if bind == "" || bind == "0.0.0.0" || bind == "::" {
		return "127.0.0.1"
	}
	return bind
}

// baseURL is where the manager reaches this instance's llama-server. A running
// process is reached at the host it was started with, so changing the host
// setting only takes effect on the next start instead of pointing health
// checks at an address the running server isn't listening on.
func (inst *Instance) baseURL() string {
	inst.mu.Lock()
	host := ""
	if inst.proc != nil {
		host = inst.proc.host
	}
	inst.mu.Unlock()
	if host == "" {
		inst.cfg.mu.RLock()
		host = connectHost(inst.cfg.Host)
		inst.cfg.mu.RUnlock()
	}
	return "http://" + net.JoinHostPort(host, strconv.Itoa(inst.conf.Port))
}

func (inst *Instance) CheckHealth(ctx context.Context) bool {
	inst.cfg.mu.RLock()
	timeout := inst.cfg.HealthCheckTimeout.Duration
	inst.cfg.mu.RUnlock()
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	path := "/health"
	if inst.conf.role == roleImageGen {
		// sd-server has no /health; this answers once the model is loaded.
		path = "/sdcpp/v1/models/status"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, inst.baseURL()+path, nil)
	if err != nil {
		return false
	}
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false
	}
	if inst.conf.role == roleImageGen {
		var s struct {
			State string `json:"state"`
		}
		return json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&s) == nil && s.State == "ready"
	}
	return true
}

// killProcess is called by the health-check loop when an instance is wedged.
// It SIGKILLs the running process without touching state; the exit goroutine
// in Start() will flip the state to Crashed, which the supervisor's restart
// loop picks up like any other crash.
func (inst *Instance) killProcess(p *proc, reason string) {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	if inst.proc != p {
		return
	}
	inst.lastError = reason
	log.Printf("[%s] killing process (pid %d): %s", inst.conf.Name, p.cmd.Process.Pid, reason)
	killGroup(p.cmd.Process)
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
	if inst.State() != StateRunning || inst.conf.role == roleImageGen {
		return nil
	}
	url := inst.baseURL() + "/metrics"
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
