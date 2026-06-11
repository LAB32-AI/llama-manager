package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func newManagerForTest(t *testing.T, serverBin string, restartDelay time.Duration) *Manager {
	t.Helper()
	dir := t.TempDir()
	cfg := &Config{
		ServerBin:           serverBin,
		ManagerPort:         0,
		RestartDelay:        duration{restartDelay},
		MaxRestarts:         5,
		HealthCheckInterval: duration{30 * time.Second},
		GPUBackend:          "metal",
		Host:                "127.0.0.1",
		NGL:                 1,
		ContextLength:       128,
		path:                filepath.Join(dir, "config.yaml"),
	}
	cfg.Instances = []InstanceConf{
		{Name: "t1", Model: "/dev/null", Port: 0, GPUIDs: []int{0}},
	}
	return NewManager(cfg)
}

// failingBin returns an absolute path to a binary that always exits with
// non-zero status. /usr/bin/false is universal on darwin and linux.
func failingBin(t *testing.T) string {
	t.Helper()
	for _, p := range []string{"/usr/bin/false", "/bin/false"} {
		if _, err := exec.LookPath(p); err == nil {
			return p
		}
	}
	t.Skipf("no false binary on %s", runtime.GOOS)
	return ""
}

func TestStopInstanceUnknownReturnsError(t *testing.T) {
	m := newManagerForTest(t, failingBin(t), 50*time.Millisecond)
	if err := m.StopInstance("does-not-exist"); err == nil {
		t.Fatal("expected error for unknown instance")
	}
}

func TestStartInstanceUnknownReturnsError(t *testing.T) {
	m := newManagerForTest(t, failingBin(t), 50*time.Millisecond)
	if err := m.StartInstance("does-not-exist"); err == nil {
		t.Fatal("expected error for unknown instance")
	}
}

// TestStopDuringCooldownExitsSupervisor reproduces the original UI bug:
// the supervisor goroutine must exit promptly when Stop is called between
// restart attempts (during the RestartDelay sleep). Before the fix, only
// m.stopCh broke the cooldown; per-instance Stop did not, so the supervisor
// kept restarting the process.
func TestStopDuringCooldownExitsSupervisor(t *testing.T) {
	const cooldown = 2 * time.Second
	m := newManagerForTest(t, failingBin(t), cooldown)
	defer m.Shutdown()

	if err := m.StartInstance("t1"); err != nil {
		t.Fatalf("StartInstance: %v", err)
	}

	// Wait until we observe at least one crash + state == Restarting.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if m.Get("t1").State() == StateRestarting {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if m.Get("t1").State() != StateRestarting {
		t.Fatalf("expected StateRestarting, got %s", m.Get("t1").State())
	}

	// The bug: this call used to sit dormant during the 2s cooldown,
	// and then the supervisor would start a new process. After the fix,
	// the supervisor must wake immediately via ctx.Done() and exit.
	start := time.Now()
	if err := m.StopInstance("t1"); err != nil {
		t.Fatalf("StopInstance: %v", err)
	}

	// Wait briefly for supervisor goroutine to drain.
	settled := false
	for time.Now().Before(start.Add(500 * time.Millisecond)) {
		m.mu.RLock()
		_, stillSupervised := m.supervisor["t1"]
		m.mu.RUnlock()
		if !stillSupervised {
			settled = true
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !settled {
		t.Fatal("supervisor did not exit after StopInstance; cooldown still blocking")
	}

	if m.Get("t1").State() != StateStopped {
		t.Errorf("state = %s, want StateStopped", m.Get("t1").State())
	}

	// Confirm no further restart fires within one cooldown window.
	pidBefore := m.Get("t1").restartCount
	time.Sleep(cooldown + 500*time.Millisecond)
	pidAfter := m.Get("t1").restartCount
	if pidAfter != pidBefore {
		t.Errorf("restart count grew after Stop: %d -> %d (auto-restart still active)", pidBefore, pidAfter)
	}
}

// TestHealthCheckKillsWedgedProcess proves that when an instance's process is
// alive but unresponsive to /health, the health-check loop demotes the state
// and then force-kills the process so the supervisor's restart loop picks it
// up. Before this fix, a wedged llama-server would show "running" forever and
// the manager would never act.
func TestHealthCheckKillsWedgedProcess(t *testing.T) {
	dir := t.TempDir()
	binPath := filepath.Join(dir, "fake-server.sh")
	if err := os.WriteFile(binPath, []byte("#!/bin/sh\nexec sleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	cfg := &Config{
		ServerBin:           binPath,
		ManagerPort:         0,
		RestartDelay:        duration{10 * time.Second},
		MaxRestarts:         5,
		HealthCheckInterval: duration{50 * time.Millisecond},
		HealthCheckTimeout:  duration{20 * time.Millisecond},
		UnhealthyAfter:      2,
		GPUBackend:          "metal",
		Host:                "127.0.0.1",
		NGL:                 1,
		ContextLength:       128,
		path:                filepath.Join(dir, "config.yaml"),
	}
	cfg.Instances = []InstanceConf{
		{Name: "wedged", Model: "/dev/null", Port: 65500, GPUIDs: []int{0}},
	}
	m := NewManager(cfg)
	defer m.Shutdown()

	if err := m.StartInstance("wedged"); err != nil {
		t.Fatalf("StartInstance: %v", err)
	}

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if m.Get("wedged").RestartCount() > 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if got := m.Get("wedged").RestartCount(); got == 0 {
		t.Fatalf("expected health-check kill to bump restart count, but it stayed 0")
	}

	inst := m.Get("wedged")
	inst.mu.Lock()
	le := inst.lastError
	inst.mu.Unlock()
	if !strings.Contains(le, "health checks failed") {
		t.Errorf("lastError = %q, want it to mention 'health checks failed'", le)
	}
}

// TestUnhealthyAfterZeroDisablesKill verifies that setting unhealthy_after to 0
// stops the health-check loop from force-killing a wedged process, even though
// the state still demotes to unhealthy.
func TestUnhealthyAfterZeroDisablesKill(t *testing.T) {
	dir := t.TempDir()
	binPath := filepath.Join(dir, "fake-server.sh")
	if err := os.WriteFile(binPath, []byte("#!/bin/sh\nexec sleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	cfg := &Config{
		ServerBin:           binPath,
		ManagerPort:         0,
		RestartDelay:        duration{10 * time.Second},
		MaxRestarts:         5,
		HealthCheckInterval: duration{30 * time.Millisecond},
		HealthCheckTimeout:  duration{15 * time.Millisecond},
		UnhealthyAfter:      0,
		GPUBackend:          "metal",
		Host:                "127.0.0.1",
		NGL:                 1,
		ContextLength:       128,
		path:                filepath.Join(dir, "config.yaml"),
	}
	cfg.Instances = []InstanceConf{
		{Name: "wedged", Model: "/dev/null", Port: 65501, GPUIDs: []int{0}},
	}
	m := NewManager(cfg)
	defer m.Shutdown()

	if err := m.StartInstance("wedged"); err != nil {
		t.Fatal(err)
	}

	time.Sleep(400 * time.Millisecond)
	if got := m.Get("wedged").RestartCount(); got != 0 {
		t.Fatalf("with unhealthy_after=0, expected no restart, got count=%d", got)
	}
}

func waitForState(t *testing.T, m *Manager, name string, want InstanceState, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if m.Get(name).State() == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("instance %q state = %s, want %s within %s", name, m.Get(name).State(), want, timeout)
}

// TestRestartReSpawnsInstance reproduces the /restart bug: RestartInstance
// stopped the running child, but the old supervisor goroutine's teardown
// (inst.Stop() on ctx.Done) raced the new Start and killed the fresh process,
// leaving the instance Stopped and unsupervised. After the fix,
// detachSupervisor waits for the old goroutine to exit before re-supervising,
// so restart reliably brings the process back.
func TestRestartReSpawnsInstance(t *testing.T) {
	dir := t.TempDir()
	binPath := filepath.Join(dir, "fake-server.sh")
	if err := os.WriteFile(binPath, []byte("#!/bin/sh\nexec sleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{
		ServerBin:           binPath,
		ManagerPort:         0,
		RestartDelay:        duration{50 * time.Millisecond},
		MaxRestarts:         5,
		HealthCheckInterval: duration{30 * time.Second},
		HealthCheckTimeout:  duration{20 * time.Millisecond},
		UnhealthyAfter:      0, // don't let health checks kill the unresponsive fake
		GPUBackend:          "metal",
		Host:                "127.0.0.1",
		NGL:                 1,
		ContextLength:       128,
		path:                filepath.Join(dir, "config.yaml"),
	}
	cfg.Instances = []InstanceConf{
		{Name: "r1", Model: "/dev/null", Port: 0, GPUIDs: []int{0}},
	}
	m := NewManager(cfg)
	defer m.Shutdown()

	if err := m.StartInstance("r1"); err != nil {
		t.Fatalf("StartInstance: %v", err)
	}
	waitForState(t, m, "r1", StateStarting, time.Second)

	if err := m.RestartInstance("r1"); err != nil {
		t.Fatalf("RestartInstance: %v", err)
	}

	// After restart the instance must be alive and supervised, not Stopped.
	waitForState(t, m, "r1", StateStarting, time.Second)
	m.mu.RLock()
	_, supervised := m.supervisor["r1"]
	m.mu.RUnlock()
	if !supervised {
		t.Fatal("instance not supervised after restart")
	}

	// And it must stay up: the old goroutine must not kill it moments later.
	time.Sleep(300 * time.Millisecond)
	if st := m.Get("r1").State(); st == StateStopped {
		t.Fatalf("instance fell back to Stopped after restart (re-spawn race)")
	}
}

// TestStartAllRespectsAutoStart proves boot only launches instances flagged
// auto_start, so a rig with several configured-but-mutually-exclusive models
// doesn't try to start them all at once on reboot.
func TestStartAllRespectsAutoStart(t *testing.T) {
	dir := t.TempDir()
	binPath := filepath.Join(dir, "fake-server.sh")
	if err := os.WriteFile(binPath, []byte("#!/bin/sh\nexec sleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{
		ServerBin:           binPath,
		ManagerPort:         0,
		RestartDelay:        duration{time.Second},
		MaxRestarts:         5,
		HealthCheckInterval: duration{30 * time.Second},
		HealthCheckTimeout:  duration{20 * time.Millisecond},
		UnhealthyAfter:      0,
		GPUBackend:          "metal",
		Host:                "127.0.0.1",
		NGL:                 1,
		ContextLength:       128,
		path:                filepath.Join(dir, "config.yaml"),
	}
	cfg.Instances = []InstanceConf{
		{Name: "boots", Model: "/dev/null", Port: 0, GPUIDs: []int{0}, AutoStart: true},
		{Name: "dormant", Model: "/dev/null", Port: 0, GPUIDs: []int{0}, AutoStart: false},
	}
	m := NewManager(cfg)
	defer m.Shutdown()

	m.StartAll()
	waitForState(t, m, "boots", StateStarting, time.Second)

	m.mu.RLock()
	_, bootsSup := m.supervisor["boots"]
	_, dormantSup := m.supervisor["dormant"]
	m.mu.RUnlock()
	if !bootsSup {
		t.Error("auto_start instance was not supervised by StartAll")
	}
	if dormantSup {
		t.Error("non-auto_start instance was started by StartAll")
	}
	if st := m.Get("dormant").State(); st != StateStopped {
		t.Errorf("dormant instance state = %s, want stopped", st)
	}
}

// TestStableRunResetsRestartBudget proves that an instance which stays up past
// the stable window and then crashes does not burn its restart budget: such
// transient crashes (e.g. an intermittent GPU/backend abort) recover
// indefinitely instead of accumulating toward MaxRestarts and being given up on.
// Without the reset, a binary that crashes every round would hit MaxRestarts and
// the supervisor would detach.
func TestStableRunResetsRestartBudget(t *testing.T) {
	dir := t.TempDir()
	binPath := filepath.Join(dir, "blip.sh")
	// Stays up ~150ms (> the shortened stable window below), then exits non-zero
	// — emulating a process that runs fine for a while then hits a transient abort.
	if err := os.WriteFile(binPath, []byte("#!/bin/sh\nsleep 0.15\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	old := restartStableWindow
	restartStableWindow = 50 * time.Millisecond
	defer func() { restartStableWindow = old }()

	cfg := &Config{
		ServerBin:           binPath,
		ManagerPort:         0,
		RestartDelay:        duration{10 * time.Millisecond},
		MaxRestarts:         3,
		HealthCheckInterval: duration{30 * time.Second},
		UnhealthyAfter:      0,
		GPUBackend:          "metal",
		Host:                "127.0.0.1",
		NGL:                 1,
		ContextLength:       128,
		path:                filepath.Join(dir, "config.yaml"),
	}
	cfg.Instances = []InstanceConf{
		{Name: "blip", Model: "/dev/null", Port: 0, GPUIDs: []int{0}},
	}
	m := NewManager(cfg)
	defer m.Shutdown()

	if err := m.StartInstance("blip"); err != nil {
		t.Fatalf("StartInstance: %v", err)
	}

	// Let it cycle through several crash/restart rounds (~150ms up + ~10ms delay
	// each). Far more than MaxRestarts=3 rounds elapse; without the stable-run
	// reset the supervisor would have given up and detached by now.
	time.Sleep(900 * time.Millisecond)

	m.mu.RLock()
	_, supervised := m.supervisor["blip"]
	m.mu.RUnlock()
	if !supervised {
		t.Fatal("supervisor gave up on an instance that kept recovering after stable runs")
	}
	if c := m.Get("blip").RestartCount(); c >= cfg.MaxRestarts {
		t.Fatalf("restart count = %d, should stay below MaxRestarts (%d) via stable-run reset", c, cfg.MaxRestarts)
	}
}

func TestAddInstanceDeduplicatesInManager(t *testing.T) {
	m := newManagerForTest(t, failingBin(t), time.Second)
	ic := InstanceConf{Name: "new", Model: "/m", Port: 9091, GPUIDs: []int{0}}
	if err := m.AddInstance(ic); err != nil {
		t.Fatalf("first AddInstance: %v", err)
	}
	if err := m.AddInstance(ic); err == nil {
		t.Fatal("expected duplicate error on second AddInstance")
	}
}
