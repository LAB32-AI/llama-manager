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
