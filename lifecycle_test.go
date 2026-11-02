package main

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// slowReap delays every exit goroutine after the process is reaped, emulating
// the seconds a large multi-GPU model takes to release VRAM after SIGKILL.
func slowReap(t *testing.T, d time.Duration) {
	t.Helper()
	old := testHookAfterWait
	testHookAfterWait = func() { time.Sleep(d) }
	t.Cleanup(func() { testHookAfterWait = old })
}

func assertOneTrackedProcess(t *testing.T, rig *fakeRig, inst *Instance) {
	t.Helper()
	waitFor(t, 2*time.Second, "exactly one live fake server", func() bool { return len(rig.livePids(t)) == 1 })
	live := rig.livePids(t)
	if got := inst.currentPid(); got != live[0] {
		t.Fatalf("manager tracks pid %d, but the live process is %d (untracked orphan)", got, live[0])
	}
	if st := inst.State(); st != StateStarting && st != StateRunning {
		t.Fatalf("state = %s with a live process, want starting/running", st)
	}
}

// TestStopThenStartDuringSlowReapKeepsOneTrackedProcess is the P0 regression:
// Stop marks the instance stopped while the old process is still being torn
// down; a Start in that window used to spawn a second process, which the old
// exit goroutine then marked crashed and forgot — leaving an untracked
// llama-server on the GPU and a StopInstance that hung forever.
func TestStopThenStartDuringSlowReapKeepsOneTrackedProcess(t *testing.T) {
	slowReap(t, 300*time.Millisecond)
	rig := newFakeRig(t, "healthy", "f1")
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("f1")

	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 2*time.Second, "first process", func() bool { return len(rig.livePids(t)) == 1 })

	stopErr := make(chan error, 1)
	go func() { stopErr <- m.StopInstance("f1") }()
	waitFor(t, time.Second, "state stopped while old process is exiting", func() bool { return inst.State() == StateStopped })

	if err := within(t, 3*time.Second, "StartInstance", func() error { return m.StartInstance("f1") }); err != nil {
		t.Fatalf("StartInstance: %v", err)
	}
	select {
	case err := <-stopErr:
		if err != nil {
			t.Fatalf("StopInstance: %v", err)
		}
	default:
		t.Fatal("StartInstance returned before the in-flight StopInstance finished")
	}

	time.Sleep(500 * time.Millisecond) // past the old process's delayed exit handling
	assertOneTrackedProcess(t, rig, inst)

	if err := within(t, 3*time.Second, "final StopInstance", func() error { return m.StopInstance("f1") }); err != nil {
		t.Fatal(err)
	}
	if live := rig.livePids(t); len(live) != 0 {
		t.Fatalf("live processes after stop: %v", live)
	}
	if st := inst.State(); st != StateStopped {
		t.Fatalf("state = %s after stop, want stopped", st)
	}
}

// TestConcurrentRestartsLeaveOneTrackedProcess covers the double-clicked
// Restart (or "restart all" overlapping a per-row restart).
func TestConcurrentRestartsLeaveOneTrackedProcess(t *testing.T) {
	slowReap(t, 150*time.Millisecond)
	rig := newFakeRig(t, "healthy", "f1")
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("f1")

	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 2*time.Second, "first process", func() bool { return len(rig.livePids(t)) == 1 })

	var wg sync.WaitGroup
	errs := make(chan error, 3)
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- m.RestartInstance("f1")
		}()
		time.Sleep(20 * time.Millisecond)
	}
	_ = within(t, 5*time.Second, "concurrent restarts", func() error { wg.Wait(); return nil })
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("RestartInstance: %v", err)
		}
	}

	time.Sleep(300 * time.Millisecond)
	assertOneTrackedProcess(t, rig, inst)

	if err := within(t, 3*time.Second, "StopInstance", func() error { return m.StopInstance("f1") }); err != nil {
		t.Fatal(err)
	}
	if live := rig.livePids(t); len(live) != 0 {
		t.Fatalf("live processes after stop: %v", live)
	}
}

// TestStopWaitsForProcessExit pins the contract the P0 fix relies on.
func TestStopWaitsForProcessExit(t *testing.T) {
	slowReap(t, 200*time.Millisecond)
	rig := newFakeRig(t, "healthy", "f1")
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("f1")

	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 2*time.Second, "process", func() bool { return inst.currentPid() != 0 })

	start := time.Now()
	if err := m.StopInstance("f1"); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < 200*time.Millisecond {
		t.Fatalf("StopInstance returned after %s, before the process was reaped", elapsed)
	}
	if pid := inst.currentPid(); pid != 0 {
		t.Fatalf("instance still tracks pid %d after StopInstance", pid)
	}
}

// TestStartRefusesWhilePreviousProcessExiting checks the low-level guard: even
// a caller that bypasses the manager can't start a second process.
func TestStartRefusesWhilePreviousProcessExiting(t *testing.T) {
	slowReap(t, 300*time.Millisecond)
	rig := newFakeRig(t, "healthy", "f1")
	inst := NewInstance(rig.cfg.Instances[0], rig.cfg)

	p, err := inst.Start()
	if err != nil {
		t.Fatal(err)
	}
	inst.mu.Lock()
	inst.state = StateStopped
	_ = p.cmd.Process.Kill()
	inst.mu.Unlock()

	if _, err := inst.Start(); err == nil {
		t.Fatal("Start succeeded while the previous process had not been reaped")
	}
	<-p.exited
	p2, err := inst.Start()
	if err != nil {
		t.Fatalf("Start after reap: %v", err)
	}
	_ = inst.Stop()
	<-p2.exited
}

// TestConcurrentStartsThenStopStaysStopped is the P1 #1 regression: two
// concurrent Starts used to register two supervisors, one of which nobody could
// cancel; after the next crash it restarted the instance behind a successful
// Stop.
func TestConcurrentStartsThenStopStaysStopped(t *testing.T) {
	rig := newFakeRig(t, "crash-after", "f1")
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("f1")

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = m.StartInstance("f1")
		}()
	}
	_ = within(t, 5*time.Second, "concurrent starts", func() error { wg.Wait(); return nil })
	waitFor(t, 2*time.Second, "a crash and restart cycle", func() bool { return inst.RestartCount() > 0 })

	if err := within(t, 3*time.Second, "StopInstance", func() error { return m.StopInstance("f1") }); err != nil {
		t.Fatal(err)
	}
	count := inst.RestartCount()
	time.Sleep(600 * time.Millisecond) // many restart_delay (50ms) + crash (100ms) cycles
	if st := inst.State(); st != StateStopped {
		t.Fatalf("state = %s after Stop, want stopped (orphaned supervisor revived it)", st)
	}
	if got := inst.RestartCount(); got != count {
		t.Fatalf("restart count moved %d -> %d after Stop", count, got)
	}
	if m.isSupervised("f1") {
		t.Fatal("instance still supervised after Stop")
	}
	if live := rig.livePids(t); len(live) != 0 {
		t.Fatalf("live processes after stop: %v", live)
	}
}

// TestRemoveDoesNotDetachReplacementSupervisor: supervisors are keyed by
// instance, so tearing down an old instance can't cancel the supervisor of a
// new one registered under the same name (the PUT edit flow).
func TestRemoveDoesNotDetachReplacementSupervisor(t *testing.T) {
	rig := newFakeRig(t, "healthy", "f1")
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	old := m.Get("f1")

	m.mu.Lock()
	delete(m.byName, "f1")
	m.instances = nil
	m.mu.Unlock()
	if err := m.AddInstance(rig.cfg.Instances[0]); err != nil {
		t.Fatal(err)
	}
	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 2*time.Second, "replacement process", func() bool { return len(rig.livePids(t)) == 1 })

	old.opMu.Lock()
	m.detachSupervisor(old)
	_ = old.Stop()
	old.opMu.Unlock()

	if !m.isSupervised("f1") {
		t.Fatal("tearing down the old instance detached the replacement's supervisor")
	}
	if len(rig.livePids(t)) != 1 {
		t.Fatal("replacement process was stopped")
	}
}

// TestRandomLifecycleOpsNeverOverlapProcesses hammers one instance with
// concurrent start/stop/restart and asserts, by sampling the real process
// table, that two llama-servers never run for the same instance.
func TestRandomLifecycleOpsNeverOverlapProcesses(t *testing.T) {
	slowReap(t, 30*time.Millisecond)
	rig := newFakeRig(t, "healthy", "f1")
	m := NewManager(rig.cfg)
	defer m.Shutdown()

	stop := make(chan struct{})
	var maxLive int
	var samplerDone sync.WaitGroup
	samplerDone.Add(1)
	go func() {
		defer samplerDone.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			if n := len(rig.livePids(t)); n > maxLive {
				maxLive = n
			}
			time.Sleep(2 * time.Millisecond)
		}
	}()

	ops := []func(string) error{m.StartInstance, m.StopInstance, m.RestartInstance}
	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 12; i++ {
				_ = ops[(g+i)%len(ops)]("f1")
			}
		}(g)
	}
	_ = within(t, 30*time.Second, "random lifecycle ops", func() error { wg.Wait(); return nil })
	if err := within(t, 3*time.Second, "final StopInstance", func() error { return m.StopInstance("f1") }); err != nil {
		t.Fatal(err)
	}
	close(stop)
	samplerDone.Wait()

	if maxLive > 1 {
		t.Fatalf("observed %d live processes for one instance at the same time", maxLive)
	}
	if live := rig.livePids(t); len(live) != 0 {
		t.Fatalf("live processes after final stop: %v", live)
	}
}

// TestStartFailureReturnsErrorAndMarksCrashed is part of the P1 #4 regression:
// a first start with a broken binary used to return "ok" and leave the
// instance looking stopped, with no error anywhere.
func TestStartFailureReturnsErrorAndMarksCrashed(t *testing.T) {
	rig := newFakeRig(t, "healthy", "f1")
	rig.cfg.ServerBin = filepath.Join(t.TempDir(), "missing-llama-server")
	m := NewManager(rig.cfg)
	defer m.Shutdown()

	if err := m.StartInstance("f1"); err == nil {
		t.Fatal("StartInstance succeeded with a missing binary")
	}
	st := m.Get("f1").Status()
	if st.State != StateCrashed {
		t.Errorf("state = %s, want crashed", st.State)
	}
	if !strings.HasPrefix(st.LastError, "start failed:") {
		t.Errorf("lastError = %q, want a 'start failed:' reason", st.LastError)
	}
	if m.isSupervised("f1") {
		t.Error("a supervisor was left running for an instance that never started")
	}
}

// TestRestartAttemptStartFailureEndsCrashed is the other half of P1 #4: when a
// restart attempt couldn't exec the binary, supervision silently ended and the
// instance stayed "restarting" forever.
func TestRestartAttemptStartFailureEndsCrashed(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "crashing.sh")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	rig := newFakeRig(t, "healthy", "f1")
	rig.cfg.ServerBin = bin
	rig.cfg.RestartDelay = duration{150 * time.Millisecond}
	rig.cfg.MaxRestarts = 3
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("f1")

	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 2*time.Second, "restarting", func() bool { return inst.State() == StateRestarting })
	if err := os.Remove(bin); err != nil {
		t.Fatal(err)
	}

	waitFor(t, 3*time.Second, "supervisor to give up", func() bool { return !m.isSupervised("f1") })
	st := inst.Status()
	if st.State != StateCrashed {
		t.Fatalf("state = %s, want crashed", st.State)
	}
	if !strings.HasPrefix(st.LastError, "start failed:") {
		t.Fatalf("lastError = %q, want a 'start failed:' reason", st.LastError)
	}
}

// TestRestartAttemptRetriesUntilBinaryReturns: a start failure inside the
// restart loop is retried like a crash, so a binary that is briefly missing
// (being rebuilt) recovers without operator action.
func TestRestartAttemptRetriesUntilBinaryReturns(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "server.sh")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	rig := newFakeRig(t, "healthy", "f1")
	rig.cfg.ServerBin = bin
	rig.cfg.RestartDelay = duration{100 * time.Millisecond}
	rig.cfg.MaxRestarts = 20
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("f1")

	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 2*time.Second, "restarting", func() bool { return inst.State() == StateRestarting })
	if err := os.Remove(bin); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 3*time.Second, "a failed start attempt", func() bool {
		return strings.HasPrefix(inst.Status().LastError, "start failed:")
	})

	tmp := bin + ".tmp"
	if err := os.WriteFile(tmp, []byte("#!/bin/sh\nexec sleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(tmp, bin); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 3*time.Second, "recovery once the binary is back", func() bool { return inst.State() == StateStarting })
	if !m.isSupervised("f1") {
		t.Fatal("recovered instance is not supervised")
	}
}
