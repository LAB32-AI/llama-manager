package main

import (
	"testing"
	"time"
)

// TestStopDuringInFlightHealthCheckEndsStopped is the P1 #2 regression: a
// health check in flight when the user pressed Stop came back failed (the
// server was gone) and the loop, acting on the state it read before the check,
// demoted the instance to "unhealthy" or tagged it with a false crash error.
func TestStopDuringInFlightHealthCheckEndsStopped(t *testing.T) {
	for i := 0; i < 3; i++ {
		rig := newFakeRig(t, "slow-health", "f1")
		rig.cfg.HealthCheckInterval = duration{40 * time.Millisecond}
		rig.cfg.HealthCheckTimeout = duration{3 * time.Second}
		rig.cfg.UnhealthyAfter = 1
		m := NewManager(rig.cfg)
		inst := m.Get("f1")

		if err := m.StartInstance("f1"); err != nil {
			t.Fatal(err)
		}
		waitFor(t, 3*time.Second, "running", func() bool { return inst.State() == StateRunning })
		time.Sleep(150 * time.Millisecond) // a slow (2s) check is now in flight

		if err := m.StopInstance("f1"); err != nil {
			t.Fatal(err)
		}
		time.Sleep(300 * time.Millisecond) // let the aborted check return

		st := inst.Status()
		if st.State != StateStopped {
			t.Fatalf("round %d: state = %s after Stop, want stopped", i, st.State)
		}
		if st.LastError != "" {
			t.Fatalf("round %d: lastError = %q after a clean Stop", i, st.LastError)
		}
		m.Shutdown()
	}
}

func TestTransitionRequiresCurrentRunAndState(t *testing.T) {
	inst := NewInstance(InstanceConf{Name: "x"}, &Config{})
	p := &proc{}
	inst.proc = p
	inst.state = StateRunning

	if inst.transition(&proc{}, StateRunning, StateUnhealthy) {
		t.Fatal("transition applied for a run that is not current")
	}
	if inst.transition(p, StateStarting, StateRunning) {
		t.Fatal("transition applied from the wrong state")
	}
	if !inst.transition(p, StateRunning, StateUnhealthy) || inst.State() != StateUnhealthy {
		t.Fatal("valid transition was not applied")
	}
	inst.proc = nil
	inst.state = StateCrashed
	if !inst.transition(nil, StateCrashed, StateRestarting) {
		t.Fatal("transition with no process was not applied")
	}
}

// TestHealthCheckDoesNotKillWhileStarting is the P1 #3 regression: a model that
// takes longer than unhealthy_after x interval to load (or an -hf download) was
// killed from the starting state and crash-looped until max_restarts.
func TestHealthCheckDoesNotKillWhileStarting(t *testing.T) {
	rig := newFakeRig(t, "never-ready", "loading")
	rig.cfg.HealthCheckInterval = duration{30 * time.Millisecond}
	rig.cfg.HealthCheckTimeout = duration{100 * time.Millisecond}
	rig.cfg.UnhealthyAfter = 2
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("loading")

	if err := m.StartInstance("loading"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 2*time.Second, "process", func() bool { return inst.currentPid() != 0 })
	pid := inst.currentPid()

	time.Sleep(700 * time.Millisecond) // ~20 failed checks, 10x unhealthy_after
	if st := inst.State(); st != StateStarting {
		t.Fatalf("state = %s, want starting", st)
	}
	if got := inst.RestartCount(); got != 0 {
		t.Fatalf("loading instance was restarted %d time(s)", got)
	}
	if got := inst.currentPid(); got != pid {
		t.Fatalf("loading process was replaced (%d -> %d)", pid, got)
	}
}

// TestReadyInstanceMarkedRunningBeforeFullInterval: with the default 30s
// interval, a model that finished loading used to stay "starting" (chat 503)
// for up to 30s; while starting the loop now polls at startingPollInterval.
func TestReadyInstanceMarkedRunningBeforeFullInterval(t *testing.T) {
	old := startingPollInterval
	startingPollInterval = 50 * time.Millisecond
	defer func() { startingPollInterval = old }()

	rig := newFakeRig(t, "healthy", "f1")
	rig.cfg.HealthCheckInterval = duration{30 * time.Second}
	m := NewManager(rig.cfg)
	defer m.Shutdown()

	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 2*time.Second, "running well before the 30s interval", func() bool { return m.Get("f1").State() == StateRunning })
}

func TestNextHealthDelay(t *testing.T) {
	if got := nextHealthDelay(30*time.Second, StateStarting); got != startingPollInterval {
		t.Errorf("starting: got %s, want %s", got, startingPollInterval)
	}
	if got := nextHealthDelay(30*time.Second, StateRunning); got != 30*time.Second {
		t.Errorf("running: got %s, want 30s", got)
	}
	if got := nextHealthDelay(time.Second, StateStarting); got != time.Second {
		t.Errorf("short interval while starting: got %s, want 1s", got)
	}
}
