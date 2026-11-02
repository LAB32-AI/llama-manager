//go:build unix

package main

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func pidAlive(pid int) bool { return syscall.Kill(pid, 0) == nil }

func readPidFile(t *testing.T, path string) int {
	t.Helper()
	var pid int
	waitFor(t, 2*time.Second, "pid file "+path, func() bool {
		b, err := os.ReadFile(path)
		if err != nil {
			return false
		}
		pid, err = strconv.Atoi(strings.TrimSpace(string(b)))
		return err == nil && pid > 0
	})
	return pid
}

// TestStopIsGraceful (P1 #6): Stop sends SIGTERM first so llama-server can
// finish in-flight requests and release its GPU contexts cleanly.
func TestStopIsGraceful(t *testing.T) {
	rig := newFakeRig(t, "healthy", "f1")
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("f1")

	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 2*time.Second, "fake server ready", func() bool { return len(rig.livePids(t)) == 1 })
	pid := inst.currentPid()
	if err := m.StopInstance("f1"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(rig.pidDir, strconv.Itoa(pid)+".term")); err != nil {
		t.Fatal("process did not receive SIGTERM before exiting")
	}
}

// TestStopEscalatesToSIGKILL: a server that ignores SIGTERM is killed after the
// grace period, and Stop still returns.
func TestStopEscalatesToSIGKILL(t *testing.T) {
	old := stopGrace
	stopGrace = 200 * time.Millisecond
	defer func() { stopGrace = old }()

	rig := newFakeRig(t, "ignore-term", "f1")
	m := NewManager(rig.cfg)
	defer m.Shutdown()

	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 2*time.Second, "process", func() bool { return len(rig.livePids(t)) == 1 })
	if err := within(t, 3*time.Second, "StopInstance", func() error { return m.StopInstance("f1") }); err != nil {
		t.Fatal(err)
	}
	if live := rig.livePids(t); len(live) != 0 {
		t.Fatalf("SIGTERM-ignoring process survived Stop: %v", live)
	}
}

// TestStopKillsWrapperChildren: with server_bin a wrapper script that doesn't
// exec, killing only the shell used to orphan the real server.
func TestStopKillsWrapperChildren(t *testing.T) {
	dir := t.TempDir()
	pidFile := filepath.Join(dir, "child.pid")
	bin := filepath.Join(dir, "wrapper.sh")
	if err := writeExec(bin, "#!/bin/sh\nsleep 300 &\necho $! > "+pidFile+"\nwait\n"); err != nil {
		t.Fatal(err)
	}
	rig := newFakeRig(t, "healthy", "f1")
	rig.cfg.ServerBin = bin
	m := NewManager(rig.cfg)
	defer m.Shutdown()

	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	child := readPidFile(t, pidFile)
	if err := within(t, 3*time.Second, "StopInstance", func() error { return m.StopInstance("f1") }); err != nil {
		t.Fatal(err)
	}
	waitFor(t, time.Second, "wrapper's child to die", func() bool { return !pidAlive(child) })
}

// TestCrashedWrapperChildrenAreReaped: if the wrapper itself dies, whatever it
// left in its process group is killed, even when it still holds the log pipes.
func TestCrashedWrapperChildrenAreReaped(t *testing.T) {
	old := outputWaitDelay
	outputWaitDelay = 200 * time.Millisecond
	defer func() { outputWaitDelay = old }()

	dir := t.TempDir()
	pidFile := filepath.Join(dir, "child.pid")
	bin := filepath.Join(dir, "wrapper.sh")
	if err := writeExec(bin, "#!/bin/sh\nsleep 300 &\necho $! > "+pidFile+"\nexit 1\n"); err != nil {
		t.Fatal(err)
	}
	rig := newFakeRig(t, "healthy", "f1")
	rig.cfg.ServerBin = bin
	rig.cfg.RestartDelay = duration{10 * time.Second}
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("f1")

	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	child := readPidFile(t, pidFile)
	waitFor(t, 3*time.Second, "crash detected despite the held pipes", func() bool { return inst.State() == StateRestarting })
	waitFor(t, time.Second, "orphaned child to be killed", func() bool { return !pidAlive(child) })
}

func TestChildRunsInOwnProcessGroup(t *testing.T) {
	rig := newFakeRig(t, "healthy", "f1")
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("f1")

	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 2*time.Second, "process", func() bool { return inst.currentPid() != 0 })
	pid := inst.currentPid()
	pgid, err := syscall.Getpgid(pid)
	if err != nil {
		t.Fatal(err)
	}
	if pgid != pid {
		t.Fatalf("child pgid = %d, want its own group %d", pgid, pid)
	}
}

// TestShutdownStopsInstancesInParallel: each Stop can take up to stopGrace, so
// stopping N models one after another made SIGTERM handling take N x grace.
func TestShutdownStopsInstancesInParallel(t *testing.T) {
	old := stopGrace
	stopGrace = 400 * time.Millisecond
	defer func() { stopGrace = old }()

	rig := newFakeRig(t, "ignore-term", "a", "b", "c")
	m := NewManager(rig.cfg)
	for _, n := range []string{"a", "b", "c"} {
		if err := m.StartInstance(n); err != nil {
			t.Fatal(err)
		}
	}
	waitFor(t, 3*time.Second, "three processes", func() bool { return len(rig.livePids(t)) == 3 })

	start := time.Now()
	m.Shutdown()
	if elapsed := time.Since(start); elapsed > 1100*time.Millisecond {
		t.Fatalf("Shutdown took %s for 3 instances with a %s grace; stops are not parallel", elapsed, stopGrace)
	}
	if live := rig.livePids(t); len(live) != 0 {
		t.Fatalf("live processes after Shutdown: %v", live)
	}
}
