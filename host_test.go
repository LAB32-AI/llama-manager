package main

import (
	"net"
	"net/url"
	"strconv"
	"testing"
	"time"
)

func TestConnectHost(t *testing.T) {
	for bind, want := range map[string]string{"": "127.0.0.1", "0.0.0.0": "127.0.0.1", "::": "127.0.0.1", "::1": "::1", "10.0.0.5": "10.0.0.5"} {
		if got := connectHost(bind); got != want {
			t.Errorf("connectHost(%q) = %q, want %q", bind, got, want)
		}
	}
}

// TestIPv6HostReachesRunning (P2): URLs were built as "http://%s:%d", which is
// invalid for an IPv6 host ("http://::1:9090"), so every health check failed
// and the instance was killed in a loop.
func TestIPv6HostReachesRunning(t *testing.T) {
	ln, err := net.Listen("tcp", "[::1]:0")
	if err != nil {
		t.Skip("no IPv6 loopback")
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()

	old := startingPollInterval
	startingPollInterval = 50 * time.Millisecond
	defer func() { startingPollInterval = old }()

	rig := newFakeRig(t, "healthy", "v6")
	rig.cfg.Host = "::1"
	rig.cfg.Instances[0].Port = port
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("v6")

	if err := m.StartInstance("v6"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 3*time.Second, "running over IPv6", func() bool { return inst.State() == StateRunning })
	u, err := url.Parse(inst.baseURL())
	if err != nil || u.Hostname() != "::1" || u.Port() != strconv.Itoa(port) {
		t.Fatalf("baseURL = %q (%v)", inst.baseURL(), err)
	}
}

// TestHostChangeDoesNotRetargetRunningInstance (P2): health checks re-read the
// host setting on every call, so changing it pointed them at an address the
// running server wasn't bound to, and it was killed ~unhealthy_after checks
// later. The new host now applies from the next start.
func TestHostChangeDoesNotRetargetRunningInstance(t *testing.T) {
	rig := newFakeRig(t, "healthy", "f1")
	rig.cfg.HealthCheckInterval = duration{30 * time.Millisecond}
	rig.cfg.HealthCheckTimeout = duration{200 * time.Millisecond}
	rig.cfg.UnhealthyAfter = 2
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("f1")

	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 3*time.Second, "running", func() bool { return inst.State() == StateRunning })
	pid := inst.currentPid()

	if err := rig.cfg.UpdateSettings(Settings{Host: "192.0.2.1"}); err != nil { // TEST-NET-1, unreachable
		t.Fatal(err)
	}
	time.Sleep(500 * time.Millisecond) // ~15 health checks
	if st := inst.State(); st != StateRunning || inst.currentPid() != pid {
		t.Fatalf("running instance disturbed by a host change: state=%s pid %d -> %d", st, pid, inst.currentPid())
	}

	if err := m.StopInstance("f1"); err != nil {
		t.Fatal(err)
	}
	if u, _ := url.Parse(inst.baseURL()); u.Hostname() != "192.0.2.1" {
		t.Fatalf("stopped instance baseURL = %q, want the new host", inst.baseURL())
	}
}
