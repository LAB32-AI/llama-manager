package main

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// writeRunConfig writes a config with one auto_start fake instance.
func writeRunConfig(t *testing.T, managerPort int) (path string, rig *fakeRig) {
	t.Helper()
	hermeticCaches(t) // run() sweeps the HF cache on start
	rig = newFakeRig(t, "healthy", "boot")
	body := fmt.Sprintf(`server_bin: %s
manager_port: %d
gpu_backend: metal
host: 127.0.0.1
restart_delay: 50ms
instances:
  - name: boot
    model: /dev/null
    port: %d
    gpu_ids: [0]
    auto_start: true
    extra_args: ["--fake-mode", "healthy", "--fake-pid-dir", %q]
`, rig.cfg.ServerBin, managerPort, rig.cfg.Instances[0].Port, rig.pidDir)
	path = filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path, rig
}

// TestRunFailsBeforeSpawningWhenPortTaken (P2): instances were started before
// the port was bound, so a taken port made main exit with auto_start
// llama-servers still running (and, outside systemd, orphaned).
func TestRunFailsBeforeSpawningWhenPortTaken(t *testing.T) {
	ln, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	path, rig := writeRunConfig(t, ln.Addr().(*net.TCPAddr).Port)

	if err := run(path, make(chan os.Signal, 1), nil); err == nil {
		t.Fatal("run succeeded on a taken port")
	}
	time.Sleep(200 * time.Millisecond)
	if live := rig.livePids(t); len(live) != 0 {
		t.Fatalf("llama-server spawned despite the bind failure: %v", live)
	}
}

// TestRunShutsDownCleanlyOnSignal drives the real entrypoint: boot starts the
// auto_start instance, a signal stops HTTP first, then every child.
func TestRunShutsDownCleanlyOnSignal(t *testing.T) {
	path, rig := writeRunConfig(t, freePort(t))
	stop := make(chan os.Signal, 1)
	addrCh := make(chan string, 1)
	done := make(chan error, 1)
	go func() { done <- run(path, stop, func(a string) { addrCh <- a }) }()

	var addr string
	select {
	case addr = <-addrCh:
	case err := <-done:
		t.Fatalf("run exited early: %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("run never became ready")
	}
	_, port, _ := net.SplitHostPort(addr)
	resp, err := http.Get("http://127.0.0.1:" + port + "/api/status")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	waitFor(t, 3*time.Second, "auto_start instance", func() bool { return len(rig.livePids(t)) == 1 })

	stop <- syscall.SIGTERM
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("run: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("run did not return after the signal")
	}
	if live := rig.livePids(t); len(live) != 0 {
		t.Fatalf("children alive after shutdown: %v", live)
	}
	if _, err := http.Get("http://127.0.0.1:" + port + "/api/status"); err == nil {
		t.Fatal("HTTP still served after shutdown")
	}
}

// TestNoStartAfterShutdown: a /start arriving while the manager shuts down
// used to spawn a process just before exit (and call wg.Add after wg.Wait).
func TestNoStartAfterShutdown(t *testing.T) {
	rig := newFakeRig(t, "healthy", "f1")
	m := NewManager(rig.cfg)
	m.Shutdown()
	if err := m.StartInstance("f1"); err == nil {
		t.Fatal("StartInstance succeeded after Shutdown")
	}
	time.Sleep(100 * time.Millisecond)
	if live := rig.livePids(t); len(live) != 0 {
		t.Fatalf("process spawned after Shutdown: %v", live)
	}
	m.Shutdown() // idempotent
}
