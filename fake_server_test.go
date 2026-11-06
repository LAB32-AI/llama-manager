package main

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

// The test binary doubles as a fake llama-server: TestMain sets fakeServerEnv
// for every child the tests spawn, so a Config whose ServerBin is the test
// binary launches runFakeServer instead of the test suite. Behaviour is chosen
// with --fake-* flags passed through the instance's extra_args.
const fakeServerEnv = "LLAMA_MANAGER_FAKE_SERVER"

func TestMain(m *testing.M) {
	if os.Getenv(fakeServerEnv) == "1" {
		runFakeServer(os.Args[1:])
		os.Exit(0)
	}
	os.Setenv(fakeServerEnv, "1")
	// A -race binary sleeps 1s on a clean exit; skip that for the fake servers.
	os.Setenv("GORACE", strings.TrimSpace(os.Getenv("GORACE")+" atexit_sleep_ms=0"))
	os.Exit(m.Run())
}

func fakeArg(args []string, name, def string) string {
	for i := 0; i < len(args)-1; i++ {
		if args[i] == name {
			return args[i+1]
		}
	}
	return def
}

// runFakeServer modes:
//
//	healthy            /health always 200
//	never-ready        /health always 503 (model still loading)
//	wedge-after-ready  first /health 200, later requests hang forever
//	slow-health        first /health 200, later requests take 2s
//	long-line          print a 200 KiB line, then "after-long-line", then healthy
//	ignore-term        ignore SIGTERM, otherwise healthy
//	crash-after        healthy, then exit 1 after --fake-crash-after (default 100ms)
func runFakeServer(args []string) {
	if len(args) > 0 && (args[0] == "funnel" || args[0] == "status") {
		runFakeTailscale(args)
		return
	}
	if port := fakeArg(args, "--listen-port", ""); port != "" {
		runFakeSDServer(fakeArg(args, "--listen-ip", "127.0.0.1"), port)
		return
	}
	if conf := fakeArg(args, "--config", ""); conf != "" {
		runFakeAudioCpp(conf)
		return
	}
	mode := fakeArg(args, "--fake-mode", "healthy")
	port := fakeArg(args, "--port", "0")
	pidDir := fakeArg(args, "--fake-pid-dir", "")
	termDelay, _ := time.ParseDuration(fakeArg(args, "--fake-term-delay", "0s"))

	sigCh := make(chan os.Signal, 1)
	if mode == "ignore-term" {
		signal.Ignore(syscall.SIGTERM)
	} else {
		signal.Notify(sigCh, syscall.SIGTERM)
	}
	go func() {
		<-sigCh
		if pidDir != "" {
			_ = os.WriteFile(filepath.Join(pidDir, strconv.Itoa(os.Getpid())+".term"), nil, 0o644)
		}
		time.Sleep(termDelay)
		os.Exit(0)
	}()
	// Written only once signal handling is set up, so tests that wait for the
	// pid file can rely on SIGTERM being observed.
	if pidDir != "" {
		_ = os.WriteFile(filepath.Join(pidDir, strconv.Itoa(os.Getpid())), nil, 0o644)
	}

	if mode == "crash-after" {
		d, _ := time.ParseDuration(fakeArg(args, "--fake-crash-after", "100ms"))
		go func() {
			time.Sleep(d)
			os.Exit(1)
		}()
	}

	if mode == "long-line" {
		fmt.Fprintln(os.Stderr, strings.Repeat("x", 200*1024))
		fmt.Fprintln(os.Stderr, "after-long-line")
	}

	var hits atomic.Int64
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		n := hits.Add(1)
		switch mode {
		case "never-ready":
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		case "wedge-after-ready":
			if n > 1 {
				select {}
			}
		case "slow-health":
			if n > 1 {
				time.Sleep(2 * time.Second)
			}
		}
		w.WriteHeader(http.StatusOK)
	})
	ln, err := net.Listen("tcp", net.JoinHostPort(fakeArg(args, "--host", "127.0.0.1"), port))
	if err != nil {
		fmt.Fprintln(os.Stderr, "fake server listen:", err)
		os.Exit(1)
	}
	_ = http.Serve(ln, mux)
}

func fakeServerBin(t *testing.T) string {
	t.Helper()
	bin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return bin
}

func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
}

type fakeRig struct {
	cfg    *Config
	pidDir string
}

// newFakeRig builds a Config with one instance per name, each running the fake
// server in the given mode on its own free port.
func newFakeRig(t *testing.T, mode string, names ...string) *fakeRig {
	t.Helper()
	dir := t.TempDir()
	pidDir := filepath.Join(dir, "pids")
	if err := os.Mkdir(pidDir, 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{
		ServerBin:           fakeServerBin(t),
		RestartDelay:        duration{50 * time.Millisecond},
		MaxRestarts:         5,
		HealthCheckInterval: duration{30 * time.Second},
		HealthCheckTimeout:  duration{time.Second},
		GPUBackend:          "metal",
		Host:                "127.0.0.1",
		NGL:                 1,
		ContextLength:       128,
		path:                filepath.Join(dir, "config.yaml"),
	}
	for _, name := range names {
		extra := []string{"--fake-mode", mode, "--fake-pid-dir", pidDir}
		cfg.Instances = append(cfg.Instances, InstanceConf{
			Name: name, Model: "/dev/null", Port: freePort(t), GPUIDs: []int{0}, ExtraArgs: &extra,
		})
	}
	return &fakeRig{cfg: cfg, pidDir: pidDir}
}

// livePids returns the fake servers that have been started and not yet reaped.
func (r *fakeRig) livePids(t *testing.T) []int {
	t.Helper()
	entries, err := os.ReadDir(r.pidDir)
	if err != nil {
		t.Fatal(err)
	}
	var live []int
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		if syscall.Kill(pid, 0) == nil {
			live = append(live, pid)
		}
	}
	return live
}

func waitFor(t *testing.T, timeout time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out after %s waiting for %s", timeout, what)
}

// within fails the test if fn doesn't return within timeout.
func within(t *testing.T, timeout time.Duration, what string, fn func() error) error {
	t.Helper()
	ch := make(chan error, 1)
	go func() { ch <- fn() }()
	select {
	case err := <-ch:
		return err
	case <-time.After(timeout):
		t.Fatalf("%s did not return within %s", what, timeout)
		return nil
	}
}

func (m *Manager) isSupervised(name string) bool {
	inst := m.Get(name)
	return inst != nil && m.hasSupervisor(inst)
}

func (inst *Instance) currentPid() int {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	if inst.proc == nil {
		return 0
	}
	return inst.proc.cmd.Process.Pid
}

func writeExec(path, script string) error {
	return os.WriteFile(path, []byte(script), 0o755)
}

// Fake tailscale CLI: the mode comes from fakeTailscaleMode in the
// environment; each run appends its arguments to <fakeTailscaleDir>/args and
// a stop signal leaves <fakeTailscaleDir>/stopped.
const (
	fakeTailscaleMode = "LLAMA_MANAGER_FAKE_TAILSCALE"
	fakeTailscaleDir  = "LLAMA_MANAGER_FAKE_TAILSCALE_DIR"
	fakeTailnetHost   = "fake-host.tail0000.ts.net"
)

// runFakeTailscale modes:
//
//	ok        prints the public URL like tailscale 1.104, runs until stopped
//	approval  prints the link to enable Funnel for the tailnet, runs
//	denied    prints Access denied and exits 1 (not root, no --operator)
//	quiet     prints nothing, runs (an output format we don't know)
//	crash     exits 1 at once
func runFakeTailscale(args []string) {
	if args[0] == "status" {
		fmt.Printf(`{"Self":{"DNSName":"%s.","Online":true}}`+"\n", fakeTailnetHost)
		return
	}
	dir := os.Getenv(fakeTailscaleDir)
	if dir != "" {
		f, err := os.OpenFile(filepath.Join(dir, "args"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err == nil {
			fmt.Fprintln(f, strings.Join(args, " "))
			f.Close()
		}
	}
	if dir != "" {
		_ = os.WriteFile(filepath.Join(dir, "pid"), []byte(strconv.Itoa(os.Getpid())), 0o644)
	}
	target := args[len(args)-1]
	switch os.Getenv(fakeTailscaleMode) {
	case "approval":
		fmt.Print("Funnel is not enabled on your tailnet.\nTo enable, visit:\n\n         https://login.tailscale.com/f/funnel?node=fakeNode\n\n")
	case "denied":
		fmt.Fprintln(os.Stderr, "sending serve config: Access denied: serve config denied")
		os.Exit(1)
	case "crash":
		fmt.Fprintln(os.Stderr, "boom")
		os.Exit(1)
	case "quiet":
	default:
		fmt.Printf("Available on the internet:\n\nhttps://%s/v1\n|-- proxy %s\n\nPress Ctrl+C to exit.\n", fakeTailnetHost, target)
	}
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
	<-sig
	if dir != "" {
		_ = os.WriteFile(filepath.Join(dir, "stopped"), nil, 0o644)
	}
	os.Exit(0)
}
