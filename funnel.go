package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/user"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"
)

// funnelHeader is set by tailscaled on every request that came in through
// Tailscale Funnel; it drops any copy the client sent. A LAN client that
// sends it only gets the stricter rules of a request from the internet.
const funnelHeader = "Tailscale-Funnel-Request"

const (
	defaultPublicConcurrent = 8
	macTailscaleBin         = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
)

// funnelPorts are the only ports Funnel listens on.
var funnelPorts = []int{443, 8443, 10000}

// PublicAPIConf opens /v1 (and nothing else) to the internet through
// Tailscale Funnel, at https://<machine>.<tailnet>.ts.net/v1.
type PublicAPIConf struct {
	TailscaleFunnel bool `yaml:"tailscale_funnel" json:"tailscale_funnel"`
	// TailscaleBin is the tailscale CLI; empty looks in PATH, then in the
	// macOS app.
	TailscaleBin string `yaml:"tailscale_bin,omitempty" json:"tailscale_bin,omitempty"`
	HTTPSPort    int    `yaml:"https_port,omitempty" json:"https_port,omitempty"`
	// MaxConcurrent caps requests from the internet running at once, over
	// all keys, so outside users can't take every slot.
	MaxConcurrent int `yaml:"max_concurrent,omitempty" json:"max_concurrent,omitempty"`
}

func (p PublicAPIConf) check() error {
	var errs []error
	if p.HTTPSPort != 0 && !slices.Contains(funnelPorts, p.HTTPSPort) {
		errs = append(errs, fmt.Errorf("public_api.https_port must be 443, 8443 or 10000 (the ports Funnel offers)"))
	}
	if p.MaxConcurrent < 0 {
		errs = append(errs, fmt.Errorf("public_api.max_concurrent must be >= 0"))
	}
	return errors.Join(errs...)
}

func (p PublicAPIConf) port() int {
	if p.HTTPSPort == 0 {
		return 443
	}
	return p.HTTPSPort
}

func (p PublicAPIConf) maxConcurrent() int {
	if p.MaxConcurrent == 0 {
		return defaultPublicConcurrent
	}
	return p.MaxConcurrent
}

type publicCtxKey struct{}

// publicFrom says whether r came from the internet, and from which address.
func publicFrom(r *http.Request) (bool, string) {
	ip, ok := r.Context().Value(publicCtxKey{}).(string)
	return ok, ip
}

// clientIP is the address a request came from. Behind Funnel the connection
// is tailscaled's, from loopback, and tailscaled sets X-Forwarded-For to the
// real client; the header is believed only then.
func clientIP(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() && r.Header.Get(funnelHeader) != "" {
		xff := r.Header.Values("X-Forwarded-For")
		if len(xff) > 0 {
			parts := strings.Split(xff[len(xff)-1], ",")
			if fwd := strings.TrimSpace(parts[len(parts)-1]); net.ParseIP(fwd) != nil {
				return fwd
			}
		}
	}
	return host
}

// Funnel states, as /api/status reports them.
const (
	funnelNoKey    = "no_key"
	funnelStarting = "starting"
	funnelApproval = "needs_approval"
	funnelOn       = "on"
	funnelError    = "error"
)

type funnelStatus struct {
	State      string `json:"state"`
	URL        string `json:"url,omitempty"`
	ApproveURL string `json:"approve_url,omitempty"`
	Error      string `json:"error,omitempty"`
	Port       int    `json:"port"`
	// MaxConcurrent is public_api.max_concurrent, for the UI.
	MaxConcurrent int `json:"max_concurrent"`
}

// Timings, vars so tests can shorten them.
var (
	funnelKeyPoll      = 3 * time.Second
	funnelConfirmAfter = 10 * time.Second
	funnelRetryFirst   = 2 * time.Second
	funnelRetryMax     = time.Minute
	funnelStopWait     = 5 * time.Second
)

var errFunnelNoKeys = errors.New("no API keys")

// funnel runs `tailscale funnel` in the foreground as a child, so /v1 is on
// the internet exactly while the manager runs: when the child exits, even
// killed, tailscaled drops its config, and nothing is left behind.
type funnel struct {
	bin     string
	port    int
	maxConc int
	target  string
	hasKeys func() bool

	mu       sync.Mutex
	st       funnelStatus
	live     bool   // "Available on the internet" seen; the URL follows
	lastLine string // the CLI's last output, for the error when it exits

	cancel context.CancelFunc
	done   chan struct{}
}

// newFunnel is nil unless public_api.tailscale_funnel is on.
func newFunnel(c PublicAPIConf, managerPort int, hasKeys func() bool) *funnel {
	if !c.TailscaleFunnel {
		return nil
	}
	return &funnel{
		bin:     c.TailscaleBin,
		port:    c.port(),
		maxConc: c.maxConcurrent(),
		target:  fmt.Sprintf("http://127.0.0.1:%d/v1", managerPort),
		hasKeys: hasKeys,
		st:      funnelStatus{State: funnelStarting, Port: c.port()},
	}
}

func (f *funnel) Start() {
	ctx, cancel := context.WithCancel(context.Background())
	f.cancel, f.done = cancel, make(chan struct{})
	go f.run(ctx)
}

// Close stops the child, which takes /v1 off the internet.
func (f *funnel) Close() {
	if f == nil || f.cancel == nil {
		return
	}
	f.cancel()
	select {
	case <-f.done:
	case <-time.After(funnelStopWait + 2*time.Second):
	}
}

func (f *funnel) Status() *funnelStatus {
	if f == nil {
		return nil
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	s := f.st
	s.MaxConcurrent = f.maxConc
	return &s
}

func (f *funnel) set(fn func(s *funnelStatus)) {
	f.mu.Lock()
	fn(&f.st)
	f.mu.Unlock()
}

func sleepCtx(ctx context.Context, d time.Duration) bool {
	select {
	case <-ctx.Done():
		return false
	case <-time.After(d):
		return true
	}
}

func (f *funnel) run(ctx context.Context) {
	defer close(f.done)
	backoff := funnelRetryFirst
	warned := false
	for ctx.Err() == nil {
		if !f.hasKeys() {
			if !warned {
				log.Printf("[funnel] waiting for an API key: /v1 goes on the internet only with keys (Settings > OpenAI API)")
				warned = true
			}
			f.set(func(s *funnelStatus) { *s = funnelStatus{State: funnelNoKey, Port: f.port} })
			if !sleepCtx(ctx, funnelKeyPoll) {
				return
			}
			continue
		}
		warned = false
		started := time.Now()
		err := f.runOnce(ctx)
		if ctx.Err() != nil {
			return
		}
		if errors.Is(err, errFunnelNoKeys) {
			continue
		}
		if time.Since(started) > time.Minute {
			backoff = funnelRetryFirst
		}
		f.set(func(s *funnelStatus) { s.State, s.Error, s.URL = funnelError, err.Error(), "" })
		log.Printf("[funnel] %v; trying again in %s", err, backoff)
		if !sleepCtx(ctx, backoff) {
			return
		}
		backoff = min(backoff*2, funnelRetryMax)
	}
}

func (f *funnel) resolveBin() (string, error) {
	if f.bin != "" {
		return f.bin, nil
	}
	if p, err := exec.LookPath("tailscale"); err == nil {
		return p, nil
	}
	if runtime.GOOS == "darwin" {
		if _, err := os.Stat(macTailscaleBin); err == nil {
			return macTailscaleBin, nil
		}
	}
	return "", errors.New("tailscale isn't installed here (no tailscale in PATH); install Tailscale, or set public_api.tailscale_bin in config.yaml")
}

func (f *funnel) runOnce(ctx context.Context) error {
	bin, err := f.resolveBin()
	if err != nil {
		return err
	}
	f.mu.Lock()
	f.st = funnelStatus{State: funnelStarting, Port: f.port}
	f.live, f.lastLine = false, ""
	f.mu.Unlock()

	cmd := exec.Command(bin, "funnel", fmt.Sprintf("--https=%d", f.port), "--set-path", "/v1", f.target)
	setProcAttrs(cmd)
	pr, pw := io.Pipe()
	cmd.Stdout, cmd.Stderr = pw, pw
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting %s: %w", bin, err)
	}
	log.Printf("[funnel] started %s funnel --https=%d --set-path /v1 %s (pid %d)", bin, f.port, f.target, cmd.Process.Pid)
	linesDone := make(chan struct{})
	go func() {
		defer close(linesDone)
		sc := bufio.NewScanner(pr)
		for sc.Scan() {
			f.onLine(sc.Text())
		}
		_, _ = io.Copy(io.Discard, pr)
	}()
	exited := make(chan error, 1)
	go func() {
		err := cmd.Wait()
		pw.Close()
		exited <- err
	}()

	keyTick := time.NewTicker(funnelKeyPoll)
	defer keyTick.Stop()
	confirm := time.NewTimer(funnelConfirmAfter)
	defer confirm.Stop()
	for {
		select {
		case err := <-exited:
			<-linesDone
			return f.exitErr(err)
		case <-ctx.Done():
			f.stopCmd(cmd, exited)
			log.Printf("[funnel] stopped: /v1 is off the internet")
			return ctx.Err()
		case <-keyTick.C:
			if !f.hasKeys() {
				log.Printf("[funnel] every API key was removed; taking /v1 off the internet")
				f.stopCmd(cmd, exited)
				<-linesDone
				return errFunnelNoKeys
			}
		case <-confirm.C:
			f.confirm(ctx, bin)
		}
	}
}

func (f *funnel) stopCmd(cmd *exec.Cmd, exited <-chan error) {
	terminateGroup(cmd.Process)
	select {
	case <-exited:
	case <-time.After(funnelStopWait):
		killGroup(cmd.Process)
		<-exited
	}
}

func (f *funnel) exitErr(err error) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.st.State == funnelError && f.st.Error != "" {
		return errors.New(f.st.Error)
	}
	if f.lastLine != "" {
		return fmt.Errorf("tailscale funnel stopped: %s", f.lastLine)
	}
	if err != nil {
		return fmt.Errorf("tailscale funnel stopped: %w", err)
	}
	return errors.New("tailscale funnel stopped")
}

func urlWithHost(line string, ok func(u *url.URL) bool) bool {
	u, err := url.Parse(line)
	return err == nil && u.Scheme == "https" && ok(u)
}

// onLine follows the CLI's output: a login.tailscale.com link when Funnel
// still has to be approved for the tailnet, "Available on the internet" then
// the public URL once it is on, "Access denied" when the user may not change
// tailscaled's config.
func (f *funnel) onLine(raw string) {
	line := strings.TrimSpace(raw)
	if line == "" {
		return
	}
	log.Printf("[funnel] %s", line)
	low := strings.ToLower(line)
	f.mu.Lock()
	defer f.mu.Unlock()
	switch {
	case urlWithHost(line, func(u *url.URL) bool { return u.Host == "login.tailscale.com" }):
		f.st.State, f.st.ApproveURL = funnelApproval, line
	case strings.Contains(low, "available on the internet"):
		f.live = true
	case f.live && urlWithHost(line, func(u *url.URL) bool { return strings.HasSuffix(u.Hostname(), ".ts.net") }):
		f.st.State, f.st.URL, f.st.ApproveURL, f.st.Error = funnelOn, line, "", ""
		log.Printf("[funnel] /v1 is on the internet at %s", line)
	case strings.Contains(low, "access denied") || strings.Contains(low, "permission denied") || strings.Contains(low, "--operator"):
		name := "<service user>"
		if u, err := user.Current(); err == nil {
			name = u.Username
		}
		f.st.State = funnelError
		f.st.Error = fmt.Sprintf("%s (let llama-manager's user change Tailscale's settings once with: sudo tailscale set --operator=%s)", strings.TrimSuffix(line, "."), name)
	case strings.HasPrefix(line, "|--") || strings.HasPrefix(low, "press ctrl+c") || strings.HasPrefix(low, "to enable, visit") || strings.HasPrefix(low, "funnel is not enabled"):
		// The rest of the banners above; not why it stopped.
	default:
		f.lastLine = line
	}
}

// confirm covers a CLI whose output we don't recognise: still running with
// nothing to approve after a while means Funnel is on; the URL comes from
// tailscale status.
func (f *funnel) confirm(ctx context.Context, bin string) {
	f.mu.Lock()
	starting := f.st.State == funnelStarting
	f.mu.Unlock()
	if !starting {
		return
	}
	cctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(cctx, bin, "status", "--json").Output()
	var st struct {
		Self struct {
			DNSName string `json:"DNSName"`
		} `json:"Self"`
	}
	if err != nil || json.Unmarshal(out, &st) != nil || st.Self.DNSName == "" {
		log.Printf("[funnel] couldn't read this machine's tailnet name from tailscale status: %v", err)
		return
	}
	host := strings.TrimSuffix(st.Self.DNSName, ".")
	if f.port != 443 {
		host = fmt.Sprintf("%s:%d", host, f.port)
	}
	f.set(func(s *funnelStatus) {
		if s.State == funnelStarting {
			s.State, s.URL = funnelOn, "https://"+host+"/v1"
		}
	})
}
