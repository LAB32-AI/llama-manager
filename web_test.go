package main

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newTestConfig(t *testing.T) *Config {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	cfg := &Config{
		ServerBin:           "/bin/true",
		ManagerPort:         0,
		RestartDelay:        duration{1 * time.Second},
		MaxRestarts:         1,
		HealthCheckInterval: duration{1 * time.Second},
		GPUBackend:          "vulkan",
		Host:                "127.0.0.1",
		NGL:                 99,
		ContextLength:       2048,
		AllowedHosts:        []string{"example.com"}, // httptest.NewRequest's default Host
		path:                path,
	}
	if err := os.WriteFile(path, []byte("server_bin: /bin/true\n"), 0640); err != nil {
		t.Fatal(err)
	}
	return cfg
}

func newTestServer(t *testing.T) (*WebServer, *Config) {
	t.Helper()
	cfg := newTestConfig(t)
	mgr := NewManager(cfg)
	dlm := NewDownloadManager()
	return NewWebServer(mgr, cfg, dlm), cfg
}

func TestServerBinCannotBeChangedViaSettings(t *testing.T) {
	ws, cfg := newTestServer(t)
	body := strings.NewReader(`{"server_bin":"/tmp/evil","ngl":99,"context_length":2048,"main_gpu":0,"max_restarts":1,"host":"127.0.0.1","gpu_backend":"vulkan"}`)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPut, "/api/settings", body)
	req.Header.Set("Content-Type", "application/json")
	ws.ServeHTTP(rr, req)
	if rr.Code == http.StatusOK {
		t.Errorf("expected non-200, got 200; ServerBin now %q", cfg.ServerBin)
	}
	if cfg.ServerBin != "/bin/true" {
		t.Errorf("ServerBin changed to %q", cfg.ServerBin)
	}
}

func TestSettingsAllowsUnchangedServerBin(t *testing.T) {
	ws, _ := newTestServer(t)
	body := strings.NewReader(`{"server_bin":"/bin/true","ngl":99,"context_length":2048,"main_gpu":0,"max_restarts":1,"host":"127.0.0.1","gpu_backend":"vulkan"}`)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPut, "/api/settings", body)
	req.Header.Set("Content-Type", "application/json")
	ws.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("got %d, want 200; body: %s", rr.Code, rr.Body.String())
	}
}

func multipartConfigBody(t *testing.T, yaml string) (*bytes.Buffer, string) {
	t.Helper()
	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	fw, err := mw.CreateFormFile("file", "config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.WriteString(fw, yaml); err != nil {
		t.Fatal(err)
	}
	mw.Close()
	return body, mw.FormDataContentType()
}

func postImport(t *testing.T, ws *WebServer, yaml string) *httptest.ResponseRecorder {
	t.Helper()
	body, ct := multipartConfigBody(t, yaml)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/config/import", body)
	req.Header.Set("Content-Type", ct)
	ws.ServeHTTP(rr, req)
	return rr
}

// assertConfigFileUnchanged guards the real attack surface: a rejected import
// must not have written anything to disk, where it would take effect at boot.
func assertConfigFileUnchanged(t *testing.T, cfg *Config, want string) {
	t.Helper()
	got, err := os.ReadFile(cfg.path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != want {
		t.Fatalf("config file changed by a rejected import:\n%s", got)
	}
}

func TestConfigImportRejectsServerBinChange(t *testing.T) {
	ws, cfg := newTestServer(t)
	orig, _ := os.ReadFile(cfg.path)
	rr := postImport(t, ws, "server_bin: /tmp/evil\n")
	if rr.Code != http.StatusForbidden {
		t.Errorf("got %d, want 403; body=%s", rr.Code, rr.Body.String())
	}
	if cfg.ServerBin != "/bin/true" {
		t.Errorf("ServerBin changed to %q", cfg.ServerBin)
	}
	assertConfigFileUnchanged(t, cfg, string(orig))
}

// TestConfigImportRejectsMissingServerBin is the P1 #10 regression: an import
// without server_bin passed the lock check and was written verbatim, so the
// next boot failed with "server_bin is required".
func TestConfigImportRejectsMissingServerBin(t *testing.T) {
	for _, body := range []string{"ngl: 50\n", "server_bin: \"\"\nngl: 50\n"} {
		ws, cfg := newTestServer(t)
		orig, _ := os.ReadFile(cfg.path)
		rr := postImport(t, ws, body)
		if rr.Code != http.StatusForbidden {
			t.Errorf("import %q: got %d, want 403; body=%s", body, rr.Code, rr.Body.String())
		}
		assertConfigFileUnchanged(t, cfg, string(orig))
		if _, err := loadConfig(cfg.path); err != nil {
			t.Fatalf("config no longer loads after a rejected import: %v", err)
		}
	}
}

// TestConfigImportRejectsInvalidConfig: an import that would not survive
// loadConfig (or would crash the supervisor) is refused before it reaches disk.
func TestConfigImportRejectsInvalidConfig(t *testing.T) {
	for _, extra := range []string{
		"health_check_interval: 0s\n",
		"gpu_backend: roc\n",
		"instances:\n  - name: a\n    model: /m.gguf\n    port: 9001\n",
	} {
		ws, cfg := newTestServer(t)
		orig, _ := os.ReadFile(cfg.path)
		rr := postImport(t, ws, "server_bin: /bin/true\n"+extra)
		if rr.Code != http.StatusBadRequest {
			t.Errorf("import %q: got %d, want 400; body=%s", extra, rr.Code, rr.Body.String())
		}
		assertConfigFileUnchanged(t, cfg, string(orig))
	}
}

func TestConfigImportAcceptsCompatibleConfig(t *testing.T) {
	ws, _ := newTestServer(t)
	body, ct := multipartConfigBody(t, "server_bin: /bin/true\nhost: 127.0.0.1\nngl: 50\n")
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/config/import", body)
	req.Header.Set("Content-Type", ct)
	ws.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("got %d, want 200; body=%s", rr.Code, rr.Body.String())
	}
}

// TestStartEndpointReportsStartFailure: the API used to answer {"status":"ok"}
// for a start that never happened.
func TestStartEndpointReportsStartFailure(t *testing.T) {
	ws, cfg := newTestServer(t)
	cfg.ServerBin = filepath.Join(t.TempDir(), "missing-llama-server")
	if err := ws.mgr.AddInstance(InstanceConf{Name: "x", Model: "/m.gguf", Port: 9555, GPUIDs: []int{0}}); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/api/instances/x/start", nil))
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500; body=%s", rr.Code, rr.Body.String())
	}
	if st := ws.mgr.Get("x").Status(); st.State != StateCrashed || !strings.HasPrefix(st.LastError, "start failed:") {
		t.Fatalf("status = %+v, want crashed with a start-failed reason", st)
	}
}
