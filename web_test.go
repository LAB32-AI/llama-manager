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

func TestConfigImportRejectsServerBinChange(t *testing.T) {
	ws, cfg := newTestServer(t)
	body, ct := multipartConfigBody(t, "server_bin: /tmp/evil\n")
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/config/import", body)
	req.Header.Set("Content-Type", ct)
	ws.ServeHTTP(rr, req)
	if rr.Code != http.StatusForbidden {
		t.Errorf("got %d, want 403; body=%s", rr.Code, rr.Body.String())
	}
	if cfg.ServerBin != "/bin/true" {
		t.Errorf("ServerBin changed to %q", cfg.ServerBin)
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
