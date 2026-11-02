package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestValidateInstanceName(t *testing.T) {
	for _, ok := range []string{"gpu0", "dolphin-gpu0", "GLM 4.5 Air", "a.b_c", strings.Repeat("a", 64)} {
		if err := validateInstanceName(ok); err != nil {
			t.Errorf("validateInstanceName(%q) = %v", ok, err)
		}
	}
	for _, bad := range []string{"", "all", " lead", "trail ", "x/y", "bob's-gpu", `x");alert(1);("`, "<script>", "q4 50%", "-dash", strings.Repeat("a", 65)} {
		if err := validateInstanceName(bad); err == nil {
			t.Errorf("validateInstanceName(%q) accepted", bad)
		}
	}
}

// TestUnsafeInstanceNamesRejected is the P1 #17 regression: any name was
// accepted, so "all" turned the row's Stop button into stop-everything (the
// mux routes /api/instances/all/stop to the bulk handler), and quotes broke
// out of the UI's inline onclick handlers.
func TestUnsafeInstanceNamesRejected(t *testing.T) {
	ws, _ := newTestServer(t)
	for _, name := range []string{"all", "bob's-gpu", `x');fetch('/api/models',{method:'DELETE'});('`, "x/y"} {
		rr := doJSON(t, ws, http.MethodPost, "/api/config/instances", InstanceConf{Name: name, Model: "/m.gguf", Port: 9001, GPUIDs: []int{0}})
		if rr.Code != http.StatusBadRequest {
			t.Errorf("POST name %q: got %d, want 400", name, rr.Code)
		}
	}
	if _, err := loadConfig(writeConfig(t, validBase+"instances:\n  - name: all\n    model: /m.gguf\n    port: 9001\n    gpu_ids: [0]\n")); err == nil {
		t.Error(`loadConfig accepted an instance named "all"`)
	}
}

// TestInstanceNameWithSpaceRoutesCorrectly: the path is decoded once by
// net/http; the old second url.PathUnescape broke names containing '%' and
// would mis-route encoded input.
func TestInstanceNameWithSpaceRoutesCorrectly(t *testing.T) {
	ws, cfg := newTestServer(t)
	ic := InstanceConf{Name: "q4 50", Model: "/m.gguf", Port: 9001, GPUIDs: []int{0}}
	if err := cfg.AddInstance(ic); err != nil {
		t.Fatal(err)
	}
	if err := ws.mgr.AddInstance(ic); err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/api/instances/q4%2050", nil))
	if rr.Code != http.StatusOK || !strings.Contains(rr.Body.String(), `"name":"q4 50"`) {
		t.Fatalf("GET encoded name: %d %s", rr.Code, rr.Body.String())
	}
	rr = httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodDelete, "/api/config/instances/q4%2050", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("DELETE encoded name: %d %s", rr.Code, rr.Body.String())
	}
	if ws.mgr.Get("q4 50") != nil || len(cfg.GetInstances()) != 0 {
		t.Fatal("instance not deleted")
	}
}
