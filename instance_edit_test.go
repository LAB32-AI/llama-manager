package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func newFakeWebServer(t *testing.T, rig *fakeRig) *WebServer {
	t.Helper()
	if err := writeFileAtomic(rig.cfg.path, []byte("server_bin: "+rig.cfg.ServerBin+"\n")); err != nil {
		t.Fatal(err)
	}
	rig.cfg.ManagerPort = 8080
	rig.cfg.AllowedHosts = []string{"example.com"} // httptest.NewRequest's default Host
	mgr := NewManager(rig.cfg)
	t.Cleanup(mgr.Shutdown)
	return NewWebServer(mgr, rig.cfg, NewDownloadManager())
}

func doJSON(t *testing.T, ws *WebServer, method, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	b, _ := json.Marshal(body)
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(method, path, strings.NewReader(string(b))))
	return rr
}

// TestRejectedInstanceEditKeepsRunningInstance is the P1 #12 regression: PUT
// stopped and unregistered the instance before validating, so an edit that
// failed (duplicate port) killed a running model and dropped it from the
// dashboard while the config still listed it.
func TestRejectedInstanceEditKeepsRunningInstance(t *testing.T) {
	rig := newFakeRig(t, "healthy", "a", "b")
	ws := newFakeWebServer(t, rig)
	if err := ws.mgr.StartInstance("a"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 2*time.Second, "a running", func() bool { return ws.mgr.Get("a").currentPid() != 0 })
	pid := ws.mgr.Get("a").currentPid()

	a := rig.cfg.Instances[0]
	for name, edit := range map[string]InstanceConf{
		"duplicate port": {Name: "a", Model: a.Model, Port: rig.cfg.Instances[1].Port, GPUIDs: []int{0}},
		"duplicate name": {Name: "b", Model: a.Model, Port: a.Port, GPUIDs: []int{0}},
		"manager port":   {Name: "a", Model: a.Model, Port: 8080, GPUIDs: []int{0}},
		"invalid port":   {Name: "a", Model: a.Model, Port: 70000, GPUIDs: []int{0}},
		"no gpu":         {Name: "a", Model: a.Model, Port: a.Port},
	} {
		rr := doJSON(t, ws, http.MethodPut, "/api/config/instances/a", edit)
		if rr.Code != http.StatusBadRequest {
			t.Errorf("%s: got %d, want 400 (%s)", name, rr.Code, rr.Body.String())
		}
		inst := ws.mgr.Get("a")
		if inst == nil {
			t.Fatalf("%s: rejected edit removed the instance from the manager", name)
		}
		if got := inst.currentPid(); got != pid {
			t.Fatalf("%s: rejected edit touched the running process (pid %d -> %d)", name, pid, got)
		}
	}
	if got := rig.cfg.GetInstances(); got[0].Port != a.Port || got[0].Name != "a" {
		t.Fatalf("config changed by rejected edits: %+v", got[0])
	}
}

func TestValidInstanceEditReplacesInstance(t *testing.T) {
	rig := newFakeRig(t, "healthy", "a")
	ws := newFakeWebServer(t, rig)
	a := rig.cfg.Instances[0]
	newPort := freePort(t)
	edit := InstanceConf{Name: "a", Model: a.Model, Port: newPort, GPUIDs: []int{0}, ExtraArgs: a.ExtraArgs}

	if rr := doJSON(t, ws, http.MethodPut, "/api/config/instances/a", edit); rr.Code != http.StatusOK {
		t.Fatalf("got %d: %s", rr.Code, rr.Body.String())
	}
	if got := ws.mgr.Get("a"); got == nil || got.conf.Port != newPort {
		t.Fatalf("manager instance not replaced: %+v", got)
	}
}

func TestAddInstanceRejectsInvalidConfig(t *testing.T) {
	rig := newFakeRig(t, "healthy", "a")
	ws := newFakeWebServer(t, rig)
	for name, ic := range map[string]InstanceConf{
		"manager port": {Name: "n", Model: "/m.gguf", Port: 8080, GPUIDs: []int{0}},
		"bad port":     {Name: "n", Model: "/m.gguf", Port: -1, GPUIDs: []int{0}},
		"no model":     {Name: "n", Port: 9500, GPUIDs: []int{0}},
	} {
		if rr := doJSON(t, ws, http.MethodPost, "/api/config/instances", ic); rr.Code == http.StatusOK {
			t.Errorf("%s: accepted %+v", name, ic)
		}
		if ws.mgr.Get("n") != nil {
			t.Fatalf("%s: rejected instance was added to the manager", name)
		}
	}
}

// TestInstanceEditKeepsHandSetOverrides is the P1 #13 regression: the UI's
// edit form never sends extra_args/jinja, and the full replace in
// UpdateInstance wiped them from config.yaml on any port or GPU tweak.
func TestInstanceEditKeepsHandSetOverrides(t *testing.T) {
	rig := newFakeRig(t, "healthy", "a")
	off := false
	rig.cfg.Instances[0].Jinja = &off
	ws := newFakeWebServer(t, rig)
	a := rig.cfg.Instances[0]
	wantArgs := *a.ExtraArgs

	uiPayload := map[string]any{"name": "a", "model": a.Model, "port": freePort(t), "gpu_ids": []int{0}, "auto_start": false}
	if rr := doJSON(t, ws, http.MethodPut, "/api/config/instances/a", uiPayload); rr.Code != http.StatusOK {
		t.Fatalf("got %d: %s", rr.Code, rr.Body.String())
	}
	got := rig.cfg.GetInstances()[0]
	if got.ExtraArgs == nil || strings.Join(*got.ExtraArgs, " ") != strings.Join(wantArgs, " ") {
		t.Fatalf("extra_args = %v, want %v kept", got.ExtraArgs, wantArgs)
	}
	if got.Jinja == nil || *got.Jinja {
		t.Fatalf("jinja = %v, want false kept", got.Jinja)
	}
}

func TestInstanceEditCanStillSetOverridesExplicitly(t *testing.T) {
	rig := newFakeRig(t, "healthy", "a")
	ws := newFakeWebServer(t, rig)
	a := rig.cfg.Instances[0]

	payload := map[string]any{"name": "a", "model": a.Model, "port": a.Port, "gpu_ids": []int{0}, "extra_args": nil, "jinja": true}
	if rr := doJSON(t, ws, http.MethodPut, "/api/config/instances/a", payload); rr.Code != http.StatusOK {
		t.Fatalf("got %d: %s", rr.Code, rr.Body.String())
	}
	got := rig.cfg.GetInstances()[0]
	if got.ExtraArgs != nil {
		t.Fatalf("explicit null extra_args should clear the override, got %v", *got.ExtraArgs)
	}
	if got.Jinja == nil || !*got.Jinja {
		t.Fatalf("explicit jinja=true not applied: %v", got.Jinja)
	}
}
