package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

// TestConfigImportSurvivesNextSave is the P1 #11 regression: import wrote the
// file but kept the old instances (and most settings) in memory, so the next
// save of anything wrote the stale copy back over the import.
func TestConfigImportSurvivesNextSave(t *testing.T) {
	ws, cfg := newTestServer(t)
	rr := postImport(t, ws, "server_bin: /bin/true\nallowed_hosts: [example.com]\ninstances:\n  - name: imported\n    model: /m.gguf\n    port: 9101\n    gpu_ids: [0]\n")
	if rr.Code != http.StatusOK {
		t.Fatalf("import: %d %s", rr.Code, rr.Body.String())
	}

	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"ngl":42,"context_length":2048}`))
	rr = httptest.NewRecorder()
	ws.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("settings: %d %s", rr.Code, rr.Body.String())
	}

	data, _ := os.ReadFile(cfg.path)
	if !strings.Contains(string(data), "name: imported") {
		t.Fatalf("imported instance was lost by the next save:\n%s", data)
	}
	if got := cfg.GetInstances(); len(got) != 1 || got[0].Name != "imported" {
		t.Fatalf("in-memory instances = %+v", got)
	}
	if ws.mgr.Get("imported") == nil {
		t.Fatal("imported instance is not in the manager (dashboard can't see it)")
	}
}

// TestConfigImportAppliesEverySetting: the old code skipped zero values
// (max_restarts: 0, ngl: 0) and never applied extra_args, model_dirs, jinja,
// main_gpu, unhealthy_after or health_check_timeout.
func TestConfigImportAppliesEverySetting(t *testing.T) {
	ws, cfg := newTestServer(t)
	rr := postImport(t, ws, `server_bin: /bin/true
max_restarts: 0
ngl: 0
main_gpu: 1
jinja: false
unhealthy_after: 0
health_check_timeout: 9s
extra_args: ["-np", "1"]
model_dirs: ["/models"]
`)
	if rr.Code != http.StatusOK {
		t.Fatalf("import: %d %s", rr.Code, rr.Body.String())
	}
	cfg.mu.RLock()
	defer cfg.mu.RUnlock()
	if cfg.MaxRestarts != 0 || cfg.NGL != 0 || cfg.MainGPU != 1 || cfg.Jinja || cfg.UnhealthyAfter != 0 ||
		cfg.HealthCheckTimeout.Duration != 9*time.Second ||
		!reflect.DeepEqual(cfg.ExtraArgs, []string{"-np", "1"}) || !reflect.DeepEqual(cfg.ModelDirs, []string{"/models"}) {
		t.Fatalf("settings not fully applied: %+v", cfg)
	}
}

func TestApplyLockedCopiesEveryExportedField(t *testing.T) {
	src := &Config{}
	v := reflect.ValueOf(src).Elem()
	for i := 0; i < v.NumField(); i++ {
		if !v.Type().Field(i).IsExported() {
			continue
		}
		f := v.Field(i)
		switch f.Kind() {
		case reflect.String:
			f.SetString("x")
		case reflect.Int:
			f.SetInt(7)
		case reflect.Bool:
			f.SetBool(true)
		case reflect.Slice:
			f.Set(reflect.MakeSlice(f.Type(), 1, 1))
		case reflect.Struct:
			switch f.Type() {
			case reflect.TypeOf(TikaConf{}):
				f.Set(reflect.ValueOf(TikaConf{Enabled: true, URL: "http://tika:9998", Timeout: duration{time.Minute}}))
			case reflect.TypeOf(ImageRecConf{}):
				f.Set(reflect.ValueOf(ImageRecConf{Enabled: true, Model: "qwen3.5-4b", Port: 9195, GPUIDs: []int{4}, ExtraArgs: []string{"-fa", "on"}}))
			case reflect.TypeOf(ImageGenConf{}):
				f.Set(reflect.ValueOf(ImageGenConf{Enabled: true, ServerBin: "/opt/sd-server", Port: 9196, GPUIDs: []int{3}, ExtraArgs: []string{"--flash-attn", "off"}, Timeout: duration{time.Minute},
					LoraDir: "/srv/loras", LoRAs: []string{"klein-anime"}}))
			case reflect.TypeOf(SpeechConf{}):
				f.Set(reflect.ValueOf(SpeechConf{Enabled: true, ServerBin: "/opt/audiocpp_server", Port: 9197, GPUIDs: []int{2}, Voices: []SpeechVoiceConf{{Name: "me", File: "/v.wav"}}, Timeout: duration{time.Minute}}))
			case reflect.TypeOf(PromptEnhancerConf{}):
				f.Set(reflect.ValueOf(PromptEnhancerConf{Enabled: true, Model: "unsloth/Qwen3-4B-GGUF:Q4_K_M", Port: 9198, GPUIDs: []int{4}, ContextLength: 8192, ExtraArgs: []string{"--no-mmproj"}, Timeout: duration{time.Minute}}))
			case reflect.TypeOf(ChatConf{}):
				temp, topK := 0.5, 20
				f.Set(reflect.ValueOf(ChatConf{SystemPrompt: "Be kind.", Temperature: &temp, TopK: &topK, MaxTokens: 100, Voice: "tts|me", VoiceLang: "en"}))
			case reflect.TypeOf(PublicAPIConf{}):
				f.Set(reflect.ValueOf(PublicAPIConf{TailscaleFunnel: true, TailscaleBin: "/usr/bin/tailscale", HTTPSPort: 8443, MaxConcurrent: 3}))
			default:
				f.Field(0).SetInt(int64(time.Second))
			}
		default:
			t.Fatalf("field %s: kind %s not covered by this test", v.Type().Field(i).Name, f.Kind())
		}
	}
	dst := &Config{path: "/keep"}
	dst.applyLocked(src)
	dv := reflect.ValueOf(dst).Elem()
	for i := 0; i < v.NumField(); i++ {
		if v.Type().Field(i).IsExported() && !reflect.DeepEqual(dv.Field(i).Interface(), v.Field(i).Interface()) {
			t.Errorf("field %s not copied", v.Type().Field(i).Name)
		}
	}
	if dst.path != "/keep" {
		t.Error("applyLocked overwrote the config path")
	}
}

func fakeInstanceYAML(name string, port int, mode, pidDir string) string {
	return fmt.Sprintf("  - name: %s\n    model: /dev/null\n    port: %d\n    gpu_ids: [0]\n    extra_args: [\"--fake-mode\", %q, \"--fake-pid-dir\", %q]\n", name, port, mode, pidDir)
}

// TestConfigImportReconcilesRunningInstances: unchanged instances keep their
// process, removed ones are stopped, changed running ones are restarted with
// the new settings, new ones are added stopped, and order follows the file.
func TestConfigImportReconcilesRunningInstances(t *testing.T) {
	rig := newFakeRig(t, "healthy", "keep", "change", "drop")
	if err := writeFileAtomic(rig.cfg.path, []byte("server_bin: "+rig.cfg.ServerBin+"\n")); err != nil {
		t.Fatal(err)
	}
	rig.cfg.AllowedHosts = []string{"example.com"}
	mgr := NewManager(rig.cfg)
	defer mgr.Shutdown()
	ws := NewWebServer(mgr, rig.cfg, NewDownloadManager())
	for _, n := range []string{"keep", "change", "drop"} {
		if err := mgr.StartInstance(n); err != nil {
			t.Fatal(err)
		}
	}
	waitFor(t, 3*time.Second, "three processes", func() bool { return len(rig.livePids(t)) == 3 })
	keepPid := mgr.Get("keep").currentPid()
	changePid := mgr.Get("change").currentPid()
	dropPid := mgr.Get("drop").currentPid()

	keepPort := rig.cfg.Instances[0].Port
	newPort := freePort(t)
	body := "server_bin: " + rig.cfg.ServerBin + "\nrestart_delay: 50ms\nhealth_check_timeout: 1s\ngpu_backend: metal\nhost: 127.0.0.1\nallowed_hosts: [example.com]\ninstances:\n" +
		fakeInstanceYAML("added", freePort(t), "healthy", rig.pidDir) +
		fakeInstanceYAML("keep", keepPort, "healthy", rig.pidDir) +
		fakeInstanceYAML("change", newPort, "healthy", rig.pidDir)
	rr := postImport(t, ws, body)
	if rr.Code != http.StatusOK {
		t.Fatalf("import: %d %s", rr.Code, rr.Body.String())
	}

	if got := mgr.Get("keep").currentPid(); got != keepPid {
		t.Errorf("unchanged instance was restarted (pid %d -> %d)", keepPid, got)
	}
	if mgr.Get("drop") != nil {
		t.Error("removed instance is still managed")
	}
	if pidAlive(dropPid) {
		t.Error("removed instance's process is still running")
	}
	changed := mgr.Get("change")
	if changed == nil || changed.conf.Port != newPort {
		t.Fatalf("changed instance not replaced with the new config: %+v", changed)
	}
	if got := changed.currentPid(); got == 0 || got == changePid {
		t.Errorf("changed running instance not restarted (pid %d -> %d)", changePid, got)
	}
	if st := mgr.Get("added").State(); st != StateStopped {
		t.Errorf("new instance state = %s, want stopped", st)
	}
	var order []string
	for _, inst := range mgr.Instances() {
		order = append(order, inst.conf.Name)
	}
	if !reflect.DeepEqual(order, []string{"added", "keep", "change"}) {
		t.Errorf("instance order = %v, want file order", order)
	}
}
