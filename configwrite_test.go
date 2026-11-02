package main

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newSavedConfig(t *testing.T) (*Config, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte("server_bin: /usr/bin/true\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := loadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	return cfg, path
}

// TestSaveReplacesConfigAtomically is the P1 #8 regression: saves used to
// truncate config.yaml in place, so ENOSPC or a crash mid-write left an empty
// file and a service that could no longer start. A rename-based write swaps in
// a new inode instead of ever truncating the live one.
func TestSaveReplacesConfigAtomically(t *testing.T) {
	cfg, path := newSavedConfig(t)
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.AddInstance(InstanceConf{Name: "a", Model: "/m.gguf", Port: 9001, GPUIDs: []int{0}}); err != nil {
		t.Fatal(err)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if os.SameFile(before, after) {
		t.Fatal("config was rewritten in place (same inode) instead of replaced atomically")
	}
	if after.Mode().Perm() != 0o600 {
		t.Errorf("mode = %v, want the original 0600 kept", after.Mode().Perm())
	}
	reloaded, err := loadConfig(path)
	if err != nil {
		t.Fatalf("reloading saved config: %v", err)
	}
	if len(reloaded.Instances) != 1 || reloaded.Instances[0].Name != "a" {
		t.Fatalf("reloaded instances = %+v", reloaded.Instances)
	}
	entries, _ := os.ReadDir(filepath.Dir(path))
	for _, e := range entries {
		if strings.Contains(e.Name(), ".tmp-") {
			t.Errorf("temp file left behind: %s", e.Name())
		}
	}
}

func TestSaveThroughSymlinkUpdatesTarget(t *testing.T) {
	cfg, target := newSavedConfig(t)
	link := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	cfg.path = link
	if err := cfg.AddInstance(InstanceConf{Name: "a", Model: "/m.gguf", Port: 9001, GPUIDs: []int{0}}); err != nil {
		t.Fatal(err)
	}
	fi, err := os.Lstat(link)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode()&os.ModeSymlink == 0 {
		t.Fatal("symlinked config was replaced by a regular file")
	}
	data, _ := os.ReadFile(target)
	if !strings.Contains(string(data), "name: a") {
		t.Fatalf("symlink target not updated:\n%s", data)
	}
}

// TestFailedSaveLeavesConfigIntact: when the new file can't be written the old
// one must survive untouched.
func TestFailedSaveLeavesConfigIntact(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root ignores directory permissions")
	}
	cfg, path := newSavedConfig(t)
	orig, _ := os.ReadFile(path)
	dir := filepath.Dir(path)
	if err := os.Chmod(dir, 0o500); err != nil {
		t.Fatal(err)
	}
	defer os.Chmod(dir, 0o700)

	if err := cfg.UpdateSettings(Settings{NGL: intp(5), ContextLength: intp(128), RestartDelay: "1s", HealthCheckInterval: time.Second.String()}); err == nil {
		t.Fatal("save into a read-only directory unexpectedly succeeded")
	}
	now, _ := os.ReadFile(path)
	if string(now) != string(orig) {
		t.Fatalf("config changed after a failed save:\n%s", now)
	}
}

// readOnlyConfigDir makes saves fail (the atomic write needs a temp file in the
// config directory) and restores permissions afterwards.
func readOnlyConfigDir(t *testing.T, path string) {
	t.Helper()
	if os.Geteuid() == 0 {
		t.Skip("root ignores directory permissions")
	}
	dir := filepath.Dir(path)
	if err := os.Chmod(dir, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(dir, 0o700) })
}

// TestFailedSaveRollsBackMemory (P2): mutations changed memory and then tried
// to save; when the save failed the change stayed in memory (phantom
// instances, "duplicate" errors on retry) though the file never had it.
func TestFailedSaveRollsBackMemory(t *testing.T) {
	cfg, path := newSavedConfig(t)
	keep := InstanceConf{Name: "keep", Model: "/m.gguf", Port: 9001, GPUIDs: []int{0}}
	other := InstanceConf{Name: "other", Model: "/m.gguf", Port: 9002, GPUIDs: []int{1}}
	if err := cfg.AddInstance(keep); err != nil {
		t.Fatal(err)
	}
	if err := cfg.AddInstance(other); err != nil {
		t.Fatal(err)
	}
	before := cfg.GetInstances()
	beforeSettings := cfg.GetSettings()
	readOnlyConfigDir(t, path)

	if err := cfg.AddInstance(InstanceConf{Name: "new", Model: "/m.gguf", Port: 9003, GPUIDs: []int{0}}); err == nil {
		t.Fatal("AddInstance save unexpectedly succeeded")
	}
	edited := keep
	edited.Port = 9010
	if err := cfg.UpdateInstance("keep", edited); err == nil {
		t.Fatal("UpdateInstance save unexpectedly succeeded")
	}
	if err := cfg.DeleteInstance("keep"); err == nil {
		t.Fatal("DeleteInstance save unexpectedly succeeded")
	}
	if err := cfg.UpdateSettings(Settings{NGL: intp(1), RestartDelay: "9s"}); err == nil {
		t.Fatal("UpdateSettings save unexpectedly succeeded")
	}

	if got := cfg.GetInstances(); len(got) != len(before) || got[0].Name != "keep" || got[0].Port != 9001 || got[1].Name != "other" {
		t.Fatalf("instances after failed saves = %+v, want %+v", got, before)
	}
	if s := cfg.GetSettings(); *s.NGL != *beforeSettings.NGL || s.RestartDelay != beforeSettings.RestartDelay {
		t.Fatalf("settings changed by failed save: ngl %d, restart_delay %s", *s.NGL, s.RestartDelay)
	}
}

// TestDeleteEndpointKeepsInstanceWhenSaveFails: the handler used to stop the
// process before the config delete, which could then fail.
func TestDeleteEndpointKeepsInstanceWhenSaveFails(t *testing.T) {
	rig := newFakeRig(t, "healthy", "a")
	ws := newFakeWebServer(t, rig)
	if err := ws.mgr.StartInstance("a"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 2*time.Second, "running process", func() bool { return ws.mgr.Get("a").currentPid() != 0 })
	pid := ws.mgr.Get("a").currentPid()
	readOnlyConfigDir(t, rig.cfg.path)

	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodDelete, "/api/config/instances/a", nil))
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("got %d, want 500", rr.Code)
	}
	if inst := ws.mgr.Get("a"); inst == nil || inst.currentPid() != pid {
		t.Fatal("instance was stopped although its deletion could not be saved")
	}
}

// TestSaveRefusesToRevertHandEdit (P2): saves rewrote config.yaml from the
// copy loaded at boot, so a hand edit (the only sanctioned way to change
// server_bin) was silently reverted by the next UI change.
func TestSaveRefusesToRevertHandEdit(t *testing.T) {
	cfg, path := newSavedConfig(t)
	if err := cfg.UpdateSettings(Settings{NGL: intp(12)}); err != nil {
		t.Fatalf("save after load: %v", err)
	}
	if err := cfg.UpdateSettings(Settings{NGL: intp(13)}); err != nil {
		t.Fatalf("second save (own previous write): %v", err)
	}

	handEdit := "server_bin: /opt/new/llama-server\n"
	if err := os.WriteFile(path, []byte(handEdit), 0o600); err != nil {
		t.Fatal(err)
	}
	err := cfg.UpdateSettings(Settings{NGL: intp(14)})
	if !errors.Is(err, errConfigChangedOnDisk) {
		t.Fatalf("err = %v, want errConfigChangedOnDisk", err)
	}
	if data, _ := os.ReadFile(path); string(data) != handEdit {
		t.Fatalf("hand edit was overwritten:\n%s", data)
	}
	if *cfg.GetSettings().NGL != 13 {
		t.Fatal("refused save still changed memory")
	}

	reloaded, err := loadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := reloaded.UpdateSettings(Settings{NGL: intp(14)}); err != nil {
		t.Fatalf("save after reload: %v", err)
	}
}

func TestSettingsEndpointReportsHandEditConflict(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte("server_bin: /bin/true\nallowed_hosts: [example.com]\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := loadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	ws := NewWebServer(NewManager(cfg), cfg, NewDownloadManager())
	if err := os.WriteFile(path, []byte("server_bin: /bin/true\nngl: 1\nallowed_hosts: [example.com]\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"ngl":5}`)))
	if rr.Code != http.StatusConflict {
		t.Fatalf("got %d, want 409; body=%s", rr.Code, rr.Body.String())
	}
}
