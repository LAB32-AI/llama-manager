package main

import (
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

	if err := cfg.UpdateSettings(Settings{NGL: 5, ContextLength: 128, RestartDelay: "1s", HealthCheckInterval: time.Second.String()}); err == nil {
		t.Fatal("save into a read-only directory unexpectedly succeeded")
	}
	now, _ := os.ReadFile(path)
	if string(now) != string(orig) {
		t.Fatalf("config changed after a failed save:\n%s", now)
	}
}
