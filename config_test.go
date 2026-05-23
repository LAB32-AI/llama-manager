package main

import (
	"os"
	"path/filepath"
	"testing"
)

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(body), 0640); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadConfigDefaults(t *testing.T) {
	path := writeConfig(t, "server_bin: /bin/true\n")
	cfg, err := loadConfig(path)
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.Host != "0.0.0.0" {
		t.Errorf("default Host = %q, want 0.0.0.0", cfg.Host)
	}
	if cfg.ManagerPort != 8080 {
		t.Errorf("default ManagerPort = %d, want 8080", cfg.ManagerPort)
	}
	if cfg.NGL != 99 {
		t.Errorf("default NGL = %d, want 99", cfg.NGL)
	}
	if cfg.GPUBackend != "vulkan" {
		t.Errorf("default GPUBackend = %q, want vulkan", cfg.GPUBackend)
	}
	if !cfg.Jinja {
		t.Error("default Jinja = false, want true (tool calling on by default)")
	}
}

func TestJinjaCanBeDisabled(t *testing.T) {
	path := writeConfig(t, "server_bin: /bin/true\njinja: false\n")
	cfg, err := loadConfig(path)
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.Jinja {
		t.Error("Jinja = true, want false when config sets jinja: false")
	}
}

// TestUpdateSettingsPreservesJinja verifies that a partial settings PUT (no
// jinja field, as older UI builds send) leaves the flag untouched, while an
// explicit value updates it.
func TestUpdateSettingsPreservesJinja(t *testing.T) {
	path := writeConfig(t, "server_bin: /bin/true\n")
	cfg, err := loadConfig(path)
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if !cfg.Jinja {
		t.Fatal("precondition: Jinja should default true")
	}
	// Partial update with no jinja field must not flip it off.
	if err := cfg.UpdateSettings(Settings{NGL: 99, ContextLength: 16384}); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	if !cfg.Jinja {
		t.Error("Jinja flipped to false by a partial settings update")
	}
	off := false
	if err := cfg.UpdateSettings(Settings{NGL: 99, ContextLength: 16384, Jinja: &off}); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	if cfg.Jinja {
		t.Error("Jinja still true after explicit jinja:false update")
	}
}

func TestLoadConfigMissingServerBin(t *testing.T) {
	path := writeConfig(t, "host: 127.0.0.1\n")
	if _, err := loadConfig(path); err == nil {
		t.Fatal("expected error when server_bin is missing")
	}
}

func TestLoadConfigMalformedYAML(t *testing.T) {
	path := writeConfig(t, "this is: : not valid\n  - yaml\n")
	if _, err := loadConfig(path); err == nil {
		t.Fatal("expected error for malformed yaml")
	}
}

func TestAddInstanceDeduplicatesByName(t *testing.T) {
	path := writeConfig(t, "server_bin: /bin/true\n")
	cfg, err := loadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	ic := InstanceConf{Name: "a", Model: "/m", Port: 9090, GPUIDs: []int{0}}
	if err := cfg.AddInstance(ic); err != nil {
		t.Fatalf("first AddInstance: %v", err)
	}
	if err := cfg.AddInstance(ic); err == nil {
		t.Fatal("expected duplicate-name error on second AddInstance")
	}
}

func TestAddInstanceDeduplicatesByPort(t *testing.T) {
	path := writeConfig(t, "server_bin: /bin/true\n")
	cfg, err := loadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.AddInstance(InstanceConf{Name: "a", Model: "/m", Port: 9090, GPUIDs: []int{0}}); err != nil {
		t.Fatal(err)
	}
	if err := cfg.AddInstance(InstanceConf{Name: "b", Model: "/m", Port: 9090, GPUIDs: []int{1}}); err == nil {
		t.Fatal("expected duplicate-port error")
	}
}

func TestUpdateSettingsValidation(t *testing.T) {
	path := writeConfig(t, "server_bin: /bin/true\n")
	cfg, err := loadConfig(path)
	if err != nil {
		t.Fatal(err)
	}

	good := cfg.GetSettings()
	if err := cfg.UpdateSettings(good); err != nil {
		t.Fatalf("UpdateSettings(good): %v", err)
	}

	cases := []struct {
		name string
		mut  func(s *Settings)
	}{
		{"negative NGL", func(s *Settings) { s.NGL = -1 }},
		{"negative MaxRestarts", func(s *Settings) { s.MaxRestarts = -1 }},
		{"zero ContextLength", func(s *Settings) { s.ContextLength = 0 }},
		{"negative MainGPU", func(s *Settings) { s.MainGPU = -1 }},
		{"bad GPU backend", func(s *Settings) { s.GPUBackend = "wat" }},
		{"bad duration", func(s *Settings) { s.RestartDelay = "not a duration" }},
		{"changing server_bin", func(s *Settings) { s.ServerBin = "/tmp/evil" }},
	}
	for _, c := range cases {
		s := good
		c.mut(&s)
		if err := cfg.UpdateSettings(s); err == nil {
			t.Errorf("%s: expected error, got nil", c.name)
		}
	}
}

func TestDurationRoundtripJSON(t *testing.T) {
	d := duration{}
	if err := d.UnmarshalJSON([]byte(`"30s"`)); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if d.Duration.Seconds() != 30 {
		t.Errorf("got %v, want 30s", d.Duration)
	}
	out, err := d.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != `"30s"` {
		t.Errorf("MarshalJSON = %s, want \"30s\"", out)
	}
}
