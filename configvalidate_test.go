package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const validBase = "server_bin: /usr/bin/true\n"

// TestLoadConfigRejectsInvalidValues is the P1 #9 regression: loadConfig only
// checked server_bin, so e.g. health_check_interval: 0s reached time.NewTicker
// and panicked the whole manager on the first start (a crash loop under
// systemd).
func TestLoadConfigRejectsInvalidValues(t *testing.T) {
	inst := func(extra string) string {
		return "instances:\n  - name: a\n    model: /m.gguf\n    port: 9001\n    gpu_ids: [0]\n" + extra
	}
	cases := map[string]string{
		"zero health interval":     "health_check_interval: 0s\n",
		"negative health interval": "health_check_interval: -5s\n",
		"zero restart delay":       "restart_delay: 0s\n",
		"negative max_restarts":    "max_restarts: -1\n",
		"negative unhealthy_after": "unhealthy_after: -1\n",
		"zero context":             "context_length: 0\n",
		"unknown backend":          "gpu_backend: roc\n",
		"bad manager port":         "manager_port: 70000\n",
		"missing server_bin":       "server_bin: \"\"\n",
		"instance without gpu":     "instances:\n  - name: a\n    model: /m.gguf\n    port: 9001\n",
		"instance bad port":        "instances:\n  - name: a\n    model: /m.gguf\n    port: 0\n    gpu_ids: [0]\n",
		"instance no model":        "instances:\n  - name: a\n    port: 9001\n    gpu_ids: [0]\n",
		"instance negative gpu":    "instances:\n  - name: a\n    model: /m.gguf\n    port: 9001\n    gpu_ids: [-1]\n",
		"duplicate name":           inst("  - name: a\n    model: /m.gguf\n    port: 9002\n    gpu_ids: [1]\n"),
		"duplicate port":           inst("  - name: b\n    model: /m.gguf\n    port: 9001\n    gpu_ids: [1]\n"),
		"instance on manager port": "instances:\n  - name: a\n    model: /m.gguf\n    port: 8080\n    gpu_ids: [0]\n",
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := loadConfig(writeConfig(t, validBase+body)); err == nil {
				t.Fatalf("loadConfig accepted %q", body)
			}
		})
	}
}

func TestLoadConfigAcceptsExampleConfig(t *testing.T) {
	data, err := os.ReadFile("example.config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, data, 0o640); err != nil {
		t.Fatal(err)
	}
	cfg, err := loadConfig(path)
	if err != nil {
		t.Fatalf("example.config.yaml no longer loads: %v", err)
	}
	if len(cfg.Instances) == 0 || len(cfg.Instances[0].GPUIDs) != 1 {
		t.Fatalf("legacy gpu_id not migrated: %+v", cfg.Instances)
	}
}

func TestLoadConfigReportsEveryProblem(t *testing.T) {
	_, err := loadConfig(writeConfig(t, validBase+"health_check_interval: 0s\nrestart_delay: 0s\n"))
	if err == nil || !strings.Contains(err.Error(), "health_check_interval") || !strings.Contains(err.Error(), "restart_delay") {
		t.Fatalf("error = %v, want both problems reported", err)
	}
}

func TestNextHealthDelayGuardsNonPositiveInterval(t *testing.T) {
	if got := nextHealthDelay(0, StateRunning); got <= 0 {
		t.Fatalf("nextHealthDelay(0) = %s; a non-positive delay spins the health loop", got)
	}
	if got := nextHealthDelay(-time.Second, StateRunning); got <= 0 {
		t.Fatalf("nextHealthDelay(-1s) = %s", got)
	}
}

// TestLoadConfigRejectsUnknownKeys (P2): typo'd keys were silently ignored, so
// "gpu_backed: rocm" left the backend on vulkan and every instance got the
// wrong GPU-visibility variable. The custom InstanceConf unmarshaler also
// defeated strict mode for keys inside instances.
func TestLoadConfigRejectsUnknownKeys(t *testing.T) {
	for _, body := range []string{
		"gpu_backed: rocm\n",
		"instances:\n  - name: a\n    model: /m.gguf\n    port: 9001\n    gpu_ids: [0]\n    gpuz: [1]\n",
	} {
		_, err := loadConfig(writeConfig(t, validBase+body))
		if err == nil || !strings.Contains(err.Error(), "not found") {
			t.Errorf("loadConfig(%q) err = %v, want an unknown-field error", body, err)
		}
	}
}

func TestLegacyGPUIDMigratesAndIsNotWrittenBack(t *testing.T) {
	path := writeConfig(t, validBase+"instances:\n  - name: a\n    model: /m.gguf\n    port: 9001\n    gpu_id: 3\n")
	cfg, err := loadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if ids := cfg.Instances[0].GPUIDs; len(ids) != 1 || ids[0] != 3 {
		t.Fatalf("gpu_id not migrated: %v", ids)
	}
	if err := cfg.AddInstance(InstanceConf{Name: "b", Model: "/m.gguf", Port: 9002, GPUIDs: []int{0}}); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(path)
	if strings.Contains(string(data), "gpu_id:") {
		t.Fatalf("legacy gpu_id written back:\n%s", data)
	}
	if _, err := loadConfig(path); err != nil {
		t.Fatalf("saved config no longer loads: %v", err)
	}
}

// TestGPUIDThroughMergeKey: the old custom unmarshaler only looked at direct
// keys, so gpu_id inherited through a YAML merge key was lost.
func TestGPUIDThroughMergeKey(t *testing.T) {
	body := validBase + `instances:
  - &base
    name: a
    model: /m.gguf
    port: 9001
    gpu_id: 2
  - <<: *base
    name: b
    port: 9002
`
	cfg, err := loadConfig(writeConfig(t, body))
	if err != nil {
		t.Fatal(err)
	}
	if ids := cfg.Instances[1].GPUIDs; len(ids) != 1 || ids[0] != 2 {
		t.Fatalf("merged instance gpu_ids = %v, want [2]", ids)
	}
}
