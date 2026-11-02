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
