package main

import (
	"net/http"
	"strings"
	"testing"
)

func TestValidateModelRef(t *testing.T) {
	ok := []string{
		"/models/GLM-4.5-Air-Q4_K_M.gguf",
		"relative/dir/model.gguf",
		"/opt/weird name/with spaces/model", // absolute path: -m, not an HF reference
		"bartowski/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-GGUF:IQ4_XS",
		"unsloth/GLM-4.5-Air-GGUF:UD-Q4_K_XL",
		"owner/repo",
	}
	bad := []string{
		"anything/at all:../x",
		"owner/repo; rm -rf /",
		"--model-url=http://evil.example/x",
		"owner/repo:--hf-token",
		"owner/repo:Q4 K",
		"../..",
		"owner/repo/extra",
		"just-a-name",
	}
	for _, m := range ok {
		if err := validateModelRef(m); err != nil {
			t.Errorf("validateModelRef(%q) = %v, want nil", m, err)
		}
	}
	for _, m := range bad {
		if err := validateModelRef(m); err == nil {
			t.Errorf("validateModelRef(%q) = nil, want error", m)
		}
	}
}

// TestInstanceModelIsValidatedEverywhere is the P1 #16 regression: an
// instance's model went to `llama-server -hf <model>` without validateRepo or
// validateQuant, breaking the CLAUDE.md rule for HF inputs.
func TestInstanceModelIsValidatedEverywhere(t *testing.T) {
	const evil = "anything/at all:../x"

	if _, err := loadConfig(writeConfig(t, validBase+"instances:\n  - name: a\n    model: \""+evil+"\"\n    port: 9001\n    gpu_ids: [0]\n")); err == nil {
		t.Error("loadConfig accepted an invalid HF model reference")
	}

	ws, _ := newTestServer(t)
	rr := doJSON(t, ws, http.MethodPost, "/api/config/instances", InstanceConf{Name: "a", Model: evil, Port: 9001, GPUIDs: []int{0}})
	if rr.Code != http.StatusBadRequest {
		t.Errorf("POST instance with invalid model: got %d, want 400", rr.Code)
	}

	rig := newFakeRig(t, "healthy", "a")
	rig.cfg.Instances[0].Model = evil
	inst := NewInstance(rig.cfg.Instances[0], rig.cfg)
	if _, err := inst.Start(); err == nil || !strings.Contains(err.Error(), "owner/repo") {
		t.Errorf("Instance.Start with invalid model: err = %v, want a validation error before exec", err)
	}
	if inst.currentPid() != 0 {
		t.Error("a process was started for an invalid model reference")
	}
}
