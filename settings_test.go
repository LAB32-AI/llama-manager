package main

import (
	"testing"
	"time"
)

// TestRejectedSettingsUpdateChangesNothing (P2): restart_delay was assigned
// before health_check_interval was parsed, so a PUT that failed validation
// still changed the live restart delay, and the next save persisted it.
func TestRejectedSettingsUpdateChangesNothing(t *testing.T) {
	cfg, _ := newSavedConfig(t)
	before := cfg.GetSettings()

	err := cfg.UpdateSettings(Settings{RestartDelay: "77s", HealthCheckInterval: "bogus", NGL: intp(5), ContextLength: intp(128)})
	if err == nil {
		t.Fatal("invalid health_check_interval accepted")
	}
	if after := cfg.GetSettings(); after.RestartDelay != before.RestartDelay || *after.NGL != *before.NGL {
		t.Fatalf("rejected update changed settings: restart_delay %s -> %s, ngl %d -> %d",
			before.RestartDelay, after.RestartDelay, *before.NGL, *after.NGL)
	}
}

func TestParsePositiveDuration(t *testing.T) {
	if d, err := parsePositiveDuration("x", ""); d != 0 || err != nil {
		t.Errorf(`"" = %s, %v; want 0, nil (unset)`, d, err)
	}
	if d, err := parsePositiveDuration("x", "90s"); d != 90*time.Second || err != nil {
		t.Errorf(`"90s" = %s, %v`, d, err)
	}
	for _, bad := range []string{"0s", "-1s", "soon"} {
		if _, err := parsePositiveDuration("x", bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
}

func intp(n int) *int { return &n }

// TestPartialSettingsUpdateKeepsOmittedNumbers (P2): omitted ints decoded as 0
// and were applied, so an API client changing only ngl silently set
// max_restarts to 0 (unlimited restarts) and main_gpu to 0.
func TestPartialSettingsUpdateKeepsOmittedNumbers(t *testing.T) {
	cfg, _ := newSavedConfig(t)
	if err := cfg.UpdateSettings(Settings{MaxRestarts: intp(7), MainGPU: intp(2), ContextLength: intp(4096)}); err != nil {
		t.Fatal(err)
	}
	if err := cfg.UpdateSettings(Settings{NGL: intp(42)}); err != nil {
		t.Fatal(err)
	}
	s := cfg.GetSettings()
	if *s.NGL != 42 || *s.MaxRestarts != 7 || *s.MainGPU != 2 || *s.ContextLength != 4096 {
		t.Fatalf("got ngl=%d max_restarts=%d main_gpu=%d context=%d; omitted fields must be kept",
			*s.NGL, *s.MaxRestarts, *s.MainGPU, *s.ContextLength)
	}
}
