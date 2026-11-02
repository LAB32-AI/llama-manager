package main

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// TestSettingsUpdateDuringSupervisionIsRaceFree is the P1 #7 regression: the
// restart loop and health loop read max_restarts, restart_delay,
// health_check_interval and unhealthy_after without cfg.mu while PUT
// /api/settings wrote them. Run under -race.
func TestSettingsUpdateDuringSupervisionIsRaceFree(t *testing.T) {
	rig := newFakeRig(t, "crash-after", "crashy", "steady")
	rig.cfg.RestartDelay = duration{5 * time.Millisecond}
	rig.cfg.HealthCheckInterval = duration{10 * time.Millisecond}
	rig.cfg.MaxRestarts = 0
	*rig.cfg.Instances[0].ExtraArgs = append(*rig.cfg.Instances[0].ExtraArgs, "--fake-crash-after", "20ms")
	*rig.cfg.Instances[1].ExtraArgs = []string{"--fake-mode", "healthy", "--fake-pid-dir", rig.pidDir}
	m := NewManager(rig.cfg)
	defer m.Shutdown()

	for _, n := range []string{"crashy", "steady"} {
		if err := m.StartInstance(n); err != nil {
			t.Fatal(err)
		}
	}
	waitFor(t, 3*time.Second, "a few restarts", func() bool { return m.Get("crashy").RestartCount() >= 2 })

	deadline := time.Now().Add(400 * time.Millisecond)
	for i := 0; time.Now().Before(deadline); i++ {
		err := rig.cfg.UpdateSettings(Settings{
			RestartDelay:        []string{"5ms", "7ms"}[i%2],
			HealthCheckInterval: []string{"10ms", "12ms"}[i%2],
			MaxRestarts:         i % 2,
			NGL:                 1,
			ContextLength:       128,
		})
		if err != nil {
			t.Fatal(err)
		}
		rig.cfg.mu.Lock()
		rig.cfg.UnhealthyAfter = i % 3
		rig.cfg.mu.Unlock()
		time.Sleep(time.Millisecond)
	}
}

// TestDownloadStartReadsStatusUnderJobLock: Start read job.Status under dm.mu
// only, while the worker wrote it under job.mu (P1 #7). Run under -race.
func TestDownloadStartReadsStatusUnderJobLock(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer srv.Close()
	oldAPI := hfAPIBase
	hfAPIBase = srv.URL
	defer func() { hfAPIBase = oldAPI }()

	dm := NewDownloadManager()
	job := &DownloadJob{Repo: "owner/name", Status: "downloading", cancel: func() {}}
	dm.active = job

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			job.finish("stopped", "stop")
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			_ = dm.Start("owner/name", "Q4_K_M")
		}
	}()
	wg.Wait()
}
