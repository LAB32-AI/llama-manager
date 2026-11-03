package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func stubGPUTools(t *testing.T, onPath map[string]bool, run func(name string, args ...string) ([]byte, error)) {
	t.Helper()
	oldRun, oldLook := runGPUTool, lookGPUTool
	gpuMon.mu.Lock()
	gpuMon.lastAt = time.Time{}
	gpuMon.mu.Unlock()
	runGPUTool = func(_ context.Context, name string, args ...string) ([]byte, error) { return run(name, args...) }
	lookGPUTool = func(name string) (string, error) {
		if onPath[name] {
			return "/usr/bin/" + name, nil
		}
		return "", errors.New("not found")
	}
	t.Cleanup(func() {
		runGPUTool, lookGPUTool = oldRun, oldLook
		gpuMon.mu.Lock()
		gpuMon.lastAt = time.Time{}
		gpuMon.mu.Unlock()
	})
}

// Vulkan is the default backend, and before this the dashboard showed no GPU
// stats on it at all. rocm-smi's cards are renumbered by PCI bus, the order
// the Vulkan loader (and so GGML_VK_VISIBLE_DEVICES) uses.
func TestVulkanUsesROCmSMIInPCIOrder(t *testing.T) {
	const out = `{
 "card0": {"Card Series": "MI50 on bus 11", "PCI Bus": "0000:11:00.0", "GPU use (%)": "10"},
 "card1": {"Card Series": "MI50 on bus 0B", "PCI Bus": "0000:0B:00.0", "GPU use (%)": "97"}
}`
	stubGPUTools(t, map[string]bool{"rocm-smi": true}, func(name string, _ ...string) ([]byte, error) {
		if name != "rocm-smi" {
			t.Errorf("ran %s", name)
		}
		return []byte(out), nil
	})
	stats, errMsg := readGPUs("vulkan")
	if errMsg != "" || len(stats) != 2 {
		t.Fatalf("stats %+v, err %q", stats, errMsg)
	}
	if stats[0].Index != 0 || stats[0].Name != "MI50 on bus 0B" || stats[1].Name != "MI50 on bus 11" {
		t.Errorf("want PCI bus order, not rocm-smi card order: %+v", stats)
	}
	if m := pciBusOrder([]byte(`{"card0": {"PCI Bus": "N/A"}}`)); m != nil {
		t.Errorf("a card without a bus keeps rocm-smi's order, got %v", m)
	}
}

func TestVulkanFallsBackToNvidiaSMIAndToNothing(t *testing.T) {
	stubGPUTools(t, map[string]bool{"nvidia-smi": true}, func(name string, _ ...string) ([]byte, error) {
		return []byte("0, NVIDIA RTX 3090, 88, 20000, 24576, 71, 310.5\n"), nil
	})
	stats, _ := readGPUs("vulkan")
	if len(stats) != 1 || stats[0].UtilPct != 88 || stats[0].TempC != 71 {
		t.Fatalf("nvidia-smi on vulkan: %+v", stats)
	}

	stubGPUTools(t, nil, func(name string, _ ...string) ([]byte, error) {
		t.Errorf("no tool is installed, but %s ran", name)
		return nil, errors.New("unreachable")
	})
	if stats, errMsg := readGPUs("vulkan"); len(stats) != 0 || errMsg != "" {
		t.Errorf("without a vendor tool there is nothing to show and nothing to report: %+v %q", stats, errMsg)
	}
}

// The chat polls GPUs faster while a reply is generating; with several tabs
// open the expired cache must not start one SMI process per request.
func TestReadGPUsRunsOneToolAtATime(t *testing.T) {
	var runs atomic.Int32
	stubGPUTools(t, nil, func(string, ...string) ([]byte, error) {
		runs.Add(1)
		time.Sleep(30 * time.Millisecond)
		return []byte("0, GPU, 50, 1, 2, 40, 100\n"), nil
	})
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() { defer wg.Done(); readGPUs("cuda") }()
	}
	wg.Wait()
	if n := runs.Load(); n != 1 {
		t.Errorf("nvidia-smi ran %d times for 8 concurrent requests", n)
	}
	// A backend change in settings must not be answered from the old cache.
	readGPUs("vulkan")
	if stats, _ := readGPUs("vulkan"); len(stats) != 0 {
		t.Errorf("vulkan without tools served cached cuda stats: %+v", stats)
	}
}

const ioregM2 = `+-o AGXAcceleratorG14X  <class AGXAcceleratorG14X, id 0x1000003c5, registered, matched, active, busy 0 (0 ms), retain 75>
    {
      "IOClass" = "AGXAcceleratorG14X"
      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=9564962816,"Tiler Utilization %"=44,"Renderer Utilization %"=43,"Device Utilization %"=44,"In use system memory"=2843033600}
      "model" = "Apple M2 Max"
      "gpu-core-count" = 38
    }
`

func TestParseIOReg(t *testing.T) {
	stats := parseIOReg([]byte(ioregM2), 65536)
	if len(stats) != 1 {
		t.Fatalf("got %+v", stats)
	}
	g := stats[0]
	if g.Name != "Apple M2 Max" || g.UtilPct != 44 || g.MemUsedMB != 2711 || g.MemTotalMB != 65536 || g.TempC != 0 || g.Backend != "metal" {
		t.Errorf("got %+v", g)
	}
	if got := parseIOReg([]byte("+-o IOAccelerator\n {}\n"), 0); len(got) != 0 {
		t.Errorf("an accelerator without statistics is skipped, got %+v", got)
	}
}

func TestDiskUsageGroupsModelsByFilesystem(t *testing.T) {
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a.gguf"), filepath.Join(dir, "sub", "b.gguf")
	os.MkdirAll(filepath.Dir(b), 0o755)
	os.WriteFile(a, nil, 0o644)
	os.WriteFile(b, nil, 0o644)
	models := []CachedModel{{Path: a, SizeMB: 1200}, {Path: b, SizeMB: 300}, {Path: filepath.Join(dir, "gone.gguf"), SizeMB: 5}}

	disks, idx := diskUsage(models, filepath.Join(dir, "not", "created", "yet"))
	if len(disks) != 1 {
		t.Fatalf("one temp dir is one filesystem, got %+v", disks)
	}
	d := disks[0]
	if !d.Downloads || d.ModelsMB != 1500 || d.TotalMB <= 0 || d.FreeMB <= 0 || d.FreeMB > d.TotalMB {
		t.Errorf("disk %+v", d)
	}
	if idx[0] != 0 || idx[1] != 0 || idx[2] != -1 {
		t.Errorf("model disk indices %v; a missing file is unknown", idx)
	}
	if d.Mount == "" || !filepath.IsAbs(d.Mount) {
		t.Errorf("mount %q", d.Mount)
	}
	if rel, err := filepath.Rel(d.Mount, dir); err != nil || rel == ".." || filepath.IsAbs(rel) {
		t.Errorf("mount %q should contain %q", d.Mount, dir)
	}
}

func TestModelsAPIReportsDisks(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CACHE_HOME", "")
	t.Setenv("LLAMA_CACHE", "")
	t.Setenv("HF_HUB_CACHE", filepath.Join(home, "hub"))
	dir := filepath.Join(home, "models")
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "m-Q4_K_M.gguf"), make([]byte, 2<<20), 0o644)

	ws, cfg := newTestServer(t)
	cfg.ModelDirs = []string{dir}
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/api/models", nil))
	var got struct {
		Models []CachedModel `json:"models"`
		Disks  []DiskUsage   `json:"disks"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatal(rr.Body.String())
	}
	if len(got.Disks) != 1 || !got.Disks[0].Downloads || got.Disks[0].ModelsMB != 2 {
		t.Fatalf("disks %+v", got.Disks)
	}
	if len(got.Models) != 1 || got.Models[0].Disk != 0 {
		t.Errorf("models %+v", got.Models)
	}
}
