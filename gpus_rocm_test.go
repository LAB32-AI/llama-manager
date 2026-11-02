package main

import (
	"context"
	"errors"
	"testing"
	"time"
)

func resetHIPCache(t *testing.T) {
	t.Helper()
	hipMu.Lock()
	hipMap, hipRetryAt = nil, time.Time{}
	hipMu.Unlock()
	oldRun, oldRetry := runRocminfo, hipRetryInterval
	t.Cleanup(func() {
		hipMu.Lock()
		hipMap, hipRetryAt = nil, time.Time{}
		hipMu.Unlock()
		runRocminfo, hipRetryInterval = oldRun, oldRetry
	})
}

const rocminfoTwoGPUs = `
  Name:                    gfx906
  Device Type:             GPU
  BDFID:                   4352
  Name:                    gfx906
  Device Type:             GPU
  BDFID:                   2816
`

// TestHIPIndexRetriesAfterFailure (P2): a single rocminfo failure (sync.Once)
// pinned the dashboard to rocm-smi's PCI order for the life of the process.
func TestHIPIndexRetriesAfterFailure(t *testing.T) {
	resetHIPCache(t)
	hipRetryInterval = 50 * time.Millisecond
	calls := 0
	runRocminfo = func(context.Context) ([]byte, error) {
		calls++
		if calls == 1 {
			return nil, errors.New("HSA init timed out")
		}
		return []byte(rocminfoTwoGPUs), nil
	}

	if m := hipBusIndex(); m != nil {
		t.Fatalf("first (failed) call returned %v", m)
	}
	if hipBusIndex(); calls != 1 {
		t.Fatalf("retried immediately (%d calls); should wait hipRetryInterval", calls)
	}
	time.Sleep(60 * time.Millisecond)
	m := hipBusIndex()
	if m[0x11] != 0 || m[0x0B] != 1 {
		t.Fatalf("map after retry = %v", m)
	}
	hipBusIndex()
	if calls != 2 {
		t.Fatalf("success not cached: %d calls", calls)
	}
}

const rocmSMIJSON = `{
 "card0": {"Card Series": "Navi 23 (display)", "PCI Bus": "0000:03:00.0", "GPU use (%)": "1", "VRAM Total Memory (B)": "8589934592", "VRAM Total Used Memory (B)": "1073741824"},
 "card1": {"Card series": "Vega 20 [MI50]", "PCI Bus": "0000:0B:00.0", "GPU Use (%)": "97", "VRAM Total Memory (B)": "34342961152", "VRAM Total Used Memory (B)": "17171480576", "Temperature (Sensor junction) (C)": "71.0", "Average Graphics Package Power (W)": "212.0"},
 "card2": {"Card Series": "Vega 20 [MI50]", "PCI Bus": "0000:11:00.0", "GPU use (%)": "N/A", "VRAM Total Memory (B)": "34342961152", "VRAM Total Used Memory (B)": "0"},
 "system": {"Driver version": "6.8"}
}`

// TestParseROCmSMI covers the JSON -> stats mapping, which was untested
// because it was tied to exec. Bus 0x11 is HIP 0 and 0x0B is HIP 1 (rocminfo
// order), the display card is unknown to rocminfo.
func TestParseROCmSMI(t *testing.T) {
	stats, err := parseROCmSMI([]byte(rocmSMIJSON), map[int]int{0x11: 0, 0x0B: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(stats) != 3 {
		t.Fatalf("got %d cards", len(stats))
	}
	byIdx := map[int]GPUStat{}
	for _, s := range stats {
		if _, dup := byIdx[s.Index]; dup {
			t.Fatalf("duplicate index %d: %+v", s.Index, stats) // P2: unmapped cards collided
		}
		byIdx[s.Index] = s
	}
	if g := byIdx[1]; g.Name != "Vega 20 [MI50]" || g.UtilPct != 97 || g.MemTotalMB != 32752 || g.MemUsedMB != 16376 || g.TempC != 71 || g.PowerW != 212 {
		t.Errorf("HIP 1 = %+v", g)
	}
	if g := byIdx[0]; g.MemUsedMB != 0 || g.UtilPct != 0 {
		t.Errorf("HIP 0 = %+v (N/A must read as 0)", g)
	}
	if g := byIdx[2]; g.Name != "Navi 23 (display)" {
		t.Errorf("unmapped display card should come after the HIP devices, got %+v", byIdx)
	}
}

func TestParseROCmSMIWithoutMappingKeepsCardOrder(t *testing.T) {
	stats, err := parseROCmSMI([]byte(rocmSMIJSON), nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, s := range stats {
		if s.Index != i {
			t.Fatalf("without rocminfo indices should follow cardN: %+v", stats)
		}
	}
}
