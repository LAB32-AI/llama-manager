package main

import (
	"strings"
	"testing"
)

// TestParseHIPBusIndex uses the real topology from the MI50 rig: rocminfo lists
// a CPU agent (BDFID 0, skipped) then four GPU agents whose BDFIDs decode to
// buses 0x11, 0x14, 0x0B, 0x0E — the order HIP/gpu_id uses, which is NOT the
// PCI-bus-ascending order rocm-smi reports.
func TestParseHIPBusIndex(t *testing.T) {
	const rocminfo = `
*******
Agent 1
*******
  Name:                    AMD CPU
  Device Type:             CPU
  BDFID:                   0
*******
Agent 2
*******
  Name:                    gfx906
  Device Type:             GPU
  BDFID:                   4352
*******
Agent 3
*******
  Device Type:             GPU
  BDFID:                   5120
*******
Agent 4
*******
  Device Type:             GPU
  BDFID:                   2816
*******
Agent 5
*******
  Device Type:             GPU
  BDFID:                   3584
`
	got := parseHIPBusIndex(strings.NewReader(rocminfo))
	want := map[int]int{0x11: 0, 0x14: 1, 0x0B: 2, 0x0E: 3}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for bus, idx := range want {
		if got[bus] != idx {
			t.Errorf("bus 0x%X -> %d, want %d", bus, got[bus], idx)
		}
	}
	// The CPU agent (BDFID 0) must not be assigned a HIP index.
	if _, ok := got[0]; ok {
		t.Error("CPU agent (bus 0) was mapped to a HIP index")
	}
}

func TestParsePCIBus(t *testing.T) {
	cases := map[string]int{
		"0000:0B:00.0": 0x0B,
		"0000:11:00.0": 0x11,
		"0000:14:00.0": 0x14,
		"garbage":      -1,
		"":             -1,
	}
	for in, want := range cases {
		if got := parsePCIBus(in); got != want {
			t.Errorf("parsePCIBus(%q) = %d, want %d", in, got, want)
		}
	}
}
