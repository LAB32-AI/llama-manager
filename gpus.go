package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// GPUStat is the per-card snapshot exposed by /api/gpus.
type GPUStat struct {
	Index      int     `json:"index"`
	Name       string  `json:"name"`
	Backend    string  `json:"backend"`
	UtilPct    float64 `json:"util_pct"`
	MemUsedMB  int64   `json:"mem_used_mb"`
	MemTotalMB int64   `json:"mem_total_mb"`
	TempC      float64 `json:"temp_c"`
	PowerW     float64 `json:"power_w"`
}

const gpuCacheTTL = 1500 * time.Millisecond

type gpuMonitor struct {
	mu      sync.Mutex
	last    []GPUStat
	lastAt  time.Time
	lastErr string
}

var gpuMon = &gpuMonitor{}

// readGPUs collects per-GPU stats appropriate for the configured backend.
// rocm-smi is shelled out for "rocm"/"rocm_rocr"; nvidia-smi for "cuda".
// Metal/vulkan/unknown backends return an empty slice (no portable cross-
// vendor query exists). Results are cached briefly to avoid hammering the
// SMI tools on every UI poll.
func readGPUs(backend string) ([]GPUStat, string) {
	gpuMon.mu.Lock()
	if time.Since(gpuMon.lastAt) < gpuCacheTTL {
		stats := gpuMon.last
		err := gpuMon.lastErr
		gpuMon.mu.Unlock()
		return stats, err
	}
	gpuMon.mu.Unlock()

	var (
		stats  []GPUStat
		errMsg string
	)
	switch backend {
	case "rocm", "rocm_rocr":
		stats, errMsg = readGPUsROCm()
	case "cuda":
		stats, errMsg = readGPUsCUDA()
	default:
		// no portable query for metal / vulkan / unset
	}

	gpuMon.mu.Lock()
	gpuMon.last = stats
	gpuMon.lastErr = errMsg
	gpuMon.lastAt = time.Now()
	gpuMon.mu.Unlock()
	return stats, errMsg
}

func readGPUsCUDA() ([]GPUStat, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx,
		"nvidia-smi",
		"--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
		"--format=csv,noheader,nounits",
	)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Sprintf("nvidia-smi: %v", err)
	}
	var stats []GPUStat
	rdr := csv.NewReader(strings.NewReader(string(out)))
	rdr.TrimLeadingSpace = true
	rows, err := rdr.ReadAll()
	if err != nil {
		return nil, fmt.Sprintf("nvidia-smi csv: %v", err)
	}
	for _, r := range rows {
		if len(r) < 7 {
			continue
		}
		stats = append(stats, GPUStat{
			Index:      atoi(r[0]),
			Name:       strings.TrimSpace(r[1]),
			Backend:    "cuda",
			UtilPct:    atof(r[2]),
			MemUsedMB:  int64(atof(r[3])),
			MemTotalMB: int64(atof(r[4])),
			TempC:      atof(r[5]),
			PowerW:     atof(r[6]),
		})
	}
	return stats, ""
}

// readGPUsROCm shells out to `rocm-smi --json` with several --show* flags and
// pulls the numbers we care about. rocm-smi's JSON shape is keyed by "card0",
// "card1", ..., each value being a map of strings. The exact field names have
// changed over rocm-smi versions, so we look for several likely spellings.
func readGPUsROCm() ([]GPUStat, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx,
		"rocm-smi",
		"--showproductname",
		"--showuse",
		"--showmemuse",
		"--showmeminfo", "vram",
		"--showtemp",
		"--showpower",
		"--showbus",
		"--json",
	)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Sprintf("rocm-smi: %v", err)
	}
	var raw map[string]map[string]string
	if err := json.Unmarshal(out, &raw); err != nil {
		return nil, fmt.Sprintf("rocm-smi json: %v", err)
	}

	type card struct {
		idx int
		m   map[string]string
	}
	cards := make([]card, 0, len(raw))
	for k, v := range raw {
		if !strings.HasPrefix(k, "card") {
			continue
		}
		idx, err := strconv.Atoi(strings.TrimPrefix(k, "card"))
		if err != nil {
			continue
		}
		cards = append(cards, card{idx, v})
	}
	// stable order by index
	for i := 1; i < len(cards); i++ {
		for j := i; j > 0 && cards[j-1].idx > cards[j].idx; j-- {
			cards[j-1], cards[j] = cards[j], cards[j-1]
		}
	}

	// rocm-smi numbers cards by PCI bus, but instances are placed by HIP device
	// index (gpu_id -> HIP_VISIBLE_DEVICES), which follows rocminfo's agent
	// order. Remap so the reported Index matches the gpu_id a user assigns;
	// fall back to rocm-smi's order if the mapping can't be built.
	busToHIP := hipBusIndex()

	stats := make([]GPUStat, 0, len(cards))
	for _, c := range cards {
		idx := c.idx
		if busToHIP != nil {
			if hip, ok := busToHIP[parsePCIBus(firstOf(c.m, "PCI Bus"))]; ok {
				idx = hip
			}
		}
		stats = append(stats, GPUStat{
			Index:      idx,
			Name:       firstOf(c.m, "Card Series", "Card Model", "Card SKU", "GPU"),
			Backend:    "rocm",
			UtilPct:    parseNum(firstOf(c.m, "GPU use (%)", "GPU Use (%)")),
			MemUsedMB:  int64(parseBytes(firstOf(c.m, "VRAM Total Used Memory (B)")) / (1024 * 1024)),
			MemTotalMB: int64(parseBytes(firstOf(c.m, "VRAM Total Memory (B)")) / (1024 * 1024)),
			TempC:      parseNum(firstOf(c.m, "Temperature (Sensor edge) (C)", "Temperature (Sensor junction) (C)", "Temperature (Sensor memory) (C)")),
			PowerW:     parseNum(firstOf(c.m, "Average Graphics Package Power (W)", "Current Socket Graphics Package Power (W)", "GPU Power (W)")),
		})
	}
	sort.Slice(stats, func(i, j int) bool { return stats[i].Index < stats[j].Index })
	return stats, ""
}

// hipBusIndex maps a PCI bus byte to the HIP/ROCr device index by parsing
// rocminfo's agent order (cached; rocminfo output is static for a boot). HIP
// numbers GPUs in rocminfo agent order, which differs from rocm-smi's PCI-bus
// card numbering — so an instance launched with HIP_VISIBLE_DEVICES=N can show
// up under a different "cardN" in rocm-smi. Returns nil on any failure, in
// which case callers keep rocm-smi's native order.
var (
	hipMapOnce sync.Once
	hipMap     map[int]int
)

func hipBusIndex() map[int]int {
	hipMapOnce.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, "rocminfo").Output()
		if err != nil {
			return
		}
		if m := parseHIPBusIndex(bytes.NewReader(out)); len(m) > 0 {
			hipMap = m
		}
	})
	return hipMap
}

// parseHIPBusIndex reads rocminfo output and returns pci-bus-byte -> HIP index.
// GPU agents are numbered in the order they appear; "Device Type: GPU" marks a
// GPU agent and the following "BDFID:" line carries its bus (bits 15..8).
func parseHIPBusIndex(r io.Reader) map[int]int {
	m := make(map[int]int)
	hip := 0
	curIsGPU := false
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		switch {
		case strings.HasPrefix(line, "Device Type:"):
			curIsGPU = strings.Contains(line, "GPU")
		case strings.HasPrefix(line, "BDFID:"):
			fields := strings.Fields(line)
			if curIsGPU && len(fields) >= 2 {
				if bdf, err := strconv.Atoi(fields[1]); err == nil {
					m[(bdf>>8)&0xff] = hip
					hip++
				}
			}
			curIsGPU = false
		}
	}
	return m
}

// parsePCIBus extracts the bus byte from a rocm-smi "PCI Bus" string such as
// "0000:0B:00.0" (-> 0x0B). Returns -1 if unparseable.
func parsePCIBus(s string) int {
	parts := strings.Split(strings.TrimSpace(s), ":")
	if len(parts) < 2 {
		return -1
	}
	v, err := strconv.ParseInt(parts[1], 16, 32)
	if err != nil {
		return -1
	}
	return int(v)
}

// firstOf returns the first non-empty value among the candidate keys. The
// lookup is case-insensitive because rocm-smi's JSON key casing has drifted
// across versions ("Card series" vs "Card Series", "GPU use (%)" vs
// "GPU Use (%)").
func firstOf(m map[string]string, keys ...string) string {
	for _, k := range keys {
		if v, ok := m[k]; ok && validField(v) {
			return strings.TrimSpace(v)
		}
	}
	for _, k := range keys {
		for mk, v := range m {
			if strings.EqualFold(mk, k) && validField(v) {
				return strings.TrimSpace(v)
			}
		}
	}
	return ""
}

func validField(v string) bool {
	v = strings.TrimSpace(v)
	return v != "" && v != "N/A"
}

func atoi(s string) int {
	n, _ := strconv.Atoi(strings.TrimSpace(s))
	return n
}

func atof(s string) float64 {
	s = strings.TrimSpace(s)
	if s == "" || s == "[N/A]" || strings.EqualFold(s, "n/a") {
		return 0
	}
	f, _ := strconv.ParseFloat(s, 64)
	return f
}

func parseNum(s string) float64 {
	return atof(s)
}

func parseBytes(s string) float64 {
	return atof(s)
}
