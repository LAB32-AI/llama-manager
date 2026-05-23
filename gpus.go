package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// GPUStat is the per-card snapshot exposed by /api/gpus.
type GPUStat struct {
	Index       int     `json:"index"`
	Name        string  `json:"name"`
	Backend     string  `json:"backend"`
	UtilPct     float64 `json:"util_pct"`
	MemUsedMB   int64   `json:"mem_used_mb"`
	MemTotalMB  int64   `json:"mem_total_mb"`
	TempC       float64 `json:"temp_c"`
	PowerW      float64 `json:"power_w"`
}

const gpuCacheTTL = 1500 * time.Millisecond

type gpuMonitor struct {
	mu       sync.Mutex
	last     []GPUStat
	lastAt   time.Time
	lastErr  string
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
		stats []GPUStat
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

	stats := make([]GPUStat, 0, len(cards))
	for _, c := range cards {
		stats = append(stats, GPUStat{
			Index:      c.idx,
			Name:       firstOf(c.m, "Card series", "Card SKU", "Card model", "GPU"),
			Backend:    "rocm",
			UtilPct:    parseNum(firstOf(c.m, "GPU use (%)", "GPU Use (%)")),
			MemUsedMB:  int64(parseBytes(firstOf(c.m, "VRAM Total Used Memory (B)")) / (1024 * 1024)),
			MemTotalMB: int64(parseBytes(firstOf(c.m, "VRAM Total Memory (B)")) / (1024 * 1024)),
			TempC:      parseNum(firstOf(c.m, "Temperature (Sensor edge) (C)", "Temperature (Sensor junction) (C)", "Temperature (Sensor memory) (C)")),
			PowerW:     parseNum(firstOf(c.m, "Average Graphics Package Power (W)", "Current Socket Graphics Package Power (W)", "GPU Power (W)")),
		})
	}
	return stats, ""
}

func firstOf(m map[string]string, keys ...string) string {
	for _, k := range keys {
		if v, ok := m[k]; ok && strings.TrimSpace(v) != "" && v != "N/A" {
			return strings.TrimSpace(v)
		}
	}
	return ""
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
