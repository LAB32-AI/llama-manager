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
	"regexp"
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
	fetch   sync.Mutex // one SMI run at a time; callers that waited reuse its result
	backend string
	last    []GPUStat
	lastAt  time.Time
	lastErr string
}

var gpuMon = &gpuMonitor{}

func (m *gpuMonitor) cached(backend string) ([]GPUStat, string, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.backend == backend && !m.lastAt.IsZero() && time.Since(m.lastAt) < gpuCacheTTL {
		return m.last, m.lastErr, true
	}
	return nil, "", false
}

// Seams for tests: the SMI tools are only present on real rigs.
var (
	runGPUTool = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		return exec.CommandContext(ctx, name, args...).Output()
	}
	lookGPUTool = exec.LookPath
)

// readGPUs collects per-GPU stats for the configured backend. Results are
// cached briefly so several open dashboards, and the chat's faster polling
// while a reply is generating, run the SMI tool at most once per TTL.
func readGPUs(backend string) ([]GPUStat, string) {
	if stats, errMsg, ok := gpuMon.cached(backend); ok {
		return stats, errMsg
	}
	gpuMon.fetch.Lock()
	defer gpuMon.fetch.Unlock()
	if stats, errMsg, ok := gpuMon.cached(backend); ok {
		return stats, errMsg
	}
	stats, errMsg := queryGPUs(backend)
	gpuMon.mu.Lock()
	gpuMon.backend, gpuMon.last, gpuMon.lastErr, gpuMon.lastAt = backend, stats, errMsg, time.Now()
	gpuMon.mu.Unlock()
	return stats, errMsg
}

func queryGPUs(backend string) ([]GPUStat, string) {
	switch backend {
	case "rocm", "rocm_rocr":
		return readGPUsROCm(func([]byte) map[int]int { return hipBusIndex() })
	case "cuda":
		return readGPUsCUDA()
	case "metal":
		return readGPUsMetal()
	case "vulkan":
		// Vulkan has no stats query of its own, so use the vendor tool when
		// one is installed. The Linux Vulkan loader lists discrete GPUs by PCI
		// bus, which is the order GGML_VK_VISIBLE_DEVICES indexes; nvidia-smi
		// already numbers cards that way.
		if _, err := lookGPUTool("rocm-smi"); err == nil {
			return readGPUsROCm(pciBusOrder)
		}
		if _, err := lookGPUTool("nvidia-smi"); err == nil {
			return readGPUsCUDA()
		}
	}
	return nil, ""
}

func readGPUsCUDA() ([]GPUStat, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := runGPUTool(ctx,
		"nvidia-smi",
		"--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
		"--format=csv,noheader,nounits",
	)
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
// order maps a card's PCI bus to the device index instances are placed by.
func readGPUsROCm(order func(out []byte) map[int]int) ([]GPUStat, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := runGPUTool(ctx,
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
	if err != nil {
		return nil, fmt.Sprintf("rocm-smi: %v", err)
	}
	// rocm-smi numbers cards by PCI bus, but on the ROCm backends instances
	// are placed by HIP device index (gpu_id -> HIP_VISIBLE_DEVICES), which
	// follows rocminfo's agent order. Remap so the reported Index matches the
	// gpu_id a user assigns; a nil map keeps rocm-smi's order.
	stats, err := parseROCmSMI(out, order(out))
	if err != nil {
		return nil, fmt.Sprintf("rocm-smi json: %v", err)
	}
	return stats, ""
}

// pciBusOrder numbers rocm-smi's cards by ascending PCI bus, the Vulkan
// device order. Nil when a card has no readable bus.
func pciBusOrder(out []byte) map[int]int {
	var raw map[string]map[string]string
	if json.Unmarshal(out, &raw) != nil {
		return nil
	}
	var buses []int
	for k, v := range raw {
		if !strings.HasPrefix(k, "card") {
			continue
		}
		bus := parsePCIBus(firstOf(v, "PCI Bus"))
		if bus < 0 {
			return nil
		}
		buses = append(buses, bus)
	}
	sort.Ints(buses)
	m := make(map[int]int, len(buses))
	for i, b := range buses {
		m[b] = i
	}
	return m
}

// readGPUsMetal reads Apple GPU load from the IOAccelerator registry, which
// needs no root (powermetrics does). macOS exposes no GPU temperature or
// power there, so those stay 0 and the UI leaves them out.
func readGPUsMetal() ([]GPUStat, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := runGPUTool(ctx, "ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator")
	if err != nil {
		return nil, fmt.Sprintf("ioreg: %v", err)
	}
	return parseIOReg(out, unifiedMemoryMB()), ""
}

var (
	ioregModel = regexp.MustCompile(`"model" = <?"([^"]+)"`)
	ioregUtil  = regexp.MustCompile(`"Device Utilization %"=(\d+)`)
	ioregMem   = regexp.MustCompile(`"In use system memory"=(\d+)`)
)

// parseIOReg turns `ioreg -c IOAccelerator` output into one stat per GPU.
// Apple silicon GPUs share system memory, so totalMB is the machine's RAM.
func parseIOReg(out []byte, totalMB int64) []GPUStat {
	var stats []GPUStat
	for _, block := range strings.Split(string(out), "+-o ")[1:] {
		m := ioregUtil.FindStringSubmatch(block)
		if m == nil {
			continue
		}
		name := "GPU"
		if n := ioregModel.FindStringSubmatch(block); n != nil {
			name = n[1]
		}
		s := GPUStat{Index: len(stats), Name: name, Backend: "metal", UtilPct: atof(m[1]), MemTotalMB: totalMB}
		if u := ioregMem.FindStringSubmatch(block); u != nil {
			s.MemUsedMB = int64(atof(u[1])) / (1024 * 1024)
		}
		stats = append(stats, s)
	}
	return stats
}

var unifiedMemoryMB = sync.OnceValue(func() int64 {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := runGPUTool(ctx, "sysctl", "-n", "hw.memsize")
	if err != nil {
		return 0
	}
	return int64(atof(string(out))) / (1024 * 1024)
})

// parseROCmSMI turns `rocm-smi --json` output into stats, renumbering cards by
// HIP index when busToHIP is known. Cards rocminfo doesn't list (an
// unsupported display GPU, or ones hidden from the manager) aren't
// addressable by gpu_id; they get indices after the HIP ones instead of
// keeping a rocm-smi number that could collide with a remapped card.
func parseROCmSMI(out []byte, busToHIP map[int]int) ([]GPUStat, error) {
	var raw map[string]map[string]string
	if err := json.Unmarshal(out, &raw); err != nil {
		return nil, err
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

	next := 0
	for _, hip := range busToHIP {
		if hip >= next {
			next = hip + 1
		}
	}
	stats := make([]GPUStat, 0, len(cards))
	for _, c := range cards {
		idx := c.idx
		if busToHIP != nil {
			if hip, ok := busToHIP[parsePCIBus(firstOf(c.m, "PCI Bus"))]; ok {
				idx = hip
			} else {
				idx = next
				next++
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
	return stats, nil
}

// hipBusIndex maps a PCI bus byte to the HIP/ROCr device index by parsing
// rocminfo's agent order (cached; rocminfo output is static for a boot). HIP
// numbers GPUs in rocminfo agent order, which differs from rocm-smi's PCI-bus
// card numbering — so an instance launched with HIP_VISIBLE_DEVICES=N can show
// up under a different "cardN" in rocm-smi. Returns nil on failure, in which
// case callers keep rocm-smi's native order. Only success is cached: rocminfo
// can fail transiently (HSA init timing out on a multi-GPU box, /dev/kfd not
// ready right after boot), so a failure is retried after hipRetryInterval
// rather than leaving the indices wrong until the next restart.
var (
	hipMu      sync.Mutex
	hipMap     map[int]int
	hipRetryAt time.Time

	hipRetryInterval = time.Minute
	runRocminfo      = func(ctx context.Context) ([]byte, error) {
		return exec.CommandContext(ctx, "rocminfo").Output()
	}
)

func hipBusIndex() map[int]int {
	hipMu.Lock()
	defer hipMu.Unlock()
	if hipMap != nil || time.Now().Before(hipRetryAt) {
		return hipMap
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := runRocminfo(ctx)
	if m := parseHIPBusIndex(bytes.NewReader(out)); err == nil && len(m) > 0 {
		hipMap = m
		return hipMap
	}
	hipRetryAt = time.Now().Add(hipRetryInterval)
	return nil
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
