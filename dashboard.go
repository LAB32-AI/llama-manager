package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"sync"
	"time"
)

// The dashboard page: GET /api/dashboard?range=1h|24h|7d|30d|all puts
// together the request statistics (stats.go), the activity feed (events.go)
// and what the sampler below reads every 15 s whether or not anyone is
// looking: GPU load, VRAM, temperature and power, each instance's live
// llama-server metrics, requests in flight and queued.

// sampleEvery is how often the sampler reads GPUs and instances; a var for
// tests.
var sampleEvery = 15 * time.Second

const sampleKeep = 24 * time.Hour

type gpuAgg struct {
	name               string
	totalMB            int64
	util, power        float64 // sums, for averages
	utilMax, mem, temp float64 // maxima
	powerMax           float64
	n                  int
}

type instAgg struct {
	tps, ptps          float64 // sums
	kv, proc, deferred float64 // maxima
	n                  int
}

type sampleMinute struct {
	gpus             map[int]*gpuAgg
	inst             map[string]*instAgg
	inFlight, queued float64 // maxima
}

type dashSampler struct {
	ws       *WebServer
	readGPUs func(backend string) ([]GPUStat, string)

	mu      sync.Mutex
	minutes map[int64]*sampleMinute
	gpus    []GPUStat
	gpuErr  string
	metrics map[string]*InstanceMetrics
	sampled bool

	lastFunnel string
	lastDL     string
}

func newDashSampler(ws *WebServer) *dashSampler {
	return &dashSampler{ws: ws, readGPUs: readGPUs, minutes: map[int64]*sampleMinute{}, metrics: map[string]*InstanceMetrics{}}
}

func (s *dashSampler) run(ctx context.Context) {
	t := time.NewTicker(sampleEvery)
	defer t.Stop()
	for {
		s.sample(time.Now())
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

func (s *dashSampler) sample(now time.Time) {
	ws := s.ws
	ws.cfg.mu.RLock()
	backend := ws.cfg.GPUBackend
	ws.cfg.mu.RUnlock()
	gpus, gpuErr := s.readGPUs(backend)

	metrics := map[string]*InstanceMetrics{}
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, inst := range ws.mgr.Instances() {
		if inst.State() != StateRunning || inst.conf.role == roleImageGen || inst.conf.role == roleSpeech {
			continue
		}
		wg.Add(1)
		go func(inst *Instance) {
			defer wg.Done()
			if m := inst.FetchMetrics(); m != nil {
				mu.Lock()
				metrics[inst.conf.Name] = m
				mu.Unlock()
			}
		}(inst)
	}
	wg.Wait()
	inFlight := float64(ws.mgr.stats.inFlight.Load())
	var queued float64
	for _, m := range metrics {
		queued += m.RequestsDeferred
	}

	minute := now.Unix() / 60 * 60
	s.mu.Lock()
	sm := s.minutes[minute]
	if sm == nil {
		sm = &sampleMinute{gpus: map[int]*gpuAgg{}, inst: map[string]*instAgg{}}
		s.minutes[minute] = sm
	}
	for _, g := range gpus {
		a := sm.gpus[g.Index]
		if a == nil {
			a = &gpuAgg{}
			sm.gpus[g.Index] = a
		}
		a.name, a.totalMB = g.Name, g.MemTotalMB
		a.util += g.UtilPct
		a.power += g.PowerW
		a.utilMax = math.Max(a.utilMax, g.UtilPct)
		a.mem = math.Max(a.mem, float64(g.MemUsedMB))
		a.temp = math.Max(a.temp, g.TempC)
		a.powerMax = math.Max(a.powerMax, g.PowerW)
		a.n++
	}
	for name, m := range metrics {
		a := sm.inst[name]
		if a == nil {
			a = &instAgg{}
			sm.inst[name] = a
		}
		a.tps += m.PredictedTokensSec
		a.ptps += m.PromptTokensSec
		a.kv = math.Max(a.kv, m.KVCacheUsage)
		a.proc = math.Max(a.proc, m.RequestsProcessing)
		a.deferred = math.Max(a.deferred, m.RequestsDeferred)
		a.n++
	}
	sm.inFlight = math.Max(sm.inFlight, inFlight)
	sm.queued = math.Max(sm.queued, queued)
	cut := now.Add(-sampleKeep).Unix()
	for t := range s.minutes {
		if t < cut {
			delete(s.minutes, t)
		}
	}
	s.gpus, s.gpuErr, s.metrics, s.sampled = gpus, gpuErr, metrics, true
	s.mu.Unlock()

	s.noteChanges()
}

// noteChanges puts Funnel and download changes in the activity feed.
func (s *dashSampler) noteChanges() {
	ev := s.ws.mgr.events
	if st := s.ws.funnel.Status(); st != nil && st.State != s.lastFunnel {
		s.lastFunnel = st.State
		switch st.State {
		case funnelOn:
			ev.add(evFunnel, "", "/v1 is on the internet at "+st.URL)
		case funnelApproval:
			ev.add(evFunnel, "", "Funnel waits for approval in the Tailscale admin console")
		case funnelNoKey:
			ev.add(evFunnel, "", "/v1 is off the internet until there is an API key")
		case funnelError:
			ev.add(evFunnel, "", "Funnel isn't running: "+st.Error)
		}
	}
	if s.ws.dlm != nil {
		d := s.ws.dlm.GetStatus()
		key := d.Repo + ":" + d.Quant + " " + d.Status
		if !d.Active && d.Status != "" && key != s.lastDL {
			if s.lastDL != "" || d.Status == "done" || d.Status == "failed" {
				msg := map[string]string{"done": "downloaded", "failed": "download failed", "stopped": "download stopped"}[d.Status]
				if msg == "" {
					msg = d.Status
				}
				ev.add(evDownload, "", fmt.Sprintf("%s:%s %s", d.Repo, d.Quant, msg))
			}
			s.lastDL = key
		}
	}
}

// series is a chart's values; NaN (no data) becomes null.
type series []float64

func (s series) MarshalJSON() ([]byte, error) {
	b := []byte{'['}
	for i, v := range s {
		if i > 0 {
			b = append(b, ',')
		}
		if math.IsNaN(v) || math.IsInf(v, 0) {
			b = append(b, "null"...)
		} else {
			b = strconv.AppendFloat(b, math.Round(v*100)/100, 'f', -1, 64)
		}
	}
	return append(b, ']'), nil
}

func nanSeries(n int) series {
	s := make(series, n)
	for i := range s {
		s[i] = math.NaN()
	}
	return s
}

type dashCards struct {
	Requests      int64   `json:"requests"`
	Errors        int64   `json:"errors"`
	Rejected      int64   `json:"rejected"`
	TokensIn      int64   `json:"tokens_in"`
	TokensOut     int64   `json:"tokens_out"`
	TokensCached  int64   `json:"tokens_cached"`
	TokS          float64 `json:"tok_s"`
	InFlight      int64   `json:"in_flight"`
	Queued        int64   `json:"queued"`
	UptimeSec     float64 `json:"uptime_sec"`
	HostUptimeSec float64 `json:"host_uptime_sec"`
}

type dashLatency struct {
	TTFTp50 int64 `json:"ttft_p50"`
	TTFTp95 int64 `json:"ttft_p95"`
	Durp50  int64 `json:"dur_p50"`
	Durp95  int64 `json:"dur_p95"`
	TTFTn   int64 `json:"ttft_n"`
	Durn    int64 `json:"dur_n"`
}

func latencyOf(a *statAgg) dashLatency {
	return dashLatency{TTFTp50: a.TTFT.quantile(.5), TTFTp95: a.TTFT.quantile(.95), Durp50: a.Dur.quantile(.5), Durp95: a.Dur.quantile(.95),
		TTFTn: a.TTFT.count(), Durn: a.Dur.count()}
}

type dashSeries struct {
	Requests  map[string]series `json:"requests"` // by source
	Errors    series            `json:"errors"`
	Rejected  series            `json:"rejected"`
	TokensIn  series            `json:"tokens_in"`
	TokensOut series            `json:"tokens_out"`
	TTFTp50   series            `json:"ttft_p50"`
	TTFTp95   series            `json:"ttft_p95"`
	Durp50    series            `json:"dur_p50"`
	Durp95    series            `json:"dur_p95"`
}

type dashLive struct {
	TokS       float64 `json:"tok_s"`
	PromptTokS float64 `json:"prompt_tok_s"`
	KVCache    float64 `json:"kv_cache"`
	Processing float64 `json:"processing"`
	Deferred   float64 `json:"deferred"`
}

type dashModel struct {
	Name      string        `json:"name"`
	Role      string        `json:"role,omitempty"`
	State     InstanceState `json:"state"` // "" when the instance is gone
	GPUIDs    []int         `json:"gpu_ids"`
	UptimeSec float64       `json:"uptime_sec"`
	Restarts  int           `json:"restarts"`
	LastError string        `json:"last_error,omitempty"`
	Requests  int64         `json:"requests"`
	Errors    int64         `json:"errors"`
	Rejected  int64         `json:"rejected"`
	TokensIn  int64         `json:"tokens_in"`
	TokensOut int64         `json:"tokens_out"`
	TokS      float64       `json:"tok_s"`
	Latency   dashLatency   `json:"latency"`
	Live      *dashLive     `json:"live,omitempty"`
}

type dashGPU struct {
	Index     int     `json:"index"`
	Name      string  `json:"name"`
	Now       GPUStat `json:"now"`
	Util      series  `json:"util"`
	MemMB     series  `json:"mem_mb"`
	TempC     series  `json:"temp_c"`
	PowerW    series  `json:"power_w"`
	PeakTemp  float64 `json:"peak_temp_c"`
	PeakPower float64 `json:"peak_power_w"`
	PeakUtil  float64 `json:"peak_util"`
}

// dashGauges are the sampled values, over the last hour or day.
type dashGauges struct {
	Step     int64     `json:"step"`
	T        []int64   `json:"t"`
	InFlight series    `json:"in_flight"`
	Queued   series    `json:"queued"`
	GPUs     []dashGPU `json:"gpus"`
	GPUError string    `json:"gpu_error,omitempty"`
}

type dashKey struct {
	Label     string `json:"label"`
	Requests  int64  `json:"requests"`
	TokensIn  int64  `json:"tokens_in"`
	TokensOut int64  `json:"tokens_out"`
	LastUsed  int64  `json:"last_used,omitempty"`
}

type dashAPI struct {
	Funnel     *funnelStatus `json:"funnel"`
	Keys       int           `json:"keys"`
	Internet   dashCount     `json:"internet"`
	LAN        dashCount     `json:"lan"`
	Blocked    int           `json:"blocked"`
	KeysPeriod string        `json:"keys_period"`
	TopKeys    []dashKey     `json:"top_keys"`
}

type dashCount struct {
	Requests int64 `json:"requests"`
	Errors   int64 `json:"errors"`
	Rejected int64 `json:"rejected"`
}

type dashEndpoint struct {
	Route     string      `json:"route"`
	Requests  int64       `json:"requests"`
	Errors    int64       `json:"errors"`
	Rejected  int64       `json:"rejected"`
	TokensIn  int64       `json:"tokens_in"`
	TokensOut int64       `json:"tokens_out"`
	Latency   dashLatency `json:"latency"`
}

type dashStorage struct {
	Disks       []DiskUsage `json:"disks"`
	ModelFiles  int         `json:"model_files"`
	ModelsMB    int64       `json:"models_mb"`
	ChatDBBytes int64       `json:"chat_db_bytes"`
	Chats       int64       `json:"chats"`
	Messages    int64       `json:"messages"`
	Responses   int64       `json:"responses"`
	Convs       int64       `json:"conversations"`
	Completions int64       `json:"stored_completions"`
	Pictures    int64       `json:"pictures"`
	SpeechParts int64       `json:"speech"`
	Transcribed int64       `json:"transcriptions"`
	History     bool        `json:"history"`
}

type dashResp struct {
	Range     string         `json:"range"`
	Step      int64          `json:"step"`
	T         []int64        `json:"t"`
	Now       int64          `json:"now"`
	Since     int64          `json:"since"`
	Cards     dashCards      `json:"cards"`
	Series    dashSeries     `json:"series"`
	Latency   dashLatency    `json:"latency"`
	Models    []dashModel    `json:"models"`
	Gauges    dashGauges     `json:"gauges"`
	API       dashAPI        `json:"api"`
	Endpoints []dashEndpoint `json:"endpoints"`
	Events    []dashEvent    `json:"events"`
	Storage   *dashStorage   `json:"storage"`
}

func (ws *WebServer) handleDashboard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	rng := r.URL.Query().Get("range")
	if _, ok := dashRanges[rng]; !ok {
		rng = "24h"
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(ws.dashboard(rng, time.Now()))
}

func (ws *WebServer) dashboard(rng string, now time.Time) dashResp {
	sp := spanFor(rng, now)
	st := ws.mgr.stats
	out := dashResp{Range: rng, Step: sp.step, T: sp.times(), Now: now.Unix()}
	ser := dashSeries{Requests: map[string]series{}, Errors: make(series, sp.n), Rejected: make(series, sp.n),
		TokensIn: make(series, sp.n), TokensOut: make(series, sp.n),
		TTFTp50: nanSeries(sp.n), TTFTp95: nanSeries(sp.n), Durp50: nanSeries(sp.n), Durp95: nanSeries(sp.n)}
	for _, src := range statSources {
		ser.Requests[src] = make(series, sp.n)
	}
	var total statAgg
	bySource := map[string]*statAgg{}
	byModel := map[string]*statAgg{}
	byRoute := map[string]*statAgg{}
	ttft := make([]latHist, sp.n)
	dur := make([]latHist, sp.n)
	add := func(m map[string]*statAgg, k string, a *statAgg) {
		x := m[k]
		if x == nil {
			x = &statAgg{}
			m[k] = x
		}
		x.merge(a)
	}
	sum := func(k statKey, a *statAgg) {
		total.merge(a)
		add(bySource, k.Source, a)
		add(byRoute, k.Route, a)
		if k.Model != "" {
			add(byModel, k.Model, a)
		}
	}
	st.each(sp, func(b int, k statKey, a *statAgg) {
		if rng != "all" {
			sum(k, a)
		}
		if s, ok := ser.Requests[k.Source]; ok {
			s[b] += float64(a.Requests)
		}
		ser.Errors[b] += float64(a.Errors)
		ser.Rejected[b] += float64(a.Rejected)
		ser.TokensIn[b] += float64(a.In)
		ser.TokensOut[b] += float64(a.Out)
		ttft[b].merge(&a.TTFT)
		dur[b].merge(&a.Dur)
	})
	if rng == "all" {
		out.Since = st.allTotals(sum)
	} else {
		out.Since = st.allTotals(func(statKey, *statAgg) {})
	}
	for b := 0; b < sp.n; b++ {
		if v := ttft[b].quantile(.5); v >= 0 {
			ser.TTFTp50[b], ser.TTFTp95[b] = float64(v), float64(ttft[b].quantile(.95))
		}
		if v := dur[b].quantile(.5); v >= 0 {
			ser.Durp50[b], ser.Durp95[b] = float64(v), float64(dur[b].quantile(.95))
		}
	}
	out.Series = ser
	out.Latency = latencyOf(&total)

	s := ws.sampler
	s.mu.Lock()
	metrics := s.metrics
	s.mu.Unlock()
	var queued float64
	for _, m := range metrics {
		queued += m.RequestsDeferred
	}
	out.Cards = dashCards{Requests: total.Requests, Errors: total.Errors, Rejected: total.Rejected, TokensIn: total.In, TokensOut: total.Out,
		TokensCached: total.Cached, TokS: total.tokS(), InFlight: st.inFlight.Load(), Queued: int64(queued),
		UptimeSec: now.Sub(ws.started).Seconds(), HostUptimeSec: getSystemUptime().Seconds()}

	seen := map[string]bool{}
	for _, inst := range ws.mgr.Instances() {
		is := inst.Status()
		seen[is.Name] = true
		m := dashModel{Name: is.Name, Role: is.Role, State: is.State, GPUIDs: is.GPUIDs, UptimeSec: is.UptimeSec, Restarts: is.RestartCount, LastError: is.LastError}
		fillModel(&m, byModel[is.Name])
		if lm := metrics[is.Name]; lm != nil && is.State == StateRunning {
			m.Live = &dashLive{TokS: lm.PredictedTokensSec, PromptTokS: lm.PromptTokensSec, KVCache: lm.KVCacheUsage, Processing: lm.RequestsProcessing, Deferred: lm.RequestsDeferred}
		}
		out.Models = append(out.Models, m)
	}
	var gone []string
	for name := range byModel {
		if !seen[name] {
			gone = append(gone, name)
		}
	}
	sort.Strings(gone)
	for _, name := range gone {
		m := dashModel{Name: name, GPUIDs: []int{}}
		fillModel(&m, byModel[name])
		out.Models = append(out.Models, m)
	}
	if out.Models == nil {
		out.Models = []dashModel{}
	}

	out.Gauges = s.gauges(rng, now)
	out.API = ws.dashAPI(rng, bySource, now)
	out.Endpoints = []dashEndpoint{}
	for route, a := range byRoute {
		out.Endpoints = append(out.Endpoints, dashEndpoint{Route: route, Requests: a.Requests, Errors: a.Errors, Rejected: a.Rejected,
			TokensIn: a.In, TokensOut: a.Out, Latency: latencyOf(a)})
	}
	sort.Slice(out.Endpoints, func(i, j int) bool {
		if out.Endpoints[i].Requests != out.Endpoints[j].Requests {
			return out.Endpoints[i].Requests > out.Endpoints[j].Requests
		}
		return out.Endpoints[i].Route < out.Endpoints[j].Route
	})
	out.Events = ws.mgr.events.recent(50)
	if out.Events == nil {
		out.Events = []dashEvent{}
	}
	out.Storage = ws.storageSummary(now)
	return out
}

func fillModel(m *dashModel, a *statAgg) {
	if a == nil {
		m.Latency = latencyOf(&statAgg{})
		return
	}
	m.Requests, m.Errors, m.Rejected, m.TokensIn, m.TokensOut, m.TokS = a.Requests, a.Errors, a.Rejected, a.In, a.Out, a.tokS()
	m.Latency = latencyOf(a)
}

// gauges cuts the sampled minutes into the last hour (1-minute steps) or,
// for every longer range, the last day (15-minute steps).
func (s *dashSampler) gauges(rng string, now time.Time) dashGauges {
	step, n := int64(900), 96
	if rng == "1h" {
		step, n = 60, 60
	}
	end := now.Unix() / step * step
	from := end - int64(n-1)*step
	g := dashGauges{Step: step, T: make([]int64, n), InFlight: nanSeries(n), Queued: nanSeries(n), GPUs: []dashGPU{}}
	for i := range g.T {
		g.T[i] = from + int64(i)*step
	}
	type acc struct {
		util, power, n           float64
		utilMax, mem, temp, powM float64
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	g.GPUError = s.gpuErr
	per := map[int][]acc{}
	names := map[int]string{}
	for t, sm := range s.minutes {
		if t < from || t >= end+step {
			continue
		}
		b := int((t - from) / step)
		if math.IsNaN(g.InFlight[b]) {
			g.InFlight[b], g.Queued[b] = 0, 0
		}
		g.InFlight[b] = math.Max(g.InFlight[b], sm.inFlight)
		g.Queued[b] = math.Max(g.Queued[b], sm.queued)
		for idx, a := range sm.gpus {
			if per[idx] == nil {
				per[idx] = make([]acc, n)
			}
			names[idx] = a.name
			x := &per[idx][b]
			x.util += a.util
			x.power += a.power
			x.n += float64(a.n)
			x.utilMax = math.Max(x.utilMax, a.utilMax)
			x.mem = math.Max(x.mem, a.mem)
			x.temp = math.Max(x.temp, a.temp)
			x.powM = math.Max(x.powM, a.powerMax)
		}
	}
	for _, cur := range s.gpus {
		if _, ok := per[cur.Index]; !ok {
			per[cur.Index] = make([]acc, n)
		}
	}
	idxs := make([]int, 0, len(per))
	for idx := range per {
		idxs = append(idxs, idx)
	}
	sort.Ints(idxs)
	for _, idx := range idxs {
		d := dashGPU{Index: idx, Name: names[idx], Util: nanSeries(n), MemMB: nanSeries(n), TempC: nanSeries(n), PowerW: nanSeries(n)}
		for _, cur := range s.gpus {
			if cur.Index == idx {
				d.Now = cur
				if d.Name == "" {
					d.Name = cur.Name
				}
			}
		}
		for b, x := range per[idx] {
			if x.n == 0 {
				continue
			}
			d.Util[b], d.PowerW[b] = x.util/x.n, x.power/x.n
			d.MemMB[b], d.TempC[b] = x.mem, x.temp
			d.PeakTemp = math.Max(d.PeakTemp, x.temp)
			d.PeakPower = math.Max(d.PeakPower, x.powM)
			d.PeakUtil = math.Max(d.PeakUtil, x.utilMax)
		}
		g.GPUs = append(g.GPUs, d)
	}
	return g
}

func (ws *WebServer) dashAPI(rng string, bySource map[string]*statAgg, now time.Time) dashAPI {
	a := dashAPI{Funnel: ws.funnel.Status(), Blocked: ws.limits.blockedCount(now), TopKeys: []dashKey{}}
	count := func(src string) dashCount {
		if x := bySource[src]; x != nil {
			return dashCount{x.Requests, x.Errors, x.Rejected}
		}
		return dashCount{}
	}
	a.Internet, a.LAN = count(srcInternet), count(srcLAN)
	keys := ws.apiKeyList()
	a.Keys = len(keys)
	a.KeysPeriod = "today"
	month := rng == "7d" || rng == "30d" || rng == "all"
	if month {
		a.KeysPeriod = "30 days"
	}
	for i, u := range ws.usage.forKeys(keys, now) {
		c := u.Today
		if month {
			c = u.Month
		}
		if c.Requests == 0 {
			continue
		}
		a.TopKeys = append(a.TopKeys, dashKey{Label: keys[i].label(), Requests: c.Requests, TokensIn: c.InputTokens, TokensOut: c.OutputTokens, LastUsed: u.LastUsed})
	}
	sort.Slice(a.TopKeys, func(i, j int) bool {
		ti, tj := a.TopKeys[i].TokensIn+a.TopKeys[i].TokensOut, a.TopKeys[j].TokensIn+a.TopKeys[j].TokensOut
		if ti != tj {
			return ti > tj
		}
		return a.TopKeys[i].Requests > a.TopKeys[j].Requests
	})
	if len(a.TopKeys) > 5 {
		a.TopKeys = a.TopKeys[:5]
	}
	return a
}

// storageTTL is how long the storage summary (a walk of the model folders and
// a few counts) is reused.
const storageTTL = time.Minute

func (ws *WebServer) storageSummary(now time.Time) *dashStorage {
	ws.storageMu.Lock()
	defer ws.storageMu.Unlock()
	if ws.storage != nil && now.Sub(ws.storageAt) < storageTTL {
		cp := *ws.storage
		ws.fillMade(&cp)
		return &cp
	}
	s := &dashStorage{Disks: []DiskUsage{}}
	ws.cfg.mu.RLock()
	extra := append([]string(nil), ws.cfg.ModelDirs...)
	ws.cfg.mu.RUnlock()
	if models, err := scanCachedModels(extra); err == nil {
		s.Disks, _ = diskUsage(models, getCacheDir())
		s.ModelFiles = len(models)
		for _, m := range models {
			s.ModelsMB += m.SizeMB
		}
	}
	if ws.chats != nil && ws.chats.store != nil {
		cs := ws.chats.store
		s.History = true
		for _, suffix := range []string{"", "-wal"} {
			if fi, err := os.Stat(cs.path + suffix); err == nil {
				s.ChatDBBytes += fi.Size()
			}
		}
		_ = cs.db.QueryRow(`SELECT (SELECT COUNT(*) FROM chats), (SELECT COUNT(*) FROM messages), (SELECT COUNT(*) FROM api_responses),
			(SELECT COUNT(*) FROM api_conversations), (SELECT COUNT(*) FROM api_chat_completions)`).Scan(&s.Chats, &s.Messages, &s.Responses, &s.Convs, &s.Completions)
	}
	ws.storage, ws.storageAt = s, now
	cp := *s
	ws.fillMade(&cp)
	return &cp
}

// fillMade adds what the models made, from the all-time counts.
func (ws *WebServer) fillMade(s *dashStorage) {
	s.Pictures, s.SpeechParts, s.Transcribed = 0, 0, 0
	ws.mgr.stats.allTotals(func(k statKey, a *statAgg) {
		ok := a.Requests - a.Errors - a.Rejected
		switch k.Route {
		case "images":
			s.Pictures += ok
		case "speech":
			s.SpeechParts += ok
		case "transcriptions":
			s.Transcribed += ok
		}
	})
}

// startDashboard runs the sampler and writes statistics and events to
// chats.db every statsFlushEvery, pruning them hourly.
func (ws *WebServer) startDashboard(ctx context.Context) {
	go ws.sampler.run(ctx)
	go func() {
		t := time.NewTicker(statsFlushEvery)
		defer t.Stop()
		lastPrune := time.Time{}
		for {
			select {
			case <-ctx.Done():
				return
			case now := <-t.C:
				ws.mgr.stats.flush()
				ws.mgr.events.flush()
				if now.Sub(lastPrune) >= time.Hour {
					ws.mgr.stats.prune(now)
					ws.mgr.events.prune(now)
					lastPrune = now
				}
			}
		}
	}()
}
