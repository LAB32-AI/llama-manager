package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLatHistQuantile(t *testing.T) {
	var h latHist
	if h.quantile(.5) != -1 {
		t.Error("an empty histogram has no quantile")
	}
	for i := 0; i < 90; i++ {
		h.add(150) // 100-200 ms bucket
	}
	for i := 0; i < 10; i++ {
		h.add(4000) // 3-5 s bucket
	}
	if p := h.quantile(.5); p < 100 || p > 200 {
		t.Errorf("p50 = %d", p)
	}
	if p := h.quantile(.95); p < 3000 || p > 5000 {
		t.Errorf("p95 = %d", p)
	}
	h.add(10 * 60 * 1000)
	if p := h.quantile(1); p != histBounds[len(histBounds)-1] {
		t.Errorf("slower than the last bound reads as the last bound: %d", p)
	}
}

func TestStatsPersistAcrossRestarts(t *testing.T) {
	store, err := OpenChatStore(filepath.Join(t.TempDir(), "chats.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	now := time.Now()
	ev := &eventLog{}
	s := newStatsStore(ev)
	s.attach(store.db)
	s.record(statRec{At: now, Source: srcLAN, Route: "chat", Model: "m", DurMs: 900, TTFTMs: 120, In: 10, Out: 20, Cached: 4, GenMs: 1000})
	s.record(statRec{At: now, Source: srcLAN, Route: "chat", Model: "m", Outcome: outcomeError, ErrMsg: "502 upstream down"})
	s.record(statRec{At: now.Add(-3 * time.Hour), Source: srcChat, Route: "chat", Model: "m", DurMs: 100, TTFTMs: -1, Out: 5})
	s.record(statRec{At: now.Add(-100 * 24 * time.Hour), Source: srcChat, Route: "chat", Model: "old"})
	s.flush()
	ev.attach(store.db)
	ev.flush()

	s2 := newStatsStore(nil)
	s2.attach(store.db)
	var tot statAgg
	since := s2.allTotals(func(_ statKey, a *statAgg) { tot.merge(a) })
	if tot.Requests != 4 || tot.Errors != 1 || tot.In != 10 || tot.Out != 25 || tot.Cached != 4 || since != now.Add(-100*24*time.Hour).Unix() {
		t.Errorf("all-time totals after a restart: %+v since %d", tot, since)
	}
	var day statAgg
	s2.each(spanFor("24h", now), func(_ int, _ statKey, a *statAgg) { day.merge(a) })
	if day.Requests != 3 || day.tokS() != 20 || day.TTFT.count() != 1 {
		t.Errorf("last day after a restart: %+v", day)
	}
	var hour statAgg
	s2.each(spanFor("1h", now), func(_ int, _ statKey, a *statAgg) { hour.merge(a) })
	if hour.Requests != 2 {
		t.Errorf("last hour: %+v", hour)
	}
	var month statAgg
	s2.each(spanFor("30d", now), func(_ int, _ statKey, a *statAgg) { month.merge(a) })
	if month.Requests != 3 {
		t.Errorf("past 90 days isn't kept per hour, but stays in the totals: %+v", month)
	}

	ev2 := &eventLog{}
	ev2.attach(store.db)
	if es := ev2.recent(10); len(es) != 1 || es[0].Kind != evError || !strings.Contains(es[0].Msg, "502 upstream down") || es[0].Model != "m" {
		t.Errorf("events after a restart: %+v", es)
	}

	s2.prune(now.Add(72 * time.Hour))
	var after statAgg
	s2.each(spanFor("24h", now), func(_ int, _ statKey, a *statAgg) { after.merge(a) })
	var rows int
	_ = store.db.QueryRow(`SELECT COUNT(*) FROM stats_minute`).Scan(&rows)
	if after.Requests != 0 || rows != 0 {
		t.Errorf("minutes past two days are dropped: %+v, %d rows", after, rows)
	}
}

// dashRig is a v1Rig whose fake llama-server answers like the real one: usage
// and timings, streamed or not, or a 500 for "model":"fail" requests.
func newDashRig(t *testing.T) *v1Rig {
	t.Helper()
	return newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if msgs, _ := body["messages"].([]any); len(msgs) > 0 {
			if m, _ := msgs[len(msgs)-1].(map[string]any); m["content"] == "fail please" {
				http.Error(w, "the model fell over", http.StatusInternalServerError)
				return
			}
		}
		if body["slow"] == true {
			time.Sleep(400 * time.Millisecond)
		}
		if body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n")
			w.(http.Flusher).Flush()
			time.Sleep(20 * time.Millisecond)
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":6,\"prompt_tokens_details\":{\"cached_tokens\":8}},"+
				"\"timings\":{\"prompt_n\":12,\"prompt_per_second\":500,\"predicted_n\":6,\"predicted_ms\":200,\"predicted_per_second\":30}}\n\ndata: [DONE]\n\n")
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"model": "c", "choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "hi"}, "finish_reason": "stop"}},
			"usage": map[string]any{"prompt_tokens": 7, "completion_tokens": 3}, "timings": map[string]any{"predicted_ms": 100, "predicted_n": 3}})
	})
}

func getDash(t *testing.T, vr *v1Rig, rng string) dashResp {
	t.Helper()
	_, b := vr.do(t, "GET", "/api/dashboard?range="+rng, nil)
	var d dashResp
	if err := json.Unmarshal(b, &d); err != nil {
		t.Fatalf("dashboard: %v: %s", err, b)
	}
	return d
}

func TestDashboardCountsEverySource(t *testing.T) {
	vr := newDashRig(t)
	setKeys(vr.ws, "sk-a")
	key := map[string]string{"Authorization": "Bearer sk-a"}
	chat := `{"model":"c","messages":[{"role":"user","content":"hi"}]}`
	vr.req(t, "POST", "/v1/chat/completions", chat, key)
	vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","stream":true,"messages":[{"role":"user","content":"hi"}]}`, key)
	vr.req(t, "POST", "/v1/chat/completions", chat, fromInternet("203.0.113.1", key))
	vr.req(t, "POST", "/v1/chat/completions", chat, nil)                                                                       // rejected: no key
	vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[{"role":"user","content":"fail please"}]}`, key)       // error
	vr.req(t, "GET", "/v1/models", "", key)                                                                                    // not model work
	vr.req(t, "POST", "/v1/embeddings", `{"model":"c","input":"x"}`, fromInternet("203.0.113.1", map[string]string{"X": "y"})) // rejected, internet

	c := vr.newChat(t, `{}`)
	vr.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "hello"})
	vr.waitReply(t, c.ID)

	var d dashResp
	waitFor(t, 3*time.Second, "the chat reply counted", func() bool { d = getDash(t, vr, "1h"); return d.Cards.Requests == 7 })
	cd := d.Cards
	if cd.Errors != 1 || cd.Rejected != 2 || cd.TokensIn != 7+12+7+12 || cd.TokensOut != 3+6+3+6 || cd.TokensCached != 8 {
		t.Errorf("cards: %+v", cd)
	}
	if cd.TokS <= 0 {
		t.Errorf("generation speed from the timings: %v", cd.TokS)
	}
	last := len(d.T) - 1
	if len(d.T) != 60 || d.Series.Requests[srcLAN][last] != 4 || d.Series.Requests[srcInternet][last] != 2 || d.Series.Requests[srcChat][last] != 1 ||
		d.Series.Errors[last] != 1 || d.Series.Rejected[last] != 2 {
		t.Errorf("series: %d points, %+v", len(d.T), d.Series)
	}
	if d.Latency.TTFTn != 2 || d.Latency.Durn != 4 {
		t.Errorf("latency counts streamed replies for TTFT and every finished one for duration: %+v", d.Latency)
	}
	var m *dashModel
	for i := range d.Models {
		if d.Models[i].Name == "c" {
			m = &d.Models[i]
		}
	}
	if m == nil || m.Requests != 5 || m.Errors != 1 || m.State != StateRunning {
		t.Errorf("model row: %+v", m)
	}
	routes := map[string]dashEndpoint{}
	for _, e := range d.Endpoints {
		routes[e.Route] = e
	}
	if routes["chat"].Requests != 6 || routes["embeddings"].Rejected != 1 || len(routes) != 2 {
		t.Errorf("endpoints: %+v", d.Endpoints)
	}
	if d.API.Internet.Requests != 2 || d.API.Internet.Rejected != 1 || d.API.LAN.Requests != 4 || d.API.Keys != 1 {
		t.Errorf("api: %+v", d.API)
	}
	found := false
	for _, e := range d.Events {
		if e.Kind == evError && e.Model == "c" && strings.Contains(e.Msg, "500") && strings.Contains(e.Msg, "the model fell over") {
			found = true
		}
	}
	if !found {
		t.Errorf("the failed request is in the activity feed: %+v", d.Events)
	}
	if d.Storage == nil || !d.Storage.History || d.Storage.Chats != 1 || d.Storage.Messages < 2 || d.Storage.ChatDBBytes == 0 {
		t.Errorf("storage: %+v", d.Storage)
	}
}

func TestDashboardRanges(t *testing.T) {
	vr := newDashRig(t)
	vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[]}`, nil)
	for rng, n := range map[string]int{"1h": 60, "24h": 96, "7d": 168, "30d": 120, "all": 90, "bogus": 96} {
		d := getDash(t, vr, rng)
		if len(d.T) != n || d.Cards.Requests != 1 || d.Since == 0 {
			t.Errorf("%s: %d points (want %d), %d requests, since %d", rng, len(d.T), n, d.Cards.Requests, d.Since)
		}
		if d.T[len(d.T)-1] > d.Now || d.Now-d.T[len(d.T)-1] >= d.Step {
			t.Errorf("%s: the last bucket holds now", rng)
		}
	}
}

func TestDashboardInFlight(t *testing.T) {
	vr := newDashRig(t)
	done := make(chan struct{})
	go func() {
		vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","slow":true,"messages":[]}`, nil)
		close(done)
	}()
	waitFor(t, 2*time.Second, "one request in flight", func() bool { return getDash(t, vr, "1h").Cards.InFlight == 1 })
	<-done
	if d := getDash(t, vr, "1h"); d.Cards.InFlight != 0 {
		t.Errorf("in flight after it finished: %d", d.Cards.InFlight)
	}
}

func TestDashboardSamplerKeepsGPUHistory(t *testing.T) {
	ws, _ := newTestServer(t)
	calls := 0
	ws.sampler.readGPUs = func(string) ([]GPUStat, string) {
		calls++
		return []GPUStat{
			{Index: 0, Name: "MI50", UtilPct: float64(10 * calls), MemUsedMB: int64(1000 * calls), MemTotalMB: 32768, TempC: float64(40 + calls), PowerW: 100},
			{Index: 1, Name: "MI50", UtilPct: 0, MemUsedMB: 10, MemTotalMB: 32768, TempC: 30, PowerW: 20},
		}, ""
	}
	now := time.Now()
	calls = -1
	ws.sampler.sample(now.Add(-25 * time.Hour)) // dropped once a day has passed
	calls = 0
	ws.sampler.sample(now.Add(-30 * time.Minute))
	ws.sampler.sample(now.Add(-2 * time.Minute))
	ws.sampler.sample(now)
	g := ws.sampler.gauges("1h", now)
	if len(g.T) != 60 || len(g.GPUs) != 2 {
		t.Fatalf("gauges: %d points, %d GPUs", len(g.T), len(g.GPUs))
	}
	g0 := g.GPUs[0]
	last := len(g.T) - 1
	if g0.Now.UtilPct != 30 || g0.Util[last] != 30 || g0.MemMB[last] != 3000 || g0.PeakTemp != 43 || !math.IsNaN(g0.Util[last-1]) || g0.Util[last-30] != 10 {
		t.Errorf("GPU 0: now %+v, last %v, peak %v, at -30m %v", g0.Now, g0.Util[last], g0.PeakTemp, g0.Util[last-30])
	}
	if g.InFlight[last] != 0 {
		t.Errorf("in flight sampled: %v", g.InFlight[last])
	}
	day := ws.sampler.gauges("7d", now)
	if len(day.T) != 96 || day.Step != 900 {
		t.Errorf("longer ranges show the sampled day: %d points of %ds", len(day.T), day.Step)
	}
	ws.sampler.mu.Lock()
	n := len(ws.sampler.minutes)
	ws.sampler.mu.Unlock()
	if n != 3 {
		t.Errorf("minutes kept: %d (a sample from 25 h ago is dropped)", n)
	}
	b, _ := json.Marshal(g0.Util)
	if !strings.Contains(string(b), "null") {
		t.Errorf("gaps are null in JSON: %s", b)
	}

	ws.sampler.readGPUs = func(string) ([]GPUStat, string) { return nil, "rocm-smi not found" }
	ws.sampler.sample(now)
	if g := ws.sampler.gauges("1h", now); g.GPUError != "rocm-smi not found" {
		t.Errorf("GPU error: %q", g.GPUError)
	}
}

func TestDashboardEventsFromInstances(t *testing.T) {
	has := func(mgr *Manager, kind, name string) bool {
		for _, e := range mgr.events.recent(maxEvents) {
			if e.Kind == kind && e.Model == name {
				return true
			}
		}
		return false
	}
	ok := NewManager(newFakeRig(t, "healthy", "steady").cfg)
	t.Cleanup(ok.Shutdown)
	if err := ok.StartInstance("steady"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 5*time.Second, "a ready event", func() bool { return has(ok, evReady, "steady") })
	if err := ok.StopInstance("steady"); err != nil {
		t.Fatal(err)
	}
	if !has(ok, evStopped, "steady") {
		t.Error("no stopped event")
	}

	bad := NewManager(newFakeRig(t, "crash-after", "crashy").cfg)
	t.Cleanup(bad.Shutdown)
	if err := bad.StartInstance("crashy"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 5*time.Second, "crash and restart events", func() bool { return has(bad, evCrash, "crashy") && has(bad, evRestart, "crashy") })
}

func TestDashboardNotesFunnelChanges(t *testing.T) {
	ws, _ := newTestServer(t)
	ws.funnel = newFunnel(PublicAPIConf{TailscaleFunnel: true}, 8080, func() bool { return true })
	ws.sampler.readGPUs = func(string) ([]GPUStat, string) { return nil, "" }
	ws.funnel.set(func(s *funnelStatus) { s.State, s.URL = funnelOn, "https://x.ts.net/v1" })
	ws.sampler.sample(time.Now())
	ws.sampler.sample(time.Now())
	ws.funnel.set(func(s *funnelStatus) { s.State, s.Error = funnelError, "Access denied" })
	ws.sampler.sample(time.Now())
	var got []string
	for _, e := range ws.mgr.events.recent(10) {
		if e.Kind == evFunnel {
			got = append(got, e.Msg)
		}
	}
	if len(got) != 2 || !strings.Contains(got[0], "Access denied") || !strings.Contains(got[1], "https://x.ts.net/v1") {
		t.Errorf("funnel events (newest first): %q", got)
	}
}

func TestDashboardNamesTheModelOfARefusedRequest(t *testing.T) {
	vr := newDashRig(t)
	inst := vr.ws.mgr.Get("c")
	inst.mu.Lock()
	inst.state = StateStarting
	inst.mu.Unlock()
	resp, b := vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[]}`, nil)
	wantAPIError(t, "while loading", resp, b, http.StatusServiceUnavailable, "model_not_ready")
	d := getDash(t, vr, "1h")
	if len(d.Events) == 0 || d.Events[0].Kind != evError || d.Events[0].Model != "c" {
		t.Errorf("the error names its model: %+v", d.Events)
	}
	for _, m := range d.Models {
		if m.Name == "c" && m.Errors != 1 {
			t.Errorf("model row: %+v", m)
		}
	}
}
