package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// The dashboard's request statistics: one record per request that reached,
// or was meant for, a model (the chat page, /v1 from this network and from
// the internet). They are added up per minute for two days, per hour for 90
// days and for all time, in memory and in chats.db (stats_*), so the
// dashboard's totals survive restarts.

const (
	statsMinuteKeep = 48 * time.Hour
	statsHourKeep   = 90 * 24 * time.Hour
)

// statsFlushEvery is how often changed counts are written to chats.db.
var statsFlushEvery = 30 * time.Second

// Where a request came from.
const (
	srcChat     = "chat"
	srcLAN      = "lan"
	srcInternet = "internet"
)

var statSources = []string{srcChat, srcLAN, srcInternet}

// How a request ended: a 5xx or a failed generation is an error; a 4xx (a
// wrong key, a rate limit, a bad request) is rejected.
const (
	outcomeOK = iota
	outcomeRejected
	outcomeError
)

type statRec struct {
	At      time.Time
	Source  string
	Route   string // chat, completions, responses, embeddings, rerank, messages, images, speech, transcriptions, realtime, describe
	Model   string // instance name; "" when the request never got that far
	Outcome int
	DurMs   int64
	TTFTMs  int64 // time to the first streamed byte; -1 when not streamed
	In, Out int64 // tokens
	Cached  int64 // prompt tokens served from the cache
	GenMs   int64 // time spent generating Out, for tok/s; 0 when unknown
	ErrMsg  string
}

func outcomeOf(status int) int {
	switch {
	case status >= 500:
		return outcomeError
	case status >= 400:
		return outcomeRejected
	}
	return outcomeOK
}

// histBounds are the upper bounds (ms) of the latency histogram's buckets;
// one more bucket holds everything slower.
var histBounds = [...]int64{100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 60000, 120000, 300000}

const histN = len(histBounds) + 1

type latHist [histN]int64

func (h *latHist) add(ms int64) {
	i := 0
	for i < len(histBounds) && ms > histBounds[i] {
		i++
	}
	h[i]++
}

func (h *latHist) merge(o *latHist) {
	for i := range h {
		h[i] += o[i]
	}
}

func (h *latHist) count() int64 {
	var n int64
	for _, c := range h {
		n += c
	}
	return n
}

// quantile estimates the q-th quantile (0-1) in ms, interpolating inside the
// bucket; -1 when empty.
func (h *latHist) quantile(q float64) int64 {
	n := h.count()
	if n == 0 {
		return -1
	}
	rank := q * float64(n)
	var seen float64
	for i, c := range h {
		if c == 0 {
			continue
		}
		if seen+float64(c) >= rank {
			if i == len(histBounds) {
				return histBounds[len(histBounds)-1]
			}
			lo := int64(0)
			if i > 0 {
				lo = histBounds[i-1]
			}
			frac := (rank - seen) / float64(c)
			return lo + int64(frac*float64(histBounds[i]-lo))
		}
		seen += float64(c)
	}
	return histBounds[len(histBounds)-1]
}

type statAgg struct {
	Requests int64   `json:"n"`
	Errors   int64   `json:"err,omitempty"`
	Rejected int64   `json:"rej,omitempty"`
	In       int64   `json:"in,omitempty"`
	Out      int64   `json:"out,omitempty"`
	Cached   int64   `json:"cached,omitempty"`
	GenMs    int64   `json:"gen_ms,omitempty"`
	GenTok   int64   `json:"gen_tok,omitempty"`
	DurMs    int64   `json:"dur_ms,omitempty"`
	TTFT     latHist `json:"ttft"`
	Dur      latHist `json:"dur"`
}

func (a *statAgg) addRec(r statRec) {
	a.Requests++
	switch r.Outcome {
	case outcomeError:
		a.Errors++
	case outcomeRejected:
		a.Rejected++
	}
	a.In += r.In
	a.Out += r.Out
	a.Cached += r.Cached
	if r.GenMs > 0 && r.Out > 0 {
		a.GenMs += r.GenMs
		a.GenTok += r.Out
	}
	if r.Outcome == outcomeOK {
		a.Dur.add(r.DurMs)
		a.DurMs += r.DurMs
		if r.TTFTMs >= 0 {
			a.TTFT.add(r.TTFTMs)
		}
	}
}

func (a *statAgg) merge(o *statAgg) {
	a.Requests += o.Requests
	a.Errors += o.Errors
	a.Rejected += o.Rejected
	a.In += o.In
	a.Out += o.Out
	a.Cached += o.Cached
	a.GenMs += o.GenMs
	a.GenTok += o.GenTok
	a.DurMs += o.DurMs
	a.TTFT.merge(&o.TTFT)
	a.Dur.merge(&o.Dur)
}

// tokS is the average generation speed, 0 when unknown.
func (a *statAgg) tokS() float64 {
	if a.GenMs <= 0 {
		return 0
	}
	return float64(a.GenTok) * 1000 / float64(a.GenMs)
}

type statKey struct{ Source, Route, Model string }

type statSlots map[int64]map[statKey]*statAgg

func (s statSlots) at(t int64, k statKey) *statAgg {
	m := s[t]
	if m == nil {
		m = map[statKey]*statAgg{}
		s[t] = m
	}
	a := m[k]
	if a == nil {
		a = &statAgg{}
		m[k] = a
	}
	return a
}

type dirtyStat struct {
	table byte // 'm' minute, 'h' hour, 't' total
	t     int64
	k     statKey
}

type statsStore struct {
	events   *eventLog
	inFlight atomic.Int64

	mu      sync.Mutex
	minutes statSlots
	hours   statSlots
	totals  map[statKey]*statAgg
	since   int64 // the first record ever, unix seconds; 0 before any
	dirty   map[dirtyStat]bool
	db      *sql.DB
	closed  bool
}

func newStatsStore(ev *eventLog) *statsStore {
	return &statsStore{events: ev, minutes: statSlots{}, hours: statSlots{}, totals: map[statKey]*statAgg{}, dirty: map[dirtyStat]bool{}}
}

var sourceLabels = map[string]string{srcChat: "Chat", srcLAN: "API", srcInternet: "Internet API"}

func (s *statsStore) record(r statRec) {
	if s == nil {
		return
	}
	if r.At.IsZero() {
		r.At = time.Now()
	}
	sec := r.At.Unix()
	m, h := sec/60*60, sec/3600*3600
	k := statKey{r.Source, r.Route, r.Model}
	s.mu.Lock()
	s.minutes.at(m, k).addRec(r)
	s.hours.at(h, k).addRec(r)
	t := s.totals[k]
	if t == nil {
		t = &statAgg{}
		s.totals[k] = t
	}
	t.addRec(r)
	if s.since == 0 || sec < s.since {
		s.since = sec
	}
	s.dirty[dirtyStat{'m', m, k}] = true
	s.dirty[dirtyStat{'h', h, k}] = true
	s.dirty[dirtyStat{'t', 0, k}] = true
	s.mu.Unlock()
	if r.Outcome == outcomeError {
		msg := strings.TrimSpace(r.ErrMsg)
		if msg == "" {
			msg = "failed"
		}
		if len(msg) > 300 {
			msg = msg[:300] + "…"
		}
		s.events.add(evError, r.Model, fmt.Sprintf("%s %s: %s", sourceLabels[r.Source], r.Route, msg))
	}
}

// statsOf is the manager's statistics; nil (which records nothing) without
// a manager.
func statsOf(m *Manager) *statsStore {
	if m == nil {
		return nil
	}
	return m.stats
}

// begin counts a request in flight until the returned func is called.
func (s *statsStore) begin() func() {
	if s == nil {
		return func() {}
	}
	s.inFlight.Add(1)
	var once sync.Once
	return func() { once.Do(func() { s.inFlight.Add(-1) }) }
}

// attach loads the kept counts from chats.db and writes to it from now on.
func (s *statsStore) attach(db *sql.DB) {
	if s == nil || db == nil {
		return
	}
	now := time.Now().Unix()
	loaded := statSlots{}
	loadedH := statSlots{}
	totals := map[statKey]*statAgg{}
	load := func(q string, args []any, put func(t int64, k statKey, a *statAgg)) {
		rows, err := db.Query(q, args...)
		if err != nil {
			log.Printf("dashboard: reading statistics: %v", err)
			return
		}
		defer rows.Close()
		for rows.Next() {
			var t int64
			var k statKey
			var js string
			if rows.Scan(&t, &k.Source, &k.Route, &k.Model, &js) != nil {
				continue
			}
			a := &statAgg{}
			if json.Unmarshal([]byte(js), a) == nil {
				put(t, k, a)
			}
		}
	}
	load(`SELECT t, source, route, model, agg FROM stats_minute WHERE t >= ?`, []any{now - int64(statsMinuteKeep.Seconds())},
		func(t int64, k statKey, a *statAgg) { loaded.at(t, k).merge(a) })
	load(`SELECT t, source, route, model, agg FROM stats_hour WHERE t >= ?`, []any{now - int64(statsHourKeep.Seconds())},
		func(t int64, k statKey, a *statAgg) { loadedH.at(t, k).merge(a) })
	load(`SELECT 0, source, route, model, agg FROM stats_total`, nil, func(_ int64, k statKey, a *statAgg) { totals[k] = a })
	var since sql.NullInt64
	_ = db.QueryRow(`SELECT v FROM stats_meta WHERE k = 'since'`).Scan(&since)

	s.mu.Lock()
	defer s.mu.Unlock()
	for t, m := range loaded {
		for k, a := range m {
			s.minutes.at(t, k).merge(a)
		}
	}
	for t, m := range loadedH {
		for k, a := range m {
			s.hours.at(t, k).merge(a)
		}
	}
	for k, a := range totals {
		if cur := s.totals[k]; cur != nil {
			cur.merge(a)
		} else {
			s.totals[k] = a
		}
	}
	if since.Valid && since.Int64 > 0 && (s.since == 0 || since.Int64 < s.since) {
		s.since = since.Int64
	}
	s.db = db
}

// flush writes the changed counts. On failure they stay marked for the next
// try.
func (s *statsStore) flush() {
	if s == nil {
		return
	}
	s.mu.Lock()
	db := s.db
	if db == nil || s.closed || len(s.dirty) == 0 {
		s.mu.Unlock()
		return
	}
	type row struct {
		d  dirtyStat
		js []byte
	}
	rows := make([]row, 0, len(s.dirty))
	for d := range s.dirty {
		var a *statAgg
		switch d.table {
		case 'm':
			a = s.minutes[d.t][d.k]
		case 'h':
			a = s.hours[d.t][d.k]
		default:
			a = s.totals[d.k]
		}
		if a == nil {
			continue
		}
		js, _ := json.Marshal(a)
		rows = append(rows, row{d, js})
	}
	since := s.since
	s.dirty = map[dirtyStat]bool{}
	s.mu.Unlock()

	err := func() error {
		tx, err := db.Begin()
		if err != nil {
			return err
		}
		defer tx.Rollback()
		for _, r := range rows {
			var q string
			args := []any{r.d.k.Source, r.d.k.Route, r.d.k.Model, string(r.js)}
			switch r.d.table {
			case 'm':
				q, args = `INSERT OR REPLACE INTO stats_minute (t, source, route, model, agg) VALUES (?, ?, ?, ?, ?)`, append([]any{r.d.t}, args...)
			case 'h':
				q, args = `INSERT OR REPLACE INTO stats_hour (t, source, route, model, agg) VALUES (?, ?, ?, ?, ?)`, append([]any{r.d.t}, args...)
			default:
				q = `INSERT OR REPLACE INTO stats_total (source, route, model, agg) VALUES (?, ?, ?, ?)`
			}
			if _, err := tx.Exec(q, args...); err != nil {
				return err
			}
		}
		if since > 0 {
			if _, err := tx.Exec(`INSERT OR REPLACE INTO stats_meta (k, v) VALUES ('since', ?)`, since); err != nil {
				return err
			}
		}
		return tx.Commit()
	}()
	if err != nil {
		log.Printf("dashboard: writing statistics: %v (kept for the next try)", err)
		s.mu.Lock()
		for _, r := range rows {
			s.dirty[r.d] = true
		}
		s.mu.Unlock()
	}
}

// prune drops minutes and hours past their keep, in memory and the database.
func (s *statsStore) prune(now time.Time) {
	if s == nil {
		return
	}
	mCut, hCut := now.Add(-statsMinuteKeep).Unix(), now.Add(-statsHourKeep).Unix()
	s.mu.Lock()
	for t := range s.minutes {
		if t < mCut {
			delete(s.minutes, t)
		}
	}
	for t := range s.hours {
		if t < hCut {
			delete(s.hours, t)
		}
	}
	for d := range s.dirty {
		if (d.table == 'm' && d.t < mCut) || (d.table == 'h' && d.t < hCut) {
			delete(s.dirty, d)
		}
	}
	db := s.db
	s.mu.Unlock()
	if db != nil {
		_, _ = db.Exec(`DELETE FROM stats_minute WHERE t < ?`, mCut)
		_, _ = db.Exec(`DELETE FROM stats_hour WHERE t < ?`, hCut)
	}
}

// close writes what's left, before the database closes.
func (s *statsStore) close() {
	if s == nil {
		return
	}
	s.flush()
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
}

// statSpan is a range cut into n buckets of step seconds, the last one
// holding now.
type statSpan struct {
	from, step int64
	n          int
	hours      bool // read hourly counts; else per minute
}

var dashRanges = map[string]struct {
	step  int64
	n     int
	hours bool
}{
	"1h":  {60, 60, false},
	"24h": {900, 96, false},
	"7d":  {3600, 168, true},
	"30d": {21600, 120, true},
	"all": {86400, 90, true},
}

func spanFor(rng string, now time.Time) statSpan {
	r, ok := dashRanges[rng]
	if !ok {
		r = dashRanges["24h"]
	}
	end := now.Unix() / r.step * r.step
	return statSpan{from: end - int64(r.n-1)*r.step, step: r.step, n: r.n, hours: r.hours}
}

func (sp statSpan) times() []int64 {
	t := make([]int64, sp.n)
	for i := range t {
		t[i] = sp.from + int64(i)*sp.step
	}
	return t
}

// each calls fn for every count in the span, with its bucket, under the lock:
// fn must only add up.
func (s *statsStore) each(sp statSpan, fn func(bucket int, k statKey, a *statAgg)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	slots := s.minutes
	if sp.hours {
		slots = s.hours
	}
	end := sp.from + int64(sp.n)*sp.step
	for t, m := range slots {
		// An hour slot starting before the span still overlaps it when the
		// step is shorter than an hour; it counts in the first bucket.
		if t+slotLen(sp.hours) <= sp.from || t >= end {
			continue
		}
		b := int((t - sp.from) / sp.step)
		if b < 0 {
			b = 0
		}
		for k, a := range m {
			fn(b, k, a)
		}
	}
}

func slotLen(hours bool) int64 {
	if hours {
		return 3600
	}
	return 60
}

// allTotals is every count since the first record.
func (s *statsStore) allTotals(fn func(k statKey, a *statAgg)) int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	for k, a := range s.totals {
		fn(k, a)
	}
	return s.since
}
