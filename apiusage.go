package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// usageRetention is how long per-key daily usage is kept.
const usageRetention = 90 * 24 * time.Hour

// meterWriter records a /v1 reply's status and the token counts in it: the
// last "prompt_tokens"/"input_tokens" and "completion_tokens"/"output_tokens"
// it carries (a streamed reply has them in its last event).
type meterWriter struct {
	http.ResponseWriter
	noCount bool // a reply that only shows stored or counted tokens
	status  int
	scan    bool
	stream  bool
	first   time.Time // the first byte of a streamed reply
	tail    []byte
	in, out int64
	cached  int64
	genMs   int64  // llama-server's timings.predicted_ms
	errBody []byte // the start of a 5xx reply, for the activity feed
}

// generates says whether r makes the model generate: a POST, other than
// counting tokens. Fetching a stored response shows tokens already counted.
func generates(r *http.Request) bool {
	return r.Method == http.MethodPost && r.URL.Path != "/v1/responses/input_tokens" && r.URL.Path != "/v1/messages/count_tokens"
}

// usageRe finds a token count, not one inside a JSON string (escaped quotes).
var usageRe = regexp.MustCompile(`(?:^|[^\\])"(prompt_tokens|input_tokens|completion_tokens|output_tokens|cached_tokens|predicted_ms)"\s*:\s*(\d+)`)

func (m *meterWriter) WriteHeader(code int) {
	if m.status == 0 {
		m.status = code
		ct := m.Header().Get("Content-Type")
		m.scan = !m.noCount && (strings.HasPrefix(ct, "application/json") || strings.HasPrefix(ct, "text/event-stream"))
		m.stream = strings.HasPrefix(ct, "text/event-stream")
	}
	m.ResponseWriter.WriteHeader(code)
}

func (m *meterWriter) Write(b []byte) (int, error) {
	if m.status == 0 {
		m.WriteHeader(http.StatusOK)
	}
	if m.stream && m.first.IsZero() && len(b) > 0 {
		m.first = time.Now()
	}
	if m.status >= 500 && len(m.errBody) < 2048 {
		m.errBody = append(m.errBody, b[:min(len(b), 2048-len(m.errBody))]...)
	}
	if m.scan {
		m.count(b)
	}
	return m.ResponseWriter.Write(b)
}

func (m *meterWriter) count(b []byte) {
	buf := make([]byte, 0, len(m.tail)+len(b))
	buf = append(append(buf, m.tail...), b...)
	for _, g := range usageRe.FindAllSubmatch(buf, -1) {
		n, err := strconv.ParseInt(string(g[2]), 10, 64)
		if err != nil {
			continue
		}
		switch string(g[1]) {
		case "prompt_tokens", "input_tokens":
			m.in = n
		case "cached_tokens":
			m.cached = n
		case "predicted_ms":
			m.genMs = n
		default:
			m.out = n
		}
	}
	// Keep the end, so a count split between two writes is still found.
	if len(buf) > 64 {
		buf = buf[len(buf)-64:]
	}
	m.tail = append([]byte(nil), buf...)
}

func (m *meterWriter) Flush() {
	if f, ok := m.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (m *meterWriter) Unwrap() http.ResponseWriter { return m.ResponseWriter }

// reqInfo is filled in while serveV1 handles a request: the model it went
// to, and the tokens of replies a WebSocket session sends after the upgrade,
// which meterWriter never sees.
type reqInfo struct {
	in, out atomic.Int64
	mu      sync.Mutex
	model   string
}

type reqInfoKey struct{}

// infoFrom is the request's reqInfo; a throwaway one outside serveV1.
func infoFrom(ctx context.Context) *reqInfo {
	if t, ok := ctx.Value(reqInfoKey{}).(*reqInfo); ok {
		return t
	}
	return &reqInfo{}
}

// noteModel records the instance a /v1 request reached; the first one wins.
func noteModel(ctx context.Context, name string) {
	t := infoFrom(ctx)
	t.mu.Lock()
	if t.model == "" {
		t.model = name
	}
	t.mu.Unlock()
}

func (t *reqInfo) modelName() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.model
}

// addResponse adds a Responses API response's usage.
func (t *reqInfo) addResponse(resp any) {
	m, _ := resp.(map[string]any)
	u, _ := m["usage"].(map[string]any)
	in, _ := u["input_tokens"].(float64)
	out, _ := u["output_tokens"].(float64)
	t.in.Add(int64(in))
	t.out.Add(int64(out))
	if name, _ := m["model"].(string); name != "" {
		t.mu.Lock()
		if t.model == "" {
			t.model = name
		}
		t.mu.Unlock()
	}
}

// errText is the message of a JSON error reply, else the reply's text.
func errText(b []byte) string {
	var e struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(b, &e) == nil && e.Error.Message != "" {
		return e.Error.Message
	}
	return strings.TrimSpace(string(b))
}

// usageCount is one key's use on one day.
type usageCount struct {
	Requests     int64 `json:"requests"`
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
}

func (c *usageCount) add(o usageCount) {
	c.Requests += o.Requests
	c.InputTokens += o.InputTokens
	c.OutputTokens += o.OutputTokens
}

// keyUsage is what Settings shows for a key.
type keyUsage struct {
	Today    usageCount `json:"today"`
	Month    usageCount `json:"month"` // the last 30 days, today included
	LastUsed int64      `json:"last_used,omitempty"`
}

type usageKey struct{ id, day string }

// apiUsage counts each key's requests and tokens per day. Counts wait in
// memory and are written to chats.db every half minute and on shutdown, so a
// busy API doesn't write the database on every request.
type apiUsage struct {
	store func() *apiStore

	flushMu sync.Mutex // one flush at a time
	closed  bool       // the database is closing; keep counts in memory

	mu      sync.Mutex
	pending map[usageKey]usageCount
	last    map[string]int64 // key id -> last request, unix seconds
}

func newAPIUsage(store func() *apiStore) *apiUsage {
	return &apiUsage{store: store, pending: map[usageKey]usageCount{}, last: map[string]int64{}}
}

func usageDay(t time.Time) string { return t.Format("2006-01-02") }

func (u *apiUsage) record(id string, in, out int64, now time.Time) {
	u.mu.Lock()
	defer u.mu.Unlock()
	k := usageKey{id, usageDay(now)}
	c := u.pending[k]
	c.add(usageCount{1, in, out})
	u.pending[k] = c
	u.last[id] = now.Unix()
}

// usageFlushEvery is how often waiting counts are written.
var usageFlushEvery = 30 * time.Second

// run writes the counts every usageFlushEvery until ctx ends.
func (u *apiUsage) run(ctx context.Context) {
	t := time.NewTicker(usageFlushEvery)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			u.flush()
		}
	}
}

// close writes what is waiting, before the database closes.
func (u *apiUsage) close() {
	u.flush()
	u.flushMu.Lock()
	u.closed = true
	u.flushMu.Unlock()
}

// flush writes the waiting counts. Without a database they stay in memory.
func (u *apiUsage) flush() {
	u.flushMu.Lock()
	defer u.flushMu.Unlock()
	st := u.store()
	if st == nil || u.closed {
		return
	}
	u.mu.Lock()
	pending, last := u.pending, u.last
	u.pending, u.last = map[usageKey]usageCount{}, map[string]int64{}
	u.mu.Unlock()
	var failed bool
	for k, c := range pending {
		if _, err := st.db.Exec(`INSERT INTO api_usage (key_id, day, requests, input_tokens, output_tokens, last_used) VALUES (?, ?, ?, ?, ?, ?)
			ON CONFLICT(key_id, day) DO UPDATE SET requests = requests + excluded.requests, input_tokens = input_tokens + excluded.input_tokens,
			output_tokens = output_tokens + excluded.output_tokens, last_used = MAX(last_used, excluded.last_used)`,
			k.id, k.day, c.Requests, c.InputTokens, c.OutputTokens, last[k.id]); err != nil {
			failed = true
			u.mu.Lock()
			p := u.pending[k]
			p.add(c)
			u.pending[k] = p
			u.last[k.id] = max(u.last[k.id], last[k.id])
			u.mu.Unlock()
		}
	}
	if failed {
		log.Printf("API usage: couldn't write to the database; keeping the counts in memory")
	}
}

// forKeys is each key's usage, in the keys' order.
func (u *apiUsage) forKeys(keys []APIKey, now time.Time) []keyUsage {
	out := make([]keyUsage, len(keys))
	if len(keys) == 0 {
		return out
	}
	today, from := usageDay(now), usageDay(now.AddDate(0, 0, -29))
	byID := map[string]*keyUsage{}
	for i, k := range keys {
		byID[keyID(k.Key)] = &out[i]
	}
	addDay := func(id, day string, c usageCount, last int64) {
		ku := byID[id]
		if ku == nil || day < from {
			return
		}
		ku.Month.add(c)
		if day == today {
			ku.Today.add(c)
		}
		ku.LastUsed = max(ku.LastUsed, last)
	}
	if st := u.store(); st != nil {
		rows, err := st.db.Query(`SELECT key_id, day, requests, input_tokens, output_tokens, last_used FROM api_usage WHERE day >= ?`, from)
		if err == nil {
			for rows.Next() {
				var id, day string
				var c usageCount
				var last int64
				if rows.Scan(&id, &day, &c.Requests, &c.InputTokens, &c.OutputTokens, &last) == nil {
					addDay(id, day, c, last)
				}
			}
			rows.Close()
		}
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	for k, c := range u.pending {
		addDay(k.id, k.day, c, u.last[k.id])
	}
	return out
}
