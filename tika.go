package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"path"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

const (
	defaultTikaTimeout = 2 * time.Minute
	tikaProbeEvery     = 30 * time.Second
	tikaProbeTimeout   = 3 * time.Second
	maxTikaResponse    = 64 << 20
	tikaParallel       = 2  // documents parsed at once
	tikaMaxInFlight    = 16 // parsing plus waiting; more are refused
)

// tikaClient talks to an Apache Tika server (3.3 or newer). /rmeta/markdown
// keeps headings and tables and lists embedded files separately, and works on
// both the 3.x and 4.x lines, which name their metadata keys differently.
type tikaClient struct {
	base     string
	shown    string // base with any password hidden, for the UI and the log
	timeout  time.Duration
	http     *http.Client
	slots    chan struct{}
	inFlight atomic.Int32

	mu      sync.Mutex
	probing bool
	state   tikaState
}

type tikaState struct {
	OK      bool
	Version string
	Err     string
	At      time.Time
}

// extractError is a failed /api/extract. Reason tells the browser whether
// reading the file itself is worth a try.
type extractError struct {
	Status int
	Reason string // unreachable, timeout, busy, too_large, encrypted, unreadable
	Msg    string
}

func (e *extractError) Error() string { return e.Msg }

// extracted is a document's text as the chat sends it to the model.
type extracted struct {
	Text      string `json:"text"`
	Type      string `json:"type"`
	Pages     int    `json:"pages,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
	Via       string `json:"via"`
}

func newTikaClient(c TikaConf) *tikaClient {
	if !c.Enabled || c.URL == "" {
		return nil
	}
	u, err := url.Parse(c.URL)
	if err != nil {
		return nil // validate rejects this before we get here
	}
	timeout := c.Timeout.Duration
	if timeout <= 0 {
		timeout = defaultTikaTimeout
	}
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.Proxy = nil // users' documents never go through a proxy picked up from the environment
	return &tikaClient{
		base:    strings.TrimRight(c.URL, "/"),
		shown:   strings.TrimRight(u.Redacted(), "/"),
		timeout: timeout,
		http: &http.Client{Transport: tr, CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		}},
		slots: make(chan struct{}, tikaParallel),
	}
}

// Status returns what the last check found and starts a new check in the
// background when that one is stale, so callers never wait on Tika.
func (t *tikaClient) Status() tikaState {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.probing && time.Since(t.state.At) > tikaProbeEvery {
		t.probing = true
		go t.probe()
	}
	return t.state
}

// ProbeNow checks Tika and waits for the answer.
func (t *tikaClient) ProbeNow() tikaState {
	t.mu.Lock()
	t.probing = true
	t.mu.Unlock()
	return t.probe()
}

var printableRe = regexp.MustCompile(`[^\x20-\x7e]+`)

func (t *tikaClient) probe() tikaState {
	ctx, cancel := context.WithTimeout(context.Background(), tikaProbeTimeout)
	defer cancel()
	st := tikaState{}
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, t.base+"/version", nil)
	resp, err := t.http.Do(req)
	if err == nil {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			st.OK = true
			st.Version = truncateRunes(strings.TrimSpace(printableRe.ReplaceAllString(string(b), " ")), 80)
		} else {
			err = fmt.Errorf("it answered %s", resp.Status)
		}
	}
	if err != nil {
		st.Err = shortNetErr(err)
	}
	t.mu.Lock()
	if st.OK && st.Version == "" {
		st.Version = t.state.Version
	}
	st.At = time.Now()
	t.state = st
	t.probing = false
	t.mu.Unlock()
	return st
}

// note records what a document request learned about Tika.
func (t *tikaClient) note(err error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.state.At = time.Now()
	t.state.OK = err == nil
	t.state.Err = ""
	if err != nil {
		t.state.Err = shortNetErr(err)
	}
}

func shortNetErr(err error) string {
	if errors.Is(err, context.DeadlineExceeded) {
		return "no answer in time"
	}
	var ue *url.Error
	if errors.As(err, &ue) {
		err = ue.Err
	}
	return err.Error()
}

// Extract sends one document to Tika and returns its text. ctx ends when the
// browser gives up (the user removed the file); the configured timeout caps
// a document that takes too long.
func (t *tikaClient) Extract(ctx context.Context, data []byte, name, ctype string) (extracted, error) {
	if t.inFlight.Add(1) > tikaMaxInFlight {
		t.inFlight.Add(-1)
		return extracted{}, &extractError{http.StatusServiceUnavailable, "busy", "Tika is busy with other documents. Try again in a moment."}
	}
	defer t.inFlight.Add(-1)
	select {
	case t.slots <- struct{}{}:
	case <-ctx.Done():
		return extracted{}, ctx.Err()
	}
	defer func() { <-t.slots }()

	tctx, cancel := context.WithTimeout(ctx, t.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(tctx, http.MethodPut, t.base+"/rmeta/markdown", bytes.NewReader(data))
	if err != nil {
		return extracted{}, err
	}
	req.ContentLength = int64(len(data))
	req.Header.Set("Content-Type", tikaContentType(ctype))
	// The extension helps Tika tell apart formats that share a container
	// (docx and xlsx are both zips); the real name stays here.
	req.Header.Set("Content-Disposition", `attachment; filename="document`+docExt(name)+`"`)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("writeLimit", strconv.Itoa(maxDocTextBytes)) // honoured by 3.x; 4.x ignores client limits
	failed := func(err error) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if errors.Is(tctx.Err(), context.DeadlineExceeded) {
			return &extractError{http.StatusGatewayTimeout, "timeout", fmt.Sprintf("Tika didn't finish reading %s within %s.", name, t.timeout)}
		}
		t.note(err)
		return &extractError{http.StatusBadGateway, "unreachable", fmt.Sprintf("Tika at %s didn't answer: %s.", t.shown, shortNetErr(err))}
	}
	resp, err := t.http.Do(req)
	if err != nil {
		return extracted{}, failed(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxTikaResponse+1))
	if err != nil {
		return extracted{}, failed(err)
	}
	t.note(nil)
	if resp.StatusCode != http.StatusOK {
		line, _, _ := strings.Cut(strings.TrimSpace(string(body)), "\n")
		return extracted{}, &extractError{http.StatusBadGateway, "unreadable", fmt.Sprintf("Tika answered %s: %s", resp.Status, truncateRunes(line, 200))}
	}
	if len(body) > maxTikaResponse {
		return extracted{}, &extractError{http.StatusRequestEntityTooLarge, "too_large", fmt.Sprintf("%s has more than %d MB of text. Attach a smaller part of it.", name, maxTikaResponse>>20)}
	}
	return parseRmeta(body, name)
}

func parseRmeta(body []byte, name string) (extracted, error) {
	var entries []map[string]any
	if err := json.Unmarshal(body, &entries); err != nil || len(entries) == 0 {
		return extracted{}, &extractError{http.StatusBadGateway, "unreadable", "Tika sent an answer llama-manager can't read. Check that tika.url points at Apache Tika 3.3 or newer."}
	}
	first := entries[0]
	res := extracted{Type: mediaType(metaStr(first, "Content-Type")), Via: "tika"}
	res.Pages, _ = strconv.Atoi(metaStr(first, "xmpTPg:NPages"))
	var b strings.Builder
	b.WriteString(metaStr(first, "tk:content", "X-TIKA:content"))
	for _, e := range entries[1:] {
		c := strings.TrimSpace(metaStr(e, "tk:content", "X-TIKA:content"))
		// Pictures inside a document come back as just their own name.
		if c == "" || c == strings.TrimSpace(metaStr(e, "tk:resource-name", "resourceName")) {
			continue
		}
		p := strings.TrimPrefix(metaStr(e, "tk:embedded-resource-path", "X-TIKA:embedded_resource_path"), "/")
		fmt.Fprintf(&b, "\n\n[Embedded file %s]\n%s", p, c)
	}
	limited := false
	for _, e := range entries {
		for k := range e {
			if l := strings.ToLower(k); strings.Contains(l, "write_limit") || strings.Contains(l, "write-limit") {
				limited = true
			}
		}
	}
	text, cut := cleanDocText(b.String(), res.Type == "application/pdf")
	res.Text, res.Truncated = text, cut || limited
	if res.Text == "" {
		if metaStr(first, "tk:encrypted", "X-TIKA:encrypted", "pdf:encrypted") == "true" {
			return extracted{}, &extractError{http.StatusUnprocessableEntity, "encrypted", name + " is password-protected. Remove the password and attach it again."}
		}
		if ex := containerException(first); ex != "" {
			return extracted{}, &extractError{http.StatusUnprocessableEntity, "unreadable", fmt.Sprintf("Tika couldn't read %s: %s", name, ex)}
		}
	}
	return res, nil
}

// metaStr returns the first of keys that is set. Tika 4 renamed X-TIKA:* to tk:*.
func metaStr(m map[string]any, keys ...string) string {
	for _, k := range keys {
		switch v := m[k].(type) {
		case string:
			return v
		case []any:
			if len(v) > 0 {
				if s, ok := v[0].(string); ok {
					return s
				}
			}
		}
	}
	return ""
}

func containerException(m map[string]any) string {
	for k, v := range m {
		l := strings.ToLower(k)
		if !strings.Contains(l, "exception") || !strings.Contains(l, "container") {
			continue
		}
		s, _ := v.(string)
		line, _, _ := strings.Cut(strings.TrimSpace(s), "\n")
		// "org.apache.tika.exception.TikaException: Error opening OOXML file"
		if i := strings.Index(line, ": "); i > 0 && strings.Contains(line[:i], ".") && !strings.Contains(line[:i], " ") {
			line = line[i+2:]
		}
		return truncateRunes(strings.TrimSpace(line), 200)
	}
	return ""
}

func mediaType(s string) string {
	t, _, err := mime.ParseMediaType(s)
	if err != nil {
		return ""
	}
	return t
}

func tikaContentType(ctype string) string {
	if t := mediaType(ctype); strings.Contains(t, "/") {
		return t
	}
	return "application/octet-stream"
}

var extRe = regexp.MustCompile(`^\.[a-z0-9]{1,10}$`)

func docExt(name string) string {
	if e := strings.ToLower(path.Ext(name)); extRe.MatchString(e) {
		return e
	}
	return ""
}

const docCutNote = "\n\n[llama-manager cut the text here: it sends at most 2 MB of a document's text]"

// cleanDocText tidies extracted text for the model: one kind of line break,
// no control characters or invisible spaces, no runs of blank lines. It keeps
// at most maxDocTextBytes and reports whether it had to cut. PDFs often map
// their hyphen glyph to the soft hyphen, so there it becomes a real one.
func cleanDocText(s string, pdf bool) (string, bool) {
	s = strings.ToValidUTF8(strings.ReplaceAll(s, "\r\n", "\n"), "\uFFFD")
	s = strings.Map(func(r rune) rune {
		switch r {
		case '\t', '\n':
			return r
		case '\r', '\f', '\v', 0x2028, 0x2029: // line and paragraph separators
			return '\n'
		case 0xA0, 0x2007, 0x202F: // no-break spaces
			return ' '
		case 0xAD: // soft hyphen
			if pdf {
				return '-'
			}
			return -1
		case 0xFEFF, 0x200B: // byte order mark, zero-width space
			return -1
		}
		if r < 0x20 || (r >= 0x7F && r < 0xA0) {
			return -1
		}
		return r
	}, s)
	lines := strings.Split(s, "\n")
	out := lines[:0]
	blank := true
	for _, ln := range lines {
		ln = strings.TrimRight(ln, " \t")
		if ln == "" {
			if !blank {
				out = append(out, "")
			}
			blank = true
			continue
		}
		blank = false
		out = append(out, ln)
	}
	text := strings.TrimSpace(strings.Join(out, "\n"))
	if len(text) <= maxDocTextBytes {
		return text, false
	}
	cut := maxDocTextBytes - len(docCutNote)
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	if nl := strings.LastIndexByte(text[:cut], '\n'); nl > cut*9/10 {
		cut = nl
	}
	return strings.TrimRight(text[:cut], " \t\n") + docCutNote, true
}
