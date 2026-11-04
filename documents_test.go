package main

import (
	"archive/zip"
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unicode/utf8"
)

func TestCleanDocText(t *testing.T) {
	in := "Title\r\n\r\n\r\n\r\nline\u00a0one \t\nUNIQUE\u00adMARKER\x00\x07 end\u200b\fnext page\n\n\n"
	got, cut := cleanDocText(in, true)
	want := "Title\n\nline one\nUNIQUE-MARKER end\nnext page"
	if got != want || cut {
		t.Errorf("pdf: got %q (cut %v), want %q", got, cut, want)
	}
	if got, _ := cleanDocText("co\u00adop", false); got != "coop" {
		t.Errorf("outside PDFs a soft hyphen is invisible and dropped, got %q", got)
	}

	long := strings.Repeat("ж", maxDocTextBytes) // 2 bytes per rune: twice the limit
	got, cut = cleanDocText(long, false)
	if !cut || len(got) > maxDocTextBytes || !utf8.ValidString(got) || !strings.HasSuffix(got, docCutNote) {
		t.Errorf("long text: cut %v, %d bytes, valid %v", cut, len(got), utf8.ValidString(got))
	}
}

func TestFenceOutlastsBackticksInside(t *testing.T) {
	if got := fence("plain"); got != "```\nplain\n```" {
		t.Errorf("plain: %q", got)
	}
	got := fence("a\n```go\nx\n```\n")
	if !strings.HasPrefix(got, "````\n") || !strings.HasSuffix(got, "\n````") {
		t.Errorf("a document with its own fences must get a longer one: %q", got)
	}
}

// The 3.x and 4.x lines of Tika name their metadata differently.
func TestParseRmetaBothKeyStyles(t *testing.T) {
	v3 := `[{"Content-Type":"application/pdf","xmpTPg:NPages":"3","X-TIKA:content":"Page one\u00a0text\n\n\n\nPage two"},
	        {"Content-Type":"image/jpeg","resourceName":"/docProps/thumbnail.jpeg","X-TIKA:content":"/docProps/thumbnail.jpeg  ","X-TIKA:embedded_resource_path":"/thumbnail.jpeg"},
	        {"Content-Type":"text/plain","X-TIKA:content":"inside the zip","X-TIKA:embedded_resource_path":"/readme.txt"}]`
	v4 := `[{"Content-Type":"application/pdf","xmpTPg:NPages":"3","tk:content":"Page one\u00a0text\n\n\n\nPage two"},
	        {"Content-Type":"image/jpeg","tk:resource-name":"/docProps/thumbnail.jpeg","tk:content":"/docProps/thumbnail.jpeg  \n","tk:embedded-resource-path":"/thumbnail.jpeg"},
	        {"Content-Type":"text/plain","tk:content":"inside the zip","tk:embedded-resource-path":"/readme.txt"}]`
	for name, body := range map[string]string{"3.x": v3, "4.x": v4} {
		res, err := parseRmeta([]byte(body), "a.pdf")
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		want := "Page one text\n\nPage two\n\n[Embedded file readme.txt]\ninside the zip"
		if res.Text != want || res.Pages != 3 || res.Type != "application/pdf" || res.Via != "tika" || res.Truncated {
			t.Errorf("%s: %+v", name, res)
		}
	}

	for _, tc := range []struct{ body, reason string }{
		{`[{"Content-Type":"application/pdf","pdf:encrypted":"true","tk:exception:container-exception":"org.apache.tika.exception.EncryptedDocumentException: Unable to process: document is encrypted\n\tat x"}]`, "encrypted"},
		{`[{"Content-Type":"application/pdf","X-TIKA:encrypted":"true"}]`, "encrypted"},
		{`[{"Content-Type":"application/msword","X-TIKA:EXCEPTION:container_exception":"org.apache.poi.openxml4j.exceptions.NotOfficeXmlFileException: No valid entries or contents found\n\tat y"}]`, "unreadable"},
		{`not json`, "unreadable"},
		{`[]`, "unreadable"},
	} {
		_, err := parseRmeta([]byte(tc.body), "x.doc")
		var ee *extractError
		if !errors.As(err, &ee) || ee.Reason != tc.reason {
			t.Errorf("%s: got %v, want reason %s", tc.body, err, tc.reason)
		}
		if ee != nil && ee.Reason == "unreadable" && strings.Contains(ee.Msg, "org.apache") {
			t.Errorf("Java class names should not reach the user: %q", ee.Msg)
		}
	}

	res, err := parseRmeta([]byte(`[{"Content-Type":"application/pdf","xmpTPg:NPages":"1"}]`), "scan.pdf")
	if err != nil || res.Text != "" || res.Pages != 1 {
		t.Errorf("a scan has no text and no error, so the browser can offer page images: %+v %v", res, err)
	}
	res, _ = parseRmeta([]byte(`[{"Content-Type":"text/plain","X-TIKA:content":"abc","X-TIKA:EXCEPTION:write_limit_reached":"true"}]`), "x.txt")
	if !res.Truncated {
		t.Error("Tika's write limit means the text was cut")
	}
}

// fakeTika answers like Tika 4: /version and /rmeta/markdown.
type fakeTika struct {
	*httptest.Server
	mu      sync.Mutex
	headers http.Header
	body    []byte
	reply   func(w http.ResponseWriter, r *http.Request)
}

func newFakeTika(t *testing.T) *fakeTika {
	t.Helper()
	f := &fakeTika{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/version" {
			fmt.Fprint(w, "Apache Tika 4.2.0")
			return
		}
		b, _ := io.ReadAll(r.Body)
		f.mu.Lock()
		f.headers, f.body = r.Header.Clone(), b
		reply := f.reply
		f.mu.Unlock()
		if r.Method != http.MethodPut || r.URL.Path != "/rmeta/markdown" {
			http.Error(w, "unexpected "+r.Method+" "+r.URL.Path, http.StatusNotFound)
			return
		}
		if reply != nil {
			reply(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `[{"Content-Type":"application/vnd.openxmlformats-officedocument.wordprocessingml.document","tk:content":"# Report\n\nGot %d bytes"}]`, len(b))
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *fakeTika) setReply(fn func(w http.ResponseWriter, r *http.Request)) {
	f.mu.Lock()
	f.reply = fn
	f.mu.Unlock()
}

func TestTikaClientExtract(t *testing.T) {
	f := newFakeTika(t)
	tc := newTikaClient(TikaConf{Enabled: true, URL: f.URL + "/"})
	if tc.http.Transport.(*http.Transport).Proxy != nil {
		t.Error("documents must not go through a proxy taken from the environment")
	}
	res, err := tc.Extract(context.Background(), []byte("PK fake docx"), "Отчёт за год.DOCX", "application/vnd.openxmlformats-officedocument.wordprocessingml.document; charset=x")
	if err != nil || res.Text != "# Report\n\nGot 12 bytes" {
		t.Fatalf("extract: %+v %v", res, err)
	}
	f.mu.Lock()
	h := f.headers
	f.mu.Unlock()
	if h.Get("Content-Disposition") != `attachment; filename="document.docx"` {
		t.Errorf("Tika should get only the extension, not the user's file name: %q", h.Get("Content-Disposition"))
	}
	if h.Get("Content-Type") != "application/vnd.openxmlformats-officedocument.wordprocessingml.document" || h.Get("writeLimit") == "" {
		t.Errorf("headers: %v", h)
	}
	if st := tc.Status(); !st.OK {
		t.Errorf("a successful read means Tika is up: %+v", st)
	}

	f.setReply(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "java.lang.OutOfMemoryError: heap\n\tat ...", http.StatusInternalServerError)
	})
	_, err = tc.Extract(context.Background(), []byte("x"), "a.pdf", "")
	var ee *extractError
	if !errors.As(err, &ee) || ee.Status != http.StatusBadGateway || strings.Contains(ee.Msg, "\tat") {
		t.Errorf("a Tika error should be a short 502: %v", err)
	}

	// Never follow a redirect: the document would be sent somewhere else.
	var elsewhere atomic.Int32
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { elsewhere.Add(1) }))
	defer other.Close()
	f.setReply(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, other.URL+"/rmeta/markdown", http.StatusTemporaryRedirect)
	})
	if _, err := tc.Extract(context.Background(), []byte("x"), "a.pdf", ""); err == nil || elsewhere.Load() != 0 {
		t.Errorf("a redirect is not an answer and must not be followed: %v, %d hits", err, elsewhere.Load())
	}

	slow := newTikaClient(TikaConf{Enabled: true, URL: f.URL, Timeout: duration{50 * time.Millisecond}})
	f.setReply(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	})
	_, err = slow.Extract(context.Background(), []byte("x"), "big.pdf", "")
	if !errors.As(err, &ee) || ee.Reason != "timeout" || !strings.Contains(ee.Msg, "big.pdf") {
		t.Errorf("timeout: %v", err)
	}

	// The user removed the file: no error to show, just stop.
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(30 * time.Millisecond); cancel() }()
	if _, err := tc.Extract(ctx, []byte("x"), "a.pdf", ""); !errors.Is(err, context.Canceled) {
		t.Errorf("cancel: %v", err)
	}

	f.Close()
	_, err = tc.Extract(context.Background(), []byte("x"), "a.pdf", "")
	if !errors.As(err, &ee) || ee.Reason != "unreachable" {
		t.Errorf("Tika down: %v", err)
	}
	if st := tc.Status(); st.OK || st.Err == "" {
		t.Errorf("a failed request should mark Tika down: %+v", st)
	}
}

func TestTikaProbeHidesPassword(t *testing.T) {
	f := newFakeTika(t)
	tc := newTikaClient(TikaConf{Enabled: true, URL: strings.Replace(f.URL, "http://", "http://user:hunter2@", 1)})
	if strings.Contains(tc.shown, "hunter2") {
		t.Errorf("the password must not be shown: %s", tc.shown)
	}
	if st := tc.ProbeNow(); !st.OK || st.Version != "Apache Tika 4.2.0" {
		t.Errorf("probe: %+v", st)
	}
	if newTikaClient(TikaConf{URL: f.URL}) != nil {
		t.Error("a tika section with enabled: false must not be used")
	}
}

func extractReq(ws *WebServer, body []byte, name string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/api/extract", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/pdf")
	req.Header.Set("X-File-Name", strings.ReplaceAll(name, " ", "%20"))
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, req)
	return rr
}

func TestExtractAPI(t *testing.T) {
	ws, _ := newTestServer(t)
	rr := extractReq(ws, []byte("%PDF"), "a.pdf")
	var e map[string]string
	json.Unmarshal(rr.Body.Bytes(), &e)
	if rr.Code != http.StatusServiceUnavailable || e["reason"] != "off" {
		t.Errorf("without Tika: %d %s", rr.Code, rr.Body)
	}

	f := newFakeTika(t)
	ws.tika = newTikaClient(TikaConf{Enabled: true, URL: f.URL})
	rr = extractReq(ws, []byte("%PDF-1.7 hello"), "my report.pdf")
	var res extracted
	if err := json.Unmarshal(rr.Body.Bytes(), &res); err != nil || rr.Code != http.StatusOK || res.Text != "# Report\n\nGot 14 bytes" || res.Via != "tika" {
		t.Errorf("extract: %d %s", rr.Code, rr.Body)
	}

	rr = extractReq(ws, make([]byte, maxDocBytes+1), "huge.pdf")
	json.Unmarshal(rr.Body.Bytes(), &e)
	if rr.Code != http.StatusRequestEntityTooLarge || !strings.Contains(e["error"], "huge.pdf") {
		t.Errorf("too large: %d %s", rr.Code, rr.Body)
	}
	if rr := extractReq(ws, nil, "empty.pdf"); rr.Code != http.StatusBadRequest {
		t.Errorf("empty upload: %d", rr.Code)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/extract", strings.NewReader("x"))
	req.Header.Set("Origin", "http://evil.example")
	rr = httptest.NewRecorder()
	ws.ServeHTTP(rr, req)
	if rr.Code != http.StatusForbidden {
		t.Errorf("another site's page must not be able to post documents: %d", rr.Code)
	}
}

func TestStatusReportsDocuments(t *testing.T) {
	ws, _ := newTestServer(t)
	get := func() docStatus {
		rr := httptest.NewRecorder()
		ws.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/api/status", nil))
		var s ServerStatus
		if err := json.Unmarshal(rr.Body.Bytes(), &s); err != nil {
			t.Fatal(err)
		}
		return s.Documents
	}
	if d := get(); d.Tika != nil || d.MaxDocBytes != maxDocBytes || d.MaxFiles != maxAttachments || d.MaxTotalBytes != maxAttachBytes {
		t.Errorf("without Tika: %+v", d)
	}
	f := newFakeTika(t)
	ws.tika = newTikaClient(TikaConf{Enabled: true, URL: f.URL})
	if d := get(); d.Tika == nil || d.Tika.Checked {
		t.Errorf("the first status must not wait for Tika: %+v", d.Tika)
	}
	waitFor(t, 3*time.Second, "background Tika check", func() bool { d := get(); return d.Tika.Checked && d.Tika.OK })
}

func TestTikaConfigRules(t *testing.T) {
	for in, ok := range map[string]bool{
		"tika: {enabled: true, url: http://127.0.0.1:9998}":                true,
		"tika: {enabled: true, url: 'https://tika.lan/base', timeout: 5m}": true,
		"tika: {enabled: false, url: http://x:9998}":                       true,
		"tika: {enabled: true}":                                            false,
		"tika: {enabled: true, url: 'ftp://x'}":                            false,
		"tika: {enabled: true, url: 'http://x:9998/?q=1'}":                 false,
		"tika: {enabled: true, url: http://x, timeout: -1s}":               false,
		"tika: {enabled: true, uri: http://x}":                             false,
	} {
		cfg, err := decodeConfig([]byte("server_bin: /bin/true\n" + in))
		if err == nil {
			err = cfg.validate()
		}
		if (err == nil) != ok {
			t.Errorf("%s: err = %v, want ok %v", in, err, ok)
		}
	}

	// A save must keep the section, and not add one to configs without it.
	dir := t.TempDir()
	for name, in := range map[string]string{"with": "server_bin: /bin/true\ntika:\n  enabled: true\n  url: http://127.0.0.1:9998\n  timeout: 90s\n", "without": "server_bin: /bin/true\n"} {
		path := filepath.Join(dir, name+".yaml")
		os.WriteFile(path, []byte(in), 0o600)
		cfg, err := loadConfig(path)
		if err != nil {
			t.Fatal(err)
		}
		cfg.mu.Lock()
		cfg.NGL = 12
		err = cfg.saveLocked()
		cfg.mu.Unlock()
		if err != nil {
			t.Fatal(err)
		}
		out, _ := os.ReadFile(path)
		again, err := loadConfig(path)
		if err != nil {
			t.Fatal(err)
		}
		if name == "with" && (again.Tika != TikaConf{Enabled: true, URL: "http://127.0.0.1:9998", Timeout: duration{90 * time.Second}}) {
			t.Errorf("tika section lost on save:\n%s", out)
		}
		if name == "without" && strings.Contains(string(out), "tika") {
			t.Errorf("a config without Tika gained a tika section:\n%s", out)
		}
	}
}

func TestTikaCannotBeChangedOverTheAPI(t *testing.T) {
	ws, cfg := newTestServer(t)
	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"ngl":5,"tika":{"enabled":true,"url":"http://evil.example:9998"}}`))
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, req)
	if cfg.Tika != (TikaConf{}) {
		t.Errorf("settings API changed tika: %+v", cfg.Tika)
	}

	rr = postImport(t, ws, "server_bin: /bin/true\nallowed_hosts: [example.com]\ntika:\n  enabled: true\n  url: http://evil.example:9998\n")
	if rr.Code != http.StatusForbidden || cfg.Tika != (TikaConf{}) {
		t.Errorf("an import must not point documents somewhere else: %d %s, tika %+v", rr.Code, rr.Body, cfg.Tika)
	}

	cfg.mu.Lock()
	cfg.Tika = TikaConf{Enabled: true, URL: "http://127.0.0.1:9998"}
	cfg.mu.Unlock()
	rr = postImport(t, ws, "server_bin: /bin/true\nallowed_hosts: [example.com]\ntika:\n  enabled: true\n  url: http://127.0.0.1:9998\n")
	if rr.Code != http.StatusOK {
		t.Errorf("an import with the same tika section should work: %d %s", rr.Code, rr.Body)
	}
	rr = postImport(t, ws, "server_bin: /bin/true\nallowed_hosts: [example.com]\n")
	if rr.Code != http.StatusForbidden {
		t.Errorf("an import that drops the tika section changes it: %d", rr.Code)
	}
}

// A database written before documents existed opens and keeps its files.
func TestChatStoreMigratesSchemaV1(t *testing.T) {
	path := filepath.Join(t.TempDir(), "chats.db")
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	v1 := strings.Replace(chatSchema, "	data       BLOB NOT NULL,\n", "	data       BLOB NOT NULL\n", 1)
	v1 = v1[:strings.Index(v1, "	text       TEXT")] + v1[strings.Index(v1, ");\nCREATE INDEX IF NOT EXISTS attachments_message"):]
	v1 = strings.Replace(v1, "CREATE INDEX IF NOT EXISTS attachments_variant ON attachments(variant_id);\n", "", 1)
	for _, q := range []string{v1, `PRAGMA user_version = 1`,
		`INSERT INTO chats (id, title, created_at, updated_at) VALUES ('c1', 'old', 1, 1)`,
		`INSERT INTO messages (id, chat_id, seq, role, content, created_at) VALUES ('m1', 'c1', 0, 'user', 'hi', 1)`,
		`INSERT INTO attachments (id, message_id, kind, name, mime, size, data) VALUES ('a1', 'm1', 'file', 'notes.txt', 'text/plain', 5, 'hello')`} {
		if _, err := db.Exec(q); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
	}
	db.Close()

	s, err := OpenChatStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	hist, err := s.History("c1", 5)
	if err != nil || len(hist) != 1 || string(hist[0].Attachments[0].Data) != "hello" || hist[0].Attachments[0].isDoc() {
		t.Fatalf("old attachment after upgrade: %+v %v", hist, err)
	}
	if _, err := s.AppendTurn("c1", "now a pdf", "i", []Attachment{{AttachmentMeta: AttachmentMeta{Kind: "file", Name: "a.pdf", Mime: "application/pdf", Extract: "tika"}, Data: []byte("%PDF"), Text: "the text"}}); err != nil {
		t.Fatal(err)
	}
	var v int
	s.db.QueryRow(`PRAGMA user_version`).Scan(&v)
	if v != chatSchemaVersion {
		t.Errorf("user_version = %d", v)
	}
}

func docAtt(name, mime, data, text, by string) map[string]any {
	return map[string]any{"kind": "file", "name": name, "mime": mime, "data": base64.StdEncoding.EncodeToString([]byte(data)), "text": text, "extract": by}
}

func TestChatSendsDocumentTextNotTheFile(t *testing.T) {
	rig := newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if body["stream"] != true {
			titleReply(w, "Doc")
			return
		}
		sse(w, delta("content", "ok"), finalChunk, "[DONE]")
	})
	c := rig.newChat(t, `{}`)
	resp, b := rig.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{
		"content": "What is the total?",
		"attachments": []map[string]any{
			docAtt("Счёт.pdf", "application/pdf", "%PDF-1.7 BINARY-ORIGINAL", "Total\u00a0due: 1,250.00 EUR\n```\nnot a fence\n```", "browser"),
			{"kind": "file", "name": "notes.txt", "data": base64.StdEncoding.EncodeToString([]byte("ctx=65536"))},
		},
	})
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("send: %d %s", resp.StatusCode, b)
	}
	rig.waitReply(t, c.ID)
	rig.mu.Lock()
	sent := rig.bodies[0]
	rig.mu.Unlock()
	msgs := sent["messages"].([]any)
	text := msgs[len(msgs)-1].(map[string]any)["content"].(string)
	if strings.Contains(text, "BINARY-ORIGINAL") {
		t.Error("the original file must not reach the model")
	}
	if !strings.Contains(text, "Attached file Счёт.pdf (text extracted from the document):\n````\nTotal due: 1,250.00 EUR") {
		t.Errorf("document text not inlined in a long enough fence: %q", text)
	}
	if !strings.HasSuffix(text, "What is the total?") || strings.Index(text, "notes.txt") > strings.Index(text, "What is the total?") {
		t.Errorf("files should come before the question: %q", text)
	}

	d := rig.detail(t, c.ID)
	atts := d.Messages[0].Attachments
	if len(atts) != 2 || atts[0].Extract != "browser" || atts[0].TextSize == 0 || atts[0].Size != int64(len("%PDF-1.7 BINARY-ORIGINAL")) || atts[1].Extract != "" {
		t.Fatalf("attachment meta: %+v", atts)
	}
	get := func(q string) (*http.Response, string) {
		resp, b := rig.do(t, "GET", "/api/attachments/"+atts[0].ID+q, nil)
		return resp, string(b)
	}
	r, body := get("")
	if body != "%PDF-1.7 BINARY-ORIGINAL" || r.Header.Get("Content-Type") != "application/octet-stream" ||
		!strings.Contains(r.Header.Get("Content-Disposition"), "filename*=utf-8''%D0%A1%D1%87%D1%91%D1%82.pdf") || r.Header.Get("Content-Security-Policy") == "" {
		t.Errorf("original download: %v %q", r.Header, body)
	}
	r, body = get("?as=text")
	if !strings.HasPrefix(body, "Total due") || !strings.HasPrefix(r.Header.Get("Content-Type"), "text/plain") || !strings.HasPrefix(r.Header.Get("Content-Disposition"), "inline") {
		t.Errorf("text view: %v %q", r.Header, body)
	}
}

func TestDecodeAttachmentsRejectsBadDocuments(t *testing.T) {
	enc := func(s string) string { return base64.StdEncoding.EncodeToString([]byte(s)) }
	text := func(s string) *string { return &s }
	for name, a := range map[string]sendAttachment{
		"unknown reader": {Kind: "file", Name: "a.pdf", Data: enc("x"), Text: text("t"), Extract: "magic"},
		"no text":        {Kind: "file", Name: "a.pdf", Data: enc("x"), Text: text(" \n\u00a0"), Extract: "tika"},
		"bad utf-8":      {Kind: "file", Name: "a.pdf", Data: enc("x"), Text: text("\xff\xfe"), Extract: "tika"},
		"too much text":  {Kind: "file", Name: "a.pdf", Data: enc("x"), Text: text(strings.Repeat("a", maxDocTextBytes+65<<10)), Extract: "tika"},
		"binary as text": {Kind: "file", Name: "a.bin", Data: enc("\xff\xd8\xff")},
	} {
		if _, err := decodeAttachments([]sendAttachment{a}); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	big := enc(strings.Repeat("x", 23<<20))
	var three []sendAttachment
	for i := 0; i < 3; i++ {
		three = append(three, sendAttachment{Kind: "file", Name: "a.pdf", Data: big, Text: text("t"), Extract: "browser"})
	}
	if _, err := decodeAttachments(three); err == nil || !strings.Contains(err.Error(), "add up") {
		t.Errorf("three 23 MB documents should go over the per-message total: %v", err)
	}
}

// TestRealTika runs the client against a real Tika. Start one with
// docker run -p 127.0.0.1:9998:9998 apache/tika:4.2.0, then set LM_TIKA_URL.
func TestRealTika(t *testing.T) {
	base := os.Getenv("LM_TIKA_URL")
	if base == "" {
		t.Skip("set LM_TIKA_URL to test against a real Tika")
	}
	tc := newTikaClient(TikaConf{Enabled: true, URL: base})
	if st := tc.ProbeNow(); !st.OK {
		t.Fatalf("Tika at %s: %+v", base, st)
	}
	var buf bytes.Buffer
	z := zip.NewWriter(&buf)
	for name, body := range map[string]string{
		"[Content_Types].xml": `<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>`,
		"_rels/.rels":         `<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="r1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>`,
		"word/document.xml":   `<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>REAL-TIKA-MARKER Привет</w:t></w:r></w:p></w:body></w:document>`,
	} {
		w, _ := z.Create(name)
		w.Write([]byte(body))
	}
	z.Close()
	res, err := tc.Extract(context.Background(), buf.Bytes(), "t.docx", "")
	if err != nil || !strings.Contains(res.Text, "REAL-TIKA-MARKER Привет") {
		t.Fatalf("docx: %+v %v", res, err)
	}
	_, err = tc.Extract(context.Background(), []byte("PK\x03\x04 not really a zip"), "broken.docx", "")
	var ee *extractError
	if !errors.As(err, &ee) || ee.Reason != "unreadable" {
		t.Errorf("broken docx: %v", err)
	}
	cp1251 := bytes.Repeat([]byte("\xcf\xf0\xe8\xe2\xe5\xf2 \xec\xe8\xf0, \xfd\xf2\xee \xf2\xe5\xea\xf1\xf2 \xe2 \xea\xee\xe4\xe8\xf0\xee\xe2\xea\xe5 windows-1251. "), 3)
	res, err = tc.Extract(context.Background(), cp1251, "old.txt", "text/plain")
	if err != nil || !strings.Contains(res.Text, "Привет мир") {
		t.Errorf("Windows-1251 text: %+v %v", res, err)
	}
}
