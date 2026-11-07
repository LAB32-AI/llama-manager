package main

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

func newTestStore(t *testing.T) (*ChatStore, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "chats.db")
	s, err := OpenChatStore(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s, path
}

func finishVariant(t *testing.T, s *ChatStore, id, content string) {
	t.Helper()
	if err := s.SaveVariant(Variant{ID: id, Content: content, Status: variantDone}); err != nil {
		t.Fatal(err)
	}
}

func TestChatStoreTurnsVariantsAndEdits(t *testing.T) {
	s, _ := newTestStore(t)
	c, err := s.CreateChat("", "inst", nil)
	if err != nil {
		t.Fatal(err)
	}
	t1, err := s.AppendTurn(c.ID, "How do I size a KV cache for 64k context?", "inst", nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := s.GetChat(c.ID); got.Title != "How do I size a KV cache" {
		t.Errorf("first message should name the chat, got %q", got.Title)
	}
	finishVariant(t, s, t1.VariantID, "first answer")
	t2, _ := s.AppendTurn(c.ID, "and at 128k?", "inst", nil)
	finishVariant(t, s, t2.VariantID, "second answer")

	// Regenerating the first reply drops the turns that answered it.
	r, err := s.AddVariant(c.ID, t1.MessageID, "other")
	if err != nil {
		t.Fatal(err)
	}
	finishVariant(t, s, r.VariantID, "first answer, take two")
	msgs, _ := s.Messages(c.ID)
	if len(msgs) != 2 || len(msgs[1].Variants) != 2 || msgs[1].VI != 1 || msgs[1].Instance != "other" {
		t.Fatalf("after regenerate: %+v", msgs)
	}
	if err := s.SelectVariant(c.ID, t1.MessageID, 0); err != nil {
		t.Fatal(err)
	}
	if err := s.SelectVariant(c.ID, t1.MessageID, 5); err == nil {
		t.Error("selecting a missing version should fail")
	}
	hist, _ := s.History(c.ID, 2)
	if len(hist) != 2 || hist[1].Content != "first answer" {
		t.Errorf("history should send the selected version: %+v", hist)
	}

	// Editing the user message replaces it and starts a fresh reply.
	e, err := s.EditUserMessage(c.ID, msgs[0].ID, "edited question", "inst")
	if err != nil {
		t.Fatal(err)
	}
	msgs, _ = s.Messages(c.ID)
	if len(msgs) != 2 || msgs[0].Content != "edited question" || msgs[1].ID != e.MessageID || msgs[1].Variants[0].Status != variantStreaming {
		t.Fatalf("after edit: %+v", msgs)
	}
	if _, err := s.EditUserMessage(c.ID, e.MessageID, "x", "inst"); err == nil {
		t.Error("editing a reply as if it were a user message should fail")
	}
}

func TestChatStoreHistorySkipsFailedReplies(t *testing.T) {
	s, _ := newTestStore(t)
	c, _ := s.CreateChat("t", "i", nil)
	t1, _ := s.AppendTurn(c.ID, "q1", "i", []Attachment{{AttachmentMeta: AttachmentMeta{Kind: "file", Name: "a.txt", Mime: "text/plain"}, Data: []byte("hello")}})
	if err := s.SaveVariant(Variant{ID: t1.VariantID, Status: variantError, Error: "boom"}); err != nil {
		t.Fatal(err)
	}
	t2, _ := s.AppendTurn(c.ID, "q2", "i", nil)
	hist, err := s.History(c.ID, t2.Seq)
	if err != nil {
		t.Fatal(err)
	}
	if len(hist) != 2 || hist[0].Role != "user" || hist[1].Role != "user" {
		t.Fatalf("a failed reply must not be sent back as an empty assistant turn: %+v", hist)
	}
	if len(hist[0].Attachments) != 1 || string(hist[0].Attachments[0].Data) != "hello" {
		t.Errorf("attachments should come back with their bytes: %+v", hist[0].Attachments)
	}
}

func TestChatStoreSearch(t *testing.T) {
	s, _ := newTestStore(t)
	a, _ := s.CreateChat("Sourdough maths", "i", nil)
	b, _ := s.CreateChat("Go backoff", "i", nil)
	tb, _ := s.AppendTurn(b.ID, "retry with exponential backoff please", "i", nil)
	finishVariant(t, s, tb.VariantID, "Use 100% of the budget, then cap the delay")
	_ = a

	for q, want := range map[string]string{"sourdough": a.ID, "EXPONENTIAL": b.ID, "100%": b.ID} {
		got, err := s.ListChats(q, 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 1 || got[0].ID != want {
			t.Errorf("search %q: got %+v", q, got)
		}
	}
	if got, _ := s.ListChats("10_%", 0); len(got) != 0 {
		t.Errorf("LIKE wildcards in the query must match literally, got %+v", got)
	}
	got, _ := s.ListChats("cap the delay", 0)
	if len(got) != 1 || !strings.Contains(got[0].Snippet, "cap the delay") {
		t.Errorf("snippet should show the match: %+v", got)
	}
	all, _ := s.ListChats("", 0)
	if len(all) != 2 || all[0].ID != b.ID {
		t.Errorf("list should be newest first: %+v", all)
	}
}

func TestChatStoreDeleteCascades(t *testing.T) {
	s, _ := newTestStore(t)
	c, _ := s.CreateChat("t", "i", nil)
	tr, _ := s.AppendTurn(c.ID, "q", "i", []Attachment{{AttachmentMeta: AttachmentMeta{Kind: "image", Name: "a.png", Mime: "image/png"}, Data: []byte{1, 2}}})
	msgs, _ := s.Messages(c.ID)
	attID := msgs[0].Attachments[0].ID
	if err := s.DeleteChat(c.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := s.GetAttachment(attID); err != errChatNotFound {
		t.Errorf("attachment should be deleted with its chat, got %v", err)
	}
	var n int
	s.db.QueryRow(`SELECT COUNT(*) FROM variants WHERE id = ?`, tr.VariantID).Scan(&n)
	if n != 0 {
		t.Error("variants should be deleted with their chat")
	}
	if err := s.DeleteChat(c.ID); err != errChatNotFound {
		t.Errorf("deleting twice: %v", err)
	}
}

func TestChatStoreMarksInterruptedRepliesOnOpen(t *testing.T) {
	s, path := newTestStore(t)
	c, _ := s.CreateChat("t", "i", nil)
	tr, _ := s.AppendTurn(c.ID, "q", "i", nil)
	if err := s.SaveVariant(Variant{ID: tr.VariantID, Content: "half an ans", Status: variantStreaming}); err != nil {
		t.Fatal(err)
	}
	s.Close()
	s2, err := OpenChatStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()
	msgs, _ := s2.Messages(c.ID)
	v := msgs[1].Variants[0]
	if v.Status != variantStopped || v.Content != "half an ans" || !strings.Contains(v.Error, "restarted") {
		t.Errorf("a reply cut off by a restart should be kept and marked stopped: %+v", v)
	}
}

func TestChatStoreRejectsBadParams(t *testing.T) {
	s, _ := newTestStore(t)
	c, _ := s.CreateChat("t", "i", json.RawMessage(`{"temp":0.5}`))
	bad := json.RawMessage(`{not json`)
	if _, err := s.UpdateChat(c.ID, ChatPatch{Params: &bad}); err == nil {
		t.Error("invalid params JSON should be rejected")
	}
	empty := ""
	if _, err := s.UpdateChat(c.ID, ChatPatch{Title: &empty}); err == nil {
		t.Error("an empty title should be rejected")
	}
}

func TestThinkSplitter(t *testing.T) {
	var s thinkSplitter
	var content, reasoning string
	for _, chunk := range []string{"\n<th", "ink>plan the ", "answer</thi", "nk>\n\nHello", " world"} {
		c, r := s.feed(chunk)
		content += c
		reasoning += r
	}
	c, r := s.flush()
	content, reasoning = content+c, reasoning+r
	if reasoning != "plan the answer" || content != "Hello world" {
		t.Errorf("reasoning %q, content %q", reasoning, content)
	}

	var plain thinkSplitter
	c1, _ := plain.feed("No tags <think> here")
	if c1 != "No tags <think> here" {
		t.Errorf("text that doesn't open with <think> is plain content, got %q", c1)
	}
}

func TestParseChatParams(t *testing.T) {
	p := parseChatParams(json.RawMessage(`{"temp":0.2,"top_p":"0.9","top_k":40,"max":"","seed":"","think":false,"sys":"  Be brief.  ","rep":1.1}`), ChatConf{})
	if p.System != "Be brief." || p.Think {
		t.Errorf("system/think: %+v", p)
	}
	if p.opts["temperature"] != 0.2 || p.opts["top_p"] != 0.9 || p.opts["top_k"] != 40 || p.opts["repeat_penalty"] != 1.1 {
		t.Errorf("sampling options: %+v", p.opts)
	}
	if _, ok := p.opts["max_tokens"]; ok {
		t.Error("an empty max tokens field means no limit")
	}
	if _, ok := p.opts["seed"]; ok {
		t.Error("an empty seed means random")
	}
	if !parseChatParams(nil, ChatConf{}).Think {
		t.Error("thinking defaults to on")
	}
}

func TestCleanTitle(t *testing.T) {
	for in, want := range map[string]string{
		`"KV Cache Sizing on MI50"`:              "KV Cache Sizing on MI50",
		"<think>hmm</think>\nTitle: Go Backoff.": "Go Backoff",
		"**Sourdough Hydration**":                "Sourdough Hydration",
		"f16_q8_0_cache_table_go_code":           "f16 q8 0 cache table go code",
		"28GB":                                   "",
		"":                                       "",
		"one two three four five six seven eight nine ten eleven twelve": "",
	} {
		if got := cleanTitle(in); got != want {
			t.Errorf("cleanTitle(%q) = %q, want %q", in, got, want)
		}
	}
}
