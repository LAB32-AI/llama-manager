package main

import (
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	_ "modernc.org/sqlite"
)

// ChatStore keeps the chat history in SQLite. Assistant turns hold one or more
// variants (regenerate adds one); editing a user turn drops everything after
// it, so the stored conversation is always the linear history the UI shows.
type ChatStore struct {
	db   *sql.DB
	path string
}

var errChatNotFound = errors.New("not found")

const chatSchemaVersion = 7

const chatSchema = `
CREATE TABLE IF NOT EXISTS chats (
	id         TEXT PRIMARY KEY,
	title      TEXT NOT NULL,
	instance   TEXT NOT NULL DEFAULT '',
	params     TEXT NOT NULL DEFAULT '{}',
	pinned     INTEGER NOT NULL DEFAULT 0,
	created_at INTEGER NOT NULL,
	updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS chats_updated ON chats(updated_at DESC);
CREATE TABLE IF NOT EXISTS messages (
	id         TEXT PRIMARY KEY,
	chat_id    TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
	seq        INTEGER NOT NULL,
	role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
	content    TEXT NOT NULL DEFAULT '',
	instance   TEXT NOT NULL DEFAULT '',
	vi         INTEGER NOT NULL DEFAULT 0,
	created_at INTEGER NOT NULL,
	UNIQUE (chat_id, seq)
);
CREATE TABLE IF NOT EXISTS variants (
	id           TEXT PRIMARY KEY,
	message_id   TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
	idx          INTEGER NOT NULL,
	content      TEXT NOT NULL DEFAULT '',
	reasoning    TEXT NOT NULL DEFAULT '',
	reasoning_ms INTEGER NOT NULL DEFAULT 0,
	status       TEXT NOT NULL DEFAULT 'done',
	error        TEXT NOT NULL DEFAULT '',
	stats        TEXT NOT NULL DEFAULT '{}',
	created_at   INTEGER NOT NULL,
	UNIQUE (message_id, idx)
);
CREATE TABLE IF NOT EXISTS attachments (
	id         TEXT PRIMARY KEY,
	message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
	kind       TEXT NOT NULL CHECK (kind IN ('image', 'file')),
	name       TEXT NOT NULL,
	mime       TEXT NOT NULL DEFAULT '',
	size       INTEGER NOT NULL,
	data       BLOB NOT NULL,
	text       TEXT,                       -- a document's text or an image's description; NULL for text files
	extract    TEXT NOT NULL DEFAULT '',   -- who read it: tika or browser (documents), reader (images)
	variant_id TEXT NOT NULL DEFAULT ''    -- set for a picture an image model made: the reply version it belongs to
);
CREATE INDEX IF NOT EXISTS attachments_message ON attachments(message_id);
CREATE INDEX IF NOT EXISTS attachments_variant ON attachments(variant_id);
-- The OpenAI-compatible API's state (v4): stored responses, conversations and
-- stored chat completions.
CREATE TABLE IF NOT EXISTS api_responses (
	id         TEXT PRIMARY KEY,
	created_at INTEGER NOT NULL,
	used_at    INTEGER NOT NULL DEFAULT 0,  -- last continued from: a chain in use isn't pruned
	input      TEXT NOT NULL,   -- JSON array: every input item, the history included
	response   TEXT NOT NULL    -- the response object
);
CREATE INDEX IF NOT EXISTS api_responses_created ON api_responses(created_at);
CREATE TABLE IF NOT EXISTS api_conversations (
	id         TEXT PRIMARY KEY,
	created_at INTEGER NOT NULL,
	metadata   TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS api_conversation_items (
	conv_id TEXT NOT NULL REFERENCES api_conversations(id) ON DELETE CASCADE,
	seq     INTEGER NOT NULL,
	id      TEXT NOT NULL,
	item    TEXT NOT NULL,
	PRIMARY KEY (conv_id, seq)
);
CREATE INDEX IF NOT EXISTS api_conversation_items_id ON api_conversation_items(id);
CREATE TABLE IF NOT EXISTS api_chat_completions (
	id         TEXT PRIMARY KEY,
	created_at INTEGER NOT NULL,
	model      TEXT NOT NULL,
	metadata   TEXT NOT NULL DEFAULT '{}',
	messages   TEXT NOT NULL,   -- JSON array: the request's messages
	completion TEXT NOT NULL    -- the chat.completion object
);
CREATE INDEX IF NOT EXISTS api_chat_completions_created ON api_chat_completions(created_at);
-- Each /v1 API key's use per day (v5), kept 90 days.
CREATE TABLE IF NOT EXISTS api_usage (
	key_id        TEXT NOT NULL,    -- keyID: a hash of the key, never the key
	day           TEXT NOT NULL,    -- YYYY-MM-DD, server time
	requests      INTEGER NOT NULL DEFAULT 0,
	input_tokens  INTEGER NOT NULL DEFAULT 0,
	output_tokens INTEGER NOT NULL DEFAULT 0,
	last_used     INTEGER NOT NULL DEFAULT 0,
	PRIMARY KEY (key_id, day)
);
-- The dashboard (v6): request counts per minute (2 days), per hour (90 days)
-- and for all time, each row a JSON statAgg; and the activity feed.
CREATE TABLE IF NOT EXISTS stats_minute (
	t      INTEGER NOT NULL,  -- unix time of the minute
	source TEXT NOT NULL,     -- chat, lan, internet
	route  TEXT NOT NULL,
	model  TEXT NOT NULL,
	agg    TEXT NOT NULL,
	PRIMARY KEY (t, source, route, model)
);
CREATE TABLE IF NOT EXISTS stats_hour (
	t      INTEGER NOT NULL,
	source TEXT NOT NULL,
	route  TEXT NOT NULL,
	model  TEXT NOT NULL,
	agg    TEXT NOT NULL,
	PRIMARY KEY (t, source, route, model)
);
CREATE TABLE IF NOT EXISTS stats_total (
	source TEXT NOT NULL,
	route  TEXT NOT NULL,
	model  TEXT NOT NULL,
	agg    TEXT NOT NULL,
	PRIMARY KEY (source, route, model)
);
CREATE TABLE IF NOT EXISTS stats_meta (
	k TEXT PRIMARY KEY,
	v INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS dash_events (
	id    INTEGER PRIMARY KEY AUTOINCREMENT,
	t     INTEGER NOT NULL,
	kind  TEXT NOT NULL,
	model TEXT NOT NULL DEFAULT '',
	msg   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS dash_events_t ON dash_events(t);
`

type Chat struct {
	ID        string          `json:"id"`
	Title     string          `json:"title"`
	Instance  string          `json:"instance"`
	Params    json.RawMessage `json:"params"`
	Pinned    bool            `json:"pinned"`
	CreatedAt int64           `json:"created_at"`
	UpdatedAt int64           `json:"updated_at"`
	Streaming bool            `json:"streaming,omitempty"`
	Snippet   string          `json:"snippet,omitempty"`
}

type ChatMessage struct {
	ID          string           `json:"id"`
	Seq         int              `json:"seq"`
	Role        string           `json:"role"`
	Content     string           `json:"content"`
	Instance    string           `json:"instance,omitempty"`
	VI          int              `json:"vi"`
	CreatedAt   int64            `json:"created_at"`
	Attachments []AttachmentMeta `json:"attachments,omitempty"`
	Variants    []Variant        `json:"variants,omitempty"`
}

type Variant struct {
	ID          string          `json:"id"`
	Idx         int             `json:"idx"`
	Content     string          `json:"content"`
	Reasoning   string          `json:"reasoning,omitempty"`
	ReasoningMs int64           `json:"reasoning_ms,omitempty"`
	Status      string          `json:"status"`
	Error       string          `json:"error,omitempty"`
	Stats       json.RawMessage `json:"stats,omitempty"`
	CreatedAt   int64           `json:"created_at"`
	// Images are the pictures an image model made for this version.
	Images []AttachmentMeta `json:"images,omitempty"`
}

type AttachmentMeta struct {
	ID   string `json:"id"`
	Kind string `json:"kind"`
	Name string `json:"name"`
	Mime string `json:"mime"`
	Size int64  `json:"size"`
	// Documents (PDF, Word...) keep the original in Data and what the model
	// reads in Text; Extract says who read it. An image the image reader
	// described has its description in Text and Extract "reader".
	Extract  string `json:"extract,omitempty"`
	TextSize int64  `json:"text_size,omitempty"`
}

type Attachment struct {
	AttachmentMeta
	Data []byte
	Text string
}

func (a AttachmentMeta) isDoc() bool { return a.Kind == "file" && a.Extract != "" }

// Variant statuses.
const (
	variantStreaming = "streaming"
	variantDone      = "done"
	variantStopped   = "stopped"
	variantError     = "error"
)

func OpenChatStore(path string) (*ChatStore, error) {
	q := url.Values{}
	for _, p := range []string{"foreign_keys(1)", "journal_mode(WAL)", "busy_timeout(5000)", "synchronous(NORMAL)"} {
		q.Add("_pragma", p)
	}
	db, err := sql.Open("sqlite", "file:"+path+"?"+q.Encode())
	if err != nil {
		return nil, err
	}
	// One connection: SQLite allows a single writer anyway, and serializing
	// here avoids SQLITE_BUSY between the stream writer and API requests.
	db.SetMaxOpenConns(1)
	s := &ChatStore{db: db, path: path}
	if err := s.migrate(); err != nil {
		db.Close()
		return nil, fmt.Errorf("chat history %s: %w", path, err)
	}
	_ = os.Chmod(path, 0o600)
	// A reply still marked streaming was cut off by a crash or restart.
	if _, err := db.Exec(`UPDATE variants SET status = ?, error = ? WHERE status = ?`,
		variantStopped, "interrupted: llama-manager restarted while replying", variantStreaming); err != nil {
		db.Close()
		return nil, err
	}
	return s, nil
}

func (s *ChatStore) migrate() error {
	var v int
	if err := s.db.QueryRow(`PRAGMA user_version`).Scan(&v); err != nil {
		return err
	}
	if v > chatSchemaVersion {
		return fmt.Errorf("database schema v%d is newer than this llama-manager (v%d)", v, chatSchemaVersion)
	}
	if v == 1 {
		if err := s.tx(func(tx *sql.Tx) error {
			for _, q := range []string{`ALTER TABLE attachments ADD COLUMN text TEXT`, `ALTER TABLE attachments ADD COLUMN extract TEXT NOT NULL DEFAULT ''`} {
				if _, err := tx.Exec(q); err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return fmt.Errorf("upgrading to schema v2: %w", err)
		}
	}
	if v == 1 || v == 2 {
		if _, err := s.db.Exec(`ALTER TABLE attachments ADD COLUMN variant_id TEXT NOT NULL DEFAULT ''`); err != nil {
			return fmt.Errorf("upgrading to schema v3: %w", err)
		}
	}
	if v >= 1 && v < 7 {
		if err := s.tx(dropOldDefaultParams); err != nil {
			return fmt.Errorf("upgrading to schema v7: %w", err)
		}
	}
	if _, err := s.db.Exec(chatSchema); err != nil {
		return err
	}
	_, err := s.db.Exec(fmt.Sprintf(`PRAGMA user_version = %d`, chatSchemaVersion))
	return err
}

// oldUIDefaults are the values the chat UI saved into every chat before the
// general chat settings (schema v7): sampling, an empty system prompt, no
// reply limit, a random seed.
var oldUIDefaults = map[string][]any{"temp": {0.7}, "top_p": {0.95}, "min_p": {0.05}, "top_k": {40.0}, "rep": {1.0}, "max": {"", 0.0}, "seed": {""}, "sys": {""}}

// dropOldDefaultParams removes those values from every chat, so chats made
// before v7 follow the general chat settings like new ones; a value the user
// changed stays the chat's own.
func dropOldDefaultParams(tx *sql.Tx) error {
	rows, err := tx.Query(`SELECT id, params FROM chats`)
	if err != nil {
		return err
	}
	upd := map[string]string{}
	for rows.Next() {
		var id, raw string
		if err := rows.Scan(&id, &raw); err != nil {
			rows.Close()
			return err
		}
		var m map[string]any
		d := json.NewDecoder(strings.NewReader(raw))
		d.UseNumber()
		if d.Decode(&m) != nil || m == nil {
			continue
		}
		dropped := false
		for k, defs := range oldUIDefaults {
			v, ok := m[k]
			if !ok {
				continue
			}
			if n, isNum := v.(json.Number); isNum {
				f, err := n.Float64()
				if err != nil {
					continue
				}
				v = f
			}
			if slices.Contains(defs, v) {
				delete(m, k)
				dropped = true
			}
		}
		if dropped {
			b, err := json.Marshal(m)
			if err != nil {
				continue
			}
			upd[id] = string(b)
		}
	}
	if err := rows.Close(); err != nil {
		return err
	}
	if err := rows.Err(); err != nil {
		return err
	}
	for id, p := range upd {
		if _, err := tx.Exec(`UPDATE chats SET params = ? WHERE id = ?`, p, id); err != nil {
			return err
		}
	}
	return nil
}

func (s *ChatStore) Close() error { return s.db.Close() }

func newID() string {
	b := make([]byte, 9)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func nowMs() int64 { return time.Now().UnixMilli() }

const defaultChatTitle = "New chat"

// autoTitle is the placeholder title from the first message, used until (or
// instead of) a model-written one.
func autoTitle(content string) string {
	words := strings.Fields(content)
	if len(words) == 0 {
		return defaultChatTitle
	}
	if len(words) > 7 {
		words = words[:7]
	}
	t := strings.TrimRight(strings.Join(words, " "), ".,:;!?")
	return truncateRunes(t, 80)
}

func truncateRunes(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	r := []rune(s)
	return strings.TrimSpace(string(r[:n-1])) + "…"
}

func likePattern(q string) string {
	r := strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`)
	return "%" + r.Replace(q) + "%"
}

func scanChat(row interface{ Scan(...any) error }) (Chat, error) {
	var c Chat
	var params string
	var pinned int
	err := row.Scan(&c.ID, &c.Title, &c.Instance, &params, &pinned, &c.CreatedAt, &c.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return c, errChatNotFound
	}
	c.Params = json.RawMessage(params)
	c.Pinned = pinned != 0
	return c, err
}

const chatCols = `id, title, instance, params, pinned, created_at, updated_at`

// ListChats returns chats, newest first. A non-empty q matches titles and
// message text; each hit carries a short snippet around the first match.
func (s *ChatStore) ListChats(q string, limit int) ([]Chat, error) {
	if limit <= 0 || limit > 1000 {
		limit = 500
	}
	q = strings.TrimSpace(q)
	var rows *sql.Rows
	var err error
	if q == "" {
		rows, err = s.db.Query(`SELECT `+chatCols+`, '' FROM chats ORDER BY updated_at DESC, rowid DESC LIMIT ?`, limit)
	} else {
		rows, err = s.db.Query(`SELECT `+chatCols+`, COALESCE(
				(SELECT m.content FROM messages m WHERE m.chat_id = c.id AND m.content LIKE ?1 ESCAPE '\' ORDER BY m.seq LIMIT 1),
				(SELECT v.content FROM variants v JOIN messages m ON m.id = v.message_id
				  WHERE m.chat_id = c.id AND v.content LIKE ?1 ESCAPE '\' ORDER BY m.seq LIMIT 1),
				'')
			FROM chats c
			WHERE c.title LIKE ?1 ESCAPE '\'
			   OR EXISTS (SELECT 1 FROM messages m WHERE m.chat_id = c.id AND m.content LIKE ?1 ESCAPE '\')
			   OR EXISTS (SELECT 1 FROM variants v JOIN messages m ON m.id = v.message_id
			              WHERE m.chat_id = c.id AND v.content LIKE ?1 ESCAPE '\')
			ORDER BY c.updated_at DESC, c.rowid DESC LIMIT ?2`, likePattern(q), limit)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Chat{}
	for rows.Next() {
		var c Chat
		var params, hit string
		var pinned int
		if err := rows.Scan(&c.ID, &c.Title, &c.Instance, &params, &pinned, &c.CreatedAt, &c.UpdatedAt, &hit); err != nil {
			return nil, err
		}
		c.Params = json.RawMessage(params)
		c.Pinned = pinned != 0
		if q != "" {
			c.Snippet = snippetAround(hit, q, 90)
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// snippetAround returns up to n runes of text centred on the first
// case-insensitive match of q, on one line.
func snippetAround(text, q string, n int) string {
	text = strings.Join(strings.Fields(text), " ")
	if text == "" {
		return ""
	}
	r := []rune(text)
	lower := []rune(strings.ToLower(text))
	needle := []rune(strings.ToLower(q))
	at := -1
	for i := 0; i+len(needle) <= len(lower); i++ {
		if string(lower[i:i+len(needle)]) == string(needle) {
			at = i
			break
		}
	}
	if at < 0 {
		return truncateRunes(text, n)
	}
	start := max(0, at-n/4)
	end := min(len(r), start+n)
	out := string(r[start:end])
	if start > 0 {
		out = "…" + out
	}
	if end < len(r) {
		out += "…"
	}
	return out
}

func (s *ChatStore) CreateChat(title, instance string, params json.RawMessage) (Chat, error) {
	if strings.TrimSpace(title) == "" {
		title = defaultChatTitle
	}
	if len(params) == 0 || !json.Valid(params) {
		params = json.RawMessage("{}")
	}
	now := nowMs()
	c := Chat{ID: newID(), Title: truncateRunes(strings.TrimSpace(title), 120), Instance: instance, Params: params, CreatedAt: now, UpdatedAt: now}
	_, err := s.db.Exec(`INSERT INTO chats (`+chatCols+`) VALUES (?, ?, ?, ?, 0, ?, ?)`,
		c.ID, c.Title, c.Instance, string(c.Params), c.CreatedAt, c.UpdatedAt)
	return c, err
}

func (s *ChatStore) GetChat(id string) (Chat, error) {
	return scanChat(s.db.QueryRow(`SELECT `+chatCols+` FROM chats WHERE id = ?`, id))
}

type ChatPatch struct {
	Title    *string          `json:"title"`
	Pinned   *bool            `json:"pinned"`
	Instance *string          `json:"instance"`
	Params   *json.RawMessage `json:"params"`
}

func (s *ChatStore) UpdateChat(id string, p ChatPatch) (Chat, error) {
	c, err := s.GetChat(id)
	if err != nil {
		return c, err
	}
	if p.Title != nil {
		t := strings.TrimSpace(*p.Title)
		if t == "" {
			return c, errors.New("title must not be empty")
		}
		c.Title = truncateRunes(t, 120)
	}
	if p.Pinned != nil {
		c.Pinned = *p.Pinned
	}
	if p.Instance != nil {
		c.Instance = *p.Instance
	}
	if p.Params != nil {
		if !json.Valid(*p.Params) || len(*p.Params) > 64<<10 {
			return c, errors.New("params must be a JSON object under 64 KiB")
		}
		c.Params = *p.Params
	}
	pinned := 0
	if c.Pinned {
		pinned = 1
	}
	_, err = s.db.Exec(`UPDATE chats SET title = ?, pinned = ?, instance = ?, params = ? WHERE id = ?`,
		c.Title, pinned, c.Instance, string(c.Params), id)
	return c, err
}

func (s *ChatStore) DeleteChat(id string) error {
	res, err := s.db.Exec(`DELETE FROM chats WHERE id = ?`, id)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return errChatNotFound
	}
	return nil
}

// Messages returns the conversation in order, with every variant and the
// attachment metadata (not the bytes).
func (s *ChatStore) Messages(chatID string) ([]ChatMessage, error) {
	rows, err := s.db.Query(`SELECT id, seq, role, content, instance, vi, created_at FROM messages WHERE chat_id = ? ORDER BY seq`, chatID)
	if err != nil {
		return nil, err
	}
	var msgs []ChatMessage
	byID := map[string]int{}
	for rows.Next() {
		var m ChatMessage
		if err := rows.Scan(&m.ID, &m.Seq, &m.Role, &m.Content, &m.Instance, &m.VI, &m.CreatedAt); err != nil {
			rows.Close()
			return nil, err
		}
		byID[m.ID] = len(msgs)
		msgs = append(msgs, m)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	vrows, err := s.db.Query(`SELECT v.message_id, v.id, v.idx, v.content, v.reasoning, v.reasoning_ms, v.status, v.error, v.stats, v.created_at
		FROM variants v JOIN messages m ON m.id = v.message_id WHERE m.chat_id = ? ORDER BY m.seq, v.idx`, chatID)
	if err != nil {
		return nil, err
	}
	type vpos struct{ msg, v int }
	byVariant := map[string]vpos{}
	for vrows.Next() {
		var mid, stats string
		var v Variant
		if err := vrows.Scan(&mid, &v.ID, &v.Idx, &v.Content, &v.Reasoning, &v.ReasoningMs, &v.Status, &v.Error, &stats, &v.CreatedAt); err != nil {
			vrows.Close()
			return nil, err
		}
		v.Stats = json.RawMessage(stats)
		if i, ok := byID[mid]; ok {
			byVariant[v.ID] = vpos{i, len(msgs[i].Variants)}
			msgs[i].Variants = append(msgs[i].Variants, v)
		}
	}
	vrows.Close()
	if err := vrows.Err(); err != nil {
		return nil, err
	}
	arows, err := s.db.Query(`SELECT a.message_id, a.variant_id, a.id, a.kind, a.name, a.mime, a.size, a.extract, COALESCE(length(CAST(a.text AS BLOB)), 0)
		FROM attachments a JOIN messages m ON m.id = a.message_id WHERE m.chat_id = ? ORDER BY m.seq, a.rowid`, chatID)
	if err != nil {
		return nil, err
	}
	defer arows.Close()
	for arows.Next() {
		var mid, vid string
		var a AttachmentMeta
		if err := arows.Scan(&mid, &vid, &a.ID, &a.Kind, &a.Name, &a.Mime, &a.Size, &a.Extract, &a.TextSize); err != nil {
			return nil, err
		}
		if vid != "" {
			if p, ok := byVariant[vid]; ok {
				v := &msgs[p.msg].Variants[p.v]
				v.Images = append(v.Images, a)
			}
			continue
		}
		if i, ok := byID[mid]; ok {
			msgs[i].Attachments = append(msgs[i].Attachments, a)
		}
	}
	if msgs == nil {
		msgs = []ChatMessage{}
	}
	return msgs, arows.Err()
}

// turn is what a generation needs to know about the reply it fills in.
type turn struct {
	ChatID    string
	MessageID string
	VariantID string
	Seq       int // the assistant message's position
}

func (s *ChatStore) tx(fn func(*sql.Tx) error) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	if err := fn(tx); err != nil {
		tx.Rollback()
		return err
	}
	return tx.Commit()
}

func insertAssistant(tx *sql.Tx, chatID string, seq int, instance string, now int64) (turn, error) {
	t := turn{ChatID: chatID, MessageID: newID(), VariantID: newID(), Seq: seq}
	if _, err := tx.Exec(`INSERT INTO messages (id, chat_id, seq, role, instance, vi, created_at) VALUES (?, ?, ?, 'assistant', ?, 0, ?)`,
		t.MessageID, chatID, seq, instance, now); err != nil {
		return t, err
	}
	_, err := tx.Exec(`INSERT INTO variants (id, message_id, idx, status, created_at) VALUES (?, ?, 0, ?, ?)`,
		t.VariantID, t.MessageID, variantStreaming, now)
	return t, err
}

func touchChat(tx *sql.Tx, chatID, instance string, now int64) error {
	res, err := tx.Exec(`UPDATE chats SET updated_at = ?, instance = ? WHERE id = ?`, now, instance, chatID)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return errChatNotFound
	}
	return nil
}

// AppendTurn stores a user message (with attachments) and an empty assistant
// reply for the generation to fill. The first message also names the chat.
func (s *ChatStore) AppendTurn(chatID, content, instance string, atts []Attachment) (turn, error) {
	var t turn
	err := s.tx(func(tx *sql.Tx) error {
		now := nowMs()
		if err := touchChat(tx, chatID, instance, now); err != nil {
			return err
		}
		var seq int
		if err := tx.QueryRow(`SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE chat_id = ?`, chatID).Scan(&seq); err != nil {
			return err
		}
		uid := newID()
		if _, err := tx.Exec(`INSERT INTO messages (id, chat_id, seq, role, content, created_at) VALUES (?, ?, ?, 'user', ?, ?)`,
			uid, chatID, seq, content, now); err != nil {
			return err
		}
		for _, a := range atts {
			text := sql.NullString{String: a.Text, Valid: a.Extract != ""}
			if _, err := tx.Exec(`INSERT INTO attachments (id, message_id, kind, name, mime, size, data, text, extract) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
				newID(), uid, a.Kind, a.Name, a.Mime, len(a.Data), a.Data, text, a.Extract); err != nil {
				return err
			}
		}
		if seq == 0 {
			if _, err := tx.Exec(`UPDATE chats SET title = ? WHERE id = ? AND title = ?`, autoTitle(content), chatID, defaultChatTitle); err != nil {
				return err
			}
		}
		var err error
		t, err = insertAssistant(tx, chatID, seq+1, instance, now)
		return err
	})
	return t, err
}

// AddVariant starts another version of an assistant reply. Later turns
// answered the old version, so they are dropped.
func (s *ChatStore) AddVariant(chatID, msgID, instance string) (turn, error) {
	var t turn
	err := s.tx(func(tx *sql.Tx) error {
		var role string
		if err := tx.QueryRow(`SELECT seq, role FROM messages WHERE id = ? AND chat_id = ?`, msgID, chatID).Scan(&t.Seq, &role); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return errChatNotFound
			}
			return err
		}
		if role != "assistant" {
			return errors.New("only replies can be regenerated")
		}
		now := nowMs()
		if _, err := tx.Exec(`DELETE FROM messages WHERE chat_id = ? AND seq > ?`, chatID, t.Seq); err != nil {
			return err
		}
		var idx int
		if err := tx.QueryRow(`SELECT COALESCE(MAX(idx), -1) + 1 FROM variants WHERE message_id = ?`, msgID).Scan(&idx); err != nil {
			return err
		}
		t.ChatID, t.MessageID, t.VariantID = chatID, msgID, newID()
		if _, err := tx.Exec(`INSERT INTO variants (id, message_id, idx, status, created_at) VALUES (?, ?, ?, ?, ?)`,
			t.VariantID, msgID, idx, variantStreaming, now); err != nil {
			return err
		}
		if _, err := tx.Exec(`UPDATE messages SET vi = ?, instance = ? WHERE id = ?`, idx, instance, msgID); err != nil {
			return err
		}
		return touchChat(tx, chatID, instance, now)
	})
	return t, err
}

// EditUserMessage replaces a user message's text, drops everything after it
// and opens a fresh reply.
func (s *ChatStore) EditUserMessage(chatID, msgID, content, instance string) (turn, error) {
	var t turn
	err := s.tx(func(tx *sql.Tx) error {
		var seq int
		var role string
		if err := tx.QueryRow(`SELECT seq, role FROM messages WHERE id = ? AND chat_id = ?`, msgID, chatID).Scan(&seq, &role); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return errChatNotFound
			}
			return err
		}
		if role != "user" {
			return errors.New("only your own messages can be edited")
		}
		now := nowMs()
		if _, err := tx.Exec(`UPDATE messages SET content = ? WHERE id = ?`, content, msgID); err != nil {
			return err
		}
		if _, err := tx.Exec(`DELETE FROM messages WHERE chat_id = ? AND seq > ?`, chatID, seq); err != nil {
			return err
		}
		if err := touchChat(tx, chatID, instance, now); err != nil {
			return err
		}
		var err error
		t, err = insertAssistant(tx, chatID, seq+1, instance, now)
		return err
	})
	return t, err
}

func (s *ChatStore) SelectVariant(chatID, msgID string, vi int) error {
	res, err := s.db.Exec(`UPDATE messages SET vi = ? WHERE id = ? AND chat_id = ? AND role = 'assistant'
		AND EXISTS (SELECT 1 FROM variants WHERE message_id = ? AND idx = ?)`, vi, msgID, chatID, msgID, vi)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return errChatNotFound
	}
	return nil
}

// Reply returns one version of an assistant message: variantID, or the
// selected one when it is "".
func (s *ChatStore) Reply(chatID, msgID, variantID string) (Variant, error) {
	var v Variant
	err := s.db.QueryRow(`SELECT v.id, v.idx, v.content, v.status FROM variants v JOIN messages m ON m.id = v.message_id
		WHERE m.chat_id = ? AND m.id = ? AND m.role = 'assistant' AND (v.id = ? OR (? = '' AND v.idx = m.vi))`,
		chatID, msgID, variantID, variantID).Scan(&v.ID, &v.Idx, &v.Content, &v.Status)
	if errors.Is(err, sql.ErrNoRows) {
		return v, errChatNotFound
	}
	return v, err
}

func (s *ChatStore) SaveVariant(v Variant) error {
	stats := string(v.Stats)
	if stats == "" {
		stats = "{}"
	}
	_, err := s.db.Exec(`UPDATE variants SET content = ?, reasoning = ?, reasoning_ms = ?, status = ?, error = ?, stats = ? WHERE id = ?`,
		v.Content, v.Reasoning, v.ReasoningMs, v.Status, v.Error, stats, v.ID)
	return err
}

// historyTurn is one earlier message as it is sent back to the model.
type historyTurn struct {
	Role        string
	Content     string
	Attachments []Attachment
}

// History returns the messages before position seq: user text with its
// attachments, and the selected version of each reply. Failed or empty
// replies are left out so the model never sees a blank assistant turn; a
// picture an image model made is a short note, since a chat model can't be
// sent images in a reply.
func (s *ChatStore) History(chatID string, seq int) ([]historyTurn, error) {
	rows, err := s.db.Query(`SELECT m.id, m.role, m.content, COALESCE(v.content, ''), COALESCE(v.status, ''),
			EXISTS (SELECT 1 FROM attachments a WHERE a.variant_id = v.id)
		FROM messages m LEFT JOIN variants v ON v.message_id = m.id AND v.idx = m.vi
		WHERE m.chat_id = ? AND m.seq < ? ORDER BY m.seq`, chatID, seq)
	if err != nil {
		return nil, err
	}
	var out []historyTurn
	var ids []string
	for rows.Next() {
		var id, role, content, vcontent, status string
		var pictured bool
		if err := rows.Scan(&id, &role, &content, &vcontent, &status, &pictured); err != nil {
			rows.Close()
			return nil, err
		}
		if role == "assistant" {
			if strings.TrimSpace(vcontent) == "" && pictured && status != variantError {
				vcontent = imageReplyNote
			}
			if status == variantError || strings.TrimSpace(vcontent) == "" {
				continue
			}
			content = vcontent
		}
		out = append(out, historyTurn{Role: role, Content: content})
		ids = append(ids, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	for i, id := range ids {
		if out[i].Role != "user" {
			continue
		}
		// A document's original stays on disk: the model only reads its text.
		arows, err := s.db.Query(`SELECT id, kind, name, mime, size, extract, CASE WHEN kind = 'image' OR extract = '' THEN data END, COALESCE(text, '')
			FROM attachments WHERE message_id = ? ORDER BY rowid`, id)
		if err != nil {
			return nil, err
		}
		for arows.Next() {
			var a Attachment
			if err := arows.Scan(&a.ID, &a.Kind, &a.Name, &a.Mime, &a.Size, &a.Extract, &a.Data, &a.Text); err != nil {
				arows.Close()
				return nil, err
			}
			out[i].Attachments = append(out[i].Attachments, a)
		}
		arows.Close()
	}
	return out, nil
}

// AddReplyImage stores a picture an image model made for reply version t.
func (s *ChatStore) AddReplyImage(t turn, name, mime string, data []byte) (AttachmentMeta, error) {
	a := AttachmentMeta{ID: newID(), Kind: "image", Name: name, Mime: mime, Size: int64(len(data))}
	_, err := s.db.Exec(`INSERT INTO attachments (id, message_id, variant_id, kind, name, mime, size, data) VALUES (?, ?, ?, 'image', ?, ?, ?, ?)`,
		a.ID, t.MessageID, t.VariantID, a.Name, a.Mime, a.Size, data)
	return a, err
}

// SetImageText saves the image reader's description of an image.
func (s *ChatStore) SetImageText(id, text string) error {
	_, err := s.db.Exec(`UPDATE attachments SET text = ?, extract = ? WHERE id = ? AND kind = 'image'`, text, extractReader, id)
	return err
}

func (s *ChatStore) GetAttachment(id string) (Attachment, error) {
	var a Attachment
	err := s.db.QueryRow(`SELECT id, kind, name, mime, size, extract, data, COALESCE(text, '') FROM attachments WHERE id = ?`, id).
		Scan(&a.ID, &a.Kind, &a.Name, &a.Mime, &a.Size, &a.Extract, &a.Data, &a.Text)
	if errors.Is(err, sql.ErrNoRows) {
		return a, errChatNotFound
	}
	return a, err
}

// FirstExchange returns the first user message and reply, for naming a chat.
func (s *ChatStore) FirstExchange(chatID string) (user, reply string, err error) {
	err = s.db.QueryRow(`SELECT COALESCE((SELECT content FROM messages WHERE chat_id = ?1 AND seq = 0), ''),
		COALESCE((SELECT v.content FROM messages m JOIN variants v ON v.message_id = m.id AND v.idx = m.vi WHERE m.chat_id = ?1 AND m.seq = 1), '')`, chatID).
		Scan(&user, &reply)
	return
}
