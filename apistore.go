package main

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"time"
)

// apiRetention is how long stored responses and chat completions are kept,
// as at OpenAI. Conversations are kept until deleted.
const apiRetention = 30 * 24 * time.Hour

// apiStore keeps the OpenAI API's state in the chat database: responses (for
// previous_response_id and retrieval), conversations and stored chat
// completions.
type apiStore struct{ db *sql.DB }

var errAPINotFound = errors.New("not found")

// apiStore is nil when the chat database couldn't be opened.
func (ws *WebServer) apiStore() *apiStore {
	if ws.chats == nil || ws.chats.store == nil {
		return nil
	}
	return &apiStore{db: ws.chats.store.db}
}

// needStore answers 503 when there is no database to keep API state in.
func (ws *WebServer) needStore(w http.ResponseWriter) *apiStore {
	s := ws.apiStore()
	if s == nil {
		writeAPIError(w, http.StatusServiceUnavailable, "", "", "llama-manager couldn't open its database (chats.db), so it can't keep responses, conversations or stored completions; see its log")
	}
	return s
}

func toJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func fromJSON[T any](s string) T {
	var v T
	_ = json.Unmarshal([]byte(s), &v)
	return v
}

func (s *apiStore) prune(now time.Time) {
	cut := now.Add(-apiRetention).Unix()
	_, _ = s.db.Exec(`DELETE FROM api_responses WHERE MAX(created_at, used_at) < ?`, cut)
	_, _ = s.db.Exec(`DELETE FROM api_chat_completions WHERE created_at < ?`, cut)
}

func (s *apiStore) putResponse(id string, created int64, input []map[string]any, resp map[string]any) error {
	_, err := s.db.Exec(`INSERT INTO api_responses (id, created_at, input, response) VALUES (?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET input = excluded.input, response = excluded.response`, id, created, toJSON(input), toJSON(resp))
	return err
}

func (s *apiStore) getResponse(id string) ([]map[string]any, map[string]any, error) {
	var in, out string
	err := s.db.QueryRow(`SELECT input, response FROM api_responses WHERE id = ?`, id).Scan(&in, &out)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil, errAPINotFound
	}
	if err != nil {
		return nil, nil, err
	}
	return fromJSON[[]map[string]any](in), fromJSON[map[string]any](out), nil
}

func (s *apiStore) deleteResponse(id string) error {
	res, err := s.db.Exec(`DELETE FROM api_responses WHERE id = ?`, id)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return errAPINotFound
	}
	return nil
}

// findItem looks up an item by id for an item_reference: in stored
// responses' input and output, then in conversations.
func (s *apiStore) findItem(id string) (map[string]any, error) {
	var raw string
	err := s.db.QueryRow(`SELECT value FROM api_responses, json_each(api_responses.input) WHERE json_extract(value, '$.id') = ?
		UNION ALL SELECT value FROM api_responses, json_each(json_extract(api_responses.response, '$.output')) WHERE json_extract(value, '$.id') = ?
		UNION ALL SELECT item FROM api_conversation_items WHERE id = ? LIMIT 1`, id, id, id).Scan(&raw)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, errAPINotFound
	}
	if err != nil {
		return nil, err
	}
	return fromJSON[map[string]any](raw), nil
}

func conversationObject(id string, created int64, metadata string) map[string]any {
	md := fromJSON[map[string]any](metadata)
	if md == nil {
		md = map[string]any{}
	}
	return map[string]any{"id": id, "object": "conversation", "created_at": created, "metadata": md}
}

func (s *apiStore) createConversation(id string, created int64, metadata map[string]any, items []map[string]any) (map[string]any, error) {
	if metadata == nil {
		metadata = map[string]any{}
	}
	tx, err := s.db.Begin()
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`INSERT INTO api_conversations (id, created_at, metadata) VALUES (?, ?, ?)`, id, created, toJSON(metadata)); err != nil {
		return nil, err
	}
	for i, it := range items {
		if _, err := tx.Exec(`INSERT INTO api_conversation_items (conv_id, seq, id, item) VALUES (?, ?, ?, ?)`, id, i, it["id"], toJSON(it)); err != nil {
			return nil, err
		}
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return conversationObject(id, created, toJSON(metadata)), nil
}

func (s *apiStore) getConversation(id string) (map[string]any, error) {
	var created int64
	var md string
	err := s.db.QueryRow(`SELECT created_at, metadata FROM api_conversations WHERE id = ?`, id).Scan(&created, &md)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, errAPINotFound
	}
	if err != nil {
		return nil, err
	}
	return conversationObject(id, created, md), nil
}

func (s *apiStore) updateConversation(id string, metadata map[string]any) (map[string]any, error) {
	if metadata == nil {
		metadata = map[string]any{}
	}
	res, err := s.db.Exec(`UPDATE api_conversations SET metadata = ? WHERE id = ?`, toJSON(metadata), id)
	if err != nil {
		return nil, err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return nil, errAPINotFound
	}
	return s.getConversation(id)
}

func (s *apiStore) deleteConversation(id string) error {
	res, err := s.db.Exec(`DELETE FROM api_conversations WHERE id = ?`, id)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return errAPINotFound
	}
	return nil
}

func (s *apiStore) addConversationItems(id string, items []map[string]any) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var exists int
	if err := tx.QueryRow(`SELECT COUNT(*) FROM api_conversations WHERE id = ?`, id).Scan(&exists); err != nil {
		return err
	}
	if exists == 0 {
		return errAPINotFound
	}
	var next int
	if err := tx.QueryRow(`SELECT COALESCE(MAX(seq), -1) + 1 FROM api_conversation_items WHERE conv_id = ?`, id).Scan(&next); err != nil {
		return err
	}
	for i, it := range items {
		// Item ids are unique in a conversation; one sent again gets a new id.
		var dup int
		if err := tx.QueryRow(`SELECT COUNT(*) FROM api_conversation_items WHERE conv_id = ? AND id = ?`, id, it["id"]).Scan(&dup); err != nil {
			return err
		}
		if dup > 0 {
			it["id"] = itemPrefix(fmt.Sprint(it["type"])) + randHex(12)
		}
		if _, err := tx.Exec(`INSERT INTO api_conversation_items (conv_id, seq, id, item) VALUES (?, ?, ?, ?)`, id, next+i, it["id"], toJSON(it)); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// conversationItems returns every item, oldest first.
func (s *apiStore) conversationItems(id string) ([]map[string]any, error) {
	if _, err := s.getConversation(id); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(`SELECT item FROM api_conversation_items WHERE conv_id = ? ORDER BY seq`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		var raw string
		if err := rows.Scan(&raw); err != nil {
			return nil, err
		}
		out = append(out, fromJSON[map[string]any](raw))
	}
	return out, rows.Err()
}

func (s *apiStore) deleteConversationItem(convID, itemID string) error {
	res, err := s.db.Exec(`DELETE FROM api_conversation_items WHERE conv_id = ? AND id = ?`, convID, itemID)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return errAPINotFound
	}
	return nil
}

func (s *apiStore) putChatCompletion(id string, created int64, model string, metadata map[string]any, messages []any, completion map[string]any) error {
	if metadata == nil {
		metadata = map[string]any{}
	}
	_, err := s.db.Exec(`INSERT INTO api_chat_completions (id, created_at, model, metadata, messages, completion) VALUES (?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET metadata = excluded.metadata, messages = excluded.messages, completion = excluded.completion`,
		id, created, model, toJSON(metadata), toJSON(messages), toJSON(completion))
	return err
}

type storedCompletion struct {
	id         string
	model      string
	metadata   map[string]any
	messages   []any
	completion map[string]any
}

// object is the completion as OpenAI returns a stored one: with its metadata.
func (c storedCompletion) object() map[string]any {
	out := map[string]any{}
	for k, v := range c.completion {
		out[k] = v
	}
	out["metadata"] = c.metadata
	return out
}

func scanCompletion(sc interface{ Scan(...any) error }) (storedCompletion, error) {
	var c storedCompletion
	var md, msgs, comp string
	if err := sc.Scan(&c.id, &c.model, &md, &msgs, &comp); err != nil {
		return c, err
	}
	c.metadata = fromJSON[map[string]any](md)
	if c.metadata == nil {
		c.metadata = map[string]any{}
	}
	c.messages = fromJSON[[]any](msgs)
	c.completion = fromJSON[map[string]any](comp)
	return c, nil
}

func (s *apiStore) getChatCompletion(id string) (storedCompletion, error) {
	c, err := scanCompletion(s.db.QueryRow(`SELECT id, model, metadata, messages, completion FROM api_chat_completions WHERE id = ?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return c, errAPINotFound
	}
	return c, err
}

// listChatCompletions returns the stored completions, filtered by model and
// metadata, oldest first (newest first with desc).
func (s *apiStore) listChatCompletions(model string, meta map[string]string, desc bool) ([]storedCompletion, error) {
	order := "ASC"
	if desc {
		order = "DESC"
	}
	rows, err := s.db.Query(`SELECT id, model, metadata, messages, completion FROM api_chat_completions WHERE (? = '' OR model = ?) ORDER BY created_at `+order+`, rowid `+order, model, model)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []storedCompletion
	for rows.Next() {
		c, err := scanCompletion(rows)
		if err != nil {
			return nil, err
		}
		match := true
		for k, v := range meta {
			if got, _ := c.metadata[k].(string); got != v {
				match = false
			}
		}
		if match {
			out = append(out, c)
		}
	}
	return out, rows.Err()
}

func (s *apiStore) updateChatCompletionMetadata(id string, metadata map[string]any) (storedCompletion, error) {
	if metadata == nil {
		metadata = map[string]any{}
	}
	res, err := s.db.Exec(`UPDATE api_chat_completions SET metadata = ? WHERE id = ?`, toJSON(metadata), id)
	if err != nil {
		return storedCompletion{}, err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return storedCompletion{}, errAPINotFound
	}
	return s.getChatCompletion(id)
}

func (s *apiStore) deleteChatCompletion(id string) error {
	res, err := s.db.Exec(`DELETE FROM api_chat_completions WHERE id = ?`, id)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return errAPINotFound
	}
	return nil
}

// listPage answers an OpenAI list: items in the order asked for, starting
// after the item whose id is after, at most limit (1-100, default 20).
func listPage(r *http.Request, items []map[string]any, defaultDesc bool) (map[string]any, *apiError) {
	q := r.URL.Query()
	limit := 20
	if v := q.Get("limit"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 || n > 100 {
			return nil, &apiError{status: http.StatusBadRequest, param: "limit", msg: "limit must be between 1 and 100"}
		}
		limit = n
	}
	desc := defaultDesc
	switch q.Get("order") {
	case "asc":
		desc = false
	case "desc":
		desc = true
	case "":
	default:
		return nil, &apiError{status: http.StatusBadRequest, param: "order", msg: "order must be asc or desc"}
	}
	ordered := items
	if desc {
		ordered = make([]map[string]any, len(items))
		for i, it := range items {
			ordered[len(items)-1-i] = it
		}
	}
	start := 0
	if after := q.Get("after"); after != "" {
		start = -1
		for i, it := range ordered {
			if it["id"] == after {
				start = i + 1
				break
			}
		}
		if start < 0 {
			return nil, &apiError{status: http.StatusBadRequest, param: "after", msg: "no item with id " + after}
		}
	}
	page := ordered[start:]
	hasMore := len(page) > limit
	if hasMore {
		page = page[:limit]
	}
	out := map[string]any{"object": "list", "data": page, "has_more": hasMore, "first_id": nil, "last_id": nil}
	if len(page) > 0 {
		out["first_id"], out["last_id"] = page[0]["id"], page[len(page)-1]["id"]
	}
	if page == nil {
		out["data"] = []map[string]any{}
	}
	return out, nil
}
