package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// The Responses API on top of llama-server's stateless /v1/responses: the
// manager keeps each response (for previous_response_id, retrieval and
// input_items), conversations, background responses and compaction, and
// sends llama-server the whole history every time.

// respReq is one /v1/responses request on its way to llama-server.
type respReq struct {
	inst       *Instance
	forced     *forcedTool
	upstream   map[string]any   // the body llama-server gets
	input      []map[string]any // this request's own input items, with ids
	history    []map[string]any // the items before them
	prevID     string
	convID     string
	store      bool
	background bool
	stream     bool
	echo       map[string]any // fields the response object shows as sent
	id         string         // our own id, for background responses
	kept       bool           // stored (and added to its conversation) already
	sentError  bool           // an error event went to the caller already
	local      *wsLocal       // the WebSocket connection's own responses
}

// wsLocal keeps a WebSocket connection's recent responses, so
// previous_response_id works on it with store: false too, as at OpenAI.
type wsLocal struct {
	mu    sync.Mutex
	turns map[string][]map[string]any // id -> history + input + output
	order []string
}

const wsLocalMax = 16

func (l *wsLocal) put(id string, items []map[string]any) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, ok := l.turns[id]; !ok {
		l.order = append(l.order, id)
	}
	l.turns[id] = items
	for len(l.order) > wsLocalMax {
		delete(l.turns, l.order[0])
		l.order = l.order[1:]
	}
}

func (l *wsLocal) get(id string) ([]map[string]any, bool) {
	if l == nil {
		return nil, false
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	t, ok := l.turns[id]
	return t, ok
}

func (rq *respReq) patch(r map[string]any) {
	if rq.id != "" {
		r["id"] = rq.id
	}
	for k, v := range rq.echo {
		r[k] = v
	}
	r["model"] = rq.inst.conf.Name
}

func itemPrefix(typ string) string {
	switch typ {
	case "message":
		return "msg_"
	case "function_call":
		return "fc_"
	case "function_call_output":
		return "fco_"
	case "reasoning":
		return "rs_"
	}
	return "item_"
}

// normalizeInput turns "input" into items with ids: a string becomes a user
// message, string contents become parts, files become their text and
// item_reference becomes the item it names.
func (ws *WebServer) normalizeInput(ctx context.Context, input any, st *apiStore) ([]map[string]any, *apiError) {
	switch v := input.(type) {
	case nil:
		return []map[string]any{}, nil
	case string:
		return []map[string]any{{"id": "msg_" + randHex(12), "type": "message", "role": "user",
			"content": []any{map[string]any{"type": "input_text", "text": v}}}}, nil
	case []any:
		out := make([]map[string]any, 0, len(v))
		for _, raw := range v {
			m, ok := raw.(map[string]any)
			if !ok {
				return nil, &apiError{status: http.StatusBadRequest, param: "input", msg: "input items must be objects"}
			}
			m = cloneJSON(m)
			typ, _ := m["type"].(string)
			if typ == "" && m["role"] != nil {
				typ = "message"
				m["type"] = typ
			}
			if typ == "item_reference" {
				id, _ := m["id"].(string)
				if st == nil {
					return nil, apiErrf(http.StatusServiceUnavailable, "", "item_reference needs llama-manager's database, which couldn't be opened")
				}
				it, err := st.findItem(id)
				if err != nil {
					return nil, &apiError{status: http.StatusBadRequest, param: "input", msg: fmt.Sprintf("Item with id '%s' not found.", id)}
				}
				// A copy: ids stay unique wherever it is stored again.
				it["id"] = itemPrefix(fmt.Sprint(it["type"])) + randHex(12)
				out = append(out, it)
				continue
			}
			if typ == "message" {
				role, _ := m["role"].(string)
				if s, ok := m["content"].(string); ok {
					pt := "input_text"
					if role == "assistant" {
						pt = "output_text"
					}
					m["content"] = []any{map[string]any{"type": pt, "text": s}}
				}
				parts, _ := m["content"].([]any)
				for i, p := range parts {
					part, _ := p.(map[string]any)
					if part != nil && part["type"] == "input_file" {
						text, aerr := ws.filePart(ctx, part)
						if aerr != nil {
							return nil, aerr
						}
						parts[i] = map[string]any{"type": "input_text", "text": text}
					}
				}
			}
			if id, _ := m["id"].(string); id == "" || slices.ContainsFunc(out, func(o map[string]any) bool { return o["id"] == id }) {
				m["id"] = itemPrefix(typ) + randHex(12)
			}
			out = append(out, m)
		}
		return out, nil
	}
	return nil, &apiError{status: http.StatusBadRequest, param: "input", msg: "'input' must be a string or a list of items"}
}

func toItems(v any) []map[string]any {
	list, _ := v.([]any)
	out := make([]map[string]any, 0, len(list))
	for _, x := range list {
		if m, ok := x.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

// responseHistory is every item before a turn that continues response id:
// the inputs and outputs of the chain of previous responses, oldest first.
// Only id itself must still exist: a chain goes back as far as its stored
// links do (an older one may have been deleted). Every link read counts as
// used, so a chain in use isn't pruned.
func (s *apiStore) responseHistory(id string) ([]map[string]any, error) {
	var turns [][]map[string]any
	var ids []any
	for seen := map[string]bool{}; id != "" && !seen[id]; {
		seen[id] = true
		in, resp, err := s.getResponse(id)
		if err != nil {
			if len(turns) == 0 {
				return nil, err
			}
			break
		}
		ids = append(ids, id)
		turns = append(turns, append(in, toItems(resp["output"])...))
		id, _ = resp["previous_response_id"].(string)
	}
	if len(ids) > 0 {
		_, _ = s.db.Exec(`UPDATE api_responses SET used_at = ? WHERE id IN (?`+strings.Repeat(",?", len(ids)-1)+`)`, append([]any{time.Now().Unix()}, ids...)...)
	}
	var out []map[string]any
	for i := len(turns) - 1; i >= 0; i-- {
		out = append(out, turns[i]...)
	}
	return out, nil
}

const compactionPrefix = "lmc1."

func decodeCompaction(it map[string]any) (string, *apiError) {
	enc, _ := it["encrypted_content"].(string)
	b64, ok := strings.CutPrefix(enc, compactionPrefix)
	if !ok {
		return "", &apiError{status: http.StatusBadRequest, param: "input", msg: "this compaction item wasn't made by llama-manager's /v1/responses/compact"}
	}
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return "", &apiError{status: http.StatusBadRequest, param: "input", msg: "the compaction item is damaged"}
	}
	return string(raw), nil
}

// llamaItems are the item types llama-server's /v1/responses reads.
var llamaItems = map[string]bool{"message": true, "function_call": true, "function_call_output": true, "reasoning": true}

// expandForLlama gives llama-server only what it reads: a compaction item
// becomes the summary it holds, and items of tools that don't run here
// (web search, computer use...) are left out.
func expandForLlama(items []map[string]any) ([]any, *apiError) {
	out := make([]any, 0, len(items))
	for _, it := range items {
		typ, _ := it["type"].(string)
		if typ == "compaction" {
			summary, aerr := decodeCompaction(it)
			if aerr != nil {
				return nil, aerr
			}
			out = append(out, map[string]any{"type": "message", "role": "user",
				"content": []any{map[string]any{"type": "input_text", "text": "Summary of the earlier conversation:\n" + summary}}})
			continue
		}
		if llamaItems[typ] {
			out = append(out, it)
		}
	}
	return out, nil
}

// responsesFormat moves Responses' text.format to the response_format
// llama-server reads.
func responsesFormat(body map[string]any) *apiError {
	t, _ := body["text"].(map[string]any)
	delete(body, "text")
	f, _ := t["format"].(map[string]any)
	switch typ, _ := f["type"].(string); typ {
	case "", "text":
	case "json_object":
		body["response_format"] = map[string]any{"type": "json_object"}
	case "json_schema":
		js := map[string]any{"name": f["name"], "schema": f["schema"]}
		if s, ok := f["strict"]; ok {
			js["strict"] = s
		}
		body["response_format"] = map[string]any{"type": "json_schema", "json_schema": js}
	default:
		return &apiError{status: http.StatusBadRequest, param: "text.format", msg: fmt.Sprintf("text.format type %q isn't supported", typ)}
	}
	return nil
}

// cloudOnlyResponseFields are fields that only mean something at OpenAI.
var cloudOnlyResponseFields = []string{"previous_response_id", "conversation", "store", "background", "include", "truncation",
	"metadata", "service_tier", "safety_identifier", "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention",
	"moderation", "access_programs", "context_management", "image", "max_tool_calls", "user", "stream_id", "personality"}

func (ws *WebServer) prepareResponses(ctx context.Context, body map[string]any, local *wsLocal) (*respReq, *apiError) {
	inst, aerr := ws.bodyInstance(body)
	if aerr != nil {
		return nil, aerr
	}
	rq := &respReq{inst: inst, store: true, local: local}
	if v, ok := body["store"].(bool); ok {
		rq.store = v
	}
	rq.background, _ = body["background"].(bool)
	rq.stream, _ = body["stream"].(bool)
	rq.prevID, _ = body["previous_response_id"].(string)
	switch c := body["conversation"].(type) {
	case string:
		rq.convID = c
	case map[string]any:
		rq.convID, _ = c["id"].(string)
	}
	if rq.prevID != "" && rq.convID != "" {
		return nil, &apiError{status: http.StatusBadRequest, param: "conversation", msg: "previous_response_id and conversation can't be used together"}
	}
	if p, ok := body["prompt"]; ok && p != nil {
		return nil, &apiError{status: http.StatusBadRequest, param: "prompt", msg: "prompt templates are kept in OpenAI's cloud; send instructions and input instead"}
	}
	if rq.background && !rq.store {
		return nil, &apiError{status: http.StatusBadRequest, param: "background", msg: "background mode needs store: true"}
	}
	st := ws.apiStore()
	if st == nil && (rq.convID != "" || rq.background) {
		return nil, apiErrf(http.StatusServiceUnavailable, "", "llama-manager couldn't open its database (chats.db), so it can't keep responses or conversations; see its log")
	}
	if rq.input, aerr = ws.normalizeInput(ctx, body["input"], st); aerr != nil {
		return nil, aerr
	}
	var history []map[string]any
	if rq.prevID != "" {
		if h, ok := rq.local.get(rq.prevID); ok {
			history = h
		} else if st == nil {
			return nil, apiErrf(http.StatusServiceUnavailable, "", "llama-manager couldn't open its database (chats.db), so it can't keep responses; see its log")
		} else if h, err := st.responseHistory(rq.prevID); err != nil {
			return nil, &apiError{status: http.StatusBadRequest, param: "previous_response_id", msg: fmt.Sprintf("Previous response with id '%s' not found.", rq.prevID)}
		} else {
			history = h
		}
	}
	if rq.convID != "" {
		h, err := st.conversationItems(rq.convID)
		if err != nil {
			return nil, &apiError{status: http.StatusNotFound, code: "not_found", param: "conversation", msg: fmt.Sprintf("Conversation with id '%s' not found.", rq.convID)}
		}
		history = h
	}
	rq.history = history
	items, aerr := expandForLlama(append(append([]map[string]any{}, history...), rq.input...))
	if aerr != nil {
		return nil, aerr
	}
	up := map[string]any{}
	for k, v := range body {
		up[k] = v
	}
	up["input"] = items
	if aerr := responsesFormat(up); aerr != nil {
		return nil, aerr
	}
	if aerr := rewriteToolChoice(up); aerr != nil {
		return nil, aerr
	}
	rq.forced = forceOneTool(up, true)
	for _, k := range cloudOnlyResponseFields {
		delete(up, k)
	}
	up["stream"] = true
	rq.upstream = up
	md, _ := body["metadata"].(map[string]any)
	if md == nil {
		md = map[string]any{}
	}
	rq.echo = map[string]any{"store": rq.store, "background": rq.background, "metadata": md, "previous_response_id": nil}
	if rq.prevID != "" {
		rq.echo["previous_response_id"] = rq.prevID
	}
	if rq.convID != "" {
		rq.echo["conversation"] = map[string]any{"id": rq.convID}
	}
	for _, k := range []string{"text", "tools", "tool_choice", "instructions"} {
		if v, ok := body[k]; ok {
			rq.echo[k] = v
		}
	}
	return rq, nil
}

// forcedItem turns an output message holding the forced function's
// arguments into the function call.
func (rq *respReq) forcedItem(it map[string]any) map[string]any {
	if it == nil || it["type"] != "message" {
		return it
	}
	var args []string
	for _, p := range toItems(it["content"]) {
		if s, ok := p["text"].(string); ok {
			args = append(args, s)
		}
	}
	status, _ := it["status"].(string)
	if status == "" {
		status = "completed"
	}
	return map[string]any{"type": "function_call", "id": rq.forced.itemID, "call_id": rq.forced.callID, "name": rq.forced.name,
		"arguments": strings.TrimSpace(strings.Join(args, "")), "status": status}
}

// forcedEvent turns a streamed message event into the function call's;
// keep is false for events a function call has no counterpart of.
func (rq *respReq) forcedEvent(name string, data map[string]any) (string, bool) {
	switch name {
	case "response.output_item.added", "response.output_item.done":
		if it, ok := data["item"].(map[string]any); ok {
			data["item"] = rq.forcedItem(it)
		}
	case "response.content_part.added", "response.content_part.done":
		return name, false
	case "response.output_text.delta":
		name = "response.function_call_arguments.delta"
		data["type"], data["item_id"] = name, rq.forced.itemID
		delete(data, "content_index")
		delete(data, "logprobs")
	case "response.output_text.done":
		name = "response.function_call_arguments.done"
		data["type"], data["item_id"], data["arguments"] = name, rq.forced.itemID, strings.TrimSpace(fmt.Sprint(data["text"]))
		data["name"] = rq.forced.name
		delete(data, "text")
		delete(data, "content_index")
		delete(data, "logprobs")
	}
	if r, ok := data["response"].(map[string]any); ok {
		out := toItems(r["output"])
		conv := make([]any, len(out))
		for i, it := range out {
			conv[i] = rq.forcedItem(it)
		}
		r["output"] = conv
	}
	return name, true
}

func (ws *WebServer) openResponse(ctx context.Context, rq *respReq) (*http.Response, *apiError) {
	b, _ := json.Marshal(rq.upstream)
	return postLlama(ctx, rq.inst, "/v1/responses", "application/json", b)
}

// streamResponse reads llama-server's events, fixing the response objects in
// them and passing each to emit (when set), and returns the final response.
func (ws *WebServer) streamResponse(ctx context.Context, rq *respReq, resp *http.Response, emit func(string, map[string]any)) (map[string]any, *apiError) {
	defer resp.Body.Close()
	var final map[string]any
	var done []any
	failMsg := ""
	err := readUpSSE(resp.Body, func(ev upEvent) error {
		var data map[string]any
		if json.Unmarshal([]byte(ev.data), &data) != nil {
			return nil
		}
		name := ev.event
		if name == "" {
			name, _ = data["type"].(string)
		}
		if rq.forced != nil {
			var keep bool
			if name, keep = rq.forcedEvent(name, data); !keep {
				return ctx.Err()
			}
		}
		if r, ok := data["response"].(map[string]any); ok {
			rq.patch(r)
		}
		switch name {
		case "response.output_item.done":
			done = append(done, data["item"])
		case "response.completed", "response.failed", "response.incomplete":
			final, _ = data["response"].(map[string]any)
			if final != nil {
				if out, _ := final["output"].([]any); len(out) == 0 && len(done) > 0 {
					final["output"] = done
				}
				// Stored before the caller hears it's done, so a follow-up
				// can name it at once.
				ws.keepResponse(rq, final)
			}
		case "error":
			failMsg, _ = data["message"].(string)
			if e, ok := data["error"].(map[string]any); ok {
				failMsg, _ = e["message"].(string)
			}
			rq.sentError = emit != nil
		}
		if emit != nil {
			emit(name, data)
		}
		return ctx.Err()
	})
	if final == nil {
		if ctx.Err() != nil {
			return nil, apiErrf(499, "", "cancelled")
		}
		if failMsg == "" && err != nil {
			failMsg = err.Error()
		}
		if failMsg == "" {
			failMsg = "the answer ended without a final response"
		}
		return nil, apiErrf(http.StatusBadGateway, "", "%s: %s", rq.inst.conf.Name, failMsg)
	}
	return final, nil
}

// keepResponse stores a finished response and adds the turn to its
// conversation.
func (ws *WebServer) keepResponse(rq *respReq, final map[string]any) {
	if rq.kept {
		return
	}
	rq.kept = true
	id, _ := final["id"].(string)
	if rq.local != nil && id != "" {
		rq.local.put(id, append(append(append([]map[string]any{}, rq.history...), rq.input...), toItems(final["output"])...))
	}
	st := ws.apiStore()
	if st == nil {
		return
	}
	created := time.Now().Unix()
	if f, ok := final["created_at"].(float64); ok {
		created = int64(f)
	}
	if rq.store && id != "" {
		_ = st.putResponse(id, created, rq.input, final)
	}
	if rq.convID != "" && (final["status"] == "completed" || final["status"] == "incomplete") {
		_ = st.addConversationItems(rq.convID, append(append([]map[string]any{}, rq.input...), toItems(final["output"])...))
	}
}

// bgJob is a background response being generated.
type bgJob struct {
	cancel  context.CancelFunc
	done    chan struct{}
	deleted atomic.Bool // deleted while running: don't store it again
}

func (ws *WebServer) handleV1Responses(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet && isWebSocket(r) {
		ws.responsesWS(w, r)
		return
	}
	if !allowMethods(w, r, http.MethodPost) {
		return
	}
	body, aerr := readBodyMap(w, r)
	if aerr != nil {
		aerr.write(w)
		return
	}
	rq, aerr := ws.prepareResponses(r.Context(), body, nil)
	if aerr != nil {
		aerr.write(w)
		return
	}
	if rq.background {
		ws.startBackground(w, r, rq)
		return
	}
	resp, aerr := ws.openResponse(r.Context(), rq)
	if aerr != nil {
		if r.Context().Err() == nil {
			aerr.write(w)
		}
		return
	}
	var emit func(string, map[string]any)
	if rq.stream {
		startSSE(w)
		emit = func(name string, data map[string]any) { writeUpEvent(w, upData(name, data)) }
	}
	final, aerr := ws.streamResponse(r.Context(), rq, resp, emit)
	if aerr != nil {
		if rq.stream {
			if !rq.sentError {
				writeUpEvent(w, upData("error", map[string]any{"type": "error", "code": nilIfEmpty(aerr.code), "message": aerr.msg, "param": nil}))
			}
		} else if r.Context().Err() == nil {
			aerr.write(w)
		}
		return
	}
	if !rq.stream {
		writeJSON(w, http.StatusOK, final)
	}
}

func nilIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// startBackground answers at once with a queued response and generates it
// in the background; the caller polls GET /v1/responses/{id}. With stream it
// also sends the events while the caller stays connected.
func (ws *WebServer) startBackground(w http.ResponseWriter, r *http.Request, rq *respReq) {
	st := ws.apiStore()
	rq.id = "resp_" + randHex(24)
	created := time.Now().Unix()
	queued := map[string]any{"id": rq.id, "object": "response", "created_at": created, "status": "queued", "output": []any{},
		"error": nil, "incomplete_details": nil, "usage": nil, "parallel_tool_calls": true, "tool_choice": "auto", "tools": []any{}}
	rq.patch(queued)
	if err := st.putResponse(rq.id, created, rq.input, queued); err != nil {
		writeAPIError(w, http.StatusInternalServerError, "", "", "storing the response: "+err.Error())
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	job := &bgJob{cancel: cancel, done: make(chan struct{})}
	ws.bgJobs.Store(rq.id, job)
	var mu sync.Mutex
	gone := !rq.stream
	emit := func(name string, data map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		if !gone {
			writeUpEvent(w, upData(name, data))
		}
	}
	if rq.stream {
		startSSE(w)
	}
	go func() {
		defer close(job.done)
		defer ws.bgJobs.Delete(rq.id)
		defer cancel()
		running := cloneJSON(queued)
		running["status"] = "in_progress"
		if !job.deleted.Load() {
			_ = st.putResponse(rq.id, created, rq.input, running)
		}
		resp, aerr := ws.openResponse(ctx, rq)
		var final map[string]any
		if aerr == nil {
			final, aerr = ws.streamResponse(ctx, rq, resp, emit)
		}
		if job.deleted.Load() {
			return
		}
		switch {
		case ctx.Err() != nil:
			final = cloneJSON(queued)
			final["status"] = "cancelled"
			rq.kept = false
		case aerr != nil:
			final = cloneJSON(queued)
			final["status"] = "failed"
			final["error"] = map[string]any{"code": "server_error", "message": aerr.msg}
			emit("response.failed", map[string]any{"type": "response.failed", "response": final})
		}
		ws.keepResponse(rq, final)
	}()
	if !rq.stream {
		writeJSON(w, http.StatusOK, queued)
		return
	}
	select {
	case <-job.done:
	case <-r.Context().Done():
	}
	mu.Lock()
	gone = true
	mu.Unlock()
}

// handleV1ResponseSub serves /v1/responses/{id}, {id}/cancel,
// {id}/input_items, input_tokens and compact.
func (ws *WebServer) handleV1ResponseSub(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/v1/responses/")
	switch rest {
	case "input_tokens":
		ws.responsesInputTokens(w, r)
		return
	case "compact":
		ws.responsesCompact(w, r)
		return
	}
	id, sub, _ := strings.Cut(rest, "/")
	st := ws.needStore(w)
	if st == nil {
		return
	}
	notFound := func() {
		writeAPIError(w, http.StatusNotFound, "not_found", "", fmt.Sprintf("Response with id '%s' not found.", id))
	}
	switch sub {
	case "":
		if !allowMethods(w, r, http.MethodGet, http.MethodDelete) {
			return
		}
		if r.Method == http.MethodDelete {
			if v, ok := ws.bgJobs.Load(id); ok {
				job := v.(*bgJob)
				job.deleted.Store(true)
				job.cancel()
				select {
				case <-job.done:
				case <-time.After(10 * time.Second):
				}
			}
			if err := st.deleteResponse(id); err != nil {
				notFound()
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{"id": id, "object": "response", "deleted": true})
			return
		}
		if r.URL.Query().Get("stream") == "true" {
			writeAPIError(w, http.StatusBadRequest, "", "stream", "streaming a stored response again isn't supported; fetch it without stream")
			return
		}
		_, resp, err := st.getResponse(id)
		if err != nil {
			notFound()
			return
		}
		writeJSON(w, http.StatusOK, resp)
	case "cancel":
		if !allowMethods(w, r, http.MethodPost) {
			return
		}
		_, resp, err := st.getResponse(id)
		if err != nil {
			notFound()
			return
		}
		if bg, _ := resp["background"].(bool); !bg {
			writeAPIError(w, http.StatusBadRequest, "", "", "Only background responses can be cancelled.")
			return
		}
		if job, ok := ws.bgJobs.Load(id); ok {
			job.(*bgJob).cancel()
			select {
			case <-job.(*bgJob).done:
			case <-time.After(10 * time.Second):
			}
		}
		_, resp, _ = st.getResponse(id)
		writeJSON(w, http.StatusOK, resp)
	case "input_items":
		if !allowMethods(w, r, http.MethodGet) {
			return
		}
		in, _, err := st.getResponse(id)
		if err != nil {
			notFound()
			return
		}
		page, aerr := listPage(r, in, true)
		if aerr != nil {
			aerr.write(w)
			return
		}
		writeJSON(w, http.StatusOK, page)
	default:
		writeAPIError(w, http.StatusNotFound, "", "", fmt.Sprintf("Invalid URL (%s %s)", r.Method, r.URL.Path))
	}
}

// responsesInputTokens counts a Responses request's tokens with its history
// (previous_response_id, conversation) included.
func (ws *WebServer) responsesInputTokens(w http.ResponseWriter, r *http.Request) {
	if !allowMethods(w, r, http.MethodPost) {
		return
	}
	body, aerr := readBodyMap(w, r)
	if aerr != nil {
		aerr.write(w)
		return
	}
	rq, aerr := ws.prepareResponses(r.Context(), body, nil)
	if aerr != nil {
		aerr.write(w)
		return
	}
	delete(rq.upstream, "stream")
	b, _ := json.Marshal(rq.upstream)
	relayLlama(w, r, rq.inst, "/v1/responses/input_tokens", b, relayTx{})
}

const compactPrompt = `You compact conversations so they can continue in a smaller context. Write a summary of the conversation below for the assistant to continue from: the goals, decisions, facts, names, numbers, code and file paths that matter, what was done, and what is still open. Write only the summary.`

// itemText renders an item for the compaction prompt.
func itemText(it map[string]any) string {
	switch it["type"] {
	case "message":
		role, _ := it["role"].(string)
		var parts []string
		for _, p := range toItems(it["content"]) {
			if s, ok := p["text"].(string); ok {
				parts = append(parts, s)
			}
		}
		return role + ": " + strings.Join(parts, "\n")
	case "function_call":
		return fmt.Sprintf("assistant called %v(%v)", it["name"], it["arguments"])
	case "function_call_output":
		return fmt.Sprintf("tool result: %v", it["output"])
	case "compaction":
		if s, aerr := decodeCompaction(it); aerr == nil {
			return "summary of what came before: " + s
		}
	}
	return ""
}

// responsesCompact shrinks a conversation: the model summarizes it, and the
// answer keeps the user's messages and puts the summary in a compaction item
// that later requests send back as input.
func (ws *WebServer) responsesCompact(w http.ResponseWriter, r *http.Request) {
	if !allowMethods(w, r, http.MethodPost) {
		return
	}
	body, aerr := readBodyMap(w, r)
	if aerr != nil {
		aerr.write(w)
		return
	}
	inst, aerr := ws.bodyInstance(body)
	if aerr != nil {
		aerr.write(w)
		return
	}
	st := ws.apiStore()
	items, aerr := ws.normalizeInput(r.Context(), body["input"], st)
	if aerr != nil {
		aerr.write(w)
		return
	}
	if prev, _ := body["previous_response_id"].(string); prev != "" {
		if st == nil {
			ws.needStore(w)
			return
		}
		h, err := st.responseHistory(prev)
		if err != nil {
			writeAPIError(w, http.StatusBadRequest, "", "previous_response_id", fmt.Sprintf("Previous response with id '%s' not found.", prev))
			return
		}
		items = append(h, items...)
	}
	var transcript []string
	if s, _ := body["instructions"].(string); s != "" {
		transcript = append(transcript, "instructions: "+s)
	}
	for _, it := range items {
		if t := itemText(it); t != "" {
			transcript = append(transcript, t)
		}
	}
	req, _ := json.Marshal(map[string]any{
		"messages": []any{map[string]any{"role": "system", "content": compactPrompt},
			map[string]any{"role": "user", "content": strings.Join(transcript, "\n\n")}},
		"temperature": 0.2, "max_tokens": 4096, "chat_template_kwargs": map[string]any{"enable_thinking": false},
	})
	resp, aerr := postLlama(r.Context(), inst, "/v1/chat/completions", "application/json", req)
	if aerr != nil {
		if r.Context().Err() == nil {
			aerr.write(w)
		}
		return
	}
	defer resp.Body.Close()
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage tokenUsage `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || len(out.Choices) == 0 {
		writeAPIError(w, http.StatusBadGateway, "", "", inst.conf.Name+" sent an answer llama-manager can't read")
		return
	}
	summary := cleanTranscript(out.Choices[0].Message.Content)
	var output []any
	for _, it := range items {
		if it["type"] == "message" && it["role"] == "user" {
			output = append(output, it)
		}
	}
	output = append(output, map[string]any{"id": "cmp_" + randHex(12), "type": "compaction", "created_by": "llama-manager",
		"encrypted_content": compactionPrefix + base64.StdEncoding.EncodeToString([]byte(summary))})
	writeJSON(w, http.StatusOK, map[string]any{"id": "resp_" + randHex(24), "object": "response.compaction", "created_at": time.Now().Unix(),
		"output": output, "usage": map[string]any{"input_tokens": out.Usage.PromptTokens, "output_tokens": out.Usage.CompletionTokens,
			"total_tokens": out.Usage.TotalTokens, "input_tokens_details": map[string]int{"cached_tokens": 0}, "output_tokens_details": map[string]int{"reasoning_tokens": 0}}})
}

// responsesWS is the Responses API over a WebSocket: each response.create
// message runs like a streamed POST /v1/responses, its events sent back as
// messages (with the stream_id it gave, for clients that run several).
func (ws *WebServer) responsesWS(w http.ResponseWriter, r *http.Request) {
	conn, err := wsAccept(w, r, pickProtocol(websocketProtocols(r)))
	if err != nil {
		if errors.Is(err, errWSVersion) {
			w.Header().Set("Sec-WebSocket-Version", "13")
			writeAPIError(w, http.StatusUpgradeRequired, "", "", err.Error())
			return
		}
		writeAPIError(w, http.StatusBadRequest, "", "", err.Error())
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	// Closing the connection stops what it started: cancel first, then wait.
	defer func() {
		cancel()
		wg.Wait()
		conn.Close()
	}()
	go conn.keepAlive(ctx)
	local := &wsLocal{turns: map[string][]map[string]any{}}
	send := func(sid string, v map[string]any) {
		if sid != "" {
			v["stream_id"] = sid
		}
		b, _ := json.Marshal(v)
		_ = conn.WriteText(b)
	}
	sendErr := func(sid string, e *apiError) {
		typ := "invalid_request_error"
		if e.status >= 500 {
			typ = "server_error"
		}
		send(sid, map[string]any{"type": "error", "status": e.status, "error": map[string]any{"type": typ, "code": nilIfEmpty(e.code), "message": e.msg, "param": nilIfEmpty(e.param)}})
	}
	for {
		msg, err := conn.ReadMessage()
		if err != nil {
			return
		}
		var ev map[string]any
		if json.Unmarshal(msg, &ev) != nil {
			sendErr("", apiErrf(http.StatusBadRequest, "", "messages must be JSON events"))
			continue
		}
		sid, _ := ev["stream_id"].(string)
		switch ev["type"] {
		case "response.create":
			if bg, _ := ev["background"].(bool); bg {
				sendErr(sid, &apiError{status: http.StatusBadRequest, param: "background", msg: "background isn't supported over WebSocket"})
				continue
			}
			delete(ev, "type")
			ev["stream"] = true
			wg.Add(1)
			go func() {
				defer wg.Done()
				rq, aerr := ws.prepareResponses(ctx, ev, local)
				if aerr == nil {
					var resp *http.Response
					if resp, aerr = ws.openResponse(ctx, rq); aerr == nil {
						_, aerr = ws.streamResponse(ctx, rq, resp, func(_ string, data map[string]any) { send(sid, data) })
					}
				}
				if aerr != nil && ctx.Err() == nil && (rq == nil || !rq.sentError) {
					sendErr(sid, aerr)
				}
			}()
		case "response.steer":
			sendErr(sid, apiErrf(http.StatusBadRequest, "", "response.steer isn't supported: llama-server can't change a response while it is being generated"))
		default:
			sendErr(sid, apiErrf(http.StatusBadRequest, "", "unknown event type %v; send response.create", ev["type"]))
		}
	}
}

// --- conversations -----------------------------------------------------------

func (ws *WebServer) handleV1Conversations(w http.ResponseWriter, r *http.Request) {
	st := ws.needStore(w)
	if st == nil {
		return
	}
	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/v1/conversations"), "/")
	parts := strings.Split(rest, "/")
	notFound := func(id string) {
		writeAPIError(w, http.StatusNotFound, "not_found", "", fmt.Sprintf("Conversation with id '%s' not found.", id))
	}
	readItems := func(body map[string]any) ([]map[string]any, *apiError) {
		if body["items"] == nil {
			return []map[string]any{}, nil
		}
		if _, ok := body["items"].([]any); !ok {
			return nil, &apiError{status: http.StatusBadRequest, param: "items", msg: "items must be a list"}
		}
		return ws.normalizeInput(r.Context(), body["items"], st)
	}
	switch {
	case rest == "":
		if !allowMethods(w, r, http.MethodPost) {
			return
		}
		body, aerr := readBodyMap(w, r)
		if aerr != nil {
			aerr.write(w)
			return
		}
		items, aerr := readItems(body)
		if aerr != nil {
			aerr.write(w)
			return
		}
		md, _ := body["metadata"].(map[string]any)
		c, err := st.createConversation("conv_"+randHex(24), time.Now().Unix(), md, items)
		if err != nil {
			writeAPIError(w, http.StatusInternalServerError, "", "", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, c)
	case len(parts) == 1:
		id := parts[0]
		if !allowMethods(w, r, http.MethodGet, http.MethodPost, http.MethodDelete) {
			return
		}
		var c map[string]any
		var err error
		switch r.Method {
		case http.MethodGet:
			c, err = st.getConversation(id)
		case http.MethodPost:
			body, aerr := readBodyMap(w, r)
			if aerr != nil {
				aerr.write(w)
				return
			}
			md, _ := body["metadata"].(map[string]any)
			c, err = st.updateConversation(id, md)
		case http.MethodDelete:
			if err = st.deleteConversation(id); err == nil {
				c = map[string]any{"id": id, "object": "conversation.deleted", "deleted": true}
			}
		}
		if err != nil {
			notFound(id)
			return
		}
		writeJSON(w, http.StatusOK, c)
	case len(parts) == 2 && parts[1] == "items":
		id := parts[0]
		if !allowMethods(w, r, http.MethodGet, http.MethodPost) {
			return
		}
		if r.Method == http.MethodPost {
			body, aerr := readBodyMap(w, r)
			if aerr != nil {
				aerr.write(w)
				return
			}
			items, aerr := readItems(body)
			if aerr != nil {
				aerr.write(w)
				return
			}
			if err := st.addConversationItems(id, items); err != nil {
				notFound(id)
				return
			}
			out := map[string]any{"object": "list", "data": items, "has_more": false, "first_id": nil, "last_id": nil}
			if len(items) > 0 {
				out["first_id"], out["last_id"] = items[0]["id"], items[len(items)-1]["id"]
			}
			writeJSON(w, http.StatusOK, out)
			return
		}
		items, err := st.conversationItems(id)
		if err != nil {
			notFound(id)
			return
		}
		page, aerr := listPage(r, items, true)
		if aerr != nil {
			aerr.write(w)
			return
		}
		writeJSON(w, http.StatusOK, page)
	case len(parts) == 3 && parts[1] == "items":
		id, itemID := parts[0], parts[2]
		if !allowMethods(w, r, http.MethodGet, http.MethodDelete) {
			return
		}
		items, err := st.conversationItems(id)
		if err != nil {
			notFound(id)
			return
		}
		if r.Method == http.MethodDelete {
			if err := st.deleteConversationItem(id, itemID); err != nil {
				writeAPIError(w, http.StatusNotFound, "not_found", "", fmt.Sprintf("Item with id '%s' not found.", itemID))
				return
			}
			c, _ := st.getConversation(id)
			writeJSON(w, http.StatusOK, c)
			return
		}
		for _, it := range items {
			if it["id"] == itemID {
				writeJSON(w, http.StatusOK, it)
				return
			}
		}
		writeAPIError(w, http.StatusNotFound, "not_found", "", fmt.Sprintf("Item with id '%s' not found.", itemID))
	default:
		writeAPIError(w, http.StatusNotFound, "", "", fmt.Sprintf("Invalid URL (%s %s)", r.Method, r.URL.Path))
	}
}

// --- stored chat completions -------------------------------------------------

func (ws *WebServer) storeChat(c map[string]any, tx chatTx) {
	st := ws.apiStore()
	id, _ := c["id"].(string)
	if st == nil || id == "" {
		return
	}
	created := time.Now().Unix()
	if f, ok := c["created"].(float64); ok {
		created = int64(f)
	}
	model, _ := c["model"].(string)
	_ = st.putChatCompletion(id, created, model, tx.metadata, tx.messages, c)
}

// metadataFilter reads metadata[key]=value query parameters.
func metadataFilter(r *http.Request) map[string]string {
	out := map[string]string{}
	for k, v := range r.URL.Query() {
		if key, ok := strings.CutPrefix(k, "metadata["); ok && strings.HasSuffix(key, "]") && len(v) > 0 {
			out[strings.TrimSuffix(key, "]")] = v[0]
		}
	}
	return out
}

func (ws *WebServer) handleStoredChatList(w http.ResponseWriter, r *http.Request) {
	st := ws.needStore(w)
	if st == nil {
		return
	}
	list, err := st.listChatCompletions(r.URL.Query().Get("model"), metadataFilter(r), false)
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, "", "", err.Error())
		return
	}
	items := make([]map[string]any, len(list))
	for i, c := range list {
		items[i] = c.object()
	}
	page, aerr := listPage(r, items, false)
	if aerr != nil {
		aerr.write(w)
		return
	}
	writeJSON(w, http.StatusOK, page)
}

// handleV1ChatStored serves /v1/chat/completions/{id} and {id}/messages.
func (ws *WebServer) handleV1ChatStored(w http.ResponseWriter, r *http.Request) {
	st := ws.needStore(w)
	if st == nil {
		return
	}
	id, sub, _ := strings.Cut(strings.TrimPrefix(r.URL.Path, "/v1/chat/completions/"), "/")
	notFound := func() {
		writeAPIError(w, http.StatusNotFound, "not_found", "", fmt.Sprintf("Chat completion with id '%s' not found. Only completions sent with store: true are kept.", id))
	}
	switch sub {
	case "":
		if !allowMethods(w, r, http.MethodGet, http.MethodPost, http.MethodDelete) {
			return
		}
		switch r.Method {
		case http.MethodGet:
			c, err := st.getChatCompletion(id)
			if err != nil {
				notFound()
				return
			}
			writeJSON(w, http.StatusOK, c.object())
		case http.MethodPost:
			body, aerr := readBodyMap(w, r)
			if aerr != nil {
				aerr.write(w)
				return
			}
			md, _ := body["metadata"].(map[string]any)
			c, err := st.updateChatCompletionMetadata(id, md)
			if err != nil {
				notFound()
				return
			}
			writeJSON(w, http.StatusOK, c.object())
		case http.MethodDelete:
			if err := st.deleteChatCompletion(id); err != nil {
				notFound()
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{"id": id, "object": "chat.completion.deleted", "deleted": true})
		}
	case "messages":
		if !allowMethods(w, r, http.MethodGet) {
			return
		}
		c, err := st.getChatCompletion(id)
		if err != nil {
			notFound()
			return
		}
		msgs := make([]map[string]any, 0, len(c.messages))
		for i, m := range c.messages {
			msg, _ := m.(map[string]any)
			out := map[string]any{"id": fmt.Sprintf("%s-%d", id, i), "content_parts": nil}
			for k, v := range msg {
				out[k] = v
			}
			if parts, ok := msg["content"].([]any); ok {
				out["content_parts"] = parts
				var texts []string
				for _, p := range toItems(parts) {
					if s, ok := p["text"].(string); ok {
						texts = append(texts, s)
					}
				}
				out["content"] = strings.Join(texts, "\n")
			}
			msgs = append(msgs, out)
		}
		page, aerr := listPage(r, msgs, false)
		if aerr != nil {
			aerr.write(w)
			return
		}
		writeJSON(w, http.StatusOK, page)
	default:
		writeAPIError(w, http.StatusNotFound, "", "", fmt.Sprintf("Invalid URL (%s %s)", r.Method, r.URL.Path))
	}
}

// startAPIState marks background responses cut off by a restart as failed
// and prunes stored responses and completions older than 30 days, daily.
func (ws *WebServer) startAPIState(ctx context.Context) {
	st := ws.apiStore()
	if st == nil {
		return
	}
	_, _ = st.db.Exec(`UPDATE api_responses SET response = json_set(response, '$.status', 'failed',
		'$.error', json_object('code', 'server_error', 'message', 'llama-manager restarted while generating this response'))
		WHERE json_extract(response, '$.status') IN ('queued', 'in_progress')`)
	go func() {
		for {
			st.prune(time.Now())
			select {
			case <-ctx.Done():
				return
			case <-time.After(24 * time.Hour):
			}
		}
	}()
}

var _ = errors.New
