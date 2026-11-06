package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// upEvent is one server-sent event: its name (often empty), its data and
// any comment lines (": ping").
type upEvent struct {
	event    string
	data     string
	comments []string
}

// readSSE calls fn for each event in r until EOF or fn fails.
func readUpSSE(r io.Reader, fn func(upEvent) error) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 64<<20)
	var ev upEvent
	var data []string
	flush := func() error {
		if ev.event == "" && len(data) == 0 && len(ev.comments) == 0 {
			return nil
		}
		ev.data = strings.Join(data, "\n")
		err := fn(ev)
		ev, data = upEvent{}, nil
		return err
	}
	for sc.Scan() {
		line := sc.Text()
		switch {
		case line == "":
			if err := flush(); err != nil {
				return err
			}
		case strings.HasPrefix(line, ":"):
			ev.comments = append(ev.comments, line)
		case strings.HasPrefix(line, "event:"):
			ev.event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			data = append(data, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
		}
	}
	if err := flush(); err != nil {
		return err
	}
	return sc.Err()
}

func writeUpEvent(w io.Writer, ev upEvent) {
	var b strings.Builder
	for _, c := range ev.comments {
		b.WriteString(c + "\n")
	}
	if ev.event != "" {
		b.WriteString("event: " + ev.event + "\n")
	}
	if ev.data != "" || ev.event != "" {
		for _, l := range strings.Split(ev.data, "\n") {
			b.WriteString("data: " + l + "\n")
		}
	}
	b.WriteString("\n")
	_, _ = io.WriteString(w, b.String())
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func upData(event string, v any) upEvent {
	b, _ := json.Marshal(v)
	return upEvent{event: event, data: string(b)}
}

// openAIError rewrites an upstream error answer into OpenAI's shape:
// llama-server's numeric "code" goes, and its 500/501 "not supported" answers
// (an image to a model without vision...) are the caller's mistake, so 400.
func openAIError(status int, body []byte) *apiError {
	e := &apiError{status: status}
	var raw map[string]any
	if json.Unmarshal(body, &raw) == nil {
		switch v := raw["error"].(type) {
		case map[string]any:
			e.msg, _ = v["message"].(string)
			e.code, _ = v["code"].(string)
			e.param, _ = v["param"].(string)
			if t, _ := v["type"].(string); t == "not_supported_error" {
				e.status = http.StatusBadRequest
			}
		case string:
			e.msg = v
			e.code, _ = raw["code"].(string)
		}
		if e.msg == "" {
			e.msg, _ = raw["message"].(string)
		}
	}
	if e.msg == "" {
		e.msg = strings.TrimSpace(string(body))
	}
	if e.msg == "" {
		e.msg = http.StatusText(status)
	}
	if (status == http.StatusInternalServerError || status == http.StatusNotImplemented) && strings.Contains(strings.ToLower(e.msg), "not support") {
		e.status = http.StatusBadRequest
	}
	return e
}

// relayTx rewrites what llama-server answers: json a whole answer, event
// each streamed event (returning the events to send instead).
type relayTx struct {
	json  func(map[string]any) *apiError
	event func(upEvent) []upEvent
}

// postLlama sends body to path on inst; a failure comes back as an apiError
// in OpenAI's shape.
func postLlama(ctx context.Context, inst *Instance, path, contentType string, body []byte) (*http.Response, *apiError) {
	noteModel(ctx, inst.conf.Name)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, inst.baseURL()+path, bytes.NewReader(body))
	if err != nil {
		return nil, apiErrf(http.StatusInternalServerError, "", "%v", err)
	}
	req.Header.Set("Content-Type", contentType)
	wait := inst.upstreamTimeout()
	resp, err := clientWithTimeout(wait).Do(req)
	if err != nil {
		var ue *url.Error
		if ctx.Err() == nil && errors.As(err, &ue) && ue.Timeout() {
			return nil, apiErrf(http.StatusGatewayTimeout, "", "%s didn't start answering within %s; for longer non-streamed answers raise --timeout in its extra_args, or use \"stream\": true", inst.conf.Name, wait)
		}
		return nil, apiErrf(http.StatusBadGateway, "", "proxying to %s: %v", inst.conf.Name, err)
	}
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()
		return nil, openAIError(resp.StatusCode, b)
	}
	return resp, nil
}

// relayLlama posts body to path on inst and relays the answer, through tx.
// A "stream": true body is relayed event by event as it arrives.
func relayLlama(w http.ResponseWriter, r *http.Request, inst *Instance, path string, body []byte, tx relayTx) {
	noteModel(r.Context(), inst.conf.Name)
	var probe struct {
		Stream bool `json:"stream"`
	}
	_ = json.Unmarshal(body, &probe)
	start := time.Now()
	resp, aerr := postLlama(r.Context(), inst, path, "application/json", body)
	if aerr != nil {
		if r.Context().Err() == nil {
			aerr.write(w)
		}
		return
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	w.Header().Set("X-Manager-Elapsed-Ms", strconv.FormatInt(time.Since(start).Milliseconds(), 10))
	if probe.Stream {
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(resp.StatusCode)
		if tx.event == nil {
			flusher, _ := w.(http.Flusher)
			buf := make([]byte, 16*1024)
			for {
				n, rerr := resp.Body.Read(buf)
				if n > 0 {
					if _, werr := w.Write(buf[:n]); werr != nil {
						return
					}
					if flusher != nil {
						flusher.Flush()
					}
				}
				if rerr != nil {
					return
				}
			}
		}
		_ = readUpSSE(resp.Body, func(ev upEvent) error {
			for _, out := range tx.event(ev) {
				writeUpEvent(w, out)
			}
			return r.Context().Err()
		})
		return
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		writeAPIError(w, http.StatusBadGateway, "", "", "reading llama-server's answer: "+err.Error())
		return
	}
	if tx.json != nil {
		var m map[string]any
		if json.Unmarshal(data, &m) == nil && m != nil {
			if aerr := tx.json(m); aerr != nil {
				aerr.write(w)
				return
			}
			data, _ = json.Marshal(m)
			w.Header().Del("Content-Length")
		}
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(data)
}

// readBodyMap reads a /v1 JSON body as a map.
func readBodyMap(w http.ResponseWriter, r *http.Request) (map[string]any, *apiError) {
	b, aerr := readAPIBody(w, r, maxAPIBody)
	if aerr != nil {
		return nil, aerr
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil || m == nil {
		msg := "We could not parse the JSON body of your request"
		if err != nil {
			msg += ": " + err.Error()
		}
		return nil, apiErrf(http.StatusBadRequest, "", "%s", msg)
	}
	return m, nil
}

func (ws *WebServer) bodyInstance(ctx context.Context, body map[string]any) (*Instance, *apiError) {
	name, _ := body["model"].(string)
	return ws.llamaInstance(ctx, name)
}

// --- tool choice -------------------------------------------------------------

// toolName is a tool's function name in the chat ({"function": {"name"}}) or
// Responses ({"name"}) shape.
func toolName(t map[string]any) string {
	if f, ok := t["function"].(map[string]any); ok {
		n, _ := f["name"].(string)
		return n
	}
	n, _ := t["name"].(string)
	return n
}

// rewriteToolChoice turns a tool_choice that picks tools (a function by name,
// allowed_tools) into what llama-server takes: "required" or "auto" with only
// those tools. llama-server reads tool_choice as a string only.
func rewriteToolChoice(body map[string]any) *apiError {
	tc, ok := body["tool_choice"].(map[string]any)
	if !ok {
		return nil
	}
	tools, _ := body["tools"].([]any)
	var keep map[string]bool
	mode := "required"
	switch typ, _ := tc["type"].(string); typ {
	case "function":
		name := toolName(tc)
		if name == "" {
			return &apiError{status: http.StatusBadRequest, param: "tool_choice", msg: "tool_choice names no function"}
		}
		keep = map[string]bool{name: true}
	case "allowed_tools":
		spec := tc
		if inner, ok := tc["allowed_tools"].(map[string]any); ok {
			spec = inner
		}
		if m, _ := spec["mode"].(string); m != "" {
			mode = m
		}
		if mode != "auto" && mode != "required" {
			return &apiError{status: http.StatusBadRequest, param: "tool_choice", msg: "allowed_tools mode must be auto or required"}
		}
		keep = map[string]bool{}
		list, _ := spec["tools"].([]any)
		for _, t := range list {
			if m, ok := t.(map[string]any); ok {
				keep[toolName(m)] = true
			}
		}
	default:
		return &apiError{status: http.StatusBadRequest, param: "tool_choice", msg: fmt.Sprintf("tool_choice type %q isn't supported here: only functions run on this server", typ)}
	}
	var out []any
	for _, t := range tools {
		if m, ok := t.(map[string]any); ok && keep[toolName(m)] {
			out = append(out, t)
		}
	}
	if len(out) == 0 {
		return &apiError{status: http.StatusBadRequest, param: "tool_choice", msg: "tool_choice names a tool that isn't in tools"}
	}
	body["tools"] = out
	body["tool_choice"] = mode
	return nil
}

// forcedTool is the one function a request demands.
type forcedTool struct {
	name   string
	callID string
	itemID string
}

// forceOneTool handles a request that demands one particular function
// (tool_choice naming it, or "required" with one tool). llama-server's
// "required" doesn't stop a model from talking instead, so the model writes
// the function's arguments under its JSON schema (a grammar) and the answer
// is turned into the call. instructions gets the hint (a Responses field);
// without it the hint goes into the messages.
func forceOneTool(body map[string]any, responses bool) *forcedTool {
	tools, _ := body["tools"].([]any)
	if body["tool_choice"] != "required" || len(tools) != 1 {
		return nil
	}
	t, _ := tools[0].(map[string]any)
	fn := t
	if f, ok := t["function"].(map[string]any); ok {
		fn = f
	}
	name, _ := fn["name"].(string)
	if name == "" {
		return nil
	}
	schema, _ := fn["parameters"].(map[string]any)
	if schema == nil {
		schema = map[string]any{"type": "object", "properties": map[string]any{}}
	}
	hint := fmt.Sprintf("Call the function %s: answer with only its arguments, as one JSON object.", name)
	if d, _ := fn["description"].(string); d != "" {
		hint += " The function: " + d
	}
	delete(body, "tools")
	delete(body, "tool_choice")
	delete(body, "parallel_tool_calls")
	body["response_format"] = map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "arguments", "schema": schema, "strict": true}}
	if responses {
		if ins, _ := body["instructions"].(string); ins != "" {
			hint = ins + "\n\n" + hint
		}
		body["instructions"] = hint
	} else {
		msgs, _ := body["messages"].([]any)
		var first map[string]any
		if len(msgs) > 0 {
			first, _ = msgs[0].(map[string]any)
		}
		if c, ok := first["content"].(string); ok && (first["role"] == "system" || first["role"] == "developer") {
			first["content"] = c + "\n\n" + hint
		} else {
			body["messages"] = append([]any{map[string]any{"role": "system", "content": hint}}, msgs...)
		}
	}
	return &forcedTool{name: name, callID: "call_" + randHex(12), itemID: "fc_" + randHex(12)}
}

// asToolCall turns a chat message holding the forced function's arguments
// into the tool call.
func (ft *forcedTool) asToolCall(choice map[string]any) {
	msg, _ := choice["message"].(map[string]any)
	if msg == nil {
		return
	}
	args, _ := msg["content"].(string)
	msg["tool_calls"] = []any{map[string]any{"id": ft.callID, "type": "function", "function": map[string]any{"name": ft.name, "arguments": strings.TrimSpace(args)}}}
	msg["content"] = nil
	if choice["finish_reason"] == "stop" {
		choice["finish_reason"] = "tool_calls"
	}
}

// --- files -------------------------------------------------------------------

// decodeFileData reads a file sent inline: a data: URL or bare base64.
func decodeFileData(s string) ([]byte, error) {
	if rest, ok := strings.CutPrefix(s, "data:"); ok {
		_, b64, found := strings.Cut(rest, ",")
		if !found {
			return nil, errors.New("malformed data URL")
		}
		s = b64
	}
	return base64.StdEncoding.DecodeString(strings.TrimSpace(s))
}

// fileText reads an attached file as text for the model: plain text as is,
// anything else (PDF, Word...) through Tika when it is on.
func (ws *WebServer) fileText(ctx context.Context, data []byte, name string) (string, *apiError) {
	if len(data) > maxDocBytes {
		return "", apiErrf(http.StatusRequestEntityTooLarge, "request_too_large", "%s is over %d MB", name, maxDocBytes>>20)
	}
	mime := http.DetectContentType(data)
	var text string
	switch {
	case strings.HasPrefix(mime, "text/") && utf8.Valid(data):
		text = string(data)
	case ws.tika != nil:
		ex, err := ws.tika.Extract(ctx, data, name, mime)
		if err != nil {
			return "", apiErrf(http.StatusBadRequest, "", "couldn't read %s: %v", name, err)
		}
		text = ex.Text
	default:
		return "", apiErrf(http.StatusBadRequest, "", "%s (%s) can only be read with Apache Tika, which is off: turn on the tika section in config.yaml on the server (plain-text files work without it)", name, mime)
	}
	if len(text) > maxDocTextBytes {
		cut := maxDocTextBytes
		for cut > 0 && !utf8.RuneStart(text[cut]) {
			cut--
		}
		text = text[:cut]
	}
	if name == "" {
		name = "file"
	}
	return "Attached file " + name + ":\n" + fence(text), nil
}

// filePart turns a file content part (chat "file", Responses "input_file")
// into its text.
func (ws *WebServer) filePart(ctx context.Context, f map[string]any) (string, *apiError) {
	if id, _ := f["file_id"].(string); id != "" {
		return "", apiErrf(http.StatusBadRequest, "", "file_id %s needs OpenAI's Files API, which llama-manager doesn't offer; send the file inline in file_data", id)
	}
	if u, _ := f["file_url"].(string); u != "" {
		return "", apiErrf(http.StatusBadRequest, "", "llama-manager doesn't fetch file_url; send the file inline in file_data")
	}
	raw, _ := f["file_data"].(string)
	name, _ := f["filename"].(string)
	if raw == "" {
		return "", apiErrf(http.StatusBadRequest, "", "a file part needs file_data")
	}
	data, err := decodeFileData(raw)
	if err != nil {
		return "", apiErrf(http.StatusBadRequest, "", "file_data of %s isn't base64: %v", name, err)
	}
	return ws.fileText(ctx, data, name)
}

// --- chat completions --------------------------------------------------------

// chatTx is what a chat request needs done to its answer.
type chatTx struct {
	forced   *forcedTool
	legacy   bool      // sent functions: answer function_call
	audio    *audioOut // modalities: ["audio"]
	store    bool
	metadata map[string]any
	messages []any // as sent, for a stored completion
}

// prepareChat rewrites a chat request into what llama-server takes.
func (ws *WebServer) prepareChat(ctx context.Context, body map[string]any) (chatTx, *apiError) {
	var tx chatTx
	msgs, _ := body["messages"].([]any)
	tx.messages = cloneJSON(msgs)
	if fs, ok := body["functions"].([]any); ok {
		tx.legacy = true
		var tools []any
		for _, f := range fs {
			tools = append(tools, map[string]any{"type": "function", "function": f})
		}
		body["tools"] = tools
		switch fc := body["function_call"].(type) {
		case string:
			body["tool_choice"] = fc
		case map[string]any:
			body["tool_choice"] = map[string]any{"type": "function", "function": map[string]any{"name": fc["name"]}}
		}
		delete(body, "functions")
		delete(body, "function_call")
	}
	if err := rewriteToolChoice(body); err != nil {
		return tx, err
	}
	tx.forced = forceOneTool(body, false)
	// Old-style turns: an assistant's function_call and a "function" message
	// with its result become a tool call and a tool message.
	lastCall := map[string]string{}
	for i, m := range msgs {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if fc, ok := msg["function_call"].(map[string]any); ok && role == "assistant" {
			id := fmt.Sprintf("call_legacy_%d", i)
			name, _ := fc["name"].(string)
			lastCall[name] = id
			msg["tool_calls"] = []any{map[string]any{"id": id, "type": "function", "function": fc}}
			delete(msg, "function_call")
		}
		if role == "function" {
			name, _ := msg["name"].(string)
			msg["role"] = "tool"
			msg["tool_call_id"] = lastCall[name]
			delete(msg, "name")
		}
		if a, ok := msg["audio"].(map[string]any); ok && role == "assistant" {
			id, _ := a["id"].(string)
			if msg["content"] == nil {
				msg["content"] = ws.audioTranscript(id)
			}
			delete(msg, "audio")
		}
		parts, ok := msg["content"].([]any)
		if !ok {
			continue
		}
		for j, p := range parts {
			part, _ := p.(map[string]any)
			if part == nil || part["type"] != "file" {
				continue
			}
			f, _ := part["file"].(map[string]any)
			text, aerr := ws.filePart(ctx, f)
			if aerr != nil {
				return tx, aerr
			}
			parts[j] = map[string]any{"type": "text", "text": text}
		}
	}
	if mods, ok := body["modalities"].([]any); ok {
		for _, m := range mods {
			if m == "audio" {
				ao, aerr := parseAudioOut(body["audio"])
				if aerr != nil {
					return tx, aerr
				}
				if ws.speech == nil {
					return tx, apiErrf(http.StatusBadRequest, "", "spoken replies need a voice model: turn on text_to_speech in config.yaml on the server")
				}
				if f := chatAudioFormats[ao.format]; f != "wav" && f != "pcm" && ffmpegPath() == "" {
					return tx, &apiError{status: http.StatusBadRequest, param: "audio.format", msg: fmt.Sprintf("%s audio needs ffmpeg on the server; ask for wav or pcm16, or install ffmpeg", ao.format)}
				}
				tx.audio = ao
			}
		}
	}
	delete(body, "modalities")
	delete(body, "audio")
	if s, _ := body["store"].(bool); s {
		tx.store = true
	}
	tx.metadata, _ = body["metadata"].(map[string]any)
	return tx, nil
}

func cloneJSON[T any](v T) T {
	var out T
	b, _ := json.Marshal(v)
	_ = json.Unmarshal(b, &out)
	return out
}

// toFunctionCall turns a message's (or delta's) tool calls into the old
// function_call.
func toFunctionCall(msg map[string]any) {
	calls, _ := msg["tool_calls"].([]any)
	delete(msg, "tool_calls")
	if len(calls) == 0 {
		return
	}
	c, _ := calls[0].(map[string]any)
	if f, ok := c["function"].(map[string]any); ok {
		fc := map[string]any{}
		if n, ok := f["name"].(string); ok && n != "" {
			fc["name"] = n
		}
		if a, ok := f["arguments"].(string); ok {
			fc["arguments"] = a
		}
		msg["function_call"] = fc
	}
}

func (ws *WebServer) handleV1Chat(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		ws.handleStoredChatList(w, r)
		return
	}
	if !allowMethods(w, r, http.MethodPost, http.MethodGet) {
		return
	}
	body, aerr := readBodyMap(w, r)
	if aerr != nil {
		aerr.write(w)
		return
	}
	inst, aerr := ws.bodyInstance(r.Context(), body)
	if aerr != nil {
		aerr.write(w)
		return
	}
	tx, aerr := ws.prepareChat(r.Context(), body)
	if aerr != nil {
		aerr.write(w)
		return
	}
	data, _ := json.Marshal(body)
	finish := func(c map[string]any) *apiError {
		if tx.audio != nil {
			if aerr := ws.addChatAudio(r.Context(), c, tx.audio); aerr != nil {
				return aerr
			}
		}
		if tx.store {
			ws.storeChat(c, tx)
		}
		return nil
	}
	rt := relayTx{json: func(c map[string]any) *apiError {
		if tx.forced != nil {
			for _, ch := range toItems(c["choices"]) {
				tx.forced.asToolCall(ch)
			}
		}
		if tx.legacy {
			choices, _ := c["choices"].([]any)
			for _, ch := range choices {
				choice, _ := ch.(map[string]any)
				if msg, ok := choice["message"].(map[string]any); ok {
					toFunctionCall(msg)
				}
				if choice["finish_reason"] == "tool_calls" {
					choice["finish_reason"] = "function_call"
				}
			}
		}
		return finish(c)
	}}
	if tx.forced != nil || tx.legacy || tx.audio != nil || tx.store {
		acc := &chatAccumulator{}
		started := map[float64]bool{}
		rt.event = func(ev upEvent) []upEvent {
			if ev.data == "[DONE]" {
				var out []upEvent
				final := acc.completion()
				if tx.audio != nil {
					if aerr := ws.addChatAudio(r.Context(), final, tx.audio); aerr == nil {
						if a, ok := chatMessage(final)["audio"].(map[string]any); ok {
							// The transcript went out piece by piece already.
							a = map[string]any{"id": a["id"], "data": a["data"], "expires_at": a["expires_at"]}
							out = append(out, upData("", map[string]any{"id": final["id"], "object": "chat.completion.chunk", "created": final["created"], "model": final["model"],
								"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"audio": a}, "finish_reason": nil}}}))
						}
					}
				}
				if tx.store {
					ws.storeChat(final, tx)
				}
				return append(out, ev)
			}
			var chunk map[string]any
			if json.Unmarshal([]byte(ev.data), &chunk) != nil {
				return []upEvent{ev}
			}
			if ft := tx.forced; ft != nil {
				for _, choice := range toItems(chunk["choices"]) {
					idx, _ := choice["index"].(float64)
					delta, _ := choice["delta"].(map[string]any)
					if delta != nil {
						text, _ := delta["content"].(string)
						delete(delta, "content")
						call := map[string]any{"index": 0, "function": map[string]any{"arguments": text}}
						if !started[idx] {
							started[idx] = true
							call["id"], call["type"] = ft.callID, "function"
							call["function"].(map[string]any)["name"] = ft.name
						}
						if text != "" || call["id"] != nil {
							delta["tool_calls"] = []any{call}
						}
					}
					if choice["finish_reason"] == "stop" {
						choice["finish_reason"] = "tool_calls"
					}
				}
			}
			acc.add(chunk)
			choices, _ := chunk["choices"].([]any)
			for _, ch := range choices {
				choice, _ := ch.(map[string]any)
				delta, _ := choice["delta"].(map[string]any)
				if delta != nil && tx.legacy {
					toFunctionCall(delta)
				}
				if tx.legacy && choice["finish_reason"] == "tool_calls" {
					choice["finish_reason"] = "function_call"
				}
				if delta != nil && tx.audio != nil {
					if text, ok := delta["content"].(string); ok && text != "" {
						delta["audio"] = map[string]any{"transcript": text}
						delete(delta, "content")
					}
				}
			}
			out, _ := json.Marshal(chunk)
			ev.data = string(out)
			return []upEvent{ev}
		}
	}
	relayLlama(w, r, inst, "/v1/chat/completions", data, rt)
}

func chatMessage(c map[string]any) map[string]any {
	choices, _ := c["choices"].([]any)
	if len(choices) == 0 {
		return map[string]any{}
	}
	ch, _ := choices[0].(map[string]any)
	m, _ := ch["message"].(map[string]any)
	if m == nil {
		return map[string]any{}
	}
	return m
}

// chatAccumulator rebuilds a chat.completion from its streamed chunks.
type chatAccumulator struct {
	head    map[string]any
	choices map[int]map[string]any
	order   []int
	usage   any
}

func (a *chatAccumulator) add(chunk map[string]any) {
	if a.head == nil {
		a.head = map[string]any{"id": chunk["id"], "created": chunk["created"], "model": chunk["model"], "system_fingerprint": chunk["system_fingerprint"]}
		a.choices = map[int]map[string]any{}
	}
	if u, ok := chunk["usage"]; ok && u != nil {
		a.usage = u
	}
	choices, _ := chunk["choices"].([]any)
	for _, ch := range choices {
		c, _ := ch.(map[string]any)
		idx := 0
		if f, ok := c["index"].(float64); ok {
			idx = int(f)
		}
		cur, ok := a.choices[idx]
		if !ok {
			cur = map[string]any{"index": idx, "message": map[string]any{"role": "assistant", "content": ""}, "finish_reason": nil, "logprobs": nil}
			a.choices[idx] = cur
			a.order = append(a.order, idx)
		}
		msg := cur["message"].(map[string]any)
		if d, ok := c["delta"].(map[string]any); ok {
			for _, k := range []string{"content", "reasoning_content", "refusal"} {
				if s, ok := d[k].(string); ok {
					prev, _ := msg[k].(string)
					msg[k] = prev + s
				}
			}
			if tcs, ok := d["tool_calls"].([]any); ok {
				calls, _ := msg["tool_calls"].([]any)
				for _, t := range tcs {
					tc, _ := t.(map[string]any)
					ti := 0
					if f, ok := tc["index"].(float64); ok {
						ti = int(f)
					}
					for len(calls) <= ti {
						calls = append(calls, map[string]any{"type": "function", "function": map[string]any{"name": "", "arguments": ""}})
					}
					call := calls[ti].(map[string]any)
					if id, ok := tc["id"].(string); ok {
						call["id"] = id
					}
					if f, ok := tc["function"].(map[string]any); ok {
						fn := call["function"].(map[string]any)
						if n, ok := f["name"].(string); ok {
							fn["name"] = fn["name"].(string) + n
						}
						if s, ok := f["arguments"].(string); ok {
							fn["arguments"] = fn["arguments"].(string) + s
						}
					}
				}
				msg["tool_calls"] = calls
			}
		}
		if fr, ok := c["finish_reason"]; ok && fr != nil {
			cur["finish_reason"] = fr
		}
	}
}

func (a *chatAccumulator) completion() map[string]any {
	out := map[string]any{"object": "chat.completion", "choices": []any{}}
	for k, v := range a.head {
		out[k] = v
	}
	var choices []any
	for _, idx := range a.order {
		choices = append(choices, a.choices[idx])
	}
	if choices != nil {
		out["choices"] = choices
	}
	if a.usage != nil {
		out["usage"] = a.usage
	}
	return out
}

// --- completions -------------------------------------------------------------

// promptTexts is the completion request's prompt(s) as text; token prompts are
// turned back into text by llama-server.
func promptTexts(ctx context.Context, inst *Instance, p any) ([]string, *apiError) {
	detok := func(tokens []any) (string, *apiError) {
		b, _ := json.Marshal(map[string]any{"tokens": tokens})
		resp, aerr := postLlama(ctx, inst, "/detokenize", "application/json", b)
		if aerr != nil {
			return "", aerr
		}
		defer resp.Body.Close()
		var out struct {
			Content string `json:"content"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&out)
		return out.Content, nil
	}
	switch v := p.(type) {
	case string:
		return []string{v}, nil
	case []any:
		if len(v) > 0 {
			if _, isNum := v[0].(float64); isNum {
				s, aerr := detok(v)
				return []string{s}, aerr
			}
		}
		var out []string
		for _, x := range v {
			switch y := x.(type) {
			case string:
				out = append(out, y)
			case []any:
				s, aerr := detok(y)
				if aerr != nil {
					return nil, aerr
				}
				out = append(out, s)
			}
		}
		return out, nil
	}
	return []string{""}, nil
}

func (ws *WebServer) handleV1Completions(w http.ResponseWriter, r *http.Request) {
	if !allowMethods(w, r, http.MethodPost) {
		return
	}
	body, aerr := readBodyMap(w, r)
	if aerr != nil {
		aerr.write(w)
		return
	}
	inst, aerr := ws.bodyInstance(r.Context(), body)
	if aerr != nil {
		aerr.write(w)
		return
	}
	// best_of picks the best of several server-side; llama-server has no
	// such thing (and newer builds refuse it), so the one answer is the best.
	delete(body, "best_of")
	echo, _ := body["echo"].(bool)
	delete(body, "echo")
	var prompts []string
	if echo {
		if prompts, aerr = promptTexts(r.Context(), inst, body["prompt"]); aerr != nil {
			aerr.write(w)
			return
		}
	}
	n := 1
	if f, ok := body["n"].(float64); ok && f > 1 {
		n = int(f)
	}
	promptFor := func(idx int) string {
		if len(prompts) == 0 {
			return ""
		}
		return prompts[min(idx/n, len(prompts)-1)]
	}
	if suffix, _ := body["suffix"].(string); suffix != "" {
		ws.completionInfill(w, r, inst, body, suffix, echo, prompts)
		return
	}
	delete(body, "suffix")
	data, _ := json.Marshal(body)
	var tx relayTx
	if echo {
		tx.json = func(c map[string]any) *apiError {
			choices, _ := c["choices"].([]any)
			for _, ch := range choices {
				choice, _ := ch.(map[string]any)
				idx, _ := choice["index"].(float64)
				text, _ := choice["text"].(string)
				choice["text"] = promptFor(int(idx)) + text
			}
			return nil
		}
		seen := map[int]bool{}
		tx.event = func(ev upEvent) []upEvent {
			var chunk map[string]any
			if ev.data == "[DONE]" || json.Unmarshal([]byte(ev.data), &chunk) != nil {
				return []upEvent{ev}
			}
			choices, _ := chunk["choices"].([]any)
			for _, ch := range choices {
				choice, _ := ch.(map[string]any)
				f, _ := choice["index"].(float64)
				if idx := int(f); !seen[idx] {
					seen[idx] = true
					text, _ := choice["text"].(string)
					choice["text"] = promptFor(idx) + text
				}
			}
			out, _ := json.Marshal(chunk)
			ev.data = string(out)
			return []upEvent{ev}
		}
	}
	relayLlama(w, r, inst, "/v1/completions", data, tx)
}

// completionInfill answers a completion with a suffix by llama-server's
// fill-in-the-middle (/infill): the prompt is the code before, suffix after.
func (ws *WebServer) completionInfill(w http.ResponseWriter, r *http.Request, inst *Instance, body map[string]any, suffix string, echo bool, prompts []string) {
	ps, aerr := promptTexts(r.Context(), inst, body["prompt"])
	if aerr != nil {
		aerr.write(w)
		return
	}
	if len(ps) != 1 {
		writeAPIError(w, http.StatusBadRequest, "", "prompt", "a completion with a suffix takes one prompt")
		return
	}
	if f, ok := body["n"].(float64); ok && f > 1 {
		writeAPIError(w, http.StatusBadRequest, "", "n", "a completion with a suffix makes one choice (n = 1)")
		return
	}
	stream, _ := body["stream"].(bool)
	req := map[string]any{"input_prefix": ps[0], "input_suffix": suffix, "n_predict": 16, "stream": stream, "cache_prompt": true}
	if v, ok := body["max_tokens"]; ok && v != nil {
		req["n_predict"] = v
	}
	for _, k := range []string{"temperature", "top_p", "stop", "seed", "presence_penalty", "frequency_penalty", "logit_bias"} {
		if v, ok := body[k]; ok && v != nil {
			req[k] = v
		}
	}
	data, _ := json.Marshal(req)
	resp, aerr := postLlama(r.Context(), inst, "/infill", "application/json", data)
	if aerr != nil {
		if r.Context().Err() == nil {
			aerr.write(w)
		}
		return
	}
	defer resp.Body.Close()
	id, created := "cmpl-"+randHex(12), time.Now().Unix()
	prefix := ""
	if echo {
		prefix = ps[0]
	}
	finishReason := func(m map[string]any) any {
		switch m["stop_type"] {
		case "limit":
			return "length"
		case nil, "none":
			return nil
		}
		return "stop"
	}
	usage := func(m map[string]any) map[string]any {
		p, _ := m["tokens_evaluated"].(float64)
		c, _ := m["tokens_predicted"].(float64)
		return map[string]any{"prompt_tokens": int(p), "completion_tokens": int(c), "total_tokens": int(p + c)}
	}
	chunk := func(text string, fr any) map[string]any {
		return map[string]any{"id": id, "object": "text_completion", "created": created, "model": inst.conf.Name,
			"choices": []any{map[string]any{"text": text, "index": 0, "logprobs": nil, "finish_reason": fr}}}
	}
	if !stream {
		var m map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
			writeAPIError(w, http.StatusBadGateway, "", "", inst.conf.Name+" sent an answer llama-manager can't read")
			return
		}
		content, _ := m["content"].(string)
		out := chunk(prefix+content, finishReason(m))
		out["usage"] = usage(m)
		writeJSON(w, http.StatusOK, out)
		return
	}
	includeUsage := false
	if so, ok := body["stream_options"].(map[string]any); ok {
		includeUsage, _ = so["include_usage"].(bool)
	}
	startSSE(w)
	if prefix != "" {
		writeUpEvent(w, upData("", chunk(prefix, nil)))
	}
	_ = readUpSSE(resp.Body, func(ev upEvent) error {
		var m map[string]any
		if json.Unmarshal([]byte(ev.data), &m) != nil {
			return nil
		}
		content, _ := m["content"].(string)
		stop, _ := m["stop"].(bool)
		var fr any
		if stop {
			fr = finishReason(m)
			if fr == nil {
				fr = "stop"
			}
		}
		if content != "" || stop {
			writeUpEvent(w, upData("", chunk(content, fr)))
		}
		if stop && includeUsage {
			u := chunk("", nil)
			u["choices"] = []any{}
			u["usage"] = usage(m)
			writeUpEvent(w, upData("", u))
		}
		return r.Context().Err()
	})
	writeUpEvent(w, upEvent{data: "[DONE]"})
}

// --- embeddings --------------------------------------------------------------

// shorten keeps the first n numbers of an embedding and scales it back to
// length 1, as OpenAI's dimensions does.
func shorten(v []float64, n int) []float64 {
	v = v[:n]
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	if norm := math.Sqrt(sum); norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
	return v
}

func (ws *WebServer) handleV1Embeddings(w http.ResponseWriter, r *http.Request) {
	if !allowMethods(w, r, http.MethodPost) {
		return
	}
	body, aerr := readBodyMap(w, r)
	if aerr != nil {
		aerr.write(w)
		return
	}
	inst, aerr := ws.bodyInstance(r.Context(), body)
	if aerr != nil {
		aerr.write(w)
		return
	}
	dims := 0
	if v, ok := body["dimensions"]; ok && v != nil {
		f, ok := v.(float64)
		if !ok || f < 1 || f != math.Trunc(f) {
			writeAPIError(w, http.StatusBadRequest, "", "dimensions", "dimensions must be a positive whole number")
			return
		}
		dims = int(f)
	}
	delete(body, "dimensions")
	data, _ := json.Marshal(body)
	var tx relayTx
	if dims > 0 {
		tx.json = func(m map[string]any) *apiError {
			items, _ := m["data"].([]any)
			for _, it := range items {
				e, _ := it.(map[string]any)
				switch v := e["embedding"].(type) {
				case []any:
					if dims > len(v) {
						return &apiError{status: http.StatusBadRequest, param: "dimensions", msg: fmt.Sprintf("dimensions must be at most %d for %s", len(v), inst.conf.Name)}
					}
					fs := make([]float64, len(v))
					for i, x := range v {
						fs[i], _ = x.(float64)
					}
					e["embedding"] = shorten(fs, dims)
				case string:
					raw, err := base64.StdEncoding.DecodeString(v)
					if err != nil || len(raw)%4 != 0 {
						return apiErrf(http.StatusBadGateway, "", "%s sent an embedding llama-manager can't read", inst.conf.Name)
					}
					if dims > len(raw)/4 {
						return &apiError{status: http.StatusBadRequest, param: "dimensions", msg: fmt.Sprintf("dimensions must be at most %d for %s", len(raw)/4, inst.conf.Name)}
					}
					fs := make([]float64, len(raw)/4)
					for i := range fs {
						fs[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])))
					}
					out := make([]byte, 4*dims)
					for i, x := range shorten(fs, dims) {
						binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(float32(x)))
					}
					e["embedding"] = base64.StdEncoding.EncodeToString(out)
				}
			}
			return nil
		}
	}
	relayLlama(w, r, inst, "/v1/embeddings", data, tx)
}
