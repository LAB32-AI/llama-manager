package main

import (
	"bufio"
	"crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// respFake is a llama-server /v1/responses (and chat, infill, detokenize)
// that answers "answer N" and records what it got.
type respFake struct {
	mu     sync.Mutex
	inputs [][]any          // each /v1/responses input
	bodies []map[string]any // each /v1/responses body
	n      atomic.Int64
	hold   chan struct{} // when set, /v1/responses waits for it (or the client)
}

func (f *respFake) answer(w http.ResponseWriter, r *http.Request, body map[string]any) {
	switch r.URL.Path {
	case "/v1/responses":
		f.mu.Lock()
		in, _ := body["input"].([]any)
		f.inputs = append(f.inputs, in)
		f.bodies = append(f.bodies, body)
		hold := f.hold
		f.mu.Unlock()
		n := f.n.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		item := map[string]any{"type": "message", "id": fmt.Sprintf("msg_out%d", n), "role": "assistant", "status": "completed",
			"content": []any{map[string]any{"type": "output_text", "text": fmt.Sprintf("answer %d", n), "annotations": []any{}}}}
		resp := map[string]any{"id": fmt.Sprintf("resp_up%d", n), "object": "response", "created_at": 1700000000, "status": "in_progress", "output": []any{},
			"tool_choice": body["tool_choice"], "tools": body["tools"]}
		ev := func(name string, v map[string]any) {
			b, _ := json.Marshal(v)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", name, b)
			w.(http.Flusher).Flush()
		}
		ev("response.created", map[string]any{"type": "response.created", "response": resp})
		if hold != nil {
			select {
			case <-hold:
			case <-r.Context().Done():
				return
			}
		}
		ev("response.output_item.done", map[string]any{"type": "response.output_item.done", "item": item})
		done := cloneJSON(resp)
		done["status"], done["output"] = "completed", []any{item}
		done["usage"] = map[string]any{"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
		ev("response.completed", map[string]any{"type": "response.completed", "response": done})
	case "/v1/responses/input_tokens":
		f.mu.Lock()
		f.bodies = append(f.bodies, body)
		f.mu.Unlock()
		in, _ := body["input"].([]any)
		writeJSON(w, http.StatusOK, map[string]any{"object": "response.input_tokens", "input_tokens": 10 * len(in)})
	case "/v1/chat/completions":
		writeJSON(w, http.StatusOK, map[string]any{"id": "chatcmpl-s", "object": "chat.completion", "created": 1, "model": "c",
			"choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": "SUMMARY: name is Zorbal"}, "finish_reason": "stop"}},
			"usage":   map[string]any{"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}})
	default:
		http.NotFound(w, r)
	}
}

func (f *respFake) lastInput(t *testing.T) []any {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.inputs) == 0 {
		t.Fatal("llama-server got no /v1/responses")
	}
	return f.inputs[len(f.inputs)-1]
}

func (f *respFake) lastBody() map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.bodies[len(f.bodies)-1]
}

// newStateRig is a v1 rig with a chat database (for stored state) whose
// llama-server is f.
func newStateRig(t *testing.T, f *respFake) *v1Rig {
	t.Helper()
	return newV1Rig(t, f.answer)
}

func texts(items []any) []string {
	var out []string
	for _, it := range items {
		m, _ := it.(map[string]any)
		for _, p := range toItems(m["content"]) {
			if s, ok := p["text"].(string); ok {
				out = append(out, fmt.Sprintf("%v:%s", m["role"], s))
			}
		}
	}
	return out
}

func decode(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("not JSON: %s", b)
	}
	return m
}

func TestV1ResponsesPreviousResponseID(t *testing.T) {
	f := &respFake{}
	vr := newStateRig(t, f)
	resp, b := vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"my name is Zorbal","instructions":"be brief","metadata":{"k":"v"}}`, nil)
	r1 := decode(t, b)
	if resp.StatusCode != http.StatusOK || r1["id"] != "resp_up1" || r1["status"] != "completed" || r1["store"] != true || r1["instructions"] != "be brief" ||
		r1["metadata"].(map[string]any)["k"] != "v" || r1["previous_response_id"] != nil {
		t.Fatalf("turn 1: %d %s", resp.StatusCode, b)
	}
	if got := f.lastBody(); got["stream"] != true || got["store"] != nil || got["metadata"] != nil {
		t.Errorf("llama-server got cloud-only fields or no stream: %v", got)
	}
	_, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"what is my name?","previous_response_id":"resp_up1"}`, nil)
	r2 := decode(t, b)
	if want := []string{"user:my name is Zorbal", "assistant:answer 1", "user:what is my name?"}; strings.Join(texts(f.lastInput(t)), "|") != strings.Join(want, "|") {
		t.Errorf("turn 2 input: %v", texts(f.lastInput(t)))
	}
	if r2["previous_response_id"] != "resp_up1" || f.lastBody()["previous_response_id"] != nil {
		t.Errorf("previous_response_id: %v / llama-server got %v", r2["previous_response_id"], f.lastBody()["previous_response_id"])
	}
	// Streamed, two back.
	resp, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"again","previous_response_id":"resp_up2","stream":true}`, nil)
	if resp.Header.Get("Content-Type") != "text/event-stream" || !strings.Contains(string(b), `"previous_response_id":"resp_up2"`) {
		t.Errorf("stream: %s", b)
	}
	if got := texts(f.lastInput(t)); len(got) != 5 || got[3] != "assistant:answer 2" {
		t.Errorf("turn 3 input: %v", got)
	}
	if _, b := vr.req(t, "GET", "/v1/responses/resp_up3", "", nil); decode(t, b)["status"] != "completed" {
		t.Errorf("retrieve streamed response: %s", b)
	}
	_, b = vr.req(t, "GET", "/v1/responses/resp_up2/input_items", "", nil)
	if l := decode(t, b); len(l["data"].([]any)) != 1 || l["object"] != "list" || l["has_more"] != false {
		t.Errorf("input_items: %s", b)
	}
	resp, b = vr.req(t, "POST", "/v1/responses/input_tokens", `{"model":"c","input":"x","previous_response_id":"resp_up2"}`, nil)
	if decode(t, b)["input_tokens"] != float64(50) {
		t.Errorf("input_tokens with history: %d %s", resp.StatusCode, b)
	}
	resp, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"x","previous_response_id":"resp_nope"}`, nil)
	wantAPIError(t, "unknown previous response", resp, b, http.StatusBadRequest, "")
	vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"x","store":false}`, nil)
	resp, b = vr.req(t, "GET", "/v1/responses/resp_up4", "", nil)
	wantAPIError(t, "store false", resp, b, http.StatusNotFound, "not_found")
	if resp, b := vr.req(t, "DELETE", "/v1/responses/resp_up1", "", nil); decode(t, b)["deleted"] != true {
		t.Errorf("delete: %d %s", resp.StatusCode, b)
	}
	// An older link deleted: the chain goes back as far as it still can.
	resp, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"x","previous_response_id":"resp_up2"}`, nil)
	if got := texts(f.lastInput(t)); resp.StatusCode != http.StatusOK || strings.Join(got, "|") != "user:what is my name?|assistant:answer 2|user:x" {
		t.Errorf("a chain with a deleted older link: %d %v", resp.StatusCode, got)
	}
}

func TestV1ResponsesItemReferenceAndFormat(t *testing.T) {
	f := &respFake{}
	vr := newStateRig(t, f)
	vr.req(t, "POST", "/v1/responses", `{"model":"c","input":[{"role":"user","content":"remember 42"}]}`, nil)
	_, b := vr.req(t, "GET", "/v1/responses/resp_up1/input_items", "", nil)
	id := decode(t, b)["data"].([]any)[0].(map[string]any)["id"].(string)
	vr.req(t, "POST", "/v1/responses", `{"model":"c","input":[{"type":"item_reference","id":"`+id+`"},{"type":"item_reference","id":"msg_out1"},{"role":"user","content":"which number?"}],
		"text":{"format":{"type":"json_schema","name":"n","schema":{"type":"object"},"strict":true}}}`, nil)
	if got := texts(f.lastInput(t)); strings.Join(got, "|") != "user:remember 42|assistant:answer 1|user:which number?" {
		t.Errorf("item_reference: %v", got)
	}
	body := f.lastBody()
	rf, _ := body["response_format"].(map[string]any)
	if rf["type"] != "json_schema" || rf["json_schema"].(map[string]any)["name"] != "n" || body["text"] != nil {
		t.Errorf("text.format -> response_format: %v", body)
	}
	// One function demanded: the model writes its arguments under its schema
	// and the answer becomes the call.
	_, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"go","instructions":"be brief","tools":[{"type":"function","name":"a"},{"type":"function","name":"b","parameters":{"type":"object","properties":{"x":{"type":"string"}}}}],"tool_choice":{"type":"function","name":"b"}}`, nil)
	body = f.lastBody()
	rf, _ = body["response_format"].(map[string]any)
	if body["tools"] != nil || body["tool_choice"] != nil || rf["json_schema"].(map[string]any)["schema"].(map[string]any)["properties"] == nil ||
		!strings.HasPrefix(fmt.Sprint(body["instructions"]), "be brief\n\nCall the function b") {
		t.Errorf("forced function request: %v", body)
	}
	out := decode(t, b)["output"].([]any)[0].(map[string]any)
	if out["type"] != "function_call" || out["name"] != "b" || out["arguments"] != "answer 3" || !strings.HasPrefix(fmt.Sprint(out["call_id"]), "call_") {
		t.Errorf("forced function answer: %s", b)
	}
	if tc, _ := decode(t, b)["tool_choice"].(map[string]any); tc["name"] != "b" {
		t.Errorf("the response shows tool_choice as sent: %s", b)
	}
	_, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"go","stream":true,"tools":[{"type":"function","name":"b"}],"tool_choice":"required"}`, nil)
	if s := string(b); !strings.Contains(s, "event: response.output_item.done") || !strings.Contains(s, `"type":"function_call"`) || strings.Contains(s, "content_part") || strings.Contains(s, `"type":"message"`) {
		t.Errorf("forced function stream: %s", b)
	}
	resp, b := vr.req(t, "POST", "/v1/responses", `{"model":"c","input":[{"type":"item_reference","id":"msg_nope"}]}`, nil)
	wantAPIError(t, "unknown item", resp, b, http.StatusBadRequest, "")
	resp, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"x","prompt":{"id":"pmpt_1"}}`, nil)
	wantAPIError(t, "prompt templates", resp, b, http.StatusBadRequest, "")
	resp, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"x","tools":[{"type":"function","name":"a"}],"tool_choice":{"type":"web_search"}}`, nil)
	wantAPIError(t, "hosted tool choice", resp, b, http.StatusBadRequest, "")
}

func TestV1Conversations(t *testing.T) {
	f := &respFake{}
	vr := newStateRig(t, f)
	resp, b := vr.req(t, "POST", "/v1/conversations", `{"items":[{"type":"message","role":"user","content":"my colour is teal"}],"metadata":{"a":"1"}}`, nil)
	conv := decode(t, b)
	id, _ := conv["id"].(string)
	if resp.StatusCode != http.StatusOK || !strings.HasPrefix(id, "conv_") || conv["object"] != "conversation" {
		t.Fatalf("create: %d %s", resp.StatusCode, b)
	}
	_, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","conversation":"`+id+`","input":"what colour?"}`, nil)
	if c, _ := decode(t, b)["conversation"].(map[string]any); c["id"] != id {
		t.Errorf("response.conversation: %s", b)
	}
	vr.req(t, "POST", "/v1/responses", `{"model":"c","conversation":{"id":"`+id+`"},"input":"again"}`, nil)
	if got := texts(f.lastInput(t)); strings.Join(got, "|") != "user:my colour is teal|user:what colour?|assistant:answer 1|user:again" {
		t.Errorf("second turn input: %v", got)
	}
	_, b = vr.req(t, "GET", "/v1/conversations/"+id+"/items?order=asc&limit=2", "", nil)
	page := decode(t, b)
	data := page["data"].([]any)
	if len(data) != 2 || page["has_more"] != true || data[0].(map[string]any)["role"] != "user" {
		t.Fatalf("items page: %s", b)
	}
	_, b = vr.req(t, "GET", "/v1/conversations/"+id+"/items?order=asc&after="+page["last_id"].(string), "", nil)
	if n := len(decode(t, b)["data"].([]any)); n != 3 {
		t.Errorf("next page: %d items", n)
	}
	first := data[0].(map[string]any)["id"].(string)
	if _, b := vr.req(t, "GET", "/v1/conversations/"+id+"/items/"+first, "", nil); decode(t, b)["id"] != first {
		t.Errorf("item retrieve: %s", b)
	}
	if _, b := vr.req(t, "POST", "/v1/conversations/"+id+"/items", `{"items":[{"role":"user","content":"x"}]}`, nil); len(decode(t, b)["data"].([]any)) != 1 {
		t.Errorf("items create: %s", b)
	}
	if _, b := vr.req(t, "DELETE", "/v1/conversations/"+id+"/items/"+first, "", nil); decode(t, b)["object"] != "conversation" {
		t.Errorf("item delete: %s", b)
	}
	if _, b := vr.req(t, "POST", "/v1/conversations/"+id, `{"metadata":{"a":"2"}}`, nil); decode(t, b)["metadata"].(map[string]any)["a"] != "2" {
		t.Errorf("update: %s", b)
	}
	if _, b := vr.req(t, "DELETE", "/v1/conversations/"+id, "", nil); decode(t, b)["object"] != "conversation.deleted" {
		t.Errorf("delete: %s", b)
	}
	resp, b = vr.req(t, "GET", "/v1/conversations/"+id, "", nil)
	wantAPIError(t, "deleted conversation", resp, b, http.StatusNotFound, "not_found")
	resp, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","conversation":"`+id+`","input":"x"}`, nil)
	wantAPIError(t, "response in a deleted conversation", resp, b, http.StatusNotFound, "not_found")
	resp, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","conversation":"conv_x","previous_response_id":"resp_up1","input":"x"}`, nil)
	wantAPIError(t, "conversation with previous_response_id", resp, b, http.StatusBadRequest, "")
}

func TestV1ResponsesBackground(t *testing.T) {
	f := &respFake{}
	vr := newStateRig(t, f)
	_, b := vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"x","background":true}`, nil)
	r := decode(t, b)
	id, _ := r["id"].(string)
	if r["status"] != "queued" || r["background"] != true || !strings.HasPrefix(id, "resp_") {
		t.Fatalf("queued: %s", b)
	}
	waitFor(t, 3*time.Second, "completed", func() bool {
		_, b := vr.req(t, "GET", "/v1/responses/"+id, "", nil)
		return decode(t, b)["status"] == "completed"
	})
	_, b = vr.req(t, "GET", "/v1/responses/"+id, "", nil)
	if done := decode(t, b); done["id"] != id || len(done["output"].([]any)) != 1 {
		t.Errorf("finished: %s", b)
	}

	f.mu.Lock()
	f.hold = make(chan struct{})
	f.mu.Unlock()
	t.Cleanup(func() { close(f.hold) })
	_, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"long","background":true}`, nil)
	id = decode(t, b)["id"].(string)
	waitFor(t, 3*time.Second, "in progress", func() bool {
		_, b := vr.req(t, "GET", "/v1/responses/"+id, "", nil)
		return decode(t, b)["status"] == "in_progress"
	})
	if _, b := vr.req(t, "POST", "/v1/responses/"+id+"/cancel", "", nil); decode(t, b)["status"] != "cancelled" {
		t.Errorf("cancel: %s", b)
	}
	resp, b := vr.req(t, "POST", "/v1/responses/resp_up1/cancel", "", nil)
	wantAPIError(t, "cancel a foreground response", resp, b, http.StatusNotFound, "not_found")
	resp, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"x","background":true,"store":false}`, nil)
	wantAPIError(t, "background without store", resp, b, http.StatusBadRequest, "")

	vr.ws.startAPIState(t.Context())
	_, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"cut off","background":true}`, nil)
	id = decode(t, b)["id"].(string)
	waitFor(t, 3*time.Second, "in progress", func() bool {
		_, b := vr.req(t, "GET", "/v1/responses/"+id, "", nil)
		return decode(t, b)["status"] == "in_progress"
	})
	vr.ws.startAPIState(t.Context()) // as after a restart
	if _, b := vr.req(t, "GET", "/v1/responses/"+id, "", nil); decode(t, b)["status"] != "failed" {
		t.Errorf("a response cut off by a restart: %s", b)
	}
}

func TestV1ResponsesCompact(t *testing.T) {
	f := &respFake{}
	vr := newStateRig(t, f)
	vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"my name is Zorbal"}`, nil)
	resp, b := vr.req(t, "POST", "/v1/responses/compact", `{"model":"c","previous_response_id":"resp_up1","input":"and my dog is Rex"}`, nil)
	c := decode(t, b)
	out, _ := c["output"].([]any)
	if resp.StatusCode != http.StatusOK || c["object"] != "response.compaction" || len(out) != 3 || out[2].(map[string]any)["type"] != "compaction" {
		t.Fatalf("compact: %d %s", resp.StatusCode, b)
	}
	in, _ := json.Marshal(append(out, map[string]any{"role": "user", "content": "names?"}))
	vr.req(t, "POST", "/v1/responses", `{"model":"c","input":`+string(in)+`}`, nil)
	got := texts(f.lastInput(t))
	if len(got) != 4 || got[2] != "user:Summary of the earlier conversation:\nSUMMARY: name is Zorbal" {
		t.Errorf("compacted input: %q", got)
	}
	resp, b = vr.req(t, "POST", "/v1/responses", `{"model":"c","input":[{"type":"compaction","encrypted_content":"gAAAA-from-openai"}]}`, nil)
	wantAPIError(t, "foreign compaction", resp, b, http.StatusBadRequest, "")
}

func TestV1ResponsesWebSocket(t *testing.T) {
	f := &respFake{}
	vr := newStateRig(t, f)
	setKeys(vr.ws, "sk-ws")
	conn, br := wsDial(t, vr.srv.URL, "/v1/responses", "openai-insecure-api-key.sk-ws", "")
	defer conn.Close()
	wsSend(t, conn, `{"type":"response.create","model":"c","input":"hi","stream_id":"lane1"}`, true)
	var types []string
	for {
		var ev map[string]any
		json.Unmarshal(wsRecv(t, br), &ev)
		if ev["stream_id"] != "lane1" {
			t.Fatalf("event without the lane: %v", ev)
		}
		types = append(types, ev["type"].(string))
		if ev["type"] == "response.completed" {
			break
		}
	}
	if types[0] != "response.created" {
		t.Errorf("events: %v", types)
	}
	wsSend(t, conn, `{"type":"response.steer","input":"x"}`, false)
	var ev map[string]any
	json.Unmarshal(wsRecv(t, br), &ev)
	if ev["type"] != "error" || ev["error"].(map[string]any)["message"] == "" {
		t.Errorf("steer: %v", ev)
	}
	if _, b := vr.req(t, "GET", "/v1/responses/resp_up1", "", map[string]string{"Authorization": "Bearer sk-ws"}); decode(t, b)["status"] != "completed" {
		t.Errorf("a WebSocket response is stored: %s", b)
	}
	_, _, status := wsHandshake(t, vr.srv.URL, "/v1/responses", "openai-insecure-api-key.wrong", "")
	if !strings.Contains(status, "401") {
		t.Errorf("wrong key: %s", status)
	}
	_, _, status = wsHandshake(t, vr.srv.URL, "/v1/responses", "openai-insecure-api-key.sk-ws", "http://evil.example")
	if !strings.Contains(status, "403") {
		t.Errorf("another site: %s", status)
	}
}

// wsHandshake opens a WebSocket to path, offering protocols.
func wsHandshake(t *testing.T, base, path, protocols, origin string) (net.Conn, *bufio.Reader, string) {
	t.Helper()
	host := strings.TrimPrefix(base, "http://")
	conn, err := net.Dial("tcp", host)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(conn, "GET %s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Protocol: %s\r\n", path, host, protocols)
	if origin != "" {
		fmt.Fprintf(conn, "Origin: %s\r\n", origin)
	}
	fmt.Fprint(conn, "\r\n")
	br := bufio.NewReader(conn)
	status, _ := br.ReadString('\n')
	for {
		l, err := br.ReadString('\n')
		if err != nil || l == "\r\n" {
			break
		}
	}
	if !strings.Contains(status, "101") {
		conn.Close()
	}
	return conn, br, status
}

func wsDial(t *testing.T, base, path, protocols, origin string) (net.Conn, *bufio.Reader) {
	t.Helper()
	conn, br, status := wsHandshake(t, base, path, protocols, origin)
	if !strings.Contains(status, "101") {
		t.Fatalf("handshake: %s", status)
	}
	return conn, br
}

// wsSend sends a masked text message, in two fragments with a ping between
// them when split is set.
func wsSend(t *testing.T, c net.Conn, msg string, split bool) {
	t.Helper()
	frame := func(fin bool, op byte, p []byte) []byte {
		b := []byte{op}
		if fin {
			b[0] |= 0x80
		}
		switch n := len(p); {
		case n < 126:
			b = append(b, 0x80|byte(n))
		default:
			b = append(b, 0x80|126, byte(n>>8), byte(n))
		}
		var mask [4]byte
		rand.Read(mask[:])
		b = append(b, mask[:]...)
		for i, x := range p {
			b = append(b, x^mask[i%4])
		}
		return b
	}
	var out []byte
	if split {
		half := len(msg) / 2
		out = append(out, frame(false, opText, []byte(msg[:half]))...)
		out = append(out, frame(true, opPing, []byte("hi"))...)
		out = append(out, frame(true, opCont, []byte(msg[half:]))...)
	} else {
		out = frame(true, opText, []byte(msg))
	}
	if _, err := c.Write(out); err != nil {
		t.Fatal(err)
	}
}

// wsRecv reads the next text message from the server, skipping pongs.
func wsRecv(t *testing.T, br *bufio.Reader) []byte {
	t.Helper()
	for {
		var h [2]byte
		if _, err := io.ReadFull(br, h[:]); err != nil {
			t.Fatal(err)
		}
		n := uint64(h[1] & 0x7F)
		switch n {
		case 126:
			var b [2]byte
			io.ReadFull(br, b[:])
			n = uint64(binary.BigEndian.Uint16(b[:]))
		case 127:
			var b [8]byte
			io.ReadFull(br, b[:])
			n = binary.BigEndian.Uint64(b[:])
		}
		p := make([]byte, n)
		io.ReadFull(br, p)
		if h[0]&0x0F == opText {
			return p
		}
	}
}

func TestV1ChatRewrites(t *testing.T) {
	var got []map[string]any
	var mu sync.Mutex
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		mu.Lock()
		got = append(got, body)
		mu.Unlock()
		call := []any{map[string]any{"index": 0, "id": "call_1", "type": "function", "function": map[string]any{"name": "get_time", "arguments": `{"city":"Paris"}`}}}
		if body["response_format"] != nil { // a forced function: its arguments as JSON
			if body["stream"] == true {
				sse(w, `{"id":"c9","object":"chat.completion.chunk","created":5,"model":"c","choices":[{"index":0,"delta":{"role":"assistant","content":"{\"city\":"}}]}`,
					`{"id":"c9","object":"chat.completion.chunk","created":5,"model":"c","choices":[{"index":0,"delta":{"content":"\"Paris\"}"},"finish_reason":"stop"}]}`, "[DONE]")
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{"id": "c9", "object": "chat.completion", "created": 5, "model": "c",
				"choices": []any{map[string]any{"index": 0, "finish_reason": "stop", "message": map[string]any{"role": "assistant", "content": `{"city":"Paris"}`}}}})
			return
		}
		if body["stream"] == true {
			sse(w, `{"id":"c9","object":"chat.completion.chunk","created":5,"model":"c","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":`+toJSON(call)+`},"finish_reason":null}]}`,
				`{"id":"c9","object":"chat.completion.chunk","created":5,"model":"c","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`, "[DONE]")
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"id": "c9", "object": "chat.completion", "created": 5, "model": "c",
			"choices": []any{map[string]any{"index": 0, "finish_reason": "tool_calls", "message": map[string]any{"role": "assistant", "content": nil, "tool_calls": call}}}})
	})
	last := func() map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return got[len(got)-1]
	}
	fn := `{"name":"get_time","parameters":{"type":"object"}}`
	_, b := vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[{"role":"user","content":"t"},{"role":"assistant","content":null,"function_call":{"name":"get_time","arguments":"{}"}},{"role":"function","name":"get_time","content":"14:05"}],
		"functions":[`+fn+`,{"name":"other"}],"function_call":{"name":"get_time"}}`, nil)
	msg := chatMessage(decode(t, b))
	if fc, _ := msg["function_call"].(map[string]any); fc["name"] != "get_time" || msg["tool_calls"] != nil || !strings.Contains(string(b), `"finish_reason":"function_call"`) {
		t.Errorf("legacy answer: %s", b)
	}
	up := last()
	msgs := up["messages"].([]any)
	if up["functions"] != nil || up["tools"] != nil || up["response_format"] == nil || msgs[0].(map[string]any)["role"] != "system" ||
		msgs[2].(map[string]any)["tool_calls"] == nil || msgs[3].(map[string]any)["role"] != "tool" || msgs[3].(map[string]any)["tool_call_id"] != msgs[2].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)["id"] {
		t.Errorf("legacy request: %v", up)
	}
	if fc, _ := msg["function_call"].(map[string]any); fc["arguments"] != `{"city":"Paris"}` {
		t.Errorf("forced legacy arguments: %v", msg)
	}
	_, b = vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[{"role":"system","content":"sys"}],"tools":[{"type":"function","function":{"name":"a"}},{"type":"function","function":`+fn+`}],"tool_choice":{"type":"function","function":{"name":"get_time"}}}`, nil)
	msg = chatMessage(decode(t, b))
	if tc := msg["tool_calls"].([]any)[0].(map[string]any); tc["function"].(map[string]any)["name"] != "get_time" || tc["function"].(map[string]any)["arguments"] != `{"city":"Paris"}` || msg["content"] != nil || !strings.Contains(string(b), `"finish_reason":"tool_calls"`) {
		t.Errorf("forced tool: %s", b)
	}
	if m0 := last()["messages"].([]any)[0].(map[string]any); !strings.HasPrefix(fmt.Sprint(m0["content"]), "sys\n\nCall the function get_time") {
		t.Errorf("hint joins the system message: %v", m0)
	}
	_, b = vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","stream":true,"messages":[],"tools":[{"type":"function","function":`+fn+`}],"tool_choice":"required"}`, nil)
	if s := string(b); !strings.Contains(s, `"name":"get_time"`) || !strings.Contains(s, `"arguments":"{\"city\":"`) || strings.Contains(s, `"content":"{`) || !strings.Contains(s, `"finish_reason":"tool_calls"`) {
		t.Errorf("forced tool stream: %s", b)
	}
	_, b = vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","stream":true,"messages":[],"functions":[`+fn+`]}`, nil)
	if !strings.Contains(string(b), `"function_call":{"arguments":"{\"city\":\"Paris\"}","name":"get_time"}`) || strings.Contains(string(b), "tool_calls\":[") || !strings.Contains(string(b), `"finish_reason":"function_call"`) {
		t.Errorf("legacy stream: %s", b)
	}
	vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[],"tools":[{"type":"function","function":{"name":"a"}},{"type":"function","function":{"name":"b"}}],
		"tool_choice":{"type":"allowed_tools","allowed_tools":{"mode":"auto","tools":[{"type":"function","function":{"name":"b"}}]}}}`, nil)
	if up := last(); up["tool_choice"] != "auto" || len(up["tools"].([]any)) != 1 {
		t.Errorf("allowed_tools: %v", up)
	}
	resp, b := vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[],"tools":[{"type":"function","function":{"name":"a"}}],"tool_choice":{"type":"function","function":{"name":"zzz"}}}`, nil)
	wantAPIError(t, "tool_choice names no tool", resp, b, http.StatusBadRequest, "")

	txt := base64.StdEncoding.EncodeToString([]byte("secret 4471"))
	vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[{"role":"user","content":[{"type":"file","file":{"filename":"a.txt","file_data":"data:text/plain;base64,`+txt+`"}}]}]}`, nil)
	if p := last()["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any); p["type"] != "text" || !strings.Contains(p["text"].(string), "secret 4471") {
		t.Errorf("file part: %v", p)
	}
	pdf := base64.StdEncoding.EncodeToString([]byte("%PDF-1.4\n"))
	resp, b = vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[{"role":"user","content":[{"type":"file","file":{"filename":"a.pdf","file_data":"`+pdf+`"}}]}]}`, nil)
	if e := wantAPIError(t, "PDF without Tika", resp, b, http.StatusBadRequest, ""); !strings.Contains(e.Error.Message, "Tika") {
		t.Errorf("should say Tika: %s", b)
	}
	resp, b = vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[{"role":"user","content":[{"type":"file","file":{"file_id":"file-1"}}]}]}`, nil)
	wantAPIError(t, "file_id", resp, b, http.StatusBadRequest, "")
}

func TestV1StoredChatCompletions(t *testing.T) {
	n := 0
	var mu sync.Mutex
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		mu.Lock()
		n++
		id := fmt.Sprintf("chatcmpl-%d", n)
		mu.Unlock()
		if body["stream"] == true {
			sse(w, `{"id":"`+id+`","object":"chat.completion.chunk","created":7,"model":"c","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
				`{"id":"`+id+`","object":"chat.completion.chunk","created":7,"model":"c","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}`, "[DONE]")
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"id": id, "object": "chat.completion", "created": 7, "model": "c",
			"choices": []any{map[string]any{"index": 0, "finish_reason": "stop", "message": map[string]any{"role": "assistant", "content": "Hi"}}}})
	})
	vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","store":true,"metadata":{"app":"a"},"messages":[{"role":"user","content":[{"type":"text","text":"hey"}]}]}`, nil)
	vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","store":true,"stream":true,"messages":[{"role":"user","content":"yo"}]}`, nil)
	vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[{"role":"user","content":"not stored"}]}`, nil)
	_, b := vr.req(t, "GET", "/v1/chat/completions", "", nil)
	if l := decode(t, b); len(l["data"].([]any)) != 2 || l["first_id"] != "chatcmpl-1" {
		t.Fatalf("list: %s", b)
	}
	if _, b := vr.req(t, "GET", "/v1/chat/completions?metadata[app]=a", "", nil); len(decode(t, b)["data"].([]any)) != 1 {
		t.Errorf("metadata filter: %s", b)
	}
	if _, b := vr.req(t, "GET", "/v1/chat/completions/chatcmpl-2", "", nil); chatMessage(decode(t, b))["content"] != "Hello" {
		t.Errorf("streamed completion stored whole: %s", b)
	}
	_, b = vr.req(t, "GET", "/v1/chat/completions/chatcmpl-1/messages", "", nil)
	m := decode(t, b)["data"].([]any)[0].(map[string]any)
	if m["id"] != "chatcmpl-1-0" || m["content"] != "hey" || m["content_parts"] == nil {
		t.Errorf("messages: %s", b)
	}
	if _, b := vr.req(t, "POST", "/v1/chat/completions/chatcmpl-1", `{"metadata":{"app":"b"}}`, nil); decode(t, b)["metadata"].(map[string]any)["app"] != "b" {
		t.Errorf("update: %s", b)
	}
	if _, b := vr.req(t, "DELETE", "/v1/chat/completions/chatcmpl-1", "", nil); decode(t, b)["object"] != "chat.completion.deleted" {
		t.Errorf("delete: %s", b)
	}
	resp, b := vr.req(t, "GET", "/v1/chat/completions/chatcmpl-3", "", nil)
	wantAPIError(t, "unstored", resp, b, http.StatusNotFound, "not_found")
}

func TestV1CompletionsEchoSuffixBestOf(t *testing.T) {
	var got []map[string]any
	var paths []string
	var mu sync.Mutex
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		mu.Lock()
		got, paths = append(got, body), append(paths, r.URL.Path)
		mu.Unlock()
		switch r.URL.Path {
		case "/detokenize":
			writeJSON(w, http.StatusOK, map[string]any{"content": "from tokens"})
		case "/infill":
			if body["stream"] == true {
				sse(w, `{"content":"a + b","stop":false}`, `{"content":"","stop":true,"stop_type":"eos","tokens_predicted":3,"tokens_evaluated":9}`)
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{"content": "a + b", "stop": true, "stop_type": "limit", "tokens_predicted": 3, "tokens_evaluated": 9})
		default:
			if body["stream"] == true {
				sse(w, `{"id":"x","object":"text_completion","choices":[{"index":0,"text":" Paris"}]}`, `{"id":"x","object":"text_completion","choices":[{"index":0,"text":"!"}]}`, "[DONE]")
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{"id": "x", "object": "text_completion", "choices": []any{map[string]any{"index": 0, "text": " Paris"}}})
		}
	})
	last := func() (string, map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		return paths[len(paths)-1], got[len(got)-1]
	}
	_, b := vr.req(t, "POST", "/v1/completions", `{"model":"c","prompt":"Capital:","echo":true,"best_of":3}`, nil)
	if !strings.Contains(string(b), `"text":"Capital: Paris"`) {
		t.Errorf("echo: %s", b)
	}
	if _, body := last(); body["echo"] != nil || body["best_of"] != nil {
		t.Errorf("llama-server got echo/best_of: %v", body)
	}
	_, b = vr.req(t, "POST", "/v1/completions", `{"model":"c","prompt":"Capital:","echo":true,"stream":true}`, nil)
	if !strings.Contains(string(b), `"text":"Capital: Paris"`) || strings.Count(string(b), "Capital:") != 1 {
		t.Errorf("echo stream: %s", b)
	}
	_, b = vr.req(t, "POST", "/v1/completions", `{"model":"c","prompt":[1,2,3],"echo":true}`, nil)
	if !strings.Contains(string(b), `"text":"from tokens Paris"`) {
		t.Errorf("echo of token prompt: %s", b)
	}
	_, b = vr.req(t, "POST", "/v1/completions", `{"model":"c","prompt":"def f(a, b):\n    return ","suffix":"\n","max_tokens":5}`, nil)
	c := decode(t, b)
	if p, body := last(); p != "/infill" || body["input_prefix"] != "def f(a, b):\n    return " || body["input_suffix"] != "\n" || body["n_predict"] != float64(5) {
		t.Errorf("infill request: %s %v", p, body)
	}
	if c["object"] != "text_completion" || !strings.Contains(string(b), `"finish_reason":"length"`) || !strings.Contains(string(b), `"text":"a + b"`) ||
		c["usage"].(map[string]any)["total_tokens"] != float64(12) {
		t.Errorf("infill answer: %s", b)
	}
	_, b = vr.req(t, "POST", "/v1/completions", `{"model":"c","prompt":"x","suffix":"y","stream":true,"stream_options":{"include_usage":true}}`, nil)
	if s := string(b); !strings.Contains(s, `"text":"a + b"`) || !strings.Contains(s, `"finish_reason":"stop"`) || !strings.Contains(s, `"total_tokens":12`) || !strings.HasSuffix(strings.TrimSpace(s), "data: [DONE]") {
		t.Errorf("infill stream: %s", b)
	}
}

func TestV1EmbeddingsDimensions(t *testing.T) {
	vec := []float64{3, 4, 12, 0}
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if body["dimensions"] != nil {
			t.Errorf("llama-server got dimensions")
		}
		var emb any = vec
		if body["encoding_format"] == "base64" {
			raw := make([]byte, 4*len(vec))
			for i, x := range vec {
				binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(float32(x)))
			}
			emb = base64.StdEncoding.EncodeToString(raw)
		}
		writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": []any{map[string]any{"object": "embedding", "index": 0, "embedding": emb}}})
	})
	_, b := vr.req(t, "POST", "/v1/embeddings", `{"model":"c","input":"x","dimensions":2}`, nil)
	e := decode(t, b)["data"].([]any)[0].(map[string]any)["embedding"].([]any)
	if len(e) != 2 || math.Abs(e[0].(float64)-0.6) > 1e-9 || math.Abs(e[1].(float64)-0.8) > 1e-9 {
		t.Errorf("float: %v", e)
	}
	_, b = vr.req(t, "POST", "/v1/embeddings", `{"model":"c","input":"x","dimensions":2,"encoding_format":"base64"}`, nil)
	raw, _ := base64.StdEncoding.DecodeString(decode(t, b)["data"].([]any)[0].(map[string]any)["embedding"].(string))
	if len(raw) != 8 || math.Abs(float64(math.Float32frombits(binary.LittleEndian.Uint32(raw)))-0.6) > 1e-6 {
		t.Errorf("base64: %d bytes", len(raw))
	}
	resp, b := vr.req(t, "POST", "/v1/embeddings", `{"model":"c","input":"x","dimensions":5}`, nil)
	wantAPIError(t, "too many", resp, b, http.StatusBadRequest, "")
	resp, b = vr.req(t, "POST", "/v1/embeddings", `{"model":"c","input":"x","dimensions":0}`, nil)
	wantAPIError(t, "zero", resp, b, http.StatusBadRequest, "")
}

func TestV1UpstreamErrorsInOpenAIShape(t *testing.T) {
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		switch body["model"] {
		case "c":
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]any{"code": 500, "message": "image input is not supported - hint: mmproj", "type": "server_error"}})
		}
	})
	resp, b := vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","messages":[]}`, nil)
	e := wantAPIError(t, "not supported", resp, b, http.StatusBadRequest, "")
	if e.Error.Code != nil || !strings.Contains(string(b), `"code":null`) {
		t.Errorf("a numeric code must go: %s", b)
	}
}

func TestV1ChatAudioReplies(t *testing.T) {
	sr := newSpeechRig(t, SpeechModelConf{Model: "lfm3-audio"})
	noFFmpeg(t)
	sr.replyMu.Lock()
	sr.reply = "Hello there."
	sr.replyMu.Unlock()
	resp, b := sr.do(t, "POST", "/v1/chat/completions", map[string]any{"model": "c", "modalities": []string{"text", "audio"}, "audio": map[string]any{"voice": "alloy", "format": "wav"},
		"messages": []any{map[string]any{"role": "user", "content": "hi"}}})
	msg := chatMessage(decode(t, b))
	a, _ := msg["audio"].(map[string]any)
	data, _ := base64.StdEncoding.DecodeString(fmt.Sprint(a["data"]))
	if resp.StatusCode != http.StatusOK || msg["content"] != nil || !isWAV(data) || a["transcript"] != "T" || !strings.HasPrefix(fmt.Sprint(a["id"]), "audio_") {
		t.Fatalf("audio reply: %d %s", resp.StatusCode, b)
	}
	if got := sr.ws.audioTranscript(a["id"].(string)); got != "T" {
		t.Errorf("transcript kept for the next turn: %q", got)
	}
	_, b = sr.do(t, "POST", "/v1/chat/completions", map[string]any{"model": "c", "stream": true, "modalities": []string{"text", "audio"}, "audio": map[string]any{"voice": "alloy", "format": "pcm16"},
		"messages": []any{map[string]any{"role": "user", "content": "hi"}}})
	s := string(b)
	if !strings.Contains(s, `"audio":{"transcript":"Hello there."}`) || !strings.Contains(s, `"data":"`) || strings.Count(s, "Hello there.") != 1 || strings.Contains(s, `"content":"Hello`) {
		t.Errorf("audio stream: %s", s)
	}
	resp, b = sr.do(t, "POST", "/v1/chat/completions", map[string]any{"model": "c", "modalities": []string{"audio"}, "audio": map[string]any{"format": "ogg"}, "messages": []any{}})
	wantAPIError(t, "bad format", resp, b, http.StatusBadRequest, "")
	vr := newV1Rig(t, nil)
	resp, b = vr.req(t, "POST", "/v1/chat/completions", `{"model":"c","modalities":["audio"],"audio":{"voice":"alloy","format":"wav"},"messages":[]}`, nil)
	wantAPIError(t, "no voice model", resp, b, http.StatusBadRequest, "")
}

func TestV1TranscriptionUsesLlamaServersRoute(t *testing.T) {
	var paths []string
	var mu sync.Mutex
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		mu.Lock()
		paths = append(paths, r.URL.Path)
		mu.Unlock()
		switch r.URL.Path {
		case "/props":
			writeJSON(w, http.StatusOK, map[string]any{"modalities": map[string]bool{"audio": true}})
		case "/v1/audio/transcriptions":
			writeJSON(w, http.StatusOK, map[string]any{"text": "native words."})
		case "/v1/chat/completions":
			msgs, _ := body["messages"].([]any)
			instr := ""
			if len(msgs) > 0 {
				parts, _ := msgs[0].(map[string]any)["content"].([]any)
				instr, _ = parts[1].(map[string]any)["text"].(string)
			}
			writeJSON(w, http.StatusOK, map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "chat: " + instr}}}})
		}
	})
	url := vr.srv.URL + "/v1/audio/transcriptions"
	wav := testWAV(16000, 1, 1600)
	_, b := multipartReq(t, url, map[string]string{"languages[]": "de"}, map[string][]byte{"file": wav}, nil)
	if !strings.Contains(string(b), "native words.") {
		t.Errorf("native: %s", b)
	}
	_, b = multipartReq(t, url, map[string]string{"keywords[]": "Zorbal"}, map[string][]byte{"file": wav}, nil)
	if !strings.Contains(string(b), "Zorbal") || !strings.Contains(string(b), "chat: ") {
		t.Errorf("keywords go through the chat route: %s", b)
	}
	_, b = multipartReq(t, url, map[string]string{"prompt": "earlier text"}, map[string][]byte{"file": wav}, nil)
	if !strings.Contains(string(b), "earlier text") {
		t.Errorf("prompt goes through the chat route: %s", b)
	}
	// A llama-server without the route: the chat way.
	old := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		switch r.URL.Path {
		case "/props":
			writeJSON(w, http.StatusOK, map[string]any{"modalities": map[string]bool{"audio": true}})
		case "/v1/chat/completions":
			writeJSON(w, http.StatusOK, map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "chat words."}}}})
		default:
			http.NotFound(w, r)
		}
	})
	_, b = multipartReq(t, old.srv.URL+"/v1/audio/transcriptions", nil, map[string][]byte{"file": wav}, nil)
	if !strings.Contains(string(b), "chat words.") {
		t.Errorf("fallback: %s", b)
	}
}

func TestV1RealtimeClientSecrets(t *testing.T) {
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if r.URL.Path == "/v1/realtime" {
			conn, brw, err := http.NewResponseController(w).Hijack()
			if err == nil {
				brw.WriteString("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
				brw.Flush()
				conn.Close()
			}
		}
	})
	setKeys(vr.ws, "sk-real")
	auth := map[string]string{"Authorization": "Bearer sk-real"}
	resp, b := vr.req(t, "POST", "/v1/realtime/client_secrets", `{"expires_after":{"anchor":"created_at","seconds":60},"session":{"type":"realtime","model":"c"}}`, auth)
	cs := decode(t, b)
	ek, _ := cs["value"].(string)
	if resp.StatusCode != http.StatusOK || !strings.HasPrefix(ek, "ek_") || cs["session"].(map[string]any)["type"] != "realtime" {
		t.Fatalf("client secret: %d %s", resp.StatusCode, b)
	}
	if _, _, status := wsHandshake(t, vr.srv.URL, "/v1/realtime", "realtime, openai-insecure-api-key."+ek, ""); !strings.Contains(status, "101") {
		t.Errorf("realtime with the client secret: %s", status)
	}
	resp, b = vr.req(t, "GET", "/v1/models", "", map[string]string{"Authorization": "Bearer " + ek})
	wantAPIError(t, "a client secret elsewhere", resp, b, http.StatusUnauthorized, "invalid_api_key")
	vr.ws.ephemeral.Store(ek, ephemeralKey{expires: time.Now().Add(-time.Second)})
	if _, _, status := wsHandshake(t, vr.srv.URL, "/v1/realtime", "realtime, openai-insecure-api-key."+ek, ""); !strings.Contains(status, "401") {
		t.Errorf("expired client secret: %s", status)
	}
	_, b = vr.req(t, "POST", "/v1/realtime/sessions", `{"model":"c","voice":"alloy"}`, auth)
	if s := decode(t, b); s["voice"] != "alloy" || !strings.HasPrefix(s["client_secret"].(map[string]any)["value"].(string), "ek_") {
		t.Errorf("legacy sessions: %s", b)
	}
	resp, b = vr.req(t, "POST", "/v1/realtime/client_secrets", `{"expires_after":{"seconds":5}}`, auth)
	wantAPIError(t, "too short", resp, b, http.StatusBadRequest, "")
}

func TestV1ImageVariationsAndRefusals(t *testing.T) {
	ir := newImageRig(t, ImageModelConf{Model: "flux2-klein-4b"})
	resp, b := multipartReq(t, ir.srv.URL+"/v1/images/variations", map[string]string{"n": "1", "response_format": "b64_json"}, map[string][]byte{"image": tinyPNG()}, nil)
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(b), "b64_json") {
		t.Fatalf("variation: %d %s", resp.StatusCode, b)
	}
	ir.sd.mu.Lock()
	edit := ir.sd.edits[len(ir.sd.edits)-1]
	ir.sd.mu.Unlock()
	if edit["prompt"] != variationPrompt || edit["images"] != 1 {
		t.Errorf("sd-server got %v", edit)
	}
	for name, extra := range map[string]map[string]any{"svg": {"svg": true}, "animated": {"frame_count": 4}} {
		body := map[string]any{"prompt": "x"}
		for k, v := range extra {
			body[k] = v
		}
		resp, b := ir.do(t, "POST", "/v1/images/generations", body)
		wantAPIError(t, name, resp, b, http.StatusBadRequest, "")
	}
	ir.do(t, "POST", "/v1/images/generations", map[string]any{"prompt": "x", "complexity": "high", "safety_identifier": "u", "frame_count": 1})
	ir.sd.mu.Lock()
	got := ir.sd.openai[len(ir.sd.openai)-1]
	ir.sd.mu.Unlock()
	for _, k := range []string{"complexity", "safety_identifier", "frame_count"} {
		if _, has := got[k]; has {
			t.Errorf("sd-server got %s", k)
		}
	}
}

func TestAPIStoreOpensInChatDB(t *testing.T) {
	s, err := OpenChatStore(filepath.Join(t.TempDir(), "chats.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	st := &apiStore{db: s.db}
	if err := st.putResponse("resp_old", time.Now().Add(-31*24*time.Hour).Unix(), nil, map[string]any{"id": "resp_old"}); err != nil {
		t.Fatal(err)
	}
	st.putResponse("resp_new", time.Now().Unix(), nil, map[string]any{"id": "resp_new"})
	st.prune(time.Now())
	if _, _, err := st.getResponse("resp_old"); err != errAPINotFound {
		t.Error("a 31-day-old response must be pruned")
	}
	if _, _, err := st.getResponse("resp_new"); err != nil {
		t.Error("a new response must stay")
	}
}
