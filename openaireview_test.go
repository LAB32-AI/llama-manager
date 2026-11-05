package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

// Regression tests for the second review of the /v1 state.

func TestV1WebSocketCloseStopsGeneration(t *testing.T) {
	started, cancelled := make(chan struct{}, 1), make(chan struct{}, 1)
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		started <- struct{}{}
		select {
		case <-r.Context().Done():
			cancelled <- struct{}{}
		case <-time.After(5 * time.Second):
		}
	})
	conn, _ := wsDial(t, vr.srv.URL, "/v1/responses", "realtime", "")
	wsSend(t, conn, `{"type":"response.create","model":"c","input":"long"}`, false)
	<-started
	conn.Close()
	select {
	case <-cancelled:
	case <-time.After(3 * time.Second):
		t.Fatal("llama-server kept generating after the WebSocket closed")
	}
}

func TestV1ResponseStoredBeforeCompletedEvent(t *testing.T) {
	release := make(chan struct{})
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		resp := map[string]any{"id": "resp_x1", "object": "response", "status": "completed", "output": []any{}}
		b, _ := json.Marshal(map[string]any{"type": "response.completed", "response": resp})
		fmt.Fprintf(w, "event: response.completed\ndata: %s\n\n", b)
		w.(http.Flusher).Flush()
		select { // the stream stays open a while after the final event
		case <-release:
		case <-r.Context().Done():
		}
	})
	t.Cleanup(func() { close(release) })
	conn, br := wsDial(t, vr.srv.URL, "/v1/responses", "", "")
	defer conn.Close()
	wsSend(t, conn, `{"type":"response.create","model":"c","input":"hi"}`, false)
	if ev := wsRecv(t, br); !strings.Contains(string(ev), "response.completed") {
		t.Fatalf("event: %s", ev)
	}
	if resp, b := vr.req(t, "GET", "/v1/responses/resp_x1", "", nil); resp.StatusCode != http.StatusOK {
		t.Errorf("a response must be stored before its completed event goes out: %d %s", resp.StatusCode, b)
	}
}

func TestV1ChainSurvivesAMissingLink(t *testing.T) {
	f := &respFake{}
	vr := newStateRig(t, f)
	vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"one"}`, nil)
	vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"two","previous_response_id":"resp_up1"}`, nil)
	vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"three","previous_response_id":"resp_up2"}`, nil)
	vr.req(t, "DELETE", "/v1/responses/resp_up1", "", nil)
	resp, b := vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"four","previous_response_id":"resp_up3"}`, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("continuing after an older link was deleted: %d %s", resp.StatusCode, b)
	}
	if got := texts(f.lastInput(t)); strings.Join(got, "|") != "user:two|assistant:answer 2|user:three|assistant:answer 3|user:four" {
		t.Errorf("history: %v", got)
	}
	// An old first turn of a chain in use isn't pruned.
	st := vr.ws.apiStore()
	st.db.Exec(`UPDATE api_responses SET created_at = ?`, time.Now().Add(-40*24*time.Hour).Unix())
	st.prune(time.Now())
	if _, _, err := st.getResponse("resp_up2"); err != nil {
		t.Error("a link of a chain continued just now was pruned")
	}
}

func TestV1DeleteBackgroundStaysDeleted(t *testing.T) {
	f := &respFake{hold: make(chan struct{})}
	vr := newStateRig(t, f)
	t.Cleanup(func() { close(f.hold) })
	_, b := vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"x","background":true}`, nil)
	id := decode(t, b)["id"].(string)
	waitFor(t, 3*time.Second, "in progress", func() bool {
		_, b := vr.req(t, "GET", "/v1/responses/"+id, "", nil)
		return decode(t, b)["status"] == "in_progress"
	})
	vr.req(t, "DELETE", "/v1/responses/"+id, "", nil)
	time.Sleep(100 * time.Millisecond)
	if resp, b := vr.req(t, "GET", "/v1/responses/"+id, "", nil); resp.StatusCode != http.StatusNotFound {
		t.Errorf("a deleted background response came back: %s", b)
	}
}

func TestV1ConversationItemIDsStayUnique(t *testing.T) {
	f := &respFake{}
	vr := newStateRig(t, f)
	_, b := vr.req(t, "POST", "/v1/conversations", `{"items":[{"id":"msg_a","role":"user","content":"one"},{"id":"msg_a","role":"user","content":"two"}]}`, nil)
	id := decode(t, b)["id"].(string)
	vr.req(t, "POST", "/v1/conversations/"+id+"/items", `{"items":[{"id":"msg_a","role":"user","content":"three"}]}`, nil)
	vr.req(t, "POST", "/v1/responses", `{"model":"c","conversation":"`+id+`","input":[{"type":"item_reference","id":"msg_a"}]}`, nil)
	seen := map[any]bool{}
	after := ""
	for page := 0; ; page++ {
		if page > 10 {
			t.Fatal("pagination doesn't end")
		}
		_, b := vr.req(t, "GET", "/v1/conversations/"+id+"/items?limit=2&order=asc"+after, "", nil)
		l := decode(t, b)
		for _, it := range l["data"].([]any) {
			iid := it.(map[string]any)["id"]
			if seen[iid] {
				t.Fatalf("item id %v twice", iid)
			}
			seen[iid] = true
		}
		if l["has_more"] != true {
			break
		}
		after = "&after=" + l["last_id"].(string)
	}
	if len(seen) != 5 {
		t.Errorf("%d items", len(seen))
	}
}

func TestV1IncompleteTurnJoinsConversation(t *testing.T) {
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		item := map[string]any{"type": "message", "id": "msg_o", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": "cut"}}}
		b, _ := json.Marshal(map[string]any{"type": "response.incomplete", "response": map[string]any{"id": "resp_i", "status": "incomplete", "output": []any{item}}})
		fmt.Fprintf(w, "event: response.incomplete\ndata: %s\n\n", b)
	})
	_, b := vr.req(t, "POST", "/v1/conversations", `{}`, nil)
	id := decode(t, b)["id"].(string)
	vr.req(t, "POST", "/v1/responses", `{"model":"c","conversation":"`+id+`","input":"long question"}`, nil)
	if _, b := vr.req(t, "GET", "/v1/conversations/"+id+"/items", "", nil); len(decode(t, b)["data"].([]any)) != 2 {
		t.Errorf("a cut-off turn must stay in its conversation: %s", b)
	}
}

func TestV1StreamErrorSentOnce(t *testing.T) {
	vr := newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "event: error\ndata: {\"type\":\"error\",\"message\":\"context full\"}\n\n")
	})
	_, b := vr.req(t, "POST", "/v1/responses", `{"model":"c","input":"x","stream":true}`, nil)
	if n := strings.Count(string(b), "event: error"); n != 1 {
		t.Errorf("%d error events: %s", n, b)
	}
}

func TestV1CORSPreflightAllowsDelete(t *testing.T) {
	vr := newV1Rig(t, nil)
	vr.ws.cfg.mu.Lock()
	vr.ws.cfg.CORSOrigins = []string{"http://app.lan:3000"}
	vr.ws.cfg.mu.Unlock()
	resp, _ := vr.req(t, "OPTIONS", "/v1/conversations/conv_1", "", map[string]string{"Origin": "http://app.lan:3000", "Access-Control-Request-Method": "DELETE"})
	if !strings.Contains(resp.Header.Get("Access-Control-Allow-Methods"), "DELETE") {
		t.Errorf("methods: %q", resp.Header.Get("Access-Control-Allow-Methods"))
	}
}

func TestV1ExpiredSecretsAreSwept(t *testing.T) {
	vr := newV1Rig(t, nil)
	vr.ws.ephemeral.Store("ek_old", ephemeralKey{expires: time.Now().Add(-time.Minute)})
	vr.req(t, "POST", "/v1/realtime/client_secrets", `{}`, nil)
	if _, ok := vr.ws.ephemeral.Load("ek_old"); ok {
		t.Error("an expired client secret is kept forever")
	}
}

func TestV1ChatAudioFormatNeedsFFmpeg(t *testing.T) {
	sr := newSpeechRig(t, SpeechModelConf{Model: "lfm3-audio"})
	noFFmpeg(t)
	resp, b := sr.do(t, "POST", "/v1/chat/completions", map[string]any{"model": "c", "modalities": []string{"text", "audio"},
		"audio": map[string]any{"voice": "alloy", "format": "mp3"}, "messages": []any{map[string]any{"role": "user", "content": "hi"}}})
	wantAPIError(t, "mp3 without ffmpeg", resp, b, http.StatusBadRequest, "")
}

func TestV1WebSocketKeepsStorelessResponses(t *testing.T) {
	f := &respFake{}
	vr := newStateRig(t, f)
	conn, br := wsDial(t, vr.srv.URL, "/v1/responses", "", "")
	defer conn.Close()
	next := func() {
		for {
			var ev map[string]any
			json.Unmarshal(wsRecv(t, br), &ev)
			if ev["type"] == "response.completed" || ev["type"] == "error" {
				if ev["type"] == "error" {
					t.Fatalf("error: %v", ev)
				}
				return
			}
		}
	}
	wsSend(t, conn, `{"type":"response.create","model":"c","input":"my name is Zorbal","store":false}`, false)
	next()
	wsSend(t, conn, `{"type":"response.create","model":"c","input":"name?","store":false,"previous_response_id":"resp_up1"}`, false)
	next()
	if got := texts(f.lastInput(t)); strings.Join(got, "|") != "user:my name is Zorbal|assistant:answer 1|user:name?" {
		t.Errorf("history on the connection: %v", got)
	}
	if resp, _ := vr.req(t, "GET", "/v1/responses/resp_up1", "", nil); resp.StatusCode != http.StatusNotFound {
		t.Error("store: false must not store")
	}
}

// rawFrame is a masked client frame with any first byte and payload.
func rawFrame(b0 byte, p []byte) []byte {
	out := []byte{b0, 0x80 | byte(len(p))}
	if len(p) > 125 {
		out = []byte{b0, 0x80 | 126, byte(len(p) >> 8), byte(len(p))}
	}
	mask := []byte{1, 2, 3, 4}
	out = append(out, mask...)
	for i, x := range p {
		out = append(out, x^mask[i%4])
	}
	return out
}

func closeCode(t *testing.T, br interface{ Read([]byte) (int, error) }) uint16 {
	t.Helper()
	var h [2]byte
	if _, err := br.Read(h[:1]); err != nil {
		t.Fatal(err)
	}
	br.Read(h[1:])
	p := make([]byte, h[1]&0x7F)
	for n := 0; n < len(p); {
		m, err := br.Read(p[n:])
		if err != nil {
			t.Fatal(err)
		}
		n += m
	}
	if h[0]&0x0F != opClose || len(p) < 2 {
		t.Fatalf("not a close frame: % x", h)
	}
	return binary.BigEndian.Uint16(p)
}

func TestWebSocketProtocolRules(t *testing.T) {
	vr := newV1Rig(t, nil)
	for name, c := range map[string]struct {
		frame []byte
		code  uint16
	}{
		"reserved bits":     {rawFrame(0x80|0x40|opText, []byte("{}")), 1002},
		"long ping":         {rawFrame(0x80|opPing, make([]byte, 200)), 1002},
		"fragmented ping":   {rawFrame(opPing, []byte("x")), 1002},
		"not UTF-8":         {rawFrame(0x80|opText, []byte{0xff, 0xfe}), 1007},
		"close echoes 4001": {rawFrame(0x80|opClose, binary.BigEndian.AppendUint16(nil, 4001)), 4001},
	} {
		conn, br := wsDial(t, vr.srv.URL, "/v1/responses", "", "")
		conn.Write(c.frame)
		conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		if got := closeCode(t, br); got != c.code {
			t.Errorf("%s: close %d, want %d", name, got, c.code)
		}
		conn.Close()
	}
	// Only the key offered: it is chosen, or a browser drops the connection.
	conn, br, status := wsHandshake(t, vr.srv.URL, "/v1/responses", "openai-insecure-api-key.k", "")
	if !strings.Contains(status, "101") {
		t.Fatalf("handshake: %s", status)
	}
	conn.Close()
	_ = br
	req, _ := http.NewRequest("GET", vr.srv.URL+"/v1/responses", nil)
	req.Header.Set("Upgrade", "websocket")
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
	req.Header.Set("Sec-WebSocket-Version", "8")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUpgradeRequired || resp.Header.Get("Sec-WebSocket-Version") != "13" {
		t.Errorf("old version: %d %v", resp.StatusCode, resp.Header)
	}
}

var _ sync.Mutex
