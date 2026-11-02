package main

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

// newChatRig serves upstream as instance "c" (forced to running) behind a
// WebServer, the way proxyChat reaches a real llama-server. Like llama-server,
// the fake reads the request body before answering; Go's server only notices
// a closed client connection once the body has been consumed.
func newChatRig(t *testing.T, upstream http.HandlerFunc) *WebServer {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		upstream(w, r)
	}))
	t.Cleanup(srv.Close)
	_, portStr, _ := net.SplitHostPort(strings.TrimPrefix(srv.URL, "http://"))
	port, _ := strconv.Atoi(portStr)

	ws, _ := newTestServer(t)
	if err := ws.mgr.AddInstance(InstanceConf{Name: "c", Model: "/m.gguf", Port: port, GPUIDs: []int{0}}); err != nil {
		t.Fatal(err)
	}
	inst := ws.mgr.Get("c")
	inst.mu.Lock()
	inst.state = StateRunning
	inst.mu.Unlock()
	return ws
}

// TestChatProxyCancelsUpstreamOnClientDisconnect is the P1 #18 regression: the
// proxy didn't pass the client's context, so an abandoned request kept
// llama-server generating (and, with -np 1, blocked every request behind it).
func TestChatProxyCancelsUpstreamOnClientDisconnect(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			cancelled := make(chan struct{})
			release := make(chan struct{})
			ws := newChatRig(t, func(w http.ResponseWriter, r *http.Request) {
				if stream {
					w.Header().Set("Content-Type", "text/event-stream")
					w.WriteHeader(http.StatusOK)
					w.(http.Flusher).Flush()
				}
				select {
				case <-r.Context().Done():
					close(cancelled)
				case <-release: // test failed; don't hang srv.Close
				}
			})
			t.Cleanup(func() { close(release) }) // runs before srv.Close

			ctx, cancel := context.WithCancel(context.Background())
			body := fmt.Sprintf(`{"stream":%v,"messages":[]}`, stream)
			req := httptest.NewRequest(http.MethodPost, "/api/instances/c/chat", strings.NewReader(body)).WithContext(ctx)
			done := make(chan struct{})
			go func() {
				ws.ServeHTTP(httptest.NewRecorder(), req)
				close(done)
			}()
			time.Sleep(100 * time.Millisecond)
			cancel()

			select {
			case <-cancelled:
			case <-time.After(2 * time.Second):
				t.Fatal("upstream request was not cancelled when the client went away")
			}
			select {
			case <-done:
			case <-time.After(2 * time.Second):
				t.Fatal("proxy handler did not return after the client went away")
			}
		})
	}
}

// TestChatProxyDoesNotCutLongStreams: the old 10-minute Client.Timeout also
// covered the body, truncating long streamed answers. Only the time to the
// response headers is bounded now.
func TestChatProxyDoesNotCutLongStreams(t *testing.T) {
	old := chatClient
	chatClient = &http.Client{Transport: newChatTransport(200 * time.Millisecond)}
	defer func() { chatClient = old }()

	ws := newChatRig(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for i := 0; i < 6; i++ {
			fmt.Fprintf(w, "data: chunk%d\n\n", i)
			w.(http.Flusher).Flush()
			time.Sleep(100 * time.Millisecond) // 600ms total, 3x the header timeout
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/api/instances/c/chat", strings.NewReader(`{"stream":true}`)))
	if !strings.Contains(rr.Body.String(), "data: [DONE]") {
		t.Fatalf("stream was cut short:\n%s", rr.Body.String())
	}
}

func TestChatProxyTimesOutSilentUpstream(t *testing.T) {
	old := chatClient
	chatClient = &http.Client{Transport: newChatTransport(150 * time.Millisecond)}
	defer func() { chatClient = old }()

	release := make(chan struct{})
	ws := newChatRig(t, func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	})
	t.Cleanup(func() { close(release) })
	rr := httptest.NewRecorder()
	start := time.Now()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/api/instances/c/chat", strings.NewReader(`{}`)))
	if rr.Code != http.StatusBadGateway || time.Since(start) > 2*time.Second {
		t.Fatalf("got %d after %s, want 502 shortly after the header timeout", rr.Code, time.Since(start))
	}
}
