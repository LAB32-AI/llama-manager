package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func serve(ws *WebServer, method, path, host, origin string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(`{}`))
	req.Host = host
	if origin != "" {
		req.Header.Set("Origin", origin)
	}
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, req)
	return rr
}

// TestDNSRebindingIsBlocked is the P1 #15 regression: a rebound attacker page
// sends Host and Origin that match each other, which the Origin check alone
// accepted, for state changes and for reads.
func TestDNSRebindingIsBlocked(t *testing.T) {
	ws, _ := newTestServer(t)
	const evil = "rebind.attacker.example:8080"
	if rr := serve(ws, http.MethodPost, "/api/instances/all/stop", evil, "http://"+evil); rr.Code != http.StatusForbidden {
		t.Errorf("rebound POST: got %d, want 403", rr.Code)
	}
	if rr := serve(ws, http.MethodGet, "/api/config/export", evil, ""); rr.Code != http.StatusForbidden {
		t.Errorf("rebound GET of the config: got %d, want 403", rr.Code)
	}
}

func TestHostCheckAcceptsLegitimateHosts(t *testing.T) {
	ws, cfg := newTestServer(t)
	cfg.AllowedHosts = []string{"llm.home.arpa", "Router.LAN."}
	hn, _ := os.Hostname()
	short, _, _ := strings.Cut(strings.ToLower(hn), ".")
	for _, host := range []string{
		"192.168.1.50:8080", "10.0.0.2", "[::1]:8080", "[fe80::1]:8080",
		"localhost:8080", "LOCALHOST", "app.localhost:8080",
		hn, hn + ":8080", short + ".local:8080",
		"llm.home.arpa:8080", "router.lan", "llm.home.arpa.:8080",
	} {
		if rr := serve(ws, http.MethodGet, "/api/status", host, ""); rr.Code != http.StatusOK {
			t.Errorf("host %q: got %d, want 200", host, rr.Code)
		}
	}
	for _, host := range []string{"evil.example", "llm.home.arpa.evil.example", "192.168.1.50.nip.io:8080"} {
		if rr := serve(ws, http.MethodGet, "/api/status", host, ""); rr.Code != http.StatusForbidden {
			t.Errorf("host %q: got %d, want 403", host, rr.Code)
		}
	}
}

func TestAllowedHostsWildcardDisablesHostCheck(t *testing.T) {
	ws, cfg := newTestServer(t)
	cfg.AllowedHosts = []string{"*"}
	if rr := serve(ws, http.MethodGet, "/api/status", "anything.example", ""); rr.Code != http.StatusOK {
		t.Fatalf("got %d with allowed_hosts [*]", rr.Code)
	}
}

// TestCrossOriginStateChangeRejected covers the Origin check itself, which no
// test exercised before (deleting it left the suite green).
func TestCrossOriginStateChangeRejected(t *testing.T) {
	ws, _ := newTestServer(t)
	const host = "192.168.1.50:8080"
	for _, m := range []string{http.MethodPost, http.MethodPut, http.MethodDelete} {
		for _, origin := range []string{"http://evil.example", "null", "http://192.168.1.50:9999"} {
			if rr := serve(ws, m, "/api/settings", host, origin); rr.Code != http.StatusForbidden {
				t.Errorf("%s with Origin %q: got %d, want 403", m, origin, rr.Code)
			}
		}
	}
	// Same-origin browser requests and non-browser clients (no Origin) pass.
	if rr := serve(ws, http.MethodPost, "/api/instances/all/stop", host, "http://"+host); rr.Code != http.StatusOK {
		t.Errorf("same-origin POST: got %d, want 200", rr.Code)
	}
	if rr := serve(ws, http.MethodPost, "/api/instances/all/stop", host, ""); rr.Code != http.StatusOK {
		t.Errorf("POST without Origin (curl, agents): got %d, want 200", rr.Code)
	}
}
