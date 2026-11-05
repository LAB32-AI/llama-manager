package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

// maxAPIBody bounds a /v1 request: base64 images in a chat, an image edit's
// pictures. A var so tests can lower it.
var maxAPIBody int64 = 64 << 20

// llamaRoutes are llama-server's own routes that /v1 passes on as they are,
// to the instance the body's "model" names. Chat, completions, embeddings and
// responses have their own handlers, which fill in what llama-server lacks.
var llamaRoutes = []string{"/v1/messages", "/v1/messages/count_tokens", "/v1/rerank", "/v1/reranking"}

// v1Groups are the first path segments /v1 serves, so an app given the base
// URL without /v1 (http://host:8080) still reaches them.
var v1Groups = map[string]bool{"health": true, "models": true, "chat": true, "completions": true, "embeddings": true,
	"responses": true, "conversations": true, "messages": true, "rerank": true, "reranking": true,
	"images": true, "audio": true, "realtime": true}

func isV1Route(p string) bool {
	seg, _, _ := strings.Cut(strings.TrimPrefix(p, "/v1/"), "/")
	return strings.HasPrefix(p, "/v1/") && v1Groups[seg]
}

// cloudOnly are OpenAI services that keep state in OpenAI's cloud; a local
// server has nothing to offer for them.
var cloudOnly = map[string]string{
	"files": "file storage", "uploads": "file uploads", "batches": "the Batch API",
	"vector_stores": "vector stores", "fine_tuning": "fine-tuning", "fine-tunes": "fine-tuning",
	"assistants": "the Assistants API", "threads": "the Assistants API", "moderations": "moderation",
	"evals": "evals", "graders": "graders", "containers": "code containers", "chatkit": "ChatKit",
	"videos": "video generation", "organization": "organization admin", "projects": "project admin",
	"admin": "organization admin", "agents": "hosted agents", "skills": "skills", "vaults": "credential vaults",
	"webhook_endpoints": "webhooks", "webhook_event_types": "webhooks", "decisions": "decisions",
	"content_provenance_checks": "content provenance checks", "safety": "safety reviews", "live": "the Live API",
	"audio/voices": "custom voices", "audio/voice_consents": "custom voices",
	"realtime/calls": "realtime phone and WebRTC calls", "realtime/translations": "realtime translation",
}

func (ws *WebServer) newV1Mux() *http.ServeMux {
	m := http.NewServeMux()
	m.HandleFunc("/v1/health", ws.handleV1Health)
	m.HandleFunc("/v1/models", ws.handleV1Models)
	m.HandleFunc("/v1/models/", ws.handleV1Model)
	for _, p := range llamaRoutes {
		m.HandleFunc(p, ws.handleV1Llama)
	}
	m.HandleFunc("/v1/chat/completions", ws.handleV1Chat)
	m.HandleFunc("/v1/chat/completions/", ws.handleV1ChatStored)
	m.HandleFunc("/v1/completions", ws.handleV1Completions)
	m.HandleFunc("/v1/embeddings", ws.handleV1Embeddings)
	m.HandleFunc("/v1/responses", ws.handleV1Responses)
	m.HandleFunc("/v1/responses/", ws.handleV1ResponseSub)
	m.HandleFunc("/v1/conversations", ws.handleV1Conversations)
	m.HandleFunc("/v1/conversations/", ws.handleV1Conversations)
	m.HandleFunc("/v1/realtime/client_secrets", ws.handleV1ClientSecrets)
	m.HandleFunc("/v1/realtime/sessions", ws.handleV1ClientSecrets)
	m.HandleFunc("/v1/realtime/transcription_sessions", ws.handleV1ClientSecrets)
	m.HandleFunc("/v1/images/generations", ws.handleV1Images)
	m.HandleFunc("/v1/images/edits", ws.handleV1ImageEdits)
	m.HandleFunc("/v1/images/variations", ws.handleV1ImageVariations)
	m.HandleFunc("/v1/images/files/", ws.handleV1ImageFile)
	m.HandleFunc("/v1/audio/speech", ws.handleV1Speech)
	m.HandleFunc("/v1/audio/transcriptions", ws.handleV1Transcriptions)
	m.HandleFunc("/v1/audio/translations", ws.handleV1Translations)
	m.HandleFunc("/v1/realtime", ws.handleV1Realtime)
	m.HandleFunc("/", ws.handleV1Unknown)
	return m
}

// apiError is an error answered in OpenAI's shape:
// {"error": {"message", "type", "param", "code"}}.
type apiError struct {
	status int
	code   string
	param  string
	msg    string
}

func (e *apiError) Error() string { return e.msg }

func apiErrf(status int, code, format string, a ...any) *apiError {
	return &apiError{status: status, code: code, msg: fmt.Sprintf(format, a...)}
}

func (e *apiError) write(w http.ResponseWriter) { writeAPIError(w, e.status, e.code, e.param, e.msg) }

func writeAPIError(w http.ResponseWriter, status int, code, param, msg string) {
	typ := "invalid_request_error"
	switch {
	case status == http.StatusForbidden:
		typ = "permission_error"
	case status == http.StatusTooManyRequests:
		typ = "rate_limit_error"
	case status >= 500:
		typ = "server_error"
	}
	e := map[string]any{"message": msg, "type": typ, "param": nil, "code": nil}
	if code != "" {
		e["code"] = code
	}
	if param != "" {
		e["param"] = param
	}
	h := w.Header()
	h.Del("Content-Length")
	h.Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(map[string]any{"error": e})
}

// apiErrorWriter turns the plain-text errors that shared code writes with
// http.Error (and text errors from upstream servers) into OpenAI-style JSON.
type apiErrorWriter struct {
	http.ResponseWriter
	wrote  bool
	status int
	buf    *bytes.Buffer // set while capturing a plain-text error
}

func (e *apiErrorWriter) WriteHeader(code int) {
	if e.wrote {
		return
	}
	e.wrote = true
	if code >= 400 && strings.HasPrefix(e.Header().Get("Content-Type"), "text/plain") {
		e.status, e.buf = code, &bytes.Buffer{}
		return
	}
	e.ResponseWriter.WriteHeader(code)
}

func (e *apiErrorWriter) Write(b []byte) (int, error) {
	if !e.wrote {
		e.WriteHeader(http.StatusOK)
	}
	if e.buf != nil {
		if e.buf.Len() < 64<<10 {
			e.buf.Write(b)
		}
		return len(b), nil
	}
	return e.ResponseWriter.Write(b)
}

func (e *apiErrorWriter) Flush() {
	if e.buf != nil {
		return
	}
	if f, ok := e.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Unwrap lets http.ResponseController (the WebSocket proxy) reach the
// connection.
func (e *apiErrorWriter) Unwrap() http.ResponseWriter { return e.ResponseWriter }

func (e *apiErrorWriter) finish() {
	if e.buf == nil {
		return
	}
	msg := strings.TrimSpace(e.buf.String())
	if msg == "" {
		msg = http.StatusText(e.status)
	}
	code := ""
	switch e.status {
	case http.StatusNotFound:
		if strings.Contains(msg, "model") {
			code = "model_not_found"
		}
	case http.StatusRequestEntityTooLarge:
		code = "request_too_large"
	}
	writeAPIError(e.ResponseWriter, e.status, code, "", msg)
}

func randHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// serveV1 is the OpenAI-compatible API. It keeps the dashboard's protections
// (known Host names, no cross-site requests) except for the web pages listed
// in cors_origins, and asks for an API key when api_keys has any.
func (ws *WebServer) serveV1(w http.ResponseWriter, r *http.Request) {
	ew := &apiErrorWriter{ResponseWriter: w}
	defer ew.finish()
	h := w.Header()
	h.Set("X-Request-Id", "req_"+randHex(12))
	h.Add("Vary", "Origin")
	origin := r.Header.Get("Origin")
	cors := origin != "" && ws.corsAllowed(origin)
	if cors {
		h.Set("Access-Control-Allow-Origin", origin)
		h.Set("Access-Control-Expose-Headers", "X-Request-Id, X-Manager-Elapsed-Ms")
	}
	foreign := origin != "" && !cors && !sameOrigin(origin, r.Host)
	if r.Method == http.MethodOptions {
		if foreign {
			writeAPIError(ew, http.StatusForbidden, "origin_not_allowed", "", fmt.Sprintf("web pages from %s may not call this API; add the origin to cors_origins in config.yaml on the server", origin))
			return
		}
		h.Set("Allow", "GET, POST, DELETE, OPTIONS")
		h.Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		if rh := r.Header.Get("Access-Control-Request-Headers"); rh != "" {
			h.Set("Access-Control-Allow-Headers", rh)
		}
		h.Set("Access-Control-Max-Age", "600")
		ew.WriteHeader(http.StatusNoContent)
		return
	}
	keys := ws.apiKeyList()
	keyOK := len(keys) == 0 || validAPIKey(r, keys)
	if !keyOK && r.URL.Path == "/v1/realtime" {
		_, keyOK = ws.ephemeralFor(r)
	}
	// Picture links can't be guessed, so like OpenAI's they need no key, and
	// no known Host either: apps fetch them as plain URLs. /health is public
	// like llama-server's, for monitors and container health checks.
	public := (r.Method == http.MethodGet || r.Method == http.MethodHead) &&
		(strings.HasPrefix(r.URL.Path, "/v1/images/files/") || r.URL.Path == "/v1/health")
	// A configured key the caller knows is proof enough that this isn't a
	// DNS-rebinding page, so the Host check only applies without one.
	if !public && !(len(keys) > 0 && keyOK) && !ws.hostAllowed(r.Host) {
		writeAPIError(ew, http.StatusForbidden, "host_not_allowed", "", fmt.Sprintf("unrecognized host %q; add it to allowed_hosts in config.yaml to allow it", r.Host))
		return
	}
	// Browsers send Origin with every POST and WebSocket; other pages may only
	// read, and can't see the answers without CORS headers.
	if foreign && (isWebSocket(r) || (r.Method != http.MethodGet && r.Method != http.MethodHead)) {
		writeAPIError(ew, http.StatusForbidden, "origin_not_allowed", "", fmt.Sprintf("web pages from %s may not call this API; add the origin to cors_origins in config.yaml on the server", origin))
		return
	}
	if !keyOK && !public {
		writeAPIError(ew, http.StatusUnauthorized, "invalid_api_key", "", "Incorrect API key provided. Send one of the keys set in Settings > OpenAI API as \"Authorization: Bearer <key>\".")
		return
	}
	ws.v1.ServeHTTP(ew, r)
}

func sameOrigin(origin, host string) bool {
	return strings.EqualFold(origin, "http://"+host) || strings.EqualFold(origin, "https://"+host)
}

func (ws *WebServer) corsAllowed(origin string) bool {
	origin = strings.ToLower(origin)
	ws.cfg.mu.RLock()
	defer ws.cfg.mu.RUnlock()
	return slices.Contains(ws.cfg.CORSOrigins, "*") || slices.Contains(ws.cfg.CORSOrigins, origin)
}

func (ws *WebServer) apiKeyList() []string {
	ws.cfg.mu.RLock()
	defer ws.cfg.mu.RUnlock()
	return ws.cfg.APIKeys
}

func isWebSocket(r *http.Request) bool {
	return strings.EqualFold(r.Header.Get("Upgrade"), "websocket")
}

// wsKeyProtocol carries the key of a browser's WebSocket, which can't set
// headers: OpenAI's Realtime API reads it from Sec-WebSocket-Protocol.
const wsKeyProtocol = "openai-insecure-api-key."

func websocketProtocols(r *http.Request) []string {
	var out []string
	for _, v := range r.Header.Values("Sec-WebSocket-Protocol") {
		for _, p := range strings.Split(v, ",") {
			if p = strings.TrimSpace(p); p != "" {
				out = append(out, p)
			}
		}
	}
	return out
}

// validAPIKey accepts a key sent the ways OpenAI-style clients send one:
// Authorization: Bearer (OpenAI), x-api-key (Anthropic), api-key (Azure) or a
// WebSocket subprotocol.
func validAPIKey(r *http.Request, keys []string) bool {
	ok := false
	for _, s := range presentedKeys(r) {
		for _, k := range keys {
			if subtle.ConstantTimeCompare([]byte(s), []byte(k)) == 1 {
				ok = true
			}
		}
	}
	return ok
}

func presentedKeys(r *http.Request) []string {
	var sent []string
	if a := r.Header.Get("Authorization"); a != "" {
		if scheme, tok, ok := strings.Cut(a, " "); ok && strings.EqualFold(scheme, "Bearer") {
			sent = append(sent, strings.TrimSpace(tok))
		}
	}
	for _, name := range []string{"X-Api-Key", "Api-Key"} {
		if v := r.Header.Get(name); v != "" {
			sent = append(sent, strings.TrimSpace(v))
		}
	}
	for _, p := range websocketProtocols(r) {
		if k, ok := strings.CutPrefix(p, wsKeyProtocol); ok {
			sent = append(sent, k)
		}
	}
	return sent
}

// ephemeralKey is a realtime client secret: a short-lived key a browser uses
// for /v1/realtime instead of a real one.
type ephemeralKey struct {
	expires time.Time
	model   string
}

// ephemeralFor finds a valid client secret among the keys r sends.
func (ws *WebServer) ephemeralFor(r *http.Request) (ephemeralKey, bool) {
	for _, k := range presentedKeys(r) {
		if v, ok := ws.ephemeral.Load(k); ok {
			if ek := v.(ephemeralKey); time.Now().Before(ek.expires) {
				return ek, true
			}
			ws.ephemeral.Delete(k)
		}
	}
	return ephemeralKey{}, false
}

// handleV1ClientSecrets makes a realtime client secret (and answers the older
// /realtime/sessions and /realtime/transcription_sessions the same way):
// valid for /v1/realtime for 10 minutes, or expires_after.seconds (10-7200).
func (ws *WebServer) handleV1ClientSecrets(w http.ResponseWriter, r *http.Request) {
	if !allowMethods(w, r, http.MethodPost) {
		return
	}
	body, aerr := readBodyMap(w, r)
	if aerr != nil {
		aerr.write(w)
		return
	}
	secs := 600
	if ea, ok := body["expires_after"].(map[string]any); ok {
		if f, ok := ea["seconds"].(float64); ok {
			if f < 10 || f > 7200 {
				writeAPIError(w, http.StatusBadRequest, "", "expires_after.seconds", "expires_after.seconds must be between 10 and 7200")
				return
			}
			secs = int(f)
		}
	}
	session, _ := body["session"].(map[string]any)
	legacy := r.URL.Path != "/v1/realtime/client_secrets"
	if legacy {
		session = body
	}
	if session == nil {
		session = map[string]any{}
	}
	typ, _ := session["type"].(string)
	if typ == "" {
		typ = "realtime"
		if r.URL.Path == "/v1/realtime/transcription_sessions" {
			typ = "transcription"
		}
	}
	model, _ := session["model"].(string)
	if inst := ws.findInstance(model); inst == nil {
		model = ""
	}
	now := time.Now()
	ws.ephemeral.Range(func(k, v any) bool {
		if now.After(v.(ephemeralKey).expires) {
			ws.ephemeral.Delete(k)
		}
		return true
	})
	value := "ek_" + randHex(24)
	expires := now.Add(time.Duration(secs) * time.Second)
	ws.ephemeral.Store(value, ephemeralKey{expires: expires, model: model})
	out := map[string]any{}
	for k, v := range session {
		out[k] = v
	}
	out["id"], out["type"], out["object"] = "sess_"+randHex(12), typ, "realtime.session"
	secret := map[string]any{"value": value, "expires_at": expires.Unix()}
	if legacy {
		out["client_secret"] = secret
		writeJSON(w, http.StatusOK, out)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"value": value, "expires_at": expires.Unix(), "session": out})
}

// allowMethods answers 405 unless r uses one of methods.
func allowMethods(w http.ResponseWriter, r *http.Request, methods ...string) bool {
	if slices.Contains(methods, r.Method) {
		return true
	}
	w.Header().Set("Allow", strings.Join(methods, ", "))
	writeAPIError(w, http.StatusMethodNotAllowed, "method_not_allowed", "", fmt.Sprintf("%s %s isn't supported; use %s", r.Method, r.URL.Path, strings.Join(methods, " or ")))
	return false
}

func readAPIBody(w http.ResponseWriter, r *http.Request, limit int64) ([]byte, *apiError) {
	r.Body = http.MaxBytesReader(w, r.Body, limit)
	b, err := io.ReadAll(r.Body)
	var tooBig *http.MaxBytesError
	switch {
	case errors.As(err, &tooBig):
		return nil, apiErrf(http.StatusRequestEntityTooLarge, "request_too_large", "the request is over %d MB", limit>>20)
	case err != nil:
		return nil, apiErrf(http.StatusBadRequest, "", "reading the request: %v", err)
	}
	return b, nil
}

type v1Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func modelOf(inst *Instance) v1Model {
	var created int64
	if t := inst.StartedAt(); !t.IsZero() {
		created = t.Unix()
	}
	return v1Model{ID: inst.conf.Name, Object: "model", Created: created, OwnedBy: "llama-manager"}
}

// handleV1Health answers like llama-server's /health: 200 {"status":"ok"}
// while the manager runs. With ?model=<name> it is that model's readiness:
// 200 when it is running, 503 while it loads or after a crash, 404 when it
// is stopped or unknown.
func (ws *WebServer) handleV1Health(w http.ResponseWriter, r *http.Request) {
	if !allowMethods(w, r, http.MethodGet, http.MethodHead) {
		return
	}
	if name := r.URL.Query().Get("model"); name != "" {
		inst := ws.findInstance(name)
		if inst == nil {
			writeAPIError(w, http.StatusNotFound, "model_not_found", "model", fmt.Sprintf("The model `%s` does not exist", name))
			return
		}
		if e := readyErr(inst); e != nil {
			e.write(w)
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// handleV1Models lists the running instances; each one's id is its name.
func (ws *WebServer) handleV1Models(w http.ResponseWriter, r *http.Request) {
	if !allowMethods(w, r, http.MethodGet, http.MethodHead) {
		return
	}
	models := []v1Model{}
	for _, inst := range ws.mgr.Instances() {
		if inst.State() == StateRunning {
			models = append(models, modelOf(inst))
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": models})
}

func (ws *WebServer) handleV1Model(w http.ResponseWriter, r *http.Request) {
	if !allowMethods(w, r, http.MethodGet, http.MethodHead) {
		return
	}
	id := strings.TrimPrefix(r.URL.Path, "/v1/models/")
	inst := ws.mgr.Get(id)
	if inst == nil || inst.State() != StateRunning {
		writeAPIError(w, http.StatusNotFound, "model_not_found", "model", fmt.Sprintf("The model `%s` does not exist or isn't running", id))
		return
	}
	writeJSON(w, http.StatusOK, modelOf(inst))
}

func (ws *WebServer) runningNames() string {
	var names []string
	for _, inst := range ws.mgr.Instances() {
		if inst.State() == StateRunning {
			names = append(names, inst.conf.Name)
		}
	}
	if len(names) == 0 {
		return "none"
	}
	return strings.Join(names, ", ")
}

// findInstance matches name exactly, then ignoring case.
func (ws *WebServer) findInstance(name string) *Instance {
	if name == "" {
		return nil
	}
	if inst := ws.mgr.Get(name); inst != nil {
		return inst
	}
	for _, inst := range ws.mgr.Instances() {
		if strings.EqualFold(inst.conf.Name, name) {
			return inst
		}
	}
	return nil
}

func (ws *WebServer) defaultModel() string {
	ws.cfg.mu.RLock()
	defer ws.cfg.mu.RUnlock()
	return ws.cfg.DefaultModel
}

// readyErr says why inst can't take a request now; nil when it can.
func readyErr(inst *Instance) *apiError {
	st := inst.Status()
	switch st.State {
	case StateRunning:
		return nil
	case StateStarting, StateRestarting:
		return apiErrf(http.StatusServiceUnavailable, "model_not_ready", "%s is starting (loading its model); try again in a moment", st.Name)
	case StateUnhealthy:
		return apiErrf(http.StatusServiceUnavailable, "model_not_ready", "%s isn't answering its health checks; try again in a moment", st.Name)
	case StateCrashed:
		msg := st.Name + " has crashed"
		if st.LastError != "" {
			msg += " (" + st.LastError + ")"
		}
		return apiErrf(http.StatusServiceUnavailable, "model_not_ready", "%s; see its log on the Instances page", msg)
	}
	return apiErrf(http.StatusNotFound, "model_not_found", "%s is stopped; start it on the Instances page", st.Name)
}

// llamaInstance finds the running llama-server a request's "model" names. An
// unknown or missing name goes to the fallback model when one is set.
func (ws *WebServer) llamaInstance(name string) (*Instance, *apiError) {
	inst := ws.findInstance(name)
	if inst == nil {
		dm := ws.defaultModel()
		switch {
		case dm != "":
			if inst = ws.mgr.Get(dm); inst == nil {
				return nil, apiErrf(http.StatusNotFound, "model_not_found", "the fallback model %s (Settings > OpenAI API) doesn't exist", dm)
			}
		case name == "":
			return nil, &apiError{status: http.StatusBadRequest, param: "model", msg: "you must provide a model parameter: the name of a running instance (GET /v1/models lists them)"}
		default:
			return nil, &apiError{status: http.StatusNotFound, code: "model_not_found", param: "model", msg: fmt.Sprintf(
				"The model `%s` does not exist or isn't running here. Running models: %s. To answer requests for other model names, pick a fallback model in Settings > OpenAI API.", name, ws.runningNames())}
		}
	}
	switch inst.conf.role {
	case roleImageGen:
		return nil, apiErrf(http.StatusBadRequest, "model_not_supported", "%s makes images; send its requests to /v1/images/generations", inst.conf.Name)
	case roleSpeech:
		return nil, apiErrf(http.StatusBadRequest, "model_not_supported", "%s reads text aloud; send its requests to /v1/audio/speech", inst.conf.Name)
	}
	if e := readyErr(inst); e != nil {
		return nil, e
	}
	return inst, nil
}

// handleV1Llama forwards a llama-server route (chat, completions, embeddings,
// responses, messages, rerank) to the instance its body names.
func (ws *WebServer) handleV1Llama(w http.ResponseWriter, r *http.Request) {
	if !allowMethods(w, r, http.MethodPost) {
		return
	}
	body, aerr := readAPIBody(w, r, maxAPIBody)
	if aerr != nil {
		aerr.write(w)
		return
	}
	var probe struct {
		Model any `json:"model"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		writeAPIError(w, http.StatusBadRequest, "", "", "We could not parse the JSON body of your request: "+err.Error())
		return
	}
	name, _ := probe.Model.(string)
	inst, aerr := ws.llamaInstance(name)
	if aerr != nil {
		aerr.write(w)
		return
	}
	relayLlama(w, r, inst, r.URL.Path, body, relayTx{})
}

// forwardLlama posts body to path on inst's llama-server and relays the
// answer. A "stream": true body is relayed as it arrives (SSE). The client's
// context makes a disconnect (closed tab, agent timeout) cancel the upstream
// request, so llama-server stops generating instead of keeping its slot busy
// for an answer nobody will read. Wall-clock time is reported in
// X-Manager-Elapsed-Ms (set before WriteHeader so it's delivered).
func forwardLlama(w http.ResponseWriter, r *http.Request, inst *Instance, path string, body []byte) {
	var probe struct {
		Stream bool `json:"stream"`
	}
	_ = json.Unmarshal(body, &probe)
	start := time.Now()
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, inst.baseURL()+path, bytes.NewReader(body))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	wait := inst.upstreamTimeout()
	resp, err := clientWithTimeout(wait).Do(req)
	if err != nil {
		var ue *url.Error
		if r.Context().Err() == nil && errors.As(err, &ue) && ue.Timeout() {
			http.Error(w, fmt.Sprintf("%s didn't start answering within %s; for longer non-streamed answers raise --timeout in its extra_args, or use \"stream\": true", inst.conf.Name, wait), http.StatusGatewayTimeout)
			return
		}
		http.Error(w, "proxying to llama-server: "+err.Error(), http.StatusBadGateway)
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

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "reading llama-server response: "+err.Error(), http.StatusBadGateway)
		return
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(data)
}

var apiClients sync.Map // time.Duration -> *http.Client

// clientWithTimeout is chatClient with another response-header timeout.
func clientWithTimeout(d time.Duration) *http.Client {
	if d <= 0 || d == chatTimeout {
		return chatClient
	}
	if c, ok := apiClients.Load(d); ok {
		return c.(*http.Client)
	}
	c, _ := apiClients.LoadOrStore(d, &http.Client{Transport: newChatTransport(d)})
	return c.(*http.Client)
}

// upstreamTimeout is how long a request waits for llama-server to start
// answering (a non-streamed reply starts only when it is done): the
// instance's own --timeout from its extra_args when that is longer, else
// chatTimeout.
func (inst *Instance) upstreamTimeout() time.Duration {
	extra := inst.conf.ExtraArgs
	var args []string
	if extra != nil {
		args = *extra
	} else if inst.cfg != nil {
		inst.cfg.mu.RLock()
		args = append(args, inst.cfg.ExtraArgs...)
		inst.cfg.mu.RUnlock()
	}
	d := chatTimeout
	for i, a := range args {
		v, ok := strings.CutPrefix(a, "--timeout=")
		if !ok && (a == "--timeout" || a == "-to") && i+1 < len(args) {
			v, ok = args[i+1], true
		}
		if n, err := strconv.Atoi(v); ok && err == nil && time.Duration(n)*time.Second > d {
			d = time.Duration(n) * time.Second
		}
	}
	return d
}

func (ws *WebServer) handleV1Unknown(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/v1/")
	seg, _, _ := strings.Cut(rest, "/")
	what, cloud := cloudOnly[seg]
	for k, v := range cloudOnly {
		if strings.Contains(k, "/") && (rest == k || strings.HasPrefix(rest, k+"/")) {
			what, cloud, seg = v, true, k
		}
	}
	if cloud {
		writeAPIError(w, http.StatusNotFound, "unsupported_endpoint", "", fmt.Sprintf(
			"llama-manager doesn't offer %s (/v1/%s): it is an OpenAI cloud service. It serves models, chat/completions, completions, responses, conversations, embeddings, rerank, messages, images, audio/speech, audio/transcriptions, audio/translations and realtime.", what, seg))
		return
	}
	writeAPIError(w, http.StatusNotFound, "", "", fmt.Sprintf("Invalid URL (%s %s)", r.Method, r.URL.Path))
}

// speechOnLlama is true for a voice model that runs in llama-server.
func (ws *WebServer) speechOnLlama(name string) bool {
	if ws.speech == nil {
		return false
	}
	e, ok := ws.speech.pick(name)
	return ok && e.Name == name && e.Preset.Engine != engineAudioCpp
}

// handleV1Realtime relays a Realtime API WebSocket (?model=) to llama-server's
// own /v1/realtime. Without a known model it goes to the fallback model, else
// the first running model that hears audio.
func (ws *WebServer) handleV1Realtime(w http.ResponseWriter, r *http.Request) {
	if !isWebSocket(r) {
		writeAPIError(w, http.StatusBadRequest, "", "", "/v1/realtime takes a WebSocket connection")
		return
	}
	model := r.URL.Query().Get("model")
	if ek, ok := ws.ephemeralFor(r); ok && ws.findInstance(model) == nil && ek.model != "" {
		model = ek.model
	}
	inst, aerr := ws.realtimeInstance(r.Context(), model)
	if aerr != nil {
		aerr.write(w)
		return
	}
	target, err := url.Parse(inst.baseURL())
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, "", "", err.Error())
		return
	}
	offered := websocketProtocols(r)
	var keep []string
	for _, p := range offered {
		if !strings.HasPrefix(p, wsKeyProtocol) {
			keep = append(keep, p)
		}
	}
	rp := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(target)
			pr.Out.URL.Path, pr.Out.URL.RawPath = "/v1/realtime", ""
			for _, h := range []string{"Authorization", "X-Api-Key", "Api-Key", "Cookie", "Origin"} {
				pr.Out.Header.Del(h)
			}
			pr.Out.Header.Del("Sec-Websocket-Protocol")
			if len(keep) > 0 {
				pr.Out.Header.Set("Sec-WebSocket-Protocol", strings.Join(keep, ", "))
			}
		},
		// A browser that offered subprotocols drops a connection that picks
		// none; answer with the first it offered when llama-server picks none.
		ModifyResponse: func(resp *http.Response) error {
			if resp.StatusCode == http.StatusSwitchingProtocols && resp.Header.Get("Sec-WebSocket-Protocol") == "" && len(offered) > 0 {
				resp.Header.Set("Sec-WebSocket-Protocol", pickProtocol(offered))
			}
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			writeAPIError(w, http.StatusBadGateway, "", "", fmt.Sprintf("couldn't reach %s: %v", inst.conf.Name, err))
		},
	}
	rp.ServeHTTP(w, r)
}

func (ws *WebServer) realtimeInstance(ctx context.Context, name string) (*Instance, *apiError) {
	if inst := ws.findInstance(name); inst != nil {
		if inst.conf.role == roleImageGen || (inst.conf.role == roleSpeech && !ws.speechOnLlama(inst.conf.Name)) {
			return nil, apiErrf(http.StatusBadRequest, "model_not_supported", "%s doesn't serve /v1/realtime: it doesn't run in llama-server", inst.conf.Name)
		}
		if e := readyErr(inst); e != nil {
			return nil, e
		}
		return inst, nil
	}
	return ws.audioInstance(ctx, "")
}
