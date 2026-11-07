package main

import (
	"context"
	"crypto/sha256"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"io/fs"
	"log"
	"mime"
	"net"
	"net/http"
	"os"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	maxJSONBody   = 1 << 20
	maxUploadSize = 10 << 20
	maxChatBody   = 4 << 20 // 4 MiB — chat conversations can grow.
	chatTimeout   = 10 * time.Minute
)

// chatClient forwards chat requests. It deliberately has no Client.Timeout:
// that also caps reading the body and would cut long streamed answers off
// mid-response. ResponseHeaderTimeout still bounds an upstream that accepts
// the request but never answers.
var chatClient = &http.Client{Transport: newChatTransport(chatTimeout)}

func newChatTransport(headerTimeout time.Duration) *http.Transport {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.ResponseHeaderTimeout = headerTimeout
	return t
}

//go:embed templates/index.html
var templateFS embed.FS

//go:embed static
var staticFS embed.FS

func init() {
	_ = mime.AddExtensionType(".webmanifest", "application/manifest+json")
	_ = mime.AddExtensionType(".woff2", "font/woff2")
}

// assetVersion fingerprints the embedded static files, so URLs carrying it
// can be cached forever and change whenever a file does.
var assetVersion = func() string {
	h := sha256.New()
	_ = fs.WalkDir(staticFS, "static", func(path string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			b, _ := staticFS.ReadFile(path)
			h.Write([]byte(path))
			h.Write(b)
		}
		return nil
	})
	return fmt.Sprintf("%x", h.Sum(nil))[:12]
}()

type WebServer struct {
	mgr  *Manager
	cfg  *Config
	dlm  *DownloadManager
	tmpl *template.Template
	mux  *http.ServeMux
	v1   *http.ServeMux // the OpenAI-compatible API, behind serveV1

	hostname string // lower-cased os.Hostname(), for the Host header check
	started  time.Time

	// chats is nil when the chat database couldn't be opened; the rest of
	// the manager keeps working and the chat API answers 503.
	chats *ChatService

	// tika is nil unless config.yaml has an enabled tika section.
	tika *tikaClient
	// reader is nil unless image_recognition is enabled.
	reader *imageReader
	// imageGen is nil unless image_generation is enabled.
	imageGen *imageGen
	// speech is nil unless text_to_speech is enabled.
	speech *speechService
	// enhancer is nil unless prompt_enhancer is enabled.
	enhancer *promptEnhancer

	// imageFiles keeps pictures made over /v1 for an hour, for their URLs.
	imageFiles *blobCache
	// audioCaps remembers which instances hear audio (name -> audioCap).
	audioCaps sync.Map
	// audioReplies holds spoken chat replies' transcripts (id -> audioReply).
	audioReplies sync.Map
	// bgJobs are background responses being generated (id -> *bgJob).
	bgJobs sync.Map
	// ephemeral are realtime client secrets (value -> ephemeralKey).
	ephemeral sync.Map

	// limits enforces each API key's limits and the internet's cap.
	limits *apiLimiter
	// usage counts each API key's requests and tokens.
	usage *apiUsage
	// funnel is nil unless public_api.tailscale_funnel is on.
	funnel *funnel

	// sampler reads GPUs and instances for the dashboard every 15 s.
	sampler   *dashSampler
	storageMu sync.Mutex
	storage   *dashStorage // the dashboard's storage summary, reused storageTTL
	storageAt time.Time
}

type ServerStatus struct {
	Name             string        `json:"name"`
	Uptime           string        `json:"uptime"`
	UptimeSec        float64       `json:"uptime_sec"`
	ManagerUptimeSec float64       `json:"manager_uptime_sec"`
	Port             int           `json:"port"`
	Addresses        []hostAddress `json:"addresses"`
	ChatHistory      bool          `json:"chat_history"`
	Documents        docStatus     `json:"documents"`
	// ImageModels are the image generation models; empty when
	// image_generation is off.
	ImageModels []imageModelStatus `json:"image_models"`
	// Speech are the voice models that read replies aloud; empty when
	// text_to_speech is off.
	Speech []speechModelStatus `json:"speech"`
	// PromptEnhancer expands image prompts; absent when prompt_enhancer is off.
	PromptEnhancer *enhancerStatus `json:"prompt_enhancer,omitempty"`
	// PublicAPI is Tailscale Funnel's state; absent when public_api is off.
	PublicAPI *funnelStatus `json:"public_api,omitempty"`
}

func NewWebServer(mgr *Manager, cfg *Config, dlm *DownloadManager) *WebServer {
	tmpl := template.Must(template.ParseFS(templateFS, "templates/index.html"))
	hostname, _ := os.Hostname()
	cfg.mu.RLock()
	tikaConf, readerConf, genConf, speechConf := cfg.Tika, cfg.ImageRecognition, cfg.ImageGeneration, cfg.TextToSpeech
	enhancerConf := cfg.PromptEnhancer
	publicConf, managerPort := cfg.PublicAPI, cfg.ManagerPort
	cfg.mu.RUnlock()
	ws := &WebServer{
		tika:       newTikaClient(tikaConf),
		reader:     newImageReader(readerConf, mgr),
		imageGen:   newImageGen(genConf, mgr),
		speech:     newSpeechService(speechConf, mgr),
		enhancer:   newPromptEnhancer(enhancerConf, mgr),
		imageFiles: newBlobCache(maxImageFileBytes, imageFileTTL),
		mgr:        mgr,
		cfg:        cfg,
		dlm:        dlm,
		tmpl:       tmpl,
		mux:        http.NewServeMux(),
		hostname:   strings.ToLower(strings.TrimSuffix(hostname, ".")),
		started:    time.Now(),
		limits:     newAPILimiter(),
	}
	ws.usage = newAPIUsage(ws.apiStore)
	ws.funnel = newFunnel(publicConf, managerPort, ws.hasAPIKeys)
	ws.sampler = newDashSampler(ws)
	ws.mux.HandleFunc("/", ws.handleIndex)
	ws.mux.HandleFunc("/api/status", ws.handleStatus)
	ws.mux.HandleFunc("/api/dashboard", ws.handleDashboard)
	ws.mux.HandleFunc("/api/instances", ws.handleInstances)
	ws.mux.HandleFunc("/api/metrics", ws.handleMetrics)
	ws.mux.HandleFunc("/api/gpus", ws.handleGPUs)
	ws.mux.HandleFunc("/api/instances/all/", ws.handleBulkAction)
	ws.mux.HandleFunc("/api/instances/", ws.handleInstanceAction)
	ws.mux.HandleFunc("/api/models", ws.handleModels)
	ws.mux.HandleFunc("/api/models/quants", ws.handleModelQuants)
	ws.mux.HandleFunc("/api/models/download", ws.handleModelDownload)
	ws.mux.HandleFunc("/api/models/download/status", ws.handleModelDownloadStatus)
	ws.mux.HandleFunc("/api/models/download/stop", ws.handleModelDownloadStop)
	ws.mux.HandleFunc("/api/config/instances", ws.handleConfigInstances)
	ws.mux.HandleFunc("/api/config/instances/", ws.handleConfigInstanceAction)
	ws.mux.HandleFunc("/api/config/export", ws.handleConfigExport)
	ws.mux.HandleFunc("/api/config/import", ws.handleConfigImport)
	ws.mux.HandleFunc("/api/settings", ws.handleSettings)
	ws.mux.HandleFunc("/api/chats", ws.handleChats)
	ws.mux.HandleFunc("/api/chats/", ws.handleChat)
	ws.mux.HandleFunc("/api/attachments/", ws.handleAttachment)
	ws.mux.HandleFunc("/api/extract", ws.handleExtract)
	ws.mux.HandleFunc("/api/describe", ws.handleDescribe)
	ws.mux.HandleFunc("/api/enhance", ws.handleEnhance)
	ws.mux.HandleFunc("/static/", ws.handleStatic)
	ws.v1 = ws.newV1Mux()
	return ws
}

func (ws *WebServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get(funnelHeader) != "" {
		ws.servePublic(w, r)
		return
	}
	if p := r.URL.Path; !strings.HasPrefix(p, "/v1/") && isV1Route("/v1"+p) {
		r.URL.Path, r.URL.RawPath = "/v1"+p, ""
	}
	if strings.HasPrefix(r.URL.Path, "/v1/") {
		ws.serveV1(w, r)
		return
	}
	// DNS rebinding: a page on attacker.example re-pointed at this machine
	// sends Host and Origin that match each other, so the Origin check alone
	// would pass and the browser would let the page read every response.
	// Rebinding needs a DNS name the attacker controls; only accept names we
	// know are ours.
	if !ws.hostAllowed(r.Host) {
		http.Error(w, fmt.Sprintf("forbidden: unrecognized host %q; add it to allowed_hosts in config.yaml to allow it", r.Host), http.StatusForbidden)
		return
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead && r.Method != http.MethodOptions {
		if origin := r.Header.Get("Origin"); origin != "" {
			allowed := "http://" + r.Host
			allowedTLS := "https://" + r.Host
			if origin != allowed && origin != allowedTLS {
				http.Error(w, "forbidden: origin mismatch", http.StatusForbidden)
				return
			}
		}
	}
	ws.mux.ServeHTTP(w, r)
}

// servePublic answers a request from the internet, which came in through
// Tailscale Funnel: only /v1, never the dashboard or its API, whatever
// allowed_hosts says.
func (ws *WebServer) servePublic(w http.ResponseWriter, r *http.Request) {
	p := r.URL.Path
	if !strings.HasPrefix(p, "/v1/") || strings.Contains(p, "/../") || strings.HasSuffix(p, "/..") || strings.Contains(p, "/./") {
		ip := clientIP(r)
		log.Printf("[public] %s %s %q 404 (only /v1 is open to the internet)", ip, r.Method, p)
		writeAPIError(w, http.StatusNotFound, "", "", fmt.Sprintf("Invalid URL (%s %s): only /v1 is open to the internet", r.Method, p))
		return
	}
	ws.serveV1(w, r.WithContext(context.WithValue(r.Context(), publicCtxKey{}, clientIP(r))))
}

// hostAllowed accepts IP literals, localhost, this machine's hostname (with or
// without .local) and anything listed in allowed_hosts ("*" disables the
// check).
func (ws *WebServer) hostAllowed(hostport string) bool {
	host := hostport
	if h, _, err := net.SplitHostPort(hostport); err == nil {
		host = h
	}
	host = strings.ToLower(strings.TrimSuffix(strings.Trim(host, "[]"), "."))
	if host == "" || host == "localhost" || strings.HasSuffix(host, ".localhost") || net.ParseIP(host) != nil {
		return true
	}
	// Docker's names for the host; .internal can't be registered, so no
	// outside page can rebind them.
	if host == "host.docker.internal" || host == "gateway.docker.internal" {
		return true
	}
	if hn := ws.hostname; hn != "" {
		short, _, _ := strings.Cut(hn, ".")
		if host == hn || host == short || host == hn+".local" || host == short+".local" {
			return true
		}
	}
	ws.cfg.mu.RLock()
	defer ws.cfg.mu.RUnlock()
	for _, a := range ws.cfg.AllowedHosts {
		a = strings.ToLower(strings.TrimSuffix(strings.TrimSpace(a), "."))
		if a == "*" || a == host {
			return true
		}
	}
	return false
}

func (ws *WebServer) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	ws.tmpl.Execute(w, map[string]string{"V": assetVersion})
}

func (ws *WebServer) handleStatic(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	// Vendored libraries carry their version in the folder name (pdf.js loads
	// its wasm by path, so it can't take ?v=).
	if r.URL.Query().Get("v") == assetVersion || strings.HasPrefix(r.URL.Path, "/static/vendor/") {
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	} else {
		w.Header().Set("Cache-Control", "no-cache")
	}
	w.Header().Set("X-Content-Type-Options", "nosniff")
	http.FileServer(http.FS(staticFS)).ServeHTTP(w, r)
}

func (ws *WebServer) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	hostname, _ := os.Hostname()
	uptime := getSystemUptime()
	ws.cfg.mu.RLock()
	port := ws.cfg.ManagerPort
	ws.cfg.mu.RUnlock()
	status := ServerStatus{
		Name:             hostname,
		Uptime:           formatDuration(uptime),
		UptimeSec:        uptime.Seconds(),
		ManagerUptimeSec: time.Since(ws.started).Seconds(),
		Port:             port,
		Addresses:        hostAddresses(),
		ChatHistory:      ws.chats != nil,
		Documents:        ws.docStatus(),
	}
	status.ImageModels = []imageModelStatus{}
	if ws.imageGen != nil {
		status.ImageModels = ws.imageGen.Status()
	}
	status.Speech = []speechModelStatus{}
	if ws.speech != nil {
		status.Speech = ws.speech.Status()
	}
	if ws.enhancer != nil {
		st := ws.enhancer.Status()
		status.PromptEnhancer = &st
	}
	status.PublicAPI = ws.funnel.Status()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (ws *WebServer) handleInstances(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	statuses := []InstanceStatus{} // [] rather than null when there are none
	for _, inst := range ws.mgr.Instances() {
		statuses = append(statuses, inst.Status())
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(statuses)
}

func (ws *WebServer) handleInstanceAction(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/api/instances/")
	parts := strings.SplitN(path, "/", 2)
	if len(parts) < 1 || parts[0] == "" {
		http.NotFound(w, r)
		return
	}

	name := parts[0] // r.URL.Path is already decoded; decoding again mangles '%'
	inst := ws.mgr.Get(name)
	if inst == nil {
		http.Error(w, "instance not found", http.StatusNotFound)
		return
	}

	if len(parts) == 1 || parts[1] == "" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(inst.Status())
		return
	}

	action := parts[1]

	switch action {
	case "logs":
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		lines := inst.Logs()
		n := 100
		if q := r.URL.Query().Get("n"); q != "" {
			if parsed, err := strconv.Atoi(q); err == nil && parsed > 0 {
				n = parsed
			}
		}
		if len(lines) > n {
			lines = lines[len(lines)-n:]
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(lines)

	case "start":
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if ws.fetchingModel(inst) {
			writeJSON(w, http.StatusAccepted, map[string]string{"status": "downloading"})
			return
		}
		if err := ws.mgr.StartInstance(name); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

	case "stop":
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if err := ws.mgr.StopInstance(name); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

	case "restart":
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if ws.fetchingModel(inst) {
			writeJSON(w, http.StatusAccepted, map[string]string{"status": "downloading"})
			return
		}
		if err := ws.mgr.RestartInstance(name); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

	case "chat":
		ws.proxyChat(w, r, inst)

	case "props":
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		p, err := inst.FetchProps(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		writeJSON(w, http.StatusOK, p)

	default:
		http.NotFound(w, r)
	}
}

// fetchingModel is true when inst is an image or voice model whose files are
// being downloaded (starting the download now if any is missing); it then
// starts by itself once they are in.
func (ws *WebServer) fetchingModel(inst *Instance) bool {
	switch inst.conf.role {
	case roleImageGen:
		return ws.imageGen != nil && ws.imageGen.fetch(inst.conf.Name)
	case roleSpeech:
		return ws.speech != nil && ws.speech.fetch(inst.conf.Name)
	}
	return false
}

// proxyChat forwards the request body to <inst>/v1/chat/completions on the
// underlying llama-server (forwardLlama relays streams as they arrive).
func (ws *WebServer) proxyChat(w http.ResponseWriter, r *http.Request, inst *Instance) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if s := inst.State(); s != StateRunning {
		http.Error(w, fmt.Sprintf("instance is %s, not running", s), http.StatusServiceUnavailable)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxChatBody)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "reading request: "+err.Error(), http.StatusBadRequest)
		return
	}
	forwardLlama(w, r, inst, "/v1/chat/completions", body)
}

func (ws *WebServer) handleGPUs(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	ws.cfg.mu.RLock()
	backend := ws.cfg.GPUBackend
	ws.cfg.mu.RUnlock()
	stats, errMsg := readGPUs(backend)
	w.Header().Set("Content-Type", "application/json")
	resp := map[string]interface{}{
		"backend": backend,
		"gpus":    stats,
	}
	if errMsg != "" {
		resp["error"] = errMsg
	}
	json.NewEncoder(w).Encode(resp)
}

func (ws *WebServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	instances := ws.mgr.Instances()

	type metricsResult struct {
		name    string
		metrics *InstanceMetrics
	}

	ch := make(chan metricsResult, len(instances))
	var wg sync.WaitGroup
	for _, inst := range instances {
		wg.Add(1)
		go func(inst *Instance) {
			defer wg.Done()
			m := inst.FetchMetrics()
			if m != nil {
				ch <- metricsResult{name: inst.conf.Name, metrics: m}
			}
		}(inst)
	}
	wg.Wait()
	close(ch)

	result := make(map[string]*InstanceMetrics)
	for mr := range ch {
		result[mr.name] = mr.metrics
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func (ws *WebServer) handleBulkAction(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	action := strings.TrimPrefix(r.URL.Path, "/api/instances/all/")
	switch action {
	case "start":
		for _, inst := range ws.mgr.Instances() {
			s := inst.State()
			if (s == StateStopped || s == StateCrashed) && !ws.fetchingModel(inst) {
				ws.mgr.StartInstance(inst.conf.Name)
			}
		}
	case "stop":
		for _, inst := range ws.mgr.Instances() {
			ws.mgr.StopInstance(inst.conf.Name)
		}
	case "restart":
		instances := ws.mgr.Instances()
		go func() {
			for _, inst := range instances {
				if !ws.fetchingModel(inst) {
					ws.mgr.RestartInstance(inst.conf.Name)
				}
			}
		}()
	default:
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (ws *WebServer) handleModels(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		ws.cfg.mu.RLock()
		extra := append([]string(nil), ws.cfg.ModelDirs...)
		ws.cfg.mu.RUnlock()
		models, err := scanCachedModels(extra)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		disks, idx := diskUsage(models, getCacheDir())
		for i := range models {
			models[i].Disk = idx[i]
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"cache_dir":  getCacheDir(),
			"model_dirs": extra,
			"models":     models,
			"disks":      disks,
		})
	case http.MethodDelete:
		path := r.URL.Query().Get("path")
		if path == "" {
			http.Error(w, "path parameter is required", http.StatusBadRequest)
			return
		}
		if err := deleteCachedModel(path); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (ws *WebServer) handleModelQuants(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	repo := r.URL.Query().Get("repo")
	if repo == "" {
		http.Error(w, "repo parameter is required", http.StatusBadRequest)
		return
	}
	quants, err := FetchQuants(repo)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(quants)
}

func (ws *WebServer) handleModelDownload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Repo  string `json:"repo"`
		Quant string `json:"quant"`
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Repo == "" {
		http.Error(w, "repo is required", http.StatusBadRequest)
		return
	}
	if err := ws.dlm.Start(req.Repo, req.Quant); err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (ws *WebServer) handleModelDownloadStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ws.dlm.GetStatus())
}

func (ws *WebServer) handleModelDownloadStop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	ws.dlm.Stop()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (ws *WebServer) handleConfigInstances(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ws.cfg.GetInstances())

	case http.MethodPost:
		var ic InstanceConf
		r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
		if err := json.NewDecoder(r.Body).Decode(&ic); err != nil {
			http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
			return
		}
		if err := validateInstance(ic); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if err := ws.cfg.AddInstance(ic); err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		if err := ws.mgr.AddInstance(ic); err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		ws.mgr.events.add(evConfig, ic.Name, "instance added")
		if ic.AutoStart {
			if err := ws.mgr.StartInstance(ic.Name); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ic)

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (ws *WebServer) handleConfigInstanceAction(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimPrefix(r.URL.Path, "/api/config/instances/") // already decoded
	if name == "" {
		http.NotFound(w, r)
		return
	}
	if inst := ws.mgr.Get(name); inst != nil && inst.conf.role == roleReader {
		http.Error(w, name+" is the image reader; change it in the image_recognition section of config.yaml on the server", http.StatusConflict)
		return
	}
	if inst := ws.mgr.Get(name); inst != nil && inst.conf.role == roleImageGen {
		http.Error(w, name+" is an image model; change it in the image_generation section of config.yaml on the server", http.StatusConflict)
		return
	}
	if inst := ws.mgr.Get(name); inst != nil && inst.conf.role == roleSpeech {
		http.Error(w, name+" is a voice model; change it in the text_to_speech section of config.yaml on the server", http.StatusConflict)
		return
	}
	if inst := ws.mgr.Get(name); inst != nil && inst.conf.role == rolePromptEnhancer {
		http.Error(w, name+" is the prompt enhancer; change it in the prompt_enhancer section of config.yaml on the server", http.StatusConflict)
		return
	}

	switch r.Method {
	case http.MethodPut:
		var ic InstanceConf
		var sent map[string]json.RawMessage
		r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
		body, err := io.ReadAll(r.Body)
		if err == nil {
			err = json.Unmarshal(body, &ic)
		}
		if err == nil {
			err = json.Unmarshal(body, &sent)
		}
		if err != nil {
			http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
			return
		}
		keepUnsentOverrides(&ic, sent, ws.cfg.GetInstances(), name)
		if err := validateInstance(ic); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		// Validate and persist first: a rejected edit must leave the running
		// instance alone.
		if err := ws.cfg.UpdateInstance(name, ic); err != nil {
			http.Error(w, err.Error(), configErrStatus(err, http.StatusBadRequest))
			return
		}
		ws.mgr.RemoveInstance(name)
		if err := ws.mgr.AddInstance(ic); err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		if ic.Name != name {
			ws.mgr.events.add(evConfig, ic.Name, "instance changed (was "+name+")")
		} else {
			ws.mgr.events.add(evConfig, ic.Name, "instance changed")
		}
		if ic.AutoStart {
			if err := ws.mgr.StartInstance(ic.Name); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ic)

	case http.MethodDelete:
		// Persist first: if the delete can't be saved the instance keeps running.
		if err := ws.cfg.DeleteInstance(name); err != nil {
			status := configErrStatus(err, http.StatusInternalServerError)
			if strings.HasSuffix(err.Error(), "not found") {
				status = http.StatusNotFound
			}
			http.Error(w, err.Error(), status)
			return
		}
		ws.mgr.RemoveInstance(name)
		ws.mgr.events.add(evConfig, name, "instance removed")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// configErrStatus maps a config mutation error to an HTTP status: 409 when
// the file was hand-edited since load, def otherwise.
func configErrStatus(err error, def int) int {
	if errors.Is(err, errConfigChangedOnDisk) {
		return http.StatusConflict
	}
	return def
}

// keepUnsentOverrides carries per-instance extra_args and jinja over from the
// stored instance when the edit doesn't mention them. They can only be set by
// hand in config.yaml, and a client that doesn't know about them (an older UI)
// must not wipe them. Sending the key, even as null, still sets it.
func keepUnsentOverrides(ic *InstanceConf, sent map[string]json.RawMessage, current []InstanceConf, name string) {
	for _, old := range current {
		if old.Name != name {
			continue
		}
		if _, ok := sent["extra_args"]; !ok {
			ic.ExtraArgs = old.ExtraArgs
		}
		if _, ok := sent["jinja"]; !ok {
			ic.Jinja = old.Jinja
		}
		return
	}
}

func (ws *WebServer) handleConfigExport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	ws.cfg.mu.RLock()
	path := ws.cfg.path
	ws.cfg.mu.RUnlock()
	data, err := os.ReadFile(path)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/x-yaml")
	w.Header().Set("Content-Disposition", "attachment; filename=\"config.yaml\"")
	w.Write(data)
}

func (ws *WebServer) handleConfigImport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadSize)
	file, _, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "file upload required: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "reading file: "+err.Error(), http.StatusBadRequest)
		return
	}

	incoming, err := decodeConfig(data)
	if err != nil {
		http.Error(w, "invalid yaml: "+err.Error(), http.StatusBadRequest)
		return
	}
	test := incoming

	ws.cfg.mu.Lock()
	// Omitting server_bin counts as changing it: the file would no longer load.
	if test.ServerBin != ws.cfg.ServerBin {
		ws.cfg.mu.Unlock()
		http.Error(w, "server_bin must be present and match the current value; edit the config file directly to change it", http.StatusForbidden)
		return
	}
	// The tika section decides where users' documents are sent.
	if test.Tika != ws.cfg.Tika {
		ws.cfg.mu.Unlock()
		http.Error(w, "the tika section must match the current one; edit config.yaml on the server to change where documents are sent", http.StatusForbidden)
		return
	}
	// image_recognition decides what else gets exec'd, on which port.
	if !reflect.DeepEqual(test.ImageRecognition, ws.cfg.ImageRecognition) {
		ws.cfg.mu.Unlock()
		http.Error(w, "the image_recognition section must match the current one; edit config.yaml on the server to change the image reader", http.StatusForbidden)
		return
	}
	// image_generation names a whole other binary to exec.
	if !reflect.DeepEqual(test.ImageGeneration, ws.cfg.ImageGeneration) {
		ws.cfg.mu.Unlock()
		http.Error(w, "the image_generation section must match the current one; edit config.yaml on the server to change image generation", http.StatusForbidden)
		return
	}
	// text_to_speech names binaries to exec and voice files to read.
	if !reflect.DeepEqual(test.TextToSpeech, ws.cfg.TextToSpeech) {
		ws.cfg.mu.Unlock()
		http.Error(w, "the text_to_speech section must match the current one; edit config.yaml on the server to change text to speech", http.StatusForbidden)
		return
	}
	// prompt_enhancer decides what else gets exec'd, on which port.
	if !reflect.DeepEqual(test.PromptEnhancer, ws.cfg.PromptEnhancer) {
		ws.cfg.mu.Unlock()
		http.Error(w, "the prompt_enhancer section must match the current one; edit config.yaml on the server to change the prompt enhancer", http.StatusForbidden)
		return
	}
	// public_api decides who on the internet reaches /v1, and runs tailscale.
	if test.PublicAPI != ws.cfg.PublicAPI {
		ws.cfg.mu.Unlock()
		http.Error(w, "the public_api section must match the current one; edit config.yaml on the server to change internet access", http.StatusForbidden)
		return
	}
	// cors_origins decides which web pages may call /v1 from a browser.
	if !slices.Equal(test.CORSOrigins, ws.cfg.CORSOrigins) {
		ws.cfg.mu.Unlock()
		http.Error(w, "cors_origins must match the current list; edit config.yaml on the server to change which web pages may call /v1", http.StatusForbidden)
		return
	}
	if err := test.validate(); err != nil {
		ws.cfg.mu.Unlock()
		http.Error(w, "invalid config: "+err.Error(), http.StatusBadRequest)
		return
	}
	// An import deliberately replaces the whole file, hand edits included.
	if err := writeFileAtomic(ws.cfg.path, data); err != nil {
		ws.cfg.mu.Unlock()
		http.Error(w, "writing config: "+err.Error(), http.StatusInternalServerError)
		return
	}
	ws.cfg.setDiskSumLocked(data)
	ws.cfg.applyLocked(test)
	confs := ws.cfg.managedInstancesLocked()
	ws.cfg.mu.Unlock()

	added, updated, removed := ws.mgr.ReconcileInstances(confs)
	ws.mgr.events.add(evConfig, "", fmt.Sprintf("config.yaml imported: %d instance(s) added, %d changed, %d removed", len(added), len(updated), len(removed)))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":  "ok",
		"message": fmt.Sprintf("config imported: %d instance(s) added, %d updated, %d removed", len(added), len(updated), len(removed)),
		"added":   added,
		"updated": updated,
		"removed": removed,
	})
}

func (ws *WebServer) handleSettings(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ws.settingsWithUsage())

	case http.MethodPut:
		var s Settings
		r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
		if err := json.NewDecoder(r.Body).Decode(&s); err != nil {
			http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
			return
		}
		if err := ws.cfg.UpdateSettings(s); err != nil {
			http.Error(w, err.Error(), configErrStatus(err, http.StatusBadRequest))
			return
		}
		ws.mgr.events.add(evConfig, "", "settings saved")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ws.settingsWithUsage())

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (ws *WebServer) settingsWithUsage() Settings {
	s := ws.cfg.GetSettings()
	if s.APIKeys != nil && len(*s.APIKeys) > 0 {
		s.APIUsage = ws.usage.forKeys(*s.APIKeys, time.Now())
	}
	return s
}
