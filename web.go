package main

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

const (
	maxJSONBody   = 1 << 20
	maxUploadSize = 10 << 20
	maxChatBody   = 4 << 20 // 4 MiB — chat conversations can grow.
	chatTimeout   = 10 * time.Minute
)

//go:embed templates/index.html
var templateFS embed.FS

type WebServer struct {
	mgr  *Manager
	cfg  *Config
	dlm  *DownloadManager
	tmpl *template.Template
	mux  *http.ServeMux
}

type ServerStatus struct {
	Name      string  `json:"name"`
	Uptime    string  `json:"uptime"`
	UptimeSec float64 `json:"uptime_sec"`
}

func NewWebServer(mgr *Manager, cfg *Config, dlm *DownloadManager) *WebServer {
	tmpl := template.Must(template.ParseFS(templateFS, "templates/index.html"))
	ws := &WebServer{
		mgr:  mgr,
		cfg:  cfg,
		dlm:  dlm,
		tmpl: tmpl,
		mux:  http.NewServeMux(),
	}
	ws.mux.HandleFunc("/", ws.handleIndex)
	ws.mux.HandleFunc("/api/status", ws.handleStatus)
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
	ws.mux.HandleFunc("/v1/models", ws.handleV1Models)
	ws.mux.HandleFunc("/v1/chat/completions", ws.handleV1Chat)
	return ws
}

func (ws *WebServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
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

func (ws *WebServer) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	ws.tmpl.Execute(w, nil)
}

func (ws *WebServer) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	hostname, _ := os.Hostname()
	uptime := getSystemUptime()
	status := ServerStatus{
		Name:      hostname,
		Uptime:    formatDuration(uptime),
		UptimeSec: uptime.Seconds(),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (ws *WebServer) handleInstances(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var statuses []InstanceStatus
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

	name, err := url.PathUnescape(parts[0])
	if err != nil {
		http.Error(w, "invalid instance name", http.StatusBadRequest)
		return
	}
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
		if err := ws.mgr.RestartInstance(name); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

	case "chat":
		ws.proxyChat(w, r, inst)

	default:
		http.NotFound(w, r)
	}
}

// proxyChat forwards the request body to <inst>/v1/chat/completions on the
// underlying llama-server. If the JSON body has "stream": true we switch to a
// chunked passthrough using http.Flusher so SSE events reach the browser
// immediately. Wall-clock time is reported via X-Manager-Elapsed-Ms (set
// before WriteHeader so it's actually delivered) so the UI can show
// end-to-end latency in addition to llama.cpp's internal "timings".
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

	streaming := false
	var probe struct {
		Stream bool `json:"stream"`
	}
	if err := json.Unmarshal(body, &probe); err == nil {
		streaming = probe.Stream
	}

	ws.cfg.mu.RLock()
	host := ws.cfg.Host
	ws.cfg.mu.RUnlock()
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	target := fmt.Sprintf("http://%s:%d/v1/chat/completions", host, inst.conf.Port)

	start := time.Now()
	client := &http.Client{Timeout: chatTimeout}
	resp, err := client.Post(target, "application/json", bytes.NewReader(body))
	if err != nil {
		http.Error(w, "proxying to llama-server: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	w.Header().Set("X-Manager-Elapsed-Ms", strconv.FormatInt(time.Since(start).Milliseconds(), 10))

	if streaming {
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

// handleV1Models returns the OpenAI-compatible model list. Each running
// instance becomes one model whose id is the instance name.
func (ws *WebServer) handleV1Models(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	var models []model
	for _, inst := range ws.mgr.Instances() {
		s := inst.Status()
		models = append(models, model{
			ID:      s.Name,
			Object:  "model",
			Created: 0,
			OwnedBy: "llama-manager",
		})
	}
	if models == nil {
		models = []model{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   models,
	})
}

// handleV1Chat is the OpenAI-compatible aggregate endpoint. The request body
// must include "model": "<instance-name>". We rewrite it (drop the model
// field; llama-server picks the loaded model itself) and proxy through the
// existing per-instance path, which already handles streaming.
func (ws *WebServer) handleV1Chat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxChatBody)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "reading request: "+err.Error(), http.StatusBadRequest)
		return
	}
	var preview map[string]interface{}
	if err := json.Unmarshal(body, &preview); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}
	modelName, _ := preview["model"].(string)
	if modelName == "" {
		http.Error(w, `"model" field is required (instance name)`, http.StatusBadRequest)
		return
	}
	inst := ws.mgr.Get(modelName)
	if inst == nil {
		http.Error(w, "model not found: "+modelName, http.StatusNotFound)
		return
	}
	// Re-wrap the body so proxyChat reads from a fresh reader.
	r.Body = io.NopCloser(bytes.NewReader(body))
	ws.proxyChat(w, r, inst)
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
			if s == StateStopped || s == StateCrashed {
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
				ws.mgr.RestartInstance(inst.conf.Name)
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
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"cache_dir":  getCacheDir(),
			"model_dirs": extra,
			"models":     models,
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
		if ic.Name == "" || ic.Model == "" || ic.Port == 0 {
			http.Error(w, "name, model, and port are required", http.StatusBadRequest)
			return
		}
		if len(ic.GPUIDs) == 0 {
			http.Error(w, "gpu_ids must contain at least one GPU ID", http.StatusBadRequest)
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
	rawName := strings.TrimPrefix(r.URL.Path, "/api/config/instances/")
	if rawName == "" {
		http.NotFound(w, r)
		return
	}
	name, err := url.PathUnescape(rawName)
	if err != nil {
		http.Error(w, "invalid instance name", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodPut:
		var ic InstanceConf
		r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
		if err := json.NewDecoder(r.Body).Decode(&ic); err != nil {
			http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
			return
		}
		if ic.Name == "" || ic.Model == "" || ic.Port == 0 {
			http.Error(w, "name, model, and port are required", http.StatusBadRequest)
			return
		}
		if len(ic.GPUIDs) == 0 {
			http.Error(w, "gpu_ids must contain at least one GPU ID", http.StatusBadRequest)
			return
		}
		ws.mgr.RemoveInstance(name)
		if err := ws.cfg.UpdateInstance(name, ic); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if err := ws.mgr.AddInstance(ic); err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
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
		ws.mgr.RemoveInstance(name)
		if err := ws.cfg.DeleteInstance(name); err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
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

	var test Config
	if err := yaml.Unmarshal(data, &test); err != nil {
		http.Error(w, "invalid yaml: "+err.Error(), http.StatusBadRequest)
		return
	}

	ws.cfg.mu.Lock()
	if test.ServerBin != "" && test.ServerBin != ws.cfg.ServerBin {
		ws.cfg.mu.Unlock()
		http.Error(w, "server_bin cannot be changed via import; edit the config file directly", http.StatusForbidden)
		return
	}
	if err := os.WriteFile(ws.cfg.path, data, 0640); err != nil {
		ws.cfg.mu.Unlock()
		http.Error(w, "writing config: "+err.Error(), http.StatusInternalServerError)
		return
	}
	if test.Host != "" {
		ws.cfg.Host = test.Host
	}
	if test.GPUBackend != "" {
		ws.cfg.GPUBackend = test.GPUBackend
	}
	if test.RestartDelay.Duration > 0 {
		ws.cfg.RestartDelay = test.RestartDelay
	}
	if test.HealthCheckInterval.Duration > 0 {
		ws.cfg.HealthCheckInterval = test.HealthCheckInterval
	}
	if test.MaxRestarts > 0 {
		ws.cfg.MaxRestarts = test.MaxRestarts
	}
	if test.NGL > 0 {
		ws.cfg.NGL = test.NGL
	}
	if test.ContextLength > 0 {
		ws.cfg.ContextLength = test.ContextLength
	}
	if test.CacheTypeK != "" {
		ws.cfg.CacheTypeK = test.CacheTypeK
	}
	if test.CacheTypeV != "" {
		ws.cfg.CacheTypeV = test.CacheTypeV
	}
	ws.cfg.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "message": "config imported, settings applied. restart to apply instance changes"})
}

func (ws *WebServer) handleSettings(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ws.cfg.GetSettings())

	case http.MethodPut:
		var s Settings
		r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
		if err := json.NewDecoder(r.Body).Decode(&s); err != nil {
			http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
			return
		}
		if err := ws.cfg.UpdateSettings(s); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ws.cfg.GetSettings())

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}
