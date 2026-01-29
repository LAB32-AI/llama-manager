package main

import (
	"embed"
	"encoding/json"
	"html/template"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

const (
	maxJSONBody   = 1 << 20
	maxUploadSize = 10 << 20
)

//go:embed templates/index.html
var templateFS embed.FS

type WebServer struct {
	mgr     *Manager
	cfg     *Config
	dlm     *DownloadManager
	tmpl    *template.Template
	mux     *http.ServeMux
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
		ws.mgr.StartInstance(name)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

	case "stop":
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		ws.mgr.StopInstance(name)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

	case "restart":
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		ws.mgr.RestartInstance(name)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

	default:
		http.NotFound(w, r)
	}
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
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	models, err := scanCachedModels()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"cache_dir": getCacheDir(),
		"models":    models,
	})
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
		ws.mgr.AddInstance(ic)
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
		ws.mgr.AddInstance(ic)
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
	if err := os.WriteFile(ws.cfg.path, data, 0644); err != nil {
		ws.cfg.mu.Unlock()
		http.Error(w, "writing config: "+err.Error(), http.StatusInternalServerError)
		return
	}
	if test.ServerBin != "" {
		ws.cfg.ServerBin = test.ServerBin
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
