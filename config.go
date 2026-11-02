package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	ServerBin           string         `yaml:"server_bin" json:"server_bin"`
	ManagerPort         int            `yaml:"manager_port" json:"manager_port"`
	RestartDelay        duration       `yaml:"restart_delay" json:"restart_delay"`
	MaxRestarts         int            `yaml:"max_restarts" json:"max_restarts"`
	HealthCheckInterval duration       `yaml:"health_check_interval" json:"health_check_interval"`
	HealthCheckTimeout  duration       `yaml:"health_check_timeout" json:"health_check_timeout"`
	UnhealthyAfter      int            `yaml:"unhealthy_after" json:"unhealthy_after"`
	GPUBackend          string         `yaml:"gpu_backend" json:"gpu_backend"`
	Host                string         `yaml:"host" json:"host"`
	NGL                 int            `yaml:"ngl" json:"ngl"`
	MainGPU             int            `yaml:"main_gpu" json:"main_gpu"`
	ContextLength       int            `yaml:"context_length" json:"context_length"`
	CacheTypeK          string         `yaml:"cache_type_k" json:"cache_type_k"`
	CacheTypeV          string         `yaml:"cache_type_v" json:"cache_type_v"`
	Jinja               bool           `yaml:"jinja" json:"jinja"`
	ExtraArgs           []string       `yaml:"extra_args" json:"extra_args"`
	ModelDirs           []string       `yaml:"model_dirs" json:"model_dirs"`
	AllowedHosts        []string       `yaml:"allowed_hosts" json:"allowed_hosts"`
	Instances           []InstanceConf `yaml:"instances" json:"instances"`

	mu   sync.RWMutex `yaml:"-" json:"-"`
	path string       `yaml:"-" json:"-"`
}

type InstanceConf struct {
	Name          string    `yaml:"name" json:"name"`
	Model         string    `yaml:"model" json:"model"`
	Port          int       `yaml:"port" json:"port"`
	GPUIDs        []int     `yaml:"gpu_ids" json:"gpu_ids"`
	AutoStart     bool      `yaml:"auto_start" json:"auto_start"`
	NGL           *int      `yaml:"ngl,omitempty" json:"ngl,omitempty"`
	ContextLength *int      `yaml:"context_length,omitempty" json:"context_length,omitempty"`
	CacheTypeK    *string   `yaml:"cache_type_k,omitempty" json:"cache_type_k,omitempty"`
	CacheTypeV    *string   `yaml:"cache_type_v,omitempty" json:"cache_type_v,omitempty"`
	Jinja         *bool     `yaml:"jinja,omitempty" json:"jinja,omitempty"`
	ExtraArgs     *[]string `yaml:"extra_args,omitempty" json:"extra_args,omitempty"`
}

func (ic *InstanceConf) UnmarshalYAML(value *yaml.Node) error {
	type rawConf InstanceConf
	var raw rawConf
	if err := value.Decode(&raw); err != nil {
		return err
	}
	*ic = InstanceConf(raw)

	if len(ic.GPUIDs) == 0 {
		for i := 0; i < len(value.Content)-1; i += 2 {
			if value.Content[i].Value == "gpu_id" {
				id, err := strconv.Atoi(value.Content[i+1].Value)
				if err != nil {
					return fmt.Errorf("invalid gpu_id: %w", err)
				}
				ic.GPUIDs = []int{id}
				break
			}
		}
	}

	return nil
}

// supervisorParams is a consistent snapshot of the settings the supervisor and
// health loop act on; they can change at any time via PUT /api/settings.
type supervisorParams struct {
	maxRestarts    int
	restartDelay   time.Duration
	healthInterval time.Duration
	unhealthyAfter int
}

func (cfg *Config) supervisorParams() supervisorParams {
	cfg.mu.RLock()
	defer cfg.mu.RUnlock()
	return supervisorParams{
		maxRestarts:    cfg.MaxRestarts,
		restartDelay:   cfg.RestartDelay.Duration,
		healthInterval: cfg.HealthCheckInterval.Duration,
		unhealthyAfter: cfg.UnhealthyAfter,
	}
}

func (cfg *Config) GPUEnvVar() string {
	switch cfg.GPUBackend {
	case "cuda":
		return "CUDA_VISIBLE_DEVICES"
	case "rocm":
		return "HIP_VISIBLE_DEVICES"
	case "rocm_rocr":
		return "ROCR_VISIBLE_DEVICES"
	case "metal":
		return ""
	default:
		return "GGML_VK_VISIBLE_DEVICES"
	}
}

type duration struct {
	time.Duration
}

func (d *duration) UnmarshalYAML(value *yaml.Node) error {
	var s string
	if err := value.Decode(&s); err != nil {
		return err
	}
	parsed, err := time.ParseDuration(s)
	if err != nil {
		return fmt.Errorf("invalid duration %q: %w", s, err)
	}
	d.Duration = parsed
	return nil
}

func (d duration) MarshalYAML() (interface{}, error) {
	return d.Duration.String(), nil
}

func (d duration) MarshalJSON() ([]byte, error) {
	return []byte(`"` + d.Duration.String() + `"`), nil
}

func (d *duration) UnmarshalJSON(b []byte) error {
	s := string(b)
	if len(s) < 2 {
		return fmt.Errorf("invalid duration value: %s", s)
	}
	s = s[1 : len(s)-1]
	parsed, err := time.ParseDuration(s)
	if err != nil {
		return err
	}
	d.Duration = parsed
	return nil
}

func loadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config: %w", err)
	}
	cfg, err := decodeConfig(data)
	if err != nil {
		return nil, err
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("invalid config %s: %w", path, err)
	}
	cfg.path = path
	return cfg, nil
}

// decodeConfig parses YAML on top of the built-in defaults. It does not
// validate; callers decide what to check first.
func decodeConfig(data []byte) (*Config, error) {
	cfg := &Config{
		ManagerPort:         8080,
		RestartDelay:        duration{5 * time.Second},
		MaxRestarts:         10,
		HealthCheckInterval: duration{30 * time.Second},
		HealthCheckTimeout:  duration{5 * time.Second},
		UnhealthyAfter:      3,
		GPUBackend:          "vulkan",
		Host:                "0.0.0.0",
		NGL:                 99,
		MainGPU:             0,
		ContextLength:       16384,
		CacheTypeK:          "q8_0",
		CacheTypeV:          "q8_0",
		Jinja:               true,
	}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}
	return cfg, nil
}

var validBackends = map[string]bool{"vulkan": true, "cuda": true, "rocm": true, "rocm_rocr": true, "metal": true}

// validate checks a whole config with the same rules the API applies to single
// changes, so a hand-edited or imported file can't produce a state the API
// would refuse, or one that breaks the supervisor (a zero health interval used
// to panic it).
func (cfg *Config) validate() error {
	var errs []error
	check := func(ok bool, format string, args ...any) {
		if !ok {
			errs = append(errs, fmt.Errorf(format, args...))
		}
	}
	check(cfg.ServerBin != "", "server_bin is required")
	check(validPort(cfg.ManagerPort), "manager_port must be 1-65535")
	check(cfg.RestartDelay.Duration > 0, "restart_delay must be > 0")
	check(cfg.HealthCheckInterval.Duration > 0, "health_check_interval must be > 0")
	check(cfg.HealthCheckTimeout.Duration >= 0, "health_check_timeout must be >= 0")
	check(cfg.MaxRestarts >= 0, "max_restarts must be >= 0")
	check(cfg.UnhealthyAfter >= 0, "unhealthy_after must be >= 0")
	check(cfg.NGL >= 0, "ngl must be >= 0")
	check(cfg.MainGPU >= 0, "main_gpu must be >= 0")
	check(cfg.ContextLength > 0, "context_length must be > 0")
	check(validBackends[cfg.GPUBackend], "gpu_backend %q must be one of: vulkan, cuda, rocm, rocm_rocr, metal", cfg.GPUBackend)

	names := map[string]bool{}
	ports := map[int]bool{}
	for i, ic := range cfg.Instances {
		if err := validateInstance(ic); err != nil {
			errs = append(errs, fmt.Errorf("instances[%d] (%s): %w", i, ic.Name, err))
			continue
		}
		check(!names[ic.Name], "duplicate instance name: %q", ic.Name)
		check(!ports[ic.Port], "duplicate port: %d", ic.Port)
		check(ic.Port != cfg.ManagerPort, "instance %q uses the manager's own port %d", ic.Name, ic.Port)
		names[ic.Name] = true
		ports[ic.Port] = true
	}
	return errors.Join(errs...)
}

// applyLocked replaces every persisted setting with n's, so a later save
// writes what was imported instead of the stale in-memory copy. Exported
// fields are copied by reflection so a field added to Config later can't be
// forgotten here. cfg.mu must be held for writing.
func (cfg *Config) applyLocked(n *Config) {
	dst := reflect.ValueOf(cfg).Elem()
	src := reflect.ValueOf(n).Elem()
	for i := 0; i < dst.NumField(); i++ {
		if dst.Type().Field(i).IsExported() {
			dst.Field(i).Set(src.Field(i))
		}
	}
}

// instanceNameRe keeps names safe in URL paths and HTML: letters, digits,
// space, dot, underscore and dash, starting with a letter or digit.
var instanceNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$`)

func validateInstanceName(name string) error {
	if !instanceNameRe.MatchString(name) || strings.TrimSpace(name) != name {
		return fmt.Errorf("invalid instance name %q: use letters, digits, space, . _ - (max 64)", name)
	}
	if name == "all" {
		return fmt.Errorf(`instance name "all" is reserved`)
	}
	return nil
}

func validPort(p int) bool { return p >= 1 && p <= 65535 }

// validateInstance checks one instance on its own; uniqueness across instances
// is the caller's job.
func validateInstance(ic InstanceConf) error {
	switch {
	case ic.Name == "":
		return fmt.Errorf("name is required")
	case validateInstanceName(ic.Name) != nil:
		return validateInstanceName(ic.Name)
	case ic.Model == "":
		return fmt.Errorf("model is required")
	case validateModelRef(ic.Model) != nil:
		return validateModelRef(ic.Model)
	case !validPort(ic.Port):
		return fmt.Errorf("port must be 1-65535")
	case len(ic.GPUIDs) == 0:
		return fmt.Errorf("gpu_ids must contain at least one GPU ID")
	}
	for _, id := range ic.GPUIDs {
		if id < 0 {
			return fmt.Errorf("gpu id %d must be >= 0", id)
		}
	}
	if ic.NGL != nil && *ic.NGL < 0 {
		return fmt.Errorf("ngl must be >= 0")
	}
	if ic.ContextLength != nil && *ic.ContextLength <= 0 {
		return fmt.Errorf("context_length must be > 0")
	}
	return nil
}

type Settings struct {
	ServerBin           string    `json:"server_bin"`
	ManagerPort         int       `json:"manager_port"`
	RestartDelay        string    `json:"restart_delay"`
	MaxRestarts         int       `json:"max_restarts"`
	HealthCheckInterval string    `json:"health_check_interval"`
	GPUBackend          string    `json:"gpu_backend"`
	Host                string    `json:"host"`
	NGL                 int       `json:"ngl"`
	MainGPU             int       `json:"main_gpu"`
	ContextLength       int       `json:"context_length"`
	CacheTypeK          string    `json:"cache_type_k"`
	CacheTypeV          string    `json:"cache_type_v"`
	Jinja               *bool     `json:"jinja,omitempty"`
	ExtraArgs           *[]string `json:"extra_args,omitempty"`
	// ExtraArgsLine is ExtraArgs as one shell-quoted line, for the UI's text
	// field. When sent it takes precedence over ExtraArgs.
	ExtraArgsLine *string `json:"extra_args_line,omitempty"`
}

func (cfg *Config) GetSettings() Settings {
	cfg.mu.RLock()
	defer cfg.mu.RUnlock()
	jinja := cfg.Jinja
	extra := append([]string(nil), cfg.ExtraArgs...)
	line := joinArgs(extra)
	return Settings{
		ServerBin:           cfg.ServerBin,
		ManagerPort:         cfg.ManagerPort,
		RestartDelay:        cfg.RestartDelay.Duration.String(),
		MaxRestarts:         cfg.MaxRestarts,
		HealthCheckInterval: cfg.HealthCheckInterval.Duration.String(),
		GPUBackend:          cfg.GPUBackend,
		Host:                cfg.Host,
		NGL:                 cfg.NGL,
		MainGPU:             cfg.MainGPU,
		ContextLength:       cfg.ContextLength,
		CacheTypeK:          cfg.CacheTypeK,
		CacheTypeV:          cfg.CacheTypeV,
		Jinja:               &jinja,
		ExtraArgs:           &extra,
		ExtraArgsLine:       &line,
	}
}

func (cfg *Config) UpdateSettings(s Settings) error {
	if s.ExtraArgsLine != nil {
		args, err := splitArgs(*s.ExtraArgsLine)
		if err != nil {
			return fmt.Errorf("invalid extra_args: %w", err)
		}
		s.ExtraArgs = &args
	}

	cfg.mu.Lock()
	defer cfg.mu.Unlock()

	if s.MaxRestarts < 0 {
		return fmt.Errorf("max_restarts must be >= 0")
	}
	if s.NGL < 0 {
		return fmt.Errorf("ngl must be >= 0")
	}
	if s.MainGPU < 0 {
		return fmt.Errorf("main_gpu must be >= 0")
	}
	if s.ContextLength <= 0 {
		return fmt.Errorf("context_length must be > 0")
	}
	if s.GPUBackend != "" && !validBackends[s.GPUBackend] {
		return fmt.Errorf("gpu_backend must be one of: vulkan, cuda, rocm, rocm_rocr, metal")
	}

	if s.ServerBin != "" && s.ServerBin != cfg.ServerBin {
		return fmt.Errorf("server_bin cannot be changed via the API; edit the config file directly")
	}
	if s.RestartDelay != "" {
		d, err := time.ParseDuration(s.RestartDelay)
		if err != nil {
			return fmt.Errorf("invalid restart_delay: %w", err)
		}
		if d <= 0 {
			return fmt.Errorf("restart_delay must be > 0")
		}
		cfg.RestartDelay = duration{d}
	}
	if s.HealthCheckInterval != "" {
		d, err := time.ParseDuration(s.HealthCheckInterval)
		if err != nil {
			return fmt.Errorf("invalid health_check_interval: %w", err)
		}
		if d <= 0 {
			return fmt.Errorf("health_check_interval must be > 0")
		}
		cfg.HealthCheckInterval = duration{d}
	}
	cfg.MaxRestarts = s.MaxRestarts
	if s.GPUBackend != "" {
		cfg.GPUBackend = s.GPUBackend
	}
	if s.Host != "" {
		cfg.Host = s.Host
	}
	cfg.NGL = s.NGL
	cfg.MainGPU = s.MainGPU
	cfg.ContextLength = s.ContextLength
	if s.CacheTypeK != "" {
		cfg.CacheTypeK = s.CacheTypeK
	}
	if s.CacheTypeV != "" {
		cfg.CacheTypeV = s.CacheTypeV
	}
	if s.Jinja != nil {
		cfg.Jinja = *s.Jinja
	}
	if s.ExtraArgs != nil {
		cfg.ExtraArgs = append([]string(nil), (*s.ExtraArgs)...)
	}

	return cfg.saveLocked()
}

func (cfg *Config) GetInstances() []InstanceConf {
	cfg.mu.RLock()
	defer cfg.mu.RUnlock()
	out := make([]InstanceConf, len(cfg.Instances))
	copy(out, cfg.Instances)
	return out
}

func (cfg *Config) checkInstanceLocked(ic InstanceConf) error {
	if err := validateInstance(ic); err != nil {
		return err
	}
	if ic.Port == cfg.ManagerPort {
		return fmt.Errorf("port %d is the manager's own port", ic.Port)
	}
	return nil
}

func (cfg *Config) AddInstance(ic InstanceConf) error {
	cfg.mu.Lock()
	defer cfg.mu.Unlock()
	if err := cfg.checkInstanceLocked(ic); err != nil {
		return err
	}
	for _, existing := range cfg.Instances {
		if existing.Name == ic.Name {
			return fmt.Errorf("duplicate instance name: %q", ic.Name)
		}
		if existing.Port == ic.Port {
			return fmt.Errorf("duplicate port: %d", ic.Port)
		}
	}
	cfg.Instances = append(cfg.Instances, ic)
	return cfg.saveLocked()
}

func (cfg *Config) UpdateInstance(name string, ic InstanceConf) error {
	cfg.mu.Lock()
	defer cfg.mu.Unlock()
	if err := cfg.checkInstanceLocked(ic); err != nil {
		return err
	}
	for i, existing := range cfg.Instances {
		if existing.Name == name {
			for j, other := range cfg.Instances {
				if i != j && other.Port == ic.Port {
					return fmt.Errorf("duplicate port: %d", ic.Port)
				}
				if i != j && other.Name == ic.Name {
					return fmt.Errorf("duplicate instance name: %q", ic.Name)
				}
			}
			cfg.Instances[i] = ic
			return cfg.saveLocked()
		}
	}
	return fmt.Errorf("instance %q not found", name)
}

func (cfg *Config) DeleteInstance(name string) error {
	cfg.mu.Lock()
	defer cfg.mu.Unlock()
	for i, existing := range cfg.Instances {
		if existing.Name == name {
			cfg.Instances = append(cfg.Instances[:i], cfg.Instances[i+1:]...)
			return cfg.saveLocked()
		}
	}
	return fmt.Errorf("instance %q not found", name)
}

func (cfg *Config) saveLocked() error {
	if cfg.path == "" {
		return nil
	}
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("marshaling config: %w", err)
	}
	return writeFileAtomic(cfg.path, data)
}

// writeFileAtomic replaces path with data so that a crash or a full disk
// mid-write leaves either the old file or the new one, never a truncated mix:
// write a temp file in the same directory, fsync it, then rename it over the
// original. The existing file's permissions are kept, and a symlinked config is
// updated through the link.
func writeFileAtomic(path string, data []byte) error {
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		path = resolved
	}
	mode := os.FileMode(0o640)
	if fi, err := os.Stat(path); err == nil {
		mode = fi.Mode().Perm()
	}
	dir := filepath.Dir(path)
	f, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("writing config: %w", err)
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err := f.Write(data); err != nil {
		f.Close()
		return fmt.Errorf("writing config: %w", err)
	}
	if err := f.Chmod(mode); err != nil {
		f.Close()
		return fmt.Errorf("writing config: %w", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("writing config: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("writing config: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("writing config: %w", err)
	}
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		d.Close()
	}
	return nil
}
