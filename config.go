package main

import (
	"fmt"
	"os"
	"strconv"
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
	GPUBackend          string         `yaml:"gpu_backend" json:"gpu_backend"`
	Host                string         `yaml:"host" json:"host"`
	NGL                 int            `yaml:"ngl" json:"ngl"`
	MainGPU             int            `yaml:"main_gpu" json:"main_gpu"`
	ContextLength       int            `yaml:"context_length" json:"context_length"`
	CacheTypeK          string         `yaml:"cache_type_k" json:"cache_type_k"`
	CacheTypeV          string         `yaml:"cache_type_v" json:"cache_type_v"`
	Instances           []InstanceConf `yaml:"instances" json:"instances"`

	mu   sync.RWMutex `yaml:"-" json:"-"`
	path string       `yaml:"-" json:"-"`
}

type InstanceConf struct {
	Name          string  `yaml:"name" json:"name"`
	Model         string  `yaml:"model" json:"model"`
	Port          int     `yaml:"port" json:"port"`
	GPUIDs        []int   `yaml:"gpu_ids" json:"gpu_ids"`
	NGL           *int    `yaml:"ngl,omitempty" json:"ngl,omitempty"`
	ContextLength *int    `yaml:"context_length,omitempty" json:"context_length,omitempty"`
	CacheTypeK    *string `yaml:"cache_type_k,omitempty" json:"cache_type_k,omitempty"`
	CacheTypeV    *string `yaml:"cache_type_v,omitempty" json:"cache_type_v,omitempty"`
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

	cfg := &Config{
		ManagerPort:         8080,
		RestartDelay:        duration{5 * time.Second},
		MaxRestarts:         10,
		HealthCheckInterval: duration{30 * time.Second},
		GPUBackend:          "vulkan",
		Host:                "0.0.0.0",
		NGL:                 99,
		MainGPU:             0,
		ContextLength:       16384,
		CacheTypeK:          "q8_0",
		CacheTypeV:          "q8_0",
		path:                path,
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}

	if cfg.ServerBin == "" {
		return nil, fmt.Errorf("server_bin is required")
	}

	return cfg, nil
}

type Settings struct {
	ServerBin           string `json:"server_bin"`
	ManagerPort         int    `json:"manager_port"`
	RestartDelay        string `json:"restart_delay"`
	MaxRestarts         int    `json:"max_restarts"`
	HealthCheckInterval string `json:"health_check_interval"`
	GPUBackend          string `json:"gpu_backend"`
	Host                string `json:"host"`
	NGL                 int    `json:"ngl"`
	MainGPU             int    `json:"main_gpu"`
	ContextLength       int    `json:"context_length"`
	CacheTypeK          string `json:"cache_type_k"`
	CacheTypeV          string `json:"cache_type_v"`
}

func (cfg *Config) GetSettings() Settings {
	cfg.mu.RLock()
	defer cfg.mu.RUnlock()
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
	}
}

func (cfg *Config) UpdateSettings(s Settings) error {
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
	if s.GPUBackend != "" {
		validBackends := map[string]bool{"vulkan": true, "cuda": true, "rocm": true, "rocm_rocr": true, "metal": true}
		if !validBackends[s.GPUBackend] {
			return fmt.Errorf("gpu_backend must be one of: vulkan, cuda, rocm, rocm_rocr")
		}
	}

	if s.ServerBin != "" {
		cfg.ServerBin = s.ServerBin
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

	return cfg.saveLocked()
}

func (cfg *Config) GetInstances() []InstanceConf {
	cfg.mu.RLock()
	defer cfg.mu.RUnlock()
	out := make([]InstanceConf, len(cfg.Instances))
	copy(out, cfg.Instances)
	return out
}

func (cfg *Config) AddInstance(ic InstanceConf) error {
	cfg.mu.Lock()
	defer cfg.mu.Unlock()
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
	return os.WriteFile(cfg.path, data, 0644)
}
