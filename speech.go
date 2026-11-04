package main

import (
	"bytes"
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// Text to speech (text_to_speech in config.yaml) reads chat replies aloud.
// Each voice model runs as its own instance: Qwen3-TTS and LFM3-Audio in
// llama-server (its /tts endpoint, llama.cpp v0.8.0 and newer), Ming 3.0 in
// audio.cpp's audiocpp_server. The chat's play button asks for one part of a
// reply at a time, so the first sentence plays while the rest is spoken.

const (
	roleSpeech         = "tts"
	defaultSpeechPort  = 9110
	defaultSpeechModel = "ming3-tts"
	defaultSpeechWait  = 2 * time.Minute // per part
	speechCtx          = 4096
	speechSeed         = 42
	maxSpeechWAV       = 64 << 20
	maxVoiceFileBytes  = 5 << 20
	maxSpeechInput     = 4096 // runes, /v1/audio/speech, as OpenAI's
	maxSpeechText      = 20000
	speechFirstPart    = 180 // runes; short, so playback starts soon
	speechPartRunes    = 450
	speechCacheBytes   = 128 << 20
)

type speechEngine string

const (
	engineLlama    speechEngine = "llama-server"
	engineAudioCpp speechEngine = "audio.cpp"
)

type speechVoice struct {
	ID    string `json:"id"`
	Label string `json:"label"`
	file  string // a reference clip, for models that clone a voice
}

type speechPreset struct {
	Label   string
	Ref     string // shown as the instance's model; llama-server's -hf
	License string
	Engine  speechEngine
	File    *imageFile     // audio.cpp: the model, pinned
	Family  string         // audio.cpp's model family
	Options map[string]any // audio.cpp request options for every part
	Voices  []speechVoice  // built in
	Clone   bool           // speaks with the reference clips under voices
	Langs   []string
	Frames  int // llama-server: the most audio frames one part may take
	VRAM    string
}

// speechPresets are the voice models llama-manager can run. Ming's file is
// Q8_0 (it keeps a small BF16 head): gfx906 has no BF16 math.
var speechPresets = map[string]speechPreset{
	"ming3-tts": {
		Label: "Ming 3.0 TTS 0.5B", Ref: "audio-cpp/Ming-3.0-TTS-0.5B-GGUF:Q8_0", License: "MIT", Engine: engineAudioCpp,
		File: &imageFile{Repo: "audio-cpp/Ming-3.0-TTS-0.5B-GGUF", Rev: "2eb8d37c7f10672e69c8aa33091baa04c461b66c",
			Path: "ming-3.0-tts-0.5b-q8_0.gguf", Size: 1742280896, SHA256: "fcf2aa25e8e9cf2065a936e506581735c64d6255fb7869f09fbb61a2dde09aa1"},
		Family: "ming3_tts",
		// The ip template speaks with one of the model's own voices.
		Options: map[string]any{"template_name": "ip"},
		Voices: []speechVoice{{ID: "US_Emma", Label: "Emma (American)"}, {ID: "US_James", Label: "James (American)"},
			{ID: "UK_Charlotte", Label: "Charlotte (British)"}, {ID: "UK_George", Label: "George (British)"}},
		Langs: []string{"en", "zh", "ja", "ko", "de", "fr", "es"}, VRAM: "~2 GB",
	},
	"qwen3-tts": {
		Label: "Qwen3-TTS 1.7B", Ref: "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF:Q8_0", License: "Apache-2.0", Engine: engineLlama,
		Clone: true, Langs: []string{"en", "ru", "zh", "de", "fr", "es", "it", "pt", "ja", "ko"}, Frames: 1024, VRAM: "~3 GB",
	},
	"lfm3-audio": {
		Label: "LFM3-Audio 2B", Ref: "LiquidAI/LFM3-Audio-2B-GGUF:Q8_0", License: "LFM Open License v1.0: commercial use only under US$10M revenue",
		Engine: engineLlama, Voices: []speechVoice{{ID: "female", Label: "Female"}, {ID: "male", Label: "Male"}},
		Langs:  []string{"en", "de", "fr", "es", "it", "pt", "nl", "da", "sv", "fi", "pl", "tr", "el", "ar", "he", "ja", "ko", "zh"},
		Frames: 1024, VRAM: "~3 GB",
	},
}

func speechPresetNames() []string {
	names := make([]string, 0, len(speechPresets))
	for n := range speechPresets {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

func speechPresetByKey(m string) (string, speechPreset, bool) {
	k := strings.ToLower(strings.TrimSpace(m))
	if k == "" {
		k = defaultSpeechModel
	}
	p, ok := speechPresets[k]
	return k, p, ok
}

// SpeechConf runs voice models for reading replies aloud. It picks binaries,
// their arguments, ports and files to read, so like image_generation it is
// set only in config.yaml. Model/port/gpu_ids/voices name one model, or
// Models lists several.
type SpeechConf struct {
	Enabled bool `yaml:"enabled" json:"enabled"`
	// ServerBin is audio.cpp's audiocpp_server, for ming3-tts; the
	// llama-server models use the top-level server_bin.
	ServerBin string `yaml:"server_bin,omitempty" json:"server_bin,omitempty"`
	// Backend is the one audio.cpp runs on: hip, vulkan, cuda, metal or cpu.
	// The default follows gpu_backend.
	Backend   string            `yaml:"backend,omitempty" json:"backend,omitempty"`
	Model     string            `yaml:"model,omitempty" json:"model,omitempty"`
	Port      int               `yaml:"port,omitempty" json:"port,omitempty"`
	GPUIDs    []int             `yaml:"gpu_ids,omitempty" json:"gpu_ids,omitempty"`
	ExtraArgs []string          `yaml:"extra_args,omitempty" json:"extra_args,omitempty"`
	Voices    []SpeechVoiceConf `yaml:"voices,omitempty" json:"voices,omitempty"`
	Timeout   duration          `yaml:"timeout,omitempty" json:"timeout,omitempty"`
	Models    []SpeechModelConf `yaml:"models,omitempty" json:"models,omitempty"`
}

// SpeechModelConf is one voice model of text_to_speech.models.
type SpeechModelConf struct {
	Name      string            `yaml:"name,omitempty" json:"name,omitempty"` // default: the model's name
	Model     string            `yaml:"model" json:"model"`
	Port      int               `yaml:"port,omitempty" json:"port,omitempty"` // default: 9110 + its position
	GPUIDs    []int             `yaml:"gpu_ids,omitempty" json:"gpu_ids,omitempty"`
	ExtraArgs []string          `yaml:"extra_args,omitempty" json:"extra_args,omitempty"`
	Voices    []SpeechVoiceConf `yaml:"voices,omitempty" json:"voices,omitempty"`
	Timeout   duration          `yaml:"timeout,omitempty" json:"timeout,omitempty"`
}

// SpeechVoiceConf is a voice for a model that clones one: a short recording
// (WAV or MP3, 3 to 20 seconds) of the voice to speak with.
type SpeechVoiceConf struct {
	Name string `yaml:"name" json:"name"`
	File string `yaml:"file" json:"file"`
}

var voiceNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9 ._-]{0,39}$`)

// speechEntry is one voice model as it runs.
type speechEntry struct {
	Name   string
	Key    string
	Preset speechPreset
	Port   int
	GPUIDs []int
	Extra  []string
	Voices []speechVoice
	Wait   time.Duration
}

func (c SpeechConf) rawModels() []SpeechModelConf {
	if len(c.Models) > 0 {
		return c.Models
	}
	return []SpeechModelConf{{Model: c.Model, Port: c.Port, GPUIDs: c.GPUIDs, ExtraArgs: c.ExtraArgs, Voices: c.Voices}}
}

func (c SpeechConf) entries() []speechEntry {
	raw := c.rawModels()
	out := make([]speechEntry, len(raw))
	for i, m := range raw {
		key, p, _ := speechPresetByKey(m.Model)
		e := speechEntry{Name: m.Name, Key: key, Preset: p, Port: m.Port, GPUIDs: append([]int(nil), m.GPUIDs...),
			Extra: append([]string{}, m.ExtraArgs...), Wait: m.Timeout.Duration}
		if e.Name == "" {
			e.Name = key
		}
		if e.Port == 0 {
			e.Port = defaultSpeechPort + i
		}
		if len(e.GPUIDs) == 0 {
			e.GPUIDs = []int{0}
		}
		for _, w := range []time.Duration{c.Timeout.Duration, defaultSpeechWait} {
			if e.Wait <= 0 {
				e.Wait = w
			}
		}
		e.Voices = append([]speechVoice(nil), p.Voices...)
		if p.Clone {
			e.Voices = nil
			for _, v := range m.Voices {
				e.Voices = append(e.Voices, speechVoice{ID: v.Name, Label: v.Name, file: v.File})
			}
		}
		out[i] = e
	}
	return out
}

func (c SpeechConf) entryFor(name string) (speechEntry, bool) {
	for _, e := range c.entries() {
		if e.Name == name {
			return e, true
		}
	}
	return speechEntry{}, false
}

// instanceConf is the voice model's instance. A llama-server model gets its
// own small context and one slot; the global extra_args don't apply to it.
func (e speechEntry) instanceConf() InstanceConf {
	extra := append([]string{}, e.Extra...)
	ic := InstanceConf{Name: e.Name, Model: e.Preset.Ref, Port: e.Port, GPUIDs: append([]int(nil), e.GPUIDs...), role: roleSpeech}
	if e.Preset.Engine == engineLlama {
		ctx, jinja, f16 := speechCtx, true, "f16"
		extra = append([]string{"-np", "1"}, extra...)
		ic.AutoStart, ic.ContextLength, ic.Jinja, ic.CacheTypeK, ic.CacheTypeV = true, &ctx, &jinja, &f16, &f16
	}
	ic.ExtraArgs = &extra
	return ic
}

// voice picks a voice by id; "" is the first.
func (e speechEntry) voice(id string) (speechVoice, bool) {
	if len(e.Voices) == 0 {
		return speechVoice{}, false
	}
	if id == "" {
		return e.Voices[0], true
	}
	for _, v := range e.Voices {
		if v.ID == id {
			return v, true
		}
	}
	return speechVoice{}, false
}

var speechBackends = map[string]bool{"": true, "hip": true, "rocm": true, "vulkan": true, "cuda": true, "metal": true, "cpu": true}

func (c SpeechConf) check() error {
	var errs []error
	add := func(format string, a ...any) { errs = append(errs, fmt.Errorf(format, a...)) }
	if c.Timeout.Duration < 0 {
		add("text_to_speech.timeout must be >= 0")
	}
	if !speechBackends[c.Backend] {
		add("text_to_speech.backend %q must be hip, vulkan, cuda, metal or cpu", c.Backend)
	}
	if len(c.Models) > 0 && (c.Model != "" || c.Port != 0 || len(c.GPUIDs) > 0 || len(c.ExtraArgs) > 0 || len(c.Voices) > 0) {
		add("text_to_speech: list the models under models, or set model, port, gpu_ids, voices and extra_args at the top for a single one, not both")
	}
	names, ports := map[string]bool{}, map[int]bool{}
	raw := c.rawModels()
	for i, e := range c.entries() {
		at := "text_to_speech"
		if len(c.Models) > 0 {
			at = fmt.Sprintf("text_to_speech.models[%d] (%s)", i, e.Name)
		}
		if _, _, ok := speechPresetByKey(raw[i].Model); !ok {
			add("%s: model %q is unknown; use %s", at, raw[i].Model, strings.Join(speechPresetNames(), ", "))
			continue
		}
		if err := validateInstanceName(e.Name); err != nil {
			add("%s: %v", at, err)
		}
		if raw[i].Port != 0 && !validPort(raw[i].Port) {
			add("%s: port must be 1-65535", at)
		}
		if names[e.Name] {
			add("%s: the name %q is used twice; give one a name", at, e.Name)
		}
		if ports[e.Port] {
			add("%s: port %d is used twice", at, e.Port)
		}
		names[e.Name], ports[e.Port] = true, true
		if len(e.GPUIDs) != 1 {
			add("%s: gpu_ids lists one GPU", at)
		}
		for _, id := range e.GPUIDs {
			if id < 0 {
				add("%s: gpu id %d must be >= 0", at, id)
			}
		}
		if raw[i].Timeout.Duration < 0 {
			add("%s: timeout must be >= 0", at)
		}
		if c.Enabled && e.Preset.Engine == engineAudioCpp && strings.TrimSpace(c.ServerBin) == "" {
			add("text_to_speech.server_bin is required for %s: the path to audio.cpp's audiocpp_server", e.Key)
		}
		switch {
		case !e.Preset.Clone && len(raw[i].Voices) > 0:
			add("%s: %s has its own voices (%s); voices: is only for models that clone one", at, e.Key, voiceIDs(e.Preset.Voices))
		case c.Enabled && e.Preset.Clone && len(raw[i].Voices) == 0:
			add("%s: %s speaks with a voice you give it: list at least one under voices, a WAV or MP3 of 3 to 20 seconds of clear speech", at, e.Key)
		}
		seen := map[string]bool{}
		for _, v := range raw[i].Voices {
			switch {
			case !voiceNameRe.MatchString(v.Name):
				add("%s: voice name %q: use letters, digits, space, . _ - (max 40)", at, v.Name)
			case seen[v.Name]:
				add("%s: the voice %q is listed twice", at, v.Name)
			case !filepath.IsAbs(v.File):
				add("%s: voice %s: file must be an absolute path to a WAV or MP3", at, v.Name)
			}
			seen[v.Name] = true
		}
	}
	return errors.Join(errs...)
}

func voiceIDs(vs []speechVoice) string {
	ids := make([]string, len(vs))
	for i, v := range vs {
		ids[i] = v.ID
	}
	return strings.Join(ids, ", ")
}

// clash reports an instance that would collide with a voice model.
func (c SpeechConf) clash(ic InstanceConf) error {
	if !c.Enabled {
		return nil
	}
	for _, e := range c.entries() {
		if ic.Name == e.Name {
			return fmt.Errorf("the name %q belongs to text to speech (text_to_speech in config.yaml); pick another name", e.Name)
		}
		if ic.Port == e.Port {
			return fmt.Errorf("port %d is used by text to speech (text_to_speech in config.yaml); pick another port", ic.Port)
		}
	}
	return nil
}

var errSpeechFileMissing = errors.New("its model file isn't downloaded yet; llama-manager fetches it when it starts or when you start the model")

// speechConfDir holds the config files audiocpp_server is started with.
var speechConfDir = sync.OnceValues(func() (string, error) {
	return os.MkdirTemp("", "llama-manager-speech-")
})

// audioCppBackend is audio.cpp's name for the backend the model runs on.
func audioCppBackend(c SpeechConf, gpuBackend string) string {
	if c.Backend != "" {
		return c.Backend
	}
	switch gpuBackend {
	case "rocm", "rocm_rocr":
		return "hip"
	case "cuda", "metal":
		return gpuBackend
	default:
		return "vulkan"
	}
}

// buildSpeechLaunch is audiocpp_server's command line, with the config file
// it reads; ok is false for a llama-server model, which launches like any
// instance. audiocpp_server always listens on loopback and without its web
// UI: browsers reach the model only through the manager.
func buildSpeechLaunch(conf InstanceConf, cfg *Config) (spec launchSpec, ok bool, err error) {
	cfg.mu.RLock()
	c := cfg.TextToSpeech
	backend, gpuEnv := cfg.GPUBackend, cfg.GPUEnvVar()
	cfg.mu.RUnlock()
	e, found := c.entryFor(conf.Name)
	if !found || e.Preset.Label == "" {
		return launchSpec{}, true, fmt.Errorf("%s isn't a voice model in text_to_speech", conf.Name)
	}
	if e.Preset.Engine != engineAudioCpp {
		return launchSpec{}, false, nil
	}
	local, have := e.Preset.File.localPath()
	if !have {
		return launchSpec{}, true, errSpeechFileMissing
	}
	be := audioCppBackend(c, backend)
	server := map[string]any{
		"host": "127.0.0.1", "port": conf.Port, "backend": be, "device": 0,
		"lazy_load": false, "log_request_body": false, "max_request_body_bytes": 16 << 20,
		"models": []map[string]any{{
			"id": e.Name, "family": e.Preset.Family, "path": local, "task": "tts", "mode": "offline",
			"default_request_options": e.Preset.Options,
		}},
	}
	data, _ := json.MarshalIndent(server, "", "  ")
	dir, err := speechConfDir()
	if err != nil {
		return launchSpec{}, true, fmt.Errorf("making a folder for audiocpp_server's config: %w", err)
	}
	file := filepath.Join(dir, e.Name+".json")
	if err := os.WriteFile(file, data, 0o600); err != nil {
		return launchSpec{}, true, fmt.Errorf("writing audiocpp_server's config: %w", err)
	}
	spec = launchSpec{bin: c.ServerBin, host: "127.0.0.1", args: []string{"--config", file, "--no-ui", "--log"}}
	if gpuEnv != "" && be != "cpu" {
		spec.env = append(spec.env, gpuEnv+"="+strings.Join(intsToStrings(conf.GPUIDs), ","))
		if gpuEnv == "CUDA_VISIBLE_DEVICES" {
			spec.env = append(spec.env, "CUDA_DEVICE_ORDER=PCI_BUS_ID")
		}
	}
	if conf.ExtraArgs != nil {
		spec.args = append(spec.args, *conf.ExtraArgs...)
	}
	return spec, true, nil
}

// speechService runs the voice models: it fetches audio.cpp's model file,
// starts the models and turns text into WAV. It is nil when text_to_speech
// is off.
type speechService struct {
	mgr    *Manager
	models []speechEntry
	byName map[string]speechEntry
	http   *http.Client

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	mu       sync.Mutex
	dl       map[string]*imageDownload // models downloading now
	fetchErr map[string]string
	slots    map[string]chan struct{} // one request per model at a time

	cache *speechCache
}

func newSpeechService(c SpeechConf, mgr *Manager) *speechService {
	if !c.Enabled {
		return nil
	}
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.Proxy = nil // the voice models are our own child processes
	ctx, cancel := context.WithCancel(context.Background())
	s := &speechService{
		mgr: mgr, models: c.entries(), byName: map[string]speechEntry{}, ctx: ctx, cancel: cancel,
		dl: map[string]*imageDownload{}, fetchErr: map[string]string{}, slots: map[string]chan struct{}{},
		cache: newSpeechCache(speechCacheBytes),
		http: &http.Client{Transport: tr, CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		}},
	}
	for _, e := range s.models {
		s.byName[e.Name] = e
		s.slots[e.Name] = make(chan struct{}, 1)
	}
	return s
}

// Close stops a download in progress and waits for it to give up.
func (s *speechService) Close() {
	s.cancel()
	s.wg.Wait()
}

// pick finds a voice model by instance or preset name; "" is the first.
func (s *speechService) pick(name string) (speechEntry, bool) {
	if name == "" {
		return s.models[0], true
	}
	if e, ok := s.byName[name]; ok {
		return e, true
	}
	for _, e := range s.models {
		if e.Key == name {
			return e, true
		}
	}
	return speechEntry{}, false
}

func (s *speechService) names() []string {
	out := make([]string, len(s.models))
	for i, e := range s.models {
		out[i] = e.Name
	}
	return out
}

// speechModelStatus is what /api/status says about one voice model.
type speechModelStatus struct {
	Instance string         `json:"instance"`
	Preset   string         `json:"preset"`
	Model    string         `json:"model"`
	Ref      string         `json:"ref"`
	License  string         `json:"license"`
	Engine   speechEngine   `json:"engine"`
	Voices   []speechVoice  `json:"voices"`
	Langs    []string       `json:"langs"`
	State    InstanceState  `json:"state"`
	Error    string         `json:"error,omitempty"`
	Download *imageDownload `json:"download,omitempty"`
}

func (s *speechService) Status() []speechModelStatus {
	out := make([]speechModelStatus, 0, len(s.models))
	for _, e := range s.models {
		p := e.Preset
		st := speechModelStatus{Instance: e.Name, Preset: e.Key, Model: p.Label, Ref: p.Ref, License: p.License, Engine: p.Engine,
			Voices: append([]speechVoice{}, e.Voices...), Langs: p.Langs, State: StateStopped}
		if inst := s.mgr.Get(e.Name); inst != nil {
			is := inst.Status()
			st.State, st.Error = is.State, is.LastError
		}
		s.mu.Lock()
		if d := s.dl[e.Name]; d != nil {
			cp := *d
			st.State, st.Error, st.Download = stateDownloading, "", &cp
		} else if msg := s.fetchErr[e.Name]; msg != "" && (st.State == StateStopped || st.State == StateCrashed) {
			st.Error = msg
		}
		s.mu.Unlock()
		out = append(out, st)
	}
	return out
}

// Boot starts the audio.cpp models when llama-manager starts, downloading the
// model file first if it is missing; llama-server models start by
// themselves (auto_start) and download with -hf.
func (s *speechService) Boot() {
	for _, e := range s.models {
		if e.Preset.Engine != engineAudioCpp || s.fetch(e.Name) {
			continue
		}
		if err := s.mgr.StartInstance(e.Name); err != nil {
			log.Printf("[%s] failed to start: %v", e.Name, err)
		}
	}
}

// fetch downloads an audio.cpp model's file in the background, unless it is
// already on disk or on its way, and starts the model once it is in. It
// reports whether a download is pending.
func (s *speechService) fetch(name string) bool {
	e, ok := s.byName[name]
	if !ok || e.Preset.File == nil {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.dl[name] != nil {
		return true
	}
	if s.ctx.Err() != nil {
		return false
	}
	if _, have := e.Preset.File.localPath(); have {
		return false
	}
	f := *e.Preset.File
	s.dl[name] = &imageDownload{File: path.Base(f.Path), TotalMB: f.Size >> 20}
	delete(s.fetchErr, name)
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		inst := s.mgr.Get(name)
		logf := func(msg string) {
			if inst != nil {
				inst.addLog(msg)
			}
		}
		log.Printf("[%s] downloading %s, %d MB", name, path.Base(f.Path), f.Size>>20)
		err := fetchPinnedFile(s.ctx, f, progress{log: logf, every: 10 * time.Second, bytes: func(n int64) {
			s.mu.Lock()
			if d := s.dl[name]; d != nil {
				d.DoneMB = n >> 20
			}
			s.mu.Unlock()
		}})
		s.mu.Lock()
		delete(s.dl, name)
		if err != nil {
			msg := "downloading " + path.Base(f.Path) + ": " + err.Error()
			if s.ctx.Err() != nil {
				msg = "download stopped: llama-manager is shutting down"
			}
			s.fetchErr[name] = msg
			s.mu.Unlock()
			logf(msg)
			log.Printf("[%s] %s", name, msg)
			return
		}
		s.mu.Unlock()
		log.Printf("[%s] model file downloaded", name)
		if s.ctx.Err() == nil {
			if err := s.mgr.StartInstance(name); err != nil {
				log.Printf("[%s] failed to start: %v", name, err)
			}
		}
	}()
	return true
}

// speechError carries the HTTP status a speech failure answers with.
type speechError struct {
	Status int
	Msg    string
}

func (e *speechError) Error() string { return e.Msg }

func speechErrf(status int, format string, a ...any) error {
	return &speechError{status, fmt.Sprintf(format, a...)}
}

// ready returns the model's instance when it can take a request.
func (s *speechService) ready(e speechEntry) (*Instance, error) {
	inst := s.mgr.Get(e.Name)
	if inst == nil {
		return nil, speechErrf(http.StatusServiceUnavailable, "%s isn't running on this server", e.Name)
	}
	s.mu.Lock()
	downloading := s.dl[e.Name] != nil
	s.mu.Unlock()
	if downloading {
		return nil, speechErrf(http.StatusServiceUnavailable, "%s is downloading its model; it reads aloud once that's done", e.Name)
	}
	switch st := inst.Status(); st.State {
	case StateRunning, StateUnhealthy:
		return inst, nil
	case StateStarting, StateRestarting:
		return nil, speechErrf(http.StatusServiceUnavailable, "%s is still starting (downloading or loading its model). Try again in a moment", e.Name)
	case StateCrashed:
		msg := e.Name + " has crashed"
		if st.LastError != "" {
			msg += " (" + st.LastError + ")"
		}
		return nil, speechErrf(http.StatusServiceUnavailable, "%s. Check its log on the Instances page", msg)
	default:
		return nil, speechErrf(http.StatusServiceUnavailable, "%s is stopped. Start it on the Instances page", e.Name)
	}
}

func (e speechEntry) speaks(lang string) bool { return slices.Contains(e.Preset.Langs, lang) }

// pickLang is the language to speak text in: lang when given, else a guess.
func (e speechEntry) pickLang(lang, text string) (string, error) {
	lang = strings.ToLower(strings.TrimSpace(lang))
	guessed := lang == "" || lang == "auto"
	if guessed {
		lang = detectLang(text)
	}
	if e.speaks(lang) {
		return lang, nil
	}
	name := langNames[lang]
	if name == "" {
		name = lang
	}
	if guessed {
		return "", speechErrf(http.StatusUnprocessableEntity, "%s (%s) doesn't speak %s, which this text looks like. Pick a voice of another model or a language in Chat settings", e.Name, e.Preset.Label, name)
	}
	return "", speechErrf(http.StatusBadRequest, "%s (%s) doesn't speak %s; it speaks %s", e.Name, e.Preset.Label, name, strings.Join(e.Preset.Langs, ", "))
}

var langNames = map[string]string{
	"en": "English", "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "de": "German", "fr": "French",
	"es": "Spanish", "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "da": "Danish", "sv": "Swedish", "fi": "Finnish",
	"pl": "Polish", "tr": "Turkish", "el": "Greek", "ar": "Arabic", "he": "Hebrew",
}

// speak turns one part of text into WAV with the model's voice, from the
// cache when the same part was spoken before.
func (s *speechService) speak(ctx context.Context, e speechEntry, v speechVoice, lang, text string) ([]byte, error) {
	key := speechKey(e.Name, v.ID, lang, text)
	if wav, ok := s.cache.get(key); ok {
		return wav, nil
	}
	inst, err := s.ready(e)
	if err != nil {
		return nil, err
	}
	slot := s.slots[e.Name]
	select {
	case slot <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-slot }()
	if wav, ok := s.cache.get(key); ok {
		return wav, nil // spoken while this request waited
	}
	rctx, cancel := context.WithTimeout(ctx, e.Wait)
	defer cancel()
	var wav []byte
	if e.Preset.Engine == engineAudioCpp {
		wav, err = s.speakAudioCpp(rctx, inst, e, v, lang, text)
	} else {
		wav, err = s.speakLlama(rctx, inst, e, v, lang, text)
	}
	if err != nil {
		switch {
		case ctx.Err() != nil:
			return nil, ctx.Err()
		case errors.Is(rctx.Err(), context.DeadlineExceeded):
			return nil, speechErrf(http.StatusGatewayTimeout, "%s didn't finish within %s", e.Name, e.Wait)
		}
		return nil, err
	}
	s.cache.put(key, wav)
	return wav, nil
}

func speechKey(parts ...string) string {
	h := sha256.New()
	for _, p := range parts {
		h.Write([]byte(p))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// speakLlama asks llama-server's /tts for one part.
func (s *speechService) speakLlama(ctx context.Context, inst *Instance, e speechEntry, v speechVoice, lang, text string) ([]byte, error) {
	body := map[string]any{"input": text, "lang": lang, "seed": speechSeed, "n_predict": e.Preset.Frames, "response_format": "wav"}
	if v.file != "" {
		ref, err := readVoiceFile(v)
		if err != nil {
			return nil, err
		}
		body["speaker_ref_b64"] = base64.StdEncoding.EncodeToString(ref)
	} else if v.ID != "" {
		body["voice"] = v.ID
	}
	resp, err := s.post(ctx, inst, "/tts", body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusNotFound:
		return nil, speechErrf(http.StatusBadGateway, "%s's llama-server has no /tts: text to speech needs llama.cpp v0.8.0 (b11960) or newer. Update llama.cpp and restart %s", e.Name, e.Name)
	case resp.StatusCode != http.StatusOK:
		return nil, speechErrf(http.StatusBadGateway, "%s couldn't speak: %s", e.Name, upstreamError(resp))
	}
	return readWAV(resp, e.Name)
}

// speakAudioCpp asks audiocpp_server's OpenAI-style speech endpoint.
func (s *speechService) speakAudioCpp(ctx context.Context, inst *Instance, e speechEntry, v speechVoice, lang, text string) ([]byte, error) {
	opts := map[string]any{"language": lang}
	if v.ID != "" {
		opts["voice_id"] = v.ID
	}
	body := map[string]any{"model": e.Name, "input": text, "response_format": "wav", "options": opts}
	resp, err := s.post(ctx, inst, "/v1/audio/speech", body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, speechErrf(http.StatusBadGateway, "%s couldn't speak: %s", e.Name, sdError(resp))
	}
	return readWAV(resp, e.Name)
}

func (s *speechService) post(ctx context.Context, inst *Instance, p string, body map[string]any) (*http.Response, error) {
	data, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, inst.baseURL()+p, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.http.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, err
		}
		return nil, speechErrf(http.StatusBadGateway, "couldn't reach %s: %v", inst.conf.Name, err)
	}
	return resp, nil
}

func readWAV(resp *http.Response, name string) ([]byte, error) {
	b, err := io.ReadAll(io.LimitReader(resp.Body, maxSpeechWAV+1))
	if err != nil {
		return nil, err
	}
	if len(b) > maxSpeechWAV || !isWAV(b) {
		return nil, speechErrf(http.StatusBadGateway, "%s sent audio llama-manager can't read", name)
	}
	return b, nil
}

// readVoiceFile reads a reference clip named in config.yaml.
func readVoiceFile(v speechVoice) ([]byte, error) {
	f, err := os.Open(v.file)
	if err != nil {
		return nil, speechErrf(http.StatusInternalServerError, "reading the voice %s: %v", v.ID, err)
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, maxVoiceFileBytes+1))
	switch {
	case err != nil:
		return nil, speechErrf(http.StatusInternalServerError, "reading the voice %s: %v", v.ID, err)
	case len(b) > maxVoiceFileBytes:
		return nil, speechErrf(http.StatusInternalServerError, "the voice %s is over %d MB; use 3 to 20 seconds of speech", v.ID, maxVoiceFileBytes>>20)
	case len(b) == 0:
		return nil, speechErrf(http.StatusInternalServerError, "the voice %s (%s) is empty", v.ID, v.file)
	}
	return b, nil
}

// replySpeech is a reply prepared for reading aloud.
type replySpeech struct {
	entry speechEntry
	voice speechVoice
	lang  string
	parts []string
}

func (s *speechService) prepare(model, voice, lang, text string) (replySpeech, error) {
	e, ok := s.pick(model)
	if !ok {
		return replySpeech{}, speechErrf(http.StatusNotFound, "no voice model %q (voice models: %s)", model, strings.Join(s.names(), ", "))
	}
	v, ok := e.voice(voice)
	if !ok {
		return replySpeech{}, speechErrf(http.StatusBadRequest, "%s has no voice %q; its voices: %s", e.Name, voice, voiceIDs(e.Voices))
	}
	if utf8.RuneCountInString(text) > maxSpeechText {
		text = string([]rune(text)[:maxSpeechText])
	}
	parts := speechParts(text, speechFirstPart, speechPartRunes)
	if len(parts) == 0 {
		return replySpeech{}, speechErrf(http.StatusUnprocessableEntity, "There's nothing to read aloud here: no text outside code blocks and links")
	}
	l, err := e.pickLang(lang, text)
	if err != nil {
		return replySpeech{}, err
	}
	return replySpeech{entry: e, voice: v, lang: l, parts: parts}, nil
}

func writeSpeechErr(w http.ResponseWriter, err error) {
	var se *speechError
	if errors.As(err, &se) {
		http.Error(w, se.Msg, se.Status)
		return
	}
	http.Error(w, err.Error(), http.StatusBadGateway)
}

func writeWAV(w http.ResponseWriter, wav []byte) {
	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("Content-Length", strconv.Itoa(len(wav)))
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	_, _ = w.Write(wav)
}

// handleChatSpeech reads one part of a reply aloud: POST
// /api/chats/{id}/messages/{mid}/speech with {variant_id, part, model, voice,
// lang}. The answer is WAV; X-Speech-Parts says how many parts there are.
func (ws *WebServer) handleChatSpeech(w http.ResponseWriter, r *http.Request, chatID, msgID string) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if ws.speech == nil {
		http.Error(w, "text to speech is off; turn it on in the text_to_speech section of config.yaml", http.StatusNotFound)
		return
	}
	var req struct {
		VariantID string `json:"variant_id"`
		Part      int    `json:"part"`
		Model     string `json:"model"`
		Voice     string `json:"voice"`
		Lang      string `json:"lang"`
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}
	v, err := ws.chats.store.Reply(chatID, msgID, req.VariantID)
	if err != nil {
		ws.chatErr(w, err)
		return
	}
	if v.Status == variantStreaming {
		http.Error(w, "the reply is still being written; play it when it's done", http.StatusConflict)
		return
	}
	rs, err := ws.speech.prepare(req.Model, req.Voice, req.Lang, speechText(v.Content))
	if err != nil {
		writeSpeechErr(w, err)
		return
	}
	if req.Part < 0 || req.Part >= len(rs.parts) {
		http.Error(w, fmt.Sprintf("part %d doesn't exist: the reply has %d", req.Part, len(rs.parts)), http.StatusBadRequest)
		return
	}
	wav, err := ws.speech.speak(r.Context(), rs.entry, rs.voice, rs.lang, rs.parts[req.Part])
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		writeSpeechErr(w, err)
		return
	}
	w.Header().Set("X-Speech-Parts", strconv.Itoa(len(rs.parts)))
	w.Header().Set("X-Speech-Lang", rs.lang)
	writeWAV(w, wav)
}

// openAIVoices are the voice names OpenAI clients send by default; they get
// the model's first voice.
var openAIVoices = map[string]bool{"alloy": true, "ash": true, "ballad": true, "coral": true, "echo": true, "fable": true,
	"onyx": true, "nova": true, "sage": true, "shimmer": true, "verse": true, "marin": true, "cedar": true}

// handleV1Speech is the OpenAI-compatible speech endpoint for other apps:
// {model, input, voice}, answered with one WAV. "model" names a voice model's
// instance or preset; OpenAI's names (tts-1...) or none pick the first.
// "language" (not OpenAI's) sets the language; without it it is guessed.
func (ws *WebServer) handleV1Speech(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if ws.speech == nil {
		http.Error(w, "text to speech is off; turn it on in the text_to_speech section of config.yaml", http.StatusNotFound)
		return
	}
	var req struct {
		Model    string `json:"model"`
		Input    string `json:"input"`
		Voice    string `json:"voice"`
		Language string `json:"language"`
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Input) == "" {
		http.Error(w, `"input" is required: the text to speak`, http.StatusBadRequest)
		return
	}
	if utf8.RuneCountInString(req.Input) > maxSpeechInput {
		http.Error(w, fmt.Sprintf(`"input" is over %d characters; split the text`, maxSpeechInput), http.StatusBadRequest)
		return
	}
	model := req.Model
	if _, ok := ws.speech.pick(model); !ok && (strings.HasPrefix(model, "tts-") || strings.HasSuffix(model, "-tts")) {
		model = ""
	}
	voice := req.Voice
	if e, ok := ws.speech.pick(model); ok {
		if _, has := e.voice(voice); !has && openAIVoices[strings.ToLower(voice)] {
			voice = ""
		}
	}
	rs, err := ws.speech.prepare(model, voice, req.Language, speechText(req.Input))
	if err != nil {
		writeSpeechErr(w, err)
		return
	}
	wavs := make([][]byte, 0, len(rs.parts))
	for _, p := range rs.parts {
		wav, err := ws.speech.speak(r.Context(), rs.entry, rs.voice, rs.lang, p)
		if err != nil {
			if r.Context().Err() != nil {
				return
			}
			writeSpeechErr(w, err)
			return
		}
		wavs = append(wavs, wav)
	}
	wav, err := joinWAVs(wavs)
	if err != nil {
		http.Error(w, rs.entry.Name+" sent audio llama-manager can't join: "+err.Error(), http.StatusBadGateway)
		return
	}
	writeWAV(w, wav)
}

// speechCache keeps recently spoken parts, so playing a reply again starts at
// once. It holds at most max bytes, dropping the least recently used.
type speechCache struct {
	mu   sync.Mutex
	max  int
	size int
	ll   *list.List
	m    map[string]*list.Element
}

type speechCached struct {
	key string
	wav []byte
}

func newSpeechCache(max int) *speechCache {
	return &speechCache{max: max, ll: list.New(), m: map[string]*list.Element{}}
}

func (c *speechCache) get(key string) ([]byte, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.m[key]
	if !ok {
		return nil, false
	}
	c.ll.MoveToFront(el)
	return el.Value.(*speechCached).wav, true
}

func (c *speechCache) put(key string, wav []byte) {
	if len(wav) > c.max {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.m[key]; ok {
		c.ll.MoveToFront(el)
		return
	}
	c.m[key] = c.ll.PushFront(&speechCached{key, wav})
	c.size += len(wav)
	for c.size > c.max {
		el := c.ll.Back()
		it := el.Value.(*speechCached)
		c.ll.Remove(el)
		delete(c.m, it.key)
		c.size -= len(it.wav)
	}
}
