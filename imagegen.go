package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"math"
	"math/rand/v2"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Image generation (image_generation in config.yaml) runs stable-diffusion.cpp's
// sd-server, one process per image model. Picked in the chat, a model turns
// each message into a picture saved with that version of the reply; images
// attached to the message go along as references, which edits them.

const (
	imageGenName        = "image-gen" // the instance of the single-model form
	roleImageGen        = "image_gen"
	defaultImageGenPort = 9096
	defaultImageModel   = "flux2-klein-4b"
	defaultImageWait    = 10 * time.Minute
	defaultImageSide    = 1024
	minImageSide        = 256
	maxImageSide        = 2048
	maxGenImageBytes    = 32 << 20
	imageJobFailLimit   = 10
)

// stateDownloading is what /api/status reports for an image model while its
// files are being fetched; the instance itself is stopped meanwhile.
const stateDownloading InstanceState = "downloading"

// imagePollEvery is how often a generation's job is polled. Var for tests.
var imagePollEvery = 400 * time.Millisecond

// imageFile is one file a preset loads, pinned to a commit so the size and
// hash checked after download can't drift.
type imageFile struct {
	Flag   string // the sd-server option that loads it; unused for a LoRA
	Repo   string
	Rev    string
	Path   string // inside the repo
	Size   int64
	SHA256 string
}

// editMode is what a model does with images attached to the message.
type editMode string

const (
	editNone editMode = "none" // makes images from a description only
	editYes  editMode = "yes"  // makes images, or edits the attached ones
	editOnly editMode = "only" // edits: needs an attached image
)

type imagePreset struct {
	Label   string
	Ref     string // shown as the instance's model
	License string
	Files   []imageFile
	LoRA    *imageFile // applied to every image
	Args    []string   // load options the model needs
	Steps   int
	CFG     float64
	Sampler string
	// FlowShift and Sigmas override sd-server's schedule when set.
	FlowShift float64
	Sigmas    []float64
	Align     int // width and height must be multiples of it
	Edit      editMode
	MaxRefs   int
	GPUs      int // 2: the text encoder runs on the second GPU
	Wait      time.Duration
}

// Files several presets share are downloaded once.
var (
	fileFlux2VAE = imageFile{Flag: "--vae", Repo: "Comfy-Org/Flux-2-Klein-4B", Rev: "5b3abca5c0dc1ecd4eaf3340868d09063fc1c965",
		Path: "split_files/vae/flux2-vae.safetensors", Size: 336211292, SHA256: "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3"}
	// The same Qwen3-4B text encoder serves FLUX.2 klein 4B and Z-Image (their
	// published safetensors are byte-identical).
	fileQwen3_4B = imageFile{Flag: "--llm", Repo: "unsloth/Qwen3-4B-GGUF", Rev: "22c9fc8a8c7700b76a1789366280a6a5a1ad1120",
		Path: "Qwen3-4B-Q8_0.gguf", Size: 4280405792, SHA256: "eed555233267a33c7e8ee31682762cc7751b3f6d224039086e0e846f05fffa5d"}
	fileQwenImageVAE = imageFile{Flag: "--vae", Repo: "Comfy-Org/Qwen-Image", Rev: "1f12b17be14c89b026c51a91d67c32f84bb047bc",
		Path: "split_files/vae/qwen_image_vae.safetensors", Size: 253806246, SHA256: "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"}
	fileQwen25VL = imageFile{Flag: "--llm", Repo: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", Rev: "cfa2baa09946b211c107e6e104948987a64dd2c1",
		Path: "Qwen2.5-VL-7B-Instruct.Q8_0.gguf", Size: 8098524160, SHA256: "577bfb3e41f93f00414e594c56a38bbe91207b3b636ab7faf343e34a20aa73ca"}
	fileQwen25VLVision = imageFile{Flag: "--llm_vision", Repo: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", Rev: "cfa2baa09946b211c107e6e104948987a64dd2c1",
		Path: "Qwen2.5-VL-7B-Instruct.mmproj-f16.gguf", Size: 1354162912, SHA256: "2f9b49529bf463c165223e21f10320655a74da61bb64bf7b9fa8b3892cc46926"}
	fileEditLightning = imageFile{Repo: "lightx2v/Qwen-Image-Edit-2511-Lightning", Rev: "107d238d68b8387e32843d0b4a2e24ceb0d3ad86",
		Path: "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", Size: 849608296, SHA256: "22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f"}
)

// qwenImageEdit is Qwen-Image-Edit-2511 with the 4-step Lightning LoRA: the
// diffusion model at Q4_K_M fits one 32 GB card with its text encoder; Q8_0
// needs the text encoder on a second card.
func qwenImageEdit(quant string, f imageFile, gpus int) imagePreset {
	return imagePreset{
		Label: "Qwen-Image-Edit 2511 (Lightning)", Ref: "unsloth/Qwen-Image-Edit-2511-GGUF:" + quant, License: "Apache-2.0",
		Files: []imageFile{f, fileQwenImageVAE, fileQwen25VL, fileQwen25VLVision}, LoRA: &fileEditLightning,
		// Without zero_cond_t, 2511's edits degrade badly (sd.cpp docs).
		Args:  []string{"--model-args", "qwen_image_zero_cond_t=true"},
		Steps: 4, CFG: 1, Sampler: "euler", FlowShift: 3, Align: 16, Edit: editOnly, MaxRefs: 3, GPUs: gpus, Wait: 20 * time.Minute,
	}
}

// imagePresets are the image models llama-manager can run. Each file is
// Q8_0 or F16 where it matters: gfx906 (MI50) has no BF16 or FP8 math.
var imagePresets = map[string]imagePreset{
	"flux2-klein-4b": {
		Label: "FLUX.2 klein 4B", Ref: "leejet/FLUX.2-klein-4B-GGUF:Q8_0", License: "Apache-2.0",
		Files: []imageFile{
			{Flag: "--diffusion-model", Repo: "leejet/FLUX.2-klein-4B-GGUF", Rev: "3b1f5a9dc3abb32238b053aeb3d823c30afdacbd",
				Path: "flux-2-klein-4b-Q8_0.gguf", Size: 4300629440, SHA256: "0bba6951258ec8f92d51114a8fa13e66828297bfff58a738f52729b3ef66fa28"},
			fileFlux2VAE, fileQwen3_4B,
		},
		Steps: 4, CFG: 1, Sampler: "euler", Align: 16, Edit: editYes, MaxRefs: 4,
	},
	"flux3-klein": {
		Label: "FLUX.3 Klein", Ref: "leejet/flux-3-image-klein-GGUF:Q8_0", License: "FLUX Kommunity: commercial output only under US$5M revenue",
		Files: []imageFile{
			{Flag: "--diffusion-model", Repo: "leejet/flux-3-image-klein-GGUF", Rev: "1c2ad2f9b9247ccca6506a23bdb033fdb0873321",
				Path: "flux3-image-klein-Q8_0.gguf", Size: 6151716032, SHA256: "8b71ffd749bcacbd50cc0acca4f2d31a044c94412743e1e780db99bf13f89752"},
			fileFlux2VAE,
			{Flag: "--llm", Repo: "Qwen/Qwen3-VL-4B-Instruct-GGUF", Rev: "1cd86afb9a95c410a6038ab3b40d8b578c892266",
				Path: "Qwen3VL-4B-Instruct-Q8_0.gguf", Size: 4280406144, SHA256: "054721f478bc5fa6beffb7f38eae575d45298f88cbb8d2f83ef675a727863eb1"},
		},
		Steps: 16, CFG: 1, Sampler: "euler", Align: 16, Edit: editYes, MaxRefs: 10,
	},
	"z-image-turbo": {
		Label: "Z-Image Turbo", Ref: "leejet/Z-Image-Turbo-GGUF:Q8_0", License: "Apache-2.0",
		Files: []imageFile{
			{Flag: "--diffusion-model", Repo: "leejet/Z-Image-Turbo-GGUF", Rev: "c61c0e422dc8b541b7548cf33a4ef8302b0f8085",
				Path: "z_image_turbo-Q8_0.gguf", Size: 6577440704, SHA256: "df1c5baa86d1398c979495a6072dbcee79444fdb884a2445582ba0769c44e9a1"},
			{Flag: "--vae", Repo: "Comfy-Org/Z-Image-Turbo", Rev: "6fc90a3b1b653e935a0d175e260736de25b84df5",
				Path: "split_files/vae/ae.safetensors", Size: 335304388, SHA256: "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"},
			fileQwen3_4B,
		},
		Steps: 8, CFG: 1, Sampler: "euler", Align: 16, Edit: editNone,
	},
	"qwen-image-2.1-turbo": {
		Label: "Qwen-Image 2.1 Turbo", Ref: "unsloth/Qwen-Image-2.1-Turbo-GGUF:Q8_0", License: "Qwen Research License: non-commercial",
		Files: []imageFile{
			{Flag: "--diffusion-model", Repo: "unsloth/Qwen-Image-2.1-Turbo-GGUF", Rev: "5d1f9f759ea877ae556cda88dbec958b62ad3809",
				Path: "qwen-image-2.1-turbo-Q8_0.gguf", Size: 7640860480, SHA256: "399d2ddc3636f36456e2f5bfa57187da4e38721bc046430317f42a9eab5b1712"},
			{Flag: "--vae", Repo: "Comfy-Org/Qwen-Image-2.1", Rev: "aacf31a6d48e707f352ecba714cee0680da3479b",
				Path: "vae/qwen_image_2.1_vae_bf16.safetensors", Size: 675509688, SHA256: "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"},
			{Flag: "--llm", Repo: "Qwen/Qwen3-VL-8B-Instruct-GGUF", Rev: "f982a07559d4a2f6c8744d840bf6fccab30eea96",
				Path: "Qwen3VL-8B-Instruct-Q8_0.gguf", Size: 8709519456, SHA256: "0d264b3941185d00a74f75c4245521dae088ff1efc90ab8d1754e83f5844adb0"},
			// The vision weights let it edit attached images.
			{Flag: "--llm_vision", Repo: "Qwen/Qwen3-VL-8B-Instruct-GGUF", Rev: "f982a07559d4a2f6c8744d840bf6fccab30eea96",
				Path: "mmproj-Qwen3VL-8B-Instruct-F16.gguf", Size: 1159029824, SHA256: "ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38"},
		},
		// Turbo is distilled to this fixed 8-step schedule (its model card).
		Steps: 8, CFG: 1, Sampler: "euler", Sigmas: []float64{1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0},
		Align: 32, Edit: editYes, MaxRefs: 4,
	},
	"qwen-image-edit-2511": qwenImageEdit("Q4_K_M", imageFile{Flag: "--diffusion-model", Repo: "unsloth/Qwen-Image-Edit-2511-GGUF", Rev: "52838251db242ddd8e8d0118de36c71006d7c397",
		Path: "qwen-image-edit-2511-Q4_K_M.gguf", Size: 12842830432, SHA256: "07951f61941a81e236cbd4a9b6f250d49626b3ba518f260135bf2e01a87cf1c8"}, 1),
	"qwen-image-edit-2511-q8": qwenImageEdit("Q8_0", imageFile{Flag: "--diffusion-model", Repo: "unsloth/Qwen-Image-Edit-2511-GGUF", Rev: "52838251db242ddd8e8d0118de36c71006d7c397",
		Path: "qwen-image-edit-2511-Q8_0.gguf", Size: 22354590304, SHA256: "0bf71797bf27dac40e3d8c1dc344309cb53f2a08beed6a96120dd77164b3806f"}, 2),
}

func imagePresetNames() []string {
	names := make([]string, 0, len(imagePresets))
	for n := range imagePresets {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

func presetByKey(m string) (string, imagePreset, bool) {
	k := strings.ToLower(strings.TrimSpace(m))
	if k == "" {
		k = defaultImageModel
	}
	p, ok := imagePresets[k]
	return k, p, ok
}

func (p imagePreset) files() []imageFile {
	out := append([]imageFile(nil), p.Files...)
	if p.LoRA != nil {
		out = append(out, *p.LoRA)
	}
	return out
}

func (p imagePreset) missing() []imageFile {
	var out []imageFile
	for _, f := range p.files() {
		if _, ok := f.localPath(); !ok {
			out = append(out, f)
		}
	}
	return out
}

// loraName is how sd-server names the LoRA in a request: its file name
// without the extension, inside --lora-model-dir.
func (p imagePreset) loraName() string {
	if p.LoRA == nil {
		return ""
	}
	base := path.Base(p.LoRA.Path)
	return strings.TrimSuffix(base, path.Ext(base))
}

// ImageGenConf runs sd-server for one or more image models. It picks a
// binary, its arguments and ports, so like image_recognition it is set only
// in config.yaml. Either model/port/gpu_ids/extra_args name one model, run
// as the instance image-gen, or Models lists several.
type ImageGenConf struct {
	Enabled   bool             `yaml:"enabled" json:"enabled"`
	ServerBin string           `yaml:"server_bin,omitempty" json:"server_bin,omitempty"`
	Model     string           `yaml:"model,omitempty" json:"model,omitempty"`
	Port      int              `yaml:"port,omitempty" json:"port,omitempty"`
	GPUIDs    []int            `yaml:"gpu_ids,omitempty" json:"gpu_ids,omitempty"`
	ExtraArgs []string         `yaml:"extra_args,omitempty" json:"extra_args,omitempty"`
	Timeout   duration         `yaml:"timeout,omitempty" json:"timeout,omitempty"`
	Models    []ImageModelConf `yaml:"models,omitempty" json:"models,omitempty"`
}

// ImageModelConf is one image model of image_generation.models.
type ImageModelConf struct {
	Name      string   `yaml:"name,omitempty" json:"name,omitempty"` // default: the model's name
	Model     string   `yaml:"model" json:"model"`
	Port      int      `yaml:"port,omitempty" json:"port,omitempty"` // default: 9096 + its position
	GPUIDs    []int    `yaml:"gpu_ids,omitempty" json:"gpu_ids,omitempty"`
	ExtraArgs []string `yaml:"extra_args,omitempty" json:"extra_args,omitempty"`
	Timeout   duration `yaml:"timeout,omitempty" json:"timeout,omitempty"`
}

// imageEntry is one image model as it runs: an instance with its preset.
type imageEntry struct {
	Name   string
	Key    string
	Preset imagePreset
	Port   int
	GPUIDs []int
	Extra  []string
	Wait   time.Duration
}

// rawModels is the section as a list, the single-model form included.
func (c ImageGenConf) rawModels() []ImageModelConf {
	if len(c.Models) > 0 {
		return c.Models
	}
	return []ImageModelConf{{Name: imageGenName, Model: c.Model, Port: c.Port, GPUIDs: c.GPUIDs, ExtraArgs: c.ExtraArgs}}
}

// entries resolves every image model with its defaults filled in.
func (c ImageGenConf) entries() []imageEntry {
	raw := c.rawModels()
	out := make([]imageEntry, len(raw))
	for i, m := range raw {
		key, p, _ := presetByKey(m.Model)
		e := imageEntry{Name: m.Name, Key: key, Preset: p, Port: m.Port,
			GPUIDs: append([]int(nil), m.GPUIDs...), Extra: append([]string{}, m.ExtraArgs...), Wait: m.Timeout.Duration}
		if e.Name == "" {
			e.Name = key
		}
		if e.Port == 0 {
			e.Port = defaultImageGenPort + i
		}
		if len(e.GPUIDs) == 0 {
			e.GPUIDs = []int{0}
		}
		for _, w := range []time.Duration{c.Timeout.Duration, p.Wait, defaultImageWait} {
			if e.Wait <= 0 {
				e.Wait = w
			}
		}
		out[i] = e
	}
	return out
}

func (c ImageGenConf) entryFor(name string) (imageEntry, bool) {
	for _, e := range c.entries() {
		if e.Name == name {
			return e, true
		}
	}
	return imageEntry{}, false
}

func (e imageEntry) instanceConf() InstanceConf {
	extra := append([]string{}, e.Extra...)
	return InstanceConf{Name: e.Name, Model: e.Preset.Ref, Port: e.Port, GPUIDs: append([]int(nil), e.GPUIDs...), ExtraArgs: &extra, role: roleImageGen}
}

func (c ImageGenConf) check() error {
	var errs []error
	add := func(format string, a ...any) { errs = append(errs, fmt.Errorf(format, a...)) }
	if c.Enabled && strings.TrimSpace(c.ServerBin) == "" {
		add("image_generation.server_bin is required: the path to stable-diffusion.cpp's sd-server")
	}
	if c.Timeout.Duration < 0 {
		add("image_generation.timeout must be >= 0")
	}
	if len(c.Models) > 0 && (c.Model != "" || c.Port != 0 || len(c.GPUIDs) > 0 || len(c.ExtraArgs) > 0) {
		add("image_generation: list the models under models, or set model, port, gpu_ids and extra_args at the top for a single one, not both")
	}
	names, ports := map[string]bool{}, map[int]bool{}
	raw := c.rawModels()
	for i, e := range c.entries() {
		at := "image_generation"
		if len(c.Models) > 0 {
			at = fmt.Sprintf("image_generation.models[%d] (%s)", i, e.Name)
		}
		if _, _, ok := presetByKey(raw[i].Model); !ok {
			add("%s: model %q is unknown; use %s", at, raw[i].Model, strings.Join(imagePresetNames(), ", "))
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
		switch need := max(e.Preset.GPUs, 1); {
		case len(e.GPUIDs) > 2:
			add("%s: gpu_ids lists at most two GPUs (the second runs the text encoder)", at)
		case len(e.GPUIDs) < need:
			add("%s: %s needs %d GPUs (the second runs the text encoder); list them in gpu_ids", at, e.Key, need)
		case len(e.GPUIDs) == 2 && e.GPUIDs[0] == e.GPUIDs[1]:
			add("%s: gpu_ids lists GPU %d twice", at, e.GPUIDs[0])
		}
		for _, id := range e.GPUIDs {
			if id < 0 {
				add("%s: gpu id %d must be >= 0", at, id)
			}
		}
		if raw[i].Timeout.Duration < 0 {
			add("%s: timeout must be >= 0", at)
		}
	}
	return errors.Join(errs...)
}

// clash reports an instance that would collide with an image model.
func (c ImageGenConf) clash(ic InstanceConf) error {
	if !c.Enabled {
		return nil
	}
	for _, e := range c.entries() {
		if ic.Name == e.Name {
			return fmt.Errorf("the name %q belongs to image generation (image_generation in config.yaml); pick another name", e.Name)
		}
		if ic.Port == e.Port {
			return fmt.Errorf("port %d is used by image generation (image_generation in config.yaml); pick another port", ic.Port)
		}
	}
	return nil
}

func (f imageFile) snapshotDir(rev string) string {
	return filepath.Join(huggingfaceHubDir(), "models--"+strings.ReplaceAll(f.Repo, "/", "--"), "snapshots", rev)
}

// localPath finds the file in the HF hub cache: at its pinned commit, where
// image models download it, or under main when the Models page fetched the
// same file, as long as the size matches.
func (f imageFile) localPath() (string, bool) {
	for _, rev := range []string{f.Rev, "main"} {
		p := filepath.Join(f.snapshotDir(rev), filepath.FromSlash(f.Path))
		if fi, err := os.Stat(p); err == nil && fi.Mode().IsRegular() && fi.Size() == f.Size {
			return p, true
		}
	}
	return "", false
}

var errImageFilesMissing = errors.New("its model files aren't downloaded yet; llama-manager fetches them when it starts or when you start the image model")

// sdDevice is sd.cpp's name for the n-th GPU the backend sees.
func sdDevice(backend string, n int) string {
	prefix := map[string]string{"rocm": "rocm", "rocm_rocr": "rocm", "cuda": "cuda", "vulkan": "vulkan"}[backend]
	if prefix == "" {
		prefix = "vulkan"
	}
	return prefix + strconv.Itoa(n)
}

func formatFloats(fs []float64) string {
	parts := make([]string, len(fs))
	for i, f := range fs {
		parts[i] = strconv.FormatFloat(f, 'f', -1, 64)
	}
	return strings.Join(parts, ",")
}

// buildImageGenLaunch is sd-server's command line. It always listens on
// loopback: sd-server has no auth and answers any Origin with credentials, so
// browsers and other machines reach it only through the manager.
func buildImageGenLaunch(conf InstanceConf, cfg *Config) (launchSpec, error) {
	cfg.mu.RLock()
	c := cfg.ImageGeneration
	backend, gpuEnv := cfg.GPUBackend, cfg.GPUEnvVar()
	cfg.mu.RUnlock()
	e, ok := c.entryFor(conf.Name)
	if !ok || e.Preset.Label == "" {
		return launchSpec{}, fmt.Errorf("%s isn't an image model in image_generation", conf.Name)
	}
	p := e.Preset
	spec := launchSpec{bin: c.ServerBin, host: "127.0.0.1"}
	args := []string{"--listen-ip", spec.host, "--listen-port", strconv.Itoa(conf.Port)}
	for _, f := range p.Files {
		local, ok := f.localPath()
		if !ok {
			return launchSpec{}, errImageFilesMissing
		}
		args = append(args, f.Flag, local)
	}
	if p.LoRA != nil {
		local, ok := p.LoRA.localPath()
		if !ok {
			return launchSpec{}, errImageFilesMissing
		}
		args = append(args, "--lora-model-dir", filepath.Dir(local))
	}
	args = append(args, p.Args...)
	// Defaults for /v1/images/generations callers; the chat sends its own.
	args = append(args,
		"--steps", strconv.Itoa(p.Steps),
		"--cfg-scale", strconv.FormatFloat(p.CFG, 'f', -1, 64),
		"--sampling-method", p.Sampler,
		"-W", strconv.Itoa(defaultImageSide), "-H", strconv.Itoa(defaultImageSide),
		"--seed", "-1",
	)
	if p.FlowShift > 0 {
		args = append(args, "--flow-shift", strconv.FormatFloat(p.FlowShift, 'f', -1, 64))
	}
	if len(p.Sigmas) > 0 {
		args = append(args, "--sigmas", formatFloats(p.Sigmas))
	}
	// Load every weight before listening, so "running" means ready and the
	// first image isn't slowed by reading the model from disk.
	args = append(args, "--eager-load")
	if len(conf.GPUIDs) == 2 {
		// The visibility variable renumbers the two GPUs 0 and 1.
		args = append(args, "--backend", fmt.Sprintf("all=%s,te=%s", sdDevice(backend, 0), sdDevice(backend, 1)))
	}
	if gpuEnv != "" && len(conf.GPUIDs) > 0 {
		spec.env = append(spec.env, gpuEnv+"="+strings.Join(intsToStrings(conf.GPUIDs), ","))
		if gpuEnv == "CUDA_VISIBLE_DEVICES" {
			spec.env = append(spec.env, "CUDA_DEVICE_ORDER=PCI_BUS_ID")
		}
	}
	if conf.ExtraArgs != nil {
		args = append(args, *conf.ExtraArgs...)
	}
	spec.args = args
	return spec, nil
}

// imageGen fetches the image models' files, starts them, and runs
// generations for the chat. It is nil when image_generation is off.
type imageGen struct {
	mgr    *Manager
	models []imageEntry
	byName map[string]imageEntry
	http   *http.Client

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// One download runs at a time, model after model, so a file two models
	// share is fetched once.
	mu       sync.Mutex
	running  bool
	queue    []string
	waiting  map[string]bool
	fetchErr map[string]string
	current  string
	file     string
	done     int64 // bytes of the current model's download so far
	total    int64
}

func newImageGen(c ImageGenConf, mgr *Manager) *imageGen {
	if !c.Enabled {
		return nil
	}
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.Proxy = nil // sd-server is our own child process
	ctx, cancel := context.WithCancel(context.Background())
	g := &imageGen{
		mgr: mgr, models: c.entries(), byName: map[string]imageEntry{}, ctx: ctx, cancel: cancel,
		waiting: map[string]bool{}, fetchErr: map[string]string{},
		http: &http.Client{Transport: tr, CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		}},
	}
	for _, e := range g.models {
		g.byName[e.Name] = e
	}
	return g
}

// Close stops a download in progress and waits for it to give up.
func (g *imageGen) Close() {
	g.cancel()
	g.wg.Wait()
}

func (g *imageGen) model(name string) (imageEntry, bool) {
	e, ok := g.byName[name]
	return e, ok
}

// pick finds an image model by instance name or preset name; "" is the first.
func (g *imageGen) pick(name string) (imageEntry, bool) {
	if name == "" {
		return g.models[0], true
	}
	if e, ok := g.byName[name]; ok {
		return e, true
	}
	for _, e := range g.models {
		if e.Key == name {
			return e, true
		}
	}
	return imageEntry{}, false
}

func (g *imageGen) names() []string {
	out := make([]string, len(g.models))
	for i, e := range g.models {
		out[i] = e.Name
	}
	return out
}

type imageDownload struct {
	File    string `json:"file,omitempty"`
	DoneMB  int64  `json:"done_mb"`
	TotalMB int64  `json:"total_mb"`
	Queued  bool   `json:"queued,omitempty"` // waits for another model's download
}

// imageModelStatus is what /api/status says about one image model.
type imageModelStatus struct {
	Instance string         `json:"instance"`
	Preset   string         `json:"preset"`
	Model    string         `json:"model"`
	Ref      string         `json:"ref"`
	License  string         `json:"license"`
	Steps    int            `json:"steps"`
	Edit     editMode       `json:"edit"`
	MaxRefs  int            `json:"max_refs,omitempty"`
	Align    int            `json:"align"`
	State    InstanceState  `json:"state"`
	Error    string         `json:"error,omitempty"`
	Download *imageDownload `json:"download,omitempty"`
}

func (g *imageGen) Status() []imageModelStatus {
	out := make([]imageModelStatus, 0, len(g.models))
	for _, e := range g.models {
		p := e.Preset
		st := imageModelStatus{Instance: e.Name, Preset: e.Key, Model: p.Label, Ref: p.Ref, License: p.License, Steps: p.Steps,
			Edit: p.Edit, MaxRefs: p.MaxRefs, Align: p.Align, State: StateStopped}
		if inst := g.mgr.Get(e.Name); inst != nil {
			s := inst.Status()
			st.State, st.Error = s.State, s.LastError
		}
		g.mu.Lock()
		switch {
		case g.waiting[e.Name] && g.current == e.Name:
			st.State, st.Error = stateDownloading, ""
			st.Download = &imageDownload{File: g.file, DoneMB: g.done >> 20, TotalMB: g.total >> 20}
		case g.waiting[e.Name]:
			st.State, st.Error = stateDownloading, ""
			st.Download = &imageDownload{Queued: true}
		case g.fetchErr[e.Name] != "" && (st.State == StateStopped || st.State == StateCrashed):
			st.Error = g.fetchErr[e.Name]
		}
		g.mu.Unlock()
		out = append(out, st)
	}
	return out
}

// Boot starts every image model when llama-manager starts, downloading any
// missing file first. It returns at once; downloads run in the background.
func (g *imageGen) Boot() {
	for _, e := range g.models {
		if g.fetch(e.Name) {
			continue
		}
		if err := g.mgr.StartInstance(e.Name); err != nil {
			log.Printf("[%s] failed to start: %v", e.Name, err)
		}
	}
}

// fetch queues a download of the model's missing files, unless it is already
// queued, and starts the model once they are all there. It reports whether a
// download is pending; false means every file is already on disk.
func (g *imageGen) fetch(name string) bool {
	e, ok := g.byName[name]
	if !ok {
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.waiting[name] {
		return true
	}
	if g.ctx.Err() != nil || len(e.Preset.missing()) == 0 {
		return false
	}
	g.waiting[name] = true
	delete(g.fetchErr, name)
	g.queue = append(g.queue, name)
	if !g.running {
		g.running = true
		g.wg.Add(1)
		go g.run()
	}
	return true
}

func (g *imageGen) run() {
	defer g.wg.Done()
	for {
		g.mu.Lock()
		if len(g.queue) == 0 || g.ctx.Err() != nil {
			for _, n := range g.queue {
				delete(g.waiting, n)
				g.fetchErr[n] = "download stopped: llama-manager is shutting down"
			}
			g.queue, g.running, g.current = nil, false, ""
			g.mu.Unlock()
			return
		}
		name := g.queue[0]
		g.queue = g.queue[1:]
		e := g.byName[name]
		todo := e.Preset.missing()
		g.current, g.file, g.done, g.total = name, "", 0, 0
		for _, f := range todo {
			g.total += f.Size
		}
		g.mu.Unlock()

		err := g.fetchFiles(e, todo)
		g.mu.Lock()
		delete(g.waiting, name)
		g.current = ""
		if err != nil {
			g.fetchErr[name] = err.Error()
		}
		g.mu.Unlock()
		if err == nil && g.ctx.Err() == nil {
			if err := g.mgr.StartInstance(name); err != nil {
				log.Printf("[%s] failed to start: %v", name, err)
			}
		}
	}
}

func (g *imageGen) fetchFiles(e imageEntry, todo []imageFile) error {
	inst := g.mgr.Get(e.Name)
	logf := func(s string) {
		if inst != nil {
			inst.addLog(s)
		}
	}
	if len(todo) > 0 {
		log.Printf("[%s] downloading %d model file(s), %d MB", e.Name, len(todo), g.total>>20)
	}
	var done int64
	for _, f := range todo {
		g.mu.Lock()
		g.file = path.Base(f.Path)
		g.mu.Unlock()
		base := done
		err := g.download(f, progress{log: logf, every: 10 * time.Second, bytes: func(n int64) {
			g.mu.Lock()
			g.done = base + n
			g.mu.Unlock()
		}})
		if err != nil {
			msg := "downloading " + path.Base(f.Path) + ": " + err.Error()
			if g.ctx.Err() != nil {
				msg = "download stopped: llama-manager is shutting down"
			}
			logf(msg)
			log.Printf("[%s] %s", e.Name, msg)
			return errors.New(msg)
		}
		done += f.Size
	}
	if len(todo) > 0 {
		log.Printf("[%s] model files downloaded", e.Name)
	}
	return nil
}

// download fetches one file into the hub cache at its pinned commit, checks
// its size and SHA-256, and only then gives it its final name.
func (g *imageGen) download(f imageFile, pr progress) error {
	if _, ok := f.localPath(); ok {
		return nil // another model fetched it
	}
	if err := validateRepo(f.Repo); err != nil {
		return err
	}
	dir := f.snapshotDir(f.Rev)
	dest := filepath.Join(dir, filepath.FromSlash(f.Path))
	if !strings.HasPrefix(filepath.Clean(dest), filepath.Clean(dir)+string(os.PathSeparator)) {
		return fmt.Errorf("unsafe path %q", f.Path)
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	tmp := dest + ".incomplete"
	defer os.Remove(tmp)
	pr.log(fmt.Sprintf("downloading %s from %s (%d MB)", path.Base(f.Path), f.Repo, f.Size>>20))
	u := hfResolveBase + "/" + f.Repo + "/resolve/" + f.Rev + "/" + fileURLPath(f.Path)
	if err := downloadToFile(g.ctx, u, tmp, pr); err != nil {
		return err
	}
	if err := checkFileHash(tmp, f.Size, f.SHA256); err != nil {
		return err
	}
	if err := os.Rename(tmp, dest); err != nil {
		return err
	}
	pr.log("fetched " + path.Base(f.Path))
	return nil
}

func checkFileHash(p string, size int64, want string) error {
	fh, err := os.Open(p)
	if err != nil {
		return err
	}
	defer fh.Close()
	h := sha256.New()
	n, err := io.Copy(h, fh)
	if err != nil {
		return err
	}
	if n != size {
		return fmt.Errorf("got %d bytes, expected %d", n, size)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return fmt.Errorf("the file is corrupt (SHA-256 %s, expected %s)", got, want)
	}
	return nil
}

// imageStats is what an image reply records in its variant's stats.
type imageStats struct {
	TotalMs int64 `json:"total_ms"`
	GenMs   int64 `json:"gen_ms,omitempty"`
	Width   int   `json:"w,omitempty"`
	Height  int   `json:"h,omitempty"`
	Steps   int   `json:"steps,omitempty"`
	Seed    int64 `json:"seed"`
	Refs    int   `json:"refs,omitempty"`
}

// parseImageSize reads the chat's "WxH" size, falling back to 1024x1024 for
// anything sd-server would refuse or the model can't tile.
func parseImageSize(s string, align int) (int, int) {
	if align <= 0 {
		align = 16
	}
	ws, hs, ok := strings.Cut(strings.ToLower(strings.TrimSpace(s)), "x")
	w, err1 := strconv.Atoi(ws)
	h, err2 := strconv.Atoi(hs)
	valid := func(n int) bool { return n >= minImageSide && n <= maxImageSide && n%align == 0 }
	if !ok || err1 != nil || err2 != nil || !valid(w) || !valid(h) {
		return defaultImageSide, defaultImageSide
	}
	return w, h
}

// fitImageSize keeps an edit in the shape of the image being edited: about
// as many pixels as the chosen size, in the reference's aspect ratio.
func fitImageSize(ref []byte, area, align int) (int, int, bool) {
	c, _, err := image.DecodeConfig(bytes.NewReader(ref))
	if err != nil || c.Width <= 0 || c.Height <= 0 {
		return 0, 0, false
	}
	r := float64(c.Width) / float64(c.Height)
	side := func(v float64) int {
		n := int(math.Round(v/float64(align))) * align
		return max(minImageSide, min(maxImageSide, n))
	}
	w := math.Sqrt(float64(area) * r)
	return side(w), side(w / r), true
}

type sdJob struct {
	Status        string `json:"status"`
	QueuePosition int    `json:"queue_position"`
	Started       *int64 `json:"started"`
	Completed     *int64 `json:"completed"`
	Result        *struct {
		Images []struct {
			B64 string `json:"b64_json"`
		} `json:"images"`
	} `json:"result"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
	Preview *struct {
		Step  int    `json:"step"`
		Total int    `json:"total_steps"`
		B64   string `json:"b64_json"`
	} `json:"preview"`
}

var errImageJobGone = errors.New("the image model no longer has this job (did it restart?)")

// sdError extracts sd-server's message from a failed response: {"error":
// "..."}, {"error": "...", "message": "..."} or {"error": {"message": ...}}.
func sdError(resp *http.Response) string {
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	var e map[string]any
	if json.Unmarshal(b, &e) == nil {
		var parts []string
		switch v := e["error"].(type) {
		case string:
			parts = append(parts, v)
		case map[string]any:
			if m, ok := v["message"].(string); ok {
				parts = append(parts, m)
			}
		}
		if m, ok := e["message"].(string); ok && m != "" {
			parts = append(parts, m)
		}
		if len(parts) > 0 {
			return truncateRunes(strings.Join(parts, ": "), 400)
		}
	}
	if msg := strings.TrimSpace(string(b)); msg != "" {
		return truncateRunes(msg, 400)
	}
	return resp.Status
}

func (g *imageGen) submit(ctx context.Context, inst *Instance, body map[string]any) (string, error) {
	name := inst.conf.Name
	data, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, inst.baseURL()+"/sdcpp/v1/img_gen", bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := g.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("couldn't reach %s: %w", name, err)
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusTooManyRequests:
		return "", fmt.Errorf("%s is busy: its queue is full. Try again in a moment", name)
	case resp.StatusCode != http.StatusAccepted && resp.StatusCode != http.StatusOK:
		return "", fmt.Errorf("%s refused the request: %s", name, sdError(resp))
	}
	var out struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&out); err != nil || out.ID == "" {
		return "", fmt.Errorf("%s sent an answer llama-manager can't read", name)
	}
	return out.ID, nil
}

func (g *imageGen) job(ctx context.Context, inst *Instance, id string) (sdJob, error) {
	var j sdJob
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, inst.baseURL()+"/sdcpp/v1/jobs/"+url.PathEscape(id), nil)
	if err != nil {
		return j, err
	}
	resp, err := g.http.Do(req)
	if err != nil {
		return j, err
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusNotFound, http.StatusGone:
		return j, errImageJobGone
	default:
		return j, errors.New(sdError(resp))
	}
	err = json.NewDecoder(io.LimitReader(resp.Body, 2*maxGenImageBytes)).Decode(&j)
	return j, err
}

// cancelJob asks sd-server to drop a job the chat no longer waits for, so the
// GPU moves on to the next one.
func (g *imageGen) cancelJob(inst *Instance, id string) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, inst.baseURL()+"/sdcpp/v1/jobs/"+url.PathEscape(id)+"/cancel", nil)
	if err != nil {
		return
	}
	if resp, err := g.http.Do(req); err == nil {
		resp.Body.Close()
	}
}

const imageReplyNote = "[An image was generated for this request.]"

// generateImage makes the picture for reply t: the last user message is the
// prompt and its images are references. It runs as an sd-server job, polled
// for progress, and the result is saved with this version of the reply.
func (cs *ChatService) generateImage(ctx context.Context, g *generation, inst *Instance, p chatParams) genDone {
	t := g.t
	start := time.Now()
	var st imageStats
	finish := func(status, errMsg string, img *AttachmentMeta) genDone {
		st.TotalMs = time.Since(start).Milliseconds()
		v := Variant{ID: t.VariantID, Status: status, Error: errMsg}
		v.Stats, _ = json.Marshal(st)
		if err := cs.store.SaveVariant(v); err != nil {
			log.Printf("chat %s: saving image reply: %v", t.ChatID, err)
		}
		d := genDone{MessageID: t.MessageID, VariantID: t.VariantID, Status: status, Error: errMsg, Stats: v.Stats}
		if img != nil {
			d.Images = []AttachmentMeta{*img}
		}
		return d
	}
	ig := cs.images
	name := inst.conf.Name
	if ig == nil {
		return finish(variantError, "image generation is off in config.yaml", nil)
	}
	e, ok := ig.model(name)
	if !ok {
		return finish(variantError, name+" isn't an image model", nil)
	}
	pr := e.Preset
	hist, err := cs.store.History(t.ChatID, t.Seq)
	if err != nil {
		return finish(variantError, "reading chat history: "+err.Error(), nil)
	}
	if len(hist) == 0 || hist[len(hist)-1].Role != "user" {
		return finish(variantError, "there is no message to make an image from", nil)
	}
	last := hist[len(hist)-1]
	prompt := strings.TrimSpace(last.Content)
	var refs []string
	var firstRef []byte
	for _, a := range last.Attachments {
		if a.Kind != "image" {
			return finish(variantError, fmt.Sprintf("%s makes images from a description and reference images; it can't read %s. Remove that file and send again.", name, a.Name), nil)
		}
		if firstRef == nil {
			firstRef = a.Data
		}
		refs = append(refs, "data:"+a.Mime+";base64,"+base64.StdEncoding.EncodeToString(a.Data))
	}
	switch {
	case len(refs) > 0 && pr.Edit == editNone:
		return finish(variantError, fmt.Sprintf("%s (%s) makes images from a description only and can't edit the attached image. Pick an image model that edits, or send the message without the image.", name, pr.Label), nil)
	case len(refs) == 0 && pr.Edit == editOnly:
		return finish(variantError, fmt.Sprintf("%s (%s) edits images: attach one, or press Edit on a picture, and say what to change.", name, pr.Label), nil)
	case pr.MaxRefs > 0 && len(refs) > pr.MaxRefs:
		return finish(variantError, fmt.Sprintf("%s takes at most %d images at once; attach fewer.", name, pr.MaxRefs), nil)
	case prompt == "":
		return finish(variantError, "Describe the image you want, or what to change in the attached one.", nil)
	}
	align := max(pr.Align, 16)
	w, h := parseImageSize(p.Size, align)
	if firstRef != nil {
		if fw, fh, ok := fitImageSize(firstRef, w*h, align); ok {
			w, h = fw, fh
		}
	}
	seed, ok := p.opts["seed"].(int64)
	if !ok || seed < 0 {
		seed = rand.Int64N(1 << 31)
	}
	st = imageStats{Width: w, Height: h, Steps: pr.Steps, Seed: seed, Refs: len(refs)}
	sample := map[string]any{
		"sample_steps":  pr.Steps,
		"sample_method": pr.Sampler,
		"guidance":      map[string]any{"txt_cfg": pr.CFG},
	}
	if pr.FlowShift > 0 {
		sample["flow_shift"] = pr.FlowShift
	}
	if len(pr.Sigmas) > 0 {
		sample["custom_sigmas"] = pr.Sigmas
	}
	body := map[string]any{
		"prompt": prompt, "width": w, "height": h, "seed": seed, "batch_count": 1,
		"sample_params": sample, "output_format": "png",
		// The cheap latent projection gives a step count and a rough look at
		// the picture while it forms.
		"preview": "proj", "preview_interval": 1,
	}
	if len(refs) > 0 {
		body["ref_images"] = refs
	}
	if lora := pr.loraName(); lora != "" {
		body["lora"] = []map[string]any{{"path": lora, "multiplier": 1.0}}
	}
	g.setStatus("Sending the prompt to " + name)
	id, err := ig.submit(ctx, inst, body)
	if err != nil {
		if ctx.Err() != nil {
			return finish(variantStopped, "", nil)
		}
		return finish(variantError, err.Error(), nil)
	}

	deadline := time.NewTimer(e.Wait)
	defer deadline.Stop()
	tick := time.NewTicker(imagePollEvery)
	defer tick.Stop()
	fails, lastStep, phase := 0, -1, ""
	setPhase := func(s string) {
		if s != phase {
			phase = s
			g.setStatus(s)
		}
	}
	for {
		select {
		case <-ctx.Done():
			ig.cancelJob(inst, id)
			return finish(variantStopped, "", nil)
		case <-deadline.C:
			ig.cancelJob(inst, id)
			return finish(variantError, fmt.Sprintf("%s didn't finish the image within %s", name, e.Wait), nil)
		case <-tick.C:
		}
		j, err := ig.job(ctx, inst, id)
		if err != nil {
			if ctx.Err() != nil {
				continue
			}
			if errors.Is(err, errImageJobGone) {
				return finish(variantError, name+": "+err.Error(), nil)
			}
			if fails++; fails >= imageJobFailLimit {
				return finish(variantError, fmt.Sprintf("%s stopped answering: %v", name, err), nil)
			}
			continue
		}
		fails = 0
		switch j.Status {
		case "queued":
			if j.QueuePosition > 0 {
				setPhase(fmt.Sprintf("Waiting for %s (%d ahead)", name, j.QueuePosition))
			} else {
				setPhase("Waiting for " + name)
			}
		case "generating":
			if pv := j.Preview; pv != nil && pv.Step != lastStep {
				lastStep = pv.Step
				label := fmt.Sprintf("Step %d of %d", pv.Step, pv.Total)
				if pv.Total > 0 && pv.Step >= pv.Total {
					label = "Decoding the image"
				}
				phase = label
				g.setPreview(label, "data:image/png;base64,"+pv.B64)
			} else if j.Preview == nil {
				setPhase("Reading the prompt")
			}
		case "completed":
			if j.Result == nil || len(j.Result.Images) == 0 {
				return finish(variantError, name+" finished without an image", nil)
			}
			data, err := base64.StdEncoding.DecodeString(j.Result.Images[0].B64)
			if err != nil || len(data) == 0 || len(data) > maxGenImageBytes || http.DetectContentType(data) != "image/png" {
				return finish(variantError, name+" sent an image llama-manager can't read", nil)
			}
			if j.Started != nil && j.Completed != nil {
				st.GenMs = *j.Completed - *j.Started
			}
			meta, err := cs.store.AddReplyImage(t, fmt.Sprintf("%s-%d.png", e.Key, seed), "image/png", data)
			if err != nil {
				return finish(variantError, "saving the image: "+err.Error(), nil)
			}
			return finish(variantDone, "", &meta)
		case "failed":
			msg := name + " couldn't make the image"
			if j.Error != nil && j.Error.Message != "" {
				msg += ": " + j.Error.Message
			}
			return finish(variantError, msg, nil)
		case "cancelled":
			return finish(variantStopped, "", nil)
		}
	}
}

// handleV1Images is the OpenAI-compatible image endpoint for other apps: it
// forwards to an image model's own /v1/images/generations. "model" names the
// image model's instance or preset; left out, it is the first image model.
// sd-server applies the model's defaults (steps, CFG, schedule) but not a
// LoRA, so a LoRA-based model is best used from the chat.
func (ws *WebServer) handleV1Images(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if ws.imageGen == nil {
		http.Error(w, "image generation is off; turn it on in the image_generation section of config.yaml", http.StatusNotFound)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "reading request: "+err.Error(), http.StatusBadRequest)
		return
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}
	m, _ := req["model"].(string)
	e, ok := ws.imageGen.pick(m)
	if !ok {
		http.Error(w, fmt.Sprintf("model not found: %s (image models: %s)", m, strings.Join(ws.imageGen.names(), ", ")), http.StatusNotFound)
		return
	}
	inst := ws.mgr.Get(e.Name)
	if inst == nil {
		http.Error(w, e.Name+" doesn't exist", http.StatusNotFound)
		return
	}
	if s := inst.State(); s != StateRunning {
		http.Error(w, fmt.Sprintf("%s is %s, not running", e.Name, s), http.StatusServiceUnavailable)
		return
	}
	delete(req, "model")
	data, _ := json.Marshal(req)
	up, err := http.NewRequestWithContext(r.Context(), http.MethodPost, inst.baseURL()+"/v1/images/generations", bytes.NewReader(data))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	up.Header.Set("Content-Type", "application/json")
	resp, err := chatClient.Do(up)
	if err != nil {
		http.Error(w, "proxying to "+e.Name+": "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}
