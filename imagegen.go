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
	"io"
	"log"
	"math/rand/v2"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Image generation (image_generation in config.yaml) runs stable-diffusion.cpp's
// sd-server as the instance image-gen. Picked in the chat, it turns each
// message into a picture saved with that version of the reply; images attached
// to the message go along as references, which edits them.

const (
	imageGenName        = "image-gen"
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

// stateDownloading is what /api/status reports for image-gen while its model
// files are being fetched; the instance itself is stopped meanwhile.
const stateDownloading InstanceState = "downloading"

// imagePollEvery is how often a generation's job is polled. Var for tests.
var imagePollEvery = 400 * time.Millisecond

// imageFile is one file a preset loads, pinned to a commit so the size and
// hash checked after download can't drift.
type imageFile struct {
	Flag   string // the sd-server option that loads it
	Repo   string
	Rev    string
	Path   string // inside the repo
	Size   int64
	SHA256 string
}

type imagePreset struct {
	Label   string
	Ref     string // shown as the instance's model
	Files   []imageFile
	Steps   int
	CFG     float64
	Sampler string
}

// imagePresets are the image models tested for this. FLUX.2 klein 4B (Apache
// 2.0) is distilled to 4 steps; Q8_0 keeps the gfx906 cards on f16 math.
var imagePresets = map[string]imagePreset{
	"flux2-klein-4b": {
		Label: "FLUX.2 klein 4B", Ref: "leejet/FLUX.2-klein-4B-GGUF:Q8_0", Steps: 4, CFG: 1, Sampler: "euler",
		Files: []imageFile{
			{Flag: "--diffusion-model", Repo: "leejet/FLUX.2-klein-4B-GGUF", Rev: "3b1f5a9dc3abb32238b053aeb3d823c30afdacbd",
				Path: "flux-2-klein-4b-Q8_0.gguf", Size: 4300629440, SHA256: "0bba6951258ec8f92d51114a8fa13e66828297bfff58a738f52729b3ef66fa28"},
			{Flag: "--vae", Repo: "Comfy-Org/Flux-2-Klein-4B", Rev: "5b3abca5c0dc1ecd4eaf3340868d09063fc1c965",
				Path: "split_files/vae/flux2-vae.safetensors", Size: 336211292, SHA256: "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3"},
			{Flag: "--llm", Repo: "unsloth/Qwen3-4B-GGUF", Rev: "22c9fc8a8c7700b76a1789366280a6a5a1ad1120",
				Path: "Qwen3-4B-Q8_0.gguf", Size: 4280405792, SHA256: "eed555233267a33c7e8ee31682762cc7751b3f6d224039086e0e846f05fffa5d"},
		},
	},
}

func imagePresetNames() []string {
	names := make([]string, 0, len(imagePresets))
	for n := range imagePresets {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// ImageGenConf runs sd-server as its own instance, image-gen. Like
// image_recognition it picks a binary, its arguments and a port, so it is set
// only in config.yaml.
type ImageGenConf struct {
	Enabled   bool     `yaml:"enabled" json:"enabled"`
	ServerBin string   `yaml:"server_bin,omitempty" json:"server_bin,omitempty"`
	Model     string   `yaml:"model,omitempty" json:"model,omitempty"`
	Port      int      `yaml:"port,omitempty" json:"port,omitempty"`
	GPUIDs    []int    `yaml:"gpu_ids,omitempty" json:"gpu_ids,omitempty"`
	ExtraArgs []string `yaml:"extra_args,omitempty" json:"extra_args,omitempty"`
	Timeout   duration `yaml:"timeout,omitempty" json:"timeout,omitempty"`
}

func (c ImageGenConf) preset() (imagePreset, bool) {
	m := strings.ToLower(strings.TrimSpace(c.Model))
	if m == "" {
		m = defaultImageModel
	}
	p, ok := imagePresets[m]
	return p, ok
}

func (c ImageGenConf) port() int {
	if c.Port > 0 {
		return c.Port
	}
	return defaultImageGenPort
}

func (c ImageGenConf) timeout() time.Duration {
	if c.Timeout.Duration > 0 {
		return c.Timeout.Duration
	}
	return defaultImageWait
}

// instanceConf is the image-gen instance. The manager doesn't boot it itself:
// imageGen.Boot starts it once the model files are on disk.
func (c ImageGenConf) instanceConf() InstanceConf {
	p, _ := c.preset()
	gpus := append([]int(nil), c.GPUIDs...)
	if len(gpus) == 0 {
		gpus = []int{0}
	}
	extra := append([]string{}, c.ExtraArgs...)
	return InstanceConf{Name: imageGenName, Model: p.Ref, Port: c.port(), GPUIDs: gpus, ExtraArgs: &extra, role: roleImageGen}
}

func (c ImageGenConf) check() error {
	var errs []error
	if c.Enabled && strings.TrimSpace(c.ServerBin) == "" {
		errs = append(errs, fmt.Errorf("image_generation.server_bin is required: the path to stable-diffusion.cpp's sd-server"))
	}
	if _, ok := c.preset(); !ok {
		errs = append(errs, fmt.Errorf("image_generation.model %q is unknown; use %s", c.Model, strings.Join(imagePresetNames(), " or ")))
	}
	if c.Port != 0 && !validPort(c.Port) {
		errs = append(errs, fmt.Errorf("image_generation.port must be 1-65535"))
	}
	if len(c.GPUIDs) > 1 {
		errs = append(errs, fmt.Errorf("image_generation.gpu_ids: sd-server runs on one GPU; list just one"))
	}
	for _, id := range c.GPUIDs {
		if id < 0 {
			errs = append(errs, fmt.Errorf("image_generation.gpu_ids: gpu id %d must be >= 0", id))
		}
	}
	if c.Timeout.Duration < 0 {
		errs = append(errs, fmt.Errorf("image_generation.timeout must be >= 0"))
	}
	return errors.Join(errs...)
}

// clash reports an instance that would collide with image-gen.
func (c ImageGenConf) clash(ic InstanceConf) error {
	if !c.Enabled {
		return nil
	}
	if ic.Name == imageGenName {
		return fmt.Errorf("the name %q belongs to image generation (image_generation in config.yaml); pick another name", imageGenName)
	}
	if ic.Port == c.port() {
		return fmt.Errorf("port %d is used by image generation (image_generation in config.yaml); pick another port", ic.Port)
	}
	return nil
}

func (f imageFile) snapshotDir(rev string) string {
	return filepath.Join(huggingfaceHubDir(), "models--"+strings.ReplaceAll(f.Repo, "/", "--"), "snapshots", rev)
}

// localPath finds the file in the HF hub cache: at its pinned commit, where
// image-gen downloads it, or under main when the Models page fetched the same
// file, as long as the size matches.
func (f imageFile) localPath() (string, bool) {
	for _, rev := range []string{f.Rev, "main"} {
		p := filepath.Join(f.snapshotDir(rev), filepath.FromSlash(f.Path))
		if fi, err := os.Stat(p); err == nil && fi.Mode().IsRegular() && fi.Size() == f.Size {
			return p, true
		}
	}
	return "", false
}

func (p imagePreset) missing() []imageFile {
	var out []imageFile
	for _, f := range p.Files {
		if _, ok := f.localPath(); !ok {
			out = append(out, f)
		}
	}
	return out
}

var errImageFilesMissing = errors.New("its model files aren't downloaded yet; llama-manager fetches them when it starts or when you start " + imageGenName)

// buildImageGenLaunch is sd-server's command line. It always listens on
// loopback: sd-server has no auth and answers any Origin with credentials, so
// browsers and other machines reach it only through the manager.
func buildImageGenLaunch(conf InstanceConf, cfg *Config) (launchSpec, error) {
	cfg.mu.RLock()
	c := cfg.ImageGeneration
	gpuEnv := cfg.GPUEnvVar()
	cfg.mu.RUnlock()
	p, ok := c.preset()
	if !ok {
		return launchSpec{}, fmt.Errorf("image_generation.model %q is unknown", c.Model)
	}
	spec := launchSpec{bin: c.ServerBin, host: "127.0.0.1"}
	args := []string{"--listen-ip", spec.host, "--listen-port", strconv.Itoa(conf.Port)}
	for _, f := range p.Files {
		local, ok := f.localPath()
		if !ok {
			return launchSpec{}, errImageFilesMissing
		}
		args = append(args, f.Flag, local)
	}
	// Defaults for /v1/images/generations callers; the chat sends its own.
	args = append(args,
		"--steps", strconv.Itoa(p.Steps),
		"--cfg-scale", strconv.FormatFloat(p.CFG, 'f', -1, 64),
		"--sampling-method", p.Sampler,
		"-W", strconv.Itoa(defaultImageSide), "-H", strconv.Itoa(defaultImageSide),
		"--seed", "-1",
		// Load every weight before listening, so "running" means ready and the
		// first image isn't slowed by reading 9 GB from disk.
		"--eager-load",
	)
	if gpuEnv != "" && len(conf.GPUIDs) > 0 {
		spec.env = append(spec.env, gpuEnv+"="+strconv.Itoa(conf.GPUIDs[0]))
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

// imageGen fetches image-gen's model files, starts it, and runs generations
// for the chat. It is nil when image_generation is off.
type imageGen struct {
	mgr     *Manager
	conf    ImageGenConf
	preset  imagePreset
	timeout time.Duration
	http    *http.Client

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	mu       sync.Mutex
	fetching bool
	fetchErr string
	file     string
	done     int64 // bytes fetched so far in this download, all files
	total    int64
}

func newImageGen(c ImageGenConf, mgr *Manager) *imageGen {
	if !c.Enabled {
		return nil
	}
	p, _ := c.preset()
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.Proxy = nil // sd-server is our own child process
	ctx, cancel := context.WithCancel(context.Background())
	return &imageGen{
		mgr: mgr, conf: c, preset: p, timeout: c.timeout(), ctx: ctx, cancel: cancel,
		http: &http.Client{Transport: tr, CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		}},
	}
}

// Close stops a download in progress and waits for it to give up.
func (g *imageGen) Close() {
	g.cancel()
	g.wg.Wait()
}

type imageDownload struct {
	File    string `json:"file"`
	DoneMB  int64  `json:"done_mb"`
	TotalMB int64  `json:"total_mb"`
}

// imageGenStatus is what /api/status says about image generation.
type imageGenStatus struct {
	Instance string         `json:"instance"`
	Model    string         `json:"model"`
	Ref      string         `json:"ref"`
	Steps    int            `json:"steps"`
	State    InstanceState  `json:"state"`
	Error    string         `json:"error,omitempty"`
	Download *imageDownload `json:"download,omitempty"`
}

func (g *imageGen) Status() imageGenStatus {
	st := imageGenStatus{Instance: imageGenName, Model: g.preset.Label, Ref: g.preset.Ref, Steps: g.preset.Steps, State: StateStopped}
	if inst := g.mgr.Get(imageGenName); inst != nil {
		s := inst.Status()
		st.State, st.Error = s.State, s.LastError
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.fetching {
		st.State, st.Error = stateDownloading, ""
		st.Download = &imageDownload{File: g.file, DoneMB: g.done >> 20, TotalMB: g.total >> 20}
	} else if g.fetchErr != "" && (st.State == StateStopped || st.State == StateCrashed) {
		st.Error = g.fetchErr
	}
	return st
}

// Boot starts image-gen when llama-manager starts, downloading any missing
// model file first. It returns at once; a download runs in the background.
func (g *imageGen) Boot() {
	if g.fetch() {
		return
	}
	if err := g.mgr.StartInstance(imageGenName); err != nil {
		log.Printf("[%s] failed to start: %v", imageGenName, err)
	}
}

// fetch downloads the missing model files in the background, unless that is
// already under way, and starts image-gen once they are all there. It reports
// whether a download is running; false means every file is already on disk.
func (g *imageGen) fetch() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.fetching {
		return true
	}
	todo := g.preset.missing()
	if len(todo) == 0 || g.ctx.Err() != nil {
		return false
	}
	g.fetching, g.fetchErr, g.done, g.total, g.file = true, "", 0, 0, ""
	for _, f := range todo {
		g.total += f.Size
	}
	g.wg.Add(1)
	go g.run(todo)
	return true
}

func (g *imageGen) run(todo []imageFile) {
	defer g.wg.Done()
	inst := g.mgr.Get(imageGenName)
	logf := func(s string) {
		if inst != nil {
			inst.addLog(s)
		}
	}
	log.Printf("[%s] downloading %d model file(s), %d MB", imageGenName, len(todo), g.total>>20)
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
			log.Printf("[%s] %s", imageGenName, msg)
			g.mu.Lock()
			g.fetching, g.fetchErr = false, msg
			g.mu.Unlock()
			return
		}
		done += f.Size
	}
	g.mu.Lock()
	g.fetching = false
	g.mu.Unlock()
	log.Printf("[%s] model files downloaded", imageGenName)
	if g.ctx.Err() != nil {
		return
	}
	if err := g.mgr.StartInstance(imageGenName); err != nil {
		log.Printf("[%s] failed to start: %v", imageGenName, err)
	}
}

// download fetches one file into the hub cache at its pinned commit, checks
// its size and SHA-256, and only then gives it its final name.
func (g *imageGen) download(f imageFile, pr progress) error {
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
// anything sd-server would refuse or that FLUX can't tile (multiples of 16).
func parseImageSize(s string) (int, int) {
	ws, hs, ok := strings.Cut(strings.ToLower(strings.TrimSpace(s)), "x")
	w, err1 := strconv.Atoi(ws)
	h, err2 := strconv.Atoi(hs)
	valid := func(n int) bool { return n >= minImageSide && n <= maxImageSide && n%16 == 0 }
	if !ok || err1 != nil || err2 != nil || !valid(w) || !valid(h) {
		return defaultImageSide, defaultImageSide
	}
	return w, h
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

var errImageJobGone = errors.New(imageGenName + " no longer has this job (did it restart?)")

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
	data, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, inst.baseURL()+"/sdcpp/v1/img_gen", bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := g.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("couldn't reach %s: %w", imageGenName, err)
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusTooManyRequests:
		return "", fmt.Errorf("%s is busy: its queue is full. Try again in a moment", imageGenName)
	case resp.StatusCode != http.StatusAccepted && resp.StatusCode != http.StatusOK:
		return "", fmt.Errorf("%s refused the request: %s", imageGenName, sdError(resp))
	}
	var out struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&out); err != nil || out.ID == "" {
		return "", fmt.Errorf("%s sent an answer llama-manager can't read", imageGenName)
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
	for _, a := range last.Attachments {
		if a.Kind != "image" {
			return finish(variantError, fmt.Sprintf("%s makes images from a description and reference images; it can't read %s. Remove that file and send again.", name, a.Name), nil)
		}
		refs = append(refs, "data:"+a.Mime+";base64,"+base64.StdEncoding.EncodeToString(a.Data))
	}
	if prompt == "" {
		return finish(variantError, "Describe the image you want, or what to change in the attached one.", nil)
	}
	w, h := parseImageSize(p.Size)
	seed, ok := p.opts["seed"].(int64)
	if !ok || seed < 0 {
		seed = rand.Int64N(1 << 31)
	}
	st = imageStats{Width: w, Height: h, Steps: ig.preset.Steps, Seed: seed, Refs: len(refs)}
	body := map[string]any{
		"prompt": prompt, "width": w, "height": h, "seed": seed, "batch_count": 1,
		"sample_params": map[string]any{
			"sample_steps":  ig.preset.Steps,
			"sample_method": ig.preset.Sampler,
			"guidance":      map[string]any{"txt_cfg": ig.preset.CFG},
		},
		"output_format": "png",
		// The cheap latent projection gives a step count and a rough look at
		// the picture while it forms.
		"preview": "proj", "preview_interval": 1,
	}
	if len(refs) > 0 {
		body["ref_images"] = refs
	}
	g.setStatus("Sending the prompt to " + name)
	id, err := ig.submit(ctx, inst, body)
	if err != nil {
		if ctx.Err() != nil {
			return finish(variantStopped, "", nil)
		}
		return finish(variantError, err.Error(), nil)
	}

	deadline := time.NewTimer(ig.timeout)
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
			return finish(variantError, fmt.Sprintf("%s didn't finish the image within %s", name, ig.timeout), nil)
		case <-tick.C:
		}
		j, err := ig.job(ctx, inst, id)
		if err != nil {
			if ctx.Err() != nil {
				continue
			}
			if errors.Is(err, errImageJobGone) {
				return finish(variantError, err.Error(), nil)
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
			meta, err := cs.store.AddReplyImage(t, fmt.Sprintf("%s-%d.png", strings.ReplaceAll(strings.ToLower(ig.preset.Label), " ", "-"), seed), "image/png", data)
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

// handleV1Images is the OpenAI-compatible image endpoint for other apps:
// it forwards to image-gen's own /v1/images/generations. "model" may be left
// out or name image-gen.
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
	if m, _ := req["model"].(string); m != "" && m != imageGenName && !slices.Contains(imagePresetNames(), m) {
		http.Error(w, fmt.Sprintf("model not found: %s (image models: %s)", m, imageGenName), http.StatusNotFound)
		return
	}
	inst := ws.mgr.Get(imageGenName)
	if inst == nil {
		http.Error(w, imageGenName+" doesn't exist", http.StatusNotFound)
		return
	}
	if s := inst.State(); s != StateRunning {
		http.Error(w, fmt.Sprintf("%s is %s, not running", imageGenName, s), http.StatusServiceUnavailable)
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
		http.Error(w, "proxying to "+imageGenName+": "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}
