package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeSD is enough of sd-server for the manager: model status, native jobs
// that step through previews, cancel, and the OpenAI endpoint.
type fakeSD struct {
	mu      sync.Mutex
	steps   int
	outcome string // "" completes, "failed", "hang"
	state   string // models/status; "" is ready
	hold    chan struct{}
	bodies  []map[string]any
	polls   map[string]int
	cancels []string
	openai  []map[string]any
}

func newFakeSD() *fakeSD { return &fakeSD{steps: 2, polls: map[string]int{}} }

func tinyPNG() []byte {
	var b bytes.Buffer
	_ = png.Encode(&b, image.NewRGBA(image.Rect(0, 0, 2, 2)))
	return b.Bytes()
}

func (f *fakeSD) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	pic := base64.StdEncoding.EncodeToString(tinyPNG())
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	switch {
	case r.URL.Path == "/sdcpp/v1/models/status":
		f.mu.Lock()
		st := f.state
		f.mu.Unlock()
		if st == "" {
			st = "ready"
		}
		writeJSON(w, http.StatusOK, map[string]any{"state": st})
	case r.Method == http.MethodPost && r.URL.Path == "/sdcpp/v1/img_gen":
		f.mu.Lock()
		f.bodies = append(f.bodies, body)
		id := fmt.Sprintf("job_%d", len(f.bodies))
		f.polls[id] = 0
		f.mu.Unlock()
		writeJSON(w, http.StatusAccepted, map[string]any{"id": id, "status": "queued", "poll_url": "/sdcpp/v1/jobs/" + id})
	case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/cancel"):
		id := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/sdcpp/v1/jobs/"), "/cancel")
		f.mu.Lock()
		f.cancels = append(f.cancels, id)
		f.polls[id] = -1
		f.mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]any{"id": id, "cancel_requested": true})
	case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/sdcpp/v1/jobs/"):
		f.mu.Lock()
		hold := f.hold
		f.mu.Unlock()
		if hold != nil {
			<-hold
		}
		id := strings.TrimPrefix(r.URL.Path, "/sdcpp/v1/jobs/")
		f.mu.Lock()
		defer f.mu.Unlock()
		n, ok := f.polls[id]
		if !ok {
			http.NotFound(w, r)
			return
		}
		job := map[string]any{"id": id, "status": "queued", "queue_position": 1}
		switch {
		case n < 0:
			job = map[string]any{"id": id, "status": "cancelled", "error": map[string]string{"code": "cancelled", "message": "job cancelled by client"}}
		case n == 0:
		case n <= f.steps || f.outcome == "hang":
			job = map[string]any{"id": id, "status": "generating", "started": 1000,
				"preview": map[string]any{"pass": 1, "step": min(n, f.steps), "total_steps": f.steps, "b64_json": pic}}
		case f.outcome == "failed":
			job = map[string]any{"id": id, "status": "failed", "error": map[string]string{"code": "generation_failed", "message": "out of memory"}}
		default:
			job = map[string]any{"id": id, "status": "completed", "started": 1000, "completed": 4000,
				"result": map[string]any{"output_format": "png", "images": []map[string]any{{"index": 0, "b64_json": pic}}}}
		}
		if n >= 0 {
			f.polls[id] = n + 1
		}
		writeJSON(w, http.StatusOK, job)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/images/generations":
		f.mu.Lock()
		f.openai = append(f.openai, body)
		f.mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]any{"created": 1, "data": []map[string]any{{"b64_json": pic}}})
	default:
		http.NotFound(w, r)
	}
}

func runFakeSDServer(host, port string) {
	ln, err := net.Listen("tcp", net.JoinHostPort(host, port))
	if err != nil {
		fmt.Fprintln(os.Stderr, "fake sd-server listen:", err)
		os.Exit(1)
	}
	_ = http.Serve(ln, newFakeSD())
}

type imageRig struct {
	*chatAPIRig
	sd  *fakeSD            // the first image model's
	sds map[string]*fakeSD // by instance name
}

// newImageRig is a chat rig with running image models backed by fake
// sd-servers, next to the chat model "c". With no models it is the
// single-model form: one instance, image-gen, running FLUX.2 klein 4B.
func newImageRig(t *testing.T, models ...ImageModelConf) *imageRig {
	t.Helper()
	rig := newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if body["stream"] != true {
			titleReply(w, "T")
			return
		}
		sse(w, delta("content", "ok"), finalChunk, "[DONE]")
	})
	old := imagePollEvery
	imagePollEvery = 5 * time.Millisecond
	t.Cleanup(func() { imagePollEvery = old })
	serve := func() (*fakeSD, int) {
		sd := newFakeSD()
		up := httptest.NewServer(sd)
		t.Cleanup(up.Close)
		return sd, portOf(t, up.URL)
	}
	conf := ImageGenConf{Enabled: true, ServerBin: "/bin/true"}
	fakes := map[int]*fakeSD{}
	if len(models) == 0 {
		var sd *fakeSD
		sd, conf.Port = serve()
		fakes[conf.Port] = sd
	}
	for _, m := range models {
		sd, port := serve()
		m.Port = port
		fakes[port] = sd
		conf.Models = append(conf.Models, m)
	}
	ws := rig.ws
	ws.cfg.mu.Lock()
	ws.cfg.ImageGeneration = conf
	ws.cfg.mu.Unlock()
	ir := &imageRig{chatAPIRig: rig, sds: map[string]*fakeSD{}}
	for i, e := range conf.entries() {
		if err := ws.mgr.AddInstance(e.instanceConf()); err != nil {
			t.Fatal(err)
		}
		setState(ws.mgr.Get(e.Name), StateRunning)
		ir.sds[e.Name] = fakes[e.Port]
		if i == 0 {
			ir.sd = fakes[e.Port]
		}
	}
	ws.imageGen = newImageGen(conf, ws.mgr)
	t.Cleanup(ws.imageGen.Close)
	ws.chats.images = ws.imageGen
	return ir
}

func (ir *imageRig) imageChat(t *testing.T, params map[string]any) Chat {
	t.Helper()
	return ir.chatWith(t, imageGenName, params)
}

func (ir *imageRig) chatWith(t *testing.T, instance string, params map[string]any) Chat {
	t.Helper()
	resp, b := ir.do(t, "POST", "/api/chats", map[string]any{"instance": instance, "params": params})
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("create chat: %d %s", resp.StatusCode, b)
	}
	var c Chat
	json.Unmarshal(b, &c)
	return c
}

func (ir *imageRig) send(t *testing.T, chatID, content string, atts ...map[string]any) {
	t.Helper()
	body := map[string]any{"content": content}
	if len(atts) > 0 {
		body["attachments"] = atts
	}
	if resp, b := ir.do(t, "POST", "/api/chats/"+chatID+"/messages", body); resp.StatusCode != http.StatusCreated {
		t.Fatalf("send: %d %s", resp.StatusCode, b)
	}
}

func (f *fakeSD) snapshot() (bodies []map[string]any, cancels []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]map[string]any(nil), f.bodies...), append([]string(nil), f.cancels...)
}

type sseEvent struct {
	name string
	data map[string]any
}

func readSSE(t *testing.T, resp *http.Response, until string) []sseEvent {
	t.Helper()
	var out []sseEvent
	var cur sseEvent
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64<<10), 4<<20)
	for sc.Scan() {
		line := sc.Text()
		if name, ok := strings.CutPrefix(line, "event: "); ok {
			cur = sseEvent{name: name}
		} else if data, ok := strings.CutPrefix(line, "data: "); ok {
			_ = json.Unmarshal([]byte(data), &cur.data)
			out = append(out, cur)
			if cur.name == until {
				break
			}
		}
	}
	return out
}

func TestImageGenConfigRules(t *testing.T) {
	base := "server_bin: /bin/true\n"
	on := "image_generation:\n  enabled: true\n  server_bin: /opt/sd-server\n"
	for name, tc := range map[string]struct{ yaml, want string }{
		"no binary":      {"image_generation:\n  enabled: true\n", "server_bin is required"},
		"unknown model":  {on + "  model: sdxl\n", `model "sdxl" is unknown`},
		"three gpus":     {on + "  gpu_ids: [0, 1, 2]\n", "at most two GPUs"},
		"manager port":   {"manager_port: 9096\n" + on, "manager's own port"},
		"reader port":    {"image_recognition:\n  enabled: true\n  port: 9300\n" + on + "  port: 9300\n", "image reader's port"},
		"instance port":  {on + "instances:\n  - {name: a, model: /m.gguf, port: 9096, gpu_ids: [0]}\n", "used by image generation"},
		"instance name":  {on + "instances:\n  - {name: image-gen, model: /m.gguf, port: 9100, gpu_ids: [0]}\n", "belongs to image generation"},
		"negative wait":  {on + "  timeout: -1s\n", "timeout must be >= 0"},
		"typo'd setting": {on + "  stepz: 8\n", "stepz"},
	} {
		cfg, err := decodeConfig([]byte(base + tc.yaml))
		if err == nil {
			err = cfg.validate()
		}
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want an error about %q", name, err, tc.want)
		}
	}

	cfg, err := decodeConfig([]byte(base + on + "  gpu_ids: [3]\ninstances:\n  - {name: chat, model: /m.gguf, port: 9090, gpu_ids: [0]}\n"))
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
	}
	mgr := NewManager(cfg)
	var names []string
	for _, in := range mgr.Instances() {
		names = append(names, in.conf.Name)
	}
	if !slices.Equal(names, []string{"chat", imageGenName}) {
		t.Fatalf("managed instances: %v", names)
	}
	g := mgr.Get(imageGenName)
	if s := g.Status(); s.Role != roleImageGen || s.Model != "leejet/FLUX.2-klein-4B-GGUF:Q8_0" || s.Port != 9096 || !slices.Equal(s.GPUIDs, []int{3}) {
		t.Errorf("image-gen status: %+v", s)
	}
	if g.conf.AutoStart {
		t.Error("the manager must not boot image-gen itself: its files may still be downloading")
	}
	if err := cfg.AddInstance(InstanceConf{Name: "x", Model: "/m.gguf", Port: 9096, GPUIDs: []int{0}}); err == nil {
		t.Error("an instance may not take image-gen's port")
	}
}

func placeFile(t *testing.T, f imageFile, rev string, size int64) {
	t.Helper()
	p := filepath.Join(f.snapshotDir(rev), filepath.FromSlash(f.Path))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	fh, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	if err := fh.Truncate(size); err != nil {
		t.Fatal(err)
	}
	fh.Close()
}

func TestImageGenLaunchUsesTheCachedFiles(t *testing.T) {
	hub := t.TempDir()
	t.Setenv("HF_HUB_CACHE", hub)
	cfg := newTestConfig(t)
	cfg.GPUBackend = "rocm"
	cfg.Host = "0.0.0.0"
	cfg.ExtraArgs = []string{"--spec-type", "draft-mtp"}
	cfg.ImageGeneration = ImageGenConf{Enabled: true, ServerBin: "/opt/sd/sd-server", GPUIDs: []int{2}, ExtraArgs: []string{"--flash-attn", "off"}}
	ic := cfg.ImageGeneration.entries()[0].instanceConf()
	if _, err := buildLaunch(ic, cfg); !errors.Is(err, errImageFilesMissing) {
		t.Fatalf("launch without files: %v", err)
	}
	_, p, _ := presetByKey("")
	placeFile(t, p.Files[0], p.Files[0].Rev, p.Files[0].Size)
	placeFile(t, p.Files[1], p.Files[1].Rev, p.Files[1].Size)
	placeFile(t, p.Files[2], "main", p.Files[2].Size) // fetched from the Models page
	spec, err := buildLaunch(ic, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if spec.bin != "/opt/sd/sd-server" || spec.host != "127.0.0.1" {
		t.Errorf("bin %q host %q: sd-server must listen on loopback only", spec.bin, spec.host)
	}
	line := strings.Join(spec.args, " ")
	for _, want := range []string{
		"--listen-ip 127.0.0.1 --listen-port 9096",
		"--diffusion-model " + filepath.Join(hub, "models--leejet--FLUX.2-klein-4B-GGUF", "snapshots", p.Files[0].Rev, "flux-2-klein-4b-Q8_0.gguf"),
		"--vae " + filepath.Join(hub, "models--Comfy-Org--Flux-2-Klein-4B", "snapshots", p.Files[1].Rev, "split_files", "vae", "flux2-vae.safetensors"),
		"--llm " + filepath.Join(hub, "models--unsloth--Qwen3-4B-GGUF", "snapshots", "main", "Qwen3-4B-Q8_0.gguf"),
		"--steps 4 --cfg-scale 1 --sampling-method euler",
		"--eager-load",
	} {
		if !strings.Contains(line, want) {
			t.Errorf("args lack %q:\n%s", want, line)
		}
	}
	if !strings.HasSuffix(line, "--flash-attn off") || strings.Contains(line, "draft-mtp") {
		t.Errorf("image-gen must get its own extra_args, last: %s", line)
	}
	if !slices.Equal(spec.env, []string{"HIP_VISIBLE_DEVICES=2"}) {
		t.Errorf("env: %v", spec.env)
	}
	placeFile(t, p.Files[2], "main", p.Files[2].Size-1)
	if _, err := buildLaunch(ic, cfg); !errors.Is(err, errImageFilesMissing) {
		t.Errorf("a file of the wrong size counted as downloaded: %v", err)
	}
}

func sha(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// testPreset serves three small files from a fake HuggingFace and registers
// a preset for them; corrupt names a file whose hash won't match.
func testPreset(t *testing.T, corrupt string) (imagePreset, *int) {
	t.Helper()
	t.Setenv("HF_HUB_CACHE", t.TempDir())
	files := map[string][]byte{"o/diff/resolve/r1/d.gguf": []byte("diffusion"), "o/vae/resolve/r2/sub/v.safetensors": []byte("vae!"), "o/te/resolve/r3/t.gguf": []byte("text encoder")}
	var mu sync.Mutex
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, ok := files[strings.TrimPrefix(r.URL.Path, "/")]
		if !ok {
			http.NotFound(w, r)
			return
		}
		mu.Lock()
		hits++
		mu.Unlock()
		w.Write(b)
	}))
	t.Cleanup(srv.Close)
	old := hfResolveBase
	hfResolveBase = srv.URL
	t.Cleanup(func() { hfResolveBase = old })
	f := func(flag, repo, rev, p string) imageFile {
		b := files[repo+"/resolve/"+rev+"/"+p]
		sum := sha(b)
		if p == corrupt {
			sum = sha([]byte("something else"))
		}
		return imageFile{Flag: flag, Repo: repo, Rev: rev, Path: p, Size: int64(len(b)), SHA256: sum}
	}
	p := imagePreset{Label: "Test Model", Ref: "o/diff:Q8_0", Steps: 2, CFG: 1, Sampler: "euler", Files: []imageFile{
		f("--diffusion-model", "o/diff", "r1", "d.gguf"), f("--vae", "o/vae", "r2", "sub/v.safetensors"), f("--llm", "o/te", "r3", "t.gguf")}}
	imagePresets["test"] = p
	t.Cleanup(func() { delete(imagePresets, "test") })
	oldPoll := startingPollInterval
	startingPollInterval = 20 * time.Millisecond
	t.Cleanup(func() { startingPollInterval = oldPoll })
	return p, &hits
}

func TestImageGenDownloadsChecksAndStarts(t *testing.T) {
	p, hits := testPreset(t, "")
	cfg := newFakeRig(t, "healthy").cfg
	cfg.ImageGeneration = ImageGenConf{Enabled: true, ServerBin: fakeServerBin(t), Model: "test", Port: freePort(t)}
	mgr := NewManager(cfg)
	t.Cleanup(mgr.Shutdown)
	g := newImageGen(cfg.ImageGeneration, mgr)
	t.Cleanup(g.Close)
	g.Boot()
	inst := mgr.Get(imageGenName)
	waitFor(t, 10*time.Second, "image-gen to download and run", func() bool { return inst.State() == StateRunning })
	for _, f := range p.Files {
		local, ok := f.localPath()
		if !ok || !strings.Contains(local, filepath.Join("snapshots", f.Rev)) {
			t.Errorf("%s not in the hub cache at its commit: %q", f.Path, local)
		}
		if _, err := os.Stat(local + ".incomplete"); !os.IsNotExist(err) {
			t.Errorf("%s left its temp file behind", f.Path)
		}
	}
	if logs := strings.Join(inst.Logs(), "\n"); !strings.Contains(logs, "fetched d.gguf") || !strings.Contains(logs, "fetched t.gguf") {
		t.Errorf("the download isn't in image-gen's log:\n%s", logs)
	}
	if st := g.Status()[0]; st.State != StateRunning || st.Model != "Test Model" || st.Download != nil {
		t.Errorf("status: %+v", st)
	}
	if g.fetch(imageGenName) || *hits != 3 {
		t.Errorf("files already on disk were fetched again (%d requests)", *hits)
	}
}

func TestImageGenRejectsACorruptDownload(t *testing.T) {
	p, _ := testPreset(t, "sub/v.safetensors")
	cfg := newFakeRig(t, "healthy").cfg
	cfg.ImageGeneration = ImageGenConf{Enabled: true, ServerBin: fakeServerBin(t), Model: "test", Port: freePort(t)}
	mgr := NewManager(cfg)
	t.Cleanup(mgr.Shutdown)
	g := newImageGen(cfg.ImageGeneration, mgr)
	t.Cleanup(g.Close)
	g.Boot()
	waitFor(t, 5*time.Second, "the download to fail", func() bool { return strings.Contains(g.Status()[0].Error, "corrupt") })
	if st := g.Status()[0]; st.State != StateStopped || !strings.Contains(st.Error, "v.safetensors") {
		t.Errorf("status after a bad file: %+v", st)
	}
	if _, ok := p.Files[1].localPath(); ok {
		t.Error("a file with the wrong hash was kept")
	}
	if _, ok := p.Files[0].localPath(); !ok {
		t.Error("the good file before it should be kept")
	}
	if !g.fetch(imageGenName) {
		t.Error("starting image-gen again must retry the download")
	}
}

func TestImageGenHealthWaitsForTheModel(t *testing.T) {
	sd := newFakeSD()
	sd.state = "loading"
	up := httptest.NewServer(sd)
	t.Cleanup(up.Close)
	cfg := newTestConfig(t)
	inst := NewInstance(ImageGenConf{Enabled: true, Port: portOf(t, up.URL)}.entries()[0].instanceConf(), cfg)
	if inst.CheckHealth(context.Background()) {
		t.Error("a model still loading counted as healthy")
	}
	sd.mu.Lock()
	sd.state = "ready"
	sd.mu.Unlock()
	if !inst.CheckHealth(context.Background()) {
		t.Error("a loaded model counted as unhealthy")
	}
}

func TestImageGenCannotBeChangedOverTheAPI(t *testing.T) {
	cfg := newTestConfig(t)
	section := "image_generation:\n  enabled: true\n  server_bin: /opt/sd-server\n"
	head := "server_bin: /bin/true\nallowed_hosts: [example.com]\n"
	cfg.ImageGeneration = ImageGenConf{Enabled: true, ServerBin: "/opt/sd-server"}
	os.WriteFile(cfg.path, []byte(head+section), 0o640)
	ws := NewWebServer(NewManager(cfg), cfg, NewDownloadManager())
	t.Cleanup(ws.imageGen.Close)
	for _, m := range []string{http.MethodPut, http.MethodDelete} {
		rr := httptest.NewRecorder()
		ws.ServeHTTP(rr, httptest.NewRequest(m, "/api/config/instances/image-gen", strings.NewReader(`{"name":"image-gen","model":"evil/x","port":9096,"gpu_ids":[0]}`)))
		if rr.Code != http.StatusConflict {
			t.Errorf("%s image-gen: %d %s", m, rr.Code, rr.Body)
		}
	}
	if rr := postImport(t, ws, head+section+"instances:\n  - {name: a, model: /m.gguf, port: 9090, gpu_ids: [0]}\n"); rr.Code != http.StatusOK {
		t.Fatalf("import with the same section: %d %s", rr.Code, rr.Body)
	}
	if ws.mgr.Get(imageGenName) == nil || ws.mgr.Get("a") == nil {
		t.Error("an import must keep image-gen and add the new instance")
	}
	for name, in := range map[string]string{
		"other binary": head + "image_generation:\n  enabled: true\n  server_bin: /tmp/evil\n",
		"dropped":      head,
	} {
		if rr := postImport(t, ws, in); rr.Code != http.StatusForbidden {
			t.Errorf("%s: %d %s", name, rr.Code, rr.Body)
		}
	}
}

func TestChatMakesImages(t *testing.T) {
	ir := newImageRig(t)
	ir.sd.hold = make(chan struct{})
	c := ir.imageChat(t, map[string]any{"size": "768x512", "seed": 7})
	ir.send(t, c.ID, "a red fox in snow, watercolor", imgAtt("ref.png", tinyPNG()))
	ev, err := http.Get(ir.srv.URL + "/api/chats/" + c.ID + "/events")
	if err != nil {
		t.Fatal(err)
	}
	defer ev.Body.Close()
	close(ir.sd.hold)
	events := readSSE(t, ev, "done")
	var phases []string
	var preview string
	for _, e := range events {
		if e.name == "status" || e.name == "snapshot" {
			s, _ := e.data["text"].(string)
			if s == "" {
				s, _ = e.data["status"].(string)
			}
			phases = append(phases, s)
			if p, _ := e.data["preview"].(string); p != "" {
				preview = p
			}
		}
	}
	if !slices.Contains(phases, "Step 1 of 2") || !slices.Contains(phases, "Decoding the image") || !strings.HasPrefix(preview, "data:image/png;base64,") {
		t.Errorf("progress events: %q, preview %.40q", phases, preview)
	}
	done := events[len(events)-1]
	if done.name != "done" || done.data["status"] != variantDone || len(done.data["images"].([]any)) != 1 {
		t.Fatalf("done event: %+v", done)
	}

	v := ir.waitReply(t, c.ID)
	if len(v.Images) != 1 || v.Images[0].Mime != "image/png" || v.Images[0].Name != "flux2-klein-4b-7.png" || v.Content != "" {
		t.Fatalf("reply: %+v", v)
	}
	var st imageStats
	json.Unmarshal(v.Stats, &st)
	// An edit keeps the reference's shape (square here) at about the chosen
	// size's pixel count: 768x512 is 393216 pixels, so 624x624.
	if st.Width != 624 || st.Height != 624 || st.Seed != 7 || st.Steps != 4 || st.Refs != 1 || st.GenMs != 3000 {
		t.Errorf("stats: %+v", st)
	}
	bodies, _ := ir.sd.snapshot()
	b := bodies[0]
	sp := b["sample_params"].(map[string]any)
	refs, _ := b["ref_images"].([]any)
	if b["prompt"] != "a red fox in snow, watercolor" || b["width"] != 624.0 || b["height"] != 624.0 || b["seed"] != 7.0 ||
		sp["sample_steps"] != 4.0 || sp["sample_method"] != "euler" || sp["guidance"].(map[string]any)["txt_cfg"] != 1.0 ||
		len(refs) != 1 || !strings.HasPrefix(refs[0].(string), "data:image/png;base64,") || b["preview"] != "proj" {
		t.Errorf("sd-server got: %v", b)
	}

	resp, img := ir.do(t, "GET", "/api/attachments/"+v.Images[0].ID, nil)
	if resp.Header.Get("Content-Type") != "image/png" || !bytes.Equal(img, tinyPNG()) || !strings.Contains(resp.Header.Get("Content-Security-Policy"), "sandbox") {
		t.Errorf("image download: %s %q", resp.Header, img)
	}
	if d := ir.detail(t, c.ID); d.Chat.Title != "a red fox in snow, watercolor" {
		t.Errorf("an image chat keeps the title made from the prompt, got %q", d.Chat.Title)
	}

	d := ir.detail(t, c.ID)
	reply := d.Messages[1]
	if resp, b := ir.do(t, "POST", "/api/chats/"+c.ID+"/messages/"+reply.ID+"/regenerate", map[string]any{}); resp.StatusCode != http.StatusCreated {
		t.Fatalf("regenerate: %d %s", resp.StatusCode, b)
	}
	ir.waitReply(t, c.ID)
	reply = ir.detail(t, c.ID).Messages[1]
	if len(reply.Variants) != 2 || len(reply.Variants[0].Images) != 1 || len(reply.Variants[1].Images) != 1 ||
		reply.Variants[0].Images[0].ID == reply.Variants[1].Images[0].ID || len(d.Messages[0].Attachments) != 1 {
		t.Errorf("each version keeps its own image: %+v", reply.Variants)
	}

	// A chat model asked about it next sees a note in place of the picture.
	if resp, b := ir.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "write a caption", "instance": "c"}); resp.StatusCode != http.StatusCreated {
		t.Fatalf("send to the chat model: %d %s", resp.StatusCode, b)
	}
	ir.waitReply(t, c.ID)
	ir.chatAPIRig.mu.Lock()
	msgs := ir.bodies[len(ir.bodies)-1]["messages"].([]any)
	ir.chatAPIRig.mu.Unlock()
	if len(msgs) != 3 || msgs[1].(map[string]any)["content"] != imageReplyNote {
		t.Errorf("history sent to the chat model: %v", msgs)
	}
}

func TestChatImagePicksASeedPerVersion(t *testing.T) {
	ir := newImageRig(t)
	c := ir.imageChat(t, map[string]any{"size": "1344x768"})
	ir.send(t, c.ID, "a lighthouse")
	v := ir.waitReply(t, c.ID)
	var st imageStats
	json.Unmarshal(v.Stats, &st)
	bodies, _ := ir.sd.snapshot()
	if bodies[0]["seed"] != float64(st.Seed) || st.Seed < 0 || bodies[0]["width"] != 1344.0 || bodies[0]["height"] != 768.0 {
		t.Errorf("seed %v / stats %+v", bodies[0]["seed"], st)
	}
}

func TestChatImageStopCancelsTheJob(t *testing.T) {
	ir := newImageRig(t)
	ir.sd.outcome = "hang"
	c := ir.imageChat(t, map[string]any{})
	ir.send(t, c.ID, "a slow one")
	waitFor(t, 5*time.Second, "the job to start", func() bool {
		ir.sd.mu.Lock()
		defer ir.sd.mu.Unlock()
		return ir.sd.polls["job_1"] > 2
	})
	if resp, b := ir.do(t, "POST", "/api/chats/"+c.ID+"/stop", nil); resp.StatusCode != http.StatusOK {
		t.Fatalf("stop: %d %s", resp.StatusCode, b)
	}
	if v := ir.waitReply(t, c.ID); v.Status != variantStopped || len(v.Images) != 0 {
		t.Errorf("stopped reply: %+v", v)
	}
	if _, cancels := ir.sd.snapshot(); !slices.Equal(cancels, []string{"job_1"}) {
		t.Errorf("sd-server cancels: %v", cancels)
	}
}

func TestChatImageFailures(t *testing.T) {
	ir := newImageRig(t)
	ir.sd.outcome = "failed"
	c := ir.imageChat(t, map[string]any{})
	ir.send(t, c.ID, "too big")
	if v := ir.waitReply(t, c.ID); v.Status != variantError || !strings.Contains(v.Error, "out of memory") {
		t.Errorf("failed job: %+v", v)
	}
	ir.send(t, c.ID, "", imgAtt("a.png", tinyPNG()))
	if v := ir.waitReply(t, c.ID); v.Status != variantError || !strings.Contains(v.Error, "Describe the image") {
		t.Errorf("image without a prompt: %+v", v)
	}
	ir.send(t, c.ID, "use this", map[string]any{"kind": "file", "name": "notes.txt", "mime": "text/plain", "data": base64.StdEncoding.EncodeToString([]byte("hi"))})
	if v := ir.waitReply(t, c.ID); v.Status != variantError || !strings.Contains(v.Error, "can't read notes.txt") {
		t.Errorf("text file sent to image-gen: %+v", v)
	}
	if bodies, _ := ir.sd.snapshot(); len(bodies) != 1 {
		t.Errorf("requests that can't work must not reach sd-server: %d", len(bodies))
	}
}

func TestV1ImagesProxiesToImageGen(t *testing.T) {
	ir := newImageRig(t)
	resp, b := ir.do(t, "POST", "/v1/images/generations", map[string]any{"model": imageGenName, "prompt": "a cat", "size": "512x512"})
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(b), "b64_json") {
		t.Fatalf("generations: %d %s", resp.StatusCode, b)
	}
	ir.sd.mu.Lock()
	got := ir.sd.openai[0]
	ir.sd.mu.Unlock()
	if !reflect.DeepEqual(got, map[string]any{"prompt": "a cat", "size": "512x512"}) {
		t.Errorf("sd-server got %v", got)
	}
	if resp, _ := ir.do(t, "POST", "/v1/images/generations", map[string]any{"model": "c", "prompt": "x"}); resp.StatusCode != http.StatusNotFound {
		t.Errorf("a chat model as image model: %d", resp.StatusCode)
	}
	if resp, _ := ir.do(t, "POST", "/v1/chat/completions", map[string]any{"model": imageGenName, "messages": []any{}}); resp.StatusCode != http.StatusBadRequest {
		t.Errorf("chat completions sent to image-gen: %d", resp.StatusCode)
	}
	setState(ir.ws.mgr.Get(imageGenName), StateStopped)
	if resp, _ := ir.do(t, "POST", "/v1/images/generations", map[string]any{"prompt": "x"}); resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("stopped image-gen: %d", resp.StatusCode)
	}
}

func TestParseImageSize(t *testing.T) {
	for in, want := range map[string][2]int{
		"768x512": {768, 512}, " 1344X768 ": {1344, 768}, "": {1024, 1024}, "1000x1000": {1024, 1024},
		"128x128": {1024, 1024}, "4096x4096": {1024, 1024}, "big": {1024, 1024}, "512x": {1024, 1024},
	} {
		if w, h := parseImageSize(in, 16); w != want[0] || h != want[1] {
			t.Errorf("%q: %dx%d, want %v", in, w, h, want)
		}
	}
}

func TestChatStoreMigratesSchemaV2(t *testing.T) {
	path := filepath.Join(t.TempDir(), "chats.db")
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	v2 := chatSchema
	for _, cut := range [][2]string{
		{"DEFAULT '',   -- who read it", "DEFAULT ''    -- who read it"},
		{"\tvariant_id TEXT NOT NULL DEFAULT ''    -- set for a picture an image model made: the reply version it belongs to\n", ""},
		{"CREATE INDEX IF NOT EXISTS attachments_variant ON attachments(variant_id);\n", ""},
	} {
		if !strings.Contains(v2, cut[0]) {
			t.Fatalf("schema no longer has %q", cut[0])
		}
		v2 = strings.Replace(v2, cut[0], cut[1], 1)
	}
	for _, q := range []string{v2, `PRAGMA user_version = 2`,
		`INSERT INTO chats (id, title, created_at, updated_at) VALUES ('c1', 'old', 1, 1)`,
		`INSERT INTO messages (id, chat_id, seq, role, content, created_at) VALUES ('m1', 'c1', 0, 'user', 'a boat', 1)`,
		`INSERT INTO messages (id, chat_id, seq, role, instance, vi, created_at) VALUES ('m2', 'c1', 1, 'assistant', 'image-gen', 0, 1)`,
		`INSERT INTO variants (id, message_id, idx, status, created_at) VALUES ('v1', 'm2', 0, 'done', 1)`} {
		if _, err := db.Exec(q); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
	}
	db.Close()
	s, err := OpenChatStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if _, err := s.AddReplyImage(turn{ChatID: "c1", MessageID: "m2", VariantID: "v1"}, "boat.png", "image/png", tinyPNG()); err != nil {
		t.Fatal(err)
	}
	msgs, err := s.Messages("c1")
	if err != nil || len(msgs[1].Variants[0].Images) != 1 || len(msgs[1].Attachments) != 0 {
		t.Fatalf("messages after upgrade: %+v %v", msgs, err)
	}
	hist, _ := s.History("c1", 5)
	if len(hist) != 2 || hist[1].Content != imageReplyNote || len(hist[1].Attachments) != 0 {
		t.Errorf("history: %+v", hist)
	}
	var v int
	s.db.QueryRow(`PRAGMA user_version`).Scan(&v)
	if v != chatSchemaVersion {
		t.Errorf("user_version = %d", v)
	}
}

func TestImagePresetsAreWellFormed(t *testing.T) {
	hex40, hex64 := func(s string) bool { _, err := hex.DecodeString(s); return err == nil && len(s) == 40 }, func(s string) bool { _, err := hex.DecodeString(s); return err == nil && len(s) == 64 }
	for key, p := range imagePresets {
		if p.Label == "" || p.Ref == "" || p.License == "" || p.Steps <= 0 || p.Sampler == "" || (p.Align != 16 && p.Align != 32) || p.GPUs > 2 {
			t.Errorf("%s: incomplete preset %+v", key, p)
		}
		if p.Edit != editNone && p.Edit != editYes && p.Edit != editOnly {
			t.Errorf("%s: edit mode %q", key, p.Edit)
		}
		if len(p.Sigmas) > 0 && len(p.Sigmas)-1 != p.Steps {
			t.Errorf("%s: %d sigmas for %d steps", key, len(p.Sigmas), p.Steps)
		}
		if validateInstanceName(key) != nil {
			t.Errorf("%s can't be an instance name", key)
		}
		flags := map[string]int{}
		for _, f := range p.files() {
			if validateRepo(f.Repo) != nil || !hex40(f.Rev) || !hex64(f.SHA256) || f.Size <= 0 || f.Path == "" {
				t.Errorf("%s: bad file %+v", key, f)
			}
			flags[f.Flag]++
		}
		if flags["--diffusion-model"] != 1 || flags["--vae"] != 1 || flags["--llm"] != 1 {
			t.Errorf("%s: needs one diffusion model, VAE and text encoder: %v", key, flags)
		}
		if p.LoRA != nil && (p.LoRA.Flag != "" || p.loraName() == "" || strings.Contains(p.loraName(), ".safetensors")) {
			t.Errorf("%s: LoRA %+v named %q", key, p.LoRA, p.loraName())
		}
	}
}

func TestImageGenModelsConfig(t *testing.T) {
	base := "server_bin: /bin/true\n"
	head := "image_generation:\n  enabled: true\n  server_bin: /opt/sd-server\n"
	for name, tc := range map[string]struct{ yaml, want string }{
		"both forms":        {head + "  model: flux3-klein\n  models:\n    - model: z-image-turbo\n", "not both"},
		"unknown":           {head + "  models:\n    - model: sdxl\n", `model "sdxl" is unknown`},
		"same model twice":  {head + "  models:\n    - model: flux3-klein\n    - model: flux3-klein\n", "used twice"},
		"same port":         {head + "  models:\n    - {model: flux3-klein, port: 9300}\n    - {model: z-image-turbo, port: 9300}\n", "port 9300 is used twice"},
		"q8 on one gpu":     {head + "  models:\n    - {model: qwen-image-edit-2511-q8, gpu_ids: [1]}\n", "needs 2 GPUs"},
		"same gpu twice":    {head + "  models:\n    - {model: qwen-image-edit-2511-q8, gpu_ids: [1, 1]}\n", "GPU 1 twice"},
		"two gpus on metal": {"gpu_backend: metal\n" + head + "  models:\n    - {model: qwen-image-edit-2511-q8, gpu_ids: [1, 2]}\n", "two GPUs need gpu_backend"},
		"bad name":          {head + "  models:\n    - {model: flux3-klein, name: \"a/b\"}\n", "invalid instance name"},
		"instance port":     {head + "  models:\n    - model: flux3-klein\n    - model: z-image-turbo\ninstances:\n  - {name: a, model: /m.gguf, port: 9097, gpu_ids: [0]}\n", "used by image generation"},
		"instance name":     {head + "  models:\n    - model: flux3-klein\ninstances:\n  - {name: flux3-klein, model: /m.gguf, port: 9100, gpu_ids: [0]}\n", "belongs to image generation"},
	} {
		cfg, err := decodeConfig([]byte(base + tc.yaml))
		if err == nil {
			err = cfg.validate()
		}
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want an error about %q", name, err, tc.want)
		}
	}

	cfg, err := decodeConfig([]byte(base + head + "  timeout: 3m\n  models:\n    - {model: flux3-klein, gpu_ids: [4]}\n" +
		"    - {name: edit, model: qwen-image-edit-2511, gpu_ids: [3], timeout: 30m}\n    - {model: z-image-turbo, gpu_ids: [2], port: 9300}\n" +
		"    - {model: qwen-image-edit-2511-q8, gpu_ids: [0, 1]}\n"))
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, e := range cfg.ImageGeneration.entries() {
		got = append(got, fmt.Sprintf("%s:%d:%v:%s", e.Name, e.Port, e.GPUIDs, e.Wait))
	}
	want := []string{"flux3-klein:9096:[4]:3m0s", "edit:9097:[3]:30m0s", "z-image-turbo:9300:[2]:3m0s", "qwen-image-edit-2511-q8:9099:[0 1]:3m0s"}
	if !slices.Equal(got, want) {
		t.Errorf("entries:\n got %v\nwant %v", got, want)
	}
	mgr := NewManager(cfg)
	for _, n := range []string{"flux3-klein", "edit", "z-image-turbo", "qwen-image-edit-2511-q8"} {
		if inst := mgr.Get(n); inst == nil || inst.Status().Role != roleImageGen {
			t.Errorf("%s isn't a managed image model", n)
		}
	}
	if mgr.Get(imageGenName) != nil {
		t.Error("the list form has no image-gen instance")
	}
}

func placePreset(t *testing.T, key string) imagePreset {
	t.Helper()
	p := imagePresets[key]
	for _, f := range p.files() {
		placeFile(t, f, f.Rev, f.Size)
	}
	return p
}

func TestImageGenLaunchPerModel(t *testing.T) {
	hub := t.TempDir()
	t.Setenv("HF_HUB_CACHE", hub)
	cfg := newTestConfig(t)
	cfg.ManagerPort = 8080
	cfg.GPUBackend = "rocm"
	cfg.ImageGeneration = ImageGenConf{Enabled: true, ServerBin: "/opt/sd-server", Models: []ImageModelConf{
		{Model: "qwen-image-edit-2511-q8", GPUIDs: []int{1, 2}},
		{Name: "turbo", Model: "qwen-image-2.1-turbo", GPUIDs: []int{3}, Port: 9200},
	}}
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
	}
	edit := placePreset(t, "qwen-image-edit-2511-q8")
	placePreset(t, "qwen-image-2.1-turbo")
	es := cfg.ImageGeneration.entries()

	spec, err := buildLaunch(es[0].instanceConf(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	line := strings.Join(spec.args, " ")
	for _, want := range []string{
		"--listen-port 9096",
		"--llm_vision " + filepath.Join(hub, "models--mradermacher--Qwen2.5-VL-7B-Instruct-GGUF", "snapshots", fileQwen25VLVision.Rev, "Qwen2.5-VL-7B-Instruct.mmproj-f16.gguf"),
		"--lora-model-dir " + filepath.Join(hub, "models--lightx2v--Qwen-Image-Edit-2511-Lightning", "snapshots", edit.LoRA.Rev),
		"--model-args qwen_image_zero_cond_t=true",
		"--steps 4 --cfg-scale 1 --sampling-method euler",
		"--flow-shift 3",
		"--backend all=rocm0,te=rocm1",
	} {
		if !strings.Contains(line, want) {
			t.Errorf("edit args lack %q:\n%s", want, line)
		}
	}
	if !slices.Equal(spec.env, []string{"HIP_VISIBLE_DEVICES=1,2"}) {
		t.Errorf("edit env: %v", spec.env)
	}

	spec, err = buildLaunch(es[1].instanceConf(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	line = strings.Join(spec.args, " ")
	if !strings.Contains(line, "--listen-port 9200") || !strings.Contains(line, "--sigmas 1,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0") ||
		strings.Contains(line, "--backend") || strings.Contains(line, "--lora-model-dir") || strings.Contains(line, "--flow-shift") {
		t.Errorf("turbo args: %s", line)
	}

	cfg.GPUBackend = "vulkan"
	spec, _ = buildLaunch(es[0].instanceConf(), cfg)
	if line := strings.Join(spec.args, " "); !strings.Contains(line, "--backend all=vulkan0,te=vulkan1") || !slices.Equal(spec.env, []string{"GGML_VK_VISIBLE_DEVICES=1,2"}) {
		t.Errorf("vulkan: %s %v", line, spec.env)
	}
}

func widePNG(w, h int) []byte {
	var b bytes.Buffer
	_ = png.Encode(&b, image.NewRGBA(image.Rect(0, 0, w, h)))
	return b.Bytes()
}

func TestChatImageModelsFollowTheirPresets(t *testing.T) {
	ir := newImageRig(t, ImageModelConf{Model: "z-image-turbo"}, ImageModelConf{Model: "qwen-image-edit-2511"}, ImageModelConf{Model: "qwen-image-2.1-turbo"})

	z := ir.chatWith(t, "z-image-turbo", map[string]any{})
	ir.send(t, z.ID, "make it blue", imgAtt("a.png", tinyPNG()))
	if v := ir.waitReply(t, z.ID); v.Status != variantError || !strings.Contains(v.Error, "can't edit") {
		t.Errorf("Z-Image with a picture: %+v", v)
	}
	ir.send(t, z.ID, "a blue door")
	if v := ir.waitReply(t, z.ID); v.Status != variantDone || v.Images[0].Name != "z-image-turbo-"+strings.TrimPrefix(v.Images[0].Name, "z-image-turbo-") {
		t.Errorf("Z-Image text to image: %+v", v)
	}
	zb, _ := ir.sds["z-image-turbo"].snapshot()
	if len(zb) != 1 || zb[0]["sample_params"].(map[string]any)["sample_steps"] != 8.0 || zb[0]["ref_images"] != nil || zb[0]["lora"] != nil {
		t.Errorf("Z-Image request: %v", zb)
	}

	q := ir.chatWith(t, "qwen-image-edit-2511", map[string]any{})
	ir.send(t, q.ID, "a cat")
	if v := ir.waitReply(t, q.ID); v.Status != variantError || !strings.Contains(v.Error, "edits images") {
		t.Errorf("Qwen edit without a picture: %+v", v)
	}
	ir.send(t, q.ID, "make it night", imgAtt("wide.png", widePNG(200, 100)))
	if v := ir.waitReply(t, q.ID); v.Status != variantDone {
		t.Fatalf("Qwen edit: %+v", v)
	}
	qb, _ := ir.sds["qwen-image-edit-2511"].snapshot()
	b := qb[0]
	sp := b["sample_params"].(map[string]any)
	if !reflect.DeepEqual(b["lora"], []any{map[string]any{"path": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16", "multiplier": 1.0}}) ||
		sp["flow_shift"] != 3.0 || sp["sample_steps"] != 4.0 || sp["guidance"].(map[string]any)["txt_cfg"] != 1.0 {
		t.Errorf("Qwen edit request: %v", b)
	}
	// A 2:1 picture edited at the default 1024x1024 worth of pixels.
	if b["width"] != 1456.0 || b["height"] != 720.0 {
		t.Errorf("edit size %vx%v, want 1456x720", b["width"], b["height"])
	}

	tq := ir.chatWith(t, "qwen-image-2.1-turbo", map[string]any{"size": "1152x896"})
	ir.send(t, tq.ID, "a poster that says hello")
	if v := ir.waitReply(t, tq.ID); v.Status != variantDone {
		t.Fatalf("Qwen turbo: %+v", v)
	}
	tb, _ := ir.sds["qwen-image-2.1-turbo"].snapshot()
	if s, _ := tb[0]["sample_params"].(map[string]any)["custom_sigmas"].([]any); len(s) != 9 || tb[0]["width"] != 1152.0 || tb[0]["lora"] != nil {
		t.Errorf("Qwen turbo request: %v", tb[0])
	}
}

func TestFitImageSize(t *testing.T) {
	for _, tc := range []struct{ w, h, area, align, ww, wh int }{
		{200, 100, 1024 * 1024, 16, 1456, 720},
		{100, 200, 1024 * 1024, 32, 736, 1440},
		{640, 480, 768 * 768, 16, 880, 672},
		{4000, 100, 1024 * 1024, 16, 2048, 256}, // clamped to 256..2048
	} {
		w, h, ok := fitImageSize(widePNG(tc.w, tc.h), tc.area, tc.align)
		if !ok || w != tc.ww || h != tc.wh {
			t.Errorf("%dx%d at %d px: %dx%d, want %dx%d", tc.w, tc.h, tc.area, w, h, tc.ww, tc.wh)
		}
	}
	if _, _, ok := fitImageSize([]byte("not an image"), 1<<20, 16); ok {
		t.Error("an unreadable picture must keep the chosen size")
	}
}

func TestV1ImagesPicksTheModel(t *testing.T) {
	ir := newImageRig(t, ImageModelConf{Model: "flux3-klein"}, ImageModelConf{Name: "fast", Model: "z-image-turbo"})
	for model, want := range map[string]string{"": "flux3-klein", "fast": "fast", "z-image-turbo": "fast", "flux3-klein": "flux3-klein"} {
		before := map[string]int{}
		for n, sd := range ir.sds {
			sd.mu.Lock()
			before[n] = len(sd.openai)
			sd.mu.Unlock()
		}
		if resp, b := ir.do(t, "POST", "/v1/images/generations", map[string]any{"model": model, "prompt": "x"}); resp.StatusCode != http.StatusOK {
			t.Fatalf("model %q: %d %s", model, resp.StatusCode, b)
		}
		sd := ir.sds[want]
		sd.mu.Lock()
		n := len(sd.openai)
		sd.mu.Unlock()
		if n != before[want]+1 {
			t.Errorf("model %q didn't reach %s", model, want)
		}
	}
	if resp, b := ir.do(t, "POST", "/v1/images/generations", map[string]any{"model": "sdxl", "prompt": "x"}); resp.StatusCode != http.StatusNotFound || !strings.Contains(string(b), "flux3-klein, fast") {
		t.Errorf("unknown model: %d %s", resp.StatusCode, b)
	}
}

func TestImageGenSharedFileDownloadsOnce(t *testing.T) {
	t.Setenv("HF_HUB_CACHE", t.TempDir())
	files := map[string][]byte{"o/a/resolve/r1/a.gguf": []byte("model a"), "o/b/resolve/r2/b.gguf": []byte("model b"),
		"o/vae/resolve/r3/vae.safetensors": []byte("shared vae"), "o/te/resolve/r4/te.gguf": []byte("shared te")}
	var mu sync.Mutex
	hits := map[string]int{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, ok := files[strings.TrimPrefix(r.URL.Path, "/")]
		if !ok {
			http.NotFound(w, r)
			return
		}
		mu.Lock()
		hits[r.URL.Path]++
		mu.Unlock()
		w.Write(b)
	}))
	t.Cleanup(srv.Close)
	old := hfResolveBase
	hfResolveBase = srv.URL
	t.Cleanup(func() { hfResolveBase = old })
	oldPoll := startingPollInterval
	startingPollInterval = 20 * time.Millisecond
	t.Cleanup(func() { startingPollInterval = oldPoll })
	f := func(flag, repo, rev, p string) imageFile {
		b := files[repo+"/resolve/"+rev+"/"+p]
		return imageFile{Flag: flag, Repo: repo, Rev: rev, Path: p, Size: int64(len(b)), SHA256: sha(b)}
	}
	vae, te := f("--vae", "o/vae", "r3", "vae.safetensors"), f("--llm", "o/te", "r4", "te.gguf")
	for key, model := range map[string]imageFile{"test-a": f("--diffusion-model", "o/a", "r1", "a.gguf"), "test-b": f("--diffusion-model", "o/b", "r2", "b.gguf")} {
		imagePresets[key] = imagePreset{Label: key, Ref: key, License: "test", Files: []imageFile{model, vae, te}, Steps: 2, CFG: 1, Sampler: "euler", Align: 16, Edit: editYes}
		t.Cleanup(func() { delete(imagePresets, key) })
	}
	cfg := newFakeRig(t, "healthy").cfg
	cfg.ImageGeneration = ImageGenConf{Enabled: true, ServerBin: fakeServerBin(t), Models: []ImageModelConf{
		{Model: "test-a", Port: freePort(t)}, {Model: "test-b", Port: freePort(t)}}}
	mgr := NewManager(cfg)
	t.Cleanup(mgr.Shutdown)
	g := newImageGen(cfg.ImageGeneration, mgr)
	t.Cleanup(g.Close)
	g.Boot()
	waitFor(t, 10*time.Second, "both image models to run", func() bool {
		return mgr.Get("test-a").State() == StateRunning && mgr.Get("test-b").State() == StateRunning
	})
	mu.Lock()
	defer mu.Unlock()
	for p, n := range hits {
		if n != 1 {
			t.Errorf("%s fetched %d times", p, n)
		}
	}
	if len(hits) != 4 {
		t.Errorf("fetched %v", hits)
	}
}
