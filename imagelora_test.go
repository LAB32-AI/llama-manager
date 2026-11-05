package main

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestImageLoRAsAreWellFormed(t *testing.T) {
	hex40, hex64 := func(s string) bool { _, err := hex.DecodeString(s); return err == nil && len(s) == 40 }, func(s string) bool { _, err := hex.DecodeString(s); return err == nil && len(s) == 64 }
	used := map[string]bool{}
	for _, p := range imagePresets {
		used[p.LoRA] = true
	}
	for key, l := range imageLoRAs {
		f := l.File
		if validateRepo(f.Repo) != nil || !hex40(f.Rev) || !hex64(f.SHA256) || f.Size <= 0 || f.Flag != "" || path.Ext(f.Path) != ".safetensors" {
			t.Errorf("%s: bad file %+v", key, f)
		}
		if l.Label == "" || l.License == "" || l.Weight <= 0 || l.Weight > maxLoRAWeight || len(l.Fits) == 0 {
			t.Errorf("%s: incomplete %+v", key, l)
		}
		for _, p := range l.Fits {
			if _, ok := imagePresets[p]; !ok {
				t.Errorf("%s fits an unknown preset %q", key, p)
			}
		}
		if l.Fixed != used[key] {
			t.Errorf("%s: fixed is %v, but a preset uses it: %v", key, l.Fixed, used[key])
		}
		if validateInstanceName(key) != nil || strings.ContainsAny(key, " /") {
			t.Errorf("%s can't name a file", key)
		}
	}
	if got := loraNamesFor("z-image-turbo"); !slices.Equal(got, []string{"z-classic-painting", "z-pixel-art", "z-realism"}) {
		t.Errorf("z-image-turbo's LoRAs: %v", got)
	}
	if got := loraNamesFor("qwen-image-edit-2511"); slices.Contains(got, lightningLoRA) || len(got) != 3 {
		t.Errorf("a preset's own LoRA isn't one to pick: %v", got)
	}
}

func TestImageLoRAConfigRules(t *testing.T) {
	base := "server_bin: /bin/true\n"
	head := "image_generation:\n  enabled: true\n  server_bin: /opt/sd-server\n"
	for name, tc := range map[string]struct{ yaml, want string }{
		"unknown":         {head + "  model: z-image-turbo\n  loras: [sdxl-style]\n", `"sdxl-style" isn't a built-in LoRA for z-image-turbo; use z-classic-painting, z-pixel-art, z-realism`},
		"other model's":   {head + "  model: z-image-turbo\n  loras: [klein-anime]\n", `"klein-anime" isn't a built-in LoRA`},
		"preset's own":    {head + "  models:\n    - {model: qwen-image-edit-2511, loras: [qwen-edit-lightning]}\n", `"qwen-edit-lightning" isn't a built-in LoRA`},
		"none for model":  {head + "  model: flux3-klein\n  loras: [z-realism]\n", "flux3-klein has no built-in LoRAs; put your own in lora_dir"},
		"twice":           {head + "  model: z-image-turbo\n  loras: [z-realism, z-realism]\n", "listed twice"},
		"relative folder": {head + "  model: z-image-turbo\n  lora_dir: loras\n", "lora_dir must be an absolute path"},
		"both forms":      {head + "  lora_dir: /srv/loras\n  models:\n    - model: z-image-turbo\n", "not both"},
	} {
		cfg, err := decodeConfig([]byte(base + tc.yaml))
		if err == nil {
			err = cfg.validate()
		}
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want an error about %q", name, err, tc.want)
		}
	}
	cfg, err := decodeConfig([]byte(base + head + "  models:\n    - {model: z-image-turbo, lora_dir: /srv/loras/z, loras: [z-pixel-art, z-realism]}\n" +
		"    - {model: flux3-klein, lora_dir: /srv/loras/flux3}\n    - model: flux2-klein-4b\n"))
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
	}
	es := cfg.ImageGeneration.entries()
	if es[0].loraRoot() != "/srv/loras/z" || !slices.Equal(es[0].builtinLoRAs(), []string{"z-pixel-art", "z-realism"}) || es[1].loraRoot() != "/srv/loras/flux3" || es[2].loraRoot() != "" {
		t.Errorf("entries: %+v", es)
	}
	if id := es[0].loraID("z-realism"); id != "llama-manager/z-image-turbo/z-realism.safetensors" {
		t.Errorf("id %q", id)
	}
}

// loraRig is an image model with built-in LoRAs in a temporary hub cache and
// a LoRA folder of its own.
func loraRig(t *testing.T, model string, builtins ...string) (imageEntry, string) {
	t.Helper()
	hub := filepath.Join(t.TempDir(), "hub")
	t.Setenv("HF_HUB_CACHE", hub)
	dir := filepath.Join(t.TempDir(), "loras")
	c := ImageGenConf{Enabled: true, ServerBin: "/bin/true", Models: []ImageModelConf{{Model: model, LoraDir: dir, LoRAs: builtins}}}
	if err := c.check(); err != nil {
		t.Fatal(err)
	}
	return c.entries()[0], dir
}

func TestImageLoRAFolder(t *testing.T) {
	e, dir := loraRig(t, "z-image-turbo", "z-pixel-art", "z-realism")
	if e.loraReady() {
		t.Fatal("ready before anything is in place")
	}
	if err := e.stageLoRAs(func(string) {}); !errors.Is(err, errImageFilesMissing) {
		t.Fatalf("staging without the downloads: %v", err)
	}
	pixel, real := imageLoRAs["z-pixel-art"].File, imageLoRAs["z-realism"].File
	placeFile(t, pixel, pixel.Rev, pixel.Size)
	// Fetched by the hf CLI: the snapshot file is a symlink into blobs/,
	// which sd-server would refuse to follow out of the LoRA folder.
	blob := filepath.Join(filepath.Dir(real.snapshotDir("x")), "..", "blobs", real.SHA256)
	placeAt(t, blob, real.Size)
	snap := filepath.Join(real.snapshotDir(real.Rev), real.Path)
	if err := os.MkdirAll(filepath.Dir(snap), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(blob, snap); err != nil {
		t.Fatal(err)
	}
	var logs []string
	if err := e.stageLoRAs(func(s string) { logs = append(logs, s) }); err != nil {
		t.Fatal(err)
	}
	if !e.loraReady() || len(logs) != 0 {
		t.Fatalf("ready %v, logs %v", e.loraReady(), logs)
	}
	for k, src := range map[string]string{"z-pixel-art": filepath.Join(pixel.snapshotDir(pixel.Rev), pixel.Path), "z-realism": blob} {
		dst := e.loraPath(e.loraID(k))
		a, err1 := os.Lstat(dst)
		b, err2 := os.Stat(src)
		if err1 != nil || err2 != nil || !a.Mode().IsRegular() || !os.SameFile(a, b) {
			t.Errorf("%s isn't a hard link to the hub cache's file: %v %v", k, err1, err2)
		}
	}

	// The user's own files, as sd-server would list them.
	writeFile(t, filepath.Join(dir, "mine.safetensors"), "x")
	writeFile(t, filepath.Join(dir, "styles", "Ink.SAFETENSORS"), "x")
	writeFile(t, filepath.Join(dir, "a", "b", "c", "too-deep.safetensors"), "x")
	writeFile(t, filepath.Join(dir, ".hidden.safetensors"), "x")
	writeFile(t, filepath.Join(dir, "notes.txt"), "x")
	writeFile(t, filepath.Join(dir, "q.gguf"), "x")
	writeFile(t, filepath.Join(dir, loraSubdir, "other-model", "theirs.safetensors"), "x")
	outside := filepath.Join(t.TempDir(), "outside.safetensors")
	writeFile(t, outside, "x")
	os.Symlink(outside, filepath.Join(dir, "escape.safetensors"))
	os.Symlink(filepath.Join(dir, "mine.safetensors"), filepath.Join(dir, "alias.safetensors"))
	var got []string
	for _, c := range e.loraChoices() {
		got = append(got, c.ID+"="+c.Name)
	}
	want := []string{
		"llama-manager/z-image-turbo/z-pixel-art.safetensors=Pixel art", "llama-manager/z-image-turbo/z-realism.safetensors=Realism",
		"alias.safetensors=alias", "mine.safetensors=mine", "q.gguf=q", "styles/Ink.SAFETENSORS=styles/Ink",
	}
	if !slices.Equal(got, want) {
		t.Errorf("choices:\n got %v\nwant %v", got, want)
	}

	// A LoRA dropped from config leaves the folder; a copy stands in where a
	// hard link can't be made (another disk).
	e.LoRAs = []string{"z-realism"}
	old := linkFile
	linkFile = func(string, string) error { return errors.New("cross-device link") }
	t.Cleanup(func() { linkFile = old })
	os.Remove(e.loraPath(e.loraID("z-realism")))
	logs = nil
	if err := e.stageLoRAs(func(s string) { logs = append(logs, s) }); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(e.loraPath("llama-manager/z-image-turbo/z-pixel-art.safetensors")); !os.IsNotExist(err) {
		t.Error("a LoRA no longer in config stayed in the folder")
	}
	a, _ := os.Lstat(e.loraPath(e.loraID("z-realism")))
	b, _ := os.Stat(blob)
	if a == nil || !a.Mode().IsRegular() || a.Size() != real.Size || os.SameFile(a, b) || len(logs) != 1 || !strings.Contains(logs[0], "copying") {
		t.Errorf("copy: %v %v", a, logs)
	}
	if _, err := os.Stat(filepath.Join(dir, loraSubdir, "other-model", "theirs.safetensors")); err != nil {
		t.Error("another model's LoRAs must be left alone")
	}
}

func placeAt(t *testing.T, p string, size int64) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	fh, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	fh.Truncate(size)
	fh.Close()
}

func TestImageLoRALaunch(t *testing.T) {
	e, dir := loraRig(t, "z-image-turbo", "z-pixel-art")
	cfg := newTestConfig(t)
	cfg.GPUBackend = "rocm"
	cfg.ImageGeneration = ImageGenConf{Enabled: true, ServerBin: "/opt/sd-server", Models: []ImageModelConf{{Model: "z-image-turbo", LoraDir: dir, LoRAs: []string{"z-pixel-art"}}}}
	placePreset(t, "z-image-turbo")
	if _, err := buildLaunch(e.instanceConf(), cfg); !errors.Is(err, errImageFilesMissing) {
		t.Fatalf("launch without the LoRA: %v", err)
	}
	f := imageLoRAs["z-pixel-art"].File
	placeFile(t, f, f.Rev, f.Size)
	if _, err := buildLaunch(e.instanceConf(), cfg); !errors.Is(err, errImageFilesMissing) {
		t.Fatalf("launch before the folder is staged: %v", err)
	}
	if err := e.stageLoRAs(func(string) {}); err != nil {
		t.Fatal(err)
	}
	spec, err := buildLaunch(e.instanceConf(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if line := strings.Join(spec.args, " "); !strings.Contains(line, "--lora-model-dir "+dir) {
		t.Errorf("args: %s", line)
	}
}

// fakeHF serves files by "repo/resolve/rev/path" in place of HuggingFace.
func fakeHF(t *testing.T, files map[string][]byte) func(flag, repo, rev, p string) imageFile {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if b, ok := files[strings.TrimPrefix(r.URL.Path, "/")]; ok {
			w.Write(b)
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)
	old := hfResolveBase
	hfResolveBase = srv.URL
	t.Cleanup(func() { hfResolveBase = old })
	return func(flag, repo, rev, p string) imageFile {
		b := files[repo+"/resolve/"+rev+"/"+p]
		return imageFile{Flag: flag, Repo: repo, Rev: rev, Path: p, Size: int64(len(b)), SHA256: sha(b)}
	}
}

func TestImageGenDownloadsAndStagesLoRAs(t *testing.T) {
	t.Setenv("HF_HUB_CACHE", filepath.Join(t.TempDir(), "hub"))
	f := fakeHF(t, map[string][]byte{"o/d/resolve/r1/d.gguf": []byte("diffusion"), "o/v/resolve/r2/v.safetensors": []byte("vae"),
		"o/t/resolve/r3/t.gguf": []byte("text encoder"), "o/lora/resolve/r9/style.safetensors": []byte("a style lora")})
	imagePresets["test"] = imagePreset{Label: "Test", Ref: "o/d:Q8_0", License: "test", Steps: 2, CFG: 1, Sampler: "euler", Align: 16, Edit: editYes,
		Files: []imageFile{f("--diffusion-model", "o/d", "r1", "d.gguf"), f("--vae", "o/v", "r2", "v.safetensors"), f("--llm", "o/t", "r3", "t.gguf")}}
	imageLoRAs["test-style"] = imageLoRA{Label: "Test style", Fits: []string{"test"}, Weight: 0.8, License: "test", File: f("", "o/lora", "r9", "style.safetensors")}
	t.Cleanup(func() { delete(imagePresets, "test"); delete(imageLoRAs, "test-style") })
	oldPoll := startingPollInterval
	startingPollInterval = 20 * time.Millisecond
	t.Cleanup(func() { startingPollInterval = oldPoll })
	dir := filepath.Join(t.TempDir(), "loras")
	cfg := newFakeRig(t, "healthy").cfg
	cfg.ImageGeneration = ImageGenConf{Enabled: true, ServerBin: fakeServerBin(t), Model: "test", Port: freePort(t), LoraDir: dir, LoRAs: []string{"test-style"}}
	mgr := NewManager(cfg)
	t.Cleanup(mgr.Shutdown)
	g := newImageGen(cfg.ImageGeneration, mgr)
	t.Cleanup(g.Close)
	g.Boot()
	inst := mgr.Get(imageGenName)
	waitFor(t, 10*time.Second, "image-gen to download, stage and run", func() bool { return inst.State() == StateRunning })
	if got, err := os.ReadFile(filepath.Join(dir, loraSubdir, imageGenName, "test-style.safetensors")); err != nil || string(got) != "a style lora" {
		t.Errorf("staged LoRA: %q %v", got, err)
	}
	if logs := strings.Join(inst.Logs(), "\n"); !strings.Contains(logs, "fetched style.safetensors") || !strings.Contains(logs, "LoRA folder "+dir+" is ready") {
		t.Errorf("log:\n%s", logs)
	}
	st := g.Status()[0]
	if st.LoraDir != dir || len(st.LoRAs) != 1 || st.LoRAs[0].Name != "Test style" || st.LoRAs[0].Weight != 0.8 {
		t.Errorf("status: %+v", st)
	}
	if g.fetch(imageGenName) {
		t.Error("a ready model must start without another download")
	}
}

func TestChatImageLoRAs(t *testing.T) {
	ir := newImageRig(t, ImageModelConf{Model: "z-image-turbo", LoraDir: filepath.Join(t.TempDir(), "loras"), LoRAs: []string{"z-pixel-art"}},
		ImageModelConf{Model: "qwen-image-edit-2511", LoRAs: []string{"qie-object-remover"}})
	e, _ := ir.ws.imageGen.model("z-image-turbo")
	writeFile(t, e.loraPath("mine.safetensors"), "x")
	pixel := "llama-manager/z-image-turbo/z-pixel-art.safetensors"

	_, b := ir.do(t, "GET", "/api/status", nil)
	var status struct {
		ImageModels []imageModelStatus `json:"image_models"`
	}
	json.Unmarshal(b, &status)
	if len(status.ImageModels) != 2 || len(status.ImageModels[0].LoRAs) != 2 || status.ImageModels[0].LoRAs[1].ID != "mine.safetensors" ||
		status.ImageModels[0].LoraDir == "" || status.ImageModels[1].LoraDir != "" || status.ImageModels[1].LoraFixed != "Lightning (4 steps)" {
		t.Fatalf("status: %s", b)
	}

	c := ir.chatWith(t, "z-image-turbo", map[string]any{"loras": map[string]any{
		"z-image-turbo":        []any{map[string]any{"id": pixel, "w": 0.8}, map[string]any{"id": "mine.safetensors", "w": 1.25}, map[string]any{"id": "mine.safetensors", "w": 0}},
		"qwen-image-edit-2511": []any{map[string]any{"id": "not for this model"}},
	}})
	ir.send(t, c.ID, "Pixel art style. a castle")
	v := ir.waitReply(t, c.ID)
	if v.Status != variantDone {
		t.Fatalf("reply: %+v", v)
	}
	var st imageStats
	json.Unmarshal(v.Stats, &st)
	if !slices.Equal(st.LoRAs, []string{"Pixel art 0.8", "mine 1.25"}) {
		t.Errorf("stats: %+v", st)
	}
	zb, _ := ir.sds["z-image-turbo"].snapshot()
	want := []any{map[string]any{"path": pixel, "multiplier": 0.8}, map[string]any{"path": "mine.safetensors", "multiplier": 1.25}}
	if !reflect.DeepEqual(zb[0]["lora"], want) {
		t.Errorf("lora: %v", zb[0]["lora"])
	}

	// The edit model keeps its Lightning LoRA first and adds the picked one.
	q := ir.chatWith(t, "qwen-image-edit-2511", map[string]any{"loras": map[string]any{
		"qwen-image-edit-2511": []any{map[string]any{"id": "llama-manager/qwen-image-edit-2511/qie-object-remover.safetensors", "w": 1}}}})
	ir.send(t, q.ID, "Remove the lamp", imgAtt("a.png", tinyPNG()))
	if v := ir.waitReply(t, q.ID); v.Status != variantDone {
		t.Fatalf("edit: %+v", v)
	}
	qb, _ := ir.sds["qwen-image-edit-2511"].snapshot()
	if l, _ := qb[0]["lora"].([]any); len(l) != 2 || l[0].(map[string]any)["path"] != "llama-manager/qwen-image-edit-2511/qwen-edit-lightning.safetensors" {
		t.Errorf("edit lora: %v", qb[0]["lora"])
	}

	for name, tc := range map[string]struct {
		picks []any
		want  string
	}{
		"gone":     {[]any{map[string]any{"id": "deleted.safetensors", "w": 1}}, `has no LoRA "deleted.safetensors"`},
		"too much": {[]any{map[string]any{"id": pixel, "w": 2.5}}, "between 0 and 2"},
		"too many": {[]any{map[string]any{"id": pixel}, map[string]any{"id": pixel}, map[string]any{"id": pixel}, map[string]any{"id": pixel}, map[string]any{"id": pixel}}, "at most 4 LoRAs"},
	} {
		c := ir.chatWith(t, "z-image-turbo", map[string]any{"loras": map[string]any{"z-image-turbo": tc.picks}})
		ir.send(t, c.ID, "a castle")
		if v := ir.waitReply(t, c.ID); v.Status != variantError || !strings.Contains(v.Error, tc.want) {
			t.Errorf("%s: %+v", name, v)
		}
	}
}

func TestV1ImagesLoRAs(t *testing.T) {
	ir := newImageRig(t, ImageModelConf{Model: "z-image-turbo", LoraDir: filepath.Join(t.TempDir(), "loras"), LoRAs: []string{"z-pixel-art"}},
		ImageModelConf{Model: "qwen-image-edit-2511"})
	z, _ := ir.ws.imageGen.model("z-image-turbo")
	q, _ := ir.ws.imageGen.model("qwen-image-edit-2511")
	writeFile(t, z.loraPath("mine.safetensors"), "x")
	last := func(name string) map[string]any {
		sd := ir.sds[name]
		sd.mu.Lock()
		defer sd.mu.Unlock()
		return sd.openai[len(sd.openai)-1]
	}
	extra := func(prompt string) (string, map[string]any) {
		i := strings.Index(prompt, extraArgsOpen)
		if i < 0 {
			return prompt, nil
		}
		var m map[string]any
		json.Unmarshal([]byte(strings.TrimSuffix(prompt[i+len(extraArgsOpen):], extraArgsClose)), &m)
		return prompt[:i], m
	}

	resp, b := ir.do(t, "POST", "/v1/images/generations", map[string]any{"model": "z-image-turbo",
		"prompt": `a castle <sd_cpp_extra_args>{"sample_params":{"sample_steps":6},"lora":[{"path":"mine.safetensors","multiplier":0.5}]}</sd_cpp_extra_args>`,
		"lora":   []any{map[string]any{"path": "z-pixel-art", "multiplier": 0.7}}})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	got := last("z-image-turbo")
	text, args := extra(got["prompt"].(string))
	want := []any{map[string]any{"path": z.loraPath(z.loraID("z-pixel-art")), "multiplier": 0.7}, map[string]any{"path": z.loraPath("mine.safetensors"), "multiplier": 0.5}}
	if text != "a castle " || got["lora"] != nil || !reflect.DeepEqual(args["lora"], want) || args["sample_params"] == nil {
		t.Errorf("forwarded %q, args %v", got["prompt"], args)
	}

	if resp, b := ir.do(t, "POST", "/v1/images/generations", map[string]any{"model": "qwen-image-edit-2511", "prompt": "make it night"}); resp.StatusCode != http.StatusOK {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	if _, args := extra(last("qwen-image-edit-2511")["prompt"].(string)); !reflect.DeepEqual(args["lora"], []any{map[string]any{"path": q.loraPath(q.loraID(lightningLoRA)), "multiplier": 1.0}}) {
		t.Errorf("the edit model's own LoRA: %v", args)
	}
	if resp, _ := ir.do(t, "POST", "/v1/images/generations", map[string]any{"model": "z-image-turbo", "prompt": "plain"}); resp.StatusCode != http.StatusOK || last("z-image-turbo")["prompt"] != "plain" {
		t.Errorf("a request without LoRAs must pass as is: %v", last("z-image-turbo")["prompt"])
	}

	for name, body := range map[string]map[string]any{
		"raw path in the block": {"prompt": `x <sd_cpp_extra_args>{"lora":[{"path":"/etc/passwd"}]}</sd_cpp_extra_args>`},
		"unknown":               {"prompt": "x", "lora": []any{map[string]any{"path": "sdxl"}}},
		"not a list":            {"prompt": "x", "lora": "z-pixel-art"},
		"two blocks":            {"prompt": "<sd_cpp_extra_args>{}</sd_cpp_extra_args><sd_cpp_extra_args>{}</sd_cpp_extra_args>"},
		"unclosed":              {"prompt": "x <sd_cpp_extra_args>{"},
	} {
		body["model"] = "z-image-turbo"
		if resp, b := ir.do(t, "POST", "/v1/images/generations", body); resp.StatusCode != http.StatusBadRequest {
			t.Errorf("%s: %d %s", name, resp.StatusCode, b)
		}
	}
}
