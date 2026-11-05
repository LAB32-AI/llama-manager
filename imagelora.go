package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// LoRAs for the image models. sd-server reads them from one folder per model
// (--lora-model-dir) and takes a LoRA only by its path inside that folder,
// extension included; a file whose real path is outside the folder (a
// symlink into the hub cache's blobs/, say) is skipped. So each model gets a
// LoRA folder: the user's lora_dir, where their own files go, or a folder
// next to the hub cache. Built-in LoRAs, pinned like the models' files, are
// downloaded to the hub cache and linked into the folder's llama-manager/
// subfolder.

const (
	loraSubdir     = "llama-manager" // the built-in LoRAs' subfolder of a LoRA folder
	maxLoRAPicks   = 4
	maxLoRAWeight  = 2.0
	maxUserLoRAs   = 200
	loraListMaxAge = 3 * time.Second
)

// linkFile hard-links a built-in LoRA into a LoRA folder. Var for tests.
var linkFile = os.Link

// userLoRAExts are the files listed from a LoRA folder; sd-server reads these
// and a few pickle formats llama-manager leaves out.
var userLoRAExts = map[string]bool{".safetensors": true, ".sft": true, ".gguf": true}

// imageLoRA is a built-in LoRA: a pinned file, the presets it was trained
// for, and how to prompt it.
type imageLoRA struct {
	Label   string
	File    imageFile
	Fits    []string // preset keys
	Fixed   bool     // part of a preset, always applied; not listed under loras
	Weight  float64  // suggested strength
	Trigger string   // words its images are prompted with
	Hint    string
	License string
}

var (
	loraZImage    = []string{"z-image-turbo"}
	loraKlein4B   = []string{"flux2-klein-4b"}
	loraQwenEdit  = []string{"qwen-image-edit-2511", "qwen-image-edit-2511-q8"}
	loraApache    = "Apache-2.0"
	lightningLoRA = "qwen-edit-lightning"
)

// imageLoRAs are the built-in LoRAs; image_generation's loras picks them by
// these names. All are Apache-2.0 and pinned to a commit and SHA-256.
var imageLoRAs = map[string]imageLoRA{
	// Part of the Qwen-Image-Edit presets: 4 steps instead of 40.
	lightningLoRA: {Label: "Lightning (4 steps)", Fits: loraQwenEdit, Fixed: true, Weight: 1, License: loraApache,
		File: imageFile{Repo: "lightx2v/Qwen-Image-Edit-2511-Lightning", Rev: "107d238d68b8387e32843d0b4a2e24ceb0d3ad86",
			Path: "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", Size: 849608296, SHA256: "22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f"}},

	"z-pixel-art": {Label: "Pixel art", Fits: loraZImage, Weight: 1, Trigger: "Pixel art style.", License: loraApache,
		Hint: "Retro pixel-art pictures.",
		File: imageFile{Repo: "tarn59/pixel_art_style_lora_z_image_turbo", Rev: "0a5092d1619664d94a5a36784f92db84b3ae62bd",
			Path: "pixel_art_style_z_image_turbo.safetensors", Size: 170128328, SHA256: "09b1b45ceed0202929bca528e51b50208d0160f4e9d2ba0f42cb7e739a43577f"}},
	"z-realism": {Label: "Realism", Fits: loraZImage, Weight: 1, Trigger: "Realism", License: loraApache,
		Hint: "More photographic skin, light and detail.",
		File: imageFile{Repo: "suayptalha/Z-Image-Turbo-Realism-LoRA", Rev: "8dc179cb56844d4ddc6e4527c30e98347eeae971",
			Path: "pytorch_lora_weights.safetensors", Size: 85094800, SHA256: "b38b074964f1f6564c22fee42125accd7a218a916966b5d0f0b2f63d4939c2f7"}},
	"z-classic-painting": {Label: "Classic painting", Fits: loraZImage, Weight: 1, Trigger: "class1cpa1nt", License: loraApache,
		Hint: "Old-master oil paintings: brushwork, canvas and dramatic light.",
		File: imageFile{Repo: "renderartist/Classic-Painting-Z-Image-Turbo-LoRA", Rev: "e873cc474ce517d49d09ed1f64358ff43660b276",
			Path: "Classic_Painting_Z_Image_Turbo_v1_renderartist_1750.safetensors", Size: 170128288, SHA256: "8beebb73725ec10ad21155d91c1d7667cb9a05544e8b2dfa0cbd72f5fa64a27b"}},

	"klein-pixel-art": {Label: "Pixel art sprites", Fits: loraKlein4B, Weight: 1, Trigger: "pixel art sprite, game asset, transparent background", License: loraApache,
		Hint: "Game sprites; strength 0.85 to 1.4 works.",
		File: imageFile{Repo: "Limbicnation/pixel-art-lora", Rev: "0ac8e5c3400af68228811edc324721e25fc26777",
			Path: "pytorch_lora_weights.comfyui.safetensors", Size: 325276656, SHA256: "24e938f510f5dd0c890ac8b1078f4abb87a50c9c053b2e85c44821c0f30011ad"}},
	"klein-anime": {Label: "Anime", Fits: loraKlein4B, Weight: 1, Trigger: "anime_style", License: loraApache,
		Hint: "Anime illustration style.",
		File: imageFile{Repo: "Sawata97/flux2_4b_koni_animestyle", Rev: "b392813770c6155d3ae22d3fb8062f247f5108a2",
			Path: "Flux_klein_4b_anime_Koni.safetensors", Size: 184832664, SHA256: "c5f8fef8610f8f7725ab4da0410ff6ffb09542e9a65676ebcf01527eb3af429f"}},
	"klein-sketch-to-image": {Label: "Sketch to image", Fits: loraKlein4B, Weight: 1, License: loraApache,
		Hint: "Attach a sketch and describe the finished picture.",
		File: imageFile{Repo: "ostris/sketch_to_image_klein_4b", Rev: "898fa73df3c60b8f7d83ec667345690b31c4d5da",
			Path: "sketch_to_image_klein_4b.safetensors", Size: 92426712, SHA256: "6e84275b88b602b352201231e9147c8947b63385f313c1c875467850fb36bbfd"}},

	"qie-multiple-angles": {Label: "Camera angles", Fits: loraQwenEdit, Weight: 0.9, Trigger: "<sks> front view eye-level shot medium shot", License: loraApache,
		Hint: "Shows the attached picture from another camera position: front, quarter, side or back view; low, eye-level or high angle; close-up, medium or wide shot.",
		File: imageFile{Repo: "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA", Rev: "e3066224ab74263f4a5b6179cd1a3b0a15577e44",
			Path: "qwen-image-edit-2511-multiple-angles-lora.safetensors", Size: 295140688, SHA256: "42426ded4e25fd22879d9e198b857556445ef4ca56e8da3246d0345155bb6765"}},
	"qie-anything2real": {Label: "Anything to photo", Fits: loraQwenEdit, Weight: 0.9, Trigger: "change the picture 1 to realistic photograph,", License: loraApache,
		Hint: "Turns drawings, anime and paintings into photos; strength 0.75 to 1. Describe the picture after the trigger words.",
		File: imageFile{Repo: "lrzjason/Anything2Real", Rev: "b547e090d255ac10b190204c7868b03a61faf15e",
			Path: "anything2real_2601_A_final_patched.safetensors", Size: 613580128, SHA256: "d4cbbabf5610a8f0ea3f530dc0bc9e8a254be6563c4738795bea45d247903528"}},
	"qie-object-remover": {Label: "Remove objects", Fits: loraQwenEdit, Weight: 1, Trigger: "Remove the", License: loraApache,
		Hint: "Say what to remove; the background is kept.",
		File: imageFile{Repo: "prithivMLmods/Qwen-Image-Edit-2511-Object-Remover", Rev: "3d92218a37c01592db63ff95ea95bb07b4508ddb",
			Path: "Qwen-Image-Edit-2511-Object-Remover.safetensors", Size: 236117064, SHA256: "8102b5fda35de505d3825b73e355f1df69baff466fa53800fd31e6232c447632"}},
}

func (l imageLoRA) fits(preset string) bool {
	for _, k := range l.Fits {
		if k == preset {
			return true
		}
	}
	return false
}

// loraNamesFor lists the built-in LoRAs a config may pick for a preset.
func loraNamesFor(preset string) []string {
	var out []string
	for k, l := range imageLoRAs {
		if !l.Fixed && l.fits(preset) {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

// defaultLoRARoot holds the LoRA folders of models without a lora_dir, next
// to the hub cache so built-in LoRAs can be hard links instead of copies.
func defaultLoRARoot() string {
	return filepath.Join(filepath.Dir(huggingfaceHubDir()), "llama-manager-loras")
}

// builtinLoRAs are the preset's own LoRA and those config picked for it.
func (e imageEntry) builtinLoRAs() []string {
	var out []string
	if e.Preset.LoRA != "" {
		out = append(out, e.Preset.LoRA)
	}
	return append(out, e.LoRAs...)
}

// loraRoot is the folder sd-server reads LoRAs from, "" for a model that
// has none.
func (e imageEntry) loraRoot() string {
	if e.LoraDir != "" {
		return filepath.Clean(e.LoraDir)
	}
	if len(e.builtinLoRAs()) > 0 {
		return filepath.Join(defaultLoRARoot(), e.Name)
	}
	return ""
}

// loraID is how sd-server knows a built-in LoRA: its path in the LoRA folder.
func (e imageEntry) loraID(key string) string {
	return loraSubdir + "/" + e.Name + "/" + key + strings.ToLower(path.Ext(imageLoRAs[key].File.Path))
}

func (e imageEntry) loraPath(id string) string {
	return filepath.Join(e.loraRoot(), filepath.FromSlash(id))
}

// files are every file the model needs: the preset's and its LoRAs'.
func (e imageEntry) files() []imageFile {
	out := append([]imageFile(nil), e.Preset.Files...)
	for _, k := range e.builtinLoRAs() {
		out = append(out, imageLoRAs[k].File)
	}
	return out
}

func (e imageEntry) missing() []imageFile {
	var out []imageFile
	for _, f := range e.files() {
		if _, ok := f.localPath(); !ok {
			out = append(out, f)
		}
	}
	return out
}

// loraReady says the LoRA folder exists and holds every built-in LoRA, so
// sd-server can start. It only looks; stageLoRAs does the work.
func (e imageEntry) loraReady() bool {
	root := e.loraRoot()
	if root == "" {
		return true
	}
	if fi, err := os.Stat(root); err != nil || !fi.IsDir() {
		return false
	}
	for _, k := range e.builtinLoRAs() {
		fi, err := os.Lstat(e.loraPath(e.loraID(k)))
		if err != nil || !fi.Mode().IsRegular() || fi.Size() != imageLoRAs[k].File.Size {
			return false
		}
	}
	return true
}

// stageLoRAs makes the LoRA folder and puts the built-in LoRAs in its
// llama-manager/<model>/ subfolder: hard links to the hub cache's files where
// the disk allows, copies where it doesn't. Real files either way, which
// sd-server needs. Files the model no longer uses leave that subfolder; the
// rest of the folder is the user's and isn't touched.
func (e imageEntry) stageLoRAs(logf func(string)) error {
	root := e.loraRoot()
	if root == "" {
		return nil
	}
	dir := filepath.Join(root, loraSubdir, e.Name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	want := map[string]bool{}
	for _, k := range e.builtinLoRAs() {
		l := imageLoRAs[k]
		dst := e.loraPath(e.loraID(k))
		want[filepath.Base(dst)] = true
		if fi, err := os.Lstat(dst); err == nil && fi.Mode().IsRegular() && fi.Size() == l.File.Size {
			continue
		}
		src, ok := l.File.localPath()
		if !ok {
			return errImageFilesMissing
		}
		real, err := filepath.EvalSymlinks(src)
		if err != nil {
			return err
		}
		_ = os.Remove(dst)
		if linkFile(real, dst) != nil {
			logf(fmt.Sprintf("copying the %s LoRA into %s (it is on another disk than the hub cache)", k, dir))
			if err := copyFile(real, dst); err != nil {
				return fmt.Errorf("copying the %s LoRA: %w", k, err)
			}
		}
		if fi, err := os.Stat(dst); err != nil || fi.Size() != l.File.Size {
			return fmt.Errorf("the %s LoRA in %s has the wrong size", k, dir)
		}
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, d := range ents {
		if !d.IsDir() && !want[d.Name()] {
			_ = os.Remove(filepath.Join(dir, d.Name()))
		}
	}
	return nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp := dst + ".partial"
	out, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer os.Remove(tmp)
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}

// loraChoice is a LoRA the chat can pick for a model.
type loraChoice struct {
	ID      string  `json:"id"` // its path in the LoRA folder, as sd-server knows it
	Name    string  `json:"name"`
	Key     string  `json:"key,omitempty"` // a built-in LoRA's name
	Weight  float64 `json:"weight"`
	Trigger string  `json:"trigger,omitempty"`
	Hint    string  `json:"hint,omitempty"`
	License string  `json:"license,omitempty"`
	SizeMB  int64   `json:"size_mb"`
}

// loraChoices are the built-in LoRAs config picked, then the user's files.
func (e imageEntry) loraChoices() []loraChoice {
	var out []loraChoice
	for _, k := range e.LoRAs {
		l := imageLoRAs[k]
		out = append(out, loraChoice{ID: e.loraID(k), Name: l.Label, Key: k, Weight: l.Weight, Trigger: l.Trigger, Hint: l.Hint,
			License: l.License, SizeMB: l.File.Size >> 20})
	}
	return append(out, e.userLoRAs()...)
}

// userLoRAs lists the LoRA files in the folder the way sd-server sees them:
// regular files whose real path is inside the folder, by path relative to
// it. Hidden entries, the built-ins' subfolder and anything more than two
// folders deep are left out.
func (e imageEntry) userLoRAs() []loraChoice {
	root := e.loraRoot()
	if root == "" {
		return nil
	}
	real, err := filepath.EvalSymlinks(root)
	if err != nil {
		return nil
	}
	var out []loraChoice
	_ = filepath.WalkDir(real, func(p string, d fs.DirEntry, err error) error {
		if err != nil || p == real {
			return nil
		}
		rel, _ := filepath.Rel(real, p)
		if strings.HasPrefix(d.Name(), ".") || (d.IsDir() && (rel == loraSubdir || strings.Count(rel, string(filepath.Separator)) >= 2)) {
			if d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if d.IsDir() || !userLoRAExts[strings.ToLower(filepath.Ext(p))] {
			return nil
		}
		if d.Type()&fs.ModeSymlink != 0 {
			target, err := filepath.EvalSymlinks(p)
			if err != nil || !strings.HasPrefix(target, real+string(filepath.Separator)) {
				return nil
			}
		}
		fi, err := os.Stat(p)
		if err != nil || !fi.Mode().IsRegular() {
			return nil
		}
		if len(out) >= maxUserLoRAs {
			return fs.SkipAll
		}
		id := filepath.ToSlash(rel)
		out = append(out, loraChoice{ID: id, Name: strings.TrimSuffix(id, path.Ext(id)), Weight: 1, SizeMB: fi.Size() >> 20})
		return nil
	})
	sort.Slice(out, func(i, j int) bool { return strings.ToLower(out[i].ID) < strings.ToLower(out[j].ID) })
	return out
}

// loraPick is a LoRA a request asks for: its id (or, over /v1, its built-in
// name or listed name) and strength.
type loraPick struct {
	ID string
	W  float64
}

// loraList is the model's LoRAs for /api/status, cached briefly: every open
// browser polls the status.
func (g *imageGen) loraList(e imageEntry) []loraChoice {
	g.lmu.Lock()
	defer g.lmu.Unlock()
	if c, ok := g.loras[e.Name]; ok && time.Since(c.at) < loraListMaxAge {
		return c.list
	}
	list := e.loraChoices()
	if list == nil {
		list = []loraChoice{}
	}
	g.loras[e.Name] = cachedLoRAs{at: time.Now(), list: list}
	return list
}

type cachedLoRAs struct {
	at   time.Time
	list []loraChoice
}

// loraRequest turns a request's LoRAs into sd-server's lora list, the
// preset's own first. Each must be one of the model's LoRAs as the folder is
// now. abs gives full paths, for sd-server's OpenAI route, which doesn't look
// paths up in the folder. labels name the picked ones for the reply's stats.
func (e imageEntry) loraRequest(picks []loraPick, abs bool) (list []map[string]any, labels []string, err error) {
	ref := func(id string) string {
		if abs {
			return e.loraPath(id)
		}
		return id
	}
	if k := e.Preset.LoRA; k != "" {
		list = append(list, map[string]any{"path": ref(e.loraID(k)), "multiplier": imageLoRAs[k].Weight})
	}
	if len(picks) == 0 {
		return list, nil, nil
	}
	if len(picks) > maxLoRAPicks {
		return nil, nil, fmt.Errorf("%s takes at most %d LoRAs at once", e.Name, maxLoRAPicks)
	}
	choices := e.loraChoices()
	seen := map[string]bool{}
	for _, p := range picks {
		c, ok := findLoRA(choices, p.ID)
		if !ok {
			return nil, nil, fmt.Errorf("%s has no LoRA %q (was the file moved or removed?)", e.Name, p.ID)
		}
		if math.IsNaN(p.W) || p.W < 0 || p.W > maxLoRAWeight {
			return nil, nil, fmt.Errorf("the strength of the %s LoRA must be between 0 and %g", c.Name, maxLoRAWeight)
		}
		if p.W == 0 || seen[c.ID] {
			continue
		}
		seen[c.ID] = true
		list = append(list, map[string]any{"path": ref(c.ID), "multiplier": p.W})
		labels = append(labels, c.Name+" "+strconv.FormatFloat(p.W, 'f', -1, 64))
	}
	return list, labels, nil
}

func findLoRA(choices []loraChoice, s string) (loraChoice, bool) {
	for _, c := range choices {
		if c.ID == s {
			return c, true
		}
	}
	for _, c := range choices {
		if (c.Key != "" && c.Key == s) || c.Name == s {
			return c, true
		}
	}
	return loraChoice{}, false
}

const extraArgsOpen, extraArgsClose = "<sd_cpp_extra_args>", "</sd_cpp_extra_args>"

// v1LoRAs puts the LoRAs of an OpenAI-style request where sd-server's OpenAI
// route reads them: an sd_cpp_extra_args block in the prompt. That route uses
// the paths as given, so a block the caller wrote is checked like the
// request's own "lora" field: only the model's LoRAs, sent by full path. The
// preset's own LoRA is always added.
func (e imageEntry) v1LoRAs(req map[string]any) error {
	prompt, _ := req["prompt"].(string)
	extra := map[string]any{}
	if i := strings.Index(prompt, extraArgsOpen); i >= 0 {
		j := strings.Index(prompt[i:], extraArgsClose)
		if j < 0 {
			return errors.New("the prompt's sd_cpp_extra_args block isn't closed")
		}
		j += i
		if err := json.Unmarshal([]byte(prompt[i+len(extraArgsOpen):j]), &extra); err != nil || extra == nil {
			return errors.New("sd_cpp_extra_args must hold a JSON object")
		}
		prompt = prompt[:i] + prompt[j+len(extraArgsClose):]
		if strings.Contains(prompt, extraArgsOpen) {
			return errors.New("put all sd_cpp_extra_args in one block")
		}
	}
	var picks []loraPick
	for _, src := range []any{req["lora"], extra["lora"]} {
		if src == nil {
			continue
		}
		items, ok := src.([]any)
		if !ok {
			return errors.New(`"lora" must be a list of {"path": ..., "multiplier": ...}`)
		}
		for _, it := range items {
			o, ok := it.(map[string]any)
			if !ok {
				return errors.New(`each LoRA must be an object like {"path": "z-pixel-art", "multiplier": 0.8}`)
			}
			var id string
			for _, k := range []string{"path", "id", "name"} {
				if s, ok := o[k].(string); ok && s != "" {
					id = s
					break
				}
			}
			w, ok := o["multiplier"].(float64)
			if !ok {
				w = 1
			}
			picks = append(picks, loraPick{ID: id, W: w})
		}
	}
	delete(req, "lora")
	delete(extra, "lora")
	list, _, err := e.loraRequest(picks, true)
	if err != nil {
		return err
	}
	if len(list) > 0 {
		extra["lora"] = list
	}
	if len(extra) > 0 {
		b, _ := json.Marshal(extra)
		prompt = strings.TrimSpace(prompt) + " " + extraArgsOpen + string(b) + extraArgsClose
	}
	req["prompt"] = prompt
	return nil
}
