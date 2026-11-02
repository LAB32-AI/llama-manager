package main

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

// modelPartRe matches the multi-part suffix llama.cpp uses for split GGUFs,
// e.g. -00001-of-00002.gguf. Only the first part is loadable directly;
// llama-server reads the remaining parts itself.
var modelPartRe = regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)

// isListedPart reports whether a .gguf file should appear as a model: single
// files and the first part of a split model do, later parts don't.
func isListedPart(name string) bool {
	m := modelPartRe.FindStringSubmatch(name)
	return m == nil || m[1] == "00001"
}

// modelDisplayName drops ".gguf" and any -NNNNN-of-NNNNN split suffix.
func modelDisplayName(name string) string {
	return strings.TrimSuffix(modelPartRe.ReplaceAllString(name, ".gguf"), ".gguf")
}

// modelParts returns every file of the model at path: all parts of a split
// GGUF (whichever part path names), or just path for a single file.
func modelParts(path string) []string {
	m := modelPartRe.FindStringSubmatch(filepath.Base(path))
	if m == nil {
		return []string{path}
	}
	n, _ := strconv.Atoi(m[2])
	prefix := strings.TrimSuffix(path, m[0])
	parts := make([]string, 0, n)
	for i := 1; i <= n; i++ {
		parts = append(parts, fmt.Sprintf("%s-%05d-of-%s.gguf", prefix, i, m[2]))
	}
	return parts
}

// modelSizeMB is the total size of all of a model's parts (following
// symlinks into the HF blob store).
func modelSizeMB(path string) int64 {
	var total int64
	for _, p := range modelParts(path) {
		if fi, err := os.Stat(p); err == nil {
			total += fi.Size()
		}
	}
	return total / (1024 * 1024)
}

type CachedModel struct {
	Name     string `json:"name"`
	FileName string `json:"file_name"`
	SizeMB   int64  `json:"size_mb"`
	Path     string `json:"path"`
	// Projector marks an mmproj file: listed so it can be deleted, but not
	// something an instance can load as its model.
	Projector bool `json:"projector,omitempty"`
}

// getCacheDir is where downloads are saved (shown in the Models tab).
func getCacheDir() string {
	return huggingfaceHubDir()
}

// huggingfaceHubDir resolves the HF hub cache with huggingface_hub's own
// precedence, so the manager and Python tooling share one cache. Paths are
// cleaned: delete compares prefixes and a trailing slash broke the match.
func huggingfaceHubDir() string {
	for _, env := range []string{"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"} {
		if v := os.Getenv(env); v != "" {
			return filepath.Clean(v)
		}
	}
	if v := os.Getenv("HF_HOME"); v != "" {
		return filepath.Join(v, "hub")
	}
	if v := os.Getenv("XDG_CACHE_HOME"); v != "" {
		return filepath.Join(v, "huggingface", "hub")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

// legacyLlamaCacheDirs are llama.cpp's own flat caches (llama-server -hf),
// LLAMA_CACHE first when set, so models there are listed and deletable.
func legacyLlamaCacheDirs() []string {
	home, _ := os.UserHomeDir()
	var dirs []string
	if env := os.Getenv("LLAMA_CACHE"); env != "" {
		dirs = append(dirs, filepath.Clean(env))
	}
	switch runtime.GOOS {
	case "darwin":
		dirs = append(dirs, filepath.Join(home, "Library", "Caches", "llama.cpp"))
	case "windows":
		if local := os.Getenv("LOCALAPPDATA"); local != "" {
			dirs = append(dirs, filepath.Join(local, "llama.cpp"))
		}
		dirs = append(dirs, filepath.Join(home, "AppData", "Local", "llama.cpp"))
	default:
		if xdg := os.Getenv("XDG_CACHE_HOME"); xdg != "" {
			dirs = append(dirs, filepath.Join(xdg, "llama.cpp"))
		}
		dirs = append(dirs, filepath.Join(home, ".cache", "llama.cpp"))
	}
	return dirs
}

func scanCachedModels(extraDirs []string) ([]CachedModel, error) {
	seen := make(map[string]bool)
	var models []CachedModel
	add := func(found []CachedModel) {
		for _, m := range found {
			if seen[m.Path] {
				continue
			}
			seen[m.Path] = true
			models = append(models, m)
		}
	}

	add(scanHuggingfaceCache(huggingfaceHubDir()))
	for _, dir := range legacyLlamaCacheDirs() {
		add(scanFlatGGUFDir(dir))
	}
	for _, dir := range extraDirs {
		dir = strings.TrimSpace(dir)
		if dir == "" {
			continue
		}
		add(scanModelTree(dir))
	}

	return models, nil
}

// scanModelTree walks `root` recursively and reports every .gguf file. For
// multi-part GGUFs (e.g. ...-00001-of-00004.gguf), only the first part is
// reported and the -NNNNN-of-NNNNN suffix is stripped from the display name,
// since that's the file you pass to llama-server -m. Symlinked directories,
// including root itself, are followed (models often live on another disk
// behind a link); each real directory is visited once, so loops terminate.
func scanModelTree(root string) []CachedModel {
	cleanRoot := filepath.Clean(root)
	var models []CachedModel
	visited := map[string]bool{}
	var walk func(dir string)
	walk = func(dir string) {
		real, err := filepath.EvalSymlinks(dir)
		if err != nil || visited[real] {
			return
		}
		visited[real] = true
		entries, err := os.ReadDir(dir)
		if err != nil {
			return
		}
		for _, e := range entries {
			p := filepath.Join(dir, e.Name())
			isDir := e.IsDir()
			if e.Type()&fs.ModeSymlink != 0 {
				fi, err := os.Stat(p)
				if err != nil {
					continue
				}
				isDir = fi.IsDir()
			}
			if isDir {
				walk(p)
				continue
			}
			if !strings.HasSuffix(e.Name(), ".gguf") || !isListedPart(e.Name()) {
				continue
			}
			if _, err := os.Stat(p); err != nil {
				continue
			}
			rel, err := filepath.Rel(cleanRoot, p)
			if err != nil {
				rel = e.Name()
			}
			models = append(models, CachedModel{
				Name:      modelDisplayName(rel),
				FileName:  e.Name(),
				SizeMB:    modelSizeMB(p),
				Path:      p,
				Projector: isProjector(e.Name()),
			})
		}
	}
	walk(cleanRoot)
	return models
}

func scanFlatGGUFDir(dir string) []CachedModel {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var models []CachedModel
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".gguf") || !isListedPart(e.Name()) {
			continue
		}
		p := filepath.Join(dir, e.Name())
		models = append(models, CachedModel{
			Name:      modelDisplayName(e.Name()),
			FileName:  e.Name(),
			SizeMB:    modelSizeMB(p),
			Path:      p,
			Projector: isProjector(e.Name()),
		})
	}
	return models
}

// deleteCachedModel removes a cached model, all parts of it if it is split.
// Two layouts are supported:
//   - flat legacy: <dir>/<file>.gguf — the files are unlinked.
//   - HF hub: <hub>/models--<org>--<repo>/snapshots/<rev>/[<subdir>/]<file>.gguf —
//     each symlink and the blob it points to are removed, then directories
//     left empty are pruned, up to the repo dir once no snapshot remains.
//
// The path MUST live inside a recognized cache dir; arbitrary paths are rejected.
func deleteCachedModel(path string) error {
	clean := filepath.Clean(path)
	if !strings.HasSuffix(clean, ".gguf") {
		return fmt.Errorf("path must end in .gguf")
	}

	if !pathInsideCache(clean) {
		return fmt.Errorf("path is outside the model cache")
	}

	hub := filepath.Clean(huggingfaceHubDir())
	inHub := strings.HasPrefix(clean, hub+string(os.PathSeparator))
	var firstErr error
	for _, part := range modelParts(clean) {
		var err error
		if inHub {
			err = removeHubFile(part)
		} else {
			err = os.Remove(part)
		}
		if err != nil && !os.IsNotExist(err) && firstErr == nil {
			firstErr = err
		}
	}
	if inHub {
		pruneHubDirs(filepath.Dir(clean), hub)
	}
	return firstErr
}

func pathInsideCache(clean string) bool {
	for _, root := range append(legacyLlamaCacheDirs(), huggingfaceHubDir()) {
		if rootClean := filepath.Clean(root); strings.HasPrefix(clean, rootClean+string(os.PathSeparator)) {
			return true
		}
	}
	return false
}

// removeHubFile removes a snapshot entry and, if it is a symlink, the blob it
// points to.
func removeHubFile(snapshotFile string) error {
	if target, err := filepath.EvalSymlinks(snapshotFile); err == nil && target != snapshotFile {
		_ = os.Remove(target)
	}
	return os.Remove(snapshotFile)
}

// pruneHubDirs removes directories under a repo's snapshots/ that the delete
// left empty, then the whole repo dir if no snapshot remains.
func pruneHubDirs(dir, hub string) {
	sep := string(os.PathSeparator)
	for d := dir; strings.HasPrefix(d, hub+sep) && filepath.Base(d) != "snapshots" && filepath.Dir(d) != hub; d = filepath.Dir(d) {
		if !isEmptyDir(d) {
			break
		}
		_ = os.Remove(d)
	}
	repo := dir
	for strings.HasPrefix(repo, hub+sep) && filepath.Dir(repo) != hub {
		repo = filepath.Dir(repo)
	}
	if filepath.Dir(repo) == hub && strings.HasPrefix(filepath.Base(repo), "models--") && isLeftoverRepoEmpty(repo) {
		_ = os.RemoveAll(repo)
	}
}

func isEmptyDir(p string) bool {
	entries, err := os.ReadDir(p)
	return err == nil && len(entries) == 0
}

func isLeftoverRepoEmpty(repoDir string) bool {
	snaps := filepath.Join(repoDir, "snapshots")
	entries, err := os.ReadDir(snaps)
	if err == nil && len(entries) > 0 {
		return false
	}
	return true
}

// scanHuggingfaceCache walks ~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<rev>/*.gguf
// and returns one entry per .gguf file. Snapshot files are symlinks into blobs/;
// we follow them via os.Stat so SizeMB reflects the real file size.
func scanHuggingfaceCache(root string) []CachedModel {
	var models []CachedModel
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	for _, repo := range entries {
		if !repo.IsDir() || !strings.HasPrefix(repo.Name(), "models--") {
			continue
		}
		repoDir := filepath.Join(root, repo.Name())
		snapsDir := filepath.Join(repoDir, "snapshots")
		repoName := strings.ReplaceAll(strings.TrimPrefix(repo.Name(), "models--"), "--", "/")

		_ = filepath.WalkDir(snapsDir, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				return nil
			}
			if !strings.HasSuffix(d.Name(), ".gguf") || !isListedPart(d.Name()) {
				return nil
			}
			if _, err := os.Stat(p); err != nil {
				return nil
			}
			models = append(models, CachedModel{
				Name:      repoName + "/" + modelDisplayName(d.Name()),
				FileName:  d.Name(),
				SizeMB:    modelSizeMB(p),
				Path:      p,
				Projector: isProjector(d.Name()),
			})
			return nil
		})
	}
	return models
}
