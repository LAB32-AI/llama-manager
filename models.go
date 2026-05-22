package main

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

// modelPartRe matches the multi-part suffix llama.cpp uses for split GGUFs,
// e.g. -00001-of-00002.gguf. Only the first part is loadable directly;
// llama-server reads the remaining parts itself.
var modelPartRe = regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)

type CachedModel struct {
	Name     string `json:"name"`
	FileName string `json:"file_name"`
	SizeMB   int64  `json:"size_mb"`
	Path     string `json:"path"`
}

func getCacheDir() string {
	if env := os.Getenv("LLAMA_CACHE"); env != "" {
		return env
	}
	return huggingfaceHubDir()
}

func huggingfaceHubDir() string {
	if env := os.Getenv("HF_HOME"); env != "" {
		return filepath.Join(env, "hub")
	}
	if env := os.Getenv("HUGGINGFACE_HUB_CACHE"); env != "" {
		return env
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

func legacyLlamaCacheDirs() []string {
	home, _ := os.UserHomeDir()
	var dirs []string
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
// since that's the file you pass to llama-server -m.
func scanModelTree(root string) []CachedModel {
	cleanRoot := filepath.Clean(root)
	var models []CachedModel
	_ = filepath.WalkDir(cleanRoot, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		if !strings.HasSuffix(d.Name(), ".gguf") {
			return nil
		}
		if m := modelPartRe.FindStringSubmatch(d.Name()); m != nil && m[1] != "00001" {
			return nil
		}
		info, err := os.Stat(p)
		if err != nil {
			return nil
		}
		rel, err := filepath.Rel(cleanRoot, p)
		if err != nil {
			rel = d.Name()
		}
		name := strings.TrimSuffix(rel, ".gguf")
		name = modelPartRe.ReplaceAllString(name+".gguf", "")
		name = strings.TrimSuffix(name, ".gguf")
		models = append(models, CachedModel{
			Name:     name,
			FileName: d.Name(),
			SizeMB:   info.Size() / (1024 * 1024),
			Path:     p,
		})
		return nil
	})
	return models
}

func scanFlatGGUFDir(dir string) []CachedModel {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var models []CachedModel
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".gguf") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		models = append(models, CachedModel{
			Name:     strings.TrimSuffix(e.Name(), ".gguf"),
			FileName: e.Name(),
			SizeMB:   info.Size() / (1024 * 1024),
			Path:     filepath.Join(dir, e.Name()),
		})
	}
	return models
}

// deleteCachedModel removes a single .gguf model from the cache. Two layouts
// are supported:
//   - flat legacy:  <dir>/<file>.gguf  -> just unlinks the file.
//   - HF hub:       ~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<rev>/<file>.gguf
//                   -> resolves the symlink, removes the blob it points to, removes
//                      the symlink, then prunes the snapshot/repo dir if now empty.
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

	if hubPrefix := huggingfaceHubDir(); strings.HasPrefix(clean, hubPrefix+string(os.PathSeparator)) {
		return deleteHubModel(clean, hubPrefix)
	}
	return os.Remove(clean)
}

func pathInsideCache(clean string) bool {
	for _, root := range append(legacyLlamaCacheDirs(), huggingfaceHubDir()) {
		if rootClean := filepath.Clean(root); strings.HasPrefix(clean, rootClean+string(os.PathSeparator)) {
			return true
		}
	}
	return false
}

func deleteHubModel(snapshotFile, hubRoot string) error {
	target, err := filepath.EvalSymlinks(snapshotFile)
	if err == nil && target != snapshotFile {
		_ = os.Remove(target)
	}
	if err := os.Remove(snapshotFile); err != nil && !os.IsNotExist(err) {
		return err
	}

	snapDir := filepath.Dir(snapshotFile)
	if isEmptyDir(snapDir) {
		_ = os.Remove(snapDir)
	}

	repoDir := filepath.Dir(filepath.Dir(snapDir))
	if strings.HasPrefix(filepath.Base(repoDir), "models--") &&
		filepath.Dir(repoDir) == filepath.Clean(hubRoot) &&
		isLeftoverRepoEmpty(repoDir) {
		_ = os.RemoveAll(repoDir)
	}
	return nil
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
			if !strings.HasSuffix(d.Name(), ".gguf") {
				return nil
			}
			info, err := os.Stat(p)
			if err != nil {
				return nil
			}
			models = append(models, CachedModel{
				Name:     repoName + "/" + strings.TrimSuffix(d.Name(), ".gguf"),
				FileName: d.Name(),
				SizeMB:   info.Size() / (1024 * 1024),
				Path:     p,
			})
			return nil
		})
	}
	return models
}
