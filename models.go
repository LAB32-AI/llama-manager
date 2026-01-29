package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

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
	home, _ := os.UserHomeDir()
	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(home, "Library", "Caches", "llama.cpp")
	case "windows":
		if local := os.Getenv("LOCALAPPDATA"); local != "" {
			return filepath.Join(local, "llama.cpp")
		}
		return filepath.Join(home, "AppData", "Local", "llama.cpp")
	default:
		if xdg := os.Getenv("XDG_CACHE_HOME"); xdg != "" {
			return filepath.Join(xdg, "llama.cpp")
		}
		return filepath.Join(home, ".cache", "llama.cpp")
	}
}

func scanCachedModels() ([]CachedModel, error) {
	dir := getCacheDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
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
		name := e.Name()
		name = strings.TrimSuffix(name, ".gguf")
		models = append(models, CachedModel{
			Name:     name,
			FileName: e.Name(),
			SizeMB:   info.Size() / (1024 * 1024),
			Path:     filepath.Join(dir, e.Name()),
		})
	}
	return models, nil
}
