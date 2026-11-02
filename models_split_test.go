package main

import (
	"os"
	"path/filepath"
	"testing"
)

// hermeticCaches points every cache location the scanners read at empty temp
// dirs, so tests never see (or delete from) the developer's real caches.
func hermeticCaches(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("HF_HOME", filepath.Join(home, "hf"))
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, "xdg"))
	t.Setenv("HUGGINGFACE_HUB_CACHE", "")
	t.Setenv("HF_HUB_CACHE", "")
	t.Setenv("LLAMA_CACHE", "")
	return home
}

const mib = 1024 * 1024

// downloadedSplit lays out a split model the way download.go writes it:
// <hub>/models--o--n/snapshots/main/UD-Q4_K_XL/M-UD-Q4_K_XL-0000k-of-00003.gguf
func downloadedSplit(t *testing.T) (hub, quantDir string) {
	t.Helper()
	hub = huggingfaceHubDir()
	quantDir = filepath.Join(hub, "models--o--n", "snapshots", "main", "UD-Q4_K_XL")
	for i := 1; i <= 3; i++ {
		mustWriteFile(t, filepath.Join(quantDir, "M-UD-Q4_K_XL-"+pad5(i)+"-of-00003.gguf"), 2*mib)
	}
	return hub, quantDir
}

// TestHubScanListsSplitModelOnce is the P1 #19 regression: the hub scan listed
// every shard, so the instance dropdown offered shards 2 and 3, which
// llama-server can't load ("model must be loaded with the first split").
func TestHubScanListsSplitModelOnce(t *testing.T) {
	hermeticCaches(t)
	hub, quantDir := downloadedSplit(t)

	got := scanHuggingfaceCache(hub)
	if len(got) != 1 {
		t.Fatalf("got %d entries, want 1: %+v", len(got), got)
	}
	m := got[0]
	if m.Name != "o/n/M-UD-Q4_K_XL" {
		t.Errorf("Name = %q, want o/n/M-UD-Q4_K_XL", m.Name)
	}
	if m.Path != filepath.Join(quantDir, "M-UD-Q4_K_XL-00001-of-00003.gguf") {
		t.Errorf("Path = %q, want the first part", m.Path)
	}
	if m.SizeMB != 6 {
		t.Errorf("SizeMB = %d, want 6 (all three parts)", m.SizeMB)
	}
}

// TestHubScanFollowsBlobSymlinksForSplitSize: in a real huggingface_hub cache
// snapshot entries are symlinks into blobs/.
func TestHubScanFollowsBlobSymlinksForSplitSize(t *testing.T) {
	hermeticCaches(t)
	hub := huggingfaceHubDir()
	repo := filepath.Join(hub, "models--o--n")
	snap := filepath.Join(repo, "snapshots", "abc123")
	if err := os.MkdirAll(snap, 0o755); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 2; i++ {
		blob := filepath.Join(repo, "blobs", "sha"+pad5(i))
		mustWriteFile(t, blob, 3*mib)
		if err := os.Symlink(blob, filepath.Join(snap, "Big-Q8_0-"+pad5(i)+"-of-00002.gguf")); err != nil {
			t.Fatal(err)
		}
	}
	got := scanHuggingfaceCache(hub)
	if len(got) != 1 || got[0].SizeMB != 6 || got[0].Name != "o/n/Big-Q8_0" {
		t.Fatalf("got %+v, want one 6 MB entry named o/n/Big-Q8_0", got)
	}

	if err := deleteCachedModel(got[0].Path); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(repo); !os.IsNotExist(err) {
		entries, _ := os.ReadDir(filepath.Join(repo, "blobs"))
		t.Fatalf("repo dir not removed after deleting its only model (blobs left: %d)", len(entries))
	}
}

func TestFlatCacheListsSplitModelOnce(t *testing.T) {
	dir := t.TempDir()
	for i := 1; i <= 2; i++ {
		mustWriteFile(t, filepath.Join(dir, "Big-Q8_0-"+pad5(i)+"-of-00002.gguf"), mib)
	}
	mustWriteFile(t, filepath.Join(dir, "Small-Q4_K_M.gguf"), mib)
	got := scanFlatGGUFDir(dir)
	names := map[string]int64{}
	for _, m := range got {
		names[m.Name] = m.SizeMB
	}
	if len(got) != 2 || names["Big-Q8_0"] != 2 || names["Small-Q4_K_M"] != 1 {
		t.Fatalf("got %+v", got)
	}
}

func TestModelTreeSplitSizeIsTotal(t *testing.T) {
	dir := t.TempDir()
	for i := 1; i <= 2; i++ {
		mustWriteFile(t, filepath.Join(dir, "G-Q4_K_M-"+pad5(i)+"-of-00002.gguf"), 3*mib)
	}
	got := scanModelTree(dir)
	if len(got) != 1 || got[0].SizeMB != 6 {
		t.Fatalf("got %+v, want one 6 MB entry", got)
	}
}

// TestDeleteSplitModelRemovesAllPartsAndEmptyDirs: deleting the listed entry
// used to remove only part 1, leaving many GB of unreachable shards, and the
// prune logic never reached the nested quant directory.
func TestDeleteSplitModelRemovesAllPartsAndEmptyDirs(t *testing.T) {
	hermeticCaches(t)
	hub, quantDir := downloadedSplit(t)

	if err := deleteCachedModel(filepath.Join(quantDir, "M-UD-Q4_K_XL-00001-of-00003.gguf")); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 3; i++ {
		if _, err := os.Stat(filepath.Join(quantDir, "M-UD-Q4_K_XL-"+pad5(i)+"-of-00003.gguf")); !os.IsNotExist(err) {
			t.Errorf("part %d still on disk", i)
		}
	}
	if _, err := os.Stat(filepath.Join(hub, "models--o--n")); !os.IsNotExist(err) {
		t.Error("empty repo dir left behind")
	}
}

func TestDeleteSplitModelKeepsOtherModelsInRepo(t *testing.T) {
	hermeticCaches(t)
	hub, quantDir := downloadedSplit(t)
	other := filepath.Join(hub, "models--o--n", "snapshots", "main", "Q8_0", "M-Q8_0.gguf")
	mustWriteFile(t, other, mib)

	if err := deleteCachedModel(filepath.Join(quantDir, "M-UD-Q4_K_XL-00001-of-00003.gguf")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(other); err != nil {
		t.Fatalf("unrelated model in the same repo was removed: %v", err)
	}
	if _, err := os.Stat(quantDir); !os.IsNotExist(err) {
		t.Error("emptied quant dir not pruned")
	}
	if got := scanHuggingfaceCache(hub); len(got) != 1 || got[0].Name != "o/n/M-Q8_0" {
		t.Fatalf("remaining models = %+v", got)
	}
}

// TestDeleteCachedModelRejectsPathsOutsideCache covers the only guard on
// DELETE /api/models?path=, which had no tests.
func TestDeleteCachedModelRejectsPathsOutsideCache(t *testing.T) {
	home := hermeticCaches(t)
	hub := huggingfaceHubDir()
	victim := filepath.Join(home, "precious.gguf")
	mustWriteFile(t, victim, 1)
	for _, p := range []string{
		victim,
		filepath.Join(hub, "..", "precious.gguf"),
		filepath.Join(hub, "models--o--n", "..", "..", "precious.gguf"),
		filepath.Join(hub, "models--o--n", "config.json"),
		"/etc/hosts.gguf",
	} {
		if err := deleteCachedModel(p); err == nil {
			t.Errorf("deleteCachedModel(%q) succeeded", p)
		}
	}
	if _, err := os.Stat(victim); err != nil {
		t.Fatal("file outside the cache was deleted")
	}
}

func TestModelParts(t *testing.T) {
	got := modelParts("/x/M-Q8_0-00002-of-00003.gguf")
	want := []string{"/x/M-Q8_0-00001-of-00003.gguf", "/x/M-Q8_0-00002-of-00003.gguf", "/x/M-Q8_0-00003-of-00003.gguf"}
	if len(got) != 3 || got[0] != want[0] || got[2] != want[2] {
		t.Fatalf("modelParts = %v, want %v", got, want)
	}
	if got := modelParts("/x/single.gguf"); len(got) != 1 || got[0] != "/x/single.gguf" {
		t.Fatalf("single-file modelParts = %v", got)
	}
}

// TestModelDirsFollowSymlinks (P2): WalkDir doesn't follow symlinks, so a
// model_dirs entry like /models -> /mnt/nvme/models (or a linked subdir)
// silently produced zero models.
func TestModelDirsFollowSymlinks(t *testing.T) {
	base := t.TempDir()
	real := filepath.Join(base, "nvme", "models")
	mustWriteFile(t, filepath.Join(real, "A-Q4_K_M.gguf"), mib)
	other := filepath.Join(base, "hdd", "more")
	mustWriteFile(t, filepath.Join(other, "B-Q8_0.gguf"), mib)
	if err := os.Symlink(other, filepath.Join(real, "linked")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(real, filepath.Join(real, "loop")); err != nil { // must not recurse forever
		t.Fatal(err)
	}
	root := filepath.Join(base, "models")
	if err := os.Symlink(real, root); err != nil {
		t.Fatal(err)
	}

	names := map[string]bool{}
	for _, m := range scanModelTree(root) {
		names[m.Name] = true
	}
	if !names["A-Q4_K_M"] || !names[filepath.Join("linked", "B-Q8_0")] || len(names) != 2 {
		t.Fatalf("models = %v, want A-Q4_K_M and linked/B-Q8_0 exactly once", names)
	}
}

// TestLlamaCacheIsScannedAndDeletable (P2): LLAMA_CACHE was displayed as the
// cache dir, but it was never scanned and its files couldn't be deleted
// ("path is outside the model cache").
func TestLlamaCacheIsScannedAndDeletable(t *testing.T) {
	hermeticCaches(t)
	dir := t.TempDir()
	t.Setenv("LLAMA_CACHE", dir+string(os.PathSeparator))
	model := filepath.Join(dir, "org_repo_M-Q4_K_M.gguf")
	mustWriteFile(t, model, mib)

	models, _ := scanCachedModels(nil)
	if len(models) != 1 || models[0].Path != model {
		t.Fatalf("models = %+v, want the LLAMA_CACHE file", models)
	}
	if err := deleteCachedModel(model); err != nil {
		t.Fatalf("delete from LLAMA_CACHE: %v", err)
	}
	if got := getCacheDir(); got != huggingfaceHubDir() {
		t.Fatalf("getCacheDir() = %q, want the download destination %q", got, huggingfaceHubDir())
	}
}

// TestHubDirPrecedence (P2): matches huggingface_hub: HF_HUB_CACHE, then
// HUGGINGFACE_HUB_CACHE, then HF_HOME/hub, then XDG_CACHE_HOME/huggingface/hub.
func TestHubDirPrecedence(t *testing.T) {
	home := hermeticCaches(t)
	t.Setenv("HF_HOME", "")
	t.Setenv("XDG_CACHE_HOME", "")
	if got, want := huggingfaceHubDir(), filepath.Join(home, ".cache", "huggingface", "hub"); got != want {
		t.Errorf("default = %q, want %q", got, want)
	}
	t.Setenv("XDG_CACHE_HOME", "/xdg")
	if got := huggingfaceHubDir(); got != "/xdg/huggingface/hub" {
		t.Errorf("XDG = %q", got)
	}
	t.Setenv("HF_HOME", "/hfhome")
	if got := huggingfaceHubDir(); got != "/hfhome/hub" {
		t.Errorf("HF_HOME = %q", got)
	}
	t.Setenv("HUGGINGFACE_HUB_CACHE", "/legacy/")
	if got := huggingfaceHubDir(); got != "/legacy" {
		t.Errorf("HUGGINGFACE_HUB_CACHE = %q, want cleaned /legacy", got)
	}
	t.Setenv("HF_HUB_CACHE", "/hub/")
	if got := huggingfaceHubDir(); got != "/hub" {
		t.Errorf("HF_HUB_CACHE = %q, want cleaned /hub", got)
	}
}

// TestDeleteWithTrailingSlashHubFreesBlob: an uncleaned HUGGINGFACE_HUB_CACHE
// made delete fall back to unlinking only the snapshot symlink, leaving the
// blob (the actual GBs) on disk.
func TestDeleteWithTrailingSlashHubFreesBlob(t *testing.T) {
	hermeticCaches(t)
	hub := t.TempDir()
	t.Setenv("HUGGINGFACE_HUB_CACHE", hub+string(os.PathSeparator))
	repo := filepath.Join(hub, "models--o--n")
	blob := filepath.Join(repo, "blobs", "sha")
	mustWriteFile(t, blob, mib)
	link := filepath.Join(repo, "snapshots", "abc", "M-Q4.gguf")
	if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(blob, link); err != nil {
		t.Fatal(err)
	}
	if err := deleteCachedModel(link); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(blob); !os.IsNotExist(err) {
		t.Fatal("blob left on disk")
	}
}
