package main

import (
	"os"
	"path/filepath"
	"sort"
	"testing"
)

func TestScanModelTreeFlatFiles(t *testing.T) {
	dir := t.TempDir()
	mustWriteFile(t, filepath.Join(dir, "Qwen3.6-27B-Q4_K_M.gguf"), 1024*1024)
	mustWriteFile(t, filepath.Join(dir, "README.md"), 10)

	got := scanModelTree(dir)
	if len(got) != 1 {
		t.Fatalf("got %d models, want 1: %+v", len(got), got)
	}
	if got[0].Name != "Qwen3.6-27B-Q4_K_M" {
		t.Errorf("Name = %q, want Qwen3.6-27B-Q4_K_M", got[0].Name)
	}
	if got[0].FileName != "Qwen3.6-27B-Q4_K_M.gguf" {
		t.Errorf("FileName = %q", got[0].FileName)
	}
}

func TestScanModelTreeRecursive(t *testing.T) {
	dir := t.TempDir()
	subDir := filepath.Join(dir, "gpt-oss-120b")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		t.Fatal(err)
	}
	mustWriteFile(t, filepath.Join(subDir, "gpt-oss-120b-Q4_K_M.gguf"), 1024)

	got := scanModelTree(dir)
	if len(got) != 1 {
		t.Fatalf("got %d models, want 1: %+v", len(got), got)
	}
	if got[0].Name != filepath.Join("gpt-oss-120b", "gpt-oss-120b-Q4_K_M") {
		t.Errorf("Name = %q", got[0].Name)
	}
}

// TestScanModelTreeMultiPart reproduces the user's gpt-oss-120b / MiniMax /
// GLM layouts where each model is split across multiple .gguf files. Only the
// first part is loadable; the manager must surface exactly one entry per
// split model and point at -00001-of-NNNNN.
func TestScanModelTreeMultiPart(t *testing.T) {
	dir := t.TempDir()
	mm := filepath.Join(dir, "MiniMax-M2.7-UD-IQ4_XS")
	if err := os.MkdirAll(mm, 0o755); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 4; i++ {
		name := filepath.Join(mm, "MiniMax-M2.7-UD-IQ4_XS-"+pad5(i)+"-of-00004.gguf")
		mustWriteFile(t, name, 1024)
	}
	gpt := filepath.Join(dir, "gpt-oss-120b")
	if err := os.MkdirAll(gpt, 0o755); err != nil {
		t.Fatal(err)
	}
	mustWriteFile(t, filepath.Join(gpt, "gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"), 1024)
	mustWriteFile(t, filepath.Join(gpt, "gpt-oss-120b-Q4_K_M-00002-of-00002.gguf"), 1024)
	mustWriteFile(t, filepath.Join(dir, "Standalone-Q4.gguf"), 512)

	got := scanModelTree(dir)
	sort.Slice(got, func(i, j int) bool { return got[i].Path < got[j].Path })
	if len(got) != 3 {
		t.Fatalf("want 3 entries (one per model), got %d: %+v", len(got), got)
	}

	want := map[string]string{
		filepath.Join("gpt-oss-120b", "gpt-oss-120b-Q4_K_M"):                    "gpt-oss-120b-Q4_K_M-00001-of-00002.gguf",
		filepath.Join("MiniMax-M2.7-UD-IQ4_XS", "MiniMax-M2.7-UD-IQ4_XS"):        "MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf",
		"Standalone-Q4": "Standalone-Q4.gguf",
	}
	seen := make(map[string]bool)
	for _, m := range got {
		expFile, ok := want[m.Name]
		if !ok {
			t.Errorf("unexpected model name %q (path %s)", m.Name, m.Path)
			continue
		}
		if m.FileName != expFile {
			t.Errorf("model %q: FileName = %q, want %q", m.Name, m.FileName, expFile)
		}
		seen[m.Name] = true
	}
	for k := range want {
		if !seen[k] {
			t.Errorf("missing model %q", k)
		}
	}
}

func TestScanCachedModelsHonorsExtraDirs(t *testing.T) {
	dir := t.TempDir()
	mustWriteFile(t, filepath.Join(dir, "MyModel-Q4.gguf"), 1024)

	got, err := scanCachedModels([]string{dir})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, m := range got {
		if m.Name == "MyModel-Q4" {
			found = true
		}
	}
	if !found {
		t.Errorf("scanCachedModels with extra dir did not include MyModel-Q4. Got: %+v", got)
	}
}

func mustWriteFile(t *testing.T, p string, size int) {
	t.Helper()
	if err := os.WriteFile(p, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
}

func pad5(n int) string {
	s := "00000" + itoaSmall(n)
	return s[len(s)-5:]
}

func itoaSmall(n int) string {
	if n == 0 {
		return "0"
	}
	digits := ""
	for n > 0 {
		digits = string(rune('0'+n%10)) + digits
		n /= 10
	}
	return digits
}
