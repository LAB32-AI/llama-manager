package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func siblingsServer(t *testing.T, files ...string) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var parts []string
		for _, f := range files {
			parts = append(parts, `{"rfilename":"`+f+`"}`)
		}
		w.Write([]byte(`{"siblings":[` + strings.Join(parts, ",") + `]}`))
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

// TestDotSeparatedQuants (P2): TheBloke-style names put a "." before the quant,
// which the "-"-only pattern missed, so those repos listed no quants and
// couldn't be downloaded.
func TestDotSeparatedQuants(t *testing.T) {
	api := siblingsServer(t, "llama-2-7b.Q4_K_M.gguf", "llama-2-7b.Q8_0.gguf", "README.md")
	q, err := fetchQuants("TheBloke/Llama-2-7B-GGUF", api)
	if err != nil || !reflect.DeepEqual(q, []string{"Q4_K_M", "Q8_0"}) {
		t.Fatalf("fetchQuants = %v, %v", q, err)
	}
	files, err := resolveQuantFiles(context.Background(), "TheBloke/Llama-2-7B-GGUF", "Q4_K_M", api)
	if err != nil || !reflect.DeepEqual(files, []string{"llama-2-7b.Q4_K_M.gguf"}) {
		t.Fatalf("resolveQuantFiles = %v, %v", files, err)
	}
}

// TestProjectorsAreNotQuants (P2): mmproj files were offered as quants (F32
// downloaded only a projector) and pulled in alongside matching model quants.
func TestProjectorsAreNotQuants(t *testing.T) {
	api := siblingsServer(t, "gemma-3-27b-it-BF16.gguf", "gemma-3-27b-it-Q4_K_M.gguf", "mmproj-BF16.gguf", "mmproj-F32.gguf")
	q, err := fetchQuants("unsloth/gemma-3-27b-it-GGUF", api)
	if err != nil || !reflect.DeepEqual(q, []string{"BF16", "Q4_K_M"}) {
		t.Fatalf("fetchQuants = %v, %v", q, err)
	}
	files, err := resolveQuantFiles(context.Background(), "unsloth/gemma-3-27b-it-GGUF", "BF16", api)
	if err != nil || !reflect.DeepEqual(files, []string{"gemma-3-27b-it-BF16.gguf"}) {
		t.Fatalf("resolveQuantFiles(BF16) = %v, %v; projector must not ride along", files, err)
	}
	if _, err := resolveQuantFiles(context.Background(), "unsloth/gemma-3-27b-it-GGUF", "F32", api); err == nil {
		t.Fatal("F32 (projector only) resolved to a download")
	}
}

// TestUDAndPlainQuantsStayDistinct: dropping the UD- prefix merged
// Model-UD-Q4_K_XL and Model-Q4_K_XL into one choice that downloaded both.
func TestUDAndPlainQuantsStayDistinct(t *testing.T) {
	api := siblingsServer(t, "M-UD-Q4_K_XL.gguf", "M-Q4_K_XL.gguf")
	q, _ := fetchQuants("o/n", api)
	if !reflect.DeepEqual(q, []string{"Q4_K_XL", "UD-Q4_K_XL"}) {
		t.Fatalf("fetchQuants = %v", q)
	}
	for quant, want := range map[string]string{"Q4_K_XL": "M-Q4_K_XL.gguf", "UD-Q4_K_XL": "M-UD-Q4_K_XL.gguf"} {
		files, err := resolveQuantFiles(context.Background(), "o/n", quant, api)
		if err != nil || !reflect.DeepEqual(files, []string{want}) {
			t.Errorf("resolveQuantFiles(%s) = %v, %v; want [%s]", quant, files, err, want)
		}
	}
}

func TestSingleFileRepoWithoutQuantStillDownloads(t *testing.T) {
	api := siblingsServer(t, "model.gguf", "mmproj-F16.gguf")
	files, err := resolveQuantFiles(context.Background(), "o/n", "", api)
	if err != nil || !reflect.DeepEqual(files, []string{"model.gguf"}) {
		t.Fatalf("resolveQuantFiles(no quant) = %v, %v", files, err)
	}
}

func TestScannersTagProjectors(t *testing.T) {
	hermeticCaches(t)
	snap := hubSnapshot("o/n")
	mustWriteFile(t, snap+"/M-Q4_K_M.gguf", 1)
	mustWriteFile(t, snap+"/mmproj-F16.gguf", 1)
	var model, proj bool
	for _, m := range scanHuggingfaceCache(huggingfaceHubDir()) {
		if m.FileName == "M-Q4_K_M.gguf" && !m.Projector {
			model = true
		}
		if m.FileName == "mmproj-F16.gguf" && m.Projector {
			proj = true
		}
	}
	if !model || !proj {
		t.Fatalf("model listed=%v, projector tagged=%v", model, proj)
	}
}
