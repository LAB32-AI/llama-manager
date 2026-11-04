package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"
)

// testWAV is a mono 16-bit WAV of n samples, all v.
func testWAV(rate int, v int16, n int) []byte {
	var b bytes.Buffer
	le := binary.LittleEndian
	b.WriteString("RIFF")
	binary.Write(&b, le, uint32(36+2*n))
	b.WriteString("WAVEfmt ")
	binary.Write(&b, le, uint32(16))
	binary.Write(&b, le, uint16(1))
	binary.Write(&b, le, uint16(1))
	binary.Write(&b, le, uint32(rate))
	binary.Write(&b, le, uint32(rate*2))
	binary.Write(&b, le, uint16(2))
	binary.Write(&b, le, uint16(16))
	b.WriteString("data")
	binary.Write(&b, le, uint32(2*n))
	for i := 0; i < n; i++ {
		binary.Write(&b, le, v)
	}
	return b.Bytes()
}

// fakeTTS is enough of llama-server's /tts and audiocpp_server's
// /v1/audio/speech: each answer is a WAV whose samples count the requests.
type fakeTTS struct {
	mu     sync.Mutex
	old    bool // no /tts, like llama.cpp before v0.8.0
	paths  []string
	bodies []map[string]any
}

func (f *fakeTTS) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	switch {
	case r.URL.Path == "/health":
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
	case r.Method == http.MethodPost && (r.URL.Path == "/tts" || r.URL.Path == "/v1/audio/speech"):
		f.mu.Lock()
		if r.URL.Path == "/tts" && f.old {
			f.mu.Unlock()
			http.NotFound(w, r)
			return
		}
		f.paths = append(f.paths, r.URL.Path)
		f.bodies = append(f.bodies, body)
		n := len(f.bodies)
		f.mu.Unlock()
		w.Header().Set("Content-Type", "audio/wav")
		w.Write(testWAV(24000, int16(n), 100))
	default:
		http.NotFound(w, r)
	}
}

func (f *fakeTTS) calls() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]map[string]any(nil), f.bodies...)
}

func (f *fakeTTS) path(i int) string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.paths[i]
}

// runFakeAudioCpp serves fakeTTS on the address audiocpp_server's config names.
func runFakeAudioCpp(confPath string) {
	var c struct {
		Host string `json:"host"`
		Port int    `json:"port"`
	}
	b, err := os.ReadFile(confPath)
	if err == nil {
		err = json.Unmarshal(b, &c)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "fake audiocpp_server config:", err)
		os.Exit(1)
	}
	ln, err := net.Listen("tcp", net.JoinHostPort(c.Host, strconv.Itoa(c.Port)))
	if err != nil {
		fmt.Fprintln(os.Stderr, "fake audiocpp_server listen:", err)
		os.Exit(1)
	}
	_ = http.Serve(ln, &fakeTTS{})
}

func useSpeechConfDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	old := speechConfDir
	speechConfDir = func() (string, error) { return dir, nil }
	t.Cleanup(func() { speechConfDir = old })
	return dir
}

type speechRig struct {
	*chatAPIRig
	replyMu sync.Mutex
	reply   string
	fakes   map[string]*fakeTTS // by instance name
}

// newSpeechRig is a chat rig whose model "c" answers with sr.reply, next to
// running voice models backed by fakeTTS servers.
func newSpeechRig(t *testing.T, models ...SpeechModelConf) *speechRig {
	t.Helper()
	sr := &speechRig{fakes: map[string]*fakeTTS{}}
	sr.chatAPIRig = newChatAPIRig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		if body["stream"] != true {
			titleReply(w, "T")
			return
		}
		sr.replyMu.Lock()
		text := sr.reply
		sr.replyMu.Unlock()
		sse(w, delta("content", text), finalChunk, "[DONE]")
	})
	conf := SpeechConf{Enabled: true, ServerBin: "/bin/true"}
	byPort := map[int]*fakeTTS{}
	for _, m := range models {
		f := &fakeTTS{}
		up := httptest.NewServer(f)
		t.Cleanup(up.Close)
		m.Port = portOf(t, up.URL)
		byPort[m.Port] = f
		conf.Models = append(conf.Models, m)
	}
	ws := sr.ws
	ws.cfg.mu.Lock()
	ws.cfg.TextToSpeech = conf
	ws.cfg.mu.Unlock()
	for _, e := range conf.entries() {
		if err := ws.mgr.AddInstance(e.instanceConf()); err != nil {
			t.Fatal(err)
		}
		setState(ws.mgr.Get(e.Name), StateRunning)
		sr.fakes[e.Name] = byPort[e.Port]
	}
	ws.speech = newSpeechService(conf, ws.mgr)
	t.Cleanup(ws.speech.Close)
	return sr
}

// chatReply makes a chat whose one reply is text, and returns its ids.
func (sr *speechRig) chatReply(t *testing.T, text string) (string, string) {
	t.Helper()
	sr.replyMu.Lock()
	sr.reply = text
	sr.replyMu.Unlock()
	c := sr.newChat(t, `{}`)
	if resp, b := sr.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "hi"}); resp.StatusCode != http.StatusCreated {
		t.Fatalf("send: %d %s", resp.StatusCode, b)
	}
	sr.waitReply(t, c.ID)
	d := sr.detail(t, c.ID)
	return c.ID, d.Messages[len(d.Messages)-1].ID
}

func (sr *speechRig) speak(t *testing.T, chatID, msgID string, body map[string]any) (*http.Response, []byte) {
	t.Helper()
	return sr.do(t, "POST", "/api/chats/"+chatID+"/messages/"+msgID+"/speech", body)
}

const longReply = "The first sentence is short. The second one says a little more about the topic. " +
	"A third sentence follows, so the first part fills up. Here is a fourth sentence with some more words in it. " +
	"The fifth sentence keeps going, because a real answer is often long. Sentence six adds yet another idea to the reply. " +
	"Seven is here too, with a number like 3.14 inside it. The eighth sentence is nearly the last one. " +
	"And the ninth one ends the paragraph. A tenth sentence makes sure there are several parts to play."

func TestSpeechText(t *testing.T) {
	in := "<think>plan it</think># Setup\n\nRun this:\n\n```bash\ngo build\n```\n\n" +
		"- **First**, install [Go](https://go.dev).\n- Then see https://example.com/x for `make` details\n\n" +
		"| Model | Size |\n|---|---|\n| Ming | 1.7 GB |\n\n> Note: it's *fast*.\n\nA line that wraps\nonto the next one.\n\n---\n![pic](a.png)"
	want := "Setup.\nRun this:\nFirst, install Go.\nThen see for make details.\nModel, Size.\nMing, 1.7 GB.\nNote: it's fast.\nA line that wraps onto the next one."
	if got := speechText(in); got != want {
		t.Errorf("speechText:\n got %q\nwant %q", got, want)
	}
	if got := speechText("```\nonly code\n```"); got != "" {
		t.Errorf("a reply of code only: %q", got)
	}
	if got := speechText("Answer.<think>never closed"); got != "Answer." {
		t.Errorf("unclosed reasoning: %q", got)
	}
}

func TestSpeechParts(t *testing.T) {
	parts := speechParts(longReply, speechFirstPart, speechPartRunes)
	if len(parts) < 2 {
		t.Fatalf("expected several parts, got %q", parts)
	}
	for i, p := range parts {
		limit := speechPartRunes
		if i == 0 {
			limit = speechFirstPart
		}
		if n := utf8.RuneCountInString(p); n > limit || n == 0 {
			t.Errorf("part %d has %d runes (limit %d): %q", i, n, limit, p)
		}
	}
	if got := strings.Join(parts, " "); got != longReply {
		t.Errorf("text lost while splitting:\n got %q\nwant %q", got, longReply)
	}
	if !strings.Contains(strings.Join(parts, "|"), "3.14 inside") {
		t.Errorf("a decimal point split a sentence: %q", parts)
	}
	long := strings.Repeat("word ", 300)
	for _, p := range speechParts(long, 100, 200) {
		if n := utf8.RuneCountInString(p); n > 200 || strings.HasPrefix(p, " ") {
			t.Errorf("long sentence cut badly: %d runes %q", n, p)
		}
	}
	if got := sentences("你好。今天很好。"); !slices.Equal(got, []string{"你好。", "今天很好。"}) {
		t.Errorf("CJK sentences: %q", got)
	}
}

func TestDetectLang(t *testing.T) {
	for text, want := range map[string]string{
		"The model is loaded and the reply is ready for you.":              "en",
		"Модель загружена, и ответ готов.":                                 "ru",
		"Die Antwort ist fertig und das Modell ist nicht langsam.":         "de",
		"La réponse est prête et le modèle est rapide pour vous.":          "fr",
		"La respuesta está lista y el modelo es rápido para los usuarios.": "es",
		"模型已经加载，回答准备好了。":                                                   "zh",
		"モデルは読み込まれました。":                                                    "ja",
		"모델이 로드되었습니다.":                                                     "ko",
		"OK":                                                               "en",
	} {
		if got := detectLang(text); got != want {
			t.Errorf("detectLang(%q) = %s, want %s", text, got, want)
		}
	}
}

func TestJoinWAVs(t *testing.T) {
	a, b := testWAV(24000, 1, 3), testWAV(24000, 2, 5)
	out, err := joinWAVs([][]byte{a, b})
	if err != nil {
		t.Fatal(err)
	}
	fc, data, err := wavChunks(out)
	if err != nil || len(fc) != 16 || len(data) != 16 {
		t.Fatalf("joined: fmt %d bytes, data %d bytes, %v", len(fc), len(data), err)
	}
	if got := binary.LittleEndian.Uint32(out[4:8]); int(got) != len(out)-8 {
		t.Errorf("RIFF size %d for a %d byte file", got, len(out))
	}
	if data[0] != 1 || data[6] != 2 {
		t.Errorf("samples out of order: %v", data)
	}
	if _, err := joinWAVs([][]byte{a, testWAV(44100, 1, 3)}); err == nil {
		t.Error("joined WAVs with different sample rates")
	}
	streamed := append([]byte(nil), a...)
	binary.LittleEndian.PutUint32(streamed[40:44], 0xFFFFFFFF)
	if _, data, err := wavChunks(streamed); err != nil || len(data) != 6 {
		t.Errorf("a streamed WAV's data: %d bytes, %v", len(data), err)
	}
	if isWAV([]byte("RIFF....AVI ")) {
		t.Error("an AVI counted as WAV")
	}
}

func TestSpeechConfigRules(t *testing.T) {
	voice := []SpeechVoiceConf{{Name: "me", File: "/voices/me.wav"}}
	for name, c := range map[string]struct {
		conf SpeechConf
		want string
	}{
		"unknown model":       {SpeechConf{Enabled: true, Model: "nope"}, `model "nope" is unknown`},
		"ming needs a binary": {SpeechConf{Enabled: true, Model: "ming3-tts"}, "server_bin is required for ming3-tts"},
		"qwen needs a voice":  {SpeechConf{Enabled: true, Model: "qwen3-tts"}, "list at least one under voices"},
		"own voices":          {SpeechConf{Enabled: true, Model: "lfm3-audio", Voices: voice}, "has its own voices"},
		"relative voice":      {SpeechConf{Enabled: true, Model: "qwen3-tts", Voices: []SpeechVoiceConf{{Name: "me", File: "me.wav"}}}, "absolute path"},
		"bad voice name":      {SpeechConf{Enabled: true, Model: "qwen3-tts", Voices: []SpeechVoiceConf{{Name: "../x", File: "/x.wav"}}}, `voice name "../x"`},
		"two GPUs":            {SpeechConf{Enabled: true, Model: "lfm3-audio", GPUIDs: []int{0, 1}}, "lists one GPU"},
		"bad backend":         {SpeechConf{Enabled: true, Model: "lfm3-audio", Backend: "opencl"}, "backend"},
		"both forms":          {SpeechConf{Enabled: true, Model: "lfm3-audio", Models: []SpeechModelConf{{Model: "lfm3-audio"}}}, "not both"},
		"same port":           {SpeechConf{Enabled: true, Models: []SpeechModelConf{{Model: "lfm3-audio", Port: 9200}, {Model: "qwen3-tts", Port: 9200, Voices: voice}}}, "port 9200 is used twice"},
	} {
		err := c.conf.check()
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want %q", name, err, c.want)
		}
	}
	ok := SpeechConf{Enabled: true, ServerBin: "/opt/audiocpp_server", Models: []SpeechModelConf{{Model: "ming3-tts"}, {Model: "qwen3-tts", Voices: voice}, {Model: "lfm3-audio"}}}
	if err := ok.check(); err != nil {
		t.Errorf("a good section: %v", err)
	}
	es := ok.entries()
	if es[0].Name != "ming3-tts" || es[0].Port != defaultSpeechPort || es[2].Port != defaultSpeechPort+2 || es[0].Wait != defaultSpeechWait {
		t.Errorf("defaults: %+v", es)
	}
	if v, _ := es[1].voice(""); v.ID != "me" || v.file != "/voices/me.wav" {
		t.Errorf("the configured voice: %+v", v)
	}

	cfg := newTestConfig(t)
	cfg.ImageRecognition = ImageRecConf{Enabled: true}
	cfg.TextToSpeech = SpeechConf{Enabled: true, Model: "lfm3-audio", Port: defaultReaderPort}
	if err := cfg.validate(); err == nil || !strings.Contains(err.Error(), "image reader's port") {
		t.Errorf("a port shared with the image reader: %v", err)
	}
	cfg.TextToSpeech.Port = 0
	cfg.Instances = []InstanceConf{{Name: "lfm3-audio", Model: "/m.gguf", Port: 9090, GPUIDs: []int{0}}}
	if err := cfg.validate(); err == nil || !strings.Contains(err.Error(), "belongs to text to speech") {
		t.Errorf("an instance named like a voice model: %v", err)
	}
	cfg.Instances = nil
	if err := cfg.AddInstance(InstanceConf{Name: "x", Model: "/m.gguf", Port: defaultSpeechPort, GPUIDs: []int{0}}); err == nil {
		t.Error("an instance took a voice model's port")
	}
}

func TestSpeechLaunchAudioCpp(t *testing.T) {
	t.Setenv("HF_HUB_CACHE", t.TempDir())
	dir := useSpeechConfDir(t)
	cfg := newTestConfig(t)
	cfg.GPUBackend = "rocm"
	cfg.TextToSpeech = SpeechConf{Enabled: true, ServerBin: "/opt/audiocpp_server", Model: "ming3-tts", GPUIDs: []int{3}, ExtraArgs: []string{"--threads", "4"}}
	e := cfg.TextToSpeech.entries()[0]
	if _, err := buildLaunch(e.instanceConf(), cfg); err != errSpeechFileMissing {
		t.Fatalf("launch without the model file: %v", err)
	}
	f := *e.Preset.File
	placeFile(t, f, f.Rev, f.Size)
	spec, err := buildLaunch(e.instanceConf(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	conf := filepath.Join(dir, "ming3-tts.json")
	if spec.bin != "/opt/audiocpp_server" || spec.host != "127.0.0.1" || !slices.Equal(spec.args, []string{"--config", conf, "--no-ui", "--log", "--threads", "4"}) {
		t.Errorf("command line: %s %q (host %s)", spec.bin, spec.args, spec.host)
	}
	if !slices.Contains(spec.env, "HIP_VISIBLE_DEVICES=3") {
		t.Errorf("env: %q", spec.env)
	}
	b, err := os.ReadFile(conf)
	if err != nil {
		t.Fatal(err)
	}
	if fi, _ := os.Stat(conf); fi.Mode().Perm() != 0o600 {
		t.Errorf("config file mode %v", fi.Mode().Perm())
	}
	var got struct {
		Host    string `json:"host"`
		Port    int    `json:"port"`
		Backend string `json:"backend"`
		Models  []struct {
			ID, Family, Path, Task, Mode string
			Options                      map[string]any `json:"default_request_options"`
		} `json:"models"`
	}
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	local, _ := f.localPath()
	m := got.Models[0]
	if got.Host != "127.0.0.1" || got.Port != defaultSpeechPort || got.Backend != "hip" || m.ID != "ming3-tts" || m.Family != "ming3_tts" ||
		m.Path != local || m.Task != "tts" || m.Mode != "offline" || m.Options["template_name"] != "ip" {
		t.Errorf("audiocpp_server config:\n%s", b)
	}
	cfg.TextToSpeech.Backend = "vulkan"
	buildLaunch(e.instanceConf(), cfg)
	if b, _ := os.ReadFile(conf); !strings.Contains(string(b), `"backend": "vulkan"`) {
		t.Errorf("backend override ignored:\n%s", b)
	}
	cfg.TextToSpeech.Backend = "cpu"
	if spec, _ := buildLaunch(e.instanceConf(), cfg); len(spec.env) != 0 {
		t.Errorf("a CPU run still picks GPUs: %q", spec.env)
	}
}

func TestSpeechLaunchLlamaServer(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ExtraArgs = []string{"--model-draft", "/d.gguf"}
	cfg.TextToSpeech = SpeechConf{Enabled: true, Model: "lfm3-audio", GPUIDs: []int{4}}
	ic := cfg.TextToSpeech.entries()[0].instanceConf()
	if !ic.AutoStart || ic.role != roleSpeech {
		t.Errorf("instance: %+v", ic)
	}
	spec, err := buildLaunch(ic, cfg)
	if err != nil {
		t.Fatal(err)
	}
	line := strings.Join(spec.args, " ")
	for _, want := range []string{"-hf LiquidAI/LFM3-Audio-2B-GGUF:Q8_0", "--port 9110", "-c 4096", "-ctk f16", "-ctv f16", "--jinja", "-np 1"} {
		if !strings.Contains(line, want) {
			t.Errorf("missing %q in %s", want, line)
		}
	}
	if strings.Contains(line, "--model-draft") || spec.bin != "/bin/true" {
		t.Errorf("a voice model got the chat models' flags or binary: %s %s", spec.bin, line)
	}
}

func TestChatReadsRepliesAloud(t *testing.T) {
	sr := newSpeechRig(t, SpeechModelConf{Model: "lfm3-audio"})
	chatID, msgID := sr.chatReply(t, longReply)
	parts := speechParts(longReply, speechFirstPart, speechPartRunes)

	resp, b := sr.speak(t, chatID, msgID, map[string]any{"part": 0})
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "audio/wav" || !isWAV(b) {
		t.Fatalf("part 0: %d %s %q", resp.StatusCode, resp.Header.Get("Content-Type"), b)
	}
	if resp.Header.Get("X-Speech-Parts") != strconv.Itoa(len(parts)) || resp.Header.Get("X-Speech-Lang") != "en" {
		t.Errorf("headers: %v", resp.Header)
	}
	if resp.Header.Get("X-Content-Type-Options") != "nosniff" {
		t.Error("audio served without nosniff")
	}
	fake := sr.fakes["lfm3-audio"]
	got := fake.calls()[0]
	if got["input"] != parts[0] || got["lang"] != "en" || got["voice"] != "female" || got["n_predict"] != float64(1024) || got["response_format"] != "wav" {
		t.Errorf("/tts request: %v", got)
	}
	if p := fake.path(0); p != "/tts" {
		t.Errorf("asked %s instead of /tts", p)
	}
	if resp, _ := sr.speak(t, chatID, msgID, map[string]any{"part": 1, "voice": "male"}); resp.StatusCode != http.StatusOK {
		t.Fatalf("part 1: %d", resp.StatusCode)
	}
	if got := fake.calls()[1]; got["input"] != parts[1] || got["voice"] != "male" {
		t.Errorf("part 1 request: %v", got)
	}
	if resp, again := sr.speak(t, chatID, msgID, map[string]any{"part": 0}); resp.StatusCode != http.StatusOK || !bytes.Equal(again, b) || len(fake.calls()) != 2 {
		t.Errorf("playing again should come from the cache: %d, %d calls", resp.StatusCode, len(fake.calls()))
	}
	for name, c := range map[string]struct {
		body   map[string]any
		status int
		want   string
	}{
		"unknown voice": {map[string]any{"voice": "robot"}, http.StatusBadRequest, "has no voice"},
		"no such part":  {map[string]any{"part": 99}, http.StatusBadRequest, "doesn't exist"},
		"unknown model": {map[string]any{"model": "nope"}, http.StatusNotFound, "no voice model"},
		"language":      {map[string]any{"lang": "ru"}, http.StatusBadRequest, "doesn't speak Russian"},
	} {
		resp, b := sr.speak(t, chatID, msgID, c.body)
		if resp.StatusCode != c.status || !strings.Contains(string(b), c.want) {
			t.Errorf("%s: %d %s", name, resp.StatusCode, b)
		}
	}
	if resp, _ := sr.do(t, "POST", "/api/chats/"+chatID+"/messages/nope/speech", map[string]any{}); resp.StatusCode != http.StatusNotFound {
		t.Errorf("an unknown message: %d", resp.StatusCode)
	}

	codeID, codeMsg := sr.chatReply(t, "```go\nfmt.Println(1)\n```")
	if resp, b := sr.speak(t, codeID, codeMsg, map[string]any{}); resp.StatusCode != http.StatusUnprocessableEntity || !strings.Contains(string(b), "nothing to read aloud") {
		t.Errorf("a reply of code only: %d %s", resp.StatusCode, b)
	}
	ruID, ruMsg := sr.chatReply(t, "Модель загружена, и ответ готов.")
	if resp, b := sr.speak(t, ruID, ruMsg, map[string]any{}); resp.StatusCode != http.StatusUnprocessableEntity || !strings.Contains(string(b), "doesn't speak Russian") {
		t.Errorf("Russian for a model without it: %d %s", resp.StatusCode, b)
	}
}

func TestChatSpeechClonesTheConfiguredVoice(t *testing.T) {
	clip := filepath.Join(t.TempDir(), "me.wav")
	os.WriteFile(clip, testWAV(16000, 7, 50), 0o644)
	sr := newSpeechRig(t, SpeechModelConf{Model: "qwen3-tts", Voices: []SpeechVoiceConf{{Name: "me", File: clip}}})
	chatID, msgID := sr.chatReply(t, "Модель загружена, и ответ готов.")
	resp, b := sr.speak(t, chatID, msgID, map[string]any{"model": "qwen3-tts"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	got := sr.fakes["qwen3-tts"].calls()[0]
	if got["speaker_ref_b64"] != base64.StdEncoding.EncodeToString(testWAV(16000, 7, 50)) || got["lang"] != "ru" || got["voice"] != nil {
		t.Errorf("/tts request: lang %v voice %v ref %d chars", got["lang"], got["voice"], len(fmt.Sprint(got["speaker_ref_b64"])))
	}
	os.Remove(clip)
	otherID, otherMsg := sr.chatReply(t, "Другой ответ, тоже на русском.")
	if resp, b := sr.speak(t, otherID, otherMsg, map[string]any{}); resp.StatusCode != http.StatusInternalServerError || !strings.Contains(string(b), "reading the voice me") {
		t.Errorf("a missing voice file: %d %s", resp.StatusCode, b)
	}
}

func TestChatSpeechWithAudioCpp(t *testing.T) {
	sr := newSpeechRig(t, SpeechModelConf{Model: "ming3-tts"})
	chatID, msgID := sr.chatReply(t, "Hello there. The answer is ready.")
	if resp, b := sr.speak(t, chatID, msgID, map[string]any{"voice": "UK_George"}); resp.StatusCode != http.StatusOK {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	f := sr.fakes["ming3-tts"]
	got := f.calls()[0]
	opts, _ := got["options"].(map[string]any)
	if p := f.path(0); p != "/v1/audio/speech" || got["model"] != "ming3-tts" || got["input"] != "Hello there. The answer is ready." ||
		opts["voice_id"] != "UK_George" || opts["language"] != "en" {
		t.Errorf("speech request to %s: %v", p, got)
	}
}

func TestSpeechNeedsANewLlamaServer(t *testing.T) {
	sr := newSpeechRig(t, SpeechModelConf{Model: "lfm3-audio"})
	f := sr.fakes["lfm3-audio"]
	f.mu.Lock()
	f.old = true
	f.mu.Unlock()
	chatID, msgID := sr.chatReply(t, "Hello.")
	if resp, b := sr.speak(t, chatID, msgID, map[string]any{}); resp.StatusCode != http.StatusBadGateway || !strings.Contains(string(b), "v0.8.0") {
		t.Errorf("an old llama-server: %d %s", resp.StatusCode, b)
	}
	setState(sr.ws.mgr.Get("lfm3-audio"), StateStopped)
	if resp, b := sr.speak(t, chatID, msgID, map[string]any{}); resp.StatusCode != http.StatusServiceUnavailable || !strings.Contains(string(b), "is stopped") {
		t.Errorf("a stopped voice model: %d %s", resp.StatusCode, b)
	}
}

func TestV1SpeechJoinsTheParts(t *testing.T) {
	sr := newSpeechRig(t, SpeechModelConf{Model: "lfm3-audio"}, SpeechModelConf{Model: "ming3-tts"})
	resp, b := sr.do(t, "POST", "/v1/audio/speech", map[string]any{"model": "tts-1", "input": longReply, "voice": "alloy"})
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "audio/wav" {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	calls := sr.fakes["lfm3-audio"].calls()
	n := len(speechParts(longReply, speechFirstPart, speechPartRunes))
	if len(calls) != n || calls[0]["voice"] != "female" {
		t.Fatalf("%d requests for %d parts: %v", len(calls), n, calls)
	}
	if _, data, err := wavChunks(b); err != nil || len(data) != n*200 {
		t.Errorf("joined audio: %d bytes of samples, %v", len(data), err)
	}
	if resp, _ := sr.do(t, "POST", "/v1/audio/speech", map[string]any{"model": "ming3-tts", "input": "Hi.", "voice": "US_James"}); resp.StatusCode != http.StatusOK || len(sr.fakes["ming3-tts"].calls()) != 1 {
		t.Errorf("by model name: %d", resp.StatusCode)
	}
	for name, body := range map[string]map[string]any{
		"empty":    {"input": " "},
		"too long": {"input": strings.Repeat("a", maxSpeechInput+1)},
	} {
		if resp, b := sr.do(t, "POST", "/v1/audio/speech", body); resp.StatusCode != http.StatusBadRequest {
			t.Errorf("%s: %d %s", name, resp.StatusCode, b)
		}
	}
	ws, _ := newTestServer(t)
	rr := httptest.NewRecorder()
	ws.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/v1/audio/speech", strings.NewReader(`{"input":"hi"}`)))
	if rr.Code != http.StatusNotFound {
		t.Errorf("with text to speech off: %d", rr.Code)
	}
}

func TestSpeechStatusAndChatRules(t *testing.T) {
	sr := newSpeechRig(t, SpeechModelConf{Model: "lfm3-audio"})
	var st ServerStatus
	_, b := sr.do(t, "GET", "/api/status", nil)
	json.Unmarshal(b, &st)
	if len(st.Speech) != 1 || st.Speech[0].Instance != "lfm3-audio" || st.Speech[0].State != StateRunning ||
		len(st.Speech[0].Voices) != 2 || st.Speech[0].Langs[0] != "en" || st.Speech[0].Engine != engineLlama {
		t.Errorf("status: %s", b)
	}
	c := sr.newChat(t, `{}`)
	if resp, b := sr.do(t, "POST", "/api/chats/"+c.ID+"/messages", map[string]any{"content": "hi", "instance": "lfm3-audio"}); resp.StatusCode != http.StatusConflict || !strings.Contains(string(b), "can't chat") {
		t.Errorf("chatting with a voice model: %d %s", resp.StatusCode, b)
	}
	if resp, b := sr.do(t, "POST", "/v1/chat/completions", map[string]any{"model": "lfm3-audio", "messages": []any{}}); resp.StatusCode != http.StatusBadRequest {
		t.Errorf("/v1/chat/completions with a voice model: %d %s", resp.StatusCode, b)
	}
	if m := sr.ws.mgr.Get("lfm3-audio").FetchMetrics(); m != nil {
		t.Error("metrics fetched from a voice model")
	}
}

func TestSpeechCannotBeChangedOverTheAPI(t *testing.T) {
	cfg := newTestConfig(t)
	section := "text_to_speech:\n  enabled: true\n  model: lfm3-audio\n"
	head := "server_bin: /bin/true\nallowed_hosts: [example.com]\n"
	cfg.TextToSpeech = SpeechConf{Enabled: true, Model: "lfm3-audio"}
	os.WriteFile(cfg.path, []byte(head+section), 0o640)
	ws := NewWebServer(NewManager(cfg), cfg, NewDownloadManager())
	t.Cleanup(ws.speech.Close)
	for _, m := range []string{http.MethodPut, http.MethodDelete} {
		rr := httptest.NewRecorder()
		ws.ServeHTTP(rr, httptest.NewRequest(m, "/api/config/instances/lfm3-audio", strings.NewReader(`{"name":"lfm3-audio","model":"evil/x","port":9110,"gpu_ids":[0]}`)))
		if rr.Code != http.StatusConflict {
			t.Errorf("%s lfm3-audio: %d %s", m, rr.Code, rr.Body)
		}
	}
	if rr := postImport(t, ws, head+section+"instances:\n  - {name: a, model: /m.gguf, port: 9090, gpu_ids: [0]}\n"); rr.Code != http.StatusOK {
		t.Fatalf("import with the same section: %d %s", rr.Code, rr.Body)
	}
	if ws.mgr.Get("lfm3-audio") == nil || ws.mgr.Get("a") == nil {
		t.Error("an import must keep the voice model and add the new instance")
	}
	for name, in := range map[string]string{
		"voice file": head + section + "  voices: [{name: x, file: /etc/shadow}]\n",
		"dropped":    head,
	} {
		if rr := postImport(t, ws, in); rr.Code != http.StatusForbidden {
			t.Errorf("%s: %d %s", name, rr.Code, rr.Body)
		}
	}
}

func TestSpeechDownloadsLaunchesAndSpeaks(t *testing.T) {
	useSpeechConfDir(t)
	t.Setenv("HF_HUB_CACHE", t.TempDir())
	model := []byte("a tiny voice model")
	hf := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/o/voice/resolve/r1/v.gguf" {
			http.NotFound(w, r)
			return
		}
		w.Write(model)
	}))
	t.Cleanup(hf.Close)
	old := hfResolveBase
	hfResolveBase = hf.URL
	t.Cleanup(func() { hfResolveBase = old })
	speechPresets["test-tts"] = speechPreset{Label: "Test Voice", Ref: "o/voice:Q8_0", License: "MIT", Engine: engineAudioCpp,
		File:   &imageFile{Repo: "o/voice", Rev: "r1", Path: "v.gguf", Size: int64(len(model)), SHA256: sha(model)},
		Family: "test", Voices: []speechVoice{{ID: "v1", Label: "One"}}, Langs: []string{"en"}}
	t.Cleanup(func() { delete(speechPresets, "test-tts") })
	oldPoll := startingPollInterval
	startingPollInterval = 20 * time.Millisecond
	t.Cleanup(func() { startingPollInterval = oldPoll })

	cfg := newFakeRig(t, "healthy").cfg
	cfg.TextToSpeech = SpeechConf{Enabled: true, ServerBin: fakeServerBin(t), Model: "test-tts", Port: freePort(t), Backend: "cpu"}
	mgr := NewManager(cfg)
	t.Cleanup(mgr.Shutdown)
	s := newSpeechService(cfg.TextToSpeech, mgr)
	t.Cleanup(s.Close)
	s.Boot()
	inst := mgr.Get("test-tts")
	waitFor(t, 10*time.Second, "the voice model to download and run", func() bool { return inst.State() == StateRunning })
	if st := s.Status()[0]; st.State != StateRunning || st.Download != nil || st.Model != "Test Voice" {
		t.Errorf("status: %+v", st)
	}
	if logs := strings.Join(inst.Logs(), "\n"); !strings.Contains(logs, "fetched v.gguf") {
		t.Errorf("the download isn't in the instance's log:\n%s", logs)
	}
	e, _ := s.pick("")
	v, _ := e.voice("")
	wav, err := s.speak(context.Background(), e, v, "en", "Hello from the test.")
	if err != nil || !isWAV(wav) {
		t.Fatalf("speaking through the launched server: %v", err)
	}
	if s.fetch("test-tts") {
		t.Error("a file already on disk was fetched again")
	}
}
