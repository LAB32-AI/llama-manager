package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"strings"
	"testing"
)

func multipartReq(t *testing.T, url string, fields map[string]string, files map[string][]byte, headers map[string]string) (*http.Response, []byte) {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for k, v := range fields {
		_ = mw.WriteField(k, v)
	}
	for k, v := range files {
		name, _, _ := strings.Cut(k, "#")
		fw, _ := mw.CreateFormFile(name, "upload.bin")
		_, _ = fw.Write(v)
	}
	_ = mw.Close()
	req, _ := http.NewRequest("POST", url, &buf)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp, b
}

type imagesAnswer struct {
	Created int64 `json:"created"`
	Data    []struct {
		B64 *string `json:"b64_json"`
		URL *string `json:"url"`
	} `json:"data"`
	OutputFormat string `json:"output_format"`
}

func TestV1ImagesAnswersURLAndBase64(t *testing.T) {
	ir := newImageRig(t)
	resp, b := ir.do(t, "POST", "/v1/images/generations", map[string]any{"model": "dall-e-3", "prompt": "a cat", "size": "auto", "quality": "hd", "style": "vivid", "n": 1})
	var a imagesAnswer
	if resp.StatusCode != http.StatusOK || json.Unmarshal(b, &a) != nil || len(a.Data) != 1 || a.Data[0].B64 == nil || a.Data[0].URL == nil || a.OutputFormat != "png" {
		t.Fatalf("OpenAI's model name, no response_format: %d %s", resp.StatusCode, b)
	}
	ir.sd.mu.Lock()
	got := ir.sd.openai[0]
	ir.sd.mu.Unlock()
	if _, has := got["size"]; has || got["quality"] != nil || got["style"] != nil || got["n"] != float64(1) {
		t.Errorf("sd-server got OpenAI-only fields: %v", got)
	}
	setKeys(ir.ws, "sk-img")
	img, err := http.Get(*a.Data[0].URL)
	if err != nil {
		t.Fatal(err)
	}
	pic, _ := io.ReadAll(img.Body)
	img.Body.Close()
	if img.StatusCode != http.StatusOK || img.Header.Get("Content-Type") != "image/png" || !bytes.Equal(pic, tinyPNG()) ||
		!strings.Contains(img.Header.Get("Content-Security-Policy"), "sandbox") {
		t.Errorf("image link: %d %v", img.StatusCode, img.Header)
	}
	setKeys(ir.ws)

	_, b = ir.do(t, "POST", "/v1/images/generations", map[string]any{"prompt": "x", "response_format": "b64_json"})
	a = imagesAnswer{}
	if json.Unmarshal(b, &a) != nil || len(a.Data) != 1 || a.Data[0].B64 == nil || a.Data[0].URL != nil {
		t.Errorf("b64_json: %s", b)
	}
	_, b = ir.do(t, "POST", "/v1/images/generations", map[string]any{"prompt": "x", "response_format": "url"})
	a = imagesAnswer{}
	if json.Unmarshal(b, &a) != nil || len(a.Data) != 1 || a.Data[0].B64 != nil || a.Data[0].URL == nil {
		t.Errorf("url: %s", b)
	}
	resp, b = ir.do(t, "POST", "/v1/images/generations", map[string]any{"prompt": "x", "response_format": "jpg"})
	wantAPIError(t, "bad response_format", resp, b, http.StatusBadRequest, "")
	resp, b = ir.do(t, "POST", "/v1/images/generations", map[string]any{"model": "midjourney", "prompt": "x"})
	wantAPIError(t, "unknown image model", resp, b, http.StatusNotFound, "model_not_found")
}

func TestV1ImagesStream(t *testing.T) {
	ir := newImageRig(t)
	resp, b := ir.do(t, "POST", "/v1/images/generations", map[string]any{"model": "gpt-image-1", "prompt": "x", "stream": true, "partial_images": 2})
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	ev, data, _ := strings.Cut(strings.TrimSpace(string(b)), "\n")
	var e map[string]any
	if ev != "event: image_generation.completed" || json.Unmarshal([]byte(strings.TrimPrefix(data, "data: ")), &e) != nil ||
		e["type"] != "image_generation.completed" || e["b64_json"] == nil {
		t.Errorf("event: %s", b)
	}
	ir.sd.mu.Lock()
	got := ir.sd.openai[0]
	ir.sd.mu.Unlock()
	if got["stream"] != nil || got["partial_images"] != nil {
		t.Errorf("sd-server got stream fields: %v", got)
	}
}

func TestV1ImageEdits(t *testing.T) {
	ir := newImageRig(t, ImageModelConf{Model: "flux2-klein-4b"}, ImageModelConf{Model: "z-image-turbo"})
	url := ir.srv.URL + "/v1/images/edits"
	png := tinyPNG()
	resp, b := multipartReq(t, url, map[string]string{"model": "gpt-image-1", "prompt": "make it blue", "size": "auto", "response_format": "b64_json", "quality": "high"},
		map[string][]byte{"image[]#1": png, "image[]#2": png, "mask": png}, nil)
	var a imagesAnswer
	if resp.StatusCode != http.StatusOK || json.Unmarshal(b, &a) != nil || len(a.Data) != 1 || a.Data[0].B64 == nil {
		t.Fatalf("edit: %d %s", resp.StatusCode, b)
	}
	sd := ir.sds["flux2-klein-4b"]
	sd.mu.Lock()
	edits := append([]map[string]any(nil), sd.edits...)
	sd.mu.Unlock()
	if len(edits) != 1 || edits[0]["prompt"] != "make it blue" || edits[0]["images"] != 2 || edits[0]["mask"] != 1 || edits[0]["size"] != "" || edits[0]["fields"] != 1 {
		t.Errorf("sd-server got %v", edits)
	}
	resp, b = multipartReq(t, url, map[string]string{"model": "z-image-turbo", "prompt": "x"}, map[string][]byte{"image": png}, nil)
	if e := wantAPIError(t, "model that can't edit", resp, b, http.StatusBadRequest, "model_not_supported"); !strings.Contains(e.Error.Message, "flux2-klein-4b") {
		t.Errorf("should name the models that edit: %s", b)
	}
	resp, b = multipartReq(t, url, map[string]string{"prompt": "x"}, nil, nil)
	wantAPIError(t, "no image", resp, b, http.StatusBadRequest, "")
	resp, b = multipartReq(t, url, map[string]string{"prompt": "x"}, map[string][]byte{"image": []byte("%PDF-1.4 not a picture")}, nil)
	wantAPIError(t, "not a picture", resp, b, http.StatusBadRequest, "")
	resp, b = multipartReq(t, url, map[string]string{"prompt": "x"}, map[string][]byte{"image#1": png, "image#2": png, "image#3": png, "image#4": png, "image#5": png}, nil)
	wantAPIError(t, "too many pictures", resp, b, http.StatusBadRequest, "")
	// sd-server takes LoRA paths from the prompt as given: the edit route
	// must check them like a generation.
	resp, b = multipartReq(t, url, map[string]string{"prompt": `x <sd_cpp_extra_args>{"lora":[{"path":"/etc/passwd"}]}</sd_cpp_extra_args>`}, map[string][]byte{"image": png}, nil)
	wantAPIError(t, "LoRA path in the prompt", resp, b, http.StatusBadRequest, "")
	sd.mu.Lock()
	n := len(sd.edits)
	sd.mu.Unlock()
	if n != 1 {
		t.Errorf("refused edits reached sd-server: %d", n)
	}
	req, _ := http.NewRequest("POST", url, strings.NewReader(`{"prompt":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	r2, _ := http.DefaultClient.Do(req)
	b2, _ := io.ReadAll(r2.Body)
	r2.Body.Close()
	wantAPIError(t, "JSON edit", r2, b2, http.StatusBadRequest, "")
}

// TestV1ImageEditsPickAModelThatEdits: OpenAI's model names (and none) go
// to the first image model that edits, not just the first one.
func TestV1ImageEditsPickAModelThatEdits(t *testing.T) {
	ir := newImageRig(t, ImageModelConf{Model: "z-image-turbo"}, ImageModelConf{Model: "flux2-klein-4b"})
	for _, model := range []string{"gpt-image-1", "dall-e-2", ""} {
		resp, b := multipartReq(t, ir.srv.URL+"/v1/images/edits", map[string]string{"model": model, "prompt": "x"}, map[string][]byte{"image": tinyPNG()}, nil)
		if resp.StatusCode != http.StatusOK {
			t.Errorf("%q: %d %s", model, resp.StatusCode, b)
		}
	}
	sd := ir.sds["flux2-klein-4b"]
	sd.mu.Lock()
	n := len(sd.edits)
	sd.mu.Unlock()
	if n != 3 {
		t.Errorf("flux2-klein-4b got %d edits, want 3", n)
	}
	if resp, _ := ir.do(t, "POST", "/v1/images/generations", map[string]any{"model": "dall-e-3", "prompt": "x"}); resp.StatusCode != http.StatusOK {
		t.Errorf("generations still go to the first model: %d", resp.StatusCode)
	}
	ir.sds["z-image-turbo"].mu.Lock()
	gens := len(ir.sds["z-image-turbo"].openai)
	ir.sds["z-image-turbo"].mu.Unlock()
	if gens != 1 {
		t.Errorf("z-image-turbo got %d generations, want 1", gens)
	}
}

func TestMP3DurationSkipsID3(t *testing.T) {
	frame := append([]byte{0xFF, 0xFB, 0x90, 0x00}, make([]byte, 15996)...) // 128 kbps: one second
	// An ID3 tag with a 2-byte body (10 | 2 is 10, so the size must be added,
	// not or-ed) holding a frame sync; with the two bytes after it, a scan
	// starting inside the tag reads a 32 kbps frame.
	tag := []byte{'I', 'D', '3', 4, 0, 0, 0, 0, 0, 2, 0xFF, 0xFB, 0x10, 0x00}
	if d := mp3Duration(append(tag, frame...)); math.Abs(d-1) > 0.001 {
		t.Errorf("duration %v, want 1", d)
	}
}

func TestWavToPCM(t *testing.T) {
	if pcm, err := wavToPCM(testWAV(24000, 1000, 10), 24000); err != nil || len(pcm) != 20 || int16(binary.LittleEndian.Uint16(pcm)) != 1000 {
		t.Errorf("same rate: %v %v", pcm, err)
	}
	if pcm, err := wavToPCM(testWAV(48000, -2000, 100), 24000); err != nil || len(pcm) != 100 || int16(binary.LittleEndian.Uint16(pcm[10:])) != -2000 {
		t.Errorf("48 kHz: %d %v", len(pcm), err)
	}
	// 32-bit float, stereo, 12 kHz.
	var b bytes.Buffer
	le := binary.LittleEndian
	data := new(bytes.Buffer)
	for i := 0; i < 6; i++ {
		binary.Write(data, le, math.Float32bits(0.5))
		binary.Write(data, le, math.Float32bits(0))
	}
	b.WriteString("RIFF")
	binary.Write(&b, le, uint32(36+data.Len()))
	b.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), uint16(3), uint16(2), uint32(12000), uint32(12000 * 8), uint16(8), uint16(32)} {
		binary.Write(&b, le, v)
	}
	b.WriteString("data")
	binary.Write(&b, le, uint32(data.Len()))
	b.Write(data.Bytes())
	pcm, err := wavToPCM(b.Bytes(), 24000)
	if err != nil || len(pcm) != 24 || int16(le.Uint16(pcm)) != 8192 {
		t.Errorf("float stereo: %d %v %v", len(pcm), pcm[:4], err)
	}
	if _, err := wavToPCM([]byte("nope"), 24000); err == nil {
		t.Error("not a WAV")
	}
}

func TestV1SpeechFormats(t *testing.T) {
	sr := newSpeechRig(t, SpeechModelConf{Model: "lfm3-audio"})
	noFFmpeg(t)
	speak := func(extra map[string]any) (*http.Response, []byte) {
		body := map[string]any{"model": "tts-1", "input": "Hello there.", "voice": "alloy"}
		for k, v := range extra {
			body[k] = v
		}
		return sr.do(t, "POST", "/v1/audio/speech", body)
	}
	resp, b := speak(map[string]any{"response_format": "pcm"})
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "audio/pcm" || len(b) != 200 {
		t.Errorf("pcm: %d %q %d bytes", resp.StatusCode, resp.Header.Get("Content-Type"), len(b))
	}
	resp, b = speak(nil)
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "audio/wav" || !isWAV(b) {
		t.Errorf("default (mp3) without ffmpeg is WAV: %d %q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	resp, b = speak(map[string]any{"response_format": "mp3"})
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "audio/wav" || !isWAV(b) {
		t.Errorf("mp3 without ffmpeg falls back to WAV: %d %q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	resp, b = speak(map[string]any{"response_format": "pcm", "stream_format": "sse"})
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(b), "event: speech.audio.delta") || !strings.Contains(string(b), `"type":"speech.audio.done"`) {
		t.Errorf("sse: %d %s", resp.StatusCode, b)
	}
	for name, extra := range map[string]map[string]any{
		"format": {"response_format": "ogg"}, "slow": {"speed": 0.1}, "fast": {"speed": 4.5}, "stream_format": {"stream_format": "chunks"},
	} {
		resp, b := speak(extra)
		wantAPIError(t, name, resp, b, http.StatusBadRequest, "")
	}
	resp, b = sr.do(t, "POST", "/v1/audio/speech", map[string]any{"model": "lfm3-audio", "input": "Hi.", "voice": "nobody"})
	wantAPIError(t, "unknown voice", resp, b, http.StatusBadRequest, "")

	args := fakeFFmpeg(t, "ID3fake-mp3")
	resp, b = speak(map[string]any{"response_format": "mp3", "speed": 1.5})
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "audio/mpeg" || string(b) != "ID3fake-mp3" {
		t.Errorf("mp3 through ffmpeg: %d %q %q", resp.StatusCode, resp.Header.Get("Content-Type"), b)
	}
	if a := readFile(t, args); !strings.Contains(a, "atempo=1.5000") || !strings.Contains(a, "libmp3lame") || !strings.Contains(a, "pipe:1") {
		t.Errorf("ffmpeg args: %s", a)
	}
	resp, b = speak(nil)
	if resp.Header.Get("Content-Type") != "audio/mpeg" || !strings.Contains(readFile(t, args), "libmp3lame") {
		t.Errorf("the default is mp3, like OpenAI's: %q", resp.Header.Get("Content-Type"))
	}
	fakeFFmpeg(t, "")
	resp, _ = speak(map[string]any{"response_format": "opus"})
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "audio/wav" {
		t.Errorf("a failing ffmpeg falls back to WAV: %d %q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	if got := atempo(0.25); got != "atempo=0.5,atempo=0.5000" {
		t.Errorf("atempo(0.25) = %s", got)
	}
	if got := atempo(4); got != "atempo=2,atempo=2.0000" {
		t.Errorf("atempo(4) = %s", got)
	}
}

// audioRig is a running instance "c" whose fake llama-server hears audio
// (or not) and answers transcriptions with reply.
func audioRig(t *testing.T, hears bool, reply string) *v1Rig {
	t.Helper()
	return newV1Rig(t, func(w http.ResponseWriter, r *http.Request, body map[string]any) {
		switch r.URL.Path {
		case "/props":
			writeJSON(w, http.StatusOK, map[string]any{"total_slots": 1, "modalities": map[string]bool{"vision": false, "audio": hears}})
		case "/v1/chat/completions":
			if body["stream"] == true {
				sse(w, delta("content", " Hello"), delta("content", " there."), `{"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":3,"total_tokens":43}}`, "[DONE]")
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{"choices": []map[string]any{{"message": map[string]any{"content": reply}}},
				"usage": map[string]int{"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49}})
		default:
			http.NotFound(w, r)
		}
	})
}

func TestV1Transcriptions(t *testing.T) {
	vr := audioRig(t, true, "Hello there. How are you today?")
	url := vr.srv.URL + "/v1/audio/transcriptions"
	wav := testWAV(16000, 5, 16000*4) // 4 seconds
	resp, b := multipartReq(t, url, map[string]string{"model": "whisper-1", "language": "en", "prompt": "Greetings."}, map[string][]byte{"file": wav}, nil)
	var out struct {
		Text  string         `json:"text"`
		Usage map[string]any `json:"usage"`
	}
	if resp.StatusCode != http.StatusOK || json.Unmarshal(b, &out) != nil || out.Text != "Hello there. How are you today?" || out.Usage["total_tokens"] != float64(49) {
		t.Fatalf("json: %d %s", resp.StatusCode, b)
	}
	c := vr.lastCall(t)
	msgs, _ := c.body["messages"].([]any)
	parts, _ := msgs[0].(map[string]any)["content"].([]any)
	audio, _ := parts[0].(map[string]any)["input_audio"].(map[string]any)
	instr, _ := parts[1].(map[string]any)["text"].(string)
	data, _ := base64.StdEncoding.DecodeString(audio["data"].(string))
	if c.path != "/v1/chat/completions" || audio["format"] != "wav" || !bytes.Equal(data, wav) ||
		!strings.Contains(instr, "Transcribe") || !strings.Contains(instr, "English") || !strings.Contains(instr, "Greetings.") {
		t.Errorf("llama-server got %s %v %q", c.path, audio["format"], instr)
	}

	_, b = multipartReq(t, url, map[string]string{"response_format": "text"}, map[string][]byte{"file": wav}, nil)
	if string(b) != "Hello there. How are you today?\n" {
		t.Errorf("text: %q", b)
	}
	_, b = multipartReq(t, url, map[string]string{"response_format": "srt"}, map[string][]byte{"file": wav}, nil)
	if !strings.HasPrefix(string(b), "1\n00:00:00,000 --> 00:00:01,") || !strings.Contains(string(b), "2\n00:00:01,") || !strings.Contains(string(b), " --> 00:00:04,000\nHow are you today?") {
		t.Errorf("srt: %q", b)
	}
	_, b = multipartReq(t, url, map[string]string{"response_format": "vtt"}, map[string][]byte{"file": wav}, nil)
	if !strings.HasPrefix(string(b), "WEBVTT\n\n00:00:00.000 --> ") {
		t.Errorf("vtt: %q", b)
	}
	_, b = multipartReq(t, url, map[string]string{"response_format": "verbose_json"}, map[string][]byte{"file": wav}, nil)
	var v struct {
		Task     string           `json:"task"`
		Language string           `json:"language"`
		Duration float64          `json:"duration"`
		Segments []map[string]any `json:"segments"`
	}
	if json.Unmarshal(b, &v) != nil || v.Task != "transcribe" || v.Language != "english" || v.Duration != 4 || len(v.Segments) != 2 || v.Segments[1]["end"] != float64(4) {
		t.Errorf("verbose_json: %s", b)
	}

	_, b = multipartReq(t, vr.srv.URL+"/v1/audio/translations", nil, map[string][]byte{"file": wav}, nil)
	instr = ""
	if msgs, ok := vr.lastCall(t).body["messages"].([]any); ok {
		parts, _ := msgs[0].(map[string]any)["content"].([]any)
		instr, _ = parts[1].(map[string]any)["text"].(string)
	}
	if !strings.Contains(string(b), `"text":"Hello there.`) || !strings.Contains(instr, "into English") {
		t.Errorf("translation: %s %q", b, instr)
	}

	resp, b = multipartReq(t, url, map[string]string{"stream": "true"}, map[string][]byte{"file": wav}, nil)
	s := string(b)
	if resp.Header.Get("Content-Type") != "text/event-stream" || !strings.Contains(s, `{"delta":"Hello","type":"transcript.text.delta"}`) ||
		!strings.Contains(s, `"text":"Hello there."`) || !strings.Contains(s, `"total_tokens":43`) {
		t.Errorf("stream: %s", s)
	}

	resp, b = multipartReq(t, url, map[string]string{"response_format": "docx"}, map[string][]byte{"file": wav}, nil)
	wantAPIError(t, "bad format", resp, b, http.StatusBadRequest, "")
	resp, b = multipartReq(t, url, nil, nil, nil)
	wantAPIError(t, "no file", resp, b, http.StatusBadRequest, "")
	resp, b = multipartReq(t, url, nil, map[string][]byte{"file": []byte("just some text")}, nil)
	wantAPIError(t, "not audio", resp, b, http.StatusBadRequest, "")
	resp, b = multipartReq(t, url, nil, map[string][]byte{"file": append([]byte{0xFF}, make([]byte, maxAudioUpload+10)...)}, nil)
	wantAPIError(t, "too big", resp, b, http.StatusRequestEntityTooLarge, "request_too_large")
}

func TestV1TranscriptionFormatsAndModels(t *testing.T) {
	vr := audioRig(t, true, "Hi.")
	url := vr.srv.URL + "/v1/audio/transcriptions"
	// An MP3 (MPEG-1 Layer III, 128 kbps): 16000 bytes is one second.
	mp3 := append([]byte{0xFF, 0xFB, 0x90, 0x00}, make([]byte, 15996)...)
	ogg := append([]byte("OggS"), make([]byte, 100)...)
	noFFmpeg(t)
	resp, b := multipartReq(t, url, map[string]string{"response_format": "verbose_json"}, map[string][]byte{"file": mp3}, nil)
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(b), `"duration":1`) {
		t.Errorf("mp3 goes as is: %d %s", resp.StatusCode, b)
	}
	if c := vr.lastCall(t); !strings.Contains(c.body["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)["input_audio"].(map[string]any)["format"].(string), "mp3") {
		t.Errorf("format: %v", c.body)
	}
	resp, b = multipartReq(t, url, nil, map[string][]byte{"file": ogg}, nil)
	if e := wantAPIError(t, "ogg without ffmpeg", resp, b, http.StatusBadRequest, ""); !strings.Contains(e.Error.Message, "ffmpeg") {
		t.Errorf("should say why: %s", b)
	}
	args := fakeFFmpeg(t, "abcd")
	resp, b = multipartReq(t, url, nil, map[string][]byte{"file": ogg}, nil)
	if resp.StatusCode != http.StatusOK {
		t.Errorf("ogg through ffmpeg: %d %s", resp.StatusCode, b)
	}
	sent := vr.lastCall(t).body["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)["input_audio"].(map[string]any)
	wav, _ := base64.StdEncoding.DecodeString(sent["data"].(string))
	if _, data, err := wavChunks(wav); sent["format"] != "wav" || err != nil || len(data) != 4 || !strings.Contains(readFile(t, args), "-ar 16000") {
		t.Errorf("converted: %v %v %s", sent["format"], err, readFile(t, args))
	}

	resp, b = multipartReq(t, url, map[string]string{"model": "nope-not-here"}, map[string][]byte{"file": testWAV(16000, 1, 10)}, nil)
	if resp.StatusCode != http.StatusOK {
		t.Errorf("an unknown model name picks a model that hears: %d %s", resp.StatusCode, b)
	}

	deaf := audioRig(t, false, "x")
	resp, b = multipartReq(t, deaf.srv.URL+"/v1/audio/transcriptions", nil, map[string][]byte{"file": testWAV(16000, 1, 10)}, nil)
	wantAPIError(t, "no model hears", resp, b, http.StatusNotFound, "model_not_found")
	resp, b = multipartReq(t, deaf.srv.URL+"/v1/audio/transcriptions", map[string]string{"model": "c"}, map[string][]byte{"file": testWAV(16000, 1, 10)}, nil)
	wantAPIError(t, "named deaf model", resp, b, http.StatusBadRequest, "model_not_supported")
	if deaf.nCalls() != 1 {
		t.Errorf("a deaf model must only be asked for /props, once per run: %d calls", deaf.nCalls())
	}
}

func TestSniffAudio(t *testing.T) {
	for want, b := range map[string][]byte{
		"wav": testWAV(8000, 0, 1), "mp3": []byte("ID3\x04\x00\x00"), "flac": []byte("fLaC...."), "ogg": []byte("OggS...."),
		"webm": {0x1A, 0x45, 0xDF, 0xA3, 0}, "mp4": []byte("\x00\x00\x00\x20ftypM4A "), "aac": {0xFF, 0xF1, 0x50, 0x80},
	} {
		if got := sniffAudio(b); got != want {
			t.Errorf("%s: got %q", want, got)
		}
	}
	if sniffAudio([]byte("hello")) != "" {
		t.Error("text is not audio")
	}
}
