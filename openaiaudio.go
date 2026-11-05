package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const (
	maxAudioUpload = 25 << 20 // OpenAI's limit for /v1/audio/transcriptions
	pcmRate        = 24000    // OpenAI's "pcm": 24 kHz, 16-bit, mono
)

var speechMIME = map[string]string{"mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac", "wav": "audio/wav", "pcm": "audio/pcm"}

// ffmpegPath is ffmpeg on PATH, or "" when it isn't installed. It turns WAV
// into OpenAI's other speech formats and audio files into WAV for the
// models; without it speech answers WAV and transcription takes WAV or MP3.
var ffmpegPath = sync.OnceValue(func() string {
	p, _ := exec.LookPath("ffmpeg")
	return p
})

var errNoFFmpeg = errors.New("ffmpeg isn't installed")

var warnFFmpegOnce sync.Once

// runFFmpeg converts in (written to a private temp file, so formats that need
// seeking work) with args, and returns what ffmpeg writes to stdout.
func runFFmpeg(ctx context.Context, in []byte, args ...string) ([]byte, error) {
	bin := ffmpegPath()
	if bin == "" {
		return nil, errNoFFmpeg
	}
	f, err := os.CreateTemp("", "llama-manager-audio-*")
	if err != nil {
		return nil, err
	}
	defer os.Remove(f.Name())
	if _, err := f.Write(in); err != nil {
		f.Close()
		return nil, err
	}
	if err := f.Close(); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, append(append([]string{"-hide_banner", "-loglevel", "error", "-nostdin", "-i", f.Name()}, args...), "pipe:1")...)
	var out, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg: %v: %s", err, truncateRunes(strings.TrimSpace(stderr.String()), 300))
	}
	return out.Bytes(), nil
}

// atempo is ffmpeg's tempo filter for speed; one atempo takes 0.5 to 2.
func atempo(speed float64) string {
	var parts []string
	for speed > 2 {
		parts, speed = append(parts, "atempo=2"), speed/2
	}
	for speed < 0.5 {
		parts, speed = append(parts, "atempo=0.5"), speed/0.5
	}
	return strings.Join(append(parts, "atempo="+strconv.FormatFloat(speed, 'f', 4, 64)), ",")
}

// encodeSpeech turns a WAV into the format asked for. WAV and PCM need no
// ffmpeg; the others (and speed) do, and without it the WAV is sent as is.
func encodeSpeech(ctx context.Context, wav []byte, format string, speed float64) ([]byte, string) {
	if speed == 1 {
		switch format {
		case "wav":
			return wav, speechMIME["wav"]
		case "pcm":
			if pcm, err := wavToPCM(wav, pcmRate); err == nil {
				return pcm, speechMIME["pcm"]
			}
		}
	}
	var args []string
	if speed != 1 {
		args = append(args, "-filter:a", atempo(speed))
	}
	switch format {
	case "mp3":
		args = append(args, "-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3")
	case "opus":
		args = append(args, "-c:a", "libopus", "-b:a", "64k", "-f", "ogg")
	case "aac":
		args = append(args, "-c:a", "aac", "-b:a", "128k", "-f", "adts")
	case "flac":
		args = append(args, "-c:a", "flac", "-f", "flac")
	default: // wav and pcm with a speed change
		args = append(args, "-c:a", "pcm_s16le", "-ar", strconv.Itoa(pcmRate), "-ac", "1", "-f", "s16le")
	}
	out, err := runFFmpeg(ctx, wav, args...)
	switch {
	case err == nil && format == "wav":
		return pcmToWAV(out, pcmRate), speechMIME["wav"]
	case err == nil:
		return out, speechMIME[format]
	case errors.Is(err, errNoFFmpeg):
		warnFFmpegOnce.Do(func() {
			log.Printf("ffmpeg isn't installed: /v1/audio/speech answers WAV for mp3, opus, aac and flac, and ignores speed")
		})
	default:
		log.Printf("/v1/audio/speech: %v; sending WAV", err)
	}
	if format == "pcm" {
		if pcm, err := wavToPCM(wav, pcmRate); err == nil {
			return pcm, speechMIME["pcm"]
		}
	}
	return wav, speechMIME["wav"]
}

// wavToPCM turns a WAV (8/16/24/32-bit integer or 32-bit float, any rate and
// channel count) into 16-bit mono PCM at rate.
func wavToPCM(wav []byte, rate int) ([]byte, error) {
	fc, data, err := wavChunks(wav)
	if err != nil || len(fc) < 16 {
		return nil, errNotWAV
	}
	le := binary.LittleEndian
	format, ch, sr, bits := le.Uint16(fc[0:]), int(le.Uint16(fc[2:])), int(le.Uint32(fc[4:])), int(le.Uint16(fc[14:]))
	if format == 0xFFFE && len(fc) >= 26 {
		format = le.Uint16(fc[24:])
	}
	if ch < 1 || sr < 1 {
		return nil, errNotWAV
	}
	var sample func(b []byte) float64
	switch {
	case format == 1 && bits == 8:
		sample = func(b []byte) float64 { return (float64(b[0]) - 128) / 128 }
	case format == 1 && bits == 16:
		sample = func(b []byte) float64 { return float64(int16(le.Uint16(b))) / 32768 }
	case format == 1 && bits == 24:
		sample = func(b []byte) float64 {
			return float64(int32(uint32(b[0])<<8|uint32(b[1])<<16|uint32(b[2])<<24)>>8) / 8388608
		}
	case format == 1 && bits == 32:
		sample = func(b []byte) float64 { return float64(int32(le.Uint32(b))) / 2147483648 }
	case format == 3 && bits == 32:
		sample = func(b []byte) float64 { return float64(math.Float32frombits(le.Uint32(b))) }
	default:
		return nil, fmt.Errorf("WAV format %d with %d-bit samples isn't supported", format, bits)
	}
	width := bits / 8
	step := width * ch
	n := len(data) / step
	mono := make([]float64, n)
	for i := range n {
		var s float64
		for c := range ch {
			s += sample(data[i*step+c*width:])
		}
		mono[i] = s / float64(ch)
	}
	outN := int(int64(n) * int64(rate) / int64(sr))
	out := make([]byte, 2*outN)
	for j := range outN {
		pos := float64(j) * float64(sr) / float64(rate)
		i := int(pos)
		v := mono[i]
		if i+1 < n {
			v += (mono[i+1] - v) * (pos - float64(i))
		}
		v = max(-1, min(1, v))
		le.PutUint16(out[2*j:], uint16(int16(math.Round(v*32767))))
	}
	return out, nil
}

// pcmToWAV wraps 16-bit mono PCM in a WAV header.
func pcmToWAV(pcm []byte, rate int) []byte {
	var b bytes.Buffer
	le := binary.LittleEndian
	b.WriteString("RIFF")
	_ = binary.Write(&b, le, uint32(36+len(pcm)))
	b.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), uint16(1), uint16(1), uint32(rate), uint32(rate * 2), uint16(2), uint16(16)} {
		_ = binary.Write(&b, le, v)
	}
	b.WriteString("data")
	_ = binary.Write(&b, le, uint32(len(pcm)))
	b.Write(pcm)
	return b.Bytes()
}

// handleV1Speech is OpenAI's text to speech: {model, input, voice,
// response_format, speed, stream_format}. "model" names a voice model's
// instance or preset; OpenAI's names (tts-1...) or none pick the first, and
// OpenAI's voices (alloy...) the model's first voice. "language" (not
// OpenAI's) sets the language; without it it is guessed. Without a
// response_format the answer is MP3, like OpenAI's (WAV without ffmpeg).
func (ws *WebServer) handleV1Speech(w http.ResponseWriter, r *http.Request) {
	if !allowMethods(w, r, http.MethodPost) {
		return
	}
	if ws.speech == nil {
		writeAPIError(w, http.StatusNotFound, "model_not_found", "", "text to speech is off; turn it on in the text_to_speech section of config.yaml")
		return
	}
	var req struct {
		Model          string   `json:"model"`
		Input          string   `json:"input"`
		Voice          any      `json:"voice"`
		Language       string   `json:"language"`
		ResponseFormat string   `json:"response_format"`
		Speed          *float64 `json:"speed"`
		StreamFormat   string   `json:"stream_format"`
	}
	body, aerr := readAPIBody(w, r, maxJSONBody)
	if aerr != nil {
		aerr.write(w)
		return
	}
	if err := json.Unmarshal(body, &req); err != nil {
		writeAPIError(w, http.StatusBadRequest, "", "", "invalid json: "+err.Error())
		return
	}
	if strings.TrimSpace(req.Input) == "" {
		writeAPIError(w, http.StatusBadRequest, "", "input", `"input" is required: the text to speak`)
		return
	}
	if utf8.RuneCountInString(req.Input) > maxSpeechInput {
		writeAPIError(w, http.StatusBadRequest, "", "input", fmt.Sprintf(`"input" is over %d characters; split the text`, maxSpeechInput))
		return
	}
	format := strings.ToLower(req.ResponseFormat)
	if format == "" {
		format = "mp3" // OpenAI's default; WAV when ffmpeg isn't installed
	}
	if speechMIME[format] == "" {
		writeAPIError(w, http.StatusBadRequest, "", "response_format", "response_format must be mp3, opus, aac, flac, wav or pcm")
		return
	}
	speed := 1.0
	if req.Speed != nil {
		speed = *req.Speed
	}
	if speed < 0.25 || speed > 4 {
		writeAPIError(w, http.StatusBadRequest, "", "speed", "speed must be between 0.25 and 4.0")
		return
	}
	if req.StreamFormat != "" && req.StreamFormat != "audio" && req.StreamFormat != "sse" {
		writeAPIError(w, http.StatusBadRequest, "", "stream_format", "stream_format must be audio or sse")
		return
	}
	sse := req.StreamFormat == "sse"
	model := req.Model
	if _, ok := ws.speech.pick(model); !ok && (strings.HasPrefix(model, "tts-") || strings.HasSuffix(model, "-tts")) {
		model = ""
	}
	// A custom voice is {"id": ...} in newer OpenAI clients.
	var voice string
	switch v := req.Voice.(type) {
	case string:
		voice = v
	case map[string]any:
		voice, _ = v["id"].(string)
	}
	if e, ok := ws.speech.pick(model); ok {
		if _, has := e.voice(voice); !has && openAIVoices[strings.ToLower(voice)] {
			voice = ""
		}
	}
	rs, err := ws.speech.prepare(model, voice, req.Language, speechText(req.Input))
	if err != nil {
		writeSpeechErr(w, err)
		return
	}
	// PCM at normal speed goes out part by part as it is spoken.
	if format == "pcm" && speed == 1 {
		ws.streamPCM(w, r, rs, sse)
		return
	}
	wavs := make([][]byte, 0, len(rs.parts))
	for _, p := range rs.parts {
		wav, err := ws.speech.speak(r.Context(), rs.entry, rs.voice, rs.lang, p)
		if err != nil {
			if r.Context().Err() != nil {
				return
			}
			writeSpeechErr(w, err)
			return
		}
		wavs = append(wavs, wav)
	}
	wav, err := joinWAVs(wavs)
	if err != nil {
		writeAPIError(w, http.StatusBadGateway, "", "", rs.entry.Name+" sent audio llama-manager can't join: "+err.Error())
		return
	}
	audio, mime := encodeSpeech(r.Context(), wav, format, speed)
	if sse {
		startSSE(w)
		sendSSE(w, "speech.audio.delta", map[string]any{"type": "speech.audio.delta", "audio": base64.StdEncoding.EncodeToString(audio)})
		sendSSE(w, "speech.audio.done", map[string]any{"type": "speech.audio.done", "usage": map[string]int{"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}})
		return
	}
	h := w.Header()
	h.Set("Content-Type", mime)
	h.Set("Content-Length", strconv.Itoa(len(audio)))
	h.Set("Cache-Control", "no-store")
	h.Set("X-Content-Type-Options", "nosniff")
	_, _ = w.Write(audio)
}

func (ws *WebServer) streamPCM(w http.ResponseWriter, r *http.Request, rs replySpeech, sse bool) {
	started := false
	for _, p := range rs.parts {
		wav, err := ws.speech.speak(r.Context(), rs.entry, rs.voice, rs.lang, p)
		if err == nil {
			var pcm []byte
			if pcm, err = wavToPCM(wav, pcmRate); err == nil {
				if !started {
					started = true
					if sse {
						startSSE(w)
					} else {
						w.Header().Set("Content-Type", speechMIME["pcm"])
						w.Header().Set("Cache-Control", "no-store")
						w.WriteHeader(http.StatusOK)
					}
				}
				if sse {
					sendSSE(w, "speech.audio.delta", map[string]any{"type": "speech.audio.delta", "audio": base64.StdEncoding.EncodeToString(pcm)})
				} else {
					_, _ = w.Write(pcm)
					if f, ok := w.(http.Flusher); ok {
						f.Flush()
					}
				}
				continue
			}
		}
		if r.Context().Err() != nil {
			return
		}
		if !started {
			writeSpeechErr(w, err)
			return
		}
		log.Printf("/v1/audio/speech: %v; the stream ends early", err)
		return
	}
	if sse && started {
		sendSSE(w, "speech.audio.done", map[string]any{"type": "speech.audio.done", "usage": map[string]int{"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}})
	}
}

func startSSE(w http.ResponseWriter) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
}

func sendSSE(w http.ResponseWriter, event string, v any) {
	b, _ := json.Marshal(v)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// audioCap remembers whether a run of an instance hears audio.
type audioCap struct {
	started time.Time
	audio   bool
}

// hearsAudio asks llama-server's /props whether its model takes audio input,
// once per run.
func (ws *WebServer) hearsAudio(ctx context.Context, inst *Instance) bool {
	started := inst.StartedAt()
	if v, ok := ws.audioCaps.Load(inst.conf.Name); ok && v.(audioCap).started.Equal(started) {
		return v.(audioCap).audio
	}
	p, err := inst.FetchProps(ctx)
	if err != nil {
		return false
	}
	ws.audioCaps.Store(inst.conf.Name, audioCap{started, p.Audio})
	return p.Audio
}

// audioInstance picks the model for transcription and realtime: the named
// instance, else the fallback model, else the first running model that hears
// audio (has an audio projector).
func (ws *WebServer) audioInstance(ctx context.Context, name string) (*Instance, *apiError) {
	if inst := ws.findInstance(name); inst != nil {
		if inst.conf.role == roleImageGen || (inst.conf.role == roleSpeech && !ws.speechOnLlama(inst.conf.Name)) {
			return nil, apiErrf(http.StatusBadRequest, "model_not_supported", "%s can't hear audio; use a llama-server model with an audio projector", inst.conf.Name)
		}
		if e := readyErr(inst); e != nil {
			return nil, e
		}
		if !ws.hearsAudio(ctx, inst) {
			return nil, apiErrf(http.StatusBadRequest, "model_not_supported", "%s can't hear audio: its model has no audio projector (mmproj). Use one like Voxtral or Qwen2.5-Omni", inst.conf.Name)
		}
		return inst, nil
	}
	var cands []*Instance
	if dm := ws.defaultModel(); dm != "" {
		if inst := ws.mgr.Get(dm); inst != nil {
			cands = append(cands, inst)
		}
	}
	all := ws.mgr.Instances()
	for _, inst := range all {
		if inst.conf.role == "" || inst.conf.role == roleReader {
			cands = append(cands, inst)
		}
	}
	for _, inst := range all {
		if inst.conf.role == roleSpeech && ws.speechOnLlama(inst.conf.Name) {
			cands = append(cands, inst)
		}
	}
	for _, inst := range cands {
		if inst.conf.role != roleImageGen && inst.State() == StateRunning && ws.hearsAudio(ctx, inst) {
			return inst, nil
		}
	}
	return nil, apiErrf(http.StatusNotFound, "model_not_found", "no running model can hear audio. Run a llama-server model with an audio projector (mmproj), such as Voxtral Mini 3B or Qwen2.5-Omni, to use /v1/audio/transcriptions, /v1/audio/translations and /v1/realtime")
}

// sniffAudio names an audio file's format from its first bytes.
func sniffAudio(b []byte) string {
	switch {
	case len(b) >= 12 && string(b[:4]) == "RIFF" && string(b[8:12]) == "WAVE":
		return "wav"
	case len(b) >= 4 && string(b[:4]) == "fLaC":
		return "flac"
	case len(b) >= 4 && string(b[:4]) == "OggS":
		return "ogg"
	case len(b) >= 4 && b[0] == 0x1A && b[1] == 0x45 && b[2] == 0xDF && b[3] == 0xA3:
		return "webm"
	case len(b) >= 12 && string(b[4:8]) == "ftyp":
		return "mp4"
	case len(b) >= 2 && b[0] == 0xFF && b[1]&0xF6 == 0xF0:
		return "aac"
	case len(b) >= 3 && string(b[:3]) == "ID3", len(b) >= 2 && b[0] == 0xFF && b[1]&0xE0 == 0xE0:
		return "mp3"
	}
	return ""
}

func wavDuration(b []byte) float64 {
	fc, data, err := wavChunks(b)
	if err != nil || len(fc) < 12 {
		return 0
	}
	if rate := binary.LittleEndian.Uint32(fc[8:12]); rate > 0 {
		return float64(len(data)) / float64(rate)
	}
	return 0
}

// mp3Duration estimates an MP3's length from its first frame's bit rate.
func mp3Duration(b []byte) float64 {
	off := 0
	if len(b) >= 10 && string(b[:3]) == "ID3" {
		off = 10 + (int(b[6]&0x7f)<<21 | int(b[7]&0x7f)<<14 | int(b[8]&0x7f)<<7 | int(b[9]&0x7f))
	}
	for ; off+4 <= len(b); off++ {
		if b[off] != 0xFF || b[off+1]&0xE0 != 0xE0 {
			continue
		}
		mpeg1 := b[off+1]&0x18 == 0x18
		layer3 := b[off+1]&0x06 == 0x02
		idx := int(b[off+2] >> 4)
		if !layer3 || idx == 0 || idx == 15 {
			continue
		}
		v1 := []int{0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320}
		v2 := []int{0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160}
		kbps := v2[idx]
		if mpeg1 {
			kbps = v1[idx]
		}
		return float64(len(b)-off) * 8 / float64(kbps*1000)
	}
	return 0
}

// prepareAudio gets an uploaded file ready for llama-server, which reads WAV
// and MP3: other formats go through ffmpeg (16 kHz mono WAV). It returns the
// data, its format and its length in seconds.
func prepareAudio(ctx context.Context, b []byte) ([]byte, string, float64, *apiError) {
	kind := sniffAudio(b)
	if kind == "" {
		return nil, "", 0, &apiError{status: http.StatusBadRequest, param: "file", msg: "the file isn't audio llama-manager can read; send WAV, MP3, FLAC, OGG, WebM, M4A or MP4"}
	}
	if kind == "wav" {
		return b, "wav", wavDuration(b), nil
	}
	pcm, err := runFFmpeg(ctx, b, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "s16le")
	if err == nil {
		return pcmToWAV(pcm, 16000), "wav", float64(len(pcm)) / 32000, nil
	}
	if kind == "mp3" {
		return b, "mp3", mp3Duration(b), nil
	}
	if errors.Is(err, errNoFFmpeg) {
		return nil, "", 0, &apiError{status: http.StatusBadRequest, param: "file", msg: fmt.Sprintf("%s audio has to be converted by ffmpeg, which isn't installed on the server; send WAV or MP3, or install ffmpeg", strings.ToUpper(kind))}
	}
	return nil, "", 0, &apiError{status: http.StatusBadRequest, param: "file", msg: "couldn't read the audio: " + err.Error()}
}

var whisperLangNames = map[string]string{
	"en": "english", "ru": "russian", "uk": "ukrainian", "be": "belarusian", "de": "german", "fr": "french", "es": "spanish",
	"it": "italian", "pt": "portuguese", "nl": "dutch", "pl": "polish", "cs": "czech", "sv": "swedish", "no": "norwegian",
	"da": "danish", "fi": "finnish", "tr": "turkish", "ar": "arabic", "he": "hebrew", "el": "greek", "zh": "chinese",
	"ja": "japanese", "ko": "korean", "hi": "hindi", "vi": "vietnamese", "th": "thai", "id": "indonesian", "ro": "romanian",
	"hu": "hungarian", "bg": "bulgarian", "sr": "serbian", "hr": "croatian", "kk": "kazakh", "fa": "persian",
}

func whisperLang(code string) string {
	code = strings.ToLower(strings.TrimSpace(code))
	if n, ok := whisperLangNames[code]; ok {
		return n
	}
	return code
}

func transcribeInstruction(translate bool, lang, prompt string) string {
	var b strings.Builder
	if translate {
		b.WriteString("Translate the speech in this audio into English. Reply with the English translation only: no introduction, notes or timestamps.")
	} else {
		b.WriteString("Transcribe this audio word for word, in the language that is spoken. Reply with the transcript only: no introduction, notes, timestamps or translation.")
		if n := []rune(whisperLang(lang)); len(n) > 0 {
			b.WriteString(" The speech is in " + strings.ToUpper(string(n[0])) + string(n[1:]) + ".")
		}
	}
	if p := strings.TrimSpace(prompt); p != "" {
		b.WriteString("\n\nText from just before this recording, for names, spelling and style (don't repeat it): " + p)
	}
	return b.String()
}

type tokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (u tokenUsage) openAI() map[string]any {
	return map[string]any{"type": "tokens", "input_tokens": u.PromptTokens, "output_tokens": u.CompletionTokens, "total_tokens": u.TotalTokens,
		"input_token_details": map[string]int{"text_tokens": 0, "audio_tokens": u.PromptTokens}}
}

func cleanTranscript(s string) string {
	if i := strings.LastIndex(s, "</think>"); i >= 0 {
		s = s[i+len("</think>"):]
	}
	return strings.TrimSpace(s)
}

func (ws *WebServer) handleV1Transcriptions(w http.ResponseWriter, r *http.Request) {
	ws.transcribe(w, r, false)
}

func (ws *WebServer) handleV1Translations(w http.ResponseWriter, r *http.Request) {
	ws.transcribe(w, r, true)
}

// transcribe is OpenAI's speech to text (and translation into English) on a
// llama-server model that hears audio: the audio goes in a chat message with
// an instruction. Timestamps (srt, vtt, verbose_json) are spread over the
// sentences by length, as the model gives none.
func (ws *WebServer) transcribe(w http.ResponseWriter, r *http.Request, translate bool) {
	if !allowMethods(w, r, http.MethodPost) {
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxAudioUpload+1<<20)
	if err := r.ParseMultipartForm(8 << 20); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeAPIError(w, http.StatusRequestEntityTooLarge, "request_too_large", "file", fmt.Sprintf("the audio is over %d MB; split it", maxAudioUpload>>20))
			return
		}
		writeAPIError(w, http.StatusBadRequest, "", "", "send the audio as multipart/form-data in file, like OpenAI's audio.transcriptions: "+err.Error())
		return
	}
	defer r.MultipartForm.RemoveAll()
	fhs := r.MultipartForm.File["file"]
	if len(fhs) == 0 {
		writeAPIError(w, http.StatusBadRequest, "", "file", "file is required: the audio to transcribe")
		return
	}
	f, err := fhs[0].Open()
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, "", "file", "reading file: "+err.Error())
		return
	}
	audio, err := io.ReadAll(io.LimitReader(f, maxAudioUpload+1))
	f.Close()
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, "", "file", "reading file: "+err.Error())
		return
	}
	if len(audio) > maxAudioUpload {
		writeAPIError(w, http.StatusRequestEntityTooLarge, "request_too_large", "file", fmt.Sprintf("the audio is over %d MB; split it", maxAudioUpload>>20))
		return
	}
	format := r.FormValue("response_format")
	switch format {
	case "":
		format = "json"
	case "json", "text", "srt", "verbose_json", "vtt", "diarized_json":
	default:
		writeAPIError(w, http.StatusBadRequest, "", "response_format", "response_format must be json, text, srt, verbose_json or vtt")
		return
	}
	temp := 0.0
	if t := r.FormValue("temperature"); t != "" {
		if temp, err = strconv.ParseFloat(t, 64); err != nil || temp < 0 || temp > 1 {
			writeAPIError(w, http.StatusBadRequest, "", "temperature", "temperature must be between 0 and 1")
			return
		}
	}
	lang := ""
	if !translate {
		lang = r.FormValue("language")
	}
	inst, aerr := ws.audioInstance(r.Context(), r.FormValue("model"))
	if aerr != nil {
		aerr.write(w)
		return
	}
	data, kind, dur, aerr := prepareAudio(r.Context(), audio)
	if aerr != nil {
		aerr.write(w)
		return
	}
	stream := r.FormValue("stream") == "true" && (format == "json" || format == "text")
	body, _ := json.Marshal(map[string]any{
		"messages": []map[string]any{{"role": "user", "content": []map[string]any{
			{"type": "input_audio", "input_audio": map[string]string{"data": base64.StdEncoding.EncodeToString(data), "format": kind}},
			{"type": "text", "text": transcribeInstruction(translate, lang, r.FormValue("prompt"))},
		}}},
		"temperature":          temp,
		"stream":               stream,
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
	})
	up, err := http.NewRequestWithContext(r.Context(), http.MethodPost, inst.baseURL()+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, "", "", err.Error())
		return
	}
	up.Header.Set("Content-Type", "application/json")
	resp, err := chatClient.Do(up)
	if err != nil {
		if r.Context().Err() == nil {
			writeAPIError(w, http.StatusBadGateway, "", "", fmt.Sprintf("couldn't reach %s: %v", inst.conf.Name, err))
		}
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		status := resp.StatusCode
		if status >= 500 {
			status = http.StatusBadGateway
		}
		writeAPIError(w, status, "", "", fmt.Sprintf("%s couldn't transcribe the audio: %s", inst.conf.Name, upstreamError(resp)))
		return
	}
	if stream {
		relayTranscript(w, resp.Body)
		return
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage tokenUsage `json:"usage"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 16<<20)).Decode(&out); err != nil || len(out.Choices) == 0 {
		writeAPIError(w, http.StatusBadGateway, "", "", inst.conf.Name+" sent an answer llama-manager can't read")
		return
	}
	text := cleanTranscript(out.Choices[0].Message.Content)
	task, spoken := "transcribe", lang
	if translate {
		task, spoken = "translate", "en"
	} else if spoken == "" {
		spoken = detectLang(text)
	}
	writeTranscript(w, format, task, whisperLang(spoken), dur, temp, text, out.Usage)
}

// relayTranscript turns the model's streamed chat into OpenAI's
// transcript.text.delta and transcript.text.done events.
func relayTranscript(w http.ResponseWriter, body io.Reader) {
	startSSE(w)
	var full strings.Builder
	var usage tokenUsage
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 64<<10), 4<<20)
	for sc.Scan() {
		line, ok := strings.CutPrefix(sc.Text(), "data:")
		if !ok {
			continue
		}
		line = strings.TrimSpace(line)
		if line == "[DONE]" {
			break
		}
		var c struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *tokenUsage `json:"usage"`
		}
		if json.Unmarshal([]byte(line), &c) != nil {
			continue
		}
		if c.Usage != nil {
			usage = *c.Usage
		}
		if len(c.Choices) > 0 && c.Choices[0].Delta.Content != "" {
			d := c.Choices[0].Delta.Content
			if full.Len() == 0 {
				d = strings.TrimLeft(d, " \n")
			}
			if d == "" {
				continue
			}
			full.WriteString(d)
			sendSSE(w, "transcript.text.delta", map[string]any{"type": "transcript.text.delta", "delta": d})
		}
	}
	sendSSE(w, "transcript.text.done", map[string]any{"type": "transcript.text.done", "text": strings.TrimSpace(full.String()), "usage": usage.openAI()})
}

type timedText struct {
	start, end float64
	text       string
}

// timedSentences spreads dur over the sentences of text by their length.
func timedSentences(text string, dur float64) []timedText {
	parts := sentences(text)
	if len(parts) == 0 {
		return nil
	}
	total := 0
	for _, p := range parts {
		total += utf8.RuneCountInString(p)
	}
	out := make([]timedText, len(parts))
	t := 0.0
	for i, p := range parts {
		d := dur * float64(utf8.RuneCountInString(p)) / float64(total)
		out[i] = timedText{start: t, end: t + d, text: p}
		t += d
	}
	return out
}

func subtitleTime(sec float64, sep string) string {
	ms := int64(math.Round(sec * 1000))
	return fmt.Sprintf("%02d:%02d:%02d%s%03d", ms/3600000, ms/60000%60, ms/1000%60, sep, ms%1000)
}

func round2(f float64) float64 { return math.Round(f*100) / 100 }

func writeTranscript(w http.ResponseWriter, format, task, lang string, dur, temp float64, text string, usage tokenUsage) {
	segs := timedSentences(text, dur)
	switch format {
	case "text", "srt", "vtt":
		var b strings.Builder
		switch format {
		case "text":
			b.WriteString(text + "\n")
		case "srt":
			for i, s := range segs {
				fmt.Fprintf(&b, "%d\n%s --> %s\n%s\n\n", i+1, subtitleTime(s.start, ","), subtitleTime(s.end, ","), s.text)
			}
		case "vtt":
			b.WriteString("WEBVTT\n\n")
			for _, s := range segs {
				fmt.Fprintf(&b, "%s --> %s\n%s\n\n", subtitleTime(s.start, "."), subtitleTime(s.end, "."), s.text)
			}
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = io.WriteString(w, b.String())
	case "verbose_json":
		out := make([]map[string]any, len(segs))
		for i, s := range segs {
			out[i] = map[string]any{"id": i, "seek": 0, "start": round2(s.start), "end": round2(s.end), "text": " " + s.text,
				"tokens": []int{}, "temperature": temp, "avg_logprob": 0, "compression_ratio": 0, "no_speech_prob": 0}
		}
		writeJSON(w, http.StatusOK, map[string]any{"task": task, "language": lang, "duration": round2(dur), "text": text, "segments": out,
			"usage": map[string]any{"type": "duration", "seconds": int(math.Ceil(dur))}})
	case "diarized_json":
		out := make([]map[string]any, len(segs))
		for i, s := range segs {
			out[i] = map[string]any{"type": "transcript.text.segment", "id": fmt.Sprintf("seg_%d", i), "start": round2(s.start), "end": round2(s.end), "text": s.text, "speaker": "A"}
		}
		writeJSON(w, http.StatusOK, map[string]any{"task": task, "duration": round2(dur), "text": text, "segments": out, "usage": usage.openAI()})
	default:
		writeJSON(w, http.StatusOK, map[string]any{"text": text, "usage": usage.openAI()})
	}
}
