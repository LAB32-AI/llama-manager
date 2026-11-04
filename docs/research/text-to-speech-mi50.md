# Text to speech on 5× AMD MI50 32 GB (gfx906)

Research done on 2026-11-03, for a "play" button under each chat reply.

**Status:** integrated on 2026-11-03 as `text_to_speech` with `ming3-tts` (audio.cpp),
`qwen3-tts` and `lfm3-audio` (llama-server `/tts`); see the README. Fish Drama 3 is not
integrated. MI50 speeds below are still estimates until the benchmark in section 7 runs. It follows
[image-generation-mi50.md](image-generation-mi50.md) and
[video-generation-mi50.md](video-generation-mi50.md). TTS moves fast (most models below are from
2026), so re-check versions before relying on this.

Tags:
- **measured** means a published run, always on other hardware: nobody has published TTS
  numbers on gfx906 yet.
- **estimate** means scaled to the MI50 from those runs. For single-stream decoding,
  llama.cpp on an MI50 runs at roughly 35–45% of an RTX 4090 (similar memory bandwidth, no
  matrix cores).

Quality numbers are Elo from the Artificial Analysis Speech Arena (provider-voice board,
blind votes, snapshot of early November 2026; higher is better). For scale: ElevenLabs v4 has
1334, ElevenLabs v3 1187 and OpenAI TTS-1 HD 1106.

## 1. Verdict

| Question | Answer |
|---|---|
| Possible on the MI50s? | **Yes.** TTS models are small next to LLMs (0.03–4B active parameters), and the good ones now run on ggml, the same base our llama.cpp builds use. |
| Something like sd.cpp for speech? | **Two options.** (1) **llama.cpp itself**: `llama-server` has had `POST /tts` since 2026-10-20 (PR #26603), which is the binary we already run. (2) **audio.cpp**: a separate C++/ggml engine with 30+ TTS models, an OpenAI-style `POST /v1/audio/speech` server and ROCm, Vulkan and CPU backends. |
| Latest, best models? | **Yes.** The #1 open model, **Fish Audio Drama 3** (Elo 1272, above ElevenLabs v3), fits on one MI50 in Q4 and runs in audio.cpp. **Ming 3.0 TTS 0.5B** (Elo 1188, the same as ElevenLabs v3) is MIT-licensed and only 1.7 GB. |
| Speed | Small models: faster than real time (estimate). Drama 3: about real time on an MI50 (estimate), so the first sound arrives quickly with streaming but long replies can stutter. |
| Main risk | **ROCm on gfx906 is untested** for every TTS engine. audio.cpp calls its HIP builds "test" quality, and some model docs list only CUDA, Vulkan and CPU. Vulkan is the fallback. Benchmark before coding. |
| Why not PyTorch / vLLM? | The official repos, vLLM-Omni and SGLang-Omni are PyTorch stacks. On gfx906 they need self-built wheels, and their fused attention kernels (CK, Triton) skip gfx906. People report "awfully slow" PyTorch TTS on MI50s. |

## 2. Runtimes

| Runtime | gfx906 status | Models (TTS) | API | Notes |
|---|---|---|---|---|
| **llama.cpp `llama-server`**, v0.8.0+ (2026-10-28) | Same ROCm/Vulkan build we already run. The TTS ops (conv-transpose, pad) are new and untested on gfx906. | Qwen3-TTS 0.6B/1.7B, LFM3-Audio-2B, LFM3-Omni-2B, Pocket TTS, Soprano 80M, Chatterbox (merged 2026-11-03; Multilingual not in the server yet). Open PRs: **Drama 3 (#31613)**, KaniTTS-2 (#28609). | `POST /tts` (at the root, not `/v1`). JSON or multipart; `input`, `lang`, `voice`, `speaker_ref_b64`, `seed`, `n_predict`, `response_format` `wav` or `pcm`, `stream`. Returns WAV bytes. `GET /props` lists `voices`. | No OpenAI `/v1/audio/speech` yet (planned). `--tts-speaker-dir` for named voices is planned too. Also has `/v1/realtime` voice chat (LFM3). |
| **audio.cpp `audiocpp_server`** (Apache-2.0, v0.9.3 2026-10-26, ~4k stars) | HIP "test builds", not validated on gfx906; build from source with `scripts/build_linux.sh --hip --gpu-targets gfx906`. Vulkan is also supported. | Drama 3, Ming 3.0, Breeze 2, Phonon, Fish S2 Pro, Qwen3-TTS (Base, CustomVoice, VoiceDesign), Voxtral, Kokoro, Chatterbox, VoxCPM2, CosyVoice3, Supertonic 3, Pocket TTS, Magpie, VibeVoice, IndexTTS-2… | `POST /v1/audio/speech` (OpenAI style), SSE streaming of PCM16, `GET /v1/models`, `GET /v1/audio/voices`. Config JSON with `voice_presets` and `default_voice_preset`. | Binds `127.0.0.1` by default. Has a WebUI with model management (leave `--ui-management` off). Many speed options are tuned for CUDA only (`cuda_perf`). |
| PyTorch (vLLM-Omni, SGLang-Omni, upstream repos) | Painful (see section 1) | Everything | Various | Not worth it on this rig. |
| CPU (audio.cpp CPU, ONNX) | n/a | Kokoro, Phonon, Pocket TTS, Supertonic 3 | – | Real time without a GPU. A fallback, not "real nice". |

## 3. Models

"Fits" assumes one MI50 32 GB. VRAM is file size plus about 1–3 GB of working memory
(estimate). Elo "–" means the model is not on the board.

| Model | Released | Params | Elo | Languages | Voices | License | Runtime | Download (Q8 unless noted) | Fits |
|---|---|---|---|---|---|---|---|---|---|
| **Fish Audio Drama 3** | Oct 2026 | 35B MoE, 3B active | **1272** (#1 open, #6 overall) | 83 incl. en, ru, uk, de, fr, es, zh, ja | Cloning (3–30 s clip), up to 16 speakers per text, `[laughs]`/`[sighs]`/`<whisper>` tags, director notes, edits | Fish Audio Community License: free for personal and research use; commercial only under USD 500k yearly revenue; show "Built with Fish Audio" | audio.cpp (experimental); llama.cpp PR open | Q4 experts + Q8 backbone **22.9 GiB**; Q3 18.9 GiB; Q2 15.9 GiB (Jmica/Drama-3-GGUF) | ✅ one whole card (~25–27 GB) |
| drama3-small (community) | Oct 2026 | pruned Drama 3, ~3-bit | – | ja, en only | Cloning, tags | Same as Drama 3 | Patched audio.cpp fork (CUDA) | IQ3_XXS 9.7 GiB, peak 10.3 GiB | ✅ but needs a fork |
| Breeze TTS 2 | Aug 2026 | 3B | 1226 | zh, en | Cloning, voice design | **Non-commercial** | audio.cpp | 5.1 GB | ✅ |
| **Ming 3.0 TTS 0.5B** (Ant Group) | Oct 2026 | 0.5B | **1188** (≈ ElevenLabs v3) | zh, en, ja, ko, de, fr, es (**no ru**) | 4 built-in English (US_Emma, US_James, UK_George, UK_Charlotte; `template_name=ip`), cloning, voice design from a description, emotion | **MIT** | audio.cpp (docs list CUDA, Vulkan, CPU) | **1.74 GB** (bf16 3.0 GB) | ✅ shares a card |
| **Phonon 31M** (Gradium) | Oct 2026 | 31M | **1111** (beats Kokoro) | en only | Default voice + voice files, no cloning | Apache-2.0 | audio.cpp; llama.cpp feature request #31043 | **0.09 GB** | ✅ or CPU |
| Fish Audio S2 Pro | Mar 2026 | 4B + 0.4B | 1108 | 80+ | Cloning | **Non-commercial** | audio.cpp | 6.3 GB | ✅ |
| Voxtral TTS (Mistral) | Mar 2026 | 4B | 1089 | 9 (no ru) | 20 presets | **CC-BY-NC** | audio.cpp | 4.7 GB | ✅ |
| Magpie Multilingual (NVIDIA) | Feb 2026 | 357M | 1064 | 13 locales (no ru) | Baked speakers | NVIDIA Open Model (commercial OK) | audio.cpp | 1.6 GB | ✅ |
| Kokoro 82M | Jan 2025 | 82M | 1061 | 9 (no ru) | 54 presets, blending | Apache-2.0 | audio.cpp, ONNX, CPU | 0.19 GB | ✅ or CPU |
| Chatterbox / Turbo (Resemble) | 2025–26 | 0.5B / 350M | 1022 | Turbo en; Multilingual 23 | Default voice + cloning | MIT | **llama.cpp** (Turbo only in server), audio.cpp | 2.1 / 0.7 GB | ✅ |
| **Qwen3-TTS 1.7B** | Jan 2026 | 1.7B | – (Alibaba's API version: 929–939) | **10 incl. ru**: zh, en, de, it, pt, es, ja, ko, fr, ru | Base: cloning from a reference clip (recommended, otherwise quality drops). CustomVoice: built-in speakers (audio.cpp). VoiceDesign | Apache-2.0 | **llama.cpp** (Base), audio.cpp (all three) | 1.85 GB + mmproj 0.45 GB | ✅ shares a card |
| LFM3-Audio-2B (Liquid) | Oct 2026 | 2B | – | 19 (no ru) | female / male | LFM Open License v1.0 | **llama.cpp** | Q4_K_M 1.1 GB + mmproj 0.3 GB | ✅ shares a card |
| Pocket TTS (Kyutai) | 2026 | 100M | – | en, fr, de, it, pt, es | ~26 presets per language, cloning | CC-BY-4.0 | **llama.cpp**, audio.cpp | 0.13 GB | ✅ or CPU |
| Soprano 1.1 | 2026 | 80M | – | en (normalized text) | One fixed voice | Apache-2.0 | **llama.cpp** | <0.2 GB | ✅ or CPU |
| Supertonic 3 | Apr 2026 | ~99M | – | 31 incl. ru, uk | Presets | OpenRAIL-M | audio.cpp, ONNX | 0.45 GB | CPU |
| Step Audio EditX | 2025–26 | 3B | 1097 | zh, en, ja, ko + dialects | Cloning, editing | Apache-2.0 | vLLM only (no ggml port) | – | ❌ PyTorch only |

## 4. Speed

No TTS model has a published gfx906 benchmark. RTF (real-time factor) is generation time
divided by audio length: below 1 is faster than real time. A 150-word reply is about one minute
of speech.

| Model | Published | MI50 estimate | 1 min of speech on an MI50 |
|---|---|---|---|
| Fish Drama 3 | RTF 0.13 on an H200 (measured, model card); drama3-small 2.6× real time on an RTX 4090 once generating (measured) | **RTF 0.8–1.3**: the 3B active parameters per audio frame are about the same as Drama3-small, at 35–45% of 4090 speed | ~50–80 s; first sound in a few seconds with streaming, may stutter |
| Ming 3.0 0.5B | No numbers published | RTF ~0.2–0.5 (0.5B backbone + flow head + 44.1 kHz VAE) | ~15–30 s |
| Qwen3-TTS 1.7B | RTF 0.35 on an RTX 5050, 1.3 on CPU (measured, community port) | RTF ~0.2–0.4 (an MI50 has ~3× the RTX 5050's memory bandwidth) | ~15–25 s |
| LFM3-Audio-2B | No numbers published | RTF ~0.2–0.4 (similar size to Qwen3-TTS) | ~15–25 s |
| Kokoro / Phonon / Pocket | Pocket: ~6× real time on an M4 Air CPU; Kokoro Q8: RTF 0.87 on a Rockchip RK3588 board (measured) | Much faster than real time on the server CPU or the GPU | a few seconds |

Speed tips:
- **Stream** (`stream: true` / SSE), so playback starts after the first sentence.
- **Split** replies by sentence or paragraph. That bounds memory and keeps one voice across chunks: audio.cpp anchors later chunks to the first one.

## 5. What we need

| Need | Detail |
|---|---|
| Binary | **llama.cpp v0.8.0+** (we already build it) for Qwen3-TTS, LFM3, Pocket TTS, Soprano and Chatterbox Turbo. **audio.cpp** built on the rig for Drama 3, Ming 3.0 and Phonon: `scripts/build_linux.sh --hip --gpu-targets gfx906 --target audiocpp_server`, and a `--vulkan` build as the fallback. |
| GPU | Drama 3: one whole card. All others: 2–4 GB, so they can share a card (e.g. GPU 4 with `image-reader`), or run on the CPU. |
| Disk | 0.1–3 GB per small model; 16–24 GB for Drama 3. |
| Voice | Built-in voices: Ming (4 English), Kokoro, Pocket, LFM3, Voxtral, Qwen3 CustomVoice. Clone-only setups (Qwen3 Base in llama.cpp, Drama 3) need a reference clip of 3–30 s plus its transcript. It must be a voice we may use (our own recording or a CC0 clip), not a celebrity. |
| Text cleanup | Strip markdown, code blocks, tables, URLs and the thinking section before speaking. Spell out numbers for small English models (Soprano needs normalized text). |
| Formats | llama.cpp returns WAV (24 kHz) or raw float PCM. audio.cpp returns WAV, or PCM16 over SSE. Browsers play WAV in `<audio>` directly; streaming PCM needs Web Audio. |
| gfx906 caveats | No bf16 math, so prefer Q8 packages (some still keep small bf16 parts; watch logs for CPU fallback). The HIP path is unvalidated, so try Vulkan if ROCm fails. |

## 6. Recommendation

| Goal | Pick | Why |
|---|---|---|
| **Best voice the rig can run** | **Fish Audio Drama 3**, Q4 experts + Q8 backbone, on one MI50 via audio.cpp | #1 open model, above ElevenLabs v3 in blind votes, 83 languages, emotions. Costs a whole card and runs about real time. |
| **Best start (recommended)** | **Ming 3.0 TTS 0.5B** Q8 via audio.cpp | MIT, 1.7 GB, ElevenLabs-v3-level Elo, 4 built-in voices, so no reference clip is needed. Shares a card. No Russian. |
| Russian, or no new binary | **Qwen3-TTS 1.7B** via `llama-server /tts`, with one reference voice clip | Apache, 10 languages incl. ru, runs on the binary we already supervise. Quality one tier lower. |
| English on the CPU | **Phonon 31M** (audio.cpp) or Kokoro | 90 MB, Elo 1111, no GPU. |
| Skip | Breeze 2, Fish S2 Pro, Voxtral (non-commercial); Step Audio EditX (PyTorch only); drama3-small (fork + CUDA patch) | |

**Plan:** run audio.cpp as a third managed binary, like `sd-server`, with model presets like the
image ones (`ming3-tts` default, `fish-drama3`, `phonon-en`). Speaking one model through the
OpenAI-style API then makes the model a config choice. Benchmark first.

## 7. Benchmark to run first (one MI50, before any code)

```bash
# build
git clone https://github.com/0xShug0/audio.cpp && cd audio.cpp
scripts/build_linux.sh --hip --gpu-targets gfx906 --target audiocpp_cli --target audiocpp_server
scripts/build_linux.sh --vulkan --target audiocpp_cli          # fallback

# Ming 3.0 (audio-cpp/Ming-3.0-TTS-0.5B-GGUF, ming-3.0-tts-0.5b-q8_0.gguf)
HIP_VISIBLE_DEVICES=4 build/linux-hip-release/bin/audiocpp_cli --family ming3_tts --task tts \
  --backend hip --model ming-3.0-tts-0.5b-q8_0.gguf --request-option language=en \
  --request-option template_name=ip --request-option voice_id=US_Emma --text "<two paragraphs of a real chat reply>" --out ming.wav --log

# Drama 3 (Jmica/Drama-3-GGUF, drama-3-backbone-8bit-experts-Q4_K_S.gguf)
HIP_VISIBLE_DEVICES=4 build/linux-hip-release/bin/audiocpp_cli --task tts --family fish_drama3 \
  --backend hip --mode streaming --model drama-3-backbone-8bit-experts-Q4_K_S.gguf \
  --voice-ref voice.wav --request-option 'reference_text=<transcript of voice.wav>' \
  --text "<same text>" --max-tokens 1500 --out drama.wav --log

# Qwen3-TTS through the llama.cpp we already have (v0.8.0+)
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF:Q8_0 --tts-lang en \
  --tts-speaker-file voice.wav -p "<same text>" --output qwen.wav
```

Record for each:
- generation time against audio length (RTF) and peak VRAM;
- whether the log shows CPU fallback;
- whether the result sounds right.

If HIP fails or falls back to the CPU, repeat with the Vulkan build.

## 8. Integration plan for llama-manager

- **Config.** A file-only `text_to_speech` section, like `image_generation`:
  - `server_bin` (audiocpp_server or llama-server), a model preset name, `gpu_ids`, port;
  - presets pinned by commit + SHA-256 and downloaded into the HF cache;
  - the server binds `127.0.0.1` and is reached only through the manager.
  - Check audiocpp_server's CORS/Origin behaviour before trusting it, as we did with sd-server.
- **Instance.** Managed instance `tts` (role `tts`). Health check `GET /health` (audio.cpp) or `/props` (llama.cpp).
- **API.** `POST /api/chats/{id}/messages/{mid}/speech` and an OpenAI-compatible proxy `POST /v1/audio/speech`. The manager:
  - cleans the reply text (section 5);
  - splits it into sentences;
  - streams audio to the browser.
- **Storage.** Optionally cache the finished WAV per reply version as an attachment. Accept only a RIFF/WAVE signature (`52 49 46 46 … 57 41 56 45`), next to the PNG rule in CLAUDE.md.
- **UI.**
  - A ▶ button under each reply, with pause/stop and a progress state;
  - streams through Web Audio, or plays the cached WAV in `<audio>`;
  - voice picker in settings;
  - "Built with Fish Audio" in the UI if Drama 3 is used.

## Sources

- llama.cpp:
  - [server `/tts` endpoint PR #26603](https://github.com/ggml-org/llama.cpp/pull/26603) and [Qwen3-TTS PR #26254](https://github.com/ggml-org/llama.cpp/pull/26254);
  - [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) and [tts README](https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md);
  - model PRs: [LFM3 #31257](https://github.com/ggml-org/llama.cpp/pull/31257), [Soprano #28607](https://github.com/ggml-org/llama.cpp/pull/28607), [Chatterbox #30993](https://github.com/ggml-org/llama.cpp/pull/30993), [Drama 3 #31613](https://github.com/ggml-org/llama.cpp/pull/31613), [Pocket TTS #26871](https://github.com/ggml-org/llama.cpp/pull/26871).
- audio.cpp:
  - [repo and README](https://github.com/0xShug0/audio.cpp), [model licenses](https://github.com/0xShug0/audio.cpp/blob/main/docs/model_licenses.md), [server README](https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md);
  - model docs: [Drama 3](https://github.com/0xShug0/audio.cpp/blob/main/docs/models/fish_drama3.md), [Ming 3.0](https://github.com/0xShug0/audio.cpp/blob/main/docs/models/ming3_tts.md), [Qwen3-TTS](https://github.com/0xShug0/audio.cpp/blob/main/docs/models/qwen3_tts.md);
  - [GGUF packages](https://huggingface.co/audio-cpp/audio.cpp-gguf).
- Leaderboard: [Artificial Analysis Speech Arena](https://artificialanalysis.ai/text-to-speech/leaderboard).
- Model cards and repos:
  - Fish Drama 3: [model](https://huggingface.co/fishaudio/drama-3), [license](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE), [GGUF](https://huggingface.co/Jmica/Drama-3-GGUF), [drama3-small](https://huggingface.co/pentacoxian-dev/drama3-small-GGUF);
  - Ming 3.0: [model](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B), [GGUF](https://huggingface.co/audio-cpp/Ming-3.0-TTS-0.5B-GGUF);
  - Phonon: [GGUF](https://huggingface.co/gradium/phonon-31m-en-audiocpp);
  - Qwen3-TTS: [GGUF](https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF);
  - LFM3-Audio: [GGUF](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF);
  - Pocket TTS: [voices](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning);
  - Chatterbox: [model](https://huggingface.co/ResembleAI/chatterbox).
- Guides used for cross-checking: [Pinggy (Aug 2026)](https://pinggy.io/blog/best_open_source_self_hosted_text_to_speech_models/), [Yaps (Oct 2026)](https://www.yaps.ai/blog/best-local-tts-models), [Ming vs Qwen Elo (OrcaRouter)](https://www.orcarouter.ai/blog/ming-3-0-tts-0-5b-vs-qwen-audio-3-0-tts), [Qwen3-TTS GGUF speed (community)](https://github.com/HaujetZhao/Qwen3-TTS-GGUF).
- gfx906 PyTorch/attention status: [r/LocalLLaMA snapshot](https://reddit.sentinel-team.org/posts/1s614i8/snapshots/2026-04-03T21%3A20%3A24.375333Z).
