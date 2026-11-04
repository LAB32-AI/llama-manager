# Video generation on 5× AMD MI50 32 GB (gfx906)

Research done on 2026-11-03, following [image-generation-mi50.md](image-generation-mi50.md).
Software support for gfx906 changes quickly, so re-check versions before relying on this.

**Status:** parked on 2026-11-03. Nothing is integrated in llama-manager yet. When picking
this up, start with the one-card benchmark in section 8 before writing any code.

Tags: **measured** means a real run on a gfx906 card. **estimate** means scaled from those
runs with a FLOP model (attention + linear layers at the measured ~6–9.5 TFLOPS fp16) and
not measured. The model roughly agrees with the measured clips below (within ~25% for Wan 2.2 5B).

## 1. Verdict

| Question | Answer |
|---|---|
| Is it possible? | **Yes.** Open video models run on gfx906, and there are measured MI50 clips: Wan 2.2 5B, Wan 2.2 A14B and LTX-2.3 in ComfyUI. |
| Is there something like sd.cpp for video? | **Yes, it is sd.cpp itself.** The `sd-server` we already run has a native video job API (`POST /sdcpp/v1/vid_gen`, polled like `img_gen`). It supports Wan 2.1/2.2, LTX-2.3/2.5, HunyuanVideo 1.5, MiniMax-H3 and LingBot-Video, and returns WebM, MJPEG AVI or animated WebP. |
| Can we use the latest models? | **Mostly.** The best open model, MiniMax-H3 (Jul 2026, 3rd on the text-to-video arena), needs two cards and has a territory-gated license. LTX-2.5 (Aug 2026) is gated on HuggingFace. Wan 2.2 (Apache-2.0) and LTX-2.3 download freely and fit one card. Wan 3.0, Seedance, Veo and Kling are API-only. |
| Is it fast? | **No: this is a batch job, not a chat reply.** A 2–5 s clip takes about 5 min at the low end (Wan 2.2 5B at 480p, LTX-2.3 distilled at 768×512) and 20–60 min for 720p or 14B models. One clip at a time per card. Five cards make five clips in parallel, not one clip five times faster. |
| What's the catch on gfx906? | **Attention.** A 5 s clip is 12–50k tokens. Without flash attention sd.cpp builds the full attention matrix in f32 (31–170 GB), which does not fit. Flash attention must be forced on (`--flash-attn on`): `auto` benchmarks it on gfx906 and may switch it off. The speed of ggml's flash-attention kernel on gfx906 at these lengths is **unmeasured**; that is the first thing to benchmark. |
| Audio? | LTX-2.3/2.5 and MiniMax-H3 generate synchronized audio, and Wan 2.2 S2V animates a portrait from speech. sd.cpp muxes the audio into WebM or AVI as raw PCM, which browsers may not play (see §6). |

## 2. Software options

| | stable-diffusion.cpp `sd-server` | ComfyUI (PyTorch on ROCm) | Others |
|---|---|---|---|
| Video models | Wan 2.1 (T2V, I2V, FLF2V, VACE), Wan 2.2 (TI2V 5B, T2V/I2V A14B, S2V, VACE-Fun), LTX-2.3/2.5 (with audio, latent upscaler), HunyuanVideo 1.5, MiniMax-H3 (audio, references), LingBot-Video, AnimateDiff | Everything, usually first | Wan2GP (PyTorch, low-VRAM, ROCm community support); xDiT / SGLang Diffusion (multi-GPU sequence parallel, but AMD's xDiT images target gfx942/gfx950 only) |
| gfx906 | Same build as our image server (HIP, `GPU_TARGETS=gfx906`) | Community image `mixa3607/comfyui-gfx906`. PyTorch 2.14 is the last official wheel with gfx906. | Need PyTorch, so same limits as ComfyUI |
| Attention memory | Full matrix unless flash attention is on (see §1) | Split attention chunks it, so memory is fine; this is where the measured MI50 clips come from | – |
| API | `POST /sdcpp/v1/vid_gen` → 202 `{id}`; poll `GET /sdcpp/v1/jobs/{id}`; cancel; result is one base64 container | Node-graph JSON workflows over WebSocket | – |
| Multi-GPU | Components on separate cards (`--backend`), or one model split over cards for memory. No sequence parallelism, so no speedup. | One instance per GPU | Sequence parallel only on supported (non-gfx906) stacks |
| Integration effort in llama-manager | **Low–medium**: extend the existing `image_generation` presets | High: separate Docker service and workflow files | High |

## 3. Models (as of Nov 2026)

Arena Elo is from the Artificial Analysis text-to-video arena (v2.1; closed models top out
around 1172, Wan 3.0).

| Model (release) | Size | Arena Elo | Audio | Native output | License | Downloads freely? | Fits one MI50? | sd.cpp |
|---|---|---|---|---|---|---|---|---|
| **MiniMax-H3** (Jul 2026) | 33B DiT (13B of it AdaLN, pruned GGUF is smaller) + Qwen3-VL-32B text encoder | **1160** (#1 open, 3rd overall) | Yes, stereo 32 kHz | 768p, 24 fps, 4–15 s | MiniMax H3 Community: users in the US, EU, UK and South Korea must apply; commercial terms are unclear (sources conflict) | Yes (leejet GGUF) | **No**: DiT Q4_K_M 18.8 GB (pruned 11.4 GB) + text encoder Q4_K_M 18.2 GB → 2 cards | Yes (T2VA, I2VA, FL2VA, Ref2VA) |
| **LTX-2.5** (Aug 2026) | 22B + Gemma 4 12B text encoder | 944 | Yes | Up to 1080p (4K with upscaler), 24 fps | LTX-2.x Community: free under $10M revenue | **Gated** (HF token needed for text encoder and VAEs) | Q4_K_M 15.7 GB + text encoder Q8 ~12.5 GB: tight; 2 cards comfortable | Yes (dev, distilled 8-step) |
| **LTX-2.3** (Mar 2026) | 22B + Gemma 3 12B text encoder | 839–884 | Yes | 1080p, 24 fps | LTX-2 Community: free under $10M revenue | Yes (unsloth GGUF) | Distilled Q4_K_M 14.2 GB + Gemma Q4 7.3 GB + connectors 2.3 GB + VAEs 1.8 GB ≈ 26 GB | Yes (8-step distilled, FLF2V, upscaler) |
| **Wan 2.2 TI2V 5B** (Jul 2025) | 5B + UMT5-XXL | – (older) | No | 1280×704, 24 fps, 5 s | **Apache-2.0** | Yes | **Yes, easily**: Q8 5.4 GB + UMT5 Q8 6.0 GB + VAE 1.4 GB = 12.8 GB | Yes (T2V + I2V in one model) |
| Wan 2.2 T2V / I2V A14B (Jul 2025) | 2 × 14B experts (high/low noise) | – | No | 480p/720p, 16 fps, 5 s | **Apache-2.0** | Yes | Q4_K_M 2 × 9.65 GB + UMT5 6 GB ≈ 26 GB (Q8 2 × 15.4 GB → 2 cards) | Yes, plus the lightx2v **4-step Lightning LoRAs** |
| Wan 2.2 S2V 14B | 14B + wav2vec2 | – | Driven by input speech | 480p, 16 fps | Apache-2.0 | Yes | Q8 ≈ 16 GB + UMT5 | Yes |
| HunyuanVideo 1.5 (Nov 2025) | 8.3B + Qwen2.5-VL 7B + ByT5 | – | No | 480p/720p (1080p SR), 24 fps | Tencent Hunyuan Community: **does not apply in the EU, UK, South Korea** | Yes | fp16 16.7 GB + Qwen2.5-VL Q8 8 GB (the same file our `qwen-image-edit-2511` preset uses) ≈ 28 GB | Yes (720p T2V doc; 480p step-distilled exists) |
| LingBot-Video (Jul 2026) | MoE 30B-A3B (+ refiner) / dense 1.3B | – (claims #1 on RBench) | No | 832×480, 24 fps | Apache-2.0 | Yes, bf16 only (60 GB) | MoE only after quantizing | Yes |
| Wan 3.0, Seedance 2.5, Veo, Kling, Utopai X | – | 1150–1172 | – | – | API only | **No weights** | – | – |

FastWan 2.2 TI2V 5B (Apache-2.0, DMD-distilled to **3 steps**) is the fastest Wan option, but
it ships in Diffusers layout; whether sd.cpp loads it without conversion is untested.

## 4. Benchmarks

### Measured on gfx906 (ComfyUI)

| Card | Model | Clip | Result | vs RTX 3090 |
|---|---|---|---|---|
| MI50 | Wan 2.2 TI2V 5B | 832×480, 2.5 s | **5 min 04 s** | – |
| MI50 32GB, ROCm 6.3.3 | Wan 2.2 T2V (default ComfyUI template, probably A14B) | default template | **32–35 min** | ~6% (3090: ~2 min) |
| MI50 | LTX-2.3 22B | 1280×704, 5.2 s, with audio | **20 min 18 s** | – |
| MI50 | Wan VAE decode alone | – | about 10 min (no fast conv3d path) | – |

There are no sd.cpp video numbers on gfx906 yet.

### Estimates for one MI50 with sd.cpp (±50%, assumes flash attention runs about as fast as ComfyUI's split attention)

| Model | Steps | Clip | Time per clip | Clips per hour on 5 cards |
|---|---|---|---|---|
| Wan 2.2 5B | 20, CFG 5 | 832×480, 2 s | **4–6 min** (measured 5 min for 2.5 s) | ~50–75 |
| Wan 2.2 5B | 20 | 832×480, 5 s | 12–18 min | ~17–25 |
| Wan 2.2 5B | 20 | 1280×704, 5 s | 40–60 min | ~5–7 |
| FastWan 2.2 5B (if it loads) | 3, CFG 1 | 832×480, 5 s | 3–10 min (VAE decode dominates) | ~30–100 |
| LTX-2.3 distilled | 8, CFG 1 | 768×512, 5 s + audio | **7–10 min** | ~30–40 |
| LTX-2.3 | – | 1280×704, 5 s + audio | ~20 min (measured) | ~15 |
| Wan 2.2 A14B + Lightning | 4 + 4 | 832×480, 2 s (33 frames) | 10–15 min | ~20–30 |
| Wan 2.2 A14B + Lightning | 4 + 4 | 832×480, 5 s (81 frames) | 30–40 min (in line with the measured 32–35 min) | ~8–10 |
| HunyuanVideo 1.5 480p step-distilled | 8 | 848×480, 5 s | 35–70 min (4× the tokens of Wan 5B at the same size) | ~4–8 |
| MiniMax-H3 (2 cards) | 50 (default) | 864×480, 2.3 s + audio | 30–60 min, very rough | ~2–5 per card pair |

Why video is so much slower than images: the cost of attention grows with the square of the
token count. A 1024² FLUX image is about 4k tokens; a 5 s 480p Wan 14B clip is about 33k, so
attention per step is roughly 65× larger.

## 5. Requirements

| Item | Requirement |
|---|---|
| sd.cpp build | A build from after **2026-09-21**: it fixes a SIGFPE in `ggml_conv_3d` with Wan 2.2 5B image-to-video (issue #2009). Keep `SD_WEBM=ON` (the default) so `webm` output works. |
| Flags | `--flash-attn on` (mandatory for video on gfx906), `--vae-tiling` / temporal tiling for 720p, `--offload-to-cpu` if a model is near 32 GB. |
| GPU | One card per video model (Wan 5B, LTX-2.3 Q4, Wan A14B Q4, HunyuanVideo 1.5). Two cards for MiniMax-H3 or Q8 14B/22B models, text encoder on the second card. Don't share a card with an LLM: a video job keeps the GPU at 100% for 5–60 min. |
| Model files | GGUF Q8_0/Q4_K or fp16. Avoid bf16/fp8 DiT weights (f32 math on gfx906) and int8 convrot (CPU fallback). Small bf16 VAEs are fine. |
| Disk | Wan 5B 13 GB; LTX-2.3 ~26 GB; Wan A14B Q4 + Lightning ~28 GB; HunyuanVideo 1.5 ~20 GB (+ shared Qwen2.5-VL); MiniMax-H3 ~32 GB. |
| Host RAM | 64 GB+ (`--offload-to-cpu` keeps weights in RAM). |
| Cooling | Long jobs keep the card at full load for the whole run; a 150–180 W cap costs ~10% speed and keeps hotspot temperatures down. |

## 6. Pitfalls

| Pitfall | What to do |
|---|---|
| Flash attention `auto` turns it off on gfx906 (no HIP MMA, so it benchmarks; a slow result means off → full attention matrix → out of memory) | Always pass `--flash-attn on` for video presets. |
| Speed of ggml flash attention on gfx906 at 10–50k tokens is unknown (on gfx900 forcing it on was 5× slower for images) | Benchmark first (§8). If it is far slower than ComfyUI, use ComfyUI for video. |
| Audio track is raw PCM (`A_PCM/INT/LIT` in WebM, PCM in AVI) | Browsers expect Opus/Vorbis in WebM. Test playback; if silent, remux with ffmpeg (`-c:v copy -c:a libopus`) before storing. |
| MJPEG AVI doesn't play in browsers | Use `webm` (VP8). Animated WebP works everywhere but has no audio and is large. |
| Wan 2.2 A14B runs two 14B experts in sequence | Both must be loaded (or streamed from RAM with `--offload-to-cpu`). Use Q4_K_M on one card or Q8 across two. |
| Frame counts snap to a grid | Wan/Hunyuan/LingBot: 4n+1 (33, 49, 81, 121); LTX: 8n+1 (33, 121); MiniMax-H3: 17n+5. Wan 2.2 S2V and MiniMax-H3 force their fps (16 / 24). |
| Licenses | Apache-2.0: Wan 2.2, LingBot. Revenue-capped: LTX-2.x (<$10M). Territory-limited: HunyuanVideo 1.5 (not EU/UK/KR), MiniMax-H3 (US/EU/UK/KR must apply). |
| LTX-2.5 is gated on HuggingFace | Our downloader has no HF token, so LTX-2.5 needs a token option or a manual download. LTX-2.3 is ungated. |

## 7. Recommendation

| Use case | Model | Why |
|---|---|---|
| **First video preset** | **Wan 2.2 TI2V 5B** (Q8) | Apache-2.0, one card, 12.8 GB, text-to-video and image-to-video in one model, measured ~5 min for a short 480p clip |
| Video with sound | LTX-2.3 distilled (Q4_K_M) | 8 steps, audio, ungated, one card; ~7–10 min for 5 s at 768×512 |
| Best quality, Apache | Wan 2.2 A14B + Lightning 4-step LoRAs | Best open Apache model; 10–35 min per clip |
| Best quality overall | MiniMax-H3 on 2 cards | #1 open model with audio, but slow and the license needs checking for your country |
| Skip for now | HunyuanVideo 1.5 (slow on gfx906, territory license), LingBot MoE (bf16 only, 40+8 steps), LTX-2.5 (gated) | – |

## 8. What to do before writing code

Run sd-cli once on one MI50 (same build as the image server):

```
sd-cli -M vid_gen --diffusion-model Wan2.2-TI2V-5B-Q8_0.gguf --vae Wan2.2_VAE.safetensors \
  --t5xxl umt5-xxl-encoder-Q8_0.gguf --flash-attn on --cfg-scale 5 --steps 20 \
  --sampling-method euler --flow-shift 5 -W 832 -H 480 --video-frames 49 --fps 24 \
  -p "a red fox running through snow, cinematic" -o fox.webm -v
```

Record: seconds per step, VAE decode time, peak VRAM (`rocm-smi`), and whether the WebM plays
in the browser. Repeat with `--flash-attn off` at `--video-frames 9` to see the speed gap.
If one step at 49 frames takes more than ~30 s, flash attention is the bottleneck.

## 9. What integration into llama-manager would need

The image path already supervises `sd-server`, downloads pinned files and polls jobs, so video
is an extension of `imagegen.go`:

| Area | Change |
|---|---|
| Presets | A `video` kind: `--t5xxl`, `--high-noise-diffusion-model`, `--audio-vae`, `--embeddings-connectors`, `--flash-attn on`, default frames/fps, high-noise steps/CFG, LoRAs with `is_high_noise`. |
| Request | `POST /sdcpp/v1/vid_gen` with `video_frames`, `fps`, `init_image` (image-to-video), `end_image`, `high_noise_sample_params`, `output_format: webm`; result is `result.b64_json` + `mime_type` + `frame_count`. |
| Storage | Accept a WebM signature (`1A 45 DF A3`) next to PNG (the CLAUDE.md rule says PNG only today), raise the size cap (estimate: a 5 s 720p VP8 clip is 10–40 MB), keep `nosniff` + sandbox CSP. |
| Chat UI | `<video controls>` in replies, a duration / resolution picker, "image to video" from an attached or generated image, progress with an ETA, timeouts of 60–90 min. |
| Audio | Optional ffmpeg remux to Opus if browsers won't play PCM. |

## 10. sd.cpp video reference

- `-M vid_gen` is required (even for one frame). `--video-frames N`, `--fps N`, `-i` init image, `--end-img`, `--control-video <dir>`.
- Wan: `--t5xxl umt5-xxl-encoder-Q8_0.gguf`, Wan 2.1 VAE for everything except TI2V 5B (Wan 2.2 VAE). I2V/FLF2V 2.1 also need `--clip_vision`. A14B: `--high-noise-diffusion-model`, `--high-noise-steps`, `--high-noise-cfg-scale`, `moe_boundary` 0.875 (T2V) / 0.9 (I2V).
- Lightning LoRA (lightx2v `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0`, 1.23 GB per expert): 4 + 4 steps; over the server API pass `lora: [{path, multiplier, is_high_noise}]`.
- LTX-2.3: `--llm gemma-3-12b-it-*.gguf --embeddings-connectors ...connectors.safetensors --vae ..._video_vae --audio-vae ..._audio_vae`; distilled recipe: 8 steps, CFG 1, 121 frames, 768×512, 24 fps; optional x2 latent upscaler (`--hires --hires-upscaler`).
- LTX-2.5: Gemma 4 12B "with-proj" encoder (no connectors), the **conv** video VAE only (the diffusion decoder isn't implemented).
- MiniMax-H3: `--llm qwen3vl_32b_minimax_h3-*.gguf --vae ... --audio-vae ...`, CFG 1, 24 fps fixed, width/height multiples of 32.
- HunyuanVideo 1.5: `--llm qwen_2.5_vl_7b --t5xxl byt5_small_glyphxl`, flow shift 9, 20 steps (base recipe).
- Server recipes with defaults live in `examples/server/recipes/` (`wan22-ti2v-5b`, `wan22-t2v`, `wan22-i2v`, `ltx23-distilled`, `ltx25-dev`, `minimax-h3-fl2va`, `hunyuan-video-15-t2v`, `lingbot-video-*`).
- Output: `webm` (VP8 via libwebp, needs `SD_WEBM`), `webp` (animated), `avi` (MJPEG). Audio, when the model makes it, is muxed as 16-bit PCM.

## Sources

- sd.cpp docs: https://github.com/leejet/stable-diffusion.cpp/tree/master/docs (`wan.md`, `ltx2.md`, `minimax_h3.md`, `hunyuan_video.md`, `lingbot_video.md`, `performance.md`); server API: `examples/server/api.md`; attention code: `src/core/ggml_extend.cpp`, flash-attention probe: `src/core/probe/flash_attn.cpp`, PR #2132; Wan 5B I2V crash: issue #2009 (closed 2026-09-21). Checked at commit 984eb06.
- Leaderboards: https://artificialanalysis.ai/video/leaderboard/text-to-video, https://artificialanalysis.ai/video/leaderboard/image-to-video
- Model cards: https://huggingface.co/MiniMaxAI/MiniMax-H3, https://huggingface.co/Lightricks/LTX-2.5, https://huggingface.co/Lightricks/LTX-2.3, https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B, https://huggingface.co/tencent/HunyuanVideo-1.5, https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b, https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers, https://huggingface.co/lightx2v/Wan2.2-Lightning
- GGUF files and sizes: https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF, https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF, https://huggingface.co/unsloth/LTX-2.3-GGUF, https://huggingface.co/vantagewithai/LTX-2.5-GGUF, https://huggingface.co/leejet/MiniMax-H3-GGUF, https://huggingface.co/city96/umt5-xxl-encoder-gguf
- Wan 3.0 is API-only: https://datanorth.ai/news/alibaba-launches-wan3-0-video-model
- MI50 measurements: see the sources in [image-generation-mi50.md](image-generation-mi50.md) (Wan 2.2 A14B: https://zhuanlan.zhihu.com/p/1952755076071404268); MI50 attention micro-benchmarks: https://github.com/Bosco1262/ComfyUI-aki-Deployment-for-AMD-MI50
