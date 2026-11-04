# Image generation on 5× AMD MI50 32 GB (gfx906)

Research done on 2026-11-03. Software support for gfx906 changes quickly, so
re-check versions before relying on this.

Tags: **measured** means a real run on a gfx906 card (MI50, MI60, Radeon VII
or Radeon Pro VII, which all use the same chip). **estimate** means scaled
from those runs and not measured.

## 1. Verdict

| Question | Answer |
|---|---|
| Is it possible? | **Yes.** It works today on gfx906 through ROCm or Vulkan, and there are reports from real MI50s. |
| Can we use the latest models? | **Yes.** The best open models of 2026 are small (5–9B parameters). Each fits on one 32 GB card at fp16 or Q8, and stable-diffusion.cpp and ComfyUI both support them. |
| Is it fast? | **No.** At 1024² an MI50 runs at roughly 17–20% of an RTX 3090. Turbo models take about 30 s–2 min per image. Large models with full step counts take 5–20 min or more. A short video clip takes 20–35 min. |
| Why so slow? | gfx906 has no matrix cores (MFMA/WMMA) and no working flash attention. It also has no native bf16 or fp8 math. The measured fp16 matrix-multiply speed is about 9.5 TFLOPS, against a 26.5 TFLOPS peak. |
| What's the advantage? | With 32 GB per card, models rarely need heavy quantization or offloading. Five cards can run five image workers in parallel, and one big model can be split across two cards. |
| What to integrate | **stable-diffusion.cpp `sd-server`**. It is from the same ggml family as llama.cpp and has an OpenAI-compatible API. It can be supervised like llama-server. |

## 2. Software options

| | stable-diffusion.cpp (`sd-server`) | ComfyUI (PyTorch on ROCm) | KoboldCpp / LocalAI |
|---|---|---|---|
| gfx906 | gfx906 is in the CI build targets, and the maintainer keeps it there (PR #1890, Oct 2026). | Works through community builds such as the `mixa3607/comfyui-gfx906` Docker image. | Both are built on sd.cpp. LocalAI's ROCm images were missing the gfx906 rocBLAS files. |
| Build | `-DSD_HIPBLAS=ON -DGPU_TARGETS=gfx906 -DAMDGPU_TARGETS=gfx906`, or `-DSD_VULKAN=ON` | Docker. Pass `--disable-dynamic-vram --use-split-cross-attention`. | Prebuilt binaries |
| API | `/v1/images/generations`, `/v1/images/edits`, `/sdapi/v1/txt2img`, and native jobs at `/sdcpp/v1/img_gen` | Its own node-graph API over WebSocket | A1111-style and OpenAI-style |
| Health | `GET /sdcpp/v1/models/status` returns `state: ready`. There is no `/health`. | `/system_stats` | – |
| Precision on gfx906 | fp16 is safe. Layers known to overflow are forced to f32 accumulation. | Models that only ship bf16/fp32 weights are computed in **fp32** (about 2× slower). | Same as sd.cpp |
| Multi-GPU | Each component can go on its own card: `--backend diffusion=ROCm0,te=ROCm1`. One model can be split across cards: `diffusion=ROCm0&ROCm1`. | One instance per GPU. Native multi-GPU is untested on ROCm. | Partial |
| Integration effort | Low: one more supervised process | Medium: a separate Docker service | Low |

## 3. Models (as of Nov 2026) and fit on one 32 GB MI50

"Arena Elo" is from the Artificial Analysis text-to-image arena (open-weight models).

| Model (release) | Params | Arena Elo | Fits one card? | Steps | License | fp16 on gfx906 |
|---|---|---|---|---|---|---|
| Qwen-Image-2.1 / Turbo (Sep–Oct 2026) | 7.1B | 1043 (#1 open) | Yes at fp16, with the text encoder as Q8 or on a second GPU | 40 (Turbo: 8) | Non-commercial | sd.cpp OK. ComfyUI needs `--fp16-unet`. |
| FLUX.3 Klein (Oct 2026) | 5.6B | 1043 (#1 open) | Yes, fp16 plus text encoder | 16 | Commercial output allowed under $5M revenue | sd.cpp OK. ComfyUI uses fp32. |
| Fibo 2.0 (Oct 2026) | 8.1B | 1040 | Yes | 50 (edit-turbo: 4) | CC BY-NC | ComfyUI uses fp32 |
| Ideogram 4 (Jun 2026) | 9.3B ×2 | 1025 | Text encoder on a second GPU, or Q8 | ~48 | Non-commercial | ComfyUI uses fp32 |
| Z-Image Turbo (Nov 2025) | 6.15B | 936 | Yes, fp16 easily | 8 | Apache-2.0 | fp16 works natively |
| **FLUX.2 klein 4B** / 9B (Jan 2026) | 3.9B / 9.1B | 930 (9B) | Yes | **4** | **Apache-2.0** (4B) / non-commercial (9B) | Yes |
| ERNIE-Image Turbo (Apr 2026) | 8B | 915 | Yes | 8 | Apache-2.0 | ComfyUI uses fp32 |
| Krea 2 Turbo (Jun 2026) | 12.8B | – | Yes at Q8 | 8 | Commercial under $1M revenue | Yes |
| Mage-Flow Turbo / Edit (Jul 2026) | 4.1B | – | Yes | 4 | MIT | ComfyUI uses fp32 |
| HiDream-O1 (May 2026) | 8.8B | 975 | Yes | 28–50 | MIT | ComfyUI uses fp32 |
| Qwen-Image-2512 / Edit-2511 | 20.4B | 904 | Q4 fits. Q8 needs the text encoder on a second GPU. | 50 (Lightning LoRA: 4–8) | Apache-2.0 | Black images in plain fp16. sd.cpp handles it. |
| FLUX.2 [dev] (Nov 2025) | 32B | 1000 | Q8 does not fit. Split it over 2 GPUs, or use Q4 with the text encoder on another GPU. | 28–50 | Non-commercial | Yes |
| FLUX.1 dev/schnell, SDXL, SD3.5, Chroma | 2.5–12B | older | Yes | 4–30 | Mixed | Yes |
| Wan 2.2 5B / A14B (video) | 5B / 2×14B | – | Yes / yes at Q8 | 20–40 | Apache-2.0 | Yes |
| HunyuanImage 3.0 | 83B MoE | 988 | No | – | – | – |

## 4. Benchmarks

### Measured on gfx906

| Card | Tool | Model | Res / steps | Result | vs RTX 3090 |
|---|---|---|---|---|---|
| Radeon VII | A1111, ROCm 5.7 | SD1.5 | 512², 20 | 6.4 it/s (~3 s per image) | ~47% (3090: 13.7 it/s) |
| Radeon VII | A1111 | SDXL | 512², 20 | 2.8 it/s | ~35% |
| Radeon VII | ComfyUI, ROCm 5.7 | SDXL | 1024², 20 | 28 s per image | ~20% |
| MI50 32GB | sd.cpp, **Vulkan**, Windows | SDXL | 1024², 20 | 3.9 s/it, 82 s per image | ~6%; Vulkan is the slow path |
| MI50 32GB at 150 W | ComfyUI, ROCm 7.2 | Z-Image Turbo | 512² / 1024² / 1536², 8 steps | 22 s / 57–60 s / 113–158 s | – |
| MI50 | ComfyUI, ZLUDA (Windows) | Z-Image Turbo Q8 | 1024², 8 | 57 s sampling, 93 s total | – |
| MI50 | ComfyUI (mixa3607 image) | Anima 2B | 40 steps | 150 s, or 100 s with TunableOp | – |
| MI50 | ComfyUI | Wan 2.2 5B video | 832×480, 2.5 s clip | 5 min 04 s | – |
| MI50 32GB | ComfyUI, ROCm 6.3.3 | Wan 2.2 T2V (probably A14B) | default template | 32–35 min per clip | ~6% (3090: ~2 min) |
| MI50 | ComfyUI | LTX-2.3 22B video | 1280×704, 5.2 s clip | 20 min 18 s | – |

### Estimates for the newest models (±50%)

These are scaled from Z-Image Turbo's measured ~7 s per step at 1024² for a 6B model.

| Model | Steps | Time per 1024² image (one MI50) | Images per hour on 5 cards |
|---|---|---|---|
| FLUX.2 klein 4B / Mage-Flow Turbo | 4 | 20–40 s | ~500–900 |
| Z-Image Turbo | 8 | ~60 s (measured) | ~300 |
| Qwen-Image-2.1 Turbo | 8 | 1–2 min | ~150–300 |
| FLUX.3 Klein | 16 | 1.5–3 min (sd.cpp fp16) / 3–5 min (ComfyUI fp32) | ~100–200 |
| Flux.1-dev | 20 | 2.5–5 min | ~60–120 |
| Qwen-Image-Edit-2511 + 8-step Lightning LoRA | 8 | 2–4 min | ~75–150 |
| Qwen-Image 20B, full 50 steps | 50 | 10–25 min | – |
| FLUX.2 [dev] 32B on 2 cards | 28+ | 10–20+ min | – |

No sd.cpp numbers on gfx906 over HIP exist for any modern DiT model. On the
related gfx900 chip, ComfyUI was faster than sd.cpp for Qwen-Image (24 vs 39
s per step). Benchmark one card before relying on the estimates above.

## 5. Requirements

| Item | Requirement |
|---|---|
| GPU | One MI50 32 GB per image worker. An image worker can share a card with a llama-server if VRAM allows, but they compete for compute. |
| ROCm | Officially unsupported since 6.4. Use 6.3.3 or a community 7.14/10.x build for gfx906 (the mixa3607 apt repo, Ubuntu 24.04). Build sd.cpp against the same ROCm as llama.cpp. |
| rocBLAS | ROCm 6.4+ packages lack the gfx906 Tensile kernel files. Copy them from an older or Arch `rocblas` package. |
| PyTorch (ComfyUI only) | **2.14 is the last official wheel with gfx906.** 2.15 (2026-10-28) dropped it. |
| Model files | **f16 or GGUF Q8_0/Q4_K.** On gfx906, sd.cpp runs bf16 and fp8 matrix multiplies in f32. int8 "convrot" files fall back to the CPU. |
| Host | 64 GB+ RAM, and 10–40 GB of disk per model family (text encoders are 8–17 GB each). |
| Cooling / power | Two MI50s generating at once reach a 95–99 °C hotspot. A 150–180 W power cap costs about 10%. |

## 6. Pitfalls specific to gfx906

| Pitfall | What to do |
|---|---|
| bf16 / fp8 weights | Convert to f16 or q8_0 (`--type q8_0` or `-M convert`). |
| `HSA_OVERRIDE_GFX_VERSION` set to gfx908 or gfx90a | It crashes. Never set it. |
| Flash attention on HIP | `--flash-attn auto` benchmarks it at startup. Forcing it on was 5× slower on gfx900. |
| ComfyUI's default attention (sub-quadratic) | It is 2.7× slower than split attention. Use `--use-split-cross-attention` and set `PYTORCH_TUNABLEOP_ENABLED=1` (about 33% faster). |
| Qwen-Image-Edit with 2 reference images (~12k tokens) | GPU page faults in about 3 of 14 runs. Text-to-image and edits with one reference image are fine. |
| Video VAE decode | There is no fast conv3d path, so Wan's VAE decode alone takes about 10 min. |
| `sd-server` echoes any `Origin` with credentials and has no auth | Bind it to 127.0.0.1 and reach it only through llama-manager. |
| Licenses | The two best models are non-commercial or capped by revenue. Fully free: Z-Image Turbo, FLUX.2 klein 4B, ERNIE-Image, Qwen-Image-2512/Edit-2511, Mage-Flow, HiDream-O1. |

## 7. Recommendation

| Use case | Model | Why |
|---|---|---|
| **Instant previews (integrated first)** | **FLUX.2 klein 4B** | 4 steps, Apache-2.0, ~9 GB with its text encoder |
| Fast default | Z-Image Turbo | Apache-2.0, 8 steps, measured ~60 s on an MI50 |
| Best quality | FLUX.3 Klein (under $5M revenue) or Qwen-Image-2.1 Turbo (non-commercial) | Tied #1 among open models |
| Image editing | Qwen-Image-Edit-2511 + Lightning LoRA, or FLUX.3 Klein / FLUX.2 klein | Best open editing |
| Video | Wan 2.2 5B | About 5 min per short clip; suited to batch jobs |

## 8. stable-diffusion.cpp reference (for the integration)

### FLUX.2 klein 4B files (all ungated, Apache-2.0)

| Part | Repo / file | Size |
|---|---|---|
| Diffusion | `leejet/FLUX.2-klein-4B-GGUF` / `flux-2-klein-4b-Q8_0.gguf` (or `-Q4_0`, 2.46 GB) | 4.30 GB |
| VAE | `Comfy-Org/Flux-2-Klein-4B` / `split_files/vae/flux2-vae.safetensors` | 0.34 GB |
| Text encoder | `unsloth/Qwen3-4B-GGUF` / `Qwen3-4B-Q8_0.gguf` | 4.28 GB |

The VAE in `black-forest-labs/FLUX.2-dev` (`ae.safetensors`) is gated. The
Comfy-Org copy is not.

Command (from `docs/flux2.md`):
`sd-cli --diffusion-model flux-2-klein-4b.gguf --vae flux2-vae.safetensors --llm qwen_3_4b.gguf -p "..." --cfg-scale 1.0 --steps 4`.
For editing, add `-r input.png --sampling-method euler`.
The klein **Base** variants use `--cfg-scale 4.0 --steps 20`.

### `sd-server`

- Startup: `--listen-ip` (default 127.0.0.1), `--listen-port` (default 1234), `--model-dir`, plus every sd-cli model and sampling flag. A model given on the command line loads before the port opens; if loading fails, the process exits with code 1.
- It cannot download models; the files must already be on disk.
- Endpoints:
  - OpenAI: `POST /v1/images/generations`, `POST /v1/images/edits`, `GET /v1/models`.
  - A1111: `/sdapi/v1/txt2img`, `/sdapi/v1/img2img`, `/sdapi/v1/progress`.
  - Native: `POST /sdcpp/v1/img_gen` → 202 `{id, status: queued, poll_url}`; `GET /sdcpp/v1/jobs/{id}`; `POST /sdcpp/v1/jobs/{id}/cancel`.
  - Models: `GET /sdcpp/v1/models/status` → `{state: empty|loading|ready|error}`.
- `img_gen` body: `prompt, negative_prompt, width, height, seed, batch_count, ref_images[] (base64 or data URL), sample_params{sample_method, sample_steps, guidance{txt_cfg}}, output_format`.
- Job status: `queued|generating|completed|failed|cancelled`. Progress is in `preview{step,total_steps}`. The image is in `result.images[].b64_json`. Errors are in `error{code,message}`.
- Generation is serialized: one worker and a FIFO queue of 64 jobs (429 when full). Finished jobs are kept for 600 s.
- Devices: HIP devices are named `ROCm<N>` and Vulkan devices `Vulkan<N>` (`--list-devices` shows them). `HIP_VISIBLE_DEVICES` and `GGML_VK_VISIBLE_DEVICES` restrict which cards it sees.

## Sources

- ROCm support matrix: https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html, https://github.com/ROCm/TheRock/issues/1844
- PyTorch 2.15 drops gfx906: https://github.com/pytorch/pytorch/releases/tag/v2.15.0
- Community gfx906 builds: https://github.com/mixa3607/ML-gfx906, https://skyne98.github.io/wiki-gfx906/
- stable-diffusion.cpp: https://github.com/leejet/stable-diffusion.cpp (README, `docs/backend.md`, `docs/flux2.md`, `examples/server/api.md`); MI50 SDXL run: https://github.com/leejet/stable-diffusion.cpp/issues/673
- ComfyUI on MI50: https://github.com/mixa3607/ML-gfx906/issues/27, https://github.com/Bosco1262/ComfyUI-aki-Deployment-for-AMD-MI50
- SD benchmark data: https://vladmandic.github.io/sd-data/pages/benchmark-data.json; Wan 2.2 on MI50: https://zhuanlan.zhihu.com/p/1952755076071404268
- Arena: https://artificialanalysis.ai/image/leaderboard/text-to-image
- Model cards: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B, https://huggingface.co/black-forest-labs/flux-3-image-klein, https://huggingface.co/Qwen/Qwen-Image-2.1, https://huggingface.co/ideogram-ai/ideogram-4-fp8
