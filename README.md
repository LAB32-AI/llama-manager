# llama-manager

Supervise multiple `llama.cpp` server processes (one per GPU) from a single Go
binary with a web UI and JSON API.

## Build

```bash
go build -o llama-manager .
```

## Configuration

Copy the example config and edit it:

```bash
cp example.config.yaml config.yaml
```

Key fields:

```yaml
server_bin: /usr/local/bin/llama-server   # path to your llama.cpp binary
manager_port: 8080
host: "0.0.0.0"                           # where llama-server instances listen
gpu_backend: vulkan                       # vulkan, cuda, rocm, rocm_rocr, metal
instances:
  - name: my-model
    model: /path/to/model.gguf            # local file or HF id (owner/repo)
    port: 9090
    gpu_ids: [0]
```

This service is designed to be shared on a trusted LAN: the dashboard and
`/v1` API listen on all interfaces at `manager_port`. `host` only sets where
the llama-server instances listen (`0.0.0.0` by default, so they are
reachable directly too; use `127.0.0.1` to expose them only through the
manager). There is no authentication; only run it on networks you trust.
Cross-site requests from a browser are blocked by an `Origin` check, and
requests whose `Host` header isn't an IP address, `localhost`, this machine's
hostname or an entry in `allowed_hosts` are refused (this stops DNS-rebinding
attacks). If you reach the manager through a custom DNS name, add it to
`allowed_hosts`.
`server_bin` cannot be changed through the API or config import — edit the
config file on disk and restart if you need to point at a different binary.
The same goes for the optional `tika` section (see
[Files and documents in the chat](#files-and-documents-in-the-chat)).

## Run

```bash
./llama-manager -config config.yaml
```

Then open `http://<host-ip>:8080/` (or `http://localhost:8080/` from the same machine).

### Chat history

The Chat page keeps every conversation in SQLite, in `chats.db` next to the
config file (override with `-chat-db /path/to/chats.db`). Replies are generated
by the server, so they keep going and are saved when a browser tab closes or a
phone locks; reopening the chat picks the stream back up. Anyone who can reach
the dashboard sees the same history, matching the LAN trust model. If the
database can't be opened, the manager still runs and only the chat API is off.

### GPU stats

GPU load, temperature, power and VRAM show on the Instances page and in a strip
above every chat. They come from `rocm-smi` (AMD) or `nvidia-smi` (NVIDIA). With
`gpu_backend: vulkan` the manager uses whichever of the two is installed and
numbers cards by PCI bus, the order Vulkan uses. On macOS (`metal`) it reads GPU
load from `ioreg`; macOS doesn't expose GPU temperature without root.

### Files and documents in the chat

The paperclip (or drag and drop, or paste) attaches up to 8 files per message:

- **Images** (PNG, JPEG, GIF, WebP, up to 10 MB) go to models with a vision
  projector (`mmproj`). llama-server can't decode WebP, so the browser turns
  WebP into PNG when you attach it. With the [image reader](#image-reader) on, models
  without one get a description of the image instead.
- **Text files** (code, logs, CSV, Markdown...) in UTF-8 or UTF-16. Over 2 MB,
  only the first 2 MB is sent.
- **Documents** up to 32 MB. The model gets their text (at most 2 MB of it),
  with headings, tables, and page, slide and sheet markers where the format has
  them. The chat keeps the original file too: click it to download it, or
  click "Text" to see exactly what the model read.

Who reads a document:

| | Formats |
|---|---|
| The browser, always available | PDF, Word `.docx`, PowerPoint `.pptx`, Excel `.xlsx` `.xls` `.xlsb`, LibreOffice `.odt` `.ods` `.odp`, RTF |
| [Apache Tika](https://tika.apache.org), if set up | all of the above, plus old Word and PowerPoint (`.doc`, `.ppt`), Outlook `.msg`, `.eml`, EPUB, ZIP and other archives, text in other encodings (Windows-1251, Latin-1...) and many more |

A PDF without text (a scan) is sent as page images when the instance has a
vision projector, or when the image reader is on (it then describes each page).
With Tika's `-full` image, Tika reads scans with OCR instead, for any model; its
OCR knows English, German, French, Italian, Spanish and Japanese.

The chip on each attached file shows its approximate size in tokens and warns
when the files won't fit the instance's context; llama-server refuses a
message that doesn't fit.

#### Setting up Tika

Run Tika on the llama-manager machine (users' browsers need nothing):

```bash
docker run -d --name tika --restart unless-stopped \
  -p 127.0.0.1:9998:9998 --read-only --tmpfs /tmp -m 2g \
  apache/tika:4.2.0          # or apache/tika:4.2.0-full for OCR (1.4 GB image)
```

Then add this to `config.yaml` and restart llama-manager:

```yaml
tika:
  enabled: true
  url: http://127.0.0.1:9998
  # timeout: 2m              # per document; OCR of long scans needs more
```

Settings → Documents and images shows whether Tika answers. If it stops answering, the
browser reads the formats it knows, so the chat keeps working; others are
refused with a message until Tika is back. PowerPoint `.pptx` files are always
read by the browser first, because Tika can jumble their slide order.

Notes:

- Tika 4.x needs about 2 GB of memory: with less, it warns that its parser
  processes can run out of memory. `apache/tika:3.3.3.0` works in 1 GB.
  llama-manager works with Tika 3.3 and newer.
- Keep Tika on `127.0.0.1`: it has no login. To share one Tika between several
  llama-manager machines, publish it on the LAN instead and point each
  `tika.url` at it.
- Keep the image updated. Tika parses untrusted files, and old versions have
  known flaws (CVE-2025-66516, a crafted PDF reading files, is fixed since 3.2.2).
- Like `server_bin`, the `tika` section can only be changed in `config.yaml`
  on disk: the settings API and config import refuse to change where users'
  documents are sent.
- llama-manager sends Tika only the file and its extension, never the
  original file name, and never follows redirects or environment proxies.

#### Image reader

A model without a vision projector can't see images. The image reader is a
small vision model that llama-manager runs as one more instance,
`image-reader`: it writes out an attached image's text (tables as Markdown),
then describes what the image shows, and models that can't see get that
description with the message. Models with a vision projector still get the
image itself. Add this to `config.yaml` and restart llama-manager:

```yaml
image_recognition:
  enabled: true
  model: qwen3.5-2b      # the default; or qwen3.5-4b
  gpu_ids: [4]           # default [0]
  # port: 9095           # the default; must not be an instance's port
```

| `model` | Download | VRAM | Notes |
|---|---|---|---|
| `qwen3.5-2b` (default) | 2.0 GB | about 3 GB | `unsloth/Qwen3.5-2B-GGUF:Q4_K_M`; the fastest; read an invoice, a chart, handwriting and a photo exactly in our tests |
| `qwen3.5-4b` | 3.4 GB | about 4.5 GB | `unsloth/Qwen3.5-4B-GGUF:Q4_K_M`; better published scores on documents and on questions about charts, about 1.5 times slower |

Any other vision model works too: `owner/repo:QUANT` (llama-server downloads
its `mmproj` with it), or a local `.gguf` with `"--mmproj", "/path/mmproj.gguf"`
in `image_recognition.extra_args`. The first start downloads the model; the
instance shows as starting until then.

How it works:

- An image attached while the chat's model can't see is described right away
  (the chip says "Reading with image-reader"). Images attached earlier, or
  while a model that sees was picked, are described when a message goes to a
  model that can't see; the reply shows "image-reader is reading..." meanwhile.
- Each description is saved with the image, so it is made once. In a sent
  message, "Text" on an image shows the description.
- A scanned PDF becomes page images that the reader describes.
- The instance starts with llama-manager, shows on the Instances page and in
  the GPU strip, and can be stopped and started there. It can't be edited or
  deleted from the UI.
- The reader starts with `--image-min-tokens 256`: below that, small images
  (an icon, one line of text) made Qwen3.5 2B invent things that weren't
  there. An `--image-min-tokens` in `image_recognition.extra_args` replaces it.
- llama-server picks which projector file to download with the model; recent
  builds take `mmproj-BF16.gguf`, older ones `mmproj-F16.gguf` (the log line
  "loaded multimodal model" names it). Cards without BF16 support, like the
  MI50 (gfx906), are made for F16. If reading images is slow there, pin it:
  `extra_args: ["--mmproj-url", "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-F16.gguf"]`.
- Like `tika`, the `image_recognition` section can only be changed in
  `config.yaml` on disk; the settings API and config import refuse to change
  it. Its `extra_args` are its own: the global `extra_args` don't apply to it.

### Image generation

llama-manager can run [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s
`sd-server` for one or more image models, each as its own instance on its own
GPU. Pick one in the chat and each message is a prompt: the reply is a picture,
saved in the chat. With a model that edits, attach an image (or press Edit on
one it made) and say what to change. "Make another" gives a new version with a
new seed. Other apps can use `POST /v1/images/generations` (OpenAI style) on
the manager's port, with `"model"` set to the image model's name.

Build `sd-server` for your GPUs, for example on the MI50s (gfx906). Use the
same backend as `gpu_backend` (HIP for `rocm`, `-DSD_VULKAN=ON` for `vulkan`),
or the GPU choice won't apply:

```bash
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp && mkdir build && cd build
cmake .. -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DSD_HIPBLAS=ON -DGPU_TARGETS=gfx906 -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build . --config Release   # sd-server lands in build/bin
```

Then add this to `config.yaml` and restart llama-manager:

```yaml
image_generation:
  enabled: true
  server_bin: /opt/stable-diffusion.cpp/build/bin/sd-server
  models:
    - {model: flux3-klein, gpu_ids: [4]}            # instance "flux3-klein", port 9096
    - {model: qwen-image-edit-2511, gpu_ids: [3]}   # instance "qwen-image-edit-2511", port 9097
    # - {name: fast, model: z-image-turbo, gpu_ids: [2], port: 9300, timeout: 5m}
    # - {model: qwen-image-edit-2511-q8, gpu_ids: [0, 1]}   # two GPUs: the text encoder on the second
    # - model: z-image-turbo                    # with LoRAs: see LoRAs below
    #   gpu_ids: [2]
    #   lora_dir: /home/dev/loras/z-image       # your own LoRA files
    #   loras: [z-pixel-art, z-realism]         # built-in LoRAs to download
```

Each entry takes `model` and optionally `name` (default: the model's name),
`gpu_ids` (default `[0]`), `port` (default 9096 + its position), `timeout` (per
image), `extra_args` (its own; the global ones don't apply), `lora_dir` and
`loras`. The older single-model form still works: those keys at the top of
`image_generation`, run as the instance `image-gen`.

| `model` | Download | VRAM | Steps | Edits | License |
|---|---|---|---|---|---|
| `flux2-klein-4b` (single-model default) | 8.9 GB | about 10 GB | 4 | yes, up to 4 images | Apache-2.0 |
| `flux3-klein` | 10.8 GB | about 12 GB | 16 | yes, up to 10 images | FLUX Kommunity: commercial output only under US$5M revenue |
| `z-image-turbo` | 11.2 GB | about 12 GB | 8 | no: text to image only | Apache-2.0 |
| `qwen-image-2.1-turbo` | 18.2 GB | about 20 GB | 8 | yes, up to 4 images | Qwen Research License: non-commercial |
| `qwen-image-edit-2511` | 23.4 GB | about 24 GB | 4 (Lightning) | edits only, up to 3 images | Apache-2.0 |
| `qwen-image-edit-2511-q8` | 32.9 GB | about 26 GB + 9 GB on two GPUs | 4 (Lightning) | edits only, up to 3 images | Apache-2.0 |

Files several models share (the FLUX.2 VAE, the Qwen3-4B text encoder) are
downloaded once. The files:

- `flux2-klein-4b`: FLUX.2 klein 4B Q8_0, Qwen3-4B Q8_0, FLUX.2 VAE.
- `flux3-klein`: FLUX.3 Image Klein Q8_0, Qwen3-VL-4B Q8_0, FLUX.2 VAE.
- `z-image-turbo`: Z-Image Turbo Q8_0, Qwen3-4B Q8_0, FLUX.1 VAE.
- `qwen-image-2.1-turbo`: unsloth's Qwen-Image 2.1 Turbo Q8_0, Qwen3-VL-8B Q8_0
  and its F16 vision weights, the Qwen-Image 2.1 VAE. It runs Turbo's fixed
  8-step schedule.
- `qwen-image-edit-2511`: Qwen-Image-Edit-2511 Q4_K_M (`-q8`: Q8_0),
  Qwen2.5-VL-7B Q8_0 and its F16 vision weights, the Qwen-Image VAE, and
  lightx2v's 4-step Lightning LoRA, applied to every image. It starts with
  `qwen_image_zero_cond_t=true`, without which 2511's edits degrade.

How it works:

- The first start downloads the files into the HuggingFace cache, each pinned
  to a commit and checked against its SHA-256, one model after another. A file
  already fetched from the Models page (same name and size under `main`) is used
  as is. Meanwhile the model shows as downloading, with progress in its log, and
  it starts by itself when its files are in.
- `sd-server` always listens on 127.0.0.1: it has no auth and answers any
  browser origin, so the chat and other apps reach it only through
  llama-manager. It loads every weight before it listens (`--eager-load`), so
  "running" means ready.
- With two GPUs in `gpu_ids`, the model runs on the first and its text encoder
  on the second (`--backend all=rocm0,te=rocm1`, or `vulkan0`/`cuda0`).
- The chat sends the size picked in the composer (1024 × 1024 by default) and
  the chat's seed setting; with no seed, every image gets a random one, shown
  under it. An edit keeps the shape of the attached image, at about the picked
  size's number of pixels. While an image forms, the reply shows the step and a
  rough preview; Stop cancels the job in `sd-server`.
- A model that can't edit refuses an attached image instead of ignoring it, and
  an edit-only model refuses a message without one; the composer warns first.
- A chat model asked about the image later sees the note "[An image was
  generated for this request.]", not the picture.
- `/v1/images/generations` uses each model's own defaults (steps, CFG,
  schedule) and LoRA (Lightning for the Qwen edit models).
- On the MI50 expect roughly 20–40 s per 1024 × 1024 image with the 4B models
  and minutes with the 20B Qwen edit model (estimates: see
  `docs/research/image-generation-mi50.md`). The presets use Q8_0, Q4_K_M or
  F16 files where it matters: gfx906 has no BF16 or FP8 math.
- Like `image_recognition`, the `image_generation` section can only be changed
  in `config.yaml` on disk; the settings API and config import refuse to
  change it, and the UI can't edit or delete the image models.

#### LoRAs

A LoRA adds a style or a skill to an image model. In the chat, open Chat
settings (or press the LoRA button next to the image size) to pick up to four
for the image model, each with a strength from 0 to 2. The choice is saved with
the chat, for each image model. The LoRAs used and their strengths are shown
under each image. Many LoRAs are trained on trigger words: the picker shows
them, and "Add to message" puts them in the prompt.

Built-in LoRAs are pinned like the models (commit and SHA-256) and downloaded
with the model's files when you list them under `loras`. All are Apache-2.0:

| `loras` name | For `model` | What it does | Prompt with |
|---|---|---|---|
| `z-pixel-art` | `z-image-turbo` | Retro pixel art (162 MB) | `Pixel art style.` |
| `z-realism` | `z-image-turbo` | More photographic skin, light and detail (81 MB) | `Realism` |
| `z-classic-painting` | `z-image-turbo` | Old-master oil painting (162 MB) | `class1cpa1nt` |
| `klein-pixel-art` | `flux2-klein-4b` | Game sprites; strength 0.85 to 1.4 (310 MB) | `pixel art sprite, game asset, transparent background` |
| `klein-anime` | `flux2-klein-4b` | Anime illustration (176 MB) | `anime_style` |
| `klein-sketch-to-image` | `flux2-klein-4b` | Attach a sketch, describe the finished picture (88 MB) | |
| `qie-multiple-angles` | `qwen-image-edit-2511`, `-q8` | The attached picture from another camera position (281 MB) | `<sks> front view eye-level shot medium shot` |
| `qie-anything2real` | `qwen-image-edit-2511`, `-q8` | Drawings, anime and paintings to photos (585 MB) | `change the picture 1 to realistic photograph, ...` |
| `qie-object-remover` | `qwen-image-edit-2511`, `-q8` | Removes what you name, keeps the background (225 MB) | `Remove the ...` |

Your own LoRAs go in the model's `lora_dir`, a folder on the server:
`.safetensors` (or `.gguf`) files, in subfolders up to two levels deep. They
show up in the picker within a few seconds, without a restart. A LoRA only
works with the model family it was trained for (a Z-Image LoRA on Z-Image
Turbo, a FLUX.2 klein 4B LoRA on `flux2-klein-4b`), so give each model its own
folder. sd.cpp reads LoRA, LoHa and LoKr files in the kohya, diffusers and
ComfyUI layouts.

How it works:

- sd-server reads LoRAs from one folder per model (`--lora-model-dir`) and
  takes only real files inside it. So built-in LoRAs are linked into the
  folder's `llama-manager/<model>/` subfolder: hard links to the HuggingFace
  cache's copies (no extra disk), or copies when the folder is on another disk.
  llama-manager removes files from that subfolder when they leave `loras`;
  leave it to llama-manager. A model without `lora_dir` keeps its built-in
  LoRAs in `llama-manager-loras/<model>/` next to the HuggingFace cache.
- Every LoRA a chat or `/v1/images/generations` asks for must be one the model
  lists; anything else is refused. Over `/v1`, add
  `"lora": [{"path": "z-pixel-art", "multiplier": 0.8}]` (a built-in name, or
  a file's path in `lora_dir`).
- The models are quantized, so sd.cpp applies LoRAs while sampling instead of
  merging them into the weights: each step is somewhat slower and uses more
  VRAM, but LoRAs change from one image to the next without reloading.

### Text to speech

With a voice model, a play button under each chat reply reads it aloud. The
voice and language are picked under Chat settings → Read aloud (saved in the
browser). Other apps can use `POST /v1/audio/speech` (OpenAI style, MP3 by
default with ffmpeg installed, else WAV; see
[OpenAI-compatible API](#openai-compatible-api)) on the manager's port, with `"model"` set to the voice model's name;
OpenAI's names (`tts-1`, voice `alloy`...) get the first model and its first
voice.

| `model` | Runs in | Download | VRAM | Voices | Languages | License |
|---|---|---|---|---|---|---|
| `ming3-tts` (default) | audio.cpp | 1.7 GB | about 2 GB | Emma, James (American), Charlotte, George (British) | en, zh, ja, ko, de, fr, es | MIT |
| `qwen3-tts` | llama-server | 2.3 GB | about 3 GB | yours: a 3–20 s recording per voice | en, ru, zh, de, fr, es, it, pt, ja, ko | Apache-2.0 |
| `lfm3-audio` | llama-server | 2.2 GB | about 3 GB | female, male | en, de, fr, es, it, pt, nl, da, sv, fi, pl, tr, el, ar, he, ja, ko, zh | LFM Open License: commercial use only under US$10M revenue |

Ming 3.0 sounds best (as good as ElevenLabs v3 in blind votes; see
`docs/research/text-to-speech-mi50.md`), but it doesn't speak Russian; use
`qwen3-tts` for that. `qwen3-tts` and `lfm3-audio` run in llama-server's
`/tts`, which needs llama.cpp v0.8.0 (b11960) or newer. `ming3-tts` runs in
[audio.cpp](https://github.com/0xShug0/audio.cpp)'s `audiocpp_server`; build it
for your GPUs, for example on the MI50s:

```bash
git clone https://github.com/0xShug0/audio.cpp && cd audio.cpp
scripts/build_linux.sh --backend hip --gpu-targets gfx906 --model-set custom --models ming3_tts \
  --deployment-build --target audiocpp_server
# or --backend vulkan if the HIP build misbehaves on gfx906 (audio.cpp calls HIP a test build)
```

Then add this to `config.yaml` and restart llama-manager:

```yaml
text_to_speech:
  enabled: true
  server_bin: /opt/audio.cpp/build/linux-hip-release/bin/audiocpp_server   # for ming3-tts
  # backend: vulkan     # audio.cpp's backend; default follows gpu_backend (rocm: hip)
  models:
    - {model: ming3-tts, gpu_ids: [4]}       # instance "ming3-tts", port 9110
    - {model: lfm3-audio, gpu_ids: [4]}      # port 9111
    - model: qwen3-tts                       # port 9112
      gpu_ids: [4]
      voices:
        - {name: me, file: /home/dev/voices/me.wav}   # WAV or MP3, 3–20 s of clear speech
```

Each entry takes `model` and optionally `name` (default: the model's name),
`gpu_ids` (one GPU, default `[0]`), `port` (default 9110 + its position),
`timeout` (per part, default 2m), `extra_args` (its own) and, for `qwen3-tts`
only, `voices`. A single model can also be set at the top of `text_to_speech`
(`model`, `gpu_ids`, `port`, `voices`, `extra_args`).

How it works:

- The manager turns the reply into plain speech: code blocks, images and links'
  URLs are left out, lists and headings become sentences, tables are read row
  by row. It splits the text into parts (a short first one, so playback starts
  after a sentence) and the browser plays each part while the next is spoken.
  Parts already spoken are kept in memory (up to 128 MB), so playing a reply
  again starts at once.
- The language is guessed from the reply (script, then common words for Latin
  languages) unless one is picked in Chat settings. A model asked for a
  language it doesn't speak says so instead of reading it badly.
- `lfm3-audio` and `qwen3-tts` run in llama-server with their own small
  context (4096), one slot and F16 KV cache; the global `extra_args` don't
  apply. llama-server downloads them with `-hf` on the first start.
- `ming3-tts`'s file is downloaded into the HuggingFace cache pinned to a
  commit and checked against its SHA-256, like the image models. The manager
  writes `audiocpp_server`'s config (into a private temp folder) and starts it
  on 127.0.0.1 with `--no-ui`, so it is reached only through llama-manager.
- The voice models show on the Instances page and can be stopped and started
  there, but not edited, deleted or picked as a chat model.
- If a llama-server voice model answers "this server does not support audio
  generation", it started without its audio part (the `mmproj` file next to
  the model): check its log, or name the file with `--mmproj-url` in its
  `extra_args`.
- MI50 speed is not measured yet; the estimate is 2–5 times faster than real
  time. Like the other model sections, `text_to_speech` can only be changed in
  `config.yaml` on disk: it names binaries to run and voice files to read.

## OpenAI-compatible API

Apps that speak the OpenAI API (the OpenAI SDKs, Open WebUI, LibreChat,
opencode, aider, Codex CLI, n8n...) use llama-manager as their server: set the
base URL to `http://<host>:8080/v1` (apps given the URL without `/v1` work
too) and the model to an instance's name. `GET /v1/models` lists the running
instances.

| Endpoint | Served by | Notes |
|---|---|---|
| `GET /v1/models`, `GET /v1/models/{id}` | llama-manager | Running instances only; `created` is when the instance started. |
| `POST /v1/chat/completions` | llama-server | Streaming, tools, `response_format`, images (with an mmproj) and the rest as llama-server supports them. |
| `POST /v1/completions` | llama-server | Legacy text completions. |
| `POST /v1/responses`, `/v1/responses/input_tokens` | llama-server | The Responses API (Codex CLI, the Agents SDK). Stateless: llama-server keeps no stored responses. |
| `POST /v1/embeddings` | llama-server | Needs an embedding model started with `--embeddings` in its `extra_args`. |
| `POST /v1/rerank` | llama-server | Not OpenAI's, but Open WebUI and others use it; needs `--reranking`. |
| `POST /v1/messages`, `/v1/messages/count_tokens` | llama-server | Anthropic's Messages API, for apps written against it. |
| `POST /v1/images/generations` | sd-server | `dall-e-*` and `gpt-image-*` names get the first image model. Without `response_format` each picture comes as both `b64_json` and `url` (kept for an hour, no key needed: the link can't be guessed). `stream: true` sends `image_generation.completed` events. |
| `POST /v1/images/edits` | sd-server | Multipart like OpenAI's: `image[]` (or `image`), `mask`, `prompt`. Needs a model that edits (FLUX.2 klein, Qwen-Image-Edit...). |
| `POST /v1/audio/speech` | voice models | `response_format` `mp3` (the default, as at OpenAI), `opus`, `aac`, `flac` need ffmpeg; `wav` and `pcm` don't. `speed` needs ffmpeg; `stream_format: "sse"`. |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | a model that hears | See below. |
| `GET /v1/realtime` (WebSocket) | llama-server | llama-server's own Realtime API (LFM3), relayed. |

Errors come as OpenAI's `{"error": {"message", "type", "param", "code"}}`, and
every answer carries an `X-Request-Id`. OpenAI's cloud services (files, batches,
vector stores, fine-tuning, assistants, moderation...) answer 404 with code
`unsupported_endpoint`.

Settings → OpenAI API:

- **API keys**: empty (the default) accepts any key or none. With keys set,
  apps must send one as `Authorization: Bearer <key>` (`x-api-key` and
  `api-key` work too; a browser WebSocket sends it as the subprotocol
  `openai-insecure-api-key.<key>`). Only `/v1` asks for it: the dashboard
  stays open to everyone who can reach the port, so expose only `/v1` if you
  put the manager on the internet. Set the global `host` to `127.0.0.1` so the
  llama-servers themselves can't be reached around the key.
- **Fallback model**: answers requests for model names that aren't running
  here, for apps with a fixed model (`gpt-4o`...). Off by default: those get
  404 `model_not_found`.

`cors_origins` in `config.yaml` lists the web pages that may call `/v1` from a
browser, such as a web chat app on another port (`["http://app.lan:3000"]`,
or `["*"]` for any page). Apps that aren't web pages don't need it. Like
`server_bin`, it can only be changed in the file. Apps in Docker can reach the
host as `host.docker.internal`; other DNS names go in `allowed_hosts` (or send
a valid API key, which stands in for that check).

**Speech to text.** `/v1/audio/transcriptions` and `/v1/audio/translations`
send the audio to a llama-server model that hears it: a model with an audio
projector (`mmproj`), such as Voxtral Mini 3B
(`ggml-org/Voxtral-Mini-3B-2507-GGUF`) or Qwen2.5-Omni 3B/7B, run as an
ordinary instance. `model` picks it by name; `whisper-1` and other names get
the fallback model if it hears, else the first running model that does
(llama-server's `/props` says which). `response_format` can be `json`, `text`,
`srt`, `vtt` or `verbose_json`; the model gives no timestamps, so subtitles
spread the length of the audio over the sentences. `stream: true` sends
`transcript.text.delta` events. llama-server reads WAV and MP3; FLAC, OGG,
WebM and M4A need ffmpeg on the server, which converts them. Files are limited
to 25 MB, as at OpenAI.

**ffmpeg** (optional, `apt install ffmpeg`) is found on `PATH`. Without it
speech answers WAV for every format (MP3, the default, included) and ignores
`speed`.

## Install as systemd service

```bash
bash service_install.sh
```

Run it as your normal user: it builds the binary with your Go toolchain, then
asks for `sudo` to install it to `/usr/local/bin`, copies `example.config.yaml`
to `/etc/llama-manager/config.yaml` (only if no config exists yet) and enables
the service under your account. On a fresh install the service is enabled but
not started; edit the config, then `sudo systemctl start llama-manager`.
Re-running the installer upgrades in place and restarts the service.

View logs:

```bash
journalctl -u llama-manager -f
```

## Development

```bash
go test -race ./...
go vet ./...
```
