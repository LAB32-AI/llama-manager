# Installing llama-manager on a new server

A runbook for setting up the whole service on a fresh machine, or moving it
from an old one: the GPU stack, every program llama-manager runs, the models,
and the files to carry over. It is written for the production rig (5× AMD
Instinct MI50 32 GB, gfx906, Ubuntu); notes say where other GPUs differ.

Steps:

1. [Inventory: what lives where](#1-inventory-what-lives-where)
2. [On the old server: record versions and copy files](#2-on-the-old-server-record-versions-and-copy-files)
3. [GPU drivers and ROCm](#3-gpu-drivers-and-rocm)
4. [Build tools and Go](#4-build-tools-and-go)
5. [llama.cpp (llama-server)](#5-llamacpp-llama-server)
6. [stable-diffusion.cpp (sd-server), for image generation](#6-stable-diffusioncpp-sd-server-for-image-generation)
7. [audio.cpp (audiocpp_server), for the Ming voice](#7-audiocpp-audiocpp_server-for-the-ming-voice)
8. [Optional services: ffmpeg, Tika, Tailscale](#8-optional-services-ffmpeg-tika-tailscale)
9. [llama-manager itself](#9-llama-manager-itself)
10. [config.yaml](#10-configyaml)
11. [Models: download or copy](#11-models-download-or-copy)
12. [First start and checks](#12-first-start-and-checks)
13. [Troubleshooting](#13-troubleshooting)
14. [Updating later](#14-updating-later)

## 1. Inventory: what lives where

| What | Where (defaults) | Carry over? |
|---|---|---|
| llama-manager binary | `/usr/local/bin/llama-manager` | No: rebuild from the repo |
| systemd unit | `/etc/systemd/system/llama-manager.service` (runs as the installing user) | No: the installer writes it |
| **Config** | `/etc/llama-manager/config.yaml` | **Yes** |
| **Chat history, stored API responses, API key usage, dashboard statistics** | `/etc/llama-manager/chats.db` (SQLite, WAL mode; next to the config unless `-chat-db` is given) | **Yes** |
| llama-server | `server_bin` in the config (e.g. `~/workspace/llama.cpp/build/bin/llama-server`) | No: rebuild |
| sd-server | `image_generation.server_bin` (e.g. `/opt/stable-diffusion.cpp/build/bin/sd-server`) | No: rebuild |
| audiocpp_server | `text_to_speech.server_bin` (e.g. `/opt/audio.cpp/build/linux-hip-release/bin/audiocpp_server`) | No: rebuild |
| Models fetched with `-hf` (chat instances, image reader, prompt enhancer, Qwen3-TTS, LFM3-Audio) | `~/.cache/huggingface/hub/` of the service user (newer llama.cpp); older builds and `LLAMA_CACHE` use `~/.cache/llama.cpp/` | Optional: copying saves the downloads |
| Image and Ming voice model files (pinned by commit and SHA-256) | `~/.cache/huggingface/hub/` (or `HF_HUB_CACHE`) | Optional: copying saves the downloads |
| Built-in LoRAs for models without `lora_dir` | `~/.cache/huggingface/llama-manager-loras/` | No: re-made from the HF cache |
| **Your own LoRAs** | each image model's `lora_dir` (e.g. `/home/dev/loras/z-image`) | **Yes** |
| **Voice recordings for qwen3-tts** | `text_to_speech…voices[].file` (e.g. `/home/dev/voices/me.wav`) | **Yes** |
| GGUF files outside the cache | the folders in `model_dirs` and any `.gguf` path in `instances[].model` | **Yes**, if you use them |
| API keys, allowed hosts, CORS | inside `config.yaml` | (with the config) |
| Apache Tika | a Docker container, no state | No: re-run the container |
| Tailscale identity | `/var/lib/tailscale/` | No: sign the new machine in (see §8) |

## 2. On the old server: record versions and copy files

Write down what runs today, so the new server can start from the same,
known-good versions (gfx906 support changes between releases):

```bash
cat /opt/rocm/.info/version                     # ROCm version
apt list --installed 2>/dev/null | grep -Ei '^(rocm|hip|rocblas|amdgpu)' > ~/old-rocm-packages.txt
lsb_release -ds; uname -r                       # OS and kernel
go version
grep -E 'server_bin|gpu_backend' /etc/llama-manager/config.yaml
"$(grep -m1 '^server_bin' /etc/llama-manager/config.yaml | awk '{print $2}')" --version   # llama.cpp build number
git -C ~/workspace/llama.cpp rev-parse HEAD     # adjust the paths to yours
git -C /opt/stable-diffusion.cpp rev-parse HEAD
git -C /opt/audio.cpp rev-parse HEAD
git -C ~/workspace/llama-manager rev-parse HEAD # this repo
tailscale version; docker images apache/tika
rocm-smi --showproductname; rocm-smi --showmeminfo vram
```

Copy the state (stop the service first, so `chats.db` and its `-wal` file are
consistent):

```bash
sudo systemctl stop llama-manager
mkdir -p ~/migrate
sudo cp -a /etc/llama-manager/config.yaml /etc/llama-manager/chats.db* ~/migrate/
cp -a ~/voices ~/loras ~/migrate/ 2>/dev/null   # your voice recordings and LoRA folders, wherever they are
sudo chown -R "$USER" ~/migrate
sudo systemctl start llama-manager              # the old server can keep running until the switch
```

Without stopping, `sqlite3 /etc/llama-manager/chats.db ".backup '$HOME/migrate/chats.db'"`
makes a consistent copy of a live database.

Models are large (an image model is 9 to 33 GB, a 24B chat model about 13 GB).
Copying the caches over the network is usually faster than downloading them
again, and it keeps exactly the same files:

```bash
# from the new server, once the service user exists there
rsync -a --info=progress2 old-server:.cache/huggingface/ ~/.cache/huggingface/
rsync -a --info=progress2 old-server:.cache/llama.cpp/ ~/.cache/llama.cpp/      # if it exists
```

## 3. GPU drivers and ROCm

AMD dropped gfx906 (MI50/MI60, Radeon VII) from official ROCm support after
6.3; see `docs/research/image-generation-mi50.md` §5 for the details. Use the
same ROCm version as the old server (§2). If you start fresh, ROCm 6.3.x on
Ubuntu 24.04 is the safe choice; community builds (7.x/10.x for gfx906) also
exist.

- Install ROCm with AMD's `amdgpu-install` package for your ROCm version and
  Ubuntu release, from `repo.radeon.com` (`sudo amdgpu-install --usecase=rocm`),
  then reboot.
- ROCm packages from 6.4 on lack rocBLAS's gfx906 kernel files; with those,
  copy them from a 6.3 or Arch `rocblas` package (same research doc).
- Let the service user use the GPUs: `sudo usermod -aG render,video $USER`, then
  log out and in.
- Check: `rocminfo | grep -c gfx906` prints 5, and `rocm-smi` lists five cards
  with 32 GB each.

With `gpu_backend: vulkan` instead, install the Mesa RADV driver
(`sudo apt install mesa-vulkan-drivers vulkan-tools`) and check with
`vulkaninfo --summary`. Every program below must then be built for Vulkan, and
`rocm-smi` (from ROCm) is still what the dashboard reads GPU stats from.

## 4. Build tools and Go

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git curl pkg-config \
  libcurl4-openssl-dev libssl-dev python3 sqlite3
```

llama-manager needs Go 1.25.6 or newer (`go.mod`). Install it from
<https://go.dev/dl/>, for example:

```bash
curl -LO https://go.dev/dl/go1.25.6.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.25.6.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.profile && . ~/.profile
go version
```

## 5. llama.cpp (llama-server)

llama-server runs the chat models, the image reader, the prompt enhancer and
the Qwen3-TTS and LFM3-Audio voices. They need a recent build: llama.cpp
v0.8.0 (b11960) or newer for `/tts`, and Qwen3.5 support (the `qwen35`
architecture) for the image reader and the prompt enhancer.

```bash
git clone https://github.com/ggml-org/llama.cpp ~/workspace/llama.cpp
cd ~/workspace/llama.cpp
# git checkout <the commit from §2>, or stay on the latest
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
build/bin/llama-server --version
```

For Vulkan, replace the cmake line with
`cmake -S . -B build -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release`
(needs `libvulkan-dev` and `glslc`).

`server_bin` in the config points at `build/bin/llama-server`.

## 6. stable-diffusion.cpp (sd-server), for image generation

Build it with the same backend as `gpu_backend` (HIP for `rocm`), or the GPU
choice won't apply:

```bash
sudo git clone --recursive https://github.com/leejet/stable-diffusion.cpp /opt/stable-diffusion.cpp
sudo chown -R "$USER" /opt/stable-diffusion.cpp
cd /opt/stable-diffusion.cpp && mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DSD_HIPBLAS=ON -DGPU_TARGETS=gfx906 -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build . --config Release   # sd-server lands in build/bin
```

If `clang` isn't on `PATH`, use ROCm's: `-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++`. For Vulkan: `-DSD_VULKAN=ON`
instead of the HIP flags.

## 7. audio.cpp (audiocpp_server), for the Ming voice

Only needed for the `ming3-tts` voice (Qwen3-TTS and LFM3-Audio run in
llama-server):

```bash
sudo git clone https://github.com/0xShug0/audio.cpp /opt/audio.cpp
sudo chown -R "$USER" /opt/audio.cpp && cd /opt/audio.cpp
scripts/build_linux.sh --backend hip --gpu-targets gfx906 --model-set custom --models ming3_tts \
  --deployment-build --target audiocpp_server
# or --backend vulkan if the HIP build misbehaves on gfx906 (audio.cpp calls HIP a test build)
```

The binary is `build/linux-hip-release/bin/audiocpp_server` (a Vulkan build
writes its own `build/linux-…-release` folder).

## 8. Optional services: ffmpeg, Tika, Tailscale

**ffmpeg** (MP3/Opus/AAC/FLAC speech, `speed`, transcribing FLAC/OGG/WebM/M4A):

```bash
sudo apt install -y ffmpeg
```

**Apache Tika** (old Office formats, email, EPUB, archives, other encodings;
`-full` adds OCR). Install Docker, then:

```bash
docker run -d --name tika --restart unless-stopped \
  -p 127.0.0.1:9998:9998 --read-only --tmpfs /tmp -m 2g \
  apache/tika:4.2.0          # or the tag from §2; apache/tika:4.2.0-full for OCR
```

**Tailscale Funnel** (`/v1` on the internet). A new machine is a new Tailscale
node:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up                         # sign in to the same tailnet
sudo tailscale set --operator=$USER       # the service user; without it Settings shows "Access denied"
```

The public address is `https://<machine name>.<tailnet>.ts.net/v1`. To keep
the old address for the people who use it, remove the old machine in the
Tailscale admin console and give the new one the same name (or rename it
there). The first start asks once to turn on Funnel and HTTPS for the tailnet
if they are off: Settings → Internet access shows the link.

## 9. llama-manager itself

```bash
git clone <this repository> ~/workspace/llama-manager
cd ~/workspace/llama-manager
# git checkout <the commit from §2>, or stay on the latest
bash service_install.sh
```

Run it as the user the service should run as (not root). It builds the
binary, installs it to `/usr/local/bin`, creates `/etc/llama-manager/` owned by
that user, installs the systemd unit and enables it. On a fresh machine it
copies `example.config.yaml` and does **not** start the service.

Then put the old state in place:

```bash
sudo systemctl stop llama-manager 2>/dev/null
sudo install -m 640 -o "$USER" -g "$(id -gn)" ~/migrate/config.yaml /etc/llama-manager/config.yaml
sudo install -m 640 -o "$USER" -g "$(id -gn)" ~/migrate/chats.db /etc/llama-manager/chats.db
# chats.db-wal / chats.db-shm only exist if the copy was made while it ran; copy them too then
```

A database from an older llama-manager is upgraded on the first start (the
schema version is in `chatstore.go`).

## 10. config.yaml

Go through the copied config (or `example.config.yaml`, which documents every
key) and fix whatever differs on the new machine:

- `server_bin`, `image_generation.server_bin`, `text_to_speech.server_bin`:
  the paths from §5–7.
- `gpu_backend`: `rocm` for the HIP builds, `vulkan` for Vulkan builds.
- `gpu_ids` everywhere: they follow the order the backend numbers the cards.
- `lora_dir`, `voices[].file`, `model_dirs`, local `.gguf` paths: where you
  copied them.
- `allowed_hosts`: add the new machine's DNS name if people reach it by name.
- `public_api`, `tika`: keep them if you set those services up in §8.

The file-only sections (`server_bin`, `tika`, `image_recognition`,
`image_generation`, `text_to_speech`, `prompt_enhancer`, `public_api`,
`cors_origins`) can't be changed from the UI or by importing a config: edit the
file and restart.

A full production layout for five MI50s, with the service models together on
GPU 4 (about 31 of 32 GB there):

```yaml
server_bin: /home/dev/workspace/llama.cpp/build/bin/llama-server
manager_port: 8080
gpu_backend: rocm
host: "0.0.0.0"
ngl: 99
context_length: 16384
cache_type_k: q8_0
cache_type_v: q8_0
jinja: true

api_keys: []                    # or your named keys, copied from the old config

tika:
  enabled: true
  url: http://127.0.0.1:9998

image_recognition:              # Qwen3.5-2B, ~3 GB
  enabled: true
  model: qwen3.5-2b
  gpu_ids: [4]

prompt_enhancer:                # FLUX.3 klein's prompt upsampler, ~7-8 GB
  enabled: true
  model: flux3-klein-pup
  gpu_ids: [4]

image_generation:
  enabled: true
  server_bin: /opt/stable-diffusion.cpp/build/bin/sd-server
  models:
    - {model: flux3-klein, gpu_ids: [4]}           # ~12 GB

text_to_speech:
  enabled: true
  server_bin: /opt/audio.cpp/build/linux-hip-release/bin/audiocpp_server
  models:
    - {model: ming3-tts, gpu_ids: [4]}             # ~2 GB
    - model: qwen3-tts                             # ~3 GB
      gpu_ids: [4]
      voices:
        - {name: me, file: /home/dev/voices/me.wav}

public_api:
  tailscale_funnel: true

instances:                      # the chat models, one per GPU 0-3
  - {name: chat-gpu0, model: "owner/repo-GGUF:QUANT", port: 9090, gpu_ids: [0], auto_start: true}
  - {name: chat-gpu1, model: "owner/repo-GGUF:QUANT", port: 9091, gpu_ids: [1], auto_start: true}
  - {name: chat-gpu2, model: "owner/repo-GGUF:QUANT", port: 9092, gpu_ids: [2], auto_start: true}
  - {name: chat-gpu3, model: "owner/repo-GGUF:QUANT", port: 9093, gpu_ids: [3], auto_start: true}
```

Ports the manager uses by default (keep chat instances off them): 8080 (the
manager), 9095 (image reader), 9096 + n (image models), 9110 + n (voice
models), 9120 (prompt enhancer); Tika 9998 on 127.0.0.1.

VRAM per service model, to plan which card takes what: image reader 3 GB
(`qwen3.5-4b`: 4.5 GB), prompt enhancer 7–8 GB (`--no-mmproj` saves about
1 GB if you never enhance edits), FLUX.3 klein 12 GB, FLUX.2 klein 4B 10 GB,
Z-Image Turbo 12 GB, Qwen-Image 2.1 Turbo 20 GB, Qwen-Image-Edit-2511 24 GB,
Ming 2 GB, Qwen3-TTS 3 GB, LFM3-Audio 3 GB. Leave 1–2 GB free per card for
spikes; `rocm-smi --showmeminfo vram` shows what is really used.

## 11. Models: download or copy

Nothing has to be downloaded by hand: on the first start llama-manager starts
every instance, and each fetches what it lacks.

| Model | Fetched by | Size |
|---|---|---|
| Chat instances | llama-server `-hf` (or a local `.gguf`) | whatever you picked (a 24B IQ4_XS is about 13 GB) |
| Image reader `qwen3.5-2b` | llama-server `-hf unsloth/Qwen3.5-2B-GGUF:Q4_K_M` + mmproj | 2.0 GB |
| Prompt enhancer `flux3-klein-pup` | llama-server `-hf AtomicChat/FLUX-3-Image-Klein-pup-GGUF:AD-IQ4_XS` + its F16 mmproj | 5.5 + 0.9 GB |
| Image models | llama-manager, pinned files, one model after another | flux3-klein 10.8 GB, flux2-klein-4b 8.9 GB, z-image-turbo 11.2 GB, qwen-image-2.1-turbo 18.2 GB, qwen-image-edit-2511 23.4 GB (`-q8` 32.9 GB) |
| Built-in LoRAs listed under `loras` | llama-manager, pinned, with their model | 81–585 MB each |
| `ming3-tts` | llama-manager, pinned | 1.7 GB |
| `qwen3-tts`, `lfm3-audio` | llama-server `-hf` | 2.3 GB, 2.2 GB |

Meanwhile instances show as starting (llama-server downloads) or downloading
(image and voice files, with progress in their log). Copying the caches from
the old server (§2) skips all of this.

To fetch the llama-server models before the first start, run each `-hf` once
as the service user, e.g.
`~/workspace/llama.cpp/build/bin/llama-server -hf AtomicChat/FLUX-3-Image-Klein-pup-GGUF:AD-IQ4_XS --port 9999`
and stop it with Ctrl+C when it says it is listening.

## 12. First start and checks

```bash
sudo systemctl start llama-manager
journalctl -u llama-manager -f
```

Then open `http://<server>:8080/` and check:

- [ ] **Instances**: every instance reaches Running (the first start can take a
  while for downloads). Open the log of any that crashes.
- [ ] **Settings → Documents, images and speech**: Tika answers, the image
  reader, image models, prompt enhancer and voice models are all green.
- [ ] **Chat**: a chat model answers; an attached image is described for a
  model that can't see.
- [ ] **Image**: pick `flux3-klein`, write "a lake", press **Enhance**: a long
  description replaces it. Send it: a picture comes back.
- [ ] **Edit**: attach a picture, write a change, press Enhance, send.
- [ ] **Read aloud**: the play button under a reply speaks.
- [ ] **History**: old chats from `chats.db` are listed; the Dashboard shows the
  old statistics.
- [ ] **API**: `curl http://<server>:8080/v1/models` lists the running models;
  Settings → OpenAI API shows the keys.
- [ ] **Internet**: Settings → Internet access shows the public address, and
  `curl https://<machine>.<tailnet>.ts.net/v1/models -H "Authorization: Bearer <key>"`
  answers from outside.
- [ ] **GPU stats**: the GPU strip and Dashboard show load, VRAM and temperature
  for all five cards.
- [ ] Reboot once: everything with `auto_start` (and every service model) comes
  back by itself.

## 13. Troubleshooting

- **An instance crashes**: its log on the Instances page shows llama-server's
  own error. `journalctl -u llama-manager` has the manager's side.
- **"permission denied" on /dev/kfd or /dev/dri**: the service user isn't in the
  `render` and `video` groups (§3); restart the service after adding them.
- **Out of memory on load**: too much on one card. Move a model to another GPU,
  lower `context_length`, or for the prompt enhancer add `--no-mmproj` or use
  a smaller file.
- **The image reader or the prompt enhancer crashes on gfx906** (both are
  Qwen3.5 models, a new architecture): try a newer llama.cpp. One community
  gfx906 build had to fix a crash in Qwen3.5's attention on ROCm (SOLVE_TRI);
  a Vulkan build of llama-server is the fallback.
- **The image reader is slow**: it may have downloaded a BF16 projector, which
  gfx906 can't run fast; pin the F16 one (README, Image reader).
- **An image model doesn't finish downloading**: its log names the file. A
  download that fails its SHA-256 check is deleted; start the model on the
  Instances page to fetch it again.
- **Settings shows Funnel "Access denied"**: run `sudo tailscale set
  --operator=<service user>` (§8).
- **"config.yaml was changed on disk after llama-manager loaded it"** when
  saving in the UI: the file was edited by hand while the manager ran; restart
  the service to load it.

## 14. Updating later

```bash
cd ~/workspace/llama-manager && git pull && bash service_install.sh   # rebuilds and restarts
cd ~/workspace/llama.cpp && git pull && cmake --build build --config Release && sudo systemctl restart llama-manager
```

Rebuild sd-server and audio.cpp the same way when needed. Before updating
ROCm or llama.cpp on gfx906, note the working versions (§2) so you can go back.
