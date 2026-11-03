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
