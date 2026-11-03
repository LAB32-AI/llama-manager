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
