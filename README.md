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
host: "0.0.0.0"                           # bind address (LAN-shared by default)
gpu_backend: vulkan                       # vulkan, cuda, rocm, rocm_rocr, metal
instances:
  - name: my-model
    model: /path/to/model.gguf            # local file or HF id (owner/repo)
    port: 9090
    gpu_ids: [0]
```

This service is designed to be shared on a trusted LAN — it binds `0.0.0.0`
by default. There is no authentication; only run it on networks you trust.
Cross-site requests from a browser are blocked by an `Origin` check.
`server_bin` cannot be changed through the API or config import — edit the
config file on disk and restart if you need to point at a different binary.

## Run

```bash
./llama-manager -config config.yaml
```

Then open `http://<host-ip>:8080/` (or `http://localhost:8080/` from the same machine).

## Install as systemd service

```bash
sudo bash service_install.sh
```

This builds the binary, installs it to `/usr/local/bin`, copies
`example.config.yaml` to `/etc/llama-manager/config.yaml` (only if no config
exists yet), and enables the service under the invoking user.

View logs:

```bash
journalctl -u llama-manager -f
```

## Development

```bash
go test -race ./...
go vet ./...
```
