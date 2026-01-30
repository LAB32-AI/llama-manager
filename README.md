# llama-manager

Manage multiple llama.cpp instances on your server.

## Configuration

Copy and edit the example config:

```bash
cp config.yaml.example config.yaml
```

Key settings in `config.yaml`:

```yaml
server_bin: /usr/local/bin/llama-server
gpu_backend: vulkan  # or metal
instances:
  - name: my-model
    model: /path/to/model.gguf
    port: 9090
    gpu_ids: [0]
```

## Install as systemd service

```bash
sudo bash service_install.sh
```

This builds the binary, installs it to `/usr/local/bin`, creates a `llama-manager` system user, copies config to `/etc/llama-manager/`, and enables the service.

View logs:

```bash
journalctl -u llama-manager -f
```
