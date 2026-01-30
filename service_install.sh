#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Error: this script must be run as root"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Building llama-manager binary..."
cd "$SCRIPT_DIR"
go build -o llama-manager .

echo "==> Installing binary to /usr/local/bin/llama-manager..."
cp llama-manager /usr/local/bin/llama-manager
chmod 755 /usr/local/bin/llama-manager

echo "==> Creating llama-manager system user..."
if ! id -u llama-manager &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin llama-manager
    echo "    User created."
else
    echo "    User already exists, skipping."
fi

echo "==> Creating config directory /etc/llama-manager/..."
mkdir -p /etc/llama-manager
chown llama-manager:llama-manager /etc/llama-manager

echo "==> Installing config file..."
if [ ! -f /etc/llama-manager/config.yaml ]; then
    cp "$SCRIPT_DIR/config.yaml" /etc/llama-manager/config.yaml
    chown llama-manager:llama-manager /etc/llama-manager/config.yaml
    chmod 644 /etc/llama-manager/config.yaml
    echo "    config.yaml installed."
else
    echo "    config.yaml already exists, skipping to avoid overwriting."
fi

echo "==> Installing systemd service..."
cp "$SCRIPT_DIR/llama-manager.service" /etc/systemd/system/llama-manager.service
chmod 644 /etc/systemd/system/llama-manager.service

echo "==> Reloading systemd daemon..."
systemctl daemon-reload

echo "==> Enabling and starting llama-manager service..."
systemctl enable llama-manager
systemctl start llama-manager

echo "==> Service status:"
systemctl status llama-manager --no-pager

echo ""
echo "Done. View logs with: journalctl -u llama-manager -f"
