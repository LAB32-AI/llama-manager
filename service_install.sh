#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_USER="${SUDO_USER:-$(whoami)}"
CURRENT_GROUP="$(id -gn "$CURRENT_USER")"

if [ ! -f "$SCRIPT_DIR/llama-manager" ]; then
    echo "==> Building llama-manager binary..."
    cd "$SCRIPT_DIR"
    go build -o llama-manager .
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "==> Re-running as root..."
    exec sudo "$0" "$@"
fi

echo "==> Installing as user: $CURRENT_USER (group: $CURRENT_GROUP)"

echo "==> Installing binary to /usr/local/bin/llama-manager..."
cp "$SCRIPT_DIR/llama-manager" /usr/local/bin/llama-manager
chmod 755 /usr/local/bin/llama-manager

echo "==> Creating config directory /etc/llama-manager/..."
mkdir -p /etc/llama-manager
chown "$CURRENT_USER:$CURRENT_GROUP" /etc/llama-manager

echo "==> Installing config file..."
if [ ! -f /etc/llama-manager/config.yaml ]; then
    cp "$SCRIPT_DIR/config.yaml" /etc/llama-manager/config.yaml
    chown "$CURRENT_USER:$CURRENT_GROUP" /etc/llama-manager/config.yaml
    chmod 644 /etc/llama-manager/config.yaml
    echo "    config.yaml installed."
else
    echo "    config.yaml already exists, skipping to avoid overwriting."
fi

echo "==> Installing systemd service..."
sed "s/^User=.*/User=$CURRENT_USER/" "$SCRIPT_DIR/llama-manager.service" \
    | sed "s/^Group=.*/Group=$CURRENT_GROUP/" \
    > /etc/systemd/system/llama-manager.service
chmod 644 /etc/systemd/system/llama-manager.service

echo "==> Reloading systemd daemon..."
systemctl daemon-reload

echo "==> Enabling and starting llama-manager service..."
systemctl enable llama-manager
systemctl restart llama-manager

echo ""
echo "==> Service status:"
systemctl status llama-manager --no-pager

echo ""
echo "Done. View logs with: journalctl -u llama-manager -f"
