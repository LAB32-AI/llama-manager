#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

# Set when re-run under sudo after building as the invoking user.
BUILT=0
if [ "${1:-}" = "--built" ]; then
    BUILT=1
    shift
fi

# Install locations; overridable for testing.
BIN_DIR="${BIN_DIR:-/usr/local/bin}"
CONF_DIR="${CONF_DIR:-/etc/llama-manager}"
UNIT_DIR="${UNIT_DIR:-/etc/systemd/system}"

# The service runs as SERVICE_USER, else the user who invoked sudo.
CURRENT_USER="${SERVICE_USER:-${SUDO_USER:-$(whoami)}}"

if [ "$CURRENT_USER" = "root" ]; then
    echo "ERROR: refusing to install the service as root: llama-server parses" >&2
    echo "downloaded model files and should not run with full privileges." >&2
    echo "Run this via sudo from your own account, or set SERVICE_USER=<user>." >&2
    exit 1
fi
if ! id "$CURRENT_USER" >/dev/null 2>&1; then
    echo "ERROR: user '$CURRENT_USER' does not exist on this system." >&2
    echo "Set SERVICE_USER to a valid account or run as that user." >&2
    exit 1
fi
CURRENT_GROUP="$(id -gn "$CURRENT_USER")"

if [ ! -f "$SCRIPT_DIR/example.config.yaml" ]; then
    echo "ERROR: $SCRIPT_DIR/example.config.yaml not found." >&2
    echo "Run this installer from the repository root." >&2
    exit 1
fi

# Always rebuild when Go is available (incremental, so cheap): re-running the
# installer after `git pull` must not reinstall a stale binary. Build as the
# invoking user so the binary in the repo isn't left owned by root.
if [ "$BUILT" = 1 ]; then
    :
elif command -v go >/dev/null 2>&1; then
    echo "==> Building llama-manager binary..."
    if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
        sudo -u "$SUDO_USER" env "PATH=$PATH" bash -c 'cd "$1" && go build -o llama-manager .' _ "$SCRIPT_DIR"
    else
        (cd "$SCRIPT_DIR" && go build -o llama-manager .)
    fi
elif [ -f "$SCRIPT_DIR/llama-manager" ]; then
    echo "WARNING: go not found; installing the existing $SCRIPT_DIR/llama-manager, which may be out of date." >&2
else
    echo "ERROR: go not found and no prebuilt $SCRIPT_DIR/llama-manager to install." >&2
    exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "==> Re-running as root..."
    exec sudo bash "$SELF" --built "$@"
fi

echo "==> Installing as user: $CURRENT_USER (group: $CURRENT_GROUP)"

echo "==> Installing binary to $BIN_DIR/llama-manager..."
# Write a new file and rename it into place: the running service keeps its old
# inode, whereas cp over a running executable fails ("Text file busy") or
# corrupts the live process.
mkdir -p "$BIN_DIR"
install -m 755 "$SCRIPT_DIR/llama-manager" "$BIN_DIR/llama-manager.new"
mv -f "$BIN_DIR/llama-manager.new" "$BIN_DIR/llama-manager"

echo "==> Creating config directory $CONF_DIR/..."
mkdir -p "$CONF_DIR"
chown "$CURRENT_USER:$CURRENT_GROUP" "$CONF_DIR"

NEW_CONFIG=0
echo "==> Installing config file..."
if [ ! -f "$CONF_DIR/config.yaml" ]; then
    cp "$SCRIPT_DIR/example.config.yaml" "$CONF_DIR/config.yaml"
    chown "$CURRENT_USER:$CURRENT_GROUP" "$CONF_DIR/config.yaml"
    chmod 640 "$CONF_DIR/config.yaml"
    NEW_CONFIG=1
    echo "    config.yaml installed from example.config.yaml."
else
    echo "    config.yaml already exists, skipping to avoid overwriting."
fi

echo "==> Installing systemd service..."
mkdir -p "$UNIT_DIR"
sed "s/^User=.*/User=$CURRENT_USER/" "$SCRIPT_DIR/llama-manager.service" \
    | sed "s/^Group=.*/Group=$CURRENT_GROUP/" \
    > "$UNIT_DIR/llama-manager.service"
chmod 644 "$UNIT_DIR/llama-manager.service"

echo "==> Reloading systemd daemon..."
systemctl daemon-reload

systemctl enable llama-manager
if [ "$NEW_CONFIG" = 1 ]; then
    # Don't start on the unedited example: it points at example models and
    # paths, and would begin downloading/loading on this machine right away.
    echo ""
    echo "==> Service enabled but not started."
    echo "    Edit $CONF_DIR/config.yaml (server_bin, instances), then run:"
    echo "    sudo systemctl start llama-manager"
else
    echo "==> Restarting llama-manager service..."
    systemctl restart llama-manager
    echo ""
    echo "==> Service status:"
    systemctl status llama-manager --no-pager || true
fi

echo ""
echo "Done. View logs with: journalctl -u llama-manager -f"
