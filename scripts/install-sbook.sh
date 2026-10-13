#!/usr/bin/env bash
# One-time install of cpuforge on a headless Linux box (Sbook): downloads the newest release,
# runs it as a systemd service reachable over Tailscale, and keeps it updated from GitHub.
#   bash install-sbook.sh            (uses sudo for the service file)
# Optional: CPUFORGE_REPO=owner/repo  CPUFORGE_DIR=/data/cpuforge  CPUFORGE_BIND=<ip>
set -euo pipefail

REPO="${CPUFORGE_REPO:-Sm0r3s-source/cpuforge}"
WORKDIR="${CPUFORGE_DIR:-/data/cpuforge}"
BIND="${CPUFORGE_BIND:-$(tailscale ip -4 2>/dev/null | head -1 || true)}"
BIND="${BIND:-127.0.0.1}"
BIN="$HOME/.local/bin/cpuforge"
ASSET="cpuforge-linux-x64"

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
echo "looking up the latest release of $REPO ..."
curl -fsSL -H "User-Agent: cpuforge-install" "https://api.github.com/repos/$REPO/releases/latest" -o "$tmp/rel.json"
read -r url sha_url < <(python3 - "$tmp/rel.json" "$ASSET" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); n = sys.argv[2]
a = {x["name"]: x["browser_download_url"] for x in r["assets"]}
print(a[n], a[n + ".sha256"])
PY
)
curl -fsSL "$url" -o "$tmp/bin"
curl -fsSL "$sha_url" -o "$tmp/sha"
want="$(awk '{print $1}' "$tmp/sha")"; got="$(sha256sum "$tmp/bin" | awk '{print $1}')"
[ "$want" = "$got" ] || { echo "checksum mismatch, aborting"; exit 1; }
mkdir -p "$(dirname "$BIN")"
install -m 755 "$tmp/bin" "$BIN"
"$BIN" version

sudo mkdir -p "$WORKDIR/models" "$WORKDIR/data"
sudo chown -R "$USER" "$WORKDIR"

sudo tee /etc/systemd/system/cpuforge.service >/dev/null <<UNIT
[Unit]
Description=cpuforge web app
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$WORKDIR
ExecStart=$BIN ui --no-open --bind $BIND --auto-update
Restart=always
RestartSec=5
MemoryHigh=5G
MemoryMax=6G

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now cpuforge
sleep 1
systemctl --no-pager --lines=3 status cpuforge || true
echo
echo "open  http://$BIND:7860   (models and data live in $WORKDIR)"
echo "no login: it is only reachable where $BIND is reachable. With 127.0.0.1 use: ssh -L 7860:127.0.0.1:7860 $USER@sbook"
