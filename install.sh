#!/usr/bin/env bash
# Install / upgrade the PII router on an agent-gateway host (idempotent).
#
#   sudo OLLAMA_MODEL=qwen3:4b NVIDIA_API_KEY=nvapi-... ./install.sh
#
# Required: OLLAMA_MODEL (local model for PII requests; must support tool calling),
#           NVIDIA_API_KEY (cloud backend for everything else).
# Optional: NVIDIA_MODEL (default nvidia/nemotron-3-ultra-550b-a55b), OLLAMA_BASE_URL
#           (default http://127.0.0.1:11434), PII_ROUTER_HOST (127.0.0.1), PII_ROUTER_PORT (4100),
#           LAYA_PORT (4101), PII_ROUTER_LAYA_THRESHOLD (0.4).
#
# Result: OpenAI-compatible endpoint http://$PII_ROUTER_HOST:$PII_ROUTER_PORT/v1, model "pii-router",
# authenticated with LITELLM_MASTER_KEY from /etc/pii-router/pii-router.env (generated once,
# preserved on re-run). Point the OpenShell/NemoClaw "custom" provider at it.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX=/opt/pii-router
STATE=/var/lib/pii-router
ETC=/etc/pii-router
ENV_FILE="$ETC/pii-router.env"

log() { printf '[pii-router] %s\n' "$*"; }
die() { printf '[pii-router] ERROR: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run as root"
[ -n "${OLLAMA_MODEL:-}" ] || die "OLLAMA_MODEL is required"
[ -n "${NVIDIA_API_KEY:-}" ] || die "NVIDIA_API_KEY is required"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python 3.10+ required"
"$PYTHON" -m venv --help >/dev/null 2>&1 || die "python3-venv is not installed"

id pii-router >/dev/null 2>&1 || useradd --system --home-dir "$STATE" --shell /usr/sbin/nologin pii-router
install -d -m 0755 "$PREFIX"
install -d -m 0750 -o pii-router -g pii-router "$STATE" "$STATE/hf"
install -d -m 0750 -o root -g pii-router "$ETC"

if [ ! -x "$PREFIX/venv/bin/python" ]; then
  log "creating virtualenv"
  "$PYTHON" -m venv "$PREFIX/venv"
fi
"$PREFIX/venv/bin/python" -m pip install --quiet --upgrade pip
if [ "$(uname -m)" = "x86_64" ]; then
  log "installing CPU-only torch (avoids multi-GB CUDA wheels)"
  "$PREFIX/venv/bin/python" -m pip install --quiet torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
fi
log "installing laya + litellm"
"$PREFIX/venv/bin/python" -m pip install --quiet -r "$SRC_DIR/requirements.txt"

log "installing router package"
rm -rf "$PREFIX/pii_router.new"
cp -R "$SRC_DIR/pii_router" "$PREFIX/pii_router.new"
rm -rf "$PREFIX/pii_router.new/__pycache__"
rm -rf "$PREFIX/pii_router" && mv "$PREFIX/pii_router.new" "$PREFIX/pii_router"

# Secrets: generate once, keep on re-run so the configured provider key stays valid.
existing() { [ -f "$ENV_FILE" ] && sed -n "s/^$1=//p" "$ENV_FILE" | tail -1 || true; }
MASTER_KEY="$(existing LITELLM_MASTER_KEY)"; [ -n "$MASTER_KEY" ] || MASTER_KEY="sk-pii-$(openssl rand -hex 24)"
LAYA_KEY="$(existing LAYA_API_KEY)"; [ -n "$LAYA_KEY" ] || LAYA_KEY="$(openssl rand -hex 24)"
LAYA_PORT="${LAYA_PORT:-4101}"
umask 027
cat > "$ENV_FILE.new" <<ENVEOF
# Managed by scripts/pii-router/install.sh. Re-run the installer to change values.
LITELLM_MASTER_KEY=$MASTER_KEY
NVIDIA_API_KEY=$NVIDIA_API_KEY
LAYA_API_KEY=$LAYA_KEY
LAYA_PORT=$LAYA_PORT
LAYA_MODELS=english
PII_ROUTER_LAYA_URL=http://127.0.0.1:$LAYA_PORT
PII_ROUTER_LAYA_THRESHOLD=${PII_ROUTER_LAYA_THRESHOLD:-0.4}
PII_ROUTER_HOST=${PII_ROUTER_HOST:-127.0.0.1}
PII_ROUTER_PORT=${PII_ROUTER_PORT:-4100}
OLLAMA_MODEL=$OLLAMA_MODEL
ENVEOF
chown root:pii-router "$ENV_FILE.new" && chmod 0640 "$ENV_FILE.new" && mv "$ENV_FILE.new" "$ENV_FILE"

log "rendering LiteLLM config"
OLLAMA_MODEL="$OLLAMA_MODEL" NVIDIA_MODEL="${NVIDIA_MODEL:-nvidia/nemotron-3-ultra-550b-a55b}" \
  OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}" \
  PYTHONPATH="$PREFIX" "$PREFIX/venv/bin/python" -m pii_router.render_config "$ETC/config.yaml"
chown root:pii-router "$ETC/config.yaml" && chmod 0640 "$ETC/config.yaml"

log "installing systemd units"
install -m 0644 "$SRC_DIR/systemd/pii-router-laya.service" /etc/systemd/system/pii-router-laya.service
install -m 0644 "$SRC_DIR/systemd/pii-router.service" /etc/systemd/system/pii-router.service
systemctl daemon-reload
systemctl enable pii-router-laya.service pii-router.service >/dev/null
systemctl restart pii-router-laya.service
log "waiting for Laya (first start downloads the checkpoint, ~1.7 GB)"
for _ in $(seq 1 600); do curl -sf -o /dev/null "http://127.0.0.1:$LAYA_PORT/health" && break; sleep 1; done
curl -sf -o /dev/null "http://127.0.0.1:$LAYA_PORT/health" || die "laya-serve did not become healthy (journalctl -u pii-router-laya)"
systemctl restart pii-router.service
PORT="${PII_ROUTER_PORT:-4100}"; HOST="${PII_ROUTER_HOST:-127.0.0.1}"
for _ in $(seq 1 120); do curl -sf -o /dev/null "http://127.0.0.1:$PORT/health/liveliness" && break; sleep 1; done
curl -sf -o /dev/null "http://127.0.0.1:$PORT/health/liveliness" || die "router did not become healthy (journalctl -u pii-router)"
log "ready: http://$HOST:$PORT/v1  model=pii-router  key=LITELLM_MASTER_KEY in $ENV_FILE"
