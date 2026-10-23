#!/usr/bin/env bash
# Install / upgrade the PII router on an agent-gateway host (idempotent).
#
#   sudo OLLAMA_MODEL=qwen3.5:4b NVIDIA_API_KEY=nvapi-... ./install.sh
#
# Required: OLLAMA_MODEL (local model for PII requests, e.g. qwen3.5:0.8b on 8 GB, qwen3.5:4b on 16 GB; must support tool calling),
#           and a cloud backend for everything else: NVIDIA_API_KEY (AgentGateway), OR
#           CLOUD_BASE_URL + CLOUD_MODEL [+ CLOUD_API_KEY] (PrivateInference: the customer's Vast GPU).
#           With CLOUD_BASE_URL, an NVIDIA_API_KEY becomes the FALLBACK used while the GPU is down
#           (health-checked at CLOUD_BASE_URL/models); without one, non-PII goes local while it is down.
# Optional: NVIDIA_MODEL (default nvidia/nemotron-3.5-super-vl-120b-a12b; manidae-cloud
# passes its configured default explicitly), OLLAMA_BASE_URL
#           (default http://127.0.0.1:11434), PII_ROUTER_HOST (127.0.0.1), PII_ROUTER_PORT (4100),
#           LITELLM_MASTER_KEY (the router's client key; generated once if not given),
#           LAYA_PORT (4101), PII_ROUTER_LAYA_THRESHOLD (0.4),
#           PII_ROUTER_PUBLIC_MODELS (pii-router; the names /v1/models lists),
#           PII_ROUTER_CLOUD_REASONING_EFFORT (none; GPU thinking, only with CLOUD_BASE_URL).
# PII_ROUTER_MODE=off (PrivateInference without PII routing): no PII check, no local model.
#           OLLAMA_MODEL is not needed, Laya/torch are not installed and its unit is removed;
#           everything goes to the cloud backend (or the NVIDIA fallback while it is down).
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
PII_MODE="$(printf '%s' "${PII_ROUTER_MODE:-on}" | tr '[:upper:]' '[:lower:]')"
[ "$PII_MODE" = "on" ] || [ "$PII_MODE" = "off" ] || die "PII_ROUTER_MODE must be on or off"
[ "$PII_MODE" = "off" ] || [ -n "${OLLAMA_MODEL:-}" ] || die "OLLAMA_MODEL is required (or PII_ROUTER_MODE=off)"
if [ -n "${CLOUD_BASE_URL:-}" ]; then
  [ -n "${CLOUD_MODEL:-}" ] || die "CLOUD_MODEL is required with CLOUD_BASE_URL"
else
  [ -n "${NVIDIA_API_KEY:-}" ] || die "NVIDIA_API_KEY (or CLOUD_BASE_URL) is required"
fi
NVIDIA_MODEL="${NVIDIA_MODEL:-nvidia/nemotron-3.5-super-vl-120b-a12b}"
FALLBACK_BASE_URL=""; FALLBACK_MODEL=""; FALLBACK2_MODEL=""
if [ -n "${CLOUD_BASE_URL:-}" ] && [ -n "${NVIDIA_API_KEY:-}" ]; then
  FALLBACK_BASE_URL="${NVIDIA_BASE_URL:-https://integrate.api.nvidia.com/v1}"; FALLBACK_MODEL="$NVIDIA_MODEL"
  FALLBACK2_MODEL="${NVIDIA_FALLBACK_MODEL:-}"   # second NVIDIA model, tried when the first fails
fi
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python 3.10+ required"
# `python3 -m venv --help` succeeds even without ensurepip, but creating a venv
# then fails ("ensurepip is not available") — Ubuntu 24.04's cloud image ships
# python3 without python3-venv (seen on the first live install, 2026-10-04).
if ! "$PYTHON" -c 'import ensurepip' >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    log "installing python3-venv"
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-venv >/dev/null
  fi
  "$PYTHON" -c 'import ensurepip' >/dev/null 2>&1 || die "python3-venv (ensurepip) is not installed"
fi

id pii-router >/dev/null 2>&1 || useradd --system --home-dir "$STATE" --shell /usr/sbin/nologin pii-router
install -d -m 0755 "$PREFIX"
install -d -m 0750 -o pii-router -g pii-router "$STATE" "$STATE/hf"
install -d -m 0750 -o root -g pii-router "$ETC"

# A failed earlier run can leave a venv without pip; rebuild it rather than reuse it.
if [ -x "$PREFIX/venv/bin/python" ] && ! "$PREFIX/venv/bin/python" -m pip --version >/dev/null 2>&1; then
  log "removing incomplete virtualenv"
  rm -rf "$PREFIX/venv"
fi
if [ ! -x "$PREFIX/venv/bin/python" ]; then
  log "creating virtualenv"
  "$PYTHON" -m venv "$PREFIX/venv"
fi
"$PREFIX/venv/bin/python" -m pip install --quiet --upgrade pip
if [ "$PII_MODE" = "on" ] && [ "$(uname -m)" = "x86_64" ]; then
  log "installing CPU-only torch (avoids multi-GB CUDA wheels)"
  "$PREFIX/venv/bin/python" -m pip install --quiet torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
fi
if [ "$PII_MODE" = "on" ]; then
  log "installing laya + litellm"
  "$PREFIX/venv/bin/python" -m pip install --quiet -r "$SRC_DIR/requirements.txt"
else
  log "installing litellm (PII routing off: no Laya)"
  grep -v '^laya' "$SRC_DIR/requirements.txt" > "$STATE/requirements-off.txt"
  "$PREFIX/venv/bin/python" -m pip install --quiet -r "$STATE/requirements-off.txt"
fi

log "installing router package"
rm -rf "$PREFIX/pii_router.new"
cp -R "$SRC_DIR/pii_router" "$PREFIX/pii_router.new"
rm -rf "$PREFIX/pii_router.new/__pycache__"
rm -rf "$PREFIX/pii_router" && mv "$PREFIX/pii_router.new" "$PREFIX/pii_router"

# Secrets: generate once, keep on re-run so the configured provider key stays valid.
existing() { [ -f "$ENV_FILE" ] && sed -n "s/^$1=//p" "$ENV_FILE" | tail -1 || true; }
# LITELLM_MASTER_KEY from the caller wins (PrivateInference: manidae-cloud generates it
# so it can show it to the customer); otherwise keep the existing one, else generate.
MASTER_KEY="${LITELLM_MASTER_KEY:-$(existing LITELLM_MASTER_KEY)}"; [ -n "$MASTER_KEY" ] || MASTER_KEY="sk-pii-$(openssl rand -hex 24)"
LAYA_KEY="$(existing LAYA_API_KEY)"; [ -n "$LAYA_KEY" ] || LAYA_KEY="$(openssl rand -hex 24)"
LAYA_PORT="${LAYA_PORT:-4101}"
umask 027
cat > "$ENV_FILE.new" <<ENVEOF
# Managed by scripts/pii-router/install.sh. Re-run the installer to change values.
LITELLM_MASTER_KEY=$MASTER_KEY
NVIDIA_API_KEY=${NVIDIA_API_KEY:-}
CLOUD_BASE_URL=${CLOUD_BASE_URL:-}
CLOUD_API_KEY=${CLOUD_API_KEY:-}
FALLBACK_BASE_URL=$FALLBACK_BASE_URL
FALLBACK_API_KEY=${NVIDIA_API_KEY:-}
LAYA_API_KEY=$LAYA_KEY
LAYA_PORT=$LAYA_PORT
LAYA_MODELS=english
PII_ROUTER_LAYA_URL=http://127.0.0.1:$LAYA_PORT
PII_ROUTER_LAYA_THRESHOLD=${PII_ROUTER_LAYA_THRESHOLD:-0.4}
PII_ROUTER_HOST=${PII_ROUTER_HOST:-127.0.0.1}
PII_ROUTER_PORT=${PII_ROUTER_PORT:-4100}
OLLAMA_MODEL=${OLLAMA_MODEL:-}
PII_ROUTER_MODE=$PII_MODE
PII_ROUTER_PUBLIC_MODELS=${PII_ROUTER_PUBLIC_MODELS:-pii-router}
ENVEOF
chown root:pii-router "$ENV_FILE.new" && chmod 0640 "$ENV_FILE.new" && mv "$ENV_FILE.new" "$ENV_FILE"

log "rendering LiteLLM config"
OLLAMA_MODEL="${OLLAMA_MODEL:-}" NVIDIA_MODEL="$NVIDIA_MODEL" PII_ROUTER_MODE="$PII_MODE" \
  OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}" \
  CLOUD_BASE_URL="${CLOUD_BASE_URL:-}" CLOUD_MODEL="${CLOUD_MODEL:-}" \
  FALLBACK_BASE_URL="$FALLBACK_BASE_URL" FALLBACK_MODEL="$FALLBACK_MODEL" FALLBACK2_MODEL="$FALLBACK2_MODEL" \
  PII_ROUTER_LOCAL_REASONING_EFFORT="${PII_ROUTER_LOCAL_REASONING_EFFORT-none}" \
  PII_ROUTER_CLOUD_REASONING_EFFORT="${PII_ROUTER_CLOUD_REASONING_EFFORT-none}" \
  PII_ROUTER_PUBLIC_MODELS="${PII_ROUTER_PUBLIC_MODELS:-pii-router}" \
  PYTHONPATH="$PREFIX" "$PREFIX/venv/bin/python" -m pii_router.render_config "$ETC/config.yaml"
chown root:pii-router "$ETC/config.yaml" && chmod 0640 "$ETC/config.yaml"

log "installing systemd units"
install -m 0644 "$SRC_DIR/systemd/pii-router.service" /etc/systemd/system/pii-router.service
if [ "$PII_MODE" = "on" ]; then
  install -m 0644 "$SRC_DIR/systemd/pii-router-laya.service" /etc/systemd/system/pii-router-laya.service
  systemctl daemon-reload
  systemctl enable pii-router-laya.service pii-router.service >/dev/null
  systemctl restart pii-router-laya.service
  log "waiting for Laya (first start downloads the checkpoint, ~1.7 GB)"
  for _ in $(seq 1 600); do curl -sf -o /dev/null "http://127.0.0.1:$LAYA_PORT/health" && break; sleep 1; done
  curl -sf -o /dev/null "http://127.0.0.1:$LAYA_PORT/health" || die "laya-serve did not become healthy (journalctl -u pii-router-laya)"
else
  # No unit file at all: pii-router.service Wants= Laya, which would otherwise start it.
  systemctl disable --now pii-router-laya.service >/dev/null 2>&1 || true
  rm -f /etc/systemd/system/pii-router-laya.service
  systemctl daemon-reload
  systemctl enable pii-router.service >/dev/null
fi
systemctl restart pii-router.service
PORT="${PII_ROUTER_PORT:-4100}"; HOST="${PII_ROUTER_HOST:-127.0.0.1}"
# Probe the address the router is bound to: agentgateway-setup.sh binds the Docker
# bridge IP, where 127.0.0.1 is never answered (2026-10-04 first live install).
PROBE_HOST="$HOST"; [ "$PROBE_HOST" = "0.0.0.0" ] && PROBE_HOST=127.0.0.1
for _ in $(seq 1 120); do curl -sf -o /dev/null "http://$PROBE_HOST:$PORT/health/liveliness" && break; sleep 1; done
curl -sf -o /dev/null "http://$PROBE_HOST:$PORT/health/liveliness" || die "router did not become healthy (journalctl -u pii-router)"
log "ready: http://$HOST:$PORT/v1  model=pii-router  key=LITELLM_MASTER_KEY in $ENV_FILE"
