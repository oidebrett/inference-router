#!/usr/bin/env bash
# Install / upgrade the Inference Router on a Linux host with systemd (idempotent).
#
#   sudo ./install.sh
#   sudo IR_SEED_CONFIG=./my-router.yaml IR_KEY_NVIDIA=nvapi-... ./install.sh
#
# Layout
#   /opt/inference-router/{venv,lib}     code (lib/pii_router, lib/inference_router)
#   /etc/inference-router/router.yaml    upstreams, routes, privacy (no secrets)
#   /etc/inference-router/secrets.env    LITELLM_MASTER_KEY (admin key), IR_MASK_SECRET,
#                                        LAYA_API_KEY, IR_EVENTS_TOKEN, IR_KEY_<UPSTREAM>=...
#   /etc/inference-router/router.env     IR_HOST, IR_PORT, PII_ROUTER_LAYA_URL
#   /var/lib/inference-router            state (rendered LiteLLM config, Laya checkpoint)
# Services: inference-router, inference-router-laya (unless IR_WITH_LAYA=no),
#           inference-router-reload.path (restarts the router when router.yaml or secrets.env change).
#
# Options (environment)
#   IR_HOST (127.0.0.1)  IR_PORT (4100)        where the OpenAI-compatible endpoint listens
#   IR_WITH_LAYA (yes)                          install the Laya classifier (mask_check, keep_private)
#   IR_SEED_CONFIG                              router.yaml to install if none exists yet
#   IR_KEY_<UPSTREAM ID>=...                    upstream keys to store (e.g. IR_KEY_NVIDIA)
#   IR_ADMIN_HOST (127.0.0.1) IR_ADMIN_PORT (4110)  admin UI listener (behind Pangolin SSO in production)
#   IR_ADMIN_KEY                                use this admin key instead of generating one
#   IR_PUBLIC_URL                               the public endpoint shown in the admin UI
#   IR_REF                                      (informational) the ref being installed
# Secrets are generated once and kept on re-runs. No message content is ever logged.
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
PREFIX=/opt/inference-router
ETC=/etc/inference-router
STATE=/var/lib/inference-router
SECRETS="$ETC/secrets.env"
USER_NAME=inference-router
WITH_LAYA="$(printf '%s' "${IR_WITH_LAYA:-yes}" | tr '[:upper:]' '[:lower:]')"

log() { printf '[inference-router] %s\n' "$*"; }
die() { printf '[inference-router] ERROR: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run as root"
command -v systemctl >/dev/null || die "systemd is required"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python 3.10+ required"
if ! "$PYTHON" -c 'import ensurepip' >/dev/null 2>&1; then
  command -v apt-get >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-venv >/dev/null
  "$PYTHON" -c 'import ensurepip' >/dev/null 2>&1 || die "python3-venv (ensurepip) is not installed"
fi

id "$USER_NAME" >/dev/null 2>&1 || useradd --system --home-dir "$STATE" --shell /usr/sbin/nologin "$USER_NAME"
install -d -m 0755 "$PREFIX" "$PREFIX/lib"
install -d -m 0750 -o "$USER_NAME" -g "$USER_NAME" "$ETC" "$STATE" "$STATE/hf"

if [ -x "$PREFIX/venv/bin/python" ] && ! "$PREFIX/venv/bin/python" -m pip --version >/dev/null 2>&1; then
  rm -rf "$PREFIX/venv"
fi
[ -x "$PREFIX/venv/bin/python" ] || { log "creating virtualenv"; "$PYTHON" -m venv "$PREFIX/venv"; }
"$PREFIX/venv/bin/python" -m pip install --quiet --upgrade pip
if [ "$WITH_LAYA" = yes ]; then
  if [ "$(uname -m)" = x86_64 ]; then
    log "installing CPU-only torch"
    "$PREFIX/venv/bin/python" -m pip install --quiet torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
  fi
  log "installing litellm + laya"
  "$PREFIX/venv/bin/python" -m pip install --quiet -r "$SRC/router/requirements.txt"
else
  log "installing litellm (no Laya)"
  grep -v '^laya' "$SRC/router/requirements.txt" > "$STATE/requirements-nolaya.txt"
  "$PREFIX/venv/bin/python" -m pip install --quiet -r "$STATE/requirements-nolaya.txt"
fi

log "installing code"
for pkg in pii_router inference_router ir_admin; do
  from="$SRC/router/$pkg"; [ "$pkg" = ir_admin ] && from="$SRC/admin/ir_admin"
  rm -rf "$PREFIX/lib/$pkg.new"
  cp -R "$from" "$PREFIX/lib/$pkg.new"
  rm -rf "$PREFIX/lib/$pkg.new/__pycache__"
  rm -rf "$PREFIX/lib/$pkg" && mv "$PREFIX/lib/$pkg.new" "$PREFIX/lib/$pkg"
done

# --- secrets: generate once, keep on re-runs; add upstream keys passed in --------------
umask 077
touch "$SECRETS"
get() { sed -n "s/^$1=//p" "$SECRETS" | tail -1; }
put() { # name value: replace or append
  grep -v "^$1=" "$SECRETS" > "$SECRETS.new" || true
  printf '%s=%s\n' "$1" "$2" >> "$SECRETS.new"
  mv "$SECRETS.new" "$SECRETS"
}
# The admin key: one passed in (e.g. by a platform that shows it to the customer) wins,
# else keep the existing one, else generate.
if [ -n "${IR_ADMIN_KEY:-}" ]; then put LITELLM_MASTER_KEY "$IR_ADMIN_KEY"
elif [ -z "$(get LITELLM_MASTER_KEY)" ]; then put LITELLM_MASTER_KEY "sk-ira-$(openssl rand -hex 24)"; fi
[ -n "$(get IR_MASK_SECRET)" ] || put IR_MASK_SECRET "$(openssl rand -hex 32)"
[ -n "$(get LAYA_API_KEY)" ] || put LAYA_API_KEY "$(openssl rand -hex 24)"
[ -n "$(get IR_EVENTS_TOKEN)" ] || put IR_EVENTS_TOKEN "$(openssl rand -hex 24)"
while IFS='=' read -r name value; do
  case "$name" in IR_KEY_[A-Z0-9_]*) [ -n "$value" ] && put "$name" "$value" ;; esac
done < <(env)
chown "$USER_NAME:$USER_NAME" "$SECRETS" && chmod 0600 "$SECRETS"

cat > "$ETC/router.env" <<ENVEOF
# Managed by install.sh.
IR_HOST=${IR_HOST:-127.0.0.1}
IR_PORT=${IR_PORT:-4100}
IR_ADMIN_HOST=${IR_ADMIN_HOST:-127.0.0.1}
IR_ADMIN_PORT=${IR_ADMIN_PORT:-4110}
IR_EVENTS_URL=http://$( [ "${IR_ADMIN_HOST:-127.0.0.1}" = 0.0.0.0 ] && echo 127.0.0.1 || echo "${IR_ADMIN_HOST:-127.0.0.1}"):${IR_ADMIN_PORT:-4110}/internal/events
IR_PUBLIC_URL=${IR_PUBLIC_URL:-}
IR_LAYA_AVAILABLE=$([ "$WITH_LAYA" = yes ] && echo 1 || echo 0)
IR_KEYS_DB=$STATE/keys.db
PII_ROUTER_LAYA_URL=http://127.0.0.1:4101
ENVEOF
chmod 0644 "$ETC/router.env"

if [ ! -f "$ETC/router.yaml" ]; then
  if [ -n "${IR_SEED_CONFIG:-}" ]; then
    install -m 0640 -o "$USER_NAME" -g "$USER_NAME" "$IR_SEED_CONFIG" "$ETC/router.yaml"
  else
    printf 'version: 1\nprivacy: {mode: mask}\nupstreams: []\nroutes: []\n' > "$ETC/router.yaml"
    chown "$USER_NAME:$USER_NAME" "$ETC/router.yaml" && chmod 0640 "$ETC/router.yaml"
  fi
fi
PYTHONPATH="$PREFIX/lib" "$PREFIX/venv/bin/python" -c \
  "from inference_router.config import load; c = load('$ETC/router.yaml'); print('[inference-router] config ok:', c.privacy.mode, 'routes', c.route_names())" \
  || die "router.yaml is invalid (see above)"
# Every upstream needs an entry in secrets.env, even if empty, so LiteLLM can resolve it.
for env_name in $(PYTHONPATH="$PREFIX/lib" "$PREFIX/venv/bin/python" -c \
    "from inference_router.config import load; print(' '.join(u.key_env for u in load('$ETC/router.yaml').upstreams))"); do
  grep -q "^$env_name=" "$SECRETS" || put "$env_name" "none"
done
chown "$USER_NAME:$USER_NAME" "$SECRETS" && chmod 0600 "$SECRETS"

log "installing systemd units"
install -m 0644 "$SRC/systemd/inference-router.service" "$SRC/systemd/inference-router-reload.path" \
  "$SRC/systemd/inference-router-reload.service" "$SRC/systemd/inference-router-admin.service" /etc/systemd/system/
if [ "$WITH_LAYA" = yes ]; then
  install -m 0644 "$SRC/systemd/inference-router-laya.service" /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable --now inference-router-laya.service >/dev/null
  log "waiting for Laya (first start downloads ~1.7 GB)"
  for _ in $(seq 1 900); do curl -sf -o /dev/null http://127.0.0.1:4101/health && break; sleep 1; done
  curl -sf -o /dev/null http://127.0.0.1:4101/health || die "Laya did not become healthy (journalctl -u inference-router-laya)"
else
  systemctl disable --now inference-router-laya.service >/dev/null 2>&1 || true
  rm -f /etc/systemd/system/inference-router-laya.service
  systemctl daemon-reload
fi
systemctl enable inference-router.service inference-router-reload.path inference-router-admin.service >/dev/null
systemctl restart inference-router.service inference-router-admin.service
systemctl restart inference-router-reload.path
HOST="${IR_HOST:-127.0.0.1}"; [ "$HOST" = 0.0.0.0 ] && HOST=127.0.0.1
for _ in $(seq 1 120); do curl -sf -o /dev/null "http://$HOST:${IR_PORT:-4100}/health/liveliness" && break; sleep 1; done
curl -sf -o /dev/null "http://$HOST:${IR_PORT:-4100}/health/liveliness" || die "router did not become healthy (journalctl -u inference-router)"
log "ready: http://${IR_HOST:-127.0.0.1}:${IR_PORT:-4100}/v1  (admin key: LITELLM_MASTER_KEY in $SECRETS)"
log "admin UI: http://${IR_ADMIN_HOST:-127.0.0.1}:${IR_ADMIN_PORT:-4110}/  (publish it only behind Pangolin SSO)"
