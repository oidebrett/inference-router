#!/usr/bin/env bash
# Wire the PII router into an agent-gateway host (called by manidae-cloud when
# "Use Ollama only for PII requests" is ticked). Idempotent.
#
#   sudo OLLAMA_MODEL=qwen3:4b NVIDIA_API_KEY=nvapi-... scripts/pii-router/agentgateway-setup.sh
#
# 1. Installs the router (install.sh), bound to the Docker bridge IP: sandbox
#    containers and the host reach it, the internet does not — even on hosts
#    whose firewall is inactive.
# 2. Opens the bridge -> host port in UFW (UFW's INPUT policy drops it otherwise,
#    same as NemoClaw's Ollama proxy on 11435).
# 3. Points NemoClaw at it as the "custom" provider in the controller's .env.local.
#    NemoClaw's SSRF guard refuses private endpoints unless they are listed in
#    NEMOCLAW_TRUSTED_PRIVATE_INFERENCE_HOSTS (RFC1918 only, never loopback),
#    which is why the router cannot simply listen on 127.0.0.1.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
CONTROLLER_ENV="${CONTROLLER_ENV:-/opt/openshell-controller/.env.local}"
PORT="${PII_ROUTER_PORT:-4100}"

[ -n "${OLLAMA_MODEL:-}" ] && [ -n "${NVIDIA_API_KEY:-}" ] || {
  echo "[pii-router] OLLAMA_MODEL and NVIDIA_API_KEY are both required" >&2; exit 1; }

BRIDGE_IP="${PII_ROUTER_BRIDGE_IP:-$(ip -4 -o addr show docker0 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)}"
BRIDGE_IP="${BRIDGE_IP:-172.17.0.1}"
case "$BRIDGE_IP" in
  10.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*|192.168.*) ;;
  *) echo "[pii-router] bridge IP $BRIDGE_IP is not RFC1918; NemoClaw would refuse it" >&2; exit 1 ;;
esac

PII_ROUTER_HOST="$BRIDGE_IP" PII_ROUTER_PORT="$PORT" "$DIR/install.sh"

if command -v ufw >/dev/null 2>&1; then
  ufw allow from 172.0.0.0/8 to any port "$PORT" proto tcp >/dev/null
fi

MASTER_KEY="$(sed -n 's/^LITELLM_MASTER_KEY=//p' /etc/pii-router/pii-router.env | tail -1)"
[ -n "$MASTER_KEY" ] || { echo "[pii-router] no LITELLM_MASTER_KEY after install" >&2; exit 1; }

touch "$CONTROLLER_ENV"
keys='NEMOCLAW_PROVIDER|NEMOCLAW_ENDPOINT_URL|NEMOCLAW_MODEL|COMPATIBLE_API_KEY|NEMOCLAW_PROVIDER_KEY|NEMOCLAW_TRUSTED_PRIVATE_INFERENCE_HOSTS'
grep -vE "^($keys)=" "$CONTROLLER_ENV" > "$CONTROLLER_ENV.new" || true
cat >> "$CONTROLLER_ENV.new" <<ENVEOF
NEMOCLAW_PROVIDER=custom
NEMOCLAW_ENDPOINT_URL=http://$BRIDGE_IP:$PORT/v1
NEMOCLAW_MODEL=pii-router
COMPATIBLE_API_KEY=$MASTER_KEY
NEMOCLAW_PROVIDER_KEY=$MASTER_KEY
NEMOCLAW_TRUSTED_PRIVATE_INFERENCE_HOSTS=$BRIDGE_IP
ENVEOF
chmod --reference="$CONTROLLER_ENV" "$CONTROLLER_ENV.new" 2>/dev/null || chmod 600 "$CONTROLLER_ENV.new"
mv "$CONTROLLER_ENV.new" "$CONTROLLER_ENV"
echo "[pii-router] NemoClaw custom provider -> http://$BRIDGE_IP:$PORT/v1 (model pii-router)"
