#!/usr/bin/env bash
# End-to-end check: real laya-serve (CPU) + real LiteLLM proxy with the rendered
# config + the hook, against two FAKE backends (no real model calls, no key).
#   PY=/path/to/venv/bin/python tests/e2e.sh
# The venv needs: laya[serve] litellm[proxy] fastapi uvicorn.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"; BIN="$(dirname "$PY")"
WORK="$(mktemp -d)"; KEY="sk-e2e-$RANDOM$RANDOM"; pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; rm -rf "$WORK"; }
trap cleanup EXIT
"$PY" tests/mock_upstream.py cloud 18101 >"$WORK/cloud.log" 2>&1 & pids+=($!)
"$PY" tests/mock_upstream.py local 18102 >"$WORK/local.log" 2>&1 & pids+=($!)
LAYA_HOST=127.0.0.1 LAYA_PORT=18103 LAYA_DEVICE=cpu LAYA_MODELS=english "$BIN/laya-serve" >"$WORK/laya.log" 2>&1 & pids+=($!)
OLLAMA_MODEL=qwen3:4b OLLAMA_BASE_URL=http://127.0.0.1:18102 NVIDIA_BASE_URL=http://127.0.0.1:18101/v1 \
  "$PY" -m pii_router.render_config "$WORK/config.yaml"
for i in $(seq 1 180); do curl -sf -o /dev/null http://127.0.0.1:18103/health && break; sleep 1; done
PYTHONPATH="$PWD" PII_ROUTER_LAYA_URL=http://127.0.0.1:18103 LITELLM_MASTER_KEY="$KEY" NVIDIA_API_KEY=fake \
  LITELLM_LOCAL_MODEL_COST_MAP=True "$BIN/litellm" --config "$WORK/config.yaml" --host 127.0.0.1 --port 18100 \
  >"$WORK/litellm.log" 2>&1 & pids+=($!)
for i in $(seq 1 90); do curl -sf -o /dev/null http://127.0.0.1:18100/health/liveliness && break; sleep 1; done
fail=0
check() { # message stream expected
  local out; out=$(curl -sS http://127.0.0.1:18100/v1/chat/completions -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" -d "{\"model\":\"pii-router\",\"stream\":$2,\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}]}")
  local got; got=$(echo "$out" | grep -oE "answered-by-[a-z]+" | head -1 | sed 's/answered-by-//')
  if [ "$got" = "$3" ]; then echo "PASS [$3 stream=$2] $1"; else echo "FAIL [want $3 got '${got}' stream=$2] $1"; fail=1; fi
}
check "Explain the difference between TCP and UDP." false cloud
check "Explain the difference between TCP and UDP." true cloud
check "Email ann.kelly@acme.ie about her overdue invoice." false local
check "Our employee Priya Nair has been on sick leave for depression since June; plan her return-to-work." true local
check "Please check why John Murphy's salary of 84,000 was paid late this month." false local
echo "--- decision log:"; grep "pii-router-decision" "$WORK/litellm.log" | sed 's/.*pii-router-decision //' | cut -c1-200
if grep -qE "ann.kelly|Priya|Murphy" <(grep "pii-router-decision" "$WORK/litellm.log"); then echo "FAIL: message content leaked into the decision log"; fail=1; fi
[ $fail -eq 0 ] && echo "E2E PASS" || { echo "E2E FAIL"; tail -20 "$WORK/litellm.log"; exit 1; }
