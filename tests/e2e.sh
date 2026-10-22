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
  # The local-reply marker: present on local answers, absent on cloud answers.
  if [ "$3" = local ] && ! echo "$out" | grep -q '\\u2020\|†'; then echo "FAIL: local reply has no † marker"; fail=1; fi
  if [ "$3" != local ] && echo "$out" | grep -q '\\u2020\|†'; then echo "FAIL: $3 reply carries the † marker"; fail=1; fi
}
check "Explain the difference between TCP and UDP." false cloud
check "Explain the difference between TCP and UDP." true cloud
check "Email ann.kelly@acme.ie about her overdue invoice." false local
check "Our employee Priya Nair has been on sick leave for depression since June; plan her return-to-work." true local
check "Please check why John Murphy's salary of 84,000 was paid late this month." false local
# PII only in history: still local, but marked "††" and x-pii-reason: history.
hist=$(curl -sS -D "$WORK/h.txt" http://127.0.0.1:18100/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" -d '{"model":"pii-router","messages":[{"role":"user","content":"Email ann.kelly@acme.ie about her invoice."},{"role":"assistant","content":"Done. \u2020"},{"role":"user","content":"Explain the difference between TCP and UDP."}]}')
if echo "$hist" | grep -q 'answered-by-local' && echo "$hist" | grep -q '\\u2020\\u2020\|††' && grep -qi '^x-pii-reason: history' "$WORK/h.txt"; then
  echo "PASS [history] local, †† marker, x-pii-reason: history"; else echo "FAIL [history]"; cat "$WORK/h.txt"; echo "$hist"; fail=1; fi
grep -qi '^x-pii-route: local' "$WORK/h.txt" || { echo "FAIL: no x-pii-route header"; fail=1; }

echo "--- decision log:"; grep "pii-router-decision" "$WORK/litellm.log" | sed 's/.*pii-router-decision //' | cut -c1-200
if grep -qE "ann.kelly|Priya|Murphy" <(grep "pii-router-decision" "$WORK/litellm.log"); then echo "FAIL: message content leaked into the decision log"; fail=1; fi

# PrivateInference: cloud = a GPU (down at first), fallback = NVIDIA stand-in.
kill "${pids[-1]}" 2>/dev/null || true; sleep 1
"$PY" tests/mock_upstream.py fallback 18104 >"$WORK/fallback.log" 2>&1 & pids+=($!)
OLLAMA_MODEL=qwen3:4b OLLAMA_BASE_URL=http://127.0.0.1:18102 CLOUD_BASE_URL=http://127.0.0.1:18105/v1 CLOUD_MODEL=gpu \
  FALLBACK_BASE_URL=http://127.0.0.1:18104/v1 FALLBACK_MODEL=nv "$PY" -m pii_router.render_config "$WORK/config.yaml"
PYTHONPATH="$PWD" PII_ROUTER_LAYA_URL=http://127.0.0.1:18103 LITELLM_MASTER_KEY="$KEY" CLOUD_API_KEY=gpu-key FALLBACK_API_KEY=fake \
  CLOUD_BASE_URL=http://127.0.0.1:18105/v1 FALLBACK_BASE_URL=http://127.0.0.1:18104/v1 PII_ROUTER_CLOUD_HEALTH_INTERVAL=2 \
  LITELLM_LOCAL_MODEL_COST_MAP=True "$BIN/litellm" --config "$WORK/config.yaml" --host 127.0.0.1 --port 18100 \
  >"$WORK/litellm2.log" 2>&1 & pids+=($!)
for i in $(seq 1 90); do curl -sf -o /dev/null http://127.0.0.1:18100/health/liveliness && break; sleep 1; done

for _ in 1 2; do curl -sf -o /dev/null http://127.0.0.1:18104/v1/models && break; sleep 1; done
check "Explain the difference between TCP and UDP." false fallback
check "Email ann.kelly@acme.ie about her overdue invoice." false local
"$PY" tests/mock_upstream.py gpu 18105 >"$WORK/gpu.log" 2>&1 & pids+=($!)
for i in $(seq 1 30); do curl -sf -o /dev/null http://127.0.0.1:18105/v1/models && break; sleep 1; done
sleep 3  # > PII_ROUTER_CLOUD_HEALTH_INTERVAL, so the next request re-probes
check "Explain the difference between TCP and UDP." true gpu

[ $fail -eq 0 ] && echo "E2E PASS" || { echo "E2E FAIL"; tail -20 "$WORK/litellm.log"; tail -20 "$WORK/litellm2.log" 2>/dev/null; exit 1; }
