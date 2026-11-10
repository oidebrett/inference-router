#!/usr/bin/env bash
# Masked mode end to end: real LiteLLM proxy + hook, against a FAKE echo backend
# that records what it receives. No Laya (PII_ROUTER_RESIDUAL_CHECK=none), no model.
#   PY=/path/to/venv/bin/python tests/e2e_masked.sh
# The venv needs: litellm[proxy] fastapi uvicorn.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"; BIN="$(dirname "$PY")"
WORK="$(mktemp -d)"; KEY="sk-e2e-$RANDOM$RANDOM"; pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; rm -rf "$WORK"; }
trap cleanup EXIT
"$PY" tests/mock_echo.py 18201 "$WORK/received.jsonl" >"$WORK/echo.log" 2>&1 & pids+=($!)
PII_ROUTER_MODE=masked NVIDIA_BASE_URL=http://127.0.0.1:18201/v1 NVIDIA_MODEL=echo \
  "$PY" -m pii_router.render_config "$WORK/config.yaml"
PYTHONPATH="$PWD" PII_ROUTER_MODE=masked PII_ROUTER_RESIDUAL_CHECK=none PII_ROUTER_MASK_SECRET=e2e \
  LITELLM_MASTER_KEY="$KEY" NVIDIA_API_KEY=fake LITELLM_LOCAL_MODEL_COST_MAP=True \
  "$BIN/litellm" --config "$WORK/config.yaml" --host 127.0.0.1 --port 18200 >"$WORK/litellm.log" 2>&1 & pids+=($!)
for _ in $(seq 1 90); do curl -sf -o /dev/null http://127.0.0.1:18200/health/liveliness && break; sleep 1; done
fail=0
ask() { # body-json -> response (headers in $WORK/h.txt)
  curl -sS -D "$WORK/h.txt" http://127.0.0.1:18200/v1/chat/completions -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" -d "$1"
}
pass() { echo "PASS $1"; }
bad() { echo "FAIL $1"; fail=1; }
EMAIL=ann.kelly@acme.ie; CARD="4111 1111 1111 1111"

out=$(ask "{\"model\":\"pii-router\",\"messages\":[{\"role\":\"user\",\"content\":\"Mail $EMAIL, card $CARD\"}]}")
echo "$out" | grep -q "seen: Mail $EMAIL, card $CARD" && pass "reply restored" || { bad "reply restored: $out"; }
echo "$out" | grep -q '‡\|\\u2021' && pass "masked marker" || bad "masked marker"
grep -qi '^x-pii-reason: masked' "$WORK/h.txt" && grep -qi '^x-pii-masked: 2' "$WORK/h.txt" && pass "headers" || { bad "headers"; cat "$WORK/h.txt"; }

out=$(ask "{\"model\":\"pii-router\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Mail $EMAIL now\"}]}")
joined=$(echo "$out" | sed -n 's/^data: //p' | grep -v DONE | "$PY" -c 'import sys,json; print("".join((json.loads(l)["choices"][0]["delta"].get("content") or "") for l in sys.stdin if l.strip()))')
[ "$joined" = "seen: Mail $EMAIL now ‡" ] && pass "stream restored across split chunks" || bad "stream: '$joined'"

out=$(ask "{\"model\":\"pii-router\",\"messages\":[{\"role\":\"user\",\"content\":\"TOOL: mail $EMAIL\"}]}")
args=$(echo "$out" | "$PY" -c 'import sys,json; print(json.loads(sys.stdin.read())["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])')
[ "$args" = "{\"to\": \"$EMAIL\"}" ] && pass "tool-call args restored" || bad "tool args: $args"

out=$(ask "{\"model\":\"pii-router\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"TOOL: mail $EMAIL\"}]}")
sargs=$(echo "$out" | sed -n 's/^data: //p' | grep -v DONE | "$PY" -c '
import sys, json
out = []
for line in sys.stdin:
    if not line.strip(): continue
    for c in json.loads(line)["choices"][0]["delta"].get("tool_calls") or []:
        out.append((c.get("function") or {}).get("arguments") or "")
print("".join(out))')
[ "$sargs" = "{\"to\": \"$EMAIL\"}" ] && pass "streamed tool-call args restored" || bad "streamed tool args: $sargs"

out=$(ask '{"model":"pii-router","messages":[{"role":"user","content":[{"type":"text","text":"what is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}')
echo "$out" | grep -q 'PII router' && grep -qi '^x-pii-route: blocked' "$WORK/h.txt" && pass "unreadable image blocked with a reply" || { bad "block: $out"; cat "$WORK/h.txt"; }

out=$(ask "{\"model\":\"cloud\",\"messages\":[{\"role\":\"user\",\"content\":\"direct $EMAIL\"}]}")
echo "$out" | grep -q "seen: direct $EMAIL" && pass "naming the cloud model still masks" || bad "bypass: $out"

if grep -qE "ann\.kelly|4111" "$WORK/received.jsonl"; then bad "a real value reached the backend"; else pass "backend never saw a real value ($(wc -l < "$WORK/received.jsonl" | tr -d ' ') requests)"; fi
grep -q 'data:image' "$WORK/received.jsonl" && bad "blocked request reached the backend" || pass "blocked request never reached the backend"
if grep "pii-router-decision" "$WORK/litellm.log" | grep -qE "ann\.kelly|4111"; then bad "value in decision log"; else pass "decision log has no values"; fi
echo "--- decision log:"; grep "pii-router-decision" "$WORK/litellm.log" | sed 's/.*pii-router-decision //' | cut -c1-220
[ "$fail" = 0 ] || { echo "--- litellm log tail"; tail -30 "$WORK/litellm.log"; }
exit "$fail"
