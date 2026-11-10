#!/usr/bin/env bash
# Inference Router end to end: real LiteLLM proxy + inference_router hook, against FAKE
# upstreams (tests/mock_echo.py records what it receives). No Laya, no real model.
#   PY=/path/to/venv/bin/python tests/e2e_router.sh      (venv: litellm[proxy] fastapi uvicorn)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"; BIN="$(dirname "$PY")"
WORK="$(mktemp -d)"; KEY="sk-e2e-$RANDOM$RANDOM"; pids=(); fail=0
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; rm -rf "$WORK"; }
trap cleanup EXIT
pass() { echo "PASS $1"; }
bad() { echo "FAIL $1"; fail=1; }

"$PY" tests/mock_echo.py 18301 "$WORK/public.jsonl" >"$WORK/public.log" 2>&1 & pids+=($!)
"$PY" tests/mock_echo.py 18302 "$WORK/private.jsonl" >"$WORK/private.log" 2>&1 & pids+=($!)

start_router() { # $1 = privacy mode
  cat > "$WORK/router.yaml" <<YAML
version: 1
privacy: {mode: $1, rules: {custom: [{name: CLAIM_NUMBER, pattern: 'CLM-\\d{4}-\\d{7}'}]}}
upstreams:
  - {id: dead, name: Down, base_url: http://127.0.0.1:18399/v1}
  - {id: public, name: Public API, base_url: http://127.0.0.1:18301/v1}
  - {id: office, name: Office GPU, base_url: http://127.0.0.1:18302/v1, private: true}
routes:
  - name: default
    targets: [{upstream: dead, model: m1}, {upstream: public, model: echo}, {upstream: office, model: echo}]
YAML
  PYTHONPATH="$PWD" "$PY" -m inference_router.render "$WORK/router.yaml" "$WORK/litellm.yaml"
  [ -n "${RPID:-}" ] && { kill "$RPID" 2>/dev/null || true; sleep 2; }
  PYTHONPATH="$PWD" IR_CONFIG="$WORK/router.yaml" IR_MASK_SECRET=e2e LITELLM_MASTER_KEY="$KEY" IR_KEYS_DB="$WORK/keys.db" \
    IR_KEY_DEAD=x IR_KEY_PUBLIC=x IR_KEY_OFFICE=x LITELLM_LOCAL_MODEL_COST_MAP=True \
    "$BIN/litellm" --config "$WORK/litellm.yaml" --host 127.0.0.1 --port 18300 >"$WORK/litellm-$1.log" 2>&1 &
  RPID=$!; pids+=($RPID)
  for _ in $(seq 1 90); do curl -sf -o /dev/null http://127.0.0.1:18300/health/liveliness && return; sleep 1; done
  echo "router did not start"; tail -30 "$WORK/litellm-$1.log"; exit 1
}
ask() { curl -sS -D "$WORK/h.txt" http://127.0.0.1:18300/v1/chat/completions -H "Authorization: Bearer $KEY" \
          -H "Content-Type: application/json" -d "$1"; }

start_router mask
out=$(ask '{"model":"default","messages":[{"role":"user","content":"claim CLM-2026-0048213 for ann.kelly@acme.ie"}]}')
echo "$out" | grep -q "seen: claim CLM-2026-0048213 for ann.kelly@acme.ie" && pass "masked, fell back past the dead upstream, restored" || bad "mask: $out"
grep -qi '^x-pii-reason: masked' "$WORK/h.txt" && grep -qi '^x-router-route: default' "$WORK/h.txt" && pass "headers" || { bad "headers"; cat "$WORK/h.txt"; }
grep -q 'CLM-2026\|ann.kelly' "$WORK/public.jsonl" && bad "a real value reached the upstream" || pass "upstream saw placeholders only"
grep -q '<CLAIM_NUMBER_' "$WORK/public.jsonl" && pass "custom rule placeholder sent" || bad "custom rule"
out=$(ask '{"model":"default","stream":true,"messages":[{"role":"user","content":"TOOL: mail ann.kelly@acme.ie"}]}')
args=$(echo "$out" | sed -n 's/^data: //p' | grep -v DONE | "$PY" -c '
import sys, json
print("".join(((c.get("function") or {}).get("arguments") or "") for l in sys.stdin if l.strip()
              for c in (json.loads(l)["choices"][0]["delta"].get("tool_calls") or [])))')
[ "$args" = '{"to": "ann.kelly@acme.ie"}' ] && pass "streamed tool call restored" || bad "stream tool args: $args"
out=$(ask '{"model":"gpt-5","messages":[{"role":"user","content":"hi"}]}')
echo "$out" | grep -q "Unknown model 'gpt-5'" && pass "unknown model refused" || bad "unknown: $out"
out=$(ask '{"model":"ir/default/1","messages":[{"role":"user","content":"direct ann.kelly@acme.ie"}]}')
echo "$out" | grep -q "Unknown model" && pass "chain deployment cannot be called directly" || bad "bypass: $out"
out=$(ask '{"model":"default","messages":[{"role":"user","content":[{"type":"text","text":"x"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}')
echo "$out" | grep -q "Inference Router" && grep -qi '^x-pii-route: blocked' "$WORK/h.txt" && pass "image refused" || bad "image: $out"
models=$(curl -sS http://127.0.0.1:18300/v1/models -H "Authorization: Bearer $KEY")
echo "$models" | "$PY" -c 'import sys,json; ids=[m["id"] for m in json.load(sys.stdin)["data"]]; sys.exit(0 if ids==["default"] else 1)' \
  && pass "/v1/models lists routes only" || bad "models: $models"
[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18300/v1/models -H 'Authorization: Bearer wrong')" = 401 ] \
  && pass "wrong key -> 401" || bad "wrong key"

# Per-person keys (keys.db is shared with the admin service).
read -r ANN ANN_ID BOT <<<"$(PYTHONPATH="$PWD" "$PY" -c "
from inference_router.keys import KeyStore
ks = KeyStore('$WORK/keys.db')
ann, s1 = ks.create('Ann', routes=['default'])
bot, s2 = ks.create('Bot', rpm=1)
print(s1, ann.id, s2)")"
uask() { curl -sS -D "$WORK/h.txt" http://127.0.0.1:18300/v1/chat/completions -H "Authorization: Bearer $1" \
          -H "Content-Type: application/json" -d "$2"; }
out=$(uask "$ANN" '{"model":"default","messages":[{"role":"user","content":"mail ann.kelly@acme.ie"}]}')
echo "$out" | grep -q "seen: mail ann.kelly@acme.ie" && pass "person key works, still masked + restored" || bad "person key: $out"
models=$(curl -sS http://127.0.0.1:18300/v1/models -H "Authorization: Bearer $ANN")
echo "$models" | grep -q '"default"' && pass "person key lists its routes" || bad "person models: $models"
uask "$BOT" '{"model":"default","messages":[{"role":"user","content":"hi"}]}' >/dev/null
[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18300/v1/chat/completions -H "Authorization: Bearer $BOT" -H 'Content-Type: application/json' -d '{"model":"default","messages":[{"role":"user","content":"hi"}]}')" = 429 ] \
  && pass "rate limit -> 429" || bad "rate limit"
PYTHONPATH="$PWD" "$PY" -c "from inference_router.keys import KeyStore; KeyStore('$WORK/keys.db').revoke($ANN_ID)"
[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18300/v1/chat/completions -H "Authorization: Bearer $ANN" -H 'Content-Type: application/json' -d '{"model":"default","messages":[{"role":"user","content":"hi"}]}')" = 401 ] \
  && pass "revoked key -> 401 immediately" || bad "revoked key"
PYTHONPATH="$PWD" "$PY" -c "
from inference_router.keys import KeyStore
u = {k['name']: k['usage']['today'] for k in KeyStore('$WORK/keys.db').list()}
assert u['Ann']['requests'] == 1 and u['Ann']['masked'] == 1, u
print('usage', u)" && pass "usage counted per key (counts only)" || bad "usage"

: > "$WORK/private.jsonl"
start_router keep_private
out=$(ask '{"model":"default","messages":[{"role":"user","content":"mail ann.kelly@acme.ie"}]}')
grep -q 'ann.kelly@acme.ie' "$WORK/private.jsonl" && grep -qi '^x-pii-route: private' "$WORK/h.txt" \
  && pass "keep_private: personal request went unmasked to the private upstream only" || bad "keep_private: $out"
before=$(wc -l < "$WORK/public.jsonl")
ask '{"model":"default","messages":[{"role":"user","content":"Explain TCP"}]}' >/dev/null
[ "$(wc -l < "$WORK/public.jsonl")" -gt "$before" ] && pass "keep_private: clean request used the public chain" || bad "clean route"
grep -q 'ann.kelly' "$WORK/public.jsonl" && bad "personal data reached the public upstream" || pass "public upstream never saw personal data"

grep -h "ir-decision" "$WORK"/litellm-*.log | grep -qE "ann\.kelly|CLM-2026" && bad "content in decision log" || pass "decision log has no content"
[ "$fail" = 0 ] || { echo "--- logs"; tail -20 "$WORK"/litellm-*.log; }
exit "$fail"
