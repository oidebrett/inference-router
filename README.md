# PII router

An OpenAI-compatible proxy for the agent gateway. Requests that contain personal
data go to a **local Ollama model**; everything else goes to **NVIDIA**. Sandboxes
are unchanged: they keep calling `inference.local`, and the OpenShell gateway calls
this router as a NemoClaw "custom" provider.

```
sandbox → inference.local → OpenShell gateway → pii-router (127.0.0.1:4100/v1, model "pii-router")
                                                  ├─ PII     → Ollama  (OLLAMA_MODEL, e.g. qwen3.5:4b)
                                                  └─ not PII → NVIDIA  (NVIDIA_MODEL), or a GPU with NVIDIA as fallback
```

## How a request is decided

1. Messages are split into units: user, tool results and assistant turns (including
   tool-call arguments). **System prompts are not scanned.**
2. **Deterministic detectors** (`pii_router/detectors.py`): email, phone, IBAN (mod-97),
   payment card (Luhn), US SSN, UK NINO, Irish PPSN (checksum), passport and date of
   birth (keyword-anchored). IP addresses are deliberately not PII here.
3. **Laya** (`laya-serve`, local, CPU) answers one yes/no question per text chunk for
   units the detectors did not settle. P(yes) ≥ `PII_ROUTER_LAYA_THRESHOLD` (0.4) = PII.
4. Any PII unit → **local**. Verdicts are cached per message hash, so history is scanned
   once and a conversation that ever contained PII **stays local** (the PII message is
   re-sent every turn).

**Fails closed:** Laya down/slow, text beyond the chunk budget, images/files, a router
bug, or a non-chat call type all route **local**. The public model name `pii-router` is
itself configured to the local backend, so if the hook is not loaded nothing reaches the
cloud. Callers cannot bypass the check by naming `nvidia-cloud`.

The decision log (`journalctl -u pii-router | grep pii-router-decision`) records route,
reasons, Laya score and timings — never message text.

Why both detectors and Laya: in the 2026-10-04 evaluation Laya alone caught 8/12 PII
prompts (it misses numbers: IBAN, phone, card), the detectors alone 8/12, together 12/12.
The bare question "does this contain PII?" caught only 3/12 — the wording in
`policy.PII_QUESTION` matters.

Parked follow-up: a **Masked** mode that pseudonymises identifiers and keeps those conversations
on NVIDIA instead of going local — see [`DESIGN-masked-mode.md`](DESIGN-masked-mode.md).

## Where a reply went, and why (2026-10-22)

Every reply carries `x-pii-route: local|cloud|fallback` and `x-pii-reason:
current-turn|history|fail-closed|clean` headers (also on tool-call-only replies, which
have no text to mark). Local text replies end with a marker:

| Marker | Reason | Would a fresh context go to the cloud? |
|---|---|---|
| ` †`  | PII in the current turn (the last user message and everything after it) | No |
| ` ††` | PII only in earlier messages (sticky history) | **Yes**: start a new conversation |
| ` †!` | Could not be cleared (Laya down, oversize, image, cloud down with no fallback) | Not necessarily |

The router strips its own markers from assistant history before deciding and forwarding.
`PII_ROUTER_LOCAL_MARKER=` (empty) turns all markers off; `PII_ROUTER_HISTORY_MARKER` and
`PII_ROUTER_FAILCLOSED_MARKER` change the other two.

## Cloud backend and fallback

The cloud backend is NVIDIA by default (AgentGateway). For **PrivateInference** it is the
customer's own GPU: `CLOUD_BASE_URL` + `CLOUD_MODEL` (+ `CLOUD_API_KEY`). With an
`NVIDIA_API_KEY` as well, NVIDIA becomes the fallback: the router probes
`CLOUD_BASE_URL/models` (cached `PII_ROUTER_CLOUD_HEALTH_INTERVAL`, default 15 s) and sends
clean traffic to NVIDIA while the GPU is down. A failed GPU call marks it down at once; that
one request fails (retries stay off), the next is re-routed. PII never goes to either.

```bash
sudo OLLAMA_MODEL=qwen3.5:4b CLOUD_BASE_URL=https://gpu.example/v1 CLOUD_MODEL=qwen3.8-27b \
     CLOUD_API_KEY=... NVIDIA_API_KEY=nvapi-... scripts/pii-router/install.sh
```

## PII routing off (`PII_ROUTER_MODE=off`)

PrivateInference's default: the router is only an authenticated, health-checked front for
the customer's GPU. Nothing is scanned, Laya/torch are not installed (its unit is removed),
no local model is needed, and every request goes to the cloud backend, or to the NVIDIA
fallback while the GPU is down. Replies carry `x-pii-reason: unchecked` and no markers.
Overhead is the LiteLLM hop only.

## Install (on the gateway host)

```bash
sudo OLLAMA_MODEL=qwen3.5:4b NVIDIA_API_KEY=nvapi-... scripts/pii-router/install.sh
```

Idempotent; re-run to upgrade. Generated secrets (`LITELLM_MASTER_KEY`, `LAYA_API_KEY`)
in `/etc/pii-router/pii-router.env` are preserved. Units: `pii-router-laya`,
`pii-router`. Code: `/opt/pii-router`, config: `/etc/pii-router/config.yaml` (must stay
outside the package directory — see `render_config.py`).

Sizing: Laya's English checkpoint needs ~1.5–2 GB RAM on CPU alongside the Ollama model.
The local model must support **tool calling** or agent turns routed to it fail.

## Settings (env file)

| Variable | Default | |
|---|---|---|
| `PII_ROUTER_LAYA_THRESHOLD` | `0.4` | Laya P(yes) that counts as PII |
| `PII_ROUTER_MAX_CHUNKS` | `32` | Laya chunks per request (new messages only) |
| `PII_ROUTER_CHUNK_CHARS` | `1800` | chunk size (English checkpoint ≈ 512 tokens) |
| `PII_ROUTER_OVERSIZE_ROUTE` | `local` | route when new text exceeds the chunk budget |
| `PII_ROUTER_MEDIA_ROUTE` | `local` | route for images/audio/files (cannot be inspected) |
| `PII_ROUTER_LAYA_TIMEOUT` | `15` | seconds |
| `PII_ROUTER_LAYA_MODEL` | `english` | `multilingual` for non-English traffic |
| `PII_ROUTER_LOCAL_REASONING_EFFORT` | `none` | sent to Ollama as `reasoning_effort`; `none` turns thinking off (qwen3.5:4b on CPU: 3 s vs ~5 min). Empty = model default |
| `PII_ROUTER_CLOUD_REASONING_EFFORT` | `none` | only with `CLOUD_BASE_URL` (a vLLM GPU): sent as `reasoning_effort`; `none` = no thinking (Qwen3.8-27B: 0.5 s vs ~5 s). A caller's own `reasoning_effort` still wins. Empty = model default |
| `PII_ROUTER_PUBLIC_MODELS` | `pii-router` | comma-separated names callers use; each routes like `pii-router`, and `GET /v1/models` lists only these. `pii-router` always works |

## Tests

```bash
python -m pytest scripts/pii-router/tests          # unit tests (needs litellm + httpx)
PY=/path/to/venv/bin/python scripts/pii-router/tests/e2e.sh   # real Laya + LiteLLM, fake backends
```

## Authentication

`pii_router/auth.py` is LiteLLM's `custom_auth`: the master key (`LITELLM_MASTER_KEY`) is the only
valid key, and a wrong or missing key gets `401 Invalid API key`. (Without it LiteLLM, which runs
here with no database, answers a wrong key with `400 No connected db`.)

## Detection limits

Pattern detectors (email, phone, IBAN, card, PPSN/NINO/SSN, passport, date of birth) are reliable. A name
together with health, salary or HR details is judged by Laya alone, which is deterministic but sensitive
to phrasing: "on sick leave for depression; plan her return-to-work" scores 0.80, "off with depression,
suggest one step" 0.30 (missed at the 0.4 threshold); "Name three prime numbers" scores 0.70 (false alarm).
Treat PII routing as best effort (live test 2026-10-23, manidae-cloud docs/private-inference-plan.md §12).
