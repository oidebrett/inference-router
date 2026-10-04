# PII router

An OpenAI-compatible proxy for the agent gateway. Requests that contain personal
data go to a **local Ollama model**; everything else goes to **NVIDIA**. Sandboxes
are unchanged: they keep calling `inference.local`, and the OpenShell gateway calls
this router as a NemoClaw "custom" provider.

```
sandbox → inference.local → OpenShell gateway → pii-router (127.0.0.1:4100/v1, model "pii-router")
                                                  ├─ PII     → Ollama  (OLLAMA_MODEL, e.g. qwen3:4b)
                                                  └─ not PII → NVIDIA  (NVIDIA_MODEL)
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

## Install (on the gateway host)

```bash
sudo OLLAMA_MODEL=qwen3:4b NVIDIA_API_KEY=nvapi-... scripts/pii-router/install.sh
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

## Tests

```bash
python -m pytest scripts/pii-router/tests          # unit tests (needs litellm + httpx)
PY=/path/to/venv/bin/python scripts/pii-router/tests/e2e.sh   # real Laya + LiteLLM, fake backends
```
