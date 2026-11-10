# Inference Router

One OpenAI-compatible endpoint for a company. It routes each request to any OpenAI-compatible
upstream (NVIDIA, OpenRouter, OpenAI, Anthropic's OpenAI-compatible API, a self-hosted vLLM,
later a rented Vast.ai GPU or office machines through a tunnel), with ordered fallback, and can
mask personal data or keep personal requests on private upstreams. It never stores message content.

Design: [docs/DESIGN.md](docs/DESIGN.md). Status: phase 1 in progress (see below).

## What works now

- `router.yaml`: upstreams, routes (model names with ordered fallback targets), privacy mode
  (`off`, `mask`, `mask_check`, `keep_private`), built-in PII detectors, custom regex rules with
  keywords and checksums, exceptions.
- Masking with placeholders restored in replies, tool calls and streams; refusals with an
  explanation; fail closed (checker down, unreadable media, slow rule, unknown model).
- Admin web UI: live metadata feed and upstream cards, upstream Test, the Experimenter
  (dry run or send-for-real, shown in the browser only), settings editor.
- Not yet: per-person API keys (phase 1, next), Vast GPU upstream, tunnelled machines, PAIR.

## Install (Linux, systemd)

```bash
git clone https://github.com/oidebrett/inference-router && cd inference-router
sudo IR_KEY_NVIDIA=nvapi-... IR_SEED_CONFIG=./my-router.yaml ./install.sh
```

- API: `http://127.0.0.1:4100/v1` (set `IR_HOST`/`IR_PORT`). Publish it as e.g. `inference.<domain>`
  through Pangolin, `/v1/*` only.
- Admin UI: `http://127.0.0.1:4110/`. Publish it only behind Pangolin SSO (separate hostname), or
  reach it over SSH: `ssh -L 4110:127.0.0.1:4110 <host>`, then open http://127.0.0.1:4110.
- Admin key: `sudo sed -n 's/^LITELLM_MASTER_KEY=//p' /etc/inference-router/secrets.env`.
- `IR_WITH_LAYA=no` skips the Laya classifier (~1.7 GB; needed for `mask_check` and the Laya part of
  `keep_private`).

Example `router.yaml`:

```yaml
version: 1
privacy:
  mode: mask
  rules:
    custom:
      - {name: CLAIM_NUMBER, pattern: 'CLM-\d{4}-\d{7}', keywords: [claim]}
    exceptions: ['@example.com']
upstreams:
  - {id: nvidia, name: NVIDIA, base_url: https://integrate.api.nvidia.com/v1}
  - {id: openrouter, name: OpenRouter, base_url: https://openrouter.ai/api/v1}
routes:
  - name: default
    targets:
      - {upstream: nvidia, model: nvidia/nemotron-3.5-super-vl-120b-a12b}
      - {upstream: openrouter, model: openai/gpt-4o-mini}
```

Keys never go in `router.yaml`: they live in `/etc/inference-router/secrets.env` as
`IR_KEY_<UPSTREAM ID>` (set them in the admin UI or pass them to `install.sh`).

## Layout

| Path | |
|---|---|
| `router/inference_router/` | config, render (router.yaml → LiteLLM), detection, hook, auth |
| `router/pii_router/` | detectors, masking, Laya client, policy (imported from openshell_controller with history) |
| `admin/ir_admin/` | admin API and web UI |
| `install.sh`, `systemd/` | installer and units |

## Tests

```bash
cd router && python -m pytest -q tests                      # unit tests
cd admin && PYTHONPATH=.:../router python -m pytest -q tests
PY=$(which python) router/tests/e2e_router.sh               # real LiteLLM proxy, fake upstreams
```
