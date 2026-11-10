# Inference Router — design

Status: draft, 2026-11-10. Nothing here is built yet. The router core it starts from is in
production as `openshell_controller/scripts/pii-router` and was verified live on 2026-11-09
(see [Starting point](#starting-point)).

## 1. What it is

A company wants **one OpenAI-compatible endpoint** that all its people, agents and apps use.
Behind it, an Inference Router sends each request to the right **upstream**: a public API
(NVIDIA, OpenRouter, Anthropic, OpenAI…), a GPU rented on Vast.ai, or the company's own
machines reached through a tunnel. On the way it can **mask personal data** or **keep
personal requests on private hardware**, and it gives every person or app **their own key**.

It replaces manidae-cloud's *Private Inference* package, which bundled one fixed design
(rent a Vast GPU, NVIDIA fallback, PII checkbox) into a long deploy form. Here the deploy
form only creates the box; everything else is configured afterwards in the router's own UI.

### Goals

- Any OpenAI-compatible upstream, configured in the UI, with fallback between them.
- Per-person / per-team / per-app API keys with limits, revocation and usage counts.
- Privacy options: off, mask, mask + check, keep private (route to a trusted upstream).
- Custom PII rules (regex) for company identifiers.
- Show masking and routing working **without storing message content** (the Experimenter).
- Private upstreams (Vast GPU, office machines, an NVIDIA PAIR cluster) reached only through
  outbound tunnels.

### Non-goals

- Storing prompts or replies. **No content logging, anywhere** (decision 2026-11-10: no extra
  confidential-data surface, no retention or access problem to solve).
- Pooling GPU memory or splitting one request across machines (PAIR does not either).
- Being an inference engine. Engines are upstreams (vLLM, Ollama, PAIR, hosted APIs).

## 2. Architecture

```
people / agents / apps
   │  HTTPS, Authorization: Bearer <their key>
   ▼
inference.<domain>  ── Pangolin (Traefik, Let's Encrypt) + CrowdSec ──►  Inference Router
                                                                          │  key check → route → privacy
                                                                          ├─► NVIDIA / OpenRouter / Anthropic / OpenAI / any OpenAI-compatible API   (direct HTTPS)
                                                                          ├─► Vast.ai GPU (vLLM)            ◄── Newt tunnel (outbound from the GPU)
                                                                          ├─► office GPU / NVIDIA PAIR      ◄── Newt tunnel (outbound from the office)
                                                                          └─► local model on this box (optional, small)
admin ──► router.<domain>  (admin UI, behind Pangolin SSO)
```

- **Pangolin** is the front door (public HTTPS, certificates, CrowdSec) and the **tunnel hub**.
  Private upstreams run **Newt** (WireGuard), which connects *out* to Pangolin: no inbound ports
  on the GPU or in the office. This is how Private Inference already reaches its Vast GPU
  (manidae `gpu_supervisor.py`: vLLM bound to 127.0.0.1, Newt on the instance).
  Open question: make tunnelled upstreams reachable **only by the router**, not as public
  Pangolin resources (today the GPU is a public `gpu.<domain>` resource protected by its vLLM key).
- **Router core: LiteLLM proxy + our hook**, as today. LiteLLM gives the OpenAI-compatible API,
  provider translation (e.g. OpenAI format → Anthropic), streaming, fallbacks and **virtual keys**
  (needs Postgres). Our hook adds key-independent routing decisions and privacy.
- **Admin service** (new, small, Python/FastAPI) serves the admin UI and the Experimenter, writes
  the router config, and drives LiteLLM's management API for keys and usage.

## 3. Concepts

### Upstreams

An upstream is an OpenAI-compatible endpoint plus how to reach it.

| Field | Notes |
|---|---|
| name | shown on its card |
| type / preset | NVIDIA, OpenRouter, Anthropic API, OpenAI API, other OpenAI-compatible, Vast GPU (managed), connected machine (tunnel), PAIR cluster (tunnel), local model |
| base URL, API key | key stored encrypted at rest, never shown again |
| models | discovered from `/v1/models` where possible, editable |
| private | **trusted to see personal data** (own hardware); used by "keep private" |
| enabled | toggle |

A **Test** button on each card sends a fixed harmless prompt and shows latency and the reply.
Hosted APIs do not get `reasoning_effort` by default (not every hosted model accepts
`"none"`); self-hosted vLLM defaults to no thinking (measured: ~0.5 s vs ~5 s on Qwen3.8 27B).

### Routes

The model names callers use (`default`, `fast`, `coder`…), each an ordered list of
upstream + model, tried in order (fallback). `/v1/models` lists route names only, never
upstream details. Later: cost/capability tiers (simple → efficient model, complex → capable
one) — done inside the router, after masking, so a classifier only ever sees masked text.

### Keys (consumers)

| | |
|---|---|
| owner | person, team or app |
| shown once | stored hashed |
| allowed routes | e.g. HR may only use `default` |
| privacy level | per key, overrides the default (HR = mask + check) |
| limits | requests/min, tokens/day, optional budget |
| expiry, revoke, rotate | |
| usage | **counts only**: requests, tokens, masked values, blocked — never content |

Built (2026-11-10) as our own SQLite key store (`router/inference_router/keys.py`) rather than LiteLLM
virtual keys, which need Postgres plus Prisma engine downloads on every box. Keys need only a name: defaults
are all routes, the router's privacy, 60 requests/minute, no token limit, no expiry. The master key is the admin key.

### Privacy

| Level | What leaves the box | Needs |
|---|---|---|
| off | the request as sent | — |
| **mask** | detected values replaced by placeholders `<EMAIL_7f3a2c1b>`; real values restored in the reply | nothing extra |
| mask + check | as mask; Laya re-checks the masked text, still-personal requests are **refused** with an explanatory reply | Laya (~2 GB RAM, ~0.6 s/new message on CPU) |
| keep private | personal requests go **only to upstreams marked private** (office GPU, PAIR, Vast GPU if trusted, local model); others go anywhere | at least one private upstream |

Masking (shipped in the router 2026-11-09, unchanged in principle):
- placeholders are `HMAC(secret, normalised value)`: stable across turns (consistent history,
  provider prompt cache keeps working), unguessable without the secret, nothing stored between requests;
- restored in message text, tool-call arguments (JSON-escaped) and streamed chunks (split placeholders held back);
- placeholder look-alikes in incoming text are neutralised first (`<` → `‹`); only this request's placeholders are restored;
- system prompts are masked too (agent memory can carry PII there);
- the Laya check sees placeholders replaced by plain phrases ("an email address"): with raw placeholders
  Laya scored masked text as personal (0.22–0.93, swinging with the placeholder's random hex); with
  phrases, maskable requests scored ≤0.28 and a name with health/salary details 0.68–0.82 (threshold 0.4);
- fails closed: checker down, oversize, unreadable media, router error, non-chat call, no reachable
  upstream → refused (or kept private), never sent unmasked;
- limits: pseudonymisation, not anonymisation; **names and addresses are not masked** (left to the check).

Replies carry `x-pii-route` and `x-pii-reason` (`masked|clean|current-turn|history|fail-closed`) and
`x-pii-masked: <n>`; masked replies end with ` ‡` (configurable, off for strict-JSON clients).

### PII rules

**Privacy → Rules** in the UI:

1. **Built-in detectors**, each toggleable: email, phone, IBAN (mod-97), payment card (Luhn), US SSN,
   UK NINO, Irish PPSN (checksum), passport, date of birth (keyword-anchored).
2. **Custom rules** for company identifiers: name (becomes the placeholder type, e.g. `CLAIM_NUMBER`),
   regex, optional validator (Luhn, mod-97, decodes-to-known-PII), optional keyword anchor.
3. **Exceptions**: values never masked (support@company.com, example.com, internal ticket formats).
4. **Rule packs** (YAML, importable): insurance, healthcare, finance, legal — the idea from piiguard,
   whose results showed seed packs help most on unseen identifier formats.

Safeguards: a per-match time limit (`regex` module); a rule that times out counts as a match
(masked, never leaked). Saving runs the rule against a clean corpus (logs, UUIDs, timestamps,
versions, tracking numbers) and warns about false positives. Rules are plain YAML so they can be
kept in git and reviewed as a diff.

Later, from piiguard: decode-then-validate for disguised values (base64, hex, percent-encoding,
reversed or spelled-out digits, full-width digits, odd separators).

### No content logging; the Experimenter instead

The router never writes prompts or replies to disk. The per-request **decision line** in the
journal holds metadata only (route, reason, types and counts masked, timings) and can be turned off.
The live requests feed in the UI is metadata only, in memory, gone on restart.

To **see** masking and routing work, the **Experimenter** (admin only):

- input: a message or a short conversation (tool calls included); choose route and key to simulate;
- **dry run by default**, same code path as real traffic, nothing forwarded or stored. It shows:
  1. detected values highlighted and labelled (built-in and custom rules);
  2. the text that would leave the box, with placeholders;
  3. the Laya score and verdict (mask + check);
  4. the decision: which upstream and why (masked → OpenRouter, blocked, kept private → office GPU, fallback);
- optional **send for real**: shows the model's raw reply (placeholders) next to the restored reply,
  in the browser only;
- example library (email + phone, IBAN + PPSN, a named person with health details, PII only in history,
  a tool call) so demos never need real personal data;
- the custom-rule editor uses it to test a regex before saving.

Endpoint: `POST /admin/dry-run`, admin auth, rate-limited, no persistence.

## 4. Admin UI

A web app (not a desktop app) with the layout of NVIDIA PAIR, which fits a router well.

| Screen | Content |
|---|---|
| **Overview** | left: live requests feed (model, key, upstream, badges: "3 masked", "blocked", "kept private"); right: **upstream cards** (status, models, requests/min, errors, latency; for tunnelled GPUs live GPU/memory charts); lines from requests to the card that served them |
| **Add upstream** | presets (NVIDIA, OpenRouter, Anthropic, OpenAI, other API, rent a Vast GPU, connect a machine, connect a PAIR cluster, personal subscription) |
| **Connect a machine** | one-time code + one-line install command for the office machine (installs Newt, registers the upstream); the card appears when the tunnel is up. PAIR's PIN pairing idea, over the internet via Pangolin |
| **Endpoints** | base URL, curl example, copy buttons |
| **Keys** | create (shown once), limits, privacy level, expiry, revoke, usage counts |
| **Routes** | route names → ordered upstream/model lists |
| **Privacy** | level, rules, exceptions, packs |
| **Experimenter** | as above |
| **Settings** | admin users (Pangolin SSO), tunnels, backups of the config |

PAIR is Apache-2.0; its React components can be reused with attribution, not its NVIDIA branding.
Stack: React + Tailwind, dark theme, served by the admin service.

## 5. Upstream types in detail

- **Public APIs**: NVIDIA (`integrate.api.nvidia.com/v1`), OpenRouter (`openrouter.ai/api/v1`),
  any other OpenAI-compatible URL; Anthropic and OpenAI APIs through LiteLLM's provider translation.
  Verified as router backends live: NVIDIA, OpenRouter (`openai/gpt-4o-mini`).
- **Vast.ai GPU (managed)**: the router rents a GPU on the customer's Vast account, starts vLLM with
  Newt, watches it (rebid, replace, budget), destroys it on removal. Code exists in manidae-cloud
  (`vastai_client.py`, `gpu_supervisor.py`) and moves here. Cold start measured 2026-11-09: ~25 min on
  a Bulgarian RTX PRO 6000 (31 GB weights download, compile, warm-up); earlier runs ~5 min. While it
  starts, the route's next upstream answers.
  Spend controls (today manidae only has a max price per hour plus the customer's prepaid credit,
  checked before each rental with a warning under 24 h of runway): add a **total spend limit**
  (per day / month) that stops renting when reached, **stop when idle** (release the GPU after N
  minutes without traffic, rent again on the next request, fallback answers during the cold start),
  and an optional **schedule** (e.g. office hours only). Recovery as in manidae's supervisor: rebid
  +30% on the same instance when outbid, on-demand for 24 h after two outbids in an hour, replace
  lost/unhealthy GPUs with the same key and tunnel, pause after 6 rentals in an hour, sweep orphans.
  Outbid → rebid → on-demand has only been unit-tested, never seen live.
- **Connected machine**: anything serving an OpenAI-compatible API on the office network (vLLM, Ollama,
  LM Studio, llama.cpp), reached through Newt.
- **NVIDIA PAIR cluster**: PAIR pools Ollama / LM Studio / llama.cpp / vLLM across machines on one LAN
  and exposes OpenAI-compatible endpoints. Treat the cluster as one private upstream behind Newt.
  PAIR's own docs: pairing is plaintext authenticated only by a PIN, host/GPU telemetry is
  unauthenticated on the subnet, the proxy serves plaintext to the local machine. So expose it **only**
  through the tunnel, with the router's keys in front; read its telemetry only through the tunnel.
  PAIR is very new: optional integration, never a dependency.
- **Local model on the box**: optional, small; on CPU-only servers a local model is 3–15 min per agent
  turn (openshell_controller `docs/local-inference-feasibility.md`), so "keep private" should normally
  point at real GPUs.
- **Personal subscription (Claude / ChatGPT)**: technically possible (Claude `setup-token` ran live on
  2026-10-09; ChatGPT device-code sign-in started but was never verified end to end). It is a personal
  account: sharing it across a company may breach the provider's terms and risks suspension, every
  request draws on one person's usage cap, and the token is a whole-account credential. So: offered only
  as a **personal upstream usable only by keys owned by the admin who connected it**, clearly labelled,
  with automatic fallback when the cap is hit. Shared use should use Anthropic / OpenAI **API keys**.
  Check the providers' current terms before shipping this.
  **Built (2026-11-10):** upstream kinds `claude_subscription` / `chatgpt_subscription`
  (`router/inference_router/subscriptions.py`); routes with one get a shared chain
  `ir/<route>/shared/<n>` without it, used for every key not marked `subscriptions`; the router
  never starts a ChatGPT sign-in itself (only the admin UI does).

## 6. Security notes

- Upstream keys encrypted at rest; consumer keys stored hashed; nothing secret in logs.
- Admin UI and `/admin/*` only behind Pangolin SSO plus the admin key; never on the public endpoint.
- Callers cannot bypass privacy by naming an upstream model directly; every chat request goes through
  the hook. If the hook is not loaded, the public names point at a deployment that only returns a
  "router not active" message (as today).
- No content logging (above). Usage is counts only.
- Masking is pseudonymisation; say so to customers.

## 7. Starting point

`openshell_controller/scripts/pii-router` (merged to `gatewaydashboard` as of `c6c5fa0`):
LiteLLM 1.104.0 proxy, hook (`hook.py`), detectors with checksums, Laya client (laya 0.3.26), masking
(`mask.py`), install script with modes `off|strict|masked`, residual check/action, generic
`CLOUD_*` and `FALLBACK_*` endpoints, 140 unit tests and two end-to-end scripts.

Verified live on 2026-11-09 (Oracle VPS, curl only, 9 cases each): against NVIDIA, OpenRouter and a
Vast RTX PRO 6000 (direct port, not yet through Newt): email/phone/IBAN/PPSN masked and restored,
tool calls (normal and streamed), multi-turn history, name + salary / sick leave refused, GPU destroyed
→ fallback still masked. Masking-only (no Laya) also verified.

Users today: AgentGateway (`agentgateway-setup.sh`, PII routing to Ollama) and manidae Private
Inference (preview only, `enabled: false`).

## 8. Plan

| Phase | Scope |
|---|---|
| 0 | This repo. Extract `scripts/pii-router` with its git history; CI running the unit and e2e tests. openshell_controller installs the router from here (pinned tag) instead of carrying a copy. |
| 1 | Config file (upstreams, routes, privacy, rules) → LiteLLM config; LiteLLM virtual keys with Postgres; admin service; UI: Overview (metadata feed + upstream cards), Add upstream (public API presets), Endpoints, Keys, Privacy + Rules, **Experimenter**. Enough for "one endpoint over NVIDIA / OpenRouter with masking and per-person keys". |
| 2 | manidae-cloud package **Inference Router** with a minimal form (domain, admin login); replaces Private Inference. |
| 3 | Vast GPU upstream (move manidae's supervisor), GPU charts for tunnelled upstreams, tunnelled upstreams reachable only by the router. |
| 4 | Connect a machine (one-time code + Newt), PAIR cluster upstream. |
| 5 | Cost/capability routes; disguised-value decoders; rule packs; personal subscriptions (after a terms check). |

## 9. Open questions

1. Can Pangolin expose a tunnelled resource only to the router (no public hostname)? If not, how do
   we keep tunnelled upstreams private (Pangolin access rules, a Newt-only network, or WireGuard direct)?
2. LiteLLM virtual keys need Postgres and its admin features are partly enterprise-licensed: confirm
   which key/budget features are in the open-source build at the pinned version.
3. Where the router config lives: a YAML file owned by the admin service (git-friendly) vs LiteLLM's DB.
4. Repo visibility (created private) and licence.
5. Whether AgentGateway keeps a router on the agent box or points at a shared Inference Router.
