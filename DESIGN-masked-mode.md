# PII router — "Masked" mode (design, not implemented)

Status: **proposal, parked 2026-10-04.** Builds on the shipped router in this directory
(`README.md`). Nothing here changes current behaviour.

## 1. Problem

Today, once a conversation contains personal data, every turn of it goes to the local
Ollama model ("Strict" behaviour). On a CPU-only Hetzner Cloud VPS that is slow — an
agent turn re-reads ~11–12k prompt tokens at ~105 tok/s, so a new session's first reply
takes ~2 minutes — and the small local model is much weaker than NVIDIA's, especially at
tool use. Often the only personal data is a single identifier ("email ann@acme.ie to
confirm the appointment"), and the task itself does not need the real value to be seen
by the model.

## 2. Idea

Before a request leaves the box, replace personal-data values with **placeholders**
(`<EMAIL_7f3a2c1b>`), send the masked request to NVIDIA, and swap the real values back
into the reply — including tool-call arguments — before the agent sees it. The real
values never leave the box; the conversation keeps NVIDIA's speed and quality.

Anything that still looks personal after masking (e.g. "an employee on sick leave for
depression") falls back to the local model, exactly as today.

## 3. Goals / non-goals

Goals
- Move conversations whose PII is maskable back to the cloud model.
- Keep the fail-closed property: when in doubt, local.
- Agent tool calls receive the real values (e.g. `send_email` gets the real address).
- No change to Strict mode, which stays the default when PII routing is on.

Non-goals
- True anonymisation. This is **pseudonymisation** (see §9).
- Masking the system prompt or agent memory (separate open question, §11).
- Using the local LLM to rewrite text (rejected, §5).

## 4. Modes (user-facing)

Replace the single tickbox under Ollama with a three-way choice:

| Mode | Behaviour | Marker on replies |
|---|---|---|
| **Off** | One provider only (current rules). | — |
| **Strict** *(default when routing is on)* | Today's behaviour: any PII → whole conversation local. | `†` on local replies |
| **Masked** | Mask maskable PII → NVIDIA; anything still personal after masking → local. | `†` local, `‡` masked-cloud |

Validation rules are unchanged: Strict and Masked both need an NVIDIA key and an Ollama
model, and neither can be combined with a custom endpoint.

## 5. Finding what to mask

Masking needs **spans** (which characters are PII), not a yes/no verdict.

- **Laya cannot provide spans.** It answers yes/no per chunk. It stays as the post-mask
  safety gate (§6.3).
- **The local LLM should not do the masking.** A 0.8B model rewriting text can miss an item,
  change meaning or invent text — one miss is a leak — and it is slow.
- **Stage 1: the existing pattern detectors**, refactored to return spans: email, phone,
  IBAN (mod-97), payment card (Luhn), US SSN, UK NINO, Irish PPSN, passport, date of birth.
  Exact, fast, already tested against dev/infra false positives.
- **Stage 2: a small named-entity (NER) model** on CPU for person names, addresses and
  organisations (e.g. a GLiNER-style PII model or spaCy/Presidio). Needs its own
  evaluation set before it is trusted.

## 6. Request flow (Masked mode)

```
request ─▶ extract units (user / tool / assistant, as today)
        ─▶ find spans (detectors; stage 2: + NER)
        ─▶ mint placeholders, build per-request map  placeholder → value
        ─▶ masked request
        ─▶ re-check gate: detectors + Laya on the MASKED text
              ├─ still personal ─▶ LOCAL (unmasked original), reason "residual_pii"
              └─ clean ──────────▶ NVIDIA (masked), plus a short system note (§6.4)
        ─▶ response ─▶ unmask content + tool-call arguments ─▶ agent
```

### 6.1 Placeholders: stable, stateless

`<TYPE_xxxxxxxx>` where `xxxxxxxx` = first 8 hex chars of
`HMAC-SHA256(deployment_secret, normalised_value)`.

- **Stable across turns** without storing anything: the same email always gets the same
  placeholder, so the cloud model sees a consistent history.
- **Reversible without a database:** the agent resends the full history every turn, and the
  history holds the real values (the agent stored the unmasked replies), so each request's
  own spans rebuild the map. A placeholder in the reply that is not in the request's map
  is left as-is (visible, never a leak).
- 8 hex chars per type; on a collision within one request, extend to 12.
- `deployment_secret` lives in `/etc/pii-router/pii-router.env`, generated once.

### 6.2 Escaping placeholder look-alikes

Text that already contains something shaped like `<EMAIL_…>` (from a user, a web page, a
tool result) must be neutralised before masking (e.g. `<EMAIL_` → `‹EMAIL_`), so only
router-minted placeholders are ever reversed. Otherwise prompt-injected text could make
the router splice a real value into a tool call it chose.

### 6.3 Re-check gate (keeps fail-closed)

After masking, run the detectors (must be clean by construction) and **Laya** on the masked
text with the normal threshold. A positive result sends the original request **local**.
This is what keeps "sick leave for depression"-type content local even when the name is
masked.

### 6.4 Telling the cloud model about placeholders

Append one short system message: placeholders like `<EMAIL_7f3a2c1b>` stand for real
values; reproduce them exactly, never invent new ones, and use them in tool arguments.

## 7. Unmasking the reply

- **Non-streaming:** replace placeholders in `message.content` and in every
  `tool_calls[].function.arguments` (a JSON string — replace inside string values, then
  re-validate it parses).
- **Streaming:** a placeholder can be split across chunks. Hold back text from a `<` until
  the closing `>` or 32 characters, then flush it, substituted or verbatim. Tool-call
  argument deltas get the same treatment per tool-call index.
- **Marker:** `‡` appended to masked-cloud text replies (same rules as `†`: never on
  tool-call-only replies; configurable, empty disables).

## 8. Stickiness and caching

- Masked mode replaces "sticky local" with "sticky masked": every turn passes through the
  masker; only `residual_pii` (or any failure) makes a turn local.
- The verdict cache keys on the **masked** text, so Laya runs once per new message as today.
- Once a turn of a conversation falls back to local for `residual_pii`, that message's
  cached verdict keeps the conversation local (same as Strict).

## 9. Privacy and legal position

- Under GDPR, data masked with a reversible key that exists (on the box) is
  **pseudonymised, still personal data** — it reduces exposure; it is not anonymisation.
  Hence opt-in, never the default, and the UI must say so.
- Quasi-identifiers (employer + town + condition) can identify someone without a name.
  The Laya re-check catches some of this; it is not a guarantee.
- The cloud model still sees the *task and context*, just not the masked values.

## 10. Failure handling (all fail closed)

| Failure | Result |
|---|---|
| Span detection or masking throws | local, reason `mask_error` |
| Re-check gate positive / Laya down | local, reason `residual_pii` / `laya_unavailable` |
| Masked request > chunk budget | local, reason `oversize_unscanned` |
| Reply contains unknown placeholder | passed through verbatim (no leak) |
| Tool-call arguments no longer parse after unmasking | retry the turn locally (never hand the agent a broken or still-masked tool call), reason `unmask_error` |
| Images / files | local (cannot be masked), as today |

Decision log adds `mode`, `masked: {"email": 2, "iban": 1}` (counts only) and the
fallback reason. Never values or placeholders.

## 11. Open questions

1. **Agent memory and the system prompt.** The router does not scan system prompts.
   If OpenClaw/Hermes re-inject remembered PII there (`memory-core`, `MEMORY.md`), Masked
   *and* Strict would send it to the cloud. Settle with the controlled memory test (write a
   test `MEMORY.md`, new session, check `system_hits`) before shipping Masked mode; Masked
   mode would likely need to mask system prompts too.
2. Does NemoClaw/OpenClaw rely on response fields we'd alter (usage, logprobs, citations)?
3. Should Masked mode be per-deployment only, or switchable per sandbox from the controller?
4. Do agents' tool schemas with format validation (e.g. `format: email`) see placeholders
   before unmasking? (They should not — unmasking happens before the agent parses.)

## 12. Testing plan

- Unit: span detectors; placeholder stability across turns; look-alike escaping; unmasking
  of content, tool-call JSON, and streamed text with placeholders split at every offset.
- **Leak test (the key one):** e2e with the fake NVIDIA backend recording every request
  body; assert no detector fires on anything the cloud received, for a labelled set of
  multi-turn conversations.
- Fallback tests: semantic PII still goes local; Laya down → local; malformed tool JSON →
  local retry.
- Live: OpenClaw + Hermes, including a tool that uses a masked value (send email to a test
  address), on a fresh hcloud box.

## 13. Staging and effort

1. **Stage 1 — structured identifiers only** (detectors → spans, placeholders, unmasking
   incl. streaming and tool calls, re-check gate, `‡`, manidae three-way mode, tests).
   Roughly 2–3 days. Moves "email/IBAN/phone"-type conversations back to the cloud; names
   and health details still go local.
2. **Stage 2 — names, addresses, organisations** via a CPU NER model, with its own labelled
   evaluation set and threshold tuning. Roughly 1–2 days plus evaluation.

Prerequisite for both: answer open question 1 (memory / system prompt).
