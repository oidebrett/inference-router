"""PII masking proxy for the Claude apps gateway (Anthropic Messages API).

The gateway's upstream points here (`base_url`); this service masks personal data in
each request before it goes to Anthropic and puts the real values back in the reply.
It reuses the inference router's masking unchanged: pii_router.mask (placeholders,
restoring, streaming) and inference_router.detection (built-in + custom regex rules).

    gateway --(x-api-key: MASKER_KEY)--> masker --(x-api-key: ANTHROPIC_API_KEY)--> Anthropic

Rules come from one admin-edited JSON file (PII_RULES_FILE), re-read when it changes:

    {"mode": "mask", "rules": {"builtin": {...}, "custom": [...], "exceptions": [...]}}

`rules` uses the inference router's privacy.rules schema. An invalid file is refused
as a whole and the last good rules stay active.

Anthropic specifics:
  * thinking / redacted_thinking blocks are signed: passed through byte for byte,
    never masked or restored;
  * tool_use inputs are masked in history and restored in replies, including the
    streamed input_json_delta, so tools act on the real values;
  * count_tokens is masked like messages, so token counts match what is sent.

Nothing logs a value or message text: one line per request with the user, the model
and how many values of each type were masked.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import time

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from inference_router.config import ConfigError, _parse_rules
from inference_router.detection import DetectionTimeout, Detector
from pii_router.mask import SYSTEM_NOTE, MaskSession, StreamRestorer, restore_text

log = logging.getLogger("anthropic-masker")

SIGNED_BLOCKS = {"thinking", "redacted_thinking"}
MEDIA_BLOCKS = {"image", "document"}
REQUEST_HEADERS = {"anthropic-version", "anthropic-beta", "content-type", "accept"}
RESPONSE_HEADER_PREFIXES = ("anthropic-", "request-id", "retry-after", "x-should-retry")


def _error(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse({"type": "error", "error": {"type": kind, "message": message}}, status_code=status)


# --- rules file ------------------------------------------------------------------

class RulesFile:
    """The admin's JSON rules, reloaded when the file's mtime changes."""

    def __init__(self, path: str):
        self.path = path
        self.mtime = None
        self.mode = "off"
        self.detector: Detector | None = None
        self.refresh(first=True)

    def refresh(self, first: bool = False) -> None:
        try:
            mtime = os.stat(self.path).st_mtime
        except OSError as error:
            if first:
                raise
            log.error("pii rules file unreadable, keeping previous rules: %s", error)
            return
        if mtime == self.mtime:
            return
        self.mtime = mtime
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            mode, detector = parse_rules(data)
        except (ConfigError, ValueError, OSError) as error:
            if first:
                raise
            log.error("pii rules file refused, keeping previous rules: %s", error)
            return
        self.mode, self.detector = mode, detector
        log.info("pii rules loaded from %s: mode=%s", self.path, mode)


def parse_rules(data) -> tuple[str, Detector]:
    if not isinstance(data, dict):
        raise ConfigError("the rules file must be a JSON object")
    unknown = set(data) - {"mode", "rules"}
    if unknown:
        raise ConfigError(f"unknown keys: {', '.join(sorted(unknown))}")
    mode = data.get("mode", "mask")
    if mode not in ("mask", "off"):
        raise ConfigError("mode must be 'mask' or 'off'")
    return mode, Detector(_parse_rules(data.get("rules") or {}))


# --- request side ------------------------------------------------------------------

class RequestMasker:
    """Masks one Anthropic Messages request body with one shared placeholder map."""

    def __init__(self, session: MaskSession):
        self.session = session
        self.media = 0   # image/document blocks sent unmasked

    def value(self, value):
        if isinstance(value, str):
            return self.session.mask_text(value)
        if isinstance(value, list):
            return [self.value(v) for v in value]
        if isinstance(value, dict):
            return {k: self.value(v) for k, v in value.items()}
        return value

    def blocks(self, content):
        if isinstance(content, str):
            return self.session.mask_text(content)
        if not isinstance(content, list):
            return content
        out = []
        for block in content:
            kind = block.get("type") if isinstance(block, dict) else None
            if kind == "text" and isinstance(block.get("text"), str):
                block = {**block, "text": self.session.mask_text(block["text"])}
            elif kind == "tool_use":
                block = {**block, "input": self.value(block.get("input"))}
            elif kind == "tool_result":
                block = {**block, "content": self.blocks(block.get("content"))}
            elif kind in MEDIA_BLOCKS:
                self.media += 1
            # SIGNED_BLOCKS and anything unknown pass through untouched.
            out.append(block)
        return out

    def body(self, body: dict) -> dict:
        body = dict(body)
        if "system" in body:
            body["system"] = self.blocks(body["system"])
        body["messages"] = [{**m, "content": self.blocks(m.get("content"))} if isinstance(m, dict) else m
                            for m in body.get("messages") or []]
        if self.session.mapping:
            body["system"] = with_note(body.get("system"))
        return body


def with_note(system):
    """The system prompt with the placeholder note appended (fixed text, cache friendly)."""
    if not system:
        return SYSTEM_NOTE
    if isinstance(system, str):
        return system.rstrip() + "\n\n" + SYSTEM_NOTE
    return list(system) + [{"type": "text", "text": SYSTEM_NOTE}]


# --- reply side ----------------------------------------------------------------------

def restore_value(value, mapping):
    if isinstance(value, str):
        return restore_text(value, mapping)
    if isinstance(value, list):
        return [restore_value(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: restore_value(v, mapping) for k, v in value.items()}
    return value


def restore_message(message: dict, mapping: dict) -> dict:
    content = []
    for block in message.get("content") or []:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text":
            block = {**block, "text": restore_text(block.get("text") or "", mapping)}
        elif kind == "tool_use":
            block = {**block, "input": restore_value(block.get("input"), mapping)}
        content.append(block)
    return {**message, "content": content}


class StreamRestore:
    """Restores placeholders in a stream of Anthropic SSE events.

    One restorer per content block: text_delta text as plain text, input_json_delta
    partial_json as JSON-escaped text. Text held back at a block's end (an unclosed
    `<`) is flushed as one extra delta just before that block's content_block_stop.
    """

    def __init__(self, mapping: dict):
        self.mapping = mapping
        self.restorers: dict[int, tuple[str, StreamRestorer]] = {}

    def event(self, data: dict) -> list[dict]:
        kind = data.get("type")
        index = data.get("index")
        if kind == "content_block_delta":
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                text = self._feed(index, "text_delta", delta.get("text") or "", json_escape=False)
                return [{**data, "delta": {**delta, "text": text}}] if text else []
            if delta.get("type") == "input_json_delta":
                text = self._feed(index, "input_json_delta", delta.get("partial_json") or "", json_escape=True)
                return [{**data, "delta": {**delta, "partial_json": text}}] if text else []
            return [data]
        if kind == "content_block_stop" and index in self.restorers:
            delta_type, restorer = self.restorers.pop(index)
            rest = restorer.flush()
            if rest:
                field = "text" if delta_type == "text_delta" else "partial_json"
                return [{"type": "content_block_delta", "index": index,
                         "delta": {"type": delta_type, field: rest}}, data]
        return [data]

    def _feed(self, index, delta_type, text, json_escape):
        if index not in self.restorers:
            self.restorers[index] = (delta_type, StreamRestorer(self.mapping, json_escape))
        return self.restorers[index][1].feed(text)


def sse(data: dict) -> bytes:
    return f"event: {data.get('type')}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


# --- the service -----------------------------------------------------------------------

class Masker:
    def __init__(self, env=os.environ):
        self.masker_key = env.get("MASKER_KEY", "")
        self.anthropic_key = env.get("ANTHROPIC_API_KEY", "")
        if not self.masker_key or not self.anthropic_key:
            raise SystemExit("MASKER_KEY and ANTHROPIC_API_KEY must both be set")
        secret = env.get("MASK_SECRET", "")
        if not secret:
            log.warning("MASK_SECRET is not set; placeholders change after a restart")
        self.secret = secret.encode() if secret else secrets.token_bytes(32)
        self.upstream = env.get("MASKER_UPSTREAM", "https://api.anthropic.com").rstrip("/")
        self.rules = RulesFile(env.get("PII_RULES_FILE", "/etc/pii-masker/pii-rules.json"))
        self.dump_dir = env.get("MASKER_DUMP_DIR", "")   # spike only: write masked bodies here
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    def authorised(self, request: Request) -> bool:
        given = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ")
        return bool(given) and hmac.compare_digest(given.encode(), self.masker_key.encode())

    async def handle(self, request: Request) -> Response:
        started = time.monotonic()
        if not self.authorised(request):
            return _error(401, "authentication_error", "invalid x-api-key for the masking proxy")
        try:
            body = json.loads(await request.body())
        except ValueError:
            return _error(400, "invalid_request_error", "request body is not JSON")
        if not isinstance(body, dict):
            return _error(400, "invalid_request_error", "request body must be a JSON object")

        self.rules.refresh()
        session = MaskSession(self.secret, finder=self.rules.detector.find_spans)
        masker = RequestMasker(session)
        if self.rules.mode == "mask":
            try:
                body = masker.body(body)
            except DetectionTimeout:
                self._log(request, body, session, masker, 400, started)
                return _error(400, "invalid_request_error",
                              "a PII rule took too long on this request; it was not sent. Ask the admin to fix the rule.")
        self._dump(request, body)

        headers = {k: v for k, v in request.headers.items() if k.lower() in REQUEST_HEADERS}
        headers["x-api-key"] = self.anthropic_key
        url = self.upstream + request.url.path + (f"?{request.url.query}" if request.url.query else "")
        upstream = await self.client.send(
            self.client.build_request("POST", url, headers=headers, content=json.dumps(body).encode()), stream=True)
        out_headers = {k: v for k, v in upstream.headers.items()
                       if k.lower().startswith(RESPONSE_HEADER_PREFIXES)}
        mapping = session.mapping
        streaming = "text/event-stream" in upstream.headers.get("content-type", "")

        if not streaming:
            raw = await upstream.aread()
            await upstream.aclose()
            self._log(request, body, session, masker, upstream.status_code, started)
            if mapping and upstream.status_code == 200 and request.url.path == "/v1/messages":
                try:
                    raw = json.dumps(restore_message(json.loads(raw), mapping)).encode()
                except ValueError:
                    pass
            return Response(raw, status_code=upstream.status_code, headers=out_headers,
                            media_type=upstream.headers.get("content-type", "application/json"))

        async def events():
            restore = StreamRestore(mapping)
            buffer = b""
            try:
                async for chunk in upstream.aiter_bytes():
                    buffer += chunk
                    while b"\n\n" in buffer:
                        raw_event, buffer = buffer.split(b"\n\n", 1)
                        data_lines = [line[5:].strip() for line in raw_event.split(b"\n") if line.startswith(b"data:")]
                        if not data_lines:
                            continue
                        try:
                            data = json.loads(b"\n".join(data_lines))
                        except ValueError:
                            yield raw_event + b"\n\n"
                            continue
                        for out in (restore.event(data) if mapping else [data]):
                            yield sse(out)
            finally:
                await upstream.aclose()
                self._log(request, body, session, masker, upstream.status_code, started)

        return StreamingResponse(events(), status_code=upstream.status_code, headers=out_headers,
                                 media_type="text/event-stream")

    def _log(self, request, body, session, masker, status, started):
        log.info(json.dumps({
            "evt": "masked_request", "path": request.url.path,
            "user": request.headers.get("x-claude-gateway-user-email") or request.headers.get("x-claude-gateway-user-id"),
            "model": body.get("model"), "stream": bool(body.get("stream")), "mode": self.rules.mode,
            "masked": session.counts, "unmasked_media": masker.media, "status": status,
            "ms": round((time.monotonic() - started) * 1000)}))

    def _dump(self, request, body):
        if not self.dump_dir:
            return
        name = f"{int(time.time() * 1000)}{request.url.path.replace('/', '_')}.json"
        with open(os.path.join(self.dump_dir, name), "w", encoding="utf-8") as fh:
            json.dump(body, fh, ensure_ascii=False, indent=1)


def create_app(env=os.environ) -> Starlette:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    masker = Masker(env)

    async def health(_request):
        return JSONResponse({"ok": True, "mode": masker.rules.mode})

    async def not_found(_request):
        return _error(404, "not_found_error", "only /v1/messages and /v1/messages/count_tokens are proxied")

    return Starlette(routes=[
        Route("/v1/messages", masker.handle, methods=["POST"]),
        Route("/v1/messages/count_tokens", masker.handle, methods=["POST"]),
        Route("/healthz", health, methods=["GET"]),
        Route("/{path:path}", not_found, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
    ])


def app_factory() -> Starlette:   # uvicorn --factory anthropic_masker.app:app_factory
    return create_app()
