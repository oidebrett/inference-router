"""LiteLLM proxy hook: pick the cloud, fallback or local deployment for each request.

Wired in config.yaml as ``litellm_settings.callbacks: pii_router.hook.proxy_handler_instance``.

The public model name (``pii-router``) is itself configured to the LOCAL
deployment, so if this hook is ever not loaded or raises, traffic still stays
local. The hook decides for EVERY chat request, whatever model name the client
asked for, so a caller cannot bypass the check by naming a cloud deployment.

Clean traffic goes to the cloud backend. When a cloud health URL is configured
(PrivateInference: the customer's Vast GPU) and the cloud is down, clean traffic
goes to the fallback backend (NVIDIA) if there is one, otherwise local. PII
always goes local.

Every reply says where it was answered and, for local, why:
  headers  x-pii-route: local|cloud|fallback   x-pii-reason: current-turn|history|fail-closed|clean
  markers  " †" PII in this turn, " ††" PII only in earlier messages (a fresh
           context would go back to the cloud), " †!" could not be cleared.
The router strips its own markers from assistant history before deciding and
forwarding, because clients re-send the whole conversation every turn.

PII_ROUTER_MODE=off (PrivateInference without PII routing): nothing is scanned
and there is no local model; every request goes to the cloud backend, or the
fallback while the cloud is down (x-pii-reason: unchecked, no markers).

PII_ROUTER_MODE=masked: detected values are replaced with placeholders
(pii_router/mask.py), the masked request goes to the cloud, and the real values
are put back into the reply, tool-call arguments included. Personal content that
masking cannot cover (Laya still says yes on the masked text) is BLOCKED with an
explanatory reply (PII_ROUTER_RESIDUAL_ACTION=block, the default; no local model
needed) or sent unmasked to the local model (=local).
  headers  x-pii-route: cloud|fallback|blocked|local
           x-pii-reason: masked|clean|current-turn|history|fail-closed
           x-pii-masked: number of distinct values masked
  marker   " ‡" on replies to masked requests (PII_ROUTER_MASKED_MARKER).
PII_ROUTER_MODE=on is the same as strict (the original behaviour above).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from collections import OrderedDict

from litellm.integrations.custom_logger import CustomLogger

from .health import CloudHealth
from .laya_client import LayaHttpScorer
from .mask import MaskSession, StreamRestorer, add_system_note, restore_json_text, restore_text
from .policy import CLOUD, CURRENT_TURN, FAIL_CLOSED, HISTORY, LOCAL, Router, Settings

log = logging.getLogger("pii_router")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False

CHAT_CALL_TYPES = {"completion", "acompletion"}
FALLBACK = "fallback"
CLEAN = "clean"
UNCHECKED = "unchecked"  # PII_ROUTER_MODE=off: not scanned
LEGACY_CLOUD_MODEL = "nvidia-cloud"  # deployment name before 2026-10-22
BLOCKED = "blocked"
MASKED = "masked"
STRICT = "strict"
MODE_OFF = "off"

BLOCK_MESSAGES = {
    CURRENT_TURN: (
        "[PII router] This message was not sent to the AI model. It contains personal "
        "information that could not be hidden automatically, for example a person described "
        "by name together with health, money or work details. Please rephrase it without "
        "identifying the person."),
    HISTORY: (
        "[PII router] This conversation was not sent to the AI model. An earlier message "
        "contains personal information that could not be hidden automatically. Please start "
        "a new conversation to continue."),
    FAIL_CLOSED: (
        "[PII router] This request was not sent to the AI model because it could not be "
        "checked for personal information (the checker is unavailable, the request is too "
        "large, or it contains an image or file it cannot read). Please try again shortly."),
}
BLOCK_MESSAGES["cloud_unavailable"] = (
    "[PII router] The AI model is not reachable right now, so this request was not sent. "
    "Please try again shortly.")


def _mode(env) -> str:
    raw = (env.get("PII_ROUTER_MODE") or "on").strip().lower()
    if raw == MODE_OFF:
        return MODE_OFF
    if raw == MASKED:
        return MASKED
    return STRICT  # "on", "strict" and anything unknown keep the original behaviour


class PiiRouterHook(CustomLogger):
    def __init__(self, router: Router | None = None, env=os.environ, health: CloudHealth | None = None):
        super().__init__()
        self.cloud_model = env.get("PII_ROUTER_CLOUD_MODEL_NAME", "cloud")
        self.local_model = env.get("PII_ROUTER_LOCAL_MODEL_NAME", "ollama-local")
        self.fallback_model = (env.get("PII_ROUTER_FALLBACK_MODEL_NAME", "cloud-fallback")
                               if (env.get("FALLBACK_BASE_URL") or "").strip() else None)
        self.remote_models = {m for m in (self.cloud_model, self.fallback_model, LEGACY_CLOUD_MODEL,
                                          "cloud-fallback-2") if m}
        self.mode = _mode(env)
        self.pii_off = self.mode == MODE_OFF
        self.blocked_model = env.get("PII_ROUTER_BLOCKED_MODEL_NAME", "pii-blocked")
        action = (env.get("PII_ROUTER_RESIDUAL_ACTION") or "block").strip().lower()
        self.residual_action = LOCAL if action == LOCAL else BLOCKED
        secret = (env.get("PII_ROUTER_MASK_SECRET") or "").strip()
        if self.mode == MASKED and not secret:
            # Still safe, but placeholders change after every restart.
            log.warning("pii-router: PII_ROUTER_MASK_SECRET is not set; using a per-process secret")
        self.mask_secret = secret.encode("utf-8") if secret else secrets.token_bytes(32)
        # placeholder maps of requests in flight: id -> (created, mapping). Never logged.
        self._maps: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.router = router or Router(LayaHttpScorer.from_env(env), Settings.from_env(env))
        self.health = health if health is not None else CloudHealth.from_env(env)
        # Appended to replies from the LOCAL model so a user or agent can see where an
        # answer came from and why. PII_ROUTER_LOCAL_MARKER= (empty) disables all
        # markers, e.g. when clients expect strict JSON; the headers stay.
        base = env.get("PII_ROUTER_LOCAL_MARKER", " †")
        self.markers = {
            CURRENT_TURN: base,
            HISTORY: env.get("PII_ROUTER_HISTORY_MARKER", " ††") if base else "",
            FAIL_CLOSED: env.get("PII_ROUTER_FAILCLOSED_MARKER", " †!") if base else "",
        }
        self.masked_marker = env.get("PII_ROUTER_MASKED_MARKER", " ‡") if base else ""
        # Longest first, so " ††" is stripped whole rather than as " †" + "†".
        self._strip = sorted({m for m in [*self.markers.values(), self.masked_marker] if m},
                             key=len, reverse=True)

    # --- helpers ---------------------------------------------------------------

    @staticmethod
    def _meta(data) -> dict:
        return ((data or {}).get("metadata") or {}) if isinstance((data or {}).get("metadata"), dict) else {}

    def _route(self, data) -> str | None:
        route = self._meta(data).get("pii_router_route")
        if route is None and (data or {}).get("model") == self.local_model:
            return LOCAL
        return route

    def _marker(self, data) -> str:
        if self._route(data) != LOCAL:
            return ""
        return self.markers.get(self._meta(data).get("pii_router_reason") or FAIL_CLOSED, "")

    def _strip_text(self, text: str) -> str:
        for marker in self._strip:
            if text.endswith(marker):
                return text[: -len(marker)]
        return text

    def _strip_markers(self, messages) -> None:
        for message in messages or []:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = self._strip_text(content)
            elif isinstance(content, list):
                for part in reversed(content):
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        part["text"] = self._strip_text(part["text"])
                        break

    def _log(self, fields: dict) -> None:
        log.info("pii-router-decision " + json.dumps({"ts": round(time.time(), 3), **fields}, sort_keys=True))

    def _set_meta(self, data, route: str, reason: str) -> None:
        if isinstance(data.get("metadata"), dict) or data.get("metadata") is None:
            data.setdefault("metadata", {})
            data["metadata"]["pii_router_route"] = route
            data["metadata"]["pii_router_reason"] = reason

    # --- masked mode -----------------------------------------------------------

    MAP_TTL = 900.0
    MAP_MAX = 2000

    def _keep_map(self, data, mapping: dict) -> None:
        now = time.time()
        while self._maps and (len(self._maps) >= self.MAP_MAX
                              or now - next(iter(self._maps.values()))[0] > self.MAP_TTL):
            self._maps.popitem(last=False)
        map_id = secrets.token_hex(12)
        self._maps[map_id] = (now, mapping)
        data["metadata"]["pii_router_map"] = map_id

    def _peek_map(self, data) -> dict:
        entry = self._maps.get(self._meta(data).get("pii_router_map") or "")
        return entry[1] if entry else {}

    def _drop_map(self, data) -> None:
        self._maps.pop(self._meta(data).get("pii_router_map") or "", None)

    def _block(self, data, reason: str, fields: dict) -> dict:
        """Answer with an explanation instead of calling any model. The request's
        messages are replaced, so nothing personal can be forwarded by mistake."""
        data["model"] = self.blocked_model
        data["mock_response"] = BLOCK_MESSAGES.get(reason, BLOCK_MESSAGES[FAIL_CLOSED])
        data["messages"] = [{"role": "user", "content": "blocked by the PII router"}]
        for key in ("tools", "tool_choice", "functions", "function_call", "response_format"):
            data.pop(key, None)
        self._set_meta(data, BLOCKED, reason)
        fields.update(route=BLOCKED, block_reason=reason, target=data["model"])
        self._log(fields)
        return data

    async def _route_masked(self, data, call_type, requested):
        fields: dict = {"mode": MASKED, "requested": requested}
        if call_type not in CHAT_CALL_TYPES:
            fields["reasons"] = [f"unsupported_call_type:{call_type}"]
            return self._block(data, FAIL_CLOSED, fields)
        try:
            self._strip_markers(data.get("messages"))
            session = MaskSession(self.mask_secret)
            masked = session.mask_messages(data.get("messages") or [])
            decision = await asyncio.to_thread(self.router.decide, masked)
            fields.update(decision.log_fields())
        except Exception as error:  # noqa: BLE001 — fail closed on any bug
            fields.update(reasons=["router_error"], error=type(error).__name__)
            return self._block(data, FAIL_CLOSED, fields)
        fields.update(masked_types=dict(sorted(session.counts.items())), masked_values=len(session.mapping))
        if fields["route"] == LOCAL:
            # Personal content left after masking (Laya), opaque media, or not checkable.
            reason = fields.get("local_reason") or FAIL_CLOSED
            if self.residual_action == LOCAL:
                data["model"] = self.local_model   # the ORIGINAL messages, unmasked, stay on the box
                self._set_meta(data, LOCAL, reason)
                fields.update(route=LOCAL, target=data["model"])
                self._log(fields)
                return data
            return self._block(data, reason, fields)
        route = CLOUD
        if self.health is not None and not await asyncio.to_thread(self.health.healthy):
            if not self.fallback_model:
                fields["reasons"] = list(fields.get("reasons") or []) + ["cloud_unavailable"]
                return self._block(data, "cloud_unavailable", fields)
            route = FALLBACK
        data["messages"] = add_system_note(masked) if session.mapping else masked
        data["model"] = self.cloud_model if route == CLOUD else self.fallback_model
        self._set_meta(data, route, MASKED if session.mapping else CLEAN)
        if session.mapping:
            self._keep_map(data, session.mapping)
            data["metadata"]["pii_router_masked"] = len(session.mapping)
        fields.update(route=route, local_reason=None, target=data["model"])
        self._log(fields)
        return data

    def _restore_message(self, message, mapping: dict) -> None:
        for attr in ("content", "reasoning_content"):
            value = getattr(message, attr, None)
            if isinstance(value, str):
                setattr(message, attr, restore_text(value, mapping))
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            if function is not None and isinstance(getattr(function, "arguments", None), str):
                function.arguments = restore_json_text(function.arguments, mapping)

    # --- LiteLLM hooks ---------------------------------------------------------

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        requested = data.get("model")
        if self.pii_off:
            return await self._route_unchecked(data, call_type, requested)
        if self.mode == MASKED:
            return await self._route_masked(data, call_type, requested)
        if call_type not in CHAT_CALL_TYPES:
            # Only chat completions are inspected. Anything else that targets our
            # deployments is pinned local rather than passed through unchecked.
            if requested in self.remote_models:
                data["model"] = self.local_model
            self._set_meta(data, LOCAL, FAIL_CLOSED)
            self._log({"route": LOCAL, "reasons": [f"unsupported_call_type:{call_type}"], "requested": requested})
            return data
        try:
            self._strip_markers(data.get("messages"))
            decision = await asyncio.to_thread(self.router.decide, data.get("messages") or [])
            fields = decision.log_fields()
        except Exception as error:  # noqa: BLE001 — fail closed on any bug
            fields = {"route": LOCAL, "reasons": ["router_error"], "error": type(error).__name__,
                      "local_reason": FAIL_CLOSED}
        route, reason = fields["route"], fields.get("local_reason") or CLEAN
        if route == CLOUD and self.health is not None and not await asyncio.to_thread(self.health.healthy):
            if self.fallback_model:
                route = FALLBACK
            else:
                route, reason = LOCAL, FAIL_CLOSED
                fields["reasons"] = list(fields.get("reasons") or []) + ["cloud_unavailable"]
        data["model"] = {CLOUD: self.cloud_model, FALLBACK: self.fallback_model}.get(route, self.local_model)
        self._set_meta(data, route, reason)
        fields.update(route=route, local_reason=reason if route == LOCAL else None,
                      requested=requested, target=data["model"])
        self._log(fields)
        return data

    async def _route_unchecked(self, data, call_type, requested):
        """PII routing off: cloud, or the fallback while the cloud is down. Never local."""
        route = CLOUD
        if self.health is not None and self.fallback_model and not await asyncio.to_thread(self.health.healthy):
            route = FALLBACK
        data["model"] = self.cloud_model if route == CLOUD else self.fallback_model
        self._set_meta(data, route, UNCHECKED)
        self._log({"route": route, "reasons": ["pii_routing_off"], "call_type": call_type,
                   "requested": requested, "target": data["model"]})
        return data

    async def async_post_call_failure_hook(self, request_data, original_exception, user_api_key_dict,
                                           traceback_str=None):
        # A failed cloud call marks the cloud down so the NEXT request is re-routed.
        # This request is not retried elsewhere (num_retries: 0).
        if self.health is not None and self._route(request_data) == CLOUD:
            self.health.mark_down()
            self._log({"event": "cloud_call_failed", "error": type(original_exception).__name__})
        self._drop_map(request_data)
        return None

    async def async_post_call_response_headers_hook(self, data, user_api_key_dict, response,
                                                    request_headers=None, litellm_call_info=None):
        route = self._route(data)
        if route is None:
            return None
        reason = self._meta(data).get("pii_router_reason") or CLEAN
        if route == BLOCKED or reason == MASKED:
            out = {"x-pii-route": route, "x-pii-reason": reason}
            if reason == MASKED:
                out["x-pii-masked"] = str(self._meta(data).get("pii_router_masked") or 0)
            return out
        if self.mode == MASKED and route == LOCAL:
            return {"x-pii-route": LOCAL, "x-pii-reason": reason}
        if reason == UNCHECKED:
            return {"x-pii-route": route, "x-pii-reason": UNCHECKED}
        return {"x-pii-route": route, "x-pii-reason": reason if route == LOCAL else CLEAN}

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        mapping = self._peek_map(data)
        if mapping:
            try:
                for choice in response.choices:
                    self._restore_message(choice.message, mapping)
                message = response.choices[0].message
                if self.masked_marker and isinstance(message.content, str) and message.content.strip():
                    message.content = message.content + self.masked_marker
            except (AttributeError, IndexError, TypeError):
                pass
            finally:
                self._drop_map(data)
            return response
        marker = self._marker(data)
        if not marker:
            return response
        try:
            message = response.choices[0].message
            if isinstance(message.content, str) and message.content.strip():
                message.content = message.content + marker
        except (AttributeError, IndexError, TypeError):
            pass  # not a chat completion; leave it alone
        return response

    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        mapping = self._peek_map(request_data)
        if mapping:
            try:
                async for chunk in self._restore_stream(response, mapping):
                    yield chunk
            finally:
                self._drop_map(request_data)
            return
        marker = self._marker(request_data)
        marked = not marker
        saw_text = False
        async for chunk in response:
            if not marked:
                try:
                    choice = chunk.choices[0]
                    delta_text = getattr(choice.delta, "content", None)
                    if isinstance(delta_text, str) and delta_text.strip():
                        saw_text = True
                    if choice.finish_reason and saw_text:
                        choice.delta.content = (delta_text or "") + marker
                        marked = True
                except (AttributeError, IndexError, TypeError):
                    pass
            yield chunk


    async def _restore_stream(self, response, mapping: dict):
        """Restore placeholders in streamed content and tool-call argument deltas.
        Text that may be the start of a split placeholder is held back until it closes."""
        content: dict[int, StreamRestorer] = {}
        args: dict[tuple[int, int], StreamRestorer] = {}
        saw_text: dict[int, bool] = {}
        async for chunk in response:
            try:
                for choice in chunk.choices:
                    i = getattr(choice, "index", 0) or 0
                    delta = choice.delta
                    text = getattr(delta, "content", None)
                    if isinstance(text, str) and text:
                        out = content.setdefault(i, StreamRestorer(mapping)).feed(text)
                        delta.content = out
                        saw_text[i] = saw_text.get(i, False) or bool(text.strip())
                    for call in getattr(delta, "tool_calls", None) or []:
                        function = getattr(call, "function", None)
                        piece = getattr(function, "arguments", None) if function is not None else None
                        if isinstance(piece, str) and piece:
                            key = (i, getattr(call, "index", 0) or 0)
                            function.arguments = args.setdefault(key, StreamRestorer(mapping, True)).feed(piece)
                    if choice.finish_reason:
                        rest = content.pop(i).flush() if i in content else ""
                        tail = self.masked_marker if (saw_text.get(i) and self.masked_marker) else ""
                        if rest or tail:
                            delta.content = (getattr(delta, "content", None) or "") + rest + tail
                        for key in [k for k in args if k[0] == i]:
                            left = args.pop(key).flush()
                            if left:
                                self._append_tool_args(delta, key[1], left)
            except (AttributeError, IndexError, TypeError):
                pass
            yield chunk

    @staticmethod
    def _append_tool_args(delta, index: int, text: str) -> None:
        for call in getattr(delta, "tool_calls", None) or []:
            if (getattr(call, "index", 0) or 0) == index and getattr(call, "function", None) is not None:
                call.function.arguments = (call.function.arguments or "") + text
                return
        from litellm.types.utils import ChatCompletionDeltaToolCall, Function
        extra = ChatCompletionDeltaToolCall(index=index, function=Function(arguments=text))
        delta.tool_calls = list(getattr(delta, "tool_calls", None) or []) + [extra]


proxy_handler_instance = PiiRouterHook()
