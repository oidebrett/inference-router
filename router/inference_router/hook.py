"""LiteLLM proxy hook for the Inference Router.

Wired by render.py as ``litellm_settings.callbacks: inference_router.hook.proxy_handler_instance``.

For every chat request:
  1. the requested model must be a route name (router.yaml); anything else is refused;
  2. privacy (router.yaml privacy.mode):
       off           request passed on unchanged
       mask          detected values -> placeholders; refused only if it cannot be checked
                     (opaque media, oversize, a rule that timed out)
       mask_check    mask, then Laya re-checks the masked text; still personal -> refused
       keep_private  personal requests go UNMASKED to the route's private upstreams only
                     (refused if the route has none); the rest go to any upstream
  3. the route's fallback chain is chosen (render.py), so LiteLLM tries targets in order.
Replies to masked requests get the real values back (text, tool-call arguments, streams).

Refusals never call an upstream: they are answered by LiteLLM's mock_response with a
"[Inference Router] ..." explanation, and the request's messages are replaced so
nothing personal can be forwarded by mistake.

No message content is logged or stored. Each request produces one metadata event
(route, decision, reason, types/counts masked, timings) for the admin service's live
feed, held there in memory only.

Headers: x-router-route, x-pii-route (upstream|private|blocked), x-pii-reason,
x-pii-masked. Markers (configurable, empty disables): " ‡" masked, " †" answered privately.

Personal subscriptions (the admin's Claude or ChatGPT plan, see subscriptions.py) are only
in a route's chain for the admin key and keys allowed to use them (ir_subscriptions);
everyone else gets the route's shared chain.
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

from pii_router.extract import Unit  # noqa: F401  (re-exported for tests)
from pii_router.laya_client import LayaHttpScorer
from pii_router.mask import (MaskSession, StreamRestorer, add_system_note, for_residual_check,
                             restore_json_text, restore_text)
from pii_router.policy import CURRENT_TURN, FAIL_CLOSED, HISTORY, LOCAL, Router, Settings

from .config import Config, load
from .detection import DetectionTimeout, Detector
from .keys import KeyStore
from .render import BLOCKED, chains
from .subscriptions import is_claude_token, never_prompt_for_chatgpt_login, with_claude_identity

never_prompt_for_chatgpt_login()

log = logging.getLogger("inference_router")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

CHAT_CALL_TYPES = {"completion", "acompletion"}
UPSTREAM, PRIVATE, BLOCKED_ROUTE = "upstream", "private", "blocked"
MASKED, CLEAN, PASSTHROUGH = "masked", "clean", "unchecked"

BLOCK_MESSAGES = {
    CURRENT_TURN: ("[Inference Router] This message was not sent to the AI model. It contains personal "
                   "information that could not be hidden automatically, for example a person described by "
                   "name together with health, money or work details. Please rephrase it without "
                   "identifying the person."),
    HISTORY: ("[Inference Router] This conversation was not sent to the AI model. An earlier message "
              "contains personal information that could not be hidden automatically. Please start a new "
              "conversation to continue."),
    FAIL_CLOSED: ("[Inference Router] This request was not sent to the AI model because it could not be "
                  "checked for personal information (the checker is unavailable, the request is too large, "
                  "or it contains an image or file it cannot read). Please try again shortly."),
    "no_private_upstream": ("[Inference Router] This request contains personal information and this route "
                            "has no private upstream to answer it, so it was not sent."),
    "no_upstream": "[Inference Router] This route has no enabled upstream, so the request was not sent.",
    "subscription_only": ("[Inference Router] This route only uses the admin's personal subscription, which "
                          "this API key may not use, so the request was not sent."),
    "laya_unavailable": ("[Inference Router] This key requires the personal-data check (mask + check), but the "
                         "checker is not installed on this router, so the request was not sent."),
}


class _NoLaya:
    def score(self, texts):
        return [0.0 for _ in texts]


class RouterHook(CustomLogger):
    MAP_TTL = 900.0
    MAP_MAX = 2000

    def __init__(self, config: Config | None = None, env=os.environ, scorer=None, keystore=None):
        super().__init__()
        self.env = env
        self.config = config or load(env.get("IR_CONFIG"))
        self.chains = chains(self.config, env)
        self.detector = Detector(self.config.privacy.rules)
        # Two deciders: without Laya (mask: detectors only) and with it (mask_check, and
        # keep_private when privacy.laya is on). Laya is available when configured or when
        # the installer put it on the box (IR_LAYA_AVAILABLE=1), so a key can ask for it.
        plain = Settings.from_env(env)
        plain.laya_enabled = False
        self.router_plain = Router(_NoLaya(), plain, detect_fn=self.detector.detect)
        self.laya_available = self.config.privacy.laya or env.get("IR_LAYA_AVAILABLE") == "1" or scorer is not None
        self.router_laya = None
        if self.laya_available:
            with_laya = Settings.from_env(env)
            with_laya.laya_enabled = True
            self.router_laya = Router(scorer or LayaHttpScorer.from_env(env), with_laya, detect_fn=self.detector.detect)
        self._keystore = keystore
        secret = (env.get("IR_MASK_SECRET") or "").strip()
        if self.config.privacy.mode in ("mask", "mask_check") and not secret:
            log.warning("inference-router: IR_MASK_SECRET is not set; placeholders change after a restart")
        self.mask_secret = secret.encode() if secret else secrets.token_bytes(32)
        self.masked_marker = env.get("IR_MASKED_MARKER", " ‡")
        self.private_marker = env.get("IR_PRIVATE_MARKER", " †")
        self._strip = sorted({m for m in (self.masked_marker, self.private_marker) if m}, key=len, reverse=True)
        self._maps: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.events_url = (env.get("IR_EVENTS_URL") or "").strip()
        self.events_token = env.get("IR_EVENTS_TOKEN") or ""
        self._events_client = None

    # --- metadata events (no content) -------------------------------------------

    def _emit(self, event: dict) -> None:
        log.info("ir-decision " + json.dumps(event, sort_keys=True))
        if not self.events_url:
            return
        try:
            asyncio.get_running_loop().create_task(self._post(event))
        except RuntimeError:
            pass

    async def _post(self, event: dict) -> None:
        try:
            import httpx
            if self._events_client is None:
                self._events_client = httpx.AsyncClient(timeout=2.0)
            await self._events_client.post(self.events_url, json=event,
                                           headers={"Authorization": f"Bearer {self.events_token}"})
        except Exception:  # noqa: BLE001 — the feed is best effort
            pass

    def keystore(self) -> KeyStore | None:
        if self._keystore is None:
            try:
                self._keystore = KeyStore()
            except Exception:  # noqa: BLE001 — usage counting must never break a request
                self._keystore = False
        return self._keystore or None

    def _finish(self, data, outcome: str, tokens: int = 0, **extra) -> None:
        event = dict(self._meta(data).get("ir_event") or {})
        if not event or event.get("_done"):
            return
        self._meta(data)["ir_event"]["_done"] = True
        event.pop("_done", None)
        if event.get("decision") == BLOCKED_ROUTE:
            tokens = 0   # refusals never reach a model; LiteLLM's mock reply reports token counts
        key_id = self._meta(data).get("ir_key_id")
        if key_id is not None and self.keystore():
            try:
                self.keystore().record(key_id, tokens=tokens, masked=event.get("reason") == MASKED,
                                       blocked=event.get("decision") == BLOCKED_ROUTE)
            except Exception:  # noqa: BLE001
                pass
        if tokens:
            event["tokens"] = tokens
        event.update(outcome=outcome, total_ms=round((time.time() - event.get("ts", time.time())) * 1000), **extra)
        self._emit(event)

    # --- helpers ---------------------------------------------------------------

    @staticmethod
    def _meta(data) -> dict:
        if not isinstance((data or {}).get("metadata"), dict):
            if isinstance(data, dict):
                data["metadata"] = {}
            else:
                return {}
        return data["metadata"]

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

    def _keep_map(self, data, mapping: dict) -> None:
        now = time.time()
        while self._maps and (len(self._maps) >= self.MAP_MAX
                              or now - next(iter(self._maps.values()))[0] > self.MAP_TTL):
            self._maps.popitem(last=False)
        map_id = secrets.token_hex(12)
        self._maps[map_id] = (now, mapping)
        self._meta(data)["ir_map"] = map_id

    def _peek_map(self, data) -> dict:
        entry = self._maps.get(self._meta(data).get("ir_map") or "")
        return entry[1] if entry else {}

    def _drop_map(self, data) -> None:
        self._maps.pop(self._meta(data).get("ir_map") or "", None)

    def _set(self, data, decision: str, reason: str, event: dict) -> None:
        meta = self._meta(data)
        meta.update(ir_decision=decision, ir_reason=reason)
        event.update(decision=decision, reason=reason)
        meta["ir_event"] = event

    def _block(self, data, reason: str, event: dict, message: str | None = None) -> dict:
        data["model"] = BLOCKED
        data["mock_response"] = message or BLOCK_MESSAGES.get(reason, BLOCK_MESSAGES[FAIL_CLOSED])
        data["messages"] = [{"role": "user", "content": "blocked by the Inference Router"}]
        for key in ("tools", "tool_choice", "functions", "function_call", "response_format"):
            data.pop(key, None)
        self._set(data, BLOCKED_ROUTE, reason, event)
        return data

    # --- the decision -----------------------------------------------------------

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        started = time.time()
        requested = data.get("model")
        route = self.config.route(requested)
        key_meta = dict(getattr(user_api_key_dict, "metadata", None) or {})
        mode = key_meta.get("ir_privacy") or self.config.privacy.mode
        event = {"ts": round(started, 3), "route": requested if route else None, "mode": mode,
                 "stream": bool(data.get("stream")), "key": key_meta.get("ir_key_name")}
        self._meta(data)["ir_key_id"] = key_meta.get("ir_key_id")
        if route is None:
            names = ", ".join(self.config.route_names()) or "none configured"
            return self._block(data, "unknown_model", event,
                               f"[Inference Router] Unknown model '{requested}'. Available: {names}.")
        if key_meta.get("ir_routes") is not None and route.name not in key_meta["ir_routes"]:
            return self._block(data, "route_not_allowed", event,
                               f"[Inference Router] This API key may not use '{route.name}'. "
                               f"Allowed: {', '.join(key_meta['ir_routes']) or 'none'}.")
        if call_type not in CHAT_CALL_TYPES:
            event["reasons"] = [f"unsupported_call_type:{call_type}"]
            return self._block(data, FAIL_CLOSED, event)
        if mode == "mask_check" and self.router_laya is None:
            return self._block(data, "laya_unavailable", event)
        chain = dict(self.chains[route.name])
        if not key_meta.get("ir_subscriptions"):
            chain["all"] = chain["shared"]
        try:
            self._strip_markers(data.get("messages"))
            messages = data.get("messages") or []
            if mode == "off":
                decision, reason, target, masked = UPSTREAM, PASSTHROUGH, chain["all"], None
            elif mode in ("mask", "mask_check"):
                session = MaskSession(self.mask_secret, finder=self.detector.find_spans)
                masked = session.mask_messages(messages)
                decider = self.router_laya if mode == "mask_check" else self.router_plain
                verdict = await asyncio.to_thread(decider.decide, for_residual_check(masked))
                event.update(masked=dict(sorted(session.counts.items())), masked_values=len(session.mapping),
                             laya_max=verdict.log_fields()["laya_max"], reasons=verdict.reasons)
                if verdict.route == LOCAL:
                    return self._block(data, verdict.local_reason or FAIL_CLOSED, event)
                decision, reason, target = UPSTREAM, (MASKED if session.mapping else CLEAN), chain["all"]
                if session.mapping:
                    data["messages"] = add_system_note(masked)
                    self._keep_map(data, session.mapping)
            else:  # keep_private
                decider = self.router_laya if (self.config.privacy.laya and self.router_laya) else self.router_plain
                verdict = await asyncio.to_thread(decider.decide, messages)
                event.update(laya_max=verdict.log_fields()["laya_max"], reasons=verdict.reasons)
                if verdict.route == LOCAL:
                    if not chain["private"]:
                        return self._block(data, "no_private_upstream", event)
                    decision, reason, target = PRIVATE, verdict.local_reason or FAIL_CLOSED, chain["private"]
                else:
                    decision, reason, target = UPSTREAM, CLEAN, chain["all"]
        except DetectionTimeout as error:
            event["reasons"] = [f"rule_timeout:{error}"]
            return self._block(data, FAIL_CLOSED, event)
        except Exception as error:  # noqa: BLE001 — fail closed on any bug
            event["reasons"] = ["router_error"]
            event["error"] = type(error).__name__
            return self._block(data, FAIL_CLOSED, event)
        if not target:
            reason = "subscription_only" if self.chains[route.name]["all"] and decision == UPSTREAM else "no_upstream"
            return self._block(data, reason, event)
        data["model"] = target[0]
        event.update(target=target[0], decide_ms=round((time.time() - started) * 1000, 1))
        self._set(data, decision, reason, event)
        return data

    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        """After LiteLLM picked the deployment: a Claude setup token needs Claude Code's
        identity as the first system line (current models refuse setup tokens without it)."""
        if is_claude_token(kwargs.get("api_key")) and isinstance(kwargs.get("messages"), list):
            kwargs["messages"] = with_claude_identity(kwargs["messages"])
        return kwargs

    # --- replies ----------------------------------------------------------------

    async def async_post_call_response_headers_hook(self, data, user_api_key_dict, response,
                                                    request_headers=None, litellm_call_info=None):
        meta = self._meta(data)
        if "ir_decision" not in meta:
            return None
        out = {"x-pii-route": meta["ir_decision"], "x-pii-reason": meta.get("ir_reason") or CLEAN}
        event = meta.get("ir_event") or {}
        if event.get("route"):
            out["x-router-route"] = event["route"]
        if meta.get("ir_reason") == MASKED:
            out["x-pii-masked"] = str(event.get("masked_values") or 0)
        return out

    async def async_post_call_failure_hook(self, request_data, original_exception, user_api_key_dict,
                                           traceback_str=None):
        self._drop_map(request_data)
        self._finish(request_data, "error", error=type(original_exception).__name__)
        return None

    def _marker(self, data) -> str:
        meta = self._meta(data)
        if meta.get("ir_reason") == MASKED:
            return self.masked_marker
        if meta.get("ir_decision") == PRIVATE:
            return self.private_marker
        return ""

    @staticmethod
    def _restore_message(message, mapping: dict) -> None:
        for attr in ("content", "reasoning_content"):
            value = getattr(message, attr, None)
            if isinstance(value, str):
                setattr(message, attr, restore_text(value, mapping))
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            if function is not None and isinstance(getattr(function, "arguments", None), str):
                function.arguments = restore_json_text(function.arguments, mapping)

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        mapping = self._peek_map(data)
        marker = self._marker(data)
        try:
            if mapping:
                for choice in response.choices:
                    self._restore_message(choice.message, mapping)
            message = response.choices[0].message
            if marker and isinstance(message.content, str) and message.content.strip():
                message.content = message.content + marker
        except (AttributeError, IndexError, TypeError):
            pass
        finally:
            self._drop_map(data)
            usage = getattr(response, "usage", None)
            self._finish(data, "ok", tokens=int(getattr(usage, "total_tokens", 0) or 0),
                         served_by=getattr(response, "model", None))
        return response

    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        mapping = self._peek_map(request_data)
        marker = self._marker(request_data)
        try:
            tokens = 0
            async for chunk in self._restore_stream(response, mapping, marker):
                usage = getattr(chunk, "usage", None)
                if usage is not None and getattr(usage, "total_tokens", None):
                    tokens = int(usage.total_tokens)
                yield chunk
            self._finish(request_data, "ok", tokens=tokens)
        except Exception as error:
            self._finish(request_data, "error", error=type(error).__name__)
            raise
        finally:
            self._drop_map(request_data)

    async def _restore_stream(self, response, mapping: dict, marker: str):
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
                        saw_text[i] = saw_text.get(i, False) or bool(text.strip())
                        if mapping:
                            delta.content = content.setdefault(i, StreamRestorer(mapping)).feed(text)
                    if mapping:
                        for call in getattr(delta, "tool_calls", None) or []:
                            function = getattr(call, "function", None)
                            piece = getattr(function, "arguments", None) if function is not None else None
                            if isinstance(piece, str) and piece:
                                key = (i, getattr(call, "index", 0) or 0)
                                function.arguments = args.setdefault(key, StreamRestorer(mapping, True)).feed(piece)
                    if choice.finish_reason:
                        rest = content.pop(i).flush() if i in content else ""
                        tail = marker if saw_text.get(i) else ""
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


def _instance():
    try:
        return RouterHook()
    except Exception as error:  # noqa: BLE001
        # Without a valid config the router must not run at all: LiteLLM then fails to
        # start, and nothing can reach an upstream unchecked.
        raise RuntimeError(f"Inference Router: cannot load config: {error}") from error


proxy_handler_instance = _instance() if os.environ.get("IR_CONFIG") or os.path.exists(
    "/etc/inference-router/router.yaml") else None
