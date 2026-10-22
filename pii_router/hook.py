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
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from litellm.integrations.custom_logger import CustomLogger

from .health import CloudHealth
from .laya_client import LayaHttpScorer
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


class PiiRouterHook(CustomLogger):
    def __init__(self, router: Router | None = None, env=os.environ, health: CloudHealth | None = None):
        super().__init__()
        self.cloud_model = env.get("PII_ROUTER_CLOUD_MODEL_NAME", "cloud")
        self.local_model = env.get("PII_ROUTER_LOCAL_MODEL_NAME", "ollama-local")
        self.fallback_model = (env.get("PII_ROUTER_FALLBACK_MODEL_NAME", "cloud-fallback")
                               if (env.get("FALLBACK_BASE_URL") or "").strip() else None)
        self.remote_models = {m for m in (self.cloud_model, self.fallback_model, LEGACY_CLOUD_MODEL,
                                          "cloud-fallback-2") if m}
        self.pii_off = (env.get("PII_ROUTER_MODE") or "on").strip().lower() == "off"
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
        # Longest first, so " ††" is stripped whole rather than as " †" + "†".
        self._strip = sorted({m for m in self.markers.values() if m}, key=len, reverse=True)

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

    # --- LiteLLM hooks ---------------------------------------------------------

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        requested = data.get("model")
        if self.pii_off:
            return await self._route_unchecked(data, call_type, requested)
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
        return None

    async def async_post_call_response_headers_hook(self, data, user_api_key_dict, response,
                                                    request_headers=None, litellm_call_info=None):
        route = self._route(data)
        if route is None:
            return None
        reason = self._meta(data).get("pii_router_reason") or CLEAN
        if reason == UNCHECKED:
            return {"x-pii-route": route, "x-pii-reason": UNCHECKED}
        return {"x-pii-route": route, "x-pii-reason": reason if route == LOCAL else CLEAN}

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
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


proxy_handler_instance = PiiRouterHook()
