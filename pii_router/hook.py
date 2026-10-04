"""LiteLLM proxy hook: pick the cloud or local deployment for each request.

Wired in config.yaml as ``litellm_settings.callbacks: pii_router.hook.proxy_handler_instance``.

The public model name (``pii-router``) is itself configured to the LOCAL
deployment, so if this hook is ever not loaded or raises, traffic still stays
local. The hook decides for EVERY chat request, whatever model name the client
asked for, so a caller cannot bypass the check by naming the cloud deployment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from litellm.integrations.custom_logger import CustomLogger

from .laya_client import LayaHttpScorer
from .policy import CLOUD, LOCAL, Router, Settings

log = logging.getLogger("pii_router")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False

CHAT_CALL_TYPES = {"completion", "acompletion"}


class PiiRouterHook(CustomLogger):
    def __init__(self, router: Router | None = None, env=os.environ):
        super().__init__()
        self.cloud_model = env.get("PII_ROUTER_CLOUD_MODEL_NAME", "nvidia-cloud")
        self.local_model = env.get("PII_ROUTER_LOCAL_MODEL_NAME", "ollama-local")
        self.router = router or Router(LayaHttpScorer.from_env(env), Settings.from_env(env))

    def _log(self, fields: dict) -> None:
        log.info("pii-router-decision " + json.dumps({"ts": round(time.time(), 3), **fields}, sort_keys=True))

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        requested = data.get("model")
        if call_type not in CHAT_CALL_TYPES:
            # Only chat completions are inspected. Anything else that targets our
            # deployments is pinned local rather than passed through unchecked.
            if requested in (self.cloud_model,):
                data["model"] = self.local_model
            self._log({"route": LOCAL, "reasons": [f"unsupported_call_type:{call_type}"], "requested": requested})
            return data
        try:
            decision = await asyncio.to_thread(self.router.decide, data.get("messages") or [])
            fields = decision.log_fields()
        except Exception as error:  # noqa: BLE001 — fail closed on any bug
            fields = {"route": LOCAL, "reasons": ["router_error"], "error": type(error).__name__}
        data["model"] = self.cloud_model if fields["route"] == CLOUD else self.local_model
        fields["requested"] = requested
        fields["target"] = data["model"]
        self._log(fields)
        return data


proxy_handler_instance = PiiRouterHook()
