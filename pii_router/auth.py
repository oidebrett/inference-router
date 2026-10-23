"""LiteLLM custom auth for the PII router: one key, clear errors, a short model list.

Wired in config.yaml as ``general_settings.custom_auth: pii_router.auth.user_api_key_auth``.

The router runs without a database, so LiteLLM's own check answers any key that is
not the master key with ``400 No connected db``. Here a wrong or missing key gets
``401 Invalid API key`` instead, which is what OpenAI-compatible clients expect.

``GET /v1/models`` lists only the public model names (PII_ROUTER_PUBLIC_MODELS,
default ``pii-router``). The internal deployments (cloud, fallbacks, local) are
picked by the hook, never by the caller, so listing them only misleads tools that
choose a model from the list.
"""

from __future__ import annotations

import hmac
import os

from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth

DEFAULT_PUBLIC_MODELS = "pii-router"


def public_models(env=os.environ) -> list[str]:
    names = [n.strip() for n in (env.get("PII_ROUTER_PUBLIC_MODELS") or DEFAULT_PUBLIC_MODELS).split(",")]
    return [n for n in names if n]


def _bare(key: str | None) -> str:
    key = (key or "").strip()
    return key[7:].strip() if key.lower().startswith("bearer ") else key


def check(api_key: str | None, path: str, env=os.environ) -> UserAPIKeyAuth:
    master = env.get("LITELLM_MASTER_KEY") or ""
    key = _bare(api_key)
    if not master or not key or not hmac.compare_digest(key.encode(), master.encode()):
        raise ProxyException(message="Invalid API key", type="auth_error", param="api_key", code=401)
    if path.rstrip("/").endswith("/models"):
        # A model-scoped (non-admin) identity: LiteLLM filters the list to these names.
        return UserAPIKeyAuth(api_key=key, models=public_models(env))
    return UserAPIKeyAuth(api_key=key, user_role=LitellmUserRoles.PROXY_ADMIN)


async def user_api_key_auth(request, api_key: str) -> UserAPIKeyAuth:
    return check(api_key, request.url.path)
