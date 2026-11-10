"""LiteLLM custom auth: the admin (master) key, 401 for anything else, and a model
list of route names only. Per-person keys (LiteLLM virtual keys) come in a later phase."""

from __future__ import annotations

import hmac
import os

from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth

from .config import ConfigError, load


def _bare(key: str | None) -> str:
    key = (key or "").strip()
    return key[7:].strip() if key.lower().startswith("bearer ") else key


def _route_names() -> list[str]:
    try:
        return load().route_names()
    except (OSError, ConfigError):
        return []


def check(api_key: str | None, path: str, env=os.environ, route_names=None) -> UserAPIKeyAuth:
    master = env.get("LITELLM_MASTER_KEY") or ""
    key = _bare(api_key)
    if not master or not key or not hmac.compare_digest(key.encode(), master.encode()):
        raise ProxyException(message="Invalid API key", type="auth_error", param="api_key", code=401)
    if path.rstrip("/").endswith("/models"):
        return UserAPIKeyAuth(api_key=key, models=route_names if route_names is not None else _route_names())
    return UserAPIKeyAuth(api_key=key, user_role=LitellmUserRoles.PROXY_ADMIN)


async def user_api_key_auth(request, api_key: str) -> UserAPIKeyAuth:
    return check(api_key, request.url.path)
