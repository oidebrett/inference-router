"""LiteLLM custom auth: the admin key, or a per-person key from keys.db.

- admin key (LITELLM_MASTER_KEY): everything, all routes;
- a person/team/app key (sk-ir-…): checked by hash, must not be revoked or expired, then
  its rate limit (requests/minute) and daily token limit; 401 / 429 with a clear message.
  The key's allowed routes and privacy level ride along in the auth object's metadata and
  are enforced by the hook. /v1/models lists only the routes the key may use.

The admin key, and person keys with ``subscriptions`` on, may be routed to the admin's
personal Claude/ChatGPT subscription upstreams (ir_subscriptions); others are not.
"""

from __future__ import annotations

import hmac
import os

from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth

from .config import ConfigError, load
from .keys import KeyStore

_store: KeyStore | None = None


def store() -> KeyStore:
    global _store
    if _store is None:
        _store = KeyStore()
    return _store


def _bare(key: str | None) -> str:
    key = (key or "").strip()
    return key[7:].strip() if key.lower().startswith("bearer ") else key


def _route_names() -> list[str]:
    try:
        return load().route_names()
    except (OSError, ConfigError):
        return []


def _deny(message: str, code: int = 401):
    raise ProxyException(message=message, type="auth_error", param="api_key", code=code)


def check(api_key: str | None, path: str, env=os.environ, route_names=None, keystore=None) -> UserAPIKeyAuth:
    key = _bare(api_key)
    if not key:
        _deny("Missing API key")
    names = route_names if route_names is not None else _route_names()
    master = env.get("LITELLM_MASTER_KEY") or ""
    if master and hmac.compare_digest(key.encode(), master.encode()):
        if path.rstrip("/").endswith("/models"):
            return UserAPIKeyAuth(api_key=key, models=names)
        return UserAPIKeyAuth(api_key=key, user_role=LitellmUserRoles.PROXY_ADMIN,
                              metadata={"ir_key_name": "admin", "ir_subscriptions": True})
    ks = keystore or store()
    record = ks.verify(key)
    if record is None:
        _deny("Invalid API key")
    if not record.active:
        _deny("This API key has been revoked or has expired")
    allowed = [n for n in names if record.allows(n)]
    meta = {"ir_key_id": record.id, "ir_key_name": record.name, "ir_privacy": record.privacy,
            "ir_routes": list(record.routes) if record.routes is not None else None,
            "ir_subscriptions": record.subscriptions}
    if path.rstrip("/").endswith("/models"):
        return UserAPIKeyAuth(api_key=f"ir-key-{record.id}", models=allowed, metadata=meta)
    reason = ks.allow_request(record)
    if reason:
        _deny(f"Slow down: {reason}", code=429)
    return UserAPIKeyAuth(api_key=f"ir-key-{record.id}", key_alias=record.name,
                          user_role=LitellmUserRoles.INTERNAL_USER, metadata=meta)


async def user_api_key_auth(request, api_key: str) -> UserAPIKeyAuth:
    return check(api_key, request.url.path)
