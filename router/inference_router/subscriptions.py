"""Personal subscriptions as upstreams: the admin's own Claude or ChatGPT plan.

Two upstream kinds use a person's subscription instead of an API key:

  claude_subscription   a ``claude setup-token`` token (sk-ant-oat01-...) in secrets.env.
                        LiteLLM's anthropic provider sends it as a Bearer token; the router
                        adds the Claude Code request shape (beta header + identity line)
                        that Anthropic requires for setup tokens on current models.
  chatgpt_subscription  "Sign in with ChatGPT" (the Codex device-code flow), run from the
                        admin UI. The tokens live in CHATGPT_TOKEN_DIR/auth.json, which
                        LiteLLM's chatgpt provider reads and refreshes.

A subscription belongs to one person and its terms and usage limits are theirs, so only
the admin key and keys the admin marks "may use my subscriptions" are routed to it;
everyone else gets the route's other targets. Put a shared upstream after it in the route
so requests fall back when the plan's limit is reached.

The router never starts a sign-in itself: LiteLLM's authenticator would otherwise block a
request for up to 15 minutes waiting for someone to type a device code.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

CLAUDE = "claude_subscription"
CHATGPT = "chatgpt_subscription"
KINDS = (CLAUDE, CHATGPT)

CLAUDE_BASE_URL = "https://api.anthropic.com"
CHATGPT_BASE_URL = "https://chatgpt.com/backend-api/codex"
CLAUDE_TOKEN_PREFIX = "sk-ant-oat"
CLAUDE_BETAS = "claude-code-20250219,oauth-2025-04-20"
CLAUDE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

DEFAULT_CHATGPT_DIR = "/var/lib/inference-router/chatgpt"
CHATGPT_AUTH_BASE = "https://auth.openai.com"
CHATGPT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"     # the Codex CLI's public client id
CHATGPT_VERIFY_URL = f"{CHATGPT_AUTH_BASE}/codex/device"
# Written by the admin after a sign-in or sign-out (never by the router, whose token
# refreshes rewrite auth.json); inference-router-reload.path restarts the router on it.
SIGNED_IN_STAMP = "signed-in"


def chatgpt_dir(env=os.environ) -> Path:
    return Path(env.get("CHATGPT_TOKEN_DIR") or DEFAULT_CHATGPT_DIR)


# --- Claude ------------------------------------------------------------------------------

def is_claude_token(value: str | None) -> bool:
    return bool(value) and value.startswith(CLAUDE_TOKEN_PREFIX)


def with_claude_identity(messages: list) -> list:
    """Messages with the Claude Code identity as the first system line (once)."""
    messages = list(messages or [])
    first = messages[0] if messages else None
    if isinstance(first, dict) and first.get("role") == "system" and str(first.get("content") or "").startswith(CLAUDE_IDENTITY):
        return messages
    return [{"role": "system", "content": CLAUDE_IDENTITY}] + messages


# --- ChatGPT tokens ------------------------------------------------------------------------

def _claims(token: str | None) -> dict:
    try:
        payload = (token or "").split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except Exception:  # noqa: BLE001
        return {}


def auth_record(tokens: dict) -> dict:
    """The auth.json record LiteLLM's chatgpt Authenticator reads."""
    access, id_token = tokens.get("access_token"), tokens.get("id_token")
    exp = _claims(access).get("exp")
    auth = _claims(id_token or access).get("https://api.openai.com/auth") or {}
    return {"access_token": access, "refresh_token": tokens.get("refresh_token"), "id_token": id_token,
            "expires_at": int(exp) if isinstance(exp, (int, float)) else None,
            "account_id": auth.get("chatgpt_account_id") if isinstance(auth, dict) else None}


def chatgpt_status(directory: Path) -> dict:
    """Signed in or not, and who (email and plan only; never a token)."""
    try:
        data = json.loads((directory / "auth.json").read_text())
    except (OSError, ValueError):
        return {"signed_in": False}
    if not data.get("refresh_token"):
        return {"signed_in": False}
    claims = _claims(data.get("id_token"))
    auth = claims.get("https://api.openai.com/auth") or {}
    return {"signed_in": True, "email": claims.get("email"),
            "plan": auth.get("chatgpt_plan_type") if isinstance(auth, dict) else None}


def chatgpt_signed_in(env=os.environ) -> bool:
    return chatgpt_status(chatgpt_dir(env))["signed_in"]


def save_chatgpt_tokens(directory: Path, tokens: dict) -> None:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = directory / "auth.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(auth_record(tokens), fh)
    os.chmod(tmp, 0o600)
    os.replace(tmp, directory / "auth.json")
    (directory / SIGNED_IN_STAMP).write_text(str(time.time()))


def sign_out_chatgpt(directory: Path) -> None:
    for name in ("auth.json", "auth.json.tmp"):
        try:
            (directory / name).unlink()
        except FileNotFoundError:
            pass
    if directory.exists():
        (directory / SIGNED_IN_STAMP).write_text(str(time.time()))


# --- ChatGPT device-code sign-in (admin service) ------------------------------------------

async def start_device_login(client) -> dict:
    r = await client.post(f"{CHATGPT_AUTH_BASE}/api/accounts/deviceauth/usercode", json={"client_id": CHATGPT_CLIENT_ID})
    r.raise_for_status()
    data = r.json()
    code, device_id = data.get("user_code") or data.get("usercode"), data.get("device_auth_id")
    if not code or not device_id:
        raise ValueError("unexpected device-code response")
    return {"user_code": code, "device_auth_id": device_id, "interval": int(data.get("interval") or 5),
            "verify_url": CHATGPT_VERIFY_URL}


async def poll_device_login(client, pending: dict) -> dict | None:
    """Tokens once the person has entered the code, else None (still waiting)."""
    r = await client.post(f"{CHATGPT_AUTH_BASE}/api/accounts/deviceauth/token",
                          json={"device_auth_id": pending["device_auth_id"], "user_code": pending["user_code"]})
    if r.status_code in (403, 404):
        return None
    r.raise_for_status()
    code = r.json()
    if not (code.get("authorization_code") and code.get("code_verifier")):
        return None
    r = await client.post(f"{CHATGPT_AUTH_BASE}/oauth/token", data={
        "grant_type": "authorization_code", "code": code["authorization_code"],
        "redirect_uri": f"{CHATGPT_AUTH_BASE}/deviceauth/callback", "client_id": CHATGPT_CLIENT_ID,
        "code_verifier": code["code_verifier"]})
    r.raise_for_status()
    tokens = r.json()
    if not (tokens.get("access_token") and tokens.get("refresh_token")):
        raise ValueError("sign-in response had no tokens")
    return tokens


def never_prompt_for_chatgpt_login() -> None:
    """Make LiteLLM fail fast (so the route falls back) instead of starting a device-code
    login inside a request when the saved ChatGPT sign-in is missing or revoked."""
    try:
        from litellm.llms.chatgpt.authenticator import Authenticator
        from litellm.llms.chatgpt.common_utils import GetAccessTokenError
    except ImportError:
        return

    def _refuse(self):
        raise GetAccessTokenError(status_code=401, message="ChatGPT is not signed in: sign in again in the router admin")

    Authenticator._login_device_code = _refuse
    Authenticator._wait_for_access_token = lambda self, timeout_seconds: None
