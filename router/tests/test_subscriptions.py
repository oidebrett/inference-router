"""Personal subscriptions (the admin's Claude / ChatGPT plan) as upstreams."""

import asyncio
import base64
import json
from types import SimpleNamespace as NS

import pytest

from inference_router import render, subscriptions as subs
from inference_router.auth import check
from inference_router.config import ConfigError, parse
from inference_router.hook import RouterHook
from inference_router.keys import KeyStore

from tests.test_policy import FakeLaya, user


def _jwt(claims: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"h.{body}.s"


def sub_cfg(chatgpt=False, only_subscription=False):
    ups = [{"id": "my-claude", "name": "My Claude plan", "kind": "claude_subscription"},
           {"id": "nvidia", "name": "NVIDIA", "base_url": "https://integrate.api.nvidia.com/v1"}]
    claude_route = [{"upstream": "my-claude", "model": "claude-sonnet-4-6"}]
    if not only_subscription:
        claude_route.append({"upstream": "nvidia", "model": "nvidia/nemotron"})
    routes = [{"name": "claude", "targets": claude_route},
              {"name": "default", "targets": [{"upstream": "nvidia", "model": "nvidia/nemotron"}]}]
    if chatgpt:
        ups.append({"id": "my-chatgpt", "name": "My ChatGPT plan", "kind": "chatgpt_subscription"})
        routes.append({"name": "gpt", "targets": [{"upstream": "my-chatgpt", "model": "gpt-5.5"},
                                                  {"upstream": "nvidia", "model": "nvidia/nemotron"}]})
    return {"version": 1, "privacy": {"mode": "mask"}, "upstreams": ups, "routes": routes}


@pytest.fixture()
def signed_out(tmp_path):
    return {"CHATGPT_TOKEN_DIR": str(tmp_path / "nobody"), "IR_MASK_SECRET": "s"}


@pytest.fixture()
def signed_in(tmp_path):
    d = tmp_path / "chatgpt"
    subs.save_chatgpt_tokens(d, {"access_token": _jwt({"exp": 9999999999}), "refresh_token": "r",
                                 "id_token": _jwt({"email": "admin@acme.ie",
                                                   "https://api.openai.com/auth": {"chatgpt_plan_type": "plus",
                                                                                   "chatgpt_account_id": "acc"}})})
    return {"CHATGPT_TOKEN_DIR": str(d), "IR_MASK_SECRET": "s"}


# --- config ----------------------------------------------------------------------------

def test_subscription_upstreams_have_fixed_urls_and_no_private():
    c = parse(sub_cfg(chatgpt=True))
    assert c.upstream("my-claude").base_url == subs.CLAUDE_BASE_URL and c.upstream("my-claude").subscription
    assert c.upstream("my-chatgpt").base_url == subs.CHATGPT_BASE_URL
    assert c.upstream("my-claude").effective_reasoning_effort() == ""
    bad = sub_cfg()
    bad["upstreams"][0]["private"] = True
    with pytest.raises(ConfigError, match="cannot be marked private"):
        parse(bad)
    bad = sub_cfg()
    bad["upstreams"][0]["base_url"] = "https://evil.example.com"
    with pytest.raises(ConfigError, match="always uses"):
        parse(bad)
    bad = sub_cfg(chatgpt=True)
    bad["upstreams"].append({"id": "second", "kind": "chatgpt_subscription"})
    with pytest.raises(ConfigError, match="only one ChatGPT"):
        parse(bad)
    with pytest.raises(ConfigError, match="kind must be"):
        parse({**sub_cfg(), "upstreams": [{"id": "x", "kind": "gemini_subscription"}], "routes": []})


# --- render ----------------------------------------------------------------------------

def test_render_claude_params_and_shared_chain(signed_out):
    out = render.build(parse(sub_cfg()), env=signed_out)
    by_name = {m["model_name"]: m["litellm_params"] for m in out["model_list"]}
    claude = by_name["ir/claude/0"]
    assert claude["model"] == "anthropic/claude-sonnet-4-6" and claude["api_key"] == "os.environ/IR_KEY_MY_CLAUDE"
    assert claude["extra_headers"] == {"anthropic-beta": subs.CLAUDE_BETAS} and "api_base" not in claude
    assert by_name["ir/claude/shared/0"]["model"] == "openai/nvidia/nemotron"
    assert "ir/default/shared/0" not in by_name                     # no subscription, no extra chain
    assert {"ir/claude/0": ["ir/claude/1"]} in out["router_settings"]["fallbacks"]
    chains = render.chains(parse(sub_cfg()), env=signed_out)
    assert chains["claude"]["shared"] == ["ir/claude/shared/0"] and chains["default"]["shared"] == ["ir/default/0"]


def test_chatgpt_target_left_out_until_signed_in(signed_out, signed_in):
    c = parse(sub_cfg(chatgpt=True))
    assert render.chains(c, env=signed_out)["gpt"]["all"] == ["ir/gpt/0"]        # only the NVIDIA fallback
    names = {m["model_name"]: m["litellm_params"] for m in render.build(c, env=signed_in)["model_list"]}
    assert names["ir/gpt/0"] == {"model": "chatgpt/gpt-5.5"} and names["ir/gpt/1"]["model"] == "openai/nvidia/nemotron"


# --- who may use a subscription ----------------------------------------------------------

def pre(h, meta, model="claude"):
    return asyncio.run(h.async_pre_call_hook(NS(metadata=meta), None,
                                             {"model": model, "messages": [user("hello")]}, "acompletion"))


def test_only_admin_and_allowed_keys_reach_the_subscription(signed_out):
    h = RouterHook(config=parse(sub_cfg()), env=signed_out, scorer=FakeLaya())
    assert pre(h, {"ir_key_name": "admin", "ir_subscriptions": True})["model"] == "ir/claude/0"
    assert pre(h, {"ir_key_id": 3, "ir_subscriptions": True})["model"] == "ir/claude/0"
    assert pre(h, {"ir_key_id": 4})["model"] == "ir/claude/shared/0"           # skips to NVIDIA
    assert pre(h, {})["model"] == "ir/claude/shared/0"


def test_subscription_only_route_refuses_other_keys(signed_out):
    h = RouterHook(config=parse(sub_cfg(only_subscription=True)), env=signed_out, scorer=FakeLaya())
    data = pre(h, {"ir_key_id": 4})
    assert data["model"] == "ir/blocked" and "personal subscription" in data["mock_response"]
    assert pre(h, {"ir_subscriptions": True})["model"] == "ir/claude/0"


def test_auth_metadata_carries_the_permission(tmp_path):
    ks = KeyStore(str(tmp_path / "k.db"))
    _, plain = ks.create("Bob")
    key, mine = ks.create("Admin laptop", subscriptions=True)
    env = {"LITELLM_MASTER_KEY": "sk-admin"}
    assert check("sk-admin", "/v1/chat/completions", env, ["claude"], ks).metadata["ir_subscriptions"] is True
    assert check(mine, "/v1/chat/completions", env, ["claude"], ks).metadata["ir_subscriptions"] is True
    assert check(plain, "/v1/chat/completions", env, ["claude"], ks).metadata["ir_subscriptions"] is False
    assert ks.update(key.id, subscriptions=False).subscriptions is False


def test_old_keys_db_gets_the_column(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE keys (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, prefix TEXT NOT NULL, "
               "hash TEXT NOT NULL UNIQUE, created REAL NOT NULL, expires REAL, revoked REAL, routes TEXT, "
               "privacy TEXT, rpm INTEGER, tpd INTEGER, last_used REAL)")
    db.execute("INSERT INTO keys (name, prefix, hash, created) VALUES ('old', 'sk-ir-x', 'h', 1)")
    db.commit()
    db.close()
    ks = KeyStore(path)
    assert ks.get(1).subscriptions is False


# --- Claude request shape and ChatGPT sign-in state --------------------------------------

def test_claude_identity_added_once_for_setup_tokens(signed_out):
    h = RouterHook(config=parse(sub_cfg()), env=signed_out, scorer=FakeLaya())
    msgs = [{"role": "system", "content": "Be brief."}, user("hi")]
    out = asyncio.run(h.async_pre_call_deployment_hook({"api_key": "sk-ant-oat01-abc", "messages": msgs}, None))
    assert out["messages"][0] == {"role": "system", "content": subs.CLAUDE_IDENTITY} and out["messages"][1:] == msgs
    again = asyncio.run(h.async_pre_call_deployment_hook(dict(out), None))
    assert again["messages"] == out["messages"]
    other = asyncio.run(h.async_pre_call_deployment_hook({"api_key": "nvapi-x", "messages": msgs}, None))
    assert other["messages"] == msgs


def test_chatgpt_status_shows_who_never_tokens(signed_out, signed_in):
    from pathlib import Path
    assert subs.chatgpt_status(Path(signed_out["CHATGPT_TOKEN_DIR"])) == {"signed_in": False}
    d = Path(signed_in["CHATGPT_TOKEN_DIR"])
    status = subs.chatgpt_status(d)
    assert status == {"signed_in": True, "email": "admin@acme.ie", "plan": "plus"}
    record = json.loads((d / "auth.json").read_text())
    assert record["account_id"] == "acc" and record["expires_at"] == 9999999999
    assert oct((d / "auth.json").stat().st_mode)[-3:] == "600" and (d / subs.SIGNED_IN_STAMP).exists()
    subs.sign_out_chatgpt(d)
    assert subs.chatgpt_status(d) == {"signed_in": False}


def test_router_never_starts_a_device_login():
    from litellm.llms.chatgpt.authenticator import Authenticator
    from litellm.llms.chatgpt.common_utils import GetAccessTokenError
    with pytest.raises(GetAccessTokenError, match="not signed in"):
        Authenticator._login_device_code(object())
