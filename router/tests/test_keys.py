"""Per-person keys: store, auth and hook enforcement."""

import asyncio
import json
import time
from types import SimpleNamespace as NS

import pytest
from litellm.proxy._types import ProxyException

from inference_router.auth import check
from inference_router.config import parse
from inference_router.hook import RouterHook
from inference_router.keys import DEFAULT_RPM, KeyError_, KeyStore

from tests.test_inference_router import cfg
from tests.test_policy import FakeLaya, user

ROUTES = ["default", "fast"]


@pytest.fixture()
def ks(tmp_path):
    return KeyStore(str(tmp_path / "keys.db"))


def test_defaults_need_only_a_name(ks):
    key, secret = ks.create("Ann (finance)")
    assert secret.startswith("sk-ir-") and len(secret) > 40
    assert key.routes is None and key.privacy is None and key.rpm == DEFAULT_RPM and key.tpd is None
    assert key.expires is None and key.active
    assert ks.verify(secret).id == key.id and ks.verify(secret + "x") is None
    assert secret not in json.dumps(ks.list())          # only a prefix is ever listed


def test_validation(ks):
    for kwargs in ({"name": ""}, {"name": "x", "privacy": "encrypt"}, {"name": "x", "rpm": 0},
                   {"name": "x", "routes": "default"}, {"name": "x", "expires_days": -1}):
        with pytest.raises(KeyError_):
            ks.create(**kwargs)


def test_revoke_expiry_and_update(ks):
    key, secret = ks.create("temp", expires_days=1)
    assert ks.revoke(key.id) and not ks.verify(secret).active
    key2, _ = ks.create("x")
    updated = ks.update(key2.id, routes=["fast"], privacy="mask_check", rpm=None)
    assert updated.routes == ("fast",) and updated.privacy == "mask_check" and updated.rpm is None


def test_rate_limit_and_daily_tokens(ks):
    key, _ = ks.create("burst", rpm=2, tpd=100)
    now = time.time()
    assert ks.allow_request(key, now) is None and ks.allow_request(key, now) is None
    assert "2 requests per minute" in ks.allow_request(key, now)
    assert ks.allow_request(key, now + 61) is None      # window moved on
    ks.record(key.id, tokens=150)
    assert "daily token limit" in ks.allow_request(key, now + 200)


def test_usage_is_counts_only(ks):
    key, _ = ks.create("u")
    ks.record(key.id, tokens=10, masked=True)
    ks.record(key.id, blocked=True)
    usage = next(k for k in ks.list() if k["id"] == key.id)["usage"]["today"]
    assert usage == {"requests": 2, "tokens": 10, "masked": 1, "blocked": 1}


# --- auth ---------------------------------------------------------------------------------

ENV = {"LITELLM_MASTER_KEY": "sk-admin"}


def test_auth_admin_user_revoked_and_models(ks):
    assert check("Bearer sk-admin", "/v1/chat/completions", ENV, ROUTES, ks).metadata["ir_key_name"] == "admin"
    key, secret = ks.create("Bob", routes=["fast"])
    auth = check(f"Bearer {secret}", "/v1/chat/completions", ENV, ROUTES, ks)
    assert auth.metadata["ir_key_id"] == key.id and auth.metadata["ir_routes"] == ["fast"]
    assert check(secret, "/v1/models", ENV, ROUTES, ks).models == ["fast"]
    for bad in ("", "sk-ir-nope", "random"):
        with pytest.raises(ProxyException) as e:
            check(bad, "/v1/chat/completions", ENV, ROUTES, ks)
        assert str(e.value.code) == "401"
    ks.revoke(key.id)
    with pytest.raises(ProxyException, match="revoked"):
        check(secret, "/v1/chat/completions", ENV, ROUTES, ks)


def test_auth_rate_limit_is_429(ks):
    _, secret = ks.create("fast caller", rpm=1)
    check(secret, "/v1/chat/completions", ENV, ROUTES, ks)
    with pytest.raises(ProxyException) as e:
        check(secret, "/v1/chat/completions", ENV, ROUTES, ks)
    assert str(e.value.code) == "429"


# --- hook ----------------------------------------------------------------------------------

def hook(data, ks, laya=True):
    return RouterHook(config=parse(data), env={"IR_MASK_SECRET": "s"}, scorer=FakeLaya() if laya else None, keystore=ks)


def pre(h, messages, meta, model="default"):
    data = {"model": model, "messages": messages}
    return asyncio.run(h.async_pre_call_hook(NS(metadata=meta), None, data, "acompletion"))


def test_route_not_allowed_for_key(ks):
    data = pre(hook(cfg(), ks), [user("hi")], {"ir_key_id": 1, "ir_routes": ["fast"]})
    assert data["model"] == "ir/blocked" and "may not use 'default'" in data["mock_response"]


def test_key_privacy_overrides_router_setting(ks):
    h = hook(cfg(mode="off"), ks)
    msg = [user("Sarah is on sick leave, mail ann@acme.ie")]
    assert "ann@acme.ie" in json.dumps(pre(h, msg, {})["messages"])               # router: off
    assert pre(h, msg, {"ir_privacy": "mask_check"})["model"] == "ir/blocked"     # HR key: mask + check
    masked = pre(h, msg, {"ir_privacy": "mask"})                                   # mask: no Laya, sent masked
    assert masked["model"] == "ir/default/0" and "ann@acme.ie" not in json.dumps(masked["messages"])


def test_mask_check_key_without_laya_is_refused(ks):
    h = hook(cfg(mode="mask"), ks, laya=False)
    data = pre(h, [user("hi")], {"ir_privacy": "mask_check"})
    assert data["model"] == "ir/blocked" and "checker is not installed" in data["mock_response"]


def test_usage_recorded_on_success(ks):
    key, _ = ks.create("Ann")
    h = hook(cfg(), ks)
    h._emit = lambda e: None
    data = pre(h, [user("mail ann@acme.ie")], {"ir_key_id": key.id, "ir_key_name": "Ann"})
    resp = NS(model="x", usage=NS(total_tokens=42),
              choices=[NS(message=NS(content="ok", reasoning_content=None, tool_calls=None))])
    asyncio.run(h.async_post_call_success_hook(data, None, resp))
    usage = next(k for k in ks.list() if k["id"] == key.id)["usage"]["today"]
    assert usage == {"requests": 1, "tokens": 42, "masked": 1, "blocked": 0}


def test_refusals_count_no_tokens(ks):
    key, _ = ks.create("HR", privacy="mask_check")
    h = hook(cfg(), ks)
    h._emit = lambda e: None
    data = pre(h, [user("Sarah is on sick leave")], {"ir_key_id": key.id, "ir_privacy": "mask_check"})
    resp = NS(model="x", usage=NS(total_tokens=30),
              choices=[NS(message=NS(content="refused", reasoning_content=None, tool_calls=None))])
    asyncio.run(h.async_post_call_success_hook(data, None, resp))
    usage = next(k for k in ks.list() if k["id"] == key.id)["usage"]["today"]
    assert usage == {"requests": 1, "tokens": 0, "masked": 0, "blocked": 1}
