"""Inference Router core: config, render, detection with custom rules, and the hook."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from inference_router import render
from inference_router.config import ConfigError, parse
from inference_router.detection import DetectionTimeout, Detector
from inference_router.hook import BLOCK_MESSAGES, RouterHook
from pii_router.mask import PLACEHOLDER
from pii_router.policy import CURRENT_TURN

from tests.test_policy import FakeLaya, user


def cfg(mode="mask", laya=None, rules=None, private=True, extra_routes=()):
    privacy = {"mode": mode}
    if laya is not None:
        privacy["laya"] = laya
    if rules:
        privacy["rules"] = rules
    return {
        "version": 1,
        "privacy": privacy,
        "upstreams": [
            {"id": "nvidia", "name": "NVIDIA", "base_url": "https://integrate.api.nvidia.com/v1"},
            {"id": "openrouter", "name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1"},
            {"id": "office", "name": "Office GPU", "base_url": "https://gpu.example.com/v1", "private": private},
        ],
        "routes": [
            {"name": "default", "targets": [{"upstream": "nvidia", "model": "nvidia/nemotron"},
                                            {"upstream": "openrouter", "model": "openai/gpt-4o-mini"},
                                            {"upstream": "office", "model": "qwen3.8-27b"}]},
            *extra_routes,
        ],
    }


def hook(data, laya=None):
    return RouterHook(config=parse(data), env={"IR_MASK_SECRET": "s"}, scorer=laya or FakeLaya())


def pre(h, messages, model="default", call_type="acompletion", **extra):
    data = {"model": model, "messages": messages, **extra}
    return asyncio.run(h.async_pre_call_hook(None, None, data, call_type))


def sent(data):
    return json.dumps(data["messages"])


# --- config -------------------------------------------------------------------------

def test_valid_config_parses_and_keys_come_from_ids():
    c = parse(cfg())
    assert c.route_names() == ["default"]
    assert c.upstream("openrouter").key_env == "IR_KEY_OPENROUTER"
    assert c.upstream("nvidia").hosted and not c.upstream("office").hosted


@pytest.mark.parametrize("change,message", [
    (lambda d: d["privacy"].update(mode="encrypt"), "privacy.mode"),
    (lambda d: d["upstreams"][0].update(base_url="ftp://x"), "base_url"),
    (lambda d: d["upstreams"][0].update(base_url="http://api.example.com/v1"), "plain http"),
    (lambda d: d["upstreams"][0].update(base_url="https://u:p@x.com/v1"), "base_url"),
    (lambda d: d["upstreams"][1].update(id="nvidia"), "unique"),
    (lambda d: d["routes"][0]["targets"].append({"upstream": "ghost", "model": "m"}), "ghost"),
    (lambda d: d["upstreams"][0].update(secret="x"), "unknown field"),
    (lambda d: d["privacy"].update(mode="mask_check", laya=False), "laya"),
    (lambda d: d["privacy"].update(rules={"custom": [{"name": "claim", "pattern": "x"}]}), "UPPER_CASE"),
    (lambda d: d["privacy"].update(rules={"custom": [{"name": "CLAIM", "pattern": "("}]}), "valid regular"),
])
def test_invalid_config_is_refused_whole(change, message):
    data = cfg()
    change(data)
    with pytest.raises(ConfigError, match=message):
        parse(data)


def test_keep_private_needs_a_private_upstream():
    with pytest.raises(ConfigError, match="private"):
        parse(cfg(mode="keep_private", private=False))


# --- render --------------------------------------------------------------------------

def test_render_chains_fallbacks_and_fails_closed():
    c = parse(cfg())
    out = render.build(c)
    by_name = {m["model_name"]: m["litellm_params"] for m in out["model_list"]}
    assert by_name["default"]["mock_response"]           # the public name never reaches an upstream
    assert by_name["ir/default/0"]["api_base"] == "https://integrate.api.nvidia.com/v1"
    assert by_name["ir/default/0"]["api_key"] == "os.environ/IR_KEY_NVIDIA"
    assert "extra_body" not in by_name["ir/default/1"]    # hosted API: no reasoning_effort
    assert by_name["ir/default/2"]["extra_body"] == {"reasoning_effort": "none"}  # self-hosted
    assert by_name["ir/default/private/0"]["model"] == "openai/qwen3.8-27b"
    assert {"ir/default/0": ["ir/default/1", "ir/default/2"]} in out["router_settings"]["fallbacks"]
    assert out["litellm_settings"]["num_retries"] == 0


def test_disabled_upstreams_are_left_out():
    data = cfg()
    data["upstreams"][0]["enabled"] = False
    assert render.chains(parse(data))["default"]["all"] == ["ir/default/0", "ir/default/1"]


# --- detection --------------------------------------------------------------------------

def test_custom_rule_keywords_validator_and_exceptions():
    rules = parse(cfg(rules={
        "custom": [{"name": "CLAIM_NUMBER", "pattern": r"CLM-\d{4}-\d{7}", "keywords": ["claim"]},
                   {"name": "CARD_REF", "pattern": r"\b\d{16}\b", "validator": "luhn"}],
        "exceptions": ["support@acme.ie", "@example.org"],
        "builtin": {"phone": False},
    })).privacy.rules
    d = Detector(rules)
    assert d.detect("claim CLM-2026-0048213 please") == ["claim_number"]
    assert d.detect("ref CLM-2026-0048213") == []                       # no keyword nearby
    assert d.detect("card 4111111111111111") == ["card_ref", "payment_card"]
    assert d.detect("ref 4111111111111112") == []                       # Luhn fails
    assert d.detect("mail support@acme.ie or bob@example.org") == []    # exceptions
    assert d.detect("mail ann@acme.ie") == ["email"]
    assert d.detect("call +353 87 123 4567") == []                      # built-in phone switched off


def test_slow_rule_fails_closed():
    rules = parse(cfg(rules={"custom": [{"name": "SLOW", "pattern": r"(a|aa)+b"}]})).privacy.rules
    with pytest.raises(DetectionTimeout):
        Detector(rules).find_spans("a" * 60)
    h = hook(cfg(rules={"custom": [{"name": "SLOW", "pattern": r"(a|aa)+b"}]}))
    data = pre(h, [user("a" * 60)])
    assert data["model"] == "ir/blocked"


# --- hook -----------------------------------------------------------------------------

def test_mask_mode_masks_custom_and_builtin_and_routes_to_the_chain():
    h = hook(cfg(rules={"custom": [{"name": "CLAIM_NUMBER", "pattern": r"CLM-\d{4}-\d{7}"}]}))
    data = pre(h, [user("claim CLM-2026-0048213 for ann@acme.ie")])
    assert data["model"] == "ir/default/0"
    assert "CLM-2026" not in sent(data) and "ann@acme.ie" not in sent(data)
    assert "<CLAIM_NUMBER_" in sent(data)
    headers = asyncio.run(h.async_post_call_response_headers_hook(data, None, None))
    assert headers == {"x-pii-route": "upstream", "x-pii-reason": "masked", "x-router-route": "default",
                       "x-pii-masked": "2"}


def test_reply_restored_and_marked():
    h = hook(cfg())
    data = pre(h, [user("mail ann@acme.ie")])
    token = next(iter(h._maps.values()))[1]
    ph = next(k for k, v in token.items() if v == "ann@acme.ie")
    call = NS(function=NS(arguments=json.dumps({"to": ph})))
    resp = NS(model="x", choices=[NS(message=NS(content=f"to {ph}", reasoning_content=None, tool_calls=[call]))])
    out = asyncio.run(h.async_post_call_success_hook(data, None, resp)).choices[0].message
    assert out.content == "to ann@acme.ie ‡"
    assert json.loads(out.tool_calls[0].function.arguments) == {"to": "ann@acme.ie"}


def test_off_mode_passes_through():
    data = pre(hook(cfg(mode="off")), [user("mail ann@acme.ie")])
    assert data["model"] == "ir/default/0" and "ann@acme.ie" in sent(data)


def test_mask_check_blocks_residual_personal_content():
    data = pre(hook(cfg(mode="mask_check")), [user("Sarah is on sick leave, mail ann@acme.ie")])
    assert data["model"] == "ir/blocked" and data["mock_response"] == BLOCK_MESSAGES[CURRENT_TURN]
    assert "Sarah" not in sent(data)


def test_keep_private_sends_personal_requests_unmasked_to_private_upstreams_only():
    h = hook(cfg(mode="keep_private", laya=True))
    data = pre(h, [user("Sarah is on sick leave, mail ann@acme.ie")])
    assert data["model"] == "ir/default/private/0" and "ann@acme.ie" in sent(data)
    assert pre(h, [user("Explain TCP")])["model"] == "ir/default/0"


def test_keep_private_without_private_target_on_route_blocks():
    extra = ({"name": "public", "targets": [{"upstream": "nvidia", "model": "m"}]},)
    data = pre(hook(cfg(mode="keep_private", extra_routes=extra)), [user("mail ann@acme.ie")], model="public")
    assert data["model"] == "ir/blocked" and "private upstream" in data["mock_response"]


def test_unknown_model_and_non_chat_are_refused():
    h = hook(cfg())
    data = pre(h, [user("hi")], model="gpt-5")
    assert data["model"] == "ir/blocked" and "Available: default" in data["mock_response"]
    assert pre(h, [user("hi")], call_type="aembedding")["model"] == "ir/blocked"


def test_caller_cannot_name_a_chain_deployment_directly():
    data = pre(hook(cfg()), [user("mail ann@acme.ie")], model="ir/default/0")
    assert data["model"] == "ir/blocked"


def test_events_carry_no_content():
    events = []
    h = hook(cfg())
    h._emit = events.append
    data = pre(h, [user("mail ann@acme.ie, card 4111 1111 1111 1111")])
    resp = NS(model="x", choices=[NS(message=NS(content="ok", reasoning_content=None, tool_calls=None))])
    asyncio.run(h.async_post_call_success_hook(data, None, resp))
    assert len(events) == 1
    blob = json.dumps(events)
    assert "ann@acme.ie" not in blob and "4111" not in blob and "<EMAIL_" not in blob
    assert events[0]["masked"] == {"email": 1, "payment_card": 1} and events[0]["outcome"] == "ok"
    assert PLACEHOLDER.search(sent(data))
