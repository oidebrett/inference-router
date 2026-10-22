"""Local-route reasons, markers, headers, marker stripping and cloud fallback (2026-10-22)."""

import asyncio
from types import SimpleNamespace as NS

from pii_router import render_config
from pii_router.health import CloudHealth
from pii_router.hook import PiiRouterHook
from pii_router.policy import CURRENT_TURN, FAIL_CLOSED, HISTORY, Router

from tests.test_policy import FakeLaya, user


def assistant(text):
    return {"role": "assistant", "content": text}


class FakeHealth:
    def __init__(self, up=True):
        self.up = up
        self.marked_down = False

    def healthy(self):
        return self.up

    def mark_down(self):
        self.marked_down = True
        self.up = False


def hook(env=None, health=None, laya=None):
    return PiiRouterHook(router=Router(laya or FakeLaya()), env=env or {}, health=health)


def run(h, messages, call_type="acompletion"):
    return asyncio.run(h.async_pre_call_hook(None, None, {"model": "pii-router", "messages": messages}, call_type))


def reply(h, data, text="ok"):
    resp = NS(choices=[NS(message=NS(content=text, tool_calls=None))])
    return asyncio.run(h.async_post_call_success_hook(data, None, resp)).choices[0].message.content


def headers(h, data):
    return asyncio.run(h.async_post_call_response_headers_hook(data, None, None))


FALLBACK_ENV = {"FALLBACK_BASE_URL": "https://integrate.api.nvidia.com/v1"}


# --- reasons + markers --------------------------------------------------------

def test_pii_in_this_turn_is_current_turn():
    d = Router(FakeLaya()).decide([user("email ann@acme.ie")])
    assert d.local_reason == CURRENT_TURN


def test_pii_only_in_history_says_history():
    router = Router(FakeLaya())
    msgs = [user("email ann@acme.ie"), assistant("Done."), user("Explain TCP")]
    d = router.decide(msgs)
    assert d.route == "local" and d.local_reason == HISTORY


def test_agent_loop_tool_result_after_pii_user_turn_is_still_current_turn():
    msgs = [user("email ann@acme.ie the invoice"),
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "invoice 42 is ready"}]
    assert Router(FakeLaya()).decide(msgs).local_reason == CURRENT_TURN


def test_laya_down_is_fail_closed_even_with_pii_history():
    laya = FakeLaya()
    router = Router(laya)
    router.decide([user("email ann@acme.ie")])
    laya.fail = True
    d = router.decide([user("email ann@acme.ie"), assistant("Done."), user("Explain TCP")])
    assert d.local_reason == FAIL_CLOSED  # a fresh context would not go to the cloud either


def test_image_in_this_turn_is_fail_closed():
    msg = {"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                         {"type": "image_url", "image_url": {"url": "data:,"}}]}
    assert Router(FakeLaya()).decide([msg]).local_reason == FAIL_CLOSED


def test_markers_per_reason():
    h = hook()
    assert reply(h, run(h, [user("email ann@acme.ie")]), "Hi") == "Hi †"
    hist = [user("email bob@acme.ie"), assistant("Done."), user("Explain TCP")]
    assert reply(h, run(h, hist), "TCP") == "TCP ††"
    hl = hook(laya=FakeLaya(fail=True))
    assert reply(hl, run(hl, [user("Explain TCP")]), "TCP") == "TCP †!"
    assert reply(h, run(h, [user("Explain UDP")]), "UDP") == "UDP"


def test_markers_are_stripped_from_history_before_routing_and_forwarding():
    h = hook()
    msgs = [user("Explain TCP"), assistant("TCP is reliable ††"), user("And UDP?")]
    data = run(h, msgs)
    assert data["messages"][1]["content"] == "TCP is reliable"
    assert data["model"] == "cloud"
    parts = [user("hi"), {"role": "assistant", "content": [{"type": "text", "text": "Hello †"}]}, user("Explain TCP")]
    assert run(h, parts)["messages"][1]["content"][0]["text"] == "Hello"


def test_user_text_ending_in_dagger_is_not_stripped():
    data = run(hook(), [user("footnote †")])
    assert data["messages"][0]["content"] == "footnote †"


def test_empty_local_marker_disables_all_markers():
    h = hook(env={"PII_ROUTER_LOCAL_MARKER": ""})
    hist = [user("email bob@acme.ie"), assistant("Done."), user("Explain TCP")]
    assert reply(h, run(h, hist), "TCP") == "TCP"
    assert headers(h, run(h, hist)) == {"x-pii-route": "local", "x-pii-reason": "history"}


# --- headers --------------------------------------------------------------------

def test_headers_report_route_and_reason():
    h = hook()
    assert headers(h, run(h, [user("email ann@acme.ie")])) == {"x-pii-route": "local", "x-pii-reason": "current-turn"}
    assert headers(h, run(h, [user("Explain TCP")])) == {"x-pii-route": "cloud", "x-pii-reason": "clean"}
    h2 = hook(env=FALLBACK_ENV, health=FakeHealth(up=False))
    assert headers(h2, run(h2, [user("Explain TCP")])) == {"x-pii-route": "fallback", "x-pii-reason": "clean"}


def test_unsupported_call_type_headers_say_local():
    h = hook()
    data = asyncio.run(h.async_pre_call_hook(None, None, {"model": "cloud", "input": "x"}, "aembedding"))
    assert data["model"] == "ollama-local"
    assert headers(h, data) == {"x-pii-route": "local", "x-pii-reason": "fail-closed"}


# --- fallback ---------------------------------------------------------------------

def test_cloud_down_sends_clean_traffic_to_fallback():
    h = hook(env=FALLBACK_ENV, health=FakeHealth(up=False))
    assert run(h, [user("Explain TCP")])["model"] == "cloud-fallback"


def test_cloud_down_without_fallback_goes_local_fail_closed():
    h = hook(health=FakeHealth(up=False))
    data = run(h, [user("Explain TCP")])
    assert data["model"] == "ollama-local"
    assert reply(h, data, "TCP") == "TCP †!"


def test_pii_never_goes_to_fallback():
    h = hook(env=FALLBACK_ENV, health=FakeHealth(up=False))
    assert run(h, [user("email ann@acme.ie")])["model"] == "ollama-local"


def test_cloud_healthy_uses_cloud_and_no_health_means_always_cloud():
    assert run(hook(env=FALLBACK_ENV, health=FakeHealth(up=True)), [user("Explain TCP")])["model"] == "cloud"
    assert run(hook(), [user("Explain TCP")])["model"] == "cloud"  # AgentGateway: no health URL


def test_fallback_and_legacy_names_cannot_be_used_to_bypass():
    h = hook(env=FALLBACK_ENV, health=FakeHealth())
    for name in ("cloud-fallback", "nvidia-cloud", "cloud"):
        data = {"model": name, "messages": [user("email ann@acme.ie")]}
        assert asyncio.run(h.async_pre_call_hook(None, None, data, "acompletion"))["model"] == "ollama-local"


def test_failed_cloud_call_marks_cloud_down():
    health = FakeHealth()
    h = hook(env=FALLBACK_ENV, health=health)
    data = run(h, [user("Explain TCP")])
    asyncio.run(h.async_post_call_failure_hook(data, ConnectionError("gpu gone"), None))
    assert health.marked_down
    assert run(h, [user("Explain UDP")])["model"] == "cloud-fallback"


def test_failed_local_call_does_not_mark_cloud_down():
    health = FakeHealth()
    h = hook(env=FALLBACK_ENV, health=health)
    data = run(h, [user("email ann@acme.ie")])
    asyncio.run(h.async_post_call_failure_hook(data, ConnectionError("ollama"), None))
    assert not health.marked_down


# --- CloudHealth --------------------------------------------------------------------

def test_health_probe_is_cached_for_the_interval():
    calls, now = [], [0.0]
    health = CloudHealth("http://gpu/v1/models", interval=15, probe=lambda: calls.append(1) or True,
                         clock=lambda: now[0])
    assert health.healthy() and health.healthy()
    assert len(calls) == 1
    now[0] = 16
    health.healthy()
    assert len(calls) == 2


def test_mark_down_holds_until_the_next_probe():
    now = [0.0]
    health = CloudHealth("http://gpu/v1/models", interval=15, probe=lambda: True, clock=lambda: now[0])
    assert health.healthy()
    health.mark_down()
    assert not health.healthy()
    now[0] = 20
    assert health.healthy()


def test_health_from_env():
    assert CloudHealth.from_env({}) is None
    assert CloudHealth.from_env({"FALLBACK_BASE_URL": "x"}) is None  # NVIDIA-only: nothing to probe
    h = CloudHealth.from_env({"FALLBACK_BASE_URL": "x", "CLOUD_BASE_URL": "https://gpu.example/v1/",
                              "CLOUD_API_KEY": "k"})
    assert h.url == "https://gpu.example/v1/models" and h.api_key == "k"
    assert CloudHealth.from_env({"PII_ROUTER_CLOUD_HEALTH_URL": "http://h/x"}).url == "http://h/x"


def test_unreachable_health_url_is_unhealthy():
    assert CloudHealth("http://127.0.0.1:9/v1/models", timeout=0.5).healthy() is False


# --- render_config ---------------------------------------------------------------------

def _names(cfg):
    return {m["model_name"]: m["litellm_params"] for m in cfg["model_list"]}


def test_render_default_is_nvidia_without_fallback():
    m = _names(render_config.build({"OLLAMA_MODEL": "qwen3.5:4b"}))
    assert m["cloud"]["api_base"] == "https://integrate.api.nvidia.com/v1"
    assert m["cloud"]["api_key"] == "os.environ/NVIDIA_API_KEY"
    assert "cloud-fallback" not in m
    assert m["pii-router"] == m["ollama-local"]  # public name stays local


def test_render_vast_cloud_with_nvidia_fallback():
    m = _names(render_config.build({
        "OLLAMA_MODEL": "qwen3.5:4b", "CLOUD_BASE_URL": "https://gpu.ex/v1/", "CLOUD_MODEL": "qwen3.8-27b",
        "FALLBACK_BASE_URL": "https://integrate.api.nvidia.com/v1", "FALLBACK_MODEL": "nvidia/x"}))
    assert m["cloud"] == {"model": "openai/qwen3.8-27b", "api_base": "https://gpu.ex/v1",
                          "api_key": "os.environ/CLOUD_API_KEY"}
    assert m["cloud-fallback"]["api_key"] == "os.environ/FALLBACK_API_KEY"


def test_render_requires_models_for_custom_backends():
    import pytest
    with pytest.raises(SystemExit):
        render_config.build({"OLLAMA_MODEL": "m", "CLOUD_BASE_URL": "https://gpu/v1"})
    with pytest.raises(SystemExit):
        render_config.build({"OLLAMA_MODEL": "m", "FALLBACK_BASE_URL": "https://n/v1"})
