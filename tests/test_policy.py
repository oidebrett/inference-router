import asyncio
import json
import logging

import pytest

from pii_router.extract import chunks, message_units
from pii_router.hook import PiiRouterHook
from pii_router.policy import CLOUD, LOCAL, Router, Settings


class FakeLaya:
    """Scores text by keyword so tests do not need the real model."""

    def __init__(self, pii_words=("salary", "diagnosed", "sick leave"), fail=False):
        self.pii_words = pii_words
        self.fail = fail
        self.calls = []

    def score(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise ConnectionError("laya-serve down")
        return [0.9 if any(w in t.lower() for w in self.pii_words) else 0.05 for t in texts]


def user(text):
    return {"role": "user", "content": text}


def test_clean_request_goes_to_cloud():
    d = Router(FakeLaya()).decide([{"role": "system", "content": "You are helpful."}, user("Explain TCP vs UDP")])
    assert d.route == CLOUD and d.reasons == []


def test_structured_pii_goes_local_without_asking_laya():
    laya = FakeLaya()
    d = Router(laya).decide([user("email ann@acme.ie about the bill")])
    assert d.route == LOCAL and "email" in d.reasons
    assert laya.calls == []  # detectors settled it


def test_semantic_pii_goes_local_via_laya():
    d = Router(FakeLaya()).decide([user("Why was John's salary late?")])
    assert d.route == LOCAL and d.reasons == ["laya"]


def test_system_prompt_is_not_scanned():
    d = Router(FakeLaya()).decide([{"role": "system", "content": "Contact admin@acme.ie for help"}, user("hi")])
    assert d.route == CLOUD


def test_laya_failure_fails_closed():
    d = Router(FakeLaya(fail=True)).decide([user("Explain TCP vs UDP")])
    assert d.route == LOCAL and d.reasons == ["laya_unavailable"]


def test_laya_failure_is_not_cached_as_clean():
    laya = FakeLaya(fail=True)
    router = Router(laya)
    msgs = [user("Explain TCP vs UDP")]
    assert router.decide(msgs).route == LOCAL
    laya.fail = False
    assert router.decide(msgs).route == CLOUD  # re-scanned once Laya is back


def test_pii_is_sticky_for_the_conversation():
    router = Router(FakeLaya())
    history = [user("Plan Priya's return from sick leave")]
    assert router.decide(history).route == LOCAL
    history += [{"role": "assistant", "content": "Here is a plan."}, user("Now summarise that in 3 bullets")]
    d = router.decide(history)
    assert d.route == LOCAL and "sticky" in d.reasons
    assert d.cached_units == 1


def test_history_is_scanned_once():
    laya = FakeLaya()
    router = Router(laya)
    history = [user("Explain TCP vs UDP")]
    router.decide(history)
    history += [{"role": "assistant", "content": "TCP is reliable."}, user("And QUIC?")]
    router.decide(history)
    assert [len(c) for c in laya.calls] == [1, 2]  # second call only the 2 new messages


def test_tool_results_and_tool_call_arguments_are_scanned():
    router = Router(FakeLaya())
    tool_result = [user("read the file"), {"role": "tool", "tool_call_id": "1", "content": "card 4111 1111 1111 1111"}]
    assert router.decide(tool_result).route == LOCAL
    call = [user("send it"), {"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "send_email", "arguments": '{"to": "ann@acme.ie"}'}}]}]
    assert Router(FakeLaya()).decide(call).route == LOCAL


def test_images_fail_closed_by_default():
    msg = {"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}
    assert Router(FakeLaya()).decide([msg]).reasons == ["opaque_media"]
    assert Router(FakeLaya(), Settings(media_route=CLOUD)).decide([msg]).route == CLOUD


def test_oversize_routes_local_and_partial_scan_is_not_cached_clean():
    router = Router(FakeLaya(), Settings(max_chunks=2, chunk_chars=100))
    big = user("benign text " * 200)  # ~2400 chars -> many chunks
    d = router.decide([big])
    assert d.route == LOCAL and "oversize_unscanned" in d.reasons
    assert router.cache.get(message_units([big])[0].key) is None


def test_chunks_overlap_and_cover_everything():
    text = "".join(str(i % 10) for i in range(5000))
    parts = chunks(text, 1800, 200)
    assert parts[0] == text[:1800] and parts[-1].endswith(text[-50:])
    assert all(len(p) <= 1800 for p in parts)


def test_settings_from_env_rejects_bad_values():
    s = Settings.from_env({"PII_ROUTER_LAYA_THRESHOLD": "nope", "PII_ROUTER_OVERSIZE_ROUTE": "sideways"})
    assert s.threshold == 0.4 and s.oversize_route == LOCAL


# --- LiteLLM hook ---------------------------------------------------------

def _run(hook, data, call_type="acompletion"):
    return asyncio.run(hook.async_pre_call_hook(None, None, data, call_type))


def test_hook_routes_and_logs_without_content(caplog):
    hook = PiiRouterHook(router=Router(FakeLaya()), env={})
    logging.getLogger("pii_router").propagate = True
    with caplog.at_level(logging.INFO, logger="pii_router"):
        out = _run(hook, {"model": "pii-router", "messages": [user("email ann@acme.ie")]})
    assert out["model"] == "ollama-local"
    line = caplog.records[-1].getMessage()
    assert "ann@acme.ie" not in line and "email ann" not in line
    payload = json.loads(line.split(" ", 1)[1])
    assert payload["route"] == LOCAL and payload["target"] == "ollama-local"
    logging.getLogger("pii_router").propagate = False


def test_hook_cannot_be_bypassed_by_naming_the_cloud_model():
    hook = PiiRouterHook(router=Router(FakeLaya()), env={})
    out = _run(hook, {"model": "nvidia-cloud", "messages": [user("email ann@acme.ie")]})
    assert out["model"] == "ollama-local"


def test_hook_sends_clean_traffic_to_cloud():
    hook = PiiRouterHook(router=Router(FakeLaya()), env={})
    assert _run(hook, {"model": "pii-router", "messages": [user("Explain TCP")]})["model"] == "nvidia-cloud"


def test_hook_fails_closed_on_router_bug():
    class Broken:
        def decide(self, messages):
            raise RuntimeError("boom")
    hook = PiiRouterHook(router=Broken(), env={})
    assert _run(hook, {"model": "pii-router", "messages": [user("Explain TCP")]})["model"] == "ollama-local"


@pytest.mark.parametrize("call_type", ["aembedding", "aresponses"])
def test_unsupported_call_types_never_reach_the_cloud(call_type):
    hook = PiiRouterHook(router=Router(FakeLaya()), env={})
    assert _run(hook, {"model": "nvidia-cloud", "input": "x"}, call_type)["model"] == "ollama-local"
