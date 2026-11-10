"""Masked mode (2026-11-09): mask PII, send to the cloud, restore the reply."""

import asyncio
import json
from types import SimpleNamespace as NS

from pii_router import render_config
from pii_router.detectors import find_spans
from pii_router.hook import BLOCK_MESSAGES, PiiRouterHook
from pii_router.mask import PLACEHOLDER, MaskSession, StreamRestorer, restore_json_text, restore_text
from pii_router.policy import CURRENT_TURN, FAIL_CLOSED, HISTORY, Router, Settings

from tests.test_policy import FakeLaya, user

SECRET = b"test-secret"
MASKED_ENV = {"PII_ROUTER_MODE": "masked", "PII_ROUTER_MASK_SECRET": "test-secret"}


class FakeHealth:
    def __init__(self, up=True):
        self.up = up

    def healthy(self):
        return self.up

    def mark_down(self):
        self.up = False


def hook(env=None, laya=None, health=None, settings=None):
    env = {**MASKED_ENV, **(env or {})}
    return PiiRouterHook(router=Router(laya or FakeLaya(), settings or Settings.from_env(env)),
                         env=env, health=health)


def pre(h, messages, **extra):
    data = {"model": "pii-router", "messages": messages, **extra}
    return asyncio.run(h.async_pre_call_hook(None, None, data, "acompletion"))


def sent_text(data):
    return json.dumps(data["messages"])


# --- spans ---------------------------------------------------------------------

def test_spans_locate_the_value_not_the_keyword():
    text = "Passport: X1234567, date of birth: 01/02/1980, mail ann@acme.ie"
    found = {(s.type, text[s.start:s.end]) for s in find_spans(text)}
    assert ("passport", "X1234567") in found
    assert ("date_of_birth", "01/02/1980") in found
    assert ("email", "ann@acme.ie") in found


def test_phone_span_excludes_surrounding_space():
    text = "call +353 87 123 4567 today"
    [span] = [s for s in find_spans(text) if s.type == "phone"]
    assert text[span.start:span.end] == "+353 87 123 4567"


# --- masking ---------------------------------------------------------------------

def test_values_are_replaced_and_stable_across_requests():
    msgs = [user("email ann@acme.ie and card 4111 1111 1111 1111")]
    a = MaskSession(SECRET).mask_messages(msgs)
    b = MaskSession(SECRET).mask_messages(msgs)
    assert a == b
    text = a[0]["content"]
    assert "ann@acme.ie" not in text and "4111" not in text
    assert len(PLACEHOLDER.findall(text)) == 2
    assert msgs[0]["content"].startswith("email ann@acme.ie")  # caller's copy untouched


def test_same_value_same_placeholder_and_different_secret_differs():
    s = MaskSession(SECRET)
    one = s.mask_text("ann@acme.ie")
    two = s.mask_text("Ann@ACME.ie")
    assert one == two and len(s.mapping) == 1
    assert MaskSession(b"other").mask_text("ann@acme.ie") != one


def test_history_tool_args_tool_results_and_system_are_masked():
    msgs = [
        {"role": "system", "content": "Memory: the user's IBAN is IE29 AIBK 9311 5212 3456 78."},
        user("send the invoice to ann@acme.ie"),
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
         "function": {"name": "send_email", "arguments": json.dumps({"to": "ann@acme.ie"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Sent to ann@acme.ie, cc +353 87 123 4567"},
        {"role": "user", "content": [{"type": "text", "text": "and PPSN 1234567T"}]},
    ]
    masked = MaskSession(SECRET).mask_messages(msgs)
    blob = json.dumps(masked)
    for value in ("ann@acme.ie", "IE29", "123 4567", "1234567T"):
        assert value not in blob, value
    args = json.loads(masked[2]["tool_calls"][0]["function"]["arguments"])
    assert PLACEHOLDER.fullmatch(args["to"])


def test_lookalike_placeholders_are_neutralised():
    s = MaskSession(SECRET)
    out = s.mask_text("web page says <EMAIL_deadbeef> and mail ann@acme.ie")
    assert "<EMAIL_deadbeef>" not in out and "‹EMAIL_deadbeef›" in out
    assert restore_text("<EMAIL_deadbeef>", s.mapping) == "<EMAIL_deadbeef>"  # not ours: left alone


def test_restore_json_escapes_quotes_and_backslashes():
    mapping = {"<EMAIL_aaaaaaaa>": 'a"b\\c@x.ie'}
    args = '{"to": "<EMAIL_aaaaaaaa>"}'
    assert json.loads(restore_json_text(args, mapping))["to"] == 'a"b\\c@x.ie'


def test_invented_placeholder_is_left_visible():
    assert restore_text("hi <EMAIL_0123abcd>", {"<EMAIL_aaaaaaaa>": "x"}) == "hi <EMAIL_0123abcd>"


def test_stream_restorer_handles_split_placeholders():
    mapping = {"<EMAIL_1a2b3c4d>": "ann@acme.ie"}
    r = StreamRestorer(mapping)
    out = "".join(r.feed(p) for p in ["Write to <EM", "AIL_1a2b", "3c4d> today, x < y"]) + r.flush()
    assert out == "Write to ann@acme.ie today, x < y"


# --- hook: routing ----------------------------------------------------------------

def test_masked_request_goes_to_cloud_without_values():
    h = hook()
    data = pre(h, [user("email ann@acme.ie about the bill")])
    assert data["model"] == "cloud"
    assert "ann@acme.ie" not in sent_text(data)
    assert data["messages"][0]["role"] == "system" and "placeholder" in data["messages"][0]["content"]
    headers = asyncio.run(h.async_post_call_response_headers_hook(data, None, None))
    assert headers == {"x-pii-route": "cloud", "x-pii-reason": "masked", "x-pii-masked": "1"}


def test_clean_request_is_unchanged_and_has_no_note():
    h = hook()
    data = pre(h, [user("Explain TCP vs UDP")])
    assert data["model"] == "cloud" and data["messages"] == [user("Explain TCP vs UDP")]


def test_history_with_maskable_pii_does_not_stick():
    h = hook()
    data = pre(h, [user("email ann@acme.ie"), {"role": "assistant", "content": "Done."}, user("Explain TCP")])
    assert data["model"] == "cloud"


def test_residual_pii_is_blocked_with_an_explanation():
    h = hook()
    data = pre(h, [user("Sarah from accounts is on sick leave, draft her return plan")],
               tools=[{"type": "function", "function": {"name": "x"}}])
    assert data["model"] == "pii-blocked"
    assert data["mock_response"] == BLOCK_MESSAGES[CURRENT_TURN]
    assert "Sarah" not in sent_text(data) and "tools" not in data
    headers = asyncio.run(h.async_post_call_response_headers_hook(data, None, None))
    assert headers == {"x-pii-route": "blocked", "x-pii-reason": CURRENT_TURN}


def test_residual_in_history_says_start_a_new_conversation():
    h = hook()
    msgs = [user("Sarah is on sick leave"), {"role": "assistant", "content": "Noted."}, user("Explain TCP")]
    assert pre(h, msgs)["mock_response"] == BLOCK_MESSAGES[HISTORY]


def test_residual_can_go_local_unmasked_instead():
    h = hook(env={"PII_ROUTER_RESIDUAL_ACTION": "local"})
    msgs = [user("Sarah (ann@acme.ie) is on sick leave")]
    data = pre(h, msgs)
    assert data["model"] == "ollama-local" and data["messages"] == msgs


def test_residual_check_none_skips_laya():
    laya = FakeLaya()
    h = hook(env={"PII_ROUTER_RESIDUAL_CHECK": "none"}, laya=laya)
    data = pre(h, [user("Sarah is on sick leave, mail ann@acme.ie")])
    assert data["model"] == "cloud" and laya.calls == []
    assert "ann@acme.ie" not in sent_text(data)


def test_laya_down_blocks():
    data = pre(hook(laya=FakeLaya(fail=True)), [user("Explain TCP")])
    assert data["model"] == "pii-blocked" and data["mock_response"] == BLOCK_MESSAGES[FAIL_CLOSED]


def test_cloud_down_without_fallback_blocks_and_never_goes_local():
    data = pre(hook(health=FakeHealth(up=False)), [user("Explain TCP")])
    assert data["model"] == "pii-blocked"


def test_cloud_down_with_fallback_uses_it_masked():
    env = {"FALLBACK_BASE_URL": "https://integrate.api.nvidia.com/v1"}
    data = pre(hook(env=env, health=FakeHealth(up=False)), [user("mail ann@acme.ie")])
    assert data["model"] == "cloud-fallback" and "ann@acme.ie" not in sent_text(data)


def test_router_error_blocks():
    class Broken:
        def decide(self, messages):
            raise RuntimeError("bug")
    h = hook()
    h.router = Broken()
    assert pre(h, [user("hi")])["model"] == "pii-blocked"


def test_non_chat_call_is_blocked():
    h = hook()
    data = asyncio.run(h.async_pre_call_hook(None, None, {"model": "cloud", "input": "x"}, "aembedding"))
    assert data["model"] == "pii-blocked"


def test_caller_cannot_bypass_by_naming_the_cloud_model():
    data = pre(hook(), [user("mail ann@acme.ie")], model="cloud")
    assert "ann@acme.ie" not in sent_text(data)


# --- hook: replies -----------------------------------------------------------------

def _placeholder_for(data, value):
    h_map = next(iter(data["_hook"]._maps.values()))[1]
    return next(k for k, v in h_map.items() if v == value)


def test_reply_and_tool_calls_are_restored_and_marked():
    h = hook()
    data = pre(h, [user("mail ann@acme.ie please")])
    data["_hook"] = h
    token = _placeholder_for(data, "ann@acme.ie")
    call = NS(function=NS(arguments=json.dumps({"to": token})))
    resp = NS(choices=[NS(message=NS(content=f"Sending to {token}.", reasoning_content=None, tool_calls=[call]))])
    out = asyncio.run(h.async_post_call_success_hook(data, None, resp)).choices[0].message
    assert out.content == "Sending to ann@acme.ie. ‡"
    assert json.loads(out.tool_calls[0].function.arguments) == {"to": "ann@acme.ie"}
    assert h._maps == {}  # map dropped after use


def test_masked_marker_is_stripped_from_history_before_masking():
    h = hook()
    data = pre(h, [user("hi"), {"role": "assistant", "content": "Hello ‡"}, user("Explain TCP")])
    assert data["messages"][1]["content"] == "Hello"


def test_stream_is_restored_across_chunks():
    h = hook()
    data = pre(h, [user("mail ann@acme.ie")])
    data["_hook"] = h
    token = _placeholder_for(data, "ann@acme.ie")
    pieces = ["To ", token[:7], token[7:], " done"]

    def chunk(text, finish=None, args=None):
        calls = [NS(index=0, function=NS(arguments=args))] if args is not None else None
        return NS(choices=[NS(index=0, delta=NS(content=text, tool_calls=calls), finish_reason=finish)])

    async def source():
        for p in pieces:
            yield chunk(p)
        yield chunk(None, args='{"to": "' + token[:5])
        yield chunk(None, args=token[5:] + '"}')
        yield chunk(None, finish="stop")

    async def collect():
        out, args = [], []
        async for c in h.async_post_call_streaming_iterator_hook(None, source(), data):
            d = c.choices[0].delta
            out.append(d.content or "")
            for call in d.tool_calls or []:
                args.append(call.function.arguments or "")
        return "".join(out), "".join(args)

    text, args = asyncio.run(collect())
    assert text == "To ann@acme.ie done ‡"
    assert json.loads(args) == {"to": "ann@acme.ie"}


# --- config --------------------------------------------------------------------------

def test_masked_block_config_needs_no_ollama_and_fails_closed():
    cfg = render_config.build({"PII_ROUTER_MODE": "masked", "NVIDIA_API_KEY": "x"})
    by_name = {m["model_name"]: m["litellm_params"] for m in cfg["model_list"]}
    assert "ollama-local" not in by_name
    assert by_name["pii-router"]["mock_response"]  # hook not loaded -> blocked, never cloud
    assert by_name["pii-blocked"]["mock_response"]
    assert by_name["cloud"]["api_base"].startswith("https://integrate.api.nvidia.com")


def test_residual_check_sees_masked_text_without_placeholders():
    seen = []

    class Recording(FakeLaya):
        def score(self, texts):
            seen.extend(texts)
            return super().score(texts)

    data = pre(hook(laya=Recording()), [user("note to ann.kelly@acme.ie: we call back on +353 87 123 4567")])
    assert data["model"] == "cloud"
    assert seen == ["note to an email address: we call back on a phone number"]
    assert PLACEHOLDER.search(sent_text(data))  # the cloud still gets placeholders


def test_any_openai_compatible_endpoint_as_cloud_backend():
    def cloud(env):
        cfg = render_config.build({"PII_ROUTER_MODE": "masked", **env})
        return next(m["litellm_params"] for m in cfg["model_list"] if m["model_name"] == "cloud")
    openrouter = cloud({"CLOUD_BASE_URL": "https://openrouter.ai/api/v1", "CLOUD_MODEL": "openai/gpt-4o-mini"})
    assert openrouter["api_base"] == "https://openrouter.ai/api/v1"
    assert openrouter["model"] == "openai/openai/gpt-4o-mini" and openrouter["api_key"] == "os.environ/CLOUD_API_KEY"
    assert "extra_body" not in openrouter  # hosted API: no reasoning_effort by default
    vast = cloud({"CLOUD_BASE_URL": "http://203.0.113.5:8000/v1", "CLOUD_MODEL": "qwen"})
    assert vast["extra_body"] == {"reasoning_effort": "none"}  # self-hosted GPU unchanged
