"""Anthropic Messages masking proxy: request masking, reply restoring, rules file."""

import json

import pytest

from anthropic_masker.app import (MASKER_NOTE, RequestMasker, RulesFile, StreamRestore, mask_request, parse_rules,
                                  restore_message, with_note)
from inference_router.config import ConfigError
from pii_router.mask import MaskSession

EMAIL = "alice.smith@acme.ie"


def session(rules=None):
    _, detector = parse_rules(rules or {"mode": "mask"})
    return MaskSession(b"test-secret", finder=detector.find_spans)


def request_body():
    return {
        "model": "claude-haiku-5-5",
        "system": [{"type": "text", "text": f"User email: {EMAIL}", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": f"Write {EMAIL} into notes.txt"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": f"I should write {EMAIL} and <EMAIL_deadbeef>", "signature": "sig=="},
                {"type": "tool_use", "id": "t1", "name": "Write",
                 "input": {"file_path": "notes.txt", "content": f"contact: {EMAIL}", "n": 3}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": f"wrote {EMAIL}"}]},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]},
        ],
    }


def test_same_placeholder_everywhere_and_note_added():
    s = session()
    masker = RequestMasker(s)
    body = masker.body(request_body())
    text = json.dumps(body)
    without_signed = json.dumps([body["system"], body["messages"][0], body["messages"][1]["content"][1:],
                                 body["messages"][2]])
    assert EMAIL not in without_signed                                # only the signed thinking block keeps it
    [placeholder] = list(s.mapping)
    assert s.mapping[placeholder] == EMAIL and s.counts == {"email": 1}
    assert body["system"][0]["text"] == f"User email: {placeholder}"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["system"][-1] == {"type": "text", "text": MASKER_NOTE}
    assert body["messages"][0]["content"] == f"Write {placeholder} into notes.txt"
    tool_use = body["messages"][1]["content"][1]
    assert tool_use["input"] == {"file_path": "notes.txt", "content": f"contact: {placeholder}", "n": 3}
    assert body["messages"][2]["content"][0]["content"][0]["text"] == f"wrote {placeholder}"
    assert masker.media == 1 and text.count(placeholder) == 4


def test_signed_thinking_is_byte_identical():
    original = request_body()["messages"][1]["content"][0]
    body = RequestMasker(session()).body(request_body())
    assert body["messages"][1]["content"][0] == original


def test_nothing_masked_means_body_unchanged():
    body = {"model": "m", "system": "Be brief.", "messages": [{"role": "user", "content": "hello"}]}
    assert RequestMasker(session()).body(body) == body


def test_note_on_string_or_missing_system():
    assert with_note(None) == MASKER_NOTE
    assert with_note("Be brief. ") == "Be brief.\n\n" + MASKER_NOTE


def test_custom_rule_and_exception():
    rules = {"mode": "mask", "rules": {"custom": [{"name": "CLAIM_NUMBER", "pattern": "CLM-\\d{4}-\\d{7}"}],
                                       "exceptions": ["@acme-public.com"]}}
    s = session(rules)
    out = s.mask_text("claim CLM-2026-0001234 from bob@acme-public.com")
    assert out.startswith("claim <CLAIM_NUMBER_") and "bob@acme-public.com" in out


def test_restore_non_streaming_text_and_tool_input():
    s = session()
    s.mask_text(EMAIL)
    [ph] = list(s.mapping)
    msg = {"content": [{"type": "thinking", "thinking": ph, "signature": "x"},
                       {"type": "text", "text": f"Done: {ph}"},
                       {"type": "tool_use", "input": {"content": [f"to {ph}"]}}]}
    out = restore_message(msg, s.mapping)
    assert out["content"][0]["thinking"] == ph                      # never restored
    assert out["content"][1]["text"] == f"Done: {EMAIL}"
    assert out["content"][2]["input"] == {"content": [f"to {EMAIL}"]}


def test_foreign_placeholder_is_not_restored():
    s = session()
    s.mask_text(EMAIL)
    msg = {"content": [{"type": "text", "text": "<EMAIL_00000000> stays"}]}
    assert restore_message(msg, s.mapping)["content"][0]["text"] == "<EMAIL_00000000> stays"


def _stream(events, mapping):
    r = StreamRestore(mapping)
    out = []
    for e in events:
        out += r.event(e)
    return out


def test_stream_text_split_across_deltas():
    s = session()
    s.mask_text(EMAIL)
    [ph] = list(s.mapping)
    parts = ["Saved for ", ph[:4], ph[4:12], ph[12:] + "."]
    events = [{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}]
    events += [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": p}} for p in parts]
    events += [{"type": "content_block_stop", "index": 0}]
    out = _stream(events, s.mapping)
    text = "".join(e["delta"]["text"] for e in out if e["type"] == "content_block_delta")
    assert text == f"Saved for {EMAIL}." and out[-1] == {"type": "content_block_stop", "index": 0}


def test_stream_tool_json_split_and_valid():
    s = session()
    s.mask_text('o"neil@acme.ie')       # a value that needs JSON escaping
    [ph] = list(s.mapping)
    full = json.dumps({"file_path": "a.txt", "content": f"to {ph}"})
    parts = [full[:20], full[20:31], full[31:]]
    events = [{"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": p}}
              for p in parts] + [{"type": "content_block_stop", "index": 1}]
    out = _stream(events, s.mapping)
    joined = "".join(e["delta"]["partial_json"] for e in out if e["type"] == "content_block_delta")
    assert json.loads(joined)["content"] == "to " + s.mapping[ph]


def test_held_text_flushed_before_block_stop():
    s = session()
    events = [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "a < b"}},
              {"type": "content_block_stop", "index": 0}]
    out = _stream(events, s.mapping or {"<X_00000000>": "y"})
    assert "".join(e["delta"]["text"] for e in out if e["type"] == "content_block_delta") == "a < b"
    assert out[-1]["type"] == "content_block_stop"


def test_thinking_deltas_untouched():
    e = {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "<EMAIL_1234abcd>"}}
    assert _stream([e], {"<EMAIL_1234abcd>": "x@y.ie"}) == [e]


def test_invalid_rules_refused_and_previous_kept(tmp_path):
    path = tmp_path / "pii-rules.json"
    path.write_text(json.dumps({"mode": "mask"}))
    rules = RulesFile(str(path))
    good = rules.detector
    path.write_text(json.dumps({"mode": "mask", "rules": {"custom": [{"name": "bad", "pattern": "("}]}}))
    import os
    os.utime(path, (1, 2))
    rules.refresh()
    assert rules.detector is good and rules.mode == "mask"
    with pytest.raises(ConfigError):
        parse_rules({"mode": "strict"})


def test_end_to_end_streaming_through_the_app(tmp_path):
    """The whole app against a fake Anthropic: auth, key swap, masked body out, restored SSE back."""
    import httpx
    from starlette.testclient import TestClient
    from anthropic_masker.app import create_app

    rules = tmp_path / "pii-rules.json"
    rules.write_text(json.dumps({"mode": "mask"}))
    seen = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-api-key")
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        ph = seen["body"]["messages"][0]["content"].split()[-1]
        events = [{"type": "message_start", "message": {"id": "m", "content": []}},
                  {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                  {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi " + ph[:6]}},
                  {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": ph[6:]}},
                  {"type": "content_block_stop", "index": 0},
                  {"type": "message_stop"}]
        body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream", "request-id": "req_1"}, text=body)

    env = {"MASKER_KEY": "mk", "ANTHROPIC_API_KEY": "sk-real", "MASK_SECRET": "s", "PII_RULES_FILE": str(rules)}
    app = create_app(env)
    masker = app.routes[0].endpoint.__self__
    masker.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    client = TestClient(app)

    req = {"model": "claude-haiku-5-5", "stream": True, "messages": [{"role": "user", "content": f"say hi to {EMAIL}"}]}
    assert client.post("/v1/messages", json=req, headers={"x-api-key": "wrong"}).status_code == 401
    r = client.post("/v1/messages?beta=true", json=req, headers={"x-api-key": "mk", "anthropic-version": "2023-06-01",
                                                                  "x-claude-gateway-user-email": "alice@oae.test"})
    assert r.status_code == 200 and r.headers["request-id"] == "req_1"
    assert seen["key"] == "sk-real" and seen["url"].endswith("/v1/messages?beta=true")
    assert EMAIL not in json.dumps(seen["body"]) and seen["body"]["system"] == MASKER_NOTE
    text = "".join(json.loads(line[5:])["delta"]["text"] for line in r.text.splitlines()
                   if line.startswith("data:") and '"text_delta"' in line)
    assert text == f"Hi {EMAIL}"
    assert client.get("/v1/models").status_code == 404


def test_value_found_once_is_masked_everywhere():
    """A keyword-gated rule must not leak the value where the keyword is absent (spike leak)."""
    rules = {"mode": "mask", "rules": {"custom": [{"name": "CLAIM_NUMBER", "pattern": "CLM-\\d{4}-\\d{7}",
                                                   "keywords": ["claim"]}]}}
    _, detector = parse_rules(rules)
    body = {"model": "m", "messages": [
        {"role": "assistant", "content": [{"type": "text", "text": "Stored as CLM-2026-0007777 in memory."}]},
        {"role": "user", "content": "her claim is CLM-2026-0007777"}]}
    masked, session, _ = mask_request(b"k", detector, body)
    assert "CLM-2026-0007777" not in json.dumps(masked)
    [token] = list(session.mapping)
    assert token.startswith("<CLAIM_NUMBER_") and json.dumps(masked).count(token) == 2
    # the same token as a single pass would give, so placeholders stay stable across turns
    single = MaskSession(b"k", finder=detector.find_spans)
    single.mask_text("claim CLM-2026-0007777")
    assert list(single.mapping) == [token]


def test_note_never_says_the_model_cannot_see():
    assert "cannot see" not in MASKER_NOTE and "placeholder" in MASKER_NOTE.lower()
