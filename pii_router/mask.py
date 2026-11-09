"""Masked mode: replace detected PII with placeholders before a request leaves the
box, and put the real values back into the reply.

Placeholders look like ``<EMAIL_7f3a2c1b>``: the type plus the first 8 hex chars
of HMAC-SHA256(secret, normalised value). That makes them

* stable across turns without storing anything: agents re-send the whole
  conversation (with the real values) every turn, so each request rebuilds its
  own map, and the cloud model sees the same placeholder for the same value;
* stable for the provider's prompt cache, because the masked history does not
  change between turns;
* unguessable without the deployment secret.

Only placeholders minted for THIS request are ever restored. Text that already
looks like a placeholder (from a user, a web page, a tool result) is neutralised
first (``<`` -> ``‹``), so outside text cannot make the router splice a real value
into the reply.

The map (placeholder -> real value) lives only in memory for the duration of one
request and is never logged.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field

from .detectors import Span, find_spans

PLACEHOLDER = re.compile(r"<([A-Z][A-Z_]*_[0-9a-f]{8,16})>")
_LOOKALIKE = re.compile(r"<([A-Z][A-Z_]*_[0-9a-f]{8,16})>")
_ESCAPED_OPEN = "‹"   # ‹
_ESCAPED_CLOSE = "›"  # ›

# Roles whose text is masked. System/developer prompts are masked too: agent
# memory (e.g. OpenClaw MEMORY.md) can carry PII from earlier sessions there.
MASKED_ROLES = {"system", "developer", "user", "assistant", "tool", "function"}

SYSTEM_NOTE = (
    "Some values in this conversation have been replaced with placeholders such as "
    "<EMAIL_1a2b3c4d>. Each placeholder stands for one real value that you cannot see. "
    "Wherever that value is needed, including in tool-call arguments, write the placeholder "
    "exactly as it appears, character for character. Never invent new placeholders, never "
    "change one, and never try to guess the hidden value."
)


def _normalise(kind: str, value: str) -> str:
    if kind == "email":
        return value.strip().lower()
    # Numbers and codes: the same identifier written with different spacing or
    # separators gets the same placeholder.
    return re.sub(r"[\s.\-/()]", "", value).upper()


def _merge(spans: list[Span]) -> list[Span]:
    """Overlapping spans become their union, so no partial value survives."""
    merged: list[Span] = []
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        if merged and span.start < merged[-1].end:
            last = merged[-1]
            if span.end > last.end:
                merged[-1] = Span(last.type, last.start, span.end)
            continue
        merged.append(span)
    return merged


@dataclass
class MaskSession:
    """Masks every string of one request with a shared map."""

    secret: bytes
    mapping: dict[str, str] = field(default_factory=dict)   # placeholder -> real value
    counts: dict[str, int] = field(default_factory=dict)    # type -> values masked (for the log)
    _by_value: dict[tuple[str, str], str] = field(default_factory=dict)

    def _placeholder(self, kind: str, value: str) -> str:
        key = (kind, _normalise(kind, value))
        existing = self._by_value.get(key)
        if existing:
            return existing
        digest = hmac.new(self.secret, f"{kind}\x00{key[1]}".encode("utf-8"), hashlib.sha256).hexdigest()
        for size in (8, 12, 16):
            token = f"<{kind.upper()}_{digest[:size]}>"
            if token not in self.mapping:
                break
        self.mapping[token] = value
        self._by_value[key] = token
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return token

    def mask_text(self, text: str) -> str:
        if not text:
            return text
        text = _LOOKALIKE.sub(lambda m: f"{_ESCAPED_OPEN}{m.group(1)}{_ESCAPED_CLOSE}", text)
        spans = _merge(find_spans(text))
        if not spans:
            return text
        out, cursor = [], 0
        for span in spans:
            out.append(text[cursor:span.start])
            out.append(self._placeholder(span.type, text[span.start:span.end]))
            cursor = span.end
        out.append(text[cursor:])
        return "".join(out)

    # --- request side ------------------------------------------------------

    def _mask_content(self, content):
        if isinstance(content, str):
            return self.mask_text(content)
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(self.mask_text(part))
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append({**part, "text": self.mask_text(part["text"])})
                else:
                    parts.append(part)
            return parts
        return content

    def mask_messages(self, messages) -> list:
        """Return a masked deep copy; the caller's messages are not changed."""
        masked = copy.deepcopy(list(messages or []))
        for message in masked:
            if not isinstance(message, dict) or message.get("role") not in MASKED_ROLES:
                continue
            if "content" in message:
                message["content"] = self._mask_content(message["content"])
            for call in message.get("tool_calls") or []:
                function = (call or {}).get("function")
                if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                    # Arguments are JSON text. A placeholder contains no quote or
                    # backslash, so swapping a value inside a JSON string keeps it valid.
                    function["arguments"] = self.mask_text(function["arguments"])
            if isinstance(message.get("function_call"), dict):
                fc = message["function_call"]
                if isinstance(fc.get("arguments"), str):
                    fc["arguments"] = self.mask_text(fc["arguments"])
        return masked


_CHECK_STRIP = re.compile(r"[ \t]*<[A-Z][A-Z_]*_[0-9a-f]{8,16}>")


def for_residual_check(masked_messages: list) -> list:
    """The masked messages with placeholders REMOVED, for the residual (Laya) check.

    The question there is "is anything personal left besides the identifiers we
    already hid?". Laya reads a placeholder itself as personal data, and its score
    swings with the placeholder's random hex (the same sentence scored 0.22 and 0.51
    live on 2026-11-09). With them removed, maskable requests scored 0.04-0.13 while
    a name with health or salary details still scored 0.68-0.80. Only used for the
    check; the cloud still receives the placeholders."""
    out = copy.deepcopy(list(masked_messages or []))

    def strip(text):
        return _CHECK_STRIP.sub("", text) if isinstance(text, str) else text

    for message in out:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = strip(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part["text"] = strip(part["text"])
        for call in message.get("tool_calls") or []:
            function = (call or {}).get("function")
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                function["arguments"] = strip(function["arguments"])
    return out


def add_system_note(messages: list) -> list:
    """Tell the cloud model how to treat placeholders (only when something was masked)."""
    note = {"role": "system", "content": SYSTEM_NOTE}
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system" \
            and isinstance(messages[0].get("content"), str):
        first = dict(messages[0])
        first["content"] = first["content"].rstrip() + "\n\n" + SYSTEM_NOTE
        return [first] + list(messages[1:])
    return [note] + list(messages)


# --- reply side ----------------------------------------------------------------

def restore_text(text: str, mapping: dict[str, str]) -> str:
    if not text or not mapping:
        return text
    return PLACEHOLDER.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)


def restore_json_text(text: str, mapping: dict[str, str]) -> str:
    """Restore inside JSON text (tool-call arguments): each value is JSON-escaped,
    so a value with a quote or backslash cannot break the arguments' structure."""
    if not text or not mapping:
        return text
    return PLACEHOLDER.sub(
        lambda m: json.dumps(mapping[m.group(0)])[1:-1] if m.group(0) in mapping else m.group(0), text)


class StreamRestorer:
    """Restores placeholders in a stream of text deltas.

    A placeholder can be split across chunks, so text from an unclosed ``<`` is
    held back until it closes or grows past the longest possible placeholder.
    """

    MAX_HOLD = 48

    def __init__(self, mapping: dict[str, str], json_escape: bool = False):
        self.mapping = mapping
        self.restore = restore_json_text if json_escape else restore_text
        self.pending = ""

    def feed(self, text: str) -> str:
        if not text:
            return ""
        buffer = self.pending + text
        cut = buffer.rfind("<")
        if cut != -1 and ">" not in buffer[cut:] and len(buffer) - cut < self.MAX_HOLD:
            self.pending, ready = buffer[cut:], buffer[:cut]
        else:
            self.pending, ready = "", buffer
        return self.restore(ready, self.mapping)

    def flush(self) -> str:
        rest, self.pending = self.pending, ""
        return self.restore(rest, self.mapping)
