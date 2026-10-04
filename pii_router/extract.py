"""Turn an OpenAI chat request into the units the router scans.

Scanned: user, tool/function results, and assistant turns (including tool-call
arguments — `send_email(to="ann@…")` is PII going to the cloud too). Skipped:
system/developer prompts, which are the agent's own long instructions and
would otherwise dominate every decision.

Each message becomes one ``Unit`` keyed by a content hash, so a conversation's
history is scanned once and later turns only pay for their new messages. That
per-message verdict cache is also what makes PII "sticky": the message that
carried the PII is re-sent with every later turn, and its cached verdict keeps
routing the whole conversation local.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

SCANNED_ROLES = {"user", "tool", "function", "assistant"}


@dataclass(frozen=True)
class Unit:
    key: str            # sha256 of role + content; never logged with the content
    text: str
    has_opaque_media: bool  # image/audio/file parts we cannot inspect


def _part_text(part) -> tuple[str, bool]:
    if isinstance(part, str):
        return part, False
    if not isinstance(part, dict):
        return "", False
    kind = part.get("type")
    if kind in ("text", "input_text", "output_text"):
        return str(part.get("text", "")), False
    if kind in ("image_url", "input_image", "image", "input_audio", "audio", "file", "input_file"):
        return "", True
    return "", False


def _content_text(content) -> tuple[str, bool]:
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        texts, opaque = [], False
        for part in content:
            text, is_opaque = _part_text(part)
            if text:
                texts.append(text)
            opaque = opaque or is_opaque
        return "\n".join(texts), opaque
    return str(content), False


def message_units(messages) -> list[Unit]:
    units: list[Unit] = []
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") not in SCANNED_ROLES:
            continue
        text, opaque = _content_text(message.get("content"))
        for call in message.get("tool_calls") or []:
            function = (call or {}).get("function") or {}
            args = function.get("arguments")
            if args:
                text += "\n" + (args if isinstance(args, str) else json.dumps(args))
        if not text.strip() and not opaque:
            continue
        digest = hashlib.sha256(
            (str(message.get("role")) + "\x00" + text + ("\x00media" if opaque else "")).encode("utf-8", "replace")
        ).hexdigest()
        units.append(Unit(key=digest, text=text, has_opaque_media=opaque))
    return units


def chunks(text: str, size: int = 1800, overlap: int = 200) -> list[str]:
    """Split ``text`` for Laya's English checkpoint (512 tokens ≈ 2k chars)."""
    text = text.strip()
    size = max(int(size), 50)
    # Overlap must stay below the chunk size or the window never advances
    # (an env-configured PII_ROUTER_CHUNK_CHARS=100 looped forever).
    overlap = max(0, min(int(overlap), size // 4))
    if len(text) <= size:
        return [text] if text else []
    out, start, step = [], 0, size - overlap
    while start < len(text):
        out.append(text[start:start + size])
        start += step
    return out
