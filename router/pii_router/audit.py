"""Audit log for masked mode: what actually left the box, and what came back.

Opt-in (PII_ROUTER_AUDIT=on). Every request sent to the cloud is written exactly as
sent, i.e. with placeholders, and every reply exactly as received, BEFORE the real
values are put back. Real values are never written: requests that are blocked or
sent to the local model are logged as a decision only, because their messages
still contain the real values.

Note that the file holds everything else the cloud provider saw (names and other
text the detectors do not mask), so it is kept small (the last PII_ROUTER_AUDIT_KEEP
entries, default 50) and readable only by root and the pii-router user.

View it in a terminal:

    sudo pii-router-audit            # last 5 requests, placeholders highlighted
    sudo pii-router-audit -n 20      # last 20
    sudo pii-router-audit --full     # whole messages, including system prompts
    sudo pii-router-audit --follow   # keep watching
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time

DEFAULT_FILE = "/var/lib/pii-router/audit.jsonl"
_PLACEHOLDER = re.compile(r"<[A-Z][A-Z_]*_[0-9a-f]{8,16}>")


class AuditLog:
    def __init__(self, path: str, keep: int = 50):
        self.path = path
        self.keep = max(int(keep), 1)
        self._lock = threading.Lock()
        self._lines = None  # counted lazily

    @classmethod
    def from_env(cls, env=os.environ) -> "AuditLog | None":
        if (env.get("PII_ROUTER_AUDIT") or "off").strip().lower() not in ("on", "1", "true", "yes"):
            return None
        try:
            keep = int(env.get("PII_ROUTER_AUDIT_KEEP") or 50)
        except ValueError:
            keep = 50
        return cls(env.get("PII_ROUTER_AUDIT_FILE") or DEFAULT_FILE, keep)

    def write(self, entry: dict) -> None:
        """Append one entry; never raises (auditing must not break a request)."""
        try:
            line = json.dumps({"ts": round(time.time(), 3), **entry}, ensure_ascii=False) + "\n"
            with self._lock:
                old = os.umask(0o027)
                try:
                    with open(self.path, "a", encoding="utf-8") as fh:
                        fh.write(line)
                finally:
                    os.umask(old)
                if self._lines is None:
                    with open(self.path, encoding="utf-8") as fh:
                        self._lines = sum(1 for _ in fh)
                else:
                    self._lines += 1
                # Requests and replies are two lines each: trim to keep*2 lines.
                if self._lines > self.keep * 4:
                    with open(self.path, encoding="utf-8") as fh:
                        tail = fh.readlines()[-self.keep * 2:]
                    tmp = self.path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as fh:
                        fh.writelines(tail)
                    os.replace(tmp, self.path)
                    self._lines = len(tail)
        except Exception as error:  # noqa: BLE001
            print(f"pii-router: audit write failed: {type(error).__name__}", file=sys.stderr)


# --- terminal viewer -------------------------------------------------------------

def _colour(enabled: bool):
    def paint(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if enabled else text
    return paint


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else json.dumps(content)


def _clip(text: str, full: bool, limit: int = 600) -> str:
    if full or len(text) <= limit:
        return text
    return text[: limit // 2] + f" … [{len(text) - limit} chars] … " + text[-limit // 2:]


def _show_request(entry: dict, full: bool, paint) -> None:
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.get("ts", 0)))
    route, reason = entry.get("route"), entry.get("reason")
    tone = "31" if route == "blocked" else "33" if reason == "masked" else "32"
    print(paint(f"━━ {when}  {entry.get('id', '')}  route={route}  reason={reason}", "1;" + tone))
    if entry.get("masked_types"):
        print(f"   masked: {entry['masked_types']}  (placeholders sent instead of the values)")
    if "messages" not in entry:
        print(paint("   messages not logged: they still contain the real values "
                    "(blocked or kept on this box)", "2"))
        return
    print(f"   sent to {entry.get('target')}: {len(entry['messages'])} messages")
    for message in entry["messages"]:
        role = message.get("role")
        text = _text_of(message.get("content"))
        for call in message.get("tool_calls") or []:
            fn = (call or {}).get("function") or {}
            text += f"\n[tool call {fn.get('name')}] {fn.get('arguments')}"
        if not full and role in ("system", "developer") and not _PLACEHOLDER.search(text):
            print(paint(f"   [{role}] ({len(text)} chars, no placeholders; --full to show)", "2"))
            continue
        shown = _PLACEHOLDER.sub(lambda m: paint(m.group(0), "1;35"), _clip(text, full))
        print(f"   [{role}] {shown}")


def _show_reply(entry: dict, full: bool, paint) -> None:
    text = entry.get("content") or ""
    for call in entry.get("tool_calls") or []:
        text += f"\n[tool call {call.get('name')}] {call.get('arguments')}"
    shown = _PLACEHOLDER.sub(lambda m: paint(m.group(0), "1;35"), _clip(text, full))
    print(f"   reply from the model (before real values were put back): {shown}")
    print()


def _render(lines, full: bool, paint) -> None:
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("kind") == "reply":
            _show_reply(entry, full, paint)
        else:
            _show_request(entry, full, paint)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Show what the PII router sent to the cloud.")
    parser.add_argument("-n", type=int, default=5, help="number of requests to show (default 5)")
    parser.add_argument("--full", action="store_true", help="show whole messages, system prompts included")
    parser.add_argument("--follow", "-f", action="store_true", help="keep watching for new requests")
    parser.add_argument("--file", default=os.environ.get("PII_ROUTER_AUDIT_FILE") or DEFAULT_FILE)
    args = parser.parse_args(argv)
    paint = _colour(sys.stdout.isatty())
    try:
        with open(args.file, encoding="utf-8") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        print(f"no audit file at {args.file}: set PII_ROUTER_AUDIT=on and restart the router", file=sys.stderr)
        return 1
    except PermissionError:
        print(f"cannot read {args.file}: run with sudo", file=sys.stderr)
        return 1
    requests = [i for i, l in enumerate(lines) if '"kind": "request"' in l]
    start = requests[-args.n] if len(requests) >= args.n else 0
    _render(lines[start:], args.full, paint)
    if args.follow:
        seen = len(lines)
        while True:
            time.sleep(1)
            with open(args.file, encoding="utf-8") as fh:
                lines = fh.readlines()
            if len(lines) < seen:  # trimmed
                seen = 0
            _render(lines[seen:], args.full, paint)
            sys.stdout.flush()
            seen = len(lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
