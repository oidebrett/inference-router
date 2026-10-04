"""Routing decision: LOCAL (Ollama) or CLOUD (NVIDIA) for one chat request.

Pure logic, no LiteLLM — so it can be tested directly. The rule is a union:
a message is PII if ANY deterministic detector fires OR Laya's yes-probability
reaches the threshold. The request goes local if ANY scanned message is PII.

Fail closed: a Laya error or timeout, more text than the Laya budget covers,
or media we cannot inspect all route LOCAL. The cloud is only used when every
message was positively cleared.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol

from .detectors import detect
from .extract import Unit, chunks, message_units

LOCAL = "local"
CLOUD = "cloud"

# Wording matters a lot: the bare "Does this contain PII?" caught 3/12 in the
# 2026-10-04 spike; this descriptive form caught 8/12 on its own and 12/12
# combined with the detectors at threshold 0.4.
PII_QUESTION = {
    "type": "noul",
    "instructions": (
        "Does this text contain personal data about a specific real individual, such as a name "
        "together with contact details, an address, an ID or account number, or health, "
        "financial or employment details?"
    ),
}


class LayaScorer(Protocol):
    def score(self, texts: list[str]) -> list[float]:
        """Return P(yes) per text. Raise on any failure."""


@dataclass
class Settings:
    threshold: float = 0.4
    max_chunks: int = 32          # per request, across new messages
    chunk_chars: int = 1800
    oversize_route: str = LOCAL   # when new text exceeds max_chunks
    media_route: str = LOCAL      # images/audio/files we cannot inspect
    cache_size: int = 20000

    @classmethod
    def from_env(cls, env=os.environ) -> "Settings":
        def _f(name, default, cast):
            raw = env.get(name)
            try:
                return cast(raw) if raw not in (None, "") else default
            except ValueError:
                return default
        route = lambda v: v if v in (LOCAL, CLOUD) else LOCAL  # noqa: E731
        return cls(
            threshold=_f("PII_ROUTER_LAYA_THRESHOLD", 0.4, float),
            max_chunks=_f("PII_ROUTER_MAX_CHUNKS", 32, int),
            chunk_chars=_f("PII_ROUTER_CHUNK_CHARS", 1800, int),
            oversize_route=route(env.get("PII_ROUTER_OVERSIZE_ROUTE", LOCAL)),
            media_route=route(env.get("PII_ROUTER_MEDIA_ROUTE", LOCAL)),
            cache_size=_f("PII_ROUTER_CACHE_SIZE", 20000, int),
        )


@dataclass
class Decision:
    route: str
    reasons: list[str] = field(default_factory=list)
    laya_max: float | None = None
    scanned_units: int = 0
    cached_units: int = 0
    laya_chunks: int = 0
    elapsed_ms: float = 0.0
    # DIAGNOSTIC ONLY (does not affect routing): detector hits inside system/
    # developer prompts, which are not scanned for routing. Used to find out
    # whether agent memory (e.g. OpenClaw memory-core / MEMORY.md) re-injects PII
    # from earlier sessions through the system prompt.
    system_hits: list[str] = field(default_factory=list)

    def log_fields(self) -> dict:
        """What the decision log may contain: never message text."""
        return {
            "route": self.route,
            "reasons": self.reasons,
            "laya_max": None if self.laya_max is None else round(self.laya_max, 3),
            "scanned_units": self.scanned_units,
            "cached_units": self.cached_units,
            "laya_chunks": self.laya_chunks,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "system_hits": self.system_hits,
        }


class _VerdictCache:
    """LRU of message-hash -> (is_pii, reasons). Holds hashes only, no text."""

    def __init__(self, size: int):
        self.size = size
        self._data: OrderedDict[str, tuple[bool, tuple[str, ...]]] = OrderedDict()

    def get(self, key):
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def put(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.size:
            self._data.popitem(last=False)


class Router:
    def __init__(self, scorer: LayaScorer, settings: Settings | None = None):
        self.scorer = scorer
        self.settings = settings or Settings()
        self.cache = _VerdictCache(self.settings.cache_size)

    def decide(self, messages) -> Decision:
        started = time.perf_counter()
        s = self.settings
        decision = Decision(route=CLOUD)
        reasons: set[str] = set()
        pending: list[Unit] = []

        system_text = "\n".join(
            str(m.get("content") if isinstance(m.get("content"), str) else "")
            for m in messages or [] if isinstance(m, dict) and m.get("role") in ("system", "developer"))
        decision.system_hits = detect(system_text)

        for unit in message_units(messages):
            cached = self.cache.get(unit.key)
            if cached is not None:
                decision.cached_units += 1
                if cached[0]:
                    reasons.update(cached[1])
                    reasons.add("sticky")
                continue
            pending.append(unit)
        decision.scanned_units = len(pending)

        # Deterministic pass over everything new: cheap, and covers what Laya misses.
        unit_hits: dict[str, list[str]] = {}
        for unit in pending:
            hits = detect(unit.text)
            if unit.has_opaque_media and s.media_route == LOCAL:
                hits = hits + ["opaque_media"]
            unit_hits[unit.key] = hits

        # Laya pass over units the detectors did not already settle.
        undecided = [u for u in pending if not unit_hits[u.key]]
        plan: list[tuple[str, str]] = []  # (unit key, chunk)
        chunks_needed: dict[str, int] = {}
        for unit in undecided:
            pieces = chunks(unit.text, s.chunk_chars)
            chunks_needed[unit.key] = len(pieces)
            for piece in pieces:
                plan.append((unit.key, piece))

        laya_scores: dict[str, float] = {}
        laya_failed = False
        if len(plan) > s.max_chunks:
            if s.oversize_route == LOCAL:
                reasons.add("oversize_unscanned")
            plan = plan[: s.max_chunks]
        chunks_scored: dict[str, int] = {}
        for key, _ in plan:
            chunks_scored[key] = chunks_scored.get(key, 0) + 1
        if plan:
            try:
                scores = self.scorer.score([piece for _, piece in plan])
                if len(scores) != len(plan):
                    raise ValueError("scorer returned the wrong number of scores")
                for (key, _), score in zip(plan, scores):
                    laya_scores[key] = max(laya_scores.get(key, 0.0), float(score))
                decision.laya_chunks = len(plan)
                decision.laya_max = max(scores) if scores else None
            except Exception:  # noqa: BLE001 — any scorer failure fails closed
                laya_failed = True
                reasons.add("laya_unavailable")

        for unit in pending:
            hits = unit_hits[unit.key]
            if hits:
                self.cache.put(unit.key, (True, tuple(hits)))
                reasons.update(hits)
                continue
            if laya_failed or unit.key not in laya_scores:
                # Not cached: a later turn retries once Laya is back / budget allows.
                continue
            fully_scanned = chunks_scored.get(unit.key, 0) == chunks_needed.get(unit.key, 0)
            if laya_scores[unit.key] >= s.threshold:
                self.cache.put(unit.key, (True, ("laya",)))  # PII in any part is enough
                reasons.add("laya")
            elif fully_scanned:
                self.cache.put(unit.key, (False, ()))
            # A partly scanned unit is never cached as clean.

        if reasons:
            decision.route = LOCAL
        decision.reasons = sorted(reasons)
        decision.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return decision
