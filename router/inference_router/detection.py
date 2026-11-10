"""PII detection for the router: built-in detectors (with checksums) plus the admin's
custom rules, minus exceptions.

Custom rules run with a per-match time limit. A rule that times out raises
DetectionTimeout and the hook refuses the request: a pathological pattern can make
the router slow or strict, never leak a value.
"""

from __future__ import annotations

import regex

from pii_router.detectors import Span, _digits, _iban_ok, _luhn_ok
from pii_router.detectors import find_spans as builtin_spans

from .config import Rules

RULE_TIMEOUT_S = 0.25
KEYWORD_WINDOW = 40   # characters before a match in which a keyword must appear


class DetectionTimeout(Exception):
    pass


_VALIDATE = {
    "luhn": lambda v: _luhn_ok(_digits(v)),
    "mod97": _iban_ok,
}


class Detector:
    def __init__(self, rules: Rules | None = None):
        self.rules = rules or Rules()
        self._custom = []
        for rule in self.rules.custom:
            flags = regex.IGNORECASE if rule.ignore_case else 0
            keywords = [k.lower() for k in rule.keywords]
            self._custom.append((rule, regex.compile(rule.pattern, flags), keywords))
        self._exact = {e.lower() for e in self.rules.exceptions if not e.startswith("@")}
        self._domains = tuple(e.lower() for e in self.rules.exceptions if e.startswith("@"))

    def _excepted(self, value: str) -> bool:
        v = value.strip().lower()
        return v in self._exact or (bool(self._domains) and v.endswith(self._domains))

    def find_spans(self, text: str) -> list[Span]:
        if not text:
            return []
        spans = [s for s in builtin_spans(text) if self.rules.builtin_enabled(s.type)]
        for rule, pattern, keywords in self._custom:
            try:
                for m in pattern.finditer(text, timeout=RULE_TIMEOUT_S):
                    value = m.group(0)
                    if not value.strip():
                        continue
                    if rule.validator and not _VALIDATE[rule.validator](value):
                        continue
                    if keywords:
                        before = text[max(0, m.start() - KEYWORD_WINDOW):m.start()].lower()
                        if not any(k in before for k in keywords):
                            continue
                    spans.append(Span(rule.name.lower(), m.start(), m.end()))
            except TimeoutError as error:
                raise DetectionTimeout(rule.name) from error
        spans = [s for s in spans if not self._excepted(text[s.start:s.end])]
        return sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))

    def detect(self, text: str) -> list[str]:
        return sorted({s.type for s in self.find_spans(text)})
