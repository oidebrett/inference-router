"""Deterministic PII detectors.

Laya judges meaning (a name next to health, salary or address details) but
misses structured identifiers: in the 2026-10-04 spike it scored an IBAN, a
phone number and a card number as not-PII. These detectors cover exactly that
gap. Each returns the names of the detectors that fired, never the matched
text, so nothing sensitive reaches the decision log.

Checksums are validated where the format has one (IBAN mod-97, card Luhn,
Irish PPSN mod-23) so ordinary long numbers in code, logs and timestamps do not
trip them. IP addresses are deliberately NOT treated as PII here: infra and
coding prompts are full of them, and flagging them would push most devops work
to the local model. Laya still sees the text.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# RFC 2606 / 6761 documentation domains: addresses there are placeholders.
_EXAMPLE_DOMAINS = re.compile(r"@(?:[\w-]+\.)*example\.(?:com|org|net)\b|@localhost\b", re.I)
# The domain must end in a letter TLD: `root@203.0.113.67` is an ssh target, not mail.
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")

# A phone number must LOOK like one: a leading + or 0, or separators between
# digit groups. Bare runs of digits (timestamps, ids) are not phone numbers.
# A number may end a sentence ("call +353 86 123 4567."): a full stop after it counts as the end
# unless a digit follows, which still keeps versions and dotted ids out (Claude Gateway, 2026-11-11:
# a phone number before a full stop reached Anthropic unmasked).
_PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?\(?\d{2,5}\)?(?:[\s.-]\d{2,5}){1,4}(?!\w|\.\d)"
    r"|(?<![\w.])(?:\+\d{8,15}|0\d{8,11})(?!\w|\.\d)"
)
_DATE_LIKE = re.compile(r"^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}$")
# Dates and times are blanked before the phone check. OpenClaw 2026.9.x adds
# "Reference UTC: 2026-10-04 13:20 UTC" to messages, and "2026-10-04 13" read
# as a 10-digit phone number: every conversation's first turn was routed local
# (live, 2026-10-04). Only the phone check uses the masked text; the
# date-of-birth detector still sees the original.
_DATETIME = re.compile(
    r"\b(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|\b\d{1,2}[./-]\d{1,2}[./-](?:19|20)?\d{2}\b"
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\b"
)

_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_US_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_UK_NINO = re.compile(r"\b(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b")
_IE_PPSN = re.compile(r"\b(\d{7})([A-W])([A-IW]?)\b", re.I)
_PASSPORT = re.compile(r"passport[^.\n]{0,40}?\b([A-Z]{1,2}\d{6,9})\b", re.I)
_DOB = re.compile(
    r"\b(?:d\.?o\.?b\.?|date of birth|born on)\b[\s:]{0,3}"
    r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})",
    re.I,
)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number][::-1]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    for i, d in enumerate(digits):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_ok(candidate: str) -> bool:
    iban = candidate.replace(" ", "").upper()
    if not 15 <= len(iban) <= 34:
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


def _ppsn_ok(digits: str, check: str, extra: str) -> bool:
    weights = [8, 7, 6, 5, 4, 3, 2]
    total = sum(int(d) * w for d, w in zip(digits, weights))
    if extra:
        total += (ord(extra.upper()) - 64 if extra.upper() != "W" else 0) * 9
    expected = total % 23
    letter = "W" if expected == 0 else chr(64 + expected)
    return letter == check.upper()


class Span(NamedTuple):
    """One detected value: its type and where it sits in the text (end exclusive).

    Masked mode needs locations, not just names. The span is never logged."""
    type: str
    start: int
    end: int


def _phone_spans(text: str) -> list[Span]:
    # Dates and times are blanked with spaces of the same length, so offsets in
    # the blanked text are offsets in the original.
    blanked = _DATETIME.sub(lambda m: " " * len(m.group()), text)
    out: list[Span] = []
    for match in _PHONE.finditer(blanked):
        raw = match.group()
        stripped = raw.strip()
        digits = _digits(stripped)
        if not 9 <= len(digits) <= 15:
            continue
        if _DATE_LIKE.match(stripped):
            continue
        # Unprefixed numbers need 2+ separators: "30000-32767" is a port range.
        if stripped.startswith(("+", "0", "(")) or len(re.findall(r"\d[\s.-]\d", stripped)) >= 2:
            # A separated run that is really a card/IBAN is reported by those detectors.
            if _luhn_ok(digits) and len(digits) >= 13:
                continue
            lead = len(raw) - len(raw.lstrip())
            begin = match.start() + lead
            out.append(Span("phone", begin, begin + len(stripped)))
    return out


def find_spans(text: str) -> list[Span]:
    """Every detected value with its location, sorted by position.

    Overlapping spans are kept as found; the masker merges them."""
    if not text:
        return []
    spans: list[Span] = []
    for match in _EMAIL.finditer(text):
        if not _EXAMPLE_DOMAINS.search(match.group()):
            spans.append(Span("email", match.start(), match.end()))
    spans += [Span("iban", m.start(), m.end()) for m in _IBAN.finditer(text) if _iban_ok(m.group())]
    spans += [Span("payment_card", m.start(), m.end()) for m in _CARD.finditer(text)
              if _luhn_ok(_digits(m.group()))]
    spans += [Span("us_ssn", m.start(), m.end()) for m in _US_SSN.finditer(text)]
    spans += [Span("uk_nino", m.start(), m.end()) for m in _UK_NINO.finditer(text)]
    spans += [Span("ie_ppsn", m.start(), m.end()) for m in _IE_PPSN.finditer(text) if _ppsn_ok(*m.groups())]
    # Passport and date of birth are keyword-anchored: mask the value, keep the keyword.
    spans += [Span("passport", m.start(1), m.end(1)) for m in _PASSPORT.finditer(text)]
    spans += [Span("date_of_birth", m.start(1), m.end(1)) for m in _DOB.finditer(text)]
    spans += _phone_spans(text)
    return sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))


def detect(text: str) -> list[str]:
    """Return the sorted names of the detectors that fire on ``text``."""
    return sorted({span.type for span in find_spans(text)})
