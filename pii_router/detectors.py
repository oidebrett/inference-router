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

# RFC 2606 / 6761 documentation domains: addresses there are placeholders.
_EXAMPLE_DOMAINS = re.compile(r"@(?:[\w-]+\.)*example\.(?:com|org|net)\b|@localhost\b", re.I)
# The domain must end in a letter TLD: `root@203.0.113.67` is an ssh target, not mail.
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")

# A phone number must LOOK like one: a leading + or 0, or separators between
# digit groups. Bare runs of digits (timestamps, ids) are not phone numbers.
_PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?\(?\d{2,5}\)?(?:[\s.-]\d{2,5}){1,4}(?![\w.])"
    r"|(?<![\w.])(?:\+\d{8,15}|0\d{8,11})(?![\w.])"
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
    r"(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})",
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


def _phone_hits(text: str) -> bool:
    text = _DATETIME.sub(lambda m: " " * len(m.group()), text)
    for match in _PHONE.finditer(text):
        raw = match.group().strip()
        digits = _digits(raw)
        if not 9 <= len(digits) <= 15:
            continue
        if _DATE_LIKE.match(raw):
            continue
        # Unprefixed numbers need 2+ separators: "30000-32767" is a port range.
        if raw.startswith(("+", "0", "(")) or len(re.findall(r"\d[\s.-]\d", raw)) >= 2:
            # A separated run that is really a card/IBAN is reported by those detectors.
            if _luhn_ok(digits) and len(digits) >= 13:
                continue
            return True
    return False


def detect(text: str) -> list[str]:
    """Return the sorted names of the detectors that fire on ``text``."""
    if not text:
        return []
    hits: set[str] = set()

    for match in _EMAIL.finditer(text):
        if not _EXAMPLE_DOMAINS.search(match.group()):
            hits.add("email")
            break
    if any(_iban_ok(m.group()) for m in _IBAN.finditer(text)):
        hits.add("iban")
    if any(_luhn_ok(_digits(m.group())) for m in _CARD.finditer(text)):
        hits.add("payment_card")
    if _US_SSN.search(text):
        hits.add("us_ssn")
    if _UK_NINO.search(text):
        hits.add("uk_nino")
    if any(_ppsn_ok(*m.groups()) for m in _IE_PPSN.finditer(text)):
        hits.add("ie_ppsn")
    if _PASSPORT.search(text):
        hits.add("passport")
    if _DOB.search(text):
        hits.add("date_of_birth")
    if _phone_hits(text):
        hits.add("phone")
    return sorted(hits)
