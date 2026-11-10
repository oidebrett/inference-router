import pytest

from pii_router.detectors import detect
from tests.cases import NEGATIVE, SEMANTIC, STRUCTURED


@pytest.mark.parametrize("text,expected", STRUCTURED)
def test_structured_pii_is_caught(text, expected):
    assert expected in detect(text)


@pytest.mark.parametrize("text", NEGATIVE)
def test_dev_and_general_text_does_not_trip_detectors(text):
    assert detect(text) == []


@pytest.mark.parametrize("text", SEMANTIC)
def test_semantic_pii_is_left_to_laya(text):
    # Documents the division of labour: nothing structured here, so Laya decides.
    assert detect(text) == []


def test_checksums_reject_lookalikes():
    assert "iban" not in detect("Reference IE29 AIBK 9311 5212 3456 79")      # mod-97 fails
    assert "payment_card" not in detect("Tracking 4111 1111 1111 1112")       # Luhn fails
    assert "ie_ppsn" not in detect("Code 1234567AA")                          # PPSN check fails


def test_detector_output_never_contains_the_matched_text():
    hits = detect("mail sarah.oconnor@gmail.com, card 4111 1111 1111 1111")
    assert all("@" not in h and not any(c.isdigit() for c in h) for h in hits)
