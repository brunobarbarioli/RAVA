from __future__ import annotations

from rava.agent.providers import _parse_confidence
from rava.metrics.calibration import parse_confidence


def test_provider_parse_confidence_from_json_key():
    text = '{"answer":"ok","confidence":0.73,"citations":[]}'
    assert _parse_confidence(text) == 0.73


def test_provider_parse_confidence_from_plain_text():
    assert _parse_confidence("Confidence: 0.73") == 0.73
    assert _parse_confidence("confidence=0.73") == 0.73


def test_calibration_parse_confidence_from_json_key():
    text = '{"answer":"ok","confidence":0.73,"citations":[]}'
    assert parse_confidence(text) == 0.73


def test_calibration_parse_confidence_from_plain_text():
    assert parse_confidence("Confidence: 0.73") == 0.73
    assert parse_confidence("confidence=0.73") == 0.73
