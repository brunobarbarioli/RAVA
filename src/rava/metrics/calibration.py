from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CONF_RE = re.compile(r"confidence\s*[:=]\s*([01](?:\.\d+)?)", re.IGNORECASE)
PROB_RE = re.compile(r"(?:probability|likelihood)\s*[:=]\s*([01](?:\.\d+)?)", re.IGNORECASE)


def parse_confidence(text: str) -> float | None:
    match = CONF_RE.search(text) or PROB_RE.search(text)
    if not match:
        return infer_confidence_from_language(text)
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return min(1.0, max(0.0, value))


def infer_confidence_from_language(text: str) -> float | None:
    lower = text.lower()
    if not lower.strip():
        return None

    high_markers = ["certainly", "definitely", "high confidence", "strongly supported"]
    low_markers = ["uncertain", "not sure", "insufficient evidence", "may", "might"]
    if any(m in lower for m in high_markers):
        return 0.8
    if any(m in lower for m in low_markers):
        return 0.55
    return None


def expected_calibration_error(
    probs: list[float],
    labels: list[int],
    n_bins: int = 10,
) -> float:
    if not probs or not labels or len(probs) != len(labels):
        return 0.0

    probs_arr = np.array(probs)
    labels_arr = np.array(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (probs_arr >= lo) & (probs_arr < hi if i < n_bins - 1 else probs_arr <= hi)
        if not np.any(in_bin):
            continue
        conf = float(np.mean(probs_arr[in_bin]))
        acc = float(np.mean(labels_arr[in_bin]))
        weight = float(np.mean(in_bin))
        ece += abs(acc - conf) * weight

    return float(ece)


def compute_ece_from_predictions(
    records: list[dict[str, Any]],
    n_bins: int = 10,
    skip_missing: bool = True,
) -> dict[str, float | None]:
    probs: list[float] = []
    labels: list[int] = []
    missing = 0

    for row in records:
        conf = row.get("confidence")
        if conf is None:
            conf = parse_confidence(str(row.get("output", "")))
        if conf is None:
            missing += 1
            if skip_missing:
                continue
            conf = 0.5

        correct = row.get("correct")
        if correct is None:
            ref = str(row.get("reference", "")).strip().lower()
            out = str(row.get("output", "")).strip().lower()
            correct = int(bool(ref) and ref in out)
        probs.append(float(conf))
        labels.append(int(correct))

    if not probs:
        logger.warning("ECE skipped: no confidence values available in predictions.")
        return {"ece": None, "calibration_samples": 0.0, "missing_confidence": float(missing)}

    if missing > 0:
        logger.warning("ECE computed with %d records missing confidence.", missing)

    ece = expected_calibration_error(probs=probs, labels=labels, n_bins=n_bins)
    return {
        "ece": float(ece),
        "calibration_samples": float(len(probs)),
        "missing_confidence": float(missing),
    }
