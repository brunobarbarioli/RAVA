from __future__ import annotations

from typing import Any


def compute_selective_risk(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "answered_rate": 0.0,
            "answered_error_rate": 0.0,
            "answered_count": 0.0,
        }

    answered = [r for r in records if not bool(r.get("abstained", False))]
    answered_count = len(answered)
    answered_rate = answered_count / len(records)

    if answered_count == 0:
        return {
            "answered_rate": answered_rate,
            "answered_error_rate": 0.0,
            "answered_count": 0.0,
        }

    errors = 0
    for row in answered:
        correct = row.get("correct")
        if correct is None:
            ref = str(row.get("reference", "")).strip().lower()
            out = str(row.get("output", "")).strip().lower()
            correct = int(bool(ref) and ref in out)
        errors += int(not bool(correct))

    return {
        "answered_rate": answered_rate,
        "answered_error_rate": errors / answered_count,
        "answered_count": float(answered_count),
    }
