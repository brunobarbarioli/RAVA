from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_violation_rates(verdict_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not verdict_rows:
        return {
            "hard_violation_rate": 0.0,
            "soft_violation_rate": 0.0,
            "prevention_hard_violation_rate": 0.0,
            "detection_hard_violation_rate": 0.0,
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(verdict_rows):
        rid = str(row.get("record_id", "record"))
        event = str(row.get("event", "event"))
        action_key = str(row.get("action_key", f"{rid}:{event}:{i}"))
        grouped[action_key].append(row)

    hard_violations = 0
    soft_violations = 0
    total = len(grouped)

    for rows in grouped.values():
        hard_fail = any(
            str(r.get("constraint_type", "")).upper() == "HARD"
            and str(r.get("verdict", "")).upper() == "FAIL"
            for r in rows
        )
        soft_fail = any(
            str(r.get("constraint_type", "")).upper() == "SOFT"
            and str(r.get("verdict", "")).upper() == "FAIL"
            for r in rows
        )
        hard_violations += int(hard_fail)
        soft_violations += int(soft_fail)

    prevention_rows = [r for r in verdict_rows if str(r.get("layer", "")).lower() in {"pre", "runtime"}]
    detection_rows = [r for r in verdict_rows if str(r.get("layer", "")).lower() == "posthoc"]

    def _hard_rate(rows: list[dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        grouped_local: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for i, row in enumerate(rows):
            rid = str(row.get("record_id", "record"))
            event = str(row.get("event", "event"))
            action_key = str(row.get("action_key", f"{rid}:{event}:{i}"))
            grouped_local[action_key].append(row)
        fail_count = 0
        for local in grouped_local.values():
            if any(
                str(r.get("constraint_type", "")).upper() == "HARD"
                and str(r.get("verdict", "")).upper() == "FAIL"
                for r in local
            ):
                fail_count += 1
        return fail_count / max(len(grouped_local), 1)

    return {
        "hard_violation_rate": hard_violations / max(total, 1),
        "soft_violation_rate": soft_violations / max(total, 1),
        "prevention_hard_violation_rate": _hard_rate(prevention_rows),
        "detection_hard_violation_rate": _hard_rate(detection_rows),
    }
