from __future__ import annotations

from typing import Any


def compute_abstention_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"abstention_rate": 0.0}
    abstained = sum(1 for row in records if bool(row.get("abstained", False)))
    return {"abstention_rate": abstained / len(records)}
