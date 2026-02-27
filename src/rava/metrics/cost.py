from __future__ import annotations

from typing import Any


# Approximate placeholder rates for relative frontier analysis.
COST_PER_1K_TOKENS_USD = {
    "ministral-3-cloud": 0.12,
    "qwen3-next": 0.28,
}


def _estimate_tokens(text: str) -> float:
    words = len(text.split())
    # Lightweight token approximation for model-agnostic accounting.
    return max(1.0, words * 1.3)


def compute_cost_metrics(records: list[dict[str, Any]], model_name: str | None = None) -> dict[str, float]:
    if not records:
        return {
            "estimated_input_tokens": 0.0,
            "estimated_output_tokens": 0.0,
            "estimated_total_tokens": 0.0,
            "estimated_cost_usd": 0.0,
        }

    input_tokens = 0.0
    output_tokens = 0.0
    for row in records:
        input_tokens += _estimate_tokens(str(row.get("input", "")))
        output_tokens += _estimate_tokens(str(row.get("output", "")))

    total_tokens = input_tokens + output_tokens
    rate = COST_PER_1K_TOKENS_USD.get(str(model_name or ""), 0.0)
    cost = (total_tokens / 1000.0) * rate

    return {
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens": total_tokens,
        "estimated_cost_usd": cost,
    }
