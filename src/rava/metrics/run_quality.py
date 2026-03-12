from __future__ import annotations

from typing import Any


def _is_generation_step(row: dict[str, Any]) -> bool:
    return str(row.get("action", "")) == "langchain_agent_generation"


def compute_run_quality_metrics(
    trajectory_rows: list[dict[str, Any]],
    runner_generation_calls: int | None = None,
    runner_generation_error_count: int | None = None,
    runner_error_taxonomy_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    generation_rows = [r for r in trajectory_rows if _is_generation_step(r)]

    fallback_count = 0
    error_count = 0
    taxonomy_counts: dict[str, int] = {}
    for row in generation_rows:
        metadata = row.get("metadata", {}) or {}
        mode = str(metadata.get("mode", "")).lower()
        if "fallback" in mode:
            fallback_count += 1
        if metadata.get("error"):
            error_count += 1
        taxonomy = metadata.get("error_taxonomy")
        if taxonomy:
            key = str(taxonomy)
            taxonomy_counts[key] = taxonomy_counts.get(key, 0) + 1

    if runner_error_taxonomy_counts:
        for key, value in runner_error_taxonomy_counts.items():
            count = int(value)
            taxonomy_counts[str(key)] = max(taxonomy_counts.get(str(key), 0), count)

    total_calls = len(generation_rows)
    if runner_generation_calls is not None:
        total_calls = max(total_calls, int(runner_generation_calls))

    total_errors = error_count
    if runner_generation_error_count is not None:
        total_errors = max(total_errors, int(runner_generation_error_count))

    total = float(max(0, total_calls))
    if total <= 0.0:
        return {
            "model_generation_calls": 0.0,
            "generation_fallback_rate": 0.0,
            "generation_error_count": float(total_errors),
            "api_failure_rate": 0.0,
            "api_failure_taxonomy_counts": taxonomy_counts,
        }

    return {
        "model_generation_calls": total,
        "generation_fallback_rate": fallback_count / total,
        "generation_error_count": float(total_errors),
        "api_failure_rate": float(total_errors) / total,
        "api_failure_taxonomy_counts": taxonomy_counts,
    }


def assess_run_quality_for_model_comparison(
    metrics: dict[str, Any],
    max_fallback_rate: float = 0.05,
    max_api_failure_rate: float = 0.02,
) -> dict[str, Any]:
    fallback_rate = float(metrics.get("generation_fallback_rate", 0.0))
    api_failure_rate = float(metrics.get("api_failure_rate", 0.0))

    reasons: list[str] = []
    if fallback_rate > max_fallback_rate:
        reasons.append(f"fallback_rate>{max_fallback_rate:.2f}")
    if api_failure_rate > max_api_failure_rate:
        reasons.append(f"api_failure_rate>{max_api_failure_rate:.2f}")

    return {
        "valid_for_model_comparison": len(reasons) == 0,
        "run_quality_reasons": reasons,
    }
