from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rava.experiments.runner import run_sweep


def _collect_run_metrics(root: Path, wall_time_s: float) -> dict[str, Any]:
    metrics_paths = sorted(root.glob("**/metrics.json"))
    timing_paths = {p.parent: p for p in root.glob("**/timing.json")}

    total_predictions = 0.0
    total_generation_calls = 0.0
    total_generation_errors = 0.0
    weighted_latency_sum = 0.0
    weighted_fallback_sum = 0.0

    runtime_sum_s = 0.0
    for metrics_path in metrics_paths:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        n_pred = float(payload.get("num_predictions", 0.0))
        total_predictions += n_pred
        total_generation_calls += float(payload.get("model_generation_calls", 0.0))
        total_generation_errors += float(payload.get("generation_error_count", 0.0))
        weighted_latency_sum += float(payload.get("latency_avg_ms", 0.0)) * n_pred
        weighted_fallback_sum += float(payload.get("generation_fallback_rate", 0.0)) * n_pred

        timing_path = timing_paths.get(metrics_path.parent)
        if timing_path and timing_path.exists():
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            runtime_sum_s += float(timing.get("runtime_seconds", 0.0))

    mean_latency_ms = weighted_latency_sum / total_predictions if total_predictions > 0 else 0.0
    fallback_rate = weighted_fallback_sum / total_predictions if total_predictions > 0 else 0.0
    predictions_per_sec = total_predictions / wall_time_s if wall_time_s > 0 else 0.0
    api_calls_per_prediction = (
        total_generation_calls / total_predictions if total_predictions > 0 else 0.0
    )
    api_failure_rate = (
        total_generation_errors / total_generation_calls if total_generation_calls > 0 else 0.0
    )

    return {
        "runs_root": str(root),
        "num_run_cells": len(metrics_paths),
        "wall_time_seconds": wall_time_s,
        "accumulated_run_runtime_seconds": runtime_sum_s,
        "total_predictions": total_predictions,
        "predictions_per_second": predictions_per_sec,
        "mean_example_latency_ms": mean_latency_ms,
        "api_calls_per_prediction": api_calls_per_prediction,
        "api_failure_rate": api_failure_rate,
        "fallback_rate": fallback_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark legacy vs LangGraph backend runtime.")
    parser.add_argument(
        "--sweep-config",
        default="configs/experiments/smoke.yaml",
        help="Sweep config used for both backend runs.",
    )
    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Base config file.",
    )
    parser.add_argument(
        "--max-concurrent-runs",
        type=int,
        default=2,
        help="Run-level concurrency.",
    )
    parser.add_argument(
        "--example-parallelism-per-run",
        type=int,
        default=2,
        help="Per-run example concurrency.",
    )
    parser.add_argument(
        "--stage",
        choices=["full", "canary", "calibration"],
        default="full",
        help="Stage forwarded to run_experiment.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/benchmarks/agent_runtime_benchmark.json",
        help="Where to write benchmark report JSON.",
    )
    args = parser.parse_args()

    results: dict[str, Any] = {}
    for backend in ("legacy_python", "langgraph"):
        started = time.time()
        root = run_sweep(
            sweep_config_path=args.sweep_config,
            base_config_path=args.base_config,
            stage=args.stage,
            provider_preflight_enabled=False,
            max_concurrent_runs=args.max_concurrent_runs,
            agentic_backend=backend,
            example_parallelism_per_run=args.example_parallelism_per_run,
            resume_mode="fresh",
        )
        wall = time.time() - started
        results[backend] = _collect_run_metrics(Path(root), wall)

    legacy_wall = float(results["legacy_python"]["wall_time_seconds"])
    langgraph_wall = float(results["langgraph"]["wall_time_seconds"])
    speedup = ((legacy_wall - langgraph_wall) / legacy_wall) if legacy_wall > 0 else 0.0
    results["comparison"] = {
        "wall_time_reduction_fraction": speedup,
        "wall_time_reduction_percent": speedup * 100.0,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
