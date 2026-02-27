from __future__ import annotations

from typing import Any


def compute_latency_metrics(trajectory_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not trajectory_rows:
        return {
            "latency_avg_ms": 0.0,
            "latency_p95_ms": 0.0,
            "layer_pre_ms": 0.0,
            "layer_runtime_ms": 0.0,
            "layer_posthoc_ms": 0.0,
        }

    durs = [float(r.get("duration_ms", 0.0)) for r in trajectory_rows]
    durs_sorted = sorted(durs)
    p95_idx = int(0.95 * (len(durs_sorted) - 1))

    def layer_sum(layer: str) -> float:
        return sum(float(r.get("duration_ms", 0.0)) for r in trajectory_rows if str(r.get("phase")) == layer)

    return {
        "latency_avg_ms": sum(durs) / len(durs),
        "latency_p95_ms": durs_sorted[p95_idx],
        "layer_pre_ms": layer_sum("pre"),
        "layer_runtime_ms": layer_sum("runtime"),
        "layer_posthoc_ms": layer_sum("posthoc"),
    }
