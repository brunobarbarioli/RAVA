from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rava.utils.serialization import read_json


def _collect_run_rows(runs_root: str | Path) -> list[dict[str, Any]]:
    root = Path(runs_root)
    rows: list[dict[str, Any]] = []

    for metrics_path in root.glob("**/metrics.json"):
        run_dir = metrics_path.parent
        parts = run_dir.parts
        try:
            seed = parts[-1]
            config = parts[-2]
            model = parts[-3]
            domain = parts[-4]
            timestamp = parts[-5]
        except Exception:
            continue

        metrics = read_json(metrics_path)
        report_path = run_dir / "report.json"
        report = read_json(report_path) if report_path.exists() else {}

        score = {}
        if isinstance(report, dict):
            if isinstance(report.get("score"), dict):
                score = report["score"]
            elif "R" in report:
                score = {"R": report.get("R"), "tier": report.get("tier")}

        row = {
            "timestamp": timestamp,
            "domain": domain,
            "model": model,
            "config": config,
            "seed": int(seed),
            **metrics,
            "R": score.get("R"),
            "tier": score.get("tier"),
        }
        rows.append(row)

    return rows


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    tex_path = path.with_suffix(".tex")
    if df.empty:
        tex_path.write_text(df.to_latex(index=False), encoding="utf-8")
    else:
        tex_path.write_text(df.to_latex(index=False, float_format=lambda x: f"{x:.3f}"), encoding="utf-8")


def _bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1.0 - alpha / 2))
    return (low, high)


def _paired_permutation_pvalue(a: list[float], b: list[float], n_perm: int = 5000, seed: int = 42) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    diffs = np.array(b, dtype=float) - np.array(a, dtype=float)
    observed = abs(float(np.mean(diffs)))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diffs), replace=True)
        stat = abs(float(np.mean(diffs * signs)))
        if stat >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def _aggregate_domain_table(sub: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, config), grp in sub.groupby(["model", "config"]):
        r_vals = [float(v) for v in grp["R"].dropna().tolist()]
        r_low, r_high = _bootstrap_ci(r_vals) if r_vals else (0.0, 0.0)
        rows.append(
            {
                "model": model,
                "config": config,
                "n_runs": int(len(grp)),
                "R_mean": float(np.mean(r_vals)) if r_vals else 0.0,
                "R_ci_low": r_low,
                "R_ci_high": r_high,
                "hard_violation_rate": float(grp["hard_violation_rate"].mean()),
                "soft_violation_rate": float(grp["soft_violation_rate"].mean()),
                "claim_precision": float(grp["claim_precision"].mean()),
                "source_attribution_score": float(grp["source_attribution_score"].mean()),
                "abstention_rate": float(grp.get("abstention_rate", pd.Series(dtype=float)).mean()) if "abstention_rate" in grp else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "config"]) if rows else pd.DataFrame()


def _make_significance_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (domain, model), grp in df.groupby(["domain", "model"]):
        pivot = (
            grp[grp["config"].isin(["none", "full"])]
            .groupby(["seed", "config"], as_index=False)["R"]
            .mean()
            .pivot(index="seed", columns="config", values="R")
        )
        if "none" not in pivot.columns or "full" not in pivot.columns:
            continue
        none_vals = [float(v) for v in pivot["none"].dropna().tolist()]
        full_vals = [float(v) for v in pivot["full"].dropna().tolist()]
        n = min(len(none_vals), len(full_vals))
        none_vals = none_vals[:n]
        full_vals = full_vals[:n]
        if n == 0:
            continue
        delta = float(np.mean(np.array(full_vals) - np.array(none_vals)))
        pvalue = _paired_permutation_pvalue(none_vals, full_vals)
        rows.append(
            {
                "domain": domain,
                "model": model,
                "n_paired_seeds": n,
                "R_delta_full_minus_none": delta,
                "p_value_permutation": pvalue,
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "model"]) if rows else pd.DataFrame()


def make_tables(runs_root: str | Path = "runs", output_dir: str | Path = "outputs/tables") -> list[Path]:
    rows = _collect_run_rows(runs_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    if not rows:
        empty = pd.DataFrame(columns=["domain", "model", "config", "seed", "R", "tier"])
        out = out_dir / "ablation.csv"
        _write_table(empty, out)
        outputs.append(out)
        return outputs

    df = pd.DataFrame(rows)
    for col, default in [
        ("abstention_rate", 0.0),
        ("prevention_hard_violation_rate", 0.0),
        ("detection_hard_violation_rate", 0.0),
        ("estimated_cost_usd", 0.0),
        ("estimated_total_tokens", 0.0),
        ("R", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default

    # Domain tables with confidence intervals.
    for domain in ["healthcare", "finance", "hr"]:
        sub = df[df["domain"] == domain]
        if sub.empty:
            continue
        agg = _aggregate_domain_table(sub)
        out = out_dir / f"{domain}.csv"
        _write_table(agg, out)
        outputs.append(out)

    # Ablation across domains/configs.
    ablation = (
        df.groupby(["domain", "config"], as_index=False)
        .agg(
            R_mean=("R", "mean"),
            hard_violation_rate=("hard_violation_rate", "mean"),
            soft_violation_rate=("soft_violation_rate", "mean"),
            claim_precision=("claim_precision", "mean"),
            source_attribution_score=("source_attribution_score", "mean"),
            abstention_rate=("abstention_rate", "mean"),
        )
        .sort_values(["domain", "config"])
    )
    ablation_path = out_dir / "ablation.csv"
    _write_table(ablation, ablation_path)
    outputs.append(ablation_path)

    # Latency table.
    latency = (
        df.groupby(["domain", "model", "config"], as_index=False)
        .agg(
            latency_avg_ms=("latency_avg_ms", "mean"),
            latency_p95_ms=("latency_p95_ms", "mean"),
            layer_pre_ms=("layer_pre_ms", "mean"),
            layer_runtime_ms=("layer_runtime_ms", "mean"),
            layer_posthoc_ms=("layer_posthoc_ms", "mean"),
        )
        .sort_values(["domain", "model", "config"])
    )
    latency_path = out_dir / "latency.csv"
    _write_table(latency, latency_path)
    outputs.append(latency_path)

    # Prevention vs detection split.
    prev_det = (
        df.groupby(["domain", "model", "config"], as_index=False)
        .agg(
            prevention_hard_violation_rate=("prevention_hard_violation_rate", "mean"),
            detection_hard_violation_rate=("detection_hard_violation_rate", "mean"),
        )
        .sort_values(["domain", "model", "config"])
    )
    prev_det_path = out_dir / "prevention_detection.csv"
    _write_table(prev_det, prev_det_path)
    outputs.append(prev_det_path)

    # Cost-assurance frontier.
    cost = (
        df.groupby(["domain", "model", "config"], as_index=False)
        .agg(
            R_mean=("R", "mean"),
            estimated_cost_usd=("estimated_cost_usd", "mean"),
            estimated_total_tokens=("estimated_total_tokens", "mean"),
        )
        .sort_values(["domain", "model", "config"])
    )
    cost["cost_per_R"] = cost.apply(
        lambda r: float(r["estimated_cost_usd"]) / max(float(r["R_mean"]), 1e-6), axis=1
    )
    cost_path = out_dir / "cost_frontier.csv"
    _write_table(cost, cost_path)
    outputs.append(cost_path)

    # Paired significance for full vs none.
    significance = _make_significance_table(df)
    significance_path = out_dir / "significance.csv"
    _write_table(significance, significance_path)
    outputs.append(significance_path)

    return outputs
