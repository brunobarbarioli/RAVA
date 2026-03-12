from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rava.metrics.factuality import compute_claim_precision
from rava.metrics.violations import compute_violation_rates
from rava.utils.serialization import read_json, read_jsonl


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

        score: dict[str, Any] = {}
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
            "run_dir": str(run_dir),
            **metrics,
            "sweep_stage": str(metrics.get("sweep_stage", "unknown")),
            "certification_eligible": bool(metrics.get("certification_eligible", True)),
            "dataset_profile": str(metrics.get("dataset_profile", "unknown")),
            "benchmark_role": str(metrics.get("benchmark_role", "mixed")),
            "R": score.get("R"),
            "R_raw": score.get("R_raw", score.get("R")),
            "R_certified": score.get("R_certified"),
            "R_operational_raw": score.get("R_operational_raw"),
            "R_operational_certified": score.get("R_operational_certified"),
            "R_audited_raw": score.get("R_audited_raw", score.get("R_raw", score.get("R"))),
            "R_audited_certified": score.get("R_audited_certified", score.get("R_certified")),
            "R_prevent": score.get("R_prevent"),
            "R_detect": score.get("R_detect"),
            "tier": score.get("tier"),
            "certification_reasons": score.get("certification_reasons", []),
            "valid_for_model_comparison": bool(metrics.get("valid_for_model_comparison", True)),
        }
        rows.append(row)

    return rows


def _collect_per_dataset_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()

    for _, run_row in df.iterrows():
        run_dir = Path(str(run_row["run_dir"]))
        pred_file = run_dir / "predictions.jsonl"
        verdict_file = run_dir / "verdicts.jsonl"
        traj_file = run_dir / "trajectory.jsonl"
        if not pred_file.exists():
            continue
        predictions = read_jsonl(pred_file)
        verdicts = read_jsonl(verdict_file) if verdict_file.exists() else []
        trajectories = read_jsonl(traj_file) if traj_file.exists() else []

        datasets = sorted({str(r.get("dataset", "unknown")) for r in predictions})
        for dataset in datasets:
            pred_ds = [r for r in predictions if str(r.get("dataset", "unknown")) == dataset]
            verdict_ds = [r for r in verdicts if str(r.get("dataset", "unknown")) == dataset]
            traj_ds = [r for r in trajectories if str(r.get("dataset", "unknown")) == dataset]
            vio = compute_violation_rates(verdict_ds)
            fac = compute_claim_precision(pred_ds)
            rows.append(
                {
                    "timestamp": run_row["timestamp"],
                    "domain": run_row["domain"],
                    "model": run_row["model"],
                    "config": run_row["config"],
                    "seed": int(run_row["seed"]),
                    "dataset": dataset,
                    "n_predictions": float(len(pred_ds)),
                    "hard_violation_rate": vio.get("hard_violation_rate", 0.0),
                    "soft_violation_rate": vio.get("soft_violation_rate", 0.0),
                    "claim_precision": fac.get("claim_precision", 0.0),
                    "generation_fallback_rate": run_row.get("generation_fallback_rate", 0.0),
                    "valid_for_model_comparison": bool(run_row.get("valid_for_model_comparison", True)),
                    "latency_avg_ms": float(np.mean([float(r.get("duration_ms", 0.0)) for r in traj_ds])) if traj_ds else 0.0,
                }
            )

    return pd.DataFrame(rows)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    tex_path = path.with_suffix(".tex")
    if df.empty:
        tex_path.write_text(df.to_latex(index=False, escape=True), encoding="utf-8")
    else:
        tex_path.write_text(
            df.to_latex(index=False, float_format=lambda x: f"{x:.3f}", escape=True),
            encoding="utf-8",
        )


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


def _cohens_dz(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    diffs = np.array(b, dtype=float) - np.array(a, dtype=float)
    std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    if std == 0.0:
        return 0.0
    return float(np.mean(diffs) / std)


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0 for _ in range(m)]
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        value = min(1.0, p_values[idx] * factor)
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def _aggregate_domain_table(
    sub: pd.DataFrame,
    *,
    r_col: str,
    hard_col: str,
    soft_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, config), grp in sub.groupby(["model", "config"]):
        r_vals = [float(v) for v in grp[r_col].dropna().tolist()]
        r_low, r_high = _bootstrap_ci(r_vals) if r_vals else (0.0, 0.0)
        rows.append(
            {
                "model": model,
                "config": config,
                "n_runs": int(len(grp)),
                "R_mean": float(np.mean(r_vals)) if r_vals else 0.0,
                "R_ci_low": r_low,
                "R_ci_high": r_high,
                "R_prevent_mean": float(grp["R_prevent"].mean()) if "R_prevent" in grp else np.nan,
                "R_detect_mean": float(grp["R_detect"].mean()) if "R_detect" in grp else np.nan,
                "hard_violation_rate": float(grp[hard_col].mean()),
                "soft_violation_rate": float(grp[soft_col].mean()),
                "claim_precision": float(grp["claim_precision"].mean()),
                "source_attribution_score": float(grp["source_attribution_score"].mean()),
                "abstention_rate": float(grp["abstention_rate"].mean()) if "abstention_rate" in grp else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "config"]) if rows else pd.DataFrame()


def _make_significance_table(
    df: pd.DataFrame,
    *,
    r_col: str,
    min_paired_seeds: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (domain, model), grp in df.groupby(["domain", "model"]):
        pivot = (
            grp[grp["config"].isin(["none", "full"])]
            .groupby(["seed", "config"], as_index=False)[r_col]
            .mean()
            .pivot(index="seed", columns="config", values=r_col)
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
        pvalue = _paired_permutation_pvalue(none_vals, full_vals) if n >= min_paired_seeds else float("nan")
        effect_size_dz = _cohens_dz(none_vals, full_vals)
        rows.append(
            {
                "domain": domain,
                "model": model,
                "n_paired_seeds": n,
                "meets_min_seed_requirement": n >= min_paired_seeds,
                "R_delta_full_minus_none": delta,
                "p_value_permutation": pvalue,
                "effect_size_dz": effect_size_dz,
            }
        )
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["domain", "model"]).reset_index(drop=True)
    valid_mask = out["p_value_permutation"].notna()
    if bool(valid_mask.any()):
        raw_vals = [float(v) for v in out.loc[valid_mask, "p_value_permutation"].tolist()]
        holm_vals = _holm_adjust(raw_vals)
        out.loc[valid_mask, "p_value_holm"] = holm_vals
        out.loc[valid_mask, "significant_holm_0p05"] = out.loc[valid_mask, "p_value_holm"] <= 0.05
    else:
        out["p_value_holm"] = np.nan
        out["significant_holm_0p05"] = False
    out["significant_holm_0p05"] = out["significant_holm_0p05"].fillna(False)
    out["underpowered"] = ~out["meets_min_seed_requirement"].astype(bool)
    return out


def make_tables(
    runs_root: str | Path = "runs",
    output_dir: str | Path = "outputs/tables",
    certified_only: bool = True,
    comparison_track: str = "audited",
    benchmark_role: str | None = None,
) -> list[Path]:
    track = str(comparison_track).strip().lower()
    if track not in {"audited", "operational", "both"}:
        raise ValueError("comparison_track must be one of: audited, operational, both")
    primary_track = "audited" if track in {"audited", "both"} else "operational"
    r_col = "R_audited_raw" if primary_track == "audited" else "R_operational_raw"
    r_cert_col = "R_audited_certified" if primary_track == "audited" else "R_operational_certified"
    hard_col = "audited_hard_violation_rate" if primary_track == "audited" else "operational_hard_violation_rate"
    soft_col = "audited_soft_violation_rate" if primary_track == "audited" else "operational_soft_violation_rate"

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
    if benchmark_role is not None:
        df = df[df["benchmark_role"].astype(str) == str(benchmark_role)]
    for col, default in [
        ("hard_violation_rate", 0.0),
        ("soft_violation_rate", 0.0),
        ("claim_precision", 0.0),
        ("source_attribution_score", 0.0),
        ("abstention_rate", 0.0),
        ("latency_avg_ms", 0.0),
        ("latency_p95_ms", 0.0),
        ("layer_pre_ms", 0.0),
        ("layer_runtime_ms", 0.0),
        ("layer_posthoc_ms", 0.0),
        ("prevention_hard_violation_rate", 0.0),
        ("detection_hard_violation_rate", 0.0),
        ("estimated_cost_usd", 0.0),
        ("estimated_total_tokens", 0.0),
        ("generation_fallback_rate", 0.0),
        ("api_failure_rate", 0.0),
        ("R", np.nan),
        ("R_raw", np.nan),
        ("R_certified", np.nan),
        ("R_operational_raw", np.nan),
        ("R_operational_certified", np.nan),
        ("R_audited_raw", np.nan),
        ("R_audited_certified", np.nan),
        ("R_prevent", np.nan),
        ("R_detect", np.nan),
        ("valid_for_model_comparison", True),
        ("sweep_stage", "unknown"),
        ("operational_hard_violation_rate", np.nan),
        ("operational_soft_violation_rate", np.nan),
        ("audited_hard_violation_rate", np.nan),
        ("audited_soft_violation_rate", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default
    if df["R_operational_raw"].isna().all():
        df["R_operational_raw"] = df["R_raw"]
    if df["R_operational_certified"].isna().all():
        df["R_operational_certified"] = df["R_certified"]
    if df["R_audited_raw"].isna().all():
        df["R_audited_raw"] = df["R_raw"]
    if df["R_audited_certified"].isna().all():
        df["R_audited_certified"] = df["R_certified"]
    if df["operational_hard_violation_rate"].isna().all():
        df["operational_hard_violation_rate"] = df["hard_violation_rate"]
    if df["operational_soft_violation_rate"].isna().all():
        df["operational_soft_violation_rate"] = df["soft_violation_rate"]
    if df["audited_hard_violation_rate"].isna().all():
        df["audited_hard_violation_rate"] = df.get("residual_hard_violation_rate", df["hard_violation_rate"])
    if df["audited_soft_violation_rate"].isna().all():
        df["audited_soft_violation_rate"] = df.get("residual_soft_violation_rate", df["soft_violation_rate"])

    valid_df = df[df["valid_for_model_comparison"] == True]  # noqa: E712
    capability_df = valid_df.copy()
    if certified_only:
        capability_df = capability_df[
            (capability_df["sweep_stage"].astype(str).str.lower() == "full")
            & capability_df[r_cert_col].notna()
        ]

    warning_path = out_dir / "WARNING_NO_VALID_RUNS_FOR_MODEL_COMPARISON.txt"
    if capability_df.empty:
        warning_path.write_text(
            "No certified full-stage runs passed validity gates for model comparison. Capability tables are intentionally empty.\n",
            encoding="utf-8",
        )
        outputs.append(warning_path)
    elif warning_path.exists():
        warning_path.unlink()

    # Domain tables with confidence intervals (model-capability view).
    for domain in ["healthcare", "finance", "hr"]:
        sub = capability_df[capability_df["domain"] == domain]
        agg = _aggregate_domain_table(sub, r_col=r_col, hard_col=hard_col, soft_col=soft_col) if not sub.empty else pd.DataFrame(
            columns=[
                "model",
                "config",
                "n_runs",
                "R_mean",
                "R_ci_low",
                "R_ci_high",
                "R_prevent_mean",
                "R_detect_mean",
                "hard_violation_rate",
                "soft_violation_rate",
                "claim_precision",
                "source_attribution_score",
                "abstention_rate",
            ]
        )
        out = out_dir / f"{domain}.csv"
        _write_table(agg, out)
        outputs.append(out)

    # Ablation across domains/configs.
    ablation = (
        capability_df.groupby(["domain", "config"], as_index=False)
        .agg(
            R_mean=(r_col, "mean"),
            R_prevent_mean=("R_prevent", "mean"),
            R_detect_mean=("R_detect", "mean"),
            hard_violation_rate=(hard_col, "mean"),
            soft_violation_rate=(soft_col, "mean"),
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
        capability_df.groupby(["domain", "model", "config"], as_index=False)
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
        capability_df.groupby(["domain", "model", "config"], as_index=False)
        .agg(
            prevention_hard_violation_rate=("prevention_hard_violation_rate", "mean"),
            detection_hard_violation_rate=("detection_hard_violation_rate", "mean"),
            R_prevent=("R_prevent", "mean"),
            R_detect=("R_detect", "mean"),
        )
        .sort_values(["domain", "model", "config"])
    )
    prev_det_path = out_dir / "prevention_detection.csv"
    _write_table(prev_det, prev_det_path)
    outputs.append(prev_det_path)

    # Cost-assurance frontier.
    cost = (
        capability_df.groupby(["domain", "model", "config"], as_index=False)
        .agg(
            R_mean=(r_col, "mean"),
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

    # Paired significance for full vs none with seed-requirement check.
    significance = _make_significance_table(capability_df, r_col=r_col, min_paired_seeds=5)
    if significance.empty:
        significance = pd.DataFrame(
            columns=[
                "domain",
                "model",
                "n_paired_seeds",
                "meets_min_seed_requirement",
                "R_delta_full_minus_none",
                "p_value_permutation",
                "effect_size_dz",
                "p_value_holm",
                "significant_holm_0p05",
                "underpowered",
            ]
        )
    significance_path = out_dir / "significance.csv"
    _write_table(significance, significance_path)
    outputs.append(significance_path)

    # Dataset-level analysis.
    per_dataset = _collect_per_dataset_rows(capability_df)
    if not per_dataset.empty:
        per_dataset_agg = (
            per_dataset.groupby(["domain", "dataset", "model", "config"], as_index=False)
            .agg(
                n_predictions=("n_predictions", "sum"),
                hard_violation_rate=("hard_violation_rate", "mean"),
                soft_violation_rate=("soft_violation_rate", "mean"),
                claim_precision=("claim_precision", "mean"),
                latency_avg_ms=("latency_avg_ms", "mean"),
            )
            .sort_values(["domain", "dataset", "model", "config"])
        )
    else:
        per_dataset_agg = pd.DataFrame(
            columns=[
                "domain",
                "dataset",
                "model",
                "config",
                "n_predictions",
                "hard_violation_rate",
                "soft_violation_rate",
                "claim_precision",
                "latency_avg_ms",
            ]
        )
    per_dataset_path = out_dir / "per_dataset.csv"
    _write_table(per_dataset_agg, per_dataset_path)
    outputs.append(per_dataset_path)

    supplemental_stress = per_dataset_agg[
        per_dataset_agg["dataset"].astype(str).str.contains("agentic_stress_hr", na=False)
    ].copy() if not per_dataset_agg.empty else pd.DataFrame()
    if supplemental_stress.empty:
        supplemental_stress = pd.DataFrame(
            columns=[
                "domain",
                "dataset",
                "model",
                "config",
                "n_predictions",
                "hard_violation_rate",
                "soft_violation_rate",
                "claim_precision",
                "latency_avg_ms",
            ]
        )
    supplemental_stress_path = out_dir / "supplemental_stress.csv"
    _write_table(supplemental_stress, supplemental_stress_path)
    outputs.append(supplemental_stress_path)

    # Separate dashboards for model capability vs infrastructure reliability.
    capability = (
        capability_df.groupby(["model"], as_index=False)
        .agg(
            n_runs=(r_col, "count"),
            R_mean=(r_col, "mean"),
            R_certified_mean=(r_cert_col, "mean"),
            R_prevent_mean=("R_prevent", "mean"),
            R_detect_mean=("R_detect", "mean"),
            hard_violation_rate=(hard_col, "mean"),
            claim_precision=("claim_precision", "mean"),
        )
        .sort_values("R_mean", ascending=False) if not capability_df.empty else pd.DataFrame(
            columns=[
                "model",
                "n_runs",
                "R_mean",
                "R_certified_mean",
                "R_prevent_mean",
                "R_detect_mean",
                "hard_violation_rate",
                "claim_precision",
            ]
        )
    )
    capability_path = out_dir / "model_capability.csv"
    _write_table(capability, capability_path)
    outputs.append(capability_path)

    exploratory = (
        valid_df.groupby(["domain", "model", "config"], as_index=False)
        .agg(
            n_runs=(r_col, "count"),
            R_raw_mean=(r_col, "mean"),
            hard_violation_rate=(hard_col, "mean"),
            claim_precision=("claim_precision", "mean"),
            generation_fallback_rate=("generation_fallback_rate", "mean"),
            api_failure_rate=("api_failure_rate", "mean"),
        )
        .sort_values(["domain", "model", "config"])
    ) if not valid_df.empty else pd.DataFrame(
        columns=[
            "domain",
            "model",
            "config",
            "n_runs",
            "R_raw_mean",
            "hard_violation_rate",
            "claim_precision",
            "generation_fallback_rate",
            "api_failure_rate",
        ]
    )
    exploratory_path = out_dir / "exploratory_raw.csv"
    _write_table(exploratory, exploratory_path)
    outputs.append(exploratory_path)

    infra = (
        df.groupby(["model"], as_index=False)
        .agg(
            n_runs=("R", "count"),
            valid_run_rate=("valid_for_model_comparison", "mean"),
            generation_fallback_rate=("generation_fallback_rate", "mean"),
            api_failure_rate=("api_failure_rate", "mean"),
            abstention_rate=("abstention_rate", "mean"),
            latency_avg_ms=("latency_avg_ms", "mean"),
        )
        .sort_values("valid_run_rate", ascending=False)
    )
    infra_path = out_dir / "infrastructure_reliability.csv"
    _write_table(infra, infra_path)
    outputs.append(infra_path)

    if track == "both":
        op_df = valid_df.copy()
        op_cap = op_df[
            (op_df["sweep_stage"].astype(str).str.lower() == "full")
            & op_df["R_operational_certified"].notna()
        ] if certified_only else op_df
        op_ablation = (
            op_cap.groupby(["domain", "config"], as_index=False)
            .agg(
                R_mean=("R_operational_raw", "mean"),
                hard_violation_rate=("operational_hard_violation_rate", "mean"),
                soft_violation_rate=("operational_soft_violation_rate", "mean"),
                claim_precision=("claim_precision", "mean"),
            )
            .sort_values(["domain", "config"])
        ) if not op_cap.empty else pd.DataFrame(
            columns=["domain", "config", "R_mean", "hard_violation_rate", "soft_violation_rate", "claim_precision"]
        )
        op_path = out_dir / "operational_ablation.csv"
        _write_table(op_ablation, op_path)
        outputs.append(op_path)

    return outputs
