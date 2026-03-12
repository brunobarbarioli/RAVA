from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from rava.experiments.tables import make_tables
from rava.utils.serialization import read_json, write_json


@dataclass
class EvidenceSummary:
    runs_root: str
    n_runs: int
    valid_run_rate: float
    min_seed_coverage_rate: float
    powered_comparison_rate: float
    score_0_to_100: float
    grade: str
    notes: list[str]


def _collect_rows(runs_root: str | Path) -> pd.DataFrame:
    root = Path(runs_root)
    rows: list[dict[str, Any]] = []
    for metrics_path in root.glob("**/metrics.json"):
        run_dir = metrics_path.parent
        parts = run_dir.parts
        try:
            seed = int(parts[-1])
            config = parts[-2]
            model = parts[-3]
            domain = parts[-4]
        except Exception:
            continue
        metrics = read_json(metrics_path)
        report = read_json(run_dir / "report.json") if (run_dir / "report.json").exists() else {}
        score = report.get("score", {}) if isinstance(report, dict) else {}
        rows.append(
            {
                "domain": domain,
                "model": model,
                "config": config,
                "seed": seed,
                "R": score.get("R"),
                "R_certified": score.get("R_certified"),
                "R_operational_raw": score.get("R_operational_raw"),
                "R_operational_certified": score.get("R_operational_certified"),
                "R_audited_raw": score.get("R_audited_raw", score.get("R")),
                "R_audited_certified": score.get("R_audited_certified", score.get("R_certified")),
                "tier": score.get("tier"),
                "certification_reasons": score.get("certification_reasons", []),
                "certification_reasons_operational": score.get("certification_reasons_operational", []),
                "certification_reasons_audited": score.get("certification_reasons_audited", score.get("certification_reasons", [])),
                "certification_status": score.get("certification_status", "unknown"),
                "sweep_stage": str(metrics.get("sweep_stage", "unknown")),
                "certification_eligible": bool(metrics.get("certification_eligible", True)),
                "dataset_profile": str(metrics.get("dataset_profile", "unknown")),
                "benchmark_role": str(metrics.get("benchmark_role", "mixed")),
                "valid_for_model_comparison": bool(metrics.get("valid_for_model_comparison", True)),
                "generation_fallback_rate": float(metrics.get("generation_fallback_rate", 0.0)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _grade(score_0_to_100: float) -> str:
    if score_0_to_100 >= 92:
        return "A"
    if score_0_to_100 >= 85:
        return "A-"
    if score_0_to_100 >= 78:
        return "B+"
    if score_0_to_100 >= 72:
        return "B"
    if score_0_to_100 >= 66:
        return "B-"
    if score_0_to_100 >= 60:
        return "C+"
    if score_0_to_100 >= 54:
        return "C"
    if score_0_to_100 >= 48:
        return "C-"
    return "D"


def _cap_grade(current: str, cap: str) -> str:
    ordering = {
        "A": 9,
        "A-": 8,
        "B+": 7,
        "B": 6,
        "B-": 5,
        "C+": 4,
        "C": 3,
        "C-": 2,
        "D": 1,
    }
    if ordering.get(current, 1) <= ordering.get(cap, 1):
        return current
    return cap


def summarize_evidence(
    runs_root: str | Path,
    output_path: str | Path | None = None,
    comparison_track: str = "audited",
    benchmark_role: str | None = None,
) -> dict[str, Any]:
    track = str(comparison_track).strip().lower()
    if track not in {"audited", "operational"}:
        raise ValueError("comparison_track must be one of: audited, operational")
    r_col = "R_audited_raw" if track == "audited" else "R_operational_raw"
    r_cert_col = "R_audited_certified" if track == "audited" else "R_operational_certified"
    reason_col = "certification_reasons_audited" if track == "audited" else "certification_reasons_operational"

    # Ensure significance and derived tables exist.
    temp_tables_dir = Path("outputs") / "tables" / "_evidence_tmp"
    make_tables(
        runs_root=runs_root,
        output_dir=temp_tables_dir,
        comparison_track=track,
        benchmark_role=benchmark_role,
    )

    df = _collect_rows(runs_root)
    if benchmark_role is not None and not df.empty:
        df = df[df["benchmark_role"].astype(str) == str(benchmark_role)]
    if df.empty:
        summary = EvidenceSummary(
            runs_root=str(runs_root),
            n_runs=0,
            valid_run_rate=0.0,
            min_seed_coverage_rate=0.0,
            powered_comparison_rate=0.0,
            score_0_to_100=0.0,
            grade="D",
            notes=["No run artifacts found."],
        )
        payload = summary.__dict__
        if output_path:
            write_json(output_path, payload)
        return payload

    valid_run_rate = float(df["valid_for_model_comparison"].mean())
    certified_df = df[df[r_cert_col].notna()]
    certified_run_rate = float(len(certified_df) / len(df)) if len(df) else 0.0
    seed_cov = (
        df.groupby(["domain", "model", "config"])["seed"]
        .nunique()
        .reset_index(name="n_seeds")
    )
    min_seed_coverage_rate = float((seed_cov["n_seeds"] >= 5).mean()) if not seed_cov.empty else 0.0
    certified_seed_cov = (
        certified_df.groupby(["domain", "model", "config"])["seed"]
        .nunique()
        .reset_index(name="n_seeds")
        if not certified_df.empty
        else pd.DataFrame(columns=["domain", "model", "config", "n_seeds"])
    )
    certified_seed_coverage_rate = (
        float((certified_seed_cov["n_seeds"] >= 5).mean()) if not certified_seed_cov.empty else 0.0
    )

    sig_path = temp_tables_dir / "significance.csv"
    powered_comparison_rate = 0.0
    if sig_path.exists():
        sig_df = pd.read_csv(sig_path)
        if not sig_df.empty and "meets_min_seed_requirement" in sig_df.columns:
            powered_comparison_rate = float(sig_df["meets_min_seed_requirement"].astype(float).mean())

    matrix_path = Path(runs_root) / "matrix_completion.json"
    matrix_completion_rate = 0.0
    matrix_expected = 0
    matrix_completed = 0
    matrix_missing = 0
    if matrix_path.exists():
        matrix_payload = read_json(matrix_path)
        matrix_expected = int(matrix_payload.get("expected_cells", 0))
        matrix_completed = int(matrix_payload.get("completed_cells", 0))
        matrix_missing = int(matrix_payload.get("missing_cells", 0))
        matrix_completion_rate = float(matrix_payload.get("completion_rate", 0.0))

    # Weighted EMNLP-readiness evidence score.
    score_0_to_100 = 100.0 * (
        0.25 * valid_run_rate
        + 0.30 * certified_run_rate
        + 0.20 * min_seed_coverage_rate
        + 0.15 * powered_comparison_rate
        + 0.10 * matrix_completion_rate
    )
    grade = _grade(score_0_to_100)
    if valid_run_rate < 0.95 or certified_run_rate <= 0.0:
        grade = "D"
    if matrix_completion_rate < 1.0:
        grade = _cap_grade(grade, "A-")

    notes: list[str] = []
    if valid_run_rate < 0.95:
        notes.append("Run-quality gate failed for a material fraction of runs.")
    if certified_run_rate <= 0.0:
        notes.append("No certified runs available; capability evidence is non-certifying.")
    if certified_seed_coverage_rate < 1.0:
        notes.append("Certified comparisons do not all meet the >=5 seed replication target.")
    if min_seed_coverage_rate < 1.0:
        notes.append("Some comparisons do not meet the >=5 seed replication target.")
    if powered_comparison_rate < 1.0:
        notes.append("Some ablation significance comparisons are underpowered.")
    if matrix_completion_rate < 1.0:
        notes.append("Full-stage matrix is incomplete; missing cells reduce comparison validity.")

    reason_counts: dict[str, int] = {}
    for reasons in df[reason_col].tolist():
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            key = str(reason)
            reason_counts[key] = reason_counts.get(key, 0) + 1

    summary = EvidenceSummary(
        runs_root=str(runs_root),
        n_runs=int(len(df)),
        valid_run_rate=valid_run_rate,
        min_seed_coverage_rate=min_seed_coverage_rate,
        powered_comparison_rate=powered_comparison_rate,
        score_0_to_100=score_0_to_100,
        grade=grade,
        notes=notes,
    )
    payload = summary.__dict__
    payload["certified_run_rate"] = certified_run_rate
    payload["certified_seed_coverage_rate"] = certified_seed_coverage_rate
    payload["certification_reason_counts"] = reason_counts
    payload["comparison_track"] = track
    payload["matrix_expected_cells"] = matrix_expected
    payload["matrix_completed_cells"] = matrix_completed
    payload["matrix_missing_cells"] = matrix_missing
    payload["matrix_completion_rate"] = matrix_completion_rate
    payload["primary_reliability_column"] = r_col
    payload["primary_certified_column"] = r_cert_col
    payload["benchmark_role"] = benchmark_role
    if output_path:
        write_json(output_path, payload)
    return payload
