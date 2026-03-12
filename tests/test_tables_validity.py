import json
from pathlib import Path

import pandas as pd

from rava.experiments.tables import make_tables


def test_tables_do_not_substitute_invalid_runs(tmp_path: Path):
    run_dir = tmp_path / "runs" / "20260101_000000" / "healthcare" / "mock-v1" / "none" / "42"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "hard_violation_rate": 0.1,
        "soft_violation_rate": 0.1,
        "claim_precision": 0.8,
        "source_attribution_score": 0.7,
        "abstention_rate": 0.0,
        "latency_avg_ms": 10.0,
        "latency_p95_ms": 12.0,
        "layer_pre_ms": 0.0,
        "layer_runtime_ms": 0.0,
        "layer_posthoc_ms": 0.0,
        "prevention_hard_violation_rate": 0.1,
        "detection_hard_violation_rate": 0.1,
        "estimated_cost_usd": 0.01,
        "estimated_total_tokens": 100.0,
        "generation_fallback_rate": 1.0,
        "api_failure_rate": 1.0,
        "valid_for_model_comparison": False,
    }
    report = {
        "score": {
            "R": 0.75,
            "R_raw": 0.75,
            "R_certified": None,
            "R_prevent": 0.7,
            "R_detect": 0.8,
            "tier": "Non-certifying (Infrastructure/Data Invalid)",
        }
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    out_dir = tmp_path / "tables"
    make_tables(runs_root=tmp_path / "runs", output_dir=out_dir)

    warning_path = out_dir / "WARNING_NO_VALID_RUNS_FOR_MODEL_COMPARISON.txt"
    assert warning_path.exists()

    capability = pd.read_csv(out_dir / "model_capability.csv")
    assert capability.empty

    infra = pd.read_csv(out_dir / "infrastructure_reliability.csv")
    assert len(infra) == 1


def test_tables_certified_only_filters_noncertifying_rows(tmp_path: Path):
    base = tmp_path / "runs" / "20260101_000000" / "healthcare" / "mock-v1"
    run_a = base / "none" / "42"
    run_b = base / "full" / "42"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    metrics_valid = {
        "hard_violation_rate": 0.0,
        "soft_violation_rate": 0.0,
        "claim_precision": 0.8,
        "source_attribution_score": 0.7,
        "abstention_rate": 0.0,
        "latency_avg_ms": 10.0,
        "latency_p95_ms": 12.0,
        "layer_pre_ms": 0.0,
        "layer_runtime_ms": 0.0,
        "layer_posthoc_ms": 0.0,
        "prevention_hard_violation_rate": 0.0,
        "detection_hard_violation_rate": 0.0,
        "estimated_cost_usd": 0.01,
        "estimated_total_tokens": 100.0,
        "generation_fallback_rate": 0.0,
        "api_failure_rate": 0.0,
        "valid_for_model_comparison": True,
    }
    report_noncert = {
        "score": {
            "R": 0.65,
            "R_raw": 0.65,
            "R_certified": None,
            "R_prevent": 0.6,
            "R_detect": 0.7,
            "tier": "Non-certifying (Infrastructure/Data Invalid)",
        }
    }
    report_cert = {
        "score": {
            "R": 0.80,
            "R_raw": 0.80,
            "R_certified": 0.80,
            "R_prevent": 0.8,
            "R_detect": 0.8,
            "tier": "Tier 1 (Advisory)",
        }
    }
    metrics_noncert = dict(metrics_valid)
    metrics_noncert["sweep_stage"] = "full"
    metrics_cert = dict(metrics_valid)
    metrics_cert["sweep_stage"] = "full"

    (run_a / "metrics.json").write_text(json.dumps(metrics_noncert), encoding="utf-8")
    (run_a / "report.json").write_text(json.dumps(report_noncert), encoding="utf-8")
    (run_b / "metrics.json").write_text(json.dumps(metrics_cert), encoding="utf-8")
    (run_b / "report.json").write_text(json.dumps(report_cert), encoding="utf-8")

    out_dir = tmp_path / "tables"
    make_tables(runs_root=tmp_path / "runs", output_dir=out_dir, certified_only=True)
    capability = pd.read_csv(out_dir / "model_capability.csv")
    assert len(capability) == 1
    assert capability.iloc[0]["n_runs"] == 1

    make_tables(runs_root=tmp_path / "runs", output_dir=out_dir, certified_only=False)
    capability_all = pd.read_csv(out_dir / "model_capability.csv")
    assert len(capability_all) == 1
    assert capability_all.iloc[0]["n_runs"] == 2


def test_tables_default_to_audited_track(tmp_path: Path):
    run_dir = tmp_path / "runs" / "20260101_000000" / "finance" / "mock-v1" / "full" / "42"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "hard_violation_rate": 0.0,
        "soft_violation_rate": 0.0,
        "operational_hard_violation_rate": 0.0,
        "operational_soft_violation_rate": 0.0,
        "audited_hard_violation_rate": 0.6,
        "audited_soft_violation_rate": 0.4,
        "claim_precision": 0.8,
        "source_attribution_score": 0.7,
        "abstention_rate": 0.0,
        "latency_avg_ms": 10.0,
        "latency_p95_ms": 12.0,
        "layer_pre_ms": 0.0,
        "layer_runtime_ms": 0.0,
        "layer_posthoc_ms": 0.0,
        "prevention_hard_violation_rate": 0.0,
        "detection_hard_violation_rate": 0.0,
        "estimated_cost_usd": 0.01,
        "estimated_total_tokens": 100.0,
        "generation_fallback_rate": 0.0,
        "api_failure_rate": 0.0,
        "valid_for_model_comparison": True,
        "sweep_stage": "full",
    }
    report = {
        "score": {
            "R": 0.30,
            "R_raw": 0.30,
            "R_certified": 0.30,
            "R_operational_raw": 0.90,
            "R_operational_certified": 0.90,
            "R_audited_raw": 0.30,
            "R_audited_certified": 0.30,
            "R_prevent": 0.5,
            "R_detect": 0.5,
            "tier": "Tier 3 (Human-in-the-Loop Required)",
        }
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    out_dir = tmp_path / "tables"
    make_tables(runs_root=tmp_path / "runs", output_dir=out_dir, certified_only=True)
    capability = pd.read_csv(out_dir / "model_capability.csv")
    assert len(capability) == 1
    assert float(capability.iloc[0]["R_mean"]) == 0.30


def test_tables_can_filter_primary_benchmark_role(tmp_path: Path):
    base = tmp_path / "runs" / "20260101_000000"
    primary_run = base / "healthcare" / "mock-v1" / "full" / "42"
    secondary_run = base / "finance" / "mock-v1" / "full" / "42"
    primary_run.mkdir(parents=True, exist_ok=True)
    secondary_run.mkdir(parents=True, exist_ok=True)

    common_metrics = {
        "hard_violation_rate": 0.0,
        "soft_violation_rate": 0.0,
        "claim_precision": 0.8,
        "source_attribution_score": 0.7,
        "abstention_rate": 0.0,
        "latency_avg_ms": 10.0,
        "latency_p95_ms": 12.0,
        "generation_fallback_rate": 0.0,
        "api_failure_rate": 0.0,
        "valid_for_model_comparison": True,
        "sweep_stage": "full",
    }
    report = {
        "score": {
            "R": 0.8,
            "R_raw": 0.8,
            "R_certified": 0.8,
            "R_audited_raw": 0.8,
            "R_audited_certified": 0.8,
            "tier": "Tier 1 (Advisory)",
        }
    }

    primary_metrics = dict(common_metrics, benchmark_role="primary_certification")
    secondary_metrics = dict(common_metrics, benchmark_role="diagnostic_secondary")
    (primary_run / "metrics.json").write_text(json.dumps(primary_metrics), encoding="utf-8")
    (primary_run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (secondary_run / "metrics.json").write_text(json.dumps(secondary_metrics), encoding="utf-8")
    (secondary_run / "report.json").write_text(json.dumps(report), encoding="utf-8")

    out_dir = tmp_path / "tables"
    make_tables(
        runs_root=tmp_path / "runs",
        output_dir=out_dir,
        certified_only=True,
        benchmark_role="primary_certification",
    )
    capability = pd.read_csv(out_dir / "model_capability.csv")
    assert len(capability) == 1
    assert int(capability.iloc[0]["n_runs"]) == 1
