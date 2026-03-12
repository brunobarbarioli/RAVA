import json
from pathlib import Path

from rava.experiments.evidence import summarize_evidence


def test_evidence_grade_capped_to_d_when_validity_low(tmp_path: Path):
    run_dir = tmp_path / "runs" / "20260101_000000" / "finance" / "mock-v1" / "none" / "42"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "valid_for_model_comparison": False,
        "generation_fallback_rate": 1.0,
    }
    report = {"score": {"R": 0.9, "tier": "Non-certifying (Infrastructure/Data Invalid)"}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    payload = summarize_evidence(runs_root=tmp_path / "runs")
    assert payload["valid_run_rate"] == 0.0
    assert payload["grade"] == "D"
