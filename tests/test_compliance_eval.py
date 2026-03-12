from pathlib import Path

from rava.experiments.compliance_eval import evaluate_compliance_challenges


def test_compliance_challenge_summary_contains_ci_fields(tmp_path: Path):
    output_path = tmp_path / "compliance.json"
    summary = evaluate_compliance_challenges(
        challenge_root="configs/compliance_challenges",
        domains=["healthcare", "finance", "hr"],
        output_path=output_path,
    )
    assert output_path.exists()
    assert summary["overall"]["n_cases"] > 0
    assert "r_pre_ci_low" in summary["overall"]
    assert "r_rt_ci_high" in summary["overall"]
    assert "q_hat_ci_high" in summary["overall"]
    assert set(summary["domains"].keys()) == {"healthcare", "finance", "hr"}
