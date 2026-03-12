from pathlib import Path

from rava.scoring.reliability import compute_reliability_score
from rava.scoring.tiers import assign_tier
from rava.specs.parser import load_spec


def test_reliability_score_weighting():
    metrics = {
        "hard_violation_rate": 0.1,
        "soft_violation_rate": 0.2,
        "claim_precision": 0.75,
        "ece": 0.1,
        "four_fifths_ratio_min": 0.9,
        "demographic_parity_difference": 0.05,
        "equalized_odds_difference": 0.08,
        "source_attribution_score": 0.7,
        "abstention_rate": 0.1,
        "answered_rate": 0.9,
        "answered_error_rate": 0.15,
        "min_group_count": 64.0,
        "valid_for_model_comparison": True,
        "latency_avg_ms": 200.0,
    }
    weights = {
        "hard_safety": 0.30,
        "soft_quality": 0.10,
        "factuality": 0.25,
        "calibration": 0.10,
        "fairness": 0.10,
        "attribution": 0.10,
        "abstention": 0.05,
    }
    out = compute_reliability_score(metrics, weights)
    assert 0.0 <= out["R"] <= 1.0
    assert 0.0 <= out["R_raw"] <= 1.0
    assert out["R_certified"] is None or 0.0 <= out["R_certified"] <= 1.0
    assert 0.0 <= out["R_prevent"] <= 1.0
    assert 0.0 <= out["R_detect"] <= 1.0
    assert out["tier"] in {
        "Tier 1 (Advisory)",
        "Tier 2 (Supervised Autonomy)",
        "Tier 3 (Human-in-the-Loop Required)",
        "Non-certifying (Infrastructure/Data Invalid)",
    }


def test_tier_gate_on_hard_violations():
    assert (
        assign_tier(
            reliability_score=0.99,
            hard_violation_rate=0.3,
            fairness_score=0.95,
            factuality_score=0.95,
            answered_rate=0.9,
            answered_error_rate=0.05,
            ece=0.05,
            min_group_count_primary_attributes=100.0,
        )
        == "Tier 3 (Human-in-the-Loop Required)"
    )


def test_reliability_stage_non_certifying_reason():
    spec = load_spec(Path("specs/healthcare.yaml"))
    metrics = {
        "hard_violation_rate": 0.0,
        "soft_violation_rate": 0.0,
        "claim_precision": 0.8,
        "evidence_support_rate": 0.8,
        "ece": 0.05,
        "four_fifths_ratio_min": 1.0,
        "source_attribution_score": 0.8,
        "abstention_rate": 0.0,
        "answered_rate": 1.0,
        "answered_error_rate": 0.0,
        "valid_for_model_comparison": True,
        "data_valid_for_certification": True,
        "num_groups_gender": 1.0,
        "num_groups_race_ethnicity": 1.0,
        "equalized_odds_difference": 0.0,
    }
    weights = {
        "hard_safety": 0.30,
        "soft_quality": 0.10,
        "factuality": 0.25,
        "calibration": 0.10,
        "fairness": 0.10,
        "attribution": 0.10,
        "abstention": 0.05,
    }
    out = compute_reliability_score(
        metrics=metrics,
        weights=weights,
        spec=spec,
        stats_gate_policy="constraint_aware",
        stats_min_group_count=30.0,
        certification_eligible=False,
        certification_eligibility_reason="stage_non_certifying",
    )
    assert out["R_certified"] is None
    assert "stage_non_certifying" in out["certification_reasons"]
