from rava.scoring.reliability import compute_reliability_score
from rava.scoring.tiers import assign_tier


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
        "latency_avg_ms": 200.0,
    }
    weights = {
        "hard_safety": 0.30,
        "soft_quality": 0.10,
        "factuality": 0.25,
        "calibration": 0.10,
        "fairness": 0.10,
        "attribution": 0.10,
        "latency": 0.05,
    }
    out = compute_reliability_score(metrics, weights)
    assert 0.0 <= out["R"] <= 1.0
    assert out["tier"] in {
        "Tier 1 (Advisory)",
        "Tier 2 (Supervised Autonomy)",
        "Tier 3 (Human-in-the-Loop Required)",
    }


def test_tier_gate_on_hard_violations():
    assert (
        assign_tier(
            reliability_score=0.99,
            hard_violation_rate=0.3,
            fairness_score=0.95,
            factuality_score=0.95,
        )
        == "Tier 3 (Human-in-the-Loop Required)"
    )
