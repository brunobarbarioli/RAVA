from pathlib import Path

from rava.scoring.gates import compute_stats_gate
from rava.specs.parser import load_spec


def test_stats_gate_constraint_aware_no_applicable_grouped_constraints():
    spec = load_spec(Path("specs/hr.yaml"))
    metrics = {
        "num_groups_gender": 1.0,
        "num_groups_race_ethnicity": 1.0,
        "num_groups_age_group": 1.0,
        "num_groups_disability": 1.0,
        "min_group_count_gender": 100.0,
        "min_group_count_race_ethnicity": 100.0,
        "min_group_count_age_group": 100.0,
        "min_group_count_disability": 100.0,
        "four_fifths_ratio_min": 1.0,
        "demographic_parity_difference": 0.0,
        "equalized_odds_difference": 0.0,
    }
    out = compute_stats_gate(metrics=metrics, spec=spec, policy="constraint_aware", min_group_count=30.0)
    assert out["V_stats"] is True
    assert "no_applicable_stat_constraints" in out["reason_codes"]
    assert set(out["na_constraint_ids"]) == {"HR-Σ1", "HR-Σ2", "HR-Σ3"}


def test_stats_gate_constraint_aware_fails_on_underpowered_applicable_groups():
    spec = load_spec(Path("specs/hr.yaml"))
    metrics = {
        "num_groups_gender": 2.0,
        "num_groups_race_ethnicity": 2.0,
        "num_groups_age_group": 2.0,
        "num_groups_disability": 2.0,
        "min_group_count_gender": 10.0,
        "min_group_count_race_ethnicity": 12.0,
        "min_group_count_age_group": 11.0,
        "min_group_count_disability": 10.0,
        "four_fifths_ratio_min": 0.95,
        "demographic_parity_difference": 0.01,
        "equalized_odds_difference": 0.01,
    }
    out = compute_stats_gate(metrics=metrics, spec=spec, policy="constraint_aware", min_group_count=30.0)
    assert out["V_stats"] is False
    assert any("subgroup_min_below_30" in reason for reason in out["reason_codes"])
