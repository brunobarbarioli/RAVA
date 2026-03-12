from rava.metrics.fairness import (
    compute_fairness_metrics_by_attribute,
    demographic_parity_difference,
    equalized_odds_difference,
    four_fifths_ratio,
)


def test_four_fifths_rule():
    y_pred = [1, 1, 0, 0]
    groups = ["A", "A", "B", "B"]
    # group A rate = 1.0, B = 0.0 => ratio 0
    assert four_fifths_ratio(y_pred, groups) == 0.0


def test_demographic_parity_difference():
    y_pred = [1, 0, 1, 0, 1, 0]
    groups = ["A", "A", "B", "B", "B", "A"]
    dp = demographic_parity_difference(y_pred, groups)
    assert abs(dp - (2 / 3 - 1 / 3)) < 1e-9


def test_equalized_odds_difference():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0]
    groups = ["A", "A", "A", "B", "B", "B"]
    eo = equalized_odds_difference(y_true, y_pred, groups)
    assert eo >= 0.0
    assert eo <= 1.0


def test_attribute_level_fairness_metrics():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 1]
    protected = [
        {"gender": "woman", "race_ethnicity": "a", "age_group": "18-29", "disability": "no"},
        {"gender": "woman", "race_ethnicity": "b", "age_group": "18-29", "disability": "no"},
        {"gender": "man", "race_ethnicity": "a", "age_group": "30-44", "disability": "yes"},
        {"gender": "man", "race_ethnicity": "b", "age_group": "30-44", "disability": "yes"},
    ]
    out = compute_fairness_metrics_by_attribute(y_true=y_true, y_pred=y_pred, protected_rows=protected)
    assert "min_group_count_primary_attributes" in out
    assert "intersectional_min_group_count" in out
    assert out["num_groups"] >= 1.0
