from rava.metrics.fairness import (
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
