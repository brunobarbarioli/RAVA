from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def _group_indices(groups: Iterable[str]) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = defaultdict(list)
    for idx, g in enumerate(groups):
        mapping[str(g)].append(idx)
    return mapping


def _selection_rate(y_pred: list[int], idxs: list[int]) -> float:
    if not idxs:
        return 0.0
    return sum(int(y_pred[i]) for i in idxs) / len(idxs)


def four_fifths_ratio(y_pred: list[int], groups: list[str]) -> float:
    gmap = _group_indices(groups)
    if len(gmap) <= 1:
        return 1.0
    rates = [_selection_rate(y_pred, idxs) for idxs in gmap.values()]
    max_rate = max(rates)
    min_rate = min(rates)
    if max_rate == 0:
        return 1.0
    return min_rate / max_rate


def demographic_parity_difference(y_pred: list[int], groups: list[str]) -> float:
    gmap = _group_indices(groups)
    if len(gmap) <= 1:
        return 0.0
    rates = [_selection_rate(y_pred, idxs) for idxs in gmap.values()]
    return max(rates) - min(rates)


def _true_positive_rate(y_true: list[int], y_pred: list[int], idxs: list[int]) -> float:
    positives = [i for i in idxs if int(y_true[i]) == 1]
    if not positives:
        return 0.0
    tp = sum(1 for i in positives if int(y_pred[i]) == 1)
    return tp / len(positives)


def _false_positive_rate(y_true: list[int], y_pred: list[int], idxs: list[int]) -> float:
    negatives = [i for i in idxs if int(y_true[i]) == 0]
    if not negatives:
        return 0.0
    fp = sum(1 for i in negatives if int(y_pred[i]) == 1)
    return fp / len(negatives)


def equalized_odds_difference(y_true: list[int], y_pred: list[int], groups: list[str]) -> float:
    gmap = _group_indices(groups)
    if len(gmap) <= 1:
        return 0.0

    tprs = [_true_positive_rate(y_true, y_pred, idxs) for idxs in gmap.values()]
    fprs = [_false_positive_rate(y_true, y_pred, idxs) for idxs in gmap.values()]
    return max(max(tprs) - min(tprs), max(fprs) - min(fprs))


def compute_fairness_metrics(y_true: list[int], y_pred: list[int], groups: list[str]) -> dict[str, float]:
    return {
        "four_fifths_ratio_min": four_fifths_ratio(y_pred=y_pred, groups=groups),
        "demographic_parity_difference": demographic_parity_difference(y_pred=y_pred, groups=groups),
        "equalized_odds_difference": equalized_odds_difference(y_true=y_true, y_pred=y_pred, groups=groups),
    }
