from __future__ import annotations

from collections import defaultdict
from typing import Iterable

INVALID_GROUP_VALUES = {"", "na", "n/a", "none", "null", "unknown", "nan"}


def _group_indices(groups: Iterable[str]) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = defaultdict(list)
    for idx, g in enumerate(groups):
        mapping[str(g)].append(idx)
    return mapping


def _valid_group_label(value: str) -> bool:
    return str(value).strip().lower() not in INVALID_GROUP_VALUES


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
    valid_idxs = [i for i, g in enumerate(groups) if _valid_group_label(str(g))]
    if valid_idxs:
        y_true = [int(y_true[i]) for i in valid_idxs]
        y_pred = [int(y_pred[i]) for i in valid_idxs]
        groups = [str(groups[i]) for i in valid_idxs]
    else:
        y_true = []
        y_pred = []
        groups = []

    gmap = _group_indices(groups)
    group_sizes = [len(v) for v in gmap.values()] if gmap else []
    min_group = min(group_sizes) if group_sizes else 0
    return {
        "four_fifths_ratio_min": four_fifths_ratio(y_pred=y_pred, groups=groups),
        "demographic_parity_difference": demographic_parity_difference(y_pred=y_pred, groups=groups),
        "equalized_odds_difference": equalized_odds_difference(y_true=y_true, y_pred=y_pred, groups=groups),
        "num_groups": float(len(gmap)),
        "min_group_count": float(min_group),
    }


def compute_fairness_metrics_by_attribute(
    y_true: list[int],
    y_pred: list[int],
    protected_rows: list[dict[str, str]],
    primary_fields: tuple[str, ...] = ("gender", "race_ethnicity", "age_group", "disability"),
) -> dict[str, float]:
    if not y_true or not y_pred or not protected_rows or len(y_true) != len(y_pred) or len(y_true) != len(protected_rows):
        return {
            "four_fifths_ratio_min": 1.0,
            "demographic_parity_difference": 0.0,
            "equalized_odds_difference": 0.0,
            "num_groups": 1.0,
            "min_group_count": 0.0,
            "intersectional_num_groups": 1.0,
            "intersectional_min_group_count": 0.0,
            "min_group_count_primary_attributes": 0.0,
        }

    metrics: dict[str, float] = {}
    ratios: list[float] = []
    dp_diffs: list[float] = []
    eo_diffs: list[float] = []
    per_field_mins: list[float] = []

    for field in primary_fields:
        groups = [str(row.get(field, "na")) for row in protected_rows]
        base = compute_fairness_metrics(y_true=y_true, y_pred=y_pred, groups=groups)
        metrics[f"four_fifths_ratio_{field}"] = base["four_fifths_ratio_min"]
        metrics[f"demographic_parity_difference_{field}"] = base["demographic_parity_difference"]
        metrics[f"equalized_odds_difference_{field}"] = base["equalized_odds_difference"]
        metrics[f"num_groups_{field}"] = base["num_groups"]
        metrics[f"min_group_count_{field}"] = base["min_group_count"]
        ratios.append(base["four_fifths_ratio_min"])
        dp_diffs.append(base["demographic_parity_difference"])
        eo_diffs.append(base["equalized_odds_difference"])
        per_field_mins.append(base["min_group_count"])

    intersectional = []
    for row in protected_rows:
        tokens = [str(row.get(field, "na")) for field in primary_fields]
        if any(not _valid_group_label(tok) for tok in tokens):
            intersectional.append("unknown")
        else:
            intersectional.append("|".join(tokens))
    inter = compute_fairness_metrics(y_true=y_true, y_pred=y_pred, groups=intersectional)

    metrics["four_fifths_ratio_min"] = min(ratios) if ratios else 1.0
    metrics["demographic_parity_difference"] = max(dp_diffs) if dp_diffs else 0.0
    metrics["equalized_odds_difference"] = max(eo_diffs) if eo_diffs else 0.0
    metrics["num_groups"] = inter["num_groups"]
    metrics["min_group_count"] = inter["min_group_count"]
    metrics["intersectional_num_groups"] = inter["num_groups"]
    metrics["intersectional_min_group_count"] = inter["min_group_count"]
    metrics["min_group_count_primary_attributes"] = min(per_field_mins) if per_field_mins else 0.0
    return metrics
