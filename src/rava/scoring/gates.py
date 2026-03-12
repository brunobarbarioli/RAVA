from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rava.specs.schema import ConstraintType, Specification


INVALID_GROUP_VALUES = {"", "na", "n/a", "none", "null", "unknown", "nan"}


@dataclass
class StatsConstraintResult:
    constraint_id: str
    metric: str | None
    threshold: float | None
    grouping_fields: list[str]
    applicable: bool
    status: str
    reason: str
    metric_value: float | None
    min_group_count_observed: float | None
    observed_num_groups: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "metric": self.metric,
            "threshold": self.threshold,
            "grouping_fields": list(self.grouping_fields),
            "applicable": bool(self.applicable),
            "status": self.status,
            "reason": self.reason,
            "metric_value": self.metric_value,
            "min_group_count_observed": self.min_group_count_observed,
            "observed_num_groups": dict(self.observed_num_groups),
        }


def _metric_passes(metric_name: str, metric_value: float, threshold: float) -> bool:
    key = metric_name.lower().strip()
    if "ratio" in key or "precision" in key or "recall" in key or "accuracy" in key:
        return float(metric_value) >= float(threshold)
    return float(metric_value) <= float(threshold)


def _get_metric_value(metrics: dict[str, Any], key: str | None) -> float | None:
    if not key:
        return None
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_valid_group_label(label: Any) -> bool:
    text = str(label).strip().lower()
    return text not in INVALID_GROUP_VALUES


def _constraint_aware_gate(
    metrics: dict[str, Any],
    spec: Specification | None,
    min_group_count: float,
) -> dict[str, Any]:
    if spec is None:
        return {
            "V_stats": False,
            "policy": "constraint_aware",
            "reason_codes": ["spec_missing_for_constraint_aware_gate"],
            "applicable_constraint_ids": [],
            "na_constraint_ids": [],
            "failed_constraint_ids": [],
            "passed_constraint_ids": [],
            "constraint_results": [],
        }

    stat_constraints = [c for c in spec.constraints if c.type == ConstraintType.STATISTICAL]
    if not stat_constraints:
        return {
            "V_stats": True,
            "policy": "constraint_aware",
            "reason_codes": ["no_statistical_constraints"],
            "applicable_constraint_ids": [],
            "na_constraint_ids": [],
            "failed_constraint_ids": [],
            "passed_constraint_ids": [],
            "constraint_results": [],
        }

    results: list[StatsConstraintResult] = []
    for constraint in stat_constraints:
        metric_name = constraint.metric
        threshold = constraint.threshold
        metric_value = _get_metric_value(metrics, metric_name)
        grouping_fields = list(constraint.grouping_fields or [])

        if grouping_fields:
            num_groups: dict[str, float] = {}
            min_counts: list[float] = []
            applicable = True
            for field in grouping_fields:
                num_key = f"num_groups_{field}"
                min_key = f"min_group_count_{field}"
                observed_num_groups = float(metrics.get(num_key, 0.0))
                observed_min_count = float(metrics.get(min_key, 0.0))
                num_groups[field] = observed_num_groups
                min_counts.append(observed_min_count)
                if observed_num_groups < 2.0:
                    applicable = False

            if not applicable:
                results.append(
                    StatsConstraintResult(
                        constraint_id=constraint.id,
                        metric=metric_name,
                        threshold=threshold,
                        grouping_fields=grouping_fields,
                        applicable=False,
                        status="NA",
                        reason="insufficient_group_diversity",
                        metric_value=metric_value,
                        min_group_count_observed=min(min_counts) if min_counts else None,
                        observed_num_groups=num_groups,
                    )
                )
                continue

            min_observed = min(min_counts) if min_counts else 0.0
            if min_observed < float(min_group_count):
                results.append(
                    StatsConstraintResult(
                        constraint_id=constraint.id,
                        metric=metric_name,
                        threshold=threshold,
                        grouping_fields=grouping_fields,
                        applicable=True,
                        status="FAIL",
                        reason=f"subgroup_min_below_{int(min_group_count)}",
                        metric_value=metric_value,
                        min_group_count_observed=min_observed,
                        observed_num_groups=num_groups,
                    )
                )
                continue

            if metric_value is None or threshold is None:
                results.append(
                    StatsConstraintResult(
                        constraint_id=constraint.id,
                        metric=metric_name,
                        threshold=threshold,
                        grouping_fields=grouping_fields,
                        applicable=True,
                        status="FAIL",
                        reason="metric_or_threshold_missing",
                        metric_value=metric_value,
                        min_group_count_observed=min_observed,
                        observed_num_groups=num_groups,
                    )
                )
                continue

            metric_ok = _metric_passes(metric_name or "", metric_value, float(threshold))
            results.append(
                StatsConstraintResult(
                    constraint_id=constraint.id,
                    metric=metric_name,
                    threshold=threshold,
                    grouping_fields=grouping_fields,
                    applicable=True,
                    status="PASS" if metric_ok else "FAIL",
                    reason="metric_threshold_passed" if metric_ok else "metric_threshold_failed",
                    metric_value=metric_value,
                    min_group_count_observed=min_observed,
                    observed_num_groups=num_groups,
                )
            )
            continue

        # Non-grouped statistical constraints
        if metric_value is None or threshold is None:
            results.append(
                StatsConstraintResult(
                    constraint_id=constraint.id,
                    metric=metric_name,
                    threshold=threshold,
                    grouping_fields=[],
                    applicable=True,
                    status="FAIL",
                    reason="metric_or_threshold_missing",
                    metric_value=metric_value,
                    min_group_count_observed=None,
                    observed_num_groups={},
                )
            )
            continue

        metric_ok = _metric_passes(metric_name or "", metric_value, float(threshold))
        results.append(
            StatsConstraintResult(
                constraint_id=constraint.id,
                metric=metric_name,
                threshold=threshold,
                grouping_fields=[],
                applicable=True,
                status="PASS" if metric_ok else "FAIL",
                reason="metric_threshold_passed" if metric_ok else "metric_threshold_failed",
                metric_value=metric_value,
                min_group_count_observed=None,
                observed_num_groups={},
            )
        )

    applicable = [r for r in results if r.applicable]
    failed = [r for r in applicable if r.status == "FAIL"]
    passed = [r for r in applicable if r.status == "PASS"]
    na = [r for r in results if not r.applicable]

    reason_codes: list[str] = []
    if not applicable:
        reason_codes.append("no_applicable_stat_constraints")
    reason_codes.extend(f"{r.constraint_id}:{r.reason}" for r in failed)

    return {
        "V_stats": len(failed) == 0,
        "policy": "constraint_aware",
        "reason_codes": reason_codes,
        "applicable_constraint_ids": [r.constraint_id for r in applicable],
        "na_constraint_ids": [r.constraint_id for r in na],
        "failed_constraint_ids": [r.constraint_id for r in failed],
        "passed_constraint_ids": [r.constraint_id for r in passed],
        "constraint_results": [r.to_dict() for r in results],
    }


def _universal_gate(metrics: dict[str, Any], min_group_count: float) -> dict[str, Any]:
    subgroup_min_primary = float(
        metrics.get("min_group_count_primary_attributes", metrics.get("min_group_count", 0.0))
    )
    passed = subgroup_min_primary >= float(min_group_count)
    reason_codes: list[str] = []
    if not passed:
        reason_codes.append(f"subgroup_min_below_{int(min_group_count)}")
    return {
        "V_stats": passed,
        "policy": "universal",
        "reason_codes": reason_codes,
        "applicable_constraint_ids": [],
        "na_constraint_ids": [],
        "failed_constraint_ids": [] if passed else ["universal_subgroup_floor"],
        "passed_constraint_ids": ["universal_subgroup_floor"] if passed else [],
        "constraint_results": [
            {
                "constraint_id": "universal_subgroup_floor",
                "metric": "min_group_count_primary_attributes",
                "threshold": float(min_group_count),
                "grouping_fields": ["gender", "race_ethnicity", "age_group", "disability"],
                "applicable": True,
                "status": "PASS" if passed else "FAIL",
                "reason": "metric_threshold_passed" if passed else "metric_threshold_failed",
                "metric_value": subgroup_min_primary,
                "min_group_count_observed": subgroup_min_primary,
                "observed_num_groups": {
                    "gender": float(metrics.get("num_groups_gender", 0.0)),
                    "race_ethnicity": float(metrics.get("num_groups_race_ethnicity", 0.0)),
                    "age_group": float(metrics.get("num_groups_age_group", 0.0)),
                    "disability": float(metrics.get("num_groups_disability", 0.0)),
                },
            }
        ],
    }


def compute_stats_gate(
    metrics: dict[str, Any],
    spec: Specification | None,
    policy: str = "constraint_aware",
    min_group_count: float = 30.0,
) -> dict[str, Any]:
    selected = str(policy or "constraint_aware").strip().lower()
    if selected == "universal":
        return _universal_gate(metrics=metrics, min_group_count=min_group_count)
    return _constraint_aware_gate(metrics=metrics, spec=spec, min_group_count=min_group_count)

