from __future__ import annotations

from collections import defaultdict
from typing import Any


def _group_by_action(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(rows):
        rid = str(row.get("record_id", "record"))
        event = str(row.get("event", "event"))
        action_key = str(row.get("action_key", f"{rid}:{event}:{i}"))
        grouped[action_key].append(row)
    return grouped


def _group_by_record(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rid = str(row.get("record_id", "record"))
        grouped[rid].append(row)
    return grouped


def _hard_soft_fail(rows: list[dict[str, Any]]) -> tuple[bool, bool]:
    hard_fail = any(
        str(r.get("constraint_type", "")).upper() == "HARD"
        and str(r.get("verdict", "")).upper() == "FAIL"
        for r in rows
    )
    soft_fail = any(
        str(r.get("constraint_type", "")).upper() == "SOFT"
        and str(r.get("verdict", "")).upper() == "FAIL"
        for r in rows
    )
    return hard_fail, soft_fail


def _hard_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    grouped_local = _group_by_action(rows)
    fail_count = 0
    for local_rows in grouped_local.values():
        hard_fail, _ = _hard_soft_fail(local_rows)
        if hard_fail:
            fail_count += 1
    return fail_count / max(len(grouped_local), 1)


def _select_final_record_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if bool(row.get("final_selected", False))]
    if selected:
        return selected
    rounds = [
        int(row.get("audit_iteration"))
        for row in rows
        if row.get("audit_iteration") is not None
    ]
    if rounds:
        last_round = max(rounds)
        selected_round = [row for row in rows if int(row.get("audit_iteration", -1)) == last_round]
        if selected_round:
            return selected_round
    return rows


def compute_violation_rates(
    verdict_rows: list[dict[str, Any]],
    halted_record_ids: set[str] | list[str] | None = None,
    record_ids: set[str] | list[str] | None = None,
) -> dict[str, float]:
    halted_set = {str(x) for x in (halted_record_ids or set()) if str(x)}
    record_set = {str(x) for x in (record_ids or set()) if str(x)}
    if not verdict_rows and not halted_set and not record_set:
        return {
            "hard_violation_rate": 0.0,
            "soft_violation_rate": 0.0,
            "operational_hard_violation_rate": 0.0,
            "operational_soft_violation_rate": 0.0,
            "residual_hard_violation_rate": 0.0,
            "residual_soft_violation_rate": 0.0,
            "prevention_hard_violation_rate": 0.0,
            "detection_hard_violation_rate": 0.0,
            "prevented_hard_incident_rate": 0.0,
            "detected_hard_incident_rate": 0.0,
            "unmitigated_halt_rate": 0.0,
        }

    grouped = _group_by_action(verdict_rows)
    hard_violations = 0
    soft_violations = 0
    total_actions = len(grouped)

    for rows in grouped.values():
        hard_fail, soft_fail = _hard_soft_fail(rows)
        hard_violations += int(hard_fail)
        soft_violations += int(soft_fail)

    prevention_rows = [r for r in verdict_rows if str(r.get("layer", "")).lower() in {"pre", "runtime"}]
    detection_rows = [
        r
        for r in verdict_rows
        if str(r.get("layer", "")).lower() in {"posthoc", "audited_posthoc"}
    ]

    posthoc_by_record = _group_by_record(detection_rows)
    record_universe = set(record_set)
    if not record_universe:
        record_universe = set(posthoc_by_record.keys())
        if not record_universe:
            record_universe = {str(r.get("record_id", "record")) for r in verdict_rows}
    record_universe.update(halted_set)

    residual_hard = 0
    residual_soft = 0
    unmitigated_halt = 0
    for rid in sorted(record_universe):
        final_rows = _select_final_record_rows(posthoc_by_record.get(rid, []))
        hard_fail, soft_fail = _hard_soft_fail(final_rows)
        if rid in halted_set and not hard_fail:
            hard_fail = True
            unmitigated_halt += 1
        residual_hard += int(hard_fail)
        residual_soft += int(soft_fail)

    denom_records = max(len(record_universe), 1)
    operational_hard = hard_violations / max(total_actions, 1)
    operational_soft = soft_violations / max(total_actions, 1)
    prevention_hard = _hard_rate(prevention_rows)
    detection_hard = _hard_rate(detection_rows)

    return {
        "hard_violation_rate": operational_hard,
        "soft_violation_rate": operational_soft,
        "operational_hard_violation_rate": operational_hard,
        "operational_soft_violation_rate": operational_soft,
        "residual_hard_violation_rate": residual_hard / float(denom_records),
        "residual_soft_violation_rate": residual_soft / float(denom_records),
        "prevention_hard_violation_rate": prevention_hard,
        "detection_hard_violation_rate": detection_hard,
        "prevented_hard_incident_rate": prevention_hard,
        "detected_hard_incident_rate": detection_hard,
        "unmitigated_halt_rate": unmitigated_halt / float(denom_records),
    }
