from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rava.specs.parser import load_spec
from rava.utils.serialization import read_json, read_jsonl, write_json
from rava.verification.pre_execution import PreExecutionVerifier
from rava.verification.runtime_monitor import RuntimeMonitor


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / float(total)
    denom = 1.0 + (z * z / total)
    center = (p + (z * z) / (2.0 * total)) / denom
    spread = (
        z
        * math.sqrt((p * (1.0 - p) / total) + ((z * z) / (4.0 * total * total)))
        / denom
    )
    return (max(0.0, center - spread), min(1.0, center + spread))


def _domain_challenge_path(challenge_root: str | Path, domain: str) -> Path:
    return Path(challenge_root) / f"{domain}.jsonl"


def _evaluate_domain_cases(domain: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    spec = load_spec(Path("specs") / f"{domain}.yaml")
    pre = PreExecutionVerifier(spec)
    runtime = RuntimeMonitor(spec)

    rows: list[dict[str, Any]] = []
    pre_hits = 0
    runtime_hits = 0
    joint_misses = 0

    for case in cases:
        expected_ids = {str(x) for x in case.get("expected_hard_ids", []) if str(x).strip()}
        if not expected_ids:
            continue
        event = str(case.get("event", "final_answer"))
        context = case.get("context", {}) or {}
        text = str(case.get("text", ""))
        action_text = str(case.get("action_text", text))
        observation_text = str(case.get("observation_text", text))

        pre_res = pre.verify(event=event, text=text, context=context)
        runtime_res = runtime.monitor(
            action_text=action_text,
            observation_text=observation_text,
            context=context,
            event=event,
        )

        pre_detected_ids = sorted(expected_ids.intersection(set(pre_res.violated_constraint_ids)))
        runtime_fail_ids = {
            str(row.get("constraint_id"))
            for row in runtime_res.verdict_rows
            if str(row.get("constraint_type")) == "HARD" and str(row.get("verdict")) == "FAIL"
        }
        runtime_detected_ids = sorted(expected_ids.intersection(runtime_fail_ids))

        pre_detected = bool(pre_detected_ids)
        runtime_detected = bool(runtime_detected_ids)
        joint_miss = not pre_detected and not runtime_detected
        pre_hits += int(pre_detected)
        runtime_hits += int(runtime_detected)
        joint_misses += int(joint_miss)

        rows.append(
            {
                "id": str(case.get("id", "")),
                "event": event,
                "expected_hard_ids": sorted(expected_ids),
                "pre_detected": pre_detected,
                "runtime_detected": runtime_detected,
                "joint_miss": joint_miss,
                "pre_detected_ids": pre_detected_ids,
                "runtime_detected_ids": runtime_detected_ids,
            }
        )

    total = len(rows)
    pre_ci = _wilson_interval(pre_hits, total)
    runtime_ci = _wilson_interval(runtime_hits, total)
    miss_ci = _wilson_interval(joint_misses, total)
    return {
        "domain": domain,
        "n_cases": total,
        "r_pre": pre_hits / float(total) if total else 0.0,
        "r_pre_ci_low": pre_ci[0],
        "r_pre_ci_high": pre_ci[1],
        "r_rt": runtime_hits / float(total) if total else 0.0,
        "r_rt_ci_low": runtime_ci[0],
        "r_rt_ci_high": runtime_ci[1],
        "q_hat": joint_misses / float(total) if total else 0.0,
        "q_hat_ci_low": miss_ci[0],
        "q_hat_ci_high": miss_ci[1],
        "rows": rows,
    }


def evaluate_compliance_challenges(
    challenge_root: str | Path = "configs/compliance_challenges",
    domains: list[str] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    selected_domains = domains or ["healthcare", "finance", "hr"]
    by_domain: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []

    for domain in selected_domains:
        path = _domain_challenge_path(challenge_root, domain)
        if not path.exists():
            continue
        payload = _evaluate_domain_cases(domain=domain, cases=read_jsonl(path))
        by_domain[domain] = payload
        all_rows.extend(payload.get("rows", []))

    total = len(all_rows)
    pre_hits = sum(1 for row in all_rows if bool(row.get("pre_detected")))
    runtime_hits = sum(1 for row in all_rows if bool(row.get("runtime_detected")))
    joint_misses = sum(1 for row in all_rows if bool(row.get("joint_miss")))
    pre_ci = _wilson_interval(pre_hits, total)
    runtime_ci = _wilson_interval(runtime_hits, total)
    miss_ci = _wilson_interval(joint_misses, total)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "challenge_root": str(challenge_root),
        "domains": by_domain,
        "overall": {
            "n_cases": total,
            "r_pre": pre_hits / float(total) if total else 0.0,
            "r_pre_ci_low": pre_ci[0],
            "r_pre_ci_high": pre_ci[1],
            "r_rt": runtime_hits / float(total) if total else 0.0,
            "r_rt_ci_low": runtime_ci[0],
            "r_rt_ci_high": runtime_ci[1],
            "q_hat": joint_misses / float(total) if total else 0.0,
            "q_hat_ci_low": miss_ci[0],
            "q_hat_ci_high": miss_ci[1],
        },
    }
    if output_path is not None:
        write_json(output_path, summary)
    return summary


def load_compliance_summary(path: str | Path) -> dict[str, Any]:
    return read_json(path)
