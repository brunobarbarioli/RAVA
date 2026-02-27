from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rava.specs.schema import ConstraintType, Specification, Verdict, VerificationAction
from rava.verification.classifiers.rule_based import get_predicate
from rava.verification.judges.mock_judge import HeuristicRiskJudge, Judge, JudgeResult, MockJudge
from rava.verification.state.advice_boundary_tracker import AdviceBoundaryTracker
from rava.verification.state.phi_leak_tracker import PhiLeakTracker


@dataclass
class RuntimeResult:
    action: VerificationAction
    verdict_rows: list[dict[str, Any]]
    hard_fail_ids: list[str]


class RuntimeMonitor:
    def __init__(
        self,
        spec: Specification,
        judge: Judge | None = None,
        secondary_judge: Judge | None = None,
    ):
        self.spec = spec
        self.judge = judge or MockJudge()
        self.secondary_judge = secondary_judge or HeuristicRiskJudge()
        self.phi_tracker = PhiLeakTracker()
        self.advice_tracker = AdviceBoundaryTracker()

    def monitor(
        self,
        action_text: str,
        observation_text: str,
        context: dict[str, Any],
        event: str = "tool_call",
    ) -> RuntimeResult:
        rows: list[dict[str, Any]] = []
        hard_fails: list[str] = []
        soft_fails: list[str] = []
        combined = f"{action_text}\n{observation_text}".strip()
        combined_context = {**context, "event": event}

        for c in self.spec.constraints:
            if c.type == ConstraintType.STATISTICAL:
                continue
            if event not in c.trigger.events:
                continue
            predicate = get_predicate(c.predicate)
            verdict, details = predicate(combined, combined_context)
            row = {
                "event": event,
                "constraint_id": c.id,
                "constraint_type": c.type.value,
                "verdict": verdict.value,
                "details": details,
                "layer": "runtime",
            }
            rows.append(row)
            if c.type == ConstraintType.HARD and verdict == Verdict.FAIL:
                hard_fails.append(c.id)
            if c.type == ConstraintType.SOFT and verdict == Verdict.FAIL:
                soft_fails.append(c.id)

        primary = self.judge.judge(
            action=action_text,
            observation=observation_text,
            input_text=str(context.get("input", "")),
        )
        secondary = self.secondary_judge.judge(
            action=action_text,
            observation=observation_text,
            input_text=str(context.get("input", "")),
        )
        rows.extend(self._judge_rows(event=event, primary=primary, secondary=secondary))

        if primary.verdict == Verdict.FAIL or secondary.verdict == Verdict.FAIL:
            hard_fails.append("RUNTIME-JUDGE")
        elif primary.verdict == Verdict.UNCERTAIN or secondary.verdict == Verdict.UNCERTAIN:
            soft_fails.append("RUNTIME-JUDGE-UNCERTAIN")

        phi_verdict, _ = get_predicate("no_phi_pii_leak")(combined, combined_context)
        boundary_health, _ = get_predicate("advice_boundary_healthcare")(combined, combined_context)
        boundary_finance, _ = get_predicate("advice_boundary_finance")(combined, combined_context)
        self.phi_tracker.record(phi_verdict == Verdict.FAIL)
        self.advice_tracker.record(
            boundary_health == Verdict.FAIL or boundary_finance == Verdict.FAIL
        )
        rows.append(
            {
                "event": event,
                "constraint_id": "RUNTIME-STATE",
                "constraint_type": "SOFT",
                "verdict": Verdict.PASS.value,
                "details": {
                    **self.phi_tracker.snapshot(),
                    **self.advice_tracker.snapshot(),
                },
                "layer": "runtime",
            }
        )

        if hard_fails:
            action = VerificationAction.HALT
        elif soft_fails:
            action = VerificationAction.FLAG
        else:
            action = VerificationAction.APPROVE

        return RuntimeResult(action=action, verdict_rows=rows, hard_fail_ids=hard_fails)

    @staticmethod
    def _judge_rows(event: str, primary: JudgeResult, secondary: JudgeResult) -> list[dict[str, Any]]:
        if primary.verdict == Verdict.FAIL or secondary.verdict == Verdict.FAIL:
            aggregate = Verdict.FAIL
            agg_reason = "Consensus runtime judge blocked action due to at least one FAIL."
        elif primary.verdict == Verdict.PASS and secondary.verdict == Verdict.PASS:
            aggregate = Verdict.PASS
            agg_reason = "Consensus runtime judge approved action."
        else:
            aggregate = Verdict.UNCERTAIN
            agg_reason = "Consensus runtime judge marked action uncertain."

        return [
            {
                "event": event,
                "constraint_id": "RUNTIME-JUDGE-1",
                "constraint_type": "HARD",
                "verdict": primary.verdict.value,
                "details": {"rationale": primary.rationale, "confidence": primary.confidence},
                "layer": "runtime",
            },
            {
                "event": event,
                "constraint_id": "RUNTIME-JUDGE-2",
                "constraint_type": "HARD",
                "verdict": secondary.verdict.value,
                "details": {"rationale": secondary.rationale, "confidence": secondary.confidence},
                "layer": "runtime",
            },
            {
                "event": event,
                "constraint_id": "RUNTIME-JUDGE",
                "constraint_type": "HARD",
                "verdict": aggregate.value,
                "details": {
                    "rationale": agg_reason,
                    "primary_confidence": primary.confidence,
                    "secondary_confidence": secondary.confidence,
                },
                "layer": "runtime",
            },
        ]
