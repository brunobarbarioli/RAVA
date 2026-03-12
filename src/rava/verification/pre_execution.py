from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rava.specs.schema import ConstraintType, Specification, Verdict, VerificationAction
from rava.verification.classifiers.rule_based import get_predicate

try:
    from langchain_core.runnables import RunnableLambda
except Exception:  # pragma: no cover - optional dependency guard
    RunnableLambda = None


class _LocalRunnable:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, payload: dict[str, Any]):
        return self._fn(payload)


@dataclass
class PreExecutionResult:
    action: VerificationAction
    verdict_rows: list[dict[str, Any]]
    violated_constraint_ids: list[str]


class PreExecutionVerifier:
    def __init__(self, spec: Specification):
        self.spec = spec

    def verify(self, event: str, text: str, context: dict[str, Any]) -> PreExecutionResult:
        rows: list[dict[str, Any]] = []
        hard_fails: list[str] = []
        soft_fails: list[str] = []

        for c in self.spec.constraints:
            if c.type == ConstraintType.STATISTICAL:
                continue
            if event not in c.trigger.events:
                continue

            predicate = get_predicate(c.predicate)
            verdict, details = predicate(text, context)
            row = {
                "event": event,
                "constraint_id": c.id,
                "constraint_type": c.type.value,
                "verdict": verdict.value,
                "details": details,
                "layer": "pre",
            }
            rows.append(row)
            if c.type == ConstraintType.HARD and verdict == Verdict.FAIL:
                hard_fails.append(c.id)
            if c.type == ConstraintType.SOFT and verdict == Verdict.FAIL:
                soft_fails.append(c.id)

        if hard_fails:
            # Escalate to BLOCK when multiple hard failures are detected.
            if len(hard_fails) > 1:
                action = VerificationAction.BLOCK
            else:
                action = VerificationAction.MODIFY
        elif soft_fails:
            action = VerificationAction.FLAG
        else:
            action = VerificationAction.APPROVE

        return PreExecutionResult(
            action=action,
            verdict_rows=rows,
            violated_constraint_ids=hard_fails + soft_fails,
        )

    def as_runnable(self):
        def _invoke(payload: dict[str, Any]) -> PreExecutionResult:
            return self.verify(
                event=str(payload.get("event", "final_answer")),
                text=str(payload.get("text", "")),
                context=payload.get("context", {}) or {},
            )

        if RunnableLambda is not None:
            return RunnableLambda(_invoke)
        return _LocalRunnable(_invoke)


def build_repair_prompt(original_prompt: str, violated_ids: list[str]) -> str:
    joined = ", ".join(violated_ids)
    return (
        f"{original_prompt}\n\n"
        "Revise the answer to satisfy verification constraints. "
        f"Violated constraints: {joined}. "
        "Add safety disclaimers, remove unsafe content, and provide citations when needed."
    )
