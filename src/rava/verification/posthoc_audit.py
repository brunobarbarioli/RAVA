from __future__ import annotations

from typing import Any

from rava.metrics.factuality import extract_atomic_claims, extract_claim_records
from rava.specs.compose import evaluate_expression
from rava.specs.schema import ConstraintType, Specification, Verdict
from rava.verification.classifiers.rule_based import get_predicate


class PostHocAuditor:
    def __init__(self, spec: Specification):
        self.spec = spec

    def audit(
        self,
        final_text: str,
        context: dict[str, Any],
        aggregate_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        aggregate_metrics = aggregate_metrics or {}
        rows: list[dict[str, Any]] = []

        for c in self.spec.constraints:
            if c.type in {ConstraintType.HARD, ConstraintType.SOFT}:
                if "final_answer" not in c.trigger.events:
                    continue
                predicate = get_predicate(c.predicate)
                verdict, details = predicate(final_text, context)
            else:
                predicate = get_predicate(c.predicate)
                metric_value = aggregate_metrics.get(c.metric) if c.metric else None
                verdict, details = predicate(
                    final_text,
                    {
                        **context,
                        "metric_name": c.metric,
                        "metric_value": metric_value,
                        "threshold": c.threshold,
                    },
                )

            rows.append(
                {
                    "event": "posthoc",
                    "constraint_id": c.id,
                    "constraint_type": c.type.value,
                    "verdict": verdict.value,
                    "details": details,
                    "layer": "posthoc",
                }
            )

        claims = extract_atomic_claims(final_text)
        claim_rows = []
        claim_graph_nodes: list[dict[str, Any]] = []
        claim_graph_edges: list[dict[str, Any]] = []
        retrieval_context = str(context.get("retrieval_context", ""))
        extracted = extract_claim_records(final_text, retrieval_context)
        for idx, row in enumerate(extracted):
            claim = str(row["claim"])
            verified = bool(row["verified"])
            claim_rows.append({"claim": claim, "verified": verified, "has_citation": bool(row["has_citation"])})
            claim_graph_nodes.append(
                {
                    "id": f"claim-{idx}",
                    "type": "claim",
                    "text": claim,
                    "verified": verified,
                    "has_citation": bool(row["has_citation"]),
                }
            )
            claim_graph_edges.append(
                {
                    "source": f"claim-{idx}",
                    "target": "retrieval-context",
                    "relation": "supported_by" if verified else "unsupported_by_context",
                }
            )

        if retrieval_context:
            claim_graph_nodes.append({"id": "retrieval-context", "type": "evidence", "text": retrieval_context[:500]})
        elif claims:
            # Keep graph schema stable even when no retrieval context is present.
            claim_graph_nodes.append({"id": "retrieval-context", "type": "evidence", "text": ""})

        hard_fail = sum(
            1
            for r in rows
            if str(r.get("constraint_type")) == "HARD" and str(r.get("verdict")) == "FAIL"
        )
        soft_fail = sum(
            1
            for r in rows
            if str(r.get("constraint_type")) == "SOFT" and str(r.get("verdict")) == "FAIL"
        )

        overall_verdict = None
        if self.spec.composition:
            atom_map = {r["constraint_id"]: Verdict(str(r["verdict"])) for r in rows if r.get("constraint_id")}
            overall_verdict = evaluate_expression(self.spec.composition, atom_map).value

        return {
            "domain": self.spec.domain,
            "constraint_verdicts": rows,
            "claims": claim_rows,
            "claim_graph": {
                "nodes": claim_graph_nodes,
                "edges": claim_graph_edges,
            },
            "summary": {
                "hard_failures": hard_fail,
                "soft_failures": soft_fail,
                "total_constraints_checked": len(rows),
                "overall_spec_verdict": overall_verdict,
            },
        }
