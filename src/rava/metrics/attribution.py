from __future__ import annotations

import re

from rava.metrics.factuality import extract_claim_records


CIT_RE = re.compile(r"(\[[^\]]+\]|https?://\S+|doi:\S+)", re.IGNORECASE)


def source_attribution_score(text: str) -> float:
    claim_rows = extract_claim_records(text)
    if not claim_rows:
        return 0.0
    attributed = sum(1 for row in claim_rows if bool(row.get("has_citation")))
    return attributed / len(claim_rows)


def compute_source_attribution(records: list[dict]) -> dict[str, float]:
    if not records:
        return {
            "source_attribution_score": 0.0,
            "evidence_support_rate": 0.0,
        }

    citation_scores: list[float] = []
    support_scores: list[float] = []
    for row in records:
        output = str(row.get("output", ""))
        retrieval = str(row.get("retrieval_context", ""))
        claim_rows = extract_claim_records(output, retrieval)
        if not claim_rows:
            citation_scores.append(0.0)
            support_scores.append(0.0)
            continue
        citation_scores.append(sum(1 for c in claim_rows if c["has_citation"]) / len(claim_rows))
        support_scores.append(sum(1 for c in claim_rows if c["verified"]) / len(claim_rows))

    return {
        "source_attribution_score": sum(citation_scores) / len(citation_scores),
        "evidence_support_rate": sum(support_scores) / len(support_scores),
    }
