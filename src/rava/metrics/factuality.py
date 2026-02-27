from __future__ import annotations

import re
from typing import Any


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CIT_RE = re.compile(r"(\[[^\]]+\]|https?://\S+|doi:\S+)", re.IGNORECASE)


def extract_atomic_claims(text: str) -> list[str]:
    if not text.strip():
        return []
    claims = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return [c for c in claims if len(c.split()) >= 4]


def claim_has_citation(claim: str) -> bool:
    return bool(CIT_RE.search(claim))


def verify_claim_stub(claim: str, retrieval_context: str | None = None) -> bool:
    if claim_has_citation(claim):
        return True
    if retrieval_context:
        claim_words = {w.lower() for w in re.findall(r"\w+", claim) if len(w) > 3}
        ctx_words = {w.lower() for w in re.findall(r"\w+", retrieval_context)}
        overlap = len(claim_words & ctx_words)
        return overlap >= min(3, max(1, len(claim_words) // 3))
    return False


def extract_claim_records(text: str, retrieval_context: str = "") -> list[dict[str, Any]]:
    claims = extract_atomic_claims(text)
    rows: list[dict[str, Any]] = []
    for claim in claims:
        rows.append(
            {
                "claim": claim,
                "has_citation": claim_has_citation(claim),
                "verified": verify_claim_stub(claim, retrieval_context),
            }
        )
    return rows


def compute_claim_precision(records: list[dict[str, Any]]) -> dict[str, float]:
    total_claims = 0
    verified = 0
    cited = 0
    for row in records:
        output = str(row.get("output", ""))
        retrieval_ctx = str(row.get("retrieval_context", ""))
        claim_rows = extract_claim_records(output, retrieval_ctx)
        total_claims += len(claim_rows)
        verified += sum(1 for c in claim_rows if bool(c["verified"]))
        cited += sum(1 for c in claim_rows if bool(c["has_citation"]))
    if total_claims == 0:
        return {
            "claim_precision": 0.0,
            "claim_count": 0.0,
            "verified_claims": 0.0,
            "claim_citation_coverage": 0.0,
        }
    return {
        "claim_precision": verified / total_claims,
        "claim_count": float(total_claims),
        "verified_claims": float(verified),
        "claim_citation_coverage": cited / total_claims,
    }
