from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
CITATION_RE = re.compile(r"(\[[^\]]+\]|https?://\S+|doi:\S+)", re.IGNORECASE)


class StructuredAgentOutput(BaseModel):
    answer: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list)
    abstain: bool = False
    abstain_reason: str | None = None


def structured_output_instructions() -> str:
    return (
        "Return ONLY JSON with keys: answer (string), confidence (0..1), citations (list of strings), "
        "abstain (bool), abstain_reason (string|null). Use conservative confidence: "
        "0.2-0.5 when evidence is limited, >0.7 only with strong cited support. "
        "If safety or evidence is insufficient, set abstain=true."
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text.strip():
        return None

    match = JSON_BLOCK_RE.search(text)
    candidate = match.group(1) if match else text.strip()

    if not match:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def parse_structured_output(raw_text: str) -> StructuredAgentOutput:
    parsed = _extract_json_object(raw_text)
    if parsed is not None:
        try:
            return StructuredAgentOutput.model_validate(parsed)
        except Exception:
            pass

    inferred_citations = [m.group(0) for m in CITATION_RE.finditer(raw_text)]
    return StructuredAgentOutput(
        answer=raw_text.strip(),
        confidence=None,
        citations=inferred_citations,
        abstain=False,
    )


def render_final_text(output: StructuredAgentOutput) -> str:
    if output.abstain:
        reason = output.abstain_reason or "Insufficient evidence for a safe answer."
        return (
            "I cannot provide a definitive answer safely with the available evidence. "
            f"Reason: {reason}"
        )

    text = output.answer.strip()
    if output.citations:
        citation_line = "Sources: " + ", ".join(output.citations)
        if citation_line not in text:
            text = (text + "\n" + citation_line).strip()
    if output.confidence is not None:
        text = f"{text}\nConfidence: {output.confidence:.2f}"
    return text
