from __future__ import annotations

from typing import Any

from rava.specs.schema import Verdict


class HFClassifierHook:
    """Optional extension point for Hugging Face classifier-backed predicates."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or "distilbert-base-uncased"

    def predict(self, text: str, labels: list[str]) -> tuple[Verdict, dict[str, Any]]:
        # Placeholder: wire transformers pipeline here if desired.
        return Verdict.UNCERTAIN, {
            "message": "HF classifier hook not active in default setup.",
            "model": self.model_name,
            "labels": labels,
        }
