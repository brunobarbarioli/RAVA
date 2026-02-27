from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rava.tools.base import Tool, ToolResult


class PubMedRetriever(Tool):
    name = "pubmed_retriever"

    def __init__(self, cache_path: str | Path = "data/raw/pubmed_corpus.jsonl"):
        self.cache_path = Path(cache_path)
        self.corpus = self._load_corpus()

    def _load_corpus(self) -> list[dict[str, Any]]:
        if not self.cache_path.exists():
            return []
        rows = []
        with self.cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return rows

    def _score(self, query: str, text: str) -> float:
        q_tokens = re.findall(r"\w+", query.lower())
        t_tokens = re.findall(r"\w+", text.lower())
        if not q_tokens or not t_tokens:
            return 0.0
        q_counts = Counter(q_tokens)
        t_counts = Counter(t_tokens)
        overlap = sum(min(q_counts[token], t_counts[token]) for token in q_counts)
        return overlap / max(len(q_tokens), 1)

    def run(self, query: str, context: dict[str, Any] | None = None) -> ToolResult:
        if not self.corpus:
            return ToolResult(
                ok=False,
                observation=(
                    "PubMed cache not configured. Add data/raw/pubmed_corpus.jsonl "
                    "or plug in a real PubMed API retriever."
                ),
                metadata={"tool": self.name},
            )
        scored = []
        for row in self.corpus:
            body = f"{row.get('title', '')} {row.get('abstract', '')}"
            scored.append((self._score(query, body), row))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [r for s, r in scored[:3] if s > 0]
        if not top:
            return ToolResult(ok=True, observation="No relevant PubMed records found.", metadata={"tool": self.name})
        text = "\n".join(
            [
                f"PMID={r.get('pmid', 'NA')} TITLE={r.get('title', '')[:120]}"
                for r in top
            ]
        )
        return ToolResult(ok=True, observation=text, metadata={"tool": self.name, "hits": len(top)})
