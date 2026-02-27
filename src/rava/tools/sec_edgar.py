from __future__ import annotations

from pathlib import Path
from typing import Any

from rava.tools.base import Tool, ToolResult


class SECEdgarRetriever(Tool):
    name = "sec_edgar_retriever"

    def __init__(self, filings_dir: str | Path = "data/raw/sec_filings"):
        self.filings_dir = Path(filings_dir)

    def run(self, query: str, context: dict[str, Any] | None = None) -> ToolResult:
        if not self.filings_dir.exists():
            return ToolResult(
                ok=False,
                observation=(
                    "SEC filings cache not configured. Add local filings under "
                    "data/raw/sec_filings or use a custom EDGAR fetcher."
                ),
                metadata={"tool": self.name},
            )

        hits: list[str] = []
        for path in self.filings_dir.glob("*.txt"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if query.lower() in text.lower():
                snippet = text[:240].replace("\n", " ")
                hits.append(f"{path.name}: {snippet}")
            if len(hits) >= 3:
                break

        if not hits:
            return ToolResult(ok=True, observation="No matching SEC filing passages.", metadata={"tool": self.name})

        return ToolResult(ok=True, observation="\n".join(hits), metadata={"tool": self.name, "hits": len(hits)})
