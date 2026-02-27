from __future__ import annotations

from pathlib import Path

import pandas as pd

from rava.tools.base import Tool, ToolResult


class PriceLookupTool(Tool):
    name = "price_lookup"

    def __init__(self, prices_csv: str | Path = "data/raw/prices/prices.csv"):
        self.prices_csv = Path(prices_csv)

    def run(self, query: str, context: dict | None = None) -> ToolResult:
        if not self.prices_csv.exists():
            return ToolResult(
                ok=False,
                observation=(
                    "Price cache not configured. Place CSV at data/raw/prices/prices.csv "
                    "with columns: symbol,date,close"
                ),
                metadata={"tool": self.name},
            )

        try:
            df = pd.read_csv(self.prices_csv)
        except Exception as exc:
            return ToolResult(ok=False, observation=f"Failed reading prices CSV: {exc}", metadata={"tool": self.name})

        symbol = query.strip().upper().split()[0] if query.strip() else ""
        if not symbol:
            return ToolResult(ok=False, observation="Provide a symbol in query.", metadata={"tool": self.name})

        subset = df[df["symbol"].astype(str).str.upper() == symbol].sort_values("date")
        if subset.empty:
            return ToolResult(ok=True, observation=f"No cached prices for {symbol}", metadata={"tool": self.name})

        last = subset.iloc[-1]
        return ToolResult(
            ok=True,
            observation=f"{symbol} latest close {last['close']} on {last['date']}",
            metadata={"tool": self.name, "rows": int(len(subset))},
        )
