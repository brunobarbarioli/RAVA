#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or append cached prices CSV for price lookup tool")
    parser.add_argument("--output", default="data/raw/prices/prices.csv")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--date", default="2025-01-02")
    parser.add_argument("--close", type=float, default=185.64)
    args = parser.parse_args()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([{"symbol": args.symbol.upper(), "date": args.date, "close": args.close}])
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, row], ignore_index=True)
    else:
        df = row
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
