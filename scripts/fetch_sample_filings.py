#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sample SEC filing snippets for offline testing")
    parser.add_argument("--output-dir", default="data/raw/sec_filings")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "sample_10k.txt": "Item 1A Risk Factors: Revenue may decline due to macroeconomic volatility.",
        "sample_10q.txt": "Management Discussion: We observed increased operating margin and cash flow.",
    }
    for name, content in samples.items():
        (out_dir / name).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
