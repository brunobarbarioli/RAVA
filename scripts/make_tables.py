#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rava.experiments.tables import make_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables from run metrics")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument(
        "--certified-only",
        action="store_true",
        default=True,
        help="Use only certified full-stage runs for capability tables (default).",
    )
    parser.add_argument(
        "--include-noncertified",
        action="store_true",
        help="Include non-certified rows in capability tables.",
    )
    parser.add_argument(
        "--comparison-track",
        default="audited",
        choices=["audited", "operational", "both"],
        help="Primary comparison track for capability/significance tables.",
    )
    args = parser.parse_args()

    certified_only = bool(args.certified_only) and not bool(args.include_noncertified)
    paths = make_tables(
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        certified_only=certified_only,
        comparison_track=args.comparison_track,
    )
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
