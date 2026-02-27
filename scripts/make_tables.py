#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rava.experiments.tables import make_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables from run metrics")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default="outputs/tables")
    args = parser.parse_args()

    paths = make_tables(runs_root=args.runs_root, output_dir=args.output_dir)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
