#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rava.experiments.plots import make_result_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate result plots from final experiment tables")
    parser.add_argument("--tables-dir", default="outputs/tables/final_a6_dual_model_full_v2")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--prefix", default="final_a6")
    args = parser.parse_args()

    for path in make_result_plots(
        tables_dir=args.tables_dir,
        output_dir=args.output_dir,
        prefix=args.prefix,
    ):
        print(path)


if __name__ == "__main__":
    main()
