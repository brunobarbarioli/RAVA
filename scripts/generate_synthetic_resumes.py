#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rava.experiments.datasets import generate_synthetic_resumes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic resume dataset")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/synthetic/synthetic_resumes.jsonl")
    parser.add_argument("--augment-with-llm", action="store_true")
    args = parser.parse_args()

    generate_synthetic_resumes(
        output_path=args.output,
        n=args.n,
        seed=args.seed,
        augment_with_llm=args.augment_with_llm,
    )


if __name__ == "__main__":
    main()
