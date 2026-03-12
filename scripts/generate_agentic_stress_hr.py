#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rava.experiments.agentic_stress import generate_agentic_stress_hr_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic agentic HR stress dataset")
    parser.add_argument("--n", type=int, default=600, help="Number of examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-path",
        default="data/synthetic/agentic_stress_hr.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()
    out = generate_agentic_stress_hr_dataset(output_path=args.output_path, n=args.n, seed=args.seed)
    print(out)


if __name__ == "__main__":
    main()

