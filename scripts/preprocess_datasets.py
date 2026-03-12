#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rava.experiments.datasets import preprocess_datasets
from rava.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess RAVA datasets into common JSONL format")
    parser.add_argument("--domains", default="healthcare,finance,hr", help="Comma-separated domain list")
    parser.add_argument("--profile", default="core", help="Dataset profile: core or enhanced")
    parser.add_argument("--split-strategy", default="preserve", help="Split strategy: preserve, random, temporal")
    parser.add_argument("--disallow-toy-fallback", action="store_true", help="Fail if toy fallback is required.")
    args = parser.parse_args()

    setup_logging()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    preprocess_datasets(
        domains=domains,
        profile=args.profile,
        split_strategy=args.split_strategy,
        disallow_toy_fallback=args.disallow_toy_fallback,
    )


if __name__ == "__main__":
    main()
