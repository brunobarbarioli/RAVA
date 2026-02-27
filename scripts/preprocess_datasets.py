#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rava.experiments.datasets import preprocess_datasets
from rava.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess RAVA datasets into common JSONL format")
    parser.add_argument("--domains", default="healthcare,finance,hr", help="Comma-separated domain list")
    parser.add_argument("--profile", default="core", help="Dataset profile: core or enhanced")
    args = parser.parse_args()

    setup_logging()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    preprocess_datasets(domains=domains, profile=args.profile)


if __name__ == "__main__":
    main()
