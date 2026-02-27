#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rava.experiments.datasets import download_datasets, export_dataset_card
from rava.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RAVA datasets")
    parser.add_argument("--domains", default="healthcare,finance,hr", help="Comma-separated domain list")
    parser.add_argument("--force", action="store_true", help="Force redownload")
    parser.add_argument("--profile", default="core", help="Dataset profile: core or enhanced")
    args = parser.parse_args()

    setup_logging()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    download_datasets(domains=domains, force=args.force, profile=args.profile)
    export_dataset_card()


if __name__ == "__main__":
    main()
