#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from rava.experiments.runner import evaluate_run_dir
from rava.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a completed run directory")
    parser.add_argument("run_dir")
    parser.add_argument("--domain", default=None)
    args = parser.parse_args()

    setup_logging()
    summary = evaluate_run_dir(args.run_dir, domain=args.domain)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
