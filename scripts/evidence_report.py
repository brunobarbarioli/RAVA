#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from rava.experiments.evidence import summarize_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize empirical evidence quality for a run root")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", default="outputs/evidence_report.json")
    parser.add_argument(
        "--comparison-track",
        default="audited",
        choices=["audited", "operational"],
        help="Primary evidence track.",
    )
    args = parser.parse_args()

    payload = summarize_evidence(
        runs_root=args.runs_root,
        output_path=args.output,
        comparison_track=args.comparison_track,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
