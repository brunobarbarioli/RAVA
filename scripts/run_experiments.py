#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rava.experiments.runner import run_sweep
from rava.logging import capture_environment, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAVA experiment sweep")
    parser.add_argument("--sweep-config", default="configs/experiments/smoke.yaml")
    parser.add_argument("--base-config", default="configs/base.yaml")
    args = parser.parse_args()

    setup_logging()
    root = run_sweep(sweep_config_path=args.sweep_config, base_config_path=args.base_config)
    capture_environment(root / "environment.json")
    print(root)


if __name__ == "__main__":
    main()
