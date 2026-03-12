#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rava.experiments.runner import run_sweep
from rava.logging import capture_environment, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAVA experiment sweep")
    parser.add_argument("--sweep-config", default="configs/experiments/smoke.yaml")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--provider-preflight-enabled", default=None)
    parser.add_argument("--provider-preflight-min-success-rate", type=float, default=None)
    parser.add_argument("--provider-preflight-abort-on-fail", default=None)
    parser.add_argument("--max-concurrent-runs", type=int, default=None)
    parser.add_argument("--stage", default="full", choices=["calibration", "canary", "full"])
    parser.add_argument(
        "--max-inflight-per-model",
        default=None,
        help="Per-model inflight caps, e.g. ministral-3-cloud=2,qwen3-next=1",
    )
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--qwen-burst-concurrency", type=int, default=None)
    parser.add_argument("--qwen-degrade-concurrency", type=int, default=None)
    parser.add_argument("--resume-mode", default=None, choices=["missing_only", "fresh"])
    parser.add_argument("--agentic-backend", default=None, choices=["langgraph", "legacy_python"])
    parser.add_argument("--example-parallelism-per-run", type=int, default=None)
    parser.add_argument("--async-model-invocation", action="store_true", default=False)
    parser.add_argument("--sync-model-invocation", action="store_true", default=False)
    args = parser.parse_args()

    setup_logging()
    enabled = None
    abort_on_fail = None
    if args.provider_preflight_enabled is not None:
        enabled = str(args.provider_preflight_enabled).lower() in {"1", "true", "yes", "on"}
    if args.provider_preflight_abort_on_fail is not None:
        abort_on_fail = str(args.provider_preflight_abort_on_fail).lower() in {"1", "true", "yes", "on"}
    root = run_sweep(
        sweep_config_path=args.sweep_config,
        base_config_path=args.base_config,
        provider_preflight_enabled=enabled,
        provider_preflight_min_success_rate=args.provider_preflight_min_success_rate,
        provider_preflight_abort_on_fail=abort_on_fail,
        max_concurrent_runs=args.max_concurrent_runs,
        stage=args.stage,
        max_inflight_per_model=args.max_inflight_per_model,
        time_budget_hours=args.time_budget_hours,
        qwen_burst_concurrency=args.qwen_burst_concurrency,
        qwen_degrade_concurrency=args.qwen_degrade_concurrency,
        resume_mode=args.resume_mode,
        agentic_backend=args.agentic_backend,
        example_parallelism_per_run=args.example_parallelism_per_run,
        async_model_invocation=(False if args.sync_model_invocation else (True if args.async_model_invocation else None)),
    )
    capture_environment(root / "environment.json")
    print(root)


if __name__ == "__main__":
    main()
