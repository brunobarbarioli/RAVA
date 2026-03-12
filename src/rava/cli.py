from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from rava.agent.providers import build_provider, provider_healthcheck
from rava.agent.react_agent import run_agent_example
from rava.experiments.baselines import get_verification_config
from rava.experiments.compliance_eval import evaluate_compliance_challenges
from rava.experiments.datasets import (
    DOMAIN_TO_DATASETS,
    download_datasets,
    export_dataset_card,
    generate_synthetic_resumes,
    preprocess_datasets,
)
from rava.experiments.evidence import summarize_evidence
from rava.experiments.paper_artifacts import make_paper_artifacts
from rava.experiments.plots import make_result_plots
from rava.experiments.runner import evaluate_run_dir, run_sweep, summarize_sweep_quality
from rava.experiments.tables import make_tables
from rava.logging import capture_environment, setup_logging
from rava.specs.parser import load_spec
from rava.utils.serialization import read_json, read_yaml, write_json, write_jsonl
from rava.utils.env import load_environment

app = typer.Typer(help="RAVA CLI")
load_environment()


def _normalize_option_default(value):
    return None if value.__class__.__name__ == "OptionInfo" else value


def _canary_summary_passes(summary: dict[str, object], thresholds: dict[str, object]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if float(summary.get("valid_run_rate", 0.0)) < float(thresholds.get("valid_run_rate", 0.95)):
        failures.append("valid_run_rate")
    if float(summary.get("api_failure_rate", 1.0)) > float(thresholds.get("api_failure_rate", 0.02)):
        failures.append("api_failure_rate")
    if float(summary.get("fallback_rate", 1.0)) > float(thresholds.get("fallback_rate", 0.05)):
        failures.append("fallback_rate")
    return (len(failures) == 0), failures


def _load_existing_canary_summary(resume_root: str | None) -> dict[str, object] | None:
    if not resume_root:
        return None
    path = Path(resume_root) / "canary_quality_summary.json"
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


@app.command("download_datasets")
def cmd_download_datasets(
    domains: str = typer.Option("healthcare,finance,hr", help="Comma-separated domains"),
    force: bool = typer.Option(False, help="Force redownload"),
    profile: str = typer.Option(
        "core",
        help="Dataset profile: core, enhanced, paper_hybrid, paper6_fast, paper3_mini, primary_certification, final_a6, diagnostic_secondary, supplemental_stress",
    ),
):
    selected = [d.strip() for d in domains.split(",") if d.strip()]
    setup_logging()
    download_datasets(domains=selected, force=force, profile=profile)
    export_dataset_card()
    print("Dataset download phase complete.")


@app.command("preprocess_datasets")
def cmd_preprocess_datasets(
    domains: str = typer.Option("healthcare,finance,hr", help="Comma-separated domains"),
    profile: str = typer.Option(
        "core",
        help="Dataset profile: core, enhanced, paper_hybrid, paper6_fast, paper3_mini, primary_certification, final_a6, diagnostic_secondary, supplemental_stress",
    ),
    split_strategy: str = typer.Option("preserve", help="Split strategy: preserve, random, temporal"),
    disallow_toy_fallback: bool = typer.Option(False, help="Fail if toy fallback would be used."),
):
    selected = [d.strip() for d in domains.split(",") if d.strip()]
    setup_logging()
    outputs = preprocess_datasets(
        domains=selected,
        profile=profile,
        split_strategy=split_strategy,
        disallow_toy_fallback=disallow_toy_fallback,
    )
    print(f"Preprocessed {len(outputs)} dataset files.")


@app.command("run_agent")
def cmd_run_agent(
    domain: str = typer.Option("healthcare", help="Domain name"),
    text: str = typer.Argument(..., help="Single input text"),
    model_config: str = typer.Option("configs/models/mock.yaml", help="Model config yaml"),
    verification_config: str = typer.Option("full", help="none/pre/runtime/posthoc/full"),
    agentic_backend: str = typer.Option("langgraph", help="Agent backend: langgraph|legacy_python"),
    output_path: str = typer.Option("outputs/single_agent_run.json", help="Where to write result JSON"),
):
    setup_logging()
    model_cfg = read_yaml(model_config)
    provider = build_provider(model_cfg)
    spec = load_spec(Path("specs") / f"{domain}.yaml")
    vcfg = get_verification_config(verification_config)

    example = {
        "id": "single-example",
        "domain": domain,
        "task": "single_query",
        "input": text,
        "reference": None,
        "metadata": {},
        "split": "inference",
    }

    result = run_agent_example(
        example=example,
        provider=provider,
        spec=spec,
        verification_cfg=vcfg,
        run_id="single-run",
        agentic_backend=agentic_backend,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, result)
    print(json.dumps(result["prediction"], indent=2))


@app.command("run_experiment")
def cmd_run_experiment(
    sweep_config: str = typer.Option("configs/experiments/smoke.yaml", help="Sweep config file"),
    base_config: str = typer.Option("configs/base.yaml", help="Base config file"),
    stage: str = typer.Option("full", help="Execution stage: calibration|canary|full"),
    provider_preflight_enabled: Optional[bool] = typer.Option(
        None, help="Override preflight enable/disable from sweep config."
    ),
    provider_preflight_min_success_rate: Optional[float] = typer.Option(
        None, help="Override preflight min success rate."
    ),
    provider_preflight_abort_on_fail: Optional[bool] = typer.Option(
        None, help="Override whether to abort sweep on preflight failure."
    ),
    max_concurrent_runs: Optional[int] = typer.Option(
        None,
        min=1,
        help="Override max number of concurrent run tasks.",
    ),
    max_inflight_per_model: Optional[str] = typer.Option(
        None,
        help="Override per-model inflight caps, e.g. ministral-3-cloud=2,qwen3-next=1",
    ),
    time_budget_hours: Optional[float] = typer.Option(
        None,
        min=0.5,
        help="Override runtime budget (hours).",
    ),
    qwen_burst_concurrency: Optional[int] = typer.Option(
        None,
        min=1,
        help="Override qwen burst inflight cap.",
    ),
    qwen_degrade_concurrency: Optional[int] = typer.Option(
        None,
        min=1,
        help="Override qwen degraded inflight cap.",
    ),
    resume_mode: Optional[str] = typer.Option(
        None,
        help="Resume strategy: missing_only or fresh.",
    ),
    agentic_backend: Optional[str] = typer.Option(
        None,
        help="Agent backend override: langgraph|legacy_python",
    ),
    example_parallelism_per_run: Optional[int] = typer.Option(
        None,
        min=1,
        help="Parallelism per run over examples.",
    ),
    async_model_invocation: Optional[bool] = typer.Option(
        None,
        "--async-model-invocation/--sync-model-invocation",
        help="Override async model invocation mode.",
    ),
):
    setup_logging()
    provider_preflight_enabled = _normalize_option_default(provider_preflight_enabled)
    provider_preflight_min_success_rate = _normalize_option_default(provider_preflight_min_success_rate)
    provider_preflight_abort_on_fail = _normalize_option_default(provider_preflight_abort_on_fail)
    max_concurrent_runs = _normalize_option_default(max_concurrent_runs)
    max_inflight_per_model = _normalize_option_default(max_inflight_per_model)
    time_budget_hours = _normalize_option_default(time_budget_hours)
    qwen_burst_concurrency = _normalize_option_default(qwen_burst_concurrency)
    qwen_degrade_concurrency = _normalize_option_default(qwen_degrade_concurrency)
    resume_mode = _normalize_option_default(resume_mode)
    agentic_backend = _normalize_option_default(agentic_backend)
    example_parallelism_per_run = _normalize_option_default(example_parallelism_per_run)
    async_model_invocation = _normalize_option_default(async_model_invocation)
    stage = stage.strip().lower()
    if stage not in {"calibration", "full", "canary"}:
        raise typer.BadParameter("stage must be 'calibration', 'canary' or 'full'")
    if resume_mode is not None and resume_mode.strip().lower() not in {"missing_only", "fresh"}:
        raise typer.BadParameter("resume-mode must be 'missing_only' or 'fresh'")
    if agentic_backend is not None and agentic_backend.strip().lower() not in {"langgraph", "legacy_python"}:
        raise typer.BadParameter("agentic-backend must be 'langgraph' or 'legacy_python'")

    sweep_cfg = read_yaml(sweep_config)
    cert_policy = str(sweep_cfg.get("certification_stage_policy", "full_only"))
    stats_policy = str(sweep_cfg.get("stats_gate_policy", "constraint_aware"))
    stats_floor = float(sweep_cfg.get("stats_min_group_count", 30.0))
    print(
        json.dumps(
            {
                "stage": stage,
                "certification_stage_policy": cert_policy,
                "stats_gate_policy": stats_policy,
                "stats_min_group_count": stats_floor,
                "resume_mode": resume_mode or str(sweep_cfg.get("resume_mode", "missing_only")),
                "agentic_backend": agentic_backend or str(sweep_cfg.get("agentic_backend", "langgraph")),
                "example_parallelism_per_run": (
                    int(example_parallelism_per_run)
                    if example_parallelism_per_run is not None
                    else int(sweep_cfg.get("example_parallelism_per_run", 1))
                ),
                "async_model_invocation": (
                    bool(async_model_invocation)
                    if async_model_invocation is not None
                    else bool(sweep_cfg.get("async_model_invocation", False))
                ),
            },
            indent=2,
        )
    )
    if stage == "full" and bool(sweep_cfg.get("canary_enabled", False)):
        thresholds = sweep_cfg.get(
            "canary_pass_thresholds",
            {"valid_run_rate": 0.95, "api_failure_rate": 0.02, "fallback_rate": 0.05},
        )
        existing_canary = _load_existing_canary_summary(sweep_cfg.get("resume_root"))
        canary_ok, canary_failures = _canary_summary_passes(existing_canary or {}, thresholds) if existing_canary else (False, ["missing"])
        if existing_canary is not None and canary_ok:
            print("Skipping canary: existing passing canary summary found in resume_root.")
            print(json.dumps(existing_canary, indent=2))
        else:
            canary_root = run_sweep(
                sweep_config_path=sweep_config,
                base_config_path=base_config,
                provider_preflight_enabled=provider_preflight_enabled,
                provider_preflight_min_success_rate=provider_preflight_min_success_rate,
                provider_preflight_abort_on_fail=provider_preflight_abort_on_fail,
                max_concurrent_runs=max_concurrent_runs,
                stage="canary",
                max_inflight_per_model=max_inflight_per_model,
                time_budget_hours=time_budget_hours,
                qwen_burst_concurrency=qwen_burst_concurrency,
                qwen_degrade_concurrency=qwen_degrade_concurrency,
                resume_mode=resume_mode,
                agentic_backend=agentic_backend,
                example_parallelism_per_run=example_parallelism_per_run,
                async_model_invocation=async_model_invocation,
            )
            canary_summary = summarize_sweep_quality(canary_root)
            write_json(Path(canary_root) / "canary_quality_summary.json", canary_summary)
            failed_checks: list[str] = []
            if canary_summary["valid_run_rate"] < float(thresholds.get("valid_run_rate", 0.95)):
                failed_checks.append(
                    f"valid_run_rate={canary_summary['valid_run_rate']:.3f} < {float(thresholds.get('valid_run_rate', 0.95)):.3f}"
                )
            if canary_summary["api_failure_rate"] > float(thresholds.get("api_failure_rate", 0.02)):
                failed_checks.append(
                    f"api_failure_rate={canary_summary['api_failure_rate']:.3f} > {float(thresholds.get('api_failure_rate', 0.02)):.3f}"
                )
            if canary_summary["fallback_rate"] > float(thresholds.get("fallback_rate", 0.05)):
                failed_checks.append(
                    f"fallback_rate={canary_summary['fallback_rate']:.3f} > {float(thresholds.get('fallback_rate', 0.05)):.3f}"
                )
            print(f"Canary complete: {canary_root}")
            print(json.dumps(canary_summary, indent=2))
            if failed_checks:
                raise RuntimeError("Canary gate failed: " + "; ".join(failed_checks))

    root = run_sweep(
        sweep_config_path=sweep_config,
        base_config_path=base_config,
        provider_preflight_enabled=provider_preflight_enabled,
        provider_preflight_min_success_rate=provider_preflight_min_success_rate,
        provider_preflight_abort_on_fail=provider_preflight_abort_on_fail,
        max_concurrent_runs=max_concurrent_runs,
        stage=stage,
        max_inflight_per_model=max_inflight_per_model,
        time_budget_hours=time_budget_hours,
        qwen_burst_concurrency=qwen_burst_concurrency,
        qwen_degrade_concurrency=qwen_degrade_concurrency,
        resume_mode=resume_mode,
        agentic_backend=agentic_backend,
        example_parallelism_per_run=example_parallelism_per_run,
        async_model_invocation=async_model_invocation,
    )
    capture_environment(root / "environment.json")
    print(f"Completed runs in: {root}")


@app.command("preflight_provider")
def cmd_preflight_provider(
    model_config: str = typer.Option(..., help="Model config yaml"),
    n_probes: int = typer.Option(5, help="Number of health probes"),
    timeout: int = typer.Option(30, help="Timeout per probe (seconds)"),
):
    setup_logging()
    model_cfg = read_yaml(model_config)
    provider = build_provider(model_cfg)
    payload = provider_healthcheck(provider=provider, n_probes=n_probes, timeout=timeout)
    print(json.dumps(payload, indent=2))


@app.command("evaluate")
def cmd_evaluate(
    run_dir: str = typer.Argument(..., help="Run directory containing predictions/verdicts/trajectory"),
    domain: Optional[str] = typer.Option(None, help="Optional domain override"),
):
    setup_logging()
    summary = evaluate_run_dir(run_dir, domain=domain)
    print(json.dumps(summary, indent=2))


@app.command("make_tables")
def cmd_make_tables(
    runs_root: str = typer.Option("runs", help="Root runs directory"),
    output_dir: str = typer.Option("outputs/tables", help="Output table directory"),
    certified_only: bool = typer.Option(
        True,
        "--certified-only/--include-noncertified",
        help="Use only full-stage certified runs for capability tables (default).",
    ),
    comparison_track: str = typer.Option(
        "audited",
        help="Capability comparison track: audited|operational|both",
    ),
    benchmark_role: Optional[str] = typer.Option(
        None,
        help="Optional benchmark-role filter: primary_certification|diagnostic_secondary|mixed",
    ),
):
    setup_logging()
    outputs = make_tables(
        runs_root=runs_root,
        output_dir=output_dir,
        certified_only=certified_only,
        comparison_track=comparison_track,
        benchmark_role=benchmark_role,
    )
    print("Generated table files:")
    for path in outputs:
        print(f"- {path}")


@app.command("evidence_report")
def cmd_evidence_report(
    runs_root: str = typer.Option(..., help="Run root directory to analyze"),
    output_path: str = typer.Option("outputs/evidence_report.json", help="Where to write summary JSON"),
    comparison_track: str = typer.Option(
        "audited",
        help="Evidence comparison track: audited|operational",
    ),
    benchmark_role: Optional[str] = typer.Option(
        None,
        help="Optional benchmark-role filter: primary_certification|diagnostic_secondary|mixed",
    ),
):
    setup_logging()
    summary = summarize_evidence(
        runs_root=runs_root,
        output_path=output_path,
        comparison_track=comparison_track,
        benchmark_role=benchmark_role,
    )
    print(json.dumps(summary, indent=2))


@app.command("evaluate_compliance_challenges")
def cmd_evaluate_compliance_challenges(
    challenge_root: str = typer.Option(
        "configs/compliance_challenges",
        help="Directory containing per-domain compliance challenge JSONL files.",
    ),
    domains: str = typer.Option("healthcare,finance,hr", help="Comma-separated domains"),
    output_path: str = typer.Option(
        "outputs/compliance/challenge_summary.json",
        help="Where to write the compliance challenge summary JSON.",
    ),
):
    setup_logging()
    selected = [d.strip() for d in domains.split(",") if d.strip()]
    summary = evaluate_compliance_challenges(
        challenge_root=challenge_root,
        domains=selected,
        output_path=output_path,
    )
    print(json.dumps(summary, indent=2))


@app.command("make_paper_artifacts")
def cmd_make_paper_artifacts(
    output_dir: str = typer.Option("outputs/paper_generated", help="Directory for generated LaTeX artifacts"),
    base_config: str = typer.Option("configs/base.yaml", help="Base config file"),
    sweep_configs: str = typer.Option(
        "configs/experiments/paper6_power12_ollama.yaml,configs/experiments/ollama_single_model_3domain_power2h.yaml",
        help="Comma-separated sweep config paths to summarize in implementation facts.",
    ),
    model_configs: str = typer.Option(
        "configs/models/ollama_ministral3_cloud.yaml,configs/models/ollama_qwen3_next.yaml,configs/models/mock.yaml",
        help="Comma-separated model config paths to summarize in implementation facts.",
    ),
    compliance_summary: str = typer.Option(
        "outputs/compliance/challenge_summary.json",
        help="Optional compliance challenge summary JSON for Appendix D artifacts.",
    ),
):
    setup_logging()
    outputs = make_paper_artifacts(
        output_dir=output_dir,
        base_config_path=base_config,
        sweep_config_paths=[p.strip() for p in sweep_configs.split(",") if p.strip()],
        model_config_paths=[p.strip() for p in model_configs.split(",") if p.strip()],
        compliance_summary_path=compliance_summary if Path(compliance_summary).exists() else None,
    )
    print("Generated paper artifacts:")
    for path in outputs:
        print(f"- {path}")


@app.command("make_result_plots")
def cmd_make_result_plots(
    tables_dir: str = typer.Option(
        "outputs/tables/final_a6_dual_model_full_v2",
        help="Directory containing result CSV tables.",
    ),
    output_dir: str = typer.Option("figures", help="Directory for generated plot PDFs"),
    prefix: str = typer.Option("final_a6", help="Filename prefix for plot outputs"),
):
    setup_logging()
    outputs = make_result_plots(tables_dir=tables_dir, output_dir=output_dir, prefix=prefix)
    print("Generated result plots:")
    for path in outputs:
        print(f"- {path}")


@app.command("generate_synthetic_resumes")
def cmd_generate_synthetic_resumes(
    n: int = typer.Option(2000, help="Number of resumes"),
    seed: int = typer.Option(42, help="Random seed"),
    output_path: str = typer.Option("data/synthetic/synthetic_resumes.jsonl", help="Output JSONL path"),
):
    setup_logging()
    out = generate_synthetic_resumes(output_path=output_path, n=n, seed=seed)
    print(f"Generated synthetic resumes at {out}")


@app.command("generate_agentic_stress_hr")
def cmd_generate_agentic_stress_hr(
    n: int = typer.Option(600, help="Number of stress examples"),
    seed: int = typer.Option(42, help="Random seed"),
    output_path: str = typer.Option(
        "data/synthetic/agentic_stress_hr.jsonl",
        help="Output JSONL path",
    ),
):
    setup_logging()
    from rava.experiments.agentic_stress import generate_agentic_stress_hr_dataset

    out = generate_agentic_stress_hr_dataset(output_path=output_path, n=n, seed=seed)
    print(f"Generated agentic stress HR dataset at {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
