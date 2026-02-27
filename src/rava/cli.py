from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from rava.agent.providers import build_provider
from rava.agent.react_agent import run_agent_example
from rava.experiments.baselines import get_verification_config
from rava.experiments.datasets import (
    DOMAIN_TO_DATASETS,
    download_datasets,
    export_dataset_card,
    generate_synthetic_resumes,
    preprocess_datasets,
)
from rava.experiments.runner import evaluate_run_dir, run_sweep
from rava.experiments.tables import make_tables
from rava.logging import capture_environment, setup_logging
from rava.specs.parser import load_spec
from rava.utils.serialization import read_yaml, write_json, write_jsonl
from rava.utils.env import load_environment

app = typer.Typer(help="RAVA CLI")
load_environment()


@app.command("download_datasets")
def cmd_download_datasets(
    domains: str = typer.Option("healthcare,finance,hr", help="Comma-separated domains"),
    force: bool = typer.Option(False, help="Force redownload"),
    profile: str = typer.Option("core", help="Dataset profile: core or enhanced"),
):
    selected = [d.strip() for d in domains.split(",") if d.strip()]
    setup_logging()
    download_datasets(domains=selected, force=force, profile=profile)
    export_dataset_card()
    print("Dataset download phase complete.")


@app.command("preprocess_datasets")
def cmd_preprocess_datasets(
    domains: str = typer.Option("healthcare,finance,hr", help="Comma-separated domains"),
    profile: str = typer.Option("core", help="Dataset profile: core or enhanced"),
):
    selected = [d.strip() for d in domains.split(",") if d.strip()]
    setup_logging()
    outputs = preprocess_datasets(domains=selected, profile=profile)
    print(f"Preprocessed {len(outputs)} dataset files.")


@app.command("run_agent")
def cmd_run_agent(
    domain: str = typer.Option("healthcare", help="Domain name"),
    text: str = typer.Argument(..., help="Single input text"),
    model_config: str = typer.Option("configs/models/mock.yaml", help="Model config yaml"),
    verification_config: str = typer.Option("full", help="none/pre/runtime/posthoc/full"),
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
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, result)
    print(json.dumps(result["prediction"], indent=2))


@app.command("run_experiment")
def cmd_run_experiment(
    sweep_config: str = typer.Option("configs/experiments/smoke.yaml", help="Sweep config file"),
    base_config: str = typer.Option("configs/base.yaml", help="Base config file"),
):
    setup_logging()
    root = run_sweep(sweep_config_path=sweep_config, base_config_path=base_config)
    capture_environment(root / "environment.json")
    print(f"Completed runs in: {root}")


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
):
    setup_logging()
    outputs = make_tables(runs_root=runs_root, output_dir=output_dir)
    print("Generated table files:")
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
