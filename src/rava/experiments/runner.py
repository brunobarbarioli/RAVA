from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rava.agent.providers import build_provider
from rava.agent.react_agent import run_agent_example
from rava.experiments.baselines import get_verification_config
from rava.experiments.datasets import get_datasets_for_domain, load_processed_dataset, preprocess_dataset
from rava.experiments.stress import augment_examples_for_stress
from rava.logging import JsonlLogger
from rava.metrics.abstention import compute_abstention_metrics
from rava.metrics.attribution import compute_source_attribution
from rava.metrics.calibration import compute_ece_from_predictions
from rava.metrics.cost import compute_cost_metrics
from rava.metrics.factuality import compute_claim_precision
from rava.metrics.fairness import compute_fairness_metrics
from rava.metrics.latency import compute_latency_metrics
from rava.metrics.violations import compute_violation_rates
from rava.scoring.reliability import compute_reliability_score
from rava.scoring.weights import load_domain_weights
from rava.specs.parser import load_spec
from rava.utils.seeding import set_seed
from rava.utils.serialization import read_yaml, write_json, write_jsonl

logger = logging.getLogger(__name__)


def _predict_labels(predictions: list[dict[str, Any]], domain: str) -> tuple[list[int], list[int], list[str]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    groups: list[str] = []

    for p in predictions:
        meta = p.get("metadata", {}) or {}
        protected = meta.get("protected_attributes") or meta.get("demographics") or {}
        if not isinstance(protected, dict):
            protected = {}
        group_key = "|".join(
            [
                str(protected.get("gender", "na")),
                str(protected.get("race_ethnicity", "na")),
                str(protected.get("age_group", "na")),
                str(protected.get("disability", "na")),
            ]
        )
        groups.append(group_key)

        ref = str(p.get("reference", "")).lower()
        out = str(p.get("output", "")).lower()

        if domain == "hr":
            y_true.append(1 if any(k in ref for k in ["mid", "high", "qualified", "strong"]) else 0)
            y_pred.append(1 if any(k in out for k in ["recommend", "qualified", "advance", "hire", "strong"]) else 0)
        else:
            c = p.get("correct")
            if c is None:
                c = int(bool(ref) and ref in out)
            y_true.append(int(c))
            y_pred.append(int(c))

    return y_true, y_pred, groups


def compute_metrics_for_run(
    predictions: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    domain: str,
    model_name: str | None = None,
    ece_bins: int = 10,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics.update(compute_violation_rates(verdicts))
    metrics.update(compute_claim_precision(predictions))
    metrics.update(compute_source_attribution(predictions))
    metrics.update(compute_latency_metrics(trajectories))
    metrics.update(compute_abstention_metrics(predictions))
    metrics.update(compute_cost_metrics(predictions, model_name=model_name))

    cal = compute_ece_from_predictions(predictions, n_bins=ece_bins, skip_missing=True)
    metrics.update(cal)

    y_true, y_pred, groups = _predict_labels(predictions, domain)
    if len(set(groups)) > 1:
        metrics.update(compute_fairness_metrics(y_true=y_true, y_pred=y_pred, groups=groups))
    else:
        metrics.update(
            {
                "four_fifths_ratio_min": 1.0,
                "demographic_parity_difference": 0.0,
                "equalized_odds_difference": 0.0,
            }
        )

    return metrics


def evaluate_run_dir(run_dir: str | Path, domain: str | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    predictions = []
    verdicts = []
    trajectories = []

    pred_file = run_path / "predictions.jsonl"
    verdict_file = run_path / "verdicts.jsonl"
    traj_file = run_path / "trajectory.jsonl"

    if pred_file.exists():
        from rava.utils.serialization import read_jsonl

        predictions = read_jsonl(pred_file)
    if verdict_file.exists():
        from rava.utils.serialization import read_jsonl

        verdicts = read_jsonl(verdict_file)
    if traj_file.exists():
        from rava.utils.serialization import read_jsonl

        trajectories = read_jsonl(traj_file)

    inferred_domain = domain
    if inferred_domain is None and predictions:
        inferred_domain = str(predictions[0].get("domain", "healthcare"))
    inferred_domain = inferred_domain or "healthcare"

    model_name = str(predictions[0].get("model", "")) if predictions else None
    metrics = compute_metrics_for_run(
        predictions,
        verdicts,
        trajectories,
        domain=inferred_domain,
        model_name=model_name,
    )
    weights_path = Path("configs/domains") / f"{inferred_domain}.yaml"
    weights = load_domain_weights(inferred_domain, config_path=weights_path if weights_path.exists() else None)
    score = compute_reliability_score(metrics, weights)

    summary = {**metrics, **score}
    write_json(run_path / "metrics.json", metrics)
    write_json(run_path / "report.json", summary)
    return summary


def _ensure_processed_data(domain: str, profile: str = "core") -> list[str]:
    datasets = get_datasets_for_domain(domain, profile=profile)
    for dataset in datasets:
        path = Path("data/processed") / domain / dataset / "data.jsonl"
        if not path.exists():
            preprocess_dataset(dataset)
    return datasets


def run_sweep(
    sweep_config_path: str | Path = "configs/experiments/smoke.yaml",
    base_config_path: str | Path = "configs/base.yaml",
) -> Path:
    base_cfg = read_yaml(base_config_path)
    sweep_cfg = read_yaml(sweep_config_path)

    seeds = list(sweep_cfg.get("seeds", base_cfg["experiments"]["seeds"]))
    verification_configs = list(sweep_cfg.get("verification_configs", base_cfg["experiments"]["verification_configs"]))
    domains = list(sweep_cfg.get("domains", ["healthcare", "finance", "hr"]))
    dataset_profile = str(sweep_cfg.get("dataset_profile", "core"))
    stress_test = bool(sweep_cfg.get("stress_test", False))
    model_paths = list(sweep_cfg.get("models", ["configs/models/mock.yaml"]))
    max_examples = int(sweep_cfg.get("max_examples_per_dataset", base_cfg["experiments"]["max_examples_per_dataset"]))
    ece_bins = int(base_cfg.get("metrics", {}).get("ece_bins", 10))

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root = Path(base_cfg["runtime"]["runs_dir"]) / stamp
    root.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        datasets = _ensure_processed_data(domain, profile=dataset_profile)
        spec = load_spec(Path("specs") / f"{domain}.yaml")

        for model_cfg_path in model_paths:
            model_cfg = read_yaml(model_cfg_path)
            provider = build_provider(model_cfg)
            model_name = str(model_cfg.get("name", model_cfg.get("model", "model")))

            for cfg_name in verification_configs:
                vcfg = get_verification_config(cfg_name)

                for seed in seeds:
                    set_seed(int(seed))
                    run_dir = root / domain / model_name / cfg_name / str(seed)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    run_logger = JsonlLogger(run_dir / "events.jsonl")

                    predictions: list[dict[str, Any]] = []
                    verdicts: list[dict[str, Any]] = []
                    trajectories: list[dict[str, Any]] = []
                    reports: list[dict[str, Any]] = []

                    started = time.time()
                    run_id = f"{stamp}-{domain}-{model_name}-{cfg_name}-{seed}"
                    run_logger.log(
                        "run_started",
                        {
                            "run_id": run_id,
                            "domain": domain,
                            "model": model_name,
                            "verification_config": cfg_name,
                            "seed": int(seed),
                        },
                    )

                    for dataset in datasets:
                        examples = load_processed_dataset(domain=domain, dataset=dataset)
                        examples = examples[:max_examples]
                        if stress_test and examples:
                            examples = augment_examples_for_stress(
                                examples=examples,
                                domain=domain,
                                seed=int(seed),
                            )
                        for ex in examples:
                            run_logger.log(
                                "example_started",
                                {"run_id": run_id, "dataset": dataset, "example_id": ex.get("id")},
                            )
                            result = run_agent_example(
                                example=ex,
                                provider=provider,
                                spec=spec,
                                verification_cfg=vcfg,
                                run_id=run_id,
                            )

                            pred = result["prediction"]
                            pred["dataset"] = dataset
                            pred["model"] = model_name
                            pred["verification_config"] = cfg_name
                            pred["seed"] = int(seed)
                            predictions.append(pred)

                            for row in result["verdicts"]:
                                row["dataset"] = dataset
                                row["model"] = model_name
                                row["verification_config"] = cfg_name
                                row["seed"] = int(seed)
                                verdicts.append(row)

                            for row in result["trajectory"]:
                                row["dataset"] = dataset
                                row["model"] = model_name
                                row["verification_config"] = cfg_name
                                row["seed"] = int(seed)
                                trajectories.append(row)

                            reports.append(result["report"])
                            run_logger.log(
                                "example_finished",
                                {"run_id": run_id, "dataset": dataset, "example_id": ex.get("id")},
                            )

                    metrics = compute_metrics_for_run(
                        predictions=predictions,
                        verdicts=verdicts,
                        trajectories=trajectories,
                        domain=domain,
                        model_name=model_name,
                        ece_bins=ece_bins,
                    )
                    weights = load_domain_weights(domain, Path("configs/domains") / f"{domain}.yaml")
                    score = compute_reliability_score(metrics, weights)

                    runtime_seconds = time.time() - started
                    timing = {
                        "runtime_seconds": runtime_seconds,
                        "num_predictions": len(predictions),
                        "avg_prediction_latency_ms": metrics.get("latency_avg_ms", 0.0),
                    }

                    write_jsonl(run_dir / "trajectory.jsonl", trajectories)
                    write_jsonl(run_dir / "predictions.jsonl", predictions)
                    write_jsonl(run_dir / "verdicts.jsonl", verdicts)
                    write_json(run_dir / "metrics.json", metrics)
                    write_json(run_dir / "report.json", {"metrics": metrics, "score": score, "reports": reports})
                    write_json(run_dir / "timing.json", timing)
                    run_logger.log(
                        "run_finished",
                        {
                            "run_id": run_id,
                            "num_predictions": len(predictions),
                            "reliability_score": score["R"],
                            "tier": score["tier"],
                            "runtime_seconds": runtime_seconds,
                        },
                    )

                    logger.info(
                        "Run complete: %s (n=%d, R=%.4f, tier=%s)",
                        run_dir,
                        len(predictions),
                        score["R"],
                        score["tier"],
                    )

    return root
