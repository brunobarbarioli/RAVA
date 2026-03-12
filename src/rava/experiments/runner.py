from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rava.agent.providers import build_provider, classify_provider_error, provider_error_metadata, provider_healthcheck
from rava.agent.react_agent import AgentGenerationFailedError, run_agent_example
from rava.experiments.baselines import get_verification_config
from rava.experiments.datasets import (
    get_datasets_for_domain,
    load_dataset_manifest,
    load_processed_dataset,
    preprocess_dataset,
)
from rava.experiments.stress import augment_examples_for_stress
from rava.logging import JsonlLogger
from rava.metrics.abstention import compute_abstention_metrics
from rava.metrics.attribution import compute_source_attribution
from rava.metrics.calibration import (
    compute_ece_from_predictions,
    fit_confidence_map,
    load_confidence_map,
    write_confidence_map,
)
from rava.metrics.cost import compute_cost_metrics
from rava.metrics.factuality import compute_claim_precision
from rava.metrics.fairness import compute_fairness_metrics_by_attribute
from rava.metrics.latency import compute_latency_metrics
from rava.metrics.run_quality import assess_run_quality_for_model_comparison, compute_run_quality_metrics
from rava.metrics.selective_risk import compute_selective_risk
from rava.metrics.violations import compute_violation_rates
from rava.scoring.reliability import compute_reliability_score
from rava.scoring.weights import load_domain_weights
from rava.specs.parser import load_spec
from rava.utils.hashing import stable_hash
from rava.utils.seeding import set_seed
from rava.utils.serialization import read_jsonl, read_yaml, write_json, write_jsonl
from rava.verification.posthoc_audit import PostHocAuditor

logger = logging.getLogger(__name__)
_THREAD_STATE = threading.local()


@dataclass
class RunTask:
    stamp: str
    root: Path
    domain: str
    model_name: str
    model_cfg: dict[str, Any]
    verification_config: str
    seed: int
    eval_split: str | None
    datasets: list[str]
    examples_by_dataset: dict[str, list[dict[str, Any]]]
    spec: Any
    data_validity_reasons: list[str]
    data_valid_for_certification: bool
    preflight_payload: dict[str, Any]
    preflight_enabled: bool
    preflight_min_success_rate: float
    ece_bins: int
    run_quality_guard_enabled: bool
    fallback_guard_window: int
    fallback_guard_max_rate: float
    example_error_guard_window: int
    example_error_guard_max_rate: float
    max_consecutive_example_failures: int
    stress_test: bool
    seed_globally: bool
    sweep_stage: str
    stats_gate_policy: str
    stats_min_group_count: float
    certification_eligible: bool
    certification_eligibility_reason: str | None
    confidence_map_method: Any
    confidence_map_path: Any
    calibration_map_scope: str
    confidence_map_applied: Any
    confidence_map_selected: Any
    confidence_map_selection_reason: Any
    skip_if_completed: bool
    agentic_backend: str
    tool_cache_scope: str
    max_tool_iterations: int
    example_parallelism_per_run: int
    dataset_profile: str
    benchmark_role: str


def _predict_labels(predictions: list[dict[str, Any]], domain: str) -> tuple[list[int], list[int], list[dict[str, str]]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    protected_rows: list[dict[str, str]] = []

    for p in predictions:
        meta = p.get("metadata", {}) or {}
        protected = meta.get("protected_attributes") or meta.get("demographics") or {}
        if not isinstance(protected, dict):
            protected = {}
        protected_rows.append(
            {
                "gender": str(protected.get("gender", "na")),
                "race_ethnicity": str(protected.get("race_ethnicity", "na")),
                "age_group": str(protected.get("age_group", "na")),
                "disability": str(protected.get("disability", "na")),
            }
        )

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

    return y_true, y_pred, protected_rows


def _resolve_dataset_limit(limit_cfg: Any, dataset: str, default: int) -> int:
    if isinstance(limit_cfg, dict):
        value = limit_cfg.get(dataset, default)
        return int(value)
    if limit_cfg is None:
        return int(default)
    return int(limit_cfg)


def _resolve_min_dataset_rows(min_cfg: Any, dataset: str, default: int = 0) -> int:
    if isinstance(min_cfg, dict):
        value = min_cfg.get(dataset, default)
        return int(value)
    if min_cfg is None:
        return int(default)
    return int(min_cfg)


def _matches_parallelism_override(
    override: dict[str, Any],
    *,
    model_name: str,
    domain: str,
    datasets: list[str],
) -> bool:
    model_filter = str(override.get("model", "")).strip()
    if model_filter and model_filter != model_name:
        return False
    domain_filter = str(override.get("domain", "")).strip()
    if domain_filter and domain_filter != domain:
        return False
    dataset_filter = override.get("dataset")
    if dataset_filter is None:
        return True
    dataset_values = (
        [str(dataset_filter).strip()]
        if not isinstance(dataset_filter, (list, tuple, set))
        else [str(value).strip() for value in dataset_filter]
    )
    dataset_set = {value for value in dataset_values if value}
    if not dataset_set:
        return True
    return any(dataset in dataset_set for dataset in datasets)


def _resolve_example_parallelism_for_task(
    *,
    base_parallelism: int,
    overrides_cfg: Any,
    model_name: str,
    domain: str,
    datasets: list[str],
) -> int:
    if isinstance(overrides_cfg, dict):
        items: list[dict[str, Any]] = []
        for key, value in overrides_cfg.items():
            if not isinstance(key, str):
                continue
            parts = [part.strip() for part in key.split("::")]
            override: dict[str, Any] = {"value": value}
            if len(parts) >= 1 and parts[0]:
                override["model"] = parts[0]
            if len(parts) >= 2 and parts[1]:
                override["domain"] = parts[1]
            if len(parts) >= 3 and parts[2]:
                override["dataset"] = parts[2]
            items.append(override)
    elif isinstance(overrides_cfg, list):
        items = [item for item in overrides_cfg if isinstance(item, dict)]
    else:
        items = []

    resolved = int(base_parallelism)
    for override in items:
        if not _matches_parallelism_override(
            override,
            model_name=model_name,
            domain=domain,
            datasets=datasets,
        ):
            continue
        try:
            resolved = max(1, int(override.get("value", resolved)))
        except Exception:
            continue
    return resolved


def _nested_value(row: dict[str, Any], dotted_path: str) -> Any:
    cur: Any = row
    for part in dotted_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _sample_with_stratification(
    dataset: str,
    examples: list[dict[str, Any]],
    dataset_limit: int,
    stratified_cfg: dict[str, Any] | None,
    seed: int = 42,
    sampling_mode: str = "head",
) -> list[dict[str, Any]]:
    if dataset_limit <= 0 or not examples:
        return []
    if not stratified_cfg:
        if sampling_mode == "paired_fixed_panel":
            rng = random.Random(seed)
            rows = list(examples)
            rows.sort(key=lambda r: str(r.get("id", "")))
            rng.shuffle(rows)
            return rows[:dataset_limit]
        return examples[:dataset_limit]

    field = str(stratified_cfg.get("field", "")).strip()
    strict_counts = bool(stratified_cfg.get("strict", False))
    per_value = stratified_cfg.get("per_value")
    if per_value is None:
        per_value = stratified_cfg.get("counts", {})
    if not field or not isinstance(per_value, dict) or not per_value:
        if sampling_mode == "paired_fixed_panel":
            rng = random.Random(seed)
            rows = list(examples)
            rows.sort(key=lambda r: str(r.get("id", "")))
            rng.shuffle(rows)
            return rows[:dataset_limit]
        return examples[:dataset_limit]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in examples:
        value = _nested_value(row, field)
        key = str(value) if value is not None else "unknown"
        grouped.setdefault(key, []).append(row)

    for key in grouped:
        grouped[key].sort(key=lambda r: str(r.get("id", "")))
        if sampling_mode == "paired_fixed_panel":
            key_seed = int(hashlib.sha256(f"{dataset}:{key}".encode("utf-8")).hexdigest()[:8], 16)
            rng = random.Random(seed + key_seed)
            rng.shuffle(grouped[key])

    configured_total = sum(max(0, int(v)) for v in per_value.values())
    if configured_total <= 0:
        return examples[:dataset_limit]

    target_counts: dict[str, int] = {str(k): max(0, int(v)) for k, v in per_value.items()}
    if configured_total > dataset_limit:
        scaled: dict[str, int] = {}
        residuals: list[tuple[float, str]] = []
        assigned = 0
        for key, count in target_counts.items():
            exact = (float(count) * float(dataset_limit)) / float(configured_total)
            floor = int(exact)
            scaled[key] = floor
            assigned += floor
            residuals.append((exact - floor, key))
        residuals.sort(reverse=True)
        remaining = dataset_limit - assigned
        for _, key in residuals:
            if remaining <= 0:
                break
            scaled[key] += 1
            remaining -= 1
        target_counts = scaled

    if strict_counts:
        shortages = {
            str(key): max(0, int(target) - len(grouped.get(str(key), [])))
            for key, target in target_counts.items()
        }
        unmet = {key: short for key, short in shortages.items() if short > 0}
        if unmet:
            raise RuntimeError(
                f"Stratified sampling for dataset={dataset} could not satisfy requested counts: {unmet}"
            )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for key, target in target_counts.items():
        rows = grouped.get(str(key), [])
        for row in rows[:target]:
            row_id = str(row.get("id", ""))
            if row_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row_id)

    target_size = min(dataset_limit, configured_total)
    if len(selected) < target_size:
        remainder = [row for row in examples if str(row.get("id", "")) not in selected_ids]
        rng = random.Random(seed)
        rng.shuffle(remainder)
        for row in remainder:
            if len(selected) >= target_size:
                break
            selected.append(row)
            selected_ids.add(str(row.get("id", "")))

    return selected[:dataset_limit]


def _calibration_map_key(
    domain: str,
    verification_config: str | None = None,
    dataset: str | None = None,
) -> str:
    if verification_config is None and dataset is None:
        return str(domain)
    if dataset is None:
        return f"{domain}::{verification_config}"
    if verification_config is None:
        return f"{domain}::{dataset}"
    return f"{domain}::{dataset}::{verification_config}"


def _load_calibration_maps_for_model(
    model_name: str,
    domains: list[str],
    verification_configs: list[str],
    calibration_dir: Path | None,
    scope: str = "domain",
    datasets_by_domain: dict[str, list[str]] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    if calibration_dir is None:
        return {}, {}
    selected_scope = str(scope).strip().lower()
    if selected_scope not in {"domain", "domain_config", "domain_dataset_config"}:
        raise ValueError("calibration scope must be one of: domain, domain_config, domain_dataset_config")
    maps: dict[str, Any] = {}
    map_paths: dict[str, str] = {}
    for domain in domains:
        path = calibration_dir / model_name / f"{domain}.json"
        payload = load_confidence_map(path)
        if isinstance(payload, dict):
            key = _calibration_map_key(domain)
            maps[key] = payload
            map_paths[key] = str(path)
        if selected_scope in {"domain_config", "domain_dataset_config"}:
            for cfg_name in verification_configs:
                scoped_path = calibration_dir / model_name / domain / f"{cfg_name}.json"
                scoped_payload = load_confidence_map(scoped_path)
                if not isinstance(scoped_payload, dict):
                    continue
                scoped_key = _calibration_map_key(domain, cfg_name)
                maps[scoped_key] = scoped_payload
                map_paths[scoped_key] = str(scoped_path)
        if selected_scope != "domain_dataset_config":
            continue
        for dataset in datasets_by_domain.get(domain, []) if isinstance(datasets_by_domain, dict) else []:
            for cfg_name in verification_configs:
                scoped_path = calibration_dir / model_name / domain / dataset / f"{cfg_name}.json"
                scoped_payload = load_confidence_map(scoped_path)
                if not isinstance(scoped_payload, dict):
                    continue
                scoped_key = _calibration_map_key(domain, cfg_name, dataset)
                maps[scoped_key] = scoped_payload
                map_paths[scoped_key] = str(scoped_path)
    return maps, map_paths


def _calibration_map_is_eligible(payload: dict[str, Any], min_improvement: float) -> tuple[bool, str]:
    method = str(payload.get("method", "identity")).strip().lower()
    selected_method = str(payload.get("selected_method", method)).strip().lower()
    accepted = bool(payload.get("accepted", False))
    accepted_identity = bool(payload.get("accepted_identity", False))
    ece_before = payload.get("ece_before")
    ece_after = payload.get("ece_after")
    target = payload.get("target_ece")

    try:
        before = None if ece_before is None else float(ece_before)
    except (TypeError, ValueError):
        before = None
    try:
        after = None if ece_after is None else float(ece_after)
    except (TypeError, ValueError):
        after = None
    try:
        target_ece = None if target is None else float(target)
    except (TypeError, ValueError):
        target_ece = None

    improved_enough = bool(
        before is not None and after is not None and (before - after) >= float(min_improvement)
    )
    meets_target = bool(after is not None and target_ece is not None and after <= target_ece)

    if accepted and (improved_enough or meets_target):
        return True, "accepted_with_improvement_or_target"
    if method == "identity" and accepted_identity:
        return True, "accepted_identity_fallback"
    if method == "identity" and selected_method == "identity" and accepted:
        return True, "accepted_identity"
    return False, "rejected_unimproved_or_unaccepted"


def _audit_predictions_with_posthoc(
    predictions: list[dict[str, Any]],
    spec: Any | None,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if spec is None:
        return []
    auditor = PostHocAuditor(spec)
    spec_hash = stable_hash(spec.model_dump_json(), n_hex=16)
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        output = str(pred.get("output", ""))
        if not output.strip():
            continue
        example_id = str(pred.get("id", "unknown"))
        retrieval_context = str(pred.get("retrieval_context", ""))
        report: dict[str, Any]
        cache_path: Path | None = None
        if cache_dir is not None:
            cache_key = stable_hash(
                f"{spec_hash}|{stable_hash(output, n_hex=16)}|{stable_hash(retrieval_context, n_hex=16)}",
                n_hex=24,
            )
            cache_path = cache_dir / spec_hash / f"{cache_key}.json"
        if cache_path is not None and cache_path.exists():
            report = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            report = auditor.audit(
                final_text=output,
                context={
                    "input": str(pred.get("input", "")),
                    "retrieval_context": retrieval_context,
                },
                aggregate_metrics={},
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(cache_path, report)
        for verdict in report.get("constraint_verdicts", []):
            row = dict(verdict)
            row["record_id"] = example_id
            row["action_key"] = f"{example_id}:audited_posthoc:1"
            row["layer"] = "audited_posthoc"
            row["event"] = "audited_posthoc"
            row["final_selected"] = True
            rows.append(row)
    return rows


def _fit_and_persist_calibration_maps(
    root: Path,
    *,
    domains: list[str],
    datasets_by_domain: dict[str, list[str]],
    model_names: list[str],
    verification_configs: list[str],
    calibration_dir: Path,
    scope: str = "domain",
    methods: tuple[str, ...],
    target_ece: float,
    ece_bins: int,
    min_samples: int,
    min_improvement: float,
) -> dict[str, Any]:
    selected_scope = str(scope).strip().lower()
    if selected_scope not in {"domain", "domain_config", "domain_dataset_config"}:
        raise ValueError("calibration scope must be one of: domain, domain_config, domain_dataset_config")
    artifacts: dict[str, Any] = {}
    for model_name in model_names:
        for domain in domains:
            model_domain_root = root / domain / model_name
            if not model_domain_root.exists():
                continue

            domain_predictions: list[dict[str, Any]] = []
            for pred_path in model_domain_root.glob("**/predictions.jsonl"):
                try:
                    domain_predictions.extend(read_jsonl(pred_path))
                except Exception:
                    logger.exception("Failed reading predictions for calibration map: %s", pred_path)

            if domain_predictions:
                domain_payload = fit_confidence_map(
                    records=domain_predictions,
                    methods=methods,
                    target_ece=target_ece,
                    n_bins=ece_bins,
                    min_samples=min_samples,
                    min_improvement=min_improvement,
                )
                domain_payload["model"] = model_name
                domain_payload["domain"] = domain
                domain_payload["scope"] = "domain"
                domain_payload["fitted_at_utc"] = datetime.now(timezone.utc).isoformat()
                domain_target_path = calibration_dir / model_name / f"{domain}.json"
                write_confidence_map(domain_target_path, domain_payload)
                artifacts[f"{model_name}:{domain}"] = {
                    "path": str(domain_target_path),
                    "scope": "domain",
                    "accepted": bool(domain_payload.get("accepted", False)),
                    "ece_before": domain_payload.get("ece_before"),
                    "ece_after": domain_payload.get("ece_after"),
                    "method": domain_payload.get("method"),
                    "selected_method": domain_payload.get("selected_method"),
                    "selection_reason": domain_payload.get("selection_reason"),
                    "n_samples": domain_payload.get("n_samples"),
                    "reason": domain_payload.get("reason"),
                }

            if selected_scope in {"domain_config", "domain_dataset_config"}:
                for cfg_name in verification_configs:
                    cfg_root = model_domain_root / cfg_name
                    if not cfg_root.exists():
                        continue
                    cfg_predictions: list[dict[str, Any]] = []
                    for pred_path in cfg_root.glob("**/predictions.jsonl"):
                        try:
                            cfg_predictions.extend(read_jsonl(pred_path))
                        except Exception:
                            logger.exception("Failed reading predictions for calibration map: %s", pred_path)
                    if not cfg_predictions:
                        continue

                    if selected_scope == "domain_config":
                        cfg_payload = fit_confidence_map(
                            records=cfg_predictions,
                            methods=methods,
                            target_ece=target_ece,
                            n_bins=ece_bins,
                            min_samples=min_samples,
                            min_improvement=min_improvement,
                        )
                        cfg_payload["model"] = model_name
                        cfg_payload["domain"] = domain
                        cfg_payload["verification_config"] = cfg_name
                        cfg_payload["scope"] = "domain_config"
                        cfg_payload["fitted_at_utc"] = datetime.now(timezone.utc).isoformat()
                        cfg_target_path = calibration_dir / model_name / domain / f"{cfg_name}.json"
                        write_confidence_map(cfg_target_path, cfg_payload)
                        artifacts[f"{model_name}:{domain}:{cfg_name}"] = {
                            "path": str(cfg_target_path),
                            "scope": "domain_config",
                            "verification_config": cfg_name,
                            "accepted": bool(cfg_payload.get("accepted", False)),
                            "ece_before": cfg_payload.get("ece_before"),
                            "ece_after": cfg_payload.get("ece_after"),
                            "method": cfg_payload.get("method"),
                            "selected_method": cfg_payload.get("selected_method"),
                            "selection_reason": cfg_payload.get("selection_reason"),
                            "n_samples": cfg_payload.get("n_samples"),
                            "reason": cfg_payload.get("reason"),
                        }
                        continue

                    for dataset in datasets_by_domain.get(domain, []):
                        dataset_predictions = [
                            row
                            for row in cfg_predictions
                            if str(row.get("dataset", "")) == str(dataset)
                        ]
                        if not dataset_predictions:
                            continue
                        cfg_payload = fit_confidence_map(
                            records=dataset_predictions,
                            methods=methods,
                            target_ece=target_ece,
                            n_bins=ece_bins,
                            min_samples=min_samples,
                            min_improvement=min_improvement,
                        )
                        cfg_payload["model"] = model_name
                        cfg_payload["domain"] = domain
                        cfg_payload["dataset"] = dataset
                        cfg_payload["verification_config"] = cfg_name
                        cfg_payload["scope"] = "domain_dataset_config"
                        cfg_payload["fitted_at_utc"] = datetime.now(timezone.utc).isoformat()
                        cfg_target_path = calibration_dir / model_name / domain / dataset / f"{cfg_name}.json"
                        write_confidence_map(cfg_target_path, cfg_payload)
                        artifacts[f"{model_name}:{domain}:{dataset}:{cfg_name}"] = {
                            "path": str(cfg_target_path),
                            "scope": "domain_dataset_config",
                            "dataset": dataset,
                            "verification_config": cfg_name,
                            "accepted": bool(cfg_payload.get("accepted", False)),
                            "ece_before": cfg_payload.get("ece_before"),
                            "ece_after": cfg_payload.get("ece_after"),
                            "method": cfg_payload.get("method"),
                            "selected_method": cfg_payload.get("selected_method"),
                            "selection_reason": cfg_payload.get("selection_reason"),
                            "n_samples": cfg_payload.get("n_samples"),
                            "reason": cfg_payload.get("reason"),
                        }
    write_json(calibration_dir / "index.json", artifacts)
    return artifacts


def _validate_manifest_requirements(
    domain: str,
    dataset: str,
    eval_split: str | None,
    disallow_toy_fallback: bool,
    min_dataset_rows_cfg: Any,
) -> dict[str, Any]:
    manifest = load_dataset_manifest(domain=domain, dataset=dataset) or {}
    source_type = str(manifest.get("source_type", "unknown"))
    row_counts = manifest.get("row_count_by_split", {}) if isinstance(manifest.get("row_count_by_split"), dict) else {}
    total_rows = int(manifest.get("total_rows", 0))
    split_key = str(eval_split) if eval_split else None
    available = int(row_counts.get(split_key, 0)) if split_key else total_rows
    required = _resolve_min_dataset_rows(min_dataset_rows_cfg, dataset=dataset, default=0)

    if disallow_toy_fallback and source_type == "toy":
        raise RuntimeError(f"Dataset {domain}/{dataset} manifest indicates toy source; disallowed for this sweep.")
    if required > 0 and available < required:
        suffix = f" split='{split_key}'" if split_key else ""
        raise RuntimeError(
            f"Dataset {domain}/{dataset} has {available} rows{suffix}, below required minimum {required}."
        )
    return {
        "dataset": dataset,
        "source_type": source_type,
        "available_rows": available,
        "required_rows": required,
        "manifest": manifest,
    }


def compute_metrics_for_run(
    predictions: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    domain: str,
    spec: Any | None = None,
    audited_verdicts: list[dict[str, Any]] | None = None,
    model_name: str | None = None,
    ece_bins: int = 10,
    data_valid_for_certification: bool = True,
    data_validity_reasons: list[str] | None = None,
    runner_generation_calls: int | None = None,
    runner_generation_error_count: int | None = None,
    runner_error_taxonomy_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"num_predictions": float(len(predictions))}
    record_ids = {str(row.get("id", "")) for row in predictions if str(row.get("id", "")).strip()}
    halted_ids = {
        str(row.get("id", ""))
        for row in predictions
        if bool(row.get("runtime_halted_hard_fail", False))
    }
    operational_vio = compute_violation_rates(
        verdicts,
        halted_record_ids=halted_ids,
        record_ids=record_ids,
    )
    metrics.update(operational_vio)
    metrics["operational_hard_violation_rate"] = float(operational_vio.get("operational_hard_violation_rate", 0.0))
    metrics["operational_soft_violation_rate"] = float(operational_vio.get("operational_soft_violation_rate", 0.0))

    audited_rows = list(audited_verdicts or [])
    if not audited_rows:
        audited_rows = _audit_predictions_with_posthoc(predictions=predictions, spec=spec)
    audited_vio = compute_violation_rates(
        audited_rows,
        halted_record_ids=halted_ids,
        record_ids=record_ids,
    )
    metrics["audited_hard_violation_rate"] = float(
        audited_vio.get("residual_hard_violation_rate", audited_vio.get("hard_violation_rate", 0.0))
    )
    metrics["audited_soft_violation_rate"] = float(
        audited_vio.get("residual_soft_violation_rate", audited_vio.get("soft_violation_rate", 0.0))
    )
    metrics["audited_residual_hard_violation_rate"] = float(audited_vio.get("residual_hard_violation_rate", 0.0))
    metrics["audited_residual_soft_violation_rate"] = float(audited_vio.get("residual_soft_violation_rate", 0.0))
    metrics["audited_unmitigated_halt_rate"] = float(audited_vio.get("unmitigated_halt_rate", 0.0))
    metrics.update(compute_claim_precision(predictions))
    metrics.update(compute_source_attribution(predictions))
    metrics.update(compute_latency_metrics(trajectories))
    metrics.update(compute_abstention_metrics(predictions))
    metrics.update(compute_selective_risk(predictions))
    metrics.update(compute_cost_metrics(predictions, model_name=model_name))
    metrics.update(
        compute_run_quality_metrics(
            trajectories,
            runner_generation_calls=runner_generation_calls,
            runner_generation_error_count=runner_generation_error_count,
            runner_error_taxonomy_counts=runner_error_taxonomy_counts,
        )
    )

    cal = compute_ece_from_predictions(predictions, n_bins=ece_bins, skip_missing=True)
    metrics.update(cal)

    y_true, y_pred, protected_rows = _predict_labels(predictions, domain)
    metrics.update(
        compute_fairness_metrics_by_attribute(
            y_true=y_true,
            y_pred=y_pred,
            protected_rows=protected_rows,
        )
    )

    metrics.update(assess_run_quality_for_model_comparison(metrics))
    metrics["data_valid_for_certification"] = bool(data_valid_for_certification)
    metrics["data_validity_reasons"] = list(data_validity_reasons or [])
    metrics["audited_verdict_count"] = float(len(audited_rows))
    return metrics


def evaluate_run_dir(run_dir: str | Path, domain: str | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    predictions = []
    verdicts = []
    trajectories = []
    audited_verdicts = []

    pred_file = run_path / "predictions.jsonl"
    verdict_file = run_path / "verdicts.jsonl"
    traj_file = run_path / "trajectory.jsonl"
    audited_verdict_file = run_path / "audited_verdicts.jsonl"

    if pred_file.exists():
        from rava.utils.serialization import read_jsonl

        predictions = read_jsonl(pred_file)
    if verdict_file.exists():
        from rava.utils.serialization import read_jsonl

        verdicts = read_jsonl(verdict_file)
    if traj_file.exists():
        from rava.utils.serialization import read_jsonl

        trajectories = read_jsonl(traj_file)
    if audited_verdict_file.exists():
        from rava.utils.serialization import read_jsonl

        audited_verdicts = read_jsonl(audited_verdict_file)

    inferred_domain = domain
    if inferred_domain is None and predictions:
        inferred_domain = str(predictions[0].get("domain", "healthcare"))
    inferred_domain = inferred_domain or "healthcare"

    model_name = str(predictions[0].get("model", "")) if predictions else None
    spec_path = Path("specs") / f"{inferred_domain}.yaml"
    spec = load_spec(spec_path) if spec_path.exists() else None
    metrics = compute_metrics_for_run(
        predictions,
        verdicts,
        trajectories,
        domain=inferred_domain,
        spec=spec,
        audited_verdicts=audited_verdicts,
        model_name=model_name,
    )
    weights_path = Path("configs/domains") / f"{inferred_domain}.yaml"
    weights = load_domain_weights(inferred_domain, config_path=weights_path if weights_path.exists() else None)
    score_operational = compute_reliability_score(
        metrics=metrics,
        weights=weights,
        spec=spec,
        stats_gate_policy=str(metrics.get("stats_gate_policy", "constraint_aware")),
        stats_min_group_count=float(metrics.get("stats_min_group_count", 30.0)),
        certification_eligible=bool(metrics.get("certification_eligible", True)),
        certification_eligibility_reason=metrics.get("certification_eligibility_reason"),
        track="operational",
    )
    score_audited = compute_reliability_score(
        metrics=metrics,
        weights=weights,
        spec=spec,
        stats_gate_policy=str(metrics.get("stats_gate_policy", "constraint_aware")),
        stats_min_group_count=float(metrics.get("stats_min_group_count", 30.0)),
        certification_eligible=bool(metrics.get("certification_eligible", True)),
        certification_eligibility_reason=metrics.get("certification_eligibility_reason"),
        track="audited",
    )
    score = {
        "R": score_audited["R_raw"],
        "R_raw": score_audited["R_raw"],
        "R_certified": score_audited["R_certified"],
        "tier": score_audited["tier"],
        "R_operational_raw": score_operational["R_raw"],
        "R_operational_certified": score_operational["R_certified"],
        "R_audited_raw": score_audited["R_raw"],
        "R_audited_certified": score_audited["R_certified"],
        "gate_flags_operational": score_operational["gate_flags"],
        "gate_flags_audited": score_audited["gate_flags"],
        "certification_reasons_operational": score_operational["certification_reasons"],
        "certification_reasons_audited": score_audited["certification_reasons"],
    }

    summary = {**metrics, **score}
    write_json(run_path / "metrics.json", metrics)
    write_json(run_path / "report.json", summary)
    return summary


def _ensure_processed_data(
    domain: str,
    profile: str = "core",
    split_strategy: str = "preserve",
    disallow_toy_fallback: bool = False,
) -> list[str]:
    datasets = get_datasets_for_domain(domain, profile=profile)
    for dataset in datasets:
        path = Path("data/processed") / domain / dataset / "data.jsonl"
        if not path.exists():
            preprocess_dataset(
                dataset,
                split_strategy=split_strategy,
                disallow_toy_fallback=disallow_toy_fallback,
            )
    return datasets


def _provider_cache_key(model_cfg: dict[str, Any]) -> str:
    return json.dumps(model_cfg, sort_keys=True, default=str)


def _get_thread_provider(model_cfg: dict[str, Any]):
    cache = getattr(_THREAD_STATE, "providers", None)
    if cache is None:
        cache = {}
        setattr(_THREAD_STATE, "providers", cache)
    key = _provider_cache_key(model_cfg)
    provider = cache.get(key)
    if provider is None:
        provider = build_provider(model_cfg)
        cache[key] = provider
    return provider


def _parse_model_inflight_caps(
    value: str | dict[str, Any] | None,
    default_cap: int,
) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        caps: dict[str, int] = {}
        for key, raw in value.items():
            cap = min(max(1, int(raw)), max(1, int(default_cap)))
            caps[str(key)] = cap
        return caps
    if not str(value).strip():
        return {}

    caps = {}
    items = [chunk.strip() for chunk in str(value).split(",") if chunk.strip()]
    for item in items:
        if "=" not in item:
            continue
        model, raw_cap = item.split("=", 1)
        model = model.strip()
        if not model:
            continue
        cap = min(max(1, int(raw_cap.strip())), max(1, int(default_cap)))
        caps[model] = cap
    return caps


def _benchmark_role_for_profile(profile: str) -> str:
    profile_norm = str(profile).strip().lower()
    if profile_norm in {"primary_certification", "paper3_mini", "final_a6"}:
        return "primary_certification"
    if profile_norm == "diagnostic_secondary":
        return "diagnostic_secondary"
    return "mixed"


def summarize_sweep_quality(root: str | Path) -> dict[str, float]:
    root_path = Path(root)
    metrics_paths = list(root_path.glob("**/metrics.json"))
    if not metrics_paths:
        return {
            "n_runs": 0.0,
            "valid_run_rate": 0.0,
            "api_failure_rate": 1.0,
            "fallback_rate": 1.0,
        }

    valid = 0
    api_rates: list[float] = []
    fallback_rates: list[float] = []
    for path in metrics_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if bool(payload.get("valid_for_model_comparison", False)):
            valid += 1
        api_rates.append(float(payload.get("api_failure_rate", 0.0)))
        fallback_rates.append(float(payload.get("generation_fallback_rate", 0.0)))

    n = len(metrics_paths)
    return {
        "n_runs": float(n),
        "valid_run_rate": valid / float(n),
        "api_failure_rate": sum(api_rates) / float(n),
        "fallback_rate": sum(fallback_rates) / float(n),
    }


def _execute_run_task(task: RunTask) -> Path:
    run_dir = task.root / task.domain / task.model_name / task.verification_config / str(task.seed)
    if task.skip_if_completed and (run_dir / "metrics.json").exists():
        logger.info("Skipping completed run: %s", run_dir)
        return run_dir

    vcfg = get_verification_config(task.verification_config)

    if task.seed_globally:
        set_seed(int(task.seed))

    run_dir.mkdir(parents=True, exist_ok=True)
    run_logger = JsonlLogger(run_dir / "events.jsonl")

    predictions: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    infra_failed = False
    infra_error_taxonomy: str | None = None
    infra_error_message: str | None = None
    generation_calls = 0
    generation_error_count = 0
    total_examples_seen = 0
    consecutive_example_failures = 0
    runner_error_taxonomy_counts: dict[str, int] = {}
    fallback_calls = 0

    started = time.time()
    run_id = f"{task.stamp}-{task.domain}-{task.model_name}-{task.verification_config}-{task.seed}"
    run_logger.log(
        "run_started",
        {
            "run_id": run_id,
            "domain": task.domain,
            "model": task.model_name,
            "verification_config": task.verification_config,
            "seed": int(task.seed),
            "eval_split": task.eval_split,
        },
    )

    flattened: list[tuple[str, dict[str, Any]]] = []
    for dataset in task.datasets:
        examples = list(task.examples_by_dataset.get(dataset, []))
        if task.stress_test and examples:
            examples = augment_examples_for_stress(
                examples=examples,
                domain=task.domain,
                seed=int(task.seed),
            )
        for ex in examples:
            flattened.append((dataset, ex))

    def _invoke_example(dataset: str, ex: dict[str, Any]) -> dict[str, Any]:
        local_provider = _get_thread_provider(task.model_cfg)
        try:
            result = run_agent_example(
                example=ex,
                provider=local_provider,
                spec=task.spec,
                verification_cfg=vcfg,
                run_id=run_id,
                agentic_backend=task.agentic_backend,
                tool_cache_scope=task.tool_cache_scope,
                max_tool_iterations=task.max_tool_iterations,
            )
            return {"ok": True, "dataset": dataset, "example": ex, "result": result}
        except Exception as exc:
            return {"ok": False, "dataset": dataset, "example": ex, "error": exc}

    def _process_example_outcome(payload: dict[str, Any]) -> None:
        nonlocal generation_calls
        nonlocal generation_error_count
        nonlocal consecutive_example_failures
        nonlocal infra_failed
        nonlocal infra_error_taxonomy
        nonlocal infra_error_message
        nonlocal fallback_calls
        nonlocal total_examples_seen

        dataset = str(payload.get("dataset", "unknown"))
        ex = payload.get("example", {}) or {}
        total_examples_seen += 1

        if not bool(payload.get("ok", False)):
            exc = payload.get("error")
            if not isinstance(exc, Exception):
                exc = RuntimeError("Unknown example execution failure.")
            generation_calls += 1
            generation_error_count += 1
            consecutive_example_failures += 1
            taxonomy = classify_provider_error(exc)
            error_meta = provider_error_metadata(exc)
            runner_error_taxonomy_counts[taxonomy] = runner_error_taxonomy_counts.get(taxonomy, 0) + 1

            partial_trajectories: list[dict[str, Any]] = []
            partial_verdicts: list[dict[str, Any]] = []
            if isinstance(exc, AgentGenerationFailedError):
                partial_trajectories = list(exc.trajectory_rows)
                partial_verdicts = list(exc.verdict_rows)
            for row in partial_verdicts:
                row["dataset"] = dataset
                row["model"] = task.model_name
                row["verification_config"] = task.verification_config
                row["seed"] = int(task.seed)
                verdicts.append(row)
            if partial_trajectories:
                for row in partial_trajectories:
                    row["dataset"] = dataset
                    row["model"] = task.model_name
                    row["verification_config"] = task.verification_config
                    row["seed"] = int(task.seed)
                    trajectories.append(row)
            else:
                now = time.time()
                trajectories.append(
                    {
                        "run_id": run_id,
                        "example_id": str(ex.get("id", "unknown")),
                        "domain": task.domain,
                        "step_id": 0,
                        "phase": "agent",
                        "action": "langchain_agent_generation",
                        "observation": "GENERATION_EXCEPTION",
                        "started_at": now,
                        "ended_at": now,
                        "duration_ms": 0.0,
                        "metadata": {
                            "mode": "langchain_agent_exception",
                            "error": str(exc),
                            "error_taxonomy": taxonomy,
                            "error_class": error_meta.get("error_class"),
                            "error_message": error_meta.get("error_message"),
                            "max_tokens_used": error_meta.get("max_tokens_used"),
                            "attempt_history": error_meta.get("attempt_history", []),
                        },
                        "dataset": dataset,
                        "model": task.model_name,
                        "verification_config": task.verification_config,
                        "seed": int(task.seed),
                    }
                )

            run_logger.log(
                "example_failed",
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "example_id": ex.get("id"),
                    "taxonomy": taxonomy,
                    "error": str(exc),
                    "error_class": error_meta.get("error_class"),
                    "error_message": error_meta.get("error_message"),
                    "max_tokens_used": error_meta.get("max_tokens_used"),
                },
            )

            if (
                task.run_quality_guard_enabled
                and consecutive_example_failures >= task.max_consecutive_example_failures
            ):
                infra_failed = True
                infra_error_taxonomy = "consecutive_example_failures_guard_triggered"
                infra_error_message = (
                    f"consecutive_example_failures={consecutive_example_failures} exceeds "
                    f"threshold={task.max_consecutive_example_failures}"
                )
            elif (
                task.run_quality_guard_enabled
                and total_examples_seen >= task.example_error_guard_window
                and total_examples_seen > 0
            ):
                error_rate = generation_error_count / float(total_examples_seen)
                if error_rate > task.example_error_guard_max_rate:
                    infra_failed = True
                    infra_error_taxonomy = "example_error_rate_guard_triggered"
                    infra_error_message = (
                        f"example_error_rate={error_rate:.3f} exceeds "
                        f"threshold={task.example_error_guard_max_rate:.3f}"
                    )
            if infra_failed:
                run_logger.log(
                    "run_aborted",
                    {
                        "run_id": run_id,
                        "reason": infra_error_taxonomy,
                        "message": infra_error_message,
                    },
                )
            return

        result = payload["result"]
        pred = result["prediction"]
        pred["dataset"] = dataset
        pred["model"] = task.model_name
        pred["verification_config"] = task.verification_config
        pred["seed"] = int(task.seed)
        predictions.append(pred)

        generation_calls += 1
        consecutive_example_failures = 0
        generation_mode = str(pred.get("generation_mode", "")).lower()
        if "fallback" in generation_mode:
            fallback_calls += 1

        for row in result["verdicts"]:
            row["dataset"] = dataset
            row["model"] = task.model_name
            row["verification_config"] = task.verification_config
            row["seed"] = int(task.seed)
            verdicts.append(row)

        for row in result["trajectory"]:
            row["dataset"] = dataset
            row["model"] = task.model_name
            row["verification_config"] = task.verification_config
            row["seed"] = int(task.seed)
            trajectories.append(row)

        reports.append(result["report"])
        run_logger.log(
            "example_finished",
            {"run_id": run_id, "dataset": dataset, "example_id": ex.get("id")},
        )

        if (
            task.run_quality_guard_enabled
            and generation_calls >= task.fallback_guard_window
            and generation_calls > 0
        ):
            fallback_rate = fallback_calls / float(generation_calls)
            if fallback_rate > task.fallback_guard_max_rate:
                infra_failed = True
                infra_error_taxonomy = "fallback_guard_triggered"
                infra_error_message = (
                    f"fallback_rate={fallback_rate:.3f} exceeds "
                    f"threshold={task.fallback_guard_max_rate:.3f}"
                )
                run_logger.log(
                    "run_aborted",
                    {
                        "run_id": run_id,
                        "reason": infra_error_taxonomy,
                        "message": infra_error_message,
                    },
                )

    if task.example_parallelism_per_run <= 1:
        for dataset, ex in flattened:
            run_logger.log(
                "example_started",
                {"run_id": run_id, "dataset": dataset, "example_id": ex.get("id")},
            )
            _process_example_outcome(_invoke_example(dataset, ex))
            if infra_failed:
                break
    else:
        with ThreadPoolExecutor(
            max_workers=task.example_parallelism_per_run,
            thread_name_prefix="rava-example",
        ) as example_pool:
            idx = 0
            running: dict[Any, tuple[str, dict[str, Any]]] = {}

            def _submit_next() -> bool:
                nonlocal idx
                if idx >= len(flattened):
                    return False
                dataset, ex = flattened[idx]
                idx += 1
                run_logger.log(
                    "example_started",
                    {"run_id": run_id, "dataset": dataset, "example_id": ex.get("id")},
                )
                future = example_pool.submit(_invoke_example, dataset, ex)
                running[future] = (dataset, ex)
                return True

            while len(running) < task.example_parallelism_per_run and _submit_next():
                pass

            while running and not infra_failed:
                done, _ = wait(set(running.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    running.pop(future, None)
                    payload = future.result()
                    _process_example_outcome(payload)
                    if infra_failed:
                        break
                    while len(running) < task.example_parallelism_per_run and _submit_next():
                        pass

    dataset_rank = {name: idx for idx, name in enumerate(task.datasets)}
    predictions.sort(
        key=lambda row: (
            dataset_rank.get(str(row.get("dataset", "")), 10**9),
            str(row.get("id", "")),
        )
    )
    trajectories.sort(
        key=lambda row: (
            dataset_rank.get(str(row.get("dataset", "")), 10**9),
            str(row.get("example_id", "")),
            int(row.get("step_id", 0)),
        )
    )
    verdicts.sort(
        key=lambda row: (
            dataset_rank.get(str(row.get("dataset", "")), 10**9),
            str(row.get("record_id", "")),
            str(row.get("action_key", "")),
            str(row.get("layer", "")),
            str(row.get("constraint_id", "")),
        )
    )

    audited_rows = _audit_predictions_with_posthoc(
        predictions=predictions,
        spec=task.spec,
        cache_dir=task.root / "_cache" / "audited_posthoc",
    )
    metrics = compute_metrics_for_run(
        predictions=predictions,
        verdicts=verdicts,
        trajectories=trajectories,
        domain=task.domain,
        spec=task.spec,
        audited_verdicts=audited_rows,
        model_name=task.model_name,
        ece_bins=task.ece_bins,
        data_valid_for_certification=task.data_valid_for_certification,
        data_validity_reasons=task.data_validity_reasons,
        runner_generation_calls=generation_calls,
        runner_generation_error_count=generation_error_count,
        runner_error_taxonomy_counts=runner_error_taxonomy_counts,
    )
    metrics["example_attempt_count"] = float(total_examples_seen)
    metrics["example_error_count"] = float(generation_error_count)
    metrics["example_error_rate"] = (
        float(generation_error_count) / float(total_examples_seen) if total_examples_seen > 0 else 0.0
    )
    metrics["provider_preflight_success_rate"] = float(task.preflight_payload.get("success_rate", 1.0))
    metrics["provider_preflight_enabled"] = bool(task.preflight_enabled)
    metrics["provider_preflight_threshold"] = float(task.preflight_min_success_rate)
    metrics["sweep_stage"] = str(task.sweep_stage)
    metrics["certification_eligible"] = bool(task.certification_eligible)
    metrics["certification_eligibility_reason"] = task.certification_eligibility_reason
    metrics["stats_gate_policy"] = str(task.stats_gate_policy)
    metrics["stats_min_group_count"] = float(task.stats_min_group_count)
    metrics["calibration_map_selected"] = task.confidence_map_selected
    metrics["calibration_map_applied"] = bool(task.confidence_map_applied)
    metrics["calibration_selection_reason"] = task.confidence_map_selection_reason
    metrics["calibration_map_method"] = task.confidence_map_method
    metrics["calibration_map_path"] = task.confidence_map_path
    metrics["calibration_map_scope"] = str(task.calibration_map_scope)
    metrics["dataset_profile"] = str(task.dataset_profile)
    metrics["benchmark_role"] = str(task.benchmark_role)

    reasons = set(metrics.get("run_quality_reasons", []))
    if float(task.preflight_payload.get("success_rate", 1.0)) < task.preflight_min_success_rate:
        metrics["valid_for_model_comparison"] = False
        reasons.add(f"preflight_success_rate<{task.preflight_min_success_rate:.2f}")
    if infra_failed:
        metrics["valid_for_model_comparison"] = False
        metrics["infra_failed"] = True
        metrics["infra_error_taxonomy"] = infra_error_taxonomy
        metrics["infra_error_message"] = infra_error_message
        reasons.add("infra_failed")
        if infra_error_taxonomy:
            reasons.add(f"infra_error:{infra_error_taxonomy}")
    if not task.data_valid_for_certification:
        metrics["valid_for_model_comparison"] = False
        reasons.add("data_invalid_for_certification")
    metrics["run_quality_reasons"] = sorted(reasons)

    weights = load_domain_weights(task.domain, Path("configs/domains") / f"{task.domain}.yaml")
    score_operational = compute_reliability_score(
        metrics=metrics,
        weights=weights,
        spec=task.spec,
        stats_gate_policy=task.stats_gate_policy,
        stats_min_group_count=task.stats_min_group_count,
        certification_eligible=task.certification_eligible,
        certification_eligibility_reason=task.certification_eligibility_reason,
        track="operational",
    )
    score_audited = compute_reliability_score(
        metrics=metrics,
        weights=weights,
        spec=task.spec,
        stats_gate_policy=task.stats_gate_policy,
        stats_min_group_count=task.stats_min_group_count,
        certification_eligible=task.certification_eligible,
        certification_eligibility_reason=task.certification_eligibility_reason,
        track="audited",
    )
    score: dict[str, Any] = {
        "R": score_audited["R_raw"],
        "R_raw": score_audited["R_raw"],
        "R_certified": score_audited["R_certified"],
        "tier": score_audited["tier"],
        "certification_status": score_audited["certification_status"],
        "components": score_audited["components"],
        "missing_components": score_audited["missing_components"],
        "certification_reasons": score_audited["certification_reasons"],
        "gate_flags": score_audited["gate_flags"],
        "R_prevent": score_operational["R_prevent"],
        "R_detect": score_operational["R_detect"],
        "R_operational_raw": score_operational["R_raw"],
        "R_operational_certified": score_operational["R_certified"],
        "R_audited_raw": score_audited["R_raw"],
        "R_audited_certified": score_audited["R_certified"],
        "tier_operational": score_operational["tier"],
        "tier_audited": score_audited["tier"],
        "certification_reasons_operational": score_operational["certification_reasons"],
        "certification_reasons_audited": score_audited["certification_reasons"],
        "gate_flags_operational": score_operational["gate_flags"],
        "gate_flags_audited": score_audited["gate_flags"],
    }
    valid_for_comparison = bool(metrics.get("valid_for_model_comparison", True))
    for row in predictions:
        row["invalid_for_model_comparison"] = not valid_for_comparison

    runtime_seconds = time.time() - started
    timing = {
        "runtime_seconds": runtime_seconds,
        "num_predictions": len(predictions),
        "avg_prediction_latency_ms": metrics.get("latency_avg_ms", 0.0),
        "infra_failed": infra_failed,
    }

    write_jsonl(run_dir / "trajectory.jsonl", trajectories)
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_jsonl(run_dir / "verdicts.jsonl", verdicts)
    write_jsonl(run_dir / "audited_verdicts.jsonl", audited_rows)
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "report.json",
        {
            "metrics": metrics,
            "score": score,
            "reports": reports,
            "valid_for_model_comparison": valid_for_comparison,
            "run_quality_reasons": metrics.get("run_quality_reasons", []),
            "certification_status": score.get("certification_status"),
            "dataset_profile": task.dataset_profile,
            "benchmark_role": task.benchmark_role,
        },
    )
    write_json(run_dir / "timing.json", timing)
    run_logger.log(
        "run_finished",
        {
            "run_id": run_id,
            "num_predictions": len(predictions),
            "reliability_score": score["R_raw"],
            "tier": score["tier"],
            "runtime_seconds": runtime_seconds,
            "valid_for_model_comparison": valid_for_comparison,
        },
    )

    logger.info(
        "Run complete: %s (n=%d, R_raw=%.4f, tier=%s, valid=%s)",
        run_dir,
        len(predictions),
        score["R_raw"],
        score["tier"],
        valid_for_comparison,
    )
    return run_dir


def run_sweep(
    sweep_config_path: str | Path = "configs/experiments/smoke.yaml",
    base_config_path: str | Path = "configs/base.yaml",
    provider_preflight_enabled: bool | None = None,
    provider_preflight_min_success_rate: float | None = None,
    provider_preflight_abort_on_fail: bool | None = None,
    max_concurrent_runs: int | None = None,
    stage: str = "full",
    max_inflight_per_model: str | dict[str, Any] | None = None,
    time_budget_hours: float | None = None,
    qwen_burst_concurrency: int | None = None,
    qwen_degrade_concurrency: int | None = None,
    resume_mode: str | None = None,
    agentic_backend: str | None = None,
    example_parallelism_per_run: int | None = None,
    async_model_invocation: bool | None = None,
) -> Path:
    base_cfg = read_yaml(base_config_path)
    sweep_cfg = read_yaml(sweep_config_path)
    selected_stage = str(stage).strip().lower()
    if selected_stage not in {"calibration", "full", "canary"}:
        raise ValueError("stage must be one of: calibration, canary, full")
    logger.info("Running sweep stage=%s using config=%s", selected_stage, sweep_config_path)

    seeds = list(sweep_cfg.get("seeds", base_cfg["experiments"]["seeds"]))
    verification_configs = list(sweep_cfg.get("verification_configs", base_cfg["experiments"]["verification_configs"]))
    max_examples_cfg = sweep_cfg.get(
        "per_dataset_examples",
        sweep_cfg.get("max_examples_per_dataset", base_cfg["experiments"]["max_examples_per_dataset"]),
    )
    calibration_fit_cfg = sweep_cfg.get("calibration_fit", {})
    if not isinstance(calibration_fit_cfg, dict):
        calibration_fit_cfg = {}
    calibration_fit_enabled = bool(calibration_fit_cfg.get("enabled", False))
    calibration_fit_scope = str(calibration_fit_cfg.get("scope", "domain")).strip().lower()
    if calibration_fit_scope not in {"domain", "domain_config", "domain_dataset_config"}:
        raise ValueError("calibration_fit.scope must be one of: domain, domain_config, domain_dataset_config")
    sampling_mode = str(sweep_cfg.get("sampling_mode", "head")).strip().lower()

    if selected_stage == "calibration":
        seeds = list(sweep_cfg.get("calibration_seeds", [42]))
        verification_configs = list(sweep_cfg.get("calibration_verification_configs", ["none"]))
        default_per_domain = int(calibration_fit_cfg.get("num_examples_per_domain", 120))
        max_examples_cfg = sweep_cfg.get(
            "calibration_max_examples_per_dataset",
            max(1, int(math.ceil(default_per_domain / 2))),
        )
    elif selected_stage == "canary":
        seeds = list(sweep_cfg.get("canary_seeds", [42, 123]))
        verification_configs = list(sweep_cfg.get("canary_verification_configs", ["none", "full"]))
        max_examples_cfg = sweep_cfg.get("canary_max_examples_per_dataset", 10)
    if len(seeds) < 5:
        logger.warning(
            "Seed count is %d (<5). Statistical evidence claims should be treated as non-certifying.",
            len(seeds),
        )
    domains = list(sweep_cfg.get("domains", ["healthcare", "finance", "hr"]))
    dataset_profile = str(sweep_cfg.get("dataset_profile", "core"))
    benchmark_role = str(sweep_cfg.get("benchmark_role", _benchmark_role_for_profile(dataset_profile)))
    split_strategy = str(sweep_cfg.get("split_strategy", "preserve"))
    eval_split = sweep_cfg.get("eval_split")
    eval_split = str(eval_split) if eval_split else None
    if selected_stage == "calibration":
        eval_split = str(sweep_cfg.get("calibration_eval_split", "validation"))
    stress_test = bool(sweep_cfg.get("stress_test", False))
    disallow_toy_fallback = bool(sweep_cfg.get("disallow_toy_fallback", False))
    min_dataset_rows_cfg = sweep_cfg.get("min_dataset_rows", 0)
    model_paths = list(sweep_cfg.get("models", ["configs/models/mock.yaml"]))
    max_examples_default = int(base_cfg["experiments"]["max_examples_per_dataset"])
    default_backend = str(base_cfg.get("agentic_framework", {}).get("backend", "langgraph"))
    backend_selected = (
        str(agentic_backend).strip().lower()
        if agentic_backend is not None
        else str(sweep_cfg.get("agentic_backend", default_backend)).strip().lower()
    )
    if backend_selected not in {"langgraph", "legacy_python"}:
        raise ValueError("agentic_backend must be one of: langgraph, legacy_python")
    tool_cache_scope = str(base_cfg.get("agentic_framework", {}).get("tool_cache_scope", "example")).strip().lower()
    max_tool_iterations = int(base_cfg.get("agentic_framework", {}).get("max_tool_iterations", 64))
    example_parallelism = (
        int(example_parallelism_per_run)
        if example_parallelism_per_run is not None
        else int(
            sweep_cfg.get(
                "example_parallelism_per_run",
                base_cfg.get("experiments", {}).get("example_parallelism_per_run", 1),
            )
        )
    )
    if example_parallelism < 1:
        raise ValueError("example_parallelism_per_run must be >= 1")
    example_parallelism_overrides = sweep_cfg.get("example_parallelism_overrides", [])
    async_invocation = (
        bool(async_model_invocation)
        if async_model_invocation is not None
        else bool(sweep_cfg.get("async_model_invocation", False))
    )
    ece_bins = int(base_cfg.get("metrics", {}).get("ece_bins", 10))
    allow_generation_fallback_cfg = sweep_cfg.get("allow_generation_fallback")
    run_quality_guard_enabled = bool(sweep_cfg.get("run_quality_guard_enabled", True))
    fallback_guard_window = int(sweep_cfg.get("fallback_guard_window", 20))
    fallback_guard_max_rate = float(sweep_cfg.get("fallback_guard_max_rate", 0.05))
    example_error_guard_window = int(sweep_cfg.get("example_error_guard_window", 20))
    example_error_guard_max_rate = float(sweep_cfg.get("example_error_guard_max_rate", 0.05))
    max_consecutive_example_failures = int(sweep_cfg.get("max_consecutive_example_failures", 5))
    certification_stage_policy = str(sweep_cfg.get("certification_stage_policy", "full_only")).strip().lower()
    stats_gate_policy = str(sweep_cfg.get("stats_gate_policy", "constraint_aware")).strip().lower()
    stats_min_group_count = float(sweep_cfg.get("stats_min_group_count", 30.0))
    stratified_sampling_cfg = sweep_cfg.get("stratified_sampling", {})
    if not isinstance(stratified_sampling_cfg, dict):
        stratified_sampling_cfg = {}
    if certification_stage_policy not in {"full_only", "all"}:
        raise ValueError("certification_stage_policy must be one of: full_only, all")
    if stats_gate_policy not in {"constraint_aware", "universal"}:
        raise ValueError("stats_gate_policy must be one of: constraint_aware, universal")
    max_concurrent_runs_cfg = int(sweep_cfg.get("max_concurrent_runs", base_cfg["experiments"].get("max_concurrent_runs", 1)))
    if max_concurrent_runs is not None:
        max_concurrent_runs_cfg = int(max_concurrent_runs)
    if max_concurrent_runs_cfg < 1:
        raise ValueError("max_concurrent_runs must be >= 1")
    model_inflight_caps = _parse_model_inflight_caps(
        max_inflight_per_model if max_inflight_per_model is not None else sweep_cfg.get("max_inflight_per_model"),
        default_cap=max_concurrent_runs_cfg,
    )
    qwen_burst_cap = int(
        qwen_burst_concurrency
        if qwen_burst_concurrency is not None
        else sweep_cfg.get("qwen_burst_concurrency", 2)
    )
    qwen_degrade_cap = int(
        qwen_degrade_concurrency
        if qwen_degrade_concurrency is not None
        else sweep_cfg.get("qwen_degrade_concurrency", 1)
    )
    qwen_degrade_error_rate_threshold = float(sweep_cfg.get("qwen_degrade_error_rate_threshold", 0.03))
    qwen_degrade_window_calls = int(sweep_cfg.get("qwen_degrade_window_calls", 50))
    runtime_budget_cfg = sweep_cfg.get("runtime_budget", {})
    if not isinstance(runtime_budget_cfg, dict):
        runtime_budget_cfg = {}
    run_time_budget_hours = (
        float(time_budget_hours)
        if time_budget_hours is not None
        else float(runtime_budget_cfg.get("time_budget_hours", 12.0))
    )
    run_hard_cap_hours = float(runtime_budget_cfg.get("hard_cap_hours", 13.0))
    eta_interval_predictions = int(runtime_budget_cfg.get("eta_interval_predictions", 200))
    pause_low_priority_on_overrun = bool(runtime_budget_cfg.get("pause_low_priority_on_overrun", True))
    selected_resume_mode = (
        str(resume_mode).strip().lower()
        if resume_mode is not None
        else str(sweep_cfg.get("resume_mode", "missing_only")).strip().lower()
    )
    if selected_resume_mode not in {"missing_only", "fresh"}:
        raise ValueError("resume_mode must be one of: missing_only, fresh")

    preflight_enabled = bool(sweep_cfg.get("provider_preflight_enabled", True))
    if provider_preflight_enabled is not None:
        preflight_enabled = bool(provider_preflight_enabled)
    preflight_min_success_rate = float(sweep_cfg.get("provider_preflight_min_success_rate", 0.95))
    if provider_preflight_min_success_rate is not None:
        preflight_min_success_rate = float(provider_preflight_min_success_rate)
    preflight_abort_on_fail = bool(sweep_cfg.get("provider_preflight_abort_on_fail", True))
    if provider_preflight_abort_on_fail is not None:
        preflight_abort_on_fail = bool(provider_preflight_abort_on_fail)
    preflight_n_probes = int(
        sweep_cfg.get(
            "provider_preflight_n_probes",
            20 if selected_stage in {"canary", "calibration"} else 5,
        )
    )
    preflight_timeout = int(sweep_cfg.get("provider_preflight_timeout", 30))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    resume_root = sweep_cfg.get("resume_root")
    if selected_stage == "canary":
        default_root_name = f"{stamp}_canary"
    elif selected_stage == "calibration":
        default_root_name = f"{stamp}_calibration"
    else:
        default_root_name = stamp
    root = Path(resume_root) if resume_root else Path(base_cfg["runtime"]["runs_dir"]) / default_root_name
    root.mkdir(parents=True, exist_ok=True)

    sweep_name = str(sweep_cfg.get("name", Path(sweep_config_path).stem))
    calibration_artifacts_dir = Path(
        calibration_fit_cfg.get("artifacts_dir", Path("outputs") / "calibration_maps" / sweep_name)
    )
    calibration_min_improvement = float(calibration_fit_cfg.get("min_improvement", 0.005))

    domain_context: dict[str, dict[str, Any]] = {}
    for domain in domains:
        datasets = _ensure_processed_data(
            domain,
            profile=dataset_profile,
            split_strategy=split_strategy,
            disallow_toy_fallback=disallow_toy_fallback,
        )
        checks = [
            _validate_manifest_requirements(
                domain=domain,
                dataset=dataset,
                eval_split=eval_split,
                disallow_toy_fallback=disallow_toy_fallback,
                min_dataset_rows_cfg=min_dataset_rows_cfg,
            )
            for dataset in datasets
        ]
        data_validity_reasons: list[str] = []
        for chk in checks:
            if chk["source_type"] == "toy":
                data_validity_reasons.append(f"{chk['dataset']}:toy_source")
            if chk["source_type"] == "unknown":
                data_validity_reasons.append(f"{chk['dataset']}:unknown_source")
            if chk["required_rows"] > 0 and chk["available_rows"] < chk["required_rows"]:
                data_validity_reasons.append(
                    f"{chk['dataset']}:rows<{chk['required_rows']}"
                )
        full_examples_by_dataset: dict[str, list[dict[str, Any]]] = {}
        stage_max_examples_cfg = max_examples_cfg
        if selected_stage == "calibration" and isinstance(max_examples_cfg, int):
            per_domain_target = int(calibration_fit_cfg.get("num_examples_per_domain", max_examples_cfg))
            per_dataset_target = max(1, int(math.ceil(per_domain_target / max(1, len(datasets)))))
            stage_max_examples_cfg = {dataset: per_dataset_target for dataset in datasets}
        for dataset in datasets:
            examples = load_processed_dataset(domain=domain, dataset=dataset, split=eval_split)
            full_examples_by_dataset[dataset] = list(examples)
        domain_context[domain] = {
            "datasets": datasets,
            "spec": load_spec(Path("specs") / f"{domain}.yaml"),
            "checks": checks,
            "data_validity_reasons": data_validity_reasons,
            "full_examples_by_dataset": full_examples_by_dataset,
            "stage_max_examples_cfg": stage_max_examples_cfg,
        }
    datasets_by_domain = {
        str(domain): list(domain_context[domain]["datasets"])
        for domain in domains
    }

    seed_examples_cache: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    sampling_base_seed = int(base_cfg.get("runtime", {}).get("seed", 42))
    for domain in domains:
        datasets = list(domain_context[domain]["datasets"])
        full_examples_by_dataset = domain_context[domain]["full_examples_by_dataset"]
        stage_limit_cfg = domain_context[domain]["stage_max_examples_cfg"]
        for seed in seeds:
            sampled_by_dataset: dict[str, list[dict[str, Any]]] = {}
            for dataset in datasets:
                examples = list(full_examples_by_dataset.get(dataset, []))
                dataset_limit = _resolve_dataset_limit(
                    stage_limit_cfg, dataset=dataset, default=max_examples_default
                )
                sampled_by_dataset[dataset] = _sample_with_stratification(
                    dataset=dataset,
                    examples=examples,
                    dataset_limit=dataset_limit,
                    stratified_cfg=stratified_sampling_cfg.get(dataset)
                    if isinstance(stratified_sampling_cfg.get(dataset), dict)
                    else None,
                    seed=int(seed) + sampling_base_seed,
                    sampling_mode=sampling_mode,
                )
            seed_examples_cache[(domain, int(seed))] = sampled_by_dataset

    tasks: list[RunTask] = []
    expected_cell_dirs: list[Path] = []
    known_model_names: list[str] = []
    for model_cfg_path in model_paths:
        model_cfg = read_yaml(model_cfg_path)
        if allow_generation_fallback_cfg is not None:
            model_cfg["allow_generation_fallback"] = bool(allow_generation_fallback_cfg)
        model_cfg["async_model_invocation"] = bool(async_invocation)
        model_name = str(model_cfg.get("name", model_cfg.get("model", "model")))
        known_model_names.append(model_name)

        confidence_map_status_by_key: dict[str, dict[str, Any]] = {}
        if calibration_fit_enabled and selected_stage != "calibration":
            loaded_maps, loaded_paths = _load_calibration_maps_for_model(
                model_name=model_name,
                domains=domains,
                verification_configs=verification_configs,
                calibration_dir=calibration_artifacts_dir,
                scope=calibration_fit_scope,
                datasets_by_domain=datasets_by_domain,
            )
            for key, payload in loaded_maps.items():
                if not isinstance(payload, dict):
                    continue
                selected_method = str(payload.get("selected_method", payload.get("method", "identity"))).strip()
                eligible, reason = _calibration_map_is_eligible(
                    payload=payload,
                    min_improvement=calibration_min_improvement,
                )
                confidence_map_status_by_key[key] = {
                    "payload": payload,
                    "path": loaded_paths.get(key),
                    "selected_method": selected_method or None,
                    "method": (
                        str(payload.get("method")).strip()
                        if payload.get("method") is not None
                        else None
                    ),
                    "applied": bool(eligible),
                    "reason": reason,
                }

        provider = build_provider(model_cfg)
        scheduler_class = str(model_cfg.get("scheduler_class", "")).strip().lower()
        if model_name not in model_inflight_caps:
            if scheduler_class == "slow":
                model_inflight_caps[model_name] = min(1, max_concurrent_runs_cfg)
            elif scheduler_class == "fast":
                model_inflight_caps[model_name] = min(2, max_concurrent_runs_cfg)
        if "qwen" in model_name.lower():
            model_inflight_caps[model_name] = max(
                1,
                min(max_concurrent_runs_cfg, int(qwen_burst_cap)),
            )

        preflight_payload: dict[str, Any] = {
            "success_rate": 1.0,
            "n_probes": 0,
            "error_taxonomy_counts": {},
        }
        if preflight_enabled:
            preflight_payload = provider_healthcheck(
                provider,
                n_probes=preflight_n_probes,
                timeout=preflight_timeout,
            )
            preflight_path = root / "preflight" / f"{model_name}.json"
            write_json(preflight_path, preflight_payload)
            if float(preflight_payload.get("success_rate", 0.0)) < preflight_min_success_rate:
                message = (
                    f"Provider preflight failed for model={model_name}: "
                    f"success_rate={preflight_payload.get('success_rate')} < {preflight_min_success_rate}"
                )
                if preflight_abort_on_fail:
                    raise RuntimeError(message)
                logger.error("%s. Skipping model.", message)
                continue

        for domain in domains:
            datasets = list(domain_context[domain]["datasets"])
            spec = domain_context[domain]["spec"]
            data_validity_reasons = list(domain_context[domain]["data_validity_reasons"])
            data_valid_for_certification = len(data_validity_reasons) == 0
            task_example_parallelism = _resolve_example_parallelism_for_task(
                base_parallelism=example_parallelism,
                overrides_cfg=example_parallelism_overrides,
                model_name=model_name,
                domain=domain,
                datasets=datasets,
            )

            for cfg_name in verification_configs:
                domain_key = _calibration_map_key(domain)
                domain_status = (
                    confidence_map_status_by_key.get(domain_key)
                    if isinstance(confidence_map_status_by_key.get(domain_key), dict)
                    else None
                )
                confidence_map = None
                confidence_map_applied: Any = False
                confidence_map_selected: Any = None
                confidence_map_selection_reason: Any = "map_not_found"
                confidence_map_method: Any = None
                confidence_map_path: Any = None
                task_model_cfg = dict(model_cfg)

                if calibration_fit_scope == "domain_dataset_config":
                    selected_maps_by_dataset: dict[str, dict[str, Any]] = {}
                    selected_flags_by_dataset: dict[str, bool] = {}
                    selected_methods_by_dataset: dict[str, str | None] = {}
                    selected_paths_by_dataset: dict[str, str | None] = {}
                    selected_reasons_by_dataset: dict[str, str] = {}
                    for dataset in datasets:
                        dataset_candidates = [
                            _calibration_map_key(domain, cfg_name, dataset),
                            _calibration_map_key(domain, cfg_name),
                            _calibration_map_key(domain),
                        ]
                        chosen_status: dict[str, Any] | None = None
                        first_seen_status: dict[str, Any] | None = None
                        for candidate_key in dataset_candidates:
                            status = confidence_map_status_by_key.get(candidate_key)
                            if not isinstance(status, dict):
                                continue
                            if first_seen_status is None:
                                first_seen_status = status
                            if bool(status.get("applied", False)):
                                chosen_status = status
                                break
                        effective_status = chosen_status or first_seen_status or {}
                        payload = (
                            effective_status.get("payload")
                            if isinstance(effective_status.get("payload"), dict)
                            else None
                        )
                        if isinstance(payload, dict):
                            selected_maps_by_dataset[str(dataset)] = payload
                        selected_flags_by_dataset[str(dataset)] = bool(
                            chosen_status is not None and isinstance(payload, dict)
                        )
                        selected_methods_by_dataset[str(dataset)] = (
                            str(effective_status.get("selected_method"))
                            if effective_status.get("selected_method") is not None
                            else None
                        )
                        selected_paths_by_dataset[str(dataset)] = (
                            str(effective_status.get("path"))
                            if effective_status.get("path") is not None
                            else None
                        )
                        selected_reasons_by_dataset[str(dataset)] = (
                            str(effective_status.get("reason"))
                            if effective_status.get("reason") is not None
                            else "map_not_found"
                        )
                    confidence_map = {
                        "_scope": "domain_dataset_config",
                        "_by_dataset": selected_maps_by_dataset,
                    }
                    if isinstance(domain_status, dict) and isinstance(domain_status.get("payload"), dict):
                        confidence_map[domain] = domain_status["payload"]
                    if selected_maps_by_dataset or domain in confidence_map:
                        task_model_cfg["confidence_calibration_maps"] = confidence_map
                    else:
                        task_model_cfg.pop("confidence_calibration_maps", None)
                    confidence_map_applied = selected_flags_by_dataset
                    confidence_map_selected = selected_methods_by_dataset
                    confidence_map_selection_reason = selected_reasons_by_dataset
                    confidence_map_method = dict(selected_methods_by_dataset)
                    confidence_map_path = selected_paths_by_dataset
                else:
                    map_candidates = (
                        [_calibration_map_key(domain, cfg_name), _calibration_map_key(domain)]
                        if calibration_fit_scope == "domain_config"
                        else [_calibration_map_key(domain)]
                    )
                    chosen_status: dict[str, Any] | None = None
                    first_seen_status: dict[str, Any] | None = None
                    for candidate_key in map_candidates:
                        status = confidence_map_status_by_key.get(candidate_key)
                        if not isinstance(status, dict):
                            continue
                        if first_seen_status is None:
                            first_seen_status = status
                        if bool(status.get("applied", False)):
                            chosen_status = status
                            break
                    effective_status = chosen_status or first_seen_status or {}
                    confidence_map = (
                        effective_status.get("payload")
                        if isinstance(effective_status.get("payload"), dict)
                        else None
                    )
                    confidence_map_applied = bool(chosen_status is not None and isinstance(confidence_map, dict))
                    confidence_map_selected = (
                        str(effective_status.get("selected_method"))
                        if effective_status.get("selected_method") is not None
                        else None
                    )
                    confidence_map_selection_reason = (
                        str(effective_status.get("reason"))
                        if effective_status.get("reason") is not None
                        else "map_not_found"
                    )
                    confidence_map_method = (
                        str(effective_status.get("method"))
                        if effective_status.get("method") is not None
                        else None
                    )
                    confidence_map_path = (
                        str(effective_status.get("path"))
                        if effective_status.get("path") is not None
                        else None
                    )
                    if confidence_map_applied and isinstance(confidence_map, dict):
                        task_model_cfg["confidence_calibration_maps"] = {domain: confidence_map}
                    else:
                        task_model_cfg.pop("confidence_calibration_maps", None)
                for seed in seeds:
                    run_dir = root / domain / model_name / cfg_name / str(seed)
                    expected_cell_dirs.append(run_dir)
                    if selected_resume_mode == "missing_only" and (run_dir / "metrics.json").exists():
                        logger.info("Skipping completed run: %s", run_dir)
                        continue
                    examples_by_dataset = {
                        k: list(v)
                        for k, v in seed_examples_cache.get((domain, int(seed)), {}).items()
                    }
                    certification_eligible = True
                    certification_eligibility_reason: str | None = None
                    if certification_stage_policy == "full_only" and selected_stage != "full":
                        certification_eligible = False
                        certification_eligibility_reason = "stage_non_certifying"
                    tasks.append(
                        RunTask(
                            stamp=stamp,
                            root=root,
                            domain=domain,
                            model_name=model_name,
                            model_cfg=dict(task_model_cfg),
                            verification_config=cfg_name,
                            seed=int(seed),
                            eval_split=eval_split,
                            datasets=datasets,
                            examples_by_dataset=examples_by_dataset,
                            spec=spec,
                            data_validity_reasons=data_validity_reasons,
                            data_valid_for_certification=data_valid_for_certification,
                            preflight_payload=preflight_payload,
                            preflight_enabled=preflight_enabled,
                            preflight_min_success_rate=preflight_min_success_rate,
                            ece_bins=ece_bins,
                            run_quality_guard_enabled=run_quality_guard_enabled,
                            fallback_guard_window=fallback_guard_window,
                            fallback_guard_max_rate=fallback_guard_max_rate,
                            example_error_guard_window=example_error_guard_window,
                            example_error_guard_max_rate=example_error_guard_max_rate,
                            max_consecutive_example_failures=max_consecutive_example_failures,
                            stress_test=stress_test,
                            seed_globally=max_concurrent_runs_cfg == 1,
                            sweep_stage=selected_stage,
                            stats_gate_policy=stats_gate_policy,
                            stats_min_group_count=stats_min_group_count,
                            certification_eligible=certification_eligible,
                            certification_eligibility_reason=certification_eligibility_reason,
                            confidence_map_method=confidence_map_method,
                            confidence_map_path=confidence_map_path,
                            calibration_map_scope=calibration_fit_scope,
                            confidence_map_applied=confidence_map_applied,
                            confidence_map_selected=confidence_map_selected,
                            confidence_map_selection_reason=confidence_map_selection_reason,
                            skip_if_completed=selected_resume_mode == "missing_only",
                            agentic_backend=backend_selected,
                            tool_cache_scope=tool_cache_scope,
                            max_tool_iterations=max_tool_iterations,
                            example_parallelism_per_run=task_example_parallelism,
                            dataset_profile=dataset_profile,
                            benchmark_role=benchmark_role,
                        )
                    )

    total_planned_predictions = sum(
        len(examples)
        for task in tasks
        for examples in task.examples_by_dataset.values()
    )
    logger.info(
        "Prepared %d run tasks (%d planned predictions) for stage=%s",
        len(tasks),
        total_planned_predictions,
        selected_stage,
    )

    started_wall = time.time()
    completed_predictions = 0.0
    last_eta_check = 0.0
    qwen_window_by_model: dict[str, deque[tuple[int, int]]] = {}
    qwen_calls_by_model: dict[str, int] = {}
    qwen_errors_by_model: dict[str, int] = {}

    def _update_qwen_window(task: RunTask, metrics: dict[str, Any]) -> None:
        name = str(task.model_name)
        if "qwen" not in name.lower():
            return
        calls = int(float(metrics.get("model_generation_calls", 0.0)))
        errors = int(float(metrics.get("generation_error_count", 0.0)))
        window = qwen_window_by_model.setdefault(name, deque())
        window.append((calls, errors))
        qwen_calls_by_model[name] = qwen_calls_by_model.get(name, 0) + calls
        qwen_errors_by_model[name] = qwen_errors_by_model.get(name, 0) + errors
        while qwen_calls_by_model[name] > qwen_degrade_window_calls and window:
            oldest_calls, oldest_errors = window.popleft()
            qwen_calls_by_model[name] -= oldest_calls
            qwen_errors_by_model[name] -= oldest_errors
        calls_window = max(0, qwen_calls_by_model.get(name, 0))
        if calls_window < max(1, qwen_degrade_window_calls):
            return
        error_rate = qwen_errors_by_model.get(name, 0) / float(max(1, calls_window))
        current_cap = int(model_inflight_caps.get(name, max_concurrent_runs_cfg))
        if error_rate > qwen_degrade_error_rate_threshold and current_cap > qwen_degrade_cap:
            model_inflight_caps[name] = max(1, min(max_concurrent_runs_cfg, qwen_degrade_cap))
            logger.warning(
                "Degraded inflight cap for %s to %d (rolling error_rate=%.4f > %.4f)",
                name,
                model_inflight_caps[name],
                error_rate,
                qwen_degrade_error_rate_threshold,
            )

    def _maybe_emit_eta(pending_count: int) -> None:
        nonlocal last_eta_check
        if eta_interval_predictions <= 0:
            return
        if (completed_predictions - last_eta_check) < eta_interval_predictions:
            return
        last_eta_check = completed_predictions
        elapsed = max(1.0, time.time() - started_wall)
        rate = completed_predictions / elapsed
        remaining = max(0.0, float(total_planned_predictions) - completed_predictions)
        eta_seconds = remaining / rate if rate > 0 else float("inf")
        projected_hours = (elapsed + eta_seconds) / 3600.0 if eta_seconds != float("inf") else float("inf")
        logger.info(
            "ETA check: completed=%.0f/%d pending_runs=%d projected_total_hours=%.2f budget=%.2f hard_cap=%.2f",
            completed_predictions,
            total_planned_predictions,
            pending_count,
            projected_hours,
            run_time_budget_hours,
            run_hard_cap_hours,
        )

    if max_concurrent_runs_cfg == 1:
        for task in tasks:
            result_dir = _execute_run_task(task)
            metrics_path = result_dir / "metrics.json"
            if metrics_path.exists():
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                completed_predictions += float(payload.get("num_predictions", 0.0))
                _update_qwen_window(task, payload)
            _maybe_emit_eta(0)
    else:
        logger.info(
            "Parallel sweep enabled: %d run tasks, max_concurrent_runs=%d",
            len(tasks),
            max_concurrent_runs_cfg,
        )
        logger.warning("Global RNG seeding is disabled in parallel mode to avoid cross-thread interference.")
        pending = deque(tasks)
        inflight_by_model: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=max_concurrent_runs_cfg, thread_name_prefix="rava-run") as executor:
            running: dict[Any, RunTask] = {}
            while pending or running:
                submitted = False
                while len(running) < max_concurrent_runs_cfg and pending:
                    selected_index: int | None = None
                    for idx, candidate in enumerate(pending):
                        cap = int(model_inflight_caps.get(candidate.model_name, max_concurrent_runs_cfg))
                        cur = inflight_by_model.get(candidate.model_name, 0)
                        if cur < cap:
                            selected_index = idx
                            break
                    if selected_index is None:
                        break

                    task = pending[selected_index]
                    del pending[selected_index]
                    future = executor.submit(_execute_run_task, task)
                    running[future] = task
                    inflight_by_model[task.model_name] = inflight_by_model.get(task.model_name, 0) + 1
                    submitted = True

                if not running:
                    if pending and not submitted:
                        blocked_models = sorted({t.model_name for t in pending})
                        raise RuntimeError(
                            "Unable to schedule pending tasks due to max_inflight_per_model caps: "
                            + ",".join(blocked_models)
                        )
                    break

                done, _ = wait(set(running.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    task = running.pop(future)
                    inflight_by_model[task.model_name] = max(0, inflight_by_model.get(task.model_name, 1) - 1)
                    try:
                        result_dir = future.result()
                    except Exception:
                        logger.exception(
                            "Run failed for domain=%s model=%s config=%s seed=%d",
                            task.domain,
                            task.model_name,
                            task.verification_config,
                            int(task.seed),
                        )
                        raise

                    metrics_path = result_dir / "metrics.json"
                    if metrics_path.exists():
                        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                        completed_predictions += float(payload.get("num_predictions", 0.0))
                        _update_qwen_window(task, payload)
                    _maybe_emit_eta(len(pending))

                    elapsed_hours = (time.time() - started_wall) / 3600.0
                    if elapsed_hours > run_hard_cap_hours:
                        logger.warning("Hard runtime cap reached (%.2fh > %.2fh).", elapsed_hours, run_hard_cap_hours)
                        if pause_low_priority_on_overrun and pending:
                            pruned = [t for t in list(pending) if t.verification_config in {"posthoc", "runtime"}]
                            if pruned:
                                pending = deque([t for t in list(pending) if t.verification_config not in {"posthoc", "runtime"}])
                                logger.warning(
                                    "Paused %d low-priority pending tasks to control runtime overrun.",
                                    len(pruned),
                                )
                        for model_name in list(model_inflight_caps):
                            if "qwen" in model_name.lower():
                                model_inflight_caps[model_name] = max(
                                    1, min(max_concurrent_runs_cfg, qwen_degrade_cap)
                                )

    if selected_stage == "calibration" and calibration_fit_enabled:
        calibration_methods = tuple(
            str(item).strip().lower()
            for item in calibration_fit_cfg.get("methods", ["isotonic", "platt", "identity"])
            if str(item).strip()
        )
        fit_artifacts = _fit_and_persist_calibration_maps(
            root=root,
            domains=domains,
            datasets_by_domain=datasets_by_domain,
            model_names=known_model_names,
            verification_configs=verification_configs,
            calibration_dir=calibration_artifacts_dir,
            scope=calibration_fit_scope,
            methods=calibration_methods if calibration_methods else ("isotonic", "platt", "identity"),
            target_ece=float(calibration_fit_cfg.get("target_ece", 0.12)),
            ece_bins=ece_bins,
            min_samples=int(calibration_fit_cfg.get("min_samples", 30)),
            min_improvement=calibration_min_improvement,
        )
        write_json(root / "calibration_fit_summary.json", fit_artifacts)

    completed_cells = [
        str(path.relative_to(root))
        for path in expected_cell_dirs
        if (path / "metrics.json").exists()
    ]
    missing_cells = [
        str(path.relative_to(root))
        for path in expected_cell_dirs
        if not (path / "metrics.json").exists()
    ]
    matrix_manifest = {
        "sweep_stage": selected_stage,
        "resume_mode": selected_resume_mode,
        "expected_cells": int(len(expected_cell_dirs)),
        "completed_cells": int(len(completed_cells)),
        "missing_cells": int(len(missing_cells)),
        "completion_rate": (
            float(len(completed_cells)) / float(len(expected_cell_dirs))
            if expected_cell_dirs
            else 0.0
        ),
        "missing_cell_paths": missing_cells,
    }
    write_json(root / "matrix_completion.json", matrix_manifest)
    matrix_stub = {
        "expected": matrix_manifest["expected_cells"],
        "completed": matrix_manifest["completed_cells"],
        "missing": matrix_manifest["missing_cells"],
        "completion_rate": matrix_manifest["completion_rate"],
    }
    for rel_path in completed_cells:
        run_dir = root / rel_path
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                payload["matrix_completion"] = matrix_stub
                write_json(metrics_path, payload)
            except Exception:
                logger.exception("Failed updating matrix completion in %s", metrics_path)
        report_path = run_dir / "report.json"
        if report_path.exists():
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload["matrix_completion"] = matrix_stub
                write_json(report_path, payload)
            except Exception:
                logger.exception("Failed updating matrix completion in %s", report_path)

    return root
