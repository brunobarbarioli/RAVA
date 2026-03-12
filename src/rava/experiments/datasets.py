from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import re
import urllib.error
import urllib.request
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from rava.utils.serialization import read_jsonl, write_json, write_jsonl

logger = logging.getLogger(__name__)


DATASET_CATALOG: dict[str, dict[str, Any]] = {
    # Healthcare
    "pubmedqa": {
        "domain": "healthcare",
        "hf_id": "pubmed_qa",
        "official": "https://pubmedqa.github.io/",
        "license": "Check PubMedQA terms and usage restrictions before redistribution.",
    },
    "medqa": {
        "domain": "healthcare",
        "hf_id": None,
        "official": "https://github.com/jind11/MedQA",
        "license": "MedQA distributions vary. Use only legally obtained files.",
        "byo_required": True,
    },
    "medhalt": {
        "domain": "healthcare",
        "hf_id": None,
        "official": "https://github.com/allenai/medhalt",
        "license": "MedHalt may require acceptance of source terms.",
        "byo_required": True,
    },
    "pubhealth": {
        "domain": "healthcare",
        "hf_id": None,
        "official": "https://github.com/neulab/PUBHEALTH",
        "license": "PubHealth license and source data terms apply.",
        "byo_required": True,
    },
    "ehrsql": {
        "domain": "healthcare",
        "hf_id": None,
        "official": "https://github.com/glee4810/EHRSQL",
        "license": "EHRSQL terms and any source EHR data restrictions apply.",
        "byo_required": True,
    },
    "mimic_iv_bhc": {
        "domain": "healthcare",
        "hf_id": None,
        "official": "https://physionet.org/content/mimiciv/",
        "license": "MIMIC-IV requires credentialed access and strict data use compliance.",
        "byo_required": True,
    },
    # Finance
    "finben": {
        "domain": "finance",
        "hf_id": None,
        "official": "https://github.com/The-FinAI/Benckmarking-Large-Language-Models-on-Financial-Tasks",
        "license": "FinBen tasks are sourced from multiple datasets with varying licenses.",
        "byo_required": True,
    },
    "flue": {
        "domain": "finance",
        "hf_id": None,
        "official": "https://arxiv.org/abs/2202.12005",
        "license": "FLUE benchmark components have task-specific licenses.",
        "byo_required": True,
    },
    "convfinqa": {
        "domain": "finance",
        "hf_id": "ibm/convfinqa",
        "official": "https://github.com/czyssrs/ConvFinQA",
        "license": "ConvFinQA license must be respected for redistribution.",
    },
    "financebench": {
        "domain": "finance",
        "hf_id": None,
        "official": "https://huggingface.co/datasets/PatronusAI/financebench",
        "license": "FinanceBench terms apply.",
        "byo_required": True,
    },
    "tat_qa": {
        "domain": "finance",
        "hf_id": None,
        "official": "https://nextplusplus.github.io/TAT-QA/",
        "license": "TAT-QA dataset terms apply.",
        "byo_required": True,
    },
    "finqa": {
        "domain": "finance",
        "hf_id": None,
        "official": "https://github.com/czyssrs/FinQA",
        "license": "FinQA terms apply.",
        "byo_required": True,
    },
    # HR
    "bbq": {
        "domain": "hr",
        "hf_id": "heegyu/bbq",
        "official": "https://github.com/nyu-mll/BBQ",
        "license": "BBQ benchmark license and ethics statement apply.",
    },
    "winobias": {
        "domain": "hr",
        "hf_id": None,
        "official": "https://uclanlp.github.io/corefBias/overview",
        "license": "Use WinoBias under its stated research terms.",
        "byo_required": True,
    },
    "jigsaw_unintended_bias": {
        "domain": "hr",
        "hf_id": None,
        "official": "https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification",
        "license": "Kaggle competition terms apply. Requires Kaggle credentials.",
        "kaggle": True,
    },
    "synthetic_resumes": {
        "domain": "hr",
        "hf_id": None,
        "official": "Generated locally by this repository.",
        "license": "Synthetic data generated from templates.",
    },
    "bias_in_bios": {
        "domain": "hr",
        "hf_id": "LabHC/bias_in_bios",
        "official": "https://github.com/microsoft/biosbias",
        "license": "Bias in Bios terms apply.",
    },
    "agentic_stress_hr": {
        "domain": "hr",
        "hf_id": None,
        "official": "Generated locally by deterministic two-agent templates.",
        "license": "Synthetic data generated from templates.",
    },
    "fairjob": {
        "domain": "hr",
        "hf_id": None,
        "official": "https://github.com/microsoft/responsible-ai-toolbox-mitigations",
        "license": "FairJob benchmark terms apply.",
        "byo_required": True,
    },
    "acs_pums_hr": {
        "domain": "hr",
        "hf_id": None,
        "official": "https://www.census.gov/programs-surveys/acs/microdata.html",
        "license": "ACS PUMS public use terms apply.",
        "byo_required": True,
    },
}

PROFILE_SOURCE_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "paper_hybrid": {
        "medqa": {"hf_id": "openlifescienceai/MedQA-USMLE-4-options-hf"},
        "pubmedqa": {"hf_id": "qiaojin/PubMedQA", "hf_config": "pqa_labeled"},
        "medhalt": {"hf_id": "openlifescienceai/Med-HALT", "hf_config": "reasoning_FCT"},
        "finben": {"hf_id": "yuweiyin/FinBench"},
        "flue": {"hf_id": "SALT-NLP/FLUE-FiQA"},
        "convfinqa": {"hf_id": "AdaptLLM/ConvFinQA"},
        "bbq": {"hf_id": "heegyu/bbq"},
        "winobias": {"hf_id": "uclanlp/wino_bias"},
        # `google/jigsaw_unintended_bias` is script-only on HF (manual Kaggle files required).
        # Use a parquet mirror to keep paper_hybrid fully non-Kaggle and reproducible.
        "jigsaw_unintended_bias": {"hf_id": "TheMrguiller/jigsaw-unintended-bias-in-toxicity-classification", "kaggle": False},
    },
    "paper6_fast": {
        "medqa": {"hf_id": "openlifescienceai/MedQA-USMLE-4-options-hf"},
        "pubmedqa": {"hf_id": "qiaojin/PubMedQA", "hf_config": "pqa_labeled"},
        "finben": {"hf_id": "yuweiyin/FinBench"},
        "convfinqa": {"hf_id": "AdaptLLM/ConvFinQA"},
        "winobias": {"hf_id": "uclanlp/wino_bias"},
        "bias_in_bios": {"hf_id": "LabHC/bias_in_bios"},
    },
    "paper3_mini": {
        "pubmedqa": {"hf_id": "qiaojin/PubMedQA", "hf_config": "pqa_labeled"},
        "convfinqa": {"hf_id": "AdaptLLM/ConvFinQA"},
        "bias_in_bios": {"hf_id": "LabHC/bias_in_bios"},
    },
    "primary_certification": {
        "pubmedqa": {"hf_id": "qiaojin/PubMedQA", "hf_config": "pqa_labeled"},
        "convfinqa": {"hf_id": "AdaptLLM/ConvFinQA"},
        "bias_in_bios": {"hf_id": "LabHC/bias_in_bios"},
    },
    "final_a6": {
        "pubmedqa": {"hf_id": "qiaojin/PubMedQA", "hf_config": "pqa_labeled"},
        "medqa": {"hf_id": "openlifescienceai/MedQA-USMLE-4-options-hf"},
        "convfinqa": {"hf_id": "AdaptLLM/ConvFinQA"},
        "finben": {"hf_id": "yuweiyin/FinBench"},
        "bias_in_bios": {"hf_id": "LabHC/bias_in_bios"},
        "winobias": {"hf_id": "uclanlp/wino_bias"},
    },
    "diagnostic_secondary": {
        "medqa": {"hf_id": "openlifescienceai/MedQA-USMLE-4-options-hf"},
        "finben": {"hf_id": "yuweiyin/FinBench"},
        "winobias": {"hf_id": "uclanlp/wino_bias"},
    },
}


DOMAIN_DATASET_PROFILES: dict[str, dict[str, list[str]]] = {
    "core": {
        "healthcare": ["medqa", "pubmedqa", "medhalt"],
        "finance": ["finben", "flue", "convfinqa"],
        "hr": ["bbq", "winobias", "jigsaw_unintended_bias", "synthetic_resumes"],
    },
    "enhanced": {
        "healthcare": ["medqa", "pubmedqa", "medhalt", "pubhealth", "ehrsql", "mimic_iv_bhc"],
        "finance": ["finben", "flue", "convfinqa", "financebench", "tat_qa", "finqa"],
        "hr": ["bbq", "winobias", "jigsaw_unintended_bias", "synthetic_resumes", "bias_in_bios", "fairjob", "acs_pums_hr"],
    },
    "paper_hybrid": {
        "healthcare": ["medqa", "pubmedqa", "medhalt"],
        "finance": ["finben", "flue", "convfinqa"],
        "hr": ["bbq", "winobias", "jigsaw_unintended_bias", "synthetic_resumes"],
    },
    "paper6_fast": {
        "healthcare": ["medqa", "pubmedqa"],
        "finance": ["finben", "convfinqa"],
        "hr": ["winobias", "bias_in_bios"],
    },
    "paper3_mini": {
        "healthcare": ["pubmedqa"],
        "finance": ["convfinqa"],
        "hr": ["bias_in_bios"],
    },
    "primary_certification": {
        "healthcare": ["pubmedqa"],
        "finance": ["convfinqa"],
        "hr": ["bias_in_bios"],
    },
    "final_a6": {
        "healthcare": ["pubmedqa", "medqa"],
        "finance": ["convfinqa", "finben"],
        "hr": ["bias_in_bios", "winobias"],
    },
    "diagnostic_secondary": {
        "healthcare": ["medqa"],
        "finance": ["finben"],
        "hr": ["winobias"],
    },
    "supplemental_stress": {
        "healthcare": [],
        "finance": [],
        "hr": ["agentic_stress_hr"],
    },
}
# Backward compatible alias used by existing call sites.
DOMAIN_TO_DATASETS = DOMAIN_DATASET_PROFILES["core"]


def get_datasets_for_domain(domain: str, profile: str = "core") -> list[str]:
    profile_map = DOMAIN_DATASET_PROFILES.get(profile, DOMAIN_DATASET_PROFILES["core"])
    return list(profile_map.get(domain, []))


def _raw_dir(dataset: str, root: str | Path = "data/raw") -> Path:
    return Path(root) / dataset


def _processed_dir(domain: str, dataset: str, root: str | Path = "data/processed") -> Path:
    return Path(root) / domain / dataset


def _source_metadata_path(raw_dir: Path) -> Path:
    return raw_dir / "_source_metadata.json"


def _load_source_metadata(raw_dir: Path) -> dict[str, Any]:
    meta_path = _source_metadata_path(raw_dir)
    if not meta_path.exists():
        return {"source_type": "byo"}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {"source_type": "byo"}


def _write_source_metadata(raw_dir: Path, payload: dict[str, Any]) -> None:
    path = _source_metadata_path(raw_dir)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_dataset_source(dataset: str, profile: str) -> dict[str, Any]:
    info = dict(DATASET_CATALOG[dataset])
    override = PROFILE_SOURCE_OVERRIDES.get(profile, {}).get(dataset, {})
    if override:
        info.update(override)
    return info


def _fetch_hf_dataset_metadata(hf_id: str) -> dict[str, Any]:
    url = f"https://huggingface.co/api/datasets/{hf_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:  # nosec - static trusted endpoint
            payload = json.loads(response.read().decode("utf-8"))
            return {
                "resolved_id": payload.get("id", hf_id),
                "revision": payload.get("sha"),
            }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return {"resolved_id": hf_id, "revision": None}


def _write_license_notice(dataset: str, raw_dir: Path) -> None:
    info = DATASET_CATALOG[dataset]
    notice = (
        f"Dataset: {dataset}\n"
        f"Official source: {info.get('official')}\n"
        f"License/terms note: {info.get('license')}\n"
        "You are responsible for ensuring compliance with the dataset license/terms.\n"
    )
    (raw_dir / "LICENSE_NOTICE.txt").write_text(notice, encoding="utf-8")


def sha256sum(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_integrity(path: str | Path, expected_sha256: str | None = None) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    if expected_sha256 is None:
        return True
    return sha256sum(p) == expected_sha256


def _write_byo_instructions(dataset: str, raw_dir: Path, expected_files: list[str]) -> None:
    info = DATASET_CATALOG[dataset]
    message = [
        f"Dataset '{dataset}' requires bring-your-own files.",
        f"Official source: {info.get('official')}",
        "Place one of the following files in this directory:",
    ]
    message.extend([f"- {name}" for name in expected_files])
    message.append("Supported formats: .jsonl, .json, .csv")
    message.append("After placing files, run: rava preprocess-datasets")
    (raw_dir / "BYO_INSTRUCTIONS.md").write_text("\n".join(message), encoding="utf-8")


def _download_hf(
    dataset: str,
    hf_id: str,
    raw_dir: Path,
    hf_config: str | None = None,
) -> dict[str, Any]:
    hf_meta = _fetch_hf_dataset_metadata(hf_id)
    resolved_id = str(hf_meta.get("resolved_id", hf_id))
    revision = hf_meta.get("revision")
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        logger.warning("datasets package unavailable for %s: %s", dataset, exc)
        _write_byo_instructions(dataset, raw_dir, ["train.jsonl", "validation.jsonl", "test.jsonl"])
        return {
            "source_type": "byo",
            "hf_dataset_id": resolved_id,
            "hf_revision": revision,
            "download_status": "datasets_package_missing",
        }

    def _snapshot_fallback(error: Exception) -> dict[str, Any]:
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except Exception:
            return {
                "source_type": "byo",
                "hf_dataset_id": resolved_id,
                "hf_revision": revision,
                "hf_config": hf_config,
                "download_status": "hf_snapshot_unavailable",
                "download_error": str(error),
            }
        allow_patterns = [
            "*.jsonl",
            "*.json",
            "*.csv",
            "*.parquet",
            "**/*.jsonl",
            "**/*.json",
            "**/*.csv",
            "**/*.parquet",
        ]
        if hf_config:
            allow_patterns.extend(
                [
                    f"{hf_config}/*.jsonl",
                    f"{hf_config}/*.json",
                    f"{hf_config}/*.csv",
                    f"{hf_config}/*.parquet",
                    f"{hf_config}/**/*.jsonl",
                    f"{hf_config}/**/*.json",
                    f"{hf_config}/**/*.csv",
                    f"{hf_config}/**/*.parquet",
                ]
            )
        try:
            snapshot_download(
                repo_id=resolved_id,
                repo_type="dataset",
                local_dir=str(raw_dir),
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
            )
            data_files = [
                p
                for p in raw_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in {".jsonl", ".json", ".csv", ".parquet"}
            ]
            if not data_files:
                return {
                    "source_type": "byo",
                    "hf_dataset_id": resolved_id,
                    "hf_revision": revision,
                    "hf_config": hf_config,
                    "download_status": "hf_snapshot_no_data_files",
                    "download_error": str(error),
                }
            return {
                "source_type": "hf",
                "hf_dataset_id": resolved_id,
                "hf_revision": revision,
                "hf_config": hf_config,
                "download_status": "ok_snapshot",
            }
        except Exception as snapshot_exc:
            return {
                "source_type": "byo",
                "hf_dataset_id": resolved_id,
                "hf_revision": revision,
                "hf_config": hf_config,
                "download_status": "hf_snapshot_failed",
                "download_error": str(snapshot_exc),
            }

    try:
        if hf_config:
            ds = load_dataset(resolved_id, hf_config)
        else:
            ds = load_dataset(resolved_id)
    except Exception as exc:
        meta = _snapshot_fallback(exc)
        if meta.get("source_type") == "hf":
            return meta
        logger.warning("HF download failed for %s (%s): %s", dataset, resolved_id, exc)
        _write_byo_instructions(dataset, raw_dir, ["train.jsonl", "validation.jsonl", "test.jsonl"])
        return {
            "source_type": "byo",
            "hf_dataset_id": resolved_id,
            "hf_revision": revision,
            "hf_config": hf_config,
            "download_status": "hf_download_failed",
            "download_error": str(exc),
        }

    split_iter = ds.items() if hasattr(ds, "items") else [("train", ds)]
    for split_name, split_data in split_iter:
        rows = []
        for row in tqdm(split_data, desc=f"{dataset}:{split_name}", leave=False):
            payload = dict(row)
            payload["__split"] = split_name
            rows.append(payload)
        write_jsonl(raw_dir / f"{split_name}.jsonl", rows)
    return {
        "source_type": "hf",
        "hf_dataset_id": resolved_id,
        "hf_revision": revision,
        "hf_config": hf_config,
        "download_status": "ok",
    }


def _download_kaggle_jigsaw(raw_dir: Path) -> dict[str, Any]:
    creds_set = (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) or Path.home().joinpath(".kaggle", "kaggle.json").exists()
    if not creds_set:
        raise RuntimeError(
            "Kaggle credentials missing. Set KAGGLE_USERNAME/KAGGLE_KEY or place ~/.kaggle/kaggle.json, then rerun."
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "kaggle package not installed. Install with `pip install kaggle` and retry."
        ) from exc

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(
        competition="jigsaw-unintended-bias-in-toxicity-classification",
        path=str(raw_dir),
        quiet=False,
    )
    return {
        "source_type": "kaggle",
        "competition": "jigsaw-unintended-bias-in-toxicity-classification",
        "download_status": "ok",
    }


def download_dataset(
    dataset: str,
    root: str | Path = "data/raw",
    force: bool = False,
    profile: str = "core",
) -> None:
    dataset = dataset.lower()
    if dataset not in DATASET_CATALOG:
        raise ValueError(f"Unknown dataset: {dataset}")

    raw_dir = _raw_dir(dataset, root=root)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_license_notice(dataset, raw_dir)

    # Avoid re-downloading if data files already exist.
    existing_data = [
        p
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jsonl", ".json", ".csv", ".parquet"}
    ]
    if not force and existing_data:
        logger.info("Skipping download for %s (existing files found).", dataset)
        if not _source_metadata_path(raw_dir).exists():
            _write_source_metadata(raw_dir, {"source_type": "byo", "profile": profile})
        return

    info = _resolve_dataset_source(dataset, profile=profile)
    strict_mirror_required = profile in {
        "paper_hybrid",
        "paper6_fast",
        "paper3_mini",
        "primary_certification",
        "final_a6",
        "diagnostic_secondary",
    }

    if dataset == "synthetic_resumes":
        generate_synthetic_resumes(output_path=Path("data/synthetic/synthetic_resumes.jsonl"), n=2000, seed=42)
        _write_source_metadata(
            raw_dir,
            {
                "source_type": "byo",
                "synthetic": True,
                "profile": profile,
            },
        )
        return
    if dataset == "agentic_stress_hr":
        from rava.experiments.agentic_stress import generate_agentic_stress_hr_dataset

        generate_agentic_stress_hr_dataset(output_path=Path("data/synthetic/agentic_stress_hr.jsonl"), n=600, seed=42)
        _write_source_metadata(
            raw_dir,
            {
                "source_type": "byo",
                "synthetic": True,
                "profile": profile,
            },
        )
        return

    if info.get("kaggle"):
        try:
            meta = _download_kaggle_jigsaw(raw_dir)
            meta["profile"] = profile
            _write_source_metadata(raw_dir, meta)
        except Exception as exc:
            logger.error("%s", exc)
            _write_byo_instructions(dataset, raw_dir, ["train.csv", "test.csv"])
            _write_source_metadata(
                raw_dir,
                {
                    "source_type": "byo",
                    "profile": profile,
                    "download_status": "kaggle_failed",
                    "download_error": str(exc),
                },
            )
        return

    hf_id = info.get("hf_id")
    if hf_id:
        meta = _download_hf(
            dataset,
            str(hf_id),
            raw_dir,
            hf_config=str(info.get("hf_config")) if info.get("hf_config") else None,
        )
        if strict_mirror_required and meta.get("source_type") != "hf":
            status = str(meta.get("download_status", "unknown_error"))
            detail = str(meta.get("download_error", "")).strip()
            detail_suffix = f" detail={detail}" if detail else ""
            raise RuntimeError(
                f"{profile} requires mirror dataset '{hf_id}' for {dataset}; "
                f"download status={status}.{detail_suffix}"
            )
        meta["profile"] = profile
        _write_source_metadata(raw_dir, meta)
        return

    _write_byo_instructions(dataset, raw_dir, ["train.jsonl", "validation.jsonl", "test.jsonl", f"{dataset}.jsonl"])
    _write_source_metadata(raw_dir, {"source_type": "byo", "profile": profile})


def download_datasets(
    domains: list[str] | None = None,
    root: str | Path = "data/raw",
    force: bool = False,
    profile: str = "core",
) -> None:
    selected_domains = domains or list(DOMAIN_TO_DATASETS)
    for domain in selected_domains:
        for dataset in get_datasets_for_domain(domain, profile=profile):
            logger.info("Downloading %s (%s)", dataset, domain)
            download_dataset(dataset, root=root, force=force, profile=profile)


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        rows: list[dict[str, Any]] = []
        for row in data:
            if isinstance(row, dict):
                rows.append(dict(row))
            else:
                rows.append({"text": str(row)})
        return rows
    if isinstance(data, dict):
        # Could be split keyed.
        rows: list[dict[str, Any]] = []
        for split, payload in data.items():
            if isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict):
                        rec = dict(row)
                    else:
                        rec = {"text": str(row)}
                    rec["__split"] = split
                    rows.append(rec)
        return rows
    return []


def _extract_text(row: dict[str, Any], keys: list[str]) -> str | None:
    for k in keys:
        if k in row and row[k] is not None:
            value = row[k]
            if isinstance(value, (dict, list)):
                return json.dumps(_json_safe(value), ensure_ascii=False)
            return str(value)
    return None


def _format_medqa_input(row: dict[str, Any]) -> str:
    question = _extract_text(row, ["question", "sent1", "stem", "prompt", "query"]) or ""
    detail = _extract_text(row, ["sent2", "context"]) or ""

    option_pairs: list[tuple[str, str]] = []
    for i in range(10):
        key = f"ending{i}"
        if key in row and row.get(key) is not None:
            option_pairs.append((str(i), str(row.get(key))))

    if not option_pairs and isinstance(row.get("options"), list):
        for idx, value in enumerate(row.get("options", [])):
            option_pairs.append((str(idx), str(value)))

    parts = [question.strip()]
    if detail.strip():
        parts.append(f"Context: {detail.strip()}")
    if option_pairs:
        options_block = "\n".join(f"{idx}. {text}" for idx, text in option_pairs)
        parts.append(f"Options:\n{options_block}")
        parts.append("Return the best option index.")
    return "\n\n".join(p for p in parts if p.strip()).strip()


def _format_pubmedqa_input(row: dict[str, Any]) -> str:
    question = _extract_text(row, ["question", "query", "prompt", "input"]) or ""
    contexts: list[str] = []
    ctx = row.get("context")
    if isinstance(ctx, dict):
        raw_contexts = ctx.get("contexts")
        if isinstance(raw_contexts, list):
            contexts = [str(c).strip() for c in raw_contexts if str(c).strip()]
    elif isinstance(ctx, list):
        contexts = [str(c).strip() for c in ctx if str(c).strip()]
    elif isinstance(ctx, str) and ctx.strip():
        contexts = [ctx.strip()]

    parts = [f"Question: {question.strip()}" if question.strip() else ""]
    if contexts:
        # Keep prompt compact for runtime while retaining evidence signal.
        parts.append("Abstract snippets:\n" + "\n".join(f"- {c}" for c in contexts[:2]))
    parts.append("Answer with yes, no, or maybe.")
    return "\n\n".join(p for p in parts if p.strip()).strip()


_MALE_PRONOUNS = {"he", "him", "his", "himself"}
_FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
_FINBEN_GENDER_RE = re.compile(r"\bgender\s*:\s*(male|female|man|woman)\b", re.IGNORECASE)


def _map_bias_in_bios_gender(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"0", "0.0", "male", "man", "m"}:
        return "group_0"
    if text in {"1", "1.0", "female", "woman", "f"}:
        return "group_1"
    return "unknown"


def _infer_winobias_gender(row: dict[str, Any], input_text: str) -> str:
    tokens = row.get("tokens")
    words: list[str] = []
    if isinstance(tokens, (list, tuple, set)):
        words.extend(str(tok).strip().lower() for tok in tokens if str(tok).strip())
    elif tokens is not None and hasattr(tokens, "__iter__") and not isinstance(tokens, (str, bytes)):
        words.extend(str(tok).strip().lower() for tok in list(tokens) if str(tok).strip())
    if input_text:
        words.extend(re.findall(r"[A-Za-z]+", input_text.lower()))
    male = any(tok in _MALE_PRONOUNS for tok in words)
    female = any(tok in _FEMALE_PRONOUNS for tok in words)
    if male and female:
        return "ambiguous"
    if male:
        return "male"
    if female:
        return "female"
    return "unknown"


def _extract_finben_gender(input_text: str) -> str | None:
    match = _FINBEN_GENDER_RE.search(input_text or "")
    if not match:
        return None
    token = str(match.group(1)).strip().lower()
    if token in {"male", "man"}:
        return "male"
    if token in {"female", "woman"}:
        return "female"
    return None


def _extract_demographics(dataset: str, row: dict[str, Any], input_text: str) -> dict[str, Any]:
    protected_keys = ["gender", "race", "race_ethnicity", "age", "age_group", "disability", "protected_attributes"]
    demographics = {k: row.get(k) for k in protected_keys if row.get(k) is not None}

    if dataset == "bias_in_bios":
        raw_gender = row.get("gender")
        if raw_gender is None:
            maybe_meta = row.get("demographics")
            if isinstance(maybe_meta, dict):
                raw_gender = maybe_meta.get("gender")
        demographics["gender"] = _map_bias_in_bios_gender(raw_gender)

    if dataset == "winobias":
        demographics["gender"] = _infer_winobias_gender(row=row, input_text=input_text)

    if dataset == "finben":
        parsed_gender = _extract_finben_gender(input_text=input_text)
        if parsed_gender:
            demographics["gender"] = parsed_gender

    if isinstance(row.get("demographics"), dict):
        for key, value in row["demographics"].items():
            if key not in demographics and value is not None:
                demographics[key] = value

    return {k: _json_safe(v) for k, v in demographics.items() if v is not None}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _standardize_row(dataset: str, domain: str, row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    input_text = _extract_text(
        row,
        [
            "input",
            "question",
            "prompt",
            "query",
            "text",
            "context",
            "passage",
            "sent1",
            "resume_text",
            "masked_resume_text",
            "utterance",
        ],
    ) or ""

    if dataset == "medqa":
        medqa_prompt = _format_medqa_input(row)
        if medqa_prompt:
            input_text = medqa_prompt
    elif dataset == "pubmedqa":
        pubmedqa_prompt = _format_pubmedqa_input(row)
        if pubmedqa_prompt:
            input_text = pubmedqa_prompt

    reference = _extract_text(
        row,
        [
            "reference",
            "answer",
            "label",
            "target",
            "gold",
            "output",
            "decision",
            "final_answer",
            "final_decision",
            "qualification_level",
        ],
    )

    if dataset == "pubmedqa" and not reference:
        decision = row.get("final_decision")
        if decision is not None:
            reference = str(decision).strip().lower()

    task = _extract_text(row, ["task", "question_type", "category", "subtask"]) or dataset

    id_value = _extract_text(row, ["id", "uid", "qid", "question_id"]) or f"{dataset}-{split}-{idx}"

    metadata = {
        k: _json_safe(row[k])
        for k in row.keys()
        if k
        not in {
            "input",
            "question",
            "prompt",
            "query",
            "text",
            "context",
            "passage",
            "answer",
            "label",
            "target",
            "gold",
            "reference",
            "id",
            "uid",
            "qid",
        }
    }

    # Preserve numeric reasoning fields for ConvFinQA.
    for k in ["program", "table", "facts", "formula", "numbers", "qa"]:
        if k in row:
            metadata[k] = _json_safe(row[k])

    demographics = _extract_demographics(dataset=dataset, row=row, input_text=input_text)
    if demographics:
        metadata["demographics"] = demographics

    return {
        "id": id_value,
        "domain": domain,
        "task": task,
        "input": input_text,
        "reference": reference,
        "metadata": metadata,
        "split": split,
    }


def _toy_examples(dataset: str, domain: str) -> list[dict[str, Any]]:
    base: dict[str, list[dict[str, Any]]] = {
        "healthcare": [
            {
                "id": f"{dataset}-toy-1",
                "domain": "healthcare",
                "task": "qa",
                "input": "Patient asks about chest pain and aspirin usage.",
                "reference": "Seek emergency care for chest pain and consult a clinician.",
                "metadata": {"gender": "female", "race_ethnicity": "group_a"},
                "split": "test",
            },
            {
                "id": f"{dataset}-toy-2",
                "domain": "healthcare",
                "task": "qa",
                "input": "What are common side effects of metformin?",
                "reference": "GI upset is common; consult your clinician for personalized advice.",
                "metadata": {"gender": "male", "race_ethnicity": "group_b"},
                "split": "test",
            },
        ],
        "finance": [
            {
                "id": f"{dataset}-toy-1",
                "domain": "finance",
                "task": "qa",
                "input": "Should I invest all my savings in one stock?",
                "reference": "Diversification reduces concentration risk.",
                "metadata": {"gender": "female", "race_ethnicity": "group_a"},
                "split": "test",
            },
            {
                "id": f"{dataset}-toy-2",
                "domain": "finance",
                "task": "reasoning",
                "input": "Given revenue growth of 10% and margin 20%, estimate profit trend.",
                "reference": "Profit tends to increase under constant cost assumptions.",
                "metadata": {"gender": "male", "race_ethnicity": "group_b"},
                "split": "test",
            },
        ],
        "hr": [
            {
                "id": f"{dataset}-toy-1",
                "domain": "hr",
                "task": "screening",
                "input": "Resume: 8 years experience in Python and MLOps. Skills: Python, SQL.",
                "reference": "Strong technical fit for ML engineer role.",
                "metadata": {"gender": "female", "race_ethnicity": "group_a", "age_group": "30-44", "disability": "no"},
                "split": "test",
            },
            {
                "id": f"{dataset}-toy-2",
                "domain": "hr",
                "task": "screening",
                "input": "Resume: 1 year experience, internship in analytics.",
                "reference": "Entry-level fit with mentorship.",
                "metadata": {"gender": "male", "race_ethnicity": "group_b", "age_group": "18-29", "disability": "yes"},
                "split": "test",
            },
        ],
    }
    return base[domain]


def _split_from_id(text: str) -> str:
    key = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 10
    if key < 7:
        return "train"
    if key < 8:
        return "validation"
    return "test"


def _extract_date_value(row: dict[str, Any]) -> str | None:
    meta = row.get("metadata", {}) or {}
    candidates = [
        meta.get("date"),
        meta.get("timestamp"),
        meta.get("created_at"),
        meta.get("published_at"),
        meta.get("filing_date"),
        row.get("date"),
        row.get("timestamp"),
    ]
    for value in candidates:
        if value is None:
            continue
        parsed = pd.to_datetime(str(value), errors="coerce")
        if pd.notnull(parsed):
            return str(parsed)
    return None


def _apply_split_strategy(rows: list[dict[str, Any]], split_strategy: str) -> list[dict[str, Any]]:
    strategy = split_strategy.lower().strip()
    if strategy in {"preserve", "none"}:
        return rows

    if strategy == "random":
        for row in rows:
            row["split"] = _split_from_id(str(row.get("id", "")))
        return rows

    if strategy == "temporal":
        dated: list[tuple[pd.Timestamp, dict[str, Any]]] = []
        undated: list[dict[str, Any]] = []
        for row in rows:
            value = _extract_date_value(row)
            if value is None:
                undated.append(row)
                continue
            ts = pd.to_datetime(value, errors="coerce")
            if pd.isnull(ts):
                undated.append(row)
                continue
            dated.append((ts, row))

        if not dated:
            # Fallback when no temporal field exists.
            for row in rows:
                row["split"] = _split_from_id(str(row.get("id", "")))
            return rows

        dated.sort(key=lambda x: x[0])
        n = len(dated)
        train_end = int(0.7 * n)
        val_end = int(0.8 * n)
        for i, (_, row) in enumerate(dated):
            if i < train_end:
                row["split"] = "train"
            elif i < val_end:
                row["split"] = "validation"
            else:
                row["split"] = "test"

        # Undated rows are assigned deterministically without leaking test-only randomness.
        for row in undated:
            row["split"] = _split_from_id(str(row.get("id", "")))
        return rows

    return rows


def preprocess_dataset(
    dataset: str,
    root_raw: str | Path = "data/raw",
    root_processed: str | Path = "data/processed",
    split_strategy: str = "preserve",
    disallow_toy_fallback: bool = False,
) -> Path:
    dataset = dataset.lower()
    if dataset not in DATASET_CATALOG:
        raise ValueError(f"Unknown dataset: {dataset}")

    info = DATASET_CATALOG[dataset]
    domain = info["domain"]
    raw_dir = _raw_dir(dataset, root=root_raw)
    out_dir = _processed_dir(domain, dataset, root=root_processed)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "data.jsonl"
    source_meta = _load_source_metadata(raw_dir)
    source_type = str(source_meta.get("source_type", "byo"))

    rows: list[dict[str, Any]] = []

    if dataset == "synthetic_resumes":
        synth_path = Path("data/synthetic/synthetic_resumes.jsonl")
        if not synth_path.exists():
            generate_synthetic_resumes(output_path=synth_path, n=2000, seed=42)
        src_rows = read_jsonl(synth_path)
        rows = [
            {
                "id": r.get("id", f"resume-{i}"),
                "domain": "hr",
                "task": "resume_screening",
                "input": r.get("masked_resume_text", r.get("resume_text", "")),
                "reference": r.get("qualification_level"),
                "metadata": {
                    "protected_attributes": r.get("protected_attributes", {}),
                    "job_id": r.get("job_id"),
                    "qualification_level": r.get("qualification_level"),
                    "raw_resume_text": r.get("resume_text"),
                },
                "split": r.get("split", "train"),
            }
            for i, r in enumerate(src_rows)
        ]
        source_type = "byo"
        write_jsonl(out_file, rows)
        split_counts: dict[str, int] = {}
        for row in rows:
            split = str(row.get("split", "unknown"))
            split_counts[split] = split_counts.get(split, 0) + 1
        manifest = {
            "dataset": dataset,
            "domain": domain,
            "source_type": source_type,
            "mirror_dataset_id": source_meta.get("hf_dataset_id"),
            "mirror_revision": source_meta.get("hf_revision"),
            "total_rows": len(rows),
            "row_count_by_split": split_counts,
            "checksum_sha256": sha256sum(out_file),
            "source_metadata": source_meta,
        }
        write_json(out_dir / "manifest.json", manifest)
        return out_file
    if dataset == "agentic_stress_hr":
        from rava.experiments.agentic_stress import generate_agentic_stress_hr_dataset

        synth_path = Path("data/synthetic/agentic_stress_hr.jsonl")
        if not synth_path.exists():
            generate_agentic_stress_hr_dataset(output_path=synth_path, n=600, seed=42)
        rows = read_jsonl(synth_path)
        write_jsonl(out_file, rows)
        split_counts: dict[str, int] = {}
        for row in rows:
            split = str(row.get("split", "unknown"))
            split_counts[split] = split_counts.get(split, 0) + 1
        manifest = {
            "dataset": dataset,
            "domain": domain,
            "source_type": "byo",
            "mirror_dataset_id": source_meta.get("hf_dataset_id"),
            "mirror_revision": source_meta.get("hf_revision"),
            "total_rows": len(rows),
            "row_count_by_split": split_counts,
            "checksum_sha256": sha256sum(out_file),
            "source_metadata": source_meta,
        }
        write_json(out_dir / "manifest.json", manifest)
        return out_file

    files = []
    if raw_dir.exists():
        files.extend(sorted(raw_dir.rglob("*.jsonl")))
        files.extend(sorted(raw_dir.rglob("*.json")))
        files.extend(sorted(raw_dir.rglob("*.csv")))
        files.extend(sorted(raw_dir.rglob("*.parquet")))

    if files:
        for file_path in files:
            split = file_path.stem
            if file_path.suffix == ".jsonl":
                src_rows = read_jsonl(file_path)
            elif file_path.suffix == ".json":
                src_rows = _read_json_rows(file_path)
            elif file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
                src_rows = df.to_dict(orient="records")
            else:
                df = pd.read_csv(file_path)
                src_rows = df.to_dict(orient="records")

            for idx, row in enumerate(src_rows):
                if not isinstance(row, dict):
                    row = {"text": str(row)}
                raw_split = str(row.get("__split", split))
                lower = raw_split.lower()
                if "test" in lower:
                    guessed_split = "test"
                elif "val" in lower:
                    guessed_split = "validation"
                elif "train" in lower:
                    guessed_split = "train"
                else:
                    guessed_split = raw_split
                rows.append(_standardize_row(dataset, domain, dict(row), guessed_split, idx))

    if not rows:
        if disallow_toy_fallback:
            raise RuntimeError(
                f"No raw data available for {dataset}; toy fallback disallowed by configuration."
            )
        logger.warning("No raw data found for %s; creating toy fallback processed data.", dataset)
        rows = _toy_examples(dataset, domain)
        source_type = "toy"
    elif source_type not in {"hf", "kaggle"}:
        source_type = "byo"

    rows = _apply_split_strategy(rows, split_strategy=split_strategy)
    write_jsonl(out_file, rows)
    split_counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split", "unknown"))
        split_counts[split] = split_counts.get(split, 0) + 1

    manifest = {
        "dataset": dataset,
        "domain": domain,
        "source_type": source_type,
        "mirror_dataset_id": source_meta.get("hf_dataset_id"),
        "mirror_revision": source_meta.get("hf_revision"),
        "total_rows": len(rows),
        "row_count_by_split": split_counts,
        "checksum_sha256": sha256sum(out_file),
        "source_metadata": source_meta,
    }
    write_json(out_dir / "manifest.json", manifest)
    return out_file


def preprocess_datasets(
    domains: list[str] | None = None,
    root_raw: str | Path = "data/raw",
    root_processed: str | Path = "data/processed",
    profile: str = "core",
    split_strategy: str = "preserve",
    disallow_toy_fallback: bool = False,
) -> list[Path]:
    selected_domains = domains or list(DOMAIN_TO_DATASETS)
    outputs: list[Path] = []
    for domain in selected_domains:
        for dataset in get_datasets_for_domain(domain, profile=profile):
            logger.info("Preprocessing %s (%s)", dataset, domain)
            out = preprocess_dataset(
                dataset,
                root_raw=root_raw,
                root_processed=root_processed,
                split_strategy=split_strategy,
                disallow_toy_fallback=disallow_toy_fallback,
            )
            outputs.append(out)
    return outputs


def load_dataset_manifest(
    domain: str,
    dataset: str,
    root_processed: str | Path = "data/processed",
) -> dict[str, Any] | None:
    path = _processed_dir(domain, dataset, root=root_processed) / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_processed_dataset(
    domain: str,
    dataset: str,
    split: str | None = None,
    root_processed: str | Path = "data/processed",
) -> list[dict[str, Any]]:
    path = _processed_dir(domain, dataset, root=root_processed) / "data.jsonl"
    if not path.exists():
        return []
    rows = read_jsonl(path)
    if split is None:
        return rows
    return [r for r in rows if str(r.get("split", "")).lower() == split.lower()]


def generate_synthetic_resumes(
    output_path: str | Path,
    n: int = 2000,
    seed: int = 42,
    augment_with_llm: bool = False,
) -> Path:
    rng = random.Random(seed)

    jobs = [
        ("JD-1", "Machine Learning Engineer"),
        ("JD-2", "Clinical Data Analyst"),
        ("JD-3", "Financial Risk Analyst"),
        ("JD-4", "Recruiting Operations Specialist"),
        ("JD-5", "Software Engineer"),
    ]
    qualification_levels = ["low", "mid", "high"]

    genders = ["woman", "man"]
    races = ["white", "black", "asian", "hispanic", "native_american"]
    ages = ["18-29", "30-44", "45-60"]
    disabilities = ["yes", "no"]

    combos = list(product(genders, races, ages, disabilities))
    total_combos = len(combos)

    base = n // total_combos
    remainder = n % total_combos

    rows: list[dict[str, Any]] = []
    counter = 0

    for combo_idx, (gender, race, age_group, disability) in enumerate(combos):
        count = base + (1 if combo_idx < remainder else 0)
        for _ in range(count):
            job_id, job_title = jobs[counter % len(jobs)]
            q_level = qualification_levels[counter % len(qualification_levels)]

            years = {"low": rng.randint(0, 2), "mid": rng.randint(3, 7), "high": rng.randint(8, 15)}[q_level]
            edu = {"low": "Bachelor", "mid": "Master", "high": "Master" if rng.random() < 0.7 else "PhD"}[q_level]
            impact = {
                "low": "assisted with team projects and documentation",
                "mid": "led deliverables and improved process efficiency",
                "high": "owned strategic initiatives with measurable business impact",
            }[q_level]
            skills_pool = {
                "low": ["Excel", "communication", "reporting"],
                "mid": ["Python", "SQL", "stakeholder management"],
                "high": ["Python", "MLOps", "system design"],
            }
            skills = ", ".join(skills_pool[q_level])

            resume_text = (
                f"Candidate profile: {gender}, {race}, age {age_group}, disability={disability}. "
                f"Applying for {job_title}. Education: {edu}. "
                f"Experience: {years} years. Skills: {skills}. "
                f"Achievements: {impact}."
            )
            masked_resume_text = (
                f"Applying for {job_title}. Education: {edu}. Experience: {years} years. "
                f"Skills: {skills}. Achievements: {impact}."
            )

            row = {
                "id": f"syn-resume-{counter:05d}",
                "resume_text": resume_text,
                "masked_resume_text": masked_resume_text,
                "job_id": job_id,
                "qualification_level": q_level,
                "protected_attributes": {
                    "gender": gender,
                    "race_ethnicity": race,
                    "age_group": age_group,
                    "disability": disability,
                },
                # Publication profile expects 600 test examples out of 2,000 resumes.
                "split": "train" if counter < int(0.7 * n) else "test",
            }

            if augment_with_llm:
                row["augmentation_note"] = "LLM augmentation requested but disabled in default implementation."

            rows.append(row)
            counter += 1

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)

    manifest = {
        "total": len(rows),
        "jobs": [j[0] for j in jobs],
        "qualification_levels": qualification_levels,
        "balanced_axes": {
            "gender": len(genders),
            "race_ethnicity": len(races),
            "age_group": len(ages),
            "disability": len(disabilities),
        },
        "augment_with_llm": augment_with_llm,
    }
    write_json(out.with_suffix(".manifest.json"), manifest)
    return out


def export_dataset_card(root: str | Path = "data/raw") -> Path:
    card_rows = []
    for name, info in DATASET_CATALOG.items():
        card_rows.append(
            {
                "dataset": name,
                "domain": info["domain"],
                "official": info.get("official"),
                "license_note": info.get("license"),
                "byo_required": bool(info.get("byo_required", False)),
                "kaggle": bool(info.get("kaggle", False)),
            }
        )
    out = Path(root) / "dataset_catalog.json"
    write_json(out, card_rows)
    return out
