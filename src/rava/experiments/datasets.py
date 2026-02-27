from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
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
        "hf_id": None,
        "official": "https://github.com/microsoft/biosbias",
        "license": "Bias in Bios terms apply.",
        "byo_required": True,
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


def _download_hf(dataset: str, hf_id: str, raw_dir: Path) -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        logger.warning("datasets package unavailable for %s: %s", dataset, exc)
        _write_byo_instructions(dataset, raw_dir, ["train.jsonl", "validation.jsonl", "test.jsonl"])
        return

    try:
        ds = load_dataset(hf_id)
    except Exception as exc:
        logger.warning("HF download failed for %s (%s): %s", dataset, hf_id, exc)
        _write_byo_instructions(dataset, raw_dir, ["train.jsonl", "validation.jsonl", "test.jsonl"])
        return

    for split_name, split_data in ds.items():
        rows = []
        for row in tqdm(split_data, desc=f"{dataset}:{split_name}", leave=False):
            payload = dict(row)
            payload["__split"] = split_name
            rows.append(payload)
        write_jsonl(raw_dir / f"{split_name}.jsonl", rows)


def _download_kaggle_jigsaw(raw_dir: Path) -> None:
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


def download_dataset(dataset: str, root: str | Path = "data/raw", force: bool = False) -> None:
    dataset = dataset.lower()
    if dataset not in DATASET_CATALOG:
        raise ValueError(f"Unknown dataset: {dataset}")

    raw_dir = _raw_dir(dataset, root=root)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_license_notice(dataset, raw_dir)

    # Avoid re-downloading if data files exist.
    if not force and any(raw_dir.glob("*.jsonl")):
        logger.info("Skipping download for %s (existing files found).", dataset)
        return

    info = DATASET_CATALOG[dataset]

    if dataset == "synthetic_resumes":
        generate_synthetic_resumes(output_path=Path("data/synthetic/synthetic_resumes.jsonl"), n=2000, seed=42)
        return

    if info.get("kaggle"):
        try:
            _download_kaggle_jigsaw(raw_dir)
        except Exception as exc:
            logger.error("%s", exc)
            _write_byo_instructions(dataset, raw_dir, ["train.csv", "test.csv"])
        return

    hf_id = info.get("hf_id")
    if hf_id:
        _download_hf(dataset, hf_id, raw_dir)
        return

    _write_byo_instructions(dataset, raw_dir, ["train.jsonl", "validation.jsonl", "test.jsonl", f"{dataset}.jsonl"])


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
            download_dataset(dataset, root=root, force=force)


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [dict(row) for row in data]
    if isinstance(data, dict):
        # Could be split keyed.
        rows: list[dict[str, Any]] = []
        for split, payload in data.items():
            if isinstance(payload, list):
                for row in payload:
                    rec = dict(row)
                    rec["__split"] = split
                    rows.append(rec)
        return rows
    return []


def _extract_text(row: dict[str, Any], keys: list[str]) -> str | None:
    for k in keys:
        if k in row and row[k] is not None:
            value = row[k]
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)
    return None


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
            "resume_text",
            "masked_resume_text",
            "utterance",
        ],
    ) or ""

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
            "qualification_level",
        ],
    )

    task = _extract_text(row, ["task", "question_type", "category", "subtask"]) or dataset

    id_value = _extract_text(row, ["id", "uid", "qid", "question_id"]) or f"{dataset}-{split}-{idx}"

    protected_keys = ["gender", "race", "race_ethnicity", "age", "age_group", "disability", "protected_attributes"]
    metadata = {k: row[k] for k in row.keys() if k not in {"input", "question", "prompt", "query", "text", "context", "passage", "answer", "label", "target", "gold", "reference", "id", "uid", "qid"}}

    # Preserve numeric reasoning fields for ConvFinQA.
    for k in ["program", "table", "facts", "formula", "numbers", "qa"]:
        if k in row:
            metadata[k] = row[k]

    demographics = {k: row.get(k) for k in protected_keys if row.get(k) is not None}
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


def preprocess_dataset(dataset: str, root_raw: str | Path = "data/raw", root_processed: str | Path = "data/processed") -> Path:
    dataset = dataset.lower()
    if dataset not in DATASET_CATALOG:
        raise ValueError(f"Unknown dataset: {dataset}")

    info = DATASET_CATALOG[dataset]
    domain = info["domain"]
    raw_dir = _raw_dir(dataset, root=root_raw)
    out_dir = _processed_dir(domain, dataset, root=root_processed)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "data.jsonl"

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
        write_jsonl(out_file, rows)
        return out_file

    files = []
    if raw_dir.exists():
        files.extend(sorted(raw_dir.glob("*.jsonl")))
        files.extend(sorted(raw_dir.glob("*.json")))
        files.extend(sorted(raw_dir.glob("*.csv")))

    if files:
        for file_path in files:
            split = file_path.stem
            if file_path.suffix == ".jsonl":
                src_rows = read_jsonl(file_path)
            elif file_path.suffix == ".json":
                src_rows = _read_json_rows(file_path)
            else:
                df = pd.read_csv(file_path)
                src_rows = df.to_dict(orient="records")

            for idx, row in enumerate(src_rows):
                guessed_split = str(row.get("__split", split))
                rows.append(_standardize_row(dataset, domain, dict(row), guessed_split, idx))

    if not rows:
        logger.warning("No raw data found for %s; creating toy fallback processed data.", dataset)
        rows = _toy_examples(dataset, domain)

    write_jsonl(out_file, rows)
    return out_file


def preprocess_datasets(
    domains: list[str] | None = None,
    root_raw: str | Path = "data/raw",
    root_processed: str | Path = "data/processed",
    profile: str = "core",
) -> list[Path]:
    selected_domains = domains or list(DOMAIN_TO_DATASETS)
    outputs: list[Path] = []
    for domain in selected_domains:
        for dataset in get_datasets_for_domain(domain, profile=profile):
            logger.info("Preprocessing %s (%s)", dataset, domain)
            out = preprocess_dataset(dataset, root_raw=root_raw, root_processed=root_processed)
            outputs.append(out)
    return outputs


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
                "split": "train" if counter < int(0.8 * n) else "test",
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
