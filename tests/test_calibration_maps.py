from __future__ import annotations

from pathlib import Path

from rava.experiments.runner import run_sweep
from rava.metrics.calibration import apply_confidence_map, fit_confidence_map
from rava.agent.react_agent import _select_confidence_calibration_map
from rava.utils.serialization import read_json, write_json, write_yaml


def test_fit_confidence_map_prefers_supported_method():
    rows = []
    for idx in range(120):
        conf = 0.2 if idx < 60 else 0.85
        correct = 0 if idx < 60 else 1
        rows.append(
            {
                "id": f"x-{idx}",
                "output": f"Confidence: {conf}",
                "confidence": conf,
                "correct": correct,
                "generation_mode": "langchain_chat",
            }
        )

    fitted = fit_confidence_map(rows, methods=("isotonic", "platt"), target_ece=0.12, n_bins=10, min_samples=20)
    assert fitted["method"] in {"isotonic", "platt"}
    assert fitted["n_samples"] == 120
    assert fitted["ece_after"] <= fitted["ece_before"]

    mapped = apply_confidence_map(0.9, fitted)
    assert mapped is not None
    assert 0.0 <= mapped <= 1.0


def test_calibration_stage_writes_maps_and_full_stage_loads_them(tmp_path: Path, monkeypatch):
    cal_artifacts = tmp_path / "cal_maps"
    sweep_cfg = {
        "name": "unit_calibration",
        "domains": ["healthcare"],
        "dataset_profile": "core",
        "models": ["configs/models/mock.yaml"],
        "verification_configs": ["none"],
        "seeds": [42],
        "max_examples_per_dataset": 10,
        "provider_preflight_enabled": False,
        "allow_generation_fallback": False,
        "calibration_fit": {
            "enabled": True,
            "methods": ["isotonic", "platt"],
            "target_ece": 0.12,
            "min_samples": 10,
            "num_examples_per_domain": 40,
            "artifacts_dir": str(cal_artifacts),
        },
        "calibration_eval_split": "validation",
        "certification_stage_policy": "full_only",
        "resume_root": str(tmp_path / "runs" / "calibration_root"),
    }
    sweep_path = tmp_path / "sweep.yaml"
    write_yaml(sweep_path, sweep_cfg)

    monkeypatch.setattr("rava.experiments.runner._ensure_processed_data", lambda *args, **kwargs: ["medqa"])
    monkeypatch.setattr(
        "rava.experiments.runner.load_processed_dataset",
        lambda **kwargs: [
            {
                "id": f"example-{i}",
                "domain": "healthcare",
                "task": "medqa",
                "input": "Q",
                "reference": "a",
                "metadata": {},
                "split": kwargs.get("split") or "validation",
            }
            for i in range(40)
        ],
    )

    def _mock_agent(**kwargs):
        ex_id = str(kwargs["example"]["id"])
        idx = int(ex_id.split("-")[-1])
        conf = 0.25 if idx % 2 == 0 else 0.8
        correct = 0 if idx % 2 == 0 else 1
        return {
            "prediction": {
                "id": ex_id,
                "domain": "healthcare",
                "input": "Q",
                "reference": "a",
                "output": "answer",
                "confidence": conf,
                "correct": correct,
                "metadata": {},
                "split": "validation",
                "generation_mode": "langchain_chat",
                "generation_error": None,
            },
            "trajectory": [
                {
                    "action": "langchain_agent_generation",
                    "duration_ms": 1.0,
                    "metadata": {"mode": "langchain_chat"},
                }
            ],
            "verdicts": [],
            "report": {"summary": {}},
        }

    monkeypatch.setattr("rava.experiments.runner.run_agent_example", _mock_agent)

    cal_root = run_sweep(
        sweep_config_path=sweep_path,
        base_config_path="configs/base.yaml",
        stage="calibration",
    )

    summary = read_json(Path(cal_root) / "calibration_fit_summary.json")
    assert summary
    map_path = cal_artifacts / "mock-v1" / "healthcare.json"
    assert map_path.exists()

    captured = {"seen": False}

    def _mock_agent_full(**kwargs):
        provider = kwargs["provider"]
        maps = getattr(provider, "confidence_calibration_maps", {})
        if isinstance(maps, dict) and "healthcare" in maps:
            captured["seen"] = True
        return _mock_agent(**kwargs)

    monkeypatch.setattr("rava.experiments.runner.run_agent_example", _mock_agent_full)
    sweep_cfg["resume_root"] = str(tmp_path / "runs" / "full_root")
    write_yaml(sweep_path, sweep_cfg)

    run_sweep(
        sweep_config_path=sweep_path,
        base_config_path="configs/base.yaml",
        stage="full",
    )
    assert captured["seen"] is True


def test_identity_map_selected_when_candidates_do_not_improve():
    rows = []
    for idx in range(80):
        conf = 0.8 if idx % 2 == 0 else 0.2
        rows.append(
            {
                "id": f"id-{idx}",
                "output": f"Confidence: {conf}",
                "confidence": conf,
                "correct": 1 if idx % 2 == 0 else 0,
                "generation_mode": "langchain_chat",
            }
        )
    fitted = fit_confidence_map(
        rows,
        methods=("identity",),
        target_ece=0.01,
        n_bins=10,
        min_samples=20,
        min_improvement=0.02,
    )
    assert fitted["method"] == "identity"
    assert fitted["accepted_identity"] is True
    assert fitted["selection_reason"] == "no_candidate_improved_identity_selected"


def test_rejected_map_is_not_applied():
    mapping = {
        "method": "platt",
        "params": {"a": 3.0, "b": -1.0},
        "accepted": False,
        "accepted_identity": False,
        "ece_before": 0.08,
        "ece_after": 0.20,
        "target_ece": 0.12,
        "min_improvement": 0.005,
    }
    value = 0.64
    assert apply_confidence_map(value, mapping) == value


def test_domain_config_scope_writes_and_loads_config_specific_map(tmp_path: Path, monkeypatch):
    cal_artifacts = tmp_path / "cal_maps"
    sweep_cfg = {
        "name": "unit_calibration_domain_config",
        "domains": ["healthcare"],
        "dataset_profile": "core",
        "models": ["configs/models/mock.yaml"],
        "verification_configs": ["full"],
        "seeds": [42],
        "max_examples_per_dataset": 12,
        "provider_preflight_enabled": False,
        "allow_generation_fallback": False,
        "calibration_fit": {
            "enabled": True,
            "scope": "domain_config",
            "methods": ["isotonic", "platt", "identity"],
            "target_ece": 1.0,
            "min_samples": 10,
            "min_improvement": 0.0,
            "num_examples_per_domain": 40,
            "artifacts_dir": str(cal_artifacts),
        },
        "calibration_eval_split": "validation",
        "calibration_verification_configs": ["none", "full"],
        "certification_stage_policy": "full_only",
        "resume_mode": "fresh",
        "resume_root": str(tmp_path / "runs" / "domain_config_root"),
    }
    sweep_path = tmp_path / "sweep_domain_config.yaml"
    write_yaml(sweep_path, sweep_cfg)

    monkeypatch.setattr("rava.experiments.runner._ensure_processed_data", lambda *args, **kwargs: ["medqa"])
    monkeypatch.setattr(
        "rava.experiments.runner.load_processed_dataset",
        lambda **kwargs: [
            {
                "id": f"example-{i}",
                "domain": "healthcare",
                "task": "medqa",
                "input": "Q",
                "reference": "a",
                "metadata": {},
                "split": kwargs.get("split") or "validation",
            }
            for i in range(50)
        ],
    )

    def _mock_agent(**kwargs):
        cfg = kwargs["verification_cfg"]
        full_like = bool(cfg.pre or cfg.runtime or cfg.posthoc)
        conf = 0.81 if full_like else 0.34
        ex_id = str(kwargs["example"]["id"])
        idx = int(ex_id.split("-")[-1])
        return {
            "prediction": {
                "id": ex_id,
                "domain": "healthcare",
                "input": "Q",
                "reference": "a",
                "output": "answer",
                "confidence": conf,
                "correct": 1 if idx % 2 else 0,
                "metadata": {},
                "split": "validation",
                "generation_mode": "langchain_chat",
                "generation_error": None,
            },
            "trajectory": [
                {
                    "action": "langchain_agent_generation",
                    "duration_ms": 1.0,
                    "metadata": {"mode": "langchain_chat"},
                }
            ],
            "verdicts": [],
            "report": {"summary": {}},
        }

    monkeypatch.setattr("rava.experiments.runner.run_agent_example", _mock_agent)
    cal_root = run_sweep(
        sweep_config_path=sweep_path,
        base_config_path="configs/base.yaml",
        stage="calibration",
    )
    assert read_json(Path(cal_root) / "calibration_fit_summary.json")

    domain_map_path = cal_artifacts / "mock-v1" / "healthcare.json"
    none_map_path = cal_artifacts / "mock-v1" / "healthcare" / "none.json"
    full_map_path = cal_artifacts / "mock-v1" / "healthcare" / "full.json"
    assert domain_map_path.exists()
    assert none_map_path.exists()
    assert full_map_path.exists()

    domain_map = read_json(domain_map_path)
    full_map = read_json(full_map_path)
    domain_map["marker"] = "domain_map"
    full_map["marker"] = "full_map"
    domain_map["accepted"] = True
    full_map["accepted"] = True
    write_json(domain_map_path, domain_map)
    write_json(full_map_path, full_map)

    captured = {"marker": None}

    def _mock_agent_full(**kwargs):
        provider = kwargs["provider"]
        maps = getattr(provider, "confidence_calibration_maps", {}) or {}
        selected = maps.get("healthcare") if isinstance(maps, dict) else None
        if isinstance(selected, dict):
            captured["marker"] = selected.get("marker")
        return _mock_agent(**kwargs)

    monkeypatch.setattr("rava.experiments.runner.run_agent_example", _mock_agent_full)
    run_sweep(
        sweep_config_path=sweep_path,
        base_config_path="configs/base.yaml",
        stage="full",
    )
    assert captured["marker"] == "full_map"


def test_domain_dataset_config_scope_writes_and_loads_dataset_specific_map(tmp_path: Path, monkeypatch):
    cal_artifacts = tmp_path / "cal_maps"
    sweep_cfg = {
        "name": "unit_calibration_domain_dataset_config",
        "domains": ["healthcare"],
        "dataset_profile": "final_a6",
        "models": ["configs/models/mock.yaml"],
        "verification_configs": ["full"],
        "seeds": [42],
        "max_examples_per_dataset": 12,
        "provider_preflight_enabled": False,
        "allow_generation_fallback": False,
        "calibration_fit": {
            "enabled": True,
            "scope": "domain_dataset_config",
            "methods": ["isotonic", "platt", "identity"],
            "target_ece": 1.0,
            "min_samples": 10,
            "min_improvement": 0.0,
            "num_examples_per_domain": 40,
            "artifacts_dir": str(cal_artifacts),
        },
        "calibration_eval_split": "validation",
        "calibration_verification_configs": ["full"],
        "certification_stage_policy": "full_only",
        "resume_mode": "fresh",
        "resume_root": str(tmp_path / "runs" / "domain_dataset_config_root"),
    }
    sweep_path = tmp_path / "sweep_domain_dataset_config.yaml"
    write_yaml(sweep_path, sweep_cfg)

    monkeypatch.setattr("rava.experiments.runner._ensure_processed_data", lambda *args, **kwargs: ["pubmedqa", "medqa"])

    def _load_processed_dataset(**kwargs):
        dataset = kwargs["dataset"]
        return [
            {
                "id": f"{dataset}-{i}",
                "domain": "healthcare",
                "task": dataset,
                "input": "Q",
                "reference": "yes" if dataset == "pubmedqa" else "a",
                "metadata": {},
                "split": kwargs.get("split") or "validation",
            }
            for i in range(30)
        ]

    monkeypatch.setattr("rava.experiments.runner.load_processed_dataset", _load_processed_dataset)

    def _mock_agent(**kwargs):
        task_name = str(kwargs["example"]["task"])
        conf = 0.82 if task_name == "pubmedqa" else 0.31
        ex_id = str(kwargs["example"]["id"])
        idx = int(ex_id.split("-")[-1])
        return {
            "prediction": {
                "id": ex_id,
                "domain": "healthcare",
                "input": "Q",
                "reference": kwargs["example"]["reference"],
                "output": "answer",
                "confidence": conf,
                "correct": 1 if idx % 2 else 0,
                "metadata": {},
                "split": "validation",
                "generation_mode": "langchain_chat",
                "generation_error": None,
            },
            "trajectory": [
                {
                    "action": "langchain_agent_generation",
                    "duration_ms": 1.0,
                    "metadata": {"mode": "langchain_chat"},
                }
            ],
            "verdicts": [],
            "report": {"summary": {}},
        }

    monkeypatch.setattr("rava.experiments.runner.run_agent_example", _mock_agent)
    cal_root = run_sweep(
        sweep_config_path=sweep_path,
        base_config_path="configs/base.yaml",
        stage="calibration",
    )
    assert read_json(Path(cal_root) / "calibration_fit_summary.json")

    pubmedqa_map_path = cal_artifacts / "mock-v1" / "healthcare" / "pubmedqa" / "full.json"
    medqa_map_path = cal_artifacts / "mock-v1" / "healthcare" / "medqa" / "full.json"
    assert pubmedqa_map_path.exists()
    assert medqa_map_path.exists()

    pubmedqa_map = read_json(pubmedqa_map_path)
    medqa_map = read_json(medqa_map_path)
    pubmedqa_map["marker"] = "pubmedqa_map"
    medqa_map["marker"] = "medqa_map"
    pubmedqa_map["accepted"] = True
    medqa_map["accepted"] = True
    write_json(pubmedqa_map_path, pubmedqa_map)
    write_json(medqa_map_path, medqa_map)

    captured = {"markers": set()}

    def _mock_agent_full(**kwargs):
        provider = kwargs["provider"]
        dataset = str(kwargs["example"]["task"])
        selected = _select_confidence_calibration_map(provider=provider, domain="healthcare", dataset=dataset)
        if isinstance(selected, dict) and selected.get("marker"):
            captured["markers"].add(str(selected["marker"]))
        return _mock_agent(**kwargs)

    monkeypatch.setattr("rava.experiments.runner.run_agent_example", _mock_agent_full)
    run_sweep(
        sweep_config_path=sweep_path,
        base_config_path="configs/base.yaml",
        stage="full",
    )
    assert captured["markers"] == {"pubmedqa_map", "medqa_map"}
