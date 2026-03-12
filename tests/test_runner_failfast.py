from pathlib import Path

import pytest

from rava.experiments.runner import run_sweep
from rava.utils.serialization import read_json, write_yaml


def test_run_sweep_aborts_on_preflight_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sweep_cfg = {
        "name": "preflight_fail",
        "domains": ["healthcare"],
        "dataset_profile": "core",
        "models": ["configs/models/mock.yaml"],
        "verification_configs": ["none"],
        "seeds": [42],
        "max_examples_per_dataset": 1,
        "provider_preflight_enabled": True,
        "provider_preflight_abort_on_fail": True,
        "provider_preflight_min_success_rate": 0.95,
    }
    cfg_path = tmp_path / "sweep.yaml"
    write_yaml(cfg_path, sweep_cfg)

    monkeypatch.setattr(
        "rava.experiments.runner.provider_healthcheck",
        lambda provider, n_probes=5, timeout=30: {
            "provider": provider.name,
            "model": provider.model,
            "n_probes": n_probes,
            "success_rate": 0.0,
            "error_taxonomy_counts": {"dns_error": n_probes},
            "probes": [],
        },
    )

    with pytest.raises(RuntimeError):
        run_sweep(sweep_config_path=cfg_path, base_config_path="configs/base.yaml")


def test_run_sweep_triggers_fallback_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sweep_cfg = {
        "name": "fallback_guard",
        "domains": ["healthcare"],
        "dataset_profile": "core",
        "models": ["configs/models/mock.yaml"],
        "verification_configs": ["none"],
        "seeds": [42],
        "max_examples_per_dataset": 1,
        "provider_preflight_enabled": False,
        "run_quality_guard_enabled": True,
        "fallback_guard_window": 1,
        "fallback_guard_max_rate": 0.0,
        "resume_root": str(tmp_path / "runs_out"),
    }
    cfg_path = tmp_path / "sweep.yaml"
    write_yaml(cfg_path, sweep_cfg)

    monkeypatch.setattr("rava.experiments.runner._ensure_processed_data", lambda *args, **kwargs: ["medqa"])
    monkeypatch.setattr(
        "rava.experiments.runner.load_processed_dataset",
        lambda **kwargs: [
            {
                "id": "example-1",
                "domain": "healthcare",
                "task": "qa",
                "input": "What should I do?",
                "reference": "consult doctor",
                "metadata": {},
                "split": "test",
            }
        ],
    )
    monkeypatch.setattr(
        "rava.experiments.runner.run_agent_example",
        lambda **kwargs: {
            "prediction": {
                "id": "example-1",
                "domain": "healthcare",
                "input": "What should I do?",
                "reference": "consult doctor",
                "output": "abstain",
                "confidence": 0.2,
                "metadata": {},
                "split": "test",
                "generation_mode": "langchain_chat_fallback",
                "generation_error": "Connection error.",
            },
            "trajectory": [
                {
                    "action": "langchain_agent_generation",
                    "duration_ms": 1.0,
                    "metadata": {"mode": "langchain_chat_fallback", "error": "Connection error."},
                }
            ],
            "verdicts": [],
            "report": {"summary": {}},
        },
    )

    root = run_sweep(sweep_config_path=cfg_path, base_config_path="configs/base.yaml")
    metrics_path = next(Path(root).rglob("metrics.json"))
    metrics = read_json(metrics_path)
    assert metrics["valid_for_model_comparison"] is False
    assert metrics.get("infra_error_taxonomy") == "fallback_guard_triggered"


def test_run_sweep_fail_slow_transient_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sweep_cfg = {
        "name": "fail_slow",
        "domains": ["healthcare"],
        "dataset_profile": "core",
        "models": ["configs/models/mock.yaml"],
        "verification_configs": ["none"],
        "seeds": [42],
        "max_examples_per_dataset": 25,
        "provider_preflight_enabled": False,
        "run_quality_guard_enabled": True,
        "example_error_guard_window": 20,
        "example_error_guard_max_rate": 0.50,
        "max_consecutive_example_failures": 5,
        "resume_root": str(tmp_path / "runs_out"),
    }
    cfg_path = tmp_path / "sweep.yaml"
    write_yaml(cfg_path, sweep_cfg)

    monkeypatch.setattr("rava.experiments.runner._ensure_processed_data", lambda *args, **kwargs: ["medqa"])
    monkeypatch.setattr(
        "rava.experiments.runner.load_processed_dataset",
        lambda **kwargs: [
            {
                "id": f"example-{i}",
                "domain": "healthcare",
                "task": "qa",
                "input": "What should I do?",
                "reference": "consult doctor",
                "metadata": {},
                "split": "test",
            }
            for i in range(25)
        ],
    )

    calls = {"n": 0}

    def _fake_run_agent_example(**kwargs):
        calls["n"] += 1
        if calls["n"] in {1, 2}:
            raise RuntimeError("Read timeout")
        ex = kwargs["example"]
        return {
            "prediction": {
                "id": ex["id"],
                "domain": "healthcare",
                "input": ex["input"],
                "reference": ex["reference"],
                "output": "consult doctor",
                "confidence": 0.9,
                "metadata": {},
                "split": "test",
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

    monkeypatch.setattr("rava.experiments.runner.run_agent_example", _fake_run_agent_example)

    root = run_sweep(sweep_config_path=cfg_path, base_config_path="configs/base.yaml")
    metrics_path = next(Path(root).rglob("metrics.json"))
    metrics = read_json(metrics_path)
    assert metrics["example_error_count"] == 2.0
    assert bool(metrics.get("infra_failed", False)) is False
    assert metrics["num_predictions"] == 23.0


def test_run_sweep_aborts_on_consecutive_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sweep_cfg = {
        "name": "consecutive_failures",
        "domains": ["healthcare"],
        "dataset_profile": "core",
        "models": ["configs/models/mock.yaml"],
        "verification_configs": ["none"],
        "seeds": [42],
        "max_examples_per_dataset": 20,
        "provider_preflight_enabled": False,
        "run_quality_guard_enabled": True,
        "example_error_guard_window": 20,
        "example_error_guard_max_rate": 0.05,
        "max_consecutive_example_failures": 5,
        "resume_root": str(tmp_path / "runs_out"),
    }
    cfg_path = tmp_path / "sweep.yaml"
    write_yaml(cfg_path, sweep_cfg)

    monkeypatch.setattr("rava.experiments.runner._ensure_processed_data", lambda *args, **kwargs: ["medqa"])
    monkeypatch.setattr(
        "rava.experiments.runner.load_processed_dataset",
        lambda **kwargs: [
            {
                "id": f"example-{i}",
                "domain": "healthcare",
                "task": "qa",
                "input": "What should I do?",
                "reference": "consult doctor",
                "metadata": {},
                "split": "test",
            }
            for i in range(20)
        ],
    )
    monkeypatch.setattr("rava.experiments.runner.run_agent_example", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("Read timeout")))

    root = run_sweep(sweep_config_path=cfg_path, base_config_path="configs/base.yaml")
    metrics_path = next(Path(root).rglob("metrics.json"))
    metrics = read_json(metrics_path)
    assert metrics["valid_for_model_comparison"] is False
    assert metrics.get("infra_error_taxonomy") == "consecutive_example_failures_guard_triggered"


def test_run_sweep_canary_stage_uses_canary_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sweep_cfg = {
        "name": "canary_stage",
        "domains": ["healthcare"],
        "dataset_profile": "core",
        "models": ["configs/models/mock.yaml"],
        "verification_configs": ["none", "pre", "runtime", "posthoc", "full"],
        "seeds": [42, 123, 456],
        "max_examples_per_dataset": 30,
        "canary_seeds": [42, 123],
        "canary_verification_configs": ["none", "full"],
        "canary_max_examples_per_dataset": 10,
        "certification_stage_policy": "full_only",
        "provider_preflight_enabled": False,
        "resume_root": str(tmp_path / "runs_out"),
    }
    cfg_path = tmp_path / "sweep.yaml"
    write_yaml(cfg_path, sweep_cfg)

    monkeypatch.setattr("rava.experiments.runner._ensure_processed_data", lambda *args, **kwargs: ["medqa"])
    monkeypatch.setattr(
        "rava.experiments.runner.load_processed_dataset",
        lambda **kwargs: [
            {
                "id": f"example-{i}",
                "domain": "healthcare",
                "task": "qa",
                "input": "What should I do?",
                "reference": "consult doctor",
                "metadata": {},
                "split": "test",
            }
            for i in range(30)
        ],
    )
    monkeypatch.setattr(
        "rava.experiments.runner.run_agent_example",
        lambda **kwargs: {
            "prediction": {
                "id": kwargs["example"]["id"],
                "domain": "healthcare",
                "input": kwargs["example"]["input"],
                "reference": kwargs["example"]["reference"],
                "output": "consult doctor",
                "confidence": 0.9,
                "metadata": {},
                "split": "test",
                "generation_mode": "langchain_chat",
                "generation_error": None,
            },
            "trajectory": [{"action": "langchain_agent_generation", "duration_ms": 1.0, "metadata": {"mode": "langchain_chat"}}],
            "verdicts": [],
            "report": {"summary": {}},
        },
    )

    root = run_sweep(sweep_config_path=cfg_path, base_config_path="configs/base.yaml", stage="canary")
    metrics_paths = sorted(Path(root).rglob("metrics.json"))
    assert len(metrics_paths) == 4  # 1 domain x 1 model x 2 configs x 2 seeds
    for path in metrics_paths:
        metrics = read_json(path)
        assert metrics["num_predictions"] == 10.0
        assert metrics["sweep_stage"] == "canary"
        assert metrics["certification_eligible"] is False
