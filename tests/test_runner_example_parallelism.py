from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from rava.experiments.runner import _resolve_example_parallelism_for_task, run_sweep
from rava.utils.serialization import write_yaml


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_example_parallelism_preserves_deterministic_output_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sweep_cfg = {
        "name": "parallel_order",
        "domains": ["healthcare"],
        "dataset_profile": "core",
        "models": ["configs/models/mock.yaml"],
        "verification_configs": ["none"],
        "seeds": [42],
        "max_examples_per_dataset": 6,
        "provider_preflight_enabled": False,
        "example_parallelism_per_run": 3,
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
                "input": f"Question {i}",
                "reference": "A",
                "metadata": {},
                "split": "test",
            }
            for i in range(6)
        ],
    )

    def _fake_run_agent_example(**kwargs):
        ex = kwargs["example"]
        idx = int(str(ex["id"]).split("-")[-1])
        time.sleep(0.01 * (idx % 3))
        return {
            "prediction": {
                "id": ex["id"],
                "domain": ex["domain"],
                "input": ex["input"],
                "reference": ex["reference"],
                "output": "A",
                "confidence": 0.8,
                "abstained": False,
                "abstain_reason": None,
                "metadata": ex["metadata"],
                "split": ex["split"],
                "retrieval_context": "",
                "correct": 1,
                "timestamp": time.time(),
                "raw_model_output": "A",
                "generation_mode": "langchain_chat",
                "generation_error": None,
                "runtime_halted_hard_fail": False,
            },
            "trajectory": [
                {
                    "run_id": "unit",
                    "example_id": ex["id"],
                    "domain": ex["domain"],
                    "step_id": 1,
                    "phase": "agent",
                    "action": "langchain_agent_generation",
                    "observation": "A",
                    "started_at": 0.0,
                    "ended_at": 0.0,
                    "duration_ms": 0.0,
                    "metadata": {"mode": "langchain_chat"},
                }
            ],
            "verdicts": [],
            "report": {"summary": {}},
        }

    monkeypatch.setattr("rava.experiments.runner.run_agent_example", _fake_run_agent_example)

    root = run_sweep(sweep_config_path=cfg_path, base_config_path="configs/base.yaml")
    predictions_path = next(Path(root).rglob("predictions.jsonl"))
    prediction_rows = _read_jsonl(predictions_path)
    assert [row["id"] for row in prediction_rows] == [
        "example-0",
        "example-1",
        "example-2",
        "example-3",
        "example-4",
        "example-5",
    ]


def test_example_parallelism_override_applies_only_to_matching_slice():
    overrides = [
        {"model": "gpt-5.4", "domain": "finance", "value": 2},
        {"model": "ministral-3-cloud", "domain": "finance", "value": 3},
    ]

    assert (
        _resolve_example_parallelism_for_task(
            base_parallelism=4,
            overrides_cfg=overrides,
            model_name="gpt-5.4",
            domain="finance",
            datasets=["convfinqa", "finben"],
        )
        == 2
    )
    assert (
        _resolve_example_parallelism_for_task(
            base_parallelism=4,
            overrides_cfg=overrides,
            model_name="gpt-5.4",
            domain="healthcare",
            datasets=["pubmedqa"],
        )
        == 4
    )
