from __future__ import annotations

from collections import Counter
from pathlib import Path

from rava.agent.providers import MockProvider
from rava.agent.react_agent import run_agent_example
from rava.experiments.baselines import get_verification_config
from rava.experiments.runner import run_sweep
from rava.specs.parser import load_spec
from rava.utils.serialization import read_json, write_yaml


def _result_signature(result: dict) -> dict:
    prediction = dict(result["prediction"])
    prediction.pop("timestamp", None)
    pred_subset = {
        "output": prediction.get("output"),
        "confidence": prediction.get("confidence"),
        "abstained": prediction.get("abstained"),
        "abstain_reason": prediction.get("abstain_reason"),
        "generation_mode": prediction.get("generation_mode"),
        "generation_error": prediction.get("generation_error"),
        "runtime_halted_hard_fail": prediction.get("runtime_halted_hard_fail"),
        "posthoc_repair_attempted": prediction.get("posthoc_repair_attempted"),
        "posthoc_repair_selected": prediction.get("posthoc_repair_selected"),
    }
    verdict_counts = Counter(
        (
            str(row.get("constraint_id")),
            str(row.get("constraint_type")),
            str(row.get("verdict")),
            str(row.get("layer")),
        )
        for row in result.get("verdicts", [])
    )
    trajectory_actions = [str(row.get("action")) for row in result.get("trajectory", [])]
    summary = dict(result.get("report", {}).get("summary", {}))
    return {
        "prediction": pred_subset,
        "verdict_counts": verdict_counts,
        "trajectory_actions": trajectory_actions,
        "summary": summary,
    }


def test_langgraph_matches_legacy_for_mock_example():
    spec = load_spec("specs/healthcare.yaml")
    provider = MockProvider(model="mock-v1")
    example = {
        "id": "parity-1",
        "domain": "healthcare",
        "task": "qa",
        "input": "What should I do for mild fever?",
        "reference": "",
        "metadata": {},
        "split": "test",
    }
    verification_cfg = get_verification_config("full")

    legacy = run_agent_example(
        example=example,
        provider=provider,
        spec=spec,
        verification_cfg=verification_cfg,
        run_id="parity-run",
        agentic_backend="legacy_python",
    )
    langgraph = run_agent_example(
        example=example,
        provider=provider,
        spec=spec,
        verification_cfg=verification_cfg,
        run_id="parity-run",
        agentic_backend="langgraph",
    )

    assert _result_signature(langgraph) == _result_signature(legacy)


def test_langgraph_matches_legacy_run_level_metrics(tmp_path: Path, monkeypatch):
    sweep_cfg = {
        "name": "parity_sweep",
        "domains": ["healthcare"],
        "dataset_profile": "core",
        "models": ["configs/models/mock.yaml"],
        "verification_configs": ["none", "full"],
        "seeds": [42],
        "max_examples_per_dataset": 2,
        "provider_preflight_enabled": False,
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
                "task": "medqa",
                "input": "Question 1",
                "reference": "A",
                "metadata": {},
                "split": "test",
            },
            {
                "id": "example-2",
                "domain": "healthcare",
                "task": "medqa",
                "input": "Question 2",
                "reference": "B",
                "metadata": {},
                "split": "test",
            },
        ],
    )

    legacy_root = run_sweep(
        sweep_config_path=cfg_path,
        base_config_path="configs/base.yaml",
        agentic_backend="legacy_python",
        resume_mode="fresh",
        max_concurrent_runs=1,
    )
    langgraph_root = run_sweep(
        sweep_config_path=cfg_path,
        base_config_path="configs/base.yaml",
        agentic_backend="langgraph",
        resume_mode="fresh",
        max_concurrent_runs=1,
    )

    legacy_reports = sorted(Path(legacy_root).rglob("report.json"))
    langgraph_reports = sorted(Path(langgraph_root).rglob("report.json"))
    assert len(legacy_reports) == len(langgraph_reports) > 0

    for lpath, gpath in zip(legacy_reports, langgraph_reports):
        lpayload = read_json(lpath)
        gpayload = read_json(gpath)
        lscore = dict(lpayload.get("score", {}))
        gscore = dict(gpayload.get("score", {}))
        assert lscore.get("R_audited_raw") == gscore.get("R_audited_raw")
        assert lscore.get("R_audited_certified") == gscore.get("R_audited_certified")
        assert lscore.get("tier_audited") == gscore.get("tier_audited")
        assert lscore.get("certification_status") == gscore.get("certification_status")
