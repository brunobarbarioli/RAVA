from pathlib import Path

import rava.cli as cli_mod
from rava.utils.serialization import write_json, write_yaml


def test_run_experiment_skips_canary_when_passing_summary_exists(tmp_path: Path, monkeypatch):
    resume_root = tmp_path / "resume_root"
    resume_root.mkdir(parents=True, exist_ok=True)
    write_json(
        resume_root / "canary_quality_summary.json",
        {
            "valid_run_rate": 1.0,
            "api_failure_rate": 0.0,
            "fallback_rate": 0.0,
        },
    )
    sweep_cfg = {
        "name": "skip_canary",
        "domains": ["healthcare"],
        "dataset_profile": "primary_certification",
        "models": ["configs/models/mock.yaml"],
        "verification_configs": ["none"],
        "seeds": [42],
        "provider_preflight_enabled": False,
        "canary_enabled": True,
        "resume_root": str(resume_root),
    }
    sweep_path = tmp_path / "sweep.yaml"
    write_yaml(sweep_path, sweep_cfg)

    calls: list[str] = []

    def _fake_run_sweep(*args, **kwargs):
        calls.append(str(kwargs.get("stage", "full")))
        return resume_root

    monkeypatch.setattr(cli_mod, "run_sweep", _fake_run_sweep)
    monkeypatch.setattr(cli_mod, "capture_environment", lambda *args, **kwargs: None)

    cli_mod.cmd_run_experiment(
        sweep_config=str(sweep_path),
        base_config="configs/base.yaml",
        stage="full",
    )
    assert calls == ["full"]
