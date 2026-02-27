import json
import os
import subprocess
import sys
from pathlib import Path


def _env_with_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{existing}" if existing else src_path
    return env


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "rava.cli", "--help"],
        capture_output=True,
        text=True,
        env=_env_with_pythonpath(),
    )
    assert result.returncode == 0
    assert "RAVA CLI" in result.stdout


def test_cli_tiny_mock_run(tmp_path: Path):
    out = tmp_path / "single_run.json"
    cmd = [
        sys.executable,
        "-m",
        "rava.cli",
        "run_agent",
        "--domain",
        "healthcare",
        "--model-config",
        "configs/models/mock.yaml",
        "--verification-config",
        "full",
        "--output-path",
        str(out),
        "What should I do for severe chest pain?",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_pythonpath())
    assert result.returncode == 0, result.stderr
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "prediction" in payload
    assert payload["prediction"]["domain"] == "healthcare"
