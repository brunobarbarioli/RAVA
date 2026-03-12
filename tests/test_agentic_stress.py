from __future__ import annotations

from pathlib import Path

from rava.experiments.agentic_stress import generate_agentic_stress_hr_dataset
from rava.utils.serialization import read_jsonl


def test_agentic_stress_generation_deterministic(tmp_path: Path):
    out_a = generate_agentic_stress_hr_dataset(tmp_path / "a.jsonl", n=12, seed=7)
    out_b = generate_agentic_stress_hr_dataset(tmp_path / "b.jsonl", n=12, seed=7)
    rows_a = read_jsonl(out_a)
    rows_b = read_jsonl(out_b)
    assert rows_a == rows_b


def test_agentic_stress_schema(tmp_path: Path):
    out = generate_agentic_stress_hr_dataset(tmp_path / "stress.jsonl", n=8, seed=1)
    rows = read_jsonl(out)
    assert len(rows) == 8
    sample = rows[0]
    assert set(["id", "domain", "task", "input", "reference", "metadata", "split"]).issubset(sample.keys())
    meta = sample["metadata"]
    assert "protected_attributes" in meta
    assert "scenario_type" in meta
    assert "generator_trace" in meta
    assert "critic_verdict" in meta
