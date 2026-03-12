from pathlib import Path

import pytest

from rava.experiments.datasets import download_dataset, get_datasets_for_domain, preprocess_dataset
from rava.utils.serialization import read_json


def test_preprocess_rejects_toy_when_disallowed(tmp_path: Path):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    raw_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError):
        preprocess_dataset(
            "medqa",
            root_raw=raw_root,
            root_processed=processed_root,
            disallow_toy_fallback=True,
        )


def test_preprocess_writes_manifest_with_toy_source(tmp_path: Path):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    raw_root.mkdir(parents=True, exist_ok=True)

    out = preprocess_dataset(
        "medqa",
        root_raw=raw_root,
        root_processed=processed_root,
        disallow_toy_fallback=False,
    )
    manifest = read_json(out.parent / "manifest.json")
    assert manifest["source_type"] == "toy"
    assert manifest["total_rows"] >= 1


def test_paper_hybrid_download_fails_when_mirror_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def _fake_download_hf(dataset: str, hf_id: str, raw_dir: Path, hf_config: str | None = None) -> dict:
        return {
            "source_type": "byo",
            "hf_dataset_id": hf_id,
            "hf_revision": None,
            "hf_config": hf_config,
            "download_status": "hf_download_failed",
            "download_error": "404",
        }

    monkeypatch.setattr("rava.experiments.datasets._download_hf", _fake_download_hf)

    with pytest.raises(RuntimeError, match="requires mirror dataset"):
        download_dataset("medqa", root=tmp_path / "raw", profile="paper_hybrid")


def test_paper6_fast_download_fails_when_mirror_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def _fake_download_hf(dataset: str, hf_id: str, raw_dir: Path, hf_config: str | None = None) -> dict:
        return {
            "source_type": "byo",
            "hf_dataset_id": hf_id,
            "hf_revision": None,
            "hf_config": hf_config,
            "download_status": "hf_download_failed",
            "download_error": "404",
        }

    monkeypatch.setattr("rava.experiments.datasets._download_hf", _fake_download_hf)

    with pytest.raises(RuntimeError, match="requires mirror dataset"):
        download_dataset("bias_in_bios", root=tmp_path / "raw", profile="paper6_fast")


def test_final_a6_profile_resolves_expected_datasets():
    assert get_datasets_for_domain("healthcare", profile="final_a6") == ["pubmedqa", "medqa"]
    assert get_datasets_for_domain("finance", profile="final_a6") == ["convfinqa", "finben"]
    assert get_datasets_for_domain("hr", profile="final_a6") == ["bias_in_bios", "winobias"]
