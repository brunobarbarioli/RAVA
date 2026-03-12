from pathlib import Path

from rava.experiments.paper_artifacts import make_paper_artifacts


def test_make_paper_artifacts_writes_expected_files(tmp_path: Path):
    outputs = make_paper_artifacts(
        output_dir=tmp_path,
        base_config_path="configs/base.yaml",
        sweep_config_paths=["configs/experiments/primary_certification_ollama.yaml"],
        model_config_paths=["configs/models/mock.yaml"],
        compliance_summary_path=None,
    )
    output_names = {path.name for path in outputs}
    assert output_names == {
        "appendix_specs.tex",
        "implementation_facts.tex",
        "compliance_challenges.tex",
    }
    for path in outputs:
        assert path.exists()

    appendix_specs = (tmp_path / "appendix_specs.tex").read_text(encoding="utf-8")
    assert "HC-$\\Sigma$1" in appendix_specs
    assert "FI-$\\Sigma$1" in appendix_specs
    assert "HR-$\\Sigma$3" in appendix_specs
    assert "HC-Σ1" not in appendix_specs
