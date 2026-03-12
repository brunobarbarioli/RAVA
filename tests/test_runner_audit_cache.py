from pathlib import Path

import rava.experiments.runner as runner_mod
from rava.specs.parser import load_spec


def test_audit_cache_reuses_posthoc_report(tmp_path: Path, monkeypatch):
    spec = load_spec("specs/healthcare.yaml")
    predictions = [
        {
            "id": "ex-1",
            "input": "What should I do?",
            "output": "You definitely have pneumonia.",
            "retrieval_context": "",
        }
    ]
    calls = {"n": 0}
    real_audit = runner_mod.PostHocAuditor.audit

    def _counting_audit(self, *args, **kwargs):
        calls["n"] += 1
        return real_audit(self, *args, **kwargs)

    monkeypatch.setattr(runner_mod.PostHocAuditor, "audit", _counting_audit)
    cache_dir = tmp_path / "audit_cache"
    rows_first = runner_mod._audit_predictions_with_posthoc(predictions, spec, cache_dir=cache_dir)
    rows_second = runner_mod._audit_predictions_with_posthoc(predictions, spec, cache_dir=cache_dir)
    assert rows_first == rows_second
    assert calls["n"] == 1
