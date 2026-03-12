import socket

from rava.agent.providers import BaseProvider, GenerationResult, provider_healthcheck
from rava.metrics.run_quality import assess_run_quality_for_model_comparison, compute_run_quality_metrics


def test_run_quality_gate():
    rows = [
        {"action": "langchain_agent_generation", "metadata": {"mode": "langchain_chat"}},
        {"action": "langchain_agent_generation", "metadata": {"mode": "langchain_chat_fallback", "error": "timeout"}},
    ]
    metrics = compute_run_quality_metrics(rows)
    assert metrics["generation_fallback_rate"] == 0.5
    gate = assess_run_quality_for_model_comparison(metrics, max_fallback_rate=0.1, max_api_failure_rate=0.1)
    assert gate["valid_for_model_comparison"] is False


def test_run_quality_includes_runner_level_errors():
    metrics = compute_run_quality_metrics(
        trajectory_rows=[],
        runner_generation_calls=10,
        runner_generation_error_count=2,
        runner_error_taxonomy_counts={"timeout": 2},
    )
    assert metrics["model_generation_calls"] == 10.0
    assert metrics["generation_error_count"] == 2.0
    assert metrics["api_failure_rate"] == 0.2
    assert metrics["api_failure_taxonomy_counts"]["timeout"] == 2


class _HealthyProvider(BaseProvider):
    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        return GenerationResult(text="OK", confidence=1.0, metadata={"mode": "langchain_chat"})


class _DnsFailProvider(BaseProvider):
    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        raise RuntimeError("Connection error.") from socket.gaierror(8, "nodename nor servname provided, or not known")


def test_provider_healthcheck_success():
    provider = _HealthyProvider(model="unit-test")
    report = provider_healthcheck(provider, n_probes=3, timeout=5)
    assert report["success_rate"] == 1.0
    assert report["error_taxonomy_counts"] == {}


def test_provider_healthcheck_dns_failure_taxonomy():
    provider = _DnsFailProvider(model="unit-test")
    report = provider_healthcheck(provider, n_probes=2, timeout=5)
    assert report["success_rate"] == 0.0
    assert report["error_taxonomy_counts"].get("dns_error", 0) == 2
