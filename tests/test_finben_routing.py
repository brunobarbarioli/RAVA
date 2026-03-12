from __future__ import annotations

import pytest

from rava.agent.providers import BaseProvider, GenerationResult
from rava.agent.react_agent import run_agent_example
from rava.experiments.baselines import get_verification_config
from rava.specs.parser import load_spec


class _RoutingProvider(BaseProvider):
    name = "routing-test"

    def __init__(self):
        super().__init__(model="routing-test", allow_generation_fallback=False)
        self.generate_calls = 0
        self.generate_agent_calls = 0

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        self.generate_calls += 1
        return GenerationResult(
            text='{"answer":"Direct","confidence":0.77,"citations":[],"abstain":false,"abstain_reason":null}',
            confidence=0.77,
            metadata={"mode": "langchain_chat"},
        )

    def generate_agent(self, prompt: str, tools=None, system: str | None = None) -> GenerationResult:
        self.generate_agent_calls += 1
        return GenerationResult(
            text='{"answer":"Agent","confidence":0.66,"citations":[],"abstain":false,"abstain_reason":null}',
            confidence=0.66,
            metadata={"mode": "langchain_agent", "intermediate_steps": []},
        )


@pytest.mark.parametrize("backend", ["legacy_python", "langgraph"])
def test_finben_uses_direct_generation(backend: str):
    spec = load_spec("specs/finance.yaml")
    provider = _RoutingProvider()
    result = run_agent_example(
        example={
            "id": "finben-1",
            "domain": "finance",
            "task": "finben",
            "input": "Evaluate the company based on these financial metrics.",
            "reference": "",
            "metadata": {},
            "split": "test",
        },
        provider=provider,
        spec=spec,
        verification_cfg=get_verification_config("none"),
        run_id="routing-test",
        agentic_backend=backend,
    )

    assert provider.generate_calls == 1
    assert provider.generate_agent_calls == 0
    assert result["prediction"]["generation_mode"] == "langchain_chat"


class _FailingAgentProvider(BaseProvider):
    name = "failing-agent"

    def __init__(self):
        super().__init__(model="failing-agent", allow_generation_fallback=False)

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        raise AssertionError("generate() should not be used in this test")

    def generate_agent(self, prompt: str, tools=None, system: str | None = None) -> GenerationResult:
        from rava.agent.providers import ProviderGenerationError

        raise ProviderGenerationError(
            "Agent execution failed and fallback is disabled.",
            taxonomy="length_limit",
            original_error="Could not parse response content as the length limit was reached",
            error_class="LengthFinishReasonError",
            error_message="Could not parse response content as the length limit was reached",
            attempt_history=[{"attempt": 1, "success": False, "taxonomy": "length_limit"}],
            max_tokens_used=10000,
        )


def test_agent_failure_trajectory_preserves_error_metadata():
    spec = load_spec("specs/healthcare.yaml")
    provider = _FailingAgentProvider()

    with pytest.raises(Exception) as exc_info:
        run_agent_example(
            example={
                "id": "hc-1",
                "domain": "healthcare",
                "task": "pubmedqa",
                "input": "What is the best treatment?",
                "reference": "",
                "metadata": {},
                "split": "test",
            },
            provider=provider,
            spec=spec,
            verification_cfg=get_verification_config("none"),
            run_id="error-metadata-test",
            agentic_backend="langgraph",
        )

    trajectory_rows = exc_info.value.trajectory_rows
    assert trajectory_rows
    metadata = trajectory_rows[-1]["metadata"]
    assert metadata["error_taxonomy"] == "length_limit"
    assert metadata["error_class"] == "LengthFinishReasonError"
    assert metadata["error_message"].startswith("Could not parse response content")
    assert metadata["max_tokens_used"] == 10000
