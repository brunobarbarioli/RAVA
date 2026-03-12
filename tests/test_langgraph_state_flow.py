from __future__ import annotations

from dataclasses import dataclass

from rava.agent.providers import GenerationResult, MockProvider
from rava.agent.react_agent import run_agent_example
from rava.experiments.baselines import get_verification_config
from rava.specs.parser import load_spec
from rava.specs.schema import VerificationAction
from rava.verification.runtime_monitor import RuntimeResult


class _ToolStepProvider(MockProvider):
    def generate_agent(self, prompt: str, tools=None, system: str | None = None) -> GenerationResult:
        return GenerationResult(
            text='{"answer":"Use caution.","confidence":0.61,"citations":[],"abstain":false}',
            confidence=0.61,
            metadata={
                "mode": "langchain_agent",
                "intermediate_steps": [
                    {
                        "tool": "price_lookup",
                        "tool_input": "AAPL",
                        "observation": "AAPL close 190.0",
                        "log": "",
                    }
                ],
            },
        )


@dataclass
class _FakeRunnable:
    result: RuntimeResult

    def invoke(self, _payload):
        return self.result


class _FakeRuntimeMonitor:
    def __init__(self, _spec):
        pass

    def as_runnable(self):
        return _FakeRunnable(
            RuntimeResult(
                action=VerificationAction.HALT,
                verdict_rows=[],
                hard_fail_ids=["RUNTIME-JUDGE"],
            )
        )


def test_langgraph_halt_path_sets_runtime_halt(monkeypatch):
    import rava.agent.langgraph_runtime as langgraph_runtime

    monkeypatch.setattr(langgraph_runtime, "RuntimeMonitor", _FakeRuntimeMonitor)

    spec = load_spec("specs/finance.yaml")
    provider = _ToolStepProvider(model="mock-v1")
    result = run_agent_example(
        example={
            "id": "ex-1",
            "domain": "finance",
            "task": "qa",
            "input": "Should I buy this stock now?",
            "reference": "",
            "metadata": {},
            "split": "test",
        },
        provider=provider,
        spec=spec,
        verification_cfg=get_verification_config("runtime"),
        run_id="unit-run",
        agentic_backend="langgraph",
    )

    assert result["prediction"]["runtime_halted_hard_fail"] is True
    actions = [row["action"] for row in result["trajectory"]]
    assert "runtime_monitor" in actions
    assert "posthoc_audit" not in actions


def test_langgraph_full_flow_emits_expected_nodes():
    spec = load_spec("specs/healthcare.yaml")
    provider = MockProvider(model="mock-v1")
    result = run_agent_example(
        example={
            "id": "ex-2",
            "domain": "healthcare",
            "task": "qa",
            "input": "What are signs of dehydration?",
            "reference": "",
            "metadata": {},
            "split": "test",
        },
        provider=provider,
        spec=spec,
        verification_cfg=get_verification_config("full"),
        run_id="unit-run",
        agentic_backend="langgraph",
    )

    actions = [row["action"] for row in result["trajectory"]]
    assert "pre_tool_check" in actions
    assert "langchain_agent_generation" in actions
    assert "pre_final_check" in actions
    assert "posthoc_audit" in actions
