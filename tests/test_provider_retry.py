from __future__ import annotations

import types

import pytest

from rava.agent.providers import (
    LangChainOllamaCloudProvider,
    ProviderGenerationError,
    classify_provider_error,
)


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class LengthFinishReasonError(RuntimeError):
    pass


class _FlakyLLM:
    def __init__(self, fail_times: int, error: Exception):
        self.fail_times = fail_times
        self.error = error
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return _FakeResponse("OK. Confidence: 0.9")


def _provider_for_test(llm: object, retry_policy: dict) -> LangChainOllamaCloudProvider:
    provider = LangChainOllamaCloudProvider(
        model="unit-test",
        allow_generation_fallback=False,
        retry_policy=retry_policy,
    )
    provider._build_llm = types.MethodType(lambda self: llm, provider)  # type: ignore[method-assign]
    return provider


def test_retry_succeeds_for_retryable_taxonomy():
    llm = _FlakyLLM(fail_times=2, error=RuntimeError("Read timeout"))
    provider = _provider_for_test(
        llm,
        retry_policy={
            "max_attempts": 4,
            "initial_backoff_s": 0.0,
            "max_backoff_s": 0.0,
            "taxonomies": ["timeout"],
        },
    )
    out = provider.generate("hello")
    assert llm.calls == 3
    assert out.metadata["attempt_count"] == 3
    attempts = out.metadata["attempt_history"]
    assert attempts[0]["taxonomy"] == "timeout"
    assert attempts[1]["taxonomy"] == "timeout"
    assert attempts[2]["success"] is True


def test_retry_does_not_retry_non_retryable_taxonomy():
    llm = _FlakyLLM(fail_times=10, error=ValueError("schema validation failed"))
    provider = _provider_for_test(
        llm,
        retry_policy={
            "max_attempts": 4,
            "initial_backoff_s": 0.0,
            "max_backoff_s": 0.0,
            "taxonomies": ["timeout"],
        },
    )
    with pytest.raises(ProviderGenerationError) as exc_info:
        provider.generate("hello")
    assert llm.calls == 1
    assert exc_info.value.taxonomy == "schema_error"


def test_classify_provider_error_maps_length_limit():
    exc = LengthFinishReasonError(
        "Could not parse response content as the length limit was reached"
    )
    assert classify_provider_error(exc) == "length_limit"


def test_classify_provider_error_uses_taxonomy_attribute():
    exc = RuntimeError("wrapped")
    setattr(exc, "taxonomy", "length_limit")
    assert classify_provider_error(exc) == "length_limit"


class _AgentStub:
    def __init__(self, *, max_tokens: int):
        self.max_tokens = max_tokens

    def invoke(self, _payload):
        if self.max_tokens < 12000:
            raise LengthFinishReasonError(
                "Could not parse response content as the length limit was reached"
            )
        return {
            "structured_response": {
                "answer": "Recovered answer",
                "confidence": 0.81,
                "citations": [],
                "abstain": False,
                "abstain_reason": None,
            },
            "messages": [],
        }


def test_generate_agent_retries_with_higher_budget_on_length_limit():
    provider = LangChainOllamaCloudProvider(
        model="unit-test",
        max_tokens=10000,
        allow_generation_fallback=False,
        retry_policy={
            "max_attempts": 1,
            "initial_backoff_s": 0.0,
            "max_backoff_s": 0.0,
            "taxonomies": ["timeout"],
        },
    )
    provider._build_llm = types.MethodType(lambda self: object(), provider)  # type: ignore[method-assign]
    provider._build_llm_uncached_with_max_tokens = types.MethodType(  # type: ignore[method-assign]
        lambda self, max_tokens: types.SimpleNamespace(max_tokens=max_tokens),
        provider,
    )
    provider._build_agent = types.MethodType(  # type: ignore[method-assign]
        lambda self, llm, tools, system_prompt: _AgentStub(max_tokens=10000),
        provider,
    )
    provider._build_agent_uncached = types.MethodType(  # type: ignore[method-assign]
        lambda self, llm, tools, system_prompt, use_cache: _AgentStub(max_tokens=int(llm.max_tokens)),
        provider,
    )
    provider._invoke_with_retry = types.MethodType(  # type: ignore[method-assign]
        lambda self, invoke_fn: (invoke_fn(), [{"attempt": 1, "success": True, "taxonomy": None}]),
        provider,
    )

    out = provider.generate_agent("hello", tools=[object()])

    assert out.text.startswith("{")
    assert out.metadata["max_tokens_used"] == 12000
    assert out.metadata["length_limit_recovered"] is True
    assert any(
        row.get("retry_reason") == "length_limit" and row.get("max_tokens_used") == 12000
        for row in out.metadata["attempt_history"]
    )
