from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

from rava.agent.providers import (
    _AGENT_CACHE,
    _LLM_CACHE,
    LangChainOllamaCloudProvider,
    OpenAIProvider,
    provider_healthcheck,
)


class _FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, messages):
        return SimpleNamespace(
            content='{"answer":"ok","confidence":0.73,"citations":["mock://source"],"abstain":false,"abstain_reason":null}'
        )

    async def ainvoke(self, messages):
        return self.invoke(messages)


class _InvalidModelChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, messages):
        err = RuntimeError("The model `gpt-5.4` does not exist or is not available.")
        setattr(err, "status_code", 404)
        raise err

    async def ainvoke(self, messages):
        return self.invoke(messages)


def _clear_provider_caches() -> None:
    _LLM_CACHE.clear()
    _AGENT_CACHE.clear()


def _install_fake_langchain_openai(monkeypatch, chat_cls) -> None:
    fake_module = ModuleType("langchain_openai")
    fake_module.ChatOpenAI = chat_cls
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)


def test_openai_and_ollama_generate_structured_output_parity(monkeypatch):
    _install_fake_langchain_openai(monkeypatch, _FakeChatOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    _clear_provider_caches()

    openai_provider = OpenAIProvider(model="gpt-5.4", allow_generation_fallback=False)
    ollama_provider = LangChainOllamaCloudProvider(model="ministral-3:3b-cloud", allow_generation_fallback=False)

    openai_sync = openai_provider.generate(prompt="Say ok")
    openai_async = asyncio.run(openai_provider.agenerate(prompt="Say ok"))
    ollama_sync = ollama_provider.generate(prompt="Say ok")

    assert openai_sync.confidence == 0.73
    assert openai_async.confidence == 0.73
    assert ollama_sync.confidence == 0.73
    assert openai_sync.text == openai_async.text == ollama_sync.text
    assert openai_sync.metadata["mode"] == "langchain_chat"
    assert ollama_sync.metadata["mode"] == "langchain_chat"


def test_openai_preflight_reports_invalid_model(monkeypatch):
    _install_fake_langchain_openai(monkeypatch, _InvalidModelChatOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _clear_provider_caches()

    provider = OpenAIProvider(model="gpt-5.4", allow_generation_fallback=False)
    report = provider_healthcheck(provider=provider, n_probes=2, timeout=5)

    assert report["success_rate"] == 0.0
    assert report["error_taxonomy_counts"].get("invalid_model", 0) == 2
