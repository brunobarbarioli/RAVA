from __future__ import annotations

import sys
import types

import rava.agent.providers as providers_mod
from rava.agent.providers import LangChainOllamaCloudProvider


class _DummyTool:
    name = "dummy_tool"

    def run(self, query: str, context=None):  # pragma: no cover - not used directly in cache tests
        return types.SimpleNamespace(observation=f"obs:{query}", metadata={})


def test_llm_cache_reuses_client(monkeypatch):
    providers_mod._LLM_CACHE.clear()
    providers_mod._AGENT_CACHE.clear()
    provider = LangChainOllamaCloudProvider(model="cache-test")
    calls = {"n": 0}

    def _fake_build_llm_uncached(self):
        calls["n"] += 1
        return object()

    monkeypatch.setattr(provider, "_build_llm_uncached", types.MethodType(_fake_build_llm_uncached, provider))
    first = provider._build_llm()
    second = provider._build_llm()
    assert first is second
    assert calls["n"] == 1


def test_agent_cache_reuses_compiled_agent(monkeypatch):
    providers_mod._LLM_CACHE.clear()
    providers_mod._AGENT_CACHE.clear()
    provider = LangChainOllamaCloudProvider(model="agent-cache-test")
    llm = object()
    calls = {"n": 0}

    def _fake_create_agent(*, model, tools, system_prompt, response_format=None, **kwargs):
        calls["n"] += 1
        return {
            "model": model,
            "tools": tools,
            "system_prompt": system_prompt,
            "response_format": response_format,
        }
    fake_langchain = types.ModuleType("langchain")
    fake_langchain_agents = types.ModuleType("langchain.agents")
    fake_langchain_agents.create_agent = _fake_create_agent
    fake_langchain.agents = fake_langchain_agents
    monkeypatch.setitem(sys.modules, "langchain", fake_langchain)
    monkeypatch.setitem(sys.modules, "langchain.agents", fake_langchain_agents)

    tools = [_DummyTool()]
    first = provider._build_agent(llm=llm, tools=tools, system_prompt="sys")
    second = provider._build_agent(llm=llm, tools=tools, system_prompt="sys")
    assert first is second
    assert calls["n"] == 1
    assert first["response_format"] is not None


def test_extract_agent_payload_prefers_structured_response():
    payload = {
        "messages": [],
        "structured_response": {
            "answer": "Use caution.",
            "confidence": 0.61,
            "citations": ["mock://source"],
            "abstain": False,
            "abstain_reason": None,
        },
    }
    text, steps = LangChainOllamaCloudProvider._extract_agent_payload(payload)
    assert '"answer": "Use caution."' in text
    assert '"confidence": 0.61' in text
    assert steps == []
