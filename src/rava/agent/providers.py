from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rava.utils.hashing import stable_hash

CONF_RE = re.compile(r"confidence\s*[:=]\s*([01](?:\.\d+)?)", re.IGNORECASE)


@dataclass
class GenerationResult:
    text: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    name: str = "base"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 256):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        raise NotImplementedError

    def generate_agent(
        self,
        prompt: str,
        tools: list[Any] | None = None,
        system: str | None = None,
    ) -> GenerationResult:
        # Default fallback for providers that do not expose a tool-enabled agent runtime.
        return self.generate(prompt=prompt, system=system)


def _parse_confidence(text: str) -> float | None:
    match = CONF_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return min(1.0, max(0.0, value))


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text") or item.get("content")
                if txt:
                    parts.append(str(txt))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content)


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(override=False)


class MockProvider(BaseProvider):
    name = "mock"

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        h = stable_hash(prompt, n_hex=8)
        text = (
            "Disclaimer: This is informational and not professional advice. "
            f"Mock response ({h}) with citation (source: mock://knowledge-base). "
            "Confidence: 0.73"
        )
        return GenerationResult(text=text, confidence=0.73, metadata={"provider": self.name})


class LangChainOllamaCloudProvider(BaseProvider):
    """LangChain-backed provider for Ollama Cloud (OpenAI-compatible API)."""

    name = "ollama_cloud"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        base_url: str | None = None,
        api_key_env: str = "OLLAMA_API_KEY",
        timeout: int = 90,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        _load_dotenv_if_available()
        self.api_key_env = api_key_env
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
        self.timeout = timeout

    def _get_api_key(self) -> str:
        _load_dotenv_if_available()
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"{self.api_key_env} is not set. Add it to .env or environment variables."
            )
        return key

    def _build_llm(self) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            raise RuntimeError(
                "langchain-openai is required for Ollama Cloud provider. "
                "Install with: pip install langchain langchain-openai"
            ) from exc

        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self._get_api_key(),
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        llm = self._build_llm()
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except Exception as exc:
            raise RuntimeError("langchain-core is required for message objects.") from exc

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        response = llm.invoke(messages)
        text = _as_text(getattr(response, "content", response))
        return GenerationResult(
            text=text,
            confidence=_parse_confidence(text),
            metadata={
                "provider": self.name,
                "model": self.model,
                "base_url": self.base_url,
                "mode": "langchain_chat",
            },
        )

    def generate_agent(
        self,
        prompt: str,
        tools: list[Any] | None = None,
        system: str | None = None,
    ) -> GenerationResult:
        llm = self._build_llm()
        tools = tools or []

        # If no tools are provided, default to chat completion.
        if not tools:
            return self.generate(prompt=prompt, system=system)

        try:
            from langchain.agents import create_agent
        except Exception as exc:
            raise RuntimeError(
                "LangChain agent components unavailable. "
                "Install with: pip install langchain langchain-openai"
            ) from exc

        langchain_tools: list[Any] = []
        for rava_tool in tools:
            def _make_tool_fn(tool_obj: Any) -> Any:
                def tool_fn(query: str) -> str:
                    """Run a RAVA tool and return its textual observation."""
                    result = tool_obj.run(query, context={})
                    return result.observation

                tool_fn.__name__ = f"{tool_obj.name}_tool"
                tool_fn.__doc__ = (
                    f"RAVA tool '{tool_obj.name}' for domain retrieval/action support."
                )
                return tool_fn

            langchain_tools.append(_make_tool_fn(rava_tool))

        try:
            agent = create_agent(
                model=llm,
                tools=langchain_tools,
                system_prompt=(
                    system
                    or "You are a regulation-aware assistant. Use tools when needed and cite sources."
                ),
            )
            payload = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

            messages = payload.get("messages", []) if isinstance(payload, dict) else []
            output_text = ""
            intermediate_steps = []
            pending_inputs: dict[str, str] = {}

            for msg in messages:
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            name = str(tc.get("name", "tool"))
                            args = tc.get("args", "")
                        else:
                            name = str(getattr(tc, "name", "tool"))
                            args = getattr(tc, "args", "")
                        pending_inputs[name] = str(args)

                msg_type = msg.__class__.__name__
                if msg_type == "ToolMessage":
                    name = str(getattr(msg, "name", "tool"))
                    intermediate_steps.append(
                        {
                            "tool": name,
                            "tool_input": pending_inputs.get(name, ""),
                            "log": "",
                            "observation": _as_text(getattr(msg, "content", "")),
                        }
                    )

                if msg_type == "AIMessage":
                    candidate = _as_text(getattr(msg, "content", ""))
                    if candidate.strip():
                        output_text = candidate

            if not output_text:
                output_text = _as_text(payload)

            generation = GenerationResult(
                text=output_text,
                confidence=_parse_confidence(output_text),
                metadata={
                    "provider": self.name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "mode": "langchain_agent",
                    "intermediate_steps": intermediate_steps,
                },
            )
            return generation
        except Exception as exc:
            # Fallback to plain chat if agent execution fails (e.g., model lacks tool-calling support).
            fallback = self.generate(prompt=prompt, system=system)
            fallback.metadata["agent_fallback_reason"] = str(exc)
            fallback.metadata["mode"] = "langchain_chat_fallback"
            fallback.metadata["intermediate_steps"] = []
            return fallback


class OpenAIProvider(BaseProvider):
    name = "openai"

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return GenerationResult(
            text="OpenAIProvider placeholder output. Set up SDK integration.",
            confidence=None,
            metadata={"provider": self.name, "model": self.model},
        )


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        return GenerationResult(
            text="AnthropicProvider placeholder output. Set up SDK integration.",
            confidence=None,
            metadata={"provider": self.name, "model": self.model},
        )


class GoogleProvider(BaseProvider):
    name = "google"

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        return GenerationResult(
            text="GoogleProvider placeholder output. Set up SDK integration.",
            confidence=None,
            metadata={"provider": self.name, "model": self.model},
        )


class LocalHFProvider(BaseProvider):
    name = "local_hf"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        model_path: str | None = None,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.model_path = model_path or model

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        return GenerationResult(
            text=(
                "LocalHFProvider placeholder output. "
                "Install transformers/torch and implement model inference."
            ),
            confidence=None,
            metadata={"provider": self.name, "model_path": self.model_path},
        )


def build_provider(config: dict[str, Any]) -> BaseProvider:
    _load_dotenv_if_available()
    provider_name = str(config.get("provider") or config.get("name") or "mock").lower()
    model = str(config.get("model") or config.get("name") or "mock-v1")
    temperature = float(config.get("temperature", 0.0))
    max_tokens = int(config.get("max_tokens", 256))

    if provider_name == "mock":
        return MockProvider(model=model, temperature=temperature, max_tokens=max_tokens)
    if provider_name in {"ollama_cloud", "ollama", "ollama-cloud"}:
        return LangChainOllamaCloudProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=config.get("base_url") or os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1"),
            api_key_env=str(config.get("api_key_env", "OLLAMA_API_KEY")),
            timeout=int(config.get("timeout", 90)),
        )
    if provider_name == "openai":
        return OpenAIProvider(model=model, temperature=temperature, max_tokens=max_tokens)
    if provider_name == "anthropic":
        return AnthropicProvider(model=model, temperature=temperature, max_tokens=max_tokens)
    if provider_name == "google":
        return GoogleProvider(model=model, temperature=temperature, max_tokens=max_tokens)
    if provider_name in {"local_hf", "hf", "local"}:
        return LocalHFProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            model_path=config.get("model_path"),
        )

    raise ValueError(f"Unknown provider: {provider_name}")
