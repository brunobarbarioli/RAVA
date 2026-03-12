from __future__ import annotations

import asyncio
import json
import os
import random
import re
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from rava.agent.output_schema import StructuredAgentOutput
from rava.utils.hashing import stable_hash

CONF_RE = re.compile(
    r"(?:['\"]?confidence['\"]?)\s*[:=]\s*([01](?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass
class GenerationResult:
    text: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryPolicy:
    max_attempts: int = 1
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 8.0
    taxonomies: set[str] = field(default_factory=lambda: {"timeout", "upstream_5xx", "rate_limited", "connection_error"})

    @classmethod
    def from_config(cls, payload: dict[str, Any] | None) -> "RetryPolicy":
        if not payload:
            return cls()
        taxonomies = payload.get("taxonomies")
        if isinstance(taxonomies, (list, tuple, set)):
            taxonomy_set = {str(x).strip() for x in taxonomies if str(x).strip()}
        else:
            taxonomy_set = {"timeout", "upstream_5xx", "rate_limited", "connection_error"}
        return cls(
            max_attempts=max(1, int(payload.get("max_attempts", 1))),
            initial_backoff_s=max(0.0, float(payload.get("initial_backoff_s", 0.5))),
            max_backoff_s=max(0.0, float(payload.get("max_backoff_s", 8.0))),
            taxonomies=taxonomy_set,
        )


class ProviderGenerationError(RuntimeError):
    """Raised when provider generation fails and fallback is disabled."""

    def __init__(
        self,
        message: str,
        taxonomy: str = "unknown",
        status_code: int | None = None,
        original_error: str | None = None,
        error_class: str | None = None,
        error_message: str | None = None,
        attempt_history: list[dict[str, Any]] | None = None,
        max_tokens_used: int | None = None,
    ):
        super().__init__(message)
        self.taxonomy = taxonomy
        self.status_code = status_code
        self.original_error = original_error
        self.error_class = error_class
        self.error_message = error_message if error_message is not None else original_error
        self.attempt_history = list(attempt_history or [])
        self.max_tokens_used = max_tokens_used


class _RetryInvocationError(RuntimeError):
    def __init__(self, original_error: Exception, attempt_history: list[dict[str, Any]]):
        super().__init__(str(original_error))
        self.original_error = original_error
        self.attempt_history = attempt_history


_PROVIDER_CACHE_LOCK = threading.Lock()
_LLM_CACHE: dict[str, Any] = {}
_AGENT_CACHE: dict[str, Any] = {}


class _ToolQueryInput(BaseModel):
    query: str = Field(description="Tool query string.")


class BaseProvider(ABC):
    name: str = "base"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        allow_generation_fallback: bool = True,
        retry_policy: RetryPolicy | dict[str, Any] | None = None,
        confidence_affine: dict[str, Any] | None = None,
        confidence_calibration_maps: dict[str, Any] | None = None,
        async_model_invocation: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.allow_generation_fallback = allow_generation_fallback
        self.async_model_invocation = bool(async_model_invocation)
        if isinstance(retry_policy, RetryPolicy):
            self.retry_policy = retry_policy
        else:
            self.retry_policy = RetryPolicy.from_config(retry_policy)
        affine = confidence_affine if isinstance(confidence_affine, dict) else {}
        self.confidence_affine = {
            "offset": float(affine.get("offset", 0.0)),
            "scale": float(affine.get("scale", 1.0)),
        }
        self.confidence_calibration_maps = (
            dict(confidence_calibration_maps)
            if isinstance(confidence_calibration_maps, dict)
            else {}
        )

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

    async def agenerate(self, prompt: str, system: str | None = None) -> GenerationResult:
        return await asyncio.to_thread(self.generate, prompt, system)

    async def agenerate_agent(
        self,
        prompt: str,
        tools: list[Any] | None = None,
        system: str | None = None,
    ) -> GenerationResult:
        return await asyncio.to_thread(self.generate_agent, prompt, tools, system)


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


def _signature_hash(payload: dict[str, Any]) -> str:
    normalized = repr(sorted(payload.items()))
    return stable_hash(normalized, n_hex=24)


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(override=False)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    return chain


def classify_provider_error(exc: BaseException) -> str:
    if isinstance(exc, ProviderGenerationError):
        return exc.taxonomy
    taxonomy = getattr(exc, "taxonomy", None)
    if isinstance(taxonomy, str) and taxonomy.strip():
        return taxonomy.strip()

    chain = _exception_chain(exc)
    text = " | ".join(str(e) for e in chain).lower()
    class_names = {e.__class__.__name__.lower() for e in chain}

    status_codes: list[int] = []
    for e in chain:
        status = getattr(e, "status_code", None)
        if isinstance(status, int):
            status_codes.append(status)
        resp = getattr(e, "response", None)
        resp_status = getattr(resp, "status_code", None)
        if isinstance(resp_status, int):
            status_codes.append(resp_status)

    invalid_model_tokens = (
        "invalid model",
        "unknown model",
        "model not found",
        "does not exist",
        "unsupported model",
        "model_not_found",
        "unknown model id",
    )
    if any(tok in text for tok in invalid_model_tokens):
        return "invalid_model"
    if any(s in {400, 404} for s in status_codes) and "model" in text:
        return "invalid_model"
    if ("api key" in text or "api_key" in text or "authentication" in text) and (
        "not set" in text or "missing" in text or "invalid" in text
    ):
        return "auth_error"

    if any(s in {401, 403} for s in status_codes):
        return "auth_error"
    if any(s == 429 for s in status_codes):
        return "rate_limited"
    if any(500 <= s < 600 for s in status_codes):
        return "upstream_5xx"
    if any(400 <= s < 500 for s in status_codes):
        return "client_4xx"

    if any(isinstance(e, socket.gaierror) for e in chain) or "nodename nor servname provided" in text:
        return "dns_error"

    timeout_tokens = ("timeout", "timed out", "read timeout")
    if any(tok in text for tok in timeout_tokens):
        return "timeout"

    length_tokens = (
        "length limit was reached",
        "finish_reason=length",
        "could not parse response content as the length limit was reached",
        "response content truncated",
    )
    if "lengthfinishreasonerror" in class_names or any(tok in text for tok in length_tokens):
        return "length_limit"

    schema_tokens = ("jsondecodeerror", "validationerror", "invalid json", "schema")
    if any(tok in text for tok in schema_tokens):
        return "schema_error"

    if "connection error" in text or "connecterror" in text:
        return "connection_error"
    return "unknown"


def provider_error_metadata(exc: BaseException, max_tokens_used: int | None = None) -> dict[str, Any]:
    if isinstance(exc, ProviderGenerationError):
        return {
            "error_taxonomy": exc.taxonomy,
            "error_class": exc.error_class or exc.__class__.__name__,
            "error_message": exc.error_message or str(exc),
            "error": exc.original_error or str(exc),
            "status_code": exc.status_code,
            "attempt_history": list(exc.attempt_history),
            "max_tokens_used": exc.max_tokens_used if exc.max_tokens_used is not None else max_tokens_used,
        }

    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            status_code = response_status
    return {
        "error_taxonomy": classify_provider_error(exc),
        "error_class": exc.__class__.__name__,
        "error_message": str(exc),
        "error": str(exc),
        "status_code": status_code,
        "attempt_history": [],
        "max_tokens_used": max_tokens_used,
    }


def provider_healthcheck(
    provider: BaseProvider,
    n_probes: int = 5,
    timeout: int | None = None,
) -> dict[str, Any]:
    _load_dotenv_if_available()
    if n_probes <= 0:
        raise ValueError("n_probes must be > 0")

    original_timeout = getattr(provider, "timeout", None)
    if timeout is not None and hasattr(provider, "timeout"):
        setattr(provider, "timeout", int(timeout))

    probes: list[dict[str, Any]] = []
    success_count = 0
    error_counts: dict[str, int] = {}
    fallback_count = 0

    try:
        for idx in range(n_probes):
            started = time.time()
            prompt = f"Healthcheck probe {idx + 1}/{n_probes}: reply with exactly 'OK'."
            try:
                result = provider.generate(prompt=prompt, system="Respond with exactly OK.")
                mode = str((result.metadata or {}).get("mode", ""))
                used_fallback = "fallback" in mode.lower()
                success = bool(str(result.text).strip()) and not used_fallback
                taxonomy = None
                error_message = None
                if used_fallback:
                    fallback_count += 1
                    taxonomy = classify_provider_error(
                        RuntimeError(str((result.metadata or {}).get("error", "fallback_response")))
                    )
                    error_message = str((result.metadata or {}).get("error", "fallback_response"))
                    error_counts[taxonomy] = error_counts.get(taxonomy, 0) + 1
                elif success:
                    success_count += 1
                probes.append(
                    {
                        "probe_id": idx + 1,
                        "success": success,
                        "used_fallback": used_fallback,
                        "mode": mode,
                        "taxonomy": taxonomy,
                        "error": error_message,
                        "latency_ms": (time.time() - started) * 1000.0,
                    }
                )
            except Exception as exc:
                taxonomy = classify_provider_error(exc)
                error_counts[taxonomy] = error_counts.get(taxonomy, 0) + 1
                probes.append(
                    {
                        "probe_id": idx + 1,
                        "success": False,
                        "used_fallback": False,
                        "mode": "exception",
                        "taxonomy": taxonomy,
                        "error": str(exc),
                        "latency_ms": (time.time() - started) * 1000.0,
                    }
                )
        success_rate = success_count / float(n_probes)
        fallback_rate = fallback_count / float(n_probes)
        return {
            "provider": provider.name,
            "model": provider.model,
            "n_probes": int(n_probes),
            "n_success": int(success_count),
            "success_rate": float(success_rate),
            "fallback_rate": float(fallback_rate),
            "error_taxonomy_counts": error_counts,
            "probes": probes,
        }
    finally:
        if timeout is not None and hasattr(provider, "timeout"):
            setattr(provider, "timeout", original_timeout)


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
        allow_generation_fallback: bool = True,
        retry_policy: RetryPolicy | dict[str, Any] | None = None,
        confidence_affine: dict[str, Any] | None = None,
        confidence_calibration_maps: dict[str, Any] | None = None,
        async_model_invocation: bool = False,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            allow_generation_fallback=allow_generation_fallback,
            retry_policy=retry_policy,
            confidence_affine=confidence_affine,
            confidence_calibration_maps=confidence_calibration_maps,
            async_model_invocation=async_model_invocation,
        )
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

    def _llm_cache_key(self) -> str:
        return _signature_hash(
            {
                "provider": self.name,
                "model": self.model,
                "temperature": float(self.temperature),
                "max_tokens": int(self.max_tokens),
                "base_url": str(self.base_url),
                "timeout": int(self.timeout),
                "api_key_env": str(self.api_key_env),
            }
        )

    @staticmethod
    def _tool_signature(tools: list[Any]) -> str:
        parts: list[str] = []
        for tool_obj in tools:
            name = str(getattr(tool_obj, "name", tool_obj.__class__.__name__))
            cache_scope = str(getattr(tool_obj, "cache_scope", "")).strip().lower()
            if cache_scope == "example":
                parts.append(f"{name}:{id(tool_obj)}")
            else:
                parts.append(name)
        return "|".join(sorted(parts))

    def _agent_cache_key(self, tools: list[Any], system_prompt: str) -> str:
        return _signature_hash(
            {
                "llm": self._llm_cache_key(),
                "tool_signature": self._tool_signature(tools),
                "system_prompt": str(system_prompt),
            }
        )

    def _build_llm_uncached(self) -> Any:
        return self._build_llm_uncached_with_max_tokens(self.max_tokens)

    def _build_llm_uncached_with_max_tokens(self, max_tokens: int) -> Any:
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
            max_tokens=int(max_tokens),
            api_key=self._get_api_key(),
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=1,
        )

    def _build_llm(self) -> Any:
        key = self._llm_cache_key()
        with _PROVIDER_CACHE_LOCK:
            cached = _LLM_CACHE.get(key)
            if cached is not None:
                return cached
            llm = self._build_llm_uncached()
            _LLM_CACHE[key] = llm
            return llm

    def _wrap_tools(self, tools: list[Any]) -> list[Any]:
        langchain_tools: list[Any] = []
        for rava_tool in tools:
            def _make_tool(tool_obj: Any) -> Any:
                try:
                    from langchain_core.tools import StructuredTool
                except Exception as exc:
                    raise RuntimeError(
                        "langchain-core tools are required for tool-enabled agent runs."
                    ) from exc

                def tool_fn(query: str) -> str:
                    result = tool_obj.run(query, context={})
                    return str(result.observation)

                tool_fn.__name__ = f"{tool_obj.name}_tool"
                tool_fn.__doc__ = (
                    getattr(tool_obj, "__doc__", None)
                    or f"RAVA tool '{tool_obj.name}' for domain retrieval/action support."
                )
                return StructuredTool.from_function(
                    func=tool_fn,
                    name=str(tool_obj.name),
                    description=str(tool_fn.__doc__ or f"RAVA tool '{tool_obj.name}'."),
                    args_schema=_ToolQueryInput,
                )

            langchain_tools.append(_make_tool(rava_tool))
        return langchain_tools

    def _build_agent(self, llm: Any, tools: list[Any], system_prompt: str) -> Any:
        return self._build_agent_uncached(llm=llm, tools=tools, system_prompt=system_prompt, use_cache=True)

    def _build_agent_uncached(
        self,
        llm: Any,
        tools: list[Any],
        system_prompt: str,
        *,
        use_cache: bool,
    ) -> Any:
        try:
            from langchain.agents import create_agent
        except Exception as exc:
            raise RuntimeError(
                "LangChain agent components unavailable. "
                "Install with: pip install langchain langchain-openai"
            ) from exc

        wrapped_tools = self._wrap_tools(tools)
        if not use_cache:
            return create_agent(
                model=llm,
                tools=wrapped_tools,
                system_prompt=system_prompt,
                response_format=StructuredAgentOutput,
            )

        key = self._agent_cache_key(tools, system_prompt)
        with _PROVIDER_CACHE_LOCK:
            cached = _AGENT_CACHE.get(key)
            if cached is not None:
                return cached
            agent = create_agent(
                model=llm,
                tools=wrapped_tools,
                system_prompt=system_prompt,
                response_format=StructuredAgentOutput,
            )
            _AGENT_CACHE[key] = agent
            return agent

    def _invoke_with_retry(self, invoke_fn: Any) -> tuple[Any, list[dict[str, Any]]]:
        max_attempts = max(1, int(self.retry_policy.max_attempts))
        attempt_history: list[dict[str, Any]] = []
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            started = time.time()
            try:
                response = invoke_fn()
                attempt_history.append(
                    {
                        "attempt": attempt,
                        "success": True,
                        "taxonomy": None,
                        "latency_ms": (time.time() - started) * 1000.0,
                    }
                )
                return response, attempt_history
            except Exception as exc:  # pragma: no cover - behavior asserted through callers
                taxonomy = classify_provider_error(exc)
                attempt_history.append(
                    {
                        "attempt": attempt,
                        "success": False,
                        "taxonomy": taxonomy,
                        "error_class": exc.__class__.__name__,
                        "error": str(exc),
                        "latency_ms": (time.time() - started) * 1000.0,
                    }
                )
                last_exc = exc
                retryable = taxonomy in self.retry_policy.taxonomies and attempt < max_attempts
                if not retryable:
                    raise _RetryInvocationError(exc, attempt_history) from exc
                base = min(
                    float(self.retry_policy.max_backoff_s),
                    float(self.retry_policy.initial_backoff_s) * (2 ** (attempt - 1)),
                )
                jitter = random.uniform(0.0, max(0.001, base * 0.25))
                time.sleep(base + jitter)

        if last_exc is not None:
            raise _RetryInvocationError(last_exc, attempt_history) from last_exc
        raise _RetryInvocationError(RuntimeError("Retry invocation failed without explicit exception."), attempt_history)

    async def _invoke_with_retry_async(self, invoke_fn: Any) -> tuple[Any, list[dict[str, Any]]]:
        max_attempts = max(1, int(self.retry_policy.max_attempts))
        attempt_history: list[dict[str, Any]] = []
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            started = time.time()
            try:
                response = await invoke_fn()
                attempt_history.append(
                    {
                        "attempt": attempt,
                        "success": True,
                        "taxonomy": None,
                        "latency_ms": (time.time() - started) * 1000.0,
                    }
                )
                return response, attempt_history
            except Exception as exc:
                taxonomy = classify_provider_error(exc)
                attempt_history.append(
                    {
                        "attempt": attempt,
                        "success": False,
                        "taxonomy": taxonomy,
                        "error_class": exc.__class__.__name__,
                        "error": str(exc),
                        "latency_ms": (time.time() - started) * 1000.0,
                    }
                )
                last_exc = exc
                retryable = taxonomy in self.retry_policy.taxonomies and attempt < max_attempts
                if not retryable:
                    raise _RetryInvocationError(exc, attempt_history) from exc
                base = min(
                    float(self.retry_policy.max_backoff_s),
                    float(self.retry_policy.initial_backoff_s) * (2 ** (attempt - 1)),
                )
                jitter = random.uniform(0.0, max(0.001, base * 0.25))
                await asyncio.sleep(base + jitter)

        if last_exc is not None:
            raise _RetryInvocationError(last_exc, attempt_history) from last_exc
        raise _RetryInvocationError(RuntimeError("Retry invocation failed without explicit exception."), attempt_history)

    @staticmethod
    def _extract_agent_payload(payload: Any) -> tuple[str, list[dict[str, Any]]]:
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        output_text = ""
        intermediate_steps = []
        pending_inputs: dict[str, str] = {}
        structured_response = payload.get("structured_response") if isinstance(payload, dict) else None

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

        if structured_response is not None:
            if hasattr(structured_response, "model_dump"):
                output_text = json.dumps(
                    structured_response.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            elif isinstance(structured_response, dict):
                output_text = json.dumps(structured_response, ensure_ascii=False, sort_keys=True)
            else:
                output_text = _as_text(structured_response)

        if not output_text:
            output_text = _as_text(payload)
        return output_text, intermediate_steps

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        if self.async_model_invocation:
            return asyncio.run(self.agenerate(prompt=prompt, system=system))

        llm = self._build_llm()
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except Exception as exc:
            raise RuntimeError("langchain-core is required for message objects.") from exc

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        try:
            response, attempt_history = self._invoke_with_retry(lambda: llm.invoke(messages))
            text = _as_text(getattr(response, "content", response))
            return GenerationResult(
                text=text,
                confidence=_parse_confidence(text),
                metadata={
                    "provider": self.name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "mode": "langchain_chat",
                    "attempt_count": len(attempt_history),
                    "attempt_history": attempt_history,
                },
            )
        except Exception as exc:
            attempt_history = []
            root_exc = exc
            if isinstance(exc, _RetryInvocationError):
                attempt_history = list(exc.attempt_history)
                root_exc = exc.original_error
            taxonomy = classify_provider_error(root_exc)
            if not self.allow_generation_fallback:
                error_meta = provider_error_metadata(root_exc, max_tokens_used=int(self.max_tokens))
                raise ProviderGenerationError(
                    "Provider generation failed and fallback is disabled.",
                    taxonomy=taxonomy,
                    status_code=error_meta.get("status_code"),
                    original_error=str(root_exc),
                    error_class=str(error_meta.get("error_class") or root_exc.__class__.__name__),
                    error_message=str(error_meta.get("error_message") or str(root_exc)),
                    attempt_history=attempt_history,
                    max_tokens_used=int(self.max_tokens),
                ) from root_exc
            fallback_text = (
                "{"
                "\"answer\":\"I cannot provide a reliable answer right now due to transient model backend issues. "
                "Please retry or use a human-reviewed workflow.\","
                "\"confidence\":0.35,"
                "\"citations\":[],"
                "\"abstain\":true,"
                "\"abstain_reason\":\"backend_unavailable\""
                "}"
            )
            return GenerationResult(
                text=fallback_text,
                confidence=0.35,
                metadata={
                    "provider": self.name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "mode": "langchain_chat_fallback",
                    "error": str(root_exc),
                    "error_taxonomy": taxonomy,
                    "error_class": root_exc.__class__.__name__,
                    "error_message": str(root_exc),
                    "max_tokens_used": int(self.max_tokens),
                    "attempt_count": len(attempt_history) if attempt_history else int(self.retry_policy.max_attempts),
                    "attempt_history": attempt_history,
                },
            )

    async def agenerate(self, prompt: str, system: str | None = None) -> GenerationResult:
        llm = self._build_llm()
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except Exception as exc:
            raise RuntimeError("langchain-core is required for message objects.") from exc

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        try:
            response, attempt_history = await self._invoke_with_retry_async(lambda: llm.ainvoke(messages))
            text = _as_text(getattr(response, "content", response))
            return GenerationResult(
                text=text,
                confidence=_parse_confidence(text),
                metadata={
                    "provider": self.name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "mode": "langchain_chat",
                    "attempt_count": len(attempt_history),
                    "attempt_history": attempt_history,
                },
            )
        except Exception as exc:
            attempt_history = []
            root_exc = exc
            if isinstance(exc, _RetryInvocationError):
                attempt_history = list(exc.attempt_history)
                root_exc = exc.original_error
            taxonomy = classify_provider_error(root_exc)
            if not self.allow_generation_fallback:
                error_meta = provider_error_metadata(root_exc, max_tokens_used=int(self.max_tokens))
                raise ProviderGenerationError(
                    "Provider generation failed and fallback is disabled.",
                    taxonomy=taxonomy,
                    status_code=error_meta.get("status_code"),
                    original_error=str(root_exc),
                    error_class=str(error_meta.get("error_class") or root_exc.__class__.__name__),
                    error_message=str(error_meta.get("error_message") or str(root_exc)),
                    attempt_history=attempt_history,
                    max_tokens_used=int(self.max_tokens),
                ) from root_exc
            fallback_text = (
                "{"
                "\"answer\":\"I cannot provide a reliable answer right now due to transient model backend issues. "
                "Please retry or use a human-reviewed workflow.\","
                "\"confidence\":0.35,"
                "\"citations\":[],"
                "\"abstain\":true,"
                "\"abstain_reason\":\"backend_unavailable\""
                "}"
            )
            return GenerationResult(
                text=fallback_text,
                confidence=0.35,
                metadata={
                    "provider": self.name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "mode": "langchain_chat_fallback",
                    "error": str(root_exc),
                    "error_taxonomy": taxonomy,
                    "error_class": root_exc.__class__.__name__,
                    "error_message": str(root_exc),
                    "max_tokens_used": int(self.max_tokens),
                    "attempt_count": len(attempt_history) if attempt_history else int(self.retry_policy.max_attempts),
                    "attempt_history": attempt_history,
                },
            )

    def _length_retry_budgets(self) -> list[int]:
        current = int(self.max_tokens)
        budgets: list[int] = []
        for candidate in (12000, 16000):
            if candidate > current and candidate not in budgets:
                budgets.append(candidate)
        return budgets

    def _invoke_agent_with_temp_budget(
        self,
        *,
        prompt: str,
        tools: list[Any],
        system_prompt: str,
        max_tokens: int,
        async_mode: bool = False,
    ) -> tuple[GenerationResult, list[dict[str, Any]]]:
        llm = self._build_llm_uncached_with_max_tokens(max_tokens)
        agent = self._build_agent_uncached(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            use_cache=False,
        )
        if async_mode:
            raise RuntimeError("_invoke_agent_with_temp_budget must use async variant.")
        payload, attempt_history = self._invoke_with_retry(
            lambda: agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        )
        output_text, intermediate_steps = self._extract_agent_payload(payload)
        return (
            GenerationResult(
                text=output_text,
                confidence=_parse_confidence(output_text),
                metadata={
                    "provider": self.name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "mode": "langchain_agent",
                    "intermediate_steps": intermediate_steps,
                    "structured_response_used": True,
                    "attempt_count": len(attempt_history),
                    "attempt_history": attempt_history,
                    "max_tokens_used": int(max_tokens),
                },
            ),
            attempt_history,
        )

    async def _ainvoke_agent_with_temp_budget(
        self,
        *,
        prompt: str,
        tools: list[Any],
        system_prompt: str,
        max_tokens: int,
    ) -> tuple[GenerationResult, list[dict[str, Any]]]:
        llm = self._build_llm_uncached_with_max_tokens(max_tokens)
        agent = self._build_agent_uncached(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            use_cache=False,
        )
        payload, attempt_history = await self._invoke_with_retry_async(
            lambda: agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
        )
        output_text, intermediate_steps = self._extract_agent_payload(payload)
        return (
            GenerationResult(
                text=output_text,
                confidence=_parse_confidence(output_text),
                metadata={
                    "provider": self.name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "mode": "langchain_agent",
                    "intermediate_steps": intermediate_steps,
                    "structured_response_used": True,
                    "attempt_count": len(attempt_history),
                    "attempt_history": attempt_history,
                    "max_tokens_used": int(max_tokens),
                },
            ),
            attempt_history,
        )

    def generate_agent(
        self,
        prompt: str,
        tools: list[Any] | None = None,
        system: str | None = None,
    ) -> GenerationResult:
        if self.async_model_invocation:
            return asyncio.run(self.agenerate_agent(prompt=prompt, tools=tools, system=system))

        llm = self._build_llm()
        tools = tools or []
        if not tools:
            return self.generate(prompt=prompt, system=system)
        system_prompt = (
            system or "You are a regulation-aware assistant. Use tools when needed and cite sources."
        )

        try:
            agent = self._build_agent(llm=llm, tools=tools, system_prompt=system_prompt)
            payload, attempt_history = self._invoke_with_retry(
                lambda: agent.invoke({"messages": [{"role": "user", "content": prompt}]})
            )
            output_text, intermediate_steps = self._extract_agent_payload(payload)

            generation = GenerationResult(
                text=output_text,
                confidence=_parse_confidence(output_text),
                metadata={
                    "provider": self.name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "mode": "langchain_agent",
                    "intermediate_steps": intermediate_steps,
                    "structured_response_used": True,
                    "attempt_count": len(attempt_history),
                    "attempt_history": attempt_history,
                    "max_tokens_used": int(self.max_tokens),
                },
            )
            return generation
        except Exception as exc:
            attempt_history = []
            root_exc = exc
            if isinstance(exc, _RetryInvocationError):
                attempt_history = list(exc.attempt_history)
                root_exc = exc.original_error
            taxonomy = classify_provider_error(root_exc)
            combined_attempt_history = list(attempt_history)
            if taxonomy == "length_limit":
                for retry_budget in self._length_retry_budgets():
                    try:
                        generation, retry_attempt_history = self._invoke_agent_with_temp_budget(
                            prompt=prompt,
                            tools=tools,
                            system_prompt=system_prompt,
                            max_tokens=retry_budget,
                        )
                        combined_attempt_history.extend(
                            [
                                {
                                    **row,
                                    "retry_reason": "length_limit",
                                    "max_tokens_used": int(retry_budget),
                                }
                                for row in retry_attempt_history
                            ]
                        )
                        generation.metadata["attempt_history"] = combined_attempt_history
                        generation.metadata["attempt_count"] = len(combined_attempt_history)
                        generation.metadata["max_tokens_used"] = int(retry_budget)
                        generation.metadata["length_limit_recovered"] = True
                        return generation
                    except Exception as retry_exc:
                        retry_meta = provider_error_metadata(retry_exc, max_tokens_used=retry_budget)
                        combined_attempt_history.append(
                            {
                                "attempt": len(combined_attempt_history) + 1,
                                "success": False,
                                "taxonomy": str(retry_meta.get("error_taxonomy", "unknown")),
                                "error_class": str(retry_meta.get("error_class", retry_exc.__class__.__name__)),
                                "error": str(retry_meta.get("error_message", str(retry_exc))),
                                "latency_ms": 0.0,
                                "retry_reason": "length_limit",
                                "max_tokens_used": int(retry_budget),
                            }
                        )
                        root_exc = retry_exc
                        taxonomy = classify_provider_error(retry_exc)
            if not self.allow_generation_fallback:
                error_meta = provider_error_metadata(root_exc, max_tokens_used=int(self.max_tokens))
                raise ProviderGenerationError(
                    "Agent execution failed and fallback is disabled.",
                    taxonomy=taxonomy,
                    status_code=error_meta.get("status_code"),
                    original_error=str(root_exc),
                    error_class=str(error_meta.get("error_class") or root_exc.__class__.__name__),
                    error_message=str(error_meta.get("error_message") or str(root_exc)),
                    attempt_history=combined_attempt_history,
                    max_tokens_used=int(error_meta.get("max_tokens_used") or self.max_tokens),
                ) from root_exc
            fallback = self.generate(prompt=prompt, system=system)
            fallback.metadata["agent_fallback_reason"] = str(root_exc)
            fallback.metadata["mode"] = "langchain_chat_fallback"
            fallback.metadata["intermediate_steps"] = []
            if combined_attempt_history:
                fallback.metadata["attempt_history"] = combined_attempt_history
            fallback.metadata["error_taxonomy"] = taxonomy
            fallback.metadata["error_class"] = root_exc.__class__.__name__
            fallback.metadata["error_message"] = str(root_exc)
            return fallback

    async def agenerate_agent(
        self,
        prompt: str,
        tools: list[Any] | None = None,
        system: str | None = None,
    ) -> GenerationResult:
        llm = self._build_llm()
        tools = tools or []
        if not tools:
            return await self.agenerate(prompt=prompt, system=system)

        system_prompt = (
            system or "You are a regulation-aware assistant. Use tools when needed and cite sources."
        )
        try:
            agent = self._build_agent(llm=llm, tools=tools, system_prompt=system_prompt)
            payload, attempt_history = await self._invoke_with_retry_async(
                lambda: agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
            )
            output_text, intermediate_steps = self._extract_agent_payload(payload)
            return GenerationResult(
                text=output_text,
                confidence=_parse_confidence(output_text),
                metadata={
                    "provider": self.name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "mode": "langchain_agent",
                    "intermediate_steps": intermediate_steps,
                    "structured_response_used": True,
                    "attempt_count": len(attempt_history),
                    "attempt_history": attempt_history,
                    "max_tokens_used": int(self.max_tokens),
                },
            )
        except Exception as exc:
            attempt_history = []
            root_exc = exc
            if isinstance(exc, _RetryInvocationError):
                attempt_history = list(exc.attempt_history)
                root_exc = exc.original_error
            taxonomy = classify_provider_error(root_exc)
            combined_attempt_history = list(attempt_history)
            if taxonomy == "length_limit":
                for retry_budget in self._length_retry_budgets():
                    try:
                        generation, retry_attempt_history = await self._ainvoke_agent_with_temp_budget(
                            prompt=prompt,
                            tools=tools,
                            system_prompt=system_prompt,
                            max_tokens=retry_budget,
                        )
                        combined_attempt_history.extend(
                            [
                                {
                                    **row,
                                    "retry_reason": "length_limit",
                                    "max_tokens_used": int(retry_budget),
                                }
                                for row in retry_attempt_history
                            ]
                        )
                        generation.metadata["attempt_history"] = combined_attempt_history
                        generation.metadata["attempt_count"] = len(combined_attempt_history)
                        generation.metadata["max_tokens_used"] = int(retry_budget)
                        generation.metadata["length_limit_recovered"] = True
                        return generation
                    except Exception as retry_exc:
                        retry_meta = provider_error_metadata(retry_exc, max_tokens_used=retry_budget)
                        combined_attempt_history.append(
                            {
                                "attempt": len(combined_attempt_history) + 1,
                                "success": False,
                                "taxonomy": str(retry_meta.get("error_taxonomy", "unknown")),
                                "error_class": str(retry_meta.get("error_class", retry_exc.__class__.__name__)),
                                "error": str(retry_meta.get("error_message", str(retry_exc))),
                                "latency_ms": 0.0,
                                "retry_reason": "length_limit",
                                "max_tokens_used": int(retry_budget),
                            }
                        )
                        root_exc = retry_exc
                        taxonomy = classify_provider_error(retry_exc)
            if not self.allow_generation_fallback:
                error_meta = provider_error_metadata(root_exc, max_tokens_used=int(self.max_tokens))
                raise ProviderGenerationError(
                    "Agent execution failed and fallback is disabled.",
                    taxonomy=taxonomy,
                    status_code=error_meta.get("status_code"),
                    original_error=str(root_exc),
                    error_class=str(error_meta.get("error_class") or root_exc.__class__.__name__),
                    error_message=str(error_meta.get("error_message") or str(root_exc)),
                    attempt_history=combined_attempt_history,
                    max_tokens_used=int(error_meta.get("max_tokens_used") or self.max_tokens),
                ) from root_exc
            fallback = await self.agenerate(prompt=prompt, system=system)
            fallback.metadata["agent_fallback_reason"] = str(root_exc)
            fallback.metadata["mode"] = "langchain_chat_fallback"
            fallback.metadata["intermediate_steps"] = []
            if combined_attempt_history:
                fallback.metadata["attempt_history"] = combined_attempt_history
            fallback.metadata["error_taxonomy"] = taxonomy
            fallback.metadata["error_class"] = root_exc.__class__.__name__
            fallback.metadata["error_message"] = str(root_exc)
            return fallback


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: int = 120,
        allow_generation_fallback: bool = True,
        retry_policy: RetryPolicy | dict[str, Any] | None = None,
        confidence_affine: dict[str, Any] | None = None,
        confidence_calibration_maps: dict[str, Any] | None = None,
        async_model_invocation: bool = False,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            allow_generation_fallback=allow_generation_fallback,
            retry_policy=retry_policy,
            confidence_affine=confidence_affine,
            confidence_calibration_maps=confidence_calibration_maps,
            async_model_invocation=async_model_invocation,
        )
        _load_dotenv_if_available()
        self.api_key_env = api_key_env
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.timeout = timeout

    def _get_api_key(self) -> str:
        _load_dotenv_if_available()
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"{self.api_key_env} is not set. Add it to .env or environment variables."
            )
        return key

    def _llm_cache_key(self) -> str:
        return _signature_hash(
            {
                "provider": self.name,
                "model": self.model,
                "temperature": float(self.temperature),
                "max_tokens": int(self.max_tokens),
                "base_url": str(self.base_url),
                "timeout": int(self.timeout),
                "api_key_env": str(self.api_key_env),
            }
        )

    @staticmethod
    def _tool_signature(tools: list[Any]) -> str:
        return LangChainOllamaCloudProvider._tool_signature(tools)

    def _agent_cache_key(self, tools: list[Any], system_prompt: str) -> str:
        return _signature_hash(
            {
                "llm": self._llm_cache_key(),
                "tool_signature": self._tool_signature(tools),
                "system_prompt": str(system_prompt),
            }
        )

    def _build_llm_uncached(self) -> Any:
        return self._build_llm_uncached_with_max_tokens(self.max_tokens)

    def _build_llm_uncached_with_max_tokens(self, max_tokens: int) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            raise RuntimeError(
                "langchain-openai is required for OpenAI provider. "
                "Install with: pip install langchain langchain-openai"
            ) from exc
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": int(max_tokens),
            "api_key": self._get_api_key(),
            "timeout": self.timeout,
            "max_retries": 1,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return ChatOpenAI(**kwargs)

    def _build_llm(self) -> Any:
        key = self._llm_cache_key()
        with _PROVIDER_CACHE_LOCK:
            cached = _LLM_CACHE.get(key)
            if cached is not None:
                return cached
            llm = self._build_llm_uncached()
            _LLM_CACHE[key] = llm
            return llm

    def _wrap_tools(self, tools: list[Any]) -> list[Any]:
        return LangChainOllamaCloudProvider._wrap_tools(self, tools)

    def _build_agent(self, llm: Any, tools: list[Any], system_prompt: str) -> Any:
        return LangChainOllamaCloudProvider._build_agent(self, llm=llm, tools=tools, system_prompt=system_prompt)

    def _build_agent_uncached(
        self,
        llm: Any,
        tools: list[Any],
        system_prompt: str,
        *,
        use_cache: bool,
    ) -> Any:
        return LangChainOllamaCloudProvider._build_agent_uncached(
            self,
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            use_cache=use_cache,
        )

    def _invoke_with_retry(self, invoke_fn: Any) -> tuple[Any, list[dict[str, Any]]]:
        return LangChainOllamaCloudProvider._invoke_with_retry(self, invoke_fn)

    async def _invoke_with_retry_async(self, invoke_fn: Any) -> tuple[Any, list[dict[str, Any]]]:
        return await LangChainOllamaCloudProvider._invoke_with_retry_async(self, invoke_fn)

    @staticmethod
    def _extract_agent_payload(payload: Any) -> tuple[str, list[dict[str, Any]]]:
        return LangChainOllamaCloudProvider._extract_agent_payload(payload)

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        return LangChainOllamaCloudProvider.generate(self, prompt=prompt, system=system)

    async def agenerate(self, prompt: str, system: str | None = None) -> GenerationResult:
        return await LangChainOllamaCloudProvider.agenerate(self, prompt=prompt, system=system)

    def generate_agent(
        self,
        prompt: str,
        tools: list[Any] | None = None,
        system: str | None = None,
    ) -> GenerationResult:
        return LangChainOllamaCloudProvider.generate_agent(self, prompt=prompt, tools=tools, system=system)

    async def agenerate_agent(
        self,
        prompt: str,
        tools: list[Any] | None = None,
        system: str | None = None,
    ) -> GenerationResult:
        return await LangChainOllamaCloudProvider.agenerate_agent(self, prompt=prompt, tools=tools, system=system)


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
        allow_generation_fallback: bool = True,
        retry_policy: RetryPolicy | dict[str, Any] | None = None,
        confidence_affine: dict[str, Any] | None = None,
        confidence_calibration_maps: dict[str, Any] | None = None,
        async_model_invocation: bool = False,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            allow_generation_fallback=allow_generation_fallback,
            retry_policy=retry_policy,
            confidence_affine=confidence_affine,
            confidence_calibration_maps=confidence_calibration_maps,
            async_model_invocation=async_model_invocation,
        )
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
    async_model_invocation = bool(config.get("async_model_invocation", False))
    allow_generation_fallback = bool(config.get("allow_generation_fallback", True))
    retry_policy = RetryPolicy.from_config(config.get("retry") if isinstance(config.get("retry"), dict) else None)
    confidence_affine = config.get("confidence_affine") if isinstance(config.get("confidence_affine"), dict) else None
    confidence_calibration_maps = (
        config.get("confidence_calibration_maps")
        if isinstance(config.get("confidence_calibration_maps"), dict)
        else None
    )

    if provider_name == "mock":
        return MockProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            allow_generation_fallback=allow_generation_fallback,
            retry_policy=retry_policy,
            confidence_affine=confidence_affine,
            confidence_calibration_maps=confidence_calibration_maps,
            async_model_invocation=async_model_invocation,
        )
    if provider_name in {"ollama_cloud", "ollama", "ollama-cloud"}:
        return LangChainOllamaCloudProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=config.get("base_url") or os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1"),
            api_key_env=str(config.get("api_key_env", "OLLAMA_API_KEY")),
            timeout=int(config.get("timeout", 90)),
            allow_generation_fallback=allow_generation_fallback,
            retry_policy=retry_policy,
            confidence_affine=confidence_affine,
            confidence_calibration_maps=confidence_calibration_maps,
            async_model_invocation=async_model_invocation,
        )
    if provider_name == "openai":
        return OpenAIProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=config.get("base_url") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key_env=str(config.get("api_key_env", "OPENAI_API_KEY")),
            timeout=int(config.get("timeout", 120)),
            allow_generation_fallback=allow_generation_fallback,
            retry_policy=retry_policy,
            confidence_affine=confidence_affine,
            confidence_calibration_maps=confidence_calibration_maps,
            async_model_invocation=async_model_invocation,
        )
    if provider_name == "anthropic":
        return AnthropicProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_policy=retry_policy,
            confidence_affine=confidence_affine,
            confidence_calibration_maps=confidence_calibration_maps,
            async_model_invocation=async_model_invocation,
        )
    if provider_name == "google":
        return GoogleProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_policy=retry_policy,
            confidence_affine=confidence_affine,
            confidence_calibration_maps=confidence_calibration_maps,
            async_model_invocation=async_model_invocation,
        )
    if provider_name in {"local_hf", "hf", "local"}:
        return LocalHFProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            model_path=config.get("model_path"),
            allow_generation_fallback=allow_generation_fallback,
            confidence_affine=confidence_affine,
            confidence_calibration_maps=confidence_calibration_maps,
            async_model_invocation=async_model_invocation,
        )

    raise ValueError(f"Unknown provider: {provider_name}")
