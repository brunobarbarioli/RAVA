from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, TypedDict

from rava.agent.output_schema import parse_structured_output, render_final_text, structured_output_instructions
from rava.agent.providers import BaseProvider, classify_provider_error, provider_error_metadata
from rava.agent.trajectory import AgentTrajectory, TrajectoryStep
from rava.experiments.baselines import VerificationConfig
from rava.specs.schema import Specification, VerificationAction
from rava.tools import Tool
from rava.utils.timing import timed
from rava.verification.posthoc_audit import PostHocAuditor
from rava.verification.pre_execution import PreExecutionVerifier, build_repair_prompt
from rava.verification.runtime_monitor import RuntimeMonitor

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - fallback path when langgraph is unavailable
    END = "__end__"
    START = "__start__"
    StateGraph = None  # type: ignore[assignment]


class AgentGraphState(TypedDict, total=False):
    example: dict[str, Any]
    provider: BaseProvider
    spec: Specification
    verification_cfg: VerificationConfig
    run_id: str
    tool: Tool
    tool_cache_scope: str
    max_tool_iterations: int
    example_id: str
    domain: str
    input_text: str
    tool_query: str
    trajectory: AgentTrajectory
    verdict_rows: list[dict[str, Any]]
    action_index: int
    pre: PreExecutionVerifier
    runtime: RuntimeMonitor
    posthoc: PostHocAuditor
    pre_runnable: Any
    runtime_runnable: Any
    posthoc_runnable: Any
    agent_prompt: str
    result_text: str
    result_confidence: float | None
    result_metadata: dict[str, Any]
    intermediate_steps: list[dict[str, Any]]
    retrieval_context: str
    generation_exception: Exception | None
    generation_exception_taxonomy: str | None
    generation_exception_message: str | None
    halted: bool
    halted_report: dict[str, Any]
    final_output: str
    confidence: float | None
    abstained: bool
    abstain_reason: str | None
    output_citations: list[str]
    generation_mode: str
    generation_error: str | None
    report: dict[str, Any]
    prediction: dict[str, Any]
    posthoc_repair_attempted: bool
    posthoc_repair_selected: bool
    posthoc_initial_hard_failures: int
    posthoc_repaired_hard_failures: int | None
    selected_iteration: int
    should_pre_repair: bool
    pre_violated_ids: list[str]
    should_posthoc_repair: bool
    repaired_output: str | None
    repaired_confidence: float | None
    repaired_citations: list[str]
    repaired_mode: str | None
    repaired_error: str | None
    repaired_abstained: bool
    repaired_abstain_reason: str | None
    invalid_backend: bool
    invalid_backend_message: str | None


@dataclass
class _MemoizedTool(Tool):
    base: Tool
    cache: dict[tuple[str, str], Any]
    cache_scope: str = "example"

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.base.name

    def run(self, query: str, context: dict[str, Any] | None = None):  # type: ignore[override]
        normalized = " ".join(str(query).strip().lower().split())
        key = (self.base.name, normalized)
        if key in self.cache:
            cached = copy.deepcopy(self.cache[key])
            if isinstance(getattr(cached, "metadata", None), dict):
                cached.metadata["cache_hit"] = True
            return cached
        result = self.base.run(query=query, context=context)
        self.cache[key] = copy.deepcopy(result)
        return result


def _tag_rows(state: AgentGraphState, rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    action_index = int(state.get("action_index", 0)) + 1
    state["action_index"] = action_index
    example_id = str(state.get("example_id", "unknown"))
    action_key = f"{example_id}:{label}:{action_index}"
    for row in rows:
        row["record_id"] = example_id
        row["action_key"] = action_key
    return rows


def _add_step(
    state: AgentGraphState,
    *,
    phase: str,
    action: str,
    observation: str,
    started_at: float,
    ended_at: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    trajectory = state["trajectory"]
    trajectory.add_step(
        TrajectoryStep(
            step_id=len(trajectory.steps) + 1,
            phase=phase,
            action=action,
            observation=observation,
            started_at=started_at,
            ended_at=ended_at,
            metadata=metadata or {},
        )
    )


def _pre_tool_check_node(state: AgentGraphState) -> AgentGraphState:
    if state["verification_cfg"].pre:
        with timed("pre_tool") as t:
            pre_res = state["pre_runnable"].invoke(
                {
                    "event": "tool_call",
                    "text": f"TOOL:{state['tool'].name} QUERY:{state['tool_query']}",
                    "context": {"input": state["input_text"]},
                }
            )
        _add_step(
            state,
            phase="pre",
            action="pre_tool_check",
            observation=pre_res.action.value,
            started_at=t.started_at,
            ended_at=t.ended_at,
            metadata={"violations": pre_res.violated_constraint_ids},
        )
        state["verdict_rows"].extend(_tag_rows(state, pre_res.verdict_rows, "pre_tool"))
        if pre_res.action in {VerificationAction.MODIFY, VerificationAction.BLOCK}:
            from rava.agent.react_agent import _sanitize_query

            state["tool_query"] = _sanitize_query(str(state["tool_query"]))
    return state


def _agent_generate_node(state: AgentGraphState) -> AgentGraphState:
    from rava.agent.react_agent import (
        AgentGenerationFailedError,
        _example_dataset_name,
        _provider_generate,
        _provider_generate_agent,
        _should_use_direct_generation,
    )

    dataset_name = _example_dataset_name(state.get("example", {}) or {})
    use_direct_generation = _should_use_direct_generation(
        domain=str(state.get("domain", "")),
        dataset_name=dataset_name,
    )
    generation_action = "langchain_chat_generation" if use_direct_generation else "langchain_agent_generation"

    try:
        with timed("model_agent_generate") as t_model:
            if use_direct_generation:
                result = _provider_generate(
                    provider=state["provider"],
                    prompt=state["agent_prompt"],
                )
            else:
                result = _provider_generate_agent(
                    provider=state["provider"],
                    prompt=state["agent_prompt"],
                    tools=[state["tool"]],
                )
    except Exception as exc:
        now = time.time()
        error_meta = provider_error_metadata(exc, max_tokens_used=getattr(state["provider"], "max_tokens", None))
        _add_step(
            state,
            phase="agent",
            action=generation_action,
            observation="GENERATION_EXCEPTION",
            started_at=now,
            ended_at=now,
            metadata=dict(
                {
                    "mode": "langchain_chat_exception" if use_direct_generation else "langchain_agent_exception",
                },
                **error_meta,
            ),
        )
        state["generation_exception"] = AgentGenerationFailedError(
            "Agent generation failed before completion.",
            trajectory_rows=state["trajectory"].to_rows(),
            verdict_rows=state["verdict_rows"],
            taxonomy=str(error_meta.get("error_taxonomy", classify_provider_error(exc))),
            original_error=str(exc),
        )
        state["generation_exception_taxonomy"] = str(error_meta.get("error_taxonomy", classify_provider_error(exc)))
        state["generation_exception_message"] = str(exc)
        return state

    _add_step(
        state,
        phase="agent",
        action=generation_action,
        observation=result.text,
        started_at=t_model.started_at,
        ended_at=t_model.ended_at,
        metadata=result.metadata,
    )
    state["result_text"] = result.text
    state["result_confidence"] = result.confidence
    state["result_metadata"] = result.metadata if isinstance(result.metadata, dict) else {}
    state["intermediate_steps"] = list(state["result_metadata"].get("intermediate_steps", []))
    return state


def _runtime_monitor_loop_node(state: AgentGraphState) -> AgentGraphState:
    if state.get("generation_exception") is not None:
        return state

    retrieval: list[str] = []
    max_steps = max(1, int(state.get("max_tool_iterations", 64)))
    steps = list(state.get("intermediate_steps", []))[:max_steps]
    for step in steps:
        action_text = f"TOOL:{step.get('tool', 'unknown')} QUERY:{step.get('tool_input', '')}"
        observation_text = str(step.get("observation", ""))
        retrieval.append(observation_text)

        now = time.time()
        _add_step(
            state,
            phase="agent",
            action=action_text,
            observation=observation_text,
            started_at=now,
            ended_at=now,
            metadata={"log": step.get("log", "")},
        )

        if state["verification_cfg"].runtime:
            with timed("runtime") as t_rt:
                rt_res = state["runtime_runnable"].invoke(
                    {
                        "action_text": action_text,
                        "observation_text": observation_text,
                        "context": {"input": state["input_text"]},
                        "event": "tool_call",
                    }
                )
            _add_step(
                state,
                phase="runtime",
                action="runtime_monitor",
                observation=rt_res.action.value,
                started_at=t_rt.started_at,
                ended_at=t_rt.ended_at,
                metadata={"hard_fail_ids": rt_res.hard_fail_ids},
            )
            state["verdict_rows"].extend(_tag_rows(state, rt_res.verdict_rows, "runtime"))

            if rt_res.action == VerificationAction.HALT:
                state["halted"] = True
                state["final_output"] = "Request halted due to runtime HARD-constraint failure."
                state["report"] = {"summary": {"halted": True}, "constraint_verdicts": []}
                state["abstained"] = True
                state["abstain_reason"] = "runtime_hard_fail_halt"
                state["confidence"] = None
                state["generation_mode"] = "runtime_halt"
                state["generation_error"] = None
                break

    state["retrieval_context"] = "\n".join([x for x in retrieval if x]).strip()
    return state


def _pre_final_check_node(state: AgentGraphState) -> AgentGraphState:
    if state.get("generation_exception") is not None or bool(state.get("halted", False)):
        return state

    parsed = parse_structured_output(str(state.get("result_text", "")))
    state["final_output"] = render_final_text(parsed)
    state["confidence"] = (
        parsed.confidence if parsed.confidence is not None else state.get("result_confidence")
    )
    state["abstained"] = bool(parsed.abstain)
    state["abstain_reason"] = parsed.abstain_reason
    state["output_citations"] = list(parsed.citations)
    metadata = state.get("result_metadata", {})
    state["generation_mode"] = str(metadata.get("mode", "")) if isinstance(metadata, dict) else ""
    state["generation_error"] = (
        str(metadata.get("error", ""))
        if isinstance(metadata, dict) and metadata.get("error")
        else None
    )

    state["should_pre_repair"] = False
    state["pre_violated_ids"] = []
    if state["verification_cfg"].pre:
        with timed("pre_final") as t_pf:
            pre_final = state["pre_runnable"].invoke(
                {
                    "event": "final_answer",
                    "text": state["final_output"],
                    "context": {"input": state["input_text"]},
                }
            )
        _add_step(
            state,
            phase="pre",
            action="pre_final_check",
            observation=pre_final.action.value,
            started_at=t_pf.started_at,
            ended_at=t_pf.ended_at,
            metadata={"violations": pre_final.violated_constraint_ids},
        )
        state["verdict_rows"].extend(_tag_rows(state, pre_final.verdict_rows, "pre_final"))
        state["pre_violated_ids"] = list(pre_final.violated_constraint_ids)
        if pre_final.action in {VerificationAction.MODIFY, VerificationAction.BLOCK}:
            state["should_pre_repair"] = True
    return state


def _pre_repair_generation_node(state: AgentGraphState) -> AgentGraphState:
    if not bool(state.get("should_pre_repair", False)):
        return state

    from rava.agent.react_agent import _build_domain_repair_prompt, _example_dataset_name

    repair_prompt = (
        _build_domain_repair_prompt(
            original_prompt=str(state["agent_prompt"]) + "\nDraft response:\n" + str(state.get("final_output", "")),
            violated_ids=list(state.get("pre_violated_ids", [])),
            domain=str(state.get("domain", "")),
            dataset=_example_dataset_name(state.get("example", {}) or {}),
            verification_config_name=state["verification_cfg"].name,
        )
        + "\n"
        + structured_output_instructions()
    )
    with timed("repair_generation") as t_repair:
        from rava.agent.react_agent import _provider_generate

        repaired = _provider_generate(provider=state["provider"], prompt=repair_prompt)
    _add_step(
        state,
        phase="agent",
        action="repair_generation",
        observation=repaired.text,
        started_at=t_repair.started_at,
        ended_at=t_repair.ended_at,
        metadata=repaired.metadata,
    )
    parsed = parse_structured_output(repaired.text)
    state["final_output"] = render_final_text(parsed)
    state["confidence"] = (
        parsed.confidence
        if parsed.confidence is not None
        else (repaired.confidence if repaired.confidence is not None else state.get("confidence"))
    )
    state["abstained"] = bool(parsed.abstain)
    state["abstain_reason"] = parsed.abstain_reason
    state["output_citations"] = list(parsed.citations or state.get("output_citations", []))
    return state


def _posthoc_audit_node(state: AgentGraphState) -> AgentGraphState:
    state["report"] = {
        "domain": state["domain"],
        "constraint_verdicts": [],
        "claims": [],
        "summary": {"hard_failures": 0, "soft_failures": 0, "total_constraints_checked": 0},
    }
    state["posthoc_repair_attempted"] = False
    state["posthoc_repair_selected"] = False
    state["posthoc_initial_hard_failures"] = 0
    state["posthoc_repaired_hard_failures"] = None
    state["selected_iteration"] = 0
    state["should_posthoc_repair"] = False

    if not state["verification_cfg"].posthoc:
        return state

    with timed("posthoc") as t_post:
        initial_report = state["posthoc_runnable"].invoke(
            {
                "final_text": str(state.get("final_output", "")),
                "context": {
                    "input": state["input_text"],
                    "retrieval_context": str(state.get("retrieval_context", "")),
                },
                "aggregate_metrics": {},
            }
        )
    _add_step(
        state,
        phase="posthoc",
        action="posthoc_audit",
        observation=f"checked={initial_report['summary']['total_constraints_checked']}",
        started_at=t_post.started_at,
        ended_at=t_post.ended_at,
        metadata={"hard_failures": initial_report["summary"]["hard_failures"], "audit_iteration": 0},
    )
    rows = _tag_rows(state, initial_report.get("constraint_verdicts", []), "posthoc")
    for row in rows:
        row["audit_iteration"] = 0
        row["final_selected"] = False
    state["verdict_rows"].extend(rows)
    state["report"] = initial_report
    state["posthoc_initial_hard_failures"] = int(initial_report.get("summary", {}).get("hard_failures", 0))

    if state["verification_cfg"].name == "full" and state["posthoc_initial_hard_failures"] > 0:
        state["should_posthoc_repair"] = True
    return state


def _posthoc_repair_generation_node(state: AgentGraphState) -> AgentGraphState:
    if not bool(state.get("should_posthoc_repair", False)):
        return state

    state["posthoc_repair_attempted"] = True
    from rava.agent.react_agent import _build_domain_repair_prompt, _example_dataset_name

    violated_ids = sorted(
        {
            str(row.get("constraint_id"))
            for row in state.get("report", {}).get("constraint_verdicts", [])
            if str(row.get("constraint_type", "")).upper() == "HARD"
            and str(row.get("verdict", "")).upper() == "FAIL"
            and row.get("constraint_id")
        }
    )
    repair_prompt = (
        _build_domain_repair_prompt(
            original_prompt=str(state["agent_prompt"]) + "\nDraft response:\n" + str(state.get("final_output", "")),
            violated_ids=violated_ids,
            domain=str(state.get("domain", "")),
            dataset=_example_dataset_name(state.get("example", {}) or {}),
            verification_config_name=state["verification_cfg"].name,
        )
        + "\n"
        + structured_output_instructions()
    )
    try:
        with timed("posthoc_repair_generation") as t_repair:
            from rava.agent.react_agent import _provider_generate

            repaired = _provider_generate(provider=state["provider"], prompt=repair_prompt)
        _add_step(
            state,
            phase="agent",
            action="posthoc_repair_generation",
            observation=repaired.text,
            started_at=t_repair.started_at,
            ended_at=t_repair.ended_at,
            metadata=repaired.metadata,
        )
        parsed = parse_structured_output(repaired.text)
        state["repaired_output"] = render_final_text(parsed)
        state["repaired_confidence"] = (
            parsed.confidence
            if parsed.confidence is not None
            else repaired.confidence
        )
        state["repaired_citations"] = list(parsed.citations or state.get("output_citations", []))
        state["repaired_mode"] = (
            str(repaired.metadata.get("mode", ""))
            if isinstance(repaired.metadata, dict)
            else str(state.get("generation_mode", ""))
        )
        state["repaired_error"] = (
            str(repaired.metadata.get("error", ""))
            if isinstance(repaired.metadata, dict) and repaired.metadata.get("error")
            else None
        )
        state["repaired_abstained"] = bool(parsed.abstain)
        state["repaired_abstain_reason"] = parsed.abstain_reason
    except Exception as exc:
        now = time.time()
        _add_step(
            state,
            phase="agent",
            action="posthoc_repair_generation",
            observation="REPAIR_EXCEPTION",
            started_at=now,
            ended_at=now,
            metadata={"error": str(exc)},
        )
        state["repaired_output"] = None
    return state


def _posthoc_reaudit_node(state: AgentGraphState) -> AgentGraphState:
    repaired_output = state.get("repaired_output")
    if not repaired_output:
        return state
    with timed("posthoc_reaudit") as t_reaudit:
        repaired_report = state["posthoc_runnable"].invoke(
            {
                "final_text": str(repaired_output),
                "context": {
                    "input": state["input_text"],
                    "retrieval_context": str(state.get("retrieval_context", "")),
                },
                "aggregate_metrics": {},
            }
        )
    _add_step(
        state,
        phase="posthoc",
        action="posthoc_reaudit",
        observation=f"checked={repaired_report['summary']['total_constraints_checked']}",
        started_at=t_reaudit.started_at,
        ended_at=t_reaudit.ended_at,
        metadata={"hard_failures": repaired_report["summary"]["hard_failures"], "audit_iteration": 1},
    )
    repaired_rows = _tag_rows(state, repaired_report.get("constraint_verdicts", []), "posthoc")
    for row in repaired_rows:
        row["audit_iteration"] = 1
        row["final_selected"] = False
    state["verdict_rows"].extend(repaired_rows)

    repaired_hard = int(repaired_report.get("summary", {}).get("hard_failures", 0))
    state["posthoc_repaired_hard_failures"] = repaired_hard
    if repaired_hard <= int(state.get("posthoc_initial_hard_failures", 0)):
        state["final_output"] = str(repaired_output)
        state["confidence"] = state.get("repaired_confidence")
        state["output_citations"] = list(state.get("repaired_citations", state.get("output_citations", [])))
        state["generation_mode"] = str(state.get("repaired_mode", state.get("generation_mode", "")))
        state["generation_error"] = state.get("repaired_error")
        state["abstained"] = bool(state.get("repaired_abstained", state.get("abstained", False)))
        state["abstain_reason"] = state.get("repaired_abstain_reason")
        state["report"] = repaired_report
        state["selected_iteration"] = 1
        state["posthoc_repair_selected"] = True
    return state


def _finalize_prediction_node(state: AgentGraphState) -> AgentGraphState:
    if state.get("generation_exception") is not None:
        return state

    from rava.agent.react_agent import (
        _example_dataset_name,
        _infer_correctness,
        _normalize_confidence,
        _select_confidence_calibration_map,
    )

    dataset_name = _example_dataset_name(state["example"])

    if state.get("verification_cfg").posthoc:
        selected = int(state.get("selected_iteration", 0))
        for row in state.get("verdict_rows", []):
            if str(row.get("layer", "")).lower() == "posthoc":
                row["final_selected"] = int(row.get("audit_iteration", 0)) == selected

    state["confidence"] = _normalize_confidence(
        confidence=state.get("confidence"),
        final_output=str(state.get("final_output", "")),
        citations=list(state.get("output_citations", [])),
        domain=str(state.get("domain", "")),
        verification_enabled=bool(
            state["verification_cfg"].pre
            or state["verification_cfg"].runtime
            or state["verification_cfg"].posthoc
        ),
        dataset=dataset_name,
        confidence_affine=getattr(state["provider"], "confidence_affine", None),
        confidence_calibration_map=_select_confidence_calibration_map(
            provider=state["provider"],
            domain=str(state.get("domain", "")),
            dataset=dataset_name,
        ),
    )
    if bool(state.get("posthoc_repair_selected", False)):
        state["confidence"] = _normalize_confidence(
            confidence=state.get("confidence"),
            final_output=str(state.get("final_output", "")),
            citations=list(state.get("output_citations", [])),
            domain=str(state.get("domain", "")),
            verification_enabled=bool(
                state["verification_cfg"].pre
                or state["verification_cfg"].runtime
                or state["verification_cfg"].posthoc
            ),
            dataset=dataset_name,
            confidence_affine=getattr(state["provider"], "confidence_affine", None),
            confidence_calibration_map=_select_confidence_calibration_map(
                provider=state["provider"],
                domain=str(state.get("domain", "")),
                dataset=dataset_name,
            ),
        )

    prediction = {
        "id": state["example_id"],
        "domain": state["domain"],
        "input": state["input_text"],
        "reference": state["example"].get("reference"),
        "output": str(state.get("final_output", "")),
        "confidence": state.get("confidence"),
        "abstained": bool(state.get("abstained", False)),
        "abstain_reason": state.get("abstain_reason"),
        "metadata": state["example"].get("metadata", {}),
        "split": state["example"].get("split", "unknown"),
        "retrieval_context": str(state.get("retrieval_context", "")),
        "correct": _infer_correctness(
            example=state["example"],
            output_text=str(state.get("final_output", "")),
        ),
        "timestamp": time.time(),
        "raw_model_output": state.get("result_text", ""),
        "generation_mode": str(state.get("generation_mode", "")),
        "generation_error": state.get("generation_error"),
        "runtime_halted_hard_fail": bool(state.get("halted", False)),
        "posthoc_repair_attempted": bool(state.get("posthoc_repair_attempted", False)),
        "posthoc_repair_selected": bool(state.get("posthoc_repair_selected", False)),
        "posthoc_initial_hard_failures": int(state.get("posthoc_initial_hard_failures", 0)),
        "posthoc_repaired_hard_failures": state.get("posthoc_repaired_hard_failures"),
    }
    state["prediction"] = prediction
    return state


def _build_graph():
    if StateGraph is None:
        return None
    graph = StateGraph(AgentGraphState)
    graph.add_node("pre_tool_check", _pre_tool_check_node)
    graph.add_node("agent_generate", _agent_generate_node)
    graph.add_node("runtime_monitor_loop", _runtime_monitor_loop_node)
    graph.add_node("pre_final_check", _pre_final_check_node)
    graph.add_node("pre_repair_generation", _pre_repair_generation_node)
    graph.add_node("posthoc_audit", _posthoc_audit_node)
    graph.add_node("posthoc_repair_generation", _posthoc_repair_generation_node)
    graph.add_node("posthoc_reaudit", _posthoc_reaudit_node)
    graph.add_node("finalize_prediction", _finalize_prediction_node)

    graph.add_edge(START, "pre_tool_check")
    graph.add_edge("pre_tool_check", "agent_generate")
    graph.add_edge("agent_generate", "runtime_monitor_loop")

    def _after_runtime(state: AgentGraphState) -> str:
        if state.get("generation_exception") is not None:
            return "finalize_prediction"
        if bool(state.get("halted", False)):
            return "finalize_prediction"
        return "pre_final_check"

    graph.add_conditional_edges(
        "runtime_monitor_loop",
        _after_runtime,
        {
            "finalize_prediction": "finalize_prediction",
            "pre_final_check": "pre_final_check",
        },
    )

    def _after_pre_final(state: AgentGraphState) -> str:
        if bool(state.get("should_pre_repair", False)):
            return "pre_repair_generation"
        return "posthoc_audit"

    graph.add_conditional_edges(
        "pre_final_check",
        _after_pre_final,
        {
            "pre_repair_generation": "pre_repair_generation",
            "posthoc_audit": "posthoc_audit",
        },
    )
    graph.add_edge("pre_repair_generation", "posthoc_audit")

    def _after_posthoc(state: AgentGraphState) -> str:
        if bool(state.get("should_posthoc_repair", False)):
            return "posthoc_repair_generation"
        return "finalize_prediction"

    graph.add_conditional_edges(
        "posthoc_audit",
        _after_posthoc,
        {
            "posthoc_repair_generation": "posthoc_repair_generation",
            "finalize_prediction": "finalize_prediction",
        },
    )
    graph.add_edge("posthoc_repair_generation", "posthoc_reaudit")
    graph.add_edge("posthoc_reaudit", "finalize_prediction")
    graph.add_edge("finalize_prediction", END)
    return graph.compile()


_COMPILED_GRAPH = _build_graph()


def run_agent_example_langgraph(
    example: dict[str, Any],
    provider: BaseProvider,
    spec: Specification,
    verification_cfg: VerificationConfig,
    run_id: str,
    tool: Tool | None = None,
    tool_cache_scope: str = "example",
    max_tool_iterations: int = 64,
) -> dict[str, Any]:
    from rava.agent.react_agent import AgentGenerationFailedError, _default_tool_for_domain

    example_id = str(example.get("id", "unknown"))
    domain = str(example.get("domain", spec.domain))
    input_text = str(example.get("input", ""))
    chosen_tool = tool or _default_tool_for_domain(domain)
    pre_verifier = PreExecutionVerifier(spec)
    runtime_monitor = RuntimeMonitor(spec)
    posthoc_auditor = PostHocAuditor(spec)
    cache_scope = str(tool_cache_scope or "example").strip().lower()
    if cache_scope == "example":
        chosen_tool = _MemoizedTool(base=chosen_tool, cache={})
    tool_query = input_text

    state: AgentGraphState = {
        "example": example,
        "provider": provider,
        "spec": spec,
        "verification_cfg": verification_cfg,
        "run_id": run_id,
        "tool": chosen_tool,
        "tool_cache_scope": cache_scope,
        "max_tool_iterations": int(max(1, max_tool_iterations)),
        "example_id": example_id,
        "domain": domain,
        "input_text": input_text,
        "tool_query": tool_query,
        "trajectory": AgentTrajectory(run_id=run_id, example_id=example_id, domain=domain),
        "verdict_rows": [],
        "action_index": 0,
        "pre": pre_verifier,
        "runtime": runtime_monitor,
        "posthoc": posthoc_auditor,
        "pre_runnable": pre_verifier.as_runnable(),
        "runtime_runnable": runtime_monitor.as_runnable(),
        "posthoc_runnable": posthoc_auditor.as_runnable(),
        "agent_prompt": (
            f"Domain: {domain}\n"
            f"User input: {input_text}\n"
            f"Tool query hint: {tool_query}\n"
            "You may call tools when useful. Provide a safe, factual response with citations where possible.\n"
            f"{structured_output_instructions()}"
        ),
        "result_text": "",
        "result_confidence": None,
        "result_metadata": {},
        "intermediate_steps": [],
        "retrieval_context": "",
        "generation_exception": None,
        "generation_exception_taxonomy": None,
        "generation_exception_message": None,
        "halted": False,
        "halted_report": {"summary": {"halted": False}, "constraint_verdicts": []},
        "final_output": "",
        "confidence": None,
        "abstained": False,
        "abstain_reason": None,
        "output_citations": [],
        "generation_mode": "",
        "generation_error": None,
        "report": {
            "domain": domain,
            "constraint_verdicts": [],
            "claims": [],
            "summary": {"hard_failures": 0, "soft_failures": 0, "total_constraints_checked": 0},
        },
        "prediction": {},
        "posthoc_repair_attempted": False,
        "posthoc_repair_selected": False,
        "posthoc_initial_hard_failures": 0,
        "posthoc_repaired_hard_failures": None,
        "selected_iteration": 0,
        "should_pre_repair": False,
        "pre_violated_ids": [],
        "should_posthoc_repair": False,
        "repaired_output": None,
        "repaired_confidence": None,
        "repaired_citations": [],
        "repaired_mode": None,
        "repaired_error": None,
        "repaired_abstained": False,
        "repaired_abstain_reason": None,
    }

    if _COMPILED_GRAPH is None:  # pragma: no cover - only when langgraph import is missing
        state = _pre_tool_check_node(state)
        state = _agent_generate_node(state)
        state = _runtime_monitor_loop_node(state)
        if not state.get("halted") and state.get("generation_exception") is None:
            state = _pre_final_check_node(state)
            state = _pre_repair_generation_node(state)
            state = _posthoc_audit_node(state)
            state = _posthoc_repair_generation_node(state)
            state = _posthoc_reaudit_node(state)
        state = _finalize_prediction_node(state)
    else:
        state = _COMPILED_GRAPH.invoke(state)

    gen_exc = state.get("generation_exception")
    if isinstance(gen_exc, AgentGenerationFailedError):
        raise gen_exc
    if isinstance(gen_exc, Exception):
        taxonomy = classify_provider_error(gen_exc)
        raise AgentGenerationFailedError(
            "Agent generation failed before completion.",
            trajectory_rows=state["trajectory"].to_rows(),
            verdict_rows=state["verdict_rows"],
            taxonomy=taxonomy,
            original_error=str(gen_exc),
        ) from gen_exc

    if bool(state.get("halted", False)):
        return {
            "prediction": {
                "id": example_id,
                "domain": domain,
                "input": input_text,
                "reference": example.get("reference"),
                "output": state.get("final_output", "Request halted due to runtime HARD-constraint failure."),
                "confidence": None,
                "abstained": True,
                "abstain_reason": "runtime_hard_fail_halt",
                "metadata": example.get("metadata", {}),
                "split": example.get("split", "unknown"),
                "retrieval_context": str(state.get("retrieval_context", "")),
                "runtime_halted_hard_fail": True,
                "generation_mode": "runtime_halt",
                "generation_error": None,
            },
            "trajectory": state["trajectory"].to_rows(),
            "verdicts": state["verdict_rows"],
            "report": state.get("report", {"summary": {"halted": True}, "constraint_verdicts": []}),
        }

    return {
        "prediction": state["prediction"],
        "trajectory": state["trajectory"].to_rows(),
        "verdicts": state["verdict_rows"],
        "report": state["report"],
    }
