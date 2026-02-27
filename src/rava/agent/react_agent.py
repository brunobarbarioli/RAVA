from __future__ import annotations

import re
import time
from typing import Any

from rava.agent.providers import BaseProvider
from rava.agent.output_schema import parse_structured_output, render_final_text, structured_output_instructions
from rava.agent.trajectory import AgentTrajectory, TrajectoryStep
from rava.experiments.baselines import VerificationConfig
from rava.specs.schema import Specification, VerificationAction
from rava.tools import PriceLookupTool, PubMedRetriever, ResumeParserTool, Tool
from rava.utils.timing import timed
from rava.verification.posthoc_audit import PostHocAuditor
from rava.verification.pre_execution import PreExecutionVerifier, build_repair_prompt
from rava.verification.runtime_monitor import RuntimeMonitor


def _sanitize_query(text: str) -> str:
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[REDACTED_EMAIL]", text)
    return text


def _default_tool_for_domain(domain: str) -> Tool:
    if domain == "healthcare":
        return PubMedRetriever()
    if domain == "finance":
        return PriceLookupTool()
    return ResumeParserTool()


def run_agent_example(
    example: dict[str, Any],
    provider: BaseProvider,
    spec: Specification,
    verification_cfg: VerificationConfig,
    run_id: str,
    tool: Tool | None = None,
) -> dict[str, Any]:
    example_id = str(example.get("id", "unknown"))
    domain = str(example.get("domain", spec.domain))
    input_text = str(example.get("input", ""))

    trajectory = AgentTrajectory(run_id=run_id, example_id=example_id, domain=domain)
    verdict_rows: list[dict[str, Any]] = []
    action_index = 0

    def _tag_rows(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        nonlocal action_index
        action_index += 1
        action_key = f"{example_id}:{label}:{action_index}"
        for row in rows:
            row["record_id"] = example_id
            row["action_key"] = action_key
        return rows

    pre = PreExecutionVerifier(spec)
    runtime = RuntimeMonitor(spec)
    posthoc = PostHocAuditor(spec)

    chosen_tool = tool or _default_tool_for_domain(domain)
    tool_query = input_text

    if verification_cfg.pre:
        with timed("pre_tool") as t:
            pre_res = pre.verify(
                event="tool_call",
                text=f"TOOL:{chosen_tool.name} QUERY:{tool_query}",
                context={"input": input_text},
            )
        trajectory.add_step(
            TrajectoryStep(
                step_id=len(trajectory.steps) + 1,
                phase="pre",
                action="pre_tool_check",
                observation=pre_res.action.value,
                started_at=t.started_at,
                ended_at=t.ended_at,
                metadata={"violations": pre_res.violated_constraint_ids},
            )
        )
        verdict_rows.extend(_tag_rows(pre_res.verdict_rows, "pre_tool"))
        if pre_res.action in {VerificationAction.MODIFY, VerificationAction.BLOCK}:
            tool_query = _sanitize_query(tool_query)

    agent_prompt = (
        f"Domain: {domain}\n"
        f"User input: {input_text}\n"
        f"Tool query hint: {tool_query}\n"
        "You may call tools when useful. Provide a safe, factual response with citations where possible.\n"
        f"{structured_output_instructions()}"
    )

    with timed("model_agent_generate") as t_model:
        result = provider.generate_agent(prompt=agent_prompt, tools=[chosen_tool])
    trajectory.add_step(
        TrajectoryStep(
            step_id=len(trajectory.steps) + 1,
            phase="agent",
            action="langchain_agent_generation",
            observation=result.text,
            started_at=t_model.started_at,
            ended_at=t_model.ended_at,
            metadata=result.metadata,
        )
    )

    intermediate_steps = list(result.metadata.get("intermediate_steps", [])) if isinstance(result.metadata, dict) else []
    retrieval_context = ""

    for step in intermediate_steps:
        action_text = f"TOOL:{step.get('tool', 'unknown')} QUERY:{step.get('tool_input', '')}"
        observation_text = str(step.get("observation", ""))
        retrieval_context += (observation_text + "\n")

        now = time.time()
        trajectory.add_step(
            TrajectoryStep(
                step_id=len(trajectory.steps) + 1,
                phase="agent",
                action=action_text,
                observation=observation_text,
                started_at=now,
                ended_at=now,
                metadata={"log": step.get("log", "")},
            )
        )

        if verification_cfg.runtime:
            with timed("runtime") as t_rt:
                rt_res = runtime.monitor(
                    action_text=action_text,
                    observation_text=observation_text,
                    context={"input": input_text},
                    event="tool_call",
                )
            trajectory.add_step(
                TrajectoryStep(
                    step_id=len(trajectory.steps) + 1,
                    phase="runtime",
                    action="runtime_monitor",
                    observation=rt_res.action.value,
                    started_at=t_rt.started_at,
                    ended_at=t_rt.ended_at,
                    metadata={"hard_fail_ids": rt_res.hard_fail_ids},
                )
            )
            verdict_rows.extend(_tag_rows(rt_res.verdict_rows, "runtime"))

            if rt_res.action == VerificationAction.HALT:
                final_output = "Request halted due to runtime HARD-constraint failure."
                report = {"summary": {"halted": True}, "constraint_verdicts": []}
                return {
                    "prediction": {
                        "id": example_id,
                        "domain": domain,
                        "input": input_text,
                        "reference": example.get("reference"),
                        "output": final_output,
                        "confidence": None,
                        "metadata": example.get("metadata", {}),
                        "split": example.get("split", "unknown"),
                        "retrieval_context": retrieval_context.strip(),
                    },
                    "trajectory": trajectory.to_rows(),
                    "verdicts": verdict_rows,
                    "report": report,
                }

    parsed_output = parse_structured_output(result.text)
    final_output = render_final_text(parsed_output)
    confidence = parsed_output.confidence if parsed_output.confidence is not None else result.confidence

    if verification_cfg.pre:
        with timed("pre_final") as t_pf:
            pre_final = pre.verify(event="final_answer", text=final_output, context={"input": input_text})
        trajectory.add_step(
            TrajectoryStep(
                step_id=len(trajectory.steps) + 1,
                phase="pre",
                action="pre_final_check",
                observation=pre_final.action.value,
                started_at=t_pf.started_at,
                ended_at=t_pf.ended_at,
                metadata={"violations": pre_final.violated_constraint_ids},
            )
        )
        verdict_rows.extend(_tag_rows(pre_final.verdict_rows, "pre_final"))

        if pre_final.action in {VerificationAction.MODIFY, VerificationAction.BLOCK}:
            repair_prompt = (
                build_repair_prompt(
                    agent_prompt + "\nDraft response:\n" + final_output,
                    pre_final.violated_constraint_ids,
                )
                + "\n"
                + structured_output_instructions()
            )
            with timed("repair_generation") as t_repair:
                repaired = provider.generate(prompt=repair_prompt)
            trajectory.add_step(
                TrajectoryStep(
                    step_id=len(trajectory.steps) + 1,
                    phase="agent",
                    action="repair_generation",
                    observation=repaired.text,
                    started_at=t_repair.started_at,
                    ended_at=t_repair.ended_at,
                    metadata=repaired.metadata,
                )
            )
            repaired_structured = parse_structured_output(repaired.text)
            final_output = render_final_text(repaired_structured)
            repaired_conf = repaired_structured.confidence
            confidence = repaired_conf if repaired_conf is not None else repaired.confidence if repaired.confidence is not None else confidence

    report = {
        "domain": domain,
        "constraint_verdicts": [],
        "claims": [],
        "summary": {"hard_failures": 0, "soft_failures": 0, "total_constraints_checked": 0},
    }

    if verification_cfg.posthoc:
        with timed("posthoc") as t_post:
            report = posthoc.audit(
                final_text=final_output,
                context={"input": input_text, "retrieval_context": retrieval_context.strip()},
                aggregate_metrics={},
            )
        trajectory.add_step(
            TrajectoryStep(
                step_id=len(trajectory.steps) + 1,
                phase="posthoc",
                action="posthoc_audit",
                observation=f"checked={report['summary']['total_constraints_checked']}",
                started_at=t_post.started_at,
                ended_at=t_post.ended_at,
                metadata={"hard_failures": report["summary"]["hard_failures"]},
            )
        )
        verdict_rows.extend(_tag_rows(report.get("constraint_verdicts", []), "posthoc"))

    prediction = {
        "id": example_id,
        "domain": domain,
        "input": input_text,
        "reference": example.get("reference"),
        "output": final_output,
        "confidence": confidence,
        "abstained": bool(parsed_output.abstain),
        "abstain_reason": parsed_output.abstain_reason,
        "metadata": example.get("metadata", {}),
        "split": example.get("split", "unknown"),
        "retrieval_context": retrieval_context.strip(),
        "correct": int(str(example.get("reference", "")).lower() in final_output.lower())
        if example.get("reference")
        else None,
        "timestamp": time.time(),
        "raw_model_output": result.text,
    }

    return {
        "prediction": prediction,
        "trajectory": trajectory.to_rows(),
        "verdicts": verdict_rows,
        "report": report,
    }
