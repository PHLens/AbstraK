"""Frozen textual protocol between the canary controller and an Agent."""

from __future__ import annotations

import ast
import json
import re
import statistics
from dataclasses import dataclass
from typing import Literal

from abstrak.canary.contracts import (
    R1_AGENT_LOOP_POLICY,
    AgentLoopPolicy,
    TaskPackSpec,
    WorkerResult,
)
from abstrak.providers.contracts import ChatMessage, MessageRole

_PYTHON_FENCE = re.compile(r"```(?:python|py)[ \t]*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL)
_MARKER = re.compile(r"^[ \t]*(CONTINUE|FINISH)[ \t]*$", re.MULTILINE)


class AgentProtocolError(ValueError):
    """Raised when an Agent response cannot produce one deterministic action."""


@dataclass(frozen=True)
class AgentAction:
    candidate_source: str
    decision: Literal["continue", "finish"] | None


def parse_agent_action(
    text: str,
    *,
    policy: AgentLoopPolicy = R1_AGENT_LOOP_POLICY,
) -> AgentAction:
    """Parse one complete ModelNew source using the frozen response policy."""

    matches = list(_PYTHON_FENCE.finditer(text))
    if len(matches) != 1 or text.count("```") != 2:
        raise AgentProtocolError("response must contain exactly one fenced Python code block")
    match = matches[0]
    source = match.group(1).strip()
    if not source:
        raise AgentProtocolError("candidate code block cannot be empty")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise AgentProtocolError(f"candidate is not valid Python: {error.msg}") from None
    if not any(isinstance(node, ast.ClassDef) and node.name == "ModelNew" for node in tree.body):
        raise AgentProtocolError("candidate must define a top-level ModelNew class")

    outside = f"{text[: match.start()]}\n{text[match.end() :]}"
    if policy.response_parser == "candidate_only":
        if outside.strip():
            raise AgentProtocolError("response must contain only one fenced Python code block")
        return AgentAction(candidate_source=f"{source}\n", decision=None)

    markers = _MARKER.findall(outside)
    if len(markers) != 1:
        raise AgentProtocolError("response must contain exactly one CONTINUE or FINISH line")
    decision: Literal["continue", "finish"] = markers[0].lower()  # type: ignore[assignment]
    return AgentAction(candidate_source=f"{source}\n", decision=decision)


def build_initial_messages(
    task: TaskPackSpec,
    target_card: str,
    *,
    policy: AgentLoopPolicy = R1_AGENT_LOOP_POLICY,
) -> tuple[ChatMessage, ...]:
    """Build the only Agent-visible task payload; private cases are omitted by construction."""

    public_task = json.dumps(
        task.public_view().model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if policy.response_parser == "agent_marker":
        system_content = (
            "You are optimizing one frozen GPU task. Return a complete Python implementation "
            "that defines ModelNew in exactly one ```python code block. On a separate line, "
            "return CONTINUE to request another feedback round or FINISH to end. Never return "
            "a patch or shell command."
        )
    else:
        ceiling = policy.latency_ceiling_ms
        if ceiling is None:  # The policy contract makes this unreachable.
            raise ValueError("candidate_only policy is missing its latency ceiling")
        system_content = (
            "You are optimizing one frozen GPU task. Return a complete Python implementation "
            "that defines ModelNew in exactly one ```python code block. Return no text outside "
            "the code block and never return a patch or shell command. The controller will stop "
            f"when the candidate is correct and median dev latency is at most {ceiling:g} ms, "
            "or when the call budget is exhausted."
        )
    system = ChatMessage(
        role=MessageRole.SYSTEM,
        content=system_content,
    )
    user = ChatMessage(
        role=MessageRole.USER,
        content=f"TASK\n{public_task}\n\nTARGET CARD\n{target_card}",
    )
    return (system, user)


def protocol_error_feedback(
    error: AgentProtocolError,
    *,
    policy: AgentLoopPolicy = R1_AGENT_LOOP_POLICY,
) -> str:
    if policy.response_parser == "candidate_only":
        requirement = (
            "Return only one complete ```python block defining ModelNew, with no text outside "
            "the code block."
        )
    else:
        requirement = (
            "Return exactly one complete ```python block defining ModelNew and exactly one "
            "CONTINUE or FINISH line."
        )
    return f"PROTOCOL_ERROR\n{error}\n{requirement}"


def format_worker_feedback(
    result: WorkerResult,
    *,
    policy: AgentLoopPolicy = R1_AGENT_LOOP_POLICY,
) -> str:
    """Return a bounded dev-only feedback envelope without runtime metadata or hidden cases."""

    cases = [
        {
            "status": case.status,
            "max_abs_error": case.max_abs_error,
            "max_rel_error": case.max_rel_error,
            "error": case.error[:1000] if case.error else None,
        }
        for case in result.cases
    ]
    payload: dict[str, object] = {
        "status": result.status,
        "compiled": result.compiled,
        "correct": result.correct,
        "cases": cases,
        "static_errors": [value[:1000] for value in result.static_errors],
        "error": result.error[:2000] if result.error else None,
    }
    if result.timing_ms:
        median_latency_ms = statistics.median(result.timing_ms)
        payload["median_latency_ms"] = median_latency_ms
        if policy.stop_policy == "correct_latency":
            ceiling = policy.latency_ceiling_ms
            if ceiling is None:  # The policy contract makes this unreachable.
                raise ValueError("correct_latency policy is missing its latency ceiling")
            payload["latency_ceiling_ms"] = ceiling
            payload["meets_latency_ceiling"] = result.correct and median_latency_ms <= ceiling
    return "DEV_FEEDBACK\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
