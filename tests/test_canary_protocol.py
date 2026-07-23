from __future__ import annotations

import hashlib

import pytest

from abstrak.canary.contracts import AgentLoopPolicy, CaseResult, WorkerJob, WorkerResult
from abstrak.canary.protocol import (
    AgentProtocolError,
    build_initial_messages,
    format_worker_feedback,
    parse_agent_action,
    protocol_error_feedback,
)
from abstrak.canary.targets import get_target_stack, load_target_card
from abstrak.canary.tasks import get_task_pack, load_oracle_source


def _response(marker: str = "FINISH") -> str:
    return f"""```python
import torch
class ModelNew(torch.nn.Module):
    def forward(self, x):
        return torch.empty_like(x)
```
{marker}
"""


def _candidate_only_response() -> str:
    return """```python
import torch
class ModelNew(torch.nn.Module):
    def forward(self, x):
        return torch.empty_like(x)
```
"""


def _candidate_only_policy() -> AgentLoopPolicy:
    return AgentLoopPolicy(
        response_parser="candidate_only",
        stop_policy="correct_latency",
        final_selection="best_correct_latency",
        latency_ceiling_ms=1.25,
    )


def test_parse_agent_action_requires_one_model_and_marker() -> None:
    action = parse_agent_action(_response("CONTINUE"))

    assert action.decision == "continue"
    assert action.candidate_source.endswith("\n")
    assert "class ModelNew" in action.candidate_source


def test_default_protocol_error_feedback_is_byte_compatible() -> None:
    error = AgentProtocolError("invalid response")

    assert protocol_error_feedback(error) == (
        "PROTOCOL_ERROR\n"
        "invalid response\n"
        "Return exactly one complete ```python block defining ModelNew and exactly one "
        "CONTINUE or FINISH line."
    )


def test_parse_candidate_only_action_has_no_agent_decision() -> None:
    action = parse_agent_action(_candidate_only_response(), policy=_candidate_only_policy())

    assert action.decision is None
    assert action.candidate_source.endswith("\n")
    assert "class ModelNew" in action.candidate_source


@pytest.mark.parametrize(
    "suffix",
    ["FINISH", "CONTINUE", "explanation"],
)
def test_candidate_only_action_rejects_text_outside_the_fence(suffix: str) -> None:
    with pytest.raises(AgentProtocolError, match="only one fenced"):
        parse_agent_action(
            f"{_candidate_only_response()}{suffix}\n",
            policy=_candidate_only_policy(),
        )


@pytest.mark.parametrize(
    "text,match",
    [
        ("FINISH", "exactly one fenced"),
        ("```python\nclass Other: pass\n```\nFINISH", "ModelNew"),
        (_response(""), "exactly one CONTINUE"),
        (_response("CONTINUE\nFINISH"), "exactly one CONTINUE"),
        (_response() + "```text\nextra\n```", "exactly one fenced"),
    ],
)
def test_parse_agent_action_rejects_ambiguous_responses(text: str, match: str) -> None:
    with pytest.raises(AgentProtocolError, match=match):
        parse_agent_action(text)


def test_initial_messages_use_public_task_view_only() -> None:
    task = get_task_pack("row-reduction-scale")
    messages = build_initial_messages(task, load_target_card("triton-a100"))
    rendered = "\n".join(message.content for message in messages)

    assert messages[0].content == (
        "You are optimizing one frozen GPU task. Return a complete Python implementation "
        "that defines ModelNew in exactly one ```python code block. On a separate line, "
        "return CONTINUE to request another feedback round or FINISH to end. Never return "
        "a patch or shell command."
    )
    assert task.specification in rendered
    assert "sealed-random" not in rendered
    assert "20260718" not in rendered
    assert task.source_sha256 not in rendered


def test_candidate_only_initial_message_omits_agent_markers() -> None:
    task = get_task_pack("row-reduction-scale")
    messages = build_initial_messages(
        task,
        load_target_card("triton-a100"),
        policy=_candidate_only_policy(),
    )
    rendered = "\n".join(message.content for message in messages)

    assert "CONTINUE" not in rendered
    assert "FINISH" not in rendered
    assert "no text outside" in rendered
    assert "at most 1.25 ms" in rendered


def test_worker_feedback_omits_metadata_and_case_ids() -> None:
    task = get_task_pack("row-reduction-scale")
    source = load_oracle_source("row-reduction-scale", "triton")
    job = WorkerJob(
        job_id="feedback-job",
        kind="dev",
        task=task,
        target=get_target_stack("triton-a100"),
        case_ids=tuple(case.id for case in task.dev_cases),
        candidate_source=source,
        candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    cases = tuple(
        CaseResult(
            case_id=case_id,
            status="pass",
            correct=True,
            max_abs_error=0.0,
            max_rel_error=0.0,
            output_finite=True,
            inputs_unchanged=True,
        )
        for case_id in job.case_ids
    )
    result = WorkerResult(
        job_id=job.job_id,
        job_sha256=job.sha256,
        input_sha256=job.input_sha256,
        candidate_sha256=job.candidate_sha256,
        status="completed",
        compiled=True,
        correct=True,
        cases=cases,
        metadata={"sealed-sentinel": "must-not-leak"},
    )

    feedback = format_worker_feedback(result)

    assert "sealed-sentinel" not in feedback
    assert "dev-random" not in feedback
    assert '"correct": true' in feedback
