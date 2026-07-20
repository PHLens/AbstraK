from __future__ import annotations

import hashlib

import pytest

from abstrak.canary.contracts import CaseResult, WorkerJob, WorkerResult
from abstrak.canary.protocol import (
    AgentProtocolError,
    build_initial_messages,
    format_worker_feedback,
    parse_agent_action,
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


def test_parse_agent_action_requires_one_model_and_marker() -> None:
    action = parse_agent_action(_response("CONTINUE"))

    assert action.decision == "continue"
    assert action.candidate_source.endswith("\n")
    assert "class ModelNew" in action.candidate_source


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

    assert task.specification in rendered
    assert "sealed-random" not in rendered
    assert "20260718" not in rendered
    assert task.source_sha256 not in rendered


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
