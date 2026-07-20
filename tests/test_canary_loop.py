from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from abstrak.canary.artifacts import TrajectoryStore, verify_trajectory
from abstrak.canary.contracts import (
    AgentBudget,
    CaseResult,
    TimingSpec,
    WorkerJob,
    WorkerResult,
)
from abstrak.canary.loop import CanaryAgentLoop
from abstrak.canary.protocol import build_initial_messages
from abstrak.canary.targets import get_target_stack, load_target_card
from abstrak.canary.tasks import get_task_pack
from abstrak.providers.contracts import (
    ErrorCategory,
    NormalizedError,
    NormalizedResponse,
    NormalizedUsage,
    ProviderCallError,
)


def _model_response(request: object, text: str, index: int) -> NormalizedResponse:
    now = datetime.now(timezone.utc)
    return NormalizedResponse(
        request_id=request.request_id,  # type: ignore[attr-defined]
        attempt_id=f"attempt-{index}",
        provider_request_id=f"provider-{index}",
        provider_id="fake-provider",
        model_id="fake-model",
        provider_manifest_sha256="1" * 64,
        model_manifest_sha256="2" * 64,
        requested_model="fake-model",
        returned_model="fake-model",
        text=text,
        finish_reason="stop",
        provider_finish_reason="stop",
        usage=NormalizedUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_characters=100,
            output_characters=len(text),
            provider_reported=True,
            core_fields_complete=True,
        ),
        started_at_utc=now,
        finished_at_utc=now,
        elapsed_ms=1.0,
        logical_request_sha256="3" * 64,
        transport_request_sha256="4" * 64,
        transport_response_sha256="5" * 64,
        sanitized_transport_request={},
        raw_transport_response={},
    )


def _candidate(name: str, marker: str) -> str:
    return f"""```python
class ModelNew:
    candidate_name = {name!r}
```
{marker}
"""


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = deque(responses)
        self.requests: list[object] = []

    def complete(self, request: object) -> NormalizedResponse:
        self.requests.append(request)
        response = self.responses.popleft()
        if response == "__PROVIDER_ERROR__":
            now = datetime.now(timezone.utc)
            raise ProviderCallError(
                NormalizedError(
                    request_id=request.request_id,  # type: ignore[attr-defined]
                    attempt_id=f"attempt-{len(self.requests)}",
                    provider_id="fake-provider",
                    model_id="fake-model",
                    category=ErrorCategory.NETWORK,
                    provider_type="FakeNetworkError",
                    sanitized_message="network unavailable",
                    retryable=True,
                    request_submitted=True,
                    possibly_charged=True,
                    started_at_utc=now,
                    failed_at_utc=now,
                    elapsed_ms=1.0,
                    logical_request_sha256="3" * 64,
                    sanitized_transport_request={},
                )
            )
        return _model_response(request, response, len(self.requests))


class FakeWorker:
    def __init__(self) -> None:
        self.jobs: list[WorkerJob] = []

    def execute(self, job: WorkerJob) -> WorkerResult:
        self.jobs.append(job)
        correct = "good" in job.candidate_source
        cases = tuple(
            CaseResult(
                case_id=case_id,
                status="pass" if correct else "wrong_result",
                correct=correct,
                max_abs_error=0.0 if correct else 1.0,
                max_rel_error=0.0 if correct else 1.0,
                output_finite=True,
                inputs_unchanged=True,
            )
            for case_id in job.case_ids
        )
        timing = tuple(1.0 for _ in range(job.timing.trial_runs)) if correct and job.timing else ()
        return WorkerResult(
            job_id=job.job_id,
            job_sha256=job.sha256,
            input_sha256=job.input_sha256,
            candidate_sha256=job.candidate_sha256,
            status="completed" if correct else "wrong_result",
            compiled=True,
            correct=correct,
            cases=cases,
            timing_ms=timing,
            timing_cv=0.0 if timing else None,
            metadata={"sealed-sentinel": job.kind == "sealed"},
        )


def _loop(tmp_path: Path, responses: list[str]) -> tuple[CanaryAgentLoop, FakeClient, FakeWorker]:
    client = FakeClient(responses)
    worker = FakeWorker()
    loop = CanaryAgentLoop(
        client=client,
        worker=worker,
        store=TrajectoryStore.create(tmp_path, "study", "trajectory"),
    )
    return loop, client, worker


def _run(loop: CanaryAgentLoop):
    task = get_task_pack("row-reduction-scale")
    return loop.run(
        trajectory_id="trajectory",
        model_ref="fake-model",
        initial_messages=build_initial_messages(
            task,
            load_target_card("triton-a100"),
        ),
        task=task,
        target=get_target_stack("triton-a100"),
        budget=AgentBudget(max_calls=4, max_wall_seconds=1200.0),
        dev_timing=TimingSpec(trial_runs=3, repetitions=1),
    )


def test_invalid_response_gets_protocol_feedback_then_finishes(tmp_path: Path) -> None:
    loop, client, worker = _loop(tmp_path, ["not valid", _candidate("good", "FINISH")])

    outcome = _run(loop)

    assert outcome.status == "finished"
    assert outcome.calls == 2
    assert len(client.requests) == 2
    second_history = client.requests[1].messages  # type: ignore[attr-defined]
    assert any("PROTOCOL_ERROR" in message.content for message in second_history)
    assert [job.kind for job in worker.jobs] == ["dev", "sealed", "sealed"]


def test_continue_then_finish_keeps_sealed_results_out_of_history(tmp_path: Path) -> None:
    loop, client, worker = _loop(
        tmp_path,
        [_candidate("bad", "CONTINUE"), _candidate("good", "FINISH")],
    )

    outcome = _run(loop)

    assert outcome.status == "finished"
    assert [job.kind for job in worker.jobs] == ["dev", "dev", "sealed", "sealed"]
    assert outcome.first_candidate_sha256 != outcome.final_candidate_sha256
    rendered_history = "\n".join(
        message.content
        for request in client.requests
        for message in request.messages  # type: ignore[attr-defined]
    )
    assert "sealed-sentinel" not in rendered_history
    assert "sealed-random" not in rendered_history
    verify_trajectory(loop.store.run_directory)


def test_four_continue_actions_stop_at_call_limit(tmp_path: Path) -> None:
    loop, client, worker = _loop(
        tmp_path,
        [_candidate(f"good-{index}", "CONTINUE") for index in range(4)],
    )

    outcome = _run(loop)

    assert outcome.status == "call_limit"
    assert outcome.calls == 4
    assert len(client.requests) == 4
    assert [job.kind for job in worker.jobs].count("dev") == 4
    assert [job.kind for job in worker.jobs][-2:] == ["sealed", "sealed"]


def test_all_protocol_failures_produce_no_candidate(tmp_path: Path) -> None:
    loop, _client, worker = _loop(tmp_path, ["invalid"] * 4)

    outcome = _run(loop)

    assert outcome.status == "no_candidate"
    assert outcome.calls == 4
    assert outcome.first_candidate_sha256 is None
    assert worker.jobs == []


def test_provider_error_is_terminal_without_retry_and_still_seals_candidate(
    tmp_path: Path,
) -> None:
    loop, client, worker = _loop(
        tmp_path,
        [_candidate("bad", "CONTINUE"), "__PROVIDER_ERROR__"],
    )

    outcome = _run(loop)

    assert outcome.status == "provider_error"
    assert outcome.calls == 2
    assert len(client.requests) == 2
    assert outcome.error and "network unavailable" in outcome.error
    assert [job.kind for job in worker.jobs] == ["dev", "sealed", "sealed"]
