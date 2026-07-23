from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pytest

from abstrak.canary.artifacts import TrajectoryStore, verify_trajectory
from abstrak.canary.contracts import (
    R1_AGENT_LOOP_POLICY,
    AgentBudget,
    AgentLoopPolicy,
    CaseResult,
    TimingSpec,
    WorkerJob,
    WorkerResult,
)
from abstrak.canary.loop import CanaryAgentLoop
from abstrak.canary.protocol import build_initial_messages
from abstrak.canary.remote import FailureCategory, WorkerExecutionError
from abstrak.canary.targets import get_target_stack, load_target_card
from abstrak.canary.tasks import get_task_pack
from abstrak.providers.contracts import (
    ErrorCategory,
    NormalizedError,
    NormalizedResponse,
    NormalizedUsage,
    ProviderCallError,
)

TEST_DEV_TIMING = TimingSpec(trial_runs=3, repetitions=1)


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


def _candidate_only(name: str) -> str:
    return f"""```python
class ModelNew:
    candidate_name = {name!r}
```
"""


def _gate_policy(latency_ceiling_ms: float) -> AgentLoopPolicy:
    return AgentLoopPolicy(
        response_parser="candidate_only",
        stop_policy="correct_latency",
        final_selection="best_correct_latency",
        latency_ceiling_ms=latency_ceiling_ms,
    )


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
    def __init__(self, latency_by_candidate: dict[str, float] | None = None) -> None:
        self.jobs: list[WorkerJob] = []
        self.latency_by_candidate = latency_by_candidate or {}

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
        latency = next(
            (
                value
                for name, value in self.latency_by_candidate.items()
                if f"candidate_name = {name!r}" in job.candidate_source
            ),
            1.0,
        )
        timing = (
            tuple(latency for _ in range(job.timing.trial_runs)) if correct and job.timing else ()
        )
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


class FailingWorker:
    def __init__(self) -> None:
        self.jobs: list[WorkerJob] = []

    def execute(self, job: WorkerJob) -> WorkerResult:
        self.jobs.append(job)
        raise RuntimeError("GPU worker unavailable")


class SealedFailingWorker(FakeWorker):
    def execute(self, job: WorkerJob) -> WorkerResult:
        if job.kind == "sealed":
            self.jobs.append(job)
            raise RuntimeError("sealed worker unavailable")
        return super().execute(job)


class CandidateTransportFailureWorker(FakeWorker):
    def __init__(self, category: FailureCategory) -> None:
        super().__init__({"good": 1.0})
        self.category = category
        self.failed = False

    def execute(self, job: WorkerJob) -> WorkerResult:
        if not self.failed:
            self.failed = True
            self.jobs.append(job)
            raise WorkerExecutionError(
                self.category,
                f"candidate {self.category}",
                health={"status": "healthy"},
                job_scoped=True,
            )
        return super().execute(job)


def _loop(
    tmp_path: Path,
    responses: list[str],
    *,
    worker: FakeWorker | None = None,
) -> tuple[CanaryAgentLoop, FakeClient, FakeWorker]:
    client = FakeClient(responses)
    worker = worker or FakeWorker()
    loop = CanaryAgentLoop(
        client=client,
        worker=worker,
        store=TrajectoryStore.create(tmp_path, "study", "trajectory"),
    )
    return loop, client, worker


def _run(
    loop: CanaryAgentLoop,
    *,
    policy: AgentLoopPolicy = R1_AGENT_LOOP_POLICY,
    max_calls: int = 4,
    dev_timing: TimingSpec | None = TEST_DEV_TIMING,
):
    task = get_task_pack("row-reduction-scale")
    return loop.run(
        trajectory_id="trajectory",
        model_ref="fake-model",
        initial_messages=build_initial_messages(
            task,
            load_target_card("triton-a100"),
            policy=policy,
        ),
        task=task,
        target=get_target_stack("triton-a100"),
        budget=AgentBudget(max_calls=max_calls, max_wall_seconds=1200.0),
        dev_timing=dev_timing,
        policy=policy,
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
    event_kinds = [
        json.loads(path.read_text(encoding="utf-8"))["kind"]
        for path in sorted((loop.store.run_directory / "events").glob("*.json"))
    ]
    assert event_kinds == [
        "trajectory_started",
        "request_started",
        "response_received",
        "dev_finished",
        "request_started",
        "response_received",
        "dev_finished",
        "agent_finished",
        "sealed_finished",
        "sealed_finished",
        "trajectory_terminal",
    ]
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


def test_candidate_only_controller_stops_when_correct_latency_meets_ceiling(
    tmp_path: Path,
) -> None:
    policy = _gate_policy(1.25)
    loop, client, worker = _loop(
        tmp_path,
        [_candidate_only("bad"), _candidate_only("good-fast"), _candidate_only("unused")],
        worker=FakeWorker({"good-fast": 1.0}),
    )

    outcome = _run(loop, policy=policy, max_calls=3)

    assert outcome.status == "finished"
    assert outcome.calls == 2
    assert len(client.requests) == 2
    assert [job.kind for job in worker.jobs] == ["dev", "dev", "sealed", "sealed"]
    assert outcome.final_candidate_sha256 == outcome.dev_results[1].candidate_sha256
    verify_trajectory(loop.store.run_directory)


def test_best_correct_latency_can_select_an_earlier_candidate(tmp_path: Path) -> None:
    policy = _gate_policy(0.5)
    loop, client, worker = _loop(
        tmp_path,
        [
            _candidate_only("good-slow"),
            _candidate_only("good-fast"),
            _candidate_only("good-medium"),
        ],
        worker=FakeWorker(
            {
                "good-slow": 2.0,
                "good-fast": 1.0,
                "good-medium": 1.5,
            }
        ),
    )

    outcome = _run(loop, policy=policy, max_calls=3)

    assert outcome.status == "call_limit"
    assert outcome.calls == 3
    assert outcome.final_candidate_sha256 == outcome.dev_results[1].candidate_sha256
    assert "good-fast" in worker.jobs[-1].candidate_source
    second_history = client.requests[1].messages  # type: ignore[attr-defined]
    assert any("latency_ceiling_ms" in message.content for message in second_history)
    assert any("meets_latency_ceiling" in message.content for message in second_history)
    verify_trajectory(loop.store.run_directory)


def test_best_correct_latency_falls_back_to_last_complete_candidate(tmp_path: Path) -> None:
    policy = _gate_policy(0.5)
    loop, client, worker = _loop(
        tmp_path,
        [_candidate_only("bad-first"), "invalid", _candidate_only("bad-last")],
    )

    outcome = _run(loop, policy=policy, max_calls=3)

    assert outcome.status == "call_limit"
    assert outcome.calls == 3
    assert len(client.requests) == 3
    assert len(outcome.dev_results) == 2
    assert outcome.final_candidate_sha256 == outcome.dev_results[-1].candidate_sha256
    assert "bad-last" in worker.jobs[-1].candidate_source
    verify_trajectory(loop.store.run_directory)


def test_default_r1_policy_still_selects_the_last_candidate(tmp_path: Path) -> None:
    loop, _client, worker = _loop(
        tmp_path,
        [_candidate("good-fast", "CONTINUE"), _candidate("good-slow", "FINISH")],
        worker=FakeWorker({"good-fast": 0.5, "good-slow": 2.0}),
    )

    outcome = _run(loop)

    assert outcome.status == "finished"
    assert outcome.final_candidate_sha256 == outcome.dev_results[-1].candidate_sha256
    assert "good-slow" in worker.jobs[-1].candidate_source
    verify_trajectory(loop.store.run_directory)


def test_latency_policy_requires_dev_timing_before_first_request(tmp_path: Path) -> None:
    policy = _gate_policy(1.25)
    loop, client, worker = _loop(tmp_path, [_candidate_only("good")])

    with pytest.raises(ValueError, match="require dev timing"):
        _run(loop, policy=policy, max_calls=3, dev_timing=None)

    assert client.requests == []
    assert worker.jobs == []


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


def test_worker_infrastructure_failure_stops_further_provider_calls(tmp_path: Path) -> None:
    client = FakeClient([_candidate("first", "CONTINUE"), _candidate("must-not-run", "FINISH")])
    worker = FailingWorker()
    loop = CanaryAgentLoop(
        client=client,
        worker=worker,
        store=TrajectoryStore.create(tmp_path, "study", "trajectory"),
    )

    outcome = _run(loop)

    assert outcome.status == "worker_error"
    assert outcome.calls == 1
    assert len(client.requests) == 1
    assert [job.kind for job in worker.jobs] == ["dev"]
    verify_trajectory(loop.store.run_directory)


@pytest.mark.parametrize(
    ("category", "worker_status"),
    (("timeout", "timeout"), ("oom", "runtime_error")),
)
def test_capability_policy_returns_candidate_transport_failures_as_feedback(
    tmp_path: Path,
    category: FailureCategory,
    worker_status: str,
) -> None:
    policy = _gate_policy(1.25)
    worker = CandidateTransportFailureWorker(category)
    loop, client, _ = _loop(
        tmp_path,
        [_candidate_only("first"), _candidate_only("good")],
        worker=worker,
    )

    outcome = _run(loop, policy=policy, max_calls=3)

    assert outcome.status == "finished"
    assert outcome.calls == 2
    assert [result.status for result in outcome.dev_results] == [
        worker_status,
        "completed",
    ]
    assert outcome.dev_results[0].metadata["failure_category"] == category
    assert outcome.dev_results[0].metadata["failure_scope"] == "job"
    second_history = client.requests[1].messages  # type: ignore[attr-defined]
    assert any(worker_status in message.content for message in second_history)


def test_r1_policy_keeps_transport_timeout_terminal(tmp_path: Path) -> None:
    worker = CandidateTransportFailureWorker("timeout")
    loop, client, _ = _loop(
        tmp_path,
        [_candidate("first", "CONTINUE"), _candidate("unused", "FINISH")],
        worker=worker,
    )

    outcome = _run(loop)

    assert outcome.status == "worker_error"
    assert outcome.calls == 1
    assert len(client.requests) == 1
    assert outcome.dev_results[0].status == "worker_error"


def test_capability_policy_keeps_unscoped_transport_timeout_terminal(tmp_path: Path) -> None:
    class UnscopedTimeoutWorker(FakeWorker):
        def execute(self, job: WorkerJob) -> WorkerResult:
            self.jobs.append(job)
            raise WorkerExecutionError(
                "timeout",
                "SSH transport stalled",
                health={"status": "healthy"},
            )

    policy = _gate_policy(1.25)
    worker = UnscopedTimeoutWorker()
    loop, client, _ = _loop(
        tmp_path,
        [_candidate_only("first"), _candidate_only("unused")],
        worker=worker,
    )

    outcome = _run(loop, policy=policy, max_calls=3)

    assert outcome.status == "worker_error"
    assert outcome.calls == 1
    assert len(client.requests) == 1
    assert outcome.dev_results[0].status == "worker_error"


def test_capability_policy_keeps_unhealthy_job_timeout_terminal(tmp_path: Path) -> None:
    class UnhealthyTimeoutWorker(FakeWorker):
        def execute(self, job: WorkerJob) -> WorkerResult:
            self.jobs.append(job)
            raise WorkerExecutionError(
                "timeout",
                "job timed out before GPU health failed",
                health={"status": "unhealthy"},
                job_scoped=True,
            )

    policy = _gate_policy(1.25)
    worker = UnhealthyTimeoutWorker()
    loop, client, _ = _loop(
        tmp_path,
        [_candidate_only("first"), _candidate_only("unused")],
        worker=worker,
    )

    outcome = _run(loop, policy=policy, max_calls=3)

    assert outcome.status == "worker_error"
    assert outcome.calls == 1
    assert len(client.requests) == 1
    assert outcome.dev_results[0].status == "worker_error"


def test_sealed_worker_failure_overrides_finished_status_and_stops_sealed_jobs(
    tmp_path: Path,
) -> None:
    client = FakeClient([_candidate("good", "FINISH")])
    worker = SealedFailingWorker()
    loop = CanaryAgentLoop(
        client=client,
        worker=worker,
        store=TrajectoryStore.create(tmp_path, "study", "trajectory"),
    )

    outcome = _run(loop)

    assert outcome.status == "worker_error"
    assert outcome.error and "sealed worker unavailable" in outcome.error
    assert outcome.first_sealed_result is not None
    assert outcome.first_sealed_result.status == "worker_error"
    assert outcome.final_sealed_result is None
    assert [job.kind for job in worker.jobs] == ["dev", "sealed"]
    verify_trajectory(loop.store.run_directory)


def test_provider_response_past_wall_budget_stops_before_candidate_execution(
    tmp_path: Path,
) -> None:
    ticks = iter((0.0, 0.0, 1201.0))
    client = FakeClient([_candidate("good", "FINISH")])
    worker = FakeWorker()
    loop = CanaryAgentLoop(
        client=client,
        worker=worker,
        store=TrajectoryStore.create(tmp_path, "study", "trajectory"),
        monotonic=lambda: next(ticks),
    )

    outcome = _run(loop)

    assert outcome.status == "budget_exhausted"
    assert outcome.calls == 1
    assert outcome.first_candidate_sha256 is None
    assert worker.jobs == []
    verify_trajectory(loop.store.run_directory)
