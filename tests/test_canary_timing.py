from __future__ import annotations

import hashlib
from collections import deque

from pydantic import ValidationError

from abstrak.canary.contracts import (
    CaseResult,
    TargetStackSpec,
    TaskPackSpec,
    TimingSpec,
    WorkerJob,
    WorkerResult,
)
from abstrak.canary.tasks import get_task_pack
from abstrak.canary.timing import run_timing_protocol


def _target() -> TargetStackSpec:
    return TargetStackSpec(
        id="triton-a100",
        backend="triton",
        version="3.7.1",
        card_path="targets/triton.md",
        card_sha256="1" * 64,
        adapter="kernelbench",
    )


class FakeWorker:
    def __init__(
        self,
        measurements: list[tuple[float, float]],
        *,
        failure_at: int | None = None,
        wrong_at: int | None = None,
    ) -> None:
        self.measurements = deque(measurements)
        self.failure_at = failure_at
        self.wrong_at = wrong_at
        self.jobs: list[WorkerJob] = []

    def execute(self, job: WorkerJob) -> WorkerResult:
        self.jobs.append(job)
        invocation = len(self.jobs)
        if invocation == self.failure_at:
            raise RuntimeError("GPU worker unavailable")
        median, cv = self.measurements.popleft()
        correct = invocation != self.wrong_at
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
        samples = tuple(median for _ in range(job.timing.trial_runs)) if correct else ()
        return WorkerResult(
            job_id=job.job_id,
            job_sha256=job.sha256,
            input_sha256=job.input_sha256,
            candidate_sha256=job.candidate_sha256,
            status="completed" if correct else "wrong_result",
            compiled=True,
            correct=correct,
            cases=cases,
            timing_ms=samples,
            timing_cv=cv if correct else None,
        )


def _run(worker: FakeWorker, timing: TimingSpec | None = None):
    task: TaskPackSpec = get_task_pack("row-reduction-scale")
    return run_timing_protocol(
        worker,
        task=task,
        target=_target(),
        source="class ModelNew: pass\n",
        job_prefix="formal-cell-r1",
        timing=timing or TimingSpec(),
    )


def test_stable_protocol_preserves_jobs_hashes_and_samples() -> None:
    worker = FakeWorker([(1.0, 0.01), (1.01, 0.02), (0.99, 0.03)])

    summary = _run(worker)

    assert summary.status == "stable"
    assert summary.stable is True
    assert summary.median_ms == 1.0
    assert len(summary.attempts) == 1
    assert summary.attempts[0].process_medians_ms == (1.0, 1.01, 0.99)
    assert summary.attempts[0].process_cvs == (0.01, 0.02, 0.03)
    assert summary.attempts[0].across_process_cv is not None
    assert summary.attempts[0].across_process_cv < 0.05
    assert [job.job_id for job in summary.jobs] == [
        "formal-cell-r1-timing-a1-p1",
        "formal-cell-r1-timing-a1-p2",
        "formal-cell-r1-timing-a1-p3",
    ]
    expected_hash = hashlib.sha256(b"class ModelNew: pass\n").hexdigest()
    task = get_task_pack("row-reduction-scale")
    for job, result in zip(summary.jobs, summary.results, strict=True):
        assert job.kind == "sealed"
        assert job.case_ids == tuple(case.id for case in task.sealed_cases)
        assert job.candidate_sha256 == expected_hash == summary.candidate_sha256
        assert job.timing == TimingSpec(repetitions=1)
        assert job.timing.discard_first == 1
        assert len(result.timing_ms) == 100
        assert result.verify_for_job(job) is result
    assert summary.jobs == summary.attempts[0].jobs
    assert summary.results == summary.attempts[0].results
    try:
        summary.stable = False
    except ValidationError:
        pass
    else:
        raise AssertionError("timing summaries must be frozen")


def test_full_retry_can_recover_stability() -> None:
    worker = FakeWorker(
        [
            (1.0, 0.06),
            (1.0, 0.06),
            (1.0, 0.06),
            (2.0, 0.01),
            (2.01, 0.01),
            (1.99, 0.01),
        ]
    )

    summary = _run(worker)

    assert summary.status == "stable"
    assert summary.stable is True
    assert summary.median_ms == 2.0
    assert [attempt.status for attempt in summary.attempts] == ["unstable", "stable"]
    assert len(summary.jobs) == len(summary.results) == 6
    assert [job.job_id for job in summary.attempts[1].jobs] == [
        "formal-cell-r1-timing-a2-p1",
        "formal-cell-r1-timing-a2-p2",
        "formal-cell-r1-timing-a2-p3",
    ]


def test_retry_remains_unstable_after_two_complete_attempts() -> None:
    worker = FakeWorker(
        [
            (1.0, 0.01),
            (2.0, 0.01),
            (3.0, 0.01),
            (2.0, 0.01),
            (4.0, 0.01),
            (6.0, 0.01),
        ]
    )

    summary = _run(worker)

    assert summary.status == "unstable"
    assert summary.stable is False
    assert summary.median_ms is None
    assert [attempt.status for attempt in summary.attempts] == ["unstable", "unstable"]
    assert all(attempt.across_process_cv > 0.05 for attempt in summary.attempts)  # type: ignore[operator]
    assert len(worker.jobs) == 6


def test_worker_failure_stops_without_retry() -> None:
    worker = FakeWorker([(1.0, 0.01)], failure_at=1)

    summary = _run(worker)

    assert summary.status == "worker_failure"
    assert summary.stable is False
    assert len(summary.attempts) == 1
    assert len(summary.jobs) == len(worker.jobs) == 1
    assert summary.results == ()
    assert summary.error == "RuntimeError: GPU worker unavailable"


def test_correctness_failure_stops_without_retry() -> None:
    worker = FakeWorker([(1.0, 0.01)], wrong_at=1)

    summary = _run(worker, TimingSpec(warmup_runs=2, trial_runs=4, repetitions=3))

    assert summary.status == "correctness_failure"
    assert len(summary.jobs) == len(summary.results) == len(worker.jobs) == 1
    assert summary.results[0].status == "wrong_result"
