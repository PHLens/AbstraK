"""Clean-process timing aggregation for the formal A100 R1 protocol."""

from __future__ import annotations

import hashlib
import math
import statistics
from typing import Literal

from pydantic import Field, field_validator, model_validator

from abstrak.canary.contracts import (
    IDENTIFIER_PATTERN,
    CanaryModel,
    TargetStackSpec,
    TaskPackSpec,
    TimingSpec,
    WorkerJob,
    WorkerResult,
)
from abstrak.canary.loop import WorkerExecutor

TimingRunStatus = Literal[
    "stable",
    "unstable",
    "worker_failure",
    "correctness_failure",
]

DEFAULT_FORMAL_TIMING = TimingSpec()
_INFRASTRUCTURE_FAILURES = {"environment_error", "timeout", "worker_error"}


class TimingAttemptSummary(CanaryModel):
    """Measurements and stability decision for one complete or failed attempt."""

    attempt: int = Field(ge=1, le=2)
    status: TimingRunStatus
    stable: bool
    jobs: tuple[WorkerJob, ...] = Field(min_length=1)
    results: tuple[WorkerResult, ...] = ()
    process_medians_ms: tuple[float, ...] = ()
    process_cvs: tuple[float, ...] = ()
    across_process_cv: float | None = Field(default=None, ge=0)
    median_ms: float | None = Field(default=None, gt=0)
    error: str | None = None

    @field_validator("process_medians_ms")
    @classmethod
    def medians_are_finite_and_positive(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("process medians must be finite and positive")
        return values

    @field_validator("process_cvs", "across_process_cv")
    @classmethod
    def cvs_are_finite(cls, value: tuple[float, ...] | float | None):
        values = value if isinstance(value, tuple) else (() if value is None else (value,))
        if any(not math.isfinite(item) or item < 0 for item in values):
            raise ValueError("timing CVs must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def decision_is_consistent(self) -> TimingAttemptSummary:
        if len(self.process_medians_ms) != len(self.process_cvs):
            raise ValueError("each process median must have one process CV")
        if len(self.results) > len(self.jobs):
            raise ValueError("an attempt cannot contain more results than jobs")
        if self.stable != (self.status == "stable"):
            raise ValueError("stable must agree with status")
        complete = len(self.results) == len(self.jobs)
        has_aggregate = self.across_process_cv is not None and self.median_ms is not None
        if self.status in {"stable", "unstable"}:
            if not complete or len(self.process_medians_ms) != len(self.results):
                raise ValueError("timing decisions require a complete process set")
            if not has_aggregate or self.error is not None:
                raise ValueError("timing decisions require aggregate metrics and no error")
        elif self.error is None:
            raise ValueError("failed timing attempts require an error")
        return self


class TimingProtocolSummary(CanaryModel):
    """Terminal formal-timing decision with a complete reproducibility trail."""

    schema_version: Literal["canary-timing-protocol.v1"] = "canary-timing-protocol.v1"
    job_prefix: str = Field(pattern=IDENTIFIER_PATTERN)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    device: str = Field(pattern=r"^cuda:[0-9]+$")
    timing: TimingSpec
    status: TimingRunStatus
    stable: bool
    attempts: tuple[TimingAttemptSummary, ...] = Field(min_length=1, max_length=2)
    jobs: tuple[WorkerJob, ...] = Field(min_length=1)
    results: tuple[WorkerResult, ...] = ()
    median_ms: float | None = Field(default=None, gt=0)
    error: str | None = None

    @model_validator(mode="after")
    def terminal_decision_is_consistent(self) -> TimingProtocolSummary:
        if self.stable != (self.status == "stable"):
            raise ValueError("stable must agree with status")
        if self.jobs != tuple(job for attempt in self.attempts for job in attempt.jobs):
            raise ValueError("jobs must contain every attempt job in execution order")
        if self.results != tuple(result for attempt in self.attempts for result in attempt.results):
            raise ValueError("results must contain every returned result in execution order")
        if self.status == "stable":
            if self.median_ms != self.attempts[-1].median_ms or self.error is not None:
                raise ValueError("stable summaries require the final stable median and no error")
        elif self.median_ms is not None:
            raise ValueError("non-stable summaries cannot expose a final median")
        if self.status in {"worker_failure", "correctness_failure"}:
            if self.error is None or self.error != self.attempts[-1].error:
                raise ValueError("failed summaries require the terminal attempt error")
        elif self.error is not None:
            raise ValueError("CV stability decisions cannot contain an execution error")
        return self


def _coefficient_of_variation(values: tuple[float, ...]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean


def _job(
    *,
    task: TaskPackSpec,
    target: TargetStackSpec,
    source: str,
    candidate_sha256: str,
    job_prefix: str,
    device: str,
    timing: TimingSpec,
    attempt: int,
    repetition: int,
) -> WorkerJob:
    process_timing = timing.model_copy(update={"repetitions": 1})
    return WorkerJob(
        job_id=f"{job_prefix}-timing-a{attempt}-p{repetition}",
        kind="sealed",
        task=task,
        target=target,
        case_ids=tuple(case.id for case in task.sealed_cases),
        candidate_source=source,
        candidate_sha256=candidate_sha256,
        timing=process_timing,
        device=device,
    )


def _failed_attempt(
    *,
    attempt: int,
    status: Literal["worker_failure", "correctness_failure"],
    jobs: list[WorkerJob],
    results: list[WorkerResult],
    medians: list[float],
    cvs: list[float],
    error: str,
) -> TimingAttemptSummary:
    return TimingAttemptSummary(
        attempt=attempt,
        status=status,
        stable=False,
        jobs=tuple(jobs),
        results=tuple(results),
        process_medians_ms=tuple(medians),
        process_cvs=tuple(cvs),
        error=error,
    )


def _summary(
    *,
    job_prefix: str,
    task: TaskPackSpec,
    target: TargetStackSpec,
    candidate_sha256: str,
    device: str,
    timing: TimingSpec,
    attempts: list[TimingAttemptSummary],
) -> TimingProtocolSummary:
    terminal = attempts[-1]
    return TimingProtocolSummary(
        job_prefix=job_prefix,
        task_id=task.id,
        target_id=target.id,
        candidate_sha256=candidate_sha256,
        device=device,
        timing=timing,
        status=terminal.status,
        stable=terminal.stable,
        attempts=tuple(attempts),
        jobs=tuple(job for attempt in attempts for job in attempt.jobs),
        results=tuple(result for attempt in attempts for result in attempt.results),
        median_ms=terminal.median_ms if terminal.stable else None,
        error=terminal.error,
    )


def run_timing_protocol(
    worker: WorkerExecutor,
    *,
    task: TaskPackSpec,
    target: TargetStackSpec,
    source: str,
    job_prefix: str,
    device: str = "cuda:0",
    timing: TimingSpec = DEFAULT_FORMAL_TIMING,
) -> TimingProtocolSummary:
    """Time a correct candidate in independent processes, with one full CV retry."""

    candidate_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    attempts: list[TimingAttemptSummary] = []
    for attempt_number in (1, 2):
        jobs: list[WorkerJob] = []
        results: list[WorkerResult] = []
        process_medians: list[float] = []
        process_cvs: list[float] = []
        for repetition in range(1, timing.repetitions + 1):
            job = _job(
                task=task,
                target=target,
                source=source,
                candidate_sha256=candidate_sha256,
                job_prefix=job_prefix,
                device=device,
                timing=timing,
                attempt=attempt_number,
                repetition=repetition,
            )
            jobs.append(job)
            try:
                result = worker.execute(job)
                results.append(result)
                result.verify_for_job(job)
            except Exception as error:
                failed = _failed_attempt(
                    attempt=attempt_number,
                    status="worker_failure",
                    jobs=jobs,
                    results=results,
                    medians=process_medians,
                    cvs=process_cvs,
                    error=f"{type(error).__name__}: {error}",
                )
                attempts.append(failed)
                return _summary(
                    job_prefix=job_prefix,
                    task=task,
                    target=target,
                    candidate_sha256=candidate_sha256,
                    device=device,
                    timing=timing,
                    attempts=attempts,
                )

            if result.status != "completed" or not result.correct:
                failure_status = (
                    "worker_failure"
                    if result.status in _INFRASTRUCTURE_FAILURES
                    else "correctness_failure"
                )
                failed = _failed_attempt(
                    attempt=attempt_number,
                    status=failure_status,
                    jobs=jobs,
                    results=results,
                    medians=process_medians,
                    cvs=process_cvs,
                    error=result.error or f"timing job ended with status {result.status}",
                )
                attempts.append(failed)
                return _summary(
                    job_prefix=job_prefix,
                    task=task,
                    target=target,
                    candidate_sha256=candidate_sha256,
                    device=device,
                    timing=timing,
                    attempts=attempts,
                )

            process_medians.append(float(statistics.median(result.timing_ms)))
            assert result.timing_cv is not None
            process_cvs.append(result.timing_cv)

        median_ms = float(statistics.median(process_medians))
        across_process_cv = _coefficient_of_variation(tuple(process_medians))
        stable = (
            all(process_cv <= timing.max_cv for process_cv in process_cvs)
            and across_process_cv <= timing.max_cv
        )
        attempt = TimingAttemptSummary(
            attempt=attempt_number,
            status="stable" if stable else "unstable",
            stable=stable,
            jobs=tuple(jobs),
            results=tuple(results),
            process_medians_ms=tuple(process_medians),
            process_cvs=tuple(process_cvs),
            across_process_cv=across_process_cv,
            median_ms=median_ms,
        )
        attempts.append(attempt)
        if stable or attempt_number == 2:
            return _summary(
                job_prefix=job_prefix,
                task=task,
                target=target,
                candidate_sha256=candidate_sha256,
                device=device,
                timing=timing,
                attempts=attempts,
            )

    raise AssertionError("timing protocol did not produce a terminal summary")
