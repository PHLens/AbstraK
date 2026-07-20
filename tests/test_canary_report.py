from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest

from abstrak.canary.artifacts import verify_trajectory
from abstrak.canary.baselines import BASELINE_VARIANTS
from abstrak.canary.contracts import (
    CaseResult,
    TimingSpec,
    TrajectoryOutcome,
    WorkerJob,
    WorkerResult,
)
from abstrak.canary.gates import GateRecord
from abstrak.canary.report import (
    AnalysisReportError,
    TrajectoryArtifact,
    build_analysis_report,
    write_analysis_report,
)
from abstrak.canary.schedule import R1_TARGETS, R1_TASKS, build_r1_schedule
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import get_task_pack
from abstrak.canary.timing import TimingAttemptSummary, TimingProtocolSummary

NOW = datetime(2026, 7, 20, tzinfo=timezone.utc)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _qualified_result(identifier: str, candidate_sha256: str) -> WorkerResult:
    return WorkerResult(
        job_id=f"job-{identifier}",
        job_sha256="1" * 64,
        input_sha256="2" * 64,
        candidate_sha256=candidate_sha256,
        status="completed",
        compiled=True,
        correct=True,
        cases=(
            CaseResult(
                case_id="sealed-1",
                status="pass",
                correct=True,
                max_abs_error=0.0,
                max_rel_error=0.0,
                output_finite=True,
                inputs_unchanged=True,
            ),
        ),
    )


def _trajectory(cell, *, qualified: bool = False, status: str = "no_candidate"):
    candidate_sha256 = _hash(cell.trajectory_id)
    result = _qualified_result(str(cell.ordinal), candidate_sha256) if qualified else None
    error = "service unavailable" if status in {"provider_error", "worker_error"} else None
    outcome = TrajectoryOutcome(
        trajectory_id=cell.trajectory_id,
        status=status,
        calls=1,
        known_input_tokens=10,
        known_output_tokens=20,
        usage_complete=status != "provider_error",
        first_candidate_sha256=candidate_sha256 if qualified else None,
        final_candidate_sha256=candidate_sha256 if qualified else None,
        first_sealed_result=result,
        final_sealed_result=result,
        started_at_utc=NOW,
        finished_at_utc=NOW + timedelta(seconds=2),
        error=error,
    )
    return TrajectoryArtifact(
        cell=cell,
        outcome=outcome,
        artifact_directory=f"/artifacts/{cell.trajectory_id}",
    )


def _gate_summary(
    task_id: str,
    target_id: str,
    latency: float,
    *,
    kind: Literal["oracle", "baseline"],
) -> TimingProtocolSummary:
    task = get_task_pack(task_id)
    target = get_target_stack(target_id)
    source = f"gate source {task_id} {target_id} {latency}"
    candidate_sha256 = _hash(source)
    timing = TimingSpec(warmup_runs=1, trial_runs=2, repetitions=1)
    job = WorkerJob(
        job_id=f"gate-{task_id}-{target_id}-{kind}",
        kind=kind,
        task=task,
        target=target,
        case_ids=tuple(case.id for case in task.sealed_cases),
        candidate_source=source,
        candidate_sha256=candidate_sha256,
        timing=timing,
        device="cuda:0",
    )
    cases = tuple(
        CaseResult(
            case_id=case.id,
            status="pass",
            correct=True,
            max_abs_error=0.0,
            max_rel_error=0.0,
            output_finite=True,
            inputs_unchanged=True,
        )
        for case in task.sealed_cases
    )
    result = WorkerResult(
        job_id=job.job_id,
        job_sha256=job.sha256,
        input_sha256=job.input_sha256,
        candidate_sha256=candidate_sha256,
        status="completed",
        compiled=True,
        correct=True,
        cases=cases,
        timing_ms=(latency, latency),
        timing_cv=0.0,
    )
    attempt = TimingAttemptSummary(
        attempt=1,
        status="stable",
        stable=True,
        jobs=(job,),
        results=(result,),
        process_medians_ms=(latency,),
        process_cvs=(0.0,),
        across_process_cv=0.0,
        median_ms=latency,
    )
    return TimingProtocolSummary(
        job_prefix=f"gate-{task_id}-{target_id}-{kind}",
        task_id=task_id,
        target_id=target_id,
        candidate_sha256=candidate_sha256,
        job_kind=kind,
        device="cuda:0",
        timing=timing,
        status="stable",
        stable=True,
        attempts=(attempt,),
        jobs=(job,),
        results=(result,),
        median_ms=latency,
    )


def _gate_records() -> tuple[tuple[GateRecord, ...], tuple[GateRecord, ...]]:
    oracles = tuple(
        GateRecord(
            kind="oracle",
            task_id=task,
            target_id=target,
            source_sha256=(
                summary := _gate_summary(task, target, 0.8, kind="oracle")
            ).candidate_sha256,
            artifact_directory=f"/gates/oracle-{task}-{target}",
            summary=summary,
        )
        for task in R1_TASKS
        for target in R1_TARGETS
    )
    baselines = tuple(
        GateRecord(
            kind="baseline",
            task_id=task,
            target_id="triton-a100",
            variant=variant,
            source_sha256=(
                summary := _gate_summary(
                    task,
                    "triton-a100",
                    1.0 + BASELINE_VARIANTS.index(variant),
                    kind="baseline",
                )
            ).candidate_sha256,
            artifact_directory=f"/gates/baseline-{task}-{variant}",
            summary=summary,
        )
        for task in R1_TASKS
        for variant in BASELINE_VARIANTS
    )
    return oracles, baselines


def _matrix():
    schedule = build_r1_schedule()
    qualified_ids = {cell.trajectory_id for cell in schedule.formal[:19]}
    infrastructure = {
        schedule.formal[-3].trajectory_id: "provider_error",
        schedule.formal[-2].trajectory_id: "provider_error",
        schedule.formal[-1].trajectory_id: "worker_error",
    }
    formal = tuple(
        _trajectory(
            cell,
            qualified=cell.trajectory_id in qualified_ids,
            status=(
                "finished"
                if cell.trajectory_id in qualified_ids
                else infrastructure.get(cell.trajectory_id, "no_candidate")
            ),
        )
        for cell in schedule.formal
    )
    supported: set[str] = set()
    shakeout = []
    for cell in schedule.shakeout:
        qualifies = cell.target_id not in supported
        if qualifies:
            supported.add(cell.target_id)
        shakeout.append(
            _trajectory(
                cell,
                qualified=qualifies,
                status="finished" if qualifies else "no_candidate",
            )
        )
    oracles, baselines = _gate_records()
    return schedule, formal, tuple(shakeout), oracles, baselines


@dataclass(frozen=True)
class TimingRecord:
    trajectory_id: str
    task_id: str
    target_id: str
    candidate_labels: tuple[Literal["first", "final"], ...]
    source_sha256: str
    summary: TimingProtocolSummary


def _candidate_timing(trajectory: TrajectoryArtifact, *, status: str = "stable") -> TimingRecord:
    candidate_sha256 = trajectory.outcome.final_candidate_sha256
    assert candidate_sha256 is not None
    task = get_task_pack(trajectory.cell.task_id)
    target = get_target_stack(trajectory.cell.target_id)
    source = trajectory.cell.trajectory_id
    assert _hash(source) == candidate_sha256
    timing = TimingSpec(warmup_runs=1, trial_runs=2, repetitions=1)
    prefix = f"timing-{trajectory.cell.trajectory_id}-first-final"
    job = WorkerJob(
        job_id=f"{prefix}-timing-a1-p1",
        kind="sealed",
        task=task,
        target=target,
        case_ids=tuple(case.id for case in task.sealed_cases),
        candidate_source=source,
        candidate_sha256=candidate_sha256,
        timing=timing,
        device="cuda:0",
    )
    cv = 0.01 if status == "stable" else 0.08
    latency = 0.5
    result = WorkerResult(
        job_id=job.job_id,
        job_sha256=job.sha256,
        input_sha256=job.input_sha256,
        candidate_sha256=candidate_sha256,
        status="completed",
        compiled=True,
        correct=True,
        cases=tuple(
            CaseResult(
                case_id=case.id,
                status="pass",
                correct=True,
                max_abs_error=0.0,
                max_rel_error=0.0,
                output_finite=True,
                inputs_unchanged=True,
            )
            for case in task.sealed_cases
        ),
        timing_ms=(latency, latency),
        timing_cv=cv,
    )
    attempt = TimingAttemptSummary(
        attempt=1,
        status=status,
        stable=status == "stable",
        jobs=(job,),
        results=(result,),
        process_medians_ms=(latency,),
        process_cvs=(cv,),
        across_process_cv=cv,
        median_ms=latency,
    )
    summary = TimingProtocolSummary(
        job_prefix=prefix,
        task_id=task.id,
        target_id=target.id,
        candidate_sha256=candidate_sha256,
        job_kind="sealed",
        device="cuda:0",
        timing=timing,
        status=status,
        stable=status == "stable",
        attempts=(attempt,),
        jobs=(job,),
        results=(result,),
        median_ms=latency if status == "stable" else None,
    )
    return TimingRecord(
        trajectory_id=trajectory.outcome.trajectory_id,
        task_id=trajectory.cell.task_id,
        target_id=trajectory.cell.target_id,
        candidate_labels=("first", "final"),
        source_sha256=candidate_sha256,
        summary=summary,
    )


def _report(*, timing_records=()):
    schedule, formal, shakeout, oracles, baselines = _matrix()
    return build_analysis_report(
        formal_records=formal,
        oracle_records=oracles,
        baseline_records=baselines,
        timing_records=timing_records,
        shakeout_records=shakeout,
        schedule=schedule,
    )


def _counts(values) -> dict[str, int]:
    return {value.name: value.count for value in values}


def test_report_preserves_qualification_when_timing_is_missing_and_censors_infra() -> None:
    report = _report()

    assert report.formal_coverage.received_trajectories == 48
    assert report.formal_coverage.observed_trajectories == 45
    assert report.formal_coverage.infrastructure_censored == 3
    assert report.formal_coverage.qualified_at_first == 19
    assert report.formal_coverage.qualified_at_final == 19
    assert report.analysis.qualified_at_final == 19
    assert report.analysis.outcome == "inconclusive_infrastructure"
    assert _counts(report.timing_coverage.first_status_counts)["missing"] == 19
    assert _counts(report.timing_coverage.final_status_counts)["missing"] == 19
    qualified = tuple(record for record in report.measurements if record.qualified_at_final)
    assert len(qualified) == 19
    assert all(record.candidate_latency_ms is None for record in qualified)
    assert all(record.efficiency is None for record in qualified)


def test_deduplicated_stable_timing_populates_both_audits_but_only_final_performance() -> None:
    _, formal, _, _, _ = _matrix()
    timing = _candidate_timing(formal[0])

    report = _report(timing_records=(timing,))

    assert _counts(report.timing_coverage.first_status_counts) == {
        "stable": 1,
        "unstable": 0,
        "worker_failure": 0,
        "correctness_failure": 0,
        "missing": 18,
    }
    assert _counts(report.timing_coverage.final_status_counts)["stable"] == 1
    measurement = next(
        item
        for item in report.measurements
        if item.agent_id == formal[0].cell.agent_id
        and item.task_id == formal[0].cell.task_id
        and item.target_id == formal[0].cell.target_id
        and item.replicate == formal[0].cell.replicate
    )
    assert measurement.candidate_latency_ms == 0.5
    assert measurement.timing_cvs == (0.01,)


def test_unstable_timing_keeps_cv_evidence_without_publishing_a_latency() -> None:
    _, formal, _, _, _ = _matrix()
    report = _report(timing_records=(_candidate_timing(formal[0], status="unstable"),))

    assert _counts(report.timing_coverage.final_status_counts)["unstable"] == 1
    audit = next(
        item
        for item in report.timing_coverage.audits
        if item.trajectory_id == formal[0].outcome.trajectory_id
        and item.candidate_label == "final"
    )
    assert audit.median_ms is None
    assert audit.attempt_worst_cvs == (0.08,)
    measurement = next(
        item
        for item in report.measurements
        if item.agent_id == formal[0].cell.agent_id
        and item.task_id == formal[0].cell.task_id
        and item.target_id == formal[0].cell.target_id
        and item.replicate == formal[0].cell.replicate
    )
    assert measurement.candidate_latency_ms is None
    assert measurement.timing_cvs == (0.08,)


def test_timing_hash_mismatch_is_rejected() -> None:
    _, formal, _, _, _ = _matrix()
    timing = replace(_candidate_timing(formal[0]), source_sha256="f" * 64)

    with pytest.raises(AnalysisReportError, match="identity mismatch"):
        _report(timing_records=(timing,))


def test_analysis_report_is_sealed_and_only_identical_inputs_resume(tmp_path: Path) -> None:
    report = _report()

    directory, resumed = write_analysis_report(report, artifact_root=tmp_path)
    resumed_directory, resumed_again = write_analysis_report(report, artifact_root=tmp_path)

    assert resumed is False
    assert resumed_again is True
    assert resumed_directory == directory
    verify_trajectory(directory)
