from __future__ import annotations

from pathlib import Path

import pytest

from abstrak.canary.artifacts import verify_trajectory
from abstrak.canary.contracts import CaseResult, TimingSpec, WorkerJob, WorkerResult
from abstrak.canary.gates import (
    GateError,
    fastest_stable_baselines,
    run_baseline_gates,
    run_oracle_gates,
)
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import get_task_pack


class FakeWorker:
    def __init__(self, latency_by_marker: dict[str, float] | None = None) -> None:
        self.jobs: list[WorkerJob] = []
        self.latency_by_marker = latency_by_marker or {}

    def execute(self, job: WorkerJob) -> WorkerResult:
        self.jobs.append(job)
        latency = next(
            (
                value
                for marker, value in self.latency_by_marker.items()
                if marker in job.candidate_source
            ),
            1.0,
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
        assert job.timing is not None
        return WorkerResult(
            job_id=job.job_id,
            job_sha256=job.sha256,
            input_sha256=job.input_sha256,
            candidate_sha256=job.candidate_sha256,
            status="completed",
            compiled=True,
            correct=True,
            cases=cases,
            timing_ms=tuple(latency for _ in range(job.timing.trial_runs)),
            timing_cv=0.0,
        )


def test_oracle_gate_is_sealed_and_resumes_without_worker_calls(tmp_path: Path) -> None:
    worker = FakeWorker()
    timing = TimingSpec(warmup_runs=1, trial_runs=2, repetitions=2)
    arguments = {
        "tasks": (get_task_pack("rmsnorm-static"),),
        "targets": (get_target_stack("triton-a100"),),
        "root": tmp_path,
        "timing": timing,
    }

    first = run_oracle_gates(worker, **arguments)
    second = run_oracle_gates(worker, **arguments)

    assert first == second
    assert len(worker.jobs) == 2
    assert {job.kind for job in worker.jobs} == {"oracle"}
    verify_trajectory(first[0].artifact_directory)


def test_baseline_gate_selects_fastest_stable_variant(tmp_path: Path) -> None:
    worker = FakeWorker(
        {
            "@torch.compile": 0.8,
            "F.rms_norm": 0.5,
            "x_fp32 =": 1.0,
        }
    )
    records = run_baseline_gates(
        worker,
        tasks=(get_task_pack("rmsnorm-static"),),
        target=get_target_stack("triton-a100"),
        root=tmp_path,
        timing=TimingSpec(warmup_runs=1, trial_runs=2, repetitions=2),
    )

    selected = fastest_stable_baselines(records)

    assert len(records) == 3
    assert selected["rmsnorm-static"].variant == "vendor"
    assert {job.kind for job in worker.jobs} == {"baseline"}


def test_fastest_baseline_rejects_an_oracle_only_input(tmp_path: Path) -> None:
    records = run_oracle_gates(
        FakeWorker(),
        tasks=(get_task_pack("rmsnorm-static"),),
        targets=(get_target_stack("triton-a100"),),
        root=tmp_path,
        timing=TimingSpec(warmup_runs=1, trial_runs=2, repetitions=2),
    )

    assert fastest_stable_baselines(records) == {}


def test_resume_rejects_changed_timing_contract(tmp_path: Path) -> None:
    worker = FakeWorker()
    arguments = {
        "tasks": (get_task_pack("rmsnorm-static"),),
        "targets": (get_target_stack("triton-a100"),),
        "root": tmp_path,
    }
    run_oracle_gates(
        worker,
        timing=TimingSpec(warmup_runs=1, trial_runs=2, repetitions=2),
        **arguments,
    )

    with pytest.raises(GateError, match="does not match frozen inputs"):
        run_oracle_gates(
            worker,
            timing=TimingSpec(warmup_runs=2, trial_runs=2, repetitions=2),
            **arguments,
        )
