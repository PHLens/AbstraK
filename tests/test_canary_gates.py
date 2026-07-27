from __future__ import annotations

from pathlib import Path

import pytest

from abstrak.canary.artifacts import verify_trajectory
from abstrak.canary.contracts import CaseResult, TimingSpec, WorkerJob, WorkerResult
from abstrak.canary.gates import (
    GateError,
    GateInfrastructureError,
    fastest_stable_baselines,
    run_baseline_gates,
    run_oracle_gates,
)
from abstrak.canary.remote import WorkerExecutionError
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


class FailOnceForTaskWorker(FakeWorker):
    def __init__(self, task_id: str) -> None:
        super().__init__()
        self.task_id = task_id
        self.failed = False

    def execute(self, job: WorkerJob) -> WorkerResult:
        if job.task.id == self.task_id and not self.failed:
            self.jobs.append(job)
            self.failed = True
            raise RuntimeError("transient worker outage")
        return super().execute(job)


class JobScopedOomWorker(FakeWorker):
    def execute(self, job: WorkerJob) -> WorkerResult:
        self.jobs.append(job)
        raise WorkerExecutionError(
            "oom",
            "candidate exceeds the available device memory",
            health={"status": "healthy"},
            job_scoped=True,
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


def test_worker_failure_is_not_sealed_and_can_resume_prior_gates(tmp_path: Path) -> None:
    worker = FailOnceForTaskWorker("row-reduction-scale")
    study_id = "infrastructure-retry"
    tasks = (
        get_task_pack("rmsnorm-static"),
        get_task_pack("row-reduction-scale"),
    )
    arguments = {
        "tasks": tasks,
        "targets": (get_target_stack("triton-a100"),),
        "root": tmp_path,
        "study_id": study_id,
        "timing": TimingSpec(warmup_runs=1, trial_runs=2, repetitions=1),
    }
    successful_path = tmp_path / study_id / "oracle-rmsnorm-static-triton-a100"
    failed_path = tmp_path / study_id / "oracle-row-reduction-scale-triton-a100"

    with pytest.raises(GateInfrastructureError, match="transient worker outage"):
        run_oracle_gates(worker, **arguments)

    verify_trajectory(successful_path)
    assert not failed_path.exists()
    assert [job.task.id for job in worker.jobs] == [
        "rmsnorm-static",
        "row-reduction-scale",
    ]

    records = run_oracle_gates(worker, **arguments)

    assert len(records) == 2
    assert [job.task.id for job in worker.jobs] == [
        "rmsnorm-static",
        "row-reduction-scale",
        "row-reduction-scale",
    ]
    verify_trajectory(failed_path)


def test_job_scoped_oom_is_sealed_as_scientific_gate_failure(
    tmp_path: Path,
) -> None:
    study_id = "scientific-oom"
    gate_id = "oracle-rmsnorm-static-triton-a100"
    arguments = {
        "tasks": (get_task_pack("rmsnorm-static"),),
        "targets": (get_target_stack("triton-a100"),),
        "root": tmp_path,
        "study_id": study_id,
        "timing": TimingSpec(warmup_runs=1, trial_runs=2, repetitions=2),
    }
    worker = JobScopedOomWorker()

    records = run_oracle_gates(worker, **arguments)

    assert len(worker.jobs) == 1
    assert records[0].summary.status == "correctness_failure"
    assert records[0].summary.results[0].metadata["failure_category"] == "oom"
    verify_trajectory(tmp_path / study_id / gate_id)

    resume_worker = FakeWorker()
    assert run_oracle_gates(resume_worker, **arguments) == records
    assert resume_worker.jobs == []


def test_unsealed_staging_is_discarded_and_gate_is_reexecuted(
    tmp_path: Path,
) -> None:
    study_id = "crash-recovery"
    gate_id = "oracle-rmsnorm-static-triton-a100"
    staging = tmp_path / study_id / f"{gate_id}.incomplete"
    staging.mkdir(parents=True)
    (staging / "partial-write").write_text("controller crashed", encoding="utf-8")
    worker = FakeWorker()

    records = run_oracle_gates(
        worker,
        tasks=(get_task_pack("rmsnorm-static"),),
        targets=(get_target_stack("triton-a100"),),
        root=tmp_path,
        study_id=study_id,
        timing=TimingSpec(warmup_runs=1, trial_runs=2, repetitions=2),
    )

    final = tmp_path / study_id / gate_id
    assert len(worker.jobs) == 2
    assert not staging.exists()
    assert Path(records[0].artifact_directory) == final
    verify_trajectory(final)


def test_sealed_staging_is_promoted_without_worker_calls(tmp_path: Path) -> None:
    study_id = "sealed-staging"
    gate_id = "oracle-rmsnorm-static-triton-a100"
    arguments = {
        "tasks": (get_task_pack("rmsnorm-static"),),
        "targets": (get_target_stack("triton-a100"),),
        "root": tmp_path,
        "study_id": study_id,
        "timing": TimingSpec(warmup_runs=1, trial_runs=2, repetitions=2),
    }
    first_worker = FakeWorker()
    first = run_oracle_gates(first_worker, **arguments)
    final = tmp_path / study_id / gate_id
    staging = final.with_name(f"{gate_id}.incomplete")
    final.rename(staging)
    resume_worker = FakeWorker()

    resumed = run_oracle_gates(resume_worker, **arguments)

    assert resumed == first
    assert resume_worker.jobs == []
    assert final.is_dir()
    assert not staging.exists()
    verify_trajectory(final)


@pytest.mark.parametrize("artifact", ["final", "staging"])
def test_gate_rejects_final_or_staging_symlink(
    tmp_path: Path,
    artifact: str,
) -> None:
    study_id = "symlink-rejection"
    gate_id = "oracle-rmsnorm-static-triton-a100"
    study_root = tmp_path / study_id
    study_root.mkdir()
    destination = tmp_path / "outside"
    destination.mkdir()
    final = study_root / gate_id
    link = final if artifact == "final" else final.with_name(f"{gate_id}.incomplete")
    link.symlink_to(destination, target_is_directory=True)
    worker = FakeWorker()

    with pytest.raises(GateError, match="symbolic link"):
        run_oracle_gates(
            worker,
            tasks=(get_task_pack("rmsnorm-static"),),
            targets=(get_target_stack("triton-a100"),),
            root=tmp_path,
            study_id=study_id,
            timing=TimingSpec(warmup_runs=1, trial_runs=2, repetitions=2),
        )

    assert worker.jobs == []


def test_gate_study_rejects_artifacts_outside_the_frozen_matrix(
    tmp_path: Path,
) -> None:
    study_root = tmp_path / "unexpected-artifact"
    (study_root / "oracle-uncontracted-target").mkdir(parents=True)
    worker = FakeWorker()

    with pytest.raises(GateError, match="unexpected artifacts"):
        run_oracle_gates(
            worker,
            tasks=(get_task_pack("rmsnorm-static"),),
            targets=(get_target_stack("triton-a100"),),
            root=tmp_path,
            study_id=study_root.name,
            timing=TimingSpec(warmup_runs=1, trial_runs=2, repetitions=2),
        )

    assert worker.jobs == []
