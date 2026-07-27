from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from abstrak.canary.artifacts import TrajectoryStore, verify_trajectory
from abstrak.canary.contracts import CaseResult, WorkerJob, WorkerResult
from abstrak.canary.matrix import MatrixCell
from abstrak.canary.matrix_preflight import FORMAL_FLOOR_TIMING
from abstrak.canary.matrix_runner import MatrixTransportContext, MatrixWorkerBinding
from abstrak.canary.matrix_study import MatrixCellArtifactIdentity
from abstrak.canary.matrix_timing import (
    MatrixCandidateSourceIdentity,
    MatrixQualifiedCandidate,
    MatrixTimingError,
    MatrixTimingStudyCandidate,
    MatrixTimingStudyManifest,
    matrix_timing_artifact_id,
    run_matrix_candidate_timing,
    seal_matrix_timing_study_manifest,
)
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import get_task_pack
from abstrak.providers.contracts import sha256_json


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class FakeWorker:
    def __init__(self) -> None:
        self.jobs: list[WorkerJob] = []

    def execute(self, job: WorkerJob) -> WorkerResult:
        self.jobs.append(job)
        assert job.timing is not None
        return WorkerResult(
            job_id=job.job_id,
            job_sha256=job.sha256,
            input_sha256=job.input_sha256,
            candidate_sha256=job.candidate_sha256,
            status="completed",
            compiled=True,
            correct=True,
            cases=tuple(
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
            ),
            timing_ms=tuple(1.0 for _ in range(job.timing.trial_runs)),
            timing_cv=0.0,
        )


def _transport() -> MatrixTransportContext:
    return MatrixTransportContext(
        host="gpu.example",
        port=30554,
        worker_root="/workspace/AbstraK",
        python_executable="/workspace/venv/bin/python",
        pythonpath="/workspace/AbstraK/src",
        kernelbench_root="/workspace/KernelBench",
        asset_root="/workspace/AbstraK/benchmarks/capability-gate-a100",
        sandbox="setpriv-supervised",
        device="cuda:0",
        timeout_seconds=1200.0,
        network_isolated=False,
        filesystem_read_only=False,
    )


def _candidate(tmp_path: Path) -> MatrixQualifiedCandidate:
    task = get_task_pack("row-reduction-scale")
    target = get_target_stack("triton-a100")
    cell = MatrixCell(
        phase_id="core",
        ordinal=0,
        phase_ordinal=0,
        task_id=task.id,
        agent_id="fake-agent",
        target_id=target.id,
        replicate=1,
        target_order_index=0,
    )
    source = "class ModelNew:\n    pass\n"
    source_store = TrajectoryStore.create(
        tmp_path,
        "matrix-source-study",
        "source-attempt",
    )
    source_store.write_text("candidate.py", source)
    source_store.seal()
    source_artifact_sha256 = hashlib.sha256(
        (source_store.run_directory / "sha256sums.txt").read_bytes()
    ).hexdigest()
    raw_study_sha256 = _digest("raw-study")
    spec_sha256 = _digest("spec")
    schedule_sha256 = _digest("schedule")
    execution_context_sha256 = _digest("execution")
    attempt_identity = MatrixCellArtifactIdentity(
        study_id="matrix-timing-test",
        raw_study_sha256=raw_study_sha256,
        spec_sha256=spec_sha256,
        schedule_sha256=schedule_sha256,
        phase_id="core",
        trajectory_id=cell.trajectory_id,
        artifact_trajectory_id=cell.trajectory_id,
        attempt_index=0,
        cell=cell,
        task_sha256=sha256_json(task),
        target_sha256=sha256_json(target),
        agent_sha256=_digest("agent"),
        policy_sha256=_digest("policy"),
        budget_sha256=_digest("budget"),
        max_calls_per_trajectory=3,
        dev_timing_sha256=_digest("dev-timing"),
        model_ref="fake-model",
        initial_messages_sha256=_digest("messages"),
        device="cuda:0",
        execution_context_sha256=execution_context_sha256,
        execution_sha256=_digest("execution-spec"),
    )
    identity = MatrixCandidateSourceIdentity(
        study_id="matrix-timing-test",
        raw_study_sha256=raw_study_sha256,
        spec_sha256=spec_sha256,
        schedule_sha256=schedule_sha256,
        phase_id="core",
        phase_contract_sha256=_digest("phase-contract"),
        preflight_receipt_sha256=_digest("preflight"),
        execution_context_sha256=execution_context_sha256,
        attempt_identity=attempt_identity,
        source_artifact_sha256=source_artifact_sha256,
        candidate_labels=("first", "final"),
        candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    return MatrixQualifiedCandidate(
        identity=identity,
        source_artifact_directory=source_store.run_directory,
        source=source,
        task=task,
        target=target,
    )


def _manifest(candidate: MatrixQualifiedCandidate) -> MatrixTimingStudyManifest:
    source = candidate.identity
    return MatrixTimingStudyManifest(
        timing_study_id="matrix-timing-study",
        study_id=source.study_id,
        raw_study_sha256=source.raw_study_sha256,
        spec_sha256=source.spec_sha256,
        schedule_sha256=source.schedule_sha256,
        phase_id=source.phase_id,
        phase_contract_sha256=source.phase_contract_sha256,
        preflight_receipt_sha256=source.preflight_receipt_sha256,
        execution_context_sha256=source.execution_context_sha256,
        worker=MatrixWorkerBinding(
            worker_revision="a" * 40,
            transport=_transport(),
        ),
        device="cuda:0",
        timing=FORMAL_FLOOR_TIMING,
        candidate_count=1,
        candidates=(
            MatrixTimingStudyCandidate(
                source=source,
                artifact_id=matrix_timing_artifact_id(source),
            ),
        ),
    )


def test_matrix_candidate_timing_seals_manifest_and_resumes_without_worker_jobs(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    manifest = _manifest(candidate)
    worker = FakeWorker()
    worker.matrix_worker_binding = manifest.worker

    manifest_directory = seal_matrix_timing_study_manifest(tmp_path, manifest)
    first = run_matrix_candidate_timing(
        worker,
        artifact_root=tmp_path,
        manifest=manifest,
        candidates=(candidate,),
    )
    resumed = run_matrix_candidate_timing(
        worker,
        artifact_root=tmp_path,
        manifest=manifest,
        candidates=(candidate,),
    )

    assert first == resumed
    assert len(worker.jobs) == FORMAL_FLOOR_TIMING.repetitions
    assert all(job.timing.warmup_runs == 25 for job in worker.jobs if job.timing)
    assert all(job.timing.trial_runs == 200 for job in worker.jobs if job.timing)
    assert first[0].summary.stable
    assert first[0].timing_study_manifest_sha256 == manifest.sha256
    verify_trajectory(manifest_directory)
    verify_trajectory(first[0].artifact_directory)

    source_path = candidate.source_artifact_directory / "candidate.py"
    source_path.chmod(0o600)
    source_path.write_text(
        "tampered\n",
        encoding="utf-8",
    )
    fresh_worker = FakeWorker()
    fresh_worker.matrix_worker_binding = manifest.worker
    with pytest.raises(MatrixTimingError, match="source artifact is invalid"):
        run_matrix_candidate_timing(
            fresh_worker,
            artifact_root=tmp_path,
            manifest=manifest,
            candidates=(candidate,),
        )
    assert fresh_worker.jobs == []
