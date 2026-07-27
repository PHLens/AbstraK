from __future__ import annotations

import hashlib
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import pytest

from abstrak.canary.artifacts import TrajectoryStore, verify_trajectory
from abstrak.canary.baselines import BASELINE_VARIANTS, get_baseline_source
from abstrak.canary.capability_assets import (
    CAPABILITY_STUDY_SHA256,
    build_capability_asset_manifest,
)
from abstrak.canary.contracts import (
    CaseResult,
    TargetStackSpec,
    TaskPackSpec,
    WorkerJob,
    WorkerResult,
)
from abstrak.canary.gates import GateRecord
from abstrak.canary.manifests import PinnedStudySpec, load_study_spec
from abstrak.canary.matrix import MatrixSchedule, build_matrix_schedule
from abstrak.canary.matrix_capability_evidence import (
    CapabilityProbeInput,
    CapabilityProbeRecord,
    CapabilityProbeStudyManifest,
    build_capability_probe_study_manifest,
    capability_probe_artifact_sha256,
    resolve_capability_probe_inputs,
    run_capability_probe_study,
)
from abstrak.canary.matrix_floor_evidence import derive_task_floor_records
from abstrak.canary.matrix_launch_floor_evidence import (
    LaunchFloorProbeInfrastructureError,
    LaunchFloorProbeInput,
    LaunchFloorStudyManifest,
    MatrixLaunchFloorEvidenceError,
    build_launch_floor_study_manifest,
    derive_launch_floor_evidence,
    launch_probe_artifact_sha256,
    resolve_launch_floor_probe_input,
    run_launch_floor_probe,
)
from abstrak.canary.matrix_preflight import (
    FORMAL_FLOOR_TIMING,
    AssetManifest,
    EnvironmentManifest,
    EnvironmentObservation,
    EnvironmentProbeEvidence,
    TaskFloorRecord,
    build_pending_environment,
)
from abstrak.canary.matrix_runner import MatrixTransportContext
from abstrak.canary.remote import WorkerExecutionError
from abstrak.canary.target_adapters import validate_target_source
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import get_task_pack, load_oracle_source
from abstrak.canary.timing import TimingAttemptSummary, TimingProtocolSummary
from abstrak.providers.contracts import sha256_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_STUDY = (
    REPOSITORY_ROOT / "benchmarks" / "capability-gate-a100" / "study.json"
)
BASELINE_TARGET_ID = "tilelang-a100-full"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _transport() -> MatrixTransportContext:
    return MatrixTransportContext(
        host="root@gpu.example",
        port=30554,
        worker_root="/srv/AbstraK",
        python_executable="/srv/venv/bin/python",
        pythonpath="/srv/AbstraK/src",
        kernelbench_root="/srv/KernelBench",
        asset_root="/srv/AbstraK/benchmarks/capability-gate-a100",
        sandbox="setpriv-supervised",
        device="cuda:0",
        timeout_seconds=420.0,
        network_isolated=False,
        filesystem_read_only=False,
    )


def _verified_environment(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
) -> EnvironmentManifest:
    pending = build_pending_environment(
        pinned,
        schedule,
        controller_revision="1" * 40,
        worker_revision="2" * 40,
        transport=_transport(),
        accelerator="NVIDIA A100-SXM4-80GB",
        compute_capability="8.0",
        python_version="3.10.20",
        tilelang_version="0.1.12",
        triton_version="3.7.1",
        torch_version="2.13.0+cu126",
        cuda_version="12.6",
        driver_version="570.00",
        kernelbench_revision="3" * 40,
    )
    payload = pending.model_dump()
    payload["status"] = "verified"
    payload["verification_evidence"] = EnvironmentProbeEvidence(
        artifact_sha256=_digest("environment-probe"),
        status="pass",
        observation=EnvironmentObservation(
            worker_revision=pending.worker_revision,
            transport=pending.transport,
            accelerator=pending.accelerator,
            compute_capability=pending.compute_capability,
            python_version=pending.python_version,
            tilelang_version=pending.tilelang_version,
            triton_version=pending.triton_version,
            torch_version=pending.torch_version,
            cuda_version=pending.cuda_version,
            driver_version=pending.driver_version,
            kernelbench_revision=pending.kernelbench_revision,
        ),
    )
    return EnvironmentManifest.model_validate(payload)


def _completed_summary(
    *,
    task: TaskPackSpec,
    target: TargetStackSpec,
    kind: str,
    source: str,
    latency: float,
    generated_code_sha256: str | None = None,
    variant: str | None = None,
) -> TimingProtocolSummary:
    candidate_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    process_timing = FORMAL_FLOOR_TIMING.model_copy(update={"repetitions": 1})
    case_ids = tuple(case.id for case in task.sealed_cases)
    jobs: list[WorkerJob] = []
    results: list[WorkerResult] = []
    suffix = "oracle" if variant is None else variant
    for repetition in range(1, FORMAL_FLOOR_TIMING.repetitions + 1):
        job = WorkerJob(
            job_id=f"{kind}-{task.id}-{target.id}-{suffix}-p{repetition}",
            kind=kind,
            task=task,
            target=target,
            case_ids=case_ids,
            candidate_source=source,
            candidate_sha256=candidate_sha256,
            timing=process_timing,
            device="cuda:0",
        )
        metadata = {}
        if generated_code_sha256 is not None:
            metadata = {
                "generated_code_capture": "tilelang.get_kernel_source.v1",
                "generated_code_sha256": generated_code_sha256,
                "generated_code_size_bytes": 1024,
            }
        result = WorkerResult(
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
                for case_id in case_ids
            ),
            timing_ms=tuple(latency for _ in range(process_timing.trial_runs)),
            timing_cv=0.0,
            metadata=metadata,
        ).verify_for_job(job)
        jobs.append(job)
        results.append(result)
    attempt = TimingAttemptSummary(
        attempt=1,
        status="stable",
        stable=True,
        jobs=tuple(jobs),
        results=tuple(results),
        process_medians_ms=tuple(latency for _ in jobs),
        process_cvs=tuple(0.0 for _ in jobs),
        across_process_cv=0.0,
        median_ms=latency,
    )
    return TimingProtocolSummary(
        job_prefix=f"{kind}-{task.id}-{target.id}-{suffix}",
        task_id=task.id,
        target_id=target.id,
        candidate_sha256=candidate_sha256,
        job_kind=kind,
        device="cuda:0",
        timing=FORMAL_FLOOR_TIMING,
        status="stable",
        stable=True,
        attempts=(attempt,),
        jobs=tuple(jobs),
        results=tuple(results),
        median_ms=latency,
    )


def _seal_gate(
    root: Path,
    *,
    task: TaskPackSpec,
    target: TargetStackSpec,
    kind: str,
    source: str,
    latency: float,
    generated_code_sha256: str | None = None,
    variant: str | None = None,
) -> GateRecord:
    suffix = "oracle" if variant is None else variant
    gate_id = f"{kind}-{task.id}-{target.id}-{suffix}"
    store = TrajectoryStore.create(root, "raw-gates", gate_id)
    record = GateRecord(
        kind=kind,
        task_id=task.id,
        target_id=target.id,
        variant=variant,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        artifact_directory=str(store.run_directory),
        summary=_completed_summary(
            task=task,
            target=target,
            kind=kind,
            source=source,
            latency=latency,
            generated_code_sha256=generated_code_sha256,
            variant=variant,
        ),
    )
    store.write_json("gate-record.json", record)
    store.seal()
    return record


def _raw_gates(
    root: Path,
    assets: AssetManifest,
) -> tuple[GateRecord, ...]:
    targets = tuple(get_target_stack(item.target_id) for item in assets.targets)
    baseline_target = get_target_stack(BASELINE_TARGET_ID)
    records: list[GateRecord] = []
    for task_index, task_asset in enumerate(assets.tasks):
        task = get_task_pack(task_asset.task_id)
        expert = load_oracle_source(task.id, "tilelang")
        generated = _digest(f"{task.id}:shared-expert-codegen")
        for target in targets:
            core_latency = 0.8 if task_index % 2 == 0 else 1.2
            records.append(
                _seal_gate(
                    root,
                    task=task,
                    target=target,
                    kind="oracle",
                    source=expert,
                    latency=(
                        core_latency
                        if target.id == "tilelang-a100-core"
                        else 1.1
                    ),
                    generated_code_sha256=generated,
                )
            )
        for variant, latency in zip(
            BASELINE_VARIANTS,
            (1.0, 1.5, 2.0),
            strict=True,
        ):
            baseline = get_baseline_source(task.id, variant)
            records.append(
                _seal_gate(
                    root,
                    task=task,
                    target=baseline_target,
                    kind="baseline",
                    source=baseline.source,
                    latency=latency,
                    variant=variant,
                )
            )
    return tuple(records)


class _CapabilityWorker:
    def __init__(self, manifest: CapabilityProbeStudyManifest) -> None:
        self.matrix_worker_binding = manifest.worker
        self.identities = {item.artifact_id: item for item in manifest.probes}

    def execute(self, job: WorkerJob) -> WorkerResult:
        artifact_id = job.job_id.partition("-timing-a")[0]
        identity = self.identities[artifact_id]
        static = validate_target_source(job.candidate_source, job.target)
        assert static.valid
        metadata = dict(static.metadata)
        generated = _digest(f"{job.candidate_sha256}:capability-codegen")
        assert generated != identity.control.generated_code_sha256
        metadata.update(
            {
                "generated_code_capture": "tilelang.get_kernel_source.v1",
                "generated_code_sha256": generated,
                "generated_code_size_bytes": 1024,
            }
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
            timing_ms=tuple(2.0 for _ in range(job.timing.trial_runs)),
            timing_cv=0.0,
            static_warnings=tuple(
                f"{item.code}: {item.message}" for item in static.warnings
            ),
            metadata=metadata,
        ).verify_for_job(job)


class _LaunchWorker:
    def __init__(
        self,
        manifest: LaunchFloorStudyManifest,
        *,
        mode: str = "stable",
    ) -> None:
        self.matrix_worker_binding = manifest.worker
        self.mode = mode
        self.calls: list[WorkerJob] = []

    def execute(self, job: WorkerJob) -> WorkerResult:
        self.calls.append(job)
        if self.mode == "infrastructure":
            raise RuntimeError("SSH connection reset")
        if self.mode == "job-timeout":
            raise WorkerExecutionError(
                "timeout",
                "deterministic launch timeout",
                health={"status": "healthy"},
                job_scoped=True,
            )
        assert job.timing is not None
        if self.mode == "unstable":
            samples = tuple(
                0.025 if index % 2 == 0 else 0.075
                for index in range(job.timing.trial_runs)
            )
            timing_cv = statistics.pstdev(samples) / statistics.fmean(samples)
        else:
            samples = tuple(0.05 for _ in range(job.timing.trial_runs))
            timing_cv = 0.0
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
            timing_ms=samples,
            timing_cv=timing_cv,
            metadata={
                "generated_code_capture": "tilelang.get_kernel_source.v1",
                "generated_code_sha256": _digest("launch-probe-codegen"),
                "generated_code_size_bytes": 256,
            },
        ).verify_for_job(job)


@dataclass(frozen=True)
class LaunchFixture:
    pinned: PinnedStudySpec
    schedule: MatrixSchedule
    assets: AssetManifest
    environment: EnvironmentManifest
    gates: tuple[GateRecord, ...]
    floors: tuple[TaskFloorRecord, ...]
    capability_inputs: tuple[CapabilityProbeInput, ...]
    capability_manifest: CapabilityProbeStudyManifest
    capability_records: tuple[CapabilityProbeRecord, ...]
    probe_input: LaunchFloorProbeInput
    manifest: LaunchFloorStudyManifest


@pytest.fixture(scope="module")
def launch_fixture(tmp_path_factory: pytest.TempPathFactory) -> LaunchFixture:
    root = tmp_path_factory.mktemp("launch-floor-upstream")
    pinned = load_study_spec(
        CAPABILITY_STUDY,
        expected_sha256=CAPABILITY_STUDY_SHA256,
    )
    schedule = build_matrix_schedule(pinned.spec)
    assets = build_capability_asset_manifest(pinned, schedule)
    environment = _verified_environment(pinned, schedule)
    gates = _raw_gates(root / "gates", assets)
    assert pinned.spec.gate is not None
    floors = derive_task_floor_records(
        gates,
        assets,
        tuple(get_target_stack(item.target_id) for item in assets.targets),
        baseline_target_id=BASELINE_TARGET_ID,
        competitive_factor=pinned.spec.gate.metrics.competitive_latency_factor,
    )
    capability_inputs = resolve_capability_probe_inputs(
        assets,
        environment,
        floors,
    )
    capability_manifest = build_capability_probe_study_manifest(
        assets,
        environment,
        capability_inputs,
    )
    capability_records = run_capability_probe_study(
        _CapabilityWorker(capability_manifest),
        artifact_root=root / "capabilities",
        manifest=capability_manifest,
        inputs=capability_inputs,
    )
    probe_input = resolve_launch_floor_probe_input(assets, environment)
    manifest = build_launch_floor_study_manifest(
        pinned,
        schedule,
        assets,
        environment,
        floors,
        gates,
        capability_manifest,
        capability_records,
        probe_input,
        baseline_target_id=BASELINE_TARGET_ID,
    )
    return LaunchFixture(
        pinned=pinned,
        schedule=schedule,
        assets=assets,
        environment=environment,
        gates=gates,
        floors=floors,
        capability_inputs=capability_inputs,
        capability_manifest=capability_manifest,
        capability_records=capability_records,
        probe_input=probe_input,
        manifest=manifest,
    )


def test_recomputes_all_raw_denominators_and_emits_complete_measurements(
    tmp_path: Path,
    launch_fixture: LaunchFixture,
) -> None:
    worker = _LaunchWorker(launch_fixture.manifest)
    record = run_launch_floor_probe(
        worker,
        artifact_root=tmp_path,
        manifest=launch_fixture.manifest,
        probe_input=launch_fixture.probe_input,
    )
    evidence = derive_launch_floor_evidence(
        launch_fixture.manifest,
        launch_fixture.probe_input,
        record,
    )

    assert evidence.status == "pass"
    assert len(worker.calls) == FORMAL_FLOOR_TIMING.repetitions
    assert len(evidence.measurements) == len(launch_fixture.assets.tasks) + len(
        launch_fixture.assets.canaries
    )
    task_measurements = tuple(
        item for item in evidence.measurements if item.workload_kind == "task"
    )
    assert tuple(item.workload_id for item in task_measurements) == tuple(
        item.task_id for item in launch_fixture.assets.tasks
    )
    assert tuple(item.workload_timing_kind for item in task_measurements) == tuple(
        "expert" if index % 2 == 0 else "baseline"
        for index in range(len(launch_fixture.assets.tasks))
    )
    assert tuple(item.task_ms for item in task_measurements) == tuple(
        0.8 if index % 2 == 0 else 1.0
        for index in range(len(launch_fixture.assets.tasks))
    )
    assert tuple(item.target_id for item in task_measurements) == tuple(
        "tilelang-a100-core" if index % 2 == 0 else BASELINE_TARGET_ID
        for index in range(len(launch_fixture.assets.tasks))
    )

    canary_measurements = tuple(
        item for item in evidence.measurements if item.workload_kind == "canary"
    )
    assert tuple(item.workload_id for item in canary_measurements) == (
        "schedule",
        "mapping",
        "schedule-mapping",
    )
    assert tuple(item.target_id for item in canary_measurements) == (
        "tilelang-a100-sched",
        "tilelang-a100-map",
        "tilelang-a100-full",
    )
    assert all(item.task_ms == 2.0 for item in canary_measurements)
    minimum_records = {
        (item.identity.canary_id, item.identity.target_id): item
        for item in launch_fixture.capability_records
    }
    for measurement in canary_measurements:
        expected = minimum_records[
            (measurement.workload_id, measurement.target_id)
        ]
        assert (
            measurement.workload_artifact_sha256
            == capability_probe_artifact_sha256(expected)
        )
        assert (
            measurement.workload_timing_summary_sha256
            == sha256_json(expected.summary)
        )

    assert all(item.launch_ms == 0.05 for item in evidence.measurements)
    assert all(
        item.launch_fraction == item.launch_ms / item.task_ms
        for item in evidence.measurements
    )
    assert all(
        item.launch_source_sha256 == launch_fixture.manifest.probe.source_sha256
        for item in evidence.measurements
    )
    assert all(
        item.launch_artifact_sha256 == launch_probe_artifact_sha256(record)
        for item in evidence.measurements
    )


def test_exact_resume_never_reexecutes_the_launch_probe(
    tmp_path: Path,
    launch_fixture: LaunchFixture,
) -> None:
    worker = _LaunchWorker(launch_fixture.manifest)
    first = run_launch_floor_probe(
        worker,
        artifact_root=tmp_path,
        manifest=launch_fixture.manifest,
        probe_input=launch_fixture.probe_input,
    )
    resumed = run_launch_floor_probe(
        worker,
        artifact_root=tmp_path,
        manifest=launch_fixture.manifest,
        probe_input=launch_fixture.probe_input,
    )

    assert resumed == first
    assert len(worker.calls) == FORMAL_FLOOR_TIMING.repetitions
    verify_trajectory(first.artifact_directory)
    assert not Path(f"{first.artifact_directory}.incomplete").exists()


def test_worker_failure_is_infrastructure_only_and_leaves_no_probe_artifact(
    tmp_path: Path,
    launch_fixture: LaunchFixture,
) -> None:
    worker = _LaunchWorker(launch_fixture.manifest, mode="infrastructure")

    with pytest.raises(
        LaunchFloorProbeInfrastructureError,
        match="SSH connection reset",
    ):
        run_launch_floor_probe(
            worker,
            artifact_root=tmp_path,
            manifest=launch_fixture.manifest,
            probe_input=launch_fixture.probe_input,
        )

    study_root = tmp_path / launch_fixture.manifest.launch_study_id
    artifact_id = launch_fixture.manifest.probe.artifact_id
    assert (study_root / "study-manifest").is_dir()
    assert not (study_root / artifact_id).exists()
    assert not (study_root / f"{artifact_id}.incomplete").exists()


def test_job_scoped_timeout_is_sealed_as_scientific_failed_evidence(
    tmp_path: Path,
    launch_fixture: LaunchFixture,
) -> None:
    worker = _LaunchWorker(launch_fixture.manifest, mode="job-timeout")

    record = run_launch_floor_probe(
        worker,
        artifact_root=tmp_path,
        manifest=launch_fixture.manifest,
        probe_input=launch_fixture.probe_input,
    )
    evidence = derive_launch_floor_evidence(
        launch_fixture.manifest,
        launch_fixture.probe_input,
        record,
    )

    assert record.summary.status == "correctness_failure"
    assert record.summary.results[0].status == "timeout"
    assert evidence.status == "fail"
    assert evidence.failure_reason
    verify_trajectory(record.artifact_directory)


def test_unstable_probe_is_sealed_as_terminal_failed_evidence(
    tmp_path: Path,
    launch_fixture: LaunchFixture,
) -> None:
    worker = _LaunchWorker(launch_fixture.manifest, mode="unstable")
    record = run_launch_floor_probe(
        worker,
        artifact_root=tmp_path,
        manifest=launch_fixture.manifest,
        probe_input=launch_fixture.probe_input,
    )
    evidence = derive_launch_floor_evidence(
        launch_fixture.manifest,
        launch_fixture.probe_input,
        record,
    )

    assert record.summary.status == "unstable"
    assert len(worker.calls) == 2 * FORMAL_FLOOR_TIMING.repetitions
    assert evidence.status == "fail"
    assert evidence.measurements == ()
    assert evidence.failure_reason == "launch probe timing is unstable"
    verify_trajectory(record.artifact_directory)


def _add_checksum_valid_extra_file(directory: Path) -> None:
    directory.chmod(0o700)
    checksum = directory / "sha256sums.txt"
    checksum.chmod(0o600)
    extra = directory / "extra.txt"
    extra.write_text("unexpected but checksum-bound\n", encoding="utf-8")
    files = sorted(
        item
        for item in directory.rglob("*")
        if item.is_file() and item != checksum
    )
    checksum.write_text(
        "".join(
            f"{hashlib.sha256(item.read_bytes()).hexdigest()}  "
            f"{item.relative_to(directory).as_posix()}\n"
            for item in files
        ),
        encoding="utf-8",
    )
    verify_trajectory(directory)


@pytest.mark.parametrize("attack", ["checksum", "extra-file"])
def test_tampered_or_shape_drifted_final_artifact_fails_closed(
    tmp_path: Path,
    launch_fixture: LaunchFixture,
    attack: str,
) -> None:
    record = run_launch_floor_probe(
        _LaunchWorker(launch_fixture.manifest),
        artifact_root=tmp_path,
        manifest=launch_fixture.manifest,
        probe_input=launch_fixture.probe_input,
    )
    directory = Path(record.artifact_directory)
    if attack == "checksum":
        checksum = directory / "sha256sums.txt"
        checksum.chmod(0o600)
        checksum.write_bytes(checksum.read_bytes() + b"tampered\n")
    else:
        _add_checksum_valid_extra_file(directory)

    with pytest.raises(MatrixLaunchFloorEvidenceError):
        derive_launch_floor_evidence(
            launch_fixture.manifest,
            launch_fixture.probe_input,
            record,
        )


def test_checksum_valid_extra_file_in_sealed_staging_fails_closed(
    tmp_path: Path,
    launch_fixture: LaunchFixture,
) -> None:
    worker = _LaunchWorker(launch_fixture.manifest)
    record = run_launch_floor_probe(
        worker,
        artifact_root=tmp_path,
        manifest=launch_fixture.manifest,
        probe_input=launch_fixture.probe_input,
    )
    final = Path(record.artifact_directory)
    staging = final.with_name(f"{final.name}.incomplete")
    os.replace(final, staging)
    _add_checksum_valid_extra_file(staging)

    with pytest.raises(
        MatrixLaunchFloorEvidenceError,
        match="cannot run or resume launch probe",
    ):
        run_launch_floor_probe(
            worker,
            artifact_root=tmp_path,
            manifest=launch_fixture.manifest,
            probe_input=launch_fixture.probe_input,
        )

    assert not final.exists()
    assert staging.is_dir()
    assert len(worker.calls) == FORMAL_FLOOR_TIMING.repetitions


def test_checksum_corrupt_launch_staging_fails_closed(
    tmp_path: Path,
    launch_fixture: LaunchFixture,
) -> None:
    worker = _LaunchWorker(launch_fixture.manifest)
    record = run_launch_floor_probe(
        worker,
        artifact_root=tmp_path,
        manifest=launch_fixture.manifest,
        probe_input=launch_fixture.probe_input,
    )
    final = Path(record.artifact_directory)
    staging = final.with_name(f"{final.name}.incomplete")
    os.replace(final, staging)
    checksum = staging / "sha256sums.txt"
    checksum.chmod(0o600)
    checksum.write_bytes(checksum.read_bytes() + b"tampered\n")
    completed_calls = len(worker.calls)

    with pytest.raises(
        MatrixLaunchFloorEvidenceError,
        match="cannot run or resume launch probe",
    ):
        run_launch_floor_probe(
            worker,
            artifact_root=tmp_path,
            manifest=launch_fixture.manifest,
            probe_input=launch_fixture.probe_input,
        )

    assert len(worker.calls) == completed_calls
    assert not final.exists()
    assert staging.is_dir()


def test_build_manifest_recomputes_and_rejects_baseline_raw_drift(
    tmp_path: Path,
    launch_fixture: LaunchFixture,
) -> None:
    original_index = next(
        index
        for index, item in enumerate(launch_fixture.gates)
        if item.kind == "baseline"
        and item.task_id == launch_fixture.assets.tasks[0].task_id
        and item.variant == "compile"
    )
    original = launch_fixture.gates[original_index]
    forged_summary = original.summary.model_copy(update={"median_ms": 0.7})
    forged_record = original.model_copy(update={"summary": forged_summary})
    forged_gates = list(launch_fixture.gates)
    forged_gates[original_index] = forged_record
    with pytest.raises(
        MatrixLaunchFloorEvidenceError,
        match="cannot recompute task floors",
    ):
        build_launch_floor_study_manifest(
            launch_fixture.pinned,
            launch_fixture.schedule,
            launch_fixture.assets,
            launch_fixture.environment,
            launch_fixture.floors,
            forged_gates,
            launch_fixture.capability_manifest,
            launch_fixture.capability_records,
            launch_fixture.probe_input,
            baseline_target_id=BASELINE_TARGET_ID,
        )

    job = original.summary.jobs[0]
    resealed = _seal_gate(
        tmp_path,
        task=job.task,
        target=job.target,
        kind="baseline",
        source=job.candidate_source,
        latency=0.7,
        variant="compile",
    )
    drifted_gates = list(launch_fixture.gates)
    drifted_gates[original_index] = resealed
    with pytest.raises(
        MatrixLaunchFloorEvidenceError,
        match="task floors differ from sealed gate recomputation",
    ):
        build_launch_floor_study_manifest(
            launch_fixture.pinned,
            launch_fixture.schedule,
            launch_fixture.assets,
            launch_fixture.environment,
            launch_fixture.floors,
            drifted_gates,
            launch_fixture.capability_manifest,
            launch_fixture.capability_records,
            launch_fixture.probe_input,
            baseline_target_id=BASELINE_TARGET_ID,
        )
