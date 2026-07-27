from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import abstrak.canary.matrix_preflight_runner as preflight_runner
from abstrak.canary.artifacts import verify_trajectory
from abstrak.canary.capability_assets import (
    CAPABILITY_STUDY_SHA256,
    build_capability_asset_manifest,
)
from abstrak.canary.contracts import CaseResult, WorkerJob, WorkerResult
from abstrak.canary.manifests import PinnedStudySpec, load_study_spec
from abstrak.canary.matrix import MatrixSchedule, build_matrix_schedule
from abstrak.canary.matrix_preflight import (
    AssetManifest,
    EnvironmentManifest,
    build_pending_environment,
)
from abstrak.canary.matrix_preflight_runner import (
    MatrixPreflightArtifactError,
    MatrixPreflightInfrastructureError,
    MatrixPreflightInvalidFloorError,
    MatrixPreflightRunnerError,
    build_preflight_study_contract,
    run_matrix_preflight,
    run_or_resume_environment_probe,
)
from abstrak.canary.matrix_runner import MatrixTransportContext, MatrixWorkerBinding
from abstrak.canary.remote import WorkerExecutionError
from abstrak.canary.target_adapters import validate_target_source

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_ROOT = REPOSITORY_ROOT / "benchmarks" / "capability-gate-a100"
CAPABILITY_STUDY = CAPABILITY_ROOT / "study.json"
BASELINE_TARGET_ID = "tilelang-a100-core"
EXPECTED_PROTOCOLS = 62
EXPECTED_JOB_CEILING = 372
EXPECTED_STABLE_JOBS = 186


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


@dataclass(frozen=True)
class FrozenStudy:
    pinned: PinnedStudySpec
    schedule: MatrixSchedule
    assets: AssetManifest
    pending: EnvironmentManifest


@pytest.fixture(scope="module")
def frozen_study() -> FrozenStudy:
    pinned = load_study_spec(
        CAPABILITY_STUDY,
        expected_sha256=CAPABILITY_STUDY_SHA256,
    )
    schedule = build_matrix_schedule(pinned.spec)
    assets = build_capability_asset_manifest(pinned, schedule)
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
    return FrozenStudy(
        pinned=pinned,
        schedule=schedule,
        assets=assets,
        pending=pending,
    )


def _contract_directory(root: Path, study: FrozenStudy) -> Path:
    return root / study.pinned.spec.study_id / "preflight-contract"


def _ready_directory(root: Path, study: FrozenStudy) -> Path:
    return root / study.pinned.spec.study_id / "matrix-preflight"


def _environment_directory(root: Path, study: FrozenStudy) -> Path:
    return root / study.pinned.spec.study_id / "environment-probe"


class CombinedFakeWorker:
    def __init__(
        self,
        environment: EnvironmentManifest,
        *,
        contract_directory: Path | None = None,
        health_updates: dict[str, object] | None = None,
        transport_outage: bool = False,
        unhealthy_gpu: bool = False,
    ) -> None:
        major, minor = (int(value) for value in environment.compute_capability.split("."))
        self.matrix_worker_binding = MatrixWorkerBinding(
            worker_revision=environment.worker_revision,
            transport=environment.transport,
        )
        self.expected_hardware_substring = environment.accelerator
        self.expected_compute_capability = (major, minor)
        self.expected_python_version = environment.python_version
        self.expected_torch_version = environment.torch_version
        self.expected_torch_cuda_version = environment.cuda_version
        self.expected_triton_version = environment.triton_version
        self.expected_tilelang_version = environment.tilelang_version
        self.expected_driver_version = environment.driver_version
        self.expected_non_container_worker = environment.non_container_worker
        self.expected_kernelbench_revision = environment.kernelbench_revision
        self.environment = environment
        self.contract_directory = contract_directory
        self.health_updates = health_updates or {}
        self.transport_outage = transport_outage
        self.unhealthy_gpu = unhealthy_gpu
        self.health_calls: list[str] = []
        self.jobs: list[WorkerJob] = []

    def _assert_contract_is_sealed(self) -> None:
        if self.contract_directory is not None:
            verify_trajectory(self.contract_directory)

    def validate_environment(self, device: str) -> dict[str, object]:
        self._assert_contract_is_sealed()
        self.health_calls.append(device)
        if self.transport_outage:
            raise WorkerExecutionError(
                "health_check_failed",
                "SSH environment probe timed out",
                health={
                    "schema_version": "canary-worker-health.v1",
                    "status": "check_failed",
                    "device": device,
                    "error": "SSH environment probe timed out",
                },
            )
        if self.unhealthy_gpu:
            raise WorkerExecutionError(
                "health_unhealthy",
                "CUDA context failed",
                health={
                    "schema_version": "canary-worker-health.v1",
                    "status": "unhealthy",
                    "device": device,
                    "error": "CUDA context failed",
                },
            )
        health: dict[str, object] = {
            "schema_version": "canary-worker-health.v1",
            "status": "healthy",
            "device": device,
            "hardware": self.environment.accelerator,
            "compute_capability": list(self.expected_compute_capability),
            "python_version": self.environment.python_version,
            "torch_version": self.environment.torch_version,
            "torch_cuda_version": self.environment.cuda_version,
            "triton_version": self.environment.triton_version,
            "tilelang_version": self.environment.tilelang_version,
            "driver_version": self.environment.driver_version,
            "container_markers": [],
            "non_container_worker": True,
            "worker_revision": self.environment.worker_revision,
            "kernelbench_revision": self.environment.kernelbench_revision,
            "value": 2.0,
        }
        health.update(self.health_updates)
        return health

    def execute(self, job: WorkerJob) -> WorkerResult:
        self._assert_contract_is_sealed()
        self.jobs.append(job)
        assert job.timing is not None

        metadata: dict[str, Any] = {}
        warnings: tuple[str, ...] = ()
        if job.kind != "baseline":
            static = validate_target_source(job.candidate_source, job.target)
            assert static.valid, static.errors
            metadata.update(static.metadata)
            used_capabilities = metadata.get("used_capabilities")
            if isinstance(used_capabilities, tuple):
                metadata["used_capabilities"] = list(used_capabilities)
            warnings = tuple(f"{item.code}: {item.message}" for item in static.warnings)
            metadata.update(
                {
                    "generated_code_capture": "tilelang.get_kernel_source.v1",
                    "generated_code_sha256": _digest(f"generated:{job.candidate_sha256}"),
                    "generated_code_size_bytes": 1024,
                }
            )

        latency = 0.05 if job.kind == "sealed" else 1.0
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
            static_warnings=warnings,
            metadata=metadata,
        ).verify_for_job(job)


def _run(
    study: FrozenStudy,
    root: Path,
    factory,
    *,
    live: bool = True,
    ceiling: int = EXPECTED_JOB_CEILING,
    baseline_target_id: str = BASELINE_TARGET_ID,
):
    return run_matrix_preflight(
        study.pinned,
        study.schedule,
        study.assets,
        study.pending,
        artifact_root=root,
        asset_root=CAPABILITY_ROOT,
        baseline_target_id=baseline_target_id,
        live=live,
        expected_max_worker_jobs_per_invocation=ceiling,
        worker_factory=factory,
    )


def test_contract_freezes_exact_protocol_closure(
    frozen_study: FrozenStudy,
) -> None:
    contract = build_preflight_study_contract(
        frozen_study.pinned,
        frozen_study.schedule,
        frozen_study.assets,
        frozen_study.pending,
        baseline_target_id=BASELINE_TARGET_ID,
    )

    assert contract.protocol_count == EXPECTED_PROTOCOLS
    assert contract.max_worker_jobs_per_invocation == EXPECTED_JOB_CEILING
    assert Counter(item.kind for item in contract.protocols) == {
        "oracle": 32,
        "baseline": 24,
        "capability": 5,
        "launch": 1,
    }
    assert len({item.artifact_id for item in contract.protocols}) == EXPECTED_PROTOCOLS


def test_runtime_worker_cap_refuses_373rd_job_before_delegation_and_excludes_health() -> None:
    class RawWorker:
        def __init__(self) -> None:
            self.health_calls: list[str] = []
            self.execute_calls: list[object] = []

        def validate_environment(self, device: str) -> dict[str, object]:
            self.health_calls.append(device)
            return {"status": "healthy"}

        def execute(self, job: object) -> object:
            self.execute_calls.append(job)
            return job

    raw = RawWorker()
    worker = preflight_runner._CappedPreflightWorker(
        raw,  # type: ignore[arg-type]
        max_worker_jobs_per_invocation=EXPECTED_JOB_CEILING,
    )

    assert worker.validate_environment("cuda:0") == {"status": "healthy"}
    assert worker.worker_job_count == 0
    for job_number in range(1, EXPECTED_JOB_CEILING + 1):
        assert worker.execute(job_number) == job_number  # type: ignore[arg-type]
    assert worker.worker_job_count == EXPECTED_JOB_CEILING
    assert len(raw.execute_calls) == EXPECTED_JOB_CEILING

    assert worker.validate_environment("cuda:0") == {"status": "healthy"}
    with pytest.raises(MatrixPreflightRunnerError, match="hard cap reached"):
        worker.execute(EXPECTED_JOB_CEILING + 1)  # type: ignore[arg-type]

    assert raw.health_calls == ["cuda:0", "cuda:0"]
    assert worker.worker_job_count == EXPECTED_JOB_CEILING
    assert len(raw.execute_calls) == EXPECTED_JOB_CEILING
    assert EXPECTED_JOB_CEILING + 1 not in raw.execute_calls


def test_run_preflight_passes_capped_worker_to_producers(
    tmp_path: Path,
    frozen_study: FrozenStudy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers: list[CombinedFakeWorker] = []
    delegated_jobs: list[object] = []

    class RecordingWorker(CombinedFakeWorker):
        def execute(self, job: object) -> object:
            delegated_jobs.append(job)
            return job

    def factory(environment: EnvironmentManifest) -> CombinedFakeWorker:
        worker = RecordingWorker(environment)
        workers.append(worker)
        return worker

    def overrun_oracle_producer(worker: Any, **kwargs: Any) -> tuple[()]:
        for job_number in range(1, EXPECTED_JOB_CEILING + 2):
            worker.execute(job_number)
        raise AssertionError("the producer exceeded the frozen runtime cap")

    monkeypatch.setattr(
        preflight_runner,
        "run_oracle_gates",
        overrun_oracle_producer,
    )

    with pytest.raises(MatrixPreflightRunnerError, match="hard cap reached"):
        _run(frozen_study, tmp_path, factory)

    assert len(workers) == 1
    assert workers[0].health_calls == ["cuda:0"]
    assert delegated_jobs == list(range(1, EXPECTED_JOB_CEILING + 1))
    assert EXPECTED_JOB_CEILING + 1 not in delegated_jobs


@pytest.mark.parametrize(
    ("live", "ceiling", "message"),
    [
        (False, EXPECTED_JOB_CEILING, "live authorization"),
        (True, EXPECTED_JOB_CEILING - 1, "worker-job ceiling"),
    ],
)
def test_guards_fail_before_worker_factory(
    tmp_path: Path,
    frozen_study: FrozenStudy,
    live: bool,
    ceiling: int,
    message: str,
) -> None:
    factory_calls: list[EnvironmentManifest] = []

    def factory(environment: EnvironmentManifest) -> CombinedFakeWorker:
        factory_calls.append(environment)
        raise AssertionError("worker factory must not be called")

    with pytest.raises(MatrixPreflightRunnerError, match=message):
        _run(
            frozen_study,
            tmp_path,
            factory,
            live=live,
            ceiling=ceiling,
        )

    assert factory_calls == []


def test_ready_path_seals_contract_before_remote_and_resumes_without_remote_calls(
    tmp_path: Path,
    frozen_study: FrozenStudy,
) -> None:
    contract_directory = _contract_directory(tmp_path, frozen_study)
    factory_calls: list[EnvironmentManifest] = []
    workers: list[CombinedFakeWorker] = []

    def factory(environment: EnvironmentManifest) -> CombinedFakeWorker:
        factory_calls.append(environment)
        verify_trajectory(contract_directory)
        worker = CombinedFakeWorker(
            environment,
            contract_directory=contract_directory,
        )
        workers.append(worker)
        return worker

    result = _run(frozen_study, tmp_path, factory)

    assert result.resumed_ready_bundle is False
    assert result.bundle.receipt.status == "ready"
    assert result.bundle.environment.status == "verified"
    assert result.bundle.floor.status == "valid"
    assert len(result.oracle_gates) == 32
    assert len(result.baseline_gates) == 24
    assert len(result.capability_records) == 5
    assert result.launch_record is not None
    assert len(factory_calls) == len(workers) == 1
    assert workers[0].health_calls == ["cuda:0"]
    assert len(workers[0].jobs) == EXPECTED_STABLE_JOBS
    assert all("-timing-a1-p" in job.job_id for job in workers[0].jobs)
    assert not any("-timing-a2-" in job.job_id for job in workers[0].jobs)
    protocols = {protocol.artifact_id: protocol for protocol in result.contract.protocols}
    jobs_per_protocol: Counter[str] = Counter()
    expected_process_timing = result.contract.timing.model_copy(update={"repetitions": 1})
    expected_job_kinds = {
        "oracle": "oracle",
        "baseline": "baseline",
        "capability": "oracle",
        "launch": "sealed",
    }
    for job in workers[0].jobs:
        artifact_id, separator, _ = job.job_id.partition("-timing-")
        assert separator
        protocol = protocols[artifact_id]
        jobs_per_protocol[artifact_id] += 1
        assert job.task.id == protocol.task_id
        assert job.target.id == protocol.target_id
        assert job.candidate_sha256 == protocol.source_sha256
        assert job.kind == expected_job_kinds[protocol.kind]
        assert job.device == result.contract.pending_environment.transport.device
        assert job.timing == expected_process_timing
    assert jobs_per_protocol == Counter({artifact_id: 3 for artifact_id in protocols})
    verify_trajectory(contract_directory)
    verify_trajectory(_ready_directory(tmp_path, frozen_study))

    resumed_factory_calls: list[EnvironmentManifest] = []

    def resumed_factory(environment: EnvironmentManifest) -> CombinedFakeWorker:
        resumed_factory_calls.append(environment)
        raise AssertionError("ready resume must not construct a remote worker")

    resumed = _run(frozen_study, tmp_path, resumed_factory)

    assert resumed.resumed_ready_bundle is True
    assert resumed.bundle == result.bundle
    assert len(resumed.oracle_gates) == 32
    assert len(resumed.baseline_gates) == 24
    assert len(resumed.capability_records) == 5
    assert resumed.launch_record is not None
    assert resumed_factory_calls == []

    ready_directory = _ready_directory(tmp_path, frozen_study)
    ready_staging = ready_directory.with_name(f"{ready_directory.name}.incomplete")
    ready_directory.rename(ready_staging)
    promoted = _run(frozen_study, tmp_path, resumed_factory)
    assert promoted.resumed_ready_bundle is True
    assert promoted.bundle == result.bundle
    assert ready_directory.is_dir()
    assert not ready_staging.exists()
    assert resumed_factory_calls == []

    ready_staging.mkdir()
    with pytest.raises(
        MatrixPreflightArtifactError,
        match="final and staging ready preflight bundles both exist",
    ):
        _run(frozen_study, tmp_path, resumed_factory)
    ready_staging.rmdir()
    assert resumed_factory_calls == []

    first_oracle = next(item for item in result.contract.protocols if item.kind == "oracle")
    removed_oracle = tmp_path / result.contract.oracle_gate_study_id / first_oracle.artifact_id
    removed_oracle.chmod(0o700)
    removed_oracle.rename(tmp_path / "removed-oracle-evidence")
    with pytest.raises(
        MatrixPreflightRunnerError,
        match="raw preflight evidence artifact is missing",
    ):
        _run(frozen_study, tmp_path, resumed_factory)
    assert resumed_factory_calls == []


def test_environment_mismatch_stops_before_worker_jobs_and_no_ready_bundle(
    tmp_path: Path,
    frozen_study: FrozenStudy,
) -> None:
    workers: list[CombinedFakeWorker] = []

    def factory(environment: EnvironmentManifest) -> CombinedFakeWorker:
        worker = CombinedFakeWorker(
            environment,
            health_updates={"triton_version": "3.8.0"},
        )
        workers.append(worker)
        return worker

    with pytest.raises(MatrixPreflightInvalidFloorError, match="triton_version"):
        _run(frozen_study, tmp_path, factory)

    assert len(workers) == 1
    assert workers[0].health_calls == ["cuda:0"]
    assert workers[0].jobs == []
    verify_trajectory(_environment_directory(tmp_path, frozen_study))
    assert not _ready_directory(tmp_path, frozen_study).exists()


def test_environment_transport_outage_is_unsealed_and_retryable(
    tmp_path: Path,
    frozen_study: FrozenStudy,
) -> None:
    outage_workers: list[CombinedFakeWorker] = []

    def outage_factory(environment: EnvironmentManifest) -> CombinedFakeWorker:
        worker = CombinedFakeWorker(environment, transport_outage=True)
        outage_workers.append(worker)
        return worker

    with pytest.raises(
        MatrixPreflightInfrastructureError,
        match="environment probe timed out",
    ):
        _run(frozen_study, tmp_path, outage_factory)

    environment_directory = _environment_directory(tmp_path, frozen_study)
    assert outage_workers[0].health_calls == ["cuda:0"]
    assert outage_workers[0].jobs == []
    assert not environment_directory.exists()
    assert not environment_directory.with_name("environment-probe.incomplete").exists()

    retry_workers: list[CombinedFakeWorker] = []

    def retry_factory(environment: EnvironmentManifest) -> CombinedFakeWorker:
        worker = CombinedFakeWorker(environment)
        retry_workers.append(worker)
        return worker

    result = _run(frozen_study, tmp_path, retry_factory)

    assert result.bundle.receipt.status == "ready"
    assert retry_workers[0].health_calls == ["cuda:0"]
    assert len(retry_workers[0].jobs) == EXPECTED_STABLE_JOBS
    verify_trajectory(environment_directory)


def test_unhealthy_gpu_probe_is_unsealed_and_retryable(
    tmp_path: Path,
    frozen_study: FrozenStudy,
) -> None:
    workers: list[CombinedFakeWorker] = []

    def unhealthy_factory(environment: EnvironmentManifest) -> CombinedFakeWorker:
        worker = CombinedFakeWorker(environment, unhealthy_gpu=True)
        workers.append(worker)
        return worker

    with pytest.raises(MatrixPreflightInfrastructureError, match="CUDA context failed"):
        _run(frozen_study, tmp_path, unhealthy_factory)

    environment_directory = _environment_directory(tmp_path, frozen_study)
    assert workers[0].health_calls == ["cuda:0"]
    assert workers[0].jobs == []
    assert not environment_directory.exists()

    retry_workers: list[CombinedFakeWorker] = []

    def healthy_factory(environment: EnvironmentManifest) -> CombinedFakeWorker:
        worker = CombinedFakeWorker(environment)
        retry_workers.append(worker)
        return worker

    result = _run(frozen_study, tmp_path, healthy_factory)

    assert result.bundle.receipt.status == "ready"
    assert len(retry_workers[0].jobs) == EXPECTED_STABLE_JOBS


def test_checksum_bearing_environment_staging_fails_closed(
    tmp_path: Path,
    frozen_study: FrozenStudy,
) -> None:
    contract = build_preflight_study_contract(
        frozen_study.pinned,
        frozen_study.schedule,
        frozen_study.assets,
        frozen_study.pending,
        baseline_target_id=BASELINE_TARGET_ID,
    )
    worker = CombinedFakeWorker(frozen_study.pending)
    run_or_resume_environment_probe(
        worker,
        artifact_root=tmp_path,
        contract=contract,
    )
    final = _environment_directory(tmp_path, frozen_study)
    staging = final.with_name(f"{final.name}.incomplete")
    final.chmod(0o700)
    final.rename(staging)
    checksum = staging / "sha256sums.txt"
    checksum.chmod(0o600)
    checksum.write_bytes(checksum.read_bytes() + b"tampered\n")
    retry_worker = CombinedFakeWorker(frozen_study.pending)

    with pytest.raises(
        MatrixPreflightArtifactError,
        match="checksum-bearing preflight staging artifact is invalid",
    ):
        run_or_resume_environment_probe(
            retry_worker,
            artifact_root=tmp_path,
            contract=contract,
        )

    assert retry_worker.health_calls == []
    assert not final.exists()
    assert staging.is_dir()


def test_unknown_baseline_target_is_rejected_before_worker_factory(
    tmp_path: Path,
    frozen_study: FrozenStudy,
) -> None:
    factory_calls: list[EnvironmentManifest] = []

    def factory(environment: EnvironmentManifest) -> CombinedFakeWorker:
        factory_calls.append(environment)
        raise AssertionError("invalid baseline target must not construct a worker")

    with pytest.raises(MatrixPreflightRunnerError, match="baseline target"):
        _run(
            frozen_study,
            tmp_path,
            factory,
            baseline_target_id="tilelang-a100-missing",
        )

    assert factory_calls == []


def test_registry_drift_is_rejected_before_worker_factory(
    tmp_path: Path,
    frozen_study: FrozenStudy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_get_task_pack = preflight_runner.get_task_pack

    def drifted_get_task_pack(task_id: str):
        task = original_get_task_pack(task_id)
        if task_id == frozen_study.assets.tasks[0].task_id:
            return task.model_copy(
                update={"specification": f"{task.specification}\nregistry drift"}
            )
        return task

    monkeypatch.setattr(
        preflight_runner,
        "get_task_pack",
        drifted_get_task_pack,
    )
    factory_calls: list[EnvironmentManifest] = []

    def factory(environment: EnvironmentManifest) -> CombinedFakeWorker:
        factory_calls.append(environment)
        raise AssertionError("registry drift must not construct a worker")

    with pytest.raises(MatrixPreflightRunnerError, match="task registry differs"):
        _run(frozen_study, tmp_path, factory)

    assert factory_calls == []
