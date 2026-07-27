from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from abstrak.canary.artifacts import verify_trajectory
from abstrak.canary.capability_assets import (
    CAPABILITY_STUDY_SHA256,
    build_capability_asset_manifest,
)
from abstrak.canary.contracts import CaseResult, WorkerJob, WorkerResult
from abstrak.canary.manifests import load_study_spec
from abstrak.canary.matrix import build_matrix_schedule
from abstrak.canary.matrix_capability_evidence import (
    CapabilityProbeInfrastructureError,
    CapabilityProbeInput,
    CapabilityProbeStudyManifest,
    MatrixCapabilityEvidenceError,
    build_capability_probe_study_manifest,
    capability_probe_artifact_sha256,
    derive_capability_canary_evidence,
    resolve_capability_probe_inputs,
    run_capability_probe_study,
    seal_capability_probe_study_manifest,
)
from abstrak.canary.matrix_preflight import (
    BaselineTimingEvidence,
    EnvironmentManifest,
    EnvironmentObservation,
    EnvironmentProbeEvidence,
    ExpertCorrectnessEvidence,
    LatencyCeilingDerivation,
    TargetCodegenEvidence,
    TaskFloorRecord,
    VerifiedTaskFloorEvidence,
    build_pending_environment,
)
from abstrak.canary.matrix_runner import MatrixTransportContext
from abstrak.canary.target_adapters import validate_target_source

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_STUDY = REPOSITORY_ROOT / "benchmarks" / "capability-gate-a100" / "study.json"


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


def _verified_environment(pinned, schedule) -> EnvironmentManifest:
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


def _task_floor(task, targets) -> TaskFloorRecord:
    medians = {"compile": 1.0, "eager": 2.0, "vendor": 1.5}
    timings = tuple(
        BaselineTimingEvidence(
            variant=baseline.variant,
            source_sha256=baseline.source_sha256,
            artifact_sha256=_digest(f"{task.task_id}:{baseline.variant}:baseline-artifact"),
            timing_summary_sha256=_digest(f"{task.task_id}:{baseline.variant}:baseline-summary"),
            status="stable",
            median_ms=medians[baseline.variant],
        )
        for baseline in task.baselines
    )
    selected = next(item for item in timings if item.variant == "compile")
    evidence = VerifiedTaskFloorEvidence(
        task_id=task.task_id,
        expert_source_sha256=task.expert_source_sha256,
        expert_correctness=ExpertCorrectnessEvidence(
            artifact_sha256=_digest(f"{task.task_id}:expert-correctness"),
            task_id=task.task_id,
            task_pack_sha256=task.task_pack_sha256,
            expert_source_sha256=task.expert_source_sha256,
            status="pass",
            compiled=True,
            all_sealed_cases_passed=True,
            output_finite=True,
            inputs_unchanged=True,
            fallback_free=True,
        ),
        target_codegen=tuple(
            TargetCodegenEvidence(
                artifact_sha256=_digest(f"{task.task_id}:{target.target_id}:oracle-artifact"),
                task_id=task.task_id,
                task_pack_sha256=task.task_pack_sha256,
                target_id=target.target_id,
                target_stack_sha256=target.target_stack_sha256,
                expert_source_sha256=task.expert_source_sha256,
                status="pass",
                compiled=True,
                correct=True,
                fallback_free=True,
                generated_code_sha256=_digest(f"{task.task_id}:B-control-codegen"),
            )
            for target in targets
        ),
        baseline_timings=timings,
        selected_baseline_variant=selected.variant,
        selected_baseline_source_sha256=selected.source_sha256,
        selected_timing_summary_sha256=selected.timing_summary_sha256,
    )
    return TaskFloorRecord(
        task_id=task.task_id,
        status="valid",
        expert_source_sha256=task.expert_source_sha256,
        verified_evidence=evidence,
        ceiling=LatencyCeilingDerivation(
            l_star_ms=evidence.l_star_ms,
            competitive_factor=1.25,
            latency_ceiling_ms=evidence.l_star_ms * 1.25,
        ),
    )


@dataclass(frozen=True)
class ProbeFixture:
    assets: object
    environment: EnvironmentManifest
    floors: tuple[TaskFloorRecord, ...]
    inputs: tuple[CapabilityProbeInput, ...]
    manifest: CapabilityProbeStudyManifest


@pytest.fixture
def probe_fixture() -> ProbeFixture:
    pinned = load_study_spec(
        CAPABILITY_STUDY,
        expected_sha256=CAPABILITY_STUDY_SHA256,
    )
    schedule = build_matrix_schedule(pinned.spec)
    assets = build_capability_asset_manifest(pinned, schedule)
    environment = _verified_environment(pinned, schedule)
    floors = tuple(_task_floor(task, assets.targets) for task in assets.tasks)
    inputs = resolve_capability_probe_inputs(assets, environment, floors)
    manifest = build_capability_probe_study_manifest(assets, environment, inputs)
    return ProbeFixture(
        assets=assets,
        environment=environment,
        floors=floors,
        inputs=inputs,
        manifest=manifest,
    )


class _Worker:
    def __init__(
        self,
        manifest: CapabilityProbeStudyManifest,
        *,
        modes: dict[str, str] | None = None,
    ) -> None:
        self.matrix_worker_binding = manifest.worker
        self.identities = {item.artifact_id: item for item in manifest.probes}
        self.modes = modes or {}
        self.calls: list[WorkerJob] = []

    def _artifact_id(self, job: WorkerJob) -> str:
        suffix = "-timing-a"
        return job.job_id.partition(suffix)[0]

    def execute(self, job: WorkerJob) -> WorkerResult:
        self.calls.append(job)
        artifact_id = self._artifact_id(job)
        mode = self.modes.get(artifact_id, "pass")
        if mode == "infrastructure":
            raise RuntimeError("SSH connection reset")

        static = validate_target_source(job.candidate_source, job.target)
        assert static.valid
        metadata = dict(static.metadata)
        identity = self.identities[artifact_id]
        generated = _digest(f"{job.candidate_sha256}:generated-code")
        if mode == "equal-control":
            generated = identity.control.generated_code_sha256
        elif mode == "codegen-drift":
            generated = _digest(f"{job.job_id}:generated-code")
        if mode != "compile-error":
            metadata.update(
                {
                    "generated_code_capture": "tilelang.get_kernel_source.v1",
                    "generated_code_sha256": generated,
                    "generated_code_size_bytes": 1024,
                }
            )
        if mode == "missing-codegen":
            metadata.pop("generated_code_capture")
            metadata.pop("generated_code_sha256")
            metadata.pop("generated_code_size_bytes")
        if mode == "static-drift":
            metadata["minimum_pack_id"] = "tileops-core"
            metadata["minimum_pack_bitmask"] = 1
        elif mode == "used-drift":
            metadata["used_capabilities"] = ("T.Kernel",)

        warnings = tuple(f"{item.code}: {item.message}" for item in static.warnings)
        if mode == "compile-error":
            return WorkerResult(
                job_id=job.job_id,
                job_sha256=job.sha256,
                input_sha256=job.input_sha256,
                candidate_sha256=job.candidate_sha256,
                status="compile_error",
                static_warnings=warnings,
                metadata=metadata,
                error="RuntimeError: TileLang compilation failed",
            ).verify_for_job(job)

        cases = tuple(
            CaseResult(
                case_id=case_id,
                status=("wrong_result" if mode == "wrong-result" and index == 0 else "pass"),
                correct=not (mode == "wrong-result" and index == 0),
                max_abs_error=(1.0 if mode == "wrong-result" and index == 0 else 0.0),
                max_rel_error=(1.0 if mode == "wrong-result" and index == 0 else 0.0),
                output_finite=True,
                inputs_unchanged=True,
            )
            for index, case_id in enumerate(job.case_ids)
        )
        if mode == "wrong-result":
            return WorkerResult(
                job_id=job.job_id,
                job_sha256=job.sha256,
                input_sha256=job.input_sha256,
                candidate_sha256=job.candidate_sha256,
                status="wrong_result",
                compiled=True,
                correct=False,
                cases=cases,
                static_warnings=warnings,
                metadata=metadata,
            ).verify_for_job(job)

        assert job.timing is not None
        timing_cv = 0.04 if mode == "timing-drift" else 0.0
        return WorkerResult(
            job_id=job.job_id,
            job_sha256=job.sha256,
            input_sha256=job.input_sha256,
            candidate_sha256=job.candidate_sha256,
            status="completed",
            compiled=True,
            correct=True,
            cases=cases,
            timing_ms=tuple(2.0 for _ in range(job.timing.trial_runs)),
            timing_cv=timing_cv,
            static_warnings=warnings,
            metadata=metadata,
        ).verify_for_job(job)


def test_resolves_exact_required_targets_and_same_target_controls(
    probe_fixture: ProbeFixture,
) -> None:
    inputs = probe_fixture.inputs

    assert [(item.identity.canary_id, item.identity.target_id) for item in inputs] == [
        ("schedule", "tilelang-a100-sched"),
        ("schedule", "tilelang-a100-full"),
        ("mapping", "tilelang-a100-map"),
        ("mapping", "tilelang-a100-full"),
        ("schedule-mapping", "tilelang-a100-full"),
    ]
    assert [item.identity.minimum_pack_id for item in inputs] == [
        "tileops-sched",
        "tileops-sched",
        "tileops-map",
        "tileops-map",
        "tileops-full",
    ]
    assert all(item.identity.static.used_capabilities for item in inputs)
    assert all(
        item.identity.control.expert_source_sha256 == item.task_asset.expert_source_sha256
        for item in inputs
    )
    assert all(item.identity.worker == probe_fixture.manifest.worker for item in inputs)


def test_probe_study_is_atomic_recomputable_and_exactly_resumable(
    tmp_path: Path,
    probe_fixture: ProbeFixture,
) -> None:
    worker = _Worker(probe_fixture.manifest)

    records = run_capability_probe_study(
        worker,
        artifact_root=tmp_path,
        manifest=probe_fixture.manifest,
        inputs=probe_fixture.inputs,
    )
    evidence = derive_capability_canary_evidence(
        probe_fixture.manifest,
        probe_fixture.inputs,
        records,
    )

    assert len(worker.calls) == 5 * 3
    assert [item.canary_id for item in evidence] == [
        "schedule",
        "mapping",
        "schedule-mapping",
    ]
    assert [len(item.targets) for item in evidence] == [2, 2, 1]
    assert all(item.status == "pass" for item in evidence)
    for record in records:
        verify_trajectory(record.artifact_directory)
        assert (
            capability_probe_artifact_sha256(record)
            == hashlib.sha256(
                (Path(record.artifact_directory) / "sha256sums.txt").read_bytes()
            ).hexdigest()
        )
        assert not Path(f"{record.artifact_directory}.incomplete").exists()

    resumed = run_capability_probe_study(
        worker,
        artifact_root=tmp_path,
        manifest=probe_fixture.manifest,
        inputs=probe_fixture.inputs,
    )
    assert resumed == records
    assert len(worker.calls) == 5 * 3


@pytest.mark.parametrize("mode", ["wrong-result", "compile-error", "equal-control"])
def test_terminal_scientific_failures_are_derived_as_failed_evidence(
    tmp_path: Path,
    probe_fixture: ProbeFixture,
    mode: str,
) -> None:
    first_id = probe_fixture.inputs[0].identity.artifact_id
    worker = _Worker(probe_fixture.manifest, modes={first_id: mode})

    records = run_capability_probe_study(
        worker,
        artifact_root=tmp_path,
        manifest=probe_fixture.manifest,
        inputs=probe_fixture.inputs,
    )
    evidence = derive_capability_canary_evidence(
        probe_fixture.manifest,
        probe_fixture.inputs,
        records,
    )

    assert evidence[0].status == "fail"
    assert evidence[0].targets[0].status == "fail"
    assert evidence[0].targets[0].failure_reason
    assert Path(records[0].artifact_directory).is_dir()


def test_infrastructure_failure_never_becomes_scientific_evidence(
    tmp_path: Path,
    probe_fixture: ProbeFixture,
) -> None:
    first = probe_fixture.inputs[0].identity
    worker = _Worker(
        probe_fixture.manifest,
        modes={first.artifact_id: "infrastructure"},
    )

    with pytest.raises(CapabilityProbeInfrastructureError, match="SSH connection reset"):
        run_capability_probe_study(
            worker,
            artifact_root=tmp_path,
            manifest=probe_fixture.manifest,
            inputs=probe_fixture.inputs,
        )

    study_root = tmp_path / probe_fixture.manifest.probe_study_id
    assert (study_root / "study-manifest").is_dir()
    assert not (study_root / first.artifact_id).exists()
    assert not (study_root / f"{first.artifact_id}.incomplete").exists()

    healthy = _Worker(probe_fixture.manifest)
    records = run_capability_probe_study(
        healthy,
        artifact_root=tmp_path,
        manifest=probe_fixture.manifest,
        inputs=probe_fixture.inputs,
    )
    assert len(records) == 5


@pytest.mark.parametrize(
    "mode",
    [
        "static-drift",
        "used-drift",
        "timing-drift",
        "codegen-drift",
        "missing-codegen",
    ],
)
def test_raw_worker_metadata_and_timing_are_recomputed_before_sealing(
    tmp_path: Path,
    probe_fixture: ProbeFixture,
    mode: str,
) -> None:
    first = probe_fixture.inputs[0].identity
    worker = _Worker(probe_fixture.manifest, modes={first.artifact_id: mode})

    with pytest.raises(MatrixCapabilityEvidenceError):
        run_capability_probe_study(
            worker,
            artifact_root=tmp_path,
            manifest=probe_fixture.manifest,
            inputs=probe_fixture.inputs,
        )

    study_root = tmp_path / probe_fixture.manifest.probe_study_id
    assert not (study_root / first.artifact_id).exists()


def test_unsealed_staging_is_discarded_but_sealed_tampering_fails_closed(
    tmp_path: Path,
    probe_fixture: ProbeFixture,
) -> None:
    manifest = probe_fixture.manifest
    seal_capability_probe_study_manifest(tmp_path, manifest)
    first = probe_fixture.inputs[0].identity
    staging = tmp_path / manifest.probe_study_id / f"{first.artifact_id}.incomplete"
    staging.mkdir()
    (staging / "partial").write_text("controller crashed", encoding="utf-8")

    worker = _Worker(manifest)
    records = run_capability_probe_study(
        worker,
        artifact_root=tmp_path,
        manifest=manifest,
        inputs=probe_fixture.inputs,
    )
    assert not staging.exists()

    checksum = Path(records[0].artifact_directory) / "sha256sums.txt"
    checksum.chmod(0o600)
    checksum.write_bytes(checksum.read_bytes() + b"tampered\n")
    with pytest.raises(MatrixCapabilityEvidenceError, match="invalid"):
        derive_capability_canary_evidence(
            manifest,
            probe_fixture.inputs,
            records,
        )


def test_resume_rejects_control_identity_drift_before_worker_execution(
    tmp_path: Path,
    probe_fixture: ProbeFixture,
) -> None:
    seal_capability_probe_study_manifest(tmp_path, probe_fixture.manifest)
    first = probe_fixture.inputs[0]
    drifted_identity = first.identity.model_copy(
        update={
            "control": first.identity.control.model_copy(
                update={"generated_code_sha256": _digest("drifted-control")}
            )
        }
    )
    drifted_input = CapabilityProbeInput(
        identity=drifted_identity,
        source=first.source,
        task=first.task,
        target=first.target,
        task_asset=first.task_asset,
        target_asset=first.target_asset,
        control_evidence=first.control_evidence.model_copy(
            update={"generated_code_sha256": _digest("drifted-control")}
        ),
    )
    drifted_inputs = (drifted_input, *probe_fixture.inputs[1:])
    drifted_manifest = probe_fixture.manifest.model_copy(
        update={"probes": tuple(item.identity for item in drifted_inputs)}
    )
    worker = _Worker(drifted_manifest)

    with pytest.raises(MatrixCapabilityEvidenceError, match="differs from frozen inputs"):
        run_capability_probe_study(
            worker,
            artifact_root=tmp_path,
            manifest=drifted_manifest,
            inputs=drifted_inputs,
        )

    assert worker.calls == []


def test_derivation_requires_exact_ordered_record_coverage(
    tmp_path: Path,
    probe_fixture: ProbeFixture,
) -> None:
    records = run_capability_probe_study(
        _Worker(probe_fixture.manifest),
        artifact_root=tmp_path,
        manifest=probe_fixture.manifest,
        inputs=probe_fixture.inputs,
    )

    with pytest.raises(MatrixCapabilityEvidenceError, match="exactly cover"):
        derive_capability_canary_evidence(
            probe_fixture.manifest,
            probe_fixture.inputs,
            records[:-1],
        )
