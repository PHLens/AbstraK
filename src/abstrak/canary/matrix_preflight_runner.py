"""Crash-resistant orchestration for one manifest-driven matrix preflight."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, model_validator

from abstrak.canary.artifacts import (
    TrajectoryArtifactError,
    TrajectoryStore,
    verify_trajectory,
)
from abstrak.canary.baselines import BASELINE_VARIANTS, get_baseline_source
from abstrak.canary.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    CanaryModel,
    TargetStackSpec,
    TaskPackSpec,
    TimingSpec,
    WorkerJob,
    WorkerResult,
)
from abstrak.canary.gates import (
    GateError,
    GateInfrastructureError,
    GateRecord,
    run_baseline_gates,
    run_oracle_gates,
)
from abstrak.canary.loop import WorkerExecutor
from abstrak.canary.manifests import PinnedStudySpec
from abstrak.canary.matrix import MatrixSchedule
from abstrak.canary.matrix_capability_evidence import (
    CapabilityProbeInfrastructureError,
    CapabilityProbeRecord,
    CapabilityProbeStudyManifest,
    MatrixCapabilityEvidenceError,
    build_capability_probe_study_manifest,
    capability_probe_artifact_id,
    capability_probe_study_id,
    derive_capability_canary_evidence,
    resolve_capability_probe_inputs,
    run_capability_probe_study,
)
from abstrak.canary.matrix_environment_evidence import (
    EnvironmentProbeArtifact,
    EnvironmentProbeResult,
    EnvironmentProbeWorker,
    MatrixEnvironmentEvidenceError,
    build_environment_probe_worker,
    derive_environment_probe,
    run_environment_probe,
)
from abstrak.canary.matrix_floor_evidence import (
    MatrixFloorEvidenceError,
    derive_task_floor_records,
)
from abstrak.canary.matrix_launch_floor_evidence import (
    LaunchFloorProbeInfrastructureError,
    LaunchFloorProbeRecord,
    LaunchFloorStudyManifest,
    MatrixLaunchFloorEvidenceError,
    build_launch_floor_study_manifest,
    derive_launch_floor_evidence,
    resolve_launch_floor_probe_input,
    run_launch_floor_probe,
)
from abstrak.canary.matrix_preflight import (
    FORMAL_FLOOR_TIMING,
    AssetManifest,
    EnvironmentManifest,
    FloorManifest,
    MatrixPreflightError,
    PreflightBundle,
    StudyBoundModel,
    TaskFloorRecord,
    load_preflight_bundle,
    seal_preflight_bundle,
)
from abstrak.canary.matrix_runner import (
    MatrixExecutionContext,
    MatrixWorkerBinding,
)
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import (
    get_task_assets,
    get_task_pack,
    load_oracle_source,
    load_task_source,
)
from abstrak.canary.timing import timing_protocol_job_ceiling
from abstrak.providers.contracts import sha256_json

_STORE_DIRECTORIES = frozenset({"events", "turns", "candidates", "sealed"})
_CONTRACT_TRAJECTORY_ID = "preflight-contract"
_ENVIRONMENT_TRAJECTORY_ID = "environment-probe"
_PREFLIGHT_BUNDLE_TRAJECTORY_ID = "matrix-preflight"


class MatrixPreflightRunnerError(RuntimeError):
    """Raised when preflight inputs or durable artifacts cannot be verified."""


class MatrixPreflightArtifactError(MatrixPreflightRunnerError):
    """Raised when durable preflight evidence is missing, corrupt, or inconsistent."""


class MatrixPreflightInfrastructureError(MatrixPreflightRunnerError):
    """Raised when live infrastructure produced no scientific evidence."""


class MatrixPreflightInvalidFloorError(MatrixPreflightRunnerError):
    """Raised when terminal evidence proves that the study floor is invalid."""


class MatrixPreflightWorker(EnvironmentProbeWorker, WorkerExecutor, Protocol):
    """Combined health and job surface used by one preflight route."""


PreflightWorkerFactory = Callable[[EnvironmentManifest], MatrixPreflightWorker]


class _CappedPreflightWorker:
    """Count every GPU job across all preflight producers before delegation."""

    def __init__(
        self,
        worker: MatrixPreflightWorker,
        *,
        max_worker_jobs_per_invocation: int,
    ) -> None:
        self._worker = worker
        self._max_worker_jobs_per_invocation = max_worker_jobs_per_invocation
        self._worker_job_count = 0

    @property
    def worker_job_count(self) -> int:
        return self._worker_job_count

    def validate_environment(self, device: str) -> dict[str, object]:
        """Delegate health checks without charging them against the GPU-job cap."""

        return self._worker.validate_environment(device)

    def execute(self, job: WorkerJob) -> WorkerResult:
        if self._worker_job_count >= self._max_worker_jobs_per_invocation:
            raise MatrixPreflightRunnerError(
                "preflight worker-job hard cap reached; refusing execution "
                f"after {self._max_worker_jobs_per_invocation} jobs in this invocation"
            )
        self._worker_job_count += 1
        return self._worker.execute(job)

    def __getattr__(self, name: str) -> object:
        """Preserve the frozen environment-binding surface of the real worker."""

        return getattr(self._worker, name)


class PreflightProtocolIdentity(CanaryModel):
    """One frozen timing protocol counted by the preflight execution ceiling."""

    schema_version: Literal["abstrak-matrix-preflight-protocol.v1"] = (
        "abstrak-matrix-preflight-protocol.v1"
    )
    kind: Literal["oracle", "baseline", "capability", "launch"]
    artifact_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    variant: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    canary_id: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)

    @model_validator(mode="after")
    def optional_identity_matches_kind(self) -> PreflightProtocolIdentity:
        if (self.variant is not None) != (self.kind == "baseline"):
            raise ValueError("only baseline preflight protocols may declare a variant")
        if (self.canary_id is not None) != (self.kind == "capability"):
            raise ValueError("only capability preflight protocols may declare a canary")
        return self


class MatrixPreflightStudyContract(StudyBoundModel):
    """Complete execution closure sealed before the first remote health check."""

    schema_version: Literal["abstrak-matrix-preflight-study-contract.v1"] = (
        "abstrak-matrix-preflight-study-contract.v1"
    )
    asset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    pending_environment: EnvironmentManifest
    worker: MatrixWorkerBinding
    baseline_target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    timing: TimingSpec
    oracle_gate_study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    baseline_gate_study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    capability_probe_study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    launch_floor_study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    protocol_count: int = Field(ge=1)
    max_worker_jobs_per_invocation: int = Field(ge=1)
    protocols: tuple[PreflightProtocolIdentity, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def closure_is_self_consistent(self) -> MatrixPreflightStudyContract:
        if self.pending_environment.status != "pending":
            raise ValueError("preflight contract requires a pending environment")
        if self.pending_environment.study_binding != self.study_binding:
            raise ValueError("pending environment differs from the preflight study")
        expected_worker = MatrixWorkerBinding(
            worker_revision=self.pending_environment.worker_revision,
            transport=self.pending_environment.transport,
        )
        if self.worker != expected_worker:
            raise ValueError("preflight worker differs from the pending environment")
        if self.timing != FORMAL_FLOOR_TIMING:
            raise ValueError("preflight must use FORMAL_FLOOR_TIMING")
        if self.protocol_count != len(self.protocols):
            raise ValueError("preflight protocol count differs from its identities")
        expected_jobs = self.protocol_count * timing_protocol_job_ceiling(self.timing)
        if self.max_worker_jobs_per_invocation != expected_jobs:
            raise ValueError(
                "per-invocation preflight worker-job ceiling differs from its protocols"
            )
        artifact_ids = tuple(item.artifact_id for item in self.protocols)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("preflight protocol artifact IDs must be unique")
        expected_study_ids = (
            f"{self.study_id}-preflight-oracle-gates",
            f"{self.study_id}-preflight-baseline-gates",
            capability_probe_study_id(self.study_id),
            f"{self.study_id}-launch-floor",
        )
        actual_study_ids = (
            self.oracle_gate_study_id,
            self.baseline_gate_study_id,
            self.capability_probe_study_id,
            self.launch_floor_study_id,
        )
        if actual_study_ids != expected_study_ids:
            raise ValueError("preflight stage study IDs differ from the matrix study")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class EnvironmentProbeRecord(CanaryModel):
    """Durable raw environment observation bound to the preflight contract."""

    schema_version: Literal["abstrak-matrix-environment-probe-record.v1"] = (
        "abstrak-matrix-environment-probe-record.v1"
    )
    preflight_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    expected_environment_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact: EnvironmentProbeArtifact

    @model_validator(mode="after")
    def artifact_matches_expected_environment(self) -> EnvironmentProbeRecord:
        if self.artifact.expected_environment_sha256 != self.expected_environment_sha256:
            raise ValueError("environment artifact differs from its expected manifest")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


@dataclass(frozen=True)
class MatrixPreflightRunResult:
    """Ready bundle plus the durable stage identities that produced it."""

    contract: MatrixPreflightStudyContract
    environment_probe: EnvironmentProbeArtifact
    oracle_gates: tuple[GateRecord, ...]
    baseline_gates: tuple[GateRecord, ...]
    task_floors: tuple[TaskFloorRecord, ...]
    capability_manifest: CapabilityProbeStudyManifest | None
    capability_records: tuple[CapabilityProbeRecord, ...]
    launch_manifest: LaunchFloorStudyManifest | None
    launch_record: LaunchFloorProbeRecord | None
    bundle: PreflightBundle
    preflight_directory: Path
    resumed_ready_bundle: bool


def _study_fields(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
) -> dict[str, str]:
    if schedule.spec != pinned.spec or schedule.spec_sha256 != pinned.spec.sha256:
        raise MatrixPreflightRunnerError("matrix schedule differs from the pinned study spec")
    return {
        "study_id": pinned.spec.study_id,
        "raw_study_sha256": pinned.sha256,
        "spec_sha256": pinned.spec.sha256,
        "schedule_sha256": schedule.sha256,
    }


def _protocols(
    assets: AssetManifest,
    *,
    baseline_target_id: str,
) -> tuple[PreflightProtocolIdentity, ...]:
    protocols: list[PreflightProtocolIdentity] = []
    for task in assets.tasks:
        for target in assets.targets:
            protocols.append(
                PreflightProtocolIdentity(
                    kind="oracle",
                    artifact_id=f"oracle-{task.task_id}-{target.target_id}",
                    task_id=task.task_id,
                    target_id=target.target_id,
                    source_sha256=task.expert_source_sha256,
                )
            )
    for task in assets.tasks:
        for baseline in task.baselines:
            protocols.append(
                PreflightProtocolIdentity(
                    kind="baseline",
                    artifact_id=(
                        f"baseline-{task.task_id}-{baseline_target_id}-{baseline.variant}"
                    ),
                    task_id=task.task_id,
                    target_id=baseline_target_id,
                    source_sha256=baseline.source_sha256,
                    variant=baseline.variant,
                )
            )
    for canary in assets.canaries:
        for target_id in canary.required_target_ids:
            protocols.append(
                PreflightProtocolIdentity(
                    kind="capability",
                    artifact_id=capability_probe_artifact_id(
                        canary.canary_id,
                        target_id,
                    ),
                    task_id=canary.task_id,
                    target_id=target_id,
                    source_sha256=canary.source_sha256,
                    canary_id=canary.canary_id,
                )
            )
    if assets.launch_probe is None:
        raise MatrixPreflightRunnerError("preflight assets do not contain a frozen launch probe")
    launch = assets.launch_probe
    protocols.append(
        PreflightProtocolIdentity(
            kind="launch",
            artifact_id="launch-floor-probe",
            task_id=launch.task_id,
            target_id=launch.target_id,
            source_sha256=launch.source_sha256,
        )
    )
    return tuple(protocols)


def build_preflight_study_contract(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    assets: AssetManifest,
    pending_environment: EnvironmentManifest,
    *,
    baseline_target_id: str,
) -> MatrixPreflightStudyContract:
    """Build the exact 62-protocol contract without touching a remote worker."""

    fields = _study_fields(pinned, schedule)
    expected_binding = (
        fields["study_id"],
        fields["raw_study_sha256"],
        fields["spec_sha256"],
        fields["schedule_sha256"],
    )
    if assets.study_binding != expected_binding:
        raise MatrixPreflightRunnerError("asset manifest differs from the pinned preflight study")
    if pending_environment.study_binding != assets.study_binding:
        raise MatrixPreflightRunnerError("pending environment differs from the preflight assets")
    if baseline_target_id not in {item.target_id for item in assets.targets}:
        raise MatrixPreflightRunnerError("baseline target is not present in frozen assets")
    gate = pinned.spec.gate
    if gate is None:
        raise MatrixPreflightRunnerError("matrix preflight requires a study gate")
    if gate.metrics.max_timing_cv != FORMAL_FLOOR_TIMING.max_cv:
        raise MatrixPreflightRunnerError(
            "study timing CV threshold differs from FORMAL_FLOOR_TIMING"
        )
    protocol_values = _protocols(
        assets,
        baseline_target_id=baseline_target_id,
    )
    return MatrixPreflightStudyContract(
        **fields,
        asset_manifest_sha256=assets.sha256,
        pending_environment=pending_environment,
        worker=MatrixWorkerBinding(
            worker_revision=pending_environment.worker_revision,
            transport=pending_environment.transport,
        ),
        baseline_target_id=baseline_target_id,
        timing=FORMAL_FLOOR_TIMING,
        oracle_gate_study_id=f"{assets.study_id}-preflight-oracle-gates",
        baseline_gate_study_id=f"{assets.study_id}-preflight-baseline-gates",
        capability_probe_study_id=capability_probe_study_id(assets.study_id),
        launch_floor_study_id=f"{assets.study_id}-launch-floor",
        protocol_count=len(protocol_values),
        max_worker_jobs_per_invocation=(
            len(protocol_values) * timing_protocol_job_ceiling(FORMAL_FLOOR_TIMING)
        ),
        protocols=protocol_values,
    )


def preflight_worker_job_ceiling(
    assets: AssetManifest,
    *,
    baseline_target_id: str,
    timing: TimingSpec = FORMAL_FLOOR_TIMING,
) -> int:
    """Return the exact fresh-run ceiling without constructing a live worker."""

    if baseline_target_id not in {item.target_id for item in assets.targets}:
        raise MatrixPreflightRunnerError("baseline target is not present in frozen assets")
    return len(
        _protocols(assets, baseline_target_id=baseline_target_id)
    ) * timing_protocol_job_ceiling(timing)


def validate_preflight_run_guards(
    contract: MatrixPreflightStudyContract,
    *,
    live: bool,
    expected_max_worker_jobs_per_invocation: int,
) -> None:
    """Reject missing live consent or stale ceilings before any remote action."""

    if live is not True:
        raise MatrixPreflightRunnerError("preflight-study requires live authorization")
    if (
        isinstance(expected_max_worker_jobs_per_invocation, bool)
        or not isinstance(expected_max_worker_jobs_per_invocation, int)
        or expected_max_worker_jobs_per_invocation
        != contract.max_worker_jobs_per_invocation
    ):
        raise MatrixPreflightRunnerError(
            "expected worker-job ceiling must equal the frozen preflight ceiling "
            f"({contract.max_worker_jobs_per_invocation}) for each authorized invocation"
        )


def _verify_shape(
    directory: Path,
    *,
    expected_files: frozenset[str],
) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise MatrixPreflightArtifactError(
            f"preflight artifact is not a regular directory: {directory}"
        )
    entries = tuple(directory.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise MatrixPreflightArtifactError(
            f"preflight artifact contains a symbolic link: {directory}"
        )
    files = {item.relative_to(directory).as_posix() for item in entries if item.is_file()}
    directories = {item.relative_to(directory).as_posix() for item in entries if item.is_dir()}
    if files != expected_files or directories != _STORE_DIRECTORIES:
        raise MatrixPreflightArtifactError(
            f"preflight artifact has unexpected files or directories: {directory}"
        )


def _remove_unsealed_staging(directory: Path) -> bool:
    if not directory.exists() and not directory.is_symlink():
        return False
    if directory.is_symlink():
        raise MatrixPreflightArtifactError(
            "preflight staging artifact cannot be a symbolic link"
        )
    if not directory.is_dir():
        raise MatrixPreflightArtifactError(
            "preflight staging artifact is not a regular directory"
        )
    checksum = directory / "sha256sums.txt"
    if not checksum.exists() and not checksum.is_symlink():
        shutil.rmtree(directory)
        return True
    try:
        verify_trajectory(directory)
    except (OSError, TrajectoryArtifactError) as error:
        raise MatrixPreflightArtifactError(
            "checksum-bearing preflight staging artifact is invalid"
        ) from error
    return False


def _contract_directory(
    artifact_root: str | Path,
    study_id: str,
) -> Path:
    return Path(artifact_root).expanduser() / study_id / _CONTRACT_TRAJECTORY_ID


def _load_contract(
    directory: Path,
    expected: MatrixPreflightStudyContract,
) -> MatrixPreflightStudyContract:
    try:
        _verify_shape(
            directory,
            expected_files=frozenset({"run-manifest.json", "sha256sums.txt"}),
        )
        verify_trajectory(directory)
        actual = MatrixPreflightStudyContract.model_validate_json(
            (directory / "run-manifest.json").read_text(encoding="utf-8")
        )
    except MatrixPreflightRunnerError:
        raise
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise MatrixPreflightArtifactError(
            "sealed preflight study contract is invalid"
        ) from error
    if actual != expected:
        raise MatrixPreflightArtifactError(
            "sealed preflight study contract differs from current inputs"
        )
    return actual


def seal_preflight_study_contract(
    artifact_root: str | Path,
    contract: MatrixPreflightStudyContract,
) -> Path:
    """Atomically seal or exactly resume the pre-SSH execution contract."""

    final = _contract_directory(artifact_root, contract.study_id)
    staging = final.with_name(f"{final.name}.incomplete")
    if (final.exists() or final.is_symlink()) and (staging.exists() or staging.is_symlink()):
        raise MatrixPreflightArtifactError(
            "final and staging preflight contracts both exist"
        )
    if final.exists() or final.is_symlink():
        _load_contract(final, contract)
        return final
    if staging.exists() or staging.is_symlink():
        removed = _remove_unsealed_staging(staging)
        if not removed:
            _load_contract(staging, contract)
            os.replace(staging, final)
            return final
    try:
        store = TrajectoryStore.create(
            artifact_root,
            contract.study_id,
            f"{_CONTRACT_TRAJECTORY_ID}.incomplete",
        )
        store.write_json("run-manifest.json", contract)
        store.seal()
        if final.exists() or final.is_symlink():
            raise MatrixPreflightArtifactError(
                "preflight contract appeared during atomic staging"
            )
        os.replace(store.run_directory, final)
    except MatrixPreflightRunnerError:
        raise
    except (OSError, TrajectoryArtifactError) as error:
        raise MatrixPreflightArtifactError(
            "cannot seal preflight study contract"
        ) from error
    _load_contract(final, contract)
    return final


def _environment_manifest(
    contract: MatrixPreflightStudyContract,
) -> dict[str, str]:
    return {
        "schema_version": "abstrak-matrix-environment-probe-run.v1",
        "preflight_contract_sha256": contract.sha256,
        "expected_environment_sha256": contract.pending_environment.sha256,
    }


def _load_environment_record(
    directory: Path,
    contract: MatrixPreflightStudyContract,
) -> EnvironmentProbeRecord:
    try:
        _verify_shape(
            directory,
            expected_files=frozenset(
                {
                    "run-manifest.json",
                    "environment-probe-record.json",
                    "sha256sums.txt",
                }
            ),
        )
        verify_trajectory(directory)
        manifest = json.loads((directory / "run-manifest.json").read_text(encoding="utf-8"))
        record = EnvironmentProbeRecord.model_validate_json(
            (directory / "environment-probe-record.json").read_text(encoding="utf-8")
        )
    except MatrixPreflightRunnerError:
        raise
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise MatrixPreflightArtifactError(
            "sealed environment probe record is invalid"
        ) from error
    if manifest != _environment_manifest(contract):
        raise MatrixPreflightArtifactError(
            "environment probe run manifest differs from the preflight contract"
        )
    if (
        record.preflight_contract_sha256 != contract.sha256
        or record.expected_environment_sha256 != contract.pending_environment.sha256
    ):
        raise MatrixPreflightArtifactError(
            "environment probe record differs from the preflight contract"
        )
    try:
        derive_environment_probe(contract.pending_environment, record.artifact)
    except MatrixEnvironmentEvidenceError as error:
        raise MatrixPreflightArtifactError(
            "sealed environment probe record differs from the preflight contract"
        ) from error
    return record


def run_or_resume_environment_probe(
    worker: MatrixPreflightWorker,
    *,
    artifact_root: str | Path,
    contract: MatrixPreflightStudyContract,
) -> EnvironmentProbeResult:
    """Persist one raw health observation, while leaving outages retryable."""

    study_root = Path(artifact_root).expanduser() / contract.study_id
    final = study_root / _ENVIRONMENT_TRAJECTORY_ID
    staging = final.with_name(f"{final.name}.incomplete")
    if (final.exists() or final.is_symlink()) and (staging.exists() or staging.is_symlink()):
        raise MatrixPreflightArtifactError(
            "final and staging environment probes both exist"
        )
    if final.exists() or final.is_symlink():
        record = _load_environment_record(final, contract)
        return derive_environment_probe(
            contract.pending_environment,
            record.artifact,
        )
    if staging.exists() or staging.is_symlink():
        removed = _remove_unsealed_staging(staging)
        if not removed:
            record = _load_environment_record(staging, contract)
            os.replace(staging, final)
            return derive_environment_probe(
                contract.pending_environment,
                record.artifact,
            )

    try:
        result = run_environment_probe(contract.pending_environment, worker)
    except MatrixEnvironmentEvidenceError as error:
        raise MatrixPreflightRunnerError(f"cannot run bound environment probe: {error}") from error
    probe_error = result.artifact.probe_error
    if probe_error is not None and probe_error.category in {
        "health_check_failed",
        "health_unhealthy",
        "quarantined",
    }:
        raise MatrixPreflightInfrastructureError(probe_error.message)
    record = EnvironmentProbeRecord(
        preflight_contract_sha256=contract.sha256,
        expected_environment_sha256=contract.pending_environment.sha256,
        artifact=result.artifact,
    )
    try:
        store = TrajectoryStore.create(
            artifact_root,
            contract.study_id,
            f"{_ENVIRONMENT_TRAJECTORY_ID}.incomplete",
        )
        store.write_json("run-manifest.json", _environment_manifest(contract))
        store.write_json("environment-probe-record.json", record)
        store.seal()
        if final.exists() or final.is_symlink():
            raise MatrixPreflightArtifactError(
                "environment probe appeared during atomic staging"
            )
        os.replace(store.run_directory, final)
    except MatrixPreflightRunnerError:
        raise
    except (OSError, TrajectoryArtifactError) as error:
        raise MatrixPreflightArtifactError(
            "cannot seal environment probe record"
        ) from error
    persisted = _load_environment_record(final, contract)
    return derive_environment_probe(
        contract.pending_environment,
        persisted.artifact,
    )


def _resolve_registry_assets(
    assets: AssetManifest,
    *,
    asset_root: str | Path,
) -> tuple[tuple[TaskPackSpec, ...], tuple[TargetStackSpec, ...]]:
    tasks: list[TaskPackSpec] = []
    for binding in assets.tasks:
        task = get_task_pack(binding.task_id)
        registered = get_task_assets(binding.task_id)
        reference = load_task_source(binding.task_id, asset_root=asset_root)
        expert = load_oracle_source(
            binding.task_id,
            "tilelang",
            asset_root=asset_root,
        )
        baseline_variants = tuple(item.variant for item in binding.baselines)
        if (
            sha256_json(task) != binding.task_pack_sha256
            or registered.source.sha256 != binding.reference_source_sha256
            or registered.oracles["tilelang"].sha256 != binding.expert_source_sha256
            or hashlib.sha256(reference.encode("utf-8")).hexdigest()
            != binding.reference_source_sha256
            or hashlib.sha256(expert.encode("utf-8")).hexdigest() != binding.expert_source_sha256
            or baseline_variants != BASELINE_VARIANTS
            or any(
                get_baseline_source(binding.task_id, item.variant).source_sha256
                != item.source_sha256
                for item in binding.baselines
            )
        ):
            raise MatrixPreflightRunnerError(
                f"task registry differs from frozen assets: {binding.task_id}"
            )
        tasks.append(task)
    targets: list[TargetStackSpec] = []
    for binding in assets.targets:
        target = get_target_stack(binding.target_id)
        if (
            sha256_json(target) != binding.target_stack_sha256
            or target.card_sha256 != binding.card_sha256
        ):
            raise MatrixPreflightRunnerError(
                f"target registry differs from frozen assets: {binding.target_id}"
            )
        targets.append(target)
    return tuple(tasks), tuple(targets)


def _pending_projection(environment: EnvironmentManifest) -> EnvironmentManifest:
    payload = environment.model_dump(mode="python")
    payload.update(
        status="pending",
        verification_evidence=None,
        invalid_reason=None,
    )
    return EnvironmentManifest.model_validate(payload)


def _preflight_directory(
    artifact_root: str | Path,
    study_id: str,
) -> Path:
    return Path(artifact_root).expanduser() / study_id / _PREFLIGHT_BUNDLE_TRAJECTORY_ID


@dataclass(frozen=True)
class _RawEvidenceResumeWorker:
    """Fail-closed worker facade used while replaying already-ready evidence."""

    matrix_worker_binding: MatrixWorkerBinding

    def execute(self, job: WorkerJob) -> WorkerResult:
        raise MatrixPreflightArtifactError(
            f"ready resume attempted a new worker job because raw evidence is missing: {job.job_id}"
        )


def _require_raw_protocol_artifacts(
    artifact_root: Path,
    contract: MatrixPreflightStudyContract,
) -> None:
    """Require every raw protocol slot before invoking resume-only producers."""

    study_by_kind = {
        "oracle": contract.oracle_gate_study_id,
        "baseline": contract.baseline_gate_study_id,
        "capability": contract.capability_probe_study_id,
        "launch": contract.launch_floor_study_id,
    }
    paths = [
        artifact_root / study_by_kind[protocol.kind] / protocol.artifact_id
        for protocol in contract.protocols
    ]
    paths.extend(
        (
            artifact_root / contract.capability_probe_study_id / "study-manifest",
            artifact_root / contract.launch_floor_study_id / "study-manifest",
        )
    )
    for final in paths:
        staging = final.with_name(f"{final.name}.incomplete")
        final_present = final.exists() or final.is_symlink()
        staging_present = staging.exists() or staging.is_symlink()
        if final_present == staging_present:
            state = "both final and staging exist" if final_present else "artifact is missing"
            raise MatrixPreflightArtifactError(
                f"ready raw preflight evidence {state}: {final}"
            )
        candidate = final if final_present else staging
        if candidate.is_symlink():
            raise MatrixPreflightArtifactError(
                f"ready raw preflight evidence cannot be a symbolic link: {candidate}"
            )


def _ready_resume(
    directory: Path,
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    assets: AssetManifest,
    contract: MatrixPreflightStudyContract,
    *,
    artifact_root: Path,
    asset_root: Path,
    tasks: tuple[TaskPackSpec, ...],
    targets: tuple[TargetStackSpec, ...],
    final_directory: Path | None = None,
) -> MatrixPreflightRunResult:
    try:
        bundle = load_preflight_bundle(directory, pinned, schedule)
    except MatrixPreflightError as error:
        raise MatrixPreflightArtifactError(
            f"sealed ready preflight bundle is invalid: {directory}"
        ) from error
    if (
        bundle.assets != assets
        or _pending_projection(bundle.environment) != contract.pending_environment
    ):
        raise MatrixPreflightArtifactError(
            "ready preflight bundle differs from the sealed run contract"
        )
    environment_record = _load_environment_record(
        directory.parent / _ENVIRONMENT_TRAJECTORY_ID,
        contract,
    )
    environment_result = derive_environment_probe(
        contract.pending_environment,
        environment_record.artifact,
    )
    if environment_result.environment != bundle.environment:
        raise MatrixPreflightArtifactError(
            "ready environment differs from its durable raw probe"
        )
    environment = environment_result.environment
    resume_worker = _RawEvidenceResumeWorker(contract.worker)
    _require_raw_protocol_artifacts(artifact_root, contract)
    try:
        oracle_gates = run_oracle_gates(
            resume_worker,
            tasks=tasks,
            targets=targets,
            root=artifact_root,
            study_id=contract.oracle_gate_study_id,
            timing=FORMAL_FLOOR_TIMING,
            asset_root=asset_root,
            device=environment.transport.device,
        )
        baseline_target = next(item for item in targets if item.id == contract.baseline_target_id)
        baseline_gates = run_baseline_gates(
            resume_worker,
            tasks=tasks,
            target=baseline_target,
            root=artifact_root,
            study_id=contract.baseline_gate_study_id,
            timing=FORMAL_FLOOR_TIMING,
            device=environment.transport.device,
        )
        all_gates = oracle_gates + baseline_gates
        gate = pinned.spec.gate
        assert gate is not None
        task_floors = derive_task_floor_records(
            all_gates,
            assets,
            targets,
            baseline_target_id=contract.baseline_target_id,
            competitive_factor=gate.metrics.competitive_latency_factor,
        )
        capability_inputs = resolve_capability_probe_inputs(
            assets,
            environment,
            task_floors,
            asset_root=asset_root,
        )
        capability_manifest = build_capability_probe_study_manifest(
            assets,
            environment,
            capability_inputs,
        )
        capability_records = run_capability_probe_study(
            resume_worker,
            artifact_root=artifact_root,
            manifest=capability_manifest,
            inputs=capability_inputs,
        )
        capability_evidence = derive_capability_canary_evidence(
            capability_manifest,
            capability_inputs,
            capability_records,
        )
        launch_input = resolve_launch_floor_probe_input(
            assets,
            environment,
            asset_root=asset_root,
        )
        launch_manifest = build_launch_floor_study_manifest(
            pinned,
            schedule,
            assets,
            environment,
            task_floors,
            all_gates,
            capability_manifest,
            capability_records,
            launch_input,
            baseline_target_id=contract.baseline_target_id,
            asset_root=asset_root,
        )
        launch_record = run_launch_floor_probe(
            resume_worker,
            artifact_root=artifact_root,
            manifest=launch_manifest,
            probe_input=launch_input,
        )
        launch_floor = derive_launch_floor_evidence(
            launch_manifest,
            launch_input,
            launch_record,
        )
    except (
        CapabilityProbeInfrastructureError,
        GateError,
        MatrixCapabilityEvidenceError,
        MatrixFloorEvidenceError,
        MatrixLaunchFloorEvidenceError,
    ) as error:
        raise MatrixPreflightArtifactError(
            f"ready raw preflight evidence is invalid: {error}"
        ) from error
    reconstructed_floor = FloorManifest(
        **_study_fields(pinned, schedule),
        status="valid" if launch_floor.status == "pass" else "invalid",
        asset_manifest_sha256=assets.sha256,
        environment_manifest_sha256=environment.sha256,
        timing=FORMAL_FLOOR_TIMING,
        tasks=task_floors,
        capability_canaries=capability_evidence,
        launch_floor=launch_floor,
    )
    if reconstructed_floor != bundle.floor:
        raise MatrixPreflightArtifactError(
            "ready floor differs from recomputed durable raw evidence"
        )
    context = _execution_context(assets, environment, reconstructed_floor)
    destination = directory if final_directory is None else final_directory
    if destination != directory:
        if destination.exists() or destination.is_symlink():
            raise MatrixPreflightArtifactError(
                "ready preflight bundle appeared during staging promotion"
            )
        try:
            os.replace(directory, destination)
        except OSError as error:
            raise MatrixPreflightArtifactError(
                "cannot promote sealed ready preflight staging bundle"
            ) from error
        directory = destination
    try:
        bundle = load_preflight_bundle(
            directory,
            pinned,
            schedule,
            execution_context=context,
        )
    except MatrixPreflightError as error:
        raise MatrixPreflightArtifactError(
            "promoted ready preflight bundle is invalid"
        ) from error
    return MatrixPreflightRunResult(
        contract=contract,
        environment_probe=environment_record.artifact,
        oracle_gates=oracle_gates,
        baseline_gates=baseline_gates,
        task_floors=task_floors,
        capability_manifest=capability_manifest,
        capability_records=capability_records,
        launch_manifest=launch_manifest,
        launch_record=launch_record,
        bundle=bundle,
        preflight_directory=directory,
        resumed_ready_bundle=True,
    )


def _execution_context(
    assets: AssetManifest,
    environment: EnvironmentManifest,
    floor: FloorManifest,
) -> MatrixExecutionContext:
    return MatrixExecutionContext(
        controller_revision=environment.controller_revision,
        worker_revision=environment.worker_revision,
        transport=environment.transport,
        asset_manifest_sha256=assets.sha256,
        floor_manifest_sha256=floor.sha256,
        environment_manifest_sha256=environment.sha256,
        cache_policy=environment.cache_policy,
        gpu_jobs_serial=environment.gpu_jobs_serial,
        generated_code_remote_only=environment.generated_code_remote_only,
        non_container_worker=environment.non_container_worker,
    )


def run_matrix_preflight(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    assets: AssetManifest,
    pending_environment: EnvironmentManifest,
    *,
    artifact_root: str | Path,
    asset_root: str | Path,
    baseline_target_id: str,
    live: bool,
    expected_max_worker_jobs_per_invocation: int,
    worker_factory: PreflightWorkerFactory = build_environment_probe_worker,
) -> MatrixPreflightRunResult:
    """Run or resume every preflight stage and emit a ready-only bundle."""

    root = Path(artifact_root).expanduser().resolve()
    local_assets = Path(asset_root).expanduser().resolve()
    contract = build_preflight_study_contract(
        pinned,
        schedule,
        assets,
        pending_environment,
        baseline_target_id=baseline_target_id,
    )
    validate_preflight_run_guards(
        contract,
        live=live,
        expected_max_worker_jobs_per_invocation=(
            expected_max_worker_jobs_per_invocation
        ),
    )
    tasks, targets = _resolve_registry_assets(assets, asset_root=local_assets)
    final_bundle = _preflight_directory(root, contract.study_id)
    staging_bundle = final_bundle.with_name(f"{final_bundle.name}.incomplete")
    contract_path = _contract_directory(root, contract.study_id)
    final_present = final_bundle.exists() or final_bundle.is_symlink()
    staging_present = staging_bundle.exists() or staging_bundle.is_symlink()
    if final_present and staging_present:
        raise MatrixPreflightArtifactError(
            "final and staging ready preflight bundles both exist"
        )
    ready_candidate = final_bundle if final_present else (
        staging_bundle if staging_present else None
    )
    if ready_candidate is not None and not (
        contract_path.exists() or contract_path.is_symlink()
    ):
        raise MatrixPreflightArtifactError(
            "ready preflight bundle exists without its pre-SSH contract"
        )
    if ready_candidate == staging_bundle:
        if staging_bundle.is_symlink() or not staging_bundle.is_dir():
            raise MatrixPreflightArtifactError(
                "ready preflight staging bundle is not a regular directory"
            )
        checksum = staging_bundle / "sha256sums.txt"
        if not checksum.exists() and not checksum.is_symlink():
            shutil.rmtree(staging_bundle)
            ready_candidate = None
    seal_preflight_study_contract(root, contract)
    if ready_candidate is not None:
        return _ready_resume(
            ready_candidate,
            pinned,
            schedule,
            assets,
            contract,
            artifact_root=root,
            asset_root=local_assets,
            tasks=tasks,
            targets=targets,
            final_directory=final_bundle,
        )

    worker = _CappedPreflightWorker(
        worker_factory(pending_environment),
        max_worker_jobs_per_invocation=(
            contract.max_worker_jobs_per_invocation
        ),
    )
    environment_result = run_or_resume_environment_probe(
        worker,
        artifact_root=root,
        contract=contract,
    )
    environment = environment_result.environment
    if environment.status != "verified":
        raise MatrixPreflightInvalidFloorError(
            environment.invalid_reason or "environment probe did not pass"
        )

    try:
        oracle_gates = run_oracle_gates(
            worker,
            tasks=tasks,
            targets=targets,
            root=root,
            study_id=contract.oracle_gate_study_id,
            timing=FORMAL_FLOOR_TIMING,
            asset_root=local_assets,
            device=environment.transport.device,
        )
        baseline_target = next(item for item in targets if item.id == baseline_target_id)
        baseline_gates = run_baseline_gates(
            worker,
            tasks=tasks,
            target=baseline_target,
            root=root,
            study_id=contract.baseline_gate_study_id,
            timing=FORMAL_FLOOR_TIMING,
            device=environment.transport.device,
        )
    except GateInfrastructureError as error:
        raise MatrixPreflightInfrastructureError(str(error)) from error
    except GateError as error:
        raise MatrixPreflightArtifactError(
            f"gate preflight artifact is invalid: {error}"
        ) from error

    all_gates = oracle_gates + baseline_gates
    gate = pinned.spec.gate
    assert gate is not None
    try:
        task_floors = derive_task_floor_records(
            all_gates,
            assets,
            targets,
            baseline_target_id=baseline_target_id,
            competitive_factor=gate.metrics.competitive_latency_factor,
        )
    except MatrixFloorEvidenceError as error:
        raise MatrixPreflightInvalidFloorError(f"task floor is invalid: {error}") from error

    try:
        capability_inputs = resolve_capability_probe_inputs(
            assets,
            environment,
            task_floors,
            asset_root=local_assets,
        )
        capability_manifest = build_capability_probe_study_manifest(
            assets,
            environment,
            capability_inputs,
        )
        capability_records = run_capability_probe_study(
            worker,
            artifact_root=root,
            manifest=capability_manifest,
            inputs=capability_inputs,
        )
        capability_evidence = derive_capability_canary_evidence(
            capability_manifest,
            capability_inputs,
            capability_records,
        )
    except CapabilityProbeInfrastructureError as error:
        raise MatrixPreflightInfrastructureError(str(error)) from error
    except MatrixCapabilityEvidenceError as error:
        raise MatrixPreflightArtifactError(
            f"capability preflight evidence is invalid: {error}"
        ) from error
    if any(item.status != "pass" for item in capability_evidence):
        failed = ", ".join(item.canary_id for item in capability_evidence if item.status == "fail")
        raise MatrixPreflightInvalidFloorError(f"capability canaries failed: {failed}")

    try:
        launch_input = resolve_launch_floor_probe_input(
            assets,
            environment,
            asset_root=local_assets,
        )
        launch_manifest = build_launch_floor_study_manifest(
            pinned,
            schedule,
            assets,
            environment,
            task_floors,
            all_gates,
            capability_manifest,
            capability_records,
            launch_input,
            baseline_target_id=baseline_target_id,
            asset_root=local_assets,
        )
        launch_record = run_launch_floor_probe(
            worker,
            artifact_root=root,
            manifest=launch_manifest,
            probe_input=launch_input,
        )
        launch_floor = derive_launch_floor_evidence(
            launch_manifest,
            launch_input,
            launch_record,
        )
    except LaunchFloorProbeInfrastructureError as error:
        raise MatrixPreflightInfrastructureError(str(error)) from error
    except MatrixLaunchFloorEvidenceError as error:
        raise MatrixPreflightArtifactError(
            f"launch-floor preflight evidence is invalid: {error}"
        ) from error
    floor = FloorManifest(
        **_study_fields(pinned, schedule),
        status="valid" if launch_floor.status == "pass" else "invalid",
        asset_manifest_sha256=assets.sha256,
        environment_manifest_sha256=environment.sha256,
        timing=FORMAL_FLOOR_TIMING,
        tasks=task_floors,
        capability_canaries=capability_evidence,
        launch_floor=launch_floor,
    )
    if floor.status != "valid":
        raise MatrixPreflightInvalidFloorError(
            launch_floor.failure_reason or "launch floor did not pass"
        )
    context = _execution_context(assets, environment, floor)
    try:
        directory = seal_preflight_bundle(
            root,
            pinned,
            schedule,
            assets=assets,
            floor=floor,
            environment=environment,
            execution_context=context,
            trajectory_id=_PREFLIGHT_BUNDLE_TRAJECTORY_ID,
        )
        bundle = load_preflight_bundle(
            directory,
            pinned,
            schedule,
            execution_context=context,
        )
    except MatrixPreflightError as error:
        raise MatrixPreflightArtifactError(
            f"cannot seal ready preflight bundle: {error}"
        ) from error
    return MatrixPreflightRunResult(
        contract=contract,
        environment_probe=environment_result.artifact,
        oracle_gates=oracle_gates,
        baseline_gates=baseline_gates,
        task_floors=task_floors,
        capability_manifest=capability_manifest,
        capability_records=capability_records,
        launch_manifest=launch_manifest,
        launch_record=launch_record,
        bundle=bundle,
        preflight_directory=directory,
        resumed_ready_bundle=False,
    )
