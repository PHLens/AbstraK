"""Run and derive hash-bound capability-canary preflight evidence."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from abstrak.canary.artifacts import (
    TrajectoryArtifactError,
    TrajectoryStore,
    verify_trajectory,
)
from abstrak.canary.capabilities import get_capability_pack
from abstrak.canary.capability_assets import (
    CapabilityCanarySpec,
    get_capability_canary,
    load_capability_canary,
)
from abstrak.canary.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    CanaryModel,
    TargetStackSpec,
    TaskPackSpec,
    TimingSpec,
    WorkerResult,
)
from abstrak.canary.gates import GateRecord
from abstrak.canary.loop import WorkerExecutor
from abstrak.canary.matrix_floor_evidence import (
    MatrixFloorEvidenceError,
    validate_gate_summary,
)
from abstrak.canary.matrix_preflight import (
    FORMAL_FLOOR_TIMING,
    AssetManifest,
    CapabilityCanaryEvidence,
    CapabilityTargetEvidence,
    EnvironmentManifest,
    StudyBoundModel,
    TargetAssetBinding,
    TargetCodegenEvidence,
    TaskAssetBinding,
    TaskFloorRecord,
)
from abstrak.canary.matrix_runner import MatrixWorkerBinding
from abstrak.canary.target_adapters import validate_target_source
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import (
    CAPABILITY_GATE_ASSET_ROOT,
    get_task_pack,
)
from abstrak.canary.timing import (
    TimingProtocolSummary,
    is_proven_job_scoped_resource_failure,
    run_timing_protocol,
)
from abstrak.providers.contracts import sha256_json

_SHA256 = re.compile(SHA256_PATTERN)
_ARTIFACT_DIRECTORIES = frozenset({"events", "turns", "candidates", "sealed"})
_STATIC_METADATA_KEYS = (
    "capability_pack_id",
    "capability_pack_sha256",
    "used_capabilities",
    "minimum_pack_id",
    "minimum_pack_bitmask",
)


class MatrixCapabilityEvidenceError(ValueError):
    """Raised when capability evidence differs from frozen or raw inputs."""


class CapabilityProbeInfrastructureError(RuntimeError):
    """Raised when a probe has no terminal scientific result."""

    def __init__(self, artifact_id: str, error: str) -> None:
        super().__init__(f"capability probe infrastructure failure: {artifact_id}: {error}")
        self.artifact_id = artifact_id
        self.error = error


class CapabilityStaticIdentity(CanaryModel):
    """Controller-recomputed static capability result for one source and target."""

    schema_version: Literal["abstrak-capability-static-identity.v1"] = (
        "abstrak-capability-static-identity.v1"
    )
    capability_pack_id: str = Field(pattern=IDENTIFIER_PATTERN)
    capability_pack_sha256: str = Field(pattern=SHA256_PATTERN)
    used_capabilities: tuple[str, ...] = Field(min_length=1)
    minimum_pack_id: str = Field(pattern=IDENTIFIER_PATTERN)
    minimum_pack_bitmask: int = Field(ge=1)
    static_warnings: tuple[str, ...] = ()

    @field_validator("used_capabilities")
    @classmethod
    def capability_names_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if (
            any(not value or value.strip() != value for value in values)
            or tuple(sorted(set(values))) != values
        ):
            raise ValueError("used capabilities must be sorted unique normalized names")
        return values

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class CapabilityControlIdentity(CanaryModel):
    """Same-task, same-target B-expert codegen used as the dynamic control."""

    schema_version: Literal["abstrak-capability-control-identity.v1"] = (
        "abstrak-capability-control-identity.v1"
    )
    expert_source_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    generated_code_sha256: str = Field(pattern=SHA256_PATTERN)

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class CapabilityProbeIdentity(StudyBoundModel):
    """Complete frozen identity for one canary and required target validator."""

    schema_version: Literal["abstrak-matrix-capability-probe-identity.v1"] = (
        "abstrak-matrix-capability-probe-identity.v1"
    )
    artifact_id: str = Field(pattern=IDENTIFIER_PATTERN)
    asset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    worker: MatrixWorkerBinding
    canary_id: str = Field(pattern=IDENTIFIER_PATTERN)
    canary_spec_sha256: str = Field(pattern=SHA256_PATTERN)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_pack_sha256: str = Field(pattern=SHA256_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_stack_sha256: str = Field(pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    minimum_pack_id: str = Field(pattern=IDENTIFIER_PATTERN)
    minimum_pack_bitmask: int = Field(ge=1)
    static: CapabilityStaticIdentity
    control: CapabilityControlIdentity
    timing: TimingSpec
    device: str = Field(pattern=r"^cuda:[0-9]+$")

    @model_validator(mode="after")
    def nested_runtime_and_capability_identity_match(self) -> CapabilityProbeIdentity:
        expected_artifact_id = capability_probe_artifact_id(
            self.canary_id,
            self.target_id,
        )
        if self.artifact_id != expected_artifact_id:
            raise ValueError("capability probe artifact ID differs from canary and target")
        if self.timing != FORMAL_FLOOR_TIMING:
            raise ValueError("capability probes must use FORMAL_FLOOR_TIMING")
        if self.worker.transport.device != self.device:
            raise ValueError("capability probe device differs from worker transport")
        if (
            self.static.minimum_pack_id != self.minimum_pack_id
            or self.static.minimum_pack_bitmask != self.minimum_pack_bitmask
        ):
            raise ValueError("static validation differs from the canary minimum pack")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class CapabilityProbeStudyManifest(StudyBoundModel):
    """Ordered capability-probe set sealed before any remote execution."""

    schema_version: Literal["abstrak-matrix-capability-probe-study.v1"] = (
        "abstrak-matrix-capability-probe-study.v1"
    )
    probe_study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    asset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    worker: MatrixWorkerBinding
    timing: TimingSpec
    probe_count: int = Field(ge=1)
    probes: tuple[CapabilityProbeIdentity, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def probes_match_study(self) -> CapabilityProbeStudyManifest:
        if self.probe_study_id != capability_probe_study_id(self.study_id):
            raise ValueError("capability probe study ID differs from the matrix study")
        if self.timing != FORMAL_FLOOR_TIMING:
            raise ValueError("capability probe study must use FORMAL_FLOOR_TIMING")
        if self.probe_count != len(self.probes):
            raise ValueError("capability probe count differs from the ordered probe set")
        artifact_ids = tuple(item.artifact_id for item in self.probes)
        identities = tuple(item.sha256 for item in self.probes)
        if len(artifact_ids) != len(set(artifact_ids)) or len(identities) != len(set(identities)):
            raise ValueError("capability probe identities must be unique")
        expected = (
            self.study_binding,
            self.asset_manifest_sha256,
            self.environment_manifest_sha256,
            self.worker,
            self.timing,
        )
        if any(
            (
                item.study_binding,
                item.asset_manifest_sha256,
                item.environment_manifest_sha256,
                item.worker,
                item.timing,
            )
            != expected
            for item in self.probes
        ):
            raise ValueError("capability probes have different study or runtime bindings")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class CapabilityProbeRecord(CanaryModel):
    """One sealed raw capability run from which target evidence is recomputed."""

    schema_version: Literal["abstrak-matrix-capability-probe-record.v1"] = (
        "abstrak-matrix-capability-probe-record.v1"
    )
    probe_study_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    identity: CapabilityProbeIdentity
    artifact_directory: str = Field(min_length=1)
    summary: TimingProtocolSummary

    @property
    def sha256(self) -> str:
        return sha256_json(self)


@dataclass(frozen=True)
class CapabilityProbeInput:
    """Verified local source and registry objects for one frozen probe identity."""

    identity: CapabilityProbeIdentity
    source: str
    task: TaskPackSpec
    target: TargetStackSpec
    task_asset: TaskAssetBinding
    target_asset: TargetAssetBinding
    control_evidence: TargetCodegenEvidence

    def __post_init__(self) -> None:
        identity = self.identity
        if (
            hashlib.sha256(self.source.encode("utf-8")).hexdigest() != identity.source_sha256
            or self.task.id != identity.task_id
            or sha256_json(self.task) != identity.task_pack_sha256
            or self.target.id != identity.target_id
            or sha256_json(self.target) != identity.target_stack_sha256
            or self.task_asset.task_id != identity.task_id
            or self.task_asset.task_pack_sha256 != identity.task_pack_sha256
            or self.task_asset.expert_source_sha256 != identity.control.expert_source_sha256
            or self.target_asset.target_id != identity.target_id
            or self.target_asset.target_stack_sha256 != identity.target_stack_sha256
            or self.control_evidence.status != "pass"
            or self.control_evidence.task_id != identity.task_id
            or self.control_evidence.task_pack_sha256 != identity.task_pack_sha256
            or self.control_evidence.target_id != identity.target_id
            or self.control_evidence.target_stack_sha256 != identity.target_stack_sha256
            or self.control_evidence.expert_source_sha256 != identity.control.expert_source_sha256
            or self.control_evidence.artifact_sha256 != identity.control.artifact_sha256
            or self.control_evidence.generated_code_sha256 != identity.control.generated_code_sha256
        ):
            raise ValueError("capability probe runtime input differs from its identity")


def capability_probe_artifact_id(canary_id: str, target_id: str) -> str:
    """Return the stable artifact ID for one canary/target pair."""

    return f"capability-{canary_id}-{target_id}"


def capability_probe_study_id(study_id: str) -> str:
    """Return the dedicated artifact study ID for capability preflight."""

    return f"{study_id}-capability-probes"


def _static_identity(
    source: str,
    target: TargetStackSpec,
    canary: CapabilityCanarySpec,
) -> CapabilityStaticIdentity:
    result = validate_target_source(source, target)
    if not result.valid or result.errors:
        raise MatrixCapabilityEvidenceError(
            f"frozen canary is not valid for required target: {canary.id}/{target.id}"
        )
    metadata = dict(result.metadata)
    missing = tuple(key for key in _STATIC_METADATA_KEYS if key not in metadata)
    if missing:
        raise MatrixCapabilityEvidenceError(
            "capability validator omitted static metadata: " + ", ".join(missing)
        )
    used = metadata["used_capabilities"]
    if not isinstance(used, tuple | list) or any(not isinstance(item, str) for item in used):
        raise MatrixCapabilityEvidenceError(
            "capability validator returned invalid used capabilities"
        )
    bitmask = metadata["minimum_pack_bitmask"]
    if isinstance(bitmask, bool) or not isinstance(bitmask, int):
        raise MatrixCapabilityEvidenceError("capability validator returned an invalid bitmask")
    static = CapabilityStaticIdentity(
        capability_pack_id=str(metadata["capability_pack_id"]),
        capability_pack_sha256=str(metadata["capability_pack_sha256"]),
        used_capabilities=tuple(used),
        minimum_pack_id=str(metadata["minimum_pack_id"]),
        minimum_pack_bitmask=bitmask,
        static_warnings=tuple(f"{issue.code}: {issue.message}" for issue in result.warnings),
    )
    minimum = get_capability_pack(canary.minimum_pack_id)
    if (
        static.minimum_pack_id != canary.minimum_pack_id
        or static.minimum_pack_bitmask != minimum.bitmask
    ):
        raise MatrixCapabilityEvidenceError(
            f"canary static minimum pack differs from its spec: {canary.id}/{target.id}"
        )
    return static


def _require_verified_environment(
    assets: AssetManifest,
    environment: EnvironmentManifest,
) -> MatrixWorkerBinding:
    if environment.status != "verified" or environment.verification_evidence is None:
        raise MatrixCapabilityEvidenceError("capability probes require a verified environment")
    if assets.study_binding != environment.study_binding:
        raise MatrixCapabilityEvidenceError(
            "capability assets and environment have different study identities"
        )
    return MatrixWorkerBinding(
        worker_revision=environment.worker_revision,
        transport=environment.transport,
    )


def resolve_capability_probe_inputs(
    assets: AssetManifest,
    environment: EnvironmentManifest,
    task_floors: Iterable[TaskFloorRecord],
    *,
    asset_root: str | Path = CAPABILITY_GATE_ASSET_ROOT,
) -> tuple[CapabilityProbeInput, ...]:
    """Resolve frozen assets and B-expert controls into ordered probe inputs."""

    worker = _require_verified_environment(assets, environment)
    floor_values = tuple(task_floors)
    if tuple(item.task_id for item in floor_values) != tuple(item.task_id for item in assets.tasks):
        raise MatrixCapabilityEvidenceError(
            "task floors do not exactly cover capability assets in task order"
        )
    floors = {item.task_id: item for item in floor_values}
    task_assets = {item.task_id: item for item in assets.tasks}
    target_assets = {item.target_id: item for item in assets.targets}
    inputs: list[CapabilityProbeInput] = []
    for binding in assets.canaries:
        canary = get_capability_canary(binding.canary_id)
        source = load_capability_canary(binding.canary_id, asset_root=asset_root)
        if (
            canary.sha256 != binding.canary_spec_sha256
            or canary.task_id != binding.task_id
            or canary.source_sha256 != binding.source_sha256
            or canary.minimum_pack_id != binding.minimum_pack_id
            or get_capability_pack(canary.minimum_pack_id).bitmask
            != binding.minimum_pack_bitmask
            or canary.required_target_ids != binding.required_target_ids
        ):
            raise MatrixCapabilityEvidenceError(
                f"capability canary registry differs from asset manifest: {binding.canary_id}"
            )
        task_asset = task_assets[canary.task_id]
        task = get_task_pack(canary.task_id)
        if sha256_json(task) != task_asset.task_pack_sha256:
            raise MatrixCapabilityEvidenceError(
                f"capability canary task differs from asset manifest: {canary.id}"
            )
        floor = floors[canary.task_id]
        if floor.status != "valid" or floor.verified_evidence is None:
            raise MatrixCapabilityEvidenceError(
                f"capability control requires a valid task floor: {canary.task_id}"
            )
        if floor.expert_source_sha256 != task_asset.expert_source_sha256:
            raise MatrixCapabilityEvidenceError(
                f"capability control source differs from task assets: {canary.task_id}"
            )
        controls = {item.target_id: item for item in floor.verified_evidence.target_codegen}
        for target_id in canary.required_target_ids:
            target_asset = target_assets[target_id]
            target = get_target_stack(target_id)
            if sha256_json(target) != target_asset.target_stack_sha256:
                raise MatrixCapabilityEvidenceError(
                    f"capability target differs from asset manifest: {target_id}"
                )
            try:
                control_evidence = controls[target_id]
            except KeyError:
                raise MatrixCapabilityEvidenceError(
                    f"task floor is missing same-target control codegen: {canary.id}/{target_id}"
                ) from None
            if (
                control_evidence.status != "pass"
                or control_evidence.task_id != canary.task_id
                or control_evidence.task_pack_sha256 != task_asset.task_pack_sha256
                or control_evidence.target_id != target_id
                or control_evidence.target_stack_sha256 != target_asset.target_stack_sha256
                or control_evidence.expert_source_sha256 != task_asset.expert_source_sha256
                or control_evidence.generated_code_sha256 is None
            ):
                raise MatrixCapabilityEvidenceError(
                    f"task floor has invalid same-target control codegen: {canary.id}/{target_id}"
                )
            static = _static_identity(source, target, canary)
            minimum = get_capability_pack(canary.minimum_pack_id)
            identity = CapabilityProbeIdentity(
                study_id=assets.study_id,
                raw_study_sha256=assets.raw_study_sha256,
                spec_sha256=assets.spec_sha256,
                schedule_sha256=assets.schedule_sha256,
                artifact_id=capability_probe_artifact_id(canary.id, target_id),
                asset_manifest_sha256=assets.sha256,
                environment_manifest_sha256=environment.sha256,
                worker=worker,
                canary_id=canary.id,
                canary_spec_sha256=canary.sha256,
                task_id=canary.task_id,
                task_pack_sha256=task_asset.task_pack_sha256,
                target_id=target_id,
                target_stack_sha256=target_asset.target_stack_sha256,
                source_sha256=canary.source_sha256,
                minimum_pack_id=minimum.id,
                minimum_pack_bitmask=minimum.bitmask,
                static=static,
                control=CapabilityControlIdentity(
                    expert_source_sha256=task_asset.expert_source_sha256,
                    artifact_sha256=control_evidence.artifact_sha256,
                    generated_code_sha256=control_evidence.generated_code_sha256,
                ),
                timing=FORMAL_FLOOR_TIMING,
                device=environment.transport.device,
            )
            inputs.append(
                CapabilityProbeInput(
                    identity=identity,
                    source=source,
                    task=task,
                    target=target,
                    task_asset=task_asset,
                    target_asset=target_asset,
                    control_evidence=control_evidence,
                )
            )
    return tuple(inputs)


def build_capability_probe_study_manifest(
    assets: AssetManifest,
    environment: EnvironmentManifest,
    inputs: Iterable[CapabilityProbeInput],
) -> CapabilityProbeStudyManifest:
    """Build the ordered study-wide identity sealed before probe execution."""

    worker = _require_verified_environment(assets, environment)
    values = tuple(inputs)
    if not values:
        raise MatrixCapabilityEvidenceError("capability probe study requires probes")
    return CapabilityProbeStudyManifest(
        study_id=assets.study_id,
        raw_study_sha256=assets.raw_study_sha256,
        spec_sha256=assets.spec_sha256,
        schedule_sha256=assets.schedule_sha256,
        probe_study_id=capability_probe_study_id(assets.study_id),
        asset_manifest_sha256=assets.sha256,
        environment_manifest_sha256=environment.sha256,
        worker=worker,
        timing=FORMAL_FLOOR_TIMING,
        probe_count=len(values),
        probes=tuple(item.identity for item in values),
    )


def _verify_artifact_shape(
    directory: Path,
    *,
    record: bool,
) -> None:
    if directory.is_symlink():
        raise MatrixCapabilityEvidenceError("capability artifact cannot be a symbolic link")
    entries = tuple(directory.rglob("*"))
    if any(entry.is_symlink() for entry in entries):
        raise MatrixCapabilityEvidenceError("capability artifact cannot contain symbolic links")
    actual_files = {item.relative_to(directory).as_posix() for item in entries if item.is_file()}
    actual_directories = {
        item.relative_to(directory).as_posix() for item in entries if item.is_dir()
    }
    expected_files = {"run-manifest.json", "sha256sums.txt"}
    if record:
        expected_files.add("capability-record.json")
    if actual_files != expected_files or actual_directories != _ARTIFACT_DIRECTORIES:
        raise MatrixCapabilityEvidenceError(
            "capability artifact has unexpected files or directories"
        )


def _remove_unsealed_staging(directory: Path) -> bool:
    if not directory.exists() and not directory.is_symlink():
        return False
    if directory.is_symlink():
        raise MatrixCapabilityEvidenceError("capability staging artifact cannot be a symbolic link")
    if not directory.is_dir():
        raise MatrixCapabilityEvidenceError(
            "capability staging artifact is not a regular directory"
        )
    checksum = directory / "sha256sums.txt"
    if not checksum.exists() and not checksum.is_symlink():
        shutil.rmtree(directory)
        return True
    try:
        verify_trajectory(directory)
    except (OSError, TrajectoryArtifactError) as error:
        raise MatrixCapabilityEvidenceError(
            "checksum-bearing capability staging artifact is invalid"
        ) from error
    return False


def _load_study_manifest(
    directory: Path,
    expected: CapabilityProbeStudyManifest,
) -> CapabilityProbeStudyManifest:
    try:
        _verify_artifact_shape(directory, record=False)
        verify_trajectory(directory)
        actual = CapabilityProbeStudyManifest.model_validate_json(
            (directory / "run-manifest.json").read_text(encoding="utf-8")
        )
    except MatrixCapabilityEvidenceError:
        raise
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise MatrixCapabilityEvidenceError(
            f"capability probe study manifest is invalid: {directory}"
        ) from error
    if actual != expected:
        raise MatrixCapabilityEvidenceError(
            "capability probe study manifest differs from frozen inputs"
        )
    return actual


def seal_capability_probe_study_manifest(
    artifact_root: str | Path,
    manifest: CapabilityProbeStudyManifest,
) -> Path:
    """Atomically seal or exactly resume the study-wide probe manifest."""

    root = Path(artifact_root).expanduser()
    final = root / manifest.probe_study_id / "study-manifest"
    staging = final.with_name("study-manifest.incomplete")
    if final.exists() and staging.exists():
        raise MatrixCapabilityEvidenceError(
            "final and staging capability study manifests both exist"
        )
    if final.exists() or final.is_symlink():
        _load_study_manifest(final, manifest)
        return final
    if staging.exists() or staging.is_symlink():
        removed = _remove_unsealed_staging(staging)
        if not removed:
            _load_study_manifest(staging, manifest)
            os.replace(staging, final)
            return final
    try:
        store = TrajectoryStore.create(
            root,
            manifest.probe_study_id,
            "study-manifest.incomplete",
        )
        store.write_json("run-manifest.json", manifest)
        store.seal()
        if final.exists() or final.is_symlink():
            raise MatrixCapabilityEvidenceError(
                "capability study manifest appeared during atomic staging"
            )
        os.replace(store.run_directory, final)
    except MatrixCapabilityEvidenceError:
        raise
    except (OSError, TrajectoryArtifactError) as error:
        raise MatrixCapabilityEvidenceError(
            "cannot seal capability probe study manifest"
        ) from error
    _load_study_manifest(final, manifest)
    return final


def _validate_static_result(
    result: WorkerResult,
    expected: CapabilityStaticIdentity,
) -> None:
    if result.static_errors or tuple(result.static_warnings) != expected.static_warnings:
        raise MatrixCapabilityEvidenceError(
            "capability worker static diagnostics differ from controller validation"
        )
    metadata = result.metadata
    missing = tuple(key for key in _STATIC_METADATA_KEYS if key not in metadata)
    if missing:
        raise MatrixCapabilityEvidenceError(
            "capability worker omitted static metadata: " + ", ".join(missing)
        )
    used = metadata["used_capabilities"]
    if not isinstance(used, tuple | list) or any(not isinstance(item, str) for item in used):
        raise MatrixCapabilityEvidenceError("worker returned invalid used capabilities")
    bitmask = metadata["minimum_pack_bitmask"]
    if isinstance(bitmask, bool) or not isinstance(bitmask, int):
        raise MatrixCapabilityEvidenceError("worker returned an invalid minimum-pack bitmask")
    actual = (
        metadata["capability_pack_id"],
        metadata["capability_pack_sha256"],
        tuple(used),
        metadata["minimum_pack_id"],
        bitmask,
    )
    expected_values = (
        expected.capability_pack_id,
        expected.capability_pack_sha256,
        expected.used_capabilities,
        expected.minimum_pack_id,
        expected.minimum_pack_bitmask,
    )
    if actual != expected_values:
        raise MatrixCapabilityEvidenceError(
            "capability worker static metadata differs from controller recomputation"
        )


def _generated_code_sha256(
    results: tuple[WorkerResult, ...],
) -> str | None:
    values: list[tuple[str, int]] = []
    for result in results:
        digest = result.metadata.get("generated_code_sha256")
        capture = result.metadata.get("generated_code_capture")
        size = result.metadata.get("generated_code_size_bytes")
        present = digest is not None or capture is not None or size is not None
        if result.compiled and not present:
            raise MatrixCapabilityEvidenceError(
                "compiled capability result is missing generated-code evidence"
            )
        if not present:
            continue
        if (
            not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or capture != "tilelang.get_kernel_source.v1"
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise MatrixCapabilityEvidenceError(
                "capability result has invalid generated-code evidence"
            )
        values.append((digest, size))
    if not values:
        return None
    if len(set(values)) != 1:
        raise MatrixCapabilityEvidenceError(
            "capability generated code changed across clean worker processes"
        )
    return values[0][0]


def _expected_gate_task(input_value: CapabilityProbeInput) -> TaskAssetBinding:
    return input_value.task_asset.model_copy(
        update={"expert_source_sha256": input_value.identity.source_sha256}
    )


def _inspect_record(
    record: CapabilityProbeRecord,
    input_value: CapabilityProbeInput,
    manifest: CapabilityProbeStudyManifest,
) -> str | None:
    identity = input_value.identity
    canary = get_capability_canary(identity.canary_id)
    recomputed_static = _static_identity(input_value.source, input_value.target, canary)
    if (
        canary.sha256 != identity.canary_spec_sha256
        or canary.task_id != identity.task_id
        or canary.source_sha256 != identity.source_sha256
        or identity.target_id not in canary.required_target_ids
        or recomputed_static != identity.static
    ):
        raise MatrixCapabilityEvidenceError(
            f"capability identity differs from controller recomputation: {identity.artifact_id}"
        )
    if (
        record.probe_study_manifest_sha256 != manifest.sha256
        or record.identity != identity
        or record.summary.job_prefix != identity.artifact_id
        or record.summary.task_id != identity.task_id
        or record.summary.target_id != identity.target_id
        or record.summary.candidate_sha256 != identity.source_sha256
        or record.summary.job_kind != "oracle"
        or record.summary.device != identity.device
        or record.summary.timing != FORMAL_FLOOR_TIMING
    ):
        raise MatrixCapabilityEvidenceError(
            f"capability record differs from frozen identity: {identity.artifact_id}"
        )
    synthetic_gate = GateRecord(
        kind="oracle",
        task_id=identity.task_id,
        target_id=identity.target_id,
        source_sha256=identity.source_sha256,
        artifact_directory=record.artifact_directory,
        summary=record.summary,
    )
    try:
        validate_gate_summary(
            synthetic_gate,
            task=_expected_gate_task(input_value),
            target=input_value.target_asset,
        )
    except MatrixFloorEvidenceError as error:
        raise MatrixCapabilityEvidenceError(
            f"capability raw timing or correctness is invalid: {identity.artifact_id}: {error}"
        ) from error
    expected_process_timing = FORMAL_FLOOR_TIMING.model_copy(update={"repetitions": 1})
    for attempt in record.summary.attempts:
        for repetition, job in enumerate(attempt.jobs, start=1):
            if (
                job.job_id != f"{identity.artifact_id}-timing-a{attempt.attempt}-p{repetition}"
                or job.kind != "oracle"
                or job.task != input_value.task
                or job.target != input_value.target
                or tuple(job.case_ids) != tuple(case.id for case in input_value.task.sealed_cases)
                or job.candidate_source != input_value.source
                or job.candidate_sha256 != identity.source_sha256
                or job.timing != expected_process_timing
                or job.device != identity.device
            ):
                raise MatrixCapabilityEvidenceError(
                    f"capability worker job differs from frozen input: {job.job_id}"
                )
    for result in record.summary.results:
        if not is_proven_job_scoped_resource_failure(result):
            _validate_static_result(result, identity.static)
    generated = _generated_code_sha256(record.summary.results)
    if record.summary.status in {"stable", "unstable"} and generated is None:
        raise MatrixCapabilityEvidenceError(
            f"passing capability run has no generated code: {identity.artifact_id}"
        )
    if record.summary.status == "worker_failure":
        raise CapabilityProbeInfrastructureError(
            identity.artifact_id,
            record.summary.error or "worker failure",
        )
    return generated


def _load_probe_record(
    directory: Path,
    input_value: CapabilityProbeInput,
    manifest: CapabilityProbeStudyManifest,
) -> CapabilityProbeRecord:
    try:
        _verify_artifact_shape(directory, record=True)
        verify_trajectory(directory)
        persisted_manifest = CapabilityProbeIdentity.model_validate_json(
            (directory / "run-manifest.json").read_text(encoding="utf-8")
        )
        record = CapabilityProbeRecord.model_validate_json(
            (directory / "capability-record.json").read_text(encoding="utf-8")
        )
    except MatrixCapabilityEvidenceError:
        raise
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise MatrixCapabilityEvidenceError(
            f"sealed capability probe artifact is invalid: {directory}"
        ) from error
    expected_path = (
        directory.parent / input_value.identity.artifact_id
        if directory.name.endswith(".incomplete")
        else directory
    )
    try:
        declared = Path(record.artifact_directory).expanduser().resolve(strict=False)
        expected = expected_path.expanduser().resolve(strict=False)
    except OSError as error:
        raise MatrixCapabilityEvidenceError(
            "cannot resolve capability artifact directory identity"
        ) from error
    if persisted_manifest != input_value.identity or record.identity != persisted_manifest:
        raise MatrixCapabilityEvidenceError("sealed capability probe differs from its run manifest")
    if declared != expected:
        raise MatrixCapabilityEvidenceError(
            "capability record artifact_directory differs from its final directory"
        )
    _inspect_record(record, input_value, manifest)
    return record


def capability_probe_artifact_sha256(
    record: CapabilityProbeRecord,
) -> str:
    """Verify a sealed probe and hash its checksum-manifest bytes."""

    directory = Path(record.artifact_directory).expanduser()
    try:
        _verify_artifact_shape(directory, record=True)
        verify_trajectory(directory)
        persisted = CapabilityProbeRecord.model_validate_json(
            (directory / "capability-record.json").read_text(encoding="utf-8")
        )
        if persisted != record:
            raise MatrixCapabilityEvidenceError(
                "supplied capability record differs from its sealed artifact"
            )
        if directory.resolve(strict=True) != Path(
            persisted.artifact_directory
        ).expanduser().resolve(strict=True):
            raise MatrixCapabilityEvidenceError(
                "sealed capability record does not identify its containing directory"
            )
        checksum = (directory / "sha256sums.txt").read_bytes()
    except MatrixCapabilityEvidenceError:
        raise
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise MatrixCapabilityEvidenceError(
            f"sealed capability probe artifact is invalid: {directory}"
        ) from error
    return hashlib.sha256(checksum).hexdigest()


def _verify_probe_study_directory(
    artifact_root: str | Path,
    manifest: CapabilityProbeStudyManifest,
) -> None:
    root = Path(artifact_root).expanduser() / manifest.probe_study_id
    _load_study_manifest(root / "study-manifest", manifest)
    allowed = {"study-manifest"}
    for identity in manifest.probes:
        allowed.add(identity.artifact_id)
        allowed.add(f"{identity.artifact_id}.incomplete")
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise MatrixCapabilityEvidenceError(
            "cannot inspect capability probe study directory"
        ) from error
    if any(entry.is_symlink() for entry in entries):
        raise MatrixCapabilityEvidenceError("capability probe study cannot contain symbolic links")
    unexpected = sorted(entry.name for entry in entries if entry.name not in allowed)
    if unexpected:
        raise MatrixCapabilityEvidenceError(
            f"capability probe study contains unexpected artifacts: {unexpected}"
        )


def _run_or_resume_probe(
    worker: WorkerExecutor,
    *,
    artifact_root: str | Path,
    manifest: CapabilityProbeStudyManifest,
    input_value: CapabilityProbeInput,
) -> tuple[CapabilityProbeRecord, bool]:
    identity = input_value.identity
    study_root = Path(artifact_root).expanduser() / manifest.probe_study_id
    final = study_root / identity.artifact_id
    staging = final.with_name(f"{identity.artifact_id}.incomplete")
    if final.exists() and staging.exists():
        raise MatrixCapabilityEvidenceError(
            f"final and staging capability artifacts both exist: {identity.artifact_id}"
        )
    if final.exists() or final.is_symlink():
        return _load_probe_record(final, input_value, manifest), True
    if staging.exists() or staging.is_symlink():
        removed = _remove_unsealed_staging(staging)
        if not removed:
            record = _load_probe_record(staging, input_value, manifest)
            if final.exists() or final.is_symlink():
                raise MatrixCapabilityEvidenceError(
                    f"capability artifact appeared during staging resume: {identity.artifact_id}"
                )
            os.replace(staging, final)
            return record, True

    summary = run_timing_protocol(
        worker,
        task=input_value.task,
        target=input_value.target,
        source=input_value.source,
        job_prefix=identity.artifact_id,
        device=identity.device,
        timing=FORMAL_FLOOR_TIMING,
        job_kind="oracle",
    )
    if summary.status == "worker_failure":
        raise CapabilityProbeInfrastructureError(
            identity.artifact_id,
            summary.error or "worker failure",
        )
    final_directory = str(final)
    record = CapabilityProbeRecord(
        probe_study_manifest_sha256=manifest.sha256,
        identity=identity,
        artifact_directory=final_directory,
        summary=summary,
    )
    _inspect_record(record, input_value, manifest)
    try:
        store = TrajectoryStore.create(
            artifact_root,
            manifest.probe_study_id,
            f"{identity.artifact_id}.incomplete",
        )
        store.write_json("run-manifest.json", identity)
        store.write_json("capability-record.json", record)
        store.seal()
        if final.exists() or final.is_symlink():
            raise MatrixCapabilityEvidenceError(
                f"capability artifact appeared during atomic staging: {identity.artifact_id}"
            )
        os.replace(store.run_directory, final)
    except MatrixCapabilityEvidenceError:
        raise
    except (OSError, TrajectoryArtifactError) as error:
        raise MatrixCapabilityEvidenceError(
            f"cannot seal capability probe artifact: {identity.artifact_id}"
        ) from error
    return _load_probe_record(final, input_value, manifest), False


def run_capability_probe_study(
    worker: WorkerExecutor,
    *,
    artifact_root: str | Path,
    manifest: CapabilityProbeStudyManifest,
    inputs: Iterable[CapabilityProbeInput],
) -> tuple[CapabilityProbeRecord, ...]:
    """Run or exactly resume every ordered capability probe serially."""

    values = tuple(inputs)
    if tuple(item.identity for item in values) != manifest.probes:
        raise MatrixCapabilityEvidenceError(
            "runtime capability inputs differ from the sealed study manifest"
        )
    if getattr(worker, "matrix_worker_binding", None) != manifest.worker:
        raise MatrixCapabilityEvidenceError(
            "capability probe worker differs from the verified environment"
        )
    seal_capability_probe_study_manifest(artifact_root, manifest)
    _verify_probe_study_directory(artifact_root, manifest)
    records: list[CapabilityProbeRecord] = []
    for input_value in values:
        record, _ = _run_or_resume_probe(
            worker,
            artifact_root=artifact_root,
            manifest=manifest,
            input_value=input_value,
        )
        records.append(record)
    _verify_probe_study_directory(artifact_root, manifest)
    return tuple(records)


def _target_evidence(
    record: CapabilityProbeRecord,
    input_value: CapabilityProbeInput,
    manifest: CapabilityProbeStudyManifest,
) -> CapabilityTargetEvidence:
    generated = _inspect_record(record, input_value, manifest)
    results = record.summary.results
    compiled = bool(results) and all(result.compiled for result in results)
    correct = bool(results) and all(
        result.status == "completed" and result.correct for result in results
    )
    distinct = (
        generated is not None and generated != input_value.identity.control.generated_code_sha256
    )
    passed = compiled and correct and distinct
    if passed:
        failure_reason = None
    elif not compiled:
        failure_reason = record.summary.error or "capability source did not compile"
    elif not correct:
        failure_reason = record.summary.error or "capability source failed sealed correctness"
    else:
        failure_reason = "capability source generated the same code as its B-expert control"
    return CapabilityTargetEvidence(
        artifact_sha256=capability_probe_artifact_sha256(record),
        target_id=input_value.identity.target_id,
        target_stack_sha256=input_value.identity.target_stack_sha256,
        minimum_pack_id=input_value.identity.minimum_pack_id,
        minimum_pack_bitmask=input_value.identity.minimum_pack_bitmask,
        control_source_sha256=input_value.identity.control.expert_source_sha256,
        control_artifact_sha256=input_value.identity.control.artifact_sha256,
        status="pass" if passed else "fail",
        compiled=compiled,
        correct=correct,
        used_capabilities=input_value.identity.static.used_capabilities,
        generated_code_sha256=generated,
        control_generated_code_sha256=(input_value.identity.control.generated_code_sha256),
        failure_reason=failure_reason,
    )


def derive_capability_canary_evidence(
    manifest: CapabilityProbeStudyManifest,
    inputs: Iterable[CapabilityProbeInput],
    records: Iterable[CapabilityProbeRecord],
) -> tuple[CapabilityCanaryEvidence, ...]:
    """Recompute nested canary evidence from exact sealed raw probe artifacts."""

    input_values = tuple(inputs)
    record_values = tuple(records)
    if tuple(item.identity for item in input_values) != manifest.probes:
        raise MatrixCapabilityEvidenceError(
            "capability evidence inputs differ from the study manifest"
        )
    if len(record_values) != len(input_values):
        raise MatrixCapabilityEvidenceError(
            "capability records do not exactly cover the probe manifest"
        )
    for record, input_value in zip(record_values, input_values, strict=True):
        expected_directory = (
            Path(record.artifact_directory).parent / input_value.identity.artifact_id
        )
        if Path(record.artifact_directory) != expected_directory:
            raise MatrixCapabilityEvidenceError(
                "capability record order or artifact identity differs from the manifest"
            )
        loaded = _load_probe_record(
            Path(record.artifact_directory),
            input_value,
            manifest,
        )
        if loaded != record:
            raise MatrixCapabilityEvidenceError(
                "supplied capability record differs from sealed raw evidence"
            )

    grouped: dict[str, list[tuple[CapabilityProbeRecord, CapabilityProbeInput]]] = {}
    order: list[str] = []
    for record, input_value in zip(record_values, input_values, strict=True):
        canary_id = input_value.identity.canary_id
        if canary_id not in grouped:
            grouped[canary_id] = []
            order.append(canary_id)
        grouped[canary_id].append((record, input_value))

    evidence_values: list[CapabilityCanaryEvidence] = []
    for canary_id in order:
        pairs = grouped[canary_id]
        first = pairs[0][1].identity
        targets = tuple(
            _target_evidence(record, input_value, manifest) for record, input_value in pairs
        )
        passed = all(item.status == "pass" for item in targets)
        failure_reason = None
        if not passed:
            failed_targets = ", ".join(item.target_id for item in targets if item.status == "fail")
            failure_reason = f"capability canary failed required targets: {failed_targets}"
        evidence_values.append(
            CapabilityCanaryEvidence(
                canary_id=canary_id,
                canary_spec_sha256=first.canary_spec_sha256,
                task_id=first.task_id,
                source_sha256=first.source_sha256,
                minimum_pack_id=first.minimum_pack_id,
                minimum_pack_bitmask=first.minimum_pack_bitmask,
                status="pass" if passed else "fail",
                targets=targets,
                failure_reason=failure_reason,
            )
        )
    return tuple(evidence_values)
