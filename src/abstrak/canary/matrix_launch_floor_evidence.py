"""Run and derive hash-bound launch-floor evidence for matrix preflight."""

from __future__ import annotations

import hashlib
import json
import math
import os
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
from abstrak.canary.capabilities import CORE_PACK
from abstrak.canary.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    CanaryModel,
    TargetStackSpec,
    TaskPackSpec,
    TimingSpec,
)
from abstrak.canary.gates import GateRecord
from abstrak.canary.loop import WorkerExecutor
from abstrak.canary.manifests import PinnedStudySpec
from abstrak.canary.matrix import MatrixSchedule
from abstrak.canary.matrix_capability_evidence import (
    CapabilityProbeRecord,
    CapabilityProbeStudyManifest,
    MatrixCapabilityEvidenceError,
    build_capability_probe_study_manifest,
    capability_probe_artifact_sha256,
    derive_capability_canary_evidence,
    resolve_capability_probe_inputs,
)
from abstrak.canary.matrix_floor_evidence import (
    MatrixFloorEvidenceError,
    derive_task_floor_records,
    gate_artifact_sha256,
    require_passing_oracle,
    validate_gate_summary,
)
from abstrak.canary.matrix_floor_evidence import (
    generated_code_sha256 as gate_generated_code_sha256,
)
from abstrak.canary.matrix_preflight import (
    FORMAL_FLOOR_TIMING,
    AssetManifest,
    BaselineAssetBinding,
    EnvironmentManifest,
    LaunchFloorEvidence,
    LaunchProbeAssetBinding,
    LaunchTimingMeasurement,
    StudyBoundModel,
    TargetAssetBinding,
    TaskAssetBinding,
    TaskFloorRecord,
)
from abstrak.canary.matrix_runner import MatrixWorkerBinding
from abstrak.canary.postprocess_timing import (
    PostprocessTimingError,
    run_or_resume_candidate_timing_artifact,
)
from abstrak.canary.target_adapters import validate_target_source
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import (
    CAPABILITY_GATE_ASSET_ROOT,
    LAUNCH_FLOOR_TASK_ID,
    get_task_assets,
    get_task_pack,
    load_oracle_source,
    load_task_source,
)
from abstrak.canary.timing import TimingProtocolSummary
from abstrak.providers.contracts import sha256_json

_ARTIFACT_DIRECTORIES = frozenset({"events", "turns", "candidates", "sealed"})
_LAUNCH_ARTIFACT_ID = "launch-floor-probe"
_FLOAT_REL_TOL = 1e-12
_FLOAT_ABS_TOL = 1e-12


class MatrixLaunchFloorEvidenceError(ValueError):
    """Raised when launch-floor evidence differs from frozen or raw inputs."""


class LaunchFloorProbeInfrastructureError(MatrixLaunchFloorEvidenceError):
    """Raised when the launch probe cannot produce scientific evidence."""


def _same_float(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=_FLOAT_REL_TOL,
        abs_tol=_FLOAT_ABS_TOL,
    )


class TaskLaunchTimingInput(CanaryModel):
    """Auditable task denominator derived from its floor and core B-expert."""

    schema_version: Literal["abstrak-task-launch-timing-input.v1"] = (
        "abstrak-task-launch-timing-input.v1"
    )
    formula: Literal["min(l_star_ms, core_expert_ms)"] = (
        "min(l_star_ms, core_expert_ms)"
    )
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_floor_sha256: str = Field(pattern=SHA256_PATTERN)
    l_star_ms: float = Field(gt=0)
    selected_baseline_variant: str = Field(pattern=IDENTIFIER_PATTERN)
    selected_baseline_target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    selected_baseline_target_stack_sha256: str = Field(pattern=SHA256_PATTERN)
    selected_baseline_source_sha256: str = Field(pattern=SHA256_PATTERN)
    selected_baseline_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    selected_timing_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    core_target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    core_target_stack_sha256: str = Field(pattern=SHA256_PATTERN)
    core_expert_source_sha256: str = Field(pattern=SHA256_PATTERN)
    core_gate_record_sha256: str = Field(pattern=SHA256_PATTERN)
    core_gate_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    core_timing_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    core_expert_ms: float = Field(gt=0)
    selected_source: Literal["l-star", "core-b-expert"]
    task_ms: float = Field(gt=0)

    @field_validator("l_star_ms", "core_expert_ms", "task_ms")
    @classmethod
    def latencies_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("task launch-timing inputs must be finite")
        return value

    @model_validator(mode="after")
    def selected_latency_is_the_minimum(self) -> TaskLaunchTimingInput:
        expected_source: Literal["l-star", "core-b-expert"] = (
            "l-star" if self.l_star_ms <= self.core_expert_ms else "core-b-expert"
        )
        expected_ms = min(self.l_star_ms, self.core_expert_ms)
        if self.selected_source != expected_source or not _same_float(
            self.task_ms,
            expected_ms,
        ):
            raise ValueError("task launch denominator is not min(L_i*, core B-expert)")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class CanaryLaunchTimingInput(CanaryModel):
    """Auditable canary denominator from its stable minimum-pack target run."""

    schema_version: Literal["abstrak-canary-launch-timing-input.v1"] = (
        "abstrak-canary-launch-timing-input.v1"
    )
    formula: Literal["stable minimum-pack target median"] = (
        "stable minimum-pack target median"
    )
    canary_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    minimum_pack_id: str = Field(pattern=IDENTIFIER_PATTERN)
    minimum_pack_bitmask: int = Field(ge=1)
    minimum_target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    minimum_target_stack_sha256: str = Field(pattern=SHA256_PATTERN)
    capability_probe_study_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    capability_probe_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    capability_probe_record_sha256: str = Field(pattern=SHA256_PATTERN)
    capability_probe_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    capability_timing_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    task_ms: float = Field(gt=0)

    @field_validator("task_ms")
    @classmethod
    def latency_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("canary launch-timing input must be finite")
        return value

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class LaunchFloorProbeIdentity(StudyBoundModel):
    """Frozen launch probe, worker, and registry identity."""

    schema_version: Literal["abstrak-matrix-launch-floor-probe-identity.v1"] = (
        "abstrak-matrix-launch-floor-probe-identity.v1"
    )
    artifact_id: Literal["launch-floor-probe"] = _LAUNCH_ARTIFACT_ID
    asset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    worker: MatrixWorkerBinding
    task_id: Literal["tilelang-launch-floor-probe"] = LAUNCH_FLOOR_TASK_ID
    task_pack_sha256: str = Field(pattern=SHA256_PATTERN)
    reference_source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    target_id: Literal["tilelang-a100-core"] = CORE_PACK.target_id
    target_stack_sha256: str = Field(pattern=SHA256_PATTERN)
    target_card_sha256: str = Field(pattern=SHA256_PATTERN)
    timing: TimingSpec
    device: str = Field(pattern=r"^cuda:[0-9]+$")

    @model_validator(mode="after")
    def runtime_is_the_formal_core_probe(self) -> LaunchFloorProbeIdentity:
        if self.timing != FORMAL_FLOOR_TIMING:
            raise ValueError("launch probe must use FORMAL_FLOOR_TIMING")
        if self.worker.transport.device != self.device:
            raise ValueError("launch probe device differs from worker transport")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class LaunchFloorStudyManifest(StudyBoundModel):
    """Ordered launch-floor closure sealed before the remote probe runs."""

    schema_version: Literal["abstrak-matrix-launch-floor-study.v1"] = (
        "abstrak-matrix-launch-floor-study.v1"
    )
    launch_study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    asset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    capability_probe_study_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    worker: MatrixWorkerBinding
    timing: TimingSpec
    max_launch_fraction: float = Field(gt=0, lt=1)
    probe: LaunchFloorProbeIdentity
    workload_count: int = Field(ge=1)
    tasks: tuple[TaskLaunchTimingInput, ...] = Field(min_length=1)
    canaries: tuple[CanaryLaunchTimingInput, ...] = ()

    @field_validator("max_launch_fraction")
    @classmethod
    def fraction_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("launch-floor threshold must be finite")
        return value

    @model_validator(mode="after")
    def closure_is_complete_and_ordered(self) -> LaunchFloorStudyManifest:
        expected_study_id = f"{self.study_id}-launch-floor"
        if self.launch_study_id != expected_study_id:
            raise ValueError("launch study ID differs from the matrix study")
        if self.timing != FORMAL_FLOOR_TIMING:
            raise ValueError("launch-floor study must use FORMAL_FLOOR_TIMING")
        if self.workload_count != len(self.tasks) + len(self.canaries):
            raise ValueError("launch workload count differs from its detailed inputs")
        task_ids = tuple(item.task_id for item in self.tasks)
        canary_ids = tuple(item.canary_id for item in self.canaries)
        if len(task_ids) != len(set(task_ids)) or len(canary_ids) != len(set(canary_ids)):
            raise ValueError("launch workload identities must be unique within each kind")
        expected_runtime = (
            self.study_binding,
            self.asset_manifest_sha256,
            self.environment_manifest_sha256,
            self.worker,
            self.timing,
        )
        actual_runtime = (
            self.probe.study_binding,
            self.probe.asset_manifest_sha256,
            self.probe.environment_manifest_sha256,
            self.probe.worker,
            self.probe.timing,
        )
        if actual_runtime != expected_runtime:
            raise ValueError("launch probe has different study or runtime bindings")
        if any(
            item.capability_probe_study_manifest_sha256
            != self.capability_probe_study_manifest_sha256
            for item in self.canaries
        ):
            raise ValueError("canary timing inputs reference a different capability study")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class LaunchFloorProbeRecord(CanaryModel):
    """Sealed raw launch-probe result used to derive terminal evidence."""

    schema_version: Literal["abstrak-matrix-launch-floor-probe-record.v1"] = (
        "abstrak-matrix-launch-floor-probe-record.v1"
    )
    launch_study_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    identity: LaunchFloorProbeIdentity
    artifact_directory: str = Field(min_length=1)
    summary: TimingProtocolSummary

    @property
    def sha256(self) -> str:
        return sha256_json(self)


@dataclass(frozen=True)
class LaunchFloorProbeInput:
    """Verified local source and registry objects for the launch probe."""

    identity: LaunchFloorProbeIdentity
    source: str
    task: TaskPackSpec
    target: TargetStackSpec
    task_asset: TaskAssetBinding
    target_asset: TargetAssetBinding
    asset_binding: LaunchProbeAssetBinding

    def __post_init__(self) -> None:
        identity = self.identity
        if (
            hashlib.sha256(self.source.encode("utf-8")).hexdigest()
            != identity.source_sha256
            or self.task.id != identity.task_id
            or sha256_json(self.task) != identity.task_pack_sha256
            or self.target.id != identity.target_id
            or sha256_json(self.target) != identity.target_stack_sha256
            or self.task_asset.task_id != identity.task_id
            or self.task_asset.task_pack_sha256 != identity.task_pack_sha256
            or self.task_asset.reference_source_sha256
            != identity.reference_source_sha256
            or self.task_asset.expert_source_sha256 != identity.source_sha256
            or self.target_asset.target_id != identity.target_id
            or self.target_asset.target_stack_sha256 != identity.target_stack_sha256
            or self.target_asset.card_sha256 != identity.target_card_sha256
            or self.asset_binding.task_id != identity.task_id
            or self.asset_binding.task_pack_sha256 != identity.task_pack_sha256
            or self.asset_binding.reference_source_sha256
            != identity.reference_source_sha256
            or self.asset_binding.source_sha256 != identity.source_sha256
            or self.asset_binding.target_id != identity.target_id
            or self.asset_binding.target_stack_sha256
            != identity.target_stack_sha256
            or self.asset_binding.target_card_sha256 != identity.target_card_sha256
        ):
            raise ValueError("launch probe runtime input differs from its identity")


def _study_binding(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
) -> tuple[str, str, str, str]:
    if schedule.spec != pinned.spec or schedule.spec_sha256 != pinned.spec.sha256:
        raise MatrixLaunchFloorEvidenceError(
            "matrix schedule differs from the pinned study spec"
        )
    return (
        pinned.spec.study_id,
        pinned.sha256,
        pinned.spec.sha256,
        schedule.sha256,
    )


def _require_verified_environment(
    assets: AssetManifest,
    environment: EnvironmentManifest,
) -> MatrixWorkerBinding:
    if environment.status != "verified" or environment.verification_evidence is None:
        raise MatrixLaunchFloorEvidenceError(
            "launch floor requires a verified environment"
        )
    if assets.study_binding != environment.study_binding:
        raise MatrixLaunchFloorEvidenceError(
            "launch-floor assets and environment have different study identities"
        )
    return MatrixWorkerBinding(
        worker_revision=environment.worker_revision,
        transport=environment.transport,
    )


def resolve_launch_floor_probe_input(
    assets: AssetManifest,
    environment: EnvironmentManifest,
    *,
    asset_root: str | Path = CAPABILITY_GATE_ASSET_ROOT,
) -> LaunchFloorProbeInput:
    """Resolve the independent one-kernel probe under the frozen core target."""

    worker = _require_verified_environment(assets, environment)
    task = get_task_pack(LAUNCH_FLOOR_TASK_ID)
    registered_assets = get_task_assets(LAUNCH_FLOOR_TASK_ID)
    reference = load_task_source(LAUNCH_FLOOR_TASK_ID, asset_root=asset_root)
    source = load_oracle_source(
        LAUNCH_FLOOR_TASK_ID,
        "tilelang",
        asset_root=asset_root,
    )
    target = get_target_stack(CORE_PACK.target_id)
    try:
        target_asset = next(
            item for item in assets.targets if item.target_id == CORE_PACK.target_id
        )
    except StopIteration:
        raise MatrixLaunchFloorEvidenceError(
            "asset manifest is missing the core launch target"
        ) from None
    reference_sha256 = hashlib.sha256(reference.encode("utf-8")).hexdigest()
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if (
        reference_sha256 != task.source_sha256
        or reference_sha256 != registered_assets.source.sha256
        or source_sha256 != registered_assets.oracles["tilelang"].sha256
        or sha256_json(target) != target_asset.target_stack_sha256
        or target.card_sha256 != target_asset.card_sha256
    ):
        raise MatrixLaunchFloorEvidenceError(
            "launch probe registry differs from the study assets"
        )
    validation = validate_target_source(source, target)
    metadata = dict(validation.metadata)
    if (
        not validation.valid
        or validation.errors
        or metadata.get("minimum_pack_id") != CORE_PACK.id
        or metadata.get("minimum_pack_bitmask") != CORE_PACK.bitmask
    ):
        raise MatrixLaunchFloorEvidenceError(
            "launch probe source is not B-legal under the core target"
        )
    task_asset = TaskAssetBinding(
        task_id=task.id,
        task_pack_sha256=sha256_json(task),
        reference_source_sha256=reference_sha256,
        expert_source_sha256=source_sha256,
        baselines=(
            BaselineAssetBinding(
                variant="launch-probe",
                source_sha256=source_sha256,
            ),
        ),
    )
    asset_binding = LaunchProbeAssetBinding(
        task_id=task.id,
        task_pack_sha256=sha256_json(task),
        reference_source_sha256=reference_sha256,
        source_sha256=source_sha256,
        target_id=target.id,
        target_stack_sha256=sha256_json(target),
        target_card_sha256=target.card_sha256,
    )
    if assets.launch_probe != asset_binding:
        raise MatrixLaunchFloorEvidenceError(
            "launch probe differs from the frozen asset manifest"
        )
    identity = LaunchFloorProbeIdentity(
        study_id=assets.study_id,
        raw_study_sha256=assets.raw_study_sha256,
        spec_sha256=assets.spec_sha256,
        schedule_sha256=assets.schedule_sha256,
        asset_manifest_sha256=assets.sha256,
        environment_manifest_sha256=environment.sha256,
        worker=worker,
        task_pack_sha256=sha256_json(task),
        reference_source_sha256=reference_sha256,
        source_sha256=source_sha256,
        target_stack_sha256=sha256_json(target),
        target_card_sha256=target.card_sha256,
        timing=FORMAL_FLOOR_TIMING,
        device=environment.transport.device,
    )
    return LaunchFloorProbeInput(
        identity=identity,
        source=source,
        task=task,
        target=target,
        task_asset=task_asset,
        target_asset=target_asset,
        asset_binding=asset_binding,
    )


def _task_timing_inputs(
    assets: AssetManifest,
    task_floors: tuple[TaskFloorRecord, ...],
    core_records: tuple[GateRecord, ...],
    selected_baseline_records: tuple[GateRecord, ...],
    baseline_target: TargetAssetBinding,
) -> tuple[TaskLaunchTimingInput, ...]:
    expected_task_ids = tuple(item.task_id for item in assets.tasks)
    if tuple(item.task_id for item in task_floors) != expected_task_ids:
        raise MatrixLaunchFloorEvidenceError(
            "task floors do not exactly cover launch workloads in asset order"
        )
    expected_core_keys = tuple(
        (task_id, CORE_PACK.target_id) for task_id in expected_task_ids
    )
    actual_core_keys = tuple((item.task_id, item.target_id) for item in core_records)
    if actual_core_keys != expected_core_keys:
        raise MatrixLaunchFloorEvidenceError(
            "core B-expert gates do not exactly cover tasks in asset order"
        )
    expected_baseline_keys = tuple(
        (
            task.task_id,
            baseline_target.target_id,
            floor.verified_evidence.selected_baseline_variant
            if floor.verified_evidence is not None
            else None,
        )
        for task, floor in zip(assets.tasks, task_floors, strict=True)
    )
    actual_baseline_keys = tuple(
        (item.task_id, item.target_id, item.variant)
        for item in selected_baseline_records
    )
    if actual_baseline_keys != expected_baseline_keys:
        raise MatrixLaunchFloorEvidenceError(
            "selected baseline gates do not exactly cover tasks in asset order"
        )
    core_target = next(
        (item for item in assets.targets if item.target_id == CORE_PACK.target_id),
        None,
    )
    if core_target is None:
        raise MatrixLaunchFloorEvidenceError(
            "asset manifest is missing the core B-expert target"
        )

    inputs: list[TaskLaunchTimingInput] = []
    for task_asset, floor, record, baseline_record in zip(
        assets.tasks,
        task_floors,
        core_records,
        selected_baseline_records,
        strict=True,
    ):
        if floor.status != "valid" or floor.verified_evidence is None:
            raise MatrixLaunchFloorEvidenceError(
                f"launch timing requires a valid task floor: {task_asset.task_id}"
            )
        evidence = floor.verified_evidence
        if (
            floor.expert_source_sha256 != task_asset.expert_source_sha256
            or evidence.expert_source_sha256 != task_asset.expert_source_sha256
            or record.kind != "oracle"
            or record.variant is not None
            or record.source_sha256 != task_asset.expert_source_sha256
        ):
            raise MatrixLaunchFloorEvidenceError(
                f"core B-expert identity differs from task floor: {task_asset.task_id}"
            )
        try:
            artifact_sha256 = gate_artifact_sha256(record)
            validate_gate_summary(
                record,
                task=task_asset,
                target=core_target,
            )
            generated_code_sha256 = gate_generated_code_sha256(
                record,
                require_passing_oracle(record),
            )
        except MatrixFloorEvidenceError as error:
            raise MatrixLaunchFloorEvidenceError(
                f"core B-expert raw evidence is invalid: {task_asset.task_id}: {error}"
            ) from error
        if not record.summary.stable or record.summary.median_ms is None:
            raise MatrixLaunchFloorEvidenceError(
                f"core B-expert timing is not stable: {task_asset.task_id}"
            )
        selected = next(
            item
            for item in evidence.baseline_timings
            if item.variant == evidence.selected_baseline_variant
        )
        controls = tuple(
            item
            for item in evidence.target_codegen
            if item.target_id == CORE_PACK.target_id
        )
        if len(controls) != 1:
            raise MatrixLaunchFloorEvidenceError(
                f"task floor has no unique core codegen evidence: {task_asset.task_id}"
            )
        control = controls[0]
        try:
            baseline_artifact_sha256 = gate_artifact_sha256(baseline_record)
            validate_gate_summary(
                baseline_record,
                task=task_asset,
                target=baseline_target,
            )
        except MatrixFloorEvidenceError as error:
            raise MatrixLaunchFloorEvidenceError(
                f"selected baseline raw evidence is invalid: "
                f"{task_asset.task_id}: {error}"
            ) from error
        if (
            selected.status != "stable"
            or selected.median_ms is None
            or selected.source_sha256 != evidence.selected_baseline_source_sha256
            or selected.timing_summary_sha256
            != evidence.selected_timing_summary_sha256
            or baseline_record.kind != "baseline"
            or baseline_record.variant != selected.variant
            or baseline_record.source_sha256 != selected.source_sha256
            or baseline_record.summary.median_ms is None
            or not baseline_record.summary.stable
            or not _same_float(
                baseline_record.summary.median_ms,
                selected.median_ms,
            )
            or baseline_artifact_sha256 != selected.artifact_sha256
            or sha256_json(baseline_record.summary)
            != selected.timing_summary_sha256
            or control.status != "pass"
            or control.task_pack_sha256 != task_asset.task_pack_sha256
            or control.target_stack_sha256 != core_target.target_stack_sha256
            or control.expert_source_sha256 != task_asset.expert_source_sha256
            or control.artifact_sha256 != artifact_sha256
            or control.generated_code_sha256 != generated_code_sha256
        ):
            raise MatrixLaunchFloorEvidenceError(
                f"task-floor provenance differs from the core gate: {task_asset.task_id}"
            )
        core_ms = record.summary.median_ms
        l_star_ms = evidence.l_star_ms
        selected_source: Literal["l-star", "core-b-expert"] = (
            "l-star" if l_star_ms <= core_ms else "core-b-expert"
        )
        inputs.append(
            TaskLaunchTimingInput(
                task_id=task_asset.task_id,
                task_floor_sha256=floor.sha256,
                l_star_ms=l_star_ms,
                selected_baseline_variant=selected.variant,
                selected_baseline_target_id=baseline_target.target_id,
                selected_baseline_target_stack_sha256=(
                    baseline_target.target_stack_sha256
                ),
                selected_baseline_source_sha256=selected.source_sha256,
                selected_baseline_artifact_sha256=selected.artifact_sha256,
                selected_timing_summary_sha256=selected.timing_summary_sha256,
                core_target_id=CORE_PACK.target_id,
                core_target_stack_sha256=core_target.target_stack_sha256,
                core_expert_source_sha256=record.source_sha256,
                core_gate_record_sha256=sha256_json(record),
                core_gate_artifact_sha256=artifact_sha256,
                core_timing_summary_sha256=sha256_json(record.summary),
                core_expert_ms=core_ms,
                selected_source=selected_source,
                task_ms=min(l_star_ms, core_ms),
            )
        )
    return tuple(inputs)


def _canary_timing_inputs(
    assets: AssetManifest,
    environment: EnvironmentManifest,
    task_floors: tuple[TaskFloorRecord, ...],
    capability_manifest: CapabilityProbeStudyManifest,
    capability_records: tuple[CapabilityProbeRecord, ...],
    *,
    asset_root: str | Path,
) -> tuple[CanaryLaunchTimingInput, ...]:
    if (
        capability_manifest.study_binding != assets.study_binding
        or capability_manifest.asset_manifest_sha256 != assets.sha256
        or capability_manifest.environment_manifest_sha256 != environment.sha256
        or capability_manifest.timing != FORMAL_FLOOR_TIMING
    ):
        raise MatrixLaunchFloorEvidenceError(
            "capability probe manifest differs from launch-floor inputs"
        )
    try:
        capability_inputs = resolve_capability_probe_inputs(
            assets,
            environment,
            task_floors,
            asset_root=asset_root,
        )
        expected_manifest = build_capability_probe_study_manifest(
            assets,
            environment,
            capability_inputs,
        )
        if expected_manifest != capability_manifest:
            raise MatrixLaunchFloorEvidenceError(
                "capability probe manifest differs from frozen registry inputs"
            )
        evidence_values = derive_capability_canary_evidence(
            capability_manifest,
            capability_inputs,
            capability_records,
        )
    except MatrixLaunchFloorEvidenceError:
        raise
    except MatrixCapabilityEvidenceError as error:
        raise MatrixLaunchFloorEvidenceError(
            f"capability timing evidence is invalid: {error}"
        ) from error
    if len(capability_records) != len(capability_inputs):
        raise MatrixLaunchFloorEvidenceError(
            "capability records do not exactly cover the probe study"
        )
    if any(item.status != "pass" for item in evidence_values):
        failed = ", ".join(item.canary_id for item in evidence_values if item.status == "fail")
        raise MatrixLaunchFloorEvidenceError(
            f"launch timing requires passing capability canaries: {failed}"
        )
    by_key = {
        (input_value.identity.canary_id, input_value.identity.target_id): (
            input_value,
            record,
        )
        for input_value, record in zip(
            capability_inputs,
            capability_records,
            strict=True,
        )
    }
    inputs: list[CanaryLaunchTimingInput] = []
    for binding in assets.canaries:
        minimum_target_id = binding.required_target_ids[0]
        try:
            input_value, record = by_key[(binding.canary_id, minimum_target_id)]
        except KeyError:
            raise MatrixLaunchFloorEvidenceError(
                f"canary has no minimum-pack timing record: {binding.canary_id}"
            ) from None
        identity = input_value.identity
        if (
            identity.task_id != binding.task_id
            or identity.source_sha256 != binding.source_sha256
            or identity.target_id != minimum_target_id
            or not record.summary.stable
            or record.summary.median_ms is None
        ):
            raise MatrixLaunchFloorEvidenceError(
                f"minimum-pack canary timing is not stable: {binding.canary_id}"
            )
        try:
            artifact_sha256 = capability_probe_artifact_sha256(record)
        except MatrixCapabilityEvidenceError as error:
            raise MatrixLaunchFloorEvidenceError(
                f"capability timing artifact is invalid: {binding.canary_id}: {error}"
            ) from error
        inputs.append(
            CanaryLaunchTimingInput(
                canary_id=binding.canary_id,
                task_id=binding.task_id,
                source_sha256=binding.source_sha256,
                minimum_pack_id=identity.minimum_pack_id,
                minimum_pack_bitmask=identity.minimum_pack_bitmask,
                minimum_target_id=minimum_target_id,
                minimum_target_stack_sha256=identity.target_stack_sha256,
                capability_probe_study_manifest_sha256=capability_manifest.sha256,
                capability_probe_identity_sha256=identity.sha256,
                capability_probe_record_sha256=record.sha256,
                capability_probe_artifact_sha256=artifact_sha256,
                capability_timing_summary_sha256=sha256_json(record.summary),
                task_ms=record.summary.median_ms,
            )
        )
    return tuple(inputs)


def build_launch_floor_study_manifest(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    assets: AssetManifest,
    environment: EnvironmentManifest,
    task_floors: Iterable[TaskFloorRecord],
    gate_records: Iterable[GateRecord],
    capability_manifest: CapabilityProbeStudyManifest,
    capability_records: Iterable[CapabilityProbeRecord],
    probe_input: LaunchFloorProbeInput,
    *,
    baseline_target_id: str,
    asset_root: str | Path = CAPABILITY_GATE_ASSET_ROOT,
) -> LaunchFloorStudyManifest:
    """Freeze detailed workload denominators and the independent launch probe."""

    expected_binding = _study_binding(pinned, schedule)
    worker = _require_verified_environment(assets, environment)
    if assets.study_binding != expected_binding:
        raise MatrixLaunchFloorEvidenceError(
            "asset manifest differs from the pinned study identity"
        )
    if environment.study_binding != expected_binding:
        raise MatrixLaunchFloorEvidenceError(
            "environment manifest differs from the pinned study identity"
        )
    gate = pinned.spec.gate
    if gate is None or gate.metrics.latency_tie_fraction <= 0:
        raise MatrixLaunchFloorEvidenceError(
            "launch floor requires a positive study latency tie fraction"
        )
    expected_task_ids = tuple(
        dict.fromkeys(
            task_id for phase in pinned.spec.phases for task_id in phase.task_ids
        )
    )
    if (
        tuple(item.task_id for item in assets.tasks) != expected_task_ids
        or tuple(item.target_id for item in assets.targets) != pinned.spec.targets
    ):
        raise MatrixLaunchFloorEvidenceError(
            "launch-floor assets do not exactly cover the pinned study"
        )
    expected_probe = resolve_launch_floor_probe_input(
        assets,
        environment,
        asset_root=asset_root,
    )
    if expected_probe != probe_input:
        raise MatrixLaunchFloorEvidenceError(
            "launch probe input differs from frozen registry inputs"
        )
    task_floor_values = tuple(task_floors)
    gate_values = tuple(gate_records)
    targets = tuple(get_target_stack(item.target_id) for item in assets.targets)
    try:
        recomputed_floors = derive_task_floor_records(
            gate_values,
            assets,
            targets,
            baseline_target_id=baseline_target_id,
            competitive_factor=gate.metrics.competitive_latency_factor,
        )
    except MatrixFloorEvidenceError as error:
        raise MatrixLaunchFloorEvidenceError(
            f"cannot recompute task floors from sealed gate evidence: {error}"
        ) from error
    if recomputed_floors != task_floor_values:
        raise MatrixLaunchFloorEvidenceError(
            "task floors differ from sealed gate recomputation"
        )
    try:
        baseline_target = next(
            item for item in assets.targets if item.target_id == baseline_target_id
        )
    except StopIteration:
        raise MatrixLaunchFloorEvidenceError(
            "baseline target is not present in frozen assets"
        ) from None
    records_by_key = {
        (item.kind, item.task_id, item.target_id, item.variant): item
        for item in gate_values
    }
    core_records = tuple(
        records_by_key[("oracle", item.task_id, CORE_PACK.target_id, None)]
        for item in assets.tasks
    )
    selected_baseline_records = tuple(
        records_by_key[
            (
                "baseline",
                item.task_id,
                baseline_target_id,
                floor.verified_evidence.selected_baseline_variant,
            )
        ]
        for item, floor in zip(assets.tasks, task_floor_values, strict=True)
        if floor.verified_evidence is not None
    )
    task_inputs = _task_timing_inputs(
        assets,
        task_floor_values,
        core_records,
        selected_baseline_records,
        baseline_target,
    )
    canary_inputs = _canary_timing_inputs(
        assets,
        environment,
        task_floor_values,
        capability_manifest,
        tuple(capability_records),
        asset_root=asset_root,
    )
    return LaunchFloorStudyManifest(
        study_id=assets.study_id,
        raw_study_sha256=assets.raw_study_sha256,
        spec_sha256=assets.spec_sha256,
        schedule_sha256=assets.schedule_sha256,
        launch_study_id=f"{assets.study_id}-launch-floor",
        asset_manifest_sha256=assets.sha256,
        environment_manifest_sha256=environment.sha256,
        capability_probe_study_manifest_sha256=capability_manifest.sha256,
        worker=worker,
        timing=FORMAL_FLOOR_TIMING,
        max_launch_fraction=gate.metrics.latency_tie_fraction,
        probe=probe_input.identity,
        workload_count=len(task_inputs) + len(canary_inputs),
        tasks=task_inputs,
        canaries=canary_inputs,
    )


def _verify_artifact_shape(directory: Path, *, record: bool) -> None:
    if directory.is_symlink():
        raise MatrixLaunchFloorEvidenceError(
            "launch-floor artifact cannot be a symbolic link"
        )
    try:
        entries = tuple(directory.rglob("*"))
    except OSError as error:
        raise MatrixLaunchFloorEvidenceError(
            f"cannot inspect launch-floor artifact: {directory}"
        ) from error
    if any(entry.is_symlink() for entry in entries):
        raise MatrixLaunchFloorEvidenceError(
            "launch-floor artifact cannot contain symbolic links"
        )
    actual_files = {
        item.relative_to(directory).as_posix() for item in entries if item.is_file()
    }
    actual_directories = {
        item.relative_to(directory).as_posix() for item in entries if item.is_dir()
    }
    expected_files = {"run-manifest.json", "sha256sums.txt"}
    if record:
        expected_files.add("timing-record.json")
    if actual_files != expected_files or actual_directories != _ARTIFACT_DIRECTORIES:
        raise MatrixLaunchFloorEvidenceError(
            "launch-floor artifact has unexpected files or directories"
        )


def _remove_unsealed_staging(directory: Path) -> bool:
    if not directory.exists() and not directory.is_symlink():
        return False
    if directory.is_symlink():
        raise MatrixLaunchFloorEvidenceError(
            "launch-floor staging artifact cannot be a symbolic link"
        )
    try:
        verify_trajectory(directory)
    except (OSError, TrajectoryArtifactError):
        shutil.rmtree(directory)
        return True
    return False


def _load_study_manifest(
    directory: Path,
    expected: LaunchFloorStudyManifest,
) -> LaunchFloorStudyManifest:
    try:
        _verify_artifact_shape(directory, record=False)
        verify_trajectory(directory)
        actual = LaunchFloorStudyManifest.model_validate_json(
            (directory / "run-manifest.json").read_text(encoding="utf-8")
        )
    except MatrixLaunchFloorEvidenceError:
        raise
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise MatrixLaunchFloorEvidenceError(
            f"launch-floor study manifest is invalid: {directory}"
        ) from error
    if actual != expected:
        raise MatrixLaunchFloorEvidenceError(
            "launch-floor study manifest differs from frozen inputs"
        )
    return actual


def seal_launch_floor_study_manifest(
    artifact_root: str | Path,
    manifest: LaunchFloorStudyManifest,
) -> Path:
    """Atomically seal or exactly resume the launch-floor study manifest."""

    root = Path(artifact_root).expanduser()
    final = root / manifest.launch_study_id / "study-manifest"
    staging = final.with_name("study-manifest.incomplete")
    if final.exists() and staging.exists():
        raise MatrixLaunchFloorEvidenceError(
            "final and staging launch-floor study manifests both exist"
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
            manifest.launch_study_id,
            "study-manifest.incomplete",
        )
        store.write_json("run-manifest.json", manifest)
        store.seal()
        if final.exists() or final.is_symlink():
            raise MatrixLaunchFloorEvidenceError(
                "launch-floor study manifest appeared during atomic staging"
            )
        os.replace(store.run_directory, final)
    except MatrixLaunchFloorEvidenceError:
        raise
    except (OSError, TrajectoryArtifactError) as error:
        raise MatrixLaunchFloorEvidenceError(
            "cannot seal launch-floor study manifest"
        ) from error
    _load_study_manifest(final, manifest)
    return final


def _expected_timing_manifest(
    manifest: LaunchFloorStudyManifest,
) -> dict[str, object]:
    return {
        "schema_version": "abstrak-matrix-launch-floor-timing-manifest.v1",
        "launch_study_manifest_sha256": manifest.sha256,
        "identity": manifest.probe,
    }


def _inspect_probe_record(
    record: LaunchFloorProbeRecord,
    probe_input: LaunchFloorProbeInput,
    manifest: LaunchFloorStudyManifest,
) -> None:
    identity = probe_input.identity
    summary = record.summary
    if (
        manifest.probe != identity
        or record.launch_study_manifest_sha256 != manifest.sha256
        or record.identity != identity
        or summary.job_prefix != identity.artifact_id
        or summary.task_id != identity.task_id
        or summary.target_id != identity.target_id
        or summary.candidate_sha256 != identity.source_sha256
        or summary.job_kind != "sealed"
        or summary.device != identity.device
        or summary.timing != FORMAL_FLOOR_TIMING
    ):
        raise MatrixLaunchFloorEvidenceError(
            "launch probe record differs from its frozen identity"
        )
    synthetic_gate = GateRecord(
        kind="oracle",
        task_id=identity.task_id,
        target_id=identity.target_id,
        source_sha256=identity.source_sha256,
        artifact_directory=record.artifact_directory,
        summary=summary.model_copy(update={"job_kind": "oracle"}),
    )
    synthetic_attempts = tuple(
        attempt.model_copy(
            update={
                "jobs": tuple(
                    job.model_copy(update={"kind": "oracle"})
                    for job in attempt.jobs
                )
            }
        )
        for attempt in synthetic_gate.summary.attempts
    )
    synthetic_jobs = tuple(job for attempt in synthetic_attempts for job in attempt.jobs)
    synthetic_results = tuple(
        result.model_copy(
            update={
                "job_sha256": job.sha256,
                "input_sha256": job.input_sha256,
            }
        )
        for job, result in zip(synthetic_jobs, synthetic_gate.summary.results, strict=False)
    )
    result_offset = 0
    rebuilt_attempts = []
    for attempt in synthetic_attempts:
        count = len(attempt.results)
        rebuilt_attempts.append(
            attempt.model_copy(
                update={
                    "results": synthetic_results[result_offset : result_offset + count]
                }
            )
        )
        result_offset += count
    synthetic_gate = synthetic_gate.model_copy(
        update={
            "summary": synthetic_gate.summary.model_copy(
                update={
                    "attempts": tuple(rebuilt_attempts),
                    "jobs": synthetic_jobs,
                    "results": synthetic_results,
                }
            )
        }
    )
    try:
        validate_gate_summary(
            synthetic_gate,
            task=probe_input.task_asset,
            target=probe_input.target_asset,
        )
    except MatrixFloorEvidenceError as error:
        raise MatrixLaunchFloorEvidenceError(
            f"launch probe raw timing or correctness is invalid: {error}"
        ) from error
    expected_process_timing = FORMAL_FLOOR_TIMING.model_copy(update={"repetitions": 1})
    for attempt in summary.attempts:
        for repetition, job in enumerate(attempt.jobs, start=1):
            if (
                job.job_id
                != f"{identity.artifact_id}-timing-a{attempt.attempt}-p{repetition}"
                or job.kind != "sealed"
                or job.task != probe_input.task
                or job.target != probe_input.target
                or tuple(job.case_ids)
                != tuple(case.id for case in probe_input.task.sealed_cases)
                or job.candidate_source != probe_input.source
                or job.candidate_sha256 != identity.source_sha256
                or job.timing != expected_process_timing
                or job.device != identity.device
            ):
                raise MatrixLaunchFloorEvidenceError(
                    f"launch worker job differs from frozen input: {job.job_id}"
                )
    if summary.status in {"stable", "unstable"}:
        generated = []
        for result in summary.results:
            digest = result.metadata.get("generated_code_sha256")
            capture = result.metadata.get("generated_code_capture")
            size = result.metadata.get("generated_code_size_bytes")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or capture != "tilelang.get_kernel_source.v1"
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
            ):
                raise MatrixLaunchFloorEvidenceError(
                    "completed launch probe is missing generated-code evidence"
                )
            generated.append((digest, size))
        if not generated or len(set(generated)) != 1:
            raise MatrixLaunchFloorEvidenceError(
                "launch probe generated code changed across clean processes"
            )
    if summary.status == "worker_failure":
        raise LaunchFloorProbeInfrastructureError(
            summary.error or "launch probe worker failure"
        )


def _verify_probe_study_directory(
    artifact_root: str | Path,
    manifest: LaunchFloorStudyManifest,
) -> None:
    root = Path(artifact_root).expanduser() / manifest.launch_study_id
    _load_study_manifest(root / "study-manifest", manifest)
    allowed = {
        "study-manifest",
        manifest.probe.artifact_id,
        f"{manifest.probe.artifact_id}.incomplete",
    }
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise MatrixLaunchFloorEvidenceError(
            "cannot inspect launch-floor study directory"
        ) from error
    if any(entry.is_symlink() for entry in entries):
        raise MatrixLaunchFloorEvidenceError(
            "launch-floor study cannot contain symbolic links"
        )
    unexpected = sorted(entry.name for entry in entries if entry.name not in allowed)
    if unexpected:
        raise MatrixLaunchFloorEvidenceError(
            f"launch-floor study contains unexpected artifacts: {unexpected}"
        )


def launch_probe_artifact_sha256(record: LaunchFloorProbeRecord) -> str:
    """Verify a sealed launch probe and hash its checksum-manifest bytes."""

    directory = Path(record.artifact_directory).expanduser()
    try:
        _verify_artifact_shape(directory, record=True)
        verify_trajectory(directory)
        persisted = LaunchFloorProbeRecord.model_validate_json(
            (directory / "timing-record.json").read_text(encoding="utf-8")
        )
        if persisted != record:
            raise MatrixLaunchFloorEvidenceError(
                "supplied launch record differs from its sealed artifact"
            )
        study_directory = directory.parent / "study-manifest"
        parsed_study_manifest = LaunchFloorStudyManifest.model_validate_json(
            (study_directory / "run-manifest.json").read_text(encoding="utf-8")
        )
        sealed_study_manifest = _load_study_manifest(
            study_directory,
            parsed_study_manifest,
        )
        if record.launch_study_manifest_sha256 != sealed_study_manifest.sha256:
            raise MatrixLaunchFloorEvidenceError(
                "launch record references a different sealed study manifest"
            )
        expected_manifest = _expected_timing_manifest(sealed_study_manifest)
        actual_manifest = json.loads(
            (directory / "run-manifest.json").read_text(encoding="utf-8")
        )
        normalized_expected = json.loads(
            json.dumps(
                expected_manifest,
                default=lambda value: value.model_dump(mode="json"),
                sort_keys=True,
            )
        )
        if actual_manifest != normalized_expected:
            raise MatrixLaunchFloorEvidenceError(
                "launch timing manifest differs from its study manifest"
            )
        if directory.resolve(strict=True) != Path(
            persisted.artifact_directory
        ).expanduser().resolve(strict=True):
            raise MatrixLaunchFloorEvidenceError(
                "sealed launch record does not identify its containing directory"
            )
        checksum = (directory / "sha256sums.txt").read_bytes()
    except MatrixLaunchFloorEvidenceError:
        raise
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise MatrixLaunchFloorEvidenceError(
            f"sealed launch probe artifact is invalid: {directory}"
        ) from error
    return hashlib.sha256(checksum).hexdigest()


def run_launch_floor_probe(
    worker: WorkerExecutor,
    *,
    artifact_root: str | Path,
    manifest: LaunchFloorStudyManifest,
    probe_input: LaunchFloorProbeInput,
) -> LaunchFloorProbeRecord:
    """Run or exactly resume the independent launch-floor timing probe."""

    if probe_input.identity != manifest.probe:
        raise MatrixLaunchFloorEvidenceError(
            "runtime launch probe input differs from the study manifest"
        )
    if getattr(worker, "matrix_worker_binding", None) != manifest.worker:
        raise MatrixLaunchFloorEvidenceError(
            "launch probe worker differs from the verified environment"
        )
    seal_launch_floor_study_manifest(artifact_root, manifest)
    _verify_probe_study_directory(artifact_root, manifest)
    identity = manifest.probe
    final = (
        Path(artifact_root).expanduser()
        / manifest.launch_study_id
        / identity.artifact_id
    )

    def build_record(
        summary: TimingProtocolSummary,
        final_path: Path,
    ) -> LaunchFloorProbeRecord:
        return LaunchFloorProbeRecord(
            launch_study_manifest_sha256=manifest.sha256,
            identity=identity,
            artifact_directory=str(final_path),
            summary=summary,
        )

    def validate_record(record: LaunchFloorProbeRecord) -> None:
        expected_path = final.expanduser().resolve(strict=False)
        declared_path = Path(record.artifact_directory).expanduser().resolve(
            strict=False
        )
        if declared_path != expected_path:
            raise MatrixLaunchFloorEvidenceError(
                "launch record artifact directory differs from frozen inputs"
            )
        _inspect_probe_record(record, probe_input, manifest)

    try:
        record, _ = run_or_resume_candidate_timing_artifact(
            worker,
            root=artifact_root,
            timing_study_id=manifest.launch_study_id,
            timing_id=identity.artifact_id,
            expected_manifest=_expected_timing_manifest(manifest),
            task=probe_input.task,
            target=probe_input.target,
            source=probe_input.source,
            source_sha256=identity.source_sha256,
            timing=FORMAL_FLOOR_TIMING,
            device=identity.device,
            record_type=LaunchFloorProbeRecord,
            build_record=build_record,
            validate_record=validate_record,
        )
    except MatrixLaunchFloorEvidenceError:
        raise
    except PostprocessTimingError as error:
        raise MatrixLaunchFloorEvidenceError(
            f"cannot run or resume launch probe: {error}"
        ) from error
    _verify_probe_study_directory(artifact_root, manifest)
    launch_probe_artifact_sha256(record)
    return record


def derive_launch_floor_evidence(
    manifest: LaunchFloorStudyManifest,
    probe_input: LaunchFloorProbeInput,
    record: LaunchFloorProbeRecord,
) -> LaunchFloorEvidence:
    """Recompute terminal launch-floor evidence from sealed raw measurements."""

    study_directory = Path(record.artifact_directory).expanduser().parent
    _load_study_manifest(study_directory / "study-manifest", manifest)
    _inspect_probe_record(record, probe_input, manifest)
    artifact_sha256 = launch_probe_artifact_sha256(record)
    summary = record.summary
    if not summary.stable or summary.median_ms is None:
        reason = summary.error or f"launch probe timing is {summary.status}"
        return LaunchFloorEvidence(
            launch_study_manifest_sha256=manifest.sha256,
            probe=probe_input.asset_binding,
            artifact_sha256=artifact_sha256,
            max_launch_fraction=manifest.max_launch_fraction,
            status="fail",
            failure_reason=reason,
        )

    launch_ms = summary.median_ms
    launch_summary_sha256 = sha256_json(summary)
    measurements = tuple(
        LaunchTimingMeasurement(
            workload_kind="task",
            workload_id=item.task_id,
            workload_timing_kind=(
                "baseline" if item.selected_source == "l-star" else "expert"
            ),
            target_id=(
                item.selected_baseline_target_id
                if item.selected_source == "l-star"
                else item.core_target_id
            ),
            workload_source_sha256=(
                item.selected_baseline_source_sha256
                if item.selected_source == "l-star"
                else item.core_expert_source_sha256
            ),
            workload_artifact_sha256=(
                item.selected_baseline_artifact_sha256
                if item.selected_source == "l-star"
                else item.core_gate_artifact_sha256
            ),
            workload_timing_summary_sha256=(
                item.selected_timing_summary_sha256
                if item.selected_source == "l-star"
                else item.core_timing_summary_sha256
            ),
            launch_source_sha256=manifest.probe.source_sha256,
            launch_artifact_sha256=artifact_sha256,
            launch_timing_summary_sha256=launch_summary_sha256,
            launch_ms=launch_ms,
            task_ms=item.task_ms,
            launch_fraction=launch_ms / item.task_ms,
        )
        for item in manifest.tasks
    ) + tuple(
        LaunchTimingMeasurement(
            workload_kind="canary",
            workload_id=item.canary_id,
            workload_timing_kind="canary",
            target_id=item.minimum_target_id,
            workload_source_sha256=item.source_sha256,
            workload_artifact_sha256=item.capability_probe_artifact_sha256,
            workload_timing_summary_sha256=(
                item.capability_timing_summary_sha256
            ),
            launch_source_sha256=manifest.probe.source_sha256,
            launch_artifact_sha256=artifact_sha256,
            launch_timing_summary_sha256=launch_summary_sha256,
            launch_ms=launch_ms,
            task_ms=item.task_ms,
            launch_fraction=launch_ms / item.task_ms,
        )
        for item in manifest.canaries
    )
    failed = tuple(
        item
        for item in measurements
        if item.launch_fraction > manifest.max_launch_fraction
    )
    failure_reason = None
    if failed:
        identities = ", ".join(
            f"{item.workload_kind}:{item.workload_id}" for item in failed
        )
        failure_reason = (
            "launch floor exceeds the study latency fraction for workloads: "
            f"{identities}"
        )
    return LaunchFloorEvidence(
        launch_study_manifest_sha256=manifest.sha256,
        probe=probe_input.asset_binding,
        artifact_sha256=artifact_sha256,
        max_launch_fraction=manifest.max_launch_fraction,
        status="fail" if failed else "pass",
        measurements=measurements,
        failure_reason=failure_reason,
    )
