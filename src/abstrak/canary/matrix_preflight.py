"""Hash-bound floor and environment contracts for generic matrix studies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from abstrak.canary.artifacts import (
    TrajectoryArtifactError,
    TrajectoryStore,
    verify_trajectory,
)
from abstrak.canary.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    CanaryModel,
    TimingSpec,
)
from abstrak.canary.manifests import PinnedStudySpec
from abstrak.canary.matrix import MatrixSchedule
from abstrak.canary.matrix_runner import MatrixExecutionContext, MatrixTransportContext
from abstrak.providers.contracts import sha256_json

FloorStatus = Literal["pending", "invalid", "valid"]
EnvironmentStatus = Literal["pending", "invalid", "verified"]
BaselineTimingStatus = Literal[
    "stable",
    "unstable",
    "worker_failure",
    "correctness_failure",
]
TerminalEvidenceStatus = Literal["pass", "fail"]

FORMAL_FLOOR_TIMING = TimingSpec(
    warmup_runs=25,
    trial_runs=200,
    repetitions=3,
    max_cv=0.05,
)

_SEMANTIC_FILES = frozenset(
    {
        "asset-manifest.json",
        "environment-manifest.json",
        "floor-manifest.json",
        "execution-context.json",
        "preflight-receipt.json",
    }
)
_STORE_DIRECTORIES = frozenset({"events", "turns", "candidates", "sealed"})


class MatrixPreflightError(ValueError):
    """Raised when a matrix preflight artifact is incomplete or inconsistent."""


class StudyBoundModel(CanaryModel):
    """Shared raw/spec/schedule identity carried by every preflight manifest."""

    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    raw_study_sha256: str = Field(pattern=SHA256_PATTERN)
    spec_sha256: str = Field(pattern=SHA256_PATTERN)
    schedule_sha256: str = Field(pattern=SHA256_PATTERN)

    @property
    def study_binding(self) -> tuple[str, str, str, str]:
        return (
            self.study_id,
            self.raw_study_sha256,
            self.spec_sha256,
            self.schedule_sha256,
        )


def _study_binding(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
) -> tuple[str, str, str, str]:
    if schedule.spec != pinned.spec or schedule.spec_sha256 != pinned.spec.sha256:
        raise MatrixPreflightError("matrix schedule differs from the pinned study spec")
    return (
        pinned.spec.study_id,
        pinned.sha256,
        pinned.spec.sha256,
        schedule.sha256,
    )


def _study_fields(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
) -> dict[str, str]:
    study_id, raw_sha256, spec_sha256, schedule_sha256 = _study_binding(pinned, schedule)
    return {
        "study_id": study_id,
        "raw_study_sha256": raw_sha256,
        "spec_sha256": spec_sha256,
        "schedule_sha256": schedule_sha256,
    }


def _require_study_binding(
    value: StudyBoundModel,
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    *,
    label: str,
) -> None:
    if value.study_binding != _study_binding(pinned, schedule):
        raise MatrixPreflightError(f"{label} differs from the pinned study identity")


def _task_ids(pinned: PinnedStudySpec) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(task_id for phase in pinned.spec.phases for task_id in phase.task_ids)
    )


class BaselineAssetBinding(CanaryModel):
    """One content-addressed common baseline implementation."""

    variant: str = Field(pattern=IDENTIFIER_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)


class TaskAssetBinding(CanaryModel):
    """All frozen source identities needed to qualify one study task."""

    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_pack_sha256: str = Field(pattern=SHA256_PATTERN)
    reference_source_sha256: str = Field(pattern=SHA256_PATTERN)
    expert_source_sha256: str = Field(pattern=SHA256_PATTERN)
    baselines: tuple[BaselineAssetBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def baseline_variants_are_unique(self) -> TaskAssetBinding:
        variants = tuple(item.variant for item in self.baselines)
        if len(variants) != len(set(variants)):
            raise ValueError("task baseline variants must be unique")
        return self


class TargetAssetBinding(CanaryModel):
    """Frozen target stack and rendered-card identities for one matrix target."""

    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_stack_sha256: str = Field(pattern=SHA256_PATTERN)
    card_sha256: str = Field(pattern=SHA256_PATTERN)


class CanaryAssetBinding(CanaryModel):
    """One capability canary source and the target validators it exercises."""

    canary_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    required_target_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("required_target_ids")
    @classmethod
    def required_targets_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("canary required target IDs must be unique")
        return values


class AssetManifest(StudyBoundModel):
    """Content-addressed assets whose exact coverage is checked against a study."""

    schema_version: Literal["abstrak-matrix-asset-manifest.v1"] = "abstrak-matrix-asset-manifest.v1"
    tasks: tuple[TaskAssetBinding, ...] = Field(min_length=1)
    targets: tuple[TargetAssetBinding, ...] = Field(min_length=1)
    canaries: tuple[CanaryAssetBinding, ...] = ()

    @model_validator(mode="after")
    def identities_are_unique_and_references_are_internal(self) -> AssetManifest:
        task_ids = tuple(item.task_id for item in self.tasks)
        target_ids = tuple(item.target_id for item in self.targets)
        canary_ids = tuple(item.canary_id for item in self.canaries)
        for label, values in (
            ("task", task_ids),
            ("target", target_ids),
            ("canary", canary_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"asset {label} IDs must be unique")
        task_set = set(task_ids)
        target_set = set(target_ids)
        for canary in self.canaries:
            if canary.task_id not in task_set:
                raise ValueError("canary references an undeclared task")
            if not set(canary.required_target_ids).issubset(target_set):
                raise ValueError("canary references an undeclared target")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class EnvironmentObservation(CanaryModel):
    """Complete remote observation produced by one environment probe."""

    schema_version: Literal["abstrak-matrix-environment-observation.v1"] = (
        "abstrak-matrix-environment-observation.v1"
    )
    worker_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    transport: MatrixTransportContext
    accelerator: str = Field(min_length=1)
    compute_capability: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    python_version: str = Field(min_length=1)
    tilelang_version: str = Field(min_length=1)
    triton_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    cuda_version: str = Field(min_length=1)
    driver_version: str = Field(min_length=1)
    cache_policy: Literal["per-job-temporary"] = "per-job-temporary"
    gpu_jobs_serial: Literal[True] = True
    generated_code_remote_only: Literal[True] = True
    non_container_worker: Literal[True] = True


class EnvironmentProbeEvidence(CanaryModel):
    """Terminal environment probe with its full observation, not only a digest."""

    schema_version: Literal["abstrak-matrix-environment-probe-evidence.v1"] = (
        "abstrak-matrix-environment-probe-evidence.v1"
    )
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    status: TerminalEvidenceStatus
    observation: EnvironmentObservation | None = None
    failure_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def terminal_status_has_complete_payload(self) -> EnvironmentProbeEvidence:
        if self.status == "pass":
            if self.observation is None or self.failure_reason is not None:
                raise ValueError("passing environment probe requires an observation and no reason")
        elif self.failure_reason is None:
            raise ValueError("failed environment probe requires a reason")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class EnvironmentManifest(StudyBoundModel):
    """Expected or verified worker environment bound to one matrix study."""

    schema_version: Literal["abstrak-matrix-environment-manifest.v1"] = (
        "abstrak-matrix-environment-manifest.v1"
    )
    status: EnvironmentStatus
    controller_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    worker_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    transport: MatrixTransportContext
    accelerator: str = Field(min_length=1)
    compute_capability: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    python_version: str = Field(min_length=1)
    tilelang_version: str = Field(min_length=1)
    triton_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    cuda_version: str = Field(min_length=1)
    driver_version: str = Field(min_length=1)
    cache_policy: Literal["per-job-temporary"] = "per-job-temporary"
    gpu_jobs_serial: Literal[True] = True
    generated_code_remote_only: Literal[True] = True
    non_container_worker: Literal[True] = True
    verification_evidence: EnvironmentProbeEvidence | None = None
    invalid_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def status_matches_evidence(self) -> EnvironmentManifest:
        if self.status == "pending":
            if self.verification_evidence is not None or self.invalid_reason is not None:
                raise ValueError("pending environment cannot contain verification evidence")
        elif self.status == "verified":
            if (
                self.verification_evidence is None
                or self.verification_evidence.status != "pass"
                or self.invalid_reason is not None
            ):
                raise ValueError("verified environment requires evidence and no invalid reason")
            expected = EnvironmentObservation(
                worker_revision=self.worker_revision,
                transport=self.transport,
                accelerator=self.accelerator,
                compute_capability=self.compute_capability,
                python_version=self.python_version,
                tilelang_version=self.tilelang_version,
                triton_version=self.triton_version,
                torch_version=self.torch_version,
                cuda_version=self.cuda_version,
                driver_version=self.driver_version,
                cache_policy=self.cache_policy,
                gpu_jobs_serial=self.gpu_jobs_serial,
                generated_code_remote_only=self.generated_code_remote_only,
                non_container_worker=self.non_container_worker,
            )
            if self.verification_evidence.observation != expected:
                raise ValueError("verified environment differs from its probe observation")
        elif (
            self.verification_evidence is None
            or self.verification_evidence.status != "fail"
            or self.invalid_reason is None
            or self.invalid_reason != self.verification_evidence.failure_reason
        ):
            raise ValueError("invalid environment requires matching failed evidence and a reason")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class ExpertCorrectnessEvidence(CanaryModel):
    """Terminal sealed correctness result for one frozen expert source."""

    schema_version: Literal["abstrak-matrix-expert-correctness-evidence.v1"] = (
        "abstrak-matrix-expert-correctness-evidence.v1"
    )
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_pack_sha256: str = Field(pattern=SHA256_PATTERN)
    expert_source_sha256: str = Field(pattern=SHA256_PATTERN)
    status: TerminalEvidenceStatus
    compiled: bool
    all_sealed_cases_passed: bool
    output_finite: bool
    inputs_unchanged: bool
    fallback_free: bool
    failure_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def status_is_derived_from_checks(self) -> ExpertCorrectnessEvidence:
        passed = all(
            (
                self.compiled,
                self.all_sealed_cases_passed,
                self.output_finite,
                self.inputs_unchanged,
                self.fallback_free,
            )
        )
        expected: TerminalEvidenceStatus = "pass" if passed else "fail"
        if self.status != expected:
            raise ValueError(f"expert correctness status must be {expected}")
        if (self.status == "pass") != (self.failure_reason is None):
            raise ValueError("failed expert correctness requires exactly one failure reason")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class TargetCodegenEvidence(CanaryModel):
    """Terminal validator result and generated code for one expert/target pair."""

    schema_version: Literal["abstrak-matrix-target-codegen-evidence.v1"] = (
        "abstrak-matrix-target-codegen-evidence.v1"
    )
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_pack_sha256: str = Field(pattern=SHA256_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_stack_sha256: str = Field(pattern=SHA256_PATTERN)
    expert_source_sha256: str = Field(pattern=SHA256_PATTERN)
    status: TerminalEvidenceStatus
    compiled: bool
    correct: bool
    fallback_free: bool
    generated_code_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    failure_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def status_is_derived_from_codegen_result(self) -> TargetCodegenEvidence:
        passed = (
            self.compiled
            and self.correct
            and self.fallback_free
            and self.generated_code_sha256 is not None
        )
        expected: TerminalEvidenceStatus = "pass" if passed else "fail"
        if self.status != expected:
            raise ValueError(f"target codegen status must be {expected}")
        if (self.status == "pass") != (self.failure_reason is None):
            raise ValueError("failed target codegen requires exactly one failure reason")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class BaselineTimingEvidence(CanaryModel):
    """Terminal timing evidence for one frozen baseline source."""

    variant: str = Field(pattern=IDENTIFIER_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    timing_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    status: BaselineTimingStatus
    median_ms: float | None = Field(default=None, gt=0)

    @field_validator("median_ms")
    @classmethod
    def median_is_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("baseline median must be finite")
        return value

    @model_validator(mode="after")
    def stable_status_has_the_only_publishable_median(self) -> BaselineTimingEvidence:
        if (self.status == "stable") != (self.median_ms is not None):
            raise ValueError("only stable baseline evidence may expose a median")
        return self


class VerifiedTaskFloorEvidence(CanaryModel):
    """Explicit correctness, codegen, and baseline evidence for a valid task floor."""

    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    expert_source_sha256: str = Field(pattern=SHA256_PATTERN)
    expert_correctness: ExpertCorrectnessEvidence
    target_codegen: tuple[TargetCodegenEvidence, ...] = Field(min_length=1)
    baseline_timings: tuple[BaselineTimingEvidence, ...] = Field(min_length=1)
    selected_baseline_variant: str = Field(pattern=IDENTIFIER_PATTERN)
    selected_baseline_source_sha256: str = Field(pattern=SHA256_PATTERN)
    selected_timing_summary_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def selections_and_codegen_are_verified(self) -> VerifiedTaskFloorEvidence:
        if (
            self.expert_correctness.status != "pass"
            or self.expert_correctness.task_id != self.task_id
            or self.expert_correctness.expert_source_sha256 != self.expert_source_sha256
        ):
            raise ValueError("expert correctness differs from the task floor identity")
        target_ids = tuple(item.target_id for item in self.target_codegen)
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("task codegen evidence target IDs must be unique")
        if any(
            item.status != "pass"
            or item.task_id != self.task_id
            or item.task_pack_sha256 != self.expert_correctness.task_pack_sha256
            or item.expert_source_sha256 != self.expert_source_sha256
            for item in self.target_codegen
        ):
            raise ValueError("target codegen differs from the passing expert identity")
        generated = {item.generated_code_sha256 for item in self.target_codegen}
        if len(generated) != 1:
            raise ValueError("one expert must generate identical code under every target validator")

        variants = tuple(item.variant for item in self.baseline_timings)
        if len(variants) != len(set(variants)):
            raise ValueError("task baseline timing variants must be unique")
        selected = tuple(
            item for item in self.baseline_timings if item.variant == self.selected_baseline_variant
        )
        if len(selected) != 1 or selected[0].status != "stable":
            raise ValueError("selected L_i* baseline must have stable evidence")
        chosen = selected[0]
        if (
            chosen.source_sha256 != self.selected_baseline_source_sha256
            or chosen.timing_summary_sha256 != self.selected_timing_summary_sha256
        ):
            raise ValueError("selected baseline identity differs from its timing evidence")
        stable_medians = tuple(
            item.median_ms for item in self.baseline_timings if item.status == "stable"
        )
        assert chosen.median_ms is not None
        if chosen.median_ms != min(value for value in stable_medians if value is not None):
            raise ValueError("selected L_i* baseline is not the fastest stable baseline")
        return self

    @property
    def l_star_ms(self) -> float:
        selected = next(
            item for item in self.baseline_timings if item.variant == self.selected_baseline_variant
        )
        assert selected.median_ms is not None
        return selected.median_ms

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class LatencyCeilingDerivation(CanaryModel):
    """Auditable derivation of the Agent early-stop latency ceiling."""

    formula: Literal["l_star_ms * competitive_factor"] = "l_star_ms * competitive_factor"
    l_star_ms: float = Field(gt=0)
    competitive_factor: float = Field(ge=1)
    latency_ceiling_ms: float = Field(gt=0)

    @field_validator("l_star_ms", "competitive_factor", "latency_ceiling_ms")
    @classmethod
    def values_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("latency ceiling inputs must be finite")
        return value

    @model_validator(mode="after")
    def ceiling_matches_inputs(self) -> LatencyCeilingDerivation:
        expected = self.l_star_ms * self.competitive_factor
        if not math.isclose(self.latency_ceiling_ms, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("latency ceiling does not match L_i* and competitive factor")
        return self


class TaskFloorRecord(CanaryModel):
    """Pending, invalid, or explicitly verified floor for one study task."""

    schema_version: Literal["abstrak-matrix-task-floor.v1"] = "abstrak-matrix-task-floor.v1"
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: FloorStatus
    expert_source_sha256: str = Field(pattern=SHA256_PATTERN)
    verified_evidence: VerifiedTaskFloorEvidence | None = None
    ceiling: LatencyCeilingDerivation | None = None
    invalid_evidence_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    invalid_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def status_has_exactly_the_allowed_payload(self) -> TaskFloorRecord:
        if self.status == "pending":
            if any(
                value is not None
                for value in (
                    self.verified_evidence,
                    self.ceiling,
                    self.invalid_evidence_sha256,
                    self.invalid_reason,
                )
            ):
                raise ValueError("pending task floor cannot contain verification evidence")
        elif self.status == "invalid":
            if (
                self.invalid_evidence_sha256 is None
                or self.invalid_reason is None
                or self.verified_evidence is not None
                or self.ceiling is not None
            ):
                raise ValueError("invalid task floor requires only invalid evidence and a reason")
        else:
            if (
                self.verified_evidence is None
                or self.ceiling is None
                or self.invalid_evidence_sha256 is not None
                or self.invalid_reason is not None
            ):
                raise ValueError("valid task floor requires explicit verified evidence and ceiling")
            if (
                self.verified_evidence.task_id != self.task_id
                or self.verified_evidence.expert_source_sha256 != self.expert_source_sha256
            ):
                raise ValueError("valid task evidence differs from its floor identity")
            if not math.isclose(
                self.ceiling.l_star_ms,
                self.verified_evidence.l_star_ms,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("floor L_i* differs from the selected stable baseline")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class CapabilityTargetEvidence(CanaryModel):
    """Terminal capability-canary result for one required target validator."""

    schema_version: Literal["abstrak-matrix-capability-target-evidence.v1"] = (
        "abstrak-matrix-capability-target-evidence.v1"
    )
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_stack_sha256: str = Field(pattern=SHA256_PATTERN)
    status: TerminalEvidenceStatus
    compiled: bool
    correct: bool
    used_capabilities: tuple[str, ...] = ()
    generated_code_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    control_generated_code_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    failure_reason: str | None = Field(default=None, min_length=1)

    @field_validator("used_capabilities")
    @classmethod
    def used_capability_ids_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value or value.strip() != value for value in values):
            raise ValueError("used capability IDs must be non-empty normalized strings")
        if len(values) != len(set(values)):
            raise ValueError("used capability IDs must be unique")
        return values

    @model_validator(mode="after")
    def status_is_derived_from_capability_result(self) -> CapabilityTargetEvidence:
        codegen_is_distinct = (
            self.generated_code_sha256 is not None
            and self.control_generated_code_sha256 is not None
            and self.generated_code_sha256 != self.control_generated_code_sha256
        )
        passed = (
            self.compiled
            and self.correct
            and bool(self.used_capabilities)
            and codegen_is_distinct
        )
        expected: TerminalEvidenceStatus = "pass" if passed else "fail"
        if self.status != expected:
            raise ValueError(f"capability target status must be {expected}")
        if (self.status == "pass") != (self.failure_reason is None):
            raise ValueError("failed capability target requires exactly one failure reason")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class CapabilityCanaryEvidence(CanaryModel):
    """Exact terminal evidence for one frozen capability-canary source."""

    schema_version: Literal["abstrak-matrix-capability-canary-evidence.v1"] = (
        "abstrak-matrix-capability-canary-evidence.v1"
    )
    canary_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    status: TerminalEvidenceStatus
    targets: tuple[CapabilityTargetEvidence, ...] = Field(min_length=1)
    failure_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def status_is_derived_from_target_evidence(self) -> CapabilityCanaryEvidence:
        target_ids = tuple(item.target_id for item in self.targets)
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("capability canary target IDs must be unique")
        expected: TerminalEvidenceStatus = (
            "pass" if all(item.status == "pass" for item in self.targets) else "fail"
        )
        if self.status != expected:
            raise ValueError(f"capability canary status must be {expected}")
        if (self.status == "pass") != (self.failure_reason is None):
            raise ValueError("failed capability canary requires exactly one failure reason")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class LaunchTimingMeasurement(CanaryModel):
    """Measured launch and task time for one task or capability canary."""

    workload_kind: Literal["task", "canary"]
    workload_id: str = Field(pattern=IDENTIFIER_PATTERN)
    launch_ms: float = Field(gt=0)
    task_ms: float = Field(gt=0)

    @field_validator("launch_ms", "task_ms")
    @classmethod
    def timing_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("launch-floor timings must be finite")
        return value


class LaunchFloorEvidence(CanaryModel):
    """Terminal launch-floor assessment with the measurements used to decide it."""

    schema_version: Literal["abstrak-matrix-launch-floor-evidence.v1"] = (
        "abstrak-matrix-launch-floor-evidence.v1"
    )
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    status: TerminalEvidenceStatus
    measurements: tuple[LaunchTimingMeasurement, ...] = ()
    failure_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def terminal_status_has_measurements_or_reason(self) -> LaunchFloorEvidence:
        identities = tuple(
            (item.workload_kind, item.workload_id) for item in self.measurements
        )
        if len(identities) != len(set(identities)):
            raise ValueError("launch-floor measurement identities must be unique")
        if self.status == "pass":
            if not self.measurements or self.failure_reason is not None:
                raise ValueError("passing launch floor requires measurements and no reason")
        elif self.failure_reason is None:
            raise ValueError("failed launch floor requires a reason")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class FloorManifest(StudyBoundModel):
    """Study-wide floor whose status is uniquely derived from its evidence."""

    schema_version: Literal["abstrak-matrix-floor-manifest.v1"] = "abstrak-matrix-floor-manifest.v1"
    status: FloorStatus
    asset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    timing: TimingSpec
    tasks: tuple[TaskFloorRecord, ...] = Field(min_length=1)
    capability_canaries: tuple[CapabilityCanaryEvidence, ...] = ()
    launch_floor: LaunchFloorEvidence | None = None

    @model_validator(mode="after")
    def status_is_derived_from_complete_evidence(self) -> FloorManifest:
        if self.timing != FORMAL_FLOOR_TIMING:
            raise ValueError("floor timing must use the frozen 25/200/3 protocol")
        task_ids = tuple(item.task_id for item in self.tasks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("floor task IDs must be unique")
        canary_ids = tuple(item.canary_id for item in self.capability_canaries)
        if len(canary_ids) != len(set(canary_ids)):
            raise ValueError("floor capability canary IDs must be unique")
        terminal_failure = any(item.status == "invalid" for item in self.tasks) or any(
            item.status == "fail" for item in self.capability_canaries
        )
        terminal_failure = terminal_failure or (
            self.launch_floor is not None and self.launch_floor.status == "fail"
        )
        if terminal_failure:
            expected: FloorStatus = "invalid"
        elif (
            all(item.status == "valid" for item in self.tasks)
            and all(item.status == "pass" for item in self.capability_canaries)
            and self.launch_floor is not None
            and self.launch_floor.status == "pass"
        ):
            expected = "valid"
        else:
            expected = "pending"
        if self.status != expected:
            raise ValueError(f"floor status must be {expected} for the supplied evidence")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class PreflightEvidenceClosure(CanaryModel):
    """All typed terminal evidence that makes a preflight receipt ready."""

    schema_version: Literal["abstrak-matrix-preflight-evidence-closure.v1"] = (
        "abstrak-matrix-preflight-evidence-closure.v1"
    )
    environment_probe: EnvironmentProbeEvidence
    task_floors: tuple[VerifiedTaskFloorEvidence, ...] = Field(min_length=1)
    capability_canaries: tuple[CapabilityCanaryEvidence, ...] = ()
    launch_floor: LaunchFloorEvidence

    @model_validator(mode="after")
    def contains_only_passing_terminal_evidence(self) -> PreflightEvidenceClosure:
        if self.environment_probe.status != "pass":
            raise ValueError("evidence closure requires a passing environment probe")
        if any(
            item.expert_correctness.status != "pass"
            or any(target.status != "pass" for target in item.target_codegen)
            for item in self.task_floors
        ):
            raise ValueError("evidence closure requires passing task floor evidence")
        if any(item.status != "pass" for item in self.capability_canaries):
            raise ValueError("evidence closure requires passing capability canaries")
        if self.launch_floor.status != "pass":
            raise ValueError("evidence closure requires a passing launch floor")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _build_evidence_closure(
    environment: EnvironmentManifest,
    floor: FloorManifest,
) -> PreflightEvidenceClosure:
    if environment.status != "verified" or environment.verification_evidence is None:
        raise MatrixPreflightError("evidence closure requires a verified environment")
    if floor.status != "valid" or floor.launch_floor is None:
        raise MatrixPreflightError("evidence closure requires a valid floor")
    task_evidence = tuple(item.verified_evidence for item in floor.tasks)
    if any(item is None for item in task_evidence):
        raise MatrixPreflightError("evidence closure is missing task floor evidence")
    return PreflightEvidenceClosure(
        environment_probe=environment.verification_evidence,
        task_floors=tuple(item for item in task_evidence if item is not None),
        capability_canaries=floor.capability_canaries,
        launch_floor=floor.launch_floor,
    )


class PreflightReceipt(StudyBoundModel):
    """Ready-only receipt binding verified floor inputs to one execution context."""

    schema_version: Literal["abstrak-matrix-preflight-receipt.v1"] = (
        "abstrak-matrix-preflight-receipt.v1"
    )
    status: Literal["ready"] = "ready"
    asset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    floor_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_context_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_closure_sha256: str = Field(pattern=SHA256_PATTERN)
    task_ids: tuple[str, ...] = Field(min_length=1)
    target_ids: tuple[str, ...] = Field(min_length=1)

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class PreflightBundle(CanaryModel):
    """The five semantic records stored in one sealed preflight directory."""

    schema_version: Literal["abstrak-matrix-preflight-bundle.v1"] = (
        "abstrak-matrix-preflight-bundle.v1"
    )
    assets: AssetManifest
    environment: EnvironmentManifest
    floor: FloorManifest
    execution_context: MatrixExecutionContext
    receipt: PreflightReceipt

    @model_validator(mode="after")
    def internal_hashes_and_bindings_match(self) -> PreflightBundle:
        if not (
            self.assets.study_binding
            == self.environment.study_binding
            == self.floor.study_binding
            == self.receipt.study_binding
        ):
            raise ValueError("preflight bundle records have different study identities")
        expected_hashes = (
            self.assets.sha256,
            self.floor.sha256,
            self.environment.sha256,
            self.execution_context.sha256,
        )
        receipt_hashes = (
            self.receipt.asset_manifest_sha256,
            self.receipt.floor_manifest_sha256,
            self.receipt.environment_manifest_sha256,
            self.receipt.execution_context_sha256,
        )
        if receipt_hashes != expected_hashes:
            raise ValueError("preflight receipt hashes differ from the bundled records")
        try:
            closure = _build_evidence_closure(self.environment, self.floor)
        except MatrixPreflightError as error:
            raise ValueError("preflight bundle has an incomplete evidence closure") from error
        if self.receipt.evidence_closure_sha256 != closure.sha256:
            raise ValueError("preflight receipt differs from its evidence closure")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def build_asset_manifest(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    *,
    tasks: tuple[TaskAssetBinding, ...],
    targets: tuple[TargetAssetBinding, ...],
    canaries: tuple[CanaryAssetBinding, ...] = (),
) -> AssetManifest:
    """Construct an asset manifest only when task and target coverage is exact."""

    manifest = AssetManifest(
        **_study_fields(pinned, schedule),
        tasks=tasks,
        targets=targets,
        canaries=canaries,
    )
    _validate_asset_coverage(manifest, pinned)
    return manifest


def build_pending_environment(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    *,
    controller_revision: str,
    worker_revision: str,
    transport: MatrixTransportContext,
    accelerator: str,
    compute_capability: str,
    python_version: str,
    tilelang_version: str,
    triton_version: str,
    torch_version: str,
    cuda_version: str,
    driver_version: str,
) -> EnvironmentManifest:
    """Freeze expected environment inputs without claiming that they were observed."""

    return EnvironmentManifest(
        **_study_fields(pinned, schedule),
        status="pending",
        controller_revision=controller_revision,
        worker_revision=worker_revision,
        transport=transport,
        accelerator=accelerator,
        compute_capability=compute_capability,
        python_version=python_version,
        tilelang_version=tilelang_version,
        triton_version=triton_version,
        torch_version=torch_version,
        cuda_version=cuda_version,
        driver_version=driver_version,
    )


def build_pending_floor(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    *,
    assets: AssetManifest,
    environment: EnvironmentManifest,
) -> FloorManifest:
    """Build the only evidence-free floor state; it is always pending."""

    _require_study_binding(assets, pinned, schedule, label="asset manifest")
    _require_study_binding(environment, pinned, schedule, label="environment manifest")
    _validate_asset_coverage(assets, pinned)
    return FloorManifest(
        **_study_fields(pinned, schedule),
        status="pending",
        asset_manifest_sha256=assets.sha256,
        environment_manifest_sha256=environment.sha256,
        timing=FORMAL_FLOOR_TIMING,
        tasks=tuple(
            TaskFloorRecord(
                task_id=task.task_id,
                status="pending",
                expert_source_sha256=task.expert_source_sha256,
            )
            for task in assets.tasks
        ),
    )


def build_preflight_receipt(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    *,
    assets: AssetManifest,
    floor: FloorManifest,
    environment: EnvironmentManifest,
    execution_context: MatrixExecutionContext,
) -> PreflightReceipt:
    """Validate all evidence and return the ready receipt used to authorize execution."""

    _validate_preflight_inputs(
        pinned,
        schedule,
        assets=assets,
        floor=floor,
        environment=environment,
        execution_context=execution_context,
    )
    return PreflightReceipt(
        **_study_fields(pinned, schedule),
        asset_manifest_sha256=assets.sha256,
        floor_manifest_sha256=floor.sha256,
        environment_manifest_sha256=environment.sha256,
        execution_context_sha256=execution_context.sha256,
        evidence_closure_sha256=_build_evidence_closure(environment, floor).sha256,
        task_ids=tuple(task.task_id for task in assets.tasks),
        target_ids=tuple(target.target_id for target in assets.targets),
    )


def _validate_asset_coverage(assets: AssetManifest, pinned: PinnedStudySpec) -> None:
    expected_tasks = _task_ids(pinned)
    actual_tasks = tuple(item.task_id for item in assets.tasks)
    if actual_tasks != expected_tasks:
        raise MatrixPreflightError("asset tasks do not exactly cover the study in phase order")
    actual_targets = tuple(item.target_id for item in assets.targets)
    if actual_targets != pinned.spec.targets:
        raise MatrixPreflightError("asset targets do not exactly cover the study in axis order")


def _validate_floor_evidence(
    pinned: PinnedStudySpec,
    *,
    assets: AssetManifest,
    floor: FloorManifest,
) -> None:
    asset_tasks = {item.task_id: item for item in assets.tasks}
    if tuple(item.task_id for item in floor.tasks) != tuple(asset_tasks):
        raise MatrixPreflightError("floor tasks do not exactly cover the asset tasks")
    if pinned.spec.gate is None:
        raise MatrixPreflightError("a valid latency floor requires a study gate")
    expected_factor = pinned.spec.gate.metrics.competitive_latency_factor
    expected_targets = tuple(item.target_id for item in assets.targets)
    target_assets = {item.target_id: item for item in assets.targets}
    for record in floor.tasks:
        asset = asset_tasks[record.task_id]
        if record.expert_source_sha256 != asset.expert_source_sha256:
            raise MatrixPreflightError("floor expert source differs from the asset manifest")
        if record.status != "valid":
            continue
        assert record.verified_evidence is not None
        assert record.ceiling is not None
        evidence = record.verified_evidence
        correctness = evidence.expert_correctness
        if (
            correctness.task_id != asset.task_id
            or correctness.task_pack_sha256 != asset.task_pack_sha256
            or correctness.expert_source_sha256 != asset.expert_source_sha256
            or correctness.status != "pass"
        ):
            raise MatrixPreflightError("expert correctness evidence differs from frozen assets")
        if tuple(item.target_id for item in evidence.target_codegen) != expected_targets:
            raise MatrixPreflightError("expert codegen evidence does not cover every target")
        for codegen in evidence.target_codegen:
            target = target_assets[codegen.target_id]
            if (
                codegen.status != "pass"
                or codegen.task_id != asset.task_id
                or codegen.task_pack_sha256 != asset.task_pack_sha256
                or codegen.expert_source_sha256 != asset.expert_source_sha256
                or codegen.target_stack_sha256 != target.target_stack_sha256
            ):
                raise MatrixPreflightError("expert codegen evidence differs from frozen assets")
        expected_baselines = tuple((item.variant, item.source_sha256) for item in asset.baselines)
        actual_baselines = tuple(
            (item.variant, item.source_sha256) for item in evidence.baseline_timings
        )
        if actual_baselines != expected_baselines:
            raise MatrixPreflightError("baseline evidence does not cover every frozen baseline")
        if not math.isclose(
            record.ceiling.competitive_factor,
            expected_factor,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise MatrixPreflightError("floor competitive factor differs from the study gate")

    expected_canaries = tuple(item.canary_id for item in assets.canaries)
    actual_canaries = tuple(item.canary_id for item in floor.capability_canaries)
    if actual_canaries != expected_canaries:
        raise MatrixPreflightError("capability evidence does not exactly cover frozen canaries")
    for evidence, canary in zip(floor.capability_canaries, assets.canaries, strict=True):
        if (
            evidence.status != "pass"
            or evidence.task_id != canary.task_id
            or evidence.source_sha256 != canary.source_sha256
            or tuple(item.target_id for item in evidence.targets)
            != canary.required_target_ids
        ):
            raise MatrixPreflightError("capability canary evidence differs from frozen assets")
        for target_evidence in evidence.targets:
            if (
                target_evidence.status != "pass"
                or target_evidence.target_stack_sha256
                != target_assets[target_evidence.target_id].target_stack_sha256
            ):
                raise MatrixPreflightError("capability target evidence differs from frozen assets")

    if floor.launch_floor is None or floor.launch_floor.status != "pass":
        raise MatrixPreflightError("valid floor requires passing launch-floor evidence")
    expected_launch_coverage = tuple(("task", item.task_id) for item in assets.tasks) + tuple(
        ("canary", item.canary_id) for item in assets.canaries
    )
    actual_launch_coverage = tuple(
        (item.workload_kind, item.workload_id)
        for item in floor.launch_floor.measurements
    )
    if actual_launch_coverage != expected_launch_coverage:
        raise MatrixPreflightError("launch-floor evidence does not exactly cover frozen workloads")


def _validate_preflight_inputs(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    *,
    assets: AssetManifest,
    floor: FloorManifest,
    environment: EnvironmentManifest,
    execution_context: MatrixExecutionContext,
) -> None:
    for label, value in (
        ("asset manifest", assets),
        ("floor manifest", floor),
        ("environment manifest", environment),
    ):
        _require_study_binding(value, pinned, schedule, label=label)
    _validate_asset_coverage(assets, pinned)
    if floor.asset_manifest_sha256 != assets.sha256:
        raise MatrixPreflightError("floor does not reference the supplied asset manifest")
    if floor.environment_manifest_sha256 != environment.sha256:
        raise MatrixPreflightError("floor does not reference the supplied environment manifest")
    if floor.status != "valid":
        raise MatrixPreflightError("matrix execution requires a valid floor")
    if environment.status != "verified":
        raise MatrixPreflightError("matrix execution requires a verified environment")
    _validate_floor_evidence(pinned, assets=assets, floor=floor)

    if execution_context.controller_revision != environment.controller_revision:
        raise MatrixPreflightError("execution context controller revision mismatch")
    if execution_context.worker_revision != environment.worker_revision:
        raise MatrixPreflightError("execution context worker revision mismatch")
    if execution_context.transport != environment.transport:
        raise MatrixPreflightError("execution context transport mismatch")
    expected_hashes = (assets.sha256, floor.sha256, environment.sha256)
    context_hashes = (
        execution_context.asset_manifest_sha256,
        execution_context.floor_manifest_sha256,
        execution_context.environment_manifest_sha256,
    )
    if context_hashes != expected_hashes:
        raise MatrixPreflightError("execution context manifest hashes mismatch")
    environment_policy = (
        environment.cache_policy,
        environment.gpu_jobs_serial,
        environment.generated_code_remote_only,
        environment.non_container_worker,
    )
    context_policy = (
        execution_context.cache_policy,
        execution_context.gpu_jobs_serial,
        execution_context.generated_code_remote_only,
        execution_context.non_container_worker,
    )
    if context_policy != environment_policy:
        raise MatrixPreflightError("execution context isolation policy mismatch")


def seal_preflight_bundle(
    root: str | Path,
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    *,
    assets: AssetManifest,
    floor: FloorManifest,
    environment: EnvironmentManifest,
    execution_context: MatrixExecutionContext,
    trajectory_id: str = "matrix-preflight",
) -> Path:
    """Write and seal one ready preflight bundle with no untyped side files."""

    receipt = build_preflight_receipt(
        pinned,
        schedule,
        assets=assets,
        floor=floor,
        environment=environment,
        execution_context=execution_context,
    )
    try:
        store = TrajectoryStore.create(root, pinned.spec.study_id, trajectory_id)
        store.write_json("asset-manifest.json", assets)
        store.write_json("environment-manifest.json", environment)
        store.write_json("floor-manifest.json", floor)
        store.write_json("execution-context.json", execution_context)
        store.write_json("preflight-receipt.json", receipt)
        store.seal()
    except (OSError, TrajectoryArtifactError) as error:
        raise MatrixPreflightError("cannot write sealed matrix preflight bundle") from error
    return store.run_directory


def _validate_artifact_shape(directory: Path) -> None:
    if directory.is_symlink():
        raise MatrixPreflightError("preflight artifact directory cannot be a symbolic link")
    if not directory.is_dir():
        raise MatrixPreflightError("preflight artifact directory does not exist")
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise MatrixPreflightError("preflight artifact cannot contain symbolic links")
    actual_files = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    }
    expected_files = _SEMANTIC_FILES | {"sha256sums.txt"}
    if actual_files != expected_files:
        raise MatrixPreflightError("preflight artifact has missing or extra semantic files")
    actual_directories = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_dir()
    }
    if actual_directories != _STORE_DIRECTORIES:
        raise MatrixPreflightError("preflight artifact has missing or extra store directories")


def load_preflight_bundle(
    directory: str | Path,
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    *,
    execution_context: MatrixExecutionContext | None = None,
) -> PreflightBundle:
    """Verify and strictly parse a bundle, optionally checking an expected context."""

    root = Path(directory).expanduser()
    _validate_artifact_shape(root)
    try:
        verify_trajectory(root)
        bundle = PreflightBundle(
            assets=AssetManifest.model_validate_json(
                (root / "asset-manifest.json").read_text(encoding="utf-8")
            ),
            environment=EnvironmentManifest.model_validate_json(
                (root / "environment-manifest.json").read_text(encoding="utf-8")
            ),
            floor=FloorManifest.model_validate_json(
                (root / "floor-manifest.json").read_text(encoding="utf-8")
            ),
            execution_context=MatrixExecutionContext.model_validate_json(
                (root / "execution-context.json").read_text(encoding="utf-8")
            ),
            receipt=PreflightReceipt.model_validate_json(
                (root / "preflight-receipt.json").read_text(encoding="utf-8")
            ),
        )
    except (OSError, ValidationError, ValueError, TrajectoryArtifactError) as error:
        raise MatrixPreflightError("sealed matrix preflight bundle is invalid") from error
    if execution_context is not None and bundle.execution_context != execution_context:
        raise MatrixPreflightError("sealed execution context differs from the expected context")
    expected_receipt = build_preflight_receipt(
        pinned,
        schedule,
        assets=bundle.assets,
        floor=bundle.floor,
        environment=bundle.environment,
        execution_context=bundle.execution_context,
    )
    if bundle.receipt != expected_receipt:
        raise MatrixPreflightError("sealed preflight receipt differs from verified inputs")
    return bundle
