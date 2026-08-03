"""Offline validator and B-star builder for the anytime DSL A100 study.

The builder accepts only complete raw evidence and derives stability itself.  The repository ships
no such live evidence: unit tests use clearly synthetic values, while formal construction remains an
M9 gate.
"""

from __future__ import annotations

import json
import math
import statistics
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from abstrak.anytime.contracts import IDENTIFIER_PATTERN, SHA256_PATTERN, AnytimeModel
from abstrak.anytime.workloads import (
    BASELINE_VARIANTS,
    TARGET_IDS,
    WORKLOAD_IDS,
    AnytimeEnvironmentContract,
    AnytimeFloorPolicy,
    AnytimeWorkloadInputManifest,
    PinnedAnytimeWorkloadInputs,
    validate_anytime_workload_inputs,
)
from abstrak.providers.contracts import sha256_json


class AnytimeInvalidFloorError(ValueError):
    """Raised when B-star inputs fail closed with ``invalid_floor``."""


class AnytimeObservedEnvironment(AnytimeModel):
    """Complete M9 worker observation bound to the frozen expected environment."""

    schema_version: Literal["abstrak-anytime-environment-observation.v1"] = (
        "abstrak-anytime-environment-observation.v1"
    )
    environment_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    observation_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    controller_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    worker_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    repository_clean: Literal[True] = True
    kernelbench_checkout_clean: Literal[True] = True
    accelerator: str = Field(min_length=1)
    compute_capability: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    python_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    cuda_runtime_version: str = Field(min_length=1)
    driver_version: str = Field(pattern=r"^[0-9]+(?:\.[0-9]+)+$")
    triton_version: str = Field(min_length=1)
    tilelang_version: str = Field(min_length=1)
    cute_cutlass_dsl_version: str = Field(min_length=1)
    cuda_python_version: str = Field(min_length=1)
    cuda_bindings_version: str = Field(min_length=1)
    kernelbench_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    worker_bootstrap_sha256: str = Field(pattern=SHA256_PATTERN)
    worker_update_sha256: str = Field(pattern=SHA256_PATTERN)
    isolation_mode: str = Field(min_length=1)
    isolation_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    isolation_observed: Literal[True] = True
    lock_sha256: str = Field(pattern=SHA256_PATTERN)
    wheelhouse_archive_sha256: str = Field(pattern=SHA256_PATTERN)
    gpu_jobs_serial: Literal[True] = True
    no_mig: Literal[True] = True
    no_gpu_sharing: Literal[True] = True

    @model_validator(mode="after")
    def controller_and_worker_are_the_same_revision(self) -> AnytimeObservedEnvironment:
        if self.controller_revision != self.worker_revision:
            raise ValueError("controller and worker revisions must be identical")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeRawTimingEvidence(AnytimeModel):
    """Independent timing-block medians; stability is derived, never asserted."""

    block_medians_ms: tuple[float, ...] = Field(min_length=3, max_length=20)

    @field_validator("block_medians_ms")
    @classmethod
    def timings_are_positive_and_finite(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("timing blocks must contain only positive finite milliseconds")
        return values

    @property
    def median_ms(self) -> float:
        return float(statistics.median(self.block_medians_ms))

    @property
    def coefficient_of_variation(self) -> float:
        mean = statistics.fmean(self.block_medians_ms)
        return statistics.pstdev(self.block_medians_ms) / mean

    @property
    def relative_spread(self) -> float:
        return max(self.block_medians_ms) / min(self.block_medians_ms) - 1.0

    def stable_under(self, policy: AnytimeFloorPolicy) -> bool:
        return (
            len(self.block_medians_ms) >= policy.minimum_timing_blocks
            and self.coefficient_of_variation <= policy.max_block_cv
            and self.relative_spread <= policy.max_block_spread
        )


class AnytimeExpertFloorEvidence(AnytimeModel):
    """Complete correctness, target-use, and timing evidence for one expert cell."""

    schema_version: Literal["abstrak-anytime-expert-floor-evidence.v1"] = (
        "abstrak-anytime-expert-floor-evidence.v1"
    )
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    input_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    workload_pack_sha256: str = Field(pattern=SHA256_PATTERN)
    target_card_input_sha256: str = Field(pattern=SHA256_PATTERN)
    expert_source_input_sha256: str = Field(pattern=SHA256_PATTERN)
    observed_expert_source_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_observation_sha256: str = Field(pattern=SHA256_PATTERN)
    correctness_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    launch_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    timing_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    compiled: bool
    all_sealed_cases_passed: bool
    output_finite: bool
    inputs_unchanged: bool
    fallback_free: bool
    target_launch_verified: bool
    timing: AnytimeRawTimingEvidence

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeBaselineFloorEvidence(AnytimeModel):
    """Complete correctness/applicability/timing evidence for one common baseline."""

    schema_version: Literal["abstrak-anytime-baseline-floor-evidence.v1"] = (
        "abstrak-anytime-baseline-floor-evidence.v1"
    )
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    variant: Literal["eager", "inductor", "vendor"]
    input_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    workload_pack_sha256: str = Field(pattern=SHA256_PATTERN)
    baseline_source_input_sha256: str = Field(pattern=SHA256_PATTERN)
    observed_baseline_source_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_observation_sha256: str = Field(pattern=SHA256_PATTERN)
    applicability_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    correctness_artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    timing_artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    applicable: bool
    correct: bool | None = None
    output_finite: bool | None = None
    inputs_unchanged: bool | None = None
    timing: AnytimeRawTimingEvidence | None = None

    @model_validator(mode="after")
    def result_matches_applicability(self) -> AnytimeBaselineFloorEvidence:
        applicable_fields = (
            self.correctness_artifact_sha256,
            self.timing_artifact_sha256,
            self.correct,
            self.output_finite,
            self.inputs_unchanged,
            self.timing,
        )
        if self.applicable and any(value is None for value in applicable_fields):
            raise ValueError("applicable baseline evidence is incomplete")
        if not self.applicable and any(value is not None for value in applicable_fields):
            raise ValueError("inapplicable baseline may contain only applicability evidence")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeFloorEvidenceBundle(AnytimeModel):
    """Raw evidence closure from which the validator may derive a floor."""

    schema_version: Literal["abstrak-anytime-floor-evidence-bundle.v1"] = (
        "abstrak-anytime-floor-evidence-bundle.v1"
    )
    input_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    environment: AnytimeObservedEnvironment | None = None
    experts: tuple[AnytimeExpertFloorEvidence, ...] = ()
    baselines: tuple[AnytimeBaselineFloorEvidence, ...] = ()

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeFloorValidation(AnytimeModel):
    """Fail-closed validation result suitable for preflight reporting."""

    schema_version: Literal["abstrak-anytime-floor-validation.v1"] = (
        "abstrak-anytime-floor-validation.v1"
    )
    status: Literal["valid_floor", "invalid_floor"]
    input_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    reasons: tuple[str, ...]

    @model_validator(mode="after")
    def status_matches_reasons(self) -> AnytimeFloorValidation:
        if self.status == "valid_floor" and self.reasons:
            raise ValueError("valid_floor cannot contain failure reasons")
        if self.status == "invalid_floor" and not self.reasons:
            raise ValueError("invalid_floor requires at least one reason")
        return self


class AnytimeTargetExpertFloor(AnytimeModel):
    """One valid expert realization floor retained separately from B-star."""

    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    expert_source_sha256: str = Field(pattern=SHA256_PATTERN)
    median_ms: float = Field(gt=0)
    timing_evidence_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("median_ms")
    @classmethod
    def median_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("expert floor median must be finite")
        return value


class AnytimeBStarTaskFloor(AnytimeModel):
    """Fastest stable common baseline for one task."""

    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    selected_variant: Literal["eager", "inductor", "vendor"]
    selected_source_sha256: str = Field(pattern=SHA256_PATTERN)
    median_ms: float = Field(gt=0)
    baseline_evidence_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("median_ms")
    @classmethod
    def median_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("B-star median must be finite")
        return value


class AnytimeBStarManifest(AnytimeModel):
    """Derived valid floor.  Trusted experts are not candidates in the B-star envelope."""

    schema_version: Literal["abstrak-anytime-b-star.v1"] = "abstrak-anytime-b-star.v1"
    status: Literal["valid_floor"] = "valid_floor"
    input_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_observation_sha256: str = Field(pattern=SHA256_PATTERN)
    floor_evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    target_expert_floors: tuple[AnytimeTargetExpertFloor, ...] = Field(min_length=36)
    b_star: tuple[AnytimeBStarTaskFloor, ...] = Field(min_length=12)

    @model_validator(mode="after")
    def cardinalities_are_exact(self) -> AnytimeBStarManifest:
        expert_order = tuple((item.task_id, item.target_id) for item in self.target_expert_floors)
        expert_keys = set(expert_order)
        task_ids = tuple(item.task_id for item in self.b_star)
        expected_experts = tuple(
            (task_id, target_id) for task_id in WORKLOAD_IDS for target_id in TARGET_IDS
        )
        if expert_order != expected_experts or len(expert_keys) != 36:
            raise ValueError("valid floor requires 36 unique target expert floors")
        if task_ids != WORKLOAD_IDS or len(task_ids) != len(set(task_ids)):
            raise ValueError("valid B-star requires twelve unique task floors")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _environment_mismatches(
    expected: AnytimeEnvironmentContract,
    observed: AnytimeObservedEnvironment,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    exact_fields = (
        "accelerator",
        "compute_capability",
        "python_version",
        "torch_version",
        "cuda_runtime_version",
        "triton_version",
        "tilelang_version",
        "cute_cutlass_dsl_version",
        "cuda_python_version",
        "cuda_bindings_version",
        "kernelbench_revision",
        "worker_bootstrap_sha256",
        "worker_update_sha256",
        "isolation_mode",
        "isolation_contract_sha256",
        "lock_sha256",
        "wheelhouse_archive_sha256",
        "gpu_jobs_serial",
        "no_mig",
        "no_gpu_sharing",
    )
    for field in exact_fields:
        if getattr(expected, field) != getattr(observed, field):
            mismatches.append(f"environment mismatch: {field}")
    try:
        if _version_tuple(observed.driver_version) < _version_tuple(
            expected.minimum_driver_version
        ):
            mismatches.append("environment mismatch: driver below minimum")
    except ValueError:
        mismatches.append("environment mismatch: unparseable driver version")
    if observed.environment_contract_sha256 != expected.sha256:
        mismatches.append("environment observation binds a different contract")
    return tuple(mismatches)


def _revalidate_bundle(value: object) -> AnytimeFloorEvidenceBundle:
    try:
        if isinstance(value, AnytimeModel):
            raw = value.model_dump(mode="json")
        else:
            raw = value
        payload = json.dumps(raw, allow_nan=False, ensure_ascii=False, sort_keys=True)
        return AnytimeFloorEvidenceBundle.model_validate_json(payload)
    except (TypeError, ValueError, ValidationError) as error:
        raise AnytimeInvalidFloorError(f"invalid_floor: malformed evidence: {error}") from error


def _timing_reason(
    timing: AnytimeRawTimingEvidence,
    policy: AnytimeFloorPolicy,
    label: str,
) -> str | None:
    if not timing.stable_under(policy):
        return (
            f"unstable timing: {label}: cv={timing.coefficient_of_variation:.6f}, "
            f"spread={timing.relative_spread:.6f}"
        )
    return None


def _expert_reasons(
    manifest: AnytimeWorkloadInputManifest,
    evidence: AnytimeExpertFloorEvidence,
    environment: AnytimeObservedEnvironment,
) -> tuple[str, ...]:
    reasons: list[str] = []
    try:
        workload = manifest.workload(evidence.task_id)
        card = manifest.target_card(evidence.target_id)
        source = manifest.expert(evidence.task_id, evidence.target_id)
    except AnytimeInvalidFloorError:
        raise
    except ValueError as error:
        return (f"expert cross-reference failed: {error}",)
    expected = (
        ("input manifest", evidence.input_manifest_sha256, manifest.sha256),
        ("workload pack", evidence.workload_pack_sha256, workload.sha256),
        ("target card", evidence.target_card_input_sha256, card.sha256),
        ("expert source input", evidence.expert_source_input_sha256, source.sha256),
        ("environment contract", evidence.environment_contract_sha256, manifest.environment.sha256),
        ("environment observation", evidence.environment_observation_sha256, environment.sha256),
    )
    reasons.extend(
        f"expert hash mismatch: {evidence.task_id}/{evidence.target_id}/{label}"
        for label, actual, wanted in expected
        if actual != wanted
    )
    if not source.formal_ready or source.status != "validated_m9":
        reasons.append(
            f"expert source is not M9-validated: {evidence.task_id}/{evidence.target_id}"
        )
    elif evidence.observed_expert_source_sha256 != source.expected_source_sha256:
        reasons.append(f"expert source hash mismatch: {evidence.task_id}/{evidence.target_id}")
    checks = (
        evidence.compiled,
        evidence.all_sealed_cases_passed,
        evidence.output_finite,
        evidence.inputs_unchanged,
        evidence.fallback_free,
        evidence.target_launch_verified,
    )
    if not all(checks):
        reasons.append(f"incomplete expert gate: {evidence.task_id}/{evidence.target_id}")
    timing_reason = _timing_reason(
        evidence.timing,
        manifest.floor_policy,
        f"expert/{evidence.task_id}/{evidence.target_id}",
    )
    if timing_reason is not None:
        reasons.append(timing_reason)
    artifact_hashes = {
        evidence.correctness_artifact_sha256,
        evidence.launch_artifact_sha256,
        evidence.timing_artifact_sha256,
    }
    if len(artifact_hashes) != 3:
        reasons.append(
            f"expert gate artifacts are not distinct: {evidence.task_id}/{evidence.target_id}"
        )
    return tuple(reasons)


def _baseline_reasons(
    manifest: AnytimeWorkloadInputManifest,
    evidence: AnytimeBaselineFloorEvidence,
    environment: AnytimeObservedEnvironment,
) -> tuple[str, ...]:
    reasons: list[str] = []
    try:
        workload = manifest.workload(evidence.task_id)
        source = manifest.baseline(evidence.task_id, evidence.variant)
    except ValueError as error:
        return (f"baseline cross-reference failed: {error}",)
    expected = (
        ("input manifest", evidence.input_manifest_sha256, manifest.sha256),
        ("workload pack", evidence.workload_pack_sha256, workload.sha256),
        ("baseline source input", evidence.baseline_source_input_sha256, source.sha256),
        ("environment contract", evidence.environment_contract_sha256, manifest.environment.sha256),
        ("environment observation", evidence.environment_observation_sha256, environment.sha256),
    )
    reasons.extend(
        f"baseline hash mismatch: {evidence.task_id}/{evidence.variant}/{label}"
        for label, actual, wanted in expected
        if actual != wanted
    )
    if not source.formal_ready or source.status != "validated_m9":
        reasons.append(
            f"baseline source is not M9-validated: {evidence.task_id}/{evidence.variant}"
        )
    elif evidence.observed_baseline_source_sha256 != source.expected_source_sha256:
        reasons.append(f"baseline source hash mismatch: {evidence.task_id}/{evidence.variant}")
    if evidence.applicable and not all(
        (evidence.correct, evidence.output_finite, evidence.inputs_unchanged)
    ):
        reasons.append(f"incomplete baseline gate: {evidence.task_id}/{evidence.variant}")
    if evidence.timing is not None:
        timing_reason = _timing_reason(
            evidence.timing,
            manifest.floor_policy,
            f"baseline/{evidence.task_id}/{evidence.variant}",
        )
        if timing_reason is not None:
            reasons.append(timing_reason)
    artifact_hashes = tuple(
        value
        for value in (
            evidence.applicability_artifact_sha256,
            evidence.correctness_artifact_sha256,
            evidence.timing_artifact_sha256,
        )
        if value is not None
    )
    expected_artifact_count = 3 if evidence.applicable else 1
    if len(artifact_hashes) != expected_artifact_count or len(set(artifact_hashes)) != len(
        artifact_hashes
    ):
        reasons.append(
            f"baseline gate artifacts are not distinct: {evidence.task_id}/{evidence.variant}"
        )
    return tuple(reasons)


def validate_anytime_floor_evidence(
    inputs: PinnedAnytimeWorkloadInputs,
    evidence: object,
    *,
    repository_root: str | None = None,
) -> AnytimeFloorValidation:
    """Return ``invalid_floor`` for every incomplete, unstable, or mismatched closure."""

    try:
        trusted_inputs = validate_anytime_workload_inputs(
            inputs,
            **({} if repository_root is None else {"repository_root": repository_root}),
        )
    except ValueError as error:
        return AnytimeFloorValidation(
            status="invalid_floor",
            input_manifest_sha256=inputs.manifest_sha256,
            evidence_sha256=None,
            reasons=(f"invalid frozen inputs: {error}",),
        )
    manifest = trusted_inputs.manifest
    try:
        bundle = _revalidate_bundle(evidence)
    except AnytimeInvalidFloorError as error:
        return AnytimeFloorValidation(
            status="invalid_floor",
            input_manifest_sha256=manifest.sha256,
            evidence_sha256=None,
            reasons=(str(error),),
        )

    reasons: list[str] = []
    if bundle.input_manifest_sha256 != manifest.sha256:
        reasons.append("floor evidence binds a different input manifest")
    environment = bundle.environment
    if environment is None:
        reasons.append("missing environment observation")
    else:
        reasons.extend(_environment_mismatches(manifest.environment, environment))

    expected_experts = {
        (task.id, target_id) for task in manifest.workloads for target_id in TARGET_IDS
    }
    expert_keys = [(item.task_id, item.target_id) for item in bundle.experts]
    if len(expert_keys) != len(set(expert_keys)):
        reasons.append("duplicate task-target expert evidence")
    actual_experts = set(expert_keys)
    if actual_experts != expected_experts:
        missing = sorted(expected_experts - actual_experts)
        extra = sorted(actual_experts - expected_experts)
        reasons.append(f"expert evidence coverage mismatch: missing={missing}, extra={extra}")

    expected_baselines = {
        (task.id, variant) for task in manifest.workloads for variant in BASELINE_VARIANTS
    }
    baseline_keys = [(item.task_id, item.variant) for item in bundle.baselines]
    if len(baseline_keys) != len(set(baseline_keys)):
        reasons.append("duplicate task-variant baseline evidence")
    actual_baselines = set(baseline_keys)
    if actual_baselines != expected_baselines:
        missing = sorted(expected_baselines - actual_baselines)
        extra = sorted(actual_baselines - expected_baselines)
        reasons.append(f"baseline evidence coverage mismatch: missing={missing}, extra={extra}")
    for task in manifest.workloads:
        if not any(item.task_id == task.id and item.applicable for item in bundle.baselines):
            reasons.append(f"no applicable common baseline: {task.id}")

    if environment is not None:
        for item in bundle.experts:
            reasons.extend(_expert_reasons(manifest, item, environment))
        for item in bundle.baselines:
            reasons.extend(_baseline_reasons(manifest, item, environment))

    all_artifacts = [
        value
        for item in bundle.experts
        for value in (
            item.correctness_artifact_sha256,
            item.launch_artifact_sha256,
            item.timing_artifact_sha256,
        )
    ]
    all_artifacts.extend(
        value
        for item in bundle.baselines
        for value in (
            item.applicability_artifact_sha256,
            item.correctness_artifact_sha256,
            item.timing_artifact_sha256,
        )
        if value is not None
    )
    if len(all_artifacts) != len(set(all_artifacts)):
        reasons.append("floor gate artifact digests must be globally unique")

    return AnytimeFloorValidation(
        status="invalid_floor" if reasons else "valid_floor",
        input_manifest_sha256=manifest.sha256,
        evidence_sha256=bundle.sha256,
        reasons=tuple(reasons),
    )


def build_anytime_b_star(
    inputs: PinnedAnytimeWorkloadInputs,
    evidence: object,
    *,
    repository_root: str | None = None,
) -> AnytimeBStarManifest:
    """Build the common baseline envelope only after the complete fake/live gate validates."""

    validation = validate_anytime_floor_evidence(
        inputs,
        evidence,
        repository_root=repository_root,
    )
    if validation.status != "valid_floor":
        raise AnytimeInvalidFloorError("invalid_floor: " + validation.reasons[0])
    bundle = _revalidate_bundle(evidence)
    assert bundle.environment is not None
    manifest = inputs.manifest

    experts_by_key = {(item.task_id, item.target_id): item for item in bundle.experts}
    target_floors = tuple(
        AnytimeTargetExpertFloor(
            task_id=workload.id,
            target_id=target_id,
            expert_source_sha256=experts_by_key[
                (workload.id, target_id)
            ].observed_expert_source_sha256,
            median_ms=experts_by_key[(workload.id, target_id)].timing.median_ms,
            timing_evidence_sha256=experts_by_key[(workload.id, target_id)].sha256,
        )
        for workload in manifest.workloads
        for target_id in TARGET_IDS
    )
    baselines_by_key = {(item.task_id, item.variant): item for item in bundle.baselines}
    b_star: list[AnytimeBStarTaskFloor] = []
    for workload in manifest.workloads:
        eligible = tuple(
            item
            for variant in BASELINE_VARIANTS
            if (item := baselines_by_key[(workload.id, variant)]).applicable
        )
        if not eligible:
            raise AnytimeInvalidFloorError(
                f"invalid_floor: no applicable baseline for {workload.id}"
            )
        selected = min(
            eligible,
            key=lambda item: (
                item.timing.median_ms if item.timing is not None else math.inf,
                BASELINE_VARIANTS.index(item.variant),
            ),
        )
        assert selected.timing is not None
        b_star.append(
            AnytimeBStarTaskFloor(
                task_id=workload.id,
                selected_variant=selected.variant,
                selected_source_sha256=selected.observed_baseline_source_sha256,
                median_ms=selected.timing.median_ms,
                baseline_evidence_sha256=selected.sha256,
            )
        )
    return AnytimeBStarManifest(
        input_manifest_sha256=manifest.sha256,
        environment_contract_sha256=manifest.environment.sha256,
        environment_observation_sha256=bundle.environment.sha256,
        floor_evidence_sha256=bundle.sha256,
        target_expert_floors=target_floors,
        b_star=tuple(b_star),
    )
