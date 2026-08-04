"""Deterministic logical freeze for the anytime DSL A100 study.

Everything in this module is offline.  It builds strict study specifications,
evaluates only local dependency parameter rendering, hashes existing repository
assets, and verifies canonical JSON.  It cannot contact a provider or worker,
open an SSH connection, execute candidate source, or manufacture M9 evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from abstrak.anytime.contracts import (
    FORMAL_CHECKPOINT_CALLS,
    SHA256_PATTERN,
    SHAKEOUT_CHECKPOINT_CALLS,
    AnytimeAgentSpec,
    AnytimeCheckpointPolicy,
    AnytimeCohortSpec,
    AnytimeGenerationSpec,
    AnytimeLoopPolicy,
    AnytimeModel,
    AnytimeReasoningSpec,
    AnytimeResourceBudget,
    AnytimeStudySpec,
)
from abstrak.anytime.freeze_pins import (
    PINNED_FORMAL_STUDY_SHA256,
    PINNED_OFFLINE_FREEZE_SHA256,
    PINNED_SHAKEOUT_STUDY_SHA256,
)
from abstrak.anytime.isolation import build_anytime_process_isolation_contract
from abstrak.anytime.prompts import (
    AnytimeBasePromptPolicy,
    build_anytime_base_prompt_policy,
)
from abstrak.anytime.qualification import get_anytime_target_static_policy
from abstrak.anytime.schedule import build_anytime_schedule
from abstrak.anytime.workloads import (
    DEFAULT_CANDIDATE_MAX_MEMORY_BYTES,
    DEFAULT_CANDIDATE_MAX_WALL_SECONDS,
    DEFAULT_INPUT_MANIFEST,
    PINNED_INPUT_MANIFEST_SHA256,
    TARGET_IDS,
    WORKLOAD_IDS,
    PinnedAnytimeWorkloadInputs,
    load_anytime_workload_inputs,
)
from abstrak.providers.contracts import sha256_json
from abstrak.providers.native_conformance import evaluate_native_dependency_conformance
from abstrak.providers.native_contracts import (
    NativeDependencyConformance,
    NativeManifestBundle,
    NativeModelManifest,
    NativeProviderManifest,
    validate_anytime_agent_binding,
)

DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FREEZE_DIRECTORY = (
    DEFAULT_REPOSITORY_ROOT / "benchmarks" / "anytime-dsl-a100" / "manifests"
)
FORMAL_STUDY_FILENAME = "formal-study.json"
SHAKEOUT_STUDY_FILENAME = "shakeout-study.json"
OFFLINE_FREEZE_FILENAME = "offline-freeze.json"

STUDY_SEED = 20260803
CORE_WORKLOAD_IDS = (
    "l1-2-standard-matmul",
    "l1-8-irregular-matmul",
    "l1-40-layernorm",
    "l1-93-masked-cumsum",
    "l1-97-scaled-dot-product-attention",
    "l2-2-convtranspose2d-bias-clamp-scale",
)
RESERVE_WORKLOAD_IDS = (
    "l1-24-logsoftmax",
    "l1-85-asymmetric-depthwise-conv2d",
    "l2-14-gemm-divide-sum-scaling",
    "l2-99-matmul-gelu-softmax",
    "l2-1-conv2d-relu-biasadd",
    "l2-85-conv2d-groupnorm-scale-pool-clamp",
)
SHAKEOUT_WORKLOAD_IDS = (
    "l1-2-standard-matmul",
    "l1-40-layernorm",
    "l1-93-masked-cumsum",
    "l2-2-convtranspose2d-bias-clamp-scale",
)

# Generated manifests and freeze_pins.py are intentionally absent to avoid digest recursion.
CORE_SOURCE_ASSET_PATHS = (
    "docs/anytime-dsl-a100-implementation-plan.md",
    "pyproject.toml",
    "scripts/bootstrap-a100.sh",
    "scripts/freeze_anytime_offline.py",
    "scripts/update-worker.sh",
    "src/abstrak/anytime/__init__.py",
    "src/abstrak/anytime/analysis.py",
    "src/abstrak/anytime/artifacts.py",
    "src/abstrak/anytime/context.py",
    "src/abstrak/anytime/contracts.py",
    "src/abstrak/anytime/figures.py",
    "src/abstrak/anytime/floor.py",
    "src/abstrak/anytime/freeze.py",
    "src/abstrak/anytime/isolation.py",
    "src/abstrak/anytime/ledger.py",
    "src/abstrak/anytime/manifests.py",
    "src/abstrak/anytime/prompts.py",
    "src/abstrak/anytime/qualification.py",
    "src/abstrak/anytime/rehearsal.py",
    "src/abstrak/anytime/resume.py",
    "src/abstrak/anytime/schedule.py",
    "src/abstrak/anytime/workloads.py",
    "src/abstrak/providers/contracts.py",
    "src/abstrak/providers/native_client.py",
    "src/abstrak/providers/native_conformance.py",
    "src/abstrak/providers/native_contracts.py",
    "src/abstrak/providers/native_transport.py",
    "uv.lock",
)

M9_BLOCKERS = (
    "clean-matching-controller-and-worker-revisions-not-observed",
    "worker-environment-and-a100-resource-feasibility-not-observed",
    "trusted-expert-and-common-baseline-sources-not-materialized",
    "valid-target-expert-and-b-star-floor-not-constructed",
    "real-os-candidate-containment-not-observed",
    "real-triton-tilelang-cute-launches-not-verified",
    "deepseek-literal-xhigh-dependency-and-endpoint-conformance-not-passed",
    "gpt-literal-xhigh-endpoint-conformance-not-passed",
    "non-scoring-shakeout-not-run",
)

_SHA256 = re.compile(SHA256_PATTERN)


class AnytimeFreezeError(ValueError):
    """Raised when an M8 logical freeze is missing, changed, or inconsistent."""


def _canonical_json_bytes(value: AnytimeModel) -> bytes:
    return (
        json.dumps(
            value.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _raw_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _model_sequence_sha256(values: tuple[AnytimeModel, ...]) -> str:
    return sha256_json(tuple(value.model_dump(mode="json") for value in values))


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("must be a safe relative POSIX path")
    return value


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise AnytimeFreezeError(f"cannot inspect {label} {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise AnytimeFreezeError(f"{label} must be a regular non-symlink file: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise AnytimeFreezeError(f"cannot read {label} {path}: {error}") from error


def _resolve_repository_asset(repository_root: Path, relative_path: str) -> Path:
    unresolved = repository_root / relative_path
    if unresolved.is_symlink():
        raise AnytimeFreezeError(f"source asset must not be a symlink: {relative_path}")
    try:
        root = repository_root.resolve(strict=True)
        candidate = unresolved.resolve(strict=True)
    except OSError as error:
        raise AnytimeFreezeError(f"cannot resolve source asset {relative_path}: {error}") from error
    if candidate.parent != root and root not in candidate.parents:
        raise AnytimeFreezeError(f"source asset escapes repository root: {relative_path}")
    return candidate


def _agent(
    identifier: Literal["deepseek-v4-flash", "gpt-5.6-luna"],
    *,
    provider_id: Literal["deepseek", "openai"],
    protocol: Literal["chat_completions", "responses"],
) -> AnytimeAgentSpec:
    return AnytimeAgentSpec(
        id=identifier,
        provider_id=provider_id,
        model_ref=identifier,
        native_protocol=protocol,
        generation=AnytimeGenerationSpec(
            max_output_tokens=16384,
            reasoning=AnytimeReasoningSpec(
                requested_reasoning_effort="xhigh",
                conformance_requirement="literal_xhigh",
            ),
        ),
    )


def _loop(*, calls: Literal[4, 12]) -> AnytimeLoopPolicy:
    checkpoints = FORMAL_CHECKPOINT_CALLS if calls == 12 else SHAKEOUT_CHECKPOINT_CALLS
    return AnytimeLoopPolicy(
        budget=AnytimeResourceBudget(
            max_scientific_calls=calls,
            max_total_output_tokens=calls * 16384,
            max_compile_attempts=calls,
            max_evaluation_attempts=calls,
            max_gpu_seconds=calls * 600.0,
        ),
        checkpoints=AnytimeCheckpointPolicy(calls=checkpoints),
    )


def build_anytime_formal_study() -> AnytimeStudySpec:
    """Build the exact 198-trajectory conditional formal population."""

    deepseek = _agent(
        "deepseek-v4-flash",
        provider_id="deepseek",
        protocol="chat_completions",
    )
    gpt = _agent("gpt-5.6-luna", provider_id="openai", protocol="responses")
    loop = _loop(calls=12)
    return AnytimeStudySpec(
        study_id="anytime-dsl-a100-formal",
        study_kind="formal",
        seed=STUDY_SEED,
        agents=(deepseek, gpt),
        cohorts=(
            AnytimeCohortSpec(
                id="primary-core",
                agent_id=deepseek.id,
                task_ids=CORE_WORKLOAD_IDS,
                target_ids=TARGET_IDS,
                replicates=(1, 2, 3, 4),
                scoring=True,
                loop=loop,
            ),
            AnytimeCohortSpec(
                id="robustness-core",
                agent_id=gpt.id,
                task_ids=CORE_WORKLOAD_IDS,
                target_ids=TARGET_IDS,
                replicates=(1, 2, 3),
                scoring=True,
                loop=loop,
            ),
            AnytimeCohortSpec(
                id="primary-reserve",
                agent_id=deepseek.id,
                task_ids=RESERVE_WORKLOAD_IDS,
                target_ids=TARGET_IDS,
                replicates=(1, 2, 3, 4),
                scoring=True,
                activation="core_gate",
                loop=loop,
            ),
        ),
    )


def build_anytime_shakeout_study() -> AnytimeStudySpec:
    """Build the exact non-scoring 48-trajectory, four-call shakeout."""

    deepseek = _agent(
        "deepseek-v4-flash",
        provider_id="deepseek",
        protocol="chat_completions",
    )
    gpt = _agent("gpt-5.6-luna", provider_id="openai", protocol="responses")
    loop = _loop(calls=4)
    return AnytimeStudySpec(
        study_id="anytime-dsl-a100-shakeout",
        study_kind="shakeout",
        seed=STUDY_SEED,
        agents=(deepseek, gpt),
        cohorts=tuple(
            AnytimeCohortSpec(
                id=f"{agent.id}-shakeout",
                agent_id=agent.id,
                task_ids=SHAKEOUT_WORKLOAD_IDS,
                target_ids=TARGET_IDS,
                replicates=(1, 2),
                scoring=False,
                loop=loop,
            )
            for agent in (deepseek, gpt)
        ),
    )


def build_anytime_native_manifest_bundle(agent: AnytimeAgentSpec) -> NativeManifestBundle:
    """Build an offline native manifest matching one frozen Agent exactly."""

    if agent.id == "deepseek-v4-flash":
        provider = NativeProviderManifest(
            id="deepseek",
            protocol="chat_completions",
            litellm_provider="deepseek",
            base_url_env="DEEPSEEK_BASE_URL",
            api_key_env="DEEPSEEK_API_KEY",
        )
        api_model = "deepseek/deepseek-v4-flash"
    elif agent.id == "gpt-5.6-luna":
        provider = NativeProviderManifest(
            id="openai",
            protocol="responses",
            litellm_provider="openai",
            base_url_env="OPENAI_BASE_URL",
            api_key_env="OPENAI_API_KEY",
        )
        api_model = "openai/gpt-5.6-luna"
    else:
        raise AnytimeFreezeError(f"unsupported frozen Agent: {agent.id}")
    model = NativeModelManifest(
        id=agent.id,
        provider=provider.id,
        api_model=api_model,
        protocol=provider.protocol,
        max_output_tokens=agent.generation.max_output_tokens,
        temperature=agent.generation.temperature,
        top_p=agent.generation.top_p,
    )
    bundle = NativeManifestBundle(provider=provider, model=model)
    validate_anytime_agent_binding(agent, bundle)
    return bundle


class AnytimeTaskGroups(AnytimeModel):
    schema_version: Literal["abstrak-anytime-task-groups.v1"] = (
        "abstrak-anytime-task-groups.v1"
    )
    core: tuple[str, ...] = Field(min_length=6, max_length=6)
    reserve: tuple[str, ...] = Field(min_length=6, max_length=6)
    shakeout: tuple[str, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def groups_are_exact(self) -> AnytimeTaskGroups:
        if self.core != CORE_WORKLOAD_IDS:
            raise ValueError("core workload group differs from the frozen six")
        if self.reserve != RESERVE_WORKLOAD_IDS:
            raise ValueError("reserve workload group differs from the frozen six")
        if self.shakeout != SHAKEOUT_WORKLOAD_IDS:
            raise ValueError("shakeout workload group differs from the frozen four")
        if set(self.core).intersection(self.reserve):
            raise ValueError("core and reserve workload groups must be disjoint")
        if set((*self.core, *self.reserve)) != set(WORKLOAD_IDS):
            raise ValueError("core and reserve groups must cover the M6 workload set")
        if not set(self.shakeout).issubset(self.core):
            raise ValueError("shakeout workloads must be drawn from the shared core")
        return self


class AnytimeRandomizationPolicy(AnytimeModel):
    schema_version: Literal["abstrak-anytime-randomization.v1"] = (
        "abstrak-anytime-randomization.v1"
    )
    seed: Literal[20260803] = STUDY_SEED
    target_order: Literal["deterministic-balanced-rotation"] = (
        "deterministic-balanced-rotation"
    )
    independent_agent_replicates: Literal[True] = True
    cross_trajectory_state: Literal["forbidden"] = "forbidden"


class AnytimeTimingProtocol(AnytimeModel):
    schema_version: Literal["abstrak-anytime-timing-protocol.v1"] = (
        "abstrak-anytime-timing-protocol.v1"
    )
    role: Literal["search-selection", "formal-checkpoint"]
    clock: Literal["cuda-events"] = "cuda-events"
    warmup_runs: int = Field(ge=1, le=100)
    timed_trials: int = Field(ge=1, le=1000)
    discard_initial_trials: int = Field(ge=0, le=100)
    independent_clean_process_blocks: int = Field(ge=1, le=20)
    block_statistic: Literal["median"] = "median"
    endpoint_statistic: Literal["median-block-median"] = "median-block-median"
    input_policy: Literal["separate-fixed-timing-input"] = "separate-fixed-timing-input"
    evidence_scope: Literal["exploratory-dev", "formal-sealed"]


class AnytimeTimingPolicy(AnytimeModel):
    schema_version: Literal["abstrak-anytime-timing-policy.v1"] = (
        "abstrak-anytime-timing-policy.v1"
    )
    search: AnytimeTimingProtocol
    formal_checkpoint: AnytimeTimingProtocol

    @model_validator(mode="after")
    def protocols_are_the_frozen_pair(self) -> AnytimeTimingPolicy:
        expected_search = AnytimeTimingProtocol(
            role="search-selection",
            warmup_runs=5,
            timed_trials=100,
            discard_initial_trials=1,
            independent_clean_process_blocks=1,
            evidence_scope="exploratory-dev",
        )
        expected_formal = AnytimeTimingProtocol(
            role="formal-checkpoint",
            warmup_runs=25,
            timed_trials=200,
            discard_initial_trials=1,
            independent_clean_process_blocks=3,
            evidence_scope="formal-sealed",
        )
        if self.search != expected_search or self.formal_checkpoint != expected_formal:
            raise ValueError("timing protocols differ from the frozen 5/100 and 25/200/3 pair")
        return self


class AnytimeWinnerPolicy(AnytimeModel):
    schema_version: Literal["abstrak-anytime-winner-policy.v1"] = (
        "abstrak-anytime-winner-policy.v1"
    )
    practical_equivalence_relative_tolerance: Literal[0.05] = 0.05
    tie_rule: Literal["retain-all-targets-within-relative-band"] = (
        "retain-all-targets-within-relative-band"
    )
    p_best_role: Literal["descriptive-only"] = "descriptive-only"
    agent_replicates_are_statistical_units: Literal[True] = True
    timing_trials_are_not_statistical_units: Literal[True] = True


class AnytimeContinuationPolicy(AnytimeModel):
    schema_version: Literal["abstrak-anytime-continuation-policy.v1"] = (
        "abstrak-anytime-continuation-policy.v1"
    )
    minimum_eligible_workloads_per_target_full: Literal[8] = 8
    minimum_eligible_workloads_per_target_core: Literal[4] = 4
    minimum_distinct_stable_winner_targets: Literal[2] = 2
    minimum_stable_winner_workloads: Literal[3] = 3
    minimum_stable_winner_families: Literal[2] = 2
    iteration_endpoint_oracle_ratio: Literal[1.05] = 1.05
    common_wall_clock_oracle_ratio: Literal[1.03] = 1.03
    common_wall_clock_bootstrap_lower_ratio: Literal[1.0] = 1.0
    robustness_winner_agreement_workloads: Literal[4] = 4
    robustness_total_workloads: Literal[6] = 6
    robustness_alternative_oracle_ratio: Literal[1.03] = 1.03
    unsupported_targets_cannot_drive_gate: Literal[True] = True
    unstable_timing_cannot_drive_gate: Literal[True] = True
    reserve_activation: Literal["all-or-none-integrity-bound-core-gate"] = (
        "all-or-none-integrity-bound-core-gate"
    )


class AnytimeShakeoutPolicy(AnytimeModel):
    schema_version: Literal["abstrak-anytime-shakeout-policy.v1"] = (
        "abstrak-anytime-shakeout-policy.v1"
    )
    minimum_stable_correct_workloads_per_target: Literal[2] = 2
    minimum_distinct_families_per_target: Literal[2] = 2
    maximum_infrastructure_censoring_rate: Literal[0.05] = 0.05
    complete_target_use_evidence_required: Literal[True] = True
    exact_checkpoint_reproduction_required: Literal[True] = True
    maximum_uniform_target_card_revisions: Literal[1] = 1
    persistent_failure_stops_study: Literal[True] = True
    scoring: Literal[False] = False


class AnytimeAnalysisPolicy(AnytimeModel):
    schema_version: Literal["abstrak-anytime-analysis-policy.v1"] = (
        "abstrak-anytime-analysis-policy.v1"
    )
    implementation: Literal["anytime-analysis.v1"] = "anytime-analysis.v1"
    wall_clock_grid: Literal[
        "derive-common-support-from-verified-checkpoint-artifacts"
    ] = "derive-common-support-from-verified-checkpoint-artifacts"
    bootstrap_cluster_unit: Literal["semantic-workload-family"] = (
        "semantic-workload-family"
    )
    bootstrap_seed: Literal[20260803] = STUDY_SEED
    bootstrap_resamples: Literal[1000] = 1000
    confidence_level: Literal[0.95] = 0.95
    missing_cells_retained_in_denominator: Literal[True] = True
    formal_curves_use_independently_retimed_checkpoints: Literal[True] = True
    intermediate_dev_curves_are_exploratory: Literal[True] = True


class AnytimeEvaluationPolicy(AnytimeModel):
    schema_version: Literal["abstrak-anytime-evaluation-policy.v1"] = (
        "abstrak-anytime-evaluation-policy.v1"
    )
    timing: AnytimeTimingPolicy
    winner: AnytimeWinnerPolicy = Field(default_factory=AnytimeWinnerPolicy)
    continuation: AnytimeContinuationPolicy = Field(
        default_factory=AnytimeContinuationPolicy
    )
    shakeout: AnytimeShakeoutPolicy = Field(default_factory=AnytimeShakeoutPolicy)
    analysis: AnytimeAnalysisPolicy = Field(default_factory=AnytimeAnalysisPolicy)

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeStudyFreezeBinding(AnytimeModel):
    schema_version: Literal["abstrak-anytime-study-freeze-binding.v1"] = (
        "abstrak-anytime-study-freeze-binding.v1"
    )
    role: Literal["formal", "shakeout"]
    relative_path: str
    raw_sha256: str = Field(pattern=SHA256_PATTERN)
    canonical_spec_sha256: str = Field(pattern=SHA256_PATTERN)
    schedule_sha256: str = Field(pattern=SHA256_PATTERN)
    planned_trajectories: int = Field(ge=1)
    default_executable_trajectories: int = Field(ge=1)
    scientific_request_ceiling: int = Field(ge=1)
    operational_request_ceiling: int = Field(ge=1)

    @field_validator("relative_path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)


class AnytimeWorkloadFreezeBinding(AnytimeModel):
    schema_version: Literal["abstrak-anytime-workload-freeze-binding.v1"] = (
        "abstrak-anytime-workload-freeze-binding.v1"
    )
    relative_path: str
    raw_sha256: str = Field(pattern=SHA256_PATTERN)
    canonical_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    floor_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    isolation_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    target_static_policy_sha256: tuple[str, str, str]

    @field_validator("relative_path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)

    @field_validator("target_static_policy_sha256")
    @classmethod
    def policy_hashes_are_valid(cls, values: tuple[str, str, str]) -> tuple[str, str, str]:
        if any(_SHA256.fullmatch(value) is None for value in values):
            raise ValueError("target static-policy hashes must be SHA-256 values")
        return values


class AnytimeProviderDependencyBinding(AnytimeModel):
    schema_version: Literal["abstrak-anytime-provider-dependency-binding.v1"] = (
        "abstrak-anytime-provider-dependency-binding.v1"
    )
    agent_id: Literal["deepseek-v4-flash", "gpt-5.6-luna"]
    provider_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    model_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    conformance_sha256: str = Field(pattern=SHA256_PATTERN)
    conformance: NativeDependencyConformance

    @model_validator(mode="after")
    def conformance_hash_is_bound(self) -> AnytimeProviderDependencyBinding:
        if self.provider_manifest_sha256 != self.conformance.provider_manifest_sha256:
            raise ValueError("provider dependency binding uses a different provider manifest")
        if self.model_manifest_sha256 != self.conformance.model_manifest_sha256:
            raise ValueError("provider dependency binding uses a different model manifest")
        if self.conformance_sha256 != sha256_json(self.conformance):
            raise ValueError("provider dependency conformance hash mismatch")
        if self.conformance.study_ready is not False:
            raise ValueError("offline dependency conformance cannot authorize a study")
        return self


class AnytimeFrozenSourceAsset(AnytimeModel):
    schema_version: Literal["abstrak-anytime-source-asset.v1"] = (
        "abstrak-anytime-source-asset.v1"
    )
    relative_path: str
    raw_sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=1)

    @field_validator("relative_path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)


class AnytimeOfflineFreezeManifest(AnytimeModel):
    """Complete M8 logical freeze with an explicit non-live boundary."""

    schema_version: Literal["abstrak-anytime-offline-freeze.v1"] = (
        "abstrak-anytime-offline-freeze.v1"
    )
    freeze_id: Literal["anytime-dsl-a100-m8"] = "anytime-dsl-a100-m8"
    studies: tuple[AnytimeStudyFreezeBinding, AnytimeStudyFreezeBinding]
    task_groups: AnytimeTaskGroups
    targets: tuple[str, str, str]
    randomization: AnytimeRandomizationPolicy
    base_prompt: AnytimeBasePromptPolicy
    evaluation: AnytimeEvaluationPolicy
    workload_inputs: AnytimeWorkloadFreezeBinding
    provider_dependencies: tuple[
        AnytimeProviderDependencyBinding,
        AnytimeProviderDependencyBinding,
    ]
    source_assets: tuple[AnytimeFrozenSourceAsset, ...] = Field(min_length=1)
    source_asset_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    repository_revision_policy: Literal[
        "same-clean-controller-worker-commit-observed-in-m9"
    ] = "same-clean-controller-worker-commit-observed-in-m9"
    authorization_policy: Literal["no-live-action-authorized"] = (
        "no-live-action-authorized"
    )
    live_ready: Literal[False] = False
    provider_requests_performed: Literal[False] = False
    credentials_read: Literal[False] = False
    ssh_connections_performed: Literal[False] = False
    gpu_code_executed: Literal[False] = False
    candidate_code_executed: Literal[False] = False
    generated_code_created: Literal[False] = False
    live_environment_observed: Literal[False] = False
    live_floor_constructed: Literal[False] = False
    m9_blockers: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def closure_is_exact_and_non_live(self) -> AnytimeOfflineFreezeManifest:
        if tuple(item.role for item in self.studies) != ("formal", "shakeout"):
            raise ValueError("study bindings must be ordered formal then shakeout")
        formal, shakeout = self.studies
        if (
            formal.planned_trajectories,
            formal.default_executable_trajectories,
            formal.scientific_request_ceiling,
            formal.operational_request_ceiling,
        ) != (198, 126, 2376, 4752):
            raise ValueError("formal study binding differs from the 198/2376/4752 freeze")
        if (
            shakeout.planned_trajectories,
            shakeout.default_executable_trajectories,
            shakeout.scientific_request_ceiling,
            shakeout.operational_request_ceiling,
        ) != (48, 48, 192, 384):
            raise ValueError("shakeout study binding differs from the 48/192/384 freeze")
        if self.targets != TARGET_IDS:
            raise ValueError("target order differs from the frozen three-target axis")
        if tuple(item.agent_id for item in self.provider_dependencies) != (
            "deepseek-v4-flash",
            "gpt-5.6-luna",
        ):
            raise ValueError("provider dependency bindings use the wrong Agent order")
        paths = tuple(item.relative_path for item in self.source_assets)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("source assets must be unique and sorted by relative path")
        if not set(CORE_SOURCE_ASSET_PATHS).issubset(paths):
            raise ValueError("source asset inventory omits a required M8 implementation asset")
        if "src/abstrak/anytime/freeze_pins.py" in paths:
            raise ValueError("freeze pins must be excluded to prevent a recursive digest")
        generated_names = (
            FORMAL_STUDY_FILENAME,
            SHAKEOUT_STUDY_FILENAME,
            OFFLINE_FREEZE_FILENAME,
        )
        if any(path.endswith(generated_names) for path in paths):
            raise ValueError("generated freeze manifests cannot hash themselves")
        if self.source_asset_bundle_sha256 != _model_sequence_sha256(self.source_assets):
            raise ValueError("source asset bundle hash mismatch")
        if self.m9_blockers != M9_BLOCKERS:
            raise ValueError("M9 blockers differ from the explicit offline boundary")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


@dataclass(frozen=True)
class PinnedAnytimeOfflineFreeze:
    path: Path
    raw_sha256: str
    manifest: AnytimeOfflineFreezeManifest


@dataclass(frozen=True)
class AnytimeFreezeWriteResult:
    directory: Path
    formal_raw_sha256: str
    shakeout_raw_sha256: str
    freeze_raw_sha256: str
    manifest: AnytimeOfflineFreezeManifest


def _study_binding(
    role: Literal["formal", "shakeout"],
    filename: str,
    spec: AnytimeStudySpec,
    payload: bytes,
) -> AnytimeStudyFreezeBinding:
    schedule = build_anytime_schedule(spec)
    return AnytimeStudyFreezeBinding(
        role=role,
        relative_path=filename,
        raw_sha256=_raw_sha256(payload),
        canonical_spec_sha256=spec.sha256,
        schedule_sha256=schedule.sha256,
        planned_trajectories=schedule.expected_trajectories,
        default_executable_trajectories=len(schedule.executable_cells()),
        scientific_request_ceiling=schedule.scientific_request_ceiling,
        operational_request_ceiling=schedule.operational_request_ceiling,
    )


def _workload_binding(inputs: PinnedAnytimeWorkloadInputs) -> AnytimeWorkloadFreezeBinding:
    manifest = inputs.manifest
    isolation = build_anytime_process_isolation_contract(
        max_wall_seconds=DEFAULT_CANDIDATE_MAX_WALL_SECONDS,
        max_memory_bytes=DEFAULT_CANDIDATE_MAX_MEMORY_BYTES,
    )
    if isolation.sha256 != manifest.environment.isolation_contract_sha256:
        raise AnytimeFreezeError("M6 environment binds a different isolation contract")
    policies = tuple(
        get_anytime_target_static_policy(backend).sha256
        for backend in ("triton", "tilelang", "cute")
    )
    return AnytimeWorkloadFreezeBinding(
        relative_path=DEFAULT_INPUT_MANIFEST.relative_to(DEFAULT_REPOSITORY_ROOT).as_posix(),
        raw_sha256=inputs.raw_sha256,
        canonical_manifest_sha256=inputs.manifest_sha256,
        environment_contract_sha256=manifest.environment.sha256,
        floor_policy_sha256=sha256_json(manifest.floor_policy),
        isolation_contract_sha256=isolation.sha256,
        target_static_policy_sha256=policies,
    )


def _provider_bindings(
    formal: AnytimeStudySpec,
) -> tuple[AnytimeProviderDependencyBinding, AnytimeProviderDependencyBinding]:
    bindings: list[AnytimeProviderDependencyBinding] = []
    for agent in formal.agents:
        bundle = build_anytime_native_manifest_bundle(agent)
        conformance = evaluate_native_dependency_conformance(bundle)
        bindings.append(
            AnytimeProviderDependencyBinding(
                agent_id=agent.id,
                provider_manifest_sha256=bundle.provider_sha256,
                model_manifest_sha256=bundle.model_sha256,
                conformance_sha256=sha256_json(conformance),
                conformance=conformance,
            )
        )
    if len(bindings) != 2:
        raise AnytimeFreezeError("frozen provider axis must contain exactly two Agents")
    return (bindings[0], bindings[1])


def _source_assets(
    repository_root: Path,
    relative_paths: tuple[str, ...],
) -> tuple[AnytimeFrozenSourceAsset, ...]:
    if len(relative_paths) != len(set(relative_paths)):
        raise AnytimeFreezeError("source asset paths must be unique")
    assets: list[AnytimeFrozenSourceAsset] = []
    for relative_path in sorted(relative_paths):
        _safe_relative_path(relative_path)
        path = _resolve_repository_asset(repository_root, relative_path)
        payload = _read_regular_bytes(path, label="source asset")
        assets.append(
            AnytimeFrozenSourceAsset(
                relative_path=relative_path,
                raw_sha256=_raw_sha256(payload),
                size_bytes=len(payload),
            )
        )
    return tuple(assets)


def build_anytime_offline_freeze(
    repository_root: str | Path = DEFAULT_REPOSITORY_ROOT,
    *,
    source_asset_paths: tuple[str, ...] = CORE_SOURCE_ASSET_PATHS,
) -> AnytimeOfflineFreezeManifest:
    """Build the complete offline freeze from current reviewed repository bytes."""

    root = Path(repository_root).resolve(strict=True)
    formal = build_anytime_formal_study()
    shakeout = build_anytime_shakeout_study()
    formal_payload = _canonical_json_bytes(formal)
    shakeout_payload = _canonical_json_bytes(shakeout)
    input_path = root / DEFAULT_INPUT_MANIFEST.relative_to(DEFAULT_REPOSITORY_ROOT)
    inputs = load_anytime_workload_inputs(input_path, expected_sha256=PINNED_INPUT_MANIFEST_SHA256)
    assets = _source_assets(root, source_asset_paths)
    timing = AnytimeTimingPolicy(
        search=AnytimeTimingProtocol(
            role="search-selection",
            warmup_runs=5,
            timed_trials=100,
            discard_initial_trials=1,
            independent_clean_process_blocks=1,
            evidence_scope="exploratory-dev",
        ),
        formal_checkpoint=AnytimeTimingProtocol(
            role="formal-checkpoint",
            warmup_runs=25,
            timed_trials=200,
            discard_initial_trials=1,
            independent_clean_process_blocks=3,
            evidence_scope="formal-sealed",
        ),
    )
    return AnytimeOfflineFreezeManifest(
        studies=(
            _study_binding("formal", FORMAL_STUDY_FILENAME, formal, formal_payload),
            _study_binding("shakeout", SHAKEOUT_STUDY_FILENAME, shakeout, shakeout_payload),
        ),
        task_groups=AnytimeTaskGroups(
            core=CORE_WORKLOAD_IDS,
            reserve=RESERVE_WORKLOAD_IDS,
            shakeout=SHAKEOUT_WORKLOAD_IDS,
        ),
        targets=TARGET_IDS,
        randomization=AnytimeRandomizationPolicy(),
        base_prompt=build_anytime_base_prompt_policy(),
        evaluation=AnytimeEvaluationPolicy(timing=timing),
        workload_inputs=_workload_binding(inputs),
        provider_dependencies=_provider_bindings(formal),
        source_assets=assets,
        source_asset_bundle_sha256=_model_sequence_sha256(assets),
        m9_blockers=M9_BLOCKERS,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise AnytimeFreezeError(f"freeze output must be a regular file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o644)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_anytime_freeze_manifests(
    destination: str | Path = DEFAULT_FREEZE_DIRECTORY,
    *,
    repository_root: str | Path = DEFAULT_REPOSITORY_ROOT,
    source_asset_paths: tuple[str, ...] = CORE_SOURCE_ASSET_PATHS,
) -> AnytimeFreezeWriteResult:
    """Atomically write the two canonical studies and their offline closure."""

    directory = Path(destination).resolve()
    formal = build_anytime_formal_study()
    shakeout = build_anytime_shakeout_study()
    manifest = build_anytime_offline_freeze(
        repository_root,
        source_asset_paths=source_asset_paths,
    )
    formal_payload = _canonical_json_bytes(formal)
    shakeout_payload = _canonical_json_bytes(shakeout)
    freeze_payload = _canonical_json_bytes(manifest)
    _atomic_write(directory / FORMAL_STUDY_FILENAME, formal_payload)
    _atomic_write(directory / SHAKEOUT_STUDY_FILENAME, shakeout_payload)
    _atomic_write(directory / OFFLINE_FREEZE_FILENAME, freeze_payload)
    return AnytimeFreezeWriteResult(
        directory=directory,
        formal_raw_sha256=_raw_sha256(formal_payload),
        shakeout_raw_sha256=_raw_sha256(shakeout_payload),
        freeze_raw_sha256=_raw_sha256(freeze_payload),
        manifest=manifest,
    )


def load_anytime_offline_freeze(
    path: str | Path = DEFAULT_FREEZE_DIRECTORY / OFFLINE_FREEZE_FILENAME,
    *,
    expected_sha256: str | None = None,
) -> PinnedAnytimeOfflineFreeze:
    """Load strict freeze JSON after checking its exact raw-byte digest."""

    freeze_path = Path(path).expanduser()
    expected = expected_sha256
    default_path = DEFAULT_FREEZE_DIRECTORY / OFFLINE_FREEZE_FILENAME
    if expected is None and freeze_path.resolve() == default_path.resolve():
        expected = PINNED_OFFLINE_FREEZE_SHA256
    if expected is None or _SHA256.fullmatch(expected) is None:
        raise AnytimeFreezeError("an explicit valid offline-freeze SHA-256 is required")
    payload = _read_regular_bytes(freeze_path, label="offline freeze")
    actual = _raw_sha256(payload)
    if actual != expected:
        raise AnytimeFreezeError(
            f"offline-freeze SHA-256 mismatch: expected {expected}, found {actual}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AnytimeFreezeError("offline freeze is not UTF-8") from error
    try:
        manifest = AnytimeOfflineFreezeManifest.model_validate_json(text)
    except ValidationError as error:
        raise AnytimeFreezeError(f"invalid offline freeze: {error}") from error
    return PinnedAnytimeOfflineFreeze(
        path=freeze_path.resolve(),
        raw_sha256=actual,
        manifest=manifest,
    )


def _load_bound_study(
    directory: Path,
    binding: AnytimeStudyFreezeBinding,
) -> tuple[AnytimeStudySpec, bytes]:
    path = directory / binding.relative_path
    payload = _read_regular_bytes(path, label=f"{binding.role} study")
    if _raw_sha256(payload) != binding.raw_sha256:
        raise AnytimeFreezeError(f"{binding.role} study raw SHA-256 mismatch")
    try:
        spec = AnytimeStudySpec.model_validate_json(payload)
    except ValidationError as error:
        raise AnytimeFreezeError(f"invalid {binding.role} study: {error}") from error
    if spec.sha256 != binding.canonical_spec_sha256:
        raise AnytimeFreezeError(f"{binding.role} study canonical hash mismatch")
    schedule = build_anytime_schedule(spec)
    if schedule.sha256 != binding.schedule_sha256:
        raise AnytimeFreezeError(f"{binding.role} schedule hash mismatch")
    if payload != _canonical_json_bytes(spec):
        raise AnytimeFreezeError(f"{binding.role} study bytes are not canonical JSON")
    return spec, payload


def verify_anytime_offline_freeze(
    pinned: PinnedAnytimeOfflineFreeze,
    *,
    repository_root: str | Path = DEFAULT_REPOSITORY_ROOT,
) -> AnytimeOfflineFreezeManifest:
    """Rebuild every offline trust anchor and reject any drift or forged model copy."""

    try:
        manifest = AnytimeOfflineFreezeManifest.model_validate_json(
            json.dumps(
                pinned.manifest.model_dump(mode="json"),
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise AnytimeFreezeError(f"invalid pinned offline-freeze object: {error}") from error
    payload = _read_regular_bytes(pinned.path, label="pinned offline freeze")
    if _raw_sha256(payload) != pinned.raw_sha256:
        raise AnytimeFreezeError("offline-freeze bytes changed after load")
    try:
        disk_manifest = AnytimeOfflineFreezeManifest.model_validate_json(payload)
    except ValidationError as error:
        raise AnytimeFreezeError(f"invalid pinned offline freeze: {error}") from error
    if payload != _canonical_json_bytes(disk_manifest):
        raise AnytimeFreezeError("offline-freeze bytes are not canonical JSON")
    if disk_manifest != manifest:
        raise AnytimeFreezeError("pinned offline-freeze object differs from its disk bytes")
    directory = pinned.path.parent
    formal, formal_payload = _load_bound_study(directory, manifest.studies[0])
    shakeout, shakeout_payload = _load_bound_study(directory, manifest.studies[1])
    if formal != build_anytime_formal_study() or shakeout != build_anytime_shakeout_study():
        raise AnytimeFreezeError("bound study differs from the canonical M8 builder")
    source_paths = tuple(asset.relative_path for asset in manifest.source_assets)
    rebuilt = build_anytime_offline_freeze(
        repository_root,
        source_asset_paths=source_paths,
    )
    if manifest != rebuilt:
        raise AnytimeFreezeError("offline freeze differs from current bound inputs or code")
    if manifest.studies[0].raw_sha256 != _raw_sha256(formal_payload):
        raise AnytimeFreezeError("formal study binding changed during verification")
    if manifest.studies[1].raw_sha256 != _raw_sha256(shakeout_payload):
        raise AnytimeFreezeError("shakeout study binding changed during verification")
    return manifest


def check_anytime_freeze_manifests(
    directory: str | Path = DEFAULT_FREEZE_DIRECTORY,
    *,
    repository_root: str | Path = DEFAULT_REPOSITORY_ROOT,
    expected_formal_sha256: str | None = None,
    expected_shakeout_sha256: str | None = None,
    expected_freeze_sha256: str | None = None,
) -> AnytimeOfflineFreezeManifest:
    """Check canonical bytes, raw pins, schedules, dependencies, inputs, and assets."""

    root = Path(directory).resolve()
    formal_expected = expected_formal_sha256 or PINNED_FORMAL_STUDY_SHA256
    shakeout_expected = expected_shakeout_sha256 or PINNED_SHAKEOUT_STUDY_SHA256
    freeze_expected = expected_freeze_sha256 or PINNED_OFFLINE_FREEZE_SHA256
    for label, expected in (
        ("formal", formal_expected),
        ("shakeout", shakeout_expected),
        ("offline freeze", freeze_expected),
    ):
        if _SHA256.fullmatch(expected) is None:
            raise AnytimeFreezeError(f"{label} expected SHA-256 is invalid")
    pinned = load_anytime_offline_freeze(
        root / OFFLINE_FREEZE_FILENAME,
        expected_sha256=freeze_expected,
    )
    manifest = verify_anytime_offline_freeze(pinned, repository_root=repository_root)
    if manifest.studies[0].raw_sha256 != formal_expected:
        raise AnytimeFreezeError("formal study does not match its reviewed raw-byte pin")
    if manifest.studies[1].raw_sha256 != shakeout_expected:
        raise AnytimeFreezeError("shakeout study does not match its reviewed raw-byte pin")
    return manifest


def frozen_request_ceilings() -> tuple[tuple[str, int, int, int], ...]:
    """Return printable ceilings without reading credentials or authorizing execution."""

    formal = build_anytime_schedule(build_anytime_formal_study())
    shakeout = build_anytime_schedule(build_anytime_shakeout_study())
    return (
        (
            "formal",
            formal.expected_trajectories,
            formal.scientific_request_ceiling,
            formal.operational_request_ceiling,
        ),
        (
            "shakeout",
            shakeout.expected_trajectories,
            shakeout.scientific_request_ceiling,
            shakeout.operational_request_ceiling,
        ),
    )
