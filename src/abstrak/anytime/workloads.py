"""Frozen offline inputs for the anytime DSL A100 workload matrix.

This module intentionally contains no Torch or DSL imports.  It validates content-addressed
descriptions and source slots only; M9 is responsible for importing KernelBench, materializing
trusted implementations, and observing the worker environment.
"""

from __future__ import annotations

import hashlib
import math
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from abstrak.anytime.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    AnytimeModel,
)
from abstrak.anytime.isolation import build_anytime_process_isolation_contract
from abstrak.providers.contracts import sha256_json

KERNELBENCH_REPOSITORY = "https://github.com/ScalingIntelligence/KernelBench.git"
KERNELBENCH_REVISION = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
TARGET_IDS = ("triton-a100", "tilelang-a100", "cute-a100")
BASELINE_VARIANTS = ("eager", "inductor", "vendor")
WORKLOAD_IDS = (
    "l1-2-standard-matmul",
    "l1-8-irregular-matmul",
    "l1-40-layernorm",
    "l1-24-logsoftmax",
    "l1-93-masked-cumsum",
    "l1-97-scaled-dot-product-attention",
    "l1-85-asymmetric-depthwise-conv2d",
    "l2-14-gemm-divide-sum-scaling",
    "l2-99-matmul-gelu-softmax",
    "l2-1-conv2d-relu-biasadd",
    "l2-2-convtranspose2d-bias-clamp-scale",
    "l2-85-conv2d-groupnorm-scale-pool-clamp",
)
WORKLOAD_FAMILIES = (
    "gemm",
    "reduction",
    "scan",
    "attention",
    "convolution",
    "gemm-fusion",
    "conv-fusion",
)
WORKLOAD_SOURCE_INPUTS = (
    (
        "l1-2-standard-matmul",
        1,
        2,
        "KernelBench/level1/2_Standard_matrix_multiplication_.py",
        "136a33c4d598d07cd7931ef8c0eff4e683c221af6fc43c2d89b6632d3f9902a0",
    ),
    (
        "l1-8-irregular-matmul",
        1,
        8,
        "KernelBench/level1/8_Matmul_with_irregular_shapes_.py",
        "717f98e135d8c11b5005bd869b36843960de2c06151beb3d7dd6a760a987223c",
    ),
    (
        "l1-40-layernorm",
        1,
        40,
        "KernelBench/level1/40_LayerNorm.py",
        "8d975d7cee6946513dcfb6ba84c971e53a890f9d9c6f0a4fb828904941f25975",
    ),
    (
        "l1-24-logsoftmax",
        1,
        24,
        "KernelBench/level1/24_LogSoftmax.py",
        "2aa5cac0c12474ed17fad018fee6ccbb09ca7d200ece7981bb8a608200b81fb8",
    ),
    (
        "l1-93-masked-cumsum",
        1,
        93,
        "KernelBench/level1/93_masked_cumsum.py",
        "8b1591c5c8242968acc0677a4a1308b7a704534724782c3858b4cb0b10828769",
    ),
    (
        "l1-97-scaled-dot-product-attention",
        1,
        97,
        "KernelBench/level1/97_ScaledDotProductAttention.py",
        "a07ca5135657c2ea29f7b7a66ec1815560858ad69fa88221bfc0f82bd5648091",
    ),
    (
        "l1-85-asymmetric-depthwise-conv2d",
        1,
        85,
        "KernelBench/level1/85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel.py",
        "5538e14c57c791611e5e6a663cbed1051af510257a603100a25cbebc78b7a8aa",
    ),
    (
        "l2-14-gemm-divide-sum-scaling",
        2,
        14,
        "KernelBench/level2/14_Gemm_Divide_Sum_Scaling.py",
        "8bd6a36d6dafc8916d9b47d413f668fb7a4c25444378717a4c09f495b3891cd2",
    ),
    (
        "l2-99-matmul-gelu-softmax",
        2,
        99,
        "KernelBench/level2/99_Matmul_GELU_Softmax.py",
        "a27769675f355605c1931433193bbf36a309e050c4b27d3e6375938f22329709",
    ),
    (
        "l2-1-conv2d-relu-biasadd",
        2,
        1,
        "KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py",
        "45ee3a3db038396c982207b36812086bf1b94472ebbf229c5e76c5ea0b90f89f",
    ),
    (
        "l2-2-convtranspose2d-bias-clamp-scale",
        2,
        2,
        "KernelBench/level2/2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide.py",
        "2b69e31d12d28d291fb5b56634176269ba6513e8d240fc01d6f8a7b706abc109",
    ),
    (
        "l2-85-conv2d-groupnorm-scale-pool-clamp",
        2,
        85,
        "KernelBench/level2/85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py",
        "ba63b5b156a347f2f92a607c4a612e2bb1963dbcb9e7f04b3e4f0cb4a294fff3",
    ),
)

DEFAULT_CANDIDATE_MAX_WALL_SECONDS = 600.0
DEFAULT_CANDIDATE_MAX_MEMORY_BYTES = 64 * 1024**3
EXPECTED_WHEELHOUSE_ARCHIVE_SHA256 = (
    "ae644076dd76cd3ed8e47931e1ca4bc044881e244024556a1cb4d05767520caf"
)
ENVIRONMENT_ASSET_PATHS = {
    "worker_bootstrap_sha256": "scripts/bootstrap-a100.sh",
    "worker_update_sha256": "scripts/update-worker.sh",
    "lock_sha256": "uv.lock",
}

DEFAULT_ASSET_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_MANIFEST = (
    DEFAULT_ASSET_ROOT / "benchmarks" / "anytime-dsl-a100" / "manifests" / "inputs.json"
)

# Filled after the reviewed JSON manifest is frozen.  The loader never silently accepts changed
# default bytes; callers using another path must provide an explicit expected digest.
PINNED_INPUT_MANIFEST_SHA256 = "ad9b4c73b5f13530fdc97ffcded7968dcea7b4c448f63decf4e34828b54afaf3"

_SHA256 = re.compile(SHA256_PATTERN)
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class AnytimeWorkloadError(ValueError):
    """Raised when frozen workload inputs are malformed, mismatched, or leak private data."""


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(_SAFE_PATH_COMPONENT.fullmatch(part) is None for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("must be a safe relative POSIX path")
    return value


class AnytimeTensorInput(AnytimeModel):
    """One public tensor ABI entry; parameter values stay in the private state snapshot."""

    name: str = Field(pattern=IDENTIFIER_PATTERN)
    shape: tuple[int, ...] = Field(min_length=1)
    dtype: Literal["fp16", "bool"]
    role: Literal["argument", "state"]
    distribution: str = Field(min_length=1)

    @field_validator("shape")
    @classmethod
    def dimensions_are_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(dimension <= 0 for dimension in value):
            raise ValueError("tensor dimensions must be positive")
        return value


class AnytimeGeneratorCase(AnytimeModel):
    """One deterministic controller-owned input recipe."""

    id: str = Field(pattern=IDENTIFIER_PATTERN)
    seed: int = Field(ge=0, le=2**63 - 1)
    recipe: str = Field(min_length=1)


class AnytimeGeneratorContract(AnytimeModel):
    """Private, deterministic dev/sealed/timing generator description."""

    schema_version: Literal["abstrak-anytime-generator.v1"] = "abstrak-anytime-generator.v1"
    implementation: Literal["controller-owned-cpu-generator-fp16.v1"] = (
        "controller-owned-cpu-generator-fp16.v1"
    )
    dev_cases: tuple[AnytimeGeneratorCase, ...] = Field(min_length=2)
    sealed_cases: tuple[AnytimeGeneratorCase, ...] = Field(min_length=4)
    timing_case: AnytimeGeneratorCase
    sealed_visibility: Literal["qualifier-only"] = "qualifier-only"
    copy_inputs_per_evaluation: Literal[True] = True

    @model_validator(mode="after")
    def cases_are_disjoint(self) -> AnytimeGeneratorContract:
        cases = (*self.dev_cases, *self.sealed_cases, self.timing_case)
        identifiers = tuple(case.id for case in cases)
        seeds = tuple(case.seed for case in cases)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("generator case IDs must be unique across all partitions")
        if len(seeds) != len(set(seeds)):
            raise ValueError("generator seeds must be unique across all partitions")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeStateTransferContract(AnytimeModel):
    """Reference/candidate initialization parity, including parameterized Level-2 modules."""

    schema_version: Literal["abstrak-anytime-state-transfer.v1"] = (
        "abstrak-anytime-state-transfer.v1"
    )
    parameterized: bool
    initialization_seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    mode: Literal["stateless", "snapshot-and-clone-state-dict"]
    state_dtype: Literal["fp16"] | None = None
    reference_state_slot: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    candidate_state_slot: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    identical_state_required: Literal[True] = True
    sequential_random_construction: Literal[False] = False

    @model_validator(mode="after")
    def state_contract_is_coherent(self) -> AnytimeStateTransferContract:
        state_values = (
            self.initialization_seed,
            self.state_dtype,
            self.reference_state_slot,
            self.candidate_state_slot,
        )
        if self.parameterized:
            if self.mode != "snapshot-and-clone-state-dict" or any(
                value is None for value in state_values
            ):
                raise ValueError("parameterized workloads require one cloned state snapshot")
            if self.reference_state_slot != self.candidate_state_slot:
                raise ValueError("reference and candidate must consume the identical state slot")
        elif self.mode != "stateless" or any(value is not None for value in state_values):
            raise ValueError("stateless workloads cannot declare state snapshot fields")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeToleranceContract(AnytimeModel):
    """Frozen correctness comparison policy."""

    atol: float = Field(gt=0)
    rtol: float = Field(gt=0)
    output_dtype: Literal["fp16"] = "fp16"
    output_finite_required: Literal[True] = True
    inputs_unchanged_required: Literal[True] = True
    equal_nan: Literal[False] = False

    @field_validator("atol", "rtol")
    @classmethod
    def tolerances_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("tolerances must be finite")
        return value


class AnytimeNumericalAdversary(AnytimeModel):
    """A private sealed-case purpose, not the sealed tensor values themselves."""

    id: str = Field(pattern=IDENTIFIER_PATTERN)
    sealed_case_id: str = Field(pattern=IDENTIFIER_PATTERN)
    purpose: str = Field(min_length=1)


class AnytimeTimingInputContract(AnytimeModel):
    """Selection timing input kept disjoint from correctness feedback inputs."""

    generator_case_id: str = Field(pattern=IDENTIFIER_PATTERN)
    fresh_process: Literal[True] = True
    fresh_input_copy_per_trial: Literal[True] = True
    state_slot: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    timing_unit: Literal["milliseconds"] = "milliseconds"


class AnytimeKernelBenchLineage(AnytimeModel):
    """Exact upstream bytes from the clean pinned KernelBench checkout."""

    repository: Literal[KERNELBENCH_REPOSITORY] = KERNELBENCH_REPOSITORY
    revision: Literal[KERNELBENCH_REVISION] = KERNELBENCH_REVISION
    level: Literal[1, 2]
    problem_id: int = Field(ge=1, le=100)
    source_path: str
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_observation: Literal["offline-byte-hash-only"] = "offline-byte-hash-only"
    execution_observation: Literal["pending-m9"] = "pending-m9"

    @field_validator("source_path")
    @classmethod
    def source_path_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)


class AnytimePublicWorkload(AnytimeModel):
    """Whitelist-only Agent-visible workload payload."""

    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    family: str = Field(pattern=IDENTIFIER_PATTERN)
    specification: str = Field(min_length=1)
    inputs: tuple[AnytimeTensorInput, ...] = Field(min_length=1)
    output_shape: tuple[int, ...] = Field(min_length=1)
    tolerance: AnytimeToleranceContract


class AnytimeWorkloadPack(AnytimeModel):
    """One complete frozen FP16 KernelBench-derived workload input pack."""

    schema_version: Literal["abstrak-anytime-workload-pack.v1"] = "abstrak-anytime-workload-pack.v1"
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    family: str = Field(pattern=IDENTIFIER_PATTERN)
    specification: str = Field(min_length=1)
    lineage: AnytimeKernelBenchLineage
    inputs: tuple[AnytimeTensorInput, ...] = Field(min_length=1)
    output_shape: tuple[int, ...] = Field(min_length=1)
    generator: AnytimeGeneratorContract
    state_transfer: AnytimeStateTransferContract
    tolerance: AnytimeToleranceContract
    numerical_adversaries: tuple[AnytimeNumericalAdversary, ...] = Field(min_length=2)
    timing_input: AnytimeTimingInputContract
    precision_policy: Literal["fp16-input-state-output-fp32-accumulation"] = (
        "fp16-input-state-output-fp32-accumulation"
    )
    fixed_shape_forward_only: Literal[True] = True
    resource_feasibility_status: Literal["pending-m9", "validated-m9"] = "pending-m9"
    resource_feasibility_artifact_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )

    @field_validator("output_shape")
    @classmethod
    def output_dimensions_are_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(dimension <= 0 for dimension in value):
            raise ValueError("output dimensions must be positive")
        return value

    @model_validator(mode="after")
    def private_contracts_cross_reference(self) -> AnytimeWorkloadPack:
        if self.family not in WORKLOAD_FAMILIES:
            raise ValueError("unknown anytime workload family")
        if any(item.dtype == "fp16" for item in self.inputs) is False:
            raise ValueError("every workload must contain at least one FP16 tensor")
        if any(item.dtype not in {"fp16", "bool"} for item in self.inputs):
            raise ValueError("workload tensors must be FP16 except boolean masks")
        sealed_ids = {case.id for case in self.generator.sealed_cases}
        adversary_ids = tuple(item.id for item in self.numerical_adversaries)
        if len(adversary_ids) != len(set(adversary_ids)):
            raise ValueError("numerical adversary IDs must be unique")
        if any(item.sealed_case_id not in sealed_ids for item in self.numerical_adversaries):
            raise ValueError("numerical adversaries must reference sealed generator cases")
        if self.timing_input.generator_case_id != self.generator.timing_case.id:
            raise ValueError("timing input must reference the dedicated timing generator case")
        state_slot = self.state_transfer.reference_state_slot
        if self.timing_input.state_slot != state_slot:
            raise ValueError("timing input must use the same frozen state as correctness")
        if self.lineage.level == 2 and not self.state_transfer.parameterized:
            raise ValueError("all selected Level-2 modules require explicit state transfer")
        if (self.resource_feasibility_status == "validated-m9") != (
            self.resource_feasibility_artifact_sha256 is not None
        ):
            raise ValueError("resource feasibility status and artifact digest disagree")
        return self

    def public_view(self) -> AnytimePublicWorkload:
        """Return an explicit whitelist that cannot expose seeds, lineage, or state bytes."""

        return AnytimePublicWorkload(
            task_id=self.id,
            family=self.family,
            specification=self.specification,
            inputs=tuple(item for item in self.inputs if item.role == "argument"),
            output_shape=self.output_shape,
            tolerance=self.tolerance,
        )

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeTargetCardInput(AnytimeModel):
    """A content-addressed balanced target card with an unrelated example."""

    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    backend: Literal["triton", "tilelang", "cute"]
    source_path: str
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    example_semantics: Literal["vector-add"] = "vector-add"
    example_count: Literal[1] = 1
    example_relation: Literal["unrelated-to-study-workloads"] = "unrelated-to-study-workloads"
    balance_group: Literal["one-vector-add-example-v1"] = "one-vector-add-example-v1"

    @field_validator("source_path")
    @classmethod
    def card_path_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeExpertSourceInput(AnytimeModel):
    """One trusted expert source slot; no implementation or result is fabricated offline."""

    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_path: str
    status: Literal[
        "pending_live_materialization",
        "materialized_pending_m9",
        "validated_m9",
    ] = "pending_live_materialization"
    expected_source_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    correctness_observation: Literal["pending-m9", "passed-m9"] = "pending-m9"
    target_launch_observation: Literal["pending-m9", "passed-m9"] = "pending-m9"
    timing_observation: Literal["pending-m9", "stable-m9"] = "pending-m9"
    formal_ready: bool = False

    @field_validator("source_path")
    @classmethod
    def source_path_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)

    @model_validator(mode="after")
    def source_state_is_coherent(self) -> AnytimeExpertSourceInput:
        observations = (
            self.correctness_observation,
            self.target_launch_observation,
            self.timing_observation,
        )
        if self.status == "pending_live_materialization":
            if self.expected_source_sha256 is not None or self.formal_ready:
                raise ValueError("pending expert slots cannot claim source bytes or readiness")
            if observations != ("pending-m9", "pending-m9", "pending-m9"):
                raise ValueError("pending expert slots cannot claim live observations")
        elif self.status == "materialized_pending_m9":
            if self.expected_source_sha256 is None or self.formal_ready:
                raise ValueError("materialized experts require bytes but remain unready")
            if observations != ("pending-m9", "pending-m9", "pending-m9"):
                raise ValueError("unvalidated expert sources cannot claim live observations")
        elif (
            self.expected_source_sha256 is None
            or not self.formal_ready
            or observations != ("passed-m9", "passed-m9", "stable-m9")
        ):
            raise ValueError("validated experts require source bytes and all M9 observations")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeBaselineSourceInput(AnytimeModel):
    """One common baseline source slot, intentionally pending M9 materialization."""

    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    variant: Literal["eager", "inductor", "vendor"]
    source_path: str
    status: Literal[
        "pending_live_materialization",
        "materialized_pending_m9",
        "validated_m9",
    ] = "pending_live_materialization"
    expected_source_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    applicability_observation: Literal["pending-m9", "passed-m9"] = "pending-m9"
    correctness_observation: Literal["pending-m9", "passed-m9"] = "pending-m9"
    timing_observation: Literal["pending-m9", "stable-m9"] = "pending-m9"
    formal_ready: bool = False

    @field_validator("source_path")
    @classmethod
    def source_path_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)

    @model_validator(mode="after")
    def source_state_is_coherent(self) -> AnytimeBaselineSourceInput:
        observations = (
            self.applicability_observation,
            self.correctness_observation,
            self.timing_observation,
        )
        if self.status == "pending_live_materialization":
            if self.expected_source_sha256 is not None or self.formal_ready:
                raise ValueError("pending baseline slots cannot claim source bytes or readiness")
            if observations != ("pending-m9", "pending-m9", "pending-m9"):
                raise ValueError("pending baseline slots cannot claim live observations")
        elif self.status == "materialized_pending_m9":
            if self.expected_source_sha256 is None or self.formal_ready:
                raise ValueError("materialized baselines require bytes but remain unready")
            if observations != ("pending-m9", "pending-m9", "pending-m9"):
                raise ValueError("unvalidated baseline sources cannot claim live observations")
        elif (
            self.expected_source_sha256 is None
            or not self.formal_ready
            or observations != ("passed-m9", "passed-m9", "stable-m9")
        ):
            raise ValueError("validated baselines require source bytes and all M9 observations")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeEnvironmentContract(AnytimeModel):
    """Complete expected environment contract with explicitly pending live observations."""

    schema_version: Literal["abstrak-anytime-environment-input.v1"] = (
        "abstrak-anytime-environment-input.v1"
    )
    accelerator: Literal["NVIDIA A100 80GB"] = "NVIDIA A100 80GB"
    compute_capability: Literal["8.0"] = "8.0"
    python_version: Literal["3.10.20"] = "3.10.20"
    torch_version: Literal["2.13.0+cu126"] = "2.13.0+cu126"
    cuda_runtime_version: Literal["12.6"] = "12.6"
    minimum_driver_version: Literal["575.51.03"] = "575.51.03"
    triton_version: Literal["3.7.1"] = "3.7.1"
    tilelang_version: Literal["0.1.12"] = "0.1.12"
    cute_cutlass_dsl_version: Literal["4.6.1"] = "4.6.1"
    cuda_python_version: Literal["12.9.5"] = "12.9.5"
    cuda_bindings_version: Literal["12.9.5"] = "12.9.5"
    kernelbench_revision: Literal[KERNELBENCH_REVISION] = KERNELBENCH_REVISION
    worker_revision_policy: Literal["same-clean-git-commit-as-controller"] = (
        "same-clean-git-commit-as-controller"
    )
    worker_bootstrap_sha256: str = Field(pattern=SHA256_PATTERN)
    worker_update_sha256: str = Field(pattern=SHA256_PATTERN)
    isolation_mode: Literal["bubblewrap-distinct-processes-no-network"] = (
        "bubblewrap-distinct-processes-no-network"
    )
    candidate_max_wall_seconds: Literal[600.0] = DEFAULT_CANDIDATE_MAX_WALL_SECONDS
    candidate_max_memory_bytes: Literal[68719476736] = DEFAULT_CANDIDATE_MAX_MEMORY_BYTES
    isolation_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    lock_sha256: str = Field(pattern=SHA256_PATTERN)
    wheelhouse_archive_sha256: Literal[EXPECTED_WHEELHOUSE_ARCHIVE_SHA256] = (
        EXPECTED_WHEELHOUSE_ARCHIVE_SHA256
    )
    gpu_jobs_serial: Literal[True] = True
    no_mig: Literal[True] = True
    no_gpu_sharing: Literal[True] = True
    observation_status: Literal["pending-m9"] = "pending-m9"
    observed_environment_sha256: None = None

    @model_validator(mode="after")
    def expected_environment_is_coherent(self) -> AnytimeEnvironmentContract:
        isolation = build_anytime_process_isolation_contract(
            max_wall_seconds=self.candidate_max_wall_seconds,
            max_memory_bytes=self.candidate_max_memory_bytes,
        )
        if self.isolation_contract_sha256 != isolation.sha256:
            raise ValueError("isolation digest differs from the frozen process contract")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeFloorPolicy(AnytimeModel):
    """Frozen stability and baseline-envelope rules consumed by ``floor.py``."""

    minimum_timing_blocks: Literal[3] = 3
    max_block_cv: Literal[0.05] = 0.05
    max_block_spread: Literal[0.10] = 0.10
    b_star_rule: Literal["fastest-stable-common-baseline"] = "fastest-stable-common-baseline"
    required_baseline_variants: tuple[Literal["eager", "inductor", "vendor"], ...] = (
        "eager",
        "inductor",
        "vendor",
    )


class AnytimeWorkloadInputManifest(AnytimeModel):
    """All M6 inputs with exact 12/36/static cross-product invariants."""

    schema_version: Literal["abstrak-anytime-workload-input-manifest.v1"] = (
        "abstrak-anytime-workload-input-manifest.v1"
    )
    study_id: Literal["anytime-dsl-a100"] = "anytime-dsl-a100"
    workloads: tuple[AnytimeWorkloadPack, ...]
    target_cards: tuple[AnytimeTargetCardInput, ...]
    experts: tuple[AnytimeExpertSourceInput, ...]
    baselines: tuple[AnytimeBaselineSourceInput, ...]
    environment: AnytimeEnvironmentContract
    floor_policy: AnytimeFloorPolicy = Field(default_factory=AnytimeFloorPolicy)

    @model_validator(mode="after")
    def coverage_is_the_frozen_exact_matrix(self) -> AnytimeWorkloadInputManifest:
        workload_ids = tuple(item.id for item in self.workloads)
        if workload_ids != WORKLOAD_IDS:
            raise ValueError("workload packs must be the exact frozen ordered set of twelve")
        if len({item.sha256 for item in self.workloads}) != len(self.workloads):
            raise ValueError("workload pack hashes must be unique")
        lineage_keys = tuple(
            (item.lineage.level, item.lineage.problem_id) for item in self.workloads
        )
        if len(lineage_keys) != len(set(lineage_keys)):
            raise ValueError("KernelBench source lineage must be unique per workload")
        observed_sources = tuple(
            (
                item.id,
                item.lineage.level,
                item.lineage.problem_id,
                item.lineage.source_path,
                item.lineage.source_sha256,
            )
            for item in self.workloads
        )
        if observed_sources != WORKLOAD_SOURCE_INPUTS:
            raise ValueError("workload lineage differs from the frozen KernelBench byte set")

        target_ids = tuple(item.target_id for item in self.target_cards)
        if target_ids != TARGET_IDS:
            raise ValueError("target cards must exactly cover Triton, TileLang, and CuTe")
        if len({item.source_sha256 for item in self.target_cards}) != 3:
            raise ValueError("target cards must bind three independent source assets")
        balance = {
            (item.example_semantics, item.example_count, item.balance_group)
            for item in self.target_cards
        }
        if len(balance) != 1:
            raise ValueError("target cards must use one balanced unrelated-example design")
        expected_backends = dict(zip(TARGET_IDS, ("triton", "tilelang", "cute"), strict=True))
        if any(card.backend != expected_backends[card.target_id] for card in self.target_cards):
            raise ValueError("target card backend differs from its target ID")

        expected_expert_order = tuple(
            (task_id, target_id) for task_id in WORKLOAD_IDS for target_id in TARGET_IDS
        )
        expected_experts = {
            (task_id, target_id) for task_id in WORKLOAD_IDS for target_id in TARGET_IDS
        }
        actual_expert_order = tuple((item.task_id, item.target_id) for item in self.experts)
        actual_experts = set(actual_expert_order)
        if (
            len(self.experts) != 36
            or actual_experts != expected_experts
            or actual_expert_order != expected_expert_order
        ):
            raise ValueError("expert slots must exactly cover the 12 x 3 task-target matrix")
        expected_expert_paths = tuple(
            f"benchmarks/anytime-dsl-a100/experts/{task_id}/{target_id}.py"
            for task_id, target_id in expected_expert_order
        )
        if tuple(item.source_path for item in self.experts) != expected_expert_paths:
            raise ValueError("expert source slots must use the canonical matrix paths")

        expected_baseline_order = tuple(
            (task_id, variant) for task_id in WORKLOAD_IDS for variant in BASELINE_VARIANTS
        )
        expected_baselines = {
            (task_id, variant) for task_id in WORKLOAD_IDS for variant in BASELINE_VARIANTS
        }
        actual_baseline_order = tuple((item.task_id, item.variant) for item in self.baselines)
        actual_baselines = set(actual_baseline_order)
        if (
            len(self.baselines) != 36
            or actual_baselines != expected_baselines
            or actual_baseline_order != expected_baseline_order
        ):
            raise ValueError("baseline slots must exactly cover 12 tasks x 3 common variants")
        expected_baseline_paths = tuple(
            f"benchmarks/anytime-dsl-a100/baselines/{task_id}/{variant}.py"
            for task_id, variant in expected_baseline_order
        )
        if tuple(item.source_path for item in self.baselines) != expected_baseline_paths:
            raise ValueError("baseline source slots must use the canonical matrix paths")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)

    def workload(self, task_id: str) -> AnytimeWorkloadPack:
        for workload in self.workloads:
            if workload.id == task_id:
                return workload
        raise AnytimeWorkloadError(f"unknown workload: {task_id}")

    def target_card(self, target_id: str) -> AnytimeTargetCardInput:
        for card in self.target_cards:
            if card.target_id == target_id:
                return card
        raise AnytimeWorkloadError(f"unknown target card: {target_id}")

    def expert(self, task_id: str, target_id: str) -> AnytimeExpertSourceInput:
        for expert in self.experts:
            if expert.task_id == task_id and expert.target_id == target_id:
                return expert
        raise AnytimeWorkloadError(f"unknown expert slot: {task_id}/{target_id}")

    def baseline(self, task_id: str, variant: str) -> AnytimeBaselineSourceInput:
        for baseline in self.baselines:
            if baseline.task_id == task_id and baseline.variant == variant:
                return baseline
        raise AnytimeWorkloadError(f"unknown baseline slot: {task_id}/{variant}")


@dataclass(frozen=True)
class PinnedAnytimeWorkloadInputs:
    """Parsed inputs plus the raw-byte and canonical-model trust anchors."""

    path: Path
    raw_sha256: str
    manifest_sha256: str
    manifest: AnytimeWorkloadInputManifest


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise AnytimeWorkloadError(f"cannot inspect {label}: {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise AnytimeWorkloadError(f"{label} must be a regular non-symlink file: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise AnytimeWorkloadError(f"cannot read {label}: {path}: {error}") from error


def load_anytime_workload_inputs(
    path: str | Path = DEFAULT_INPUT_MANIFEST,
    *,
    expected_sha256: str | None = None,
) -> PinnedAnytimeWorkloadInputs:
    """Load strict UTF-8 JSON after verifying its raw bytes."""

    manifest_path = Path(path).expanduser()
    expected = expected_sha256
    if expected is None and manifest_path.resolve() == DEFAULT_INPUT_MANIFEST.resolve():
        expected = PINNED_INPUT_MANIFEST_SHA256
    if expected is None or _SHA256.fullmatch(expected) is None:
        raise AnytimeWorkloadError("an explicit valid expected manifest SHA-256 is required")
    payload = _read_regular_bytes(manifest_path, label="workload input manifest")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise AnytimeWorkloadError(
            f"workload input manifest SHA-256 mismatch: expected {expected}, found {actual}"
        )
    try:
        encoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AnytimeWorkloadError("workload input manifest is not UTF-8") from error
    try:
        manifest = AnytimeWorkloadInputManifest.model_validate_json(encoded)
    except ValidationError as error:
        raise AnytimeWorkloadError(f"invalid workload input manifest: {error}") from error
    return PinnedAnytimeWorkloadInputs(
        path=manifest_path.resolve(),
        raw_sha256=actual,
        manifest_sha256=manifest.sha256,
        manifest=manifest,
    )


def _revalidate_pinned(value: PinnedAnytimeWorkloadInputs) -> PinnedAnytimeWorkloadInputs:
    try:
        raw = _read_regular_bytes(value.path, label="pinned workload input manifest")
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        if raw_sha256 != value.raw_sha256:
            raise AnytimeWorkloadError("pinned workload manifest bytes changed after load")
        manifest = AnytimeWorkloadInputManifest.model_validate_json(raw)
    except (AttributeError, TypeError, ValueError) as error:
        raise AnytimeWorkloadError(f"invalid workload input object: {error}") from error
    if manifest != value.manifest or manifest.sha256 != value.manifest_sha256:
        raise AnytimeWorkloadError("workload input object differs from its canonical trust anchor")
    return PinnedAnytimeWorkloadInputs(
        path=value.path,
        raw_sha256=value.raw_sha256,
        manifest_sha256=value.manifest_sha256,
        manifest=manifest,
    )


def _resolve_asset(root: Path, relative_path: str) -> Path:
    candidate = root / relative_path
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise AnytimeWorkloadError(
            f"cannot resolve frozen asset {relative_path}: {error}"
        ) from error
    if resolved.parent != resolved_root and resolved_root not in resolved.parents:
        raise AnytimeWorkloadError(f"frozen asset escapes its root: {relative_path}")
    return resolved


def _verify_asset(root: Path, relative_path: str, expected_sha256: str, *, label: str) -> str:
    path = _resolve_asset(root, relative_path)
    payload = _read_regular_bytes(path, label=label)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise AnytimeWorkloadError(
            f"{label} SHA-256 mismatch for {relative_path}: "
            f"expected {expected_sha256}, found {actual}"
        )
    return payload.decode("utf-8")


def _validate_environment_assets(
    manifest: AnytimeWorkloadInputManifest,
    repository_root: Path,
) -> None:
    observed_text: dict[str, str] = {}
    for field, relative_path in ENVIRONMENT_ASSET_PATHS.items():
        observed_text[field] = _verify_asset(
            repository_root,
            relative_path,
            getattr(manifest.environment, field),
            label=f"environment input {field}",
        )
    expected_assignment = (
        f'EXPECTED_WHEELHOUSE_SHA256="{manifest.environment.wheelhouse_archive_sha256}"'
    )
    if expected_assignment not in observed_text["worker_bootstrap_sha256"]:
        raise AnytimeWorkloadError("bootstrap does not bind the frozen wheelhouse archive digest")


def _validate_materialized_sources(
    manifest: AnytimeWorkloadInputManifest,
    repository_root: Path,
) -> None:
    for source in (*manifest.experts, *manifest.baselines):
        if source.expected_source_sha256 is None:
            continue
        _verify_asset(
            repository_root,
            source.source_path,
            source.expected_source_sha256,
            label=f"materialized source {source.source_path}",
        )


def _validate_target_cards(manifest: AnytimeWorkloadInputManifest, repository_root: Path) -> None:
    backend_names = {"triton", "tilelang", "cute"}
    for card in manifest.target_cards:
        text = _verify_asset(
            repository_root,
            card.source_path,
            card.source_sha256,
            label=f"{card.target_id} target card",
        )
        lowered = text.lower()
        if "vectoradd" not in lowered and "vector add" not in lowered:
            raise AnytimeWorkloadError(
                f"{card.target_id} card lacks its declared VectorAdd example"
            )
        if "unrelated to the benchmark tasks" not in lowered:
            raise AnytimeWorkloadError(f"{card.target_id} card does not declare example separation")
        foreign = backend_names - {card.backend}
        if any(name in lowered for name in foreign):
            raise AnytimeWorkloadError(f"{card.target_id} card leaks another target stack")


def _private_tokens(manifest: AnytimeWorkloadInputManifest, task: AnytimeWorkloadPack) -> set[str]:
    tokens = {
        task.lineage.source_path,
        task.lineage.source_sha256,
        task.generator.sha256,
        task.state_transfer.sha256,
        *(case.id for case in task.generator.sealed_cases),
        *(str(case.seed) for case in task.generator.sealed_cases),
    }
    tokens.update(expert.source_path for expert in manifest.experts if expert.task_id == task.id)
    return {token for token in tokens if token}


def validate_public_workload_leakage(manifest: AnytimeWorkloadInputManifest) -> None:
    """Prove that each whitelist view excludes sealed/source/expert material."""

    for workload in manifest.workloads:
        rendered = workload.public_view().model_dump_json()
        leaked = sorted(token for token in _private_tokens(manifest, workload) if token in rendered)
        if leaked:
            raise AnytimeWorkloadError(
                f"public workload view leaks private material for {workload.id}: {leaked[0]}"
            )


def verify_kernelbench_source_inputs(
    pinned: PinnedAnytimeWorkloadInputs,
    kernelbench_root: str | Path,
) -> None:
    """Verify the twelve upstream source bytes without importing or executing them."""

    trusted = _revalidate_pinned(pinned)
    root = Path(kernelbench_root).expanduser()
    for workload in trusted.manifest.workloads:
        _verify_asset(
            root,
            workload.lineage.source_path,
            workload.lineage.source_sha256,
            label=f"{workload.id} KernelBench source",
        )


def validate_anytime_workload_inputs(
    pinned: PinnedAnytimeWorkloadInputs,
    *,
    repository_root: str | Path = DEFAULT_ASSET_ROOT,
) -> PinnedAnytimeWorkloadInputs:
    """Run pure offline schema, cross-reference, card, state-parity, and leakage checks."""

    trusted = _revalidate_pinned(pinned)
    root = Path(repository_root).expanduser()
    _validate_target_cards(trusted.manifest, root)
    _validate_environment_assets(trusted.manifest, root)
    _validate_materialized_sources(trusted.manifest, root)
    validate_public_workload_leakage(trusted.manifest)
    return trusted


def formal_readiness_issues(manifest: AnytimeWorkloadInputManifest) -> tuple[str, ...]:
    """List explicit M9 inputs that keep the formal study fail-closed."""

    issues: list[str] = []
    if manifest.environment.observation_status != "pending-m9":
        issues.append("environment status is not the expected pre-M9 pending state")
    else:
        issues.append("environment observation pending M9")
    issues.extend(
        f"expert source/correctness/launch/timing pending M9: {item.task_id}/{item.target_id}"
        for item in manifest.experts
        if not item.formal_ready
    )
    issues.extend(
        "baseline source/applicability/correctness/timing pending M9: "
        f"{item.task_id}/{item.variant}"
        for item in manifest.baselines
        if not item.formal_ready
    )
    issues.extend(
        f"resource feasibility pending M9: {item.id}"
        for item in manifest.workloads
        if item.resource_feasibility_status != "validated-m9"
    )
    return tuple(issues)


def require_formal_workload_readiness(manifest: AnytimeWorkloadInputManifest) -> None:
    """Always block the frozen offline M6 inputs until M9 evidence closes every slot."""

    issues = formal_readiness_issues(manifest)
    if issues:
        raise AnytimeWorkloadError("formal workload inputs are not ready: " + issues[0])
