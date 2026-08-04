"""Scripted, side-effect-free M8 rehearsal for the anytime DSL A100 study.

The rehearsal materializes the complete 48-trajectory shakeout schedule with a
fake provider and fake worker.  It exercises immutable attempts, one bounded
crash/retry, phase journals, checkpoints, offline qualification, invalid-floor
handling, analysis, and figures.  Candidate source is inspected but never
executed, and no provider client, credential, network, SSH, or GPU API is used.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from abstrak.anytime.analysis import (
    AnytimeAgentReplicateAxis,
    AnytimeAnalysisDataset,
    AnytimeAnalysisSpec,
    AnytimeArtifactTrust,
    AnytimeFloorArtifact,
    AnytimeTrajectoryArtifact,
    AnytimeTurnArtifact,
    AnytimeWorkloadAxis,
    build_anytime_analysis,
)
from abstrak.anytime.artifacts import (
    AnytimeAttemptAudit,
    AnytimeAttemptHeader,
    AnytimeAttemptIdentity,
    AnytimeAttemptWriter,
    AnytimeInjectedCrash,
    AnytimePersistedProviderObservation,
    AnytimeWorkerEvaluationArtifact,
    audit_anytime_attempt,
    create_anytime_retry,
    recover_anytime_attempt,
)
from abstrak.anytime.context import AnytimeSourceSnapshot
from abstrak.anytime.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    AnytimeLoopPolicy,
    AnytimeModel,
)
from abstrak.anytime.figures import (
    AnytimeAnalysisBundleManifest,
    AnytimeFigureManifest,
    verify_anytime_analysis_bundle,
    write_anytime_analysis_bundle,
)
from abstrak.anytime.floor import (
    AnytimeFloorEvidenceBundle,
    AnytimeFloorValidation,
    validate_anytime_floor_evidence,
)
from abstrak.anytime.freeze import (
    M9_BLOCKERS,
    SHAKEOUT_WORKLOAD_IDS,
    AnytimeOfflineFreezeManifest,
    PinnedAnytimeOfflineFreeze,
    build_anytime_native_manifest_bundle,
    build_anytime_shakeout_study,
    verify_anytime_offline_freeze,
)
from abstrak.anytime.isolation import (
    AnytimeCandidateInvocation,
    AnytimeOutputChannel,
    AnytimeProcessIsolationContract,
    AnytimePublicRuntime,
    AnytimePublicTaskABI,
    AnytimeTensorDescriptor,
    build_anytime_candidate_source,
    build_anytime_process_isolation_contract,
)
from abstrak.anytime.ledger import (
    AnytimeLedgerHeader,
    AnytimeQualificationPending,
    AnytimeTurnRecord,
    build_anytime_ledger_header,
    prepare_anytime_turn,
    rebuild_anytime_checkpoints,
)
from abstrak.anytime.prompts import render_anytime_base_prompt
from abstrak.anytime.qualification import (
    AnytimeCandidateQualificationBinding,
    AnytimeCuteSyntheticLaunchPayload,
    AnytimeOfflineQualificationDecision,
    AnytimeSyntheticLaunchAttestation,
    AnytimeSyntheticRuntimeObservation,
    AnytimeTileLangSyntheticLaunchPayload,
    AnytimeTritonSyntheticLaunchPayload,
    attest_anytime_synthetic_launch,
    build_anytime_qualification_binding,
    qualify_anytime_candidate_offline,
)
from abstrak.anytime.resume import (
    AnytimePhaseAudit,
    AnytimePhaseJournal,
    AnytimePhaseJournalHeader,
    audit_anytime_phase,
)
from abstrak.anytime.schedule import AnytimeScheduleCell, build_anytime_schedule
from abstrak.anytime.workloads import (
    DEFAULT_CANDIDATE_MAX_MEMORY_BYTES,
    DEFAULT_CANDIDATE_MAX_WALL_SECONDS,
    PINNED_INPUT_MANIFEST_SHA256,
    TARGET_IDS,
    PinnedAnytimeWorkloadInputs,
    formal_readiness_issues,
    load_anytime_workload_inputs,
    validate_anytime_workload_inputs,
)
from abstrak.providers.contracts import NormalizedUsage, sha256_json
from abstrak.providers.native_contracts import (
    NativeManifestBundle,
    NativeNormalizedResponse,
)

REHEARSAL_STUDY_SCOPE = "offline_synthetic_fixture"
REHEARSAL_MANIFEST_FILENAME = "rehearsal-manifest.json"
REHEARSAL_RECEIPT_FILENAME = "rehearsal-receipt.json"
ANALYSIS_DATASET_FILENAME = "analysis-dataset.json"
FLOOR_VALIDATION_FILENAME = "floor-validation.json"
ANALYSIS_DIRECTORY = "analysis"
QUALIFICATION_DIRECTORY = "qualifications"
FAKE_PROVIDER_WARNING = "offline scripted fixture; no provider request was performed"
QualificationKey = tuple[str, int, int]

_SHA256 = re.compile(SHA256_PATTERN)

_TRITON_SOURCE = """\
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x, y, output, n_elements: tl.constexpr):
    offsets = tl.program_id(0) * 128 + tl.arange(0, 128)
    mask = offsets < n_elements
    left = tl.load(x + offsets, mask=mask)
    right = tl.load(y + offsets, mask=mask)
    tl.store(output + offsets, left + right, mask=mask)

class ModelNew:
    def forward(self, x, y):
        output = torch.empty_like(x)
        n_elements = x.numel()
        add_kernel[(triton.cdiv(n_elements, 128),)](x, y, output, n_elements)
        return output
"""

_TILELANG_SOURCE = """\
import tilelang
import tilelang.language as T

@T.prim_func
def add_kernel(x, y, output):
    with T.Kernel(1, threads=128):
        T.copy(x, output)

compiled_kernel = tilelang.compile(add_kernel, target="cuda", out_idx=[2])

class ModelNew:
    def forward(self, x, y):
        return compiled_kernel(x, y)
"""

_CUTE_SOURCE = """\
import cutlass.cute as cute

@cute.kernel
def add_kernel(x, y, output):
    cute.copy(x, output)

compiled_kernel = cute.compile(add_kernel)

class ModelNew:
    def forward(self, x, y):
        return compiled_kernel(x, y)
"""

_BASE_SOURCE = {
    "triton-a100": ("triton", _TRITON_SOURCE),
    "tilelang-a100": ("tilelang", _TILELANG_SOURCE),
    "cute-a100": ("cute", _CUTE_SOURCE),
}


class AnytimeRehearsalError(ValueError):
    """Raised when a scripted rehearsal is incomplete or has been modified."""


class AnytimeOfflineQualificationFixture(AnytimeModel):
    schema_version: Literal["abstrak-anytime-offline-qualification-fixture.v2"] = (
        "abstrak-anytime-offline-qualification-fixture.v2"
    )
    evidence_scope: Literal["offline_synthetic_fixture"] = REHEARSAL_STUDY_SCOPE
    agent_id: str = Field(pattern=IDENTIFIER_PATTERN)
    workload_id: str = Field(pattern=IDENTIFIER_PATTERN)
    trajectory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    infrastructure_attempt_index: int = Field(ge=1, le=2)
    trajectory_execution_sha256: str = Field(pattern=SHA256_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    scientific_call_index: int = Field(ge=1, le=4)
    logical_request_sha256: str = Field(pattern=SHA256_PATTERN)
    provider_observation_sha256: str = Field(pattern=SHA256_PATTERN)
    invocation: AnytimeCandidateInvocation
    binding: AnytimeCandidateQualificationBinding
    isolation_contract: AnytimeProcessIsolationContract
    runtime_observation: AnytimeSyntheticRuntimeObservation
    launch_attestation: AnytimeSyntheticLaunchAttestation
    decision: AnytimeOfflineQualificationDecision
    candidate_source_executed: Literal[False] = False
    gpu_code_executed: Literal[False] = False

    @model_validator(mode="after")
    def bindings_are_consistent(self) -> AnytimeOfflineQualificationFixture:
        if self.target_id != self.binding.target_id:
            raise ValueError("qualification fixture target differs from its binding")
        if self.invocation.sha256 != self.binding.candidate_invocation_sha256:
            raise ValueError("qualification fixture invocation differs from its binding")
        if self.binding.execution_binding_sha256 != _qualification_execution_binding_sha256(
            agent_id=self.agent_id,
            workload_id=self.workload_id,
            trajectory_id=self.trajectory_id,
            infrastructure_attempt_index=self.infrastructure_attempt_index,
            trajectory_execution_sha256=self.trajectory_execution_sha256,
            target_id=self.target_id,
            scientific_call_index=self.scientific_call_index,
            logical_request_sha256=self.logical_request_sha256,
            provider_observation_sha256=self.provider_observation_sha256,
            invocation=self.invocation,
        ):
            raise ValueError("qualification fixture execution identity is not fully bound")
        if self.decision.binding_sha256 != self.binding.sha256:
            raise ValueError("qualification decision differs from its binding")
        if self.decision.isolation_contract_sha256 != self.isolation_contract.sha256:
            raise ValueError("qualification decision differs from its isolation contract")
        if self.decision.status != "pending-m9":
            raise ValueError("passing offline fixture must remain pending M9")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeRehearsalArtifactBinding(AnytimeModel):
    schema_version: Literal["abstrak-anytime-rehearsal-artifact-binding.v2"] = (
        "abstrak-anytime-rehearsal-artifact-binding.v2"
    )
    relative_path: str
    raw_sha256: str = Field(pattern=SHA256_PATTERN)
    canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    trajectory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    infrastructure_attempt_index: int = Field(ge=1, le=2)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    scientific_call_index: int = Field(ge=1, le=4)
    execution_binding_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)


class AnytimeOfflinePhaseReceipt(AnytimeModel):
    schema_version: Literal["abstrak-anytime-offline-phase-receipt.v1"] = (
        "abstrak-anytime-offline-phase-receipt.v1"
    )
    phase_id: str = Field(pattern=IDENTIFIER_PATTERN)
    agent_id: str = Field(pattern=IDENTIFIER_PATTERN)
    workload_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    expected_trajectory_ids: tuple[str, str]
    phase_header_sha256: str = Field(pattern=SHA256_PATTERN)
    journal_head_sha256: str = Field(pattern=SHA256_PATTERN)
    close_audit_sha256: str = Field(pattern=SHA256_PATTERN)
    attempt_count: int = Field(ge=2, le=3)
    scientific_calls_consumed: Literal[8] = 8
    provider_requests_submitted: Literal[8] = 8
    checkpoint_count: Literal[4] = 4

    @field_validator("expected_trajectory_ids")
    @classmethod
    def trajectories_are_sorted(cls, value: tuple[str, str]) -> tuple[str, str]:
        if value != tuple(sorted(set(value))) or len(value) != 2:
            raise ValueError("offline phase must contain two sorted replicate trajectories")
        return value

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeOfflineSideEffectBoundary(AnytimeModel):
    schema_version: Literal["abstrak-anytime-offline-side-effect-boundary.v1"] = (
        "abstrak-anytime-offline-side-effect-boundary.v1"
    )
    credentials_read: Literal[False] = False
    provider_clients_created: Literal[False] = False
    live_requests_performed: Literal[False] = False
    ssh_connections_performed: Literal[False] = False
    gpu_apis_called: Literal[False] = False
    gpu_code_executed: Literal[False] = False
    candidate_code_executed: Literal[False] = False
    model_generated_code_created: Literal[False] = False
    formal_authorization_emitted: Literal[False] = False


class AnytimeOfflineRehearsalReceipt(AnytimeModel):
    schema_version: Literal["abstrak-anytime-offline-rehearsal-receipt.v2"] = (
        "abstrak-anytime-offline-rehearsal-receipt.v2"
    )
    rehearsal_id: Literal["anytime-dsl-a100-m8-full-shakeout-fixture"] = (
        "anytime-dsl-a100-m8-full-shakeout-fixture"
    )
    evidence_scope: Literal["offline_synthetic_fixture"] = REHEARSAL_STUDY_SCOPE
    freeze_raw_sha256: str = Field(pattern=SHA256_PATTERN)
    freeze_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    shakeout_study_sha256: str = Field(pattern=SHA256_PATTERN)
    shakeout_schedule_sha256: str = Field(pattern=SHA256_PATTERN)
    planned_trajectories: Literal[48] = 48
    scientific_model_call_ceiling: Literal[192] = 192
    operational_provider_request_ceiling: Literal[384] = 384
    completed_trajectories: Literal[48] = 48
    phase_count: Literal[24] = 24
    attempt_count: Literal[49] = 49
    infrastructure_retry_count: Literal[1] = 1
    scripted_provider_response_count: Literal[192] = 192
    scripted_worker_artifact_count: Literal[192] = 192
    checkpoint_count: Literal[96] = 96
    provider_protocol_counts: tuple[tuple[str, int], tuple[str, int]]
    candidate_outcome_counts: tuple[tuple[str, int], ...]
    terminal_counts: tuple[tuple[str, int], ...]
    phase_receipts: tuple[AnytimeOfflinePhaseReceipt, ...] = Field(min_length=24, max_length=24)
    phase_receipt_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    qualifications: tuple[AnytimeRehearsalArtifactBinding, ...] = Field(
        min_length=192,
        max_length=192,
    )
    qualification_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    pending_m9_qualification_count: Literal[192] = 192
    floor_validation_raw_sha256: str = Field(pattern=SHA256_PATTERN)
    floor_validation_sha256: str = Field(pattern=SHA256_PATTERN)
    floor_gate: Literal["invalid_floor"] = "invalid_floor"
    formal_readiness: Literal["blocked"] = "blocked"
    formal_readiness_issue_count: int = Field(ge=1)
    formal_readiness_issues_sha256: str = Field(pattern=SHA256_PATTERN)
    analysis_dataset_raw_sha256: str = Field(pattern=SHA256_PATTERN)
    analysis_dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    analysis_bundle_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    analysis_trajectory_count: Literal[48] = 48
    analysis_floor_count: Literal[12] = 12
    analysis_figure_count: Literal[7] = 7
    environment_status: Literal["pending-m9"] = "pending-m9"
    next_gate: Literal["m9-trusted-gpu-preflight"] = "m9-trusted-gpu-preflight"
    m9_blockers: tuple[str, ...] = Field(min_length=1)
    side_effects: AnytimeOfflineSideEffectBoundary

    @model_validator(mode="after")
    def closure_is_exact_and_blocked(self) -> AnytimeOfflineRehearsalReceipt:
        if self.provider_protocol_counts != (
            ("chat_completions", 96),
            ("responses", 96),
        ):
            raise ValueError("scripted provider protocol counts differ from the full shakeout")
        if self.candidate_outcome_counts != (("qualification_pending", 192),):
            raise ValueError("scripted candidate outcomes differ from the frozen fixture")
        if self.terminal_counts != (("infrastructure_failure", 1), ("success", 48)):
            raise ValueError("attempt terminal counts differ from crash/retry fixture")
        if self.phase_receipt_bundle_sha256 != _model_sequence_sha256(self.phase_receipts):
            raise ValueError("phase receipt bundle hash mismatch")
        if self.qualification_bundle_sha256 != _model_sequence_sha256(self.qualifications):
            raise ValueError("qualification bundle hash mismatch")
        qualification_paths = tuple(item.relative_path for item in self.qualifications)
        if qualification_paths != tuple(sorted(set(qualification_paths))):
            raise ValueError("qualification bindings must be path-sorted and unique")
        if self.m9_blockers != M9_BLOCKERS:
            raise ValueError("offline rehearsal M9 blockers differ from the freeze")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeOfflineRehearsalFile(AnytimeModel):
    schema_version: Literal["abstrak-anytime-offline-rehearsal-file.v1"] = (
        "abstrak-anytime-offline-rehearsal-file.v1"
    )
    relative_path: str
    role: Literal[
        "analysis",
        "attempt",
        "derived-index",
        "floor-validation",
        "phase-journal",
        "qualification",
        "rehearsal-receipt",
    ]
    raw_sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=1)

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)


class AnytimeOfflineRehearsalManifest(AnytimeModel):
    schema_version: Literal["abstrak-anytime-offline-rehearsal-manifest.v1"] = (
        "abstrak-anytime-offline-rehearsal-manifest.v1"
    )
    receipt_raw_sha256: str = Field(pattern=SHA256_PATTERN)
    receipt_sha256: str = Field(pattern=SHA256_PATTERN)
    files: tuple[AnytimeOfflineRehearsalFile, ...] = Field(min_length=1)
    file_bundle_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def inventory_is_canonical(self) -> AnytimeOfflineRehearsalManifest:
        paths = tuple(item.relative_path for item in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("rehearsal file inventory must be sorted and unique")
        if REHEARSAL_MANIFEST_FILENAME in paths:
            raise ValueError("rehearsal manifest cannot inventory itself")
        if self.file_bundle_sha256 != _model_sequence_sha256(self.files):
            raise ValueError("rehearsal file bundle hash mismatch")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


@dataclass(frozen=True)
class AnytimeOfflineRehearsalResult:
    directory: Path
    receipt: AnytimeOfflineRehearsalReceipt
    manifest: AnytimeOfflineRehearsalManifest


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


def _assert_offline_rehearsal_source() -> None:
    """Reject accidental live/GPU imports or dynamic candidate execution in this module."""

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden_import_roots = {
        "cutlass",
        "litellm",
        "paramiko",
        "socket",
        "subprocess",
        "tilelang",
        "torch",
        "triton",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = (alias.name.split(".", 1)[0] for alias in node.names)
            if any(name in forbidden_import_roots for name in imported):
                raise AnytimeRehearsalError("offline rehearsal source imports a live or GPU module")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in forbidden_import_roots:
                raise AnytimeRehearsalError("offline rehearsal source imports a live or GPU module")
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id in {"compile", "eval", "exec"}:
                raise AnytimeRehearsalError("offline rehearsal source executes dynamic code")


def _canonical_json_bytes(value: Any) -> bytes:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return (
        json.dumps(
            payload,
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(root: Path, relative_path: str, value: Any) -> tuple[Path, str]:
    relative = PurePosixPath(_safe_relative_path(relative_path))
    destination = root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = _canonical_json_bytes(value)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise AnytimeRehearsalError(
            f"rehearsal artifact already exists: {relative_path}"
        ) from error
    return destination, _raw_sha256(payload)


def _read_model(path: Path, model: type[AnytimeModel]) -> Any:
    if path.is_symlink() or not path.is_file():
        raise AnytimeRehearsalError(f"rehearsal artifact is missing or unsafe: {path}")
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as error:
        raise AnytimeRehearsalError(f"invalid rehearsal artifact {path}: {error}") from error


def _target_source(target_id: str, scientific_call_index: int) -> str:
    try:
        _, source = _BASE_SOURCE[target_id]
    except KeyError:
        raise AnytimeRehearsalError(f"unsupported fixture target: {target_id}") from None
    return f"{source.rstrip()}\n\n# offline synthetic fixture call {scientific_call_index}\n"


def _phase_id(agent_id: str, workload_id: str, target_id: str) -> str:
    return f"fixture-{agent_id}-{workload_id}-{target_id}"


def _trajectory_seed(execution_sha256: str) -> int:
    return int(execution_sha256[:16], 16)


def _fixture_invocation(
    *,
    target_id: str,
    source: str,
) -> AnytimeCandidateInvocation:
    backend, _ = _BASE_SOURCE[target_id]
    return AnytimeCandidateInvocation(
        source=build_anytime_candidate_source(source),
        public_abi=AnytimePublicTaskABI(
            abi_id="offline-vector-add-fixture",
            abi_version="v1",
            entrypoint="ModelNew.forward",
            input_names=("x", "y"),
            output_count=1,
        ),
        public_runtime=AnytimePublicRuntime(
            target_id=target_id,
            backend=backend,
            runtime_id=f"offline-{backend}-runtime",
            runtime_abi_version="kernelbench-v1",
        ),
        inputs=(
            AnytimeTensorDescriptor(
                name="x",
                shape=(32, 32),
                strides=(32, 1),
                dtype="float16",
            ),
            AnytimeTensorDescriptor(
                name="y",
                shape=(32, 32),
                strides=(32, 1),
                dtype="float16",
            ),
        ),
        output_channel=AnytimeOutputChannel(
            channel_id=hashlib.sha256(
                f"offline-channel\0{target_id}\0{source}".encode()
            ).hexdigest()[:32],
            expected_output_count=1,
        ),
    )


def _qualification_execution_binding_sha256(
    *,
    agent_id: str,
    workload_id: str,
    trajectory_id: str,
    infrastructure_attempt_index: int,
    trajectory_execution_sha256: str,
    target_id: str,
    scientific_call_index: int,
    logical_request_sha256: str,
    provider_observation_sha256: str,
    invocation: AnytimeCandidateInvocation,
) -> str:
    return sha256_json(
        {
            "schema_version": "offline-fixture-execution-binding.v2",
            "agent_id": agent_id,
            "workload_id": workload_id,
            "trajectory_id": trajectory_id,
            "infrastructure_attempt_index": infrastructure_attempt_index,
            "trajectory_execution_sha256": trajectory_execution_sha256,
            "target_id": target_id,
            "scientific_call_index": scientific_call_index,
            "logical_request_sha256": logical_request_sha256,
            "provider_observation_sha256": provider_observation_sha256,
            "candidate_source_sha256": invocation.source.source_sha256,
            "output_channel_id": invocation.output_channel.channel_id,
        }
    )


def _qualification_fixture(
    *,
    inputs: PinnedAnytimeWorkloadInputs,
    cell: AnytimeScheduleCell,
    identity: AnytimeAttemptIdentity,
    scientific_call_index: int,
    logical_request_sha256: str,
    provider_observation_sha256: str,
    source: str,
) -> AnytimeOfflineQualificationFixture:
    invocation = _fixture_invocation(target_id=cell.target_id, source=source)
    card = inputs.manifest.target_card(cell.target_id)
    execution_binding = _qualification_execution_binding_sha256(
        agent_id=cell.agent_id,
        workload_id=cell.task_id,
        trajectory_id=identity.trajectory_id,
        infrastructure_attempt_index=identity.infrastructure_attempt_index,
        trajectory_execution_sha256=identity.trajectory_execution_sha256,
        target_id=cell.target_id,
        scientific_call_index=scientific_call_index,
        logical_request_sha256=logical_request_sha256,
        provider_observation_sha256=provider_observation_sha256,
        invocation=invocation,
    )
    binding = build_anytime_qualification_binding(
        invocation=invocation,
        target_stack_sha256=card.sha256,
        execution_binding_sha256=execution_binding,
    )
    isolation = build_anytime_process_isolation_contract(
        max_wall_seconds=DEFAULT_CANDIDATE_MAX_WALL_SECONDS,
        max_memory_bytes=DEFAULT_CANDIDATE_MAX_MEMORY_BYTES,
    )
    runtime = AnytimeSyntheticRuntimeObservation(
        candidate_invocation_sha256=binding.candidate_invocation_sha256,
        candidate_source_sha256=binding.candidate_source_sha256,
        execution_binding_sha256=binding.execution_binding_sha256,
        terminal_status="completed",
        expected_output_count=1,
        observed_output_count=1,
        outputs_finite=True,
        inputs_unchanged=True,
        ipc_envelope_valid=True,
        elapsed_seconds=0.0,
    )
    payload_fields = dict(
        target_id=binding.target_id,
        target_stack_sha256=binding.target_stack_sha256,
        candidate_source_sha256=binding.candidate_source_sha256,
        candidate_invocation_sha256=binding.candidate_invocation_sha256,
        execution_binding_sha256=binding.execution_binding_sha256,
        runtime_launch_count=1,
        launched_kernel_sha256=sha256_json(
            {"fixture": "launch", "execution_binding": execution_binding}
        ),
        lowered_code_sha256=sha256_json(
            {"fixture": "lowered", "execution_binding": execution_binding}
        ),
        core_operation_attributed=True,
        fallback_detected=False,
        dummy_signature_only=False,
    )
    payload_type: dict[str, type[Any]] = {
        "triton-a100": AnytimeTritonSyntheticLaunchPayload,
        "tilelang-a100": AnytimeTileLangSyntheticLaunchPayload,
        "cute-a100": AnytimeCuteSyntheticLaunchPayload,
    }
    attestation = attest_anytime_synthetic_launch(
        payload_type[cell.target_id](**payload_fields)
    )
    decision = qualify_anytime_candidate_offline(
        binding=binding,
        invocation=invocation,
        isolation_contract=isolation,
        runtime_observation=runtime,
        launch_attestation=attestation,
    )
    return AnytimeOfflineQualificationFixture(
        agent_id=cell.agent_id,
        workload_id=cell.task_id,
        trajectory_id=identity.trajectory_id,
        infrastructure_attempt_index=identity.infrastructure_attempt_index,
        trajectory_execution_sha256=identity.trajectory_execution_sha256,
        target_id=cell.target_id,
        scientific_call_index=scientific_call_index,
        logical_request_sha256=logical_request_sha256,
        provider_observation_sha256=provider_observation_sha256,
        invocation=invocation,
        binding=binding,
        isolation_contract=isolation,
        runtime_observation=runtime,
        launch_attestation=attestation,
        decision=decision,
    )


def _qualification_path(
    trajectory_id: str,
    infrastructure_attempt_index: int,
    scientific_call_index: int,
) -> str:
    return (
        f"{QUALIFICATION_DIRECTORY}/{trajectory_id}/"
        f"attempt-{infrastructure_attempt_index:02d}/call-{scientific_call_index:02d}.json"
    )


def _phase_prompt(
    inputs: PinnedAnytimeWorkloadInputs,
    *,
    task_id: str,
    target_id: str,
    repository_root: Path,
    freeze: AnytimeOfflineFreezeManifest,
) -> tuple[Any, ...]:
    return render_anytime_base_prompt(
        inputs,
        task_id=task_id,
        target_id=target_id,
        repository_root=repository_root,
        policy=freeze.base_prompt,
    )


def _phase_execution_sha256(
    *,
    freeze: AnytimeOfflineFreezeManifest,
    schedule_sha256: str,
    agent_id: str,
    workload_id: str,
    target_id: str,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[Any, ...],
) -> str:
    return sha256_json(
        {
            "schema_version": "offline-fixture-phase-execution.v1",
            "freeze_sha256": freeze.sha256,
            "schedule_sha256": schedule_sha256,
            "agent_id": agent_id,
            "workload_id": workload_id,
            "target_id": target_id,
            "loop_sha256": loop.sha256,
            "base_prompt_sha256": sha256_json(
                [message.model_dump(mode="json") for message in base_prompt]
            ),
        }
    )


def _trajectory_execution_sha256(
    *,
    phase_execution_sha256: str,
    cell: AnytimeScheduleCell,
    agent_sha256: str,
    workload_sha256: str,
    target_card_sha256: str,
    environment_sha256: str,
) -> str:
    return sha256_json(
        {
            "schema_version": "offline-fixture-trajectory-execution.v1",
            "phase_execution_sha256": phase_execution_sha256,
            "cell": cell.model_dump(mode="json"),
            "agent_sha256": agent_sha256,
            "workload_sha256": workload_sha256,
            "target_card_sha256": target_card_sha256,
            "environment_sha256": environment_sha256,
        }
    )


def _fake_native_response(
    *,
    prepared: Any,
    cell: AnytimeScheduleCell,
    infrastructure_attempt_index: int,
    bundle: NativeManifestBundle,
    freeze: AnytimeOfflineFreezeManifest,
    source: str,
) -> NativeNormalizedResponse:
    dependency = next(
        item for item in freeze.provider_dependencies if item.agent_id == cell.agent_id
    )
    call = prepared.scientific_call_index
    started = datetime(2026, 8, 3, tzinfo=timezone.utc) + timedelta(
        seconds=cell.ordinal * 10 + call,
        milliseconds=infrastructure_attempt_index,
    )
    text = f"```python\n{source.rstrip()}\n```"
    input_characters = sum(
        len(message.content) for message in prepared.logical_request.messages
    )
    input_tokens = 100 + call
    output_tokens = 32 + call
    reasoning_tokens = 16 + call
    if bundle.provider.protocol == "chat_completions":
        sanitized_request: dict[str, Any] = {
            "model": bundle.model.api_model,
            "max_completion_tokens": bundle.model.max_output_tokens,
            "stream": False,
            "thinking": {"type": "enabled"},
        }
        finish_reason = "stop"
    else:
        sanitized_request = {
            "model": bundle.model.api_model,
            "max_output_tokens": bundle.model.max_output_tokens,
            "reasoning": {"effort": "xhigh"},
            "store": False,
            "truncation": "disabled",
        }
        finish_reason = "completed"
    raw_response = {
        "id": f"offline-{cell.trajectory_id}-call-{call:02d}",
        "model": cell.agent_id,
        "status": "completed",
        "origin": "offline-synthetic-fixture",
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return NativeNormalizedResponse(
        request_id=prepared.logical_request.request_id,
        attempt_id=(
            f"offline-{cell.trajectory_id}-a{infrastructure_attempt_index}-call-{call:02d}"
        ),
        provider_request_id=raw_response["id"],
        provider_id=bundle.provider.id,
        model_id=bundle.model.id,
        protocol=bundle.provider.protocol,
        provider_manifest_sha256=bundle.provider_sha256,
        model_manifest_sha256=bundle.model_sha256,
        requested_model=bundle.model.api_model,
        returned_model=cell.agent_id,
        text=text,
        finish_reason=finish_reason,
        provider_finish_reason=finish_reason,
        usage=NormalizedUsage(
            input_tokens=input_tokens,
            cached_input_tokens=0,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=input_tokens + output_tokens,
            input_characters=input_characters,
            output_characters=len(text),
            provider_reported=True,
            core_fields_complete=True,
            raw_usage={
                "origin": "offline-synthetic-fixture",
                "input_tokens": input_tokens,
                "cached_input_tokens": 0,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
        ),
        resource_usage_complete=True,
        reasoning=dependency.conformance.reasoning,
        started_at_utc=started,
        finished_at_utc=started + timedelta(milliseconds=10),
        elapsed_ms=10.0,
        logical_request_sha256=sha256_json(prepared.logical_request),
        transport_request_sha256=sha256_json(sanitized_request),
        transport_response_sha256=sha256_json(raw_response),
        sanitized_transport_request=sanitized_request,
        raw_transport_response=raw_response,
        warnings=(FAKE_PROVIDER_WARNING,),
    )


def _fake_worker_artifact(
    *,
    provider: AnytimePersistedProviderObservation,
    cell: AnytimeScheduleCell,
    scientific_call_index: int,
    source: str,
    qualification_raw_sha256: str,
) -> AnytimeWorkerEvaluationArtifact:
    return AnytimeWorkerEvaluationArtifact(
        provider_observation_sha256=provider.sha256,
        evaluator_id="offline-scripted-worker-v2",
        evaluator_execution_sha256=sha256_json(
            {
                "schema_version": "offline-scripted-worker.v2",
                "trajectory_id": cell.trajectory_id,
                "logical_request_sha256": provider.observation.logical_request_sha256,
                "target_id": cell.target_id,
                "scientific_call_index": scientific_call_index,
                "candidate_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "qualification_raw_sha256": qualification_raw_sha256,
            }
        ),
        candidate=AnytimeQualificationPending(
            source=AnytimeSourceSnapshot.from_source(source),
            diagnostics=(
                "offline synthetic worker only; compile, correctness, and target use remain "
                "pending M9",
            ),
        ),
        observed_wall_seconds=0.02,
        qualification_artifact_sha256=qualification_raw_sha256,
    )


def _complete_scripted_attempt(
    writer: AnytimeAttemptWriter,
    *,
    rehearsal_root: Path,
    inputs: PinnedAnytimeWorkloadInputs,
    cell: AnytimeScheduleCell,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[Any, ...],
    bundle: NativeManifestBundle,
    freeze: AnytimeOfflineFreezeManifest,
    qualification_fixtures: dict[QualificationKey, AnytimeOfflineQualificationFixture],
    qualification_raw_hashes: dict[QualificationKey, str],
    qualification_bindings: list[AnytimeRehearsalArtifactBinding],
) -> AnytimeAttemptAudit:
    records: tuple[AnytimeTurnRecord, ...] = ()
    checkpoints: tuple[Any, ...] = ()
    for _ in range(loop.budget.max_scientific_calls):
        prepared = prepare_anytime_turn(
            header=writer.header.ledger_header,
            loop=loop,
            base_prompt=base_prompt,
            records=records,
            checkpoints=checkpoints,
        )
        call = prepared.scientific_call_index
        source = _target_source(cell.target_id, call)
        writer.persist_dispatch_intent(prepared)
        provider = writer.persist_native_provider_response(
            scientific_call_index=call,
            response=_fake_native_response(
                prepared=prepared,
                cell=cell,
                infrastructure_attempt_index=(
                    writer.header.identity.infrastructure_attempt_index
                ),
                bundle=bundle,
                freeze=freeze,
                source=source,
            ),
            observed_wall_seconds=0.01,
        )
        identity = writer.header.identity
        key = (identity.trajectory_id, identity.infrastructure_attempt_index, call)
        if key in qualification_fixtures:
            raise AnytimeRehearsalError("duplicate execution-bound qualification fixture")
        fixture = _qualification_fixture(
            inputs=inputs,
            cell=cell,
            identity=identity,
            scientific_call_index=call,
            logical_request_sha256=sha256_json(prepared.logical_request),
            provider_observation_sha256=provider.sha256,
            source=source,
        )
        relative_path = _qualification_path(
            identity.trajectory_id,
            identity.infrastructure_attempt_index,
            call,
        )
        _, qualification_raw_sha256 = _write_new(
            rehearsal_root,
            relative_path,
            fixture,
        )
        qualification_fixtures[key] = fixture
        qualification_raw_hashes[key] = qualification_raw_sha256
        qualification_bindings.append(
            AnytimeRehearsalArtifactBinding(
                relative_path=relative_path,
                raw_sha256=qualification_raw_sha256,
                canonical_sha256=fixture.sha256,
                trajectory_id=identity.trajectory_id,
                infrastructure_attempt_index=identity.infrastructure_attempt_index,
                target_id=cell.target_id,
                scientific_call_index=call,
                execution_binding_sha256=fixture.binding.execution_binding_sha256,
            )
        )
        worker = _fake_worker_artifact(
            provider=provider,
            cell=cell,
            scientific_call_index=call,
            source=source,
            qualification_raw_sha256=qualification_raw_sha256,
        )
        writer.persist_worker_evaluation(
            scientific_call_index=call,
            worker_artifact=worker,
        )
        record = writer.persist_derived_turn(
            loop=loop,
            base_prompt=base_prompt,
            scientific_call_index=call,
        )
        records = (*records, record)
        checkpoints = rebuild_anytime_checkpoints(
            header=writer.header.ledger_header,
            loop=loop,
            base_prompt=base_prompt,
            records=records,
        )
    writer.persist_terminal(
        loop=loop,
        base_prompt=base_prompt,
        terminal_kind="success",
        reason="call_budget_complete",
    )
    return writer.seal_and_promote(loop=loop, base_prompt=base_prompt)


def _crash_after_attempt_create(point: str) -> None:
    if point == "after_attempt_create":
        raise AnytimeInjectedCrash(point)


def _ledger_header(
    *,
    cell: AnytimeScheduleCell,
    infrastructure_attempt_index: int,
    trajectory_execution_sha256: str,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[Any, ...],
    freeze: AnytimeOfflineFreezeManifest,
    inputs: PinnedAnytimeWorkloadInputs,
) -> AnytimeLedgerHeader:
    study = build_anytime_shakeout_study()
    agent = study.agent(cell.agent_id)
    provider = next(
        item for item in freeze.provider_dependencies if item.agent_id == cell.agent_id
    )
    workload = inputs.manifest.workload(cell.task_id)
    target_card = inputs.manifest.target_card(cell.target_id)
    return build_anytime_ledger_header(
        trajectory_id=cell.trajectory_id,
        infrastructure_attempt_index=infrastructure_attempt_index,
        trajectory_execution_sha256=trajectory_execution_sha256,
        agent_binding_sha256=sha256_json(
            {
                "agent_sha256": agent.sha256,
                "provider_dependency_sha256": provider.conformance_sha256,
            }
        ),
        task_sha256=workload.sha256,
        target_sha256=target_card.sha256,
        environment_sha256=inputs.manifest.environment.sha256,
        model_ref=agent.model_ref,
        loop=loop,
        base_prompt=base_prompt,
        local_trajectory_seed=_trajectory_seed(trajectory_execution_sha256),
    )


def _run_phase(
    root: Path,
    *,
    cells: tuple[AnytimeScheduleCell, AnytimeScheduleCell],
    inject_crash: bool,
    freeze: AnytimeOfflineFreezeManifest,
    inputs: PinnedAnytimeWorkloadInputs,
    repository_root: Path,
    schedule_sha256: str,
    qualification_fixtures: dict[QualificationKey, AnytimeOfflineQualificationFixture],
    qualification_raw_hashes: dict[QualificationKey, str],
    qualification_bindings: list[AnytimeRehearsalArtifactBinding],
) -> tuple[AnytimeOfflinePhaseReceipt, dict[str, AnytimeAttemptAudit]]:
    first = cells[0]
    study = build_anytime_shakeout_study()
    loop = study.cohort(first.cohort_id).loop
    phase_id = _phase_id(first.agent_id, first.task_id, first.target_id)
    base_prompt = _phase_prompt(
        inputs,
        task_id=first.task_id,
        target_id=first.target_id,
        repository_root=repository_root,
        freeze=freeze,
    )
    phase_execution_sha256 = _phase_execution_sha256(
        freeze=freeze,
        schedule_sha256=schedule_sha256,
        agent_id=first.agent_id,
        workload_id=first.task_id,
        target_id=first.target_id,
        loop=loop,
        base_prompt=base_prompt,
    )
    expected_trajectory_ids = tuple(sorted(cell.trajectory_id for cell in cells))
    if len(expected_trajectory_ids) != 2:
        raise AnytimeRehearsalError("offline phase does not contain exactly two replicates")
    journal = AnytimePhaseJournal.create(
        root,
        study_id=study.study_id,
        phase_id=phase_id,
        phase_execution_sha256=phase_execution_sha256,
        expected_trajectory_ids=expected_trajectory_ids,
        max_attempts_per_trajectory=2,
    )
    successful: dict[str, AnytimeAttemptAudit] = {}
    bundle = build_anytime_native_manifest_bundle(study.agent(first.agent_id))
    for cell_index, cell in enumerate(cells):
        if (cell.agent_id, cell.task_id, cell.target_id) != (
            first.agent_id,
            first.task_id,
            first.target_id,
        ):
            raise AnytimeRehearsalError("phase cells do not share one prompt and loop")
        workload = inputs.manifest.workload(cell.task_id)
        target_card = inputs.manifest.target_card(cell.target_id)
        trajectory_execution = _trajectory_execution_sha256(
            phase_execution_sha256=phase_execution_sha256,
            cell=cell,
            agent_sha256=study.agent(cell.agent_id).sha256,
            workload_sha256=workload.sha256,
            target_card_sha256=target_card.sha256,
            environment_sha256=inputs.manifest.environment.sha256,
        )
        primary_identity = AnytimeAttemptIdentity(
            study_id=study.study_id,
            phase_id=phase_id,
            trajectory_id=cell.trajectory_id,
            infrastructure_attempt_index=1,
            trajectory_execution_sha256=trajectory_execution,
        )
        primary_header = _ledger_header(
            cell=cell,
            infrastructure_attempt_index=1,
            trajectory_execution_sha256=trajectory_execution,
            loop=loop,
            base_prompt=base_prompt,
            freeze=freeze,
            inputs=inputs,
        )
        use_retry = inject_crash and cell_index == 0
        if use_retry:
            try:
                AnytimeAttemptWriter.create(
                    root,
                    identity=primary_identity,
                    ledger_header=primary_header,
                    fault=_crash_after_attempt_create,
                )
            except AnytimeInjectedCrash:
                pass
            else:
                raise AnytimeRehearsalError("offline crash fixture did not trigger")
            recovered = recover_anytime_attempt(
                root,
                primary_identity,
                loop=loop,
                base_prompt=base_prompt,
            )
            if recovered.action != "finalized" or recovered.request_submitted:
                raise AnytimeRehearsalError("offline crash recovery crossed request submission")
            primary_audit = audit_anytime_attempt(
                Path(recovered.directory or ""),
                loop=loop,
                base_prompt=base_prompt,
            )
            if (
                primary_audit.terminal.terminal_kind != "infrastructure_failure"
                or not primary_audit.terminal.retry_eligible
            ):
                raise AnytimeRehearsalError("offline crash did not create a retryable tombstone")
            journal.append_attempt(
                primary_audit.identity,
                loop=loop,
                base_prompt=base_prompt,
            )
            retry_header = _ledger_header(
                cell=cell,
                infrastructure_attempt_index=2,
                trajectory_execution_sha256=trajectory_execution,
                loop=loop,
                base_prompt=base_prompt,
                freeze=freeze,
                inputs=inputs,
            )
            writer = create_anytime_retry(
                root,
                primary_audit.identity,
                ledger_header=retry_header,
                loop=loop,
                base_prompt=base_prompt,
            )
        else:
            writer = AnytimeAttemptWriter.create(
                root,
                identity=primary_identity,
                ledger_header=primary_header,
            )
        completed = _complete_scripted_attempt(
            writer,
            rehearsal_root=root,
            inputs=inputs,
            cell=cell,
            loop=loop,
            base_prompt=base_prompt,
            bundle=bundle,
            freeze=freeze,
            qualification_fixtures=qualification_fixtures,
            qualification_raw_hashes=qualification_raw_hashes,
            qualification_bindings=qualification_bindings,
        )
        journal.append_attempt(completed.identity, loop=loop, base_prompt=base_prompt)
        successful[cell.trajectory_id] = completed
    closed = journal.close(loop=loop, base_prompt=base_prompt)
    audit = audit_anytime_phase(root, journal.header, loop=loop, base_prompt=base_prompt)
    if closed.close_audit is None or audit.close_audit is None:
        raise AnytimeRehearsalError("offline phase did not produce a close audit")
    checkpoint_count = sum(
        len(tuple(Path(item.directory).glob("checkpoints/*.json")))
        for item in audit.attempts
    )
    receipt = AnytimeOfflinePhaseReceipt(
        phase_id=phase_id,
        agent_id=first.agent_id,
        workload_id=first.task_id,
        target_id=first.target_id,
        expected_trajectory_ids=expected_trajectory_ids,
        phase_header_sha256=journal.header.sha256,
        journal_head_sha256=audit.journal_head_sha256,
        close_audit_sha256=audit.close_audit.sha256,
        attempt_count=len(audit.attempts),
        scientific_calls_consumed=audit.close_audit.operational_totals.scientific_calls_consumed,
        provider_requests_submitted=(
            audit.close_audit.operational_totals.provider_requests_submitted
        ),
        checkpoint_count=checkpoint_count,
    )
    return receipt, successful


def _load_turn_records(attempt: AnytimeAttemptAudit) -> tuple[AnytimeTurnRecord, ...]:
    directory = Path(attempt.directory)
    try:
        return tuple(
            AnytimeTurnRecord.model_validate_json(path.read_bytes())
            for path in sorted(directory.glob("turns/[0-9][0-9][0-9][0-9]/turn-record.json"))
        )
    except (OSError, ValidationError) as error:
        raise AnytimeRehearsalError(f"cannot project verified turn records: {error}") from error


def _analysis_dataset(
    *,
    inputs: PinnedAnytimeWorkloadInputs,
    successful_attempts: dict[str, AnytimeAttemptAudit],
    floor_validation_raw_sha256: str,
) -> AnytimeAnalysisDataset:
    study = build_anytime_shakeout_study()
    workload_axes = tuple(
        AnytimeWorkloadAxis(
            workload_id=task_id,
            semantic_family_id=inputs.manifest.workload(task_id).family,
        )
        for task_id in SHAKEOUT_WORKLOAD_IDS
    )
    spec = AnytimeAnalysisSpec(
        study_id=study.study_id,
        study_spec_sha256=study.sha256,
        study_stage="synthetic_fixture",
        agents=tuple(
            AnytimeAgentReplicateAxis(agent_id=agent.id, replicates=(1, 2))
            for agent in study.agents
        ),
        workloads=workload_axes,
        targets=TARGET_IDS,
        max_scientific_calls=4,
        formal_checkpoints=(1, 4),
        wall_clock_budgets_seconds=(0.02, 0.08),
        winner_relative_tolerance=0.05,
        bootstrap_seed=20260803,
        bootstrap_resamples=1000,
        confidence_level=0.95,
    )
    schedule = build_anytime_schedule(study)
    by_key = {
        (cell.agent_id, cell.task_id, cell.target_id, cell.replicate): cell
        for cell in schedule.cells
    }
    trajectories: list[AnytimeTrajectoryArtifact] = []
    for agent in spec.agents:
        for workload in spec.workloads:
            for target_id in spec.targets:
                for replicate in agent.replicates:
                    cell = by_key[(agent.agent_id, workload.workload_id, target_id, replicate)]
                    attempt = successful_attempts[cell.trajectory_id]
                    turns = tuple(
                        AnytimeTurnArtifact(
                            scientific_call_index=record.scientific_call_index,
                            cumulative_wall_seconds=record.resource_snapshot.wall_seconds,
                            candidate_stage="qualification_pending",
                        )
                        for record in _load_turn_records(attempt)
                    )
                    trajectories.append(
                        AnytimeTrajectoryArtifact(
                            trust=AnytimeArtifactTrust(
                                artifact_manifest_sha256=attempt.manifest_file_sha256
                            ),
                            study_id=spec.study_id,
                            study_spec_sha256=spec.study_spec_sha256,
                            trajectory_id=cell.trajectory_id,
                            agent_id=cell.agent_id,
                            workload_id=cell.task_id,
                            semantic_family_id=workload.semantic_family_id,
                            target_id=cell.target_id,
                            replicate=cell.replicate,
                            terminal_status="complete",
                            turns=turns,
                        )
                    )
    floors = tuple(
        AnytimeFloorArtifact(
            trust=AnytimeArtifactTrust(
                artifact_manifest_sha256=floor_validation_raw_sha256
            ),
            study_id=spec.study_id,
            study_spec_sha256=spec.study_spec_sha256,
            workload_id=workload.workload_id,
            semantic_family_id=workload.semantic_family_id,
            target_id=target_id,
            status="invalid_floor",
        )
        for workload in spec.workloads
        for target_id in spec.targets
    )
    dataset = AnytimeAnalysisDataset(
        spec=spec,
        floors=floors,
        trajectories=tuple(trajectories),
    )
    build_anytime_analysis(dataset)
    return dataset


def _inventory_role(relative_path: str) -> str:
    if relative_path == REHEARSAL_RECEIPT_FILENAME:
        return "rehearsal-receipt"
    if relative_path == FLOOR_VALIDATION_FILENAME:
        return "floor-validation"
    if relative_path.startswith(f"{QUALIFICATION_DIRECTORY}/"):
        return "qualification"
    if relative_path == ANALYSIS_DATASET_FILENAME or relative_path.startswith(
        f"{ANALYSIS_DIRECTORY}/"
    ):
        return "analysis"
    if relative_path.startswith("derived/"):
        return "derived-index"
    if "/journal/" in relative_path:
        return "phase-journal"
    if relative_path.startswith("source/"):
        return "attempt"
    raise AnytimeRehearsalError(f"unexpected rehearsal artifact path: {relative_path}")


def _inventory(root: Path) -> tuple[AnytimeOfflineRehearsalFile, ...]:
    files: list[AnytimeOfflineRehearsalFile] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AnytimeRehearsalError("symbolic links are forbidden in rehearsal bundles")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == REHEARSAL_MANIFEST_FILENAME:
            continue
        size = path.stat().st_size
        if size < 1:
            raise AnytimeRehearsalError(f"empty rehearsal artifact: {relative}")
        files.append(
            AnytimeOfflineRehearsalFile(
                relative_path=relative,
                role=_inventory_role(relative),
                raw_sha256=_file_sha256(path),
                size_bytes=size,
            )
        )
    return tuple(sorted(files, key=lambda item: item.relative_path))


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise AnytimeRehearsalError("symbolic links are forbidden in rehearsal bundles")
        path.chmod(0o400 if path.is_file() else 0o500)
    root.chmod(0o500)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_anytime_offline_rehearsal(
    output_directory: str | Path,
    *,
    pinned_freeze: PinnedAnytimeOfflineFreeze,
    repository_root: str | Path,
) -> AnytimeOfflineRehearsalResult:
    """Materialize and verify the full scripted shakeout population offline."""

    _assert_offline_rehearsal_source()
    repository = Path(repository_root).expanduser().resolve(strict=True)
    freeze = verify_anytime_offline_freeze(
        pinned_freeze,
        repository_root=repository,
    )
    final = Path(output_directory).expanduser()
    staging = final.with_name(f"{final.name}.incomplete")
    if final.exists() or final.is_symlink() or staging.exists() or staging.is_symlink():
        raise AnytimeRehearsalError("rehearsal output or staging directory already exists")
    final.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging.mkdir(mode=0o700)

    input_path = repository / freeze.workload_inputs.relative_path
    inputs = validate_anytime_workload_inputs(
        load_anytime_workload_inputs(
            input_path,
            expected_sha256=PINNED_INPUT_MANIFEST_SHA256,
        ),
        repository_root=repository,
    )
    if (
        inputs.raw_sha256 != freeze.workload_inputs.raw_sha256
        or inputs.manifest_sha256 != freeze.workload_inputs.canonical_manifest_sha256
    ):
        raise AnytimeRehearsalError("rehearsal workload inputs differ from the freeze")

    qualification_fixtures: dict[QualificationKey, AnytimeOfflineQualificationFixture] = {}
    qualification_raw_hashes: dict[QualificationKey, str] = {}
    qualification_bindings: list[AnytimeRehearsalArtifactBinding] = []

    floor_bundle = AnytimeFloorEvidenceBundle(
        input_manifest_sha256=inputs.manifest_sha256
    )
    floor_validation = validate_anytime_floor_evidence(
        inputs,
        floor_bundle,
        repository_root=str(repository),
    )
    if floor_validation.status != "invalid_floor":
        raise AnytimeRehearsalError("empty M9 evidence unexpectedly constructed a valid floor")
    _, floor_validation_raw_sha256 = _write_new(
        staging,
        FLOOR_VALIDATION_FILENAME,
        floor_validation,
    )

    study = build_anytime_shakeout_study()
    schedule = build_anytime_schedule(study)
    shakeout_binding = freeze.studies[1]
    if (
        shakeout_binding.canonical_spec_sha256 != study.sha256
        or shakeout_binding.schedule_sha256 != schedule.sha256
        or shakeout_binding.planned_trajectories != len(schedule.cells)
    ):
        raise AnytimeRehearsalError("rehearsal schedule differs from the frozen shakeout")
    grouped: dict[tuple[str, str, str], list[AnytimeScheduleCell]] = defaultdict(list)
    for cell in schedule.cells:
        grouped[(cell.agent_id, cell.task_id, cell.target_id)].append(cell)

    phase_receipts: list[AnytimeOfflinePhaseReceipt] = []
    successful_attempts: dict[str, AnytimeAttemptAudit] = {}
    first_phase = True
    for agent in study.agents:
        for task_id in SHAKEOUT_WORKLOAD_IDS:
            for target_id in TARGET_IDS:
                cells = tuple(
                    sorted(
                        grouped[(agent.id, task_id, target_id)],
                        key=lambda item: item.replicate,
                    )
                )
                if len(cells) != 2:
                    raise AnytimeRehearsalError("shakeout group lacks two replicates")
                phase, attempts = _run_phase(
                    staging,
                    cells=(cells[0], cells[1]),
                    inject_crash=first_phase,
                    freeze=freeze,
                    inputs=inputs,
                    repository_root=repository,
                    schedule_sha256=schedule.sha256,
                    qualification_fixtures=qualification_fixtures,
                    qualification_raw_hashes=qualification_raw_hashes,
                    qualification_bindings=qualification_bindings,
                )
                first_phase = False
                phase_receipts.append(phase)
                overlap = set(successful_attempts).intersection(attempts)
                if overlap:
                    raise AnytimeRehearsalError("duplicate successful trajectory audit")
                successful_attempts.update(attempts)

    dataset = _analysis_dataset(
        inputs=inputs,
        successful_attempts=successful_attempts,
        floor_validation_raw_sha256=floor_validation_raw_sha256,
    )
    _, dataset_raw_sha256 = _write_new(staging, ANALYSIS_DATASET_FILENAME, dataset)
    analysis_bundle = write_anytime_analysis_bundle(
        dataset,
        staging / ANALYSIS_DIRECTORY,
        include_exploratory=True,
    )
    verify_anytime_analysis_bundle(staging / ANALYSIS_DIRECTORY)
    figure_manifest = _read_model(
        staging / ANALYSIS_DIRECTORY / "figure-manifest.json",
        AnytimeFigureManifest,
    )

    issues = formal_readiness_issues(inputs.manifest)
    terminal_counts = Counter()
    protocol_counts = Counter()
    candidate_counts = Counter()
    worker_count = 0
    checkpoint_count = 0
    for phase in phase_receipts:
        terminal_counts["success"] += 2
        if phase.attempt_count == 3:
            terminal_counts["infrastructure_failure"] += 1
    for attempt in successful_attempts.values():
        directory = Path(attempt.directory)
        checkpoint_count += len(tuple(directory.glob("checkpoints/*.json")))
        for response_path in sorted(directory.glob("turns/*/provider-native-response.json")):
            response = _read_model(response_path, NativeNormalizedResponse)
            protocol_counts[response.protocol] += 1
        for worker_path in sorted(directory.glob("turns/*/worker-result.json")):
            worker = _read_model(worker_path, AnytimeWorkerEvaluationArtifact)
            worker_count += 1
            candidate_counts[worker.candidate.kind] += 1

    receipt = AnytimeOfflineRehearsalReceipt(
        freeze_raw_sha256=pinned_freeze.raw_sha256,
        freeze_manifest_sha256=freeze.sha256,
        shakeout_study_sha256=study.sha256,
        shakeout_schedule_sha256=schedule.sha256,
        planned_trajectories=len(schedule.cells),
        scientific_model_call_ceiling=schedule.scientific_request_ceiling,
        operational_provider_request_ceiling=schedule.operational_request_ceiling,
        completed_trajectories=len(successful_attempts),
        phase_count=len(phase_receipts),
        attempt_count=sum(phase.attempt_count for phase in phase_receipts),
        infrastructure_retry_count=sum(
            phase.attempt_count - 2 for phase in phase_receipts
        ),
        scripted_provider_response_count=sum(protocol_counts.values()),
        scripted_worker_artifact_count=worker_count,
        checkpoint_count=checkpoint_count,
        provider_protocol_counts=tuple(sorted(protocol_counts.items())),
        candidate_outcome_counts=tuple(sorted(candidate_counts.items())),
        terminal_counts=tuple(sorted(terminal_counts.items())),
        phase_receipts=tuple(phase_receipts),
        phase_receipt_bundle_sha256=_model_sequence_sha256(tuple(phase_receipts)),
        qualifications=tuple(
            sorted(qualification_bindings, key=lambda item: item.relative_path)
        ),
        qualification_bundle_sha256=_model_sequence_sha256(
            tuple(sorted(qualification_bindings, key=lambda item: item.relative_path))
        ),
        pending_m9_qualification_count=sum(
            fixture.decision.status == "pending-m9"
            for fixture in qualification_fixtures.values()
        ),
        floor_validation_raw_sha256=floor_validation_raw_sha256,
        floor_validation_sha256=sha256_json(floor_validation),
        floor_gate=floor_validation.status,
        formal_readiness="blocked",
        formal_readiness_issue_count=len(issues),
        formal_readiness_issues_sha256=sha256_json(issues),
        analysis_dataset_raw_sha256=dataset_raw_sha256,
        analysis_dataset_sha256=dataset.sha256,
        analysis_bundle_manifest_sha256=analysis_bundle.sha256,
        analysis_trajectory_count=len(dataset.trajectories),
        analysis_floor_count=len(dataset.floors),
        analysis_figure_count=len(figure_manifest.figures),
        environment_status=inputs.manifest.environment.observation_status,
        m9_blockers=freeze.m9_blockers,
        side_effects=AnytimeOfflineSideEffectBoundary(),
    )
    _, receipt_raw_sha256 = _write_new(staging, REHEARSAL_RECEIPT_FILENAME, receipt)
    files = _inventory(staging)
    manifest = AnytimeOfflineRehearsalManifest(
        receipt_raw_sha256=receipt_raw_sha256,
        receipt_sha256=receipt.sha256,
        files=files,
        file_bundle_sha256=_model_sequence_sha256(files),
    )
    _write_new(staging, REHEARSAL_MANIFEST_FILENAME, manifest)
    verify_anytime_offline_rehearsal(
        staging,
        pinned_freeze=pinned_freeze,
        repository_root=repository,
    )
    _make_read_only(staging)
    os.rename(staging, final)
    _fsync_directory(final.parent)
    return AnytimeOfflineRehearsalResult(
        directory=final.resolve(),
        receipt=receipt,
        manifest=manifest,
    )


def _verify_qualification_fixture(
    fixture: AnytimeOfflineQualificationFixture,
    *,
    inputs: PinnedAnytimeWorkloadInputs,
    cell: AnytimeScheduleCell,
    identity: AnytimeAttemptIdentity,
    prepared: Any,
    provider: AnytimePersistedProviderObservation,
) -> None:
    expected_source = _target_source(cell.target_id, prepared.scientific_call_index)
    if fixture.invocation.source.text != expected_source:
        raise AnytimeRehearsalError("qualification fixture source differs from handwritten input")
    expected = _qualification_fixture(
        inputs=inputs,
        cell=cell,
        identity=identity,
        scientific_call_index=prepared.scientific_call_index,
        logical_request_sha256=sha256_json(prepared.logical_request),
        provider_observation_sha256=provider.sha256,
        source=expected_source,
    )
    if expected != fixture:
        raise AnytimeRehearsalError(
            "qualification fixture is not bound to the exact trajectory, request, and provider"
        )


def _verify_phase(
    root: Path,
    *,
    phase_receipt: AnytimeOfflinePhaseReceipt,
    freeze: AnytimeOfflineFreezeManifest,
    inputs: PinnedAnytimeWorkloadInputs,
    repository_root: Path,
    schedule_sha256: str,
    cells_by_id: dict[str, AnytimeScheduleCell],
    qualification_fixtures: dict[QualificationKey, AnytimeOfflineQualificationFixture],
    qualification_hashes: dict[QualificationKey, str],
) -> tuple[AnytimePhaseAudit, dict[str, AnytimeAttemptAudit], set[QualificationKey]]:
    study = build_anytime_shakeout_study()
    cells = tuple(cells_by_id[item] for item in phase_receipt.expected_trajectory_ids)
    first = cells[0]
    if any(
        (cell.agent_id, cell.task_id, cell.target_id)
        != (phase_receipt.agent_id, phase_receipt.workload_id, phase_receipt.target_id)
        for cell in cells
    ):
        raise AnytimeRehearsalError("phase receipt trajectories belong to another prompt group")
    loop = study.cohort(first.cohort_id).loop
    base_prompt = _phase_prompt(
        inputs,
        task_id=phase_receipt.workload_id,
        target_id=phase_receipt.target_id,
        repository_root=repository_root,
        freeze=freeze,
    )
    header_path = (
        root
        / "source"
        / study.study_id
        / phase_receipt.phase_id
        / "journal"
        / "header.json"
    )
    header = _read_model(header_path, AnytimePhaseJournalHeader)
    expected_phase_sha256 = _phase_execution_sha256(
        freeze=freeze,
        schedule_sha256=schedule_sha256,
        agent_id=phase_receipt.agent_id,
        workload_id=phase_receipt.workload_id,
        target_id=phase_receipt.target_id,
        loop=loop,
        base_prompt=base_prompt,
    )
    if (
        header.sha256 != phase_receipt.phase_header_sha256
        or header.phase_execution_sha256 != expected_phase_sha256
        or header.expected_trajectory_ids != phase_receipt.expected_trajectory_ids
    ):
        raise AnytimeRehearsalError("phase journal header differs from its receipt or prompt")
    audit = audit_anytime_phase(root, header, loop=loop, base_prompt=base_prompt)
    if audit.close_audit is None:
        raise AnytimeRehearsalError("verified phase lacks a close audit")
    checkpoint_count = sum(
        len(tuple(Path(item.directory).glob("checkpoints/*.json")))
        for item in audit.attempts
    )
    if (
        audit.journal_head_sha256 != phase_receipt.journal_head_sha256
        or audit.close_audit.sha256 != phase_receipt.close_audit_sha256
        or len(audit.attempts) != phase_receipt.attempt_count
        or audit.close_audit.operational_totals.scientific_calls_consumed
        != phase_receipt.scientific_calls_consumed
        or audit.close_audit.operational_totals.provider_requests_submitted
        != phase_receipt.provider_requests_submitted
        or checkpoint_count != phase_receipt.checkpoint_count
    ):
        raise AnytimeRehearsalError("phase receipt differs from its fresh source audit")

    successful: dict[str, AnytimeAttemptAudit] = {}
    used_qualification_keys: set[QualificationKey] = set()
    bundle = build_anytime_native_manifest_bundle(study.agent(phase_receipt.agent_id))
    for attempt in audit.attempts:
        if attempt.terminal.terminal_kind != "success":
            continue
        successful[attempt.identity.trajectory_id] = attempt
        cell = cells_by_id[attempt.identity.trajectory_id]
        directory = Path(attempt.directory)
        header = _read_model(directory / "header.json", AnytimeAttemptHeader)
        if header.identity != attempt.identity:
            raise AnytimeRehearsalError("attempt header differs from its audited identity")
        records: tuple[AnytimeTurnRecord, ...] = ()
        checkpoints: tuple[Any, ...] = ()
        response_paths = sorted(directory.glob("turns/*/provider-native-response.json"))
        worker_paths = sorted(directory.glob("turns/*/worker-result.json"))
        if len(response_paths) != 4 or len(worker_paths) != 4:
            raise AnytimeRehearsalError("successful fixture attempt lacks four scripted turns")
        for response_path, worker_path in zip(response_paths, worker_paths, strict=True):
            call = int(response_path.parent.name)
            if worker_path.parent != response_path.parent:
                raise AnytimeRehearsalError("provider and worker artifacts are not co-located")
            prepared = prepare_anytime_turn(
                header=header.ledger_header,
                loop=loop,
                base_prompt=base_prompt,
                records=records,
                checkpoints=checkpoints,
            )
            if prepared.scientific_call_index != call:
                raise AnytimeRehearsalError("turn directory does not match its prepared call")
            source = _target_source(cell.target_id, call)
            response = _read_model(response_path, NativeNormalizedResponse)
            expected_response = _fake_native_response(
                prepared=prepared,
                cell=cell,
                infrastructure_attempt_index=attempt.identity.infrastructure_attempt_index,
                bundle=bundle,
                freeze=freeze,
                source=source,
            )
            if response != expected_response:
                raise AnytimeRehearsalError(
                    "scripted provider fact differs in model, reasoning, request, or response"
                )
            persisted = _read_model(
                response_path.parent / "provider-observation.json",
                AnytimePersistedProviderObservation,
            )
            worker = _read_model(worker_path, AnytimeWorkerEvaluationArtifact)
            key = (
                attempt.identity.trajectory_id,
                attempt.identity.infrastructure_attempt_index,
                call,
            )
            fixture = qualification_fixtures.get(key)
            if fixture is None:
                raise AnytimeRehearsalError("missing execution-bound qualification fixture")
            _verify_qualification_fixture(
                fixture,
                inputs=inputs,
                cell=cell,
                identity=attempt.identity,
                prepared=prepared,
                provider=persisted,
            )
            expected_worker = _fake_worker_artifact(
                provider=persisted,
                cell=cell,
                scientific_call_index=call,
                source=source,
                qualification_raw_sha256=qualification_hashes[key],
            )
            if worker != expected_worker:
                raise AnytimeRehearsalError("scripted worker fact differs from qualification input")
            used_qualification_keys.add(key)
            record = _read_model(
                response_path.parent / "turn-record.json",
                AnytimeTurnRecord,
            )
            records = (*records, record)
            checkpoints = rebuild_anytime_checkpoints(
                header=header.ledger_header,
                loop=loop,
                base_prompt=base_prompt,
                records=records,
            )
    if set(successful) != set(phase_receipt.expected_trajectory_ids):
        raise AnytimeRehearsalError("phase does not contain two successful terminal trajectories")
    return audit, successful, used_qualification_keys


def verify_anytime_offline_rehearsal(
    directory: str | Path,
    *,
    pinned_freeze: PinnedAnytimeOfflineFreeze,
    repository_root: str | Path,
) -> AnytimeOfflineRehearsalManifest:
    """Freshly verify every byte and every cross-artifact M8 binding."""

    root = Path(directory).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise AnytimeRehearsalError("rehearsal bundle directory is missing or unsafe")
    repository = Path(repository_root).expanduser().resolve(strict=True)
    freeze = verify_anytime_offline_freeze(
        pinned_freeze,
        repository_root=repository,
    )
    manifest = _read_model(
        root / REHEARSAL_MANIFEST_FILENAME,
        AnytimeOfflineRehearsalManifest,
    )
    expected_paths = {item.relative_path for item in manifest.files} | {
        REHEARSAL_MANIFEST_FILENAME
    }
    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AnytimeRehearsalError("symbolic links are forbidden in rehearsal bundles")
        if path.is_file():
            actual_paths.add(path.relative_to(root).as_posix())
    if actual_paths != expected_paths:
        raise AnytimeRehearsalError("rehearsal file inventory differs from its manifest")
    for item in manifest.files:
        path = root.joinpath(*PurePosixPath(item.relative_path).parts)
        if path.stat().st_size != item.size_bytes or _file_sha256(path) != item.raw_sha256:
            raise AnytimeRehearsalError(
                f"rehearsal artifact checksum mismatch: {item.relative_path}"
            )

    receipt_path = root / REHEARSAL_RECEIPT_FILENAME
    receipt = _read_model(receipt_path, AnytimeOfflineRehearsalReceipt)
    if (
        _file_sha256(receipt_path) != manifest.receipt_raw_sha256
        or receipt.sha256 != manifest.receipt_sha256
        or receipt.freeze_raw_sha256 != pinned_freeze.raw_sha256
        or receipt.freeze_manifest_sha256 != freeze.sha256
        or receipt.m9_blockers != freeze.m9_blockers
    ):
        raise AnytimeRehearsalError("rehearsal receipt differs from its freeze or manifest")

    input_path = repository / freeze.workload_inputs.relative_path
    inputs = validate_anytime_workload_inputs(
        load_anytime_workload_inputs(
            input_path,
            expected_sha256=PINNED_INPUT_MANIFEST_SHA256,
        ),
        repository_root=repository,
    )
    qualifications: dict[QualificationKey, AnytimeOfflineQualificationFixture] = {}
    qualification_hashes: dict[QualificationKey, str] = {}
    observed_bindings: list[AnytimeRehearsalArtifactBinding] = []
    for binding in receipt.qualifications:
        path = root.joinpath(*PurePosixPath(binding.relative_path).parts)
        fixture = _read_model(path, AnytimeOfflineQualificationFixture)
        if (
            binding.relative_path
            != _qualification_path(
                fixture.trajectory_id,
                fixture.infrastructure_attempt_index,
                fixture.scientific_call_index,
            )
            or fixture.trajectory_id != binding.trajectory_id
            or fixture.infrastructure_attempt_index != binding.infrastructure_attempt_index
            or fixture.target_id != binding.target_id
            or fixture.scientific_call_index != binding.scientific_call_index
            or fixture.binding.execution_binding_sha256 != binding.execution_binding_sha256
            or _file_sha256(path) != binding.raw_sha256
            or fixture.sha256 != binding.canonical_sha256
        ):
            raise AnytimeRehearsalError("qualification artifact differs from its receipt binding")
        key = (
            fixture.trajectory_id,
            fixture.infrastructure_attempt_index,
            fixture.scientific_call_index,
        )
        if key in qualifications:
            raise AnytimeRehearsalError("duplicate qualification fixture")
        qualifications[key] = fixture
        qualification_hashes[key] = binding.raw_sha256
        observed_bindings.append(binding)
    if len(qualifications) != 192:
        raise AnytimeRehearsalError("qualification fixtures do not cover all execution-bound turns")
    if _model_sequence_sha256(tuple(observed_bindings)) != receipt.qualification_bundle_sha256:
        raise AnytimeRehearsalError("qualification receipt bundle is not reproducible")

    floor_path = root / FLOOR_VALIDATION_FILENAME
    floor_validation = _read_model(floor_path, AnytimeFloorValidation)
    expected_floor = validate_anytime_floor_evidence(
        inputs,
        AnytimeFloorEvidenceBundle(input_manifest_sha256=inputs.manifest_sha256),
        repository_root=str(repository),
    )
    if (
        floor_validation != expected_floor
        or floor_validation.status != "invalid_floor"
        or _file_sha256(floor_path) != receipt.floor_validation_raw_sha256
        or sha256_json(floor_validation) != receipt.floor_validation_sha256
    ):
        raise AnytimeRehearsalError("invalid-floor result is not reproducible")
    issues = formal_readiness_issues(inputs.manifest)
    if (
        len(issues) != receipt.formal_readiness_issue_count
        or sha256_json(issues) != receipt.formal_readiness_issues_sha256
    ):
        raise AnytimeRehearsalError("formal readiness blockers differ from M6 inputs")

    study = build_anytime_shakeout_study()
    schedule = build_anytime_schedule(study)
    if (
        receipt.shakeout_study_sha256 != study.sha256
        or receipt.shakeout_schedule_sha256 != schedule.sha256
        or receipt.planned_trajectories != schedule.expected_trajectories
    ):
        raise AnytimeRehearsalError("receipt is bound to another shakeout schedule")
    cells_by_id = {cell.trajectory_id: cell for cell in schedule.cells}
    expected_phase_keys = tuple(
        (agent.id, task_id, target_id)
        for agent in study.agents
        for task_id in SHAKEOUT_WORKLOAD_IDS
        for target_id in TARGET_IDS
    )
    observed_phase_keys = tuple(
        (phase.agent_id, phase.workload_id, phase.target_id)
        for phase in receipt.phase_receipts
    )
    if observed_phase_keys != expected_phase_keys:
        raise AnytimeRehearsalError("phase receipts are not the canonical full shakeout grouping")
    if _model_sequence_sha256(receipt.phase_receipts) != receipt.phase_receipt_bundle_sha256:
        raise AnytimeRehearsalError("phase receipt bundle is not reproducible")

    successful: dict[str, AnytimeAttemptAudit] = {}
    all_attempts: list[AnytimeAttemptAudit] = []
    used_qualification_keys: set[QualificationKey] = set()
    for phase in receipt.phase_receipts:
        expected_phase_id = _phase_id(phase.agent_id, phase.workload_id, phase.target_id)
        if phase.phase_id != expected_phase_id:
            raise AnytimeRehearsalError("phase ID differs from its canonical prompt group")
        audit, phase_successful, phase_qualification_keys = _verify_phase(
            root,
            phase_receipt=phase,
            freeze=freeze,
            inputs=inputs,
            repository_root=repository,
            schedule_sha256=schedule.sha256,
            cells_by_id=cells_by_id,
            qualification_fixtures=qualifications,
            qualification_hashes=qualification_hashes,
        )
        all_attempts.extend(audit.attempts)
        used_qualification_keys.update(phase_qualification_keys)
        if set(successful).intersection(phase_successful):
            raise AnytimeRehearsalError("successful trajectory appears in multiple phases")
        successful.update(phase_successful)

    expected_qualification_keys = {
        (attempt.identity.trajectory_id, attempt.identity.infrastructure_attempt_index, call)
        for attempt in successful.values()
        for call in range(1, 5)
    }
    if (
        set(qualifications) != expected_qualification_keys
        or used_qualification_keys != expected_qualification_keys
    ):
        raise AnytimeRehearsalError(
            "qualification fixtures do not cover exactly the successful turns"
        )

    terminal_counts = Counter(item.terminal.terminal_kind for item in all_attempts)
    protocol_counts = Counter()
    candidate_counts = Counter()
    checkpoint_count = 0
    worker_count = 0
    for attempt in successful.values():
        attempt_dir = Path(attempt.directory)
        checkpoint_count += len(tuple(attempt_dir.glob("checkpoints/*.json")))
        for path in attempt_dir.glob("turns/*/provider-native-response.json"):
            protocol_counts[_read_model(path, NativeNormalizedResponse).protocol] += 1
        for path in attempt_dir.glob("turns/*/worker-result.json"):
            worker = _read_model(path, AnytimeWorkerEvaluationArtifact)
            worker_count += 1
            candidate_counts[worker.candidate.kind] += 1
    if (
        len(successful) != receipt.completed_trajectories
        or len(all_attempts) != receipt.attempt_count
        or tuple(sorted(terminal_counts.items())) != receipt.terminal_counts
        or tuple(sorted(protocol_counts.items())) != receipt.provider_protocol_counts
        or tuple(sorted(candidate_counts.items())) != receipt.candidate_outcome_counts
        or sum(protocol_counts.values()) != receipt.scripted_provider_response_count
        or worker_count != receipt.scripted_worker_artifact_count
        or checkpoint_count != receipt.checkpoint_count
    ):
        raise AnytimeRehearsalError("rehearsal counts differ from fresh attempt audits")

    dataset_path = root / ANALYSIS_DATASET_FILENAME
    dataset = _read_model(dataset_path, AnytimeAnalysisDataset)
    if (
        _file_sha256(dataset_path) != receipt.analysis_dataset_raw_sha256
        or dataset.sha256 != receipt.analysis_dataset_sha256
        or len(dataset.trajectories) != receipt.analysis_trajectory_count
        or len(dataset.floors) != receipt.analysis_floor_count
    ):
        raise AnytimeRehearsalError("analysis dataset differs from its receipt")
    expected_dataset = _analysis_dataset(
        inputs=inputs,
        successful_attempts=successful,
        floor_validation_raw_sha256=receipt.floor_validation_raw_sha256,
    )
    if dataset != expected_dataset:
        raise AnytimeRehearsalError("analysis dataset is not derived from verified attempts")
    build_anytime_analysis(dataset)
    analysis_bundle: AnytimeAnalysisBundleManifest = verify_anytime_analysis_bundle(
        root / ANALYSIS_DIRECTORY
    )
    figure_manifest = _read_model(
        root / ANALYSIS_DIRECTORY / "figure-manifest.json",
        AnytimeFigureManifest,
    )
    if (
        analysis_bundle.sha256 != receipt.analysis_bundle_manifest_sha256
        or analysis_bundle.input_dataset_sha256 != dataset.sha256
        or len(figure_manifest.figures) != receipt.analysis_figure_count
    ):
        raise AnytimeRehearsalError("analysis bundle differs from verified fixture inputs")
    return manifest
