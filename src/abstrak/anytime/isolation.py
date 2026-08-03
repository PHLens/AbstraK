"""Offline-only process-isolation and IPC contracts for anytime candidates.

The models in this module describe the boundary that a later trusted worker must
enforce.  They never create a process, open a file, load a runtime, or transfer a
tensor.  In particular, a valid offline contract is *not* evidence that OS-level
containment has been observed; that gate remains explicitly pending M9.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from abstrak.anytime.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    AnytimeModel,
)
from abstrak.providers.contracts import sha256_json

AnytimeTargetBackend = Literal["triton", "tilelang", "cute"]

_PUBLIC_SYMBOL_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
_CHANNEL_PATTERN = r"^[a-f0-9]{32,128}$"
_PRIVATE_IDENTIFIER_MARKERS = (
    "credential",
    "expert",
    "oracle",
    "private",
    "provider-key",
    "reference-source",
    "sealed",
)


class AnytimeIsolationError(ValueError):
    """Raised when an untrusted IPC value violates the frozen boundary."""


def _source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _public_identifier(value: str, label: str) -> str:
    lowered = value.lower()
    if any(marker in lowered for marker in _PRIVATE_IDENTIFIER_MARKERS):
        raise ValueError(f"{label} cannot name a private benchmark capability")
    if "/" in value or "\\" in value or ":" in value or value.startswith("."):
        raise ValueError(f"{label} must be an opaque public identifier, not a path")
    return value


class AnytimeCandidateSource(AnytimeModel):
    """Exact UTF-8 source payload with a raw-byte digest."""

    schema_version: Literal["abstrak-anytime-candidate-source.v1"] = (
        "abstrak-anytime-candidate-source.v1"
    )
    text: str = Field(min_length=1, max_length=1048576)
    source_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def digest_matches_exact_text(self) -> AnytimeCandidateSource:
        if self.source_sha256 != _source_sha256(self.text):
            raise ValueError("candidate source digest does not match exact UTF-8 text")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def build_anytime_candidate_source(source: str) -> AnytimeCandidateSource:
    """Bind candidate source without normalizing whitespace or newlines."""

    return AnytimeCandidateSource(text=source, source_sha256=_source_sha256(source))


class AnytimeTensorDescriptor(AnytimeModel):
    """Public tensor metadata; storage is transferred out of band by the controller."""

    schema_version: Literal["abstrak-anytime-tensor-descriptor.v1"] = (
        "abstrak-anytime-tensor-descriptor.v1"
    )
    name: str = Field(pattern=_PUBLIC_SYMBOL_PATTERN)
    shape: tuple[int, ...] = Field(min_length=1, max_length=16)
    strides: tuple[int, ...] = Field(min_length=1, max_length=16)
    dtype: Literal[
        "float16",
        "bfloat16",
        "float32",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
    ]
    device_kind: Literal["cuda"] = "cuda"
    requires_grad: Literal[False] = False

    @field_validator("shape")
    @classmethod
    def dimensions_are_bounded(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 0 or value > 2**31 - 1 for value in values):
            raise ValueError("tensor dimensions must be bounded non-negative integers")
        return values

    @field_validator("strides")
    @classmethod
    def strides_are_bounded(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 0 or value > 2**63 - 1 for value in values):
            raise ValueError("tensor strides must be bounded non-negative integers")
        return values

    @model_validator(mode="after")
    def rank_matches_stride_count(self) -> AnytimeTensorDescriptor:
        if len(self.shape) != len(self.strides):
            raise ValueError("tensor shape and stride rank must match")
        return self


class AnytimePublicTaskABI(AnytimeModel):
    """The complete task information visible to an untrusted candidate process."""

    schema_version: Literal["abstrak-anytime-public-task-abi.v1"] = (
        "abstrak-anytime-public-task-abi.v1"
    )
    abi_id: str = Field(pattern=IDENTIFIER_PATTERN)
    abi_version: str = Field(pattern=IDENTIFIER_PATTERN)
    entrypoint: str = Field(pattern=_PUBLIC_SYMBOL_PATTERN)
    input_names: tuple[str, ...] = Field(min_length=1, max_length=32)
    output_count: int = Field(ge=1, le=16)

    @field_validator("abi_id", "abi_version")
    @classmethod
    def identifiers_are_public(cls, value: str) -> str:
        return _public_identifier(value, "task ABI identifier")

    @field_validator("input_names")
    @classmethod
    def inputs_are_public_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        symbol = re.compile(_PUBLIC_SYMBOL_PATTERN)
        if len(values) != len(set(values)):
            raise ValueError("public ABI input names must be unique")
        if any(symbol.fullmatch(value) is None for value in values):
            raise ValueError("public ABI input names must be Python symbols")
        contains_private_marker = any(
            any(marker in value.lower() for marker in _PRIVATE_IDENTIFIER_MARKERS)
            for value in values
        )
        if contains_private_marker:
            raise ValueError("public ABI input names cannot name private assets")
        return values


class AnytimePublicRuntime(AnytimeModel):
    """Public runtime identity without an executable path or repository location."""

    schema_version: Literal["abstrak-anytime-public-runtime.v1"] = (
        "abstrak-anytime-public-runtime.v1"
    )
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    backend: AnytimeTargetBackend
    runtime_id: str = Field(pattern=IDENTIFIER_PATTERN)
    runtime_abi_version: str = Field(pattern=IDENTIFIER_PATTERN)
    accelerator: Literal["a100"] = "a100"

    @field_validator("target_id", "runtime_id", "runtime_abi_version")
    @classmethod
    def identifiers_are_public(cls, value: str) -> str:
        return _public_identifier(value, "runtime identifier")


class AnytimeOutputChannel(AnytimeModel):
    """Opaque controller-owned output channel, never a filesystem path."""

    schema_version: Literal["abstrak-anytime-output-channel.v1"] = (
        "abstrak-anytime-output-channel.v1"
    )
    channel_id: str = Field(pattern=_CHANNEL_PATTERN)
    transport: Literal["controller-owned-tensor-channel.v1"] = "controller-owned-tensor-channel.v1"
    expected_output_count: int = Field(ge=1, le=16)
    candidate_may_report_timing: Literal[False] = False


class AnytimeCandidateInvocation(AnytimeModel):
    """The entire candidate-visible request; no implicit environment is allowed."""

    schema_version: Literal["abstrak-anytime-candidate-invocation.v1"] = (
        "abstrak-anytime-candidate-invocation.v1"
    )
    source: AnytimeCandidateSource
    public_abi: AnytimePublicTaskABI
    public_runtime: AnytimePublicRuntime
    inputs: tuple[AnytimeTensorDescriptor, ...] = Field(min_length=1, max_length=32)
    output_channel: AnytimeOutputChannel

    @model_validator(mode="after")
    def descriptors_match_the_public_abi(self) -> AnytimeCandidateInvocation:
        names = tuple(item.name for item in self.inputs)
        if names != self.public_abi.input_names:
            raise ValueError("candidate input descriptors must exactly follow the public ABI")
        if self.output_channel.expected_output_count != self.public_abi.output_count:
            raise ValueError("candidate output channel differs from the public ABI")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeCandidateOutput(AnytimeModel):
    """Minimal untrusted response.  Timing and qualification claims are impossible here."""

    schema_version: Literal["abstrak-anytime-candidate-output.v1"] = (
        "abstrak-anytime-candidate-output.v1"
    )
    channel_id: str = Field(pattern=_CHANNEL_PATTERN)
    outputs: tuple[AnytimeTensorDescriptor, ...] = Field(min_length=1, max_length=16)

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeQualifierInvocation(AnytimeModel):
    """Trusted qualifier request containing opaque private-asset bindings, never paths."""

    schema_version: Literal["abstrak-anytime-qualifier-invocation.v1"] = (
        "abstrak-anytime-qualifier-invocation.v1"
    )
    candidate_invocation_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_output_sha256: str = Field(pattern=SHA256_PATTERN)
    private_asset_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    reference_source_sha256: str = Field(pattern=SHA256_PATTERN)
    sealed_case_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_binding_sha256: str = Field(pattern=SHA256_PATTERN)

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeCandidateProcessPolicy(AnytimeModel):
    """Required security posture for the untrusted candidate role."""

    schema_version: Literal["abstrak-anytime-candidate-process-policy.v1"] = (
        "abstrak-anytime-candidate-process-policy.v1"
    )
    role: Literal["candidate"] = "candidate"
    principal: Literal["untrusted-candidate"] = "untrusted-candidate"
    network_access: Literal[False] = False
    host_filesystem_access: Literal[False] = False
    repository_access: Literal[False] = False
    private_asset_access: Literal[False] = False
    provider_credential_access: Literal[False] = False
    inherited_environment: Literal[False] = False
    child_processes: Literal[False] = False
    runtime_mount: Literal["read-only-public-runtime"] = "read-only-public-runtime"
    input_storage: Literal["controller-owned-copy-on-write"] = "controller-owned-copy-on-write"
    output_transport: Literal["controller-mediated"] = "controller-mediated"
    max_wall_seconds: float = Field(gt=0, le=3600)
    max_memory_bytes: int = Field(ge=1048576, le=2**50)

    @field_validator("max_wall_seconds")
    @classmethod
    def timeout_is_finite(cls, value: float) -> float:
        return _finite(value, "candidate timeout")


class AnytimeQualifierProcessPolicy(AnytimeModel):
    """Required security posture for the trusted reference/qualifier role."""

    schema_version: Literal["abstrak-anytime-qualifier-process-policy.v1"] = (
        "abstrak-anytime-qualifier-process-policy.v1"
    )
    role: Literal["reference-qualifier"] = "reference-qualifier"
    principal: Literal["trusted-reference-qualifier"] = "trusted-reference-qualifier"
    network_access: Literal[False] = False
    provider_credential_access: Literal[False] = False
    candidate_address_space_access: Literal[False] = False
    private_assets: Literal["read-only-sealed-mount"] = "read-only-sealed-mount"
    candidate_output_transport: Literal["controller-mediated"] = "controller-mediated"


class AnytimeProcessIsolationContract(AnytimeModel):
    """Frozen two-role topology and the explicit offline/live evidence boundary."""

    schema_version: Literal["abstrak-anytime-process-isolation.v1"] = (
        "abstrak-anytime-process-isolation.v1"
    )
    topology: Literal["distinct-processes"] = "distinct-processes"
    candidate: AnytimeCandidateProcessPolicy
    qualifier: AnytimeQualifierProcessPolicy
    direct_candidate_to_qualifier_ipc: Literal[False] = False
    controller_mediates_all_ipc: Literal[True] = True
    candidate_and_qualifier_share_address_space: Literal[False] = False
    offline_contract_status: Literal["contract-ready"] = "contract-ready"
    real_os_containment_status: Literal["pending-m9"] = "pending-m9"
    real_os_containment_observed: Literal[False] = False
    candidate_execution_performed: Literal[False] = False

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def build_anytime_process_isolation_contract(
    *,
    max_wall_seconds: float,
    max_memory_bytes: int,
) -> AnytimeProcessIsolationContract:
    """Create the offline policy declaration; this does not create a sandbox."""

    return AnytimeProcessIsolationContract(
        candidate=AnytimeCandidateProcessPolicy(
            max_wall_seconds=max_wall_seconds,
            max_memory_bytes=max_memory_bytes,
        ),
        qualifier=AnytimeQualifierProcessPolicy(),
    )


def _revalidate_json(model_type: type[AnytimeModel], value: object) -> AnytimeModel:
    """Revalidate even objects forged with ``model_copy(update=...)``."""

    try:
        if isinstance(value, AnytimeModel):
            payload = value.model_dump(mode="json")
        else:
            payload = value
        encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True)
        return model_type.model_validate_json(encoded)
    except (TypeError, ValueError) as error:
        raise AnytimeIsolationError(f"invalid {model_type.__name__}: {error}") from error


def verify_anytime_candidate_invocation(value: object) -> AnytimeCandidateInvocation:
    """Validate an untrusted candidate request without executing its source."""

    validated = _revalidate_json(AnytimeCandidateInvocation, value)
    assert isinstance(validated, AnytimeCandidateInvocation)
    return validated


def verify_anytime_candidate_output(
    value: object,
    *,
    invocation: AnytimeCandidateInvocation,
) -> AnytimeCandidateOutput:
    """Validate the minimal output envelope and its controller channel binding."""

    trusted_invocation = verify_anytime_candidate_invocation(invocation)
    validated = _revalidate_json(AnytimeCandidateOutput, value)
    assert isinstance(validated, AnytimeCandidateOutput)
    if validated.channel_id != trusted_invocation.output_channel.channel_id:
        raise AnytimeIsolationError("candidate output used the wrong controller channel")
    if len(validated.outputs) != trusted_invocation.output_channel.expected_output_count:
        raise AnytimeIsolationError("candidate output count differs from the public ABI")
    return validated


def verify_anytime_isolation_contract(value: object) -> AnytimeProcessIsolationContract:
    """Revalidate the two-role policy while preserving the explicit M9 pending state."""

    validated = _revalidate_json(AnytimeProcessIsolationContract, value)
    assert isinstance(validated, AnytimeProcessIsolationContract)
    return validated
