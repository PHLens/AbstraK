"""Provider-independent request, response, usage, and error contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """Strict immutable base model for records that enter experiment artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class StringEnum(str, Enum):
    """Python 3.10-compatible subset of enum.StrEnum semantics."""

    def __str__(self) -> str:
        return self.value


class MessageRole(StringEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(ContractModel):
    role: MessageRole
    content: str = Field(min_length=1)


class CompletionClientIdentity(ContractModel):
    """Resolved provider/model identity exposed before a completion call."""

    schema_version: Literal["completion-client-identity.v1"] = "completion-client-identity.v1"
    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    provider_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_model: str = Field(min_length=1)
    model_ref: str = Field(min_length=1)
    returned_model_policy: Literal["exact", "mutable_alias"]
    expected_returned_model: str | None = Field(default=None, min_length=1)
    returned_model_required: bool

    @model_validator(mode="after")
    def exact_policy_has_expected_model(self) -> CompletionClientIdentity:
        if self.returned_model_policy == "exact" and self.expected_returned_model is None:
            raise ValueError("exact returned-model policy requires an expected model")
        return self


class LogicalRequest(ContractModel):
    schema_version: Literal["logical-request.v1"] = "logical-request.v1"
    request_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    model_ref: str = Field(min_length=1)
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    trajectory_id: str | None = None
    turn_index: int | None = Field(default=None, ge=0)
    local_trajectory_seed: int | None = None


class NormalizedUsage(ContractModel):
    """Provider usage with unknown quantities represented by None, never zero."""

    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    input_characters: int = Field(ge=0)
    output_characters: int = Field(ge=0)
    provider_reported: bool
    core_fields_complete: bool
    raw_usage: dict[str, Any] | None = None

    @model_validator(mode="after")
    def flags_match_fields(self) -> NormalizedUsage:
        core_fields = (self.input_tokens, self.output_tokens, self.total_tokens)
        if self.core_fields_complete != all(value is not None for value in core_fields):
            raise ValueError("core_fields_complete does not match the token fields")
        if self.provider_reported != any(value is not None for value in core_fields):
            raise ValueError("provider_reported does not match the token fields")
        return self


class NormalizedCost(ContractModel):
    provider_reported_usd: float | None = Field(default=None, ge=0)
    estimated_usd: float | None = Field(default=None, ge=0)
    estimate_pricing_sha256: str | None = None
    currency: Literal["USD"] = "USD"

    @property
    def status(self) -> Literal["reported", "estimated", "both", "unavailable"]:
        if self.provider_reported_usd is not None and self.estimated_usd is not None:
            return "both"
        if self.provider_reported_usd is not None:
            return "reported"
        if self.estimated_usd is not None:
            return "estimated"
        return "unavailable"


class NormalizedResponse(ContractModel):
    schema_version: Literal["normalized-response.v1"] = "normalized-response.v1"
    request_id: str
    attempt_id: str
    provider_request_id: str | None = None
    provider_id: str
    model_id: str
    provider_manifest_sha256: str
    model_manifest_sha256: str
    requested_model: str
    returned_model: str | None = None
    system_fingerprint: str | None = None
    text: str
    finish_reason: str | None = None
    provider_finish_reason: str | None = None
    usage: NormalizedUsage
    cost: NormalizedCost = Field(default_factory=NormalizedCost)
    started_at_utc: datetime
    finished_at_utc: datetime
    elapsed_ms: float = Field(ge=0)
    logical_request_sha256: str
    transport_request_sha256: str
    transport_response_sha256: str
    capture_fidelity: Literal["sdk_object"] = "sdk_object"
    sanitized_transport_request: dict[str, Any]
    raw_transport_response: dict[str, Any]
    warnings: tuple[str, ...] = ()


class ErrorCategory(StringEnum):
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    CONTEXT_LENGTH = "context_length"
    CONTENT_FILTER = "content_filter"
    SERVER_ERROR = "server_error"
    MALFORMED_RESPONSE = "malformed_response"
    CANCELLED = "cancelled"
    UNKNOWN_PROVIDER_ERROR = "unknown_provider_error"


class NormalizedError(ContractModel):
    schema_version: Literal["normalized-error.v1"] = "normalized-error.v1"
    request_id: str
    attempt_id: str
    attempt_index: Literal[1] = 1
    provider_id: str
    model_id: str
    category: ErrorCategory
    http_status: int | None = None
    provider_code: str | None = None
    provider_type: str
    sanitized_message: str
    retryable: bool
    retry_after_ms: int | None = Field(default=None, ge=0)
    request_submitted: bool
    possibly_charged: bool
    partial_usage: NormalizedUsage | None = None
    started_at_utc: datetime
    failed_at_utc: datetime
    elapsed_ms: float = Field(ge=0)
    logical_request_sha256: str
    sanitized_transport_request: dict[str, Any]


class ProviderCallError(RuntimeError):
    """Raised after exactly one failed transport attempt."""

    def __init__(self, record: NormalizedError) -> None:
        super().__init__(
            f"{record.provider_id}/{record.model_id}: "
            f"{record.category.value}: {record.sanitized_message}"
        )
        self.record = record


class MalformedProviderResponse(ValueError):
    """Raised when an SDK response cannot satisfy the normalization contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a value deterministically without changing message contents."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class ConformanceCheck(ContractModel):
    name: str
    status: Literal["pass", "fail", "warn"]
    detail: str


class ConformanceReport(ContractModel):
    schema_version: Literal["provider-conformance.v1"] = "provider-conformance.v1"
    status: Literal["pass", "fail"]
    transport_ready: bool
    action_protocol_ready: bool
    pilot_ready: bool
    provider_id: str
    model_id: str
    provider_manifest_sha256: str
    model_manifest_sha256: str
    checks: tuple[ConformanceCheck, ...]
    response: NormalizedResponse | None = None
    error: NormalizedError | None = None

    @model_validator(mode="after")
    def require_terminal_record(self) -> ConformanceReport:
        if (self.response is None) == (self.error is None):
            raise ValueError("exactly one of response or error must be present")
        has_failed_check = any(check.status == "fail" for check in self.checks)
        if self.error is not None and self.status != "fail":
            raise ValueError("an error record requires status=fail")
        if self.error is not None and (
            self.transport_ready or self.action_protocol_ready or self.pilot_ready
        ):
            raise ValueError("an error report cannot be ready")
        if self.status == "pass" and has_failed_check:
            raise ValueError("a passing report cannot contain failed checks")
        if self.status == "pass" and not (self.transport_ready and self.action_protocol_ready):
            raise ValueError("status=pass requires transport and action protocol readiness")
        if self.status == "fail" and not has_failed_check:
            raise ValueError("a failing report requires at least one failed check")
        if self.pilot_ready and self.status != "pass":
            raise ValueError("pilot_ready requires status=pass")
        return self
