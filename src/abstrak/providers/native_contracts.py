"""Versioned contracts for protocol-native, single-call provider access.

These contracts are deliberately separate from ``provider.v1``.  The original
Chat Completions client and its artifact hashes remain frozen while new studies
can bind the exact provider protocol and the effective reasoning semantics.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from abstrak.anytime.contracts import AnytimeAgentSpec
from abstrak.providers.contracts import (
    ErrorCategory,
    LogicalRequest,
    NormalizedUsage,
    sha256_json,
)

IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
ENV_PATTERN = r"^[A-Z_][A-Z0-9_]*$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
EXPECTED_LITELLM_VERSION = "1.92.0"
NATIVE_DEPENDENCY_EVALUATOR_VERSION = "native-dependency-evaluator.v1"

NativeProtocol = Literal["chat_completions", "responses"]
EffectiveReasoningMode = Literal["literal_xhigh", "thinking_enabled", "unknown"]
ReasoningFidelity = Literal["literal", "collapsed", "unknown"]
NativeStudyReadiness = Literal[
    "dependency_blocked",
    "pending_endpoint_conformance",
]

REQUIRED_NATIVE_CONFORMANCE_CHECKS: dict[NativeProtocol, tuple[str, ...]] = {
    "chat_completions": (
        "litellm_version",
        "native_chat_entrypoint",
        "generation_parameters_preserved",
        "literal_xhigh_preserved",
    ),
    "responses": (
        "litellm_version",
        "native_responses_entrypoint",
        "native_responses_config",
        "generation_parameters_preserved",
        "literal_xhigh_preserved",
    ),
}


class NativeContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class NativeProviderManifest(NativeContractModel):
    """One endpoint bound to one native protocol.

    Milestone 2 intentionally supports only the two routes required by the
    anytime study.  Adding another provider/protocol pair requires a new
    dependency-conformance rule rather than an implicit LiteLLM bridge.
    """

    schema_version: Literal["abstrak-anytime-provider.v1"] = (
        "abstrak-anytime-provider.v1"
    )
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    adapter: Literal["litellm-native"] = "litellm-native"
    protocol: NativeProtocol
    litellm_provider: Literal["deepseek", "openai"]
    base_url_env: str | None = Field(default=None, pattern=ENV_PATTERN)
    api_key_env: str = Field(pattern=ENV_PATTERN)
    timeout_seconds: float = Field(default=180, gt=0, le=3600)

    @model_validator(mode="after")
    def require_native_route(self) -> NativeProviderManifest:
        expected = {
            "deepseek": "chat_completions",
            "openai": "responses",
        }[self.litellm_provider]
        if self.protocol != expected:
            raise ValueError(
                f"{self.litellm_provider} must use the {expected} native route"
            )
        return self


class NativeModelManifest(NativeContractModel):
    schema_version: Literal["abstrak-anytime-model.v1"] = "abstrak-anytime-model.v1"
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    provider: str = Field(pattern=IDENTIFIER_PATTERN)
    api_model: str = Field(min_length=1)
    protocol: NativeProtocol
    requested_reasoning_effort: Literal["xhigh"] = "xhigh"
    max_output_tokens: int = Field(ge=256, le=65536)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    model_id_policy: Literal["exact", "mutable_alias"] = "mutable_alias"
    expected_returned_model: str | None = Field(default=None, min_length=1)

    @field_validator("temperature", "top_p")
    @classmethod
    def sampling_values_are_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("sampling parameters must be finite")
        return value

    @model_validator(mode="after")
    def exact_model_requires_expected_id(self) -> NativeModelManifest:
        if self.model_id_policy == "exact" and self.expected_returned_model is None:
            raise ValueError("exact model_id_policy requires expected_returned_model")
        return self


class NativeManifestBundle(NativeContractModel):
    provider: NativeProviderManifest
    model: NativeModelManifest

    @model_validator(mode="after")
    def references_match(self) -> NativeManifestBundle:
        if self.model.provider != self.provider.id:
            raise ValueError("model provider does not match the provider manifest")
        if self.model.protocol != self.provider.protocol:
            raise ValueError("model protocol does not match the provider protocol")
        return self

    @property
    def provider_sha256(self) -> str:
        return sha256_json(self.provider)

    @property
    def model_sha256(self) -> str:
        return sha256_json(self.model)


class NativeReasoningRecord(NativeContractModel):
    """Requested and deterministically resolved provider reasoning semantics."""

    schema_version: Literal["abstrak-anytime-provider-reasoning.v1"] = (
        "abstrak-anytime-provider-reasoning.v1"
    )
    requested_effort: Literal["xhigh"] = "xhigh"
    submitted_parameter: Literal["reasoning_effort", "reasoning"]
    submitted_value: str | dict[str, str]
    effective_mode: EffectiveReasoningMode
    fidelity: ReasoningFidelity
    evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def fidelity_matches_mode(self) -> NativeReasoningRecord:
        expected: ReasoningFidelity = {
            "literal_xhigh": "literal",
            "thinking_enabled": "collapsed",
            "unknown": "unknown",
        }[self.effective_mode]
        if self.fidelity != expected:
            raise ValueError("reasoning fidelity does not match the effective mode")
        if self.submitted_parameter == "reasoning_effort":
            if self.submitted_value != "xhigh":
                raise ValueError("reasoning_effort must submit the literal xhigh string")
        elif self.submitted_value != {"effort": "xhigh"}:
            raise ValueError("reasoning must submit exactly {'effort': 'xhigh'}")
        return self


class NativeConformanceCheck(NativeContractModel):
    name: str = Field(min_length=1)
    status: Literal["pass", "fail", "warn"]
    detail: str = Field(min_length=1)


class NativeDependencyConformance(NativeContractModel):
    """Offline dependency readiness, never formal study authorization.

    A passing record only establishes that the pinned local dependency renders
    the expected native request shape.  M9 must issue a separate, endpoint-bound
    conformance receipt before a formal study runner may execute a cohort.
    """

    schema_version: Literal["abstrak-anytime-provider-conformance.v1"] = (
        "abstrak-anytime-provider-conformance.v1"
    )
    evaluator_version: Literal["native-dependency-evaluator.v1"]
    status: Literal["pass", "fail"]
    dependency_ready: bool
    study_ready: Literal[False]
    study_readiness: NativeStudyReadiness
    protocol: NativeProtocol
    provider_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    model_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    expected_litellm_version: Literal["1.92.0"]
    observed_litellm_version: str = Field(min_length=1)
    reasoning: NativeReasoningRecord
    checks: tuple[NativeConformanceCheck, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def readiness_matches_checks(self) -> NativeDependencyConformance:
        names = tuple(check.name for check in self.checks)
        expected_names = REQUIRED_NATIVE_CONFORMANCE_CHECKS[self.protocol]
        if names != expected_names:
            raise ValueError(
                "dependency checks must exactly match the ordered protocol requirements"
            )
        if len(names) != len(set(names)):
            raise ValueError("dependency checks must have unique names")
        if any(check.status == "warn" for check in self.checks):
            raise ValueError("required dependency checks cannot use warn status")

        checks_by_name = {check.name: check for check in self.checks}
        version_matches = (
            self.observed_litellm_version == self.expected_litellm_version
        )
        expected_version_status = "pass" if version_matches else "fail"
        if checks_by_name["litellm_version"].status != expected_version_status:
            raise ValueError("LiteLLM version check conflicts with observed version")
        expected_reasoning_status = (
            "pass" if self.reasoning.fidelity == "literal" else "fail"
        )
        if (
            checks_by_name["literal_xhigh_preserved"].status
            != expected_reasoning_status
        ):
            raise ValueError("literal-xhigh check conflicts with reasoning evidence")

        all_pass = all(check.status == "pass" for check in self.checks)
        if (self.status == "pass") != all_pass:
            raise ValueError("conformance status does not match its checks")
        expected_dependency_ready = (
            self.status == "pass"
            and version_matches
            and self.reasoning.fidelity == "literal"
        )
        if self.dependency_ready != expected_dependency_ready:
            raise ValueError(
                "dependency readiness requires pinned dependencies and literal reasoning"
            )
        expected_study_readiness: NativeStudyReadiness = (
            "pending_endpoint_conformance"
            if self.dependency_ready
            else "dependency_blocked"
        )
        if self.study_readiness != expected_study_readiness:
            raise ValueError("study readiness does not match dependency readiness")
        return self


class NativeClientIdentity(NativeContractModel):
    schema_version: Literal["abstrak-anytime-provider-client-identity.v1"] = (
        "abstrak-anytime-provider-client-identity.v1"
    )
    provider_id: str = Field(pattern=IDENTIFIER_PATTERN)
    model_id: str = Field(pattern=IDENTIFIER_PATTERN)
    protocol: NativeProtocol
    provider_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    model_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    requested_model: str = Field(min_length=1)
    reasoning: NativeReasoningRecord
    dependency_ready: bool
    study_ready: Literal[False]
    study_readiness: NativeStudyReadiness

    @model_validator(mode="after")
    def readiness_matches_reasoning(self) -> NativeClientIdentity:
        if self.dependency_ready and self.reasoning.fidelity != "literal":
            raise ValueError("a dependency-ready client requires literal reasoning fidelity")
        expected_study_readiness: NativeStudyReadiness = (
            "pending_endpoint_conformance"
            if self.dependency_ready
            else "dependency_blocked"
        )
        if self.study_readiness != expected_study_readiness:
            raise ValueError("study readiness does not match dependency readiness")
        return self


class NativeResolvedProviderBinding(NativeContractModel):
    """Hash-bound manifest and dependency record written before any provider call."""

    schema_version: Literal["abstrak-anytime-provider-binding.v1"] = (
        "abstrak-anytime-provider-binding.v1"
    )
    provider: NativeProviderManifest
    model: NativeModelManifest
    agent: AnytimeAgentSpec
    agent_sha256: str = Field(pattern=SHA256_PATTERN)
    provider_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    model_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    dependency_conformance: NativeDependencyConformance

    @model_validator(mode="after")
    def hashes_match_manifests(self) -> NativeResolvedProviderBinding:
        bundle = NativeManifestBundle(provider=self.provider, model=self.model)
        if self.agent_sha256 != self.agent.sha256:
            raise ValueError("Agent hash does not match the embedded M1 Agent spec")
        validate_anytime_agent_binding(self.agent, bundle)
        if self.provider_manifest_sha256 != sha256_json(self.provider):
            raise ValueError("provider manifest hash does not match the embedded manifest")
        if self.model_manifest_sha256 != sha256_json(self.model):
            raise ValueError("model manifest hash does not match the embedded manifest")
        if self.provider.protocol != self.dependency_conformance.protocol:
            raise ValueError("dependency conformance protocol does not match the provider")
        if (
            self.provider_manifest_sha256
            != self.dependency_conformance.provider_manifest_sha256
        ):
            raise ValueError("dependency conformance is bound to a different provider manifest")
        if self.model_manifest_sha256 != self.dependency_conformance.model_manifest_sha256:
            raise ValueError("dependency conformance is bound to a different model manifest")
        return self


class NativeNormalizedResponse(NativeContractModel):
    schema_version: Literal["abstrak-anytime-provider-response.v1"] = (
        "abstrak-anytime-provider-response.v1"
    )
    request_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    provider_request_id: str | None = None
    provider_id: str = Field(pattern=IDENTIFIER_PATTERN)
    model_id: str = Field(pattern=IDENTIFIER_PATTERN)
    protocol: NativeProtocol
    provider_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    model_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    requested_model: str = Field(min_length=1)
    returned_model: str | None = None
    system_fingerprint: str | None = None
    text: str = Field(min_length=1)
    finish_reason: str | None = None
    provider_finish_reason: str | None = None
    usage: NormalizedUsage
    resource_usage_complete: bool = Field(
        description=(
            "True exactly when input, cached-input, output, and reasoning token "
            "counts are all provider-known"
        )
    )
    reasoning: NativeReasoningRecord
    started_at_utc: datetime
    finished_at_utc: datetime
    elapsed_ms: float = Field(ge=0)
    logical_request_sha256: str = Field(pattern=SHA256_PATTERN)
    transport_request_sha256: str = Field(pattern=SHA256_PATTERN)
    transport_response_sha256: str = Field(pattern=SHA256_PATTERN)
    sanitized_transport_request: dict[str, Any]
    raw_transport_response: dict[str, Any]
    capture_fidelity: Literal["sdk_object"] = "sdk_object"
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def resource_flag_matches_usage(self) -> NativeNormalizedResponse:
        resource_fields = (
            self.usage.input_tokens,
            self.usage.cached_input_tokens,
            self.usage.output_tokens,
            self.usage.reasoning_tokens,
        )
        if self.resource_usage_complete != all(
            value is not None for value in resource_fields
        ):
            raise ValueError(
                "resource_usage_complete does not match all four resource token fields"
            )
        return self


class NativeNormalizedError(NativeContractModel):
    schema_version: Literal["abstrak-anytime-provider-error.v1"] = (
        "abstrak-anytime-provider-error.v1"
    )
    request_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    provider_id: str = Field(pattern=IDENTIFIER_PATTERN)
    model_id: str = Field(pattern=IDENTIFIER_PATTERN)
    protocol: NativeProtocol
    attempt_index: Literal[1] = 1
    category: ErrorCategory
    http_status: int | None = None
    provider_code: str | None = None
    provider_type: str
    sanitized_message: str
    retryable: bool
    request_submitted: bool
    possibly_charged: bool
    retry_after_ms: int | None = Field(default=None, ge=0)
    partial_usage: NormalizedUsage | None = None
    reasoning: NativeReasoningRecord
    started_at_utc: datetime
    failed_at_utc: datetime
    elapsed_ms: float = Field(ge=0)
    logical_request_sha256: str = Field(pattern=SHA256_PATTERN)
    sanitized_transport_request: dict[str, Any]

    @model_validator(mode="after")
    def submission_matches_charge_state(self) -> NativeNormalizedError:
        if not self.request_submitted and self.possibly_charged:
            raise ValueError("an unsubmitted request cannot be possibly charged")
        if not self.request_submitted and self.partial_usage is not None:
            raise ValueError("an unsubmitted request cannot report partial usage")
        return self


class NativeProviderCallError(RuntimeError):
    def __init__(self, record: NativeNormalizedError) -> None:
        super().__init__(
            f"{record.provider_id}/{record.model_id}: "
            f"{record.category.value}: {record.sanitized_message}"
        )
        self.record = record


class NativeDependencyReadinessError(RuntimeError):
    """Raised before transport when the offline dependency checks block a call."""


class NativeMalformedProviderResponse(ValueError):
    pass


class NativeAgentBindingError(ValueError):
    pass


def native_client_identity(
    bundle: NativeManifestBundle,
    conformance: NativeDependencyConformance,
) -> NativeClientIdentity:
    return NativeClientIdentity(
        provider_id=bundle.provider.id,
        model_id=bundle.model.id,
        protocol=bundle.provider.protocol,
        provider_manifest_sha256=bundle.provider_sha256,
        model_manifest_sha256=bundle.model_sha256,
        requested_model=bundle.model.api_model,
        reasoning=conformance.reasoning,
        dependency_ready=conformance.dependency_ready,
        study_ready=False,
        study_readiness=conformance.study_readiness,
    )


def validate_native_request(request: LogicalRequest, bundle: NativeManifestBundle) -> None:
    if request.model_ref != bundle.model.id:
        raise ValueError(
            f"request model_ref {request.model_ref!r} does not match {bundle.model.id!r}"
        )


def validate_anytime_agent_binding(
    agent: AnytimeAgentSpec,
    bundle: NativeManifestBundle,
) -> None:
    """Fail if a hashed M1 Agent spec drifts from its runtime provider manifests."""

    expected = {
        "provider_id": agent.provider_id,
        "model_ref": agent.model_ref,
        "native_protocol": agent.native_protocol,
        "max_output_tokens": agent.generation.max_output_tokens,
        "requested_reasoning_effort": (
            agent.generation.reasoning.requested_reasoning_effort
        ),
        "reasoning_conformance": agent.generation.reasoning.conformance_requirement,
        "temperature": agent.generation.temperature,
        "top_p": agent.generation.top_p,
    }
    actual = {
        "provider_id": bundle.provider.id,
        "model_ref": bundle.model.id,
        "native_protocol": bundle.provider.protocol,
        "max_output_tokens": bundle.model.max_output_tokens,
        "requested_reasoning_effort": bundle.model.requested_reasoning_effort,
        "reasoning_conformance": "literal_xhigh",
        "temperature": bundle.model.temperature,
        "top_p": bundle.model.top_p,
    }
    mismatches = tuple(name for name in expected if expected[name] != actual[name])
    if mismatches:
        raise NativeAgentBindingError(
            "anytime Agent spec differs from native provider manifests: "
            + ", ".join(mismatches)
        )
