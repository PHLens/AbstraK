"""Pure hash-chained ledger and checkpoint verification for anytime studies.

This module deliberately has no provider client, worker, filesystem, or Agent-loop
dependency.  It turns already-observed provider and candidate facts into immutable
records, then replays every derived field from an externally supplied header.

Provider and candidate observations are primary inputs at this layer.  Their
referenced response/evaluator artifacts must be content-verified and projected by
the crash-safe artifact layer before an attempt is sealed; this module proves
internal replay consistency, not the truth of an unaudited caller assertion.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, TypeAdapter, field_validator, model_validator

from abstrak.anytime.context import (
    AnytimeDevFeedback,
    AnytimeFeedbackInput,
    AnytimeRenderedContext,
    AnytimeSourceSnapshot,
    bound_anytime_feedback,
    build_anytime_logical_request,
    render_anytime_context,
)
from abstrak.anytime.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    AnytimeCheckpointIdentity,
    AnytimeLoopPolicy,
    AnytimeModel,
    AnytimeResourceSnapshot,
)
from abstrak.providers.contracts import ChatMessage, ErrorCategory, LogicalRequest, sha256_json


class AnytimeLedgerError(ValueError):
    """Raised when an anytime prefix cannot be derived or verified exactly."""


def _require_finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return 0.0 if value == 0.0 else value


def _validate_diagnostics(values: tuple[str, ...]) -> tuple[str, ...]:
    if any(not value for value in values):
        raise ValueError("candidate diagnostics cannot contain empty strings")
    return values


class AnytimeTokenUsage(AnytimeModel):
    """The four token axes used by the resource ledger; unknown is ``None``."""

    schema_version: Literal["abstrak-anytime-token-usage.v1"] = "abstrak-anytime-token-usage.v1"
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def cached_tokens_are_a_subset(self) -> AnytimeTokenUsage:
        if (
            self.cached_input_tokens is not None
            and self.input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise ValueError("cached input tokens cannot exceed input tokens")
        return self

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.input_tokens,
                self.cached_input_tokens,
                self.output_tokens,
                self.reasoning_tokens,
            )
        )


class AnytimeProviderSuccess(AnytimeModel):
    """A submitted request with one persisted normalized response."""

    schema_version: Literal["abstrak-anytime-provider-success.v1"] = (
        "abstrak-anytime-provider-success.v1"
    )
    kind: Literal["success"] = "success"
    request_submitted: Literal[True] = True
    logical_request_sha256: str = Field(pattern=SHA256_PATTERN)
    response_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    usage: AnytimeTokenUsage
    provider_seconds: float = Field(ge=0)

    @field_validator("provider_seconds")
    @classmethod
    def provider_time_is_finite(cls, value: float) -> float:
        return _require_finite(value, "provider seconds")

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeProviderSubmittedError(AnytimeModel):
    """A definitely submitted request that ended in a persisted terminal error."""

    schema_version: Literal["abstrak-anytime-provider-submitted-error.v1"] = (
        "abstrak-anytime-provider-submitted-error.v1"
    )
    kind: Literal["submitted_error"] = "submitted_error"
    request_submitted: Literal[True] = True
    logical_request_sha256: str = Field(pattern=SHA256_PATTERN)
    error_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    category: ErrorCategory
    possibly_charged: bool
    partial_usage: AnytimeTokenUsage | None = None
    provider_seconds: float = Field(ge=0)

    @field_validator("provider_seconds")
    @classmethod
    def provider_time_is_finite(cls, value: float) -> float:
        return _require_finite(value, "provider seconds")

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeProviderAmbiguousSubmission(AnytimeModel):
    """A dispatch that may have reached the provider and is terminal by policy."""

    schema_version: Literal["abstrak-anytime-provider-ambiguous-submission.v1"] = (
        "abstrak-anytime-provider-ambiguous-submission.v1"
    )
    kind: Literal["ambiguous_submission"] = "ambiguous_submission"
    request_submitted: Literal[True] = True
    possibly_charged: Literal[True] = True
    logical_request_sha256: str = Field(pattern=SHA256_PATTERN)
    dispatch_intent_sha256: str = Field(pattern=SHA256_PATTERN)
    provider_seconds: float = Field(ge=0)

    @field_validator("provider_seconds")
    @classmethod
    def provider_time_is_finite(cls, value: float) -> float:
        return _require_finite(value, "provider seconds")

    @property
    def sha256(self) -> str:
        return sha256_json(self)


AnytimeProviderObservation: TypeAlias = Annotated[
    AnytimeProviderSuccess | AnytimeProviderSubmittedError | AnytimeProviderAmbiguousSubmission,
    Field(discriminator="kind"),
]

_PROVIDER_OBSERVATION_ADAPTER = TypeAdapter(AnytimeProviderObservation)


class _CandidateOutcome(AnytimeModel):
    """Shared immutable behavior for all candidate outcome variants."""

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeEligibleCandidate(_CandidateOutcome):
    schema_version: Literal["abstrak-anytime-candidate-eligible.v1"] = (
        "abstrak-anytime-candidate-eligible.v1"
    )
    kind: Literal["eligible"] = "eligible"
    source: AnytimeSourceSnapshot
    target_use_verified: Literal[True] = True
    search_latency_ms: float = Field(gt=0)
    compile_seconds: float = Field(ge=0)
    evaluation_seconds: float = Field(ge=0)
    gpu_seconds: float = Field(ge=0)
    diagnostics: tuple[str, ...] = ()

    @field_validator("search_latency_ms", "compile_seconds", "evaluation_seconds", "gpu_seconds")
    @classmethod
    def times_are_finite(cls, value: float) -> float:
        return _require_finite(value, "candidate time")

    @field_validator("diagnostics")
    @classmethod
    def diagnostics_are_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_diagnostics(values)


class AnytimeParseFailure(_CandidateOutcome):
    schema_version: Literal["abstrak-anytime-candidate-parse-failure.v1"] = (
        "abstrak-anytime-candidate-parse-failure.v1"
    )
    kind: Literal["parse_failure"] = "parse_failure"
    diagnostics: tuple[str, ...] = ()
    error: str = Field(min_length=1)

    @field_validator("diagnostics")
    @classmethod
    def diagnostics_are_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_diagnostics(values)


class AnytimeOversizeSource(_CandidateOutcome):
    schema_version: Literal["abstrak-anytime-candidate-oversize-source.v1"] = (
        "abstrak-anytime-candidate-oversize-source.v1"
    )
    kind: Literal["oversize_source"] = "oversize_source"
    rejected_source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_characters: int = Field(ge=1)
    error: str = Field(min_length=1)


class AnytimeDuplicateSource(_CandidateOutcome):
    schema_version: Literal["abstrak-anytime-candidate-duplicate-source.v1"] = (
        "abstrak-anytime-candidate-duplicate-source.v1"
    )
    kind: Literal["duplicate_source"] = "duplicate_source"
    source: AnytimeSourceSnapshot
    duplicate_of_call_index: int = Field(ge=1, le=12)
    original_assessment_sha256: str = Field(pattern=SHA256_PATTERN)


class AnytimeStaticCheckFailure(_CandidateOutcome):
    schema_version: Literal["abstrak-anytime-candidate-static-check-failure.v1"] = (
        "abstrak-anytime-candidate-static-check-failure.v1"
    )
    kind: Literal["static_check_failed"] = "static_check_failed"
    source: AnytimeSourceSnapshot
    diagnostics: tuple[str, ...] = Field(min_length=1)

    @field_validator("diagnostics")
    @classmethod
    def diagnostics_are_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_diagnostics(values)


class AnytimeCompileFailure(_CandidateOutcome):
    schema_version: Literal["abstrak-anytime-candidate-compile-failure.v1"] = (
        "abstrak-anytime-candidate-compile-failure.v1"
    )
    kind: Literal["compile_error"] = "compile_error"
    source: AnytimeSourceSnapshot
    compile_seconds: float = Field(ge=0)
    diagnostics: tuple[str, ...] = ()
    error: str = Field(min_length=1)

    @field_validator("compile_seconds")
    @classmethod
    def compile_time_is_finite(cls, value: float) -> float:
        return _require_finite(value, "compile seconds")

    @field_validator("diagnostics")
    @classmethod
    def diagnostics_are_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_diagnostics(values)


class AnytimeWrongResult(_CandidateOutcome):
    schema_version: Literal["abstrak-anytime-candidate-wrong-result.v1"] = (
        "abstrak-anytime-candidate-wrong-result.v1"
    )
    kind: Literal["wrong_result"] = "wrong_result"
    source: AnytimeSourceSnapshot
    compile_seconds: float = Field(ge=0)
    evaluation_seconds: float = Field(ge=0)
    gpu_seconds: float = Field(ge=0)
    diagnostics: tuple[str, ...] = ()
    error: str | None = None

    @field_validator("compile_seconds", "evaluation_seconds", "gpu_seconds")
    @classmethod
    def times_are_finite(cls, value: float) -> float:
        return _require_finite(value, "candidate time")

    @field_validator("diagnostics")
    @classmethod
    def diagnostics_are_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_diagnostics(values)


class AnytimeCandidateTimeout(_CandidateOutcome):
    schema_version: Literal["abstrak-anytime-candidate-timeout.v1"] = (
        "abstrak-anytime-candidate-timeout.v1"
    )
    kind: Literal["timeout"] = "timeout"
    stage: Literal["compile", "evaluation"]
    source: AnytimeSourceSnapshot
    compile_seconds: float = Field(ge=0)
    evaluation_seconds: float = Field(ge=0)
    gpu_seconds: float = Field(ge=0)
    error: str = Field(min_length=1)

    @field_validator("compile_seconds", "evaluation_seconds", "gpu_seconds")
    @classmethod
    def times_are_finite(cls, value: float) -> float:
        return _require_finite(value, "candidate time")

    @model_validator(mode="after")
    def stage_matches_resources(self) -> AnytimeCandidateTimeout:
        if self.stage == "compile" and (self.evaluation_seconds != 0.0 or self.gpu_seconds != 0.0):
            raise ValueError("compile timeout cannot contain evaluation or GPU time")
        return self


class AnytimeCandidateRuntimeError(_CandidateOutcome):
    schema_version: Literal["abstrak-anytime-candidate-runtime-error.v1"] = (
        "abstrak-anytime-candidate-runtime-error.v1"
    )
    kind: Literal["runtime_error"] = "runtime_error"
    source: AnytimeSourceSnapshot
    compile_seconds: float = Field(ge=0)
    evaluation_seconds: float = Field(ge=0)
    gpu_seconds: float = Field(ge=0)
    error: str = Field(min_length=1)

    @field_validator("compile_seconds", "evaluation_seconds", "gpu_seconds")
    @classmethod
    def times_are_finite(cls, value: float) -> float:
        return _require_finite(value, "candidate time")


class AnytimeTimingUnstable(_CandidateOutcome):
    schema_version: Literal["abstrak-anytime-candidate-timing-unstable.v1"] = (
        "abstrak-anytime-candidate-timing-unstable.v1"
    )
    kind: Literal["timing_unstable"] = "timing_unstable"
    source: AnytimeSourceSnapshot
    target_use_verified: Literal[True] = True
    observed_latency_ms: float = Field(gt=0)
    compile_seconds: float = Field(ge=0)
    evaluation_seconds: float = Field(ge=0)
    gpu_seconds: float = Field(ge=0)
    diagnostics: tuple[str, ...] = Field(min_length=1)

    @field_validator("observed_latency_ms", "compile_seconds", "evaluation_seconds", "gpu_seconds")
    @classmethod
    def times_are_finite(cls, value: float) -> float:
        return _require_finite(value, "candidate time")

    @field_validator("diagnostics")
    @classmethod
    def diagnostics_are_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_diagnostics(values)


class AnytimeIneligibleCandidate(_CandidateOutcome):
    """A compiled/correct candidate rejected by target-use qualification."""

    schema_version: Literal["abstrak-anytime-candidate-ineligible.v1"] = (
        "abstrak-anytime-candidate-ineligible.v1"
    )
    kind: Literal["ineligible"] = "ineligible"
    reason: Literal["target_use_unverified", "fallback_detected", "qualification_failed"]
    source: AnytimeSourceSnapshot
    compile_seconds: float = Field(ge=0)
    evaluation_seconds: float = Field(ge=0)
    gpu_seconds: float = Field(ge=0)
    diagnostics: tuple[str, ...] = Field(min_length=1)

    @field_validator("compile_seconds", "evaluation_seconds", "gpu_seconds")
    @classmethod
    def times_are_finite(cls, value: float) -> float:
        return _require_finite(value, "candidate time")

    @field_validator("diagnostics")
    @classmethod
    def diagnostics_are_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_diagnostics(values)


class AnytimeQualificationPending(_CandidateOutcome):
    """A parsed fixture whose compile, correctness, and target use remain unobserved."""

    schema_version: Literal["abstrak-anytime-candidate-qualification-pending.v1"] = (
        "abstrak-anytime-candidate-qualification-pending.v1"
    )
    kind: Literal["qualification_pending"] = "qualification_pending"
    reason: Literal["offline_rehearsal"] = "offline_rehearsal"
    source: AnytimeSourceSnapshot
    diagnostics: tuple[str, ...] = Field(min_length=1)

    @field_validator("diagnostics")
    @classmethod
    def diagnostics_are_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_diagnostics(values)


AnytimeCandidateOutcome: TypeAlias = Annotated[
    AnytimeEligibleCandidate
    | AnytimeParseFailure
    | AnytimeOversizeSource
    | AnytimeDuplicateSource
    | AnytimeStaticCheckFailure
    | AnytimeCompileFailure
    | AnytimeWrongResult
    | AnytimeCandidateTimeout
    | AnytimeCandidateRuntimeError
    | AnytimeTimingUnstable
    | AnytimeIneligibleCandidate
    | AnytimeQualificationPending,
    Field(discriminator="kind"),
]

_CANDIDATE_OUTCOME_ADAPTER = TypeAdapter(AnytimeCandidateOutcome)


class AnytimeLedgerHeader(AnytimeModel):
    """External trust anchor for one infrastructure attempt ledger."""

    schema_version: Literal["abstrak-anytime-ledger-header.v1"] = "abstrak-anytime-ledger-header.v1"
    trajectory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    infrastructure_attempt_index: int = Field(ge=1, le=2)
    trajectory_execution_sha256: str = Field(pattern=SHA256_PATTERN)
    agent_binding_sha256: str = Field(pattern=SHA256_PATTERN)
    task_sha256: str = Field(pattern=SHA256_PATTERN)
    target_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    loop_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    context_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    base_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    model_ref: str = Field(min_length=1)
    local_trajectory_seed: int | None = Field(default=None, ge=0)

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def build_anytime_ledger_header(
    *,
    trajectory_id: str,
    infrastructure_attempt_index: int,
    trajectory_execution_sha256: str,
    agent_binding_sha256: str,
    task_sha256: str,
    target_sha256: str,
    environment_sha256: str,
    model_ref: str,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
    local_trajectory_seed: int | None = None,
) -> AnytimeLedgerHeader:
    """Construct the externally persisted header from frozen inputs."""

    if not base_prompt:
        raise AnytimeLedgerError("base prompt cannot be empty")
    if infrastructure_attempt_index > loop.infrastructure.max_attempts_per_trajectory:
        raise AnytimeLedgerError("infrastructure attempt exceeds the loop policy")
    return AnytimeLedgerHeader(
        trajectory_id=trajectory_id,
        infrastructure_attempt_index=infrastructure_attempt_index,
        trajectory_execution_sha256=trajectory_execution_sha256,
        agent_binding_sha256=agent_binding_sha256,
        task_sha256=task_sha256,
        target_sha256=target_sha256,
        environment_sha256=environment_sha256,
        loop_policy_sha256=loop.sha256,
        context_policy_sha256=sha256_json(loop.context),
        base_prompt_sha256=sha256_json(
            [message.model_dump(mode="json") for message in base_prompt]
        ),
        model_ref=model_ref,
        local_trajectory_seed=local_trajectory_seed,
    )


class AnytimeResourceDelta(AnytimeModel):
    """Redundant per-turn resource facts, exactly derivable except wall time."""

    schema_version: Literal["abstrak-anytime-resource-delta.v1"] = (
        "abstrak-anytime-resource-delta.v1"
    )
    scientific_calls_consumed: Literal[1] = 1
    provider_requests_submitted: Literal[1] = 1
    possibly_charged_requests: Literal[0, 1]
    known_input_tokens: int = Field(ge=0)
    known_cached_input_tokens: int = Field(ge=0)
    known_output_tokens: int = Field(ge=0)
    known_reasoning_tokens: int = Field(ge=0)
    usage_complete: bool
    compile_attempts: Literal[0, 1]
    evaluation_attempts: Literal[0, 1]
    provider_seconds: float = Field(ge=0)
    compile_seconds: float = Field(ge=0)
    evaluation_seconds: float = Field(ge=0)
    gpu_seconds: float = Field(ge=0)
    wall_seconds: float = Field(ge=0)

    @field_validator(
        "provider_seconds",
        "compile_seconds",
        "evaluation_seconds",
        "gpu_seconds",
        "wall_seconds",
    )
    @classmethod
    def elapsed_values_are_finite(cls, value: float) -> float:
        return _require_finite(value, "resource delta time")

    @model_validator(mode="after")
    def counts_and_times_are_coherent(self) -> AnytimeResourceDelta:
        if self.known_cached_input_tokens > self.known_input_tokens:
            raise ValueError("cached input token delta cannot exceed input token delta")
        if self.compile_attempts == 0 and self.compile_seconds != 0.0:
            raise ValueError("compile time requires a compile attempt")
        if self.evaluation_attempts == 0 and (
            self.evaluation_seconds != 0.0 or self.gpu_seconds != 0.0
        ):
            raise ValueError("evaluation/GPU time requires an evaluation attempt")
        if self.evaluation_attempts > self.compile_attempts:
            raise ValueError("evaluation requires a compile attempt")
        sequential_seconds = math.fsum(
            (self.provider_seconds, self.compile_seconds, self.evaluation_seconds)
        )
        if self.wall_seconds < sequential_seconds:
            raise ValueError("turn wall time cannot be below sequential observed time")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


TerminalReason: TypeAlias = Literal[
    "call_budget",
    "resource_cap",
    "provider_submitted_error",
    "ambiguous_submission",
]


class AnytimeTurnRecord(AnytimeModel):
    """One consumed scientific call in a header-to-turn hash chain."""

    schema_version: Literal["abstrak-anytime-turn-record.v1"] = "abstrak-anytime-turn-record.v1"
    header_sha256: str = Field(pattern=SHA256_PATTERN)
    trajectory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    infrastructure_attempt_index: int = Field(ge=1, le=2)
    scientific_call_index: int = Field(ge=1, le=12)
    trajectory_execution_sha256: str = Field(pattern=SHA256_PATTERN)
    previous_ledger_sha256: str = Field(pattern=SHA256_PATTERN)
    context: AnytimeRenderedContext
    logical_request: LogicalRequest
    provider: AnytimeProviderObservation
    candidate: AnytimeCandidateOutcome | None = None
    feedback: AnytimeDevFeedback | None = None
    incumbent_before: AnytimeSourceSnapshot | None = None
    incumbent_after: AnytimeSourceSnapshot | None = None
    observed_wall_seconds: float = Field(ge=0)
    resource_delta: AnytimeResourceDelta
    resource_snapshot: AnytimeResourceSnapshot
    terminal_reason: TerminalReason | None = None

    @field_validator("observed_wall_seconds")
    @classmethod
    def wall_time_is_finite(cls, value: float) -> float:
        return _require_finite(value, "observed wall seconds")

    @model_validator(mode="after")
    def local_identities_are_coherent(self) -> AnytimeTurnRecord:
        if self.context.scientific_call_index != self.scientific_call_index:
            raise ValueError("context call index does not match turn call index")
        if self.logical_request.turn_index != self.scientific_call_index - 1:
            raise ValueError("logical request turn index is not zero-based call index")
        if self.logical_request.trajectory_id != self.trajectory_id:
            raise ValueError("logical request trajectory does not match turn")
        if sha256_json(self.logical_request) != self.provider.logical_request_sha256:
            raise ValueError("provider observation does not bind the logical request")
        expected_incumbent_before = (
            None if self.incumbent_before is None else self.incumbent_before.source_sha256
        )
        if self.context.incumbent_candidate_sha256 != expected_incumbent_before:
            raise ValueError("context incumbent hash does not match entering incumbent")
        if self.resource_snapshot.scientific_calls_consumed != self.scientific_call_index:
            raise ValueError("resource snapshot call count does not match turn call index")
        if isinstance(self.provider, AnytimeProviderSuccess):
            if self.candidate is None or self.feedback is None:
                raise ValueError("provider success requires candidate outcome and feedback")
            if self.terminal_reason in {
                "provider_submitted_error",
                "ambiguous_submission",
            }:
                raise ValueError("provider success has an incompatible terminal reason")
        else:
            if self.candidate is not None or self.feedback is not None:
                raise ValueError("provider terminal outcome cannot contain candidate facts")
            expected_reason = (
                "provider_submitted_error"
                if isinstance(self.provider, AnytimeProviderSubmittedError)
                else "ambiguous_submission"
            )
            if self.terminal_reason != expected_reason:
                raise ValueError("provider terminal outcome has the wrong terminal reason")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeCheckpointRecord(AnytimeModel):
    """A fully derived checkpoint over one verified ledger prefix."""

    schema_version: Literal["abstrak-anytime-checkpoint-record.v1"] = (
        "abstrak-anytime-checkpoint-record.v1"
    )
    header_sha256: str = Field(pattern=SHA256_PATTERN)
    identity: AnytimeCheckpointIdentity
    incumbent: AnytimeSourceSnapshot | None = None
    resource_snapshot: AnytimeResourceSnapshot

    @model_validator(mode="after")
    def identity_matches_payload(self) -> AnytimeCheckpointRecord:
        if self.identity.scientific_call_index != self.resource_snapshot.scientific_calls_consumed:
            raise ValueError("checkpoint call does not match resource snapshot")
        incumbent_sha256 = None if self.incumbent is None else self.incumbent.source_sha256
        if self.identity.incumbent_candidate_sha256 != incumbent_sha256:
            raise ValueError("checkpoint incumbent does not match checkpoint identity")
        if self.identity.resource_snapshot_sha256 != self.resource_snapshot.sha256:
            raise ValueError("checkpoint resource snapshot hash does not match payload")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimePreparedTurn(AnytimeModel):
    """Purely reconstructed request boundary ready for an external caller."""

    schema_version: Literal["abstrak-anytime-prepared-turn.v1"] = "abstrak-anytime-prepared-turn.v1"
    header_sha256: str = Field(pattern=SHA256_PATTERN)
    scientific_call_index: int = Field(ge=1, le=12)
    previous_ledger_sha256: str = Field(pattern=SHA256_PATTERN)
    context: AnytimeRenderedContext
    logical_request: LogicalRequest

    @model_validator(mode="after")
    def request_matches_context(self) -> AnytimePreparedTurn:
        if self.context.scientific_call_index != self.scientific_call_index:
            raise ValueError("prepared context call does not match prepared turn")
        if self.logical_request.messages != self.context.messages:
            raise ValueError("prepared logical request does not contain rendered context")
        if self.logical_request.turn_index != self.scientific_call_index - 1:
            raise ValueError("prepared logical request turn is not zero based")
        return self


class AnytimeVerifiedLedger(AnytimeModel):
    """Small replay result; construction is internal to ``verify_anytime_ledger``."""

    schema_version: Literal["abstrak-anytime-verified-ledger.v1"] = (
        "abstrak-anytime-verified-ledger.v1"
    )
    header_sha256: str = Field(pattern=SHA256_PATTERN)
    records_verified: int = Field(ge=0, le=12)
    ledger_head_sha256: str = Field(pattern=SHA256_PATTERN)
    incumbent: AnytimeSourceSnapshot | None = None
    resource_snapshot: AnytimeResourceSnapshot
    terminal_reason: TerminalReason | None = None


class AnytimeAppendResult(AnytimeModel):
    """One newly derived turn and an optional policy checkpoint."""

    schema_version: Literal["abstrak-anytime-append-result.v1"] = "abstrak-anytime-append-result.v1"
    record: AnytimeTurnRecord
    checkpoint: AnytimeCheckpointRecord | None = None


def _candidate_source(
    candidate: AnytimeCandidateOutcome | None,
) -> AnytimeSourceSnapshot | None:
    if candidate is None or isinstance(candidate, (AnytimeParseFailure, AnytimeOversizeSource)):
        return None
    return candidate.source


def _provider_usage(
    provider: AnytimeProviderObservation,
) -> AnytimeTokenUsage | None:
    if isinstance(provider, AnytimeProviderSuccess):
        return provider.usage
    if isinstance(provider, AnytimeProviderSubmittedError):
        return provider.partial_usage
    return None


def _candidate_resource_facts(
    candidate: AnytimeCandidateOutcome | None,
) -> tuple[int, int, float, float, float]:
    if candidate is None or isinstance(
        candidate,
        (
            AnytimeParseFailure,
            AnytimeOversizeSource,
            AnytimeDuplicateSource,
            AnytimeStaticCheckFailure,
            AnytimeQualificationPending,
        ),
    ):
        return 0, 0, 0.0, 0.0, 0.0
    if isinstance(candidate, AnytimeCompileFailure):
        return 1, 0, candidate.compile_seconds, 0.0, 0.0
    if isinstance(candidate, AnytimeCandidateTimeout) and candidate.stage == "compile":
        return 1, 0, candidate.compile_seconds, 0.0, 0.0
    return (
        1,
        1,
        candidate.compile_seconds,
        candidate.evaluation_seconds,
        candidate.gpu_seconds,
    )


def _feedback_input_for_candidate(
    candidate: AnytimeCandidateOutcome,
) -> AnytimeFeedbackInput:
    if isinstance(candidate, AnytimeEligibleCandidate):
        return AnytimeFeedbackInput(
            status="eligible",
            compiled=True,
            correct=True,
            median_latency_ms=candidate.search_latency_ms,
            diagnostics=candidate.diagnostics,
        )
    if isinstance(candidate, AnytimeIneligibleCandidate):
        return AnytimeFeedbackInput(
            status="ineligible",
            compiled=True,
            correct=True,
            diagnostics=candidate.diagnostics,
            error=f"candidate ineligible: {candidate.reason}",
        )
    if isinstance(candidate, AnytimeQualificationPending):
        return AnytimeFeedbackInput(
            status="qualification_pending",
            diagnostics=candidate.diagnostics,
            error="candidate execution and qualification remain pending trusted M9 evaluation",
        )
    if isinstance(candidate, AnytimeParseFailure):
        return AnytimeFeedbackInput(
            status="parse_failure",
            diagnostics=candidate.diagnostics,
            error=candidate.error,
        )
    if isinstance(candidate, AnytimeOversizeSource):
        return AnytimeFeedbackInput(
            status="oversize_source",
            error=candidate.error,
        )
    if isinstance(candidate, AnytimeDuplicateSource):
        return AnytimeFeedbackInput(
            status="duplicate_source",
            error=f"source duplicates scientific call {candidate.duplicate_of_call_index}",
        )
    if isinstance(candidate, AnytimeStaticCheckFailure):
        return AnytimeFeedbackInput(
            status="static_check_failed",
            diagnostics=candidate.diagnostics,
        )
    if isinstance(candidate, AnytimeCompileFailure):
        return AnytimeFeedbackInput(
            status="compile_error",
            compiled=False,
            diagnostics=candidate.diagnostics,
            error=candidate.error,
        )
    if isinstance(candidate, AnytimeWrongResult):
        return AnytimeFeedbackInput(
            status="wrong_result",
            compiled=True,
            correct=False,
            diagnostics=candidate.diagnostics,
            error=candidate.error,
        )
    if isinstance(candidate, AnytimeCandidateTimeout):
        return AnytimeFeedbackInput(
            status="timeout",
            compiled=candidate.stage == "evaluation",
            error=candidate.error,
        )
    if isinstance(candidate, AnytimeCandidateRuntimeError):
        return AnytimeFeedbackInput(
            status="runtime_error",
            compiled=True,
            error=candidate.error,
        )
    if isinstance(candidate, AnytimeTimingUnstable):
        return AnytimeFeedbackInput(
            status="timing_unstable",
            compiled=True,
            correct=True,
            median_latency_ms=candidate.observed_latency_ms,
            diagnostics=candidate.diagnostics,
        )
    raise AssertionError("unhandled candidate outcome")


def format_anytime_dev_feedback(
    candidate: AnytimeCandidateOutcome,
    loop: AnytimeLoopPolicy,
) -> AnytimeDevFeedback:
    """Return the sole bounded Agent-visible representation of an outcome."""

    return bound_anytime_feedback(_feedback_input_for_candidate(candidate), loop.context)


def _derive_resource_delta(
    provider: AnytimeProviderObservation,
    candidate: AnytimeCandidateOutcome | None,
    observed_wall_seconds: float,
) -> AnytimeResourceDelta:
    usage = _provider_usage(provider)
    compile_attempts, evaluation_attempts, compile_seconds, evaluation_seconds, gpu_seconds = (
        _candidate_resource_facts(candidate)
    )
    possibly_charged = (
        int(provider.possibly_charged)
        if isinstance(
            provider,
            (AnytimeProviderSubmittedError, AnytimeProviderAmbiguousSubmission),
        )
        else 0
    )
    return AnytimeResourceDelta(
        possibly_charged_requests=possibly_charged,
        known_input_tokens=(
            0 if usage is None or usage.input_tokens is None else usage.input_tokens
        ),
        known_cached_input_tokens=(
            0 if usage is None or usage.cached_input_tokens is None else usage.cached_input_tokens
        ),
        known_output_tokens=(
            0 if usage is None or usage.output_tokens is None else usage.output_tokens
        ),
        known_reasoning_tokens=(
            0 if usage is None or usage.reasoning_tokens is None else usage.reasoning_tokens
        ),
        usage_complete=usage is not None and usage.complete,
        compile_attempts=compile_attempts,
        evaluation_attempts=evaluation_attempts,
        provider_seconds=provider.provider_seconds,
        compile_seconds=compile_seconds,
        evaluation_seconds=evaluation_seconds,
        gpu_seconds=gpu_seconds,
        wall_seconds=observed_wall_seconds,
    )


def fold_anytime_resources(
    deltas: tuple[AnytimeResourceDelta, ...],
) -> AnytimeResourceSnapshot:
    """Fold a single-attempt resource prefix with exact integer sums and ``fsum``."""

    return AnytimeResourceSnapshot(
        scientific_calls_consumed=sum(delta.scientific_calls_consumed for delta in deltas),
        provider_requests_submitted=sum(delta.provider_requests_submitted for delta in deltas),
        possibly_charged_requests=sum(delta.possibly_charged_requests for delta in deltas),
        known_input_tokens=sum(delta.known_input_tokens for delta in deltas),
        known_cached_input_tokens=sum(delta.known_cached_input_tokens for delta in deltas),
        known_output_tokens=sum(delta.known_output_tokens for delta in deltas),
        known_reasoning_tokens=sum(delta.known_reasoning_tokens for delta in deltas),
        usage_complete=all(delta.usage_complete for delta in deltas),
        compile_attempts=sum(delta.compile_attempts for delta in deltas),
        evaluation_attempts=sum(delta.evaluation_attempts for delta in deltas),
        provider_seconds=math.fsum(delta.provider_seconds for delta in deltas),
        compile_seconds=math.fsum(delta.compile_seconds for delta in deltas),
        evaluation_seconds=math.fsum(delta.evaluation_seconds for delta in deltas),
        gpu_seconds=math.fsum(delta.gpu_seconds for delta in deltas),
        wall_seconds=math.fsum(delta.wall_seconds for delta in deltas),
    )


def _validate_header_anchor(
    header: AnytimeLedgerHeader,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
) -> None:
    if not base_prompt:
        raise AnytimeLedgerError("base prompt cannot be empty")
    if header.loop_policy_sha256 != loop.sha256:
        raise AnytimeLedgerError("ledger header does not match the loop policy")
    if header.context_policy_sha256 != sha256_json(loop.context):
        raise AnytimeLedgerError("ledger header does not match the context policy")
    expected_base_sha256 = sha256_json([message.model_dump(mode="json") for message in base_prompt])
    if header.base_prompt_sha256 != expected_base_sha256:
        raise AnytimeLedgerError("ledger header does not match the frozen base prompt")
    if header.infrastructure_attempt_index > loop.infrastructure.max_attempts_per_trajectory:
        raise AnytimeLedgerError("ledger attempt exceeds the infrastructure policy")


def _validate_resource_budget(
    snapshot: AnytimeResourceSnapshot,
    delta: AnytimeResourceDelta,
    loop: AnytimeLoopPolicy,
) -> None:
    budget = loop.budget
    # A just-observed request or evaluation may cross a time/token cap.  The
    # consumed fact must remain recordable so replay can make this turn
    # terminal.  These bounded counters are the only excesses that prove a
    # dispatch continued after an already-reached terminal boundary.
    checks = (
        (
            snapshot.scientific_calls_consumed <= budget.max_scientific_calls,
            "scientific-call cap",
        ),
        (snapshot.compile_attempts <= budget.max_compile_attempts, "compile cap"),
        (
            snapshot.evaluation_attempts <= budget.max_evaluation_attempts,
            "evaluation cap",
        ),
    )
    for allowed, label in checks:
        if not allowed:
            raise AnytimeLedgerError(f"resource ledger exceeds {label}")


def _resource_cap_reached(
    snapshot: AnytimeResourceSnapshot,
    delta: AnytimeResourceDelta,
    loop: AnytimeLoopPolicy,
) -> bool:
    budget = loop.budget
    return any(
        (
            snapshot.known_output_tokens >= budget.max_total_output_tokens,
            delta.known_output_tokens
            > budget.max_total_output_tokens // budget.max_scientific_calls,
            snapshot.compile_attempts >= budget.max_compile_attempts,
            snapshot.evaluation_attempts >= budget.max_evaluation_attempts,
            snapshot.gpu_seconds >= budget.max_gpu_seconds,
            delta.provider_seconds > budget.max_provider_seconds_per_call,
            math.fsum((delta.compile_seconds, delta.evaluation_seconds))
            > budget.max_candidate_seconds_per_call,
            snapshot.wall_seconds >= budget.max_trajectory_wall_seconds,
        )
    )


def _checkpoint_for_record(
    header: AnytimeLedgerHeader,
    record: AnytimeTurnRecord,
) -> AnytimeCheckpointRecord:
    incumbent_sha256 = (
        None if record.incumbent_after is None else record.incumbent_after.source_sha256
    )
    identity = AnytimeCheckpointIdentity(
        trajectory_id=header.trajectory_id,
        infrastructure_attempt_index=header.infrastructure_attempt_index,
        scientific_call_index=record.scientific_call_index,
        trajectory_execution_sha256=header.trajectory_execution_sha256,
        ledger_prefix_sha256=record.sha256,
        incumbent_candidate_sha256=incumbent_sha256,
        resource_snapshot_sha256=record.resource_snapshot.sha256,
    )
    return AnytimeCheckpointRecord(
        header_sha256=header.sha256,
        identity=identity,
        incumbent=record.incumbent_after,
        resource_snapshot=record.resource_snapshot,
    )


def _revalidate_turn(record: AnytimeTurnRecord) -> AnytimeTurnRecord:
    """Re-run validators so ``model_copy(update=...)`` never crosses the trust boundary."""

    return AnytimeTurnRecord.model_validate_json(record.model_dump_json())


def _revalidate_checkpoint(
    checkpoint: AnytimeCheckpointRecord,
) -> AnytimeCheckpointRecord:
    return AnytimeCheckpointRecord.model_validate_json(checkpoint.model_dump_json())


def _canonicalize_ledger_inputs(
    *,
    header: AnytimeLedgerHeader,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
    records: tuple[AnytimeTurnRecord, ...],
    checkpoints: tuple[AnytimeCheckpointRecord, ...],
) -> tuple[
    AnytimeLedgerHeader,
    AnytimeLoopPolicy,
    tuple[ChatMessage, ...],
    tuple[AnytimeTurnRecord, ...],
    tuple[AnytimeCheckpointRecord, ...],
]:
    """Return the only instances public ledger entrypoints may read.

    ``model_copy(update=...)`` can place unvalidated nested dictionaries inside
    a typed frozen Pydantic model.  Verifying one reconstruction and then
    reading the caller's original object creates a TOCTOU boundary, so all
    subsequent reads use this same canonical tuple.
    """

    return (
        AnytimeLedgerHeader.model_validate_json(header.model_dump_json()),
        AnytimeLoopPolicy.model_validate_json(loop.model_dump_json()),
        tuple(
            ChatMessage.model_validate_json(message.model_dump_json()) for message in base_prompt
        ),
        tuple(_revalidate_turn(record) for record in records),
        tuple(_revalidate_checkpoint(value) for value in checkpoints),
    )


def _validate_candidate_history(
    candidate: AnytimeCandidateOutcome,
    *,
    scientific_call_index: int,
    source_history: dict[str, tuple[int, AnytimeSourceSnapshot, str]],
    loop: AnytimeLoopPolicy,
) -> None:
    if isinstance(candidate, AnytimeOversizeSource):
        if candidate.source_characters <= loop.context.max_candidate_source_characters:
            raise AnytimeLedgerError("oversize outcome does not exceed the source cap")
        return
    source = _candidate_source(candidate)
    if source is None:
        return
    if len(source.source) > loop.context.max_candidate_source_characters:
        raise AnytimeLedgerError("candidate source exceeds the frozen source cap")
    earlier = source_history.get(source.source_sha256)
    if isinstance(candidate, AnytimeDuplicateSource):
        if earlier is None:
            raise AnytimeLedgerError("duplicate source has no earlier assessment")
        earlier_call, earlier_source, earlier_assessment_sha256 = earlier
        if candidate.source != earlier_source:
            raise AnytimeLedgerError("duplicate source payload differs from earlier source")
        if candidate.duplicate_of_call_index != earlier_call:
            raise AnytimeLedgerError("duplicate source does not cite its first occurrence")
        if candidate.original_assessment_sha256 != earlier_assessment_sha256:
            raise AnytimeLedgerError("duplicate source assessment hash is invalid")
        return
    if earlier is not None:
        raise AnytimeLedgerError("repeated candidate source must use duplicate outcome")
    if scientific_call_index < 1:
        raise AssertionError("scientific calls are one based")


def _expected_terminal_reason(
    *,
    provider: AnytimeProviderObservation,
    scientific_call_index: int,
    snapshot: AnytimeResourceSnapshot,
    delta: AnytimeResourceDelta,
    loop: AnytimeLoopPolicy,
) -> TerminalReason | None:
    if isinstance(provider, AnytimeProviderSubmittedError):
        return "provider_submitted_error"
    if isinstance(provider, AnytimeProviderAmbiguousSubmission):
        return "ambiguous_submission"
    if scientific_call_index == loop.budget.max_scientific_calls:
        return "call_budget"
    if _resource_cap_reached(snapshot, delta, loop):
        return "resource_cap"
    return None


def verify_anytime_ledger(
    *,
    header: AnytimeLedgerHeader,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
    records: tuple[AnytimeTurnRecord, ...] = (),
    checkpoints: tuple[AnytimeCheckpointRecord, ...] = (),
) -> AnytimeVerifiedLedger:
    """Replay a complete or partial prefix and fail on any non-derived field.

    ``header``, ``loop`` and ``base_prompt`` are caller-supplied trust anchors.  The
    verifier does not accept an in-ledger replacement for any of them.
    """

    header, loop, base_prompt, records, checkpoints = _canonicalize_ledger_inputs(
        header=header,
        loop=loop,
        base_prompt=base_prompt,
        records=records,
        checkpoints=checkpoints,
    )
    _validate_header_anchor(header, loop, base_prompt)
    if len(records) > loop.budget.max_scientific_calls:
        raise AnytimeLedgerError("ledger contains more turns than the scientific budget")

    head_sha256 = header.sha256
    incumbent: AnytimeSourceSnapshot | None = None
    incumbent_key: tuple[float, int, str] | None = None
    previous_candidate: AnytimeSourceSnapshot | None = None
    previous_feedback: AnytimeDevFeedback | None = None
    deltas: list[AnytimeResourceDelta] = []
    source_history: dict[str, tuple[int, AnytimeSourceSnapshot, str]] = {}
    expected_checkpoints: list[AnytimeCheckpointRecord] = []
    terminal_reason: TerminalReason | None = None

    for scientific_call_index, record in enumerate(records, start=1):
        if terminal_reason is not None:
            raise AnytimeLedgerError("ledger continues after a terminal turn")
        if record.header_sha256 != header.sha256:
            raise AnytimeLedgerError("turn does not bind the external ledger header")
        if record.trajectory_id != header.trajectory_id:
            raise AnytimeLedgerError("turn trajectory differs from ledger header")
        if record.infrastructure_attempt_index != header.infrastructure_attempt_index:
            raise AnytimeLedgerError("turn infrastructure attempt differs from ledger header")
        if record.trajectory_execution_sha256 != header.trajectory_execution_sha256:
            raise AnytimeLedgerError("turn execution identity differs from ledger header")
        if record.scientific_call_index != scientific_call_index:
            raise AnytimeLedgerError("scientific call indices are not contiguous and one based")
        if record.previous_ledger_sha256 != head_sha256:
            raise AnytimeLedgerError("turn breaks the header-to-turn ledger hash chain")

        expected_context = render_anytime_context(
            policy=loop.context,
            scientific_call_index=scientific_call_index,
            max_scientific_calls=loop.budget.max_scientific_calls,
            base_prompt=base_prompt,
            incumbent=incumbent,
            previous_candidate=previous_candidate,
            previous_feedback=previous_feedback,
        )
        if record.context != expected_context:
            raise AnytimeLedgerError("turn context differs from full reconstructed context")
        expected_request = build_anytime_logical_request(
            trajectory_id=header.trajectory_id,
            infrastructure_attempt_index=header.infrastructure_attempt_index,
            model_ref=header.model_ref,
            context=expected_context,
            local_trajectory_seed=header.local_trajectory_seed,
        )
        if record.logical_request != expected_request:
            raise AnytimeLedgerError("logical request differs from reconstructed request")
        if record.provider.logical_request_sha256 != sha256_json(expected_request):
            raise AnytimeLedgerError("provider artifact belongs to another request")
        if record.incumbent_before != incumbent:
            raise AnytimeLedgerError("turn entering incumbent differs from replayed incumbent")

        if isinstance(record.provider, AnytimeProviderSuccess):
            if record.candidate is None or record.feedback is None:
                raise AnytimeLedgerError("provider success lacks candidate evaluation")
            _validate_candidate_history(
                record.candidate,
                scientific_call_index=scientific_call_index,
                source_history=source_history,
                loop=loop,
            )
            expected_feedback = format_anytime_dev_feedback(record.candidate, loop)
            if record.feedback != expected_feedback:
                raise AnytimeLedgerError("bounded feedback differs from candidate outcome")
            candidate_source = _candidate_source(record.candidate)
            if candidate_source is not None and not isinstance(
                record.candidate, AnytimeDuplicateSource
            ):
                source_history[candidate_source.source_sha256] = (
                    scientific_call_index,
                    candidate_source,
                    record.candidate.sha256,
                )
            if isinstance(record.candidate, AnytimeEligibleCandidate):
                candidate_key = (
                    record.candidate.search_latency_ms,
                    scientific_call_index,
                    record.candidate.source.source_sha256,
                )
                if incumbent_key is None or candidate_key < incumbent_key:
                    incumbent_key = candidate_key
                    incumbent = record.candidate.source
            previous_candidate = candidate_source
            previous_feedback = expected_feedback
        else:
            if record.candidate is not None or record.feedback is not None:
                raise AnytimeLedgerError("provider terminal turn contains candidate facts")
            previous_candidate = None
            previous_feedback = None

        if record.incumbent_after != incumbent:
            raise AnytimeLedgerError("turn incumbent differs from deterministic eligible selection")
        expected_delta = _derive_resource_delta(
            record.provider,
            record.candidate,
            record.observed_wall_seconds,
        )
        if record.resource_delta != expected_delta:
            raise AnytimeLedgerError("turn resource delta differs from observed outcome")
        deltas.append(expected_delta)
        expected_snapshot = fold_anytime_resources(tuple(deltas))
        if record.resource_snapshot != expected_snapshot:
            raise AnytimeLedgerError("cumulative resource snapshot does not match replay")
        _validate_resource_budget(expected_snapshot, expected_delta, loop)
        expected_terminal = _expected_terminal_reason(
            provider=record.provider,
            scientific_call_index=scientific_call_index,
            snapshot=expected_snapshot,
            delta=expected_delta,
            loop=loop,
        )
        if record.terminal_reason != expected_terminal:
            raise AnytimeLedgerError("turn terminal reason differs from frozen policy")
        terminal_reason = expected_terminal
        head_sha256 = record.sha256
        if scientific_call_index in loop.checkpoints.calls:
            expected_checkpoints.append(_checkpoint_for_record(header, record))

    expected_checkpoint_calls = tuple(
        checkpoint.identity.scientific_call_index for checkpoint in expected_checkpoints
    )
    observed_checkpoint_calls = tuple(
        checkpoint.identity.scientific_call_index for checkpoint in checkpoints
    )
    if observed_checkpoint_calls != expected_checkpoint_calls:
        raise AnytimeLedgerError(
            "checkpoint set is not exactly the declared calls in the verified prefix"
        )
    for observed, expected in zip(checkpoints, expected_checkpoints, strict=True):
        if observed != expected:
            raise AnytimeLedgerError("checkpoint does not match its verified ledger prefix")

    snapshot = fold_anytime_resources(tuple(deltas))
    return AnytimeVerifiedLedger(
        header_sha256=header.sha256,
        records_verified=len(records),
        ledger_head_sha256=head_sha256,
        incumbent=incumbent,
        resource_snapshot=snapshot,
        terminal_reason=terminal_reason,
    )


def rebuild_anytime_checkpoints(
    *,
    header: AnytimeLedgerHeader,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
    records: tuple[AnytimeTurnRecord, ...],
) -> tuple[AnytimeCheckpointRecord, ...]:
    """Derive all reached checkpoints, then fully replay-verify the result.

    This is the public crash-recovery boundary for the later artifact milestone;
    checkpoints beyond the existing scientific prefix are never synthesized.
    """

    header, loop, base_prompt, revalidated_records, _ = _canonicalize_ledger_inputs(
        header=header,
        loop=loop,
        base_prompt=base_prompt,
        records=records,
        checkpoints=(),
    )
    checkpoints = tuple(
        _checkpoint_for_record(header, record)
        for record in revalidated_records
        if record.scientific_call_index in loop.checkpoints.calls
    )
    verify_anytime_ledger(
        header=header,
        loop=loop,
        base_prompt=base_prompt,
        records=revalidated_records,
        checkpoints=checkpoints,
    )
    return checkpoints


def prepare_anytime_turn(
    *,
    header: AnytimeLedgerHeader,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
    records: tuple[AnytimeTurnRecord, ...] = (),
    checkpoints: tuple[AnytimeCheckpointRecord, ...] = (),
) -> AnytimePreparedTurn:
    """Reconstruct the next request from a verified prefix, without submitting it."""

    header, loop, base_prompt, records, checkpoints = _canonicalize_ledger_inputs(
        header=header,
        loop=loop,
        base_prompt=base_prompt,
        records=records,
        checkpoints=checkpoints,
    )
    verified = verify_anytime_ledger(
        header=header,
        loop=loop,
        base_prompt=base_prompt,
        records=records,
        checkpoints=checkpoints,
    )
    if verified.terminal_reason is not None:
        raise AnytimeLedgerError("cannot prepare a turn after terminal state")
    scientific_call_index = verified.records_verified + 1
    if scientific_call_index > loop.budget.max_scientific_calls:
        raise AnytimeLedgerError("scientific call budget is exhausted")
    previous_candidate = None if not records else _candidate_source(records[-1].candidate)
    previous_feedback = None if not records else records[-1].feedback
    context = render_anytime_context(
        policy=loop.context,
        scientific_call_index=scientific_call_index,
        max_scientific_calls=loop.budget.max_scientific_calls,
        base_prompt=base_prompt,
        incumbent=verified.incumbent,
        previous_candidate=previous_candidate,
        previous_feedback=previous_feedback,
    )
    logical_request = build_anytime_logical_request(
        trajectory_id=header.trajectory_id,
        infrastructure_attempt_index=header.infrastructure_attempt_index,
        model_ref=header.model_ref,
        context=context,
        local_trajectory_seed=header.local_trajectory_seed,
    )
    return AnytimePreparedTurn(
        header_sha256=header.sha256,
        scientific_call_index=scientific_call_index,
        previous_ledger_sha256=verified.ledger_head_sha256,
        context=context,
        logical_request=logical_request,
    )


def _best_eligible_key(
    records: tuple[AnytimeTurnRecord, ...],
) -> tuple[float, int, str] | None:
    keys = tuple(
        (
            record.candidate.search_latency_ms,
            record.scientific_call_index,
            record.candidate.source.source_sha256,
        )
        for record in records
        if isinstance(record.candidate, AnytimeEligibleCandidate)
    )
    return None if not keys else min(keys)


def append_anytime_turn(
    *,
    header: AnytimeLedgerHeader,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
    records: tuple[AnytimeTurnRecord, ...],
    checkpoints: tuple[AnytimeCheckpointRecord, ...],
    provider: AnytimeProviderObservation,
    candidate: AnytimeCandidateOutcome | None,
    observed_wall_seconds: float,
) -> AnytimeAppendResult:
    """Append one already-observed submitted request as a fully derived record."""

    header, loop, base_prompt, records, checkpoints = _canonicalize_ledger_inputs(
        header=header,
        loop=loop,
        base_prompt=base_prompt,
        records=records,
        checkpoints=checkpoints,
    )
    provider = _PROVIDER_OBSERVATION_ADAPTER.validate_json(provider.model_dump_json())
    if candidate is not None:
        candidate = _CANDIDATE_OUTCOME_ADAPTER.validate_json(candidate.model_dump_json())
    prepared = prepare_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=base_prompt,
        records=records,
        checkpoints=checkpoints,
    )
    if provider.logical_request_sha256 != sha256_json(prepared.logical_request):
        raise AnytimeLedgerError("provider observation does not match the prepared request")
    if isinstance(provider, AnytimeProviderSuccess):
        if candidate is None:
            raise AnytimeLedgerError("provider success requires a candidate outcome")
        feedback = format_anytime_dev_feedback(candidate, loop)
    else:
        if candidate is not None:
            raise AnytimeLedgerError("provider terminal outcome cannot evaluate a candidate")
        feedback = None

    verified_before = verify_anytime_ledger(
        header=header,
        loop=loop,
        base_prompt=base_prompt,
        records=records,
        checkpoints=checkpoints,
    )
    incumbent_after = verified_before.incumbent
    incumbent_key = _best_eligible_key(records)
    if isinstance(candidate, AnytimeEligibleCandidate):
        candidate_key = (
            candidate.search_latency_ms,
            prepared.scientific_call_index,
            candidate.source.source_sha256,
        )
        if incumbent_key is None or candidate_key < incumbent_key:
            incumbent_after = candidate.source

    delta = _derive_resource_delta(provider, candidate, observed_wall_seconds)
    snapshot = fold_anytime_resources((*tuple(record.resource_delta for record in records), delta))
    _validate_resource_budget(snapshot, delta, loop)
    terminal_reason = _expected_terminal_reason(
        provider=provider,
        scientific_call_index=prepared.scientific_call_index,
        snapshot=snapshot,
        delta=delta,
        loop=loop,
    )
    record = AnytimeTurnRecord(
        header_sha256=header.sha256,
        trajectory_id=header.trajectory_id,
        infrastructure_attempt_index=header.infrastructure_attempt_index,
        scientific_call_index=prepared.scientific_call_index,
        trajectory_execution_sha256=header.trajectory_execution_sha256,
        previous_ledger_sha256=prepared.previous_ledger_sha256,
        context=prepared.context,
        logical_request=prepared.logical_request,
        provider=provider,
        candidate=candidate,
        feedback=feedback,
        incumbent_before=verified_before.incumbent,
        incumbent_after=incumbent_after,
        observed_wall_seconds=observed_wall_seconds,
        resource_delta=delta,
        resource_snapshot=snapshot,
        terminal_reason=terminal_reason,
    )
    checkpoint = (
        _checkpoint_for_record(header, record)
        if record.scientific_call_index in loop.checkpoints.calls
        else None
    )
    next_checkpoints = checkpoints if checkpoint is None else (*checkpoints, checkpoint)
    verify_anytime_ledger(
        header=header,
        loop=loop,
        base_prompt=base_prompt,
        records=(*records, record),
        checkpoints=next_checkpoints,
    )
    return AnytimeAppendResult(record=record, checkpoint=checkpoint)
