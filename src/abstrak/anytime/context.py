"""Pure deterministic context construction for anytime trajectories."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from abstrak.anytime.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    AnytimeContextPolicy,
    AnytimeModel,
)
from abstrak.providers.contracts import (
    ChatMessage,
    LogicalRequest,
    MessageRole,
    canonical_json_bytes,
    sha256_json,
)

_COMPONENT_ORDER = (
    "base_prompt",
    "incumbent",
    "previous_candidate",
    "previous_feedback",
)
_TRUNCATION_MARKER = "...[truncated]"

FeedbackStatus = Literal[
    "eligible",
    "ineligible",
    "qualification_pending",
    "parse_failure",
    "oversize_source",
    "duplicate_source",
    "static_check_failed",
    "compile_error",
    "wrong_result",
    "timeout",
    "runtime_error",
    "timing_unstable",
]


class AnytimeContextError(ValueError):
    """Raised when a normalized context cannot be rendered without policy drift."""


class AnytimeSourceSnapshot(AnytimeModel):
    """One exact candidate source; source text is never normalized or truncated."""

    schema_version: Literal["abstrak-anytime-source-snapshot.v1"] = (
        "abstrak-anytime-source-snapshot.v1"
    )
    source: str = Field(min_length=1)
    source_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def source_matches_hash(self) -> AnytimeSourceSnapshot:
        observed = hashlib.sha256(self.source.encode("utf-8")).hexdigest()
        if observed != self.source_sha256:
            raise ValueError("candidate source does not match source_sha256")
        return self

    @classmethod
    def from_source(cls, source: str) -> AnytimeSourceSnapshot:
        return cls(
            source=source,
            source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )


class AnytimeFeedbackInput(AnytimeModel):
    """Unbounded dev-only facts projected into one bounded Agent-visible envelope."""

    schema_version: Literal["abstrak-anytime-feedback-input.v1"] = (
        "abstrak-anytime-feedback-input.v1"
    )
    status: FeedbackStatus
    compiled: bool | None = None
    correct: bool | None = None
    median_latency_ms: float | None = Field(default=None, gt=0)
    diagnostics: tuple[str, ...] = ()
    error: str | None = None

    @field_validator("median_latency_ms")
    @classmethod
    def latency_is_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("feedback latency must be finite")
        return value

    @field_validator("diagnostics")
    @classmethod
    def diagnostics_are_nonempty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value for value in values):
            raise ValueError("feedback diagnostics cannot contain empty strings")
        return values

    @model_validator(mode="after")
    def outcome_facts_are_coherent(self) -> AnytimeFeedbackInput:
        _validate_feedback_facts(
            status=self.status,
            compiled=self.compiled,
            correct=self.correct,
            median_latency_ms=self.median_latency_ms,
        )
        return self


class AnytimeDevFeedback(AnytimeModel):
    """Canonical bounded feedback that may enter the next provider request."""

    schema_version: Literal["abstrak-anytime-dev-feedback.v1"] = "abstrak-anytime-dev-feedback.v1"
    renderer_schema_version: Literal["anytime-dev-feedback.v1"] = "anytime-dev-feedback.v1"
    status: FeedbackStatus
    compiled: bool | None = None
    correct: bool | None = None
    median_latency_ms: float | None = Field(default=None, gt=0)
    diagnostics: tuple[str, ...] = ()
    diagnostics_omitted: int = Field(ge=0)
    diagnostics_truncated: int = Field(ge=0)
    error: str | None = None
    error_truncated: bool

    @field_validator("median_latency_ms")
    @classmethod
    def latency_is_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("feedback latency must be finite")
        return value

    @field_validator("diagnostics")
    @classmethod
    def diagnostics_are_nonempty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value for value in values):
            raise ValueError("feedback diagnostics cannot contain empty strings")
        return values

    @model_validator(mode="after")
    def truncation_counts_are_coherent(self) -> AnytimeDevFeedback:
        if self.diagnostics_truncated > len(self.diagnostics):
            raise ValueError("truncated diagnostic count exceeds retained diagnostics")
        if self.error is None and self.error_truncated:
            raise ValueError("missing feedback error cannot be marked truncated")
        _validate_feedback_facts(
            status=self.status,
            compiled=self.compiled,
            correct=self.correct,
            median_latency_ms=self.median_latency_ms,
        )
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _validate_feedback_facts(
    *,
    status: FeedbackStatus,
    compiled: bool | None,
    correct: bool | None,
    median_latency_ms: float | None,
) -> None:
    """Reject combinations that could contradict a candidate outcome."""

    if correct is True and compiled is not True:
        raise ValueError("correct feedback requires compiled=true")
    if median_latency_ms is not None and (compiled is not True or correct is not True):
        raise ValueError("latency feedback requires a compiled, correct candidate")
    if status == "eligible" and (
        compiled is not True or correct is not True or median_latency_ms is None
    ):
        raise ValueError("eligible feedback requires compiled, correct, and latency facts")
    if status == "timing_unstable" and (compiled is not True or correct is not True):
        raise ValueError("timing-unstable feedback requires a compiled, correct candidate")
    if status == "ineligible" and (compiled is not True or correct is not True):
        raise ValueError("ineligible feedback requires a compiled, correct candidate")
    if status == "qualification_pending" and (
        compiled is not None or correct is not None or median_latency_ms is not None
    ):
        raise ValueError("qualification-pending feedback cannot assert execution facts")
    if status == "wrong_result" and correct is not False:
        raise ValueError("wrong-result feedback requires correct=false")
    if status == "compile_error" and compiled is not False:
        raise ValueError("compile-error feedback requires compiled=false")
    if (
        status
        in {
            "parse_failure",
            "oversize_source",
            "duplicate_source",
            "static_check_failed",
        }
        and median_latency_ms is not None
    ):
        raise ValueError(f"{status} feedback cannot carry latency")


def _truncate_text(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    retained = limit - len(_TRUNCATION_MARKER)
    return f"{value[:retained]}{_TRUNCATION_MARKER}", True


def bound_anytime_feedback(
    value: AnytimeFeedbackInput,
    policy: AnytimeContextPolicy,
) -> AnytimeDevFeedback:
    """Apply the only diagnostic/error truncation algorithm used by context v1."""

    retained_inputs = value.diagnostics[: policy.max_diagnostic_items]
    retained: list[str] = []
    truncated_count = 0
    for diagnostic in retained_inputs:
        rendered, truncated = _truncate_text(
            diagnostic,
            policy.max_diagnostic_characters_per_item,
        )
        retained.append(rendered)
        truncated_count += int(truncated)
    error = None
    error_truncated = False
    if value.error is not None:
        error, error_truncated = _truncate_text(value.error, policy.max_error_characters)
    return AnytimeDevFeedback(
        renderer_schema_version=policy.feedback_schema_version,
        status=value.status,
        compiled=value.compiled,
        correct=value.correct,
        median_latency_ms=value.median_latency_ms,
        diagnostics=tuple(retained),
        diagnostics_omitted=len(value.diagnostics) - len(retained_inputs),
        diagnostics_truncated=truncated_count,
        error=error,
        error_truncated=error_truncated,
    )


class AnytimeRenderedContext(AnytimeModel):
    """Complete provider-visible messages and hashes for one scientific call."""

    schema_version: Literal["abstrak-anytime-rendered-context.v1"] = (
        "abstrak-anytime-rendered-context.v1"
    )
    renderer_version: Literal["anytime-context-renderer.v1"] = "anytime-context-renderer.v1"
    scientific_call_index: int = Field(ge=1, le=12)
    max_scientific_calls: int = Field(ge=1, le=12)
    component_order: tuple[str, ...]
    base_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    incumbent_candidate_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    previous_candidate_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    previous_feedback_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    messages: tuple[ChatMessage, ...] = Field(min_length=4)
    messages_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def identities_are_coherent(self) -> AnytimeRenderedContext:
        if self.component_order != _COMPONENT_ORDER:
            raise ValueError("rendered context components are not in canonical order")
        if self.scientific_call_index > self.max_scientific_calls:
            raise ValueError("context call index exceeds its scientific-call budget")
        observed = sha256_json([message.model_dump(mode="json") for message in self.messages])
        if observed != self.messages_sha256:
            raise ValueError("rendered context messages do not match messages_sha256")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _source_message(
    label: str,
    source: AnytimeSourceSnapshot | None,
    *,
    prefix: str = "",
) -> ChatMessage:
    if source is None:
        body = "NONE"
    else:
        body = f"sha256={source.source_sha256}\nsource-begins-next-line\n{source.source}"
    return ChatMessage(role=MessageRole.USER, content=f"{prefix}{label}\n{body}")


def render_anytime_context(
    *,
    policy: AnytimeContextPolicy,
    scientific_call_index: int,
    max_scientific_calls: int,
    base_prompt: tuple[ChatMessage, ...],
    incumbent: AnytimeSourceSnapshot | None,
    previous_candidate: AnytimeSourceSnapshot | None,
    previous_feedback: AnytimeDevFeedback | None,
) -> AnytimeRenderedContext:
    """Render one normalized context without provider history or source truncation."""

    policy = AnytimeContextPolicy.model_validate_json(policy.model_dump_json())
    base_prompt = tuple(
        ChatMessage.model_validate_json(message.model_dump_json()) for message in base_prompt
    )
    incumbent = (
        None
        if incumbent is None
        else AnytimeSourceSnapshot.model_validate_json(incumbent.model_dump_json())
    )
    previous_candidate = (
        None
        if previous_candidate is None
        else AnytimeSourceSnapshot.model_validate_json(previous_candidate.model_dump_json())
    )
    previous_feedback = (
        None
        if previous_feedback is None
        else AnytimeDevFeedback.model_validate_json(previous_feedback.model_dump_json())
    )
    if not base_prompt:
        raise AnytimeContextError("base prompt cannot be empty")
    if scientific_call_index < 1 or scientific_call_index > max_scientific_calls:
        raise AnytimeContextError("scientific call index is outside its budget")
    if scientific_call_index == 1 and any(
        value is not None for value in (incumbent, previous_candidate, previous_feedback)
    ):
        raise AnytimeContextError("first scientific call must have empty reconstructed state")
    for label, source in (
        ("incumbent", incumbent),
        ("previous candidate", previous_candidate),
    ):
        if source is not None and len(source.source) > policy.max_candidate_source_characters:
            raise AnytimeContextError(f"{label} source exceeds the frozen context limit")
    if previous_feedback is not None:
        if len(previous_feedback.diagnostics) > policy.max_diagnostic_items:
            raise AnytimeContextError("previous feedback exceeds the diagnostic item limit")
        if any(
            len(diagnostic) > policy.max_diagnostic_characters_per_item
            for diagnostic in previous_feedback.diagnostics
        ):
            raise AnytimeContextError("previous feedback exceeds the diagnostic character limit")
        if (
            previous_feedback.error is not None
            and len(previous_feedback.error) > policy.max_error_characters
        ):
            raise AnytimeContextError("previous feedback exceeds the error character limit")
    state_prefix = (
        "ANYTIME_STATE\n"
        f"scientific_call_index={scientific_call_index}\n"
        f"max_scientific_calls={max_scientific_calls}\n"
    )
    dynamic = (
        _source_message("INCUMBENT", incumbent, prefix=state_prefix),
        _source_message("PREVIOUS_CANDIDATE", previous_candidate),
        ChatMessage(
            role=MessageRole.USER,
            content=(
                "PREVIOUS_FEEDBACK\nNONE"
                if previous_feedback is None
                else "PREVIOUS_FEEDBACK\n" + canonical_json_bytes(previous_feedback).decode("utf-8")
            ),
        ),
    )
    messages = (*base_prompt, *dynamic)
    return AnytimeRenderedContext(
        renderer_version=policy.renderer_version,
        scientific_call_index=scientific_call_index,
        max_scientific_calls=max_scientific_calls,
        component_order=policy.component_order,
        base_prompt_sha256=sha256_json(
            [message.model_dump(mode="json") for message in base_prompt]
        ),
        incumbent_candidate_sha256=(None if incumbent is None else incumbent.source_sha256),
        previous_candidate_sha256=(
            None if previous_candidate is None else previous_candidate.source_sha256
        ),
        previous_feedback_sha256=(None if previous_feedback is None else previous_feedback.sha256),
        messages=messages,
        messages_sha256=sha256_json([message.model_dump(mode="json") for message in messages]),
    )


def deterministic_anytime_request_id(
    trajectory_id: str,
    infrastructure_attempt_index: int,
    scientific_call_index: int,
) -> str:
    """Return the stable artifact request ID for one attempt-local call."""

    return f"{trajectory_id}.attempt-{infrastructure_attempt_index}.call-{scientific_call_index}"


def build_anytime_logical_request(
    *,
    trajectory_id: str,
    infrastructure_attempt_index: int,
    model_ref: str,
    context: AnytimeRenderedContext,
    local_trajectory_seed: int | None = None,
) -> LogicalRequest:
    """Build the exact request; one-based scientific calls map to v1 zero-based turns."""

    context = AnytimeRenderedContext.model_validate_json(context.model_dump_json())
    if not trajectory_id or re.fullmatch(IDENTIFIER_PATTERN, trajectory_id) is None:
        raise AnytimeContextError("trajectory_id is not a safe identifier")
    if infrastructure_attempt_index not in {1, 2}:
        raise AnytimeContextError("infrastructure attempt index must be one or two")
    if local_trajectory_seed is not None and local_trajectory_seed < 0:
        raise AnytimeContextError("local trajectory seed cannot be negative")
    return LogicalRequest(
        request_id=deterministic_anytime_request_id(
            trajectory_id,
            infrastructure_attempt_index,
            context.scientific_call_index,
        ),
        model_ref=model_ref,
        messages=context.messages,
        trajectory_id=trajectory_id,
        turn_index=context.scientific_call_index - 1,
        local_trajectory_seed=local_trajectory_seed,
    )
