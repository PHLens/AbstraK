from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from abstrak.anytime.context import (
    AnytimeContextError,
    AnytimeDevFeedback,
    AnytimeFeedbackInput,
    AnytimeRenderedContext,
    AnytimeSourceSnapshot,
    bound_anytime_feedback,
    build_anytime_logical_request,
    render_anytime_context,
)
from abstrak.anytime.contracts import AnytimeContextPolicy
from abstrak.providers.contracts import ChatMessage, MessageRole, sha256_json


def _base_prompt() -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(role=MessageRole.SYSTEM, content="Optimize the frozen task."),
        ChatMessage(role=MessageRole.USER, content="TASK\nrow reduction\nTARGET\nTriton"),
    )


def test_first_call_has_canonical_explicit_none_components() -> None:
    context = render_anytime_context(
        policy=AnytimeContextPolicy(),
        scientific_call_index=1,
        max_scientific_calls=12,
        base_prompt=_base_prompt(),
        incumbent=None,
        previous_candidate=None,
        previous_feedback=None,
    )

    assert context.component_order == (
        "base_prompt",
        "incumbent",
        "previous_candidate",
        "previous_feedback",
    )
    assert context.messages[:2] == _base_prompt()
    assert [message.content for message in context.messages[2:]] == [
        "ANYTIME_STATE\nscientific_call_index=1\nmax_scientific_calls=12\nINCUMBENT\nNONE",
        "PREVIOUS_CANDIDATE\nNONE",
        "PREVIOUS_FEEDBACK\nNONE",
    ]
    assert context.incumbent_candidate_sha256 is None
    assert context.previous_candidate_sha256 is None
    assert context.previous_feedback_sha256 is None
    assert context.base_prompt_sha256 == sha256_json(
        [message.model_dump(mode="json") for message in _base_prompt()]
    )


def test_sources_are_preserved_and_request_identity_is_deterministic() -> None:
    incumbent = AnytimeSourceSnapshot.from_source("class ModelNew:\n    note = '你好'  \n")
    previous = AnytimeSourceSnapshot.from_source("class ModelNew:\r\n    pass\t \r\n")
    feedback = bound_anytime_feedback(
        AnytimeFeedbackInput(
            status="wrong_result",
            compiled=True,
            correct=False,
            diagnostics=("max error was 1.0",),
        ),
        AnytimeContextPolicy(),
    )
    context = render_anytime_context(
        policy=AnytimeContextPolicy(),
        scientific_call_index=4,
        max_scientific_calls=12,
        base_prompt=_base_prompt(),
        incumbent=incumbent,
        previous_candidate=previous,
        previous_feedback=feedback,
    )
    first = build_anytime_logical_request(
        trajectory_id="trajectory-1",
        infrastructure_attempt_index=2,
        model_ref="gpt-5.6-luna",
        context=context,
        local_trajectory_seed=7,
    )
    second = build_anytime_logical_request(
        trajectory_id="trajectory-1",
        infrastructure_attempt_index=2,
        model_ref="gpt-5.6-luna",
        context=context,
        local_trajectory_seed=7,
    )

    assert first == second
    assert first.request_id == "trajectory-1.attempt-2.call-4"
    assert first.turn_index == 3
    assert first.trajectory_id == "trajectory-1"
    assert incumbent.source in context.messages[2].content
    assert previous.source in context.messages[3].content
    assert context.incumbent_candidate_sha256 == incumbent.source_sha256
    assert context.previous_candidate_sha256 == previous.source_sha256
    assert context.previous_feedback_sha256 == feedback.sha256


def test_feedback_is_bounded_with_explicit_deterministic_truncation() -> None:
    policy = AnytimeContextPolicy(
        max_diagnostic_items=2,
        max_diagnostic_characters_per_item=64,
        max_error_characters=64,
    )
    value = AnytimeFeedbackInput(
        status="compile_error",
        compiled=False,
        correct=False,
        diagnostics=("a" * 100, "short", "omitted"),
        error="e" * 100,
    )

    first = bound_anytime_feedback(value, policy)
    second = bound_anytime_feedback(value, policy)

    assert first == second
    assert len(first.diagnostics) == 2
    assert len(first.diagnostics[0]) == 64
    assert first.diagnostics[0].endswith("...[truncated]")
    assert first.diagnostics_omitted == 1
    assert first.diagnostics_truncated == 1
    assert first.error is not None and len(first.error) == 64
    assert first.error.endswith("...[truncated]")
    assert first.error_truncated is True
    assert first.sha256 == second.sha256


@pytest.mark.parametrize(
    "feedback",
    (
        AnytimeDevFeedback(
            status="parse_failure",
            diagnostics=tuple("item" for _ in range(3)),
            diagnostics_omitted=0,
            diagnostics_truncated=0,
            error=None,
            error_truncated=False,
        ),
        AnytimeDevFeedback(
            status="parse_failure",
            diagnostics=("x" * 65,),
            diagnostics_omitted=0,
            diagnostics_truncated=0,
            error=None,
            error_truncated=False,
        ),
        AnytimeDevFeedback(
            status="parse_failure",
            diagnostics=(),
            diagnostics_omitted=0,
            diagnostics_truncated=0,
            error="x" * 65,
            error_truncated=False,
        ),
    ),
    ids=("items", "diagnostic-characters", "error-characters"),
)
def test_renderer_rejects_direct_feedback_that_bypasses_policy_bound(
    feedback: AnytimeDevFeedback,
) -> None:
    policy = AnytimeContextPolicy(
        max_diagnostic_items=2,
        max_diagnostic_characters_per_item=64,
        max_error_characters=64,
    )
    with pytest.raises(AnytimeContextError, match="previous feedback exceeds"):
        render_anytime_context(
            policy=policy,
            scientific_call_index=2,
            max_scientific_calls=4,
            base_prompt=_base_prompt(),
            incumbent=None,
            previous_candidate=None,
            previous_feedback=feedback,
        )


def test_oversize_source_rejects_instead_of_truncating() -> None:
    policy = AnytimeContextPolicy(max_candidate_source_characters=1024)
    source_text = "x" * 1025
    source = AnytimeSourceSnapshot.from_source(source_text)

    with pytest.raises(AnytimeContextError, match="exceeds"):
        render_anytime_context(
            policy=policy,
            scientific_call_index=2,
            max_scientific_calls=4,
            base_prompt=_base_prompt(),
            incumbent=None,
            previous_candidate=source,
            previous_feedback=None,
        )
    assert source.source == source_text
    assert source.source_sha256 == hashlib.sha256(source_text.encode()).hexdigest()

    exact = AnytimeSourceSnapshot.from_source("x" * 1024)
    rendered = render_anytime_context(
        policy=policy,
        scientific_call_index=2,
        max_scientific_calls=4,
        base_prompt=_base_prompt(),
        incumbent=None,
        previous_candidate=exact,
        previous_feedback=None,
    )
    assert exact.source in rendered.messages[3].content


@pytest.mark.parametrize("state_name", ("incumbent", "previous_candidate", "feedback"))
def test_first_call_rejects_nonempty_reconstructed_state(state_name: str) -> None:
    source = AnytimeSourceSnapshot.from_source("class ModelNew: pass\n")
    feedback = bound_anytime_feedback(
        AnytimeFeedbackInput(
            status="compile_error",
            compiled=False,
            error="compiler rejected candidate",
        ),
        AnytimeContextPolicy(),
    )
    values = {
        "incumbent": source if state_name == "incumbent" else None,
        "previous_candidate": source if state_name == "previous_candidate" else None,
        "previous_feedback": feedback if state_name == "feedback" else None,
    }
    with pytest.raises(AnytimeContextError, match="first scientific call"):
        render_anytime_context(
            policy=AnytimeContextPolicy(),
            scientific_call_index=1,
            max_scientific_calls=4,
            base_prompt=_base_prompt(),
            **values,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"status": "eligible", "compiled": True, "correct": True},
            "eligible feedback",
        ),
        (
            {"status": "compile_error", "compiled": True},
            "compile-error feedback",
        ),
        (
            {
                "status": "wrong_result",
                "compiled": True,
                "correct": False,
                "median_latency_ms": 1.0,
            },
            "latency feedback",
        ),
        (
            {
                "status": "qualification_pending",
                "compiled": False,
            },
            "qualification-pending feedback",
        ),
    ),
)
def test_feedback_rejects_contradictory_facts(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        AnytimeFeedbackInput(**kwargs)


def test_context_and_source_hash_tampering_fail_closed() -> None:
    source = AnytimeSourceSnapshot.from_source("class ModelNew: pass\n")
    with pytest.raises(ValidationError, match="source_sha256"):
        AnytimeSourceSnapshot(source=source.source, source_sha256="0" * 64)

    context = render_anytime_context(
        policy=AnytimeContextPolicy(),
        scientific_call_index=1,
        max_scientific_calls=4,
        base_prompt=_base_prompt(),
        incumbent=None,
        previous_candidate=None,
        previous_feedback=None,
    )
    payload = context.model_dump()
    payload["messages_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="messages_sha256"):
        AnytimeRenderedContext.model_validate(payload)


@pytest.mark.parametrize("attempt", (0, 3))
def test_request_rejects_invalid_attempt_index(attempt: int) -> None:
    context = render_anytime_context(
        policy=AnytimeContextPolicy(),
        scientific_call_index=1,
        max_scientific_calls=4,
        base_prompt=_base_prompt(),
        incumbent=None,
        previous_candidate=None,
        previous_feedback=None,
    )
    with pytest.raises(AnytimeContextError, match="attempt index"):
        build_anytime_logical_request(
            trajectory_id="trajectory-1",
            infrastructure_attempt_index=attempt,
            model_ref="deepseek-v4-flash",
            context=context,
        )
