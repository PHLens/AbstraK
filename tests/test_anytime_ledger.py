from __future__ import annotations

from typing import cast

import pytest
from pydantic import TypeAdapter, ValidationError

from abstrak.anytime.context import AnytimeSourceSnapshot
from abstrak.anytime.contracts import (
    AnytimeCheckpointPolicy,
    AnytimeLoopPolicy,
    AnytimeResourceBudget,
)
from abstrak.anytime.ledger import (
    AnytimeCandidateOutcome,
    AnytimeCandidateTimeout,
    AnytimeCompileFailure,
    AnytimeDuplicateSource,
    AnytimeEligibleCandidate,
    AnytimeIneligibleCandidate,
    AnytimeLedgerError,
    AnytimeLedgerHeader,
    AnytimeOversizeSource,
    AnytimeParseFailure,
    AnytimeProviderAmbiguousSubmission,
    AnytimeProviderObservation,
    AnytimeProviderSubmittedError,
    AnytimeProviderSuccess,
    AnytimeStaticCheckFailure,
    AnytimeTimingUnstable,
    AnytimeTokenUsage,
    AnytimeTurnRecord,
    AnytimeWrongResult,
    append_anytime_turn,
    build_anytime_ledger_header,
    prepare_anytime_turn,
    verify_anytime_ledger,
)
from abstrak.providers.contracts import ChatMessage, ErrorCategory, MessageRole, sha256_json


def _loop(
    calls: int = 12,
    *,
    compile_attempts: int | None = None,
    evaluation_attempts: int | None = None,
) -> AnytimeLoopPolicy:
    return AnytimeLoopPolicy(
        budget=AnytimeResourceBudget(
            max_scientific_calls=calls,
            max_total_output_tokens=1200,
            max_compile_attempts=(calls if compile_attempts is None else compile_attempts),
            max_evaluation_attempts=(calls if evaluation_attempts is None else evaluation_attempts),
            max_gpu_seconds=float(calls * 10),
            max_provider_seconds_per_call=10.0,
            max_candidate_seconds_per_call=10.0,
            max_trajectory_wall_seconds=1000.0,
        ),
        checkpoints=AnytimeCheckpointPolicy(calls=(1, 4, 8, 12) if calls == 12 else (1, 4)),
    )


def _base_prompt() -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(role=MessageRole.SYSTEM, content="Generate one candidate."),
        ChatMessage(role=MessageRole.USER, content="TASK\nreduce\nTARGET\nTriton"),
    )


def _header(loop: AnytimeLoopPolicy) -> AnytimeLedgerHeader:
    return build_anytime_ledger_header(
        trajectory_id="trajectory-1",
        infrastructure_attempt_index=1,
        trajectory_execution_sha256="1" * 64,
        agent_binding_sha256="2" * 64,
        task_sha256="3" * 64,
        target_sha256="4" * 64,
        environment_sha256="5" * 64,
        model_ref="gpt-5.6-luna",
        loop=loop,
        base_prompt=_base_prompt(),
        local_trajectory_seed=17,
    )


def _success_for_next(
    header: AnytimeLedgerHeader,
    loop: AnytimeLoopPolicy,
    records: tuple[AnytimeTurnRecord, ...],
    checkpoints: tuple[object, ...],
    *,
    complete_usage: bool = True,
) -> AnytimeProviderSuccess:
    prepared = prepare_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=records,
        checkpoints=checkpoints,
    )
    call = prepared.scientific_call_index
    return AnytimeProviderSuccess(
        logical_request_sha256=sha256_json(prepared.logical_request),
        response_artifact_sha256=f"{call:064x}",
        usage=AnytimeTokenUsage(
            input_tokens=100 + call,
            cached_input_tokens=0 if complete_usage else None,
            output_tokens=10,
            reasoning_tokens=5 if complete_usage else None,
        ),
        provider_seconds=0.1,
    )


def _append_candidate(
    header: AnytimeLedgerHeader,
    loop: AnytimeLoopPolicy,
    records: tuple[AnytimeTurnRecord, ...],
    checkpoints: tuple[object, ...],
    candidate: AnytimeCandidateOutcome,
) -> tuple[tuple[AnytimeTurnRecord, ...], tuple[object, ...]]:
    result = append_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=records,
        checkpoints=checkpoints,
        provider=_success_for_next(header, loop, records, checkpoints),
        candidate=candidate,
        observed_wall_seconds=1.0,
    )
    records = (*records, result.record)
    if result.checkpoint is not None:
        checkpoints = (*checkpoints, result.checkpoint)
    return records, checkpoints


def _source(name: str) -> AnytimeSourceSnapshot:
    return AnytimeSourceSnapshot.from_source(f"class ModelNew:\n    name = '{name}'\n")


def test_full_synthetic_history_reconstructs_context_resources_and_incumbent() -> None:
    loop = _loop()
    header = _header(loop)
    records: tuple[AnytimeTurnRecord, ...] = ()
    checkpoints: tuple[object, ...] = ()
    source_a = _source("a")
    eligible_a = AnytimeEligibleCandidate(
        source=source_a,
        search_latency_ms=2.0,
        compile_seconds=0.1,
        evaluation_seconds=0.2,
        gpu_seconds=0.2,
    )
    outcomes: tuple[AnytimeCandidateOutcome, ...] = (
        AnytimeParseFailure(error="no fenced candidate"),
        eligible_a,
        AnytimeDuplicateSource(
            source=source_a,
            duplicate_of_call_index=2,
            original_assessment_sha256=eligible_a.sha256,
        ),
        AnytimeCompileFailure(
            source=_source("b"),
            compile_seconds=0.3,
            error="compiler error",
        ),
        AnytimeWrongResult(
            source=_source("c"),
            compile_seconds=0.1,
            evaluation_seconds=0.2,
            gpu_seconds=0.2,
            diagnostics=("mismatch",),
        ),
        AnytimeStaticCheckFailure(source=_source("d"), diagnostics=("forbidden fallback",)),
        AnytimeCandidateTimeout(
            stage="compile",
            source=_source("e"),
            compile_seconds=0.5,
            evaluation_seconds=0.0,
            gpu_seconds=0.0,
            error="compile timeout",
        ),
        AnytimeEligibleCandidate(
            source=_source("f"),
            search_latency_ms=3.0,
            compile_seconds=0.1,
            evaluation_seconds=0.2,
            gpu_seconds=0.2,
        ),
        AnytimeTimingUnstable(
            source=_source("g"),
            observed_latency_ms=1.5,
            compile_seconds=0.1,
            evaluation_seconds=0.2,
            gpu_seconds=0.2,
            diagnostics=("confidence interval too wide",),
        ),
        AnytimeIneligibleCandidate(
            reason="fallback_detected",
            source=_source("h"),
            compile_seconds=0.1,
            evaluation_seconds=0.2,
            gpu_seconds=0.2,
            diagnostics=("framework fallback launched",),
        ),
        AnytimeEligibleCandidate(
            source=_source("j"),
            search_latency_ms=1.0,
            compile_seconds=0.1,
            evaluation_seconds=0.2,
            gpu_seconds=0.2,
        ),
        AnytimeEligibleCandidate(
            source=_source("k"),
            search_latency_ms=1.0,
            compile_seconds=0.1,
            evaluation_seconds=0.2,
            gpu_seconds=0.2,
        ),
    )

    for outcome in outcomes:
        records, checkpoints = _append_candidate(header, loop, records, checkpoints, outcome)

    verified = verify_anytime_ledger(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=records,
        checkpoints=checkpoints,
    )
    assert [checkpoint.identity.scientific_call_index for checkpoint in checkpoints] == [
        1,
        4,
        8,
        12,
    ]
    assert [
        None if checkpoint.incumbent is None else checkpoint.incumbent.source_sha256
        for checkpoint in checkpoints
    ] == [None, source_a.source_sha256, source_a.source_sha256, _source("j").source_sha256]
    assert records[0].context.previous_candidate_sha256 is None
    assert records[1].context.previous_candidate_sha256 is None
    assert records[2].context.previous_candidate_sha256 == source_a.source_sha256
    assert records[3].context.previous_candidate_sha256 == source_a.source_sha256
    assert records[2].resource_delta.compile_attempts == 0
    assert records[2].resource_delta.evaluation_attempts == 0
    assert records[2].resource_delta.gpu_seconds == 0.0
    assert verified.resource_snapshot.compile_attempts == 9
    assert verified.resource_snapshot.evaluation_attempts == 7
    assert verified.resource_snapshot.usage_complete is True
    assert verified.resource_snapshot.known_output_tokens == 120
    assert verified.incumbent == _source("j")
    assert verified.terminal_reason == "call_budget"


@pytest.mark.parametrize("kind", ("submitted_error", "ambiguous_submission"))
def test_submitted_provider_terminal_consumes_call_and_emits_reached_checkpoint(
    kind: str,
) -> None:
    loop = _loop(calls=4)
    header = _header(loop)
    prepared = prepare_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
    )
    request_sha256 = sha256_json(prepared.logical_request)
    provider: AnytimeProviderObservation
    if kind == "submitted_error":
        provider = AnytimeProviderSubmittedError(
            logical_request_sha256=request_sha256,
            error_artifact_sha256="a" * 64,
            category=ErrorCategory.TIMEOUT,
            possibly_charged=True,
            partial_usage=AnytimeTokenUsage(
                input_tokens=10,
                cached_input_tokens=None,
                output_tokens=None,
                reasoning_tokens=None,
            ),
            provider_seconds=0.5,
        )
    else:
        provider = AnytimeProviderAmbiguousSubmission(
            logical_request_sha256=request_sha256,
            dispatch_intent_sha256="b" * 64,
            provider_seconds=0.5,
        )
    appended = append_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=(),
        checkpoints=(),
        provider=provider,
        candidate=None,
        observed_wall_seconds=0.5,
    )
    assert appended.record.scientific_call_index == 1
    assert appended.record.resource_snapshot.scientific_calls_consumed == 1
    assert appended.record.resource_snapshot.possibly_charged_requests == 1
    assert appended.record.resource_snapshot.usage_complete is False
    assert appended.checkpoint is not None
    assert appended.checkpoint.incumbent is None
    with pytest.raises(AnytimeLedgerError, match="terminal"):
        prepare_anytime_turn(
            header=header,
            loop=loop,
            base_prompt=_base_prompt(),
            records=(appended.record,),
            checkpoints=(appended.checkpoint,),
        )


def test_pre_submit_failure_has_no_scientific_outcome_and_resource_cap_stops_prefix() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(AnytimeProviderObservation).validate_python(
            {"kind": "unsubmitted_error"}, strict=True
        )

    loop = _loop(calls=4, compile_attempts=1, evaluation_attempts=1)
    header = _header(loop)
    records, checkpoints = _append_candidate(
        header,
        loop,
        (),
        (),
        AnytimeEligibleCandidate(
            source=_source("cap"),
            search_latency_ms=1.0,
            compile_seconds=0.1,
            evaluation_seconds=0.2,
            gpu_seconds=0.2,
        ),
    )
    checkpoint = cast(object, checkpoints[0])
    assert records[0].terminal_reason == "resource_cap"
    assert checkpoint is not None
    with pytest.raises(AnytimeLedgerError, match="terminal"):
        prepare_anytime_turn(
            header=header,
            loop=loop,
            base_prompt=_base_prompt(),
            records=records,
            checkpoints=checkpoints,
        )


def test_oversize_outcome_carries_no_source_into_next_context() -> None:
    loop = _loop(calls=4)
    header = _header(loop)
    records, checkpoints = _append_candidate(
        header,
        loop,
        (),
        (),
        AnytimeOversizeSource(
            rejected_source_sha256="c" * 64,
            source_characters=loop.context.max_candidate_source_characters + 1,
            error="source exceeded context limit",
        ),
    )
    prepared = prepare_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=records,
        checkpoints=checkpoints,
    )
    assert prepared.context.previous_candidate_sha256 is None
    assert prepared.context.previous_feedback_sha256 == records[0].feedback.sha256
