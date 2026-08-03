from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from abstrak.anytime.context import AnytimeSourceSnapshot
from abstrak.anytime.contracts import (
    AnytimeCheckpointIdentity,
    AnytimeCheckpointPolicy,
    AnytimeLoopPolicy,
    AnytimeResourceBudget,
)
from abstrak.anytime.ledger import (
    AnytimeCheckpointRecord,
    AnytimeCompileFailure,
    AnytimeDuplicateSource,
    AnytimeEligibleCandidate,
    AnytimeLedgerError,
    AnytimeLedgerHeader,
    AnytimeParseFailure,
    AnytimeProviderAmbiguousSubmission,
    AnytimeProviderSuccess,
    AnytimeResourceDelta,
    AnytimeTokenUsage,
    AnytimeTurnRecord,
    append_anytime_turn,
    build_anytime_ledger_header,
    prepare_anytime_turn,
    rebuild_anytime_checkpoints,
    verify_anytime_ledger,
)
from abstrak.providers.contracts import ChatMessage, MessageRole, sha256_json


def _loop() -> AnytimeLoopPolicy:
    return AnytimeLoopPolicy(
        budget=AnytimeResourceBudget(
            max_scientific_calls=4,
            max_total_output_tokens=1024,
            max_compile_attempts=4,
            max_evaluation_attempts=4,
            max_gpu_seconds=40.0,
            max_provider_seconds_per_call=10.0,
            max_candidate_seconds_per_call=10.0,
            max_trajectory_wall_seconds=100.0,
        ),
        checkpoints=AnytimeCheckpointPolicy(calls=(1, 4)),
    )


def _base_prompt() -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(role=MessageRole.SYSTEM, content="Return one candidate."),
        ChatMessage(role=MessageRole.USER, content="TASK\nsoftmax\nTARGET\nTileLang"),
    )


def _header(loop: AnytimeLoopPolicy) -> AnytimeLedgerHeader:
    return build_anytime_ledger_header(
        trajectory_id="tamper-trajectory",
        infrastructure_attempt_index=1,
        trajectory_execution_sha256="1" * 64,
        agent_binding_sha256="2" * 64,
        task_sha256="3" * 64,
        target_sha256="4" * 64,
        environment_sha256="5" * 64,
        model_ref="deepseek-v4-flash",
        loop=loop,
        base_prompt=_base_prompt(),
        local_trajectory_seed=9,
    )


def _source(name: str) -> AnytimeSourceSnapshot:
    return AnytimeSourceSnapshot.from_source(f"class ModelNew:\n    value = '{name}'\n")


def _provider(
    header: AnytimeLedgerHeader,
    loop: AnytimeLoopPolicy,
    records: tuple[AnytimeTurnRecord, ...],
    checkpoints: tuple[AnytimeCheckpointRecord, ...],
    *,
    output_tokens: int = 10,
) -> AnytimeProviderSuccess:
    prepared = prepare_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=records,
        checkpoints=checkpoints,
    )
    return AnytimeProviderSuccess(
        logical_request_sha256=sha256_json(prepared.logical_request),
        response_artifact_sha256=f"{prepared.scientific_call_index:064x}",
        usage=AnytimeTokenUsage(
            input_tokens=100,
            cached_input_tokens=0,
            output_tokens=output_tokens,
            reasoning_tokens=5,
        ),
        provider_seconds=0.1,
    )


def _valid_history() -> tuple[
    AnytimeLoopPolicy,
    AnytimeLedgerHeader,
    tuple[AnytimeTurnRecord, ...],
    tuple[AnytimeCheckpointRecord, ...],
]:
    loop = _loop()
    header = _header(loop)
    records: tuple[AnytimeTurnRecord, ...] = ()
    checkpoints: tuple[AnytimeCheckpointRecord, ...] = ()
    source_a = _source("a")
    eligible_a = AnytimeEligibleCandidate(
        source=source_a,
        search_latency_ms=2.0,
        compile_seconds=0.1,
        evaluation_seconds=0.2,
        gpu_seconds=0.2,
    )
    outcomes = (
        AnytimeParseFailure(error="parse failed"),
        eligible_a,
        AnytimeDuplicateSource(
            source=source_a,
            duplicate_of_call_index=2,
            original_assessment_sha256=eligible_a.sha256,
        ),
        AnytimeEligibleCandidate(
            source=_source("faster"),
            search_latency_ms=1.0,
            compile_seconds=0.1,
            evaluation_seconds=0.2,
            gpu_seconds=0.2,
        ),
    )
    for outcome in outcomes:
        result = append_anytime_turn(
            header=header,
            loop=loop,
            base_prompt=_base_prompt(),
            records=records,
            checkpoints=checkpoints,
            provider=_provider(header, loop, records, checkpoints),
            candidate=outcome,
            observed_wall_seconds=1.0,
        )
        records = (*records, result.record)
        if result.checkpoint is not None:
            checkpoints = (*checkpoints, result.checkpoint)
    verify_anytime_ledger(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=records,
        checkpoints=checkpoints,
    )
    return loop, header, records, checkpoints


def _verify(
    loop: AnytimeLoopPolicy,
    header: AnytimeLedgerHeader,
    records: tuple[AnytimeTurnRecord, ...],
    checkpoints: tuple[AnytimeCheckpointRecord, ...],
    *,
    base_prompt: tuple[ChatMessage, ...] | None = None,
) -> None:
    verify_anytime_ledger(
        header=header,
        loop=loop,
        base_prompt=_base_prompt() if base_prompt is None else base_prompt,
        records=records,
        checkpoints=checkpoints,
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda records: (records[0], records[2], records[3]),
        lambda records: (records[0], records[2], records[1], records[3]),
        lambda records: (records[0], records[1], records[1], records[3]),
    ),
    ids=("drop", "reorder", "duplicate"),
)
def test_drop_reorder_or_duplicate_call_fails_full_prefix_replay(
    mutate: Callable[[tuple[AnytimeTurnRecord, ...]], tuple[AnytimeTurnRecord, ...]],
) -> None:
    loop, header, records, checkpoints = _valid_history()
    with pytest.raises(AnytimeLedgerError):
        _verify(loop, header, mutate(records), checkpoints)


@pytest.mark.parametrize(
    ("index", "update"),
    (
        (1, {"infrastructure_attempt_index": 2}),
        (1, {"trajectory_execution_sha256": "e" * 64}),
        (1, {"previous_ledger_sha256": "d" * 64}),
    ),
    ids=("attempt", "execution", "previous-chain"),
)
def test_turn_identity_or_chain_tamper_fails(index: int, update: dict[str, object]) -> None:
    loop, header, records, checkpoints = _valid_history()
    changed = records[index].model_copy(update=update)
    tampered = (*records[:index], changed, *records[index + 1 :])
    with pytest.raises(AnytimeLedgerError):
        _verify(loop, header, tampered, checkpoints)


def test_external_header_and_base_prompt_remain_trust_anchors() -> None:
    loop, header, records, checkpoints = _valid_history()
    changed_header = header.model_copy(update={"agent_binding_sha256": "f" * 64})
    with pytest.raises(AnytimeLedgerError, match="header"):
        _verify(loop, changed_header, records, checkpoints)

    changed_base = (
        ChatMessage(role=MessageRole.SYSTEM, content="A different prompt."),
        _base_prompt()[1],
    )
    with pytest.raises(AnytimeLedgerError, match="base prompt"):
        _verify(loop, header, records, checkpoints, base_prompt=changed_base)

    invalid_base = (
        _base_prompt()[0].model_copy(update={"content": ""}),
        _base_prompt()[1],
    )
    with pytest.raises(ValidationError):
        _verify(loop, header, records, checkpoints, base_prompt=invalid_base)


def test_context_and_logical_request_are_fully_reconstructed() -> None:
    loop, header, records, checkpoints = _valid_history()
    target = records[1]
    messages = (
        *target.context.messages[:-1],
        ChatMessage(
            role=MessageRole.USER,
            content=target.context.messages[-1].content + "\nforged",
        ),
    )
    changed_context = target.context.model_copy(
        update={
            "messages": messages,
            "messages_sha256": sha256_json(
                [message.model_dump(mode="json") for message in messages]
            ),
        }
    )
    changed = target.model_copy(update={"context": changed_context})
    with pytest.raises(AnytimeLedgerError, match="context"):
        _verify(loop, header, (records[0], changed, *records[2:]), checkpoints)

    changed_request = target.logical_request.model_copy(update={"messages": messages})
    changed_provider = target.provider.model_copy(
        update={"logical_request_sha256": sha256_json(changed_request)}
    )
    changed = target.model_copy(
        update={"logical_request": changed_request, "provider": changed_provider}
    )
    with pytest.raises(AnytimeLedgerError, match="logical request"):
        _verify(loop, header, (records[0], changed, *records[2:]), checkpoints)


@pytest.mark.parametrize(
    "update",
    (
        {"turn_index": 8},
        {"trajectory_id": "another-trajectory"},
    ),
    ids=("turn-index", "request-trajectory"),
)
def test_invalid_request_identity_cannot_bypass_with_model_copy(
    update: dict[str, object],
) -> None:
    loop, header, records, checkpoints = _valid_history()
    changed_request = records[1].logical_request.model_copy(update=update)
    changed = records[1].model_copy(update={"logical_request": changed_request})
    with pytest.raises(ValidationError):
        _verify(loop, header, (records[0], changed, *records[2:]), checkpoints)


def test_swapped_provider_response_or_request_hash_fails_closed() -> None:
    loop, header, records, checkpoints = _valid_history()
    swapped = records[2].model_copy(update={"provider": records[1].provider})
    with pytest.raises(ValidationError):
        _verify(loop, header, (*records[:2], swapped, records[3]), checkpoints)

    changed_provider = records[1].provider.model_copy(update={"logical_request_sha256": "a" * 64})
    changed = records[1].model_copy(update={"provider": changed_provider})
    with pytest.raises(ValidationError):
        _verify(loop, header, (records[0], changed, *records[2:]), checkpoints)


def test_candidate_feedback_and_incumbent_tampering_fail_after_local_rehash() -> None:
    loop, header, records, checkpoints = _valid_history()
    duplicate = records[2].candidate
    assert isinstance(duplicate, AnytimeDuplicateSource)
    changed_source = AnytimeSourceSnapshot.from_source(duplicate.source.source + "# changed\n")
    changed_candidate = duplicate.model_copy(update={"source": changed_source})
    changed_record = records[2].model_copy(update={"candidate": changed_candidate})
    with pytest.raises(AnytimeLedgerError, match="duplicate source"):
        _verify(loop, header, (*records[:2], changed_record, records[3]), checkpoints)

    feedback = records[1].feedback
    assert feedback is not None
    changed_feedback = feedback.model_copy(update={"diagnostics": ("forged fact",)})
    changed_record = records[1].model_copy(update={"feedback": changed_feedback})
    with pytest.raises(AnytimeLedgerError, match="feedback"):
        _verify(loop, header, (records[0], changed_record, *records[2:]), checkpoints)

    slower_incumbent = records[1].incumbent_after
    changed_record = records[3].model_copy(update={"incumbent_after": slower_incumbent})
    with pytest.raises(AnytimeLedgerError, match="incumbent"):
        _verify(loop, header, (*records[:3], changed_record), checkpoints)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("duplicate_of_call_index", 1),
        ("original_assessment_sha256", "f" * 64),
    ),
    ids=("citation", "assessment-hash"),
)
def test_duplicate_cache_link_is_exact(field: str, value: object) -> None:
    loop, header, records, checkpoints = _valid_history()
    duplicate = records[2].candidate
    assert isinstance(duplicate, AnytimeDuplicateSource)
    changed_candidate = duplicate.model_copy(update={field: value})
    changed_record = records[2].model_copy(update={"candidate": changed_candidate})
    with pytest.raises(AnytimeLedgerError, match="duplicate source"):
        _verify(loop, header, (*records[:2], changed_record, records[3]), checkpoints)


@pytest.mark.parametrize("usage_complete", (False,))
def test_resource_delta_snapshot_and_completeness_are_recomputed(
    usage_complete: bool,
) -> None:
    loop, header, records, checkpoints = _valid_history()
    target = records[1]
    changed_delta = target.resource_delta.model_copy(
        update={
            "known_input_tokens": target.resource_delta.known_input_tokens + 1,
            "usage_complete": usage_complete,
        }
    )
    changed_snapshot = target.resource_snapshot.model_copy(
        update={
            "known_input_tokens": target.resource_snapshot.known_input_tokens + 1,
            "usage_complete": usage_complete,
        }
    )
    changed = target.model_copy(
        update={
            "resource_delta": changed_delta,
            "resource_snapshot": changed_snapshot,
        }
    )
    with pytest.raises(AnytimeLedgerError, match="resource delta"):
        _verify(loop, header, (records[0], changed, *records[2:]), checkpoints)

    invalid_delta = target.resource_delta.model_copy(update={"provider_seconds": float("nan")})
    invalid = target.model_copy(update={"resource_delta": invalid_delta})
    with pytest.raises(ValidationError):
        _verify(loop, header, (records[0], invalid, *records[2:]), checkpoints)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda checkpoints, _records: (checkpoints[1],),
        lambda checkpoints, _records: (checkpoints[0], checkpoints[0], checkpoints[1]),
        lambda checkpoints, _records: (checkpoints[1], checkpoints[0]),
    ),
    ids=("delete", "duplicate", "reorder"),
)
def test_checkpoint_set_must_be_exact(
    mutate: Callable[
        [tuple[AnytimeCheckpointRecord, ...], tuple[AnytimeTurnRecord, ...]],
        tuple[AnytimeCheckpointRecord, ...],
    ],
) -> None:
    loop, header, records, checkpoints = _valid_history()
    with pytest.raises(AnytimeLedgerError, match="checkpoint set"):
        _verify(loop, header, records, mutate(checkpoints, records))


def test_checkpoint_prefix_resource_incumbent_and_attempt_are_recomputed() -> None:
    loop, header, records, checkpoints = _valid_history()
    final = checkpoints[1]
    identity = final.identity.model_copy(update={"ledger_prefix_sha256": "f" * 64})
    changed = final.model_copy(update={"identity": identity})
    with pytest.raises(AnytimeLedgerError, match="checkpoint"):
        _verify(loop, header, records, (checkpoints[0], changed))

    changed_snapshot = final.resource_snapshot.model_copy(
        update={"provider_seconds": final.resource_snapshot.provider_seconds + 1.0}
    )
    identity = final.identity.model_copy(
        update={"resource_snapshot_sha256": changed_snapshot.sha256}
    )
    changed = final.model_copy(update={"identity": identity, "resource_snapshot": changed_snapshot})
    with pytest.raises(AnytimeLedgerError, match="checkpoint"):
        _verify(loop, header, records, (checkpoints[0], changed))

    old_incumbent = records[1].incumbent_after
    assert old_incumbent is not None
    identity = final.identity.model_copy(
        update={"incumbent_candidate_sha256": old_incumbent.source_sha256}
    )
    changed = final.model_copy(update={"identity": identity, "incumbent": old_incumbent})
    with pytest.raises(AnytimeLedgerError, match="checkpoint"):
        _verify(loop, header, records, (checkpoints[0], changed))

    identity = final.identity.model_copy(update={"infrastructure_attempt_index": 2})
    changed = final.model_copy(update={"identity": identity})
    with pytest.raises(AnytimeLedgerError, match="checkpoint"):
        _verify(loop, header, records, (checkpoints[0], changed))


def test_undeclared_extra_checkpoint_is_rejected() -> None:
    loop, header, records, checkpoints = _valid_history()
    record = records[1]
    extra = AnytimeCheckpointRecord(
        header_sha256=header.sha256,
        identity=AnytimeCheckpointIdentity(
            trajectory_id=header.trajectory_id,
            infrastructure_attempt_index=header.infrastructure_attempt_index,
            scientific_call_index=2,
            trajectory_execution_sha256=header.trajectory_execution_sha256,
            ledger_prefix_sha256=record.sha256,
            incumbent_candidate_sha256=record.incumbent_after.source_sha256,
            resource_snapshot_sha256=record.resource_snapshot.sha256,
        ),
        incumbent=record.incumbent_after,
        resource_snapshot=record.resource_snapshot,
    )
    with pytest.raises(AnytimeLedgerError, match="checkpoint set"):
        _verify(loop, header, records, (checkpoints[0], extra, checkpoints[1]))


def test_records_after_provider_terminal_are_rejected() -> None:
    loop, header, records, _ = _valid_history()
    prepared = prepare_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
    )
    result = append_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=(),
        checkpoints=(),
        provider=AnytimeProviderAmbiguousSubmission(
            logical_request_sha256=sha256_json(prepared.logical_request),
            dispatch_intent_sha256="a" * 64,
            provider_seconds=0.1,
        ),
        candidate=None,
        observed_wall_seconds=0.1,
    )
    assert result.checkpoint is not None
    with pytest.raises(AnytimeLedgerError, match="continues after"):
        _verify(
            loop,
            header,
            (result.record, records[1]),
            (result.checkpoint,),
        )


def test_per_call_output_overage_is_recorded_and_stops_before_total_cap() -> None:
    loop = _loop()
    header = _header(loop)
    provider = _provider(
        header,
        loop,
        (),
        (),
        output_tokens=loop.budget.max_total_output_tokens // loop.budget.max_scientific_calls + 1,
    )
    appended = append_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=(),
        checkpoints=(),
        provider=provider,
        candidate=AnytimeParseFailure(error="parse failed"),
        observed_wall_seconds=1.0,
    )
    assert appended.record.resource_snapshot.known_output_tokens < (
        loop.budget.max_total_output_tokens
    )
    assert appended.record.terminal_reason == "resource_cap"


def test_public_entrypoints_only_read_canonicalized_nested_models() -> None:
    loop, header, records, checkpoints = _valid_history()
    tail = records[2]
    assert tail.candidate is not None
    noncanonical_tail = tail.model_copy(
        update={"candidate": tail.candidate.model_dump(mode="python")}
    )
    noncanonical_records = (*records[:2], noncanonical_tail)

    with pytest.warns(UserWarning, match="Pydantic serializer warnings"):
        prepared = prepare_anytime_turn(
            header=header,
            loop=loop,
            base_prompt=_base_prompt(),
            records=noncanonical_records,
            checkpoints=checkpoints[:1],
        )
    assert prepared.scientific_call_index == 4

    provider = AnytimeProviderSuccess(
        logical_request_sha256=sha256_json(prepared.logical_request),
        response_artifact_sha256="c" * 64,
        usage=AnytimeTokenUsage(
            input_tokens=1,
            cached_input_tokens=0,
            output_tokens=1,
            reasoning_tokens=0,
        ),
        provider_seconds=0.1,
    )
    with pytest.warns(UserWarning, match="Pydantic serializer warnings"):
        appended = append_anytime_turn(
            header=header,
            loop=loop,
            base_prompt=_base_prompt(),
            records=noncanonical_records,
            checkpoints=checkpoints[:1],
            provider=provider,
            candidate=AnytimeParseFailure(error="parse failed"),
            observed_wall_seconds=0.1,
        )
    assert appended.record.scientific_call_index == 4

    with pytest.warns(UserWarning, match="Pydantic serializer warnings"):
        rebuilt = rebuild_anytime_checkpoints(
            header=header,
            loop=loop,
            base_prompt=_base_prompt(),
            records=noncanonical_records,
        )
    assert rebuilt == checkpoints[:1]


def test_consumed_elapsed_overage_is_preserved_as_terminal_fact() -> None:
    loop = _loop()
    header = _header(loop)
    prepared = prepare_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
    )
    provider = AnytimeProviderSuccess(
        logical_request_sha256=sha256_json(prepared.logical_request),
        response_artifact_sha256="d" * 64,
        usage=AnytimeTokenUsage(
            input_tokens=1,
            cached_input_tokens=0,
            output_tokens=1,
            reasoning_tokens=0,
        ),
        provider_seconds=loop.budget.max_provider_seconds_per_call + 0.25,
    )

    appended = append_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=(),
        checkpoints=(),
        provider=provider,
        candidate=AnytimeParseFailure(error="parse failed"),
        observed_wall_seconds=loop.budget.max_provider_seconds_per_call + 0.25,
    )

    assert appended.record.terminal_reason == "resource_cap"
    assert appended.record.resource_snapshot.provider_seconds > (
        loop.budget.max_provider_seconds_per_call
    )
    assert appended.checkpoint is not None


def test_exact_per_call_time_limits_do_not_exhaust_the_trajectory() -> None:
    loop = _loop()
    header = _header(loop)
    prepared = prepare_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
    )
    provider_at_limit = AnytimeProviderSuccess(
        logical_request_sha256=sha256_json(prepared.logical_request),
        response_artifact_sha256="e" * 64,
        usage=AnytimeTokenUsage(
            input_tokens=1,
            cached_input_tokens=0,
            output_tokens=1,
            reasoning_tokens=0,
        ),
        provider_seconds=loop.budget.max_provider_seconds_per_call,
    )
    first = append_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=(),
        checkpoints=(),
        provider=provider_at_limit,
        candidate=AnytimeParseFailure(error="parse failed"),
        observed_wall_seconds=loop.budget.max_provider_seconds_per_call,
    )
    assert first.record.terminal_reason is None
    assert first.checkpoint is not None

    records = (first.record,)
    checkpoints = (first.checkpoint,)
    provider = _provider(header, loop, records, checkpoints, output_tokens=1)
    second = append_anytime_turn(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=records,
        checkpoints=checkpoints,
        provider=provider,
        candidate=AnytimeCompileFailure(
            source=_source("compile-at-limit"),
            compile_seconds=loop.budget.max_candidate_seconds_per_call,
            error="compile failed at limit",
        ),
        observed_wall_seconds=(
            loop.budget.max_candidate_seconds_per_call + provider.provider_seconds
        ),
    )
    assert second.record.terminal_reason is None
    assert (
        prepare_anytime_turn(
            header=header,
            loop=loop,
            base_prompt=_base_prompt(),
            records=(*records, second.record),
            checkpoints=checkpoints,
        ).scientific_call_index
        == 3
    )


def test_rebuild_checkpoints_derives_only_reached_verified_prefixes() -> None:
    loop, header, records, checkpoints = _valid_history()
    rebuilt_partial = rebuild_anytime_checkpoints(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=records[:3],
    )
    rebuilt_full = rebuild_anytime_checkpoints(
        header=header,
        loop=loop,
        base_prompt=_base_prompt(),
        records=records,
    )
    assert rebuilt_partial == checkpoints[:1]
    assert rebuilt_full == checkpoints

    changed = records[1].model_copy(update={"previous_ledger_sha256": "e" * 64})
    with pytest.raises(AnytimeLedgerError):
        rebuild_anytime_checkpoints(
            header=header,
            loop=loop,
            base_prompt=_base_prompt(),
            records=(records[0], changed, *records[2:]),
        )


def test_model_copy_cannot_insert_invalid_resource_schema() -> None:
    loop, header, records, checkpoints = _valid_history()
    invalid_delta = records[0].resource_delta.model_copy(update={"scientific_calls_consumed": 2})
    assert isinstance(invalid_delta, AnytimeResourceDelta)
    changed = records[0].model_copy(update={"resource_delta": invalid_delta})
    with pytest.raises(ValidationError):
        _verify(loop, header, (changed, *records[1:]), checkpoints)


def test_signed_zero_cannot_create_an_alternate_ledger_hash() -> None:
    loop, header, records, checkpoints = _valid_history()
    record = records[0]
    checkpoint = checkpoints[0]
    negative_delta = record.resource_delta.model_copy(update={"compile_seconds": -0.0})
    negative_snapshot = record.resource_snapshot.model_copy(update={"compile_seconds": -0.0})
    negative_record = record.model_copy(
        update={
            "resource_delta": negative_delta,
            "resource_snapshot": negative_snapshot,
        }
    )
    assert negative_record.sha256 != record.sha256
    negative_identity = checkpoint.identity.model_copy(
        update={
            "ledger_prefix_sha256": negative_record.sha256,
            "resource_snapshot_sha256": negative_snapshot.sha256,
        }
    )
    negative_checkpoint = checkpoint.model_copy(
        update={
            "identity": negative_identity,
            "resource_snapshot": negative_snapshot,
        }
    )

    with pytest.raises((AnytimeLedgerError, ValidationError)):
        _verify(loop, header, (negative_record,), (negative_checkpoint,))

    normalized_delta = AnytimeResourceDelta.model_validate_json(negative_delta.model_dump_json())
    assert normalized_delta.compile_seconds == 0.0
    assert str(normalized_delta.compile_seconds) == "0.0"
