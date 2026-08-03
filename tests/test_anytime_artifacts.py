from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from abstrak.anytime.artifacts import (
    AnytimeArtifactError,
    AnytimeAttemptIdentity,
    AnytimeAttemptManifest,
    AnytimeAttemptWriter,
    AnytimeInjectedCrash,
    AnytimeWorkerEvaluationArtifact,
    artifact_payload_sha256,
    audit_anytime_attempt,
    canonical_artifact_bytes,
    create_anytime_retry,
    recover_anytime_attempt,
)
from abstrak.anytime.contracts import (
    AnytimeCheckpointPolicy,
    AnytimeLoopPolicy,
    AnytimeResourceBudget,
)
from abstrak.anytime.ledger import (
    AnytimeCheckpointRecord,
    AnytimeLedgerHeader,
    AnytimeParseFailure,
    build_anytime_ledger_header,
    prepare_anytime_turn,
    rebuild_anytime_checkpoints,
)
from abstrak.anytime.resume import (
    AnytimePhaseJournal,
    AnytimePhaseJournalError,
    AnytimeResumeIndex,
    audit_anytime_phase,
    refresh_anytime_resume_index,
)
from abstrak.providers.contracts import (
    ChatMessage,
    ErrorCategory,
    MessageRole,
    NormalizedUsage,
    sha256_json,
)
from abstrak.providers.native_contracts import (
    NativeNormalizedError,
    NativeNormalizedResponse,
    NativeReasoningRecord,
)


def _loop() -> AnytimeLoopPolicy:
    return AnytimeLoopPolicy(
        budget=AnytimeResourceBudget(
            max_scientific_calls=4,
            max_total_output_tokens=1200,
            max_compile_attempts=4,
            max_evaluation_attempts=4,
            max_gpu_seconds=40.0,
            max_provider_seconds_per_call=10.0,
            max_candidate_seconds_per_call=10.0,
            max_trajectory_wall_seconds=1000.0,
        ),
        checkpoints=AnytimeCheckpointPolicy(calls=(1, 4)),
    )


def _base_prompt() -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(role=MessageRole.SYSTEM, content="Generate one candidate."),
        ChatMessage(role=MessageRole.USER, content="TASK\nreduce\nTARGET\nTriton"),
    )


def _ledger_header(attempt: int) -> AnytimeLedgerHeader:
    return build_anytime_ledger_header(
        trajectory_id="trajectory-1",
        infrastructure_attempt_index=attempt,
        trajectory_execution_sha256="1" * 64,
        agent_binding_sha256="2" * 64,
        task_sha256="3" * 64,
        target_sha256="4" * 64,
        environment_sha256="5" * 64,
        model_ref="gpt-5.6-luna",
        loop=_loop(),
        base_prompt=_base_prompt(),
        local_trajectory_seed=17,
    )


def _identity(attempt: int = 1, prior: str | None = None) -> AnytimeAttemptIdentity:
    return AnytimeAttemptIdentity(
        study_id="study-1",
        phase_id="core",
        trajectory_id="trajectory-1",
        infrastructure_attempt_index=attempt,
        trajectory_execution_sha256="1" * 64,
        prior_attempt_manifest_sha256=prior,
    )


def _reasoning() -> NativeReasoningRecord:
    return NativeReasoningRecord(
        submitted_parameter="reasoning",
        submitted_value={"effort": "xhigh"},
        effective_mode="literal_xhigh",
        fidelity="literal",
        evidence="offline fixture preserves literal xhigh",
    )


def _usage(*, output_tokens: int = 10) -> NormalizedUsage:
    return NormalizedUsage(
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=output_tokens,
        reasoning_tokens=5,
        total_tokens=105 + output_tokens,
        input_characters=1000,
        output_characters=20,
        provider_reported=True,
        core_fields_complete=True,
        raw_usage={"input_tokens": 100, "output_tokens": output_tokens},
    )


def _native_response(prepared, *, output_tokens: int = 10) -> NativeNormalizedResponse:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    raw = {"id": f"response-{prepared.scientific_call_index}", "status": "completed"}
    sanitized = {"model": "gpt-5.6-luna", "store": False}
    return NativeNormalizedResponse(
        request_id=prepared.logical_request.request_id,
        attempt_id=f"native-{prepared.scientific_call_index}",
        provider_request_id=f"provider-{prepared.scientific_call_index}",
        provider_id="openai-native",
        model_id="gpt-5.6-luna",
        protocol="responses",
        provider_manifest_sha256="6" * 64,
        model_manifest_sha256="7" * 64,
        requested_model="gpt-5.6-luna",
        returned_model="gpt-5.6-luna",
        text="```python\nclass ModelNew: pass\n```",
        finish_reason="completed",
        provider_finish_reason="completed",
        usage=_usage(output_tokens=output_tokens),
        resource_usage_complete=True,
        reasoning=_reasoning(),
        started_at_utc=started,
        finished_at_utc=started + timedelta(milliseconds=10),
        elapsed_ms=10.0,
        logical_request_sha256=sha256_json(prepared.logical_request),
        transport_request_sha256=sha256_json(sanitized),
        transport_response_sha256=sha256_json(raw),
        sanitized_transport_request=sanitized,
        raw_transport_response=raw,
    )


def _native_error(prepared, *, submitted: bool) -> NativeNormalizedError:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return NativeNormalizedError(
        request_id=prepared.logical_request.request_id,
        attempt_id="native-error",
        provider_id="openai-native",
        model_id="gpt-5.6-luna",
        protocol="responses",
        category=ErrorCategory.NETWORK if submitted else ErrorCategory.INVALID_REQUEST,
        provider_type="OfflineFixtureError",
        sanitized_message="offline fixture failure",
        retryable=False,
        request_submitted=submitted,
        possibly_charged=submitted,
        partial_usage=None,
        reasoning=_reasoning(),
        started_at_utc=started,
        failed_at_utc=started + timedelta(milliseconds=10),
        elapsed_ms=10.0,
        logical_request_sha256=sha256_json(prepared.logical_request),
        sanitized_transport_request={"model": "gpt-5.6-luna"},
    )


def _writer(
    root: Path,
    *,
    fault=None,
) -> AnytimeAttemptWriter:
    return AnytimeAttemptWriter.create(
        root,
        identity=_identity(),
        ledger_header=_ledger_header(1),
        fault=fault,
    )


def _append_success_turn(
    writer: AnytimeAttemptWriter,
    records: tuple,
    checkpoints: tuple[AnytimeCheckpointRecord, ...],
) -> tuple[tuple, tuple[AnytimeCheckpointRecord, ...]]:
    prepared = prepare_anytime_turn(
        header=writer.header.ledger_header,
        loop=_loop(),
        base_prompt=_base_prompt(),
        records=records,
        checkpoints=checkpoints,
    )
    writer.persist_dispatch_intent(prepared)
    provider = writer.persist_native_provider_response(
        scientific_call_index=prepared.scientific_call_index,
        response=_native_response(prepared),
        observed_wall_seconds=0.02,
    )
    writer.persist_worker_evaluation(
        scientific_call_index=prepared.scientific_call_index,
        worker_artifact=AnytimeWorkerEvaluationArtifact(
            provider_observation_sha256=provider.sha256,
            evaluator_id="offline-fixture-v1",
            evaluator_execution_sha256="9" * 64,
            candidate=AnytimeParseFailure(error=f"parse-{prepared.scientific_call_index}"),
            observed_wall_seconds=0.02,
        ),
    )
    record = writer.persist_derived_turn(
        loop=_loop(),
        base_prompt=_base_prompt(),
        scientific_call_index=prepared.scientific_call_index,
    )
    records = (*records, record)
    checkpoints = rebuild_anytime_checkpoints(
        header=writer.header.ledger_header,
        loop=_loop(),
        base_prompt=_base_prompt(),
        records=records,
    )
    return records, checkpoints


def _complete_attempt(writer: AnytimeAttemptWriter):
    records: tuple = ()
    checkpoints: tuple[AnytimeCheckpointRecord, ...] = ()
    for _ in range(4):
        records, checkpoints = _append_success_turn(writer, records, checkpoints)
    writer.persist_terminal(
        loop=_loop(),
        base_prompt=_base_prompt(),
        terminal_kind="success",
        reason="call_budget_complete",
    )
    return writer.seal_and_promote(loop=_loop(), base_prompt=_base_prompt())


def _crash_at(target):
    def inject(point):
        if point == target:
            raise AnytimeInjectedCrash(point)

    return inject


def test_completed_attempt_is_exclusive_sealed_and_freshly_audited(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    audit = _complete_attempt(writer)

    assert not writer.staging.exists()
    assert writer.final == Path(audit.directory)
    assert audit.terminal.terminal_kind == "success"
    assert audit.ledger_records == 4
    assert audit.terminal.resource_snapshot.known_output_tokens == 40
    assert (
        audit_anytime_attempt(
            writer.final,
            loop=_loop(),
            base_prompt=_base_prompt(),
        )
        == audit
    )
    with pytest.raises(AnytimeArtifactError, match="sealed attempt"):
        _writer(tmp_path)


def test_crash_before_request_seals_unsubmitted_failure_and_allows_one_retry(
    tmp_path: Path,
) -> None:
    with pytest.raises(AnytimeInjectedCrash):
        _writer(tmp_path, fault=_crash_at("after_attempt_create"))

    recovered = recover_anytime_attempt(
        tmp_path,
        _identity(),
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    assert recovered.action == "finalized"
    assert recovered.request_submitted is False
    primary = audit_anytime_attempt(
        recovered.directory,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    assert primary.terminal.terminal_kind == "infrastructure_failure"
    assert primary.terminal.retry_eligible is True

    retry = create_anytime_retry(
        tmp_path,
        _identity(),
        ledger_header=_ledger_header(2),
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    assert retry.header.identity.infrastructure_attempt_index == 2
    assert retry.header.identity.prior_attempt_manifest_sha256 == primary.manifest_file_sha256
    assert retry.staging.parent != Path(primary.directory).parent / "attempt-01.incomplete"
    with pytest.raises(AnytimeArtifactError, match="primary attempt"):
        create_anytime_retry(
            tmp_path,
            retry.header.identity,
            ledger_header=_ledger_header(2),
            loop=_loop(),
            base_prompt=_base_prompt(),
        )


def test_response_publish_crash_becomes_ambiguous_and_is_never_replayed(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path, fault=_crash_at("provider_terminal_before_publish"))
    prepared = prepare_anytime_turn(
        header=writer.header.ledger_header,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    writer.persist_dispatch_intent(prepared)
    with pytest.raises(AnytimeInjectedCrash):
        writer.persist_native_provider_response(
            scientific_call_index=1,
            response=_native_response(prepared),
        )

    decision = recover_anytime_attempt(
        tmp_path,
        _identity(),
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    assert decision.action == "finalized"
    assert decision.provider_replay_allowed is False
    assert decision.request_submitted is True
    assert decision.possibly_charged is True
    audit = audit_anytime_attempt(
        decision.directory,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    assert audit.terminal.reason == "ambiguous_submission"
    assert audit.terminal.resource_snapshot.possibly_charged_requests == 1


def test_native_unsubmitted_error_consumes_no_call_and_authorizes_retry(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    prepared = prepare_anytime_turn(
        header=writer.header.ledger_header,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    writer.persist_dispatch_intent(prepared)
    assert (
        writer.persist_native_provider_error(
            scientific_call_index=1,
            error=_native_error(prepared, submitted=False),
        )
        is None
    )

    decision = recover_anytime_attempt(
        tmp_path,
        _identity(),
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    audit = audit_anytime_attempt(
        decision.directory,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    assert audit.ledger_records == 0
    assert audit.terminal.retry_eligible is True
    assert audit.terminal.resource_snapshot.provider_requests_submitted == 0


def test_native_submitted_error_is_derived_and_scientifically_terminal(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    prepared = prepare_anytime_turn(
        header=writer.header.ledger_header,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    writer.persist_dispatch_intent(prepared)
    persisted = writer.persist_native_provider_error(
        scientific_call_index=1,
        error=_native_error(prepared, submitted=True),
    )
    assert persisted is not None
    assert persisted.observation.category == ErrorCategory.NETWORK
    assert persisted.observation.possibly_charged is True

    decision = recover_anytime_attempt(
        tmp_path,
        _identity(),
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    audit = audit_anytime_attempt(
        decision.directory,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    assert audit.terminal.terminal_kind == "scientific_failure"
    assert audit.terminal.reason == "provider_submitted_error"
    assert audit.terminal.retry_eligible is False


def test_controller_failure_has_immutable_nonretryable_tombstone(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    terminal = writer.persist_terminal(
        loop=_loop(),
        base_prompt=_base_prompt(),
        terminal_kind="controller_failure",
        reason="controller_invariant_failed",
    )
    assert terminal.retry_eligible is False
    audit = writer.seal_and_promote(loop=_loop(), base_prompt=_base_prompt())
    assert audit.terminal.terminal_kind == "controller_failure"


def test_worker_crash_resumes_only_local_evaluation(tmp_path: Path) -> None:
    writer = _writer(tmp_path, fault=_crash_at("worker_evaluation_before_publish"))
    prepared = prepare_anytime_turn(
        header=writer.header.ledger_header,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    writer.persist_dispatch_intent(prepared)
    provider = writer.persist_native_provider_response(
        scientific_call_index=1,
        response=_native_response(prepared),
    )
    worker_artifact = AnytimeWorkerEvaluationArtifact(
        provider_observation_sha256=provider.sha256,
        evaluator_id="offline-fixture-v1",
        evaluator_execution_sha256="9" * 64,
        candidate=AnytimeParseFailure(error="parse-1"),
        observed_wall_seconds=0.02,
    )
    with pytest.raises(AnytimeInjectedCrash):
        writer.persist_worker_evaluation(
            scientific_call_index=1,
            worker_artifact=worker_artifact,
        )

    decision = recover_anytime_attempt(
        tmp_path,
        _identity(),
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    assert decision.action == "resume_worker_evaluation"
    assert decision.provider_replay_allowed is False
    assert decision.scientific_call_index == 1

    resumed = AnytimeAttemptWriter(tmp_path, writer.header, create=False)
    resumed.persist_worker_evaluation(
        scientific_call_index=1,
        worker_artifact=worker_artifact,
    )
    resumed.persist_derived_turn(
        loop=_loop(),
        base_prompt=_base_prompt(),
        scientific_call_index=1,
    )
    finalized = recover_anytime_attempt(
        tmp_path,
        _identity(),
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    assert finalized.action == "finalized"
    assert not writer.staging.exists()


def test_worker_fact_must_bind_the_persisted_provider_observation(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    prepared = prepare_anytime_turn(
        header=writer.header.ledger_header,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    writer.persist_dispatch_intent(prepared)
    writer.persist_native_provider_response(
        scientific_call_index=1,
        response=_native_response(prepared),
    )
    worker = AnytimeWorkerEvaluationArtifact(
        provider_observation_sha256="0" * 64,
        evaluator_id="offline-fixture-v1",
        evaluator_execution_sha256="9" * 64,
        candidate=AnytimeParseFailure(error="parse-1"),
        observed_wall_seconds=0.02,
    )

    with pytest.raises(AnytimeArtifactError, match="another provider observation"):
        writer.persist_worker_evaluation(
            scientific_call_index=1,
            worker_artifact=worker,
        )
    assert not (writer.staging / "turns" / "0001" / "worker-result.json").exists()


def test_checkpoint_crash_is_rebuilt_then_attempt_finalizes_without_cleanup(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path, fault=_crash_at("checkpoint_before_publish"))
    prepared = prepare_anytime_turn(
        header=writer.header.ledger_header,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    writer.persist_dispatch_intent(prepared)
    provider = writer.persist_native_provider_response(
        scientific_call_index=1,
        response=_native_response(prepared),
    )
    writer.persist_worker_evaluation(
        scientific_call_index=1,
        worker_artifact=AnytimeWorkerEvaluationArtifact(
            provider_observation_sha256=provider.sha256,
            evaluator_id="offline-fixture-v1",
            evaluator_execution_sha256="9" * 64,
            candidate=AnytimeParseFailure(error="parse-1"),
            observed_wall_seconds=0.02,
        ),
    )
    with pytest.raises(AnytimeInjectedCrash):
        writer.persist_derived_turn(
            loop=_loop(),
            base_prompt=_base_prompt(),
            scientific_call_index=1,
        )

    decision = recover_anytime_attempt(
        tmp_path,
        _identity(),
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    assert decision.action == "finalized"
    final = Path(decision.directory)
    assert (final / "checkpoints" / "0001.json").is_file()
    assert not writer.staging.exists()


@pytest.mark.parametrize(
    "fault_point",
    ("seal_before_manifest_publish", "after_manifest_publish", "after_atomic_promotion"),
)
def test_seal_and_promotion_faults_resume_to_same_final_directory(
    tmp_path: Path,
    fault_point: str,
) -> None:
    writer = _writer(tmp_path, fault=_crash_at(fault_point))
    records: tuple = ()
    checkpoints: tuple[AnytimeCheckpointRecord, ...] = ()
    for _ in range(4):
        records, checkpoints = _append_success_turn(writer, records, checkpoints)
    writer.persist_terminal(
        loop=_loop(),
        base_prompt=_base_prompt(),
        terminal_kind="success",
        reason="call_budget_complete",
    )
    with pytest.raises(AnytimeInjectedCrash):
        writer.seal_and_promote(loop=_loop(), base_prompt=_base_prompt())

    decision = recover_anytime_attempt(
        tmp_path,
        _identity(),
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    assert decision.action in {"finalized", "complete"}
    assert Path(decision.directory) == writer.final
    assert not writer.staging.exists()
    audit_anytime_attempt(writer.final, loop=_loop(), base_prompt=_base_prompt())


def _refresh_manifest(directory: Path) -> None:
    manifest_path = directory / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    for item in payload["files"]:
        path = directory / item["path"]
        content = path.read_bytes()
        item["sha256"] = hashlib.sha256(content).hexdigest()
        item["size_bytes"] = len(content)
    refreshed = AnytimeAttemptManifest.model_validate_json(json.dumps(payload))
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(canonical_artifact_bytes(refreshed))


@pytest.mark.parametrize(
    "tamper",
    ("usage", "observation_hash", "candidate", "worker_candidate"),
)
def test_audit_rederives_provider_and_evaluator_facts_after_checksum_refresh(
    tmp_path: Path,
    tamper: str,
) -> None:
    writer = _writer(tmp_path)
    _complete_attempt(writer)
    turn = writer.final / "turns" / "0001"
    if tamper == "usage":
        path = turn / "provider-native-response.json"
        payload = json.loads(path.read_text())
        payload["usage"]["output_tokens"] = 11
        payload["usage"]["total_tokens"] = 116
    elif tamper == "observation_hash":
        path = turn / "provider-observation.json"
        payload = json.loads(path.read_text())
        payload["observation"]["response_artifact_sha256"] = "0" * 64
    elif tamper == "candidate":
        path = turn / "evaluation.json"
        payload = json.loads(path.read_text())
        payload["candidate"]["error"] = "swapped evaluator result"
    else:
        path = turn / "worker-result.json"
        payload = json.loads(path.read_text())
        payload["candidate"]["error"] = "swapped worker result"
    path.chmod(0o600)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _refresh_manifest(writer.final)

    with pytest.raises(AnytimeArtifactError):
        audit_anytime_attempt(writer.final, loop=_loop(), base_prompt=_base_prompt())


def _source_fingerprints(root: Path) -> dict[str, tuple[str, int]]:
    source = root / "source"
    return {
        path.relative_to(source).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mode,
        )
        for path in source.rglob("*")
        if path.is_file()
    }


def test_phase_journal_closes_with_full_audit_and_derived_index_is_disposable(
    tmp_path: Path,
) -> None:
    journal = AnytimePhaseJournal.create(
        tmp_path,
        study_id="study-1",
        phase_id="core",
        phase_execution_sha256="8" * 64,
        expected_trajectory_ids=("trajectory-1",),
        max_attempts_per_trajectory=2,
    )
    writer = _writer(tmp_path)
    attempt = _complete_attempt(writer)
    journal.append_attempt(attempt.identity, loop=_loop(), base_prompt=_base_prompt())
    source_before = _source_fingerprints(tmp_path)

    index_path = refresh_anytime_resume_index(
        tmp_path,
        journal.header,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    index = AnytimeResumeIndex.model_validate_json(index_path.read_bytes())
    assert index.trajectories[0].state == "success"
    assert "/derived/" in str(index_path)
    assert _source_fingerprints(tmp_path) == source_before

    closed = journal.close(loop=_loop(), base_prompt=_base_prompt())
    assert closed.close_audit is not None
    assert closed.close_audit.attempt_count == 1
    assert closed.close_audit.operational_totals.known_output_tokens == 40
    audit = audit_anytime_phase(
        tmp_path,
        journal.header,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    assert audit.closed is True
    with pytest.raises(AnytimePhaseJournalError, match="after phase close"):
        journal.append_attempt(attempt.identity, loop=_loop(), base_prompt=_base_prompt())


def test_phase_journal_rejects_out_of_population_before_writing_entry(
    tmp_path: Path,
) -> None:
    journal = AnytimePhaseJournal.create(
        tmp_path,
        study_id="study-1",
        phase_id="core",
        phase_execution_sha256="8" * 64,
        expected_trajectory_ids=("trajectory-1",),
        max_attempts_per_trajectory=2,
    )
    outside = AnytimeAttemptIdentity(
        study_id="study-1",
        phase_id="core",
        trajectory_id="trajectory-outside",
        infrastructure_attempt_index=1,
        trajectory_execution_sha256="1" * 64,
    )

    with pytest.raises(AnytimePhaseJournalError, match="expected trajectory"):
        journal.append_attempt(outside, loop=_loop(), base_prompt=_base_prompt())

    assert {path.name for path in journal.directory.iterdir()} == {"header.json"}


def test_phase_close_requires_bounded_retry_and_counts_both_attempts(tmp_path: Path) -> None:
    journal = AnytimePhaseJournal.create(
        tmp_path,
        study_id="study-1",
        phase_id="core",
        phase_execution_sha256="8" * 64,
        expected_trajectory_ids=("trajectory-1",),
        max_attempts_per_trajectory=2,
    )
    primary_writer = _writer(tmp_path)
    recover_anytime_attempt(
        tmp_path,
        primary_writer.header.identity,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    primary = audit_anytime_attempt(
        primary_writer.final,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    journal.append_attempt(primary.identity, loop=_loop(), base_prompt=_base_prompt())
    with pytest.raises(AnytimePhaseJournalError, match="bounded retry"):
        journal.close(loop=_loop(), base_prompt=_base_prompt())

    retry_writer = create_anytime_retry(
        tmp_path,
        primary.identity,
        ledger_header=_ledger_header(2),
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    recover_anytime_attempt(
        tmp_path,
        retry_writer.header.identity,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    retry = audit_anytime_attempt(
        retry_writer.final,
        loop=_loop(),
        base_prompt=_base_prompt(),
    )
    journal.append_attempt(retry.identity, loop=_loop(), base_prompt=_base_prompt())
    closed = journal.close(loop=_loop(), base_prompt=_base_prompt())
    assert closed.close_audit is not None
    assert closed.close_audit.operational_totals.attempts == 2
    assert closed.close_audit.terminal_counts == (("infrastructure_failure", 2),)


def test_response_artifact_helper_matches_exact_persisted_bytes() -> None:
    payload = {"unicode": "你好", "nested": [1, 2, 3]}
    assert (
        artifact_payload_sha256(payload)
        == hashlib.sha256(canonical_artifact_bytes(payload)).hexdigest()
    )
