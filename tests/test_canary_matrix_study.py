from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from abstrak.canary.artifacts import TrajectoryStore
from abstrak.canary.contracts import (
    AgentBudget,
    AgentLoopPolicy,
    CaseResult,
    TimingSpec,
    TrajectoryOutcome,
    WorkerJob,
    WorkerResult,
)
from abstrak.canary.loop import CanaryAgentLoop
from abstrak.canary.manifests import PinnedStudySpec
from abstrak.canary.matrix import MatrixStudySpec, PhaseSpec, TaskGroupSpec
from abstrak.canary.matrix_study import (
    ExistingCellAttempt,
    MatrixAttemptManifest,
    MatrixAxisIdentity,
    MatrixCellArtifactIdentity,
    MatrixCellExecutionSpec,
    MatrixStudyPlanError,
    build_matrix_attempt_manifest,
    plan_matrix_phase,
    resolve_registered_target_identity,
    resolve_registered_task_identity,
    sealed_cell_identity_loader,
)
from abstrak.canary.protocol import build_initial_messages
from abstrak.canary.targets import get_target_stack, load_target_card
from abstrak.canary.tasks import get_task_pack
from abstrak.providers.contracts import (
    ChatMessage,
    LogicalRequest,
    MessageRole,
    NormalizedResponse,
    NormalizedUsage,
    sha256_json,
)

_TEST_MESSAGES = (
    ChatMessage(role=MessageRole.SYSTEM, content="frozen system prompt"),
    ChatMessage(role=MessageRole.USER, content="frozen task and target card"),
)
_TEST_MESSAGES_SHA256 = sha256_json([message.model_dump(mode="json") for message in _TEST_MESSAGES])


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _pinned(tmp_path: Path) -> PinnedStudySpec:
    spec = MatrixStudySpec(
        study_id="matrix-plan-test",
        seed=20260723,
        agents=("agent",),
        targets=("target-a", "target-b"),
        task_groups=(TaskGroupSpec(id="group", task_ids=("task",)),),
        phases=(
            PhaseSpec(
                id="core",
                task_ids=("task",),
                replicates=(1,),
                order_policy="fixed",
                max_calls_per_trajectory=3,
                infrastructure_retries=1,
            ),
        ),
    )
    return PinnedStudySpec(
        path=tmp_path / "study.json",
        sha256=_digest("raw-study-bytes"),
        spec=spec,
    )


def _axis(kind: str):
    def resolve(identifier: str) -> MatrixAxisIdentity:
        return MatrixAxisIdentity(
            kind=kind,
            id=identifier,
            sha256=_digest(f"{kind}:{identifier}"),
        )

    return resolve


def _execution(_cell) -> MatrixCellExecutionSpec:
    return MatrixCellExecutionSpec(
        budget=AgentBudget(max_calls=3),
        policy=AgentLoopPolicy(
            response_parser="candidate_only",
            stop_policy="correct_latency",
            final_selection="best_correct_latency",
            latency_ceiling_ms=1.25,
        ),
        dev_timing=TimingSpec(repetitions=1),
        model_ref="fake-model",
        initial_messages_sha256=_TEST_MESSAGES_SHA256,
        execution_context_sha256=_digest("floor-manifest"),
    )


def _plan(pinned: PinnedStudySpec, **kwargs):
    return plan_matrix_phase(
        pinned,
        "core",
        resolve_task=_axis("task"),
        resolve_target=_axis("target"),
        resolve_agent=_axis("agent"),
        resolve_execution=_execution,
        **kwargs,
    )


def _outcome(
    identity: MatrixCellArtifactIdentity,
    *,
    status: str = "no_candidate",
    calls: int = 1,
) -> TrajectoryOutcome:
    now = datetime.now(timezone.utc)
    return TrajectoryOutcome(
        trajectory_id=identity.artifact_trajectory_id,
        status=status,
        calls=calls,
        usage_complete=True,
        started_at_utc=now,
        finished_at_utc=now,
        error=(f"{status} for test" if status in {"provider_error", "worker_error"} else None),
    )


def _attempt(
    identity: MatrixCellArtifactIdentity,
    *,
    status: str = "no_candidate",
) -> ExistingCellAttempt:
    return ExistingCellAttempt(identity=identity, outcome=_outcome(identity, status=status))


def _attempt_manifest(
    identity: MatrixCellArtifactIdentity,
    *,
    execution: MatrixCellExecutionSpec | None = None,
    initial_messages: tuple[ChatMessage, ...] = _TEST_MESSAGES,
) -> MatrixAttemptManifest:
    frozen_execution = execution or _execution(identity.cell)
    return build_matrix_attempt_manifest(
        identity,
        agent=MatrixAxisIdentity(
            kind="agent",
            id=identity.cell.agent_id,
            sha256=identity.agent_sha256,
        ),
        execution=frozen_execution,
        initial_messages=initial_messages,
    )


def _logical_request(identity: MatrixCellArtifactIdentity, turn: int) -> dict[str, object]:
    return LogicalRequest(
        model_ref=identity.model_ref,
        messages=_TEST_MESSAGES,
        trajectory_id=identity.artifact_trajectory_id,
        turn_index=turn,
    ).model_dump(mode="json")


def _seal_attempt(
    root: Path,
    identity: MatrixCellArtifactIdentity,
    outcome: TrajectoryOutcome,
) -> None:
    store = TrajectoryStore.create(
        root,
        identity.study_id,
        identity.artifact_trajectory_id,
    )
    store.write_json("run-manifest.json", _attempt_manifest(identity))
    store.write_json("cell-identity.json", identity)
    sequence = 0
    store.append_event(
        sequence,
        "trajectory_started",
        None,
        {"trajectory_id": identity.artifact_trajectory_id},
    )
    sequence += 1
    for turn in range(outcome.calls):
        request = _logical_request(identity, turn)
        store.append_event(sequence, "request_started", turn, request)
        sequence += 1
        if outcome.status == "provider_error" and turn == outcome.calls - 1:
            provider_error = {"error": "provider unavailable"}
            store.write_turn(turn, request=request, error=provider_error)
            store.append_event(sequence, "provider_error", turn, provider_error)
        else:
            response = {"text": "invalid candidate response"}
            store.write_turn(turn, request=request, response=response)
            store.append_event(sequence, "response_received", turn, response)
        sequence += 1
    if outcome.status == "no_candidate":
        store.append_event(sequence, "call_limit", outcome.calls - 1, {"calls": outcome.calls})
        sequence += 1
    store.append_event(sequence, "trajectory_terminal", None, outcome)
    store.write_json("outcome.json", outcome)
    store.seal()


class _OneResponseClient:
    def complete(self, request: object) -> NormalizedResponse:
        now = datetime.now(timezone.utc)
        return NormalizedResponse(
            request_id=request.request_id,  # type: ignore[attr-defined]
            attempt_id="attempt-1",
            provider_request_id="provider-1",
            provider_id="fake-provider",
            model_id="fake-model",
            provider_manifest_sha256="1" * 64,
            model_manifest_sha256="2" * 64,
            requested_model="fake-model",
            returned_model="fake-model",
            text="```python\nclass ModelNew:\n    pass\n```\nFINISH\n",
            finish_reason="stop",
            provider_finish_reason="stop",
            usage=NormalizedUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                input_characters=100,
                output_characters=50,
                provider_reported=True,
                core_fields_complete=True,
            ),
            started_at_utc=now,
            finished_at_utc=now,
            elapsed_ms=1.0,
            logical_request_sha256="3" * 64,
            transport_request_sha256="4" * 64,
            transport_response_sha256="5" * 64,
            sanitized_transport_request={},
            raw_transport_response={},
        )


class _PassingWorker:
    def execute(self, job: WorkerJob) -> WorkerResult:
        cases = tuple(
            CaseResult(
                case_id=case_id,
                status="pass",
                correct=True,
                max_abs_error=0.0,
                max_rel_error=0.0,
                output_finite=True,
                inputs_unchanged=True,
            )
            for case_id in job.case_ids
        )
        return WorkerResult(
            job_id=job.job_id,
            job_sha256=job.sha256,
            input_sha256=job.input_sha256,
            candidate_sha256=job.candidate_sha256,
            status="completed",
            compiled=True,
            correct=True,
            cases=cases,
        )


def test_dry_run_is_pure_and_reports_scientific_and_operational_ceilings(
    tmp_path: Path,
) -> None:
    pinned = _pinned(tmp_path)

    plan = _plan(pinned)
    repeated = _plan(pinned)

    assert not pinned.path.exists()
    assert len(plan.cells) == 2
    assert {cell.state for cell in plan.cells} == {"pending"}
    assert plan.scientific_request_ceiling == 6
    assert plan.operational_request_ceiling == 12
    assert plan.summary.expected_cells == 2
    assert plan.summary.pending_cells == 2
    assert plan.summary.retry_pending_cells == 0
    assert plan.summary.resumed_cells == 0
    assert plan.summary.retry_exhausted_cells == 0
    assert plan.summary.pending_scientific_request_ceiling == 6
    assert plan.summary.pending_operational_request_ceiling == 12
    assert plan.cells[0].identity.raw_study_sha256 == pinned.sha256
    assert plan.cells[0].identity.spec_sha256 == pinned.spec.sha256
    assert plan.cells[0].identity.execution_context_sha256 == _digest("floor-manifest")
    assert plan.cells[0].identity.attempt_index == 0
    assert plan.cells[0].identity.artifact_trajectory_id == plan.cells[0].identity.trajectory_id
    assert plan.sha256 == repeated.sha256
    assert plan.cells[0].identity.sha256 == repeated.cells[0].identity.sha256


def test_valid_existing_identity_resumes_and_missing_identity_remains_pending(
    tmp_path: Path,
) -> None:
    pinned = _pinned(tmp_path)
    baseline = _plan(pinned)
    resumed_id = baseline.cells[0].identity.trajectory_id

    def load(expected: MatrixCellArtifactIdentity) -> ExistingCellAttempt | None:
        if expected.trajectory_id == resumed_id and expected.attempt_index == 0:
            return _attempt(expected)
        return None

    plan = _plan(pinned, load_existing_attempt=load)

    assert [cell.state for cell in plan.cells] == ["resumed", "pending"]
    assert plan.summary.resumed_cells == 1
    assert plan.summary.pending_cells == 1
    assert plan.summary.retry_pending_cells == 0
    assert plan.summary.pending_scientific_request_ceiling == 3
    assert plan.summary.pending_operational_request_ceiling == 6


def test_infrastructure_failure_selects_an_independent_retry_attempt(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)
    logical_id = _plan(pinned).cells[0].identity.trajectory_id

    def load(expected: MatrixCellArtifactIdentity) -> ExistingCellAttempt | None:
        if expected.trajectory_id == logical_id and expected.attempt_index == 0:
            return _attempt(expected, status="worker_error")
        return None

    plan = _plan(pinned, load_existing_attempt=load)

    assert [cell.state for cell in plan.cells] == ["retry_pending", "pending"]
    retry = plan.cells[0].identity
    assert retry.attempt_index == 1
    assert retry.artifact_trajectory_id == f"{logical_id}.infra-1"
    assert plan.summary.pending_cells == 1
    assert plan.summary.retry_pending_cells == 1
    assert plan.summary.pending_scientific_request_ceiling == 3
    assert plan.summary.pending_operational_request_ceiling == 9


@pytest.mark.parametrize(
    ("retry_status", "expected_state"),
    (("no_candidate", "resumed"), ("provider_error", "retry_exhausted")),
)
def test_terminal_retry_is_resumed_or_marked_exhausted(
    tmp_path: Path,
    retry_status: str,
    expected_state: str,
) -> None:
    pinned = _pinned(tmp_path)
    logical_id = _plan(pinned).cells[0].identity.trajectory_id

    def load(expected: MatrixCellArtifactIdentity) -> ExistingCellAttempt | None:
        if expected.trajectory_id != logical_id:
            return None
        status = "worker_error" if expected.attempt_index == 0 else retry_status
        return _attempt(expected, status=status)

    plan = _plan(pinned, load_existing_attempt=load)

    assert plan.cells[0].state == expected_state
    assert plan.cells[0].identity.attempt_index == 1
    assert plan.summary.pending_cells == 1
    assert plan.summary.retry_pending_cells == 0
    assert plan.summary.resumed_cells == (expected_state == "resumed")
    assert plan.summary.retry_exhausted_cells == (expected_state == "retry_exhausted")
    assert plan.summary.pending_scientific_request_ceiling == 3
    assert plan.summary.pending_operational_request_ceiling == 6


@pytest.mark.parametrize("first_exists", (False, True))
def test_orphan_or_unexpected_retry_artifacts_fail_closed(
    tmp_path: Path,
    first_exists: bool,
) -> None:
    pinned = _pinned(tmp_path)
    logical_id = _plan(pinned).cells[0].identity.trajectory_id

    def load(expected: MatrixCellArtifactIdentity) -> ExistingCellAttempt | None:
        if expected.trajectory_id != logical_id:
            return None
        if expected.attempt_index == 0:
            return _attempt(expected) if first_exists else None
        return _attempt(expected)

    message = "unexpected retry" if first_exists else "orphan retry"
    with pytest.raises(MatrixStudyPlanError, match=message):
        _plan(pinned, load_existing_attempt=load)


@pytest.mark.parametrize(
    "field",
    (
        "study_id",
        "raw_study_sha256",
        "spec_sha256",
        "schedule_sha256",
        "cell",
        "task_sha256",
        "target_sha256",
        "agent_sha256",
        "policy_sha256",
        "budget_sha256",
        "max_calls_per_trajectory",
        "dev_timing_sha256",
        "model_ref",
        "initial_messages_sha256",
        "device",
        "execution_context_sha256",
        "execution_sha256",
    ),
)
def test_existing_artifact_identity_drift_fails_closed(tmp_path: Path, field: str) -> None:
    pinned = _pinned(tmp_path)

    def load(expected: MatrixCellArtifactIdentity) -> ExistingCellAttempt:
        if field == "cell":
            changed: object = expected.cell.model_copy(
                update={"target_order_index": expected.cell.target_order_index + 1}
            )
        elif field == "max_calls_per_trajectory":
            changed = 4 if expected.max_calls_per_trajectory != 4 else 3
        elif field == "device":
            changed = "cuda:1"
        elif field == "model_ref":
            changed = "drift"
        elif field == "study_id":
            changed = "drift"
        else:
            changed = "f" * 64
        changed_identity = expected.model_copy(update={field: changed})
        return _attempt(changed_identity)

    with pytest.raises(MatrixStudyPlanError, match=rf"identity drift.*{field}"):
        _plan(pinned, load_existing_attempt=load)


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("phase_id", "drift"),
        ("trajectory_id", "drift"),
        ("artifact_trajectory_id", "drift"),
        ("attempt_index", 1),
    ),
)
def test_internally_inconsistent_artifact_identity_fails_closed(
    tmp_path: Path,
    field: str,
    changed: object,
) -> None:
    pinned = _pinned(tmp_path)

    def load(expected: MatrixCellArtifactIdentity) -> ExistingCellAttempt:
        changed_identity = expected.model_copy(update={field: changed})
        return _attempt(changed_identity)

    with pytest.raises(MatrixStudyPlanError, match="cannot verify existing artifact"):
        _plan(pinned, load_existing_attempt=load)


@pytest.mark.parametrize(
    "field",
    (
        "trajectory_id",
        "cell_identity_sha256",
        "agent",
        "policy",
        "budget",
        "dev_timing",
        "model_ref",
        "initial_messages",
        "device",
        "execution_context_sha256",
    ),
)
def test_attempt_manifest_runtime_identity_drift_fails_closed(
    tmp_path: Path,
    field: str,
) -> None:
    identity = _plan(_pinned(tmp_path)).cells[0].identity
    manifest = _attempt_manifest(identity)
    changes: dict[str, object] = {
        "trajectory_id": "drift",
        "cell_identity_sha256": "f" * 64,
        "agent": manifest.agent.model_copy(update={"sha256": "f" * 64}),
        "policy": AgentLoopPolicy(),
        "budget": AgentBudget(max_calls=2),
        "dev_timing": TimingSpec(repetitions=2),
        "model_ref": "drift",
        "initial_messages": (ChatMessage(role=MessageRole.USER, content="different prompt"),),
        "device": "cuda:1",
        "execution_context_sha256": "f" * 64,
    }
    changed = manifest.model_copy(update={field: changes[field]})

    with pytest.raises(ValueError, match="attempt manifest differs from cell identity"):
        changed.verify_for_identity(identity)


def test_unreadable_partial_or_tampered_artifact_fails_closed(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)

    def load(_expected: MatrixCellArtifactIdentity) -> ExistingCellAttempt:
        raise RuntimeError("checksum manifest is missing")

    with pytest.raises(MatrixStudyPlanError, match="cannot verify existing artifact"):
        _plan(pinned, load_existing_attempt=load)


def test_shared_sealed_artifact_loader_resumes_only_a_terminal_identity(
    tmp_path: Path,
) -> None:
    pinned = _pinned(tmp_path)
    baseline = _plan(pinned)
    identity = baseline.cells[0].identity
    _seal_attempt(tmp_path, identity, _outcome(identity, calls=3))

    plan = _plan(
        pinned,
        load_existing_attempt=sealed_cell_identity_loader(tmp_path),
    )

    assert [cell.state for cell in plan.cells] == ["resumed", "pending"]


def test_shared_loader_resumes_an_artifact_written_by_the_real_agent_loop(
    tmp_path: Path,
) -> None:
    spec = MatrixStudySpec(
        study_id="real-loop-resume",
        seed=20260723,
        agents=("fake-agent",),
        targets=("triton-a100",),
        task_groups=(TaskGroupSpec(id="group", task_ids=("row-reduction-scale",)),),
        phases=(
            PhaseSpec(
                id="core",
                task_ids=("row-reduction-scale",),
                replicates=(1,),
                order_policy="fixed",
                max_calls_per_trajectory=1,
                infrastructure_retries=1,
            ),
        ),
    )
    pinned = PinnedStudySpec(
        path=tmp_path / "study.json",
        sha256=_digest("real-loop-study"),
        spec=spec,
    )
    task = get_task_pack("row-reduction-scale")
    target = get_target_stack("triton-a100")
    initial_messages = build_initial_messages(task, load_target_card(target.id))
    execution = MatrixCellExecutionSpec(
        budget=AgentBudget(max_calls=1),
        policy=AgentLoopPolicy(),
        model_ref="fake-model",
        initial_messages_sha256=sha256_json(
            [message.model_dump(mode="json") for message in initial_messages]
        ),
        execution_context_sha256=_digest("real-loop-context"),
    )

    def make_plan(*, resume: bool):
        return plan_matrix_phase(
            pinned,
            "core",
            resolve_task=resolve_registered_task_identity,
            resolve_target=resolve_registered_target_identity,
            resolve_agent=_axis("agent"),
            resolve_execution=lambda _cell: execution,
            load_existing_attempt=(sealed_cell_identity_loader(tmp_path) if resume else None),
        )

    identity = make_plan(resume=False).cells[0].identity
    store = TrajectoryStore.create(
        tmp_path,
        identity.study_id,
        identity.artifact_trajectory_id,
    )
    store.write_json(
        "run-manifest.json",
        _attempt_manifest(
            identity,
            execution=execution,
            initial_messages=initial_messages,
        ),
    )
    store.write_json("cell-identity.json", identity)
    outcome = CanaryAgentLoop(
        client=_OneResponseClient(),
        worker=_PassingWorker(),
        store=store,
    ).run(
        trajectory_id=identity.artifact_trajectory_id,
        model_ref="fake-model",
        initial_messages=initial_messages,
        task=task,
        target=target,
        budget=execution.budget,
        dev_timing=None,
        policy=execution.policy,
    )

    resumed = make_plan(resume=True)

    assert outcome.status == "finished"
    assert resumed.cells[0].state == "resumed"
    assert resumed.summary.pending_scientific_request_ceiling == 0
    assert resumed.summary.pending_operational_request_ceiling == 0


def test_shared_loader_routes_a_sealed_infrastructure_failure_to_retry(
    tmp_path: Path,
) -> None:
    pinned = _pinned(tmp_path)
    identity = _plan(pinned).cells[0].identity
    _seal_attempt(tmp_path, identity, _outcome(identity, status="provider_error"))

    plan = _plan(
        pinned,
        load_existing_attempt=sealed_cell_identity_loader(tmp_path),
    )

    assert [cell.state for cell in plan.cells] == ["retry_pending", "pending"]
    assert plan.cells[0].identity.artifact_trajectory_id == f"{identity.trajectory_id}.infra-1"


@pytest.mark.parametrize(
    ("defect", "message"),
    (
        ("missing_request", "request events differ"),
        ("missing_turn", "turn ledger is incomplete"),
        ("terminal_payload", "terminal event payload differs"),
    ),
)
def test_shared_loader_rejects_checksummed_but_semantically_invalid_artifacts(
    tmp_path: Path,
    defect: str,
    message: str,
) -> None:
    pinned = _pinned(tmp_path)
    identity = _plan(pinned).cells[0].identity
    outcome = _outcome(identity, calls=1)
    store = TrajectoryStore.create(
        tmp_path,
        identity.study_id,
        identity.artifact_trajectory_id,
    )
    store.write_json("run-manifest.json", _attempt_manifest(identity))
    store.write_json("cell-identity.json", identity)
    sequence = 0
    store.append_event(
        sequence,
        "trajectory_started",
        None,
        {"trajectory_id": identity.artifact_trajectory_id},
    )
    sequence += 1
    if defect != "missing_request":
        request = _logical_request(identity, 0)
        store.append_event(sequence, "request_started", 0, request)
        sequence += 1
    if defect == "terminal_payload":
        response = {"text": "invalid"}
        store.write_turn(0, request=request, response=response)
        store.append_event(sequence, "response_received", 0, response)
        sequence += 1
    terminal = _outcome(identity, calls=0) if defect == "terminal_payload" else outcome
    store.append_event(sequence, "trajectory_terminal", None, terminal)
    store.write_json("outcome.json", outcome)
    store.seal()

    with pytest.raises(MatrixStudyPlanError, match=message):
        _plan(
            pinned,
            load_existing_attempt=sealed_cell_identity_loader(tmp_path),
        )


def test_shared_loader_rejects_a_candidate_less_finished_outcome(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)
    identity = _plan(pinned).cells[0].identity
    outcome = _outcome(identity, status="finished", calls=1)
    store = TrajectoryStore.create(
        tmp_path,
        identity.study_id,
        identity.artifact_trajectory_id,
    )
    store.write_json("run-manifest.json", _attempt_manifest(identity))
    store.write_json("cell-identity.json", identity)
    store.append_event(
        0,
        "trajectory_started",
        None,
        {"trajectory_id": identity.artifact_trajectory_id},
    )
    request = _logical_request(identity, 0)
    response = {"text": "no candidate"}
    store.append_event(1, "request_started", 0, request)
    store.append_event(2, "response_received", 0, response)
    store.write_turn(0, request=request, response=response)
    store.append_event(3, "agent_finished", 0, {"candidate_sha256": "f" * 64})
    store.append_event(4, "trajectory_terminal", None, outcome)
    store.write_json("outcome.json", outcome)
    store.seal()

    with pytest.raises(MatrixStudyPlanError, match="finished trajectory has no candidate"):
        _plan(
            pinned,
            load_existing_attempt=sealed_cell_identity_loader(tmp_path),
        )


def test_shared_artifact_loader_rejects_partial_directories(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)
    identity = _plan(pinned).cells[0].identity
    directory = tmp_path / identity.study_id / identity.artifact_trajectory_id
    directory.mkdir(parents=True)
    (directory / "cell-identity.json").write_text("{}", encoding="utf-8")

    with pytest.raises(MatrixStudyPlanError, match="checksum manifest is missing"):
        _plan(
            pinned,
            load_existing_attempt=sealed_cell_identity_loader(tmp_path),
        )


def test_untyped_artifact_identity_fails_closed(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)

    with pytest.raises(MatrixStudyPlanError, match="loader returned dict"):
        _plan(
            pinned,
            load_existing_attempt=lambda expected: expected.model_dump(mode="json"),
        )


@pytest.mark.parametrize("missing_kind", ("task", "target", "agent"))
def test_missing_registry_resolution_fails_before_artifact_loading(
    tmp_path: Path,
    missing_kind: str,
) -> None:
    pinned = _pinned(tmp_path)
    loader_called = False

    def missing(identifier: str) -> MatrixAxisIdentity:
        raise KeyError(identifier)

    def load(_expected: MatrixCellArtifactIdentity) -> None:
        nonlocal loader_called
        loader_called = True
        return None

    resolvers = {
        "task": _axis("task"),
        "target": _axis("target"),
        "agent": _axis("agent"),
    }
    resolvers[missing_kind] = missing

    with pytest.raises(MatrixStudyPlanError, match=rf"cannot resolve {missing_kind}"):
        plan_matrix_phase(
            pinned,
            "core",
            resolve_task=resolvers["task"],
            resolve_target=resolvers["target"],
            resolve_agent=resolvers["agent"],
            resolve_execution=_execution,
            load_existing_attempt=load,
        )
    assert not loader_called


def test_resolver_identity_and_phase_budget_mismatches_fail_closed(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)

    with pytest.raises(MatrixStudyPlanError, match="task resolver identity mismatch"):
        plan_matrix_phase(
            pinned,
            "core",
            resolve_task=lambda _identifier: MatrixAxisIdentity(
                kind="task",
                id="other-task",
                sha256=_digest("other"),
            ),
            resolve_target=_axis("target"),
            resolve_agent=_axis("agent"),
            resolve_execution=_execution,
        )

    def wrong_budget(_cell) -> MatrixCellExecutionSpec:
        return MatrixCellExecutionSpec(
            budget=AgentBudget(max_calls=2),
            policy=AgentLoopPolicy(),
            model_ref="fake-model",
            initial_messages_sha256=_TEST_MESSAGES_SHA256,
            execution_context_sha256=_digest("execution-context"),
        )

    with pytest.raises(MatrixStudyPlanError, match="phase requires 3"):
        plan_matrix_phase(
            pinned,
            "core",
            resolve_task=_axis("task"),
            resolve_target=_axis("target"),
            resolve_agent=_axis("agent"),
            resolve_execution=wrong_budget,
        )


def test_shared_registry_resolvers_hash_verified_contracts() -> None:
    task = resolve_registered_task_identity("row-reduction-scale")
    target = resolve_registered_target_identity("triton-a100")

    assert task.kind == "task"
    assert task.id == "row-reduction-scale"
    assert target.kind == "target"
    assert target.id == "triton-a100"
