from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

import abstrak.canary.matrix_runner as matrix_runner
from abstrak.canary.contracts import (
    AgentBudget,
    AgentLoopPolicy,
    CaseResult,
    TimingSpec,
    WorkerJob,
    WorkerResult,
)
from abstrak.canary.manifests import PinnedStudySpec
from abstrak.canary.matrix import MatrixStudySpec, PhaseSpec, TaskGroupSpec, build_matrix_schedule
from abstrak.canary.matrix_runner import (
    MatrixAgentBinding,
    MatrixAttemptRuntime,
    MatrixExecutionContext,
    MatrixStudyRunError,
    MatrixTransportContext,
    build_matrix_phase_contract,
    run_matrix_phase,
)
from abstrak.canary.matrix_study import (
    MatrixAxisIdentity,
    MatrixCellExecutionSpec,
    resolve_registered_target_identity,
    resolve_registered_task_identity,
)
from abstrak.canary.protocol import build_initial_messages
from abstrak.canary.targets import get_target_stack, load_target_card
from abstrak.canary.tasks import get_task_pack
from abstrak.providers.contracts import (
    CompletionClientIdentity,
    ErrorCategory,
    NormalizedError,
    NormalizedResponse,
    NormalizedUsage,
    ProviderCallError,
    sha256_json,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _pinned(tmp_path: Path) -> PinnedStudySpec:
    spec = MatrixStudySpec(
        study_id="matrix-runner-test",
        seed=20260724,
        agents=("fake-agent",),
        targets=("triton-a100", "tilelang-a100"),
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
    return PinnedStudySpec(
        path=tmp_path / "study.json",
        sha256=_digest("matrix-runner-study"),
        spec=spec,
    )


def _agent(identifier: str) -> MatrixAxisIdentity:
    return _binding(identifier).axis_identity


def _client_identity() -> CompletionClientIdentity:
    return CompletionClientIdentity(
        provider_id="fake-provider",
        model_id="fake-model",
        provider_manifest_sha256="1" * 64,
        model_manifest_sha256="2" * 64,
        requested_model="fake-model",
        model_ref="fake-model",
        returned_model_policy="exact",
        expected_returned_model="fake-model",
        returned_model_required=True,
    )


def _binding(identifier: str) -> MatrixAgentBinding:
    return MatrixAgentBinding(
        agent_id=identifier,
        completion=_client_identity(),
    )


def _policy() -> AgentLoopPolicy:
    return AgentLoopPolicy(
        response_parser="candidate_only",
        stop_policy="correct_latency",
        final_selection="best_correct_latency",
        latency_ceiling_ms=2.0,
    )


def _context() -> MatrixExecutionContext:
    return MatrixExecutionContext(
        controller_revision="1" * 40,
        worker_revision="1" * 40,
        transport=MatrixTransportContext(
            host="gpu.example",
            worker_root="/srv/AbstraK",
            python_executable="/srv/venv/bin/python",
            pythonpath="/srv/AbstraK/src",
            kernelbench_root="/srv/KernelBench",
            asset_root="/srv/AbstraK/benchmarks/r1-a100",
            sandbox="setpriv-supervised",
            device="cuda:0",
            timeout_seconds=300.0,
            network_isolated=False,
            filesystem_read_only=False,
        ),
        asset_manifest_sha256=_digest("assets"),
        floor_manifest_sha256=_digest("floor"),
        environment_manifest_sha256=_digest("environment"),
    )


def _messages(target_id: str):
    return build_initial_messages(
        get_task_pack("row-reduction-scale"),
        load_target_card(target_id),
        policy=_policy(),
    )


def _execution(cell) -> MatrixCellExecutionSpec:
    messages = _messages(cell.target_id)
    return MatrixCellExecutionSpec(
        budget=AgentBudget(max_calls=1),
        policy=_policy(),
        dev_timing=TimingSpec(trial_runs=1, repetitions=1),
        model_ref="fake-model",
        initial_messages_sha256=sha256_json(
            [message.model_dump(mode="json") for message in messages]
        ),
        execution_context_sha256=_context().sha256,
    )


def _response(request: object) -> NormalizedResponse:
    now = datetime.now(timezone.utc)
    text = "```python\nclass ModelNew:\n    pass\n```\n"
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
        text=text,
        finish_reason="stop",
        provider_finish_reason="stop",
        usage=NormalizedUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_characters=100,
            output_characters=len(text),
            provider_reported=True,
            core_fields_complete=True,
        ),
        started_at_utc=now,
        finished_at_utc=now,
        elapsed_ms=1.0,
        logical_request_sha256=sha256_json(request),
        transport_request_sha256="4" * 64,
        transport_response_sha256="5" * 64,
        sanitized_transport_request={},
        raw_transport_response={},
    )


class _BoundFakeClient:
    completion_identity = _client_identity()


class _SuccessClient(_BoundFakeClient):
    def complete(self, request: object) -> NormalizedResponse:
        return _response(request)


class _ProviderFailureClient(_BoundFakeClient):
    def complete(self, request: object) -> NormalizedResponse:
        now = datetime.now(timezone.utc)
        raise ProviderCallError(
            NormalizedError(
                request_id=request.request_id,  # type: ignore[attr-defined]
                attempt_id="attempt-1",
                provider_id="fake-provider",
                model_id="fake-model",
                category=ErrorCategory.NETWORK,
                provider_type="FakeNetworkError",
                sanitized_message="network unavailable",
                retryable=True,
                request_submitted=True,
                possibly_charged=True,
                started_at_utc=now,
                failed_at_utc=now,
                elapsed_ms=1.0,
                logical_request_sha256=sha256_json(request),
                sanitized_transport_request={},
            )
        )


class _MismatchedResponseClient(_BoundFakeClient):
    def complete(self, request: object) -> NormalizedResponse:
        return _response(request).model_copy(update={"model_manifest_sha256": "9" * 64})


class _StaleResponseClient(_BoundFakeClient):
    def complete(self, request: object) -> NormalizedResponse:
        return _response(request).model_copy(
            update={"request_id": "stale-request", "logical_request_sha256": "8" * 64}
        )


class _WrongReturnedModelClient(_BoundFakeClient):
    def complete(self, request: object) -> NormalizedResponse:
        return _response(request).model_copy(update={"returned_model": "other-model"})


class _MismatchedClientIdentityClient(_SuccessClient):
    completion_identity = _client_identity().model_copy(
        update={"model_manifest_sha256": "7" * 64}
    )


class _MismatchedFailureClient(_BoundFakeClient):
    def complete(self, request: object) -> NormalizedResponse:
        now = datetime.now(timezone.utc)
        raise ProviderCallError(
            NormalizedError(
                request_id=request.request_id,  # type: ignore[attr-defined]
                attempt_id="attempt-1",
                provider_id="other-provider",
                model_id="fake-model",
                category=ErrorCategory.NETWORK,
                provider_type="FakeNetworkError",
                sanitized_message="network unavailable",
                retryable=True,
                request_submitted=True,
                possibly_charged=True,
                started_at_utc=now,
                failed_at_utc=now,
                elapsed_ms=1.0,
                logical_request_sha256=sha256_json(request),
                sanitized_transport_request={},
            )
        )


class _CrashingClient(_BoundFakeClient):
    def complete(self, _request: object) -> NormalizedResponse:
        raise RuntimeError("controller crash")


class _Activity:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0


class _PassingWorker:
    def __init__(self, activity: _Activity) -> None:
        self.activity = activity

    def execute(self, job: WorkerJob) -> WorkerResult:
        self.activity.active += 1
        self.activity.max_active = max(self.activity.max_active, self.activity.active)
        try:
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
            timing = (1.0,) if job.timing is not None else ()
            return WorkerResult(
                job_id=job.job_id,
                job_sha256=job.sha256,
                input_sha256=job.input_sha256,
                candidate_sha256=job.candidate_sha256,
                status="completed",
                compiled=True,
                correct=True,
                cases=cases,
                timing_ms=timing,
                timing_cv=0.0 if timing else None,
            )
        finally:
            self.activity.active -= 1


class _RuntimeFactory:
    def __init__(self, failures: set[str] = frozenset()) -> None:
        self.failures = failures
        self.identities = []
        self.activity = _Activity()

    def __call__(self, identity) -> MatrixAttemptRuntime:
        self.identities.append(identity)
        client = (
            _ProviderFailureClient()
            if identity.artifact_trajectory_id in self.failures
            else _SuccessClient()
        )
        return MatrixAttemptRuntime(
            task=get_task_pack(identity.cell.task_id),
            target=get_target_stack(identity.cell.target_id),
            agent_binding=_binding(identity.cell.agent_id),
            execution=_execution(identity.cell),
            initial_messages=_messages(identity.cell.target_id),
            client=client,
            worker=_PassingWorker(self.activity),
        )


def _run(
    pinned: PinnedStudySpec,
    root: Path,
    factory,
    *,
    live: bool = True,
    expected: int = 4,
    progress=None,
):
    return run_matrix_phase(
        pinned,
        "core",
        artifact_root=root,
        execution_context=_context(),
        live=live,
        expected_operational_request_ceiling=expected,
        resolve_task=resolve_registered_task_identity,
        resolve_target=resolve_registered_target_identity,
        resolve_agent=_agent,
        resolve_execution=_execution,
        runtime_factory=factory,
        progress=progress,
    )


def test_live_and_full_operational_guards_precede_artifacts_and_runtime(
    tmp_path: Path,
) -> None:
    pinned = _pinned(tmp_path)
    root = tmp_path / "artifacts"

    def unexpected(_value):
        raise AssertionError("resolver or runtime factory was called before authorization")

    def guarded_run(*, live, expected):
        return run_matrix_phase(
            pinned,
            "core",
            artifact_root=root,
            execution_context=_context(),
            live=live,
            expected_operational_request_ceiling=expected,
            resolve_task=unexpected,
            resolve_target=unexpected,
            resolve_agent=unexpected,
            resolve_execution=unexpected,
            runtime_factory=unexpected,
        )

    with pytest.raises(MatrixStudyRunError, match="requires live authorization"):
        guarded_run(live=False, expected=4)
    with pytest.raises(MatrixStudyRunError, match="requires live authorization"):
        guarded_run(live=1, expected=4)
    with pytest.raises(MatrixStudyRunError, match="must be an integer"):
        guarded_run(live=True, expected=True)
    with pytest.raises(MatrixStudyRunError, match=r"full-phase ceiling \(4\)"):
        guarded_run(live=True, expected=2)

    assert not root.exists()


def test_serial_phase_executes_once_and_then_resumes_without_live_calls(
    tmp_path: Path,
) -> None:
    pinned = _pinned(tmp_path)
    root = tmp_path / "artifacts"
    factory = _RuntimeFactory()
    progress = []

    first = _run(pinned, root, factory, progress=progress.append)
    second = _run(
        pinned,
        root,
        lambda _identity: (_ for _ in ()).throw(
            AssertionError("completed cells must not create a runtime")
        ),
    )

    expected_ids = tuple(
        cell.trajectory_id for cell in build_matrix_schedule(pinned.spec).cells_for_phase("core")
    )
    assert first.status == second.status == "complete"
    assert tuple(record.identity.artifact_trajectory_id for record in first.records) == expected_ids
    assert [record.identity.artifact_trajectory_id for record in progress] == list(expected_ids)
    assert first.newly_executed_attempts == 2
    assert first.newly_consumed_calls == 2
    assert first.newly_known_input_tokens == 20
    assert first.newly_known_output_tokens == 10
    assert first.cumulative_attempts == first.cumulative_calls == 2
    assert second.newly_executed_attempts == second.newly_consumed_calls == 0
    assert second.cumulative_attempts == second.cumulative_calls == 2
    assert second.initial_plan.resumed_cells == 2
    assert factory.activity.max_active == 1
    assert (root / pinned.spec.study_id / "phase-core-contract" / "sha256sums.txt").is_file()


def test_infrastructure_retry_requires_a_new_invocation_and_runs_before_next_cell(
    tmp_path: Path,
) -> None:
    pinned = _pinned(tmp_path)
    root = tmp_path / "artifacts"
    first_id = build_matrix_schedule(pinned.spec).cells_for_phase("core")[0].trajectory_id
    first_factory = _RuntimeFactory({first_id})

    paused = _run(pinned, root, first_factory)
    resumed_factory = _RuntimeFactory()
    resumed = _run(pinned, root, resumed_factory)

    assert paused.status == "paused_infrastructure"
    assert [record.identity.artifact_trajectory_id for record in paused.records] == [first_id]
    assert paused.final_plan.retry_pending_cells == 1
    assert paused.final_plan.pending_cells == 1
    expected_resumed_ids = [
        f"{first_id}.infra-1",
        build_matrix_schedule(pinned.spec).cells_for_phase("core")[1].trajectory_id,
    ]
    assert [record.identity.artifact_trajectory_id for record in resumed.records] == (
        expected_resumed_ids
    )
    assert resumed.status == "complete"
    assert resumed.cumulative_attempts == resumed.cumulative_calls == 3
    assert resumed.cumulative_usage_complete is False


def test_exhausted_retry_pauses_then_later_invocation_completes_other_cells(
    tmp_path: Path,
) -> None:
    pinned = _pinned(tmp_path)
    root = tmp_path / "artifacts"
    first_id = build_matrix_schedule(pinned.spec).cells_for_phase("core")[0].trajectory_id

    first = _run(pinned, root, _RuntimeFactory({first_id}))
    second = _run(pinned, root, _RuntimeFactory({f"{first_id}.infra-1"}))
    third_factory = _RuntimeFactory()
    third = _run(pinned, root, third_factory)

    assert first.status == second.status == "paused_infrastructure"
    assert second.final_plan.retry_exhausted_cells == 1
    assert second.final_plan.pending_cells == 1
    assert third.status == "incomplete_infrastructure"
    assert third.final_plan.retry_exhausted_cells == 1
    assert third.final_plan.resumed_cells == 1
    assert [identity.trajectory_id for identity in third_factory.identities] == [
        build_matrix_schedule(pinned.spec).cells_for_phase("core")[1].trajectory_id
    ]
    assert third.cumulative_attempts == third.cumulative_calls == 3


def test_partial_attempt_fails_closed_before_runtime_creation(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)
    root = tmp_path / "artifacts"
    contract = build_matrix_phase_contract(
        pinned,
        "core",
        execution_context=_context(),
        resolve_task=resolve_registered_task_identity,
        resolve_target=resolve_registered_target_identity,
        resolve_agent=_agent,
        resolve_execution=_execution,
    )
    identity = contract.plan.cells[0].identity
    directory = root / pinned.spec.study_id / identity.artifact_trajectory_id
    directory.mkdir(parents=True)
    (directory / "outcome.json").write_text("{}", encoding="utf-8")

    with pytest.raises(MatrixStudyRunError, match="initial matrix phase plan"):
        _run(
            pinned,
            root,
            lambda _identity: (_ for _ in ()).throw(
                AssertionError("partial artifacts must fail before runtime creation")
            ),
        )

    assert not (root / pinned.spec.study_id / "phase-core-contract").exists()


def test_controller_failure_seals_partial_attempt_and_stops_the_phase(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)
    root = tmp_path / "artifacts"
    created = []

    def crashing_factory(identity):
        created.append(identity.artifact_trajectory_id)
        return MatrixAttemptRuntime(
            task=get_task_pack(identity.cell.task_id),
            target=get_target_stack(identity.cell.target_id),
            agent_binding=_binding(identity.cell.agent_id),
            execution=_execution(identity.cell),
            initial_messages=_messages(identity.cell.target_id),
            client=_CrashingClient(),
            worker=_PassingWorker(_Activity()),
        )

    with pytest.raises(MatrixStudyRunError, match="attempt controller failed"):
        _run(pinned, root, crashing_factory)

    assert len(created) == 1
    directory = root / pinned.spec.study_id / created[0]
    assert (directory / "controller-error.json").is_file()
    assert (directory / "sha256sums.txt").is_file()
    assert not (directory / "outcome.json").exists()
    with pytest.raises(MatrixStudyRunError, match="initial matrix phase plan"):
        _run(pinned, root, _RuntimeFactory())


def test_runtime_identity_mismatch_fails_before_attempt_directory(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)
    root = tmp_path / "artifacts"
    first_id = build_matrix_schedule(pinned.spec).cells_for_phase("core")[0].trajectory_id

    def mismatched_factory(identity):
        execution = _execution(identity.cell).model_copy(
            update={"execution_context_sha256": _digest("drift")}
        )
        return MatrixAttemptRuntime(
            task=get_task_pack(identity.cell.task_id),
            target=get_target_stack(identity.cell.target_id),
            agent_binding=_binding(identity.cell.agent_id),
            execution=execution,
            initial_messages=_messages(identity.cell.target_id),
            client=_SuccessClient(),
            worker=_PassingWorker(_Activity()),
        )

    with pytest.raises(MatrixStudyRunError, match="runtime execution inputs differ"):
        _run(pinned, root, mismatched_factory)

    assert not (root / pinned.spec.study_id / first_id).exists()


@pytest.mark.parametrize(
    "client",
    (_MismatchedResponseClient(), _StaleResponseClient(), _WrongReturnedModelClient()),
    ids=("manifest", "request-linkage", "returned-model"),
)
def test_provider_response_must_match_hash_bound_agent_identity(
    tmp_path: Path,
    client,
) -> None:
    pinned = _pinned(tmp_path)
    root = tmp_path / "artifacts"

    def mismatched_factory(identity):
        return MatrixAttemptRuntime(
            task=get_task_pack(identity.cell.task_id),
            target=get_target_stack(identity.cell.target_id),
            agent_binding=_binding(identity.cell.agent_id),
            execution=_execution(identity.cell),
            initial_messages=_messages(identity.cell.target_id),
            client=client,
            worker=_PassingWorker(_Activity()),
        )

    with pytest.raises(MatrixStudyRunError, match="attempt controller failed"):
        _run(pinned, root, mismatched_factory)

    first_id = build_matrix_schedule(pinned.spec).cells_for_phase("core")[0].trajectory_id
    directory = root / pinned.spec.study_id / first_id
    assert (directory / "controller-error.json").is_file()
    assert not (directory / "outcome.json").exists()


def test_completion_client_identity_is_checked_before_attempt_creation(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)
    root = tmp_path / "artifacts"
    first_id = build_matrix_schedule(pinned.spec).cells_for_phase("core")[0].trajectory_id

    def mismatched_factory(identity):
        return MatrixAttemptRuntime(
            task=get_task_pack(identity.cell.task_id),
            target=get_target_stack(identity.cell.target_id),
            agent_binding=_binding(identity.cell.agent_id),
            execution=_execution(identity.cell),
            initial_messages=_messages(identity.cell.target_id),
            client=_MismatchedClientIdentityClient(),
            worker=_PassingWorker(_Activity()),
        )

    with pytest.raises(MatrixStudyRunError, match="completion client differs"):
        _run(pinned, root, mismatched_factory)

    assert not (root / pinned.spec.study_id / first_id).exists()


def test_provider_error_must_match_hash_bound_agent_identity(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)
    root = tmp_path / "artifacts"

    def mismatched_factory(identity):
        return MatrixAttemptRuntime(
            task=get_task_pack(identity.cell.task_id),
            target=get_target_stack(identity.cell.target_id),
            agent_binding=_binding(identity.cell.agent_id),
            execution=_execution(identity.cell),
            initial_messages=_messages(identity.cell.target_id),
            client=_MismatchedFailureClient(),
            worker=_PassingWorker(_Activity()),
        )

    with pytest.raises(MatrixStudyRunError, match="attempt controller failed"):
        _run(pinned, root, mismatched_factory)

    first_id = build_matrix_schedule(pinned.spec).cells_for_phase("core")[0].trajectory_id
    directory = root / pinned.spec.study_id / first_id
    assert (directory / "controller-error.json").is_file()
    assert not (directory / "outcome.json").exists()


def test_progress_waits_for_new_attempt_resume_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = _pinned(tmp_path)
    root = tmp_path / "artifacts"
    original = matrix_runner.plan_matrix_phase
    calls = 0

    def fail_first_replan(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise matrix_runner.MatrixStudyPlanError("injected resume verification failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(matrix_runner, "plan_matrix_phase", fail_first_replan)
    progress = []

    with pytest.raises(MatrixStudyRunError, match="failed resume verification"):
        _run(pinned, root, _RuntimeFactory(), progress=progress.append)

    assert progress == []
