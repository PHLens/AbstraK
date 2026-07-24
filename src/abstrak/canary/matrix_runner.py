"""Serial execution for hash-bound generic canary matrix phases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, model_validator

from abstrak.canary.artifacts import TrajectoryArtifactError, TrajectoryStore, verify_trajectory
from abstrak.canary.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    CanaryModel,
    TargetStackSpec,
    TaskPackSpec,
    TrajectoryOutcome,
)
from abstrak.canary.loop import CanaryAgentLoop, CompletionClient, WorkerExecutor
from abstrak.canary.manifests import PinnedStudySpec
from abstrak.canary.matrix import MatrixSchedule, build_matrix_schedule
from abstrak.canary.matrix_study import (
    AxisIdentityResolver,
    CellExecutionResolver,
    ExistingCellAttempt,
    ExistingCellAttemptLoader,
    MatrixAttemptManifest,
    MatrixAxisIdentity,
    MatrixCellArtifactIdentity,
    MatrixCellExecutionSpec,
    MatrixPhasePlan,
    MatrixPhasePlanSummary,
    MatrixStudyPlanError,
    build_matrix_attempt_manifest,
    plan_matrix_phase,
    sealed_cell_identity_loader,
)
from abstrak.providers.contracts import (
    ChatMessage,
    CompletionClientIdentity,
    LogicalRequest,
    NormalizedError,
    NormalizedResponse,
    ProviderCallError,
    sha256_json,
)

RunStatus = Literal["complete", "paused_infrastructure", "incomplete_infrastructure"]


class MatrixStudyRunError(RuntimeError):
    """Raised when a guarded matrix phase cannot continue deterministically."""


class MatrixAgentBinding(CanaryModel):
    """Expected provider/model identity for one hash-bound Agent axis value."""

    schema_version: Literal["abstrak-matrix-agent-binding.v1"] = (
        "abstrak-matrix-agent-binding.v1"
    )
    agent_id: str = Field(pattern=IDENTIFIER_PATTERN)
    completion: CompletionClientIdentity

    @property
    def sha256(self) -> str:
        return sha256_json(self)

    @property
    def axis_identity(self) -> MatrixAxisIdentity:
        return MatrixAxisIdentity(kind="agent", id=self.agent_id, sha256=self.sha256)


@dataclass(frozen=True)
class _BoundCompletionClient:
    client: CompletionClient
    binding: MatrixAgentBinding

    def complete(self, request: LogicalRequest) -> NormalizedResponse:
        if request.model_ref != self.binding.completion.model_ref:
            raise MatrixStudyRunError("logical request model differs from the Agent binding")
        try:
            response = self.client.complete(request)
        except ProviderCallError as error:
            self._verify_request_linkage(
                request,
                request_id=error.record.request_id,
                logical_request_sha256=error.record.logical_request_sha256,
            )
            self._verify_error(error.record)
            raise
        self._verify_request_linkage(
            request,
            request_id=response.request_id,
            logical_request_sha256=response.logical_request_sha256,
        )
        self._verify_response(response)
        return response

    def _verify_error(self, error: NormalizedError) -> None:
        expected = self.binding.completion
        if error.provider_id != expected.provider_id or error.model_id != expected.model_id:
            raise MatrixStudyRunError(
                "provider error identity differs from the Agent binding"
            )

    def _verify_request_linkage(
        self,
        request: LogicalRequest,
        *,
        request_id: str,
        logical_request_sha256: str,
    ) -> None:
        if request_id != request.request_id or logical_request_sha256 != sha256_json(request):
            raise MatrixStudyRunError("provider result differs from its logical request")

    def _verify_response(self, response: NormalizedResponse) -> None:
        expected = self.binding.completion
        checks = {
            "provider_id": response.provider_id == expected.provider_id,
            "model_id": response.model_id == expected.model_id,
            "provider_manifest_sha256": (
                response.provider_manifest_sha256 == expected.provider_manifest_sha256
            ),
            "model_manifest_sha256": (
                response.model_manifest_sha256 == expected.model_manifest_sha256
            ),
            "requested_model": response.requested_model == expected.requested_model,
            "returned_model_present": (
                not expected.returned_model_required or response.returned_model is not None
            ),
            "returned_model_exact": (
                expected.returned_model_policy != "exact"
                or response.returned_model == expected.expected_returned_model
            ),
        }
        mismatches = tuple(name for name, matches in checks.items() if not matches)
        if mismatches:
            raise MatrixStudyRunError(
                "provider response differs from the Agent binding: " + ", ".join(mismatches)
            )


@dataclass(frozen=True)
class MatrixAttemptRuntime:
    """Live objects and frozen inputs required to execute exactly one attempt."""

    task: TaskPackSpec
    target: TargetStackSpec
    agent_binding: MatrixAgentBinding
    execution: MatrixCellExecutionSpec
    initial_messages: tuple[ChatMessage, ...]
    client: CompletionClient
    worker: WorkerExecutor
    artifact_secrets: tuple[str, ...] = ()

    def manifest_for(self, identity: MatrixCellArtifactIdentity) -> MatrixAttemptManifest:
        if self.task.id != identity.cell.task_id or sha256_json(self.task) != identity.task_sha256:
            raise MatrixStudyRunError("runtime task differs from its planned cell identity")
        if (
            self.target.id != identity.cell.target_id
            or sha256_json(self.target) != identity.target_sha256
        ):
            raise MatrixStudyRunError("runtime target differs from its planned cell identity")
        if self.agent_binding.axis_identity != MatrixAxisIdentity(
            kind="agent",
            id=identity.cell.agent_id,
            sha256=identity.agent_sha256,
        ):
            raise MatrixStudyRunError("runtime Agent differs from its planned cell identity")
        if self.agent_binding.completion.model_ref != self.execution.model_ref:
            raise MatrixStudyRunError("runtime model differs from its Agent binding")
        client_identity = getattr(self.client, "completion_identity", None)
        if client_identity != self.agent_binding.completion:
            raise MatrixStudyRunError("completion client differs from its Agent binding")
        try:
            return build_matrix_attempt_manifest(
                identity,
                agent=self.agent_binding.axis_identity,
                execution=self.execution,
                initial_messages=self.initial_messages,
            )
        except ValueError as error:
            raise MatrixStudyRunError(
                "runtime execution inputs differ from their planned cell identity"
            ) from error


class AttemptRuntimeFactory(Protocol):
    def __call__(self, identity: MatrixCellArtifactIdentity) -> MatrixAttemptRuntime: ...


class AttemptProgressSink(Protocol):
    def __call__(self, record: MatrixAttemptRunRecord) -> None: ...


class MatrixTransportContext(CanaryModel):
    """Frozen SSH transport and isolation inputs for generated-code execution."""

    schema_version: Literal["abstrak-matrix-transport-context.v1"] = (
        "abstrak-matrix-transport-context.v1"
    )
    kind: Literal["ssh"] = "ssh"
    host: str = Field(min_length=1)
    worker_root: str = Field(min_length=1)
    python_executable: str = Field(min_length=1)
    pythonpath: str = Field(min_length=1)
    kernelbench_root: str = Field(min_length=1)
    asset_root: str = Field(min_length=1)
    sandbox: Literal["bubblewrap", "setpriv-supervised"]
    device: str = Field(pattern=r"^cuda:[0-9]+$")
    timeout_seconds: float = Field(gt=0)
    network_isolated: bool
    filesystem_read_only: bool


class MatrixExecutionContext(CanaryModel):
    """Sealed phase-wide provenance referenced by every cell execution identity."""

    schema_version: Literal["abstrak-matrix-execution-context.v1"] = (
        "abstrak-matrix-execution-context.v1"
    )
    controller_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    worker_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    transport: MatrixTransportContext
    asset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    floor_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    cache_policy: Literal["per-job-temporary"] = "per-job-temporary"
    gpu_jobs_serial: Literal[True] = True
    generated_code_remote_only: Literal[True] = True
    non_container_worker: Literal[True] = True

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class MatrixPhaseContract(CanaryModel):
    """Immutable attempt-zero identities for one complete phase execution contract."""

    schema_version: Literal["abstrak-matrix-phase-contract.v1"] = "abstrak-matrix-phase-contract.v1"
    execution_context: MatrixExecutionContext
    plan: MatrixPhasePlan

    @model_validator(mode="after")
    def contains_only_primary_pending_attempts(self) -> MatrixPhaseContract:
        if any(
            cell.state != "pending" or cell.identity.attempt_index != 0 for cell in self.plan.cells
        ):
            raise ValueError("phase contract must contain only attempt-zero pending cells")
        if any(
            cell.identity.execution_context_sha256 != self.execution_context.sha256
            for cell in self.plan.cells
        ):
            raise ValueError("phase cells do not reference the sealed execution context")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class MatrixAttemptRunRecord(CanaryModel):
    """One newly executed attempt in serial phase order."""

    schema_version: Literal["abstrak-matrix-attempt-run-record.v1"] = (
        "abstrak-matrix-attempt-run-record.v1"
    )
    identity: MatrixCellArtifactIdentity
    outcome_status: str
    calls: int = Field(ge=0, le=4)
    known_input_tokens: int = Field(ge=0)
    known_output_tokens: int = Field(ge=0)
    usage_complete: bool
    retryable_infrastructure: bool
    artifact_directory: str


class MatrixPhaseRunSummary(CanaryModel):
    """Terminal result of one guarded invocation over all actionable phase cells."""

    schema_version: Literal["abstrak-matrix-phase-run-summary.v1"] = (
        "abstrak-matrix-phase-run-summary.v1"
    )
    status: RunStatus
    study_id: str
    phase_id: str
    phase_contract_sha256: str
    schedule_sha256: str
    scientific_request_ceiling: int = Field(ge=1)
    operational_request_ceiling: int = Field(ge=1)
    initial_plan_sha256: str
    final_plan_sha256: str
    initial_plan: MatrixPhasePlanSummary
    final_plan: MatrixPhasePlanSummary
    newly_executed_attempts: int = Field(ge=0)
    newly_consumed_calls: int = Field(ge=0)
    newly_known_input_tokens: int = Field(ge=0)
    newly_known_output_tokens: int = Field(ge=0)
    newly_usage_complete: bool
    cumulative_attempts: int = Field(ge=0)
    cumulative_calls: int = Field(ge=0)
    cumulative_known_input_tokens: int = Field(ge=0)
    cumulative_known_output_tokens: int = Field(ge=0)
    cumulative_usage_complete: bool
    records: tuple[MatrixAttemptRunRecord, ...]

    @model_validator(mode="after")
    def terminal_counts_match_records(self) -> MatrixPhaseRunSummary:
        if self.newly_executed_attempts != len(self.records):
            raise ValueError("executed-attempt count differs from run records")
        if self.newly_consumed_calls != sum(record.calls for record in self.records):
            raise ValueError("consumed-call count differs from run records")
        if self.newly_known_input_tokens != sum(
            record.known_input_tokens for record in self.records
        ) or self.newly_known_output_tokens != sum(
            record.known_output_tokens for record in self.records
        ):
            raise ValueError("new token totals differ from run records")
        if self.newly_usage_complete != all(record.usage_complete for record in self.records):
            raise ValueError("new usage-completeness flag differs from run records")
        actionable = self.final_plan.pending_cells + self.final_plan.retry_pending_cells
        if actionable:
            expected_status: RunStatus = "paused_infrastructure"
            if not self.records or not self.records[-1].retryable_infrastructure:
                raise ValueError("phase paused without a terminal infrastructure failure")
        else:
            expected_status = (
                "incomplete_infrastructure" if self.final_plan.retry_exhausted_cells else "complete"
            )
        if self.status != expected_status:
            raise ValueError("run status differs from terminal phase state")
        if self.newly_consumed_calls > self.initial_plan.pending_operational_request_ceiling:
            raise ValueError("run consumed more calls than its initial remaining ceiling")
        if (
            self.cumulative_attempts < self.newly_executed_attempts
            or self.cumulative_calls < self.newly_consumed_calls
            or self.cumulative_known_input_tokens < self.newly_known_input_tokens
            or self.cumulative_known_output_tokens < self.newly_known_output_tokens
        ):
            raise ValueError("cumulative costs cannot be smaller than new invocation costs")
        if self.cumulative_calls > self.operational_request_ceiling:
            raise ValueError("cumulative calls exceed the frozen operational ceiling")
        return self


def _attempt_identity(
    primary: MatrixCellArtifactIdentity,
    attempt_index: int,
) -> MatrixCellArtifactIdentity:
    payload = primary.model_dump(mode="json")
    payload.update(
        {
            "attempt_index": attempt_index,
            "artifact_trajectory_id": (
                primary.trajectory_id
                if attempt_index == 0
                else f"{primary.trajectory_id}.infra-{attempt_index}"
            ),
        }
    )
    return MatrixCellArtifactIdentity.model_validate(payload)


def _assert_plan_matches_contract(
    plan: MatrixPhasePlan,
    contract: MatrixPhaseContract,
) -> None:
    primary_by_id = {cell.identity.trajectory_id: cell.identity for cell in contract.plan.cells}
    if tuple(cell.identity.trajectory_id for cell in plan.cells) != tuple(primary_by_id):
        raise MatrixStudyRunError("phase plan cells differ from the frozen phase contract")
    for cell in plan.cells:
        primary = primary_by_id[cell.identity.trajectory_id]
        if cell.identity != _attempt_identity(primary, cell.identity.attempt_index):
            raise MatrixStudyRunError(
                "phase plan execution identity differs from the frozen phase contract"
            )


def _load_contract_attempts(
    contract: MatrixPhaseContract,
    loader: ExistingCellAttemptLoader,
) -> tuple[ExistingCellAttempt, ...]:
    attempts: list[ExistingCellAttempt] = []
    for cell in contract.plan.cells:
        primary = cell.identity
        maximum_attempt = contract.plan.infrastructure_retries
        for attempt_index in range(1 + maximum_attempt):
            loaded = loader(_attempt_identity(primary, attempt_index))
            if loaded is not None:
                attempts.append(loaded)
    return tuple(attempts)


def build_matrix_phase_contract(
    pinned: PinnedStudySpec,
    phase_id: str,
    *,
    execution_context: MatrixExecutionContext,
    resolve_task: AxisIdentityResolver,
    resolve_target: AxisIdentityResolver,
    resolve_agent: AxisIdentityResolver,
    resolve_execution: CellExecutionResolver,
    schedule: MatrixSchedule | None = None,
) -> MatrixPhaseContract:
    """Build the immutable primary-attempt plan without reading or writing artifacts."""

    plan = plan_matrix_phase(
        pinned,
        phase_id,
        resolve_task=resolve_task,
        resolve_target=resolve_target,
        resolve_agent=resolve_agent,
        resolve_execution=resolve_execution,
        schedule=schedule,
    )
    return MatrixPhaseContract(execution_context=execution_context, plan=plan)


def _ensure_phase_contract(
    artifact_root: str | Path,
    contract: MatrixPhaseContract,
) -> Path:
    directory = (
        Path(artifact_root).expanduser()
        / contract.plan.study_id
        / f"phase-{contract.plan.phase_id}-contract"
    )
    if directory.exists() or directory.is_symlink():
        if directory.is_symlink():
            raise MatrixStudyRunError("phase contract artifact cannot be a symbolic link")
        try:
            verify_trajectory(directory)
            actual = MatrixPhaseContract.model_validate_json(
                (directory / "run-manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TrajectoryArtifactError) as error:
            raise MatrixStudyRunError(f"phase contract artifact is invalid: {directory}") from error
        if actual != contract:
            raise MatrixStudyRunError("existing phase contract differs from current frozen inputs")
        return directory
    try:
        store = TrajectoryStore.create(
            artifact_root,
            contract.plan.study_id,
            f"phase-{contract.plan.phase_id}-contract",
        )
        store.write_json("run-manifest.json", contract)
        store.seal()
    except (OSError, TrajectoryArtifactError) as error:
        raise MatrixStudyRunError("cannot create the immutable phase contract") from error
    return store.run_directory


def _seal_controller_failure(store: TrajectoryStore, error: Exception) -> None:
    try:
        store.write_json(
            "controller-error.json",
            {"error_type": type(error).__name__},
        )
    except TrajectoryArtifactError:
        pass
    try:
        if not (store.run_directory / "sha256sums.txt").exists():
            store.seal()
    except TrajectoryArtifactError:
        pass


def _run_attempt(
    identity: MatrixCellArtifactIdentity,
    runtime: MatrixAttemptRuntime,
    *,
    artifact_root: str | Path,
) -> tuple[TrajectoryOutcome, Path]:
    manifest = runtime.manifest_for(identity)
    try:
        store = TrajectoryStore.create(
            artifact_root,
            identity.study_id,
            identity.artifact_trajectory_id,
            secrets=runtime.artifact_secrets,
        )
    except (OSError, TrajectoryArtifactError) as error:
        raise MatrixStudyRunError(
            f"cannot create attempt artifact for {identity.artifact_trajectory_id}"
        ) from error
    try:
        store.write_json("run-manifest.json", manifest)
        store.write_json("cell-identity.json", identity)
        outcome = CanaryAgentLoop(
            client=_BoundCompletionClient(runtime.client, runtime.agent_binding),
            worker=runtime.worker,
            store=store,
        ).run(
            trajectory_id=identity.artifact_trajectory_id,
            model_ref=runtime.execution.model_ref,
            initial_messages=runtime.initial_messages,
            task=runtime.task,
            target=runtime.target,
            budget=runtime.execution.budget,
            device=runtime.execution.device,
            dev_timing=runtime.execution.dev_timing,
            policy=runtime.execution.policy,
        )
    except Exception as error:
        _seal_controller_failure(store, error)
        raise MatrixStudyRunError(
            f"attempt controller failed for {identity.artifact_trajectory_id}: "
            f"{type(error).__name__}"
        ) from error
    return outcome, store.run_directory


def run_matrix_phase(
    pinned: PinnedStudySpec,
    phase_id: str,
    *,
    artifact_root: str | Path,
    execution_context: MatrixExecutionContext,
    live: bool,
    expected_operational_request_ceiling: int,
    resolve_task: AxisIdentityResolver,
    resolve_target: AxisIdentityResolver,
    resolve_agent: AxisIdentityResolver,
    resolve_execution: CellExecutionResolver,
    runtime_factory: AttemptRuntimeFactory,
    schedule: MatrixSchedule | None = None,
    progress: AttemptProgressSink | None = None,
) -> MatrixPhaseRunSummary:
    """Run every actionable phase attempt in order after explicit live authorization."""

    if live is not True:
        raise MatrixStudyRunError("matrix phase execution requires live authorization")
    if isinstance(expected_operational_request_ceiling, bool) or not isinstance(
        expected_operational_request_ceiling, int
    ):
        raise MatrixStudyRunError("expected operational request ceiling must be an integer")
    frozen_schedule = schedule or build_matrix_schedule(pinned.spec)
    if frozen_schedule.spec != pinned.spec or frozen_schedule.spec_sha256 != pinned.spec.sha256:
        raise MatrixStudyRunError("matrix schedule differs from the pinned study spec")
    try:
        full_phase_ceiling = frozen_schedule.phase_operational_request_ceiling(phase_id)
    except ValueError as error:
        raise MatrixStudyRunError(f"unknown matrix phase: {phase_id}") from error
    if expected_operational_request_ceiling != full_phase_ceiling:
        raise MatrixStudyRunError(
            "expected operational request ceiling must equal the frozen full-phase ceiling "
            f"({full_phase_ceiling})"
        )

    loader = sealed_cell_identity_loader(artifact_root)
    try:
        contract = build_matrix_phase_contract(
            pinned,
            phase_id,
            execution_context=execution_context,
            resolve_task=resolve_task,
            resolve_target=resolve_target,
            resolve_agent=resolve_agent,
            resolve_execution=resolve_execution,
            schedule=frozen_schedule,
        )
        initial_plan = plan_matrix_phase(
            pinned,
            phase_id,
            resolve_task=resolve_task,
            resolve_target=resolve_target,
            resolve_agent=resolve_agent,
            resolve_execution=resolve_execution,
            schedule=frozen_schedule,
            load_existing_attempt=loader,
        )
    except MatrixStudyPlanError as error:
        raise MatrixStudyRunError("cannot build the initial matrix phase plan") from error
    _assert_plan_matches_contract(initial_plan, contract)
    _ensure_phase_contract(artifact_root, contract)
    maximum_new_attempts = (
        initial_plan.summary.pending_cells * (1 + initial_plan.infrastructure_retries)
        + initial_plan.summary.retry_pending_cells
    )
    records: list[MatrixAttemptRunRecord] = []
    current_plan = initial_plan
    while True:
        actionable = tuple(
            cell for cell in current_plan.cells if cell.state in {"pending", "retry_pending"}
        )
        if not actionable:
            break
        if len(records) >= maximum_new_attempts:
            raise MatrixStudyRunError("phase exceeded its bounded number of new attempts")
        identity = actionable[0].identity
        try:
            runtime = runtime_factory(identity)
        except Exception as error:
            raise MatrixStudyRunError(
                f"cannot resolve runtime for {identity.artifact_trajectory_id}: "
                f"{type(error).__name__}"
            ) from error
        if not isinstance(runtime, MatrixAttemptRuntime):
            raise MatrixStudyRunError(
                f"runtime factory returned {type(runtime).__name__}, expected MatrixAttemptRuntime"
            )
        outcome, directory = _run_attempt(
            identity,
            runtime,
            artifact_root=artifact_root,
        )
        record = MatrixAttemptRunRecord(
            identity=identity,
            outcome_status=outcome.status,
            calls=outcome.calls,
            known_input_tokens=outcome.known_input_tokens,
            known_output_tokens=outcome.known_output_tokens,
            usage_complete=outcome.usage_complete,
            retryable_infrastructure=outcome.status in {"provider_error", "worker_error"},
            artifact_directory=str(directory),
        )
        records.append(record)
        try:
            current_plan = plan_matrix_phase(
                pinned,
                phase_id,
                resolve_task=resolve_task,
                resolve_target=resolve_target,
                resolve_agent=resolve_agent,
                resolve_execution=resolve_execution,
                schedule=frozen_schedule,
                load_existing_attempt=loader,
            )
            _assert_plan_matches_contract(current_plan, contract)
        except MatrixStudyPlanError as error:
            raise MatrixStudyRunError("new attempt artifact failed resume verification") from error
        if progress is not None:
            progress(record)
        if record.retryable_infrastructure:
            break

    final_summary = current_plan.summary
    try:
        cumulative_attempts = _load_contract_attempts(contract, loader)
    except Exception as error:
        raise MatrixStudyRunError("cannot rebuild cumulative attempt cost ledger") from error
    actionable = final_summary.pending_cells + final_summary.retry_pending_cells
    if actionable:
        status: RunStatus = "paused_infrastructure"
    elif final_summary.retry_exhausted_cells:
        status = "incomplete_infrastructure"
    else:
        status = "complete"
    return MatrixPhaseRunSummary(
        status=status,
        study_id=pinned.spec.study_id,
        phase_id=phase_id,
        phase_contract_sha256=contract.sha256,
        schedule_sha256=frozen_schedule.sha256,
        scientific_request_ceiling=contract.plan.scientific_request_ceiling,
        operational_request_ceiling=contract.plan.operational_request_ceiling,
        initial_plan_sha256=initial_plan.sha256,
        final_plan_sha256=current_plan.sha256,
        initial_plan=initial_plan.summary,
        final_plan=final_summary,
        newly_executed_attempts=len(records),
        newly_consumed_calls=sum(record.calls for record in records),
        newly_known_input_tokens=sum(record.known_input_tokens for record in records),
        newly_known_output_tokens=sum(record.known_output_tokens for record in records),
        newly_usage_complete=all(record.usage_complete for record in records),
        cumulative_attempts=len(cumulative_attempts),
        cumulative_calls=sum(attempt.outcome.calls for attempt in cumulative_attempts),
        cumulative_known_input_tokens=sum(
            attempt.outcome.known_input_tokens for attempt in cumulative_attempts
        ),
        cumulative_known_output_tokens=sum(
            attempt.outcome.known_output_tokens for attempt in cumulative_attempts
        ),
        cumulative_usage_complete=all(
            attempt.outcome.usage_complete for attempt in cumulative_attempts
        ),
        records=tuple(records),
    )
