"""Pure planning and resume identity checks for generic matrix-study phases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, model_validator

from abstrak.canary.artifacts import TrajectoryArtifactError, verify_trajectory
from abstrak.canary.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    AgentBudget,
    AgentLoopPolicy,
    CanaryModel,
    TimingSpec,
    TrajectoryOutcome,
    WorkerJob,
    WorkerResult,
)
from abstrak.canary.manifests import PinnedStudySpec
from abstrak.canary.matrix import MatrixCell, MatrixSchedule, build_matrix_schedule
from abstrak.canary.targets import get_target_stack, load_target_card
from abstrak.canary.tasks import get_task_pack, load_task_source
from abstrak.providers.contracts import ChatMessage, LogicalRequest, sha256_json

AxisKind = Literal["task", "target", "agent"]
CellPlanState = Literal["pending", "retry_pending", "resumed", "retry_exhausted"]


class MatrixStudyPlanError(ValueError):
    """Raised when a phase cannot be planned without identity ambiguity."""


class MatrixAxisIdentity(CanaryModel):
    """Hash-bound registry resolution for one task, target, or Agent ID."""

    schema_version: Literal["abstrak-matrix-axis-identity.v1"] = "abstrak-matrix-axis-identity.v1"
    kind: AxisKind
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    sha256: str = Field(pattern=SHA256_PATTERN)


class MatrixCellExecutionSpec(CanaryModel):
    """Resolved loop inputs whose hashes become part of one cell identity."""

    schema_version: Literal["abstrak-matrix-cell-execution.v1"] = "abstrak-matrix-cell-execution.v1"
    budget: AgentBudget
    policy: AgentLoopPolicy
    dev_timing: TimingSpec | None = None
    model_ref: str = Field(min_length=1)
    initial_messages_sha256: str = Field(pattern=SHA256_PATTERN)
    device: str = Field(default="cuda:0", pattern=r"^cuda:[0-9]+$")
    execution_context_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def latency_policy_has_dev_timing(self) -> MatrixCellExecutionSpec:
        if (
            self.policy.stop_policy == "correct_latency"
            or self.policy.final_selection == "best_correct_latency"
        ) and self.dev_timing is None:
            raise ValueError("latency-based loop policies require dev timing")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class MatrixCellArtifactIdentity(CanaryModel):
    """Complete immutable identity required to resume one logical matrix cell."""

    schema_version: Literal["abstrak-matrix-cell-artifact-identity.v1"] = (
        "abstrak-matrix-cell-artifact-identity.v1"
    )
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    raw_study_sha256: str = Field(pattern=SHA256_PATTERN)
    spec_sha256: str = Field(pattern=SHA256_PATTERN)
    schedule_sha256: str = Field(pattern=SHA256_PATTERN)
    phase_id: str = Field(pattern=IDENTIFIER_PATTERN)
    trajectory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    artifact_trajectory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    attempt_index: int = Field(ge=0, le=1)
    cell: MatrixCell
    task_sha256: str = Field(pattern=SHA256_PATTERN)
    target_sha256: str = Field(pattern=SHA256_PATTERN)
    agent_sha256: str = Field(pattern=SHA256_PATTERN)
    policy_sha256: str = Field(pattern=SHA256_PATTERN)
    budget_sha256: str = Field(pattern=SHA256_PATTERN)
    max_calls_per_trajectory: int = Field(ge=1, le=4)
    dev_timing_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    model_ref: str = Field(min_length=1)
    initial_messages_sha256: str = Field(pattern=SHA256_PATTERN)
    device: str = Field(pattern=r"^cuda:[0-9]+$")
    execution_context_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def redundant_cell_identity_matches(self) -> MatrixCellArtifactIdentity:
        if self.phase_id != self.cell.phase_id:
            raise ValueError("artifact phase ID differs from its matrix cell")
        if self.trajectory_id != self.cell.trajectory_id:
            raise ValueError("artifact trajectory ID differs from its matrix cell")
        expected_artifact_id = _attempt_trajectory_id(self.trajectory_id, self.attempt_index)
        if self.artifact_trajectory_id != expected_artifact_id:
            raise ValueError("artifact attempt ID differs from its logical trajectory")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _messages_sha256(messages: tuple[ChatMessage, ...]) -> str:
    return sha256_json([message.model_dump(mode="json") for message in messages])


class MatrixAttemptManifest(CanaryModel):
    """Typed runtime inputs that bind a sealed attempt to its planned identity."""

    schema_version: Literal["abstrak-matrix-attempt-manifest.v1"] = (
        "abstrak-matrix-attempt-manifest.v1"
    )
    trajectory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    cell_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    agent: MatrixAxisIdentity
    policy: AgentLoopPolicy
    budget: AgentBudget
    dev_timing: TimingSpec | None = None
    model_ref: str = Field(min_length=1)
    initial_messages: tuple[ChatMessage, ...] = Field(min_length=1)
    device: str = Field(pattern=r"^cuda:[0-9]+$")
    execution_context_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def runtime_inputs_are_coherent(self) -> MatrixAttemptManifest:
        if self.agent.kind != "agent":
            raise ValueError("attempt manifest agent identity has the wrong kind")
        if (
            self.policy.stop_policy == "correct_latency"
            or self.policy.final_selection == "best_correct_latency"
        ) and self.dev_timing is None:
            raise ValueError("latency-based attempt manifest requires dev timing")
        return self

    def verify_for_identity(self, identity: MatrixCellArtifactIdentity) -> None:
        expected_timing_sha256 = None if self.dev_timing is None else sha256_json(self.dev_timing)
        execution = MatrixCellExecutionSpec(
            budget=self.budget,
            policy=self.policy,
            dev_timing=self.dev_timing,
            model_ref=self.model_ref,
            initial_messages_sha256=_messages_sha256(self.initial_messages),
            device=self.device,
            execution_context_sha256=self.execution_context_sha256,
        )
        checks = {
            "trajectory_id": self.trajectory_id == identity.artifact_trajectory_id,
            "cell_identity_sha256": self.cell_identity_sha256 == identity.sha256,
            "agent_id": self.agent.id == identity.cell.agent_id,
            "agent_sha256": self.agent.sha256 == identity.agent_sha256,
            "policy_sha256": self.policy.sha256 == identity.policy_sha256,
            "budget_sha256": sha256_json(self.budget) == identity.budget_sha256,
            "max_calls_per_trajectory": (
                self.budget.max_calls == identity.max_calls_per_trajectory
            ),
            "dev_timing_sha256": expected_timing_sha256 == identity.dev_timing_sha256,
            "model_ref": self.model_ref == identity.model_ref,
            "initial_messages_sha256": (
                _messages_sha256(self.initial_messages) == identity.initial_messages_sha256
            ),
            "device": self.device == identity.device,
            "execution_context_sha256": (
                self.execution_context_sha256 == identity.execution_context_sha256
            ),
            "execution_sha256": execution.sha256 == identity.execution_sha256,
        }
        mismatches = tuple(name for name, matches in checks.items() if not matches)
        if mismatches:
            raise ValueError(
                "attempt manifest differs from cell identity: " + ", ".join(mismatches)
            )


def build_matrix_attempt_manifest(
    identity: MatrixCellArtifactIdentity,
    *,
    agent: MatrixAxisIdentity,
    execution: MatrixCellExecutionSpec,
    initial_messages: tuple[ChatMessage, ...],
) -> MatrixAttemptManifest:
    """Build and self-verify the only runtime manifest accepted for a matrix attempt."""

    manifest = MatrixAttemptManifest(
        trajectory_id=identity.artifact_trajectory_id,
        cell_identity_sha256=identity.sha256,
        agent=agent,
        policy=execution.policy,
        budget=execution.budget,
        dev_timing=execution.dev_timing,
        model_ref=execution.model_ref,
        initial_messages=initial_messages,
        device=execution.device,
        execution_context_sha256=execution.execution_context_sha256,
    )
    manifest.verify_for_identity(identity)
    return manifest


class PlannedMatrixCell(CanaryModel):
    """One phase cell classified without performing any live action."""

    identity: MatrixCellArtifactIdentity
    state: CellPlanState

    @model_validator(mode="after")
    def state_matches_attempt(self) -> PlannedMatrixCell:
        if self.state == "pending" and self.identity.attempt_index != 0:
            raise ValueError("fresh pending cells must select attempt zero")
        if self.state == "retry_pending" and self.identity.attempt_index == 0:
            raise ValueError("retry-pending cells must select a retry attempt")
        return self


class ExistingCellAttempt(CanaryModel):
    """One verified terminal attempt loaded from an immutable artifact directory."""

    identity: MatrixCellArtifactIdentity
    outcome: TrajectoryOutcome

    @model_validator(mode="after")
    def outcome_matches_artifact(self) -> ExistingCellAttempt:
        if self.outcome.trajectory_id != self.identity.artifact_trajectory_id:
            raise ValueError("attempt outcome ID differs from its artifact identity")
        return self

    @property
    def retryable_infrastructure(self) -> bool:
        return self.outcome.status in {"provider_error", "worker_error"}


class MatrixPhasePlanSummary(CanaryModel):
    """Compact typed counts and request ceilings derived from one phase plan."""

    schema_version: Literal["abstrak-matrix-phase-plan-summary.v1"] = (
        "abstrak-matrix-phase-plan-summary.v1"
    )
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    raw_study_sha256: str = Field(pattern=SHA256_PATTERN)
    spec_sha256: str = Field(pattern=SHA256_PATTERN)
    schedule_sha256: str = Field(pattern=SHA256_PATTERN)
    phase_id: str = Field(pattern=IDENTIFIER_PATTERN)
    max_calls_per_trajectory: int = Field(ge=1, le=4)
    infrastructure_retries: int = Field(ge=0, le=1)
    expected_cells: int = Field(ge=1)
    pending_cells: int = Field(ge=0)
    retry_pending_cells: int = Field(ge=0)
    resumed_cells: int = Field(ge=0)
    retry_exhausted_cells: int = Field(ge=0)
    scientific_request_ceiling: int = Field(ge=1)
    operational_request_ceiling: int = Field(ge=1)
    pending_scientific_request_ceiling: int = Field(ge=0)
    pending_operational_request_ceiling: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_and_ceilings_are_bounded(self) -> MatrixPhasePlanSummary:
        if (
            self.pending_cells
            + self.retry_pending_cells
            + self.resumed_cells
            + self.retry_exhausted_cells
            != self.expected_cells
        ):
            raise ValueError("plan state counts must cover every expected cell")
        expected_scientific = self.expected_cells * self.max_calls_per_trajectory
        expected_operational = expected_scientific * (1 + self.infrastructure_retries)
        pending_scientific = self.pending_cells * self.max_calls_per_trajectory
        pending_operational = (
            self.pending_cells * self.max_calls_per_trajectory * (1 + self.infrastructure_retries)
            + self.retry_pending_cells * self.max_calls_per_trajectory
        )
        if (
            self.scientific_request_ceiling != expected_scientific
            or self.operational_request_ceiling != expected_operational
            or self.pending_scientific_request_ceiling != pending_scientific
            or self.pending_operational_request_ceiling != pending_operational
        ):
            raise ValueError("summary request ceilings do not match its cell counts")
        return self


class MatrixPhasePlan(CanaryModel):
    """Frozen dry-run plan for exactly one deterministic matrix phase."""

    schema_version: Literal["abstrak-matrix-phase-plan.v1"] = "abstrak-matrix-phase-plan.v1"
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    raw_study_sha256: str = Field(pattern=SHA256_PATTERN)
    spec_sha256: str = Field(pattern=SHA256_PATTERN)
    schedule_sha256: str = Field(pattern=SHA256_PATTERN)
    phase_id: str = Field(pattern=IDENTIFIER_PATTERN)
    max_calls_per_trajectory: int = Field(ge=1, le=4)
    infrastructure_retries: int = Field(ge=0, le=1)
    scientific_request_ceiling: int = Field(ge=1)
    operational_request_ceiling: int = Field(ge=1)
    cells: tuple[PlannedMatrixCell, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def cells_and_ceilings_match_plan(self) -> MatrixPhasePlan:
        expected_scientific = len(self.cells) * self.max_calls_per_trajectory
        expected_operational = expected_scientific * (1 + self.infrastructure_retries)
        if self.scientific_request_ceiling != expected_scientific:
            raise ValueError("scientific request ceiling does not match planned cells")
        if self.operational_request_ceiling != expected_operational:
            raise ValueError("operational request ceiling does not match retry policy")

        trajectory_ids = tuple(item.identity.trajectory_id for item in self.cells)
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError("phase plan contains duplicate trajectory IDs")
        phase_ordinals = tuple(item.identity.cell.phase_ordinal for item in self.cells)
        if phase_ordinals != tuple(range(len(self.cells))):
            raise ValueError("phase plan cells must follow contiguous phase order")
        for item in self.cells:
            identity = item.identity
            if (
                identity.study_id != self.study_id
                or identity.raw_study_sha256 != self.raw_study_sha256
                or identity.spec_sha256 != self.spec_sha256
                or identity.schedule_sha256 != self.schedule_sha256
                or identity.phase_id != self.phase_id
            ):
                raise ValueError("planned cell identity differs from its phase plan")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)

    @property
    def summary(self) -> MatrixPhasePlanSummary:
        pending = sum(item.state == "pending" for item in self.cells)
        retry_pending = sum(item.state == "retry_pending" for item in self.cells)
        resumed = sum(item.state == "resumed" for item in self.cells)
        retry_exhausted = sum(item.state == "retry_exhausted" for item in self.cells)
        return MatrixPhasePlanSummary(
            study_id=self.study_id,
            raw_study_sha256=self.raw_study_sha256,
            spec_sha256=self.spec_sha256,
            schedule_sha256=self.schedule_sha256,
            phase_id=self.phase_id,
            max_calls_per_trajectory=self.max_calls_per_trajectory,
            infrastructure_retries=self.infrastructure_retries,
            expected_cells=len(self.cells),
            pending_cells=pending,
            retry_pending_cells=retry_pending,
            resumed_cells=resumed,
            retry_exhausted_cells=retry_exhausted,
            scientific_request_ceiling=self.scientific_request_ceiling,
            operational_request_ceiling=self.operational_request_ceiling,
            pending_scientific_request_ceiling=pending * self.max_calls_per_trajectory,
            pending_operational_request_ceiling=(
                pending * self.max_calls_per_trajectory * (1 + self.infrastructure_retries)
                + retry_pending * self.max_calls_per_trajectory
            ),
        )


class AxisIdentityResolver(Protocol):
    def __call__(self, identifier: str) -> MatrixAxisIdentity: ...


class CellExecutionResolver(Protocol):
    def __call__(self, cell: MatrixCell) -> MatrixCellExecutionSpec: ...


class ExistingCellAttemptLoader(Protocol):
    """Return identity only for a sealed terminal artifact; ``None`` means absent."""

    def __call__(
        self,
        expected: MatrixCellArtifactIdentity,
    ) -> ExistingCellAttempt | None: ...


def resolve_registered_task_identity(identifier: str) -> MatrixAxisIdentity:
    """Resolve and byte-verify one task from the shared scoped registry."""

    task = get_task_pack(identifier)
    load_task_source(identifier)
    return MatrixAxisIdentity(kind="task", id=identifier, sha256=sha256_json(task))


def resolve_registered_target_identity(identifier: str) -> MatrixAxisIdentity:
    """Resolve and byte-verify one target card from the shared scoped registry."""

    target = get_target_stack(identifier)
    load_target_card(identifier)
    return MatrixAxisIdentity(kind="target", id=identifier, sha256=sha256_json(target))


def sealed_cell_identity_loader(
    artifact_root: str | Path,
) -> ExistingCellAttemptLoader:
    """Build a read-only loader for shared sealed trajectory artifacts."""

    root = Path(artifact_root).expanduser()

    def load(expected: MatrixCellArtifactIdentity) -> ExistingCellAttempt | None:
        directory = root / expected.study_id / expected.artifact_trajectory_id
        if not directory.exists() and not directory.is_symlink():
            return None
        try:
            resolved_root = root.resolve()
            resolved_directory = directory.resolve(strict=True)
            resolved_directory.relative_to(resolved_root)
        except (OSError, ValueError) as error:
            raise TrajectoryArtifactError(
                f"trajectory artifact escapes its root: {directory}"
            ) from error
        if not resolved_directory.is_dir():
            raise TrajectoryArtifactError(f"trajectory artifact is not a directory: {directory}")
        verify_trajectory(resolved_directory)
        manifest = MatrixAttemptManifest.model_validate_json(
            (resolved_directory / "run-manifest.json").read_text(encoding="utf-8")
        )
        identity = MatrixCellArtifactIdentity.model_validate_json(
            (resolved_directory / "cell-identity.json").read_text(encoding="utf-8")
        )
        manifest.verify_for_identity(identity)
        outcome = TrajectoryOutcome.model_validate_json(
            (resolved_directory / "outcome.json").read_text(encoding="utf-8")
        )
        if outcome.trajectory_id != expected.artifact_trajectory_id:
            raise TrajectoryArtifactError("trajectory outcome ID differs from its attempt")
        events = _validate_terminal_events(
            resolved_directory,
            identity,
            manifest,
            outcome,
        )
        _validate_terminal_artifacts(
            resolved_directory,
            identity,
            manifest,
            outcome,
            events,
        )
        return ExistingCellAttempt(identity=identity, outcome=outcome)

    return load


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrajectoryArtifactError(f"invalid trajectory JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise TrajectoryArtifactError(f"trajectory JSON is not an object: {path.name}")
    return value


def _one_event(
    events: tuple[dict[str, object], ...],
    kind: str,
    turn_index: int | None,
) -> dict[str, object]:
    matches = tuple(
        event
        for event in events
        if event.get("kind") == kind and event.get("turn_index") == turn_index
    )
    if len(matches) != 1:
        raise TrajectoryArtifactError(
            f"trajectory requires exactly one {kind} event for turn {turn_index}"
        )
    return matches[0]


def _validate_terminal_events(
    directory: Path,
    identity: MatrixCellArtifactIdentity,
    manifest: MatrixAttemptManifest,
    outcome: TrajectoryOutcome,
) -> tuple[dict[str, object], ...]:
    events_directory = directory / "events"
    if not events_directory.is_dir():
        raise TrajectoryArtifactError("terminal trajectory has no event ledger")
    event_paths = sorted(events_directory.iterdir())
    if not event_paths:
        raise TrajectoryArtifactError("terminal trajectory has no event ledger")
    if any(
        path.is_symlink() or not path.is_file() or path.suffix != ".json" for path in event_paths
    ):
        raise TrajectoryArtifactError("trajectory event ledger contains an invalid entry")
    events: list[dict[str, object]] = []
    for index, path in enumerate(event_paths):
        event = _read_json_object(path)
        if path.name != f"{index:04d}.json":
            raise TrajectoryArtifactError("trajectory event filenames are not contiguous")
        if (
            event.get("schema_version") != "canary-trajectory-event.v1"
            or event.get("sequence") != index
        ):
            raise TrajectoryArtifactError("trajectory event sequence is not contiguous")
        if event.get("payload_sha256") != sha256_json(event.get("payload")):
            raise TrajectoryArtifactError("trajectory event payload hash mismatch")
        events.append(event)
    frozen_events = tuple(events)
    if (
        events[0].get("kind") != "trajectory_started"
        or sum(event.get("kind") == "trajectory_started" for event in events) != 1
    ):
        raise TrajectoryArtifactError("trajectory event ledger has no start event")
    start_payload = events[0].get("payload")
    if (
        not isinstance(start_payload, dict)
        or start_payload.get("trajectory_id") != outcome.trajectory_id
    ):
        raise TrajectoryArtifactError("trajectory start event has the wrong identity")
    if (
        events[-1].get("kind") != "trajectory_terminal"
        or sum(event.get("kind") == "trajectory_terminal" for event in events) != 1
    ):
        raise TrajectoryArtifactError("trajectory event ledger has no terminal event")
    request_turns = tuple(
        event.get("turn_index") for event in events if event.get("kind") == "request_started"
    )
    if request_turns != tuple(range(outcome.calls)):
        raise TrajectoryArtifactError("trajectory request events differ from outcome calls")

    turns_directory = directory / "turns"
    if not turns_directory.is_dir():
        raise TrajectoryArtifactError("trajectory turn ledger is incomplete")
    turn_entries = sorted(turns_directory.iterdir())
    expected_turn_names = tuple(f"{turn:04d}" for turn in range(outcome.calls))
    if tuple(path.name for path in turn_entries) != expected_turn_names or any(
        path.is_symlink() or not path.is_dir() for path in turn_entries
    ):
        raise TrajectoryArtifactError("trajectory turn ledger is incomplete")

    provider_error_turns = 0
    for turn in range(outcome.calls):
        turn_directory = turns_directory / f"{turn:04d}"
        entries = tuple(turn_directory.iterdir())
        allowed_names = {
            "request.json",
            "response.json",
            "error.json",
            "candidate.py",
            "worker-job.json",
            "worker-result.json",
        }
        if any(
            path.is_symlink() or not path.is_file() or path.name not in allowed_names
            for path in entries
        ):
            raise TrajectoryArtifactError("trajectory turn ledger contains an invalid entry")
        request_path = turn_directory / "request.json"
        response_path = turn_directory / "response.json"
        error_path = turn_directory / "error.json"
        if not request_path.is_file() or response_path.is_file() == error_path.is_file():
            raise TrajectoryArtifactError("trajectory turn ledger is incomplete")
        request = _read_json_object(request_path)
        if _one_event(frozen_events, "request_started", turn).get("payload") != request:
            raise TrajectoryArtifactError("request event payload differs from its turn artifact")
        try:
            logical_request = LogicalRequest.model_validate(request)
        except ValueError as error:
            raise TrajectoryArtifactError(
                "trajectory contains an invalid logical request"
            ) from error
        initial_count = len(manifest.initial_messages)
        if (
            logical_request.trajectory_id != manifest.trajectory_id
            or logical_request.turn_index != turn
            or logical_request.model_ref != manifest.model_ref
            or logical_request.messages[:initial_count] != manifest.initial_messages
            or (turn == 0 and logical_request.messages != manifest.initial_messages)
        ):
            raise TrajectoryArtifactError("logical request differs from its attempt manifest")
        if response_path.is_file():
            response = _read_json_object(response_path)
            if _one_event(frozen_events, "response_received", turn).get("payload") != response:
                raise TrajectoryArtifactError(
                    "response event payload differs from its turn artifact"
                )
        else:
            provider_error_turns += 1
            provider_error = _read_json_object(error_path)
            if _one_event(frozen_events, "provider_error", turn).get("payload") != provider_error:
                raise TrajectoryArtifactError(
                    "provider error event payload differs from its turn artifact"
                )

    expected_provider_errors = 1 if outcome.status == "provider_error" else 0
    if provider_error_turns != expected_provider_errors:
        raise TrajectoryArtifactError("provider error turns differ from trajectory status")
    if events[-1].get("payload") != outcome.model_dump(mode="json"):
        raise TrajectoryArtifactError("terminal event payload differs from outcome")
    if outcome.calls > identity.max_calls_per_trajectory:
        raise TrajectoryArtifactError("trajectory calls exceed its frozen Agent budget")
    if outcome.status in {"call_limit", "no_candidate"} and (
        outcome.calls != identity.max_calls_per_trajectory
    ):
        raise TrajectoryArtifactError("call-limit trajectory did not consume its frozen budget")
    if outcome.status in {"finished", "provider_error", "worker_error"} and outcome.calls == 0:
        raise TrajectoryArtifactError(f"{outcome.status} trajectory made no Agent request")
    if (
        outcome.status == "finished"
        and sum(event.get("kind") in {"agent_finished", "controller_finished"} for event in events)
        != 1
    ):
        raise TrajectoryArtifactError("finished trajectory has no matching stop event")
    if (
        outcome.status in {"call_limit", "no_candidate"}
        and sum(event.get("kind") == "call_limit" for event in events) != 1
    ):
        raise TrajectoryArtifactError("call-limit trajectory has no matching stop event")
    if (
        outcome.status == "budget_exhausted"
        and sum(event.get("kind") == "budget_exhausted" for event in events) != 1
    ):
        raise TrajectoryArtifactError("budget-exhausted trajectory has no matching stop event")
    return frozen_events


def _validate_worker_pair(
    job_path: Path,
    result_path: Path,
    identity: MatrixCellArtifactIdentity,
    manifest: MatrixAttemptManifest,
    *,
    expected_kind: Literal["dev", "sealed"],
    expected_job_id: str,
) -> tuple[WorkerJob, WorkerResult]:
    try:
        job = WorkerJob.model_validate_json(job_path.read_text(encoding="utf-8"))
        result = WorkerResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        result.verify_for_job(job)
    except (OSError, ValueError) as error:
        raise TrajectoryArtifactError("trajectory contains an invalid worker record") from error
    expected_cases = job.task.dev_cases if expected_kind == "dev" else job.task.sealed_cases
    expected_timing = manifest.dev_timing if expected_kind == "dev" else None
    if job.kind != expected_kind or job.job_id != expected_job_id:
        raise TrajectoryArtifactError(f"trajectory worker record is not a {expected_kind} job")
    if (
        sha256_json(job.task) != identity.task_sha256
        or sha256_json(job.target) != identity.target_sha256
        or job.case_ids != tuple(case.id for case in expected_cases)
        or job.timing != expected_timing
        or job.device != manifest.device
    ):
        raise TrajectoryArtifactError("trajectory worker record differs from its cell identity")
    return job, result


def _validate_candidate_snapshots(
    directory: Path,
    outcome: TrajectoryOutcome,
) -> dict[str, str]:
    candidates_directory = directory / "candidates"
    if not candidates_directory.is_dir():
        raise TrajectoryArtifactError("trajectory candidate ledger is missing")
    hashes = {
        "first": outcome.first_candidate_sha256,
        "final": outcome.final_candidate_sha256,
    }
    expected_files = {
        f"{label}.{suffix}"
        for label, digest in hashes.items()
        if digest is not None
        for suffix in ("py", "json")
    }
    actual_files = {
        path.relative_to(candidates_directory).as_posix()
        for path in candidates_directory.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files or any(
        path.is_symlink() for path in candidates_directory.rglob("*")
    ):
        raise TrajectoryArtifactError("candidate snapshots differ from trajectory outcome")

    sources: dict[str, str] = {}
    for label, digest in hashes.items():
        if digest is None:
            continue
        source = (candidates_directory / f"{label}.py").read_text(encoding="utf-8")
        actual_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        metadata = _read_json_object(candidates_directory / f"{label}.json")
        if actual_digest != digest or metadata != {"label": label, "sha256": digest}:
            raise TrajectoryArtifactError("candidate snapshot hash mismatch")
        sources[label] = source
    return sources


def _validate_terminal_artifacts(
    directory: Path,
    identity: MatrixCellArtifactIdentity,
    manifest: MatrixAttemptManifest,
    outcome: TrajectoryOutcome,
    events: tuple[dict[str, object], ...],
) -> None:
    dev_results: list[WorkerResult] = []
    for turn in range(outcome.calls):
        turn_directory = directory / "turns" / f"{turn:04d}"
        candidate_path = turn_directory / "candidate.py"
        job_path = turn_directory / "worker-job.json"
        result_path = turn_directory / "worker-result.json"
        worker_files = (candidate_path.is_file(), job_path.is_file(), result_path.is_file())
        if any(worker_files) and not all(worker_files):
            raise TrajectoryArtifactError("trajectory dev worker ledger is incomplete")
        if not all(worker_files):
            if any(
                event.get("kind") == "dev_finished" and event.get("turn_index") == turn
                for event in events
            ):
                raise TrajectoryArtifactError("dev event has no worker artifact")
            continue
        job, result = _validate_worker_pair(
            job_path,
            result_path,
            identity,
            manifest,
            expected_kind="dev",
            expected_job_id=f"{identity.artifact_trajectory_id}-turn-{turn}-dev",
        )
        source = candidate_path.read_text(encoding="utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != job.candidate_sha256:
            raise TrajectoryArtifactError("turn candidate differs from its worker job")
        if _one_event(events, "dev_finished", turn).get("payload") != result.model_dump(
            mode="json"
        ):
            raise TrajectoryArtifactError("dev event payload differs from its worker result")
        dev_results.append(result)

    if tuple(dev_results) != outcome.dev_results:
        raise TrajectoryArtifactError("dev worker artifacts differ from trajectory outcome")
    if sum(event.get("kind") == "dev_finished" for event in events) != len(dev_results):
        raise TrajectoryArtifactError("trajectory has unmatched dev events")

    has_candidate = outcome.first_candidate_sha256 is not None
    if has_candidate != bool(dev_results):
        raise TrajectoryArtifactError("trajectory candidate state differs from its dev results")
    if outcome.status in {"finished", "call_limit", "worker_error"} and not has_candidate:
        raise TrajectoryArtifactError(f"{outcome.status} trajectory has no candidate")
    if has_candidate:
        assert outcome.first_candidate_sha256 is not None
        assert outcome.final_candidate_sha256 is not None
        if outcome.first_candidate_sha256 != dev_results[0].candidate_sha256 or (
            outcome.final_candidate_sha256
            not in {result.candidate_sha256 for result in dev_results}
        ):
            raise TrajectoryArtifactError("selected candidates differ from dev worker results")

    candidate_sources = _validate_candidate_snapshots(directory, outcome)
    declared_sealed = {
        "first": outcome.first_sealed_result,
        "final": outcome.final_sealed_result,
    }
    expected_sealed_files = {
        f"{label}/{filename}"
        for label, result in declared_sealed.items()
        if result is not None
        for filename in ("worker-job.json", "worker-result.json")
    }
    sealed_directory = directory / "sealed"
    if not sealed_directory.is_dir():
        raise TrajectoryArtifactError("trajectory sealed ledger is missing")
    actual_sealed_files = {
        path.relative_to(sealed_directory).as_posix()
        for path in sealed_directory.rglob("*")
        if path.is_file()
    }
    if actual_sealed_files != expected_sealed_files or any(
        path.is_symlink() for path in sealed_directory.rglob("*")
    ):
        raise TrajectoryArtifactError("sealed worker artifacts differ from trajectory outcome")

    for label, declared_result in declared_sealed.items():
        matching_events = tuple(
            event
            for event in events
            if event.get("kind") == "sealed_finished"
            and isinstance(event.get("payload"), dict)
            and event["payload"].get("label") == label  # type: ignore[union-attr]
        )
        if declared_result is None:
            if matching_events:
                raise TrajectoryArtifactError("sealed event has no declared worker result")
            continue
        sealed_job, sealed_result = _validate_worker_pair(
            sealed_directory / label / "worker-job.json",
            sealed_directory / label / "worker-result.json",
            identity,
            manifest,
            expected_kind="sealed",
            expected_job_id=f"{identity.artifact_trajectory_id}-{label}-sealed",
        )
        if sealed_result != declared_result:
            raise TrajectoryArtifactError("sealed worker result differs from trajectory outcome")
        expected_source = candidate_sources[label]
        if sealed_job.candidate_source != expected_source:
            raise TrajectoryArtifactError("sealed worker source differs from candidate snapshot")
        expected_event_payload = {
            "label": label,
            "result": declared_result.model_dump(mode="json"),
        }
        if len(matching_events) != 1 or matching_events[0].get("payload") != expected_event_payload:
            raise TrajectoryArtifactError("sealed event differs from its worker result")

    if sum(event.get("kind") == "sealed_finished" for event in events) != sum(
        result is not None for result in declared_sealed.values()
    ):
        raise TrajectoryArtifactError("trajectory has unmatched sealed events")
    first_sealed = outcome.first_sealed_result
    final_sealed = outcome.final_sealed_result
    infrastructure_statuses = {"environment_error", "worker_error"}
    if final_sealed is not None and first_sealed is None:
        raise TrajectoryArtifactError("final sealed result exists without a first result")
    if has_candidate and first_sealed is None:
        dev_failed = (
            outcome.status == "worker_error"
            and bool(dev_results)
            and dev_results[-1].status in infrastructure_statuses
        )
        if not dev_failed:
            raise TrajectoryArtifactError("candidate trajectory is missing sealed results")
    if first_sealed is not None:
        if first_sealed.status in infrastructure_statuses and final_sealed is not None:
            raise TrajectoryArtifactError("final sealed result follows an infrastructure failure")
        if first_sealed.status not in infrastructure_statuses and final_sealed is None:
            raise TrajectoryArtifactError("candidate trajectory is missing its final sealed result")
    if outcome.status == "worker_error" and not any(
        result.status in infrastructure_statuses
        for result in (*dev_results, first_sealed, final_sealed)
        if result is not None
    ):
        raise TrajectoryArtifactError("worker-error trajectory has no infrastructure failure")


def _attempt_trajectory_id(trajectory_id: str, attempt_index: int) -> str:
    return trajectory_id if attempt_index == 0 else f"{trajectory_id}.infra-{attempt_index}"


def _resolve_axis(
    kind: AxisKind,
    identifier: str,
    resolver: AxisIdentityResolver,
) -> MatrixAxisIdentity:
    try:
        resolved = resolver(identifier)
    except Exception as error:
        raise MatrixStudyPlanError(
            f"cannot resolve {kind} {identifier!r}: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(resolved, MatrixAxisIdentity):
        raise MatrixStudyPlanError(
            f"{kind} resolver returned {type(resolved).__name__}, expected MatrixAxisIdentity"
        )
    if resolved.kind != kind or resolved.id != identifier:
        raise MatrixStudyPlanError(
            f"{kind} resolver identity mismatch for {identifier!r}: "
            f"received {resolved.kind} {resolved.id!r}"
        )
    return resolved


def _resolve_unique_axes(
    cells: tuple[MatrixCell, ...],
    *,
    resolve_task: AxisIdentityResolver,
    resolve_target: AxisIdentityResolver,
    resolve_agent: AxisIdentityResolver,
) -> tuple[
    dict[str, MatrixAxisIdentity],
    dict[str, MatrixAxisIdentity],
    dict[str, MatrixAxisIdentity],
]:
    tasks = {
        identifier: _resolve_axis("task", identifier, resolve_task)
        for identifier in dict.fromkeys(cell.task_id for cell in cells)
    }
    targets = {
        identifier: _resolve_axis("target", identifier, resolve_target)
        for identifier in dict.fromkeys(cell.target_id for cell in cells)
    }
    agents = {
        identifier: _resolve_axis("agent", identifier, resolve_agent)
        for identifier in dict.fromkeys(cell.agent_id for cell in cells)
    }
    return tasks, targets, agents


def _resolve_execution(
    cell: MatrixCell,
    phase_max_calls: int,
    resolver: CellExecutionResolver,
) -> MatrixCellExecutionSpec:
    try:
        execution = resolver(cell)
    except Exception as error:
        raise MatrixStudyPlanError(
            f"cannot resolve execution policy for {cell.trajectory_id}: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(execution, MatrixCellExecutionSpec):
        raise MatrixStudyPlanError(
            "execution resolver returned "
            f"{type(execution).__name__}, expected MatrixCellExecutionSpec"
        )
    if execution.budget.max_calls != phase_max_calls:
        raise MatrixStudyPlanError(
            f"execution budget for {cell.trajectory_id} allows "
            f"{execution.budget.max_calls} calls; phase requires {phase_max_calls}"
        )
    return execution


def _expected_cell_identity(
    *,
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    cell: MatrixCell,
    task: MatrixAxisIdentity,
    target: MatrixAxisIdentity,
    agent: MatrixAxisIdentity,
    execution: MatrixCellExecutionSpec,
    attempt_index: int,
) -> MatrixCellArtifactIdentity:
    return MatrixCellArtifactIdentity(
        study_id=pinned.spec.study_id,
        raw_study_sha256=pinned.sha256,
        spec_sha256=pinned.spec.sha256,
        schedule_sha256=schedule.sha256,
        phase_id=cell.phase_id,
        trajectory_id=cell.trajectory_id,
        artifact_trajectory_id=_attempt_trajectory_id(cell.trajectory_id, attempt_index),
        attempt_index=attempt_index,
        cell=cell,
        task_sha256=task.sha256,
        target_sha256=target.sha256,
        agent_sha256=agent.sha256,
        policy_sha256=execution.policy.sha256,
        budget_sha256=sha256_json(execution.budget),
        max_calls_per_trajectory=execution.budget.max_calls,
        dev_timing_sha256=(
            None if execution.dev_timing is None else sha256_json(execution.dev_timing)
        ),
        model_ref=execution.model_ref,
        initial_messages_sha256=execution.initial_messages_sha256,
        device=execution.device,
        execution_context_sha256=execution.execution_context_sha256,
        execution_sha256=execution.sha256,
    )


def _load_existing_attempt(
    expected: MatrixCellArtifactIdentity,
    loader: ExistingCellAttemptLoader | None,
) -> ExistingCellAttempt | None:
    if loader is None:
        return None
    try:
        actual = loader(expected)
    except Exception as error:
        raise MatrixStudyPlanError(
            f"cannot verify existing artifact for {expected.artifact_trajectory_id}: "
            f"{type(error).__name__}: {error}"
        ) from error
    if actual is None:
        return None
    if not isinstance(actual, ExistingCellAttempt):
        raise MatrixStudyPlanError(
            f"artifact attempt loader returned {type(actual).__name__} for "
            f"{expected.artifact_trajectory_id}, expected ExistingCellAttempt"
        )
    if actual.identity != expected:
        expected_payload = expected.model_dump(mode="json")
        actual_payload = actual.identity.model_dump(mode="json")
        drifted = tuple(
            key for key in expected_payload if expected_payload[key] != actual_payload.get(key)
        )
        details = ", ".join(drifted) if drifted else "unknown fields"
        raise MatrixStudyPlanError(
            f"existing artifact identity drift for {expected.artifact_trajectory_id}: {details}"
        )
    return actual


def _plan_cell(
    *,
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    cell: MatrixCell,
    task: MatrixAxisIdentity,
    target: MatrixAxisIdentity,
    agent: MatrixAxisIdentity,
    execution: MatrixCellExecutionSpec,
    infrastructure_retries: int,
    loader: ExistingCellAttemptLoader | None,
) -> PlannedMatrixCell:
    identities = tuple(
        _expected_cell_identity(
            pinned=pinned,
            schedule=schedule,
            cell=cell,
            task=task,
            target=target,
            agent=agent,
            execution=execution,
            attempt_index=attempt_index,
        )
        for attempt_index in range(1 + infrastructure_retries)
    )
    first = _load_existing_attempt(identities[0], loader)
    retry = None if infrastructure_retries == 0 else _load_existing_attempt(identities[1], loader)
    if first is None:
        if retry is not None:
            raise MatrixStudyPlanError(
                f"orphan retry artifact exists for {identities[1].artifact_trajectory_id}"
            )
        return PlannedMatrixCell(identity=identities[0], state="pending")
    if first.retryable_infrastructure:
        if infrastructure_retries == 0:
            return PlannedMatrixCell(identity=identities[0], state="retry_exhausted")
        if retry is None:
            return PlannedMatrixCell(identity=identities[1], state="retry_pending")
        state: CellPlanState = "retry_exhausted" if retry.retryable_infrastructure else "resumed"
        return PlannedMatrixCell(identity=identities[1], state=state)
    if retry is not None:
        raise MatrixStudyPlanError(
            f"unexpected retry artifact exists for {identities[1].artifact_trajectory_id}"
        )
    return PlannedMatrixCell(identity=identities[0], state="resumed")


def plan_matrix_phase(
    pinned: PinnedStudySpec,
    phase_id: str,
    *,
    resolve_task: AxisIdentityResolver,
    resolve_target: AxisIdentityResolver,
    resolve_agent: AxisIdentityResolver,
    resolve_execution: CellExecutionResolver,
    schedule: MatrixSchedule | None = None,
    load_existing_attempt: ExistingCellAttemptLoader | None = None,
) -> MatrixPhasePlan:
    """Build one phase plan without creating artifacts or invoking live systems."""

    frozen_schedule = schedule or build_matrix_schedule(pinned.spec)
    if frozen_schedule.spec != pinned.spec or frozen_schedule.spec_sha256 != pinned.spec.sha256:
        raise MatrixStudyPlanError("matrix schedule does not belong to the pinned study spec")

    phase = pinned.spec.phase(phase_id)
    cells = frozen_schedule.cells_for_phase(phase_id)
    tasks, targets, agents = _resolve_unique_axes(
        cells,
        resolve_task=resolve_task,
        resolve_target=resolve_target,
        resolve_agent=resolve_agent,
    )

    planned: list[PlannedMatrixCell] = []
    for cell in cells:
        execution = _resolve_execution(
            cell,
            phase.max_calls_per_trajectory,
            resolve_execution,
        )
        planned.append(
            _plan_cell(
                pinned=pinned,
                schedule=frozen_schedule,
                cell=cell,
                task=tasks[cell.task_id],
                target=targets[cell.target_id],
                agent=agents[cell.agent_id],
                execution=execution,
                infrastructure_retries=phase.infrastructure_retries,
                loader=load_existing_attempt,
            )
        )

    return MatrixPhasePlan(
        study_id=pinned.spec.study_id,
        raw_study_sha256=pinned.sha256,
        spec_sha256=pinned.spec.sha256,
        schedule_sha256=frozen_schedule.sha256,
        phase_id=phase_id,
        max_calls_per_trajectory=phase.max_calls_per_trajectory,
        infrastructure_retries=phase.infrastructure_retries,
        scientific_request_ceiling=frozen_schedule.phase_request_ceiling(phase_id),
        operational_request_ceiling=frozen_schedule.phase_operational_request_ceiling(phase_id),
        cells=tuple(planned),
    )
