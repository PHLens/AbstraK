"""Fixed four-turn Agent loop for the A100 R1 canary study."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol

from abstrak.canary.artifacts import TrajectoryStore
from abstrak.canary.contracts import (
    AgentBudget,
    TargetStackSpec,
    TaskPackSpec,
    TimingSpec,
    TrajectoryOutcome,
    WorkerJob,
    WorkerResult,
)
from abstrak.canary.protocol import (
    AgentProtocolError,
    format_worker_feedback,
    parse_agent_action,
    protocol_error_feedback,
)
from abstrak.providers.contracts import (
    ChatMessage,
    LogicalRequest,
    MessageRole,
    NormalizedResponse,
    ProviderCallError,
)


class CompletionClient(Protocol):
    def complete(self, request: LogicalRequest) -> NormalizedResponse: ...


class WorkerExecutor(Protocol):
    def execute(self, job: WorkerJob) -> WorkerResult: ...


UtcNow = Callable[[], datetime]
Monotonic = Callable[[], float]
DEFAULT_AGENT_BUDGET = AgentBudget()
DEFAULT_DEV_TIMING = TimingSpec(repetitions=1)


def _candidate_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _worker_failure(job: WorkerJob, error: Exception) -> WorkerResult:
    health = getattr(error, "health", None)
    return WorkerResult(
        job_id=job.job_id,
        job_sha256=job.sha256,
        input_sha256=job.input_sha256,
        candidate_sha256=job.candidate_sha256,
        status="worker_error",
        metadata={"post_job_gpu_health": health} if health is not None else {},
        error=f"{type(error).__name__}: {error}",
    )


class CanaryAgentLoop:
    """One deterministic controller; it intentionally exposes no general tool framework."""

    def __init__(
        self,
        *,
        client: CompletionClient,
        worker: WorkerExecutor,
        store: TrajectoryStore,
        monotonic: Monotonic = time.monotonic,
        utcnow: UtcNow = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.client = client
        self.worker = worker
        self.store = store
        self.monotonic = monotonic
        self.utcnow = utcnow
        self._event_sequence = 0

    def _event(self, kind: str, turn_index: int | None, payload: Any) -> None:
        self.store.append_event(self._event_sequence, kind, turn_index, payload)
        self._event_sequence += 1

    @staticmethod
    def _job(
        *,
        trajectory_id: str,
        label: str,
        kind: str,
        source: str,
        task: TaskPackSpec,
        target: TargetStackSpec,
        timing: TimingSpec | None,
        device: str,
    ) -> WorkerJob:
        cases = task.dev_cases if kind == "dev" else task.sealed_cases
        return WorkerJob(
            job_id=f"{trajectory_id}-{label}-{kind}",
            kind=kind,
            task=task,
            target=target,
            case_ids=tuple(case.id for case in cases),
            candidate_source=source,
            candidate_sha256=_candidate_hash(source),
            timing=timing,
            device=device,
        )

    def _execute(self, job: WorkerJob) -> WorkerResult:
        try:
            result = self.worker.execute(job)
        except Exception as error:
            result = _worker_failure(job, error)
        return result.verify_for_job(job)

    def run(
        self,
        *,
        trajectory_id: str,
        model_ref: str,
        initial_messages: tuple[ChatMessage, ...],
        task: TaskPackSpec,
        target: TargetStackSpec,
        budget: AgentBudget = DEFAULT_AGENT_BUDGET,
        device: str = "cuda:0",
        dev_timing: TimingSpec | None = DEFAULT_DEV_TIMING,
    ) -> TrajectoryOutcome:
        started_at = self.utcnow()
        started_clock = self.monotonic()
        history = list(initial_messages)
        calls = 0
        known_input_tokens = 0
        known_output_tokens = 0
        usage_complete = True
        dev_results: list[WorkerResult] = []
        first_source: str | None = None
        first_hash: str | None = None
        final_source: str | None = None
        final_hash: str | None = None
        terminal_status = "call_limit"
        terminal_error: str | None = None

        self._event("trajectory_started", None, {"trajectory_id": trajectory_id})
        for turn_index in range(budget.max_calls):
            if self.monotonic() - started_clock >= budget.max_wall_seconds:
                terminal_status = "budget_exhausted"
                self._event("budget_exhausted", turn_index, {})
                break
            request = LogicalRequest(
                model_ref=model_ref,
                messages=tuple(history),
                trajectory_id=trajectory_id,
                turn_index=turn_index,
            )
            calls += 1
            self._event("request_started", turn_index, request)
            try:
                response = self.client.complete(request)
            except ProviderCallError as error:
                terminal_status = "provider_error"
                terminal_error = str(error)
                if error.record.partial_usage is None:
                    usage_complete = False
                else:
                    usage = error.record.partial_usage
                    known_input_tokens += usage.input_tokens or 0
                    known_output_tokens += usage.output_tokens or 0
                    usage_complete = usage_complete and usage.core_fields_complete
                self.store.write_turn(turn_index, request=request, error=error.record)
                self._event("provider_error", turn_index, error.record)
                break

            usage = response.usage
            known_input_tokens += usage.input_tokens or 0
            known_output_tokens += usage.output_tokens or 0
            usage_complete = usage_complete and usage.core_fields_complete
            history.append(ChatMessage(role=MessageRole.ASSISTANT, content=response.text))
            self._event("response_received", turn_index, response)
            if self.monotonic() - started_clock >= budget.max_wall_seconds:
                terminal_status = "budget_exhausted"
                self.store.write_turn(turn_index, request=request, response=response)
                self._event(
                    "budget_exhausted",
                    turn_index,
                    {"phase": "provider_response"},
                )
                break
            try:
                action = parse_agent_action(response.text)
            except AgentProtocolError as error:
                self.store.write_turn(turn_index, request=request, response=response)
                history.append(
                    ChatMessage(role=MessageRole.USER, content=protocol_error_feedback(error))
                )
                self._event("action_parse_failed", turn_index, {"error": str(error)})
                continue

            candidate_source = action.candidate_source
            candidate_sha256 = _candidate_hash(candidate_source)
            if first_source is None:
                first_source = candidate_source
                first_hash = candidate_sha256
                self.store.snapshot_candidate("first", first_source, first_hash)
            final_source = candidate_source
            final_hash = candidate_sha256
            dev_job = self._job(
                trajectory_id=trajectory_id,
                label=f"turn-{turn_index}",
                kind="dev",
                source=candidate_source,
                task=task,
                target=target,
                timing=dev_timing,
                device=device,
            )
            dev_result = self._execute(dev_job)
            dev_results.append(dev_result)
            self.store.write_turn(
                turn_index,
                request=request,
                response=response,
                candidate=candidate_source,
                worker_job=dev_job,
                worker_result=dev_result,
            )
            self._event("dev_finished", turn_index, dev_result)
            if dev_result.status in {"environment_error", "worker_error"}:
                terminal_status = "worker_error"
                terminal_error = dev_result.error
                self._event(
                    "worker_terminal",
                    turn_index,
                    {"status": dev_result.status, "error": dev_result.error},
                )
                break
            if self.monotonic() - started_clock >= budget.max_wall_seconds:
                terminal_status = "budget_exhausted"
                self._event(
                    "budget_exhausted",
                    turn_index,
                    {"phase": "dev_result"},
                )
                break
            if action.decision == "finish":
                terminal_status = "finished"
                self._event("agent_finished", turn_index, {"candidate_sha256": final_hash})
                break
            history.append(
                ChatMessage(role=MessageRole.USER, content=format_worker_feedback(dev_result))
            )
        else:
            terminal_status = "call_limit" if first_source is not None else "no_candidate"
            self._event("call_limit", budget.max_calls - 1, {"calls": calls})

        if first_source is None or final_source is None or first_hash is None or final_hash is None:
            if terminal_status not in {"provider_error", "budget_exhausted"}:
                terminal_status = "no_candidate"
            first_sealed = None
            final_sealed = None
        elif terminal_status == "worker_error":
            self.store.snapshot_candidate("final", final_source, final_hash)
            first_sealed = None
            final_sealed = None
            self._event(
                "sealed_skipped",
                None,
                {"reason": "worker infrastructure is unavailable"},
            )
        else:
            self.store.snapshot_candidate("final", final_source, final_hash)
            first_job = self._job(
                trajectory_id=trajectory_id,
                label="first",
                kind="sealed",
                source=first_source,
                task=task,
                target=target,
                timing=None,
                device=device,
            )
            first_sealed = self._execute(first_job)
            self.store.write_sealed("first", first_job, first_sealed)
            self._event("sealed_finished", None, {"label": "first", "result": first_sealed})
            if first_sealed.status in {"environment_error", "worker_error"}:
                if terminal_status != "provider_error":
                    terminal_status = "worker_error"
                    terminal_error = first_sealed.error
                final_sealed = None
                self._event(
                    "sealed_skipped",
                    None,
                    {"label": "final", "reason": "worker infrastructure is unavailable"},
                )
            else:
                final_job = self._job(
                    trajectory_id=trajectory_id,
                    label="final",
                    kind="sealed",
                    source=final_source,
                    task=task,
                    target=target,
                    timing=None,
                    device=device,
                )
                final_sealed = self._execute(final_job)
                self.store.write_sealed("final", final_job, final_sealed)
                self._event("sealed_finished", None, {"label": "final", "result": final_sealed})
                if final_sealed.status in {"environment_error", "worker_error"}:
                    if terminal_status != "provider_error":
                        terminal_status = "worker_error"
                        terminal_error = final_sealed.error

        outcome = TrajectoryOutcome(
            trajectory_id=trajectory_id,
            status=terminal_status,
            calls=calls,
            known_input_tokens=known_input_tokens,
            known_output_tokens=known_output_tokens,
            usage_complete=usage_complete,
            first_candidate_sha256=first_hash,
            final_candidate_sha256=final_hash,
            dev_results=tuple(dev_results),
            first_sealed_result=first_sealed,
            final_sealed_result=final_sealed,
            started_at_utc=started_at,
            finished_at_utc=self.utcnow(),
            error=terminal_error,
        )
        self._event("trajectory_terminal", None, outcome)
        self.store.write_json("outcome.json", outcome)
        self.store.seal()
        return outcome
