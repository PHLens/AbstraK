"""Minimal iterative model -> KernelBench -> feedback collection loop."""

from __future__ import annotations

import ast
import contextlib
import json
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from abstrak.canary.targets import load_target_card
from abstrak.evaluation.agent_contracts import (
    AgentAttemptRecord,
    AgentModelSpec,
    KernelBenchAgentStudy,
)
from abstrak.evaluation.agent_provider import (
    AgentCompletion,
    AgentCompletionClient,
    AgentMessage,
    AgentOutputTruncated,
    AgentProviderError,
    AgentUsage,
)
from abstrak.evaluation.agent_worker import AgentEvaluationJob
from abstrak.evaluation.contracts import (
    EvaluationResult,
    KernelBenchTask,
    Precision,
    TargetName,
)
from abstrak.evaluation.kernelbench import KernelBenchCheckout, TaskMaterial
from abstrak.providers.contracts import MessageRole

RUNNABLE_OUTPUT_CONTRACT = """
Return exactly one Python code block and nothing else. The block must define the complete
`ModelNew` implementation, must be directly runnable by KernelBench, and must use the requested
backend. Do not return prose, patches, shell commands, placeholders, or partial snippets.

Reason concisely and reserve enough of the response budget to emit the complete Python code block.
""".strip()

_PYTHON_BLOCK = re.compile(r"```python[ \t]*\r?\n(?P<code>.*?)```", re.DOTALL | re.IGNORECASE)

_TARGET_STACK_IDS: dict[TargetName, str] = {
    "triton": "triton-a100",
    "tilelang": "tilelang-a100",
    "cute": "cute-a100",
}
_DIAGNOSTIC_METADATA_KEYS = (
    "compilation_error_name",
    "compilation_error",
    "runtime_error_name",
    "runtime_error",
    "correctness_issue",
    "max_difference",
    "avg_difference",
    "correctness_trials",
    "error_during_performance",
)
_MAX_DIAGNOSTIC_CHARS = 4000


class AgentCollectionError(RuntimeError):
    """Raised when the local collector cannot start or persist a run."""


class AgentEvaluationTransportError(RuntimeError):
    def __init__(self, message: str, *, log: str = "") -> None:
        super().__init__(message)
        self.log = log


class ExtractedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str | None
    error: str | None


class AgentEvaluationOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result: EvaluationResult
    log: str = ""


class AgentEvaluator(Protocol):
    def evaluate(self, job: AgentEvaluationJob) -> AgentEvaluationOutcome: ...


class AgentCollectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_directory: Path
    attempts: int
    generation_status_counts: dict[str, int]
    evaluation_status_counts: dict[str, int]


ProviderFactory = Callable[[AgentModelSpec], AgentCompletionClient]
ProgressSink = Callable[[str], None]


def _stderr_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def default_agent_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid4().hex[:10]}"


def trajectory_id(model_id: str, task_ref: str, target: TargetName) -> str:
    return f"{model_id}--{task_ref}--{target}"


def extract_runnable_candidate(response: str) -> ExtractedCandidate:
    """Extract the exact response shape requested by the pilot prompt."""

    matches = list(_PYTHON_BLOCK.finditer(response))
    if len(matches) != 1:
        return ExtractedCandidate(
            code=None,
            error=f"expected exactly one Python code block, found {len(matches)}",
        )
    match = matches[0]
    if response[: match.start()].strip() or response[match.end() :].strip():
        return ExtractedCandidate(code=None, error="response contains text outside the code block")
    code = match.group("code").strip()
    if not code:
        return ExtractedCandidate(code=None, error="Python code block is empty")
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return ExtractedCandidate(
            code=None,
            error=f"candidate is not valid Python: {error.msg} at line {error.lineno}",
        )
    if not any(isinstance(node, ast.ClassDef) and node.name == "ModelNew" for node in tree.body):
        return ExtractedCandidate(code=None, error="candidate does not define top-level ModelNew")
    return ExtractedCandidate(code=f"{code}\n", error=None)


def build_initial_messages(
    checkout: KernelBenchCheckout,
    material: TaskMaterial,
    target: TargetName,
    precision: Precision,
) -> list[AgentMessage]:
    target_card = load_target_card(_TARGET_STACK_IDS[target])
    prompt = (
        checkout.zero_shot_prompt(material, target, precision)
        + "\n\nTARGET CONTRACT (must follow)\n"
        + target_card.strip()
        + "\n\n"
        + RUNNABLE_OUTPUT_CONTRACT
    )
    return [AgentMessage(role=MessageRole.USER, content=prompt)]


def _remote_timeout_result(job: AgentEvaluationJob, error: str) -> EvaluationResult:
    now = datetime.now(timezone.utc)
    return EvaluationResult(
        cell_id=job.cell_id,
        status="timeout",
        backend=job.target,
        precision=job.precision,
        error=error,
        started_at_utc=now,
        finished_at_utc=now,
    )


class SshAgentEvaluator:
    """Send one JSON worker job to a remote checkout over non-interactive SSH."""

    def __init__(
        self,
        *,
        host: str,
        worker_root: str,
        worker_python: str,
        port: int | None = None,
        ssh_executable: str = "ssh",
    ) -> None:
        self.host = host
        self.port = port
        self.worker_root = PurePosixPath(worker_root)
        self.worker_python = worker_python
        self.ssh_executable = ssh_executable

    @property
    def binding(self) -> dict[str, object]:
        return {
            "kind": "ssh",
            "host": self.host,
            "port": self.port,
            "worker_root": str(self.worker_root),
            "worker_python": self.worker_python,
        }

    @property
    def command(self) -> list[str]:
        remote = shlex.join(
            [
                "env",
                f"PYTHONPATH={self.worker_root / 'src'}",
                self.worker_python,
                "-m",
                "abstrak.evaluation.agent_worker",
            ]
        )
        command = [self.ssh_executable, "-o", "BatchMode=yes"]
        if self.port is not None:
            command.extend(["-p", str(self.port)])
        command.extend([self.host, remote])
        return command

    def evaluate(self, job: AgentEvaluationJob) -> AgentEvaluationOutcome:
        try:
            process = subprocess.run(
                self.command,
                input=job.model_dump_json(),
                check=False,
                capture_output=True,
                text=True,
                timeout=job.evaluator.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            log = f"{error.stdout or ''}{error.stderr or ''}"
            result = _remote_timeout_result(
                job, f"remote worker exceeded {job.evaluator.timeout_seconds}s"
            )
            return AgentEvaluationOutcome(result=result, log=log)
        except OSError as error:
            raise AgentEvaluationTransportError(
                f"cannot start SSH worker: {type(error).__name__}: {error}"
            ) from error

        log = process.stderr
        if process.returncode != 0:
            raise AgentEvaluationTransportError(
                f"SSH worker exited with status {process.returncode}",
                log=f"{log}\nworker stdout:\n{process.stdout}",
            )
        try:
            result = EvaluationResult.model_validate_json(process.stdout)
        except ValueError as error:
            raise AgentEvaluationTransportError(
                f"SSH worker returned invalid JSON: {error}",
                log=f"{log}\nworker stdout:\n{process.stdout}",
            ) from error
        if result.cell_id != job.cell_id:
            raise AgentEvaluationTransportError(
                "SSH worker result cell_id mismatch",
                log=f"{log}\nworker stdout:\n{process.stdout}",
            )
        return AgentEvaluationOutcome(result=result, log=log)


def _write_json(path: Path, value: object) -> None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path, run_directory: Path) -> str:
    return str(path.relative_to(run_directory))


def _response_payload(completion: AgentCompletion) -> dict[str, object]:
    return {
        "schema_version": "kernelbench-agent-response.v1",
        "text": completion.text,
        "protocol": completion.protocol,
        "provider_request_id": completion.provider_request_id,
        "returned_model": completion.returned_model,
        "input_tokens": completion.input_tokens,
        "cached_input_tokens": completion.cached_input_tokens,
        "output_tokens": completion.output_tokens,
        "reasoning_tokens": completion.reasoning_tokens,
        "elapsed_ms": completion.elapsed_ms,
        "sanitized_request": completion.sanitized_request,
        "raw_response": completion.raw_response,
    }


def _truncate_diagnostic(value: str, limit: int = _MAX_DIAGNOSTIC_CHARS) -> str:
    if len(value) <= limit:
        return value
    marker = "\n...[diagnostic truncated]...\n"
    remaining = limit - len(marker)
    head = remaining // 2
    return f"{value[:head]}{marker}{value[-(remaining - head) :]}"


def _diagnostic_feedback(result: EvaluationResult, worker_log: str) -> str | None:
    metadata = {
        key: result.metadata[key]
        for key in _DIAGNOSTIC_METADATA_KEYS
        if key in result.metadata and result.metadata[key] not in (None, "", [], {})
    }
    if metadata:
        rendered = json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str)
        return _truncate_diagnostic(rendered)
    if not result.correctness and not result.static_errors:
        stripped_log = worker_log.strip()
        if stripped_log:
            return _truncate_diagnostic(stripped_log)
    return None


def _incumbent_feedback(
    incumbent_code: str | None,
    latest_code: str | None,
) -> list[str]:
    if incumbent_code is None or incumbent_code == latest_code:
        return []
    return [
        "The latest attempt did not improve the best correct candidate. Continue from this "
        "incumbent implementation:",
        "```python",
        incumbent_code.rstrip(),
        "```",
    ]


def _evaluation_feedback(
    result: EvaluationResult,
    best_speedup: float | None,
    worker_log: str = "",
    *,
    incumbent_code: str | None = None,
    latest_code: str | None = None,
) -> str:
    lines = [
        "KernelBench evaluation result:",
        f"status: {result.status}",
        f"compiled: {str(result.compiled).lower()}",
        f"correct: {str(result.correctness).lower()}",
    ]
    if result.correctness and result.performance_ratio is not None:
        lines.append(f"speedup_vs_reference: {result.performance_ratio:.6g}")
    if best_speedup is not None:
        lines.append(f"best_correct_speedup_so_far: {best_speedup:.6g}")
    if result.static_errors:
        lines.append("static_errors: " + " | ".join(result.static_errors))
    if result.error:
        lines.append("error: " + result.error[:2000])
    diagnostics = _diagnostic_feedback(result, worker_log)
    if diagnostics:
        lines.append("diagnostics: " + diagnostics)
    lines.extend(_incumbent_feedback(incumbent_code, latest_code))
    lines.append(
        "Return a revised complete directly runnable ModelNew as exactly one Python code block."
    )
    return "\n".join(lines)


def _parse_feedback(
    error: str,
    best_speedup: float | None,
    *,
    incumbent_code: str | None = None,
) -> str:
    lines = [f"Candidate extraction failed: {error}"]
    if best_speedup is not None:
        lines.append(f"best_correct_speedup_so_far: {best_speedup:.6g}")
    lines.extend(_incumbent_feedback(incumbent_code, None))
    lines.append(
        "Return the complete directly runnable ModelNew as exactly one Python code block and "
        "nothing else."
    )
    return "\n".join(lines)


def _truncation_feedback(
    best_speedup: float | None,
    *,
    incumbent_code: str | None = None,
) -> str:
    lines = [
        "The previous response exhausted its output budget before emitting a final answer.",
        "Stop analysis now and immediately return the complete directly runnable ModelNew as "
        "exactly one Python code block and nothing else.",
    ]
    if best_speedup is not None:
        lines.insert(1, f"best_correct_speedup_so_far: {best_speedup:.6g}")
    lines.extend(_incumbent_feedback(incumbent_code, None))
    return "\n".join(lines)


def _bounded_history(
    initial_messages: Sequence[AgentMessage],
    completion: AgentCompletion,
    feedback: str,
) -> list[AgentMessage]:
    return [
        *initial_messages,
        AgentMessage(
            role=MessageRole.ASSISTANT,
            content=completion.text,
            reasoning_content=completion.reasoning_content,
        ),
        AgentMessage(role=MessageRole.USER, content=feedback),
    ]


class AgentCollectionRunner:
    """Collect a small sequential model x workload x target experiment."""

    def __init__(
        self,
        *,
        study: KernelBenchAgentStudy,
        checkout: KernelBenchCheckout,
        provider_factory: ProviderFactory,
        evaluator: AgentEvaluator,
        worker_kernelbench_root: str,
        artifact_root: str | Path,
        run_id: str | None = None,
        iterations: int | None = None,
        device: str = "cuda:0",
        progress: ProgressSink = _stderr_progress,
    ) -> None:
        self.study = study
        self.checkout = checkout
        self.provider_factory = provider_factory
        self.evaluator = evaluator
        self.worker_kernelbench_root = worker_kernelbench_root
        self.run_id = run_id or default_agent_run_id()
        self.iterations = study.iterations if iterations is None else iterations
        if not 1 <= self.iterations <= 100:
            raise AgentCollectionError("iterations must be between 1 and 100")
        self.device = device
        self.progress = progress
        self.run_directory = Path(artifact_root).expanduser().resolve() / self.run_id

    def _log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        self.progress(f"[agent {timestamp}] {message}")

    @contextlib.contextmanager
    def _heartbeat(self, label: str, *, interval_seconds: float = 30.0) -> Iterator[None]:
        stopped = threading.Event()
        started = time.monotonic()

        def report() -> None:
            while not stopped.wait(interval_seconds):
                elapsed = int(time.monotonic() - started)
                self._log(f"{label} still running elapsed={elapsed}s")

        thread = threading.Thread(target=report, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join()

    def _record(self, attempts_path: Path, record: AgentAttemptRecord) -> None:
        with attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def _paths(self, identifier: str, iteration: int) -> tuple[Path, Path, Path]:
        stem = f"iteration-{iteration:03d}"
        raw = self.run_directory / "raw"
        return (
            raw / "responses" / identifier / f"{stem}.json",
            raw / "candidates" / identifier / f"{stem}.py",
            raw / "worker-logs" / identifier / f"{stem}.log",
        )

    def _attempt(
        self,
        *,
        attempts_path: Path,
        model: AgentModelSpec,
        task: KernelBenchTask,
        material: TaskMaterial,
        target: TargetName,
        identifier: str,
        iteration: int,
        best_speedup: float | None,
        completion: AgentCompletion | None,
        generation_status: str,
        evaluation_status: str = "not_run",
        result: EvaluationResult | None = None,
        response_path: Path | None = None,
        candidate_path: Path | None = None,
        log_path: Path | None = None,
        provider_elapsed_ms: float | None = None,
        provider_usage: AgentUsage | None = None,
        error: str | None = None,
    ) -> AgentAttemptRecord:
        correct = bool(result and result.correctness)
        record = AgentAttemptRecord(
            run_id=self.run_id,
            trajectory_id=identifier,
            model_id=model.id,
            task_ref=task.ref,
            task_name=material.name,
            target=target,
            iteration=iteration,
            generation_status=generation_status,
            evaluation_status=evaluation_status,
            compiled=bool(result and result.compiled),
            correct=correct,
            candidate_runtime_ms=(result.kernel_runtime_ms if correct and result else None),
            reference_runtime_ms=(result.reference_runtime_ms if correct and result else None),
            speedup=(result.performance_ratio if correct and result else None),
            best_speedup=best_speedup,
            provider_request_id=(completion.provider_request_id if completion else None),
            returned_model=(completion.returned_model if completion else None),
            input_tokens=(
                completion.input_tokens
                if completion
                else provider_usage.input_tokens if provider_usage else None
            ),
            cached_input_tokens=(
                completion.cached_input_tokens
                if completion
                else provider_usage.cached_input_tokens if provider_usage else None
            ),
            output_tokens=(
                completion.output_tokens
                if completion
                else provider_usage.output_tokens if provider_usage else None
            ),
            reasoning_tokens=(
                completion.reasoning_tokens
                if completion
                else provider_usage.reasoning_tokens if provider_usage else None
            ),
            provider_elapsed_ms=(completion.elapsed_ms if completion else provider_elapsed_ms),
            response_path=(_relative(response_path, self.run_directory) if response_path else None),
            candidate_path=(
                _relative(candidate_path, self.run_directory) if candidate_path else None
            ),
            worker_log_path=(_relative(log_path, self.run_directory) if log_path else None),
            error=error,
        )
        self._record(attempts_path, record)
        return record

    def _run_trajectory(
        self,
        *,
        model: AgentModelSpec,
        client: AgentCompletionClient,
        task: KernelBenchTask,
        material: TaskMaterial,
        target: TargetName,
        attempts_path: Path,
    ) -> list[AgentAttemptRecord]:
        identifier = trajectory_id(model.id, task.ref, target)
        initial_messages = tuple(
            build_initial_messages(self.checkout, material, target, self.study.precision)
        )
        messages = list(initial_messages)
        records: list[AgentAttemptRecord] = []
        best_speedup: float | None = None
        incumbent_code: str | None = None
        for iteration in range(1, self.iterations + 1):
            response_path, candidate_path, log_path = self._paths(identifier, iteration)
            self._log(
                f"{identifier} iteration={iteration}/{self.iterations} provider request started "
                f"timeout={model.timeout_seconds:g}s"
            )

            def report_stream(
                detail: str,
                current_identifier: str = identifier,
                current_iteration: int = iteration,
            ) -> None:
                self._log(f"{current_identifier} iteration={current_iteration} provider {detail}")

            try:
                with self._heartbeat(f"{identifier} iteration={iteration} provider request"):
                    completion = client.complete(messages, progress=report_stream)
            except AgentOutputTruncated as error:
                self._log(
                    f"{identifier} iteration={iteration} provider output truncated; "
                    "continuing with a code-only retry turn"
                )
                _write_json(
                    response_path,
                    {
                        "schema_version": "kernelbench-agent-provider-error.v1",
                        "error_kind": "output_truncated",
                        "error": str(error),
                        "elapsed_ms": error.elapsed_ms,
                        "sanitized_request": error.sanitized_request,
                        "raw_response": error.raw_response,
                    },
                )
                records.append(
                    self._attempt(
                        attempts_path=attempts_path,
                        model=model,
                        task=task,
                        material=material,
                        target=target,
                        identifier=identifier,
                        iteration=iteration,
                        best_speedup=best_speedup,
                        completion=None,
                        generation_status="output_truncated",
                        response_path=response_path,
                        provider_elapsed_ms=error.elapsed_ms,
                        provider_usage=error.usage,
                        error=str(error),
                    )
                )
                retry_feedback = _truncation_feedback(
                    best_speedup,
                    incumbent_code=incumbent_code,
                )
                # The original request is a single user prompt. Fold the retry instruction into it
                # because there is no valid assistant turn to retain after an empty response.
                messages = [
                    initial_messages[0].model_copy(
                        update={
                            "content": f"{initial_messages[0].content}\n\n{retry_feedback}",
                        }
                    )
                ]
                continue
            except AgentProviderError as error:
                self._log(f"{identifier} iteration={iteration} provider failed: {error}")
                _write_json(
                    response_path,
                    {
                        "schema_version": "kernelbench-agent-provider-error.v1",
                        "error": str(error),
                        "elapsed_ms": error.elapsed_ms,
                        "sanitized_request": error.sanitized_request,
                        "raw_response": error.raw_response,
                    },
                )
                records.append(
                    self._attempt(
                        attempts_path=attempts_path,
                        model=model,
                        task=task,
                        material=material,
                        target=target,
                        identifier=identifier,
                        iteration=iteration,
                        best_speedup=best_speedup,
                        completion=None,
                        generation_status="provider_error",
                        response_path=response_path,
                        provider_elapsed_ms=error.elapsed_ms,
                        provider_usage=error.usage,
                        error=str(error),
                    )
                )
                break

            self._log(
                f"{identifier} iteration={iteration} provider completed "
                f"elapsed={completion.elapsed_ms / 1000:.1f}s "
                f"output_tokens={completion.output_tokens}"
            )
            _write_json(response_path, _response_payload(completion))
            extracted = extract_runnable_candidate(completion.text)
            if extracted.code is None:
                self._log(
                    f"{identifier} iteration={iteration} candidate parse failed: {extracted.error}"
                )
                records.append(
                    self._attempt(
                        attempts_path=attempts_path,
                        model=model,
                        task=task,
                        material=material,
                        target=target,
                        identifier=identifier,
                        iteration=iteration,
                        best_speedup=best_speedup,
                        completion=completion,
                        generation_status="parse_failure",
                        response_path=response_path,
                        error=extracted.error,
                    )
                )
                messages = _bounded_history(
                    initial_messages,
                    completion,
                    _parse_feedback(
                        extracted.error or "unknown error",
                        best_speedup,
                        incumbent_code=incumbent_code,
                    ),
                )
                continue

            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(extracted.code, encoding="utf-8")
            job = AgentEvaluationJob(
                cell_id=f"{identifier}--i{iteration:03d}",
                task_level=task.level,
                problem_id=task.problem_id,
                target=target,
                precision=self.study.precision,
                candidate_source=extracted.code,
                kernelbench_root=self.worker_kernelbench_root,
                device=self.device,
                evaluator=self.study.evaluator,
            )
            self._log(
                f"{identifier} iteration={iteration} SSH evaluation started "
                f"timeout={self.study.evaluator.timeout_seconds}s"
            )
            try:
                with self._heartbeat(f"{identifier} iteration={iteration} SSH evaluation"):
                    outcome = self.evaluator.evaluate(job)
            except AgentEvaluationTransportError as error:
                self._log(f"{identifier} iteration={iteration} SSH evaluation failed: {error}")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(error.log, encoding="utf-8")
                records.append(
                    self._attempt(
                        attempts_path=attempts_path,
                        model=model,
                        task=task,
                        material=material,
                        target=target,
                        identifier=identifier,
                        iteration=iteration,
                        best_speedup=best_speedup,
                        completion=completion,
                        generation_status="generated",
                        evaluation_status="transport_error",
                        response_path=response_path,
                        candidate_path=candidate_path,
                        log_path=log_path,
                        error=str(error),
                    )
                )
                break

            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(outcome.log, encoding="utf-8")
            result = outcome.result
            _write_json(log_path.with_suffix(".result.json"), result)
            if result.correctness:
                if incumbent_code is None:
                    incumbent_code = extracted.code
                if result.performance_ratio is not None and (
                    best_speedup is None or result.performance_ratio > best_speedup
                ):
                    best_speedup = result.performance_ratio
                    incumbent_code = extracted.code
            self._log(
                f"{identifier} iteration={iteration} evaluation completed "
                f"status={result.status} compiled={result.compiled} "
                f"correct={result.correctness} speedup={result.performance_ratio} "
                f"best={best_speedup}"
            )
            records.append(
                self._attempt(
                    attempts_path=attempts_path,
                    model=model,
                    task=task,
                    material=material,
                    target=target,
                    identifier=identifier,
                    iteration=iteration,
                    best_speedup=best_speedup,
                    completion=completion,
                    generation_status="generated",
                    evaluation_status=result.status,
                    result=result,
                    response_path=response_path,
                    candidate_path=candidate_path,
                    log_path=log_path,
                    error=result.error,
                )
            )
            messages = _bounded_history(
                initial_messages,
                completion,
                _evaluation_feedback(
                    result,
                    best_speedup,
                    outcome.log,
                    incumbent_code=incumbent_code,
                    latest_code=extracted.code,
                ),
            )
        return records

    def run(self) -> AgentCollectionResult:
        try:
            self.run_directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise AgentCollectionError(
                f"run directory already exists: {self.run_directory}"
            ) from None
        raw = self.run_directory / "raw"
        raw.mkdir()
        attempts_path = raw / "attempts.jsonl"
        attempts_path.touch()
        materials = {task.ref: self.checkout.load_task(task) for task in self.study.tasks}
        run_payload: dict[str, object] = {
            "schema_version": "kernelbench-agent-run.v1",
            "run_id": self.run_id,
            "study": self.study.model_dump(mode="json"),
            "study_sha256": self.study.sha256,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "iterations": self.iterations,
            "trajectory_count": self.study.trajectory_count,
            "worker": {
                **(
                    self.evaluator.binding
                    if isinstance(self.evaluator, SshAgentEvaluator)
                    else {"kind": "injected"}
                ),
                "kernelbench_root": self.worker_kernelbench_root,
                "device": self.device,
            },
        }
        _write_json(raw / "run.json", run_payload)
        self._log(
            f"run={self.run_id} started trajectories={self.study.trajectory_count} "
            f"iterations={self.iterations} max_requests="
            f"{self.study.trajectory_count * self.iterations}"
        )

        try:
            clients = {model.id: self.provider_factory(model) for model in self.study.models}
        except (OSError, ValueError) as error:
            raise AgentCollectionError(f"cannot initialize provider clients: {error}") from error
        records: list[AgentAttemptRecord] = []
        trajectory_index = 0
        for model in self.study.models:
            for task in self.study.tasks:
                for target in self.study.targets:
                    trajectory_index += 1
                    identifier = trajectory_id(model.id, task.ref, target)
                    self._log(
                        f"trajectory={trajectory_index}/{self.study.trajectory_count} "
                        f"started id={identifier}"
                    )
                    records.extend(
                        self._run_trajectory(
                            model=model,
                            client=clients[model.id],
                            task=task,
                            material=materials[task.ref],
                            target=target,
                            attempts_path=attempts_path,
                        )
                    )
                    self._log(
                        f"trajectory={trajectory_index}/{self.study.trajectory_count} "
                        f"completed id={identifier}"
                    )

        generation_counts = Counter(record.generation_status for record in records)
        evaluation_counts = Counter(record.evaluation_status for record in records)
        run_payload.update(
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "attempt_count": len(records),
                "generation_status_counts": dict(sorted(generation_counts.items())),
                "evaluation_status_counts": dict(sorted(evaluation_counts.items())),
            }
        )
        _write_json(raw / "run.json", run_payload)
        self._log(
            f"run={self.run_id} completed attempts={len(records)} "
            f"generation={dict(sorted(generation_counts.items()))} "
            f"evaluation={dict(sorted(evaluation_counts.items()))}"
        )
        return AgentCollectionResult(
            run_directory=self.run_directory,
            attempts=len(records),
            generation_status_counts=dict(generation_counts),
            evaluation_status_counts=dict(evaluation_counts),
        )
