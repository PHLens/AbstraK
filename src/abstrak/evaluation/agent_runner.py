"""Minimal iterative model -> KernelBench -> feedback collection loop."""

from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from abstrak.evaluation.agent_contracts import (
    AgentAttemptRecord,
    AgentModelSpec,
    KernelBenchAgentStudy,
)
from abstrak.evaluation.agent_provider import (
    AgentCompletion,
    AgentCompletionClient,
    AgentProviderError,
)
from abstrak.evaluation.agent_worker import AgentEvaluationJob
from abstrak.evaluation.contracts import (
    EvaluationResult,
    KernelBenchTask,
    Precision,
    TargetName,
)
from abstrak.evaluation.kernelbench import KernelBenchCheckout, TaskMaterial
from abstrak.providers.contracts import ChatMessage, MessageRole

RUNNABLE_OUTPUT_CONTRACT = """
Return exactly one Python code block and nothing else. The block must define the complete
`ModelNew` implementation, must be directly runnable by KernelBench, and must use the requested
backend. Do not return prose, patches, shell commands, placeholders, or partial snippets.
""".strip()

_PYTHON_BLOCK = re.compile(r"```python[ \t]*\r?\n(?P<code>.*?)```", re.DOTALL | re.IGNORECASE)


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
) -> list[ChatMessage]:
    prompt = (
        checkout.zero_shot_prompt(material, target, precision)
        + "\n"
        + RUNNABLE_OUTPUT_CONTRACT
    )
    return [ChatMessage(role=MessageRole.USER, content=prompt)]


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


def _evaluation_feedback(result: EvaluationResult, best_speedup: float | None) -> str:
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
    lines.append(
        "Return a revised complete directly runnable ModelNew as exactly one Python code block."
    )
    return "\n".join(lines)


def _parse_feedback(error: str, best_speedup: float | None) -> str:
    lines = [f"Candidate extraction failed: {error}"]
    if best_speedup is not None:
        lines.append(f"best_correct_speedup_so_far: {best_speedup:.6g}")
    lines.append(
        "Return the complete directly runnable ModelNew as exactly one Python code block and "
        "nothing else."
    )
    return "\n".join(lines)


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
        self.run_directory = Path(artifact_root).expanduser().resolve() / self.run_id

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
            input_tokens=(completion.input_tokens if completion else None),
            cached_input_tokens=(completion.cached_input_tokens if completion else None),
            output_tokens=(completion.output_tokens if completion else None),
            reasoning_tokens=(completion.reasoning_tokens if completion else None),
            provider_elapsed_ms=(
                completion.elapsed_ms if completion else provider_elapsed_ms
            ),
            response_path=(
                _relative(response_path, self.run_directory) if response_path else None
            ),
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
        messages = build_initial_messages(
            self.checkout, material, target, self.study.precision
        )
        records: list[AgentAttemptRecord] = []
        best_speedup: float | None = None
        for iteration in range(1, self.iterations + 1):
            response_path, candidate_path, log_path = self._paths(identifier, iteration)
            try:
                completion = client.complete(messages)
            except AgentProviderError as error:
                _write_json(
                    response_path,
                    {
                        "schema_version": "kernelbench-agent-provider-error.v1",
                        "error": str(error),
                        "elapsed_ms": error.elapsed_ms,
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
                        error=str(error),
                    )
                )
                break

            _write_json(response_path, _response_payload(completion))
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=completion.text))
            extracted = extract_runnable_candidate(completion.text)
            if extracted.code is None:
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
                messages.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=_parse_feedback(extracted.error or "unknown error", best_speedup),
                    )
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
            try:
                outcome = self.evaluator.evaluate(job)
            except AgentEvaluationTransportError as error:
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
            if result.correctness and result.performance_ratio is not None:
                best_speedup = max(best_speedup or 0.0, result.performance_ratio)
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
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=_evaluation_feedback(result, best_speedup),
                )
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
        }
        _write_json(raw / "run.json", run_payload)

        try:
            clients = {model.id: self.provider_factory(model) for model in self.study.models}
        except (OSError, ValueError) as error:
            raise AgentCollectionError(f"cannot initialize provider clients: {error}") from error
        records: list[AgentAttemptRecord] = []
        for model in self.study.models:
            for task in self.study.tasks:
                for target in self.study.targets:
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
        return AgentCollectionResult(
            run_directory=self.run_directory,
            attempts=len(records),
            generation_status_counts=dict(generation_counts),
            evaluation_status_counts=dict(evaluation_counts),
        )
