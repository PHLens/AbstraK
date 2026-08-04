"""JSON-stdin KernelBench candidate worker for remote agent evaluations."""

from __future__ import annotations

import contextlib
import json
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from abstrak.evaluation.contracts import (
    EvaluationResult,
    KernelBenchEvaluatorConfig,
    Precision,
    TargetName,
)
from abstrak.evaluation.worker import evaluate_kernelbench_task_candidate


class AgentEvaluationJob(BaseModel):
    """One self-contained evaluation job sent to a GPU worker over stdin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["kernelbench-agent-evaluation-job.v1"] = (
        "kernelbench-agent-evaluation-job.v1"
    )
    cell_id: str = Field(min_length=1)
    task_level: int = Field(ge=1, le=4)
    problem_id: int = Field(ge=1)
    target: TargetName
    precision: Precision
    candidate_source: str = Field(min_length=1)
    kernelbench_root: str = Field(min_length=1)
    device: str = Field(min_length=1)
    evaluator: KernelBenchEvaluatorConfig


def evaluate_agent_job(job: AgentEvaluationJob) -> EvaluationResult:
    """Evaluate a validated job using the shared KernelBench worker path."""

    return evaluate_kernelbench_task_candidate(
        cell_id=job.cell_id,
        task_level=job.task_level,
        problem_id=job.problem_id,
        target=job.target,
        precision=job.precision,
        candidate_source=job.candidate_source,
        kernelbench_root=job.kernelbench_root,
        device=job.device,
        num_correct_trials=job.evaluator.num_correct_trials,
        num_perf_trials=job.evaluator.num_perf_trials,
        timing_method=job.evaluator.timing_method,
        excessive_speedup_threshold=job.evaluator.excessive_speedup_threshold,
        static_check=job.evaluator.static_check,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments:
        print("agent worker accepts one JSON job on stdin and no arguments", file=sys.stderr)
        return 2
    try:
        job = AgentEvaluationJob.model_validate_json(sys.stdin.read())
    except ValidationError as error:
        print(f"invalid agent evaluation job: {error}", file=sys.stderr)
        return 2
    with contextlib.redirect_stdout(sys.stderr):
        result = evaluate_agent_job(job)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
