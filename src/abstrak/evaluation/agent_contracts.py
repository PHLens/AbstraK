"""Small contracts for the exploratory iterative KernelBench harness."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from abstrak.evaluation.contracts import (
    KernelBenchEvaluatorConfig,
    KernelBenchSource,
    KernelBenchTask,
    Precision,
    StudyError,
    StudyModel,
    TargetName,
    _validation_summary,
)
from abstrak.providers.contracts import sha256_json


class AgentModelSpec(StudyModel):
    """One model endpoint used by the exploratory harness."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    protocol: Literal["chat_completions", "responses"]
    litellm_provider: str = Field(min_length=1)
    api_model: str = Field(min_length=1)
    api_key_env: str = Field(pattern=r"^[A-Z_][A-Z0-9_]*$")
    base_url_env: str | None = Field(default=None, pattern=r"^[A-Z_][A-Z0-9_]*$")
    timeout_seconds: float = Field(default=600.0, gt=0, le=3600)


class AgentGenerationConfig(StudyModel):
    max_output_tokens: int = Field(default=16384, ge=256, le=65536)
    reasoning_effort: Literal["xhigh"] = "xhigh"
    temperature: None = None
    top_p: None = None


class KernelBenchAgentStudy(StudyModel):
    """The deliberately small model x task x target pilot matrix."""

    schema_version: Literal["kernelbench-agent-study.v1"] = "kernelbench-agent-study.v1"
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    source: KernelBenchSource
    models: tuple[AgentModelSpec, ...] = Field(min_length=1)
    targets: tuple[TargetName, ...] = Field(min_length=1)
    tasks: tuple[KernelBenchTask, ...] = Field(min_length=1)
    precision: Precision = "fp16"
    iterations: int = Field(default=4, ge=1, le=100)
    generation: AgentGenerationConfig = Field(default_factory=AgentGenerationConfig)
    evaluator: KernelBenchEvaluatorConfig = Field(default_factory=KernelBenchEvaluatorConfig)

    @field_validator("models")
    @classmethod
    def models_are_unique(cls, values: tuple[AgentModelSpec, ...]) -> tuple[AgentModelSpec, ...]:
        identifiers = tuple(item.id for item in values)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("models must be unique by id")
        return values

    @field_validator("targets")
    @classmethod
    def targets_are_unique(cls, values: tuple[TargetName, ...]) -> tuple[TargetName, ...]:
        if len(values) != len(set(values)):
            raise ValueError("targets must be unique")
        return values

    @field_validator("tasks")
    @classmethod
    def tasks_are_unique(cls, values: tuple[KernelBenchTask, ...]) -> tuple[KernelBenchTask, ...]:
        refs = tuple(item.ref for item in values)
        if len(refs) != len(set(refs)):
            raise ValueError("tasks must be unique by level and problem_id")
        return values

    @model_validator(mode="after")
    def target_precision_is_supported(self) -> KernelBenchAgentStudy:
        if "tilelang" in self.targets and self.precision == "fp32":
            raise ValueError("KernelBench TileLang requires fp16 or bf16")
        return self

    @property
    def trajectory_count(self) -> int:
        return len(self.models) * len(self.targets) * len(self.tasks)

    @property
    def request_count(self) -> int:
        return self.trajectory_count * self.iterations

    @property
    def sha256(self) -> str:
        return sha256_json(self)


GenerationStatus = Literal["generated", "provider_error", "parse_failure"]
EvaluationStatus = Literal[
    "not_run",
    "evaluated",
    "static_check_failed",
    "timeout",
    "environment_error",
    "harness_error",
    "transport_error",
]


class AgentAttemptRecord(StudyModel):
    """One model call and its optional immediate KernelBench evaluation."""

    schema_version: Literal["kernelbench-agent-attempt.v1"] = "kernelbench-agent-attempt.v1"
    run_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    task_ref: str = Field(min_length=1)
    task_name: str = Field(min_length=1)
    target: TargetName
    iteration: int = Field(ge=1)
    generation_status: GenerationStatus
    evaluation_status: EvaluationStatus = "not_run"
    compiled: bool = False
    correct: bool = False
    candidate_runtime_ms: float | None = Field(default=None, gt=0)
    reference_runtime_ms: float | None = Field(default=None, gt=0)
    speedup: float | None = Field(default=None, gt=0)
    best_speedup: float | None = Field(default=None, gt=0)
    provider_request_id: str | None = None
    returned_model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    provider_elapsed_ms: float | None = Field(default=None, ge=0)
    response_path: str | None = None
    candidate_path: str | None = None
    worker_log_path: str | None = None
    error: str | None = None

    @field_validator(
        "candidate_runtime_ms",
        "reference_runtime_ms",
        "speedup",
        "best_speedup",
        "provider_elapsed_ms",
    )
    @classmethod
    def numeric_values_are_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("numeric attempt values must be finite")
        return value

    @model_validator(mode="after")
    def performance_requires_a_correct_evaluation(self) -> AgentAttemptRecord:
        has_performance = any(
            value is not None
            for value in (
                self.candidate_runtime_ms,
                self.reference_runtime_ms,
                self.speedup,
            )
        )
        if has_performance and not self.correct:
            raise ValueError("only correct candidates may expose performance")
        if self.correct and self.evaluation_status != "evaluated":
            raise ValueError("correct candidates require an evaluated result")
        if self.speedup is not None and (
            self.candidate_runtime_ms is None or self.reference_runtime_ms is None
        ):
            raise ValueError("speedup requires candidate and reference runtime")
        if self.generation_status != "generated" and self.evaluation_status != "not_run":
            raise ValueError("failed generation cannot have an evaluation")
        return self


def load_agent_study(path: str | Path) -> KernelBenchAgentStudy:
    study_path = Path(path).expanduser()
    try:
        payload = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise StudyError(f"cannot read {study_path}: {error}") from error
    if not isinstance(payload, dict):
        raise StudyError(f"{study_path} must contain one YAML mapping")
    try:
        return KernelBenchAgentStudy.model_validate(payload)
    except ValidationError as error:
        raise StudyError(f"invalid {study_path}: {_validation_summary(error)}") from None
