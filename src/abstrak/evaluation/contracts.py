"""Strict contracts for the single-turn KernelBench screening study."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from abstrak.providers.contracts import sha256_json

TargetName = Literal["triton", "tilelang", "cute"]
Precision = Literal["fp16", "bf16", "fp32"]


class StudyError(ValueError):
    """Raised when a study manifest or its referenced inputs are invalid."""


class StudyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class KernelBenchSource(StudyModel):
    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    require_clean_checkout: bool = True


class KernelBenchTask(StudyModel):
    level: int = Field(ge=1, le=4)
    problem_id: int = Field(ge=1)
    stratum: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")

    @property
    def ref(self) -> str:
        return f"level{self.level}-problem{self.problem_id}"


class NaiveGenerationConfig(StudyModel):
    max_completion_tokens: int = Field(default=8192, ge=256, le=65536)
    temperature: float = Field(default=0.0, ge=0, le=2)
    repetitions: Literal[1] = 1


class KernelBenchEvaluatorConfig(StudyModel):
    num_correct_trials: int = Field(default=5, ge=1, le=100)
    num_perf_trials: int = Field(default=100, ge=1, le=10000)
    timing_method: Literal["cuda_event", "do_bench", "do_bench_impl", "host_time"] = "cuda_event"
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    excessive_speedup_threshold: float = Field(default=10.0, gt=1)
    static_check: bool = True


class KernelBenchNaiveStudy(StudyModel):
    schema_version: Literal["kernelbench-naive-study.v1"] = "kernelbench-naive-study.v1"
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    source: KernelBenchSource
    profiles: tuple[str, ...] = Field(min_length=1)
    targets: tuple[TargetName, ...] = Field(min_length=1)
    tasks: tuple[KernelBenchTask, ...] = Field(min_length=1)
    precision: Precision = "fp16"
    prompt_mode: Literal["kernelbench_zero_shot"] = "kernelbench_zero_shot"
    include_hardware_prompt: Literal[False] = False
    generation: NaiveGenerationConfig = Field(default_factory=NaiveGenerationConfig)
    evaluator: KernelBenchEvaluatorConfig = Field(default_factory=KernelBenchEvaluatorConfig)

    @field_validator("profiles", "targets")
    @classmethod
    def require_unique_strings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("values must be unique")
        if any(not value for value in values):
            raise ValueError("values cannot be empty")
        return values

    @field_validator("tasks")
    @classmethod
    def require_unique_tasks(
        cls, values: tuple[KernelBenchTask, ...]
    ) -> tuple[KernelBenchTask, ...]:
        refs = [task.ref for task in values]
        if len(refs) != len(set(refs)):
            raise ValueError("tasks must be unique by level and problem_id")
        return values

    @model_validator(mode="after")
    def require_backend_precision_compatibility(self) -> KernelBenchNaiveStudy:
        if "tilelang" in self.targets and self.precision == "fp32":
            raise ValueError("KernelBench TileLang requires fp16 or bf16")
        return self

    @property
    def matrix_size(self) -> int:
        return len(self.profiles) * len(self.targets) * len(self.tasks)

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class CellSpec(StudyModel):
    schema_version: Literal["kernelbench-naive-cell.v1"] = "kernelbench-naive-cell.v1"
    cell_id: str
    study_id: str
    study_sha256: str
    profile: str
    target: TargetName
    precision: Precision
    task: KernelBenchTask
    task_name: str
    task_source_sha256: str
    prompt_sha256: str
    replicate: Literal[0] = 0


class GenerationRecord(StudyModel):
    schema_version: Literal["kernelbench-naive-generation.v1"] = "kernelbench-naive-generation.v1"
    cell_id: str
    status: Literal["generated", "provider_error", "no_code_block"]
    request_id: str
    response_model: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    elapsed_ms: float | None = Field(default=None, ge=0)
    candidate_sha256: str | None = None


class EvaluationRequest(StudyModel):
    schema_version: Literal["kernelbench-naive-evaluation-request.v1"] = (
        "kernelbench-naive-evaluation-request.v1"
    )
    cell_id: str
    kernelbench_commit: str
    python_executable: str
    device: str
    evaluator: KernelBenchEvaluatorConfig
    requested_at_utc: datetime


class EvaluationResult(StudyModel):
    schema_version: Literal["kernelbench-naive-evaluation.v1"] = "kernelbench-naive-evaluation.v1"
    cell_id: str
    status: Literal[
        "evaluated",
        "no_candidate",
        "static_check_failed",
        "timeout",
        "environment_error",
        "harness_error",
    ]
    backend: TargetName
    precision: Precision
    compiled: bool = False
    correctness: bool = False
    kernel_runtime_ms: float | None = Field(default=None, gt=0)
    reference_runtime_ms: float | None = Field(default=None, gt=0)
    performance_ratio: float | None = Field(default=None, gt=0)
    fast_0: bool = False
    fast_1: bool = False
    fast_2: bool = False
    static_errors: tuple[str, ...] = ()
    static_warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    started_at_utc: datetime
    finished_at_utc: datetime


def _validation_summary(error: ValidationError) -> str:
    issues: list[str] = []
    for issue in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(segment) for segment in issue["loc"])
        issues.append(f"{location}: {issue['msg']}")
    return "; ".join(issues)


def load_study(path: str | Path) -> KernelBenchNaiveStudy:
    study_path = Path(path).expanduser()
    try:
        payload = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise StudyError(f"cannot read {study_path}: {error}") from error
    if not isinstance(payload, dict):
        raise StudyError(f"{study_path} must contain one YAML mapping")
    try:
        return KernelBenchNaiveStudy.model_validate(payload)
    except ValidationError as error:
        raise StudyError(f"invalid {study_path}: {_validation_summary(error)}") from None
