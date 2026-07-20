"""Strict immutable contracts for the A100 R1 canary harness."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from abstrak.providers.contracts import sha256_json

IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
ParameterValue = int | float | str | bool


class CanaryModel(BaseModel):
    """Base class for values that enter hashed experiment artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_relative_asset_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("must be a safe relative POSIX asset path")
    return value


class InputCaseSpec(CanaryModel):
    """One deterministic input case; task packs assign it to dev or sealed."""

    id: str = Field(pattern=IDENTIFIER_PATTERN)
    kind: Literal["random", "zero", "constant"]
    seed: int = Field(ge=0, le=2**63 - 1)
    value: float | None = None

    @model_validator(mode="after")
    def value_matches_kind(self) -> InputCaseSpec:
        if self.kind == "constant":
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("constant cases require one finite value")
        elif self.value is not None:
            raise ValueError(f"{self.kind} cases cannot declare value")
        return self


class PublicTaskSpec(CanaryModel):
    """The only task representation that may be injected into an Agent prompt."""

    id: str = Field(pattern=IDENTIFIER_PATTERN)
    specification: str = Field(min_length=1)
    dtype: Literal["fp16", "bf16", "fp32"]
    reference_precision: Literal["fp32"] = "fp32"
    input_shapes: tuple[tuple[int, ...], ...] = Field(min_length=1)
    parameters: tuple[tuple[str, ParameterValue], ...] = ()
    init_args: tuple[ParameterValue, ...] = ()
    atol: float = Field(gt=0)
    rtol: float = Field(gt=0)


class TaskPackSpec(CanaryModel):
    """Frozen public task semantics plus private case partition references."""

    schema_version: Literal["canary-task-pack.v1"] = "canary-task-pack.v1"
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    specification: str = Field(min_length=1)
    source_path: str
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    dtype: Literal["fp16", "bf16", "fp32"]
    reference_precision: Literal["fp32"] = "fp32"
    input_shapes: tuple[tuple[int, ...], ...] = Field(min_length=1)
    parameters: tuple[tuple[str, ParameterValue], ...] = ()
    init_args: tuple[ParameterValue, ...] = ()
    atol: float = Field(gt=0)
    rtol: float = Field(gt=0)
    fallback_policy: Literal["forbid_framework_ops", "allow_framework_ops"]
    dev_cases: tuple[InputCaseSpec, ...] = Field(min_length=1)
    sealed_cases: tuple[InputCaseSpec, ...] = Field(min_length=1)

    @field_validator("source_path")
    @classmethod
    def source_path_is_safe(cls, value: str) -> str:
        return _validate_relative_asset_path(value)

    @field_validator("input_shapes")
    @classmethod
    def shapes_are_positive(
        cls, value: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]:
        if any(not shape or any(dimension <= 0 for dimension in shape) for shape in value):
            raise ValueError("input shapes must contain only positive dimensions")
        return value

    @field_validator("parameters")
    @classmethod
    def parameters_are_finite(
        cls, value: tuple[tuple[str, ParameterValue], ...]
    ) -> tuple[tuple[str, ParameterValue], ...]:
        names = [name for name, _ in value]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        if any(isinstance(item, float) and not math.isfinite(item) for _, item in value):
            raise ValueError("float parameters must be finite")
        return value

    @model_validator(mode="after")
    def case_ids_are_unique(self) -> TaskPackSpec:
        identifiers = [case.id for case in (*self.dev_cases, *self.sealed_cases)]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("case IDs must be unique across dev and sealed splits")
        return self

    @property
    def all_cases(self) -> tuple[InputCaseSpec, ...]:
        return (*self.dev_cases, *self.sealed_cases)

    def cases_by_id(self) -> dict[str, InputCaseSpec]:
        return {case.id: case for case in self.all_cases}

    @property
    def parameter_map(self) -> dict[str, ParameterValue]:
        return dict(self.parameters)

    def public_view(self) -> PublicTaskSpec:
        """Return a whitelist-based prompt payload with no case or source metadata."""

        return PublicTaskSpec(
            id=self.id,
            specification=self.specification,
            dtype=self.dtype,
            reference_precision=self.reference_precision,
            input_shapes=self.input_shapes,
            parameters=self.parameters,
            init_args=self.init_args,
            atol=self.atol,
            rtol=self.rtol,
        )


class TargetStackSpec(CanaryModel):
    """One complete target stack and its frozen Agent-visible assets."""

    schema_version: Literal["canary-target-stack.v1"] = "canary-target-stack.v1"
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    backend: Literal["triton", "tilelang", "cute", "cuda"]
    version: str = Field(min_length=1)
    card_path: str
    card_sha256: str = Field(pattern=SHA256_PATTERN)
    adapter: str = Field(pattern=IDENTIFIER_PATTERN)
    allowed_assets: tuple[str, ...] = ()
    oracle_path: str | None = None
    oracle_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @field_validator("card_path", "oracle_path")
    @classmethod
    def asset_paths_are_safe(cls, value: str | None) -> str | None:
        return None if value is None else _validate_relative_asset_path(value)

    @field_validator("allowed_assets")
    @classmethod
    def allowed_asset_paths_are_safe(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(_validate_relative_asset_path(value) for value in values)
        if len(validated) != len(set(validated)):
            raise ValueError("allowed assets must be unique")
        return validated

    @model_validator(mode="after")
    def oracle_reference_is_complete(self) -> TargetStackSpec:
        if (self.oracle_path is None) != (self.oracle_sha256 is None):
            raise ValueError("oracle_path and oracle_sha256 must be supplied together")
        return self


class TimingSpec(CanaryModel):
    """One process-local timing protocol."""

    method: Literal["cuda_event"] = "cuda_event"
    warmup_runs: int = Field(default=5, ge=1, le=1000)
    trial_runs: int = Field(default=100, ge=1, le=10000)
    repetitions: int = Field(default=3, ge=1, le=20)
    max_cv: float = Field(default=0.05, gt=0, le=1)


class AgentBudget(CanaryModel):
    """Fixed resource guardrails for the synchronous rapid-study loop."""

    max_calls: int = Field(default=4, ge=1, le=4)
    max_completion_tokens_per_call: int = Field(default=8192, ge=256, le=65536)
    max_wall_seconds: float = Field(default=1200.0, gt=0, le=86400)


class WorkerJob(CanaryModel):
    """Canonical controller-to-worker request for one candidate process."""

    schema_version: Literal["canary-worker-job.v1"] = "canary-worker-job.v1"
    job_id: str = Field(pattern=IDENTIFIER_PATTERN)
    kind: Literal["dev", "sealed", "oracle", "baseline"]
    task: TaskPackSpec
    target: TargetStackSpec
    case_ids: tuple[str, ...] = Field(min_length=1)
    candidate_source: str = Field(min_length=1)
    candidate_sha256: str = Field(pattern=SHA256_PATTERN)
    timing: TimingSpec | None = None
    device: str = Field(default="cuda:0", pattern=r"^cuda:[0-9]+$")

    @model_validator(mode="after")
    def references_are_consistent(self) -> WorkerJob:
        actual_candidate_hash = hashlib.sha256(self.candidate_source.encode("utf-8")).hexdigest()
        if actual_candidate_hash != self.candidate_sha256:
            raise ValueError("candidate_source does not match candidate_sha256")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("case_ids must be unique")
        known = self.task.cases_by_id()
        unknown = sorted(set(self.case_ids) - set(known))
        if unknown:
            raise ValueError(f"unknown task case IDs: {', '.join(unknown)}")
        split = self.task.dev_cases if self.kind == "dev" else self.task.sealed_cases
        allowed = {case.id for case in split}
        disallowed = sorted(set(self.case_ids) - allowed)
        if disallowed:
            raise ValueError(f"{self.kind} job cannot use case IDs: {', '.join(disallowed)}")
        return self

    @property
    def input_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"job_id"})
        return sha256_json(payload)

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class CaseResult(CanaryModel):
    """Structured result for one explicit correctness case."""

    case_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: Literal[
        "pass",
        "wrong_result",
        "runtime_error",
        "input_mutation",
        "nonfinite_output",
    ]
    correct: bool
    max_abs_error: float | None = Field(default=None, ge=0)
    max_rel_error: float | None = Field(default=None, ge=0)
    output_finite: bool
    inputs_unchanged: bool
    error: str | None = None

    @model_validator(mode="after")
    def status_matches_fields(self) -> CaseResult:
        if self.status == "pass":
            if not self.correct or not self.output_finite or not self.inputs_unchanged:
                raise ValueError("passing cases must be correct, finite, and mutation-free")
            if self.error is not None:
                raise ValueError("passing cases cannot contain an error")
        elif self.correct:
            raise ValueError("non-passing cases cannot be correct")
        if self.status == "input_mutation" and self.inputs_unchanged:
            raise ValueError("input_mutation requires inputs_unchanged=false")
        if self.status == "nonfinite_output" and self.output_finite:
            raise ValueError("nonfinite_output requires output_finite=false")
        if self.status == "runtime_error" and not self.error:
            raise ValueError("runtime_error requires an error message")
        return self


class WorkerResult(CanaryModel):
    """One terminal worker result linked to the exact canonical input."""

    schema_version: Literal["canary-worker-result.v1"] = "canary-worker-result.v1"
    job_id: str = Field(pattern=IDENTIFIER_PATTERN)
    job_sha256: str = Field(pattern=SHA256_PATTERN)
    input_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_sha256: str = Field(pattern=SHA256_PATTERN)
    status: Literal[
        "completed",
        "static_check_failed",
        "compile_error",
        "runtime_error",
        "wrong_result",
        "timeout",
        "environment_error",
        "worker_error",
    ]
    compiled: bool = False
    correct: bool = False
    cases: tuple[CaseResult, ...] = ()
    timing_ms: tuple[float, ...] = ()
    timing_cv: float | None = Field(default=None, ge=0)
    static_errors: tuple[str, ...] = ()
    static_warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @field_validator("timing_ms")
    @classmethod
    def timings_are_positive(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("timing samples must be finite and positive")
        return values

    @model_validator(mode="after")
    def terminal_status_is_consistent(self) -> WorkerResult:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case result IDs must be unique")
        if self.timing_cv is not None and not math.isfinite(self.timing_cv):
            raise ValueError("timing_cv must be finite")
        if bool(self.timing_ms) != (self.timing_cv is not None):
            raise ValueError("timing samples and timing_cv must be supplied together")
        if self.status == "completed":
            if not self.compiled or not self.correct or not self.cases:
                raise ValueError("completed results require compiled, correct case results")
            if any(case.status != "pass" for case in self.cases):
                raise ValueError("completed results require every case to pass")
            if self.error is not None:
                raise ValueError("completed results cannot contain an error")
        else:
            if self.correct:
                raise ValueError("non-completed results cannot be correct")
            if self.status in {"compile_error", "static_check_failed"} and self.compiled:
                raise ValueError(f"{self.status} results cannot be compiled")
            if self.status == "wrong_result" and (
                not self.compiled or not self.cases or all(case.correct for case in self.cases)
            ):
                raise ValueError("wrong_result requires a compiled failing case")
            if self.status in {
                "compile_error",
                "runtime_error",
                "timeout",
                "environment_error",
                "worker_error",
            } and not self.error:
                raise ValueError(f"{self.status} requires an error message")
            if self.status == "static_check_failed" and not self.static_errors:
                raise ValueError("static_check_failed requires static_errors")
        if self.timing_ms and not self.correct:
            raise ValueError("only correct results may contain timing samples")
        return self

    def verify_for_job(self, job: WorkerJob) -> WorkerResult:
        """Fail closed unless this terminal result belongs to the exact worker job."""

        expected_links = (job.job_id, job.sha256, job.input_sha256, job.candidate_sha256)
        actual_links = (
            self.job_id,
            self.job_sha256,
            self.input_sha256,
            self.candidate_sha256,
        )
        if actual_links != expected_links:
            raise ValueError("worker result does not match job identity or content hashes")
        result_case_ids = tuple(case.case_id for case in self.cases)
        if self.status in {"completed", "wrong_result"}:
            same_cases = set(result_case_ids) == set(job.case_ids)
            if not same_cases or len(result_case_ids) != len(job.case_ids):
                raise ValueError("terminal scientific result does not cover every requested case")
        elif any(case_id not in job.case_ids for case_id in result_case_ids):
            raise ValueError("worker result contains a case not requested by the job")
        if job.timing is None and self.timing_ms:
            raise ValueError("worker returned timing samples for a job without timing")
        if job.timing is not None and self.status == "completed":
            if len(self.timing_ms) != job.timing.trial_runs:
                raise ValueError("worker timing sample count does not match the job")
        return self


class TrajectoryOutcome(CanaryModel):
    """Terminal summary returned by the fixed four-turn Agent loop."""

    schema_version: Literal["canary-trajectory-outcome.v1"] = (
        "canary-trajectory-outcome.v1"
    )
    trajectory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: Literal[
        "finished",
        "call_limit",
        "budget_exhausted",
        "provider_error",
        "worker_error",
        "no_candidate",
    ]
    calls: int = Field(ge=0, le=4)
    known_input_tokens: int = Field(default=0, ge=0)
    known_output_tokens: int = Field(default=0, ge=0)
    usage_complete: bool
    first_candidate_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    final_candidate_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    dev_results: tuple[WorkerResult, ...] = ()
    first_sealed_result: WorkerResult | None = None
    final_sealed_result: WorkerResult | None = None
    started_at_utc: datetime
    finished_at_utc: datetime
    error: str | None = None

    @model_validator(mode="after")
    def terminal_summary_is_consistent(self) -> TrajectoryOutcome:
        has_first = self.first_candidate_sha256 is not None
        has_final = self.final_candidate_sha256 is not None
        if has_first != has_final:
            raise ValueError("first and final candidate hashes must appear together")
        if self.status == "no_candidate" and has_first:
            raise ValueError("no_candidate outcomes cannot contain candidate hashes")
        if not has_first and (self.first_sealed_result or self.final_sealed_result):
            raise ValueError("sealed results require first and final candidates")
        if self.status in {"provider_error", "worker_error"} and not self.error:
            raise ValueError(f"{self.status} outcomes require an error message")
        if self.finished_at_utc < self.started_at_utc:
            raise ValueError("trajectory finish time cannot precede start time")
        return self
