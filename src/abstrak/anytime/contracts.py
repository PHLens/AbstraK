"""Strict immutable contracts for version-one anytime DSL studies."""

from __future__ import annotations

import math
import re
from itertools import product
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from abstrak.providers.contracts import sha256_json

IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
FORMAL_CHECKPOINT_CALLS = (1, 4, 8, 12)
SHAKEOUT_CHECKPOINT_CALLS = (1, 4)


class AnytimeModel(BaseModel):
    """Base class for values that enter hashed anytime-study artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AnytimeReasoningSpec(AnytimeModel):
    """Literal reasoning request that must pass provider conformance before a study runs."""

    schema_version: Literal["abstrak-anytime-reasoning.v1"] = (
        "abstrak-anytime-reasoning.v1"
    )
    requested_reasoning_effort: Literal["xhigh"]
    conformance_requirement: Literal["literal_xhigh"] = "literal_xhigh"


class AnytimeGenerationSpec(AnytimeModel):
    """Provider-independent generation intent without transport normalization."""

    schema_version: Literal["abstrak-anytime-generation.v1"] = (
        "abstrak-anytime-generation.v1"
    )
    max_output_tokens: int = Field(ge=256, le=65536)
    reasoning: AnytimeReasoningSpec
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)

    @field_validator("temperature", "top_p")
    @classmethod
    def sampling_values_are_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("sampling parameters must be finite")
        return value


class AnytimeAgentSpec(AnytimeModel):
    """One exact Agent model and its native provider protocol."""

    schema_version: Literal["abstrak-anytime-agent.v1"] = "abstrak-anytime-agent.v1"
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    provider_id: str = Field(pattern=IDENTIFIER_PATTERN)
    model_ref: str = Field(min_length=1)
    native_protocol: Literal["chat_completions", "responses"]
    generation: AnytimeGenerationSpec

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeInfrastructurePolicy(AnytimeModel):
    """Bounded whole-trajectory infrastructure attempts, including the primary attempt."""

    schema_version: Literal["abstrak-anytime-infrastructure-policy.v1"] = (
        "abstrak-anytime-infrastructure-policy.v1"
    )
    max_attempts_per_trajectory: int = Field(default=2, ge=1, le=2)
    ambiguous_submission_policy: Literal["terminal"] = "terminal"
    unsubmitted_failures_consume_scientific_calls: Literal[False] = False
    submitted_failures_consume_scientific_calls: Literal[True] = True

    @property
    def retries_per_trajectory(self) -> int:
        return self.max_attempts_per_trajectory - 1


class AnytimeResourceBudget(AnytimeModel):
    """Scientific-call and wall-time caps for one anytime trajectory."""

    schema_version: Literal["abstrak-anytime-resource-budget.v1"] = (
        "abstrak-anytime-resource-budget.v1"
    )
    max_scientific_calls: int = Field(ge=1, le=12)
    max_total_output_tokens: int = Field(ge=256, le=786432)
    max_compile_attempts: int = Field(ge=1, le=12)
    max_evaluation_attempts: int = Field(ge=1, le=12)
    max_gpu_seconds: float = Field(gt=0, le=86400)
    max_provider_seconds_per_call: float = Field(default=600.0, gt=0, le=3600)
    max_candidate_seconds_per_call: float = Field(default=600.0, gt=0, le=3600)
    max_trajectory_wall_seconds: float = Field(default=86400.0, gt=0, le=604800)

    @field_validator(
        "max_provider_seconds_per_call",
        "max_candidate_seconds_per_call",
        "max_gpu_seconds",
        "max_trajectory_wall_seconds",
    )
    @classmethod
    def resource_caps_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("resource caps must be finite")
        return value

    @model_validator(mode="after")
    def attempt_caps_are_coherent(self) -> AnytimeResourceBudget:
        if self.max_compile_attempts > self.max_scientific_calls:
            raise ValueError("compile-attempt cap cannot exceed the scientific-call cap")
        if self.max_evaluation_attempts > self.max_scientific_calls:
            raise ValueError("evaluation-attempt cap cannot exceed the scientific-call cap")
        maximum_candidate_seconds = (
            self.max_scientific_calls * self.max_candidate_seconds_per_call
        )
        if self.max_gpu_seconds > maximum_candidate_seconds:
            raise ValueError("GPU-seconds cap exceeds the candidate-time envelope")
        return self


class AnytimeContextPolicy(AnytimeModel):
    """Hash-bound normalized context semantics implemented by the later ledger milestone."""

    schema_version: Literal["abstrak-anytime-context-policy.v1"] = (
        "abstrak-anytime-context-policy.v1"
    )
    mode: Literal["reconstructed"] = "reconstructed"
    component_order: tuple[str, ...] = (
        "base_prompt",
        "incumbent",
        "previous_candidate",
        "previous_feedback",
    )
    renderer_version: Literal["anytime-context-renderer.v1"] = "anytime-context-renderer.v1"
    feedback_schema_version: Literal["anytime-dev-feedback.v1"] = "anytime-dev-feedback.v1"
    provider_session_state: Literal["forbidden"] = "forbidden"
    previous_response_id: Literal["forbidden"] = "forbidden"
    automatic_compaction: Literal["forbidden"] = "forbidden"
    max_candidate_source_characters: int = Field(default=262144, ge=1024, le=1048576)
    oversize_source_policy: Literal["reject"] = "reject"
    max_diagnostic_items: int = Field(default=16, ge=1, le=128)
    max_diagnostic_characters_per_item: int = Field(default=1000, ge=64, le=10000)
    max_error_characters: int = Field(default=2000, ge=64, le=20000)

    @field_validator("component_order")
    @classmethod
    def components_are_exact_and_ordered(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        expected = (
            "base_prompt",
            "incumbent",
            "previous_candidate",
            "previous_feedback",
        )
        if value != expected:
            raise ValueError("context components must use the frozen canonical order")
        return value


class AnytimeCheckpointPolicy(AnytimeModel):
    """One-based consumed-scientific-call indices to snapshot."""

    schema_version: Literal["abstrak-anytime-checkpoint-policy.v1"] = (
        "abstrak-anytime-checkpoint-policy.v1"
    )
    calls: tuple[int, ...] = Field(min_length=1)

    @field_validator("calls")
    @classmethod
    def calls_are_positive_unique_and_sorted(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 1 for value in values):
            raise ValueError("checkpoint calls must be positive")
        if len(values) != len(set(values)):
            raise ValueError("checkpoint calls must be unique")
        if values != tuple(sorted(values)):
            raise ValueError("checkpoint calls must be strictly increasing")
        return values


class AnytimeLoopPolicy(AnytimeModel):
    """Fixed-call anytime semantics for one trajectory."""

    schema_version: Literal["abstrak-anytime-loop-policy.v1"] = (
        "abstrak-anytime-loop-policy.v1"
    )
    stop_policy: Literal["fixed_calls"] = "fixed_calls"
    incumbent_selection: Literal["best_eligible_latency"] = "best_eligible_latency"
    budget: AnytimeResourceBudget
    infrastructure: AnytimeInfrastructurePolicy = Field(
        default_factory=AnytimeInfrastructurePolicy
    )
    context: AnytimeContextPolicy = Field(default_factory=AnytimeContextPolicy)
    checkpoints: AnytimeCheckpointPolicy

    @model_validator(mode="after")
    def checkpoints_match_budget(self) -> AnytimeLoopPolicy:
        calls = self.checkpoints.calls
        if calls[0] != 1:
            raise ValueError("checkpoint policy must include the first scientific call")
        if calls[-1] != self.budget.max_scientific_calls:
            raise ValueError("last checkpoint must equal the scientific-call budget")
        if any(call > self.budget.max_scientific_calls for call in calls):
            raise ValueError("checkpoint call exceeds the scientific-call budget")
        return self

    @property
    def scientific_request_ceiling(self) -> int:
        return self.budget.max_scientific_calls

    @property
    def operational_request_ceiling(self) -> int:
        return self.scientific_request_ceiling * self.infrastructure.max_attempts_per_trajectory

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeResourceSnapshot(AnytimeModel):
    """Cumulative resource vector after a prefix of scientific calls."""

    schema_version: Literal["abstrak-anytime-resource-snapshot.v1"] = (
        "abstrak-anytime-resource-snapshot.v1"
    )
    scientific_calls_consumed: int = Field(ge=0, le=12)
    provider_requests_submitted: int = Field(ge=0, le=12)
    possibly_charged_requests: int = Field(ge=0, le=12)
    known_input_tokens: int = Field(default=0, ge=0)
    known_cached_input_tokens: int = Field(default=0, ge=0)
    known_output_tokens: int = Field(default=0, ge=0)
    known_reasoning_tokens: int = Field(default=0, ge=0)
    usage_complete: bool = Field(
        description=(
            "True only when input, cached-input, output, and reasoning token usage was known "
            "for every submitted request in this attempt prefix"
        )
    )
    compile_attempts: int = Field(default=0, ge=0, le=12)
    evaluation_attempts: int = Field(default=0, ge=0, le=12)
    provider_seconds: float = Field(default=0.0, ge=0)
    compile_seconds: float = Field(default=0.0, ge=0)
    evaluation_seconds: float = Field(default=0.0, ge=0)
    gpu_seconds: float = Field(default=0.0, ge=0)
    wall_seconds: float = Field(default=0.0, ge=0)

    @field_validator(
        "provider_seconds",
        "compile_seconds",
        "evaluation_seconds",
        "gpu_seconds",
        "wall_seconds",
    )
    @classmethod
    def elapsed_values_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("elapsed resources must be finite")
        return 0.0 if value == 0.0 else value

    @model_validator(mode="after")
    def counts_are_coherent(self) -> AnytimeResourceSnapshot:
        if self.provider_requests_submitted != self.scientific_calls_consumed:
            raise ValueError(
                "submitted requests must equal consumed scientific calls within one attempt"
            )
        if self.possibly_charged_requests > self.provider_requests_submitted:
            raise ValueError("possibly charged requests cannot exceed submitted requests")
        if self.known_cached_input_tokens > self.known_input_tokens:
            raise ValueError("known cached input tokens cannot exceed known input tokens")
        if self.compile_attempts > self.scientific_calls_consumed:
            raise ValueError("compile attempts cannot exceed consumed scientific calls")
        if self.evaluation_attempts > self.scientific_calls_consumed:
            raise ValueError("evaluation attempts cannot exceed consumed scientific calls")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeCheckpointIdentity(AnytimeModel):
    """Immutable identity of one checkpoint over a verified ledger prefix."""

    schema_version: Literal["abstrak-anytime-checkpoint-identity.v1"] = (
        "abstrak-anytime-checkpoint-identity.v1"
    )
    trajectory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    infrastructure_attempt_index: int = Field(ge=1, le=2)
    scientific_call_index: int = Field(ge=1, le=12)
    trajectory_execution_sha256: str = Field(pattern=SHA256_PATTERN)
    ledger_prefix_sha256: str = Field(pattern=SHA256_PATTERN)
    incumbent_candidate_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    resource_snapshot_sha256: str = Field(pattern=SHA256_PATTERN)

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeCohortSpec(AnytimeModel):
    """One independently schedulable Agent/task/target/replicate cohort."""

    schema_version: Literal["abstrak-anytime-cohort.v1"] = "abstrak-anytime-cohort.v1"
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    agent_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_ids: tuple[str, ...] = Field(min_length=1)
    target_ids: tuple[str, ...] = Field(min_length=1)
    replicates: tuple[int, ...] = Field(min_length=1)
    scoring: bool
    activation: Literal["always", "core_gate"] = "always"
    order_policy: Literal["balanced_rotation"] = "balanced_rotation"
    loop: AnytimeLoopPolicy

    @field_validator("task_ids", "target_ids")
    @classmethod
    def identifiers_are_unique(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        name = getattr(info, "field_name", "axis")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must be unique")
        invalid = tuple(
            value for value in values if re.fullmatch(IDENTIFIER_PATTERN, value) is None
        )
        if invalid:
            raise ValueError(f"{name} contain invalid identifiers: {', '.join(invalid)}")
        return values

    @field_validator("replicates")
    @classmethod
    def replicates_are_positive_and_unique(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 1 for value in values):
            raise ValueError("replicates must be positive")
        if len(values) != len(set(values)):
            raise ValueError("replicates must be unique")
        return values

    @property
    def expected_trajectories(self) -> int:
        return len(self.task_ids) * len(self.target_ids) * len(self.replicates)

    @property
    def scientific_request_ceiling(self) -> int:
        return self.expected_trajectories * self.loop.scientific_request_ceiling

    @property
    def operational_request_ceiling(self) -> int:
        return self.expected_trajectories * self.loop.operational_request_ceiling


class AnytimeStudySpec(AnytimeModel):
    """Complete hashable axes and budgets for a formal or shakeout anytime study."""

    schema_version: Literal["abstrak-anytime-study-spec.v1"] = (
        "abstrak-anytime-study-spec.v1"
    )
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    study_kind: Literal["formal", "shakeout"]
    seed: int = Field(ge=0)
    agents: tuple[AnytimeAgentSpec, ...] = Field(min_length=1)
    cohorts: tuple[AnytimeCohortSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def axes_and_study_kind_are_coherent(self) -> AnytimeStudySpec:
        agent_ids = tuple(agent.id for agent in self.agents)
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("Agent IDs must be unique")
        cohort_ids = tuple(cohort.id for cohort in self.cohorts)
        if len(cohort_ids) != len(set(cohort_ids)):
            raise ValueError("cohort IDs must be unique")
        referenced_agents = {cohort.agent_id for cohort in self.cohorts}
        unknown_agents = sorted(referenced_agents - set(agent_ids))
        if unknown_agents:
            raise ValueError("cohorts reference unknown Agents: " + ", ".join(unknown_agents))
        unused_agents = sorted(set(agent_ids) - referenced_agents)
        if unused_agents:
            raise ValueError("study declares unused Agents: " + ", ".join(unused_agents))

        expected_calls = 12 if self.study_kind == "formal" else 4
        expected_checkpoints = (
            FORMAL_CHECKPOINT_CALLS
            if self.study_kind == "formal"
            else SHAKEOUT_CHECKPOINT_CALLS
        )
        for cohort in self.cohorts:
            agent = self.agent(cohort.agent_id)
            if cohort.loop.budget.max_scientific_calls != expected_calls:
                raise ValueError(f"{self.study_kind} cohorts require {expected_calls} calls")
            if cohort.loop.checkpoints.calls != expected_checkpoints:
                raise ValueError(
                    f"{self.study_kind} cohorts require checkpoints {expected_checkpoints}"
                )
            if self.study_kind == "formal" and not cohort.scoring:
                raise ValueError("formal cohorts must be scoring")
            if self.study_kind == "shakeout" and cohort.scoring:
                raise ValueError("shakeout cohorts must be non-scoring")
            if self.study_kind == "shakeout" and cohort.activation != "always":
                raise ValueError("shakeout cohorts cannot be gate-authorized")
            budget = cohort.loop.budget
            expected_output_cap = expected_calls * agent.generation.max_output_tokens
            if budget.max_total_output_tokens != expected_output_cap:
                raise ValueError(
                    "cohort output-token cap must equal calls times per-call generation cap"
                )
            if budget.max_compile_attempts != expected_calls:
                raise ValueError("cohort compile-attempt cap must equal its scientific calls")
            if budget.max_evaluation_attempts != expected_calls:
                raise ValueError("cohort evaluation-attempt cap must equal its scientific calls")

        occupied: dict[tuple[str, str, str, int], str] = {}
        for cohort in self.cohorts:
            for task_id, target_id, replicate in product(
                cohort.task_ids,
                cohort.target_ids,
                cohort.replicates,
            ):
                key = (cohort.agent_id, task_id, target_id, replicate)
                previous = occupied.setdefault(key, cohort.id)
                if previous != cohort.id:
                    raise ValueError(
                        "cohorts contain a duplicate scientific cell: "
                        f"{previous}, {cohort.id}"
                    )
        return self

    def agent(self, agent_id: str) -> AnytimeAgentSpec:
        try:
            return next(agent for agent in self.agents if agent.id == agent_id)
        except StopIteration as error:
            raise ValueError(f"unknown Agent: {agent_id}") from error

    def cohort(self, cohort_id: str) -> AnytimeCohortSpec:
        try:
            return next(cohort for cohort in self.cohorts if cohort.id == cohort_id)
        except StopIteration as error:
            raise ValueError(f"unknown cohort: {cohort_id}") from error

    @property
    def expected_trajectories(self) -> int:
        return sum(cohort.expected_trajectories for cohort in self.cohorts)

    @property
    def scientific_request_ceiling(self) -> int:
        return sum(cohort.scientific_request_ceiling for cohort in self.cohorts)

    @property
    def operational_request_ceiling(self) -> int:
        return sum(cohort.operational_request_ceiling for cohort in self.cohorts)

    @property
    def sha256(self) -> str:
        return sha256_json(self)
