"""Generic deterministic matrix contracts for canary studies."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from typing import Literal

from pydantic import Field, field_validator, model_validator

from abstrak.canary.contracts import IDENTIFIER_PATTERN, SHA256_PATTERN, CanaryModel
from abstrak.providers.contracts import sha256_json

OrderPolicy = Literal["fixed", "seeded_shuffle", "balanced_rotation"]
CoreGateOutcome = Literal[
    "provisional_go",
    "core_no_go",
    "inconclusive",
    "invalid_floor",
    "inconclusive_infrastructure",
]

_IDENTIFIER = re.compile(IDENTIFIER_PATTERN)


class MatrixSpecError(ValueError):
    """Raised when a generic matrix cannot be constructed safely."""


def _unique_identifiers(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    invalid = tuple(value for value in values if _IDENTIFIER.fullmatch(value) is None)
    if invalid:
        raise ValueError(f"{name} contain invalid identifiers: {', '.join(invalid)}")
    return values


class TaskGroupSpec(CanaryModel):
    """One mechanism group used to limit correlated winner evidence."""

    id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("task_ids")
    @classmethod
    def task_ids_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_identifiers("task group task IDs", values)


class PhaseSpec(CanaryModel):
    """One complete matrix phase with its own tasks and execution policy."""

    id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_ids: tuple[str, ...] = Field(min_length=1)
    replicates: tuple[int, ...] = Field(min_length=1)
    order_policy: OrderPolicy
    max_calls_per_trajectory: int = Field(ge=1, le=4)
    infrastructure_retries: int = Field(default=0, ge=0, le=1)

    @field_validator("task_ids")
    @classmethod
    def task_ids_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_identifiers("phase task IDs", values)

    @field_validator("replicates")
    @classmethod
    def replicates_are_positive_and_unique(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 1 for value in values):
            raise ValueError("phase replicates must be positive")
        if len(values) != len(set(values)):
            raise ValueError("phase replicates must be unique")
        return values


class MetricThresholds(CanaryModel):
    """Shared thresholds for replicate qualification and timing comparisons."""

    competitive_latency_factor: float = Field(default=1.25, ge=1)
    latency_tie_fraction: float = Field(default=0.10, ge=0, lt=1)
    max_timing_cv: float = Field(default=0.05, gt=0, le=1)

    @field_validator(
        "competitive_latency_factor",
        "latency_tie_fraction",
        "max_timing_cv",
    )
    @classmethod
    def values_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("metric thresholds must be finite")
        return value


class CoreGateThresholds(CanaryModel):
    """Thresholds for the first portfolio phase and its reserve decision."""

    min_stable_tasks: int = Field(default=3, ge=1)
    min_unique_winner_groups: int = Field(default=2, ge=1)
    min_competitive_gap_units: int = Field(default=3, ge=0)
    min_latency_gain_tasks: int = Field(default=2, ge=1)
    min_latency_regret: float = Field(default=0.10, ge=0)
    no_go_max_competitive_gap_units: int = Field(default=1, ge=0)
    no_go_max_latency_regret: float = Field(default=0.05, ge=0)
    require_zero_stable_gap_for_no_go: bool = True
    require_common_winner_for_no_go: bool = True

    @field_validator("min_latency_regret", "no_go_max_latency_regret")
    @classmethod
    def regrets_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("latency regret thresholds must be finite")
        return value


class FullGateThresholds(CanaryModel):
    """Thresholds applied after all portfolio phases have completed."""

    min_stable_tasks: int = Field(default=6, ge=1)
    min_competitive_gap_units: int = Field(default=3, ge=0)
    min_latency_regret: float = Field(default=0.10, ge=0)
    min_opportunity_tasks: int = Field(default=2, ge=1)
    min_opportunity_groups: int = Field(default=2, ge=1)
    min_unique_winner_targets: int = Field(default=2, ge=1)
    dominant_winner_min_tasks: int = Field(default=7, ge=1)
    no_go_max_competitive_gap_units: int = Field(default=1, ge=0)
    no_go_max_latency_regret: float = Field(default=0.05, ge=0)
    require_zero_stable_gap_for_no_go: bool = True

    @field_validator("min_latency_regret", "no_go_max_latency_regret")
    @classmethod
    def regrets_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("latency regret thresholds must be finite")
        return value


class PortfolioGateSpec(CanaryModel):
    """Typed two-stage gate policy without a general-purpose rule language."""

    core_phase_id: str = Field(pattern=IDENTIFIER_PATTERN)
    reserve_phase_id: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    metrics: MetricThresholds = MetricThresholds()
    core: CoreGateThresholds = CoreGateThresholds()
    full: FullGateThresholds | None = None
    reserve_on_outcomes: tuple[CoreGateOutcome, ...] = (
        "provisional_go",
        "inconclusive",
    )

    @model_validator(mode="after")
    def reserve_contract_is_coherent(self) -> PortfolioGateSpec:
        if len(self.reserve_on_outcomes) != len(set(self.reserve_on_outcomes)):
            raise ValueError("reserve outcomes must be unique")
        if self.reserve_phase_id is None:
            if self.full is not None or self.reserve_on_outcomes:
                raise ValueError(
                    "a gate without a reserve phase cannot define full thresholds "
                    "or reserve outcomes"
                )
        elif self.full is None:
            raise ValueError("a reserve phase requires full gate thresholds")
        if self.reserve_phase_id == self.core_phase_id:
            raise ValueError("core and reserve phase IDs must differ")
        return self


class MatrixStudySpec(CanaryModel):
    """Hashable axes, phases, and decision thresholds for one matrix study."""

    schema_version: Literal["abstrak-matrix-study-spec.v1"] = "abstrak-matrix-study-spec.v1"
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    seed: int
    agents: tuple[str, ...] = Field(min_length=1)
    targets: tuple[str, ...] = Field(min_length=1)
    task_groups: tuple[TaskGroupSpec, ...] = Field(min_length=1)
    phases: tuple[PhaseSpec, ...] = Field(min_length=1)
    gate: PortfolioGateSpec | None = None

    @field_validator("agents", "targets")
    @classmethod
    def axes_are_unique(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        name = getattr(info, "field_name", "axis")
        return _unique_identifiers(name, values)

    @model_validator(mode="after")
    def groups_phases_and_gate_are_coherent(self) -> MatrixStudySpec:
        group_ids = tuple(group.id for group in self.task_groups)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("task group IDs must be unique")
        phase_ids = tuple(phase.id for phase in self.phases)
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError("phase IDs must be unique")

        grouped_tasks = tuple(task_id for group in self.task_groups for task_id in group.task_ids)
        duplicate_grouped_tasks = tuple(
            task_id for task_id, count in Counter(grouped_tasks).items() if count > 1
        )
        if duplicate_grouped_tasks:
            raise ValueError(
                "tasks must belong to exactly one group: " + ", ".join(duplicate_grouped_tasks)
            )
        phase_tasks = {task_id for phase in self.phases for task_id in phase.task_ids}
        grouped_task_set = set(grouped_tasks)
        if phase_tasks != grouped_task_set:
            missing = sorted(phase_tasks - grouped_task_set)
            unused = sorted(grouped_task_set - phase_tasks)
            details = []
            if missing:
                details.append("ungrouped phase tasks: " + ", ".join(missing))
            if unused:
                details.append("grouped tasks absent from phases: " + ", ".join(unused))
            raise ValueError("task groups must exactly cover phase tasks; " + "; ".join(details))

        if self.gate is not None:
            self._validate_gate(phase_ids)
        return self

    def _validate_gate(self, phase_ids: tuple[str, ...]) -> None:
        assert self.gate is not None
        if self.gate.core_phase_id not in phase_ids:
            raise ValueError("gate core phase is not declared")
        if self.gate.reserve_phase_id is not None and self.gate.reserve_phase_id not in phase_ids:
            raise ValueError("gate reserve phase is not declared")

        core_phase = self.phase(self.gate.core_phase_id)
        core_groups = self.groups_for_tasks(core_phase.task_ids)
        core = self.gate.core
        if core.min_stable_tasks > len(core_phase.task_ids):
            raise ValueError("core min_stable_tasks exceeds core task count")
        if core.min_unique_winner_groups > len(core_groups):
            raise ValueError("core winner-group threshold exceeds core group count")
        if core.min_latency_gain_tasks > len(core_phase.task_ids):
            raise ValueError("core latency-gain threshold exceeds core task count")
        core_replicate_units = len(core_phase.task_ids) * len(core_phase.replicates)
        if core.min_competitive_gap_units > core_replicate_units:
            raise ValueError("core competitive-gap threshold exceeds available replicate units")

        reserve_id = self.gate.reserve_phase_id
        if reserve_id is None:
            return
        assert self.gate.full is not None
        reserve_phase = self.phase(reserve_id)
        if reserve_phase.replicates != core_phase.replicates:
            raise ValueError("core and reserve phases must use identical replicates")
        full_task_ids = tuple(dict.fromkeys((*core_phase.task_ids, *reserve_phase.task_ids)))
        full_groups = self.groups_for_tasks(full_task_ids)
        full = self.gate.full
        if full.min_stable_tasks > len(full_task_ids):
            raise ValueError("full min_stable_tasks exceeds full task count")
        if full.min_opportunity_tasks > len(full_task_ids):
            raise ValueError("full opportunity-task threshold exceeds full task count")
        if full.min_opportunity_groups > len(full_groups):
            raise ValueError("full opportunity-group threshold exceeds full group count")
        if full.min_unique_winner_targets > len(self.targets):
            raise ValueError("full winner-target threshold exceeds target count")
        if full.dominant_winner_min_tasks > len(full_task_ids):
            raise ValueError("full dominant-winner threshold exceeds full task count")
        full_replicate_units = len(full_task_ids) * len(core_phase.replicates)
        if full.min_competitive_gap_units > full_replicate_units:
            raise ValueError("full competitive-gap threshold exceeds available replicate units")

    @property
    def sha256(self) -> str:
        return sha256_json(self)

    @property
    def expected_trajectories(self) -> int:
        return sum(self.phase_trajectory_count(phase.id) for phase in self.phases)

    @property
    def request_ceiling(self) -> int:
        """Return the scientific Agent-call ceiling, excluding infrastructure retries."""

        return sum(self.phase_request_ceiling(phase.id) for phase in self.phases)

    @property
    def operational_request_ceiling(self) -> int:
        """Return the worst-case Agent-call ceiling including infrastructure retries."""

        return sum(self.phase_operational_request_ceiling(phase.id) for phase in self.phases)

    def phase(self, phase_id: str) -> PhaseSpec:
        try:
            return next(phase for phase in self.phases if phase.id == phase_id)
        except StopIteration as error:
            raise MatrixSpecError(f"unknown phase: {phase_id}") from error

    def phase_trajectory_count(self, phase_id: str) -> int:
        phase = self.phase(phase_id)
        return len(phase.task_ids) * len(self.agents) * len(self.targets) * len(phase.replicates)

    def phase_request_ceiling(self, phase_id: str) -> int:
        phase = self.phase(phase_id)
        return self.phase_trajectory_count(phase_id) * phase.max_calls_per_trajectory

    def phase_operational_request_ceiling(self, phase_id: str) -> int:
        phase = self.phase(phase_id)
        return self.phase_request_ceiling(phase_id) * (1 + phase.infrastructure_retries)

    def groups_for_tasks(self, task_ids: tuple[str, ...]) -> tuple[str, ...]:
        requested = set(task_ids)
        return tuple(
            group.id
            for group in self.task_groups
            if any(task_id in requested for task_id in group.task_ids)
        )


class MatrixCell(CanaryModel):
    """One trajectory in the immutable total execution order."""

    phase_id: str = Field(pattern=IDENTIFIER_PATTERN)
    ordinal: int = Field(ge=0)
    phase_ordinal: int = Field(ge=0)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    agent_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    replicate: int = Field(ge=1)
    target_order_index: int = Field(ge=0)

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return (
            self.phase_id,
            self.task_id,
            self.agent_id,
            self.target_id,
            self.replicate,
        )

    @property
    def block_key(self) -> tuple[str, str, str, int]:
        return (self.phase_id, self.task_id, self.agent_id, self.replicate)

    @property
    def trajectory_id(self) -> str:
        return f"{self.phase_id}-{self.task_id}-{self.agent_id}-{self.target_id}-r{self.replicate}"


class MatrixSchedule(CanaryModel):
    """Materialized, replayable schedule for a generic matrix study."""

    schema_version: Literal["abstrak-matrix-schedule.v1"] = "abstrak-matrix-schedule.v1"
    spec: MatrixStudySpec
    spec_sha256: str = Field(pattern=SHA256_PATTERN)
    cells: tuple[MatrixCell, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def cells_exactly_match_spec(self) -> MatrixSchedule:
        if self.spec_sha256 != self.spec.sha256:
            raise ValueError("matrix schedule spec hash mismatch")
        expected = _build_cells(self.spec)
        if self.cells != expected:
            raise ValueError("matrix cells do not match the deterministic study spec")
        trajectory_ids = tuple(cell.trajectory_id for cell in self.cells)
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError("matrix schedule contains duplicate trajectory IDs")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)

    @property
    def expected_trajectories(self) -> int:
        return len(self.cells)

    @property
    def request_ceiling(self) -> int:
        return self.spec.request_ceiling

    @property
    def operational_request_ceiling(self) -> int:
        return self.spec.operational_request_ceiling

    def cells_for_phase(self, phase_id: str) -> tuple[MatrixCell, ...]:
        self.spec.phase(phase_id)
        return tuple(cell for cell in self.cells if cell.phase_id == phase_id)

    def phase_request_ceiling(self, phase_id: str) -> int:
        return self.spec.phase_request_ceiling(phase_id)

    def phase_operational_request_ceiling(self, phase_id: str) -> int:
        return self.spec.phase_operational_request_ceiling(phase_id)


def _phase_seed(seed: int, phase_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{phase_id}".encode()).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _ordered_targets(
    *,
    policy: OrderPolicy,
    targets: tuple[str, ...],
    generator: random.Random,
    balanced_base: tuple[str, ...],
    block_index: int,
) -> tuple[str, ...]:
    if policy == "fixed":
        return targets
    if policy == "seeded_shuffle":
        shuffled = list(targets)
        generator.shuffle(shuffled)
        return tuple(shuffled)
    offset = block_index % len(balanced_base)
    return (*balanced_base[offset:], *balanced_base[:offset])


def _build_cells(spec: MatrixStudySpec) -> tuple[MatrixCell, ...]:
    cells: list[MatrixCell] = []
    for phase in spec.phases:
        generator = random.Random(_phase_seed(spec.seed, phase.id))
        balanced = list(spec.targets)
        generator.shuffle(balanced)
        balanced_base = tuple(balanced)
        phase_ordinal = 0
        block_index = 0
        for task_id in phase.task_ids:
            for agent_id in spec.agents:
                for replicate in phase.replicates:
                    ordered_targets = _ordered_targets(
                        policy=phase.order_policy,
                        targets=spec.targets,
                        generator=generator,
                        balanced_base=balanced_base,
                        block_index=block_index,
                    )
                    for target_order_index, target_id in enumerate(ordered_targets):
                        cells.append(
                            MatrixCell(
                                phase_id=phase.id,
                                ordinal=len(cells),
                                phase_ordinal=phase_ordinal,
                                task_id=task_id,
                                agent_id=agent_id,
                                target_id=target_id,
                                replicate=replicate,
                                target_order_index=target_order_index,
                            )
                        )
                        phase_ordinal += 1
                    block_index += 1
    return tuple(cells)


def build_matrix_schedule(spec: MatrixStudySpec) -> MatrixSchedule:
    """Build the exact deterministic schedule declared by ``spec``."""

    cells = _build_cells(spec)
    trajectory_ids = tuple(cell.trajectory_id for cell in cells)
    if len(trajectory_ids) != len(set(trajectory_ids)):
        duplicates = tuple(
            identifier for identifier, count in Counter(trajectory_ids).items() if count > 1
        )
        raise MatrixSpecError(
            "matrix axes produce ambiguous trajectory IDs: " + ", ".join(duplicates)
        )
    return MatrixSchedule(spec=spec, spec_sha256=spec.sha256, cells=cells)
