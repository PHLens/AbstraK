"""Offline aggregation and decision rules for the A100 R1 rapid study."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Literal

from pydantic import Field, field_validator, model_validator

from abstrak.canary.contracts import IDENTIFIER_PATTERN, CanaryModel

CellStatus = Literal[
    "stable_qualified",
    "unstable",
    "failed",
    "infrastructure_missing",
]
OracleStatus = Literal[
    "selected",
    "lower_effort_preference",
    "performance_tie",
    "no_stable_target",
    "timing_unstable",
]
StudyOutcome = Literal[
    "positive_signal",
    "provisional_negative",
    "invalid_floor",
    "inconclusive_instability",
    "inconclusive_infrastructure",
]


class AnalysisError(ValueError):
    """Raised when normalized observations do not form a valid R1 matrix."""


class TrajectoryMeasurement(CanaryModel):
    """Normalized final-candidate data from one formal trajectory artifact."""

    agent_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    replicate: int = Field(ge=1)
    terminal_status: Literal[
        "observed",
        "provider_error",
        "worker_error",
        "environment_error",
        "timeout",
    ]
    qualified_at_first: bool = False
    qualified_at_final: bool = False
    candidate_latency_ms: float | None = Field(default=None, gt=0)
    baseline_latency_ms: float = Field(gt=0)
    expert_latency_ms: float = Field(gt=0)
    timing_cvs: tuple[float, ...] = Field(default=(), max_length=2)
    calls: int = Field(ge=0, le=4)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    wall_seconds: float = Field(default=0.0, ge=0)
    gpu_seconds: float = Field(default=0.0, ge=0)

    @field_validator("timing_cvs")
    @classmethod
    def timing_cvs_are_finite(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("timing CVs must be finite and non-negative")
        return values

    @field_validator(
        "candidate_latency_ms",
        "baseline_latency_ms",
        "expert_latency_ms",
        "wall_seconds",
        "gpu_seconds",
    )
    @classmethod
    def measurements_are_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("runtime and resource measurements must be finite")
        return value

    @model_validator(mode="after")
    def scientific_fields_match_terminal_status(self) -> TrajectoryMeasurement:
        is_observed = self.terminal_status == "observed"
        if not is_observed and (self.qualified_at_first or self.qualified_at_final):
            raise ValueError("infrastructure failures cannot be marked qualified")
        has_performance = self.candidate_latency_ms is not None or bool(self.timing_cvs)
        if self.qualified_at_final != has_performance:
            raise ValueError(
                "qualified final candidates require latency and timing CVs, and failed "
                "candidates cannot contain them"
            )
        if self.qualified_at_final and (self.candidate_latency_ms is None or not self.timing_cvs):
            raise ValueError("qualified final candidates require complete timing data")
        return self

    @property
    def infrastructure_failure(self) -> bool:
        return self.terminal_status != "observed"

    @property
    def efficiency(self) -> float | None:
        if self.candidate_latency_ms is None:
            return None
        return self.baseline_latency_ms / self.candidate_latency_ms

    @property
    def target_realization(self) -> float | None:
        if self.candidate_latency_ms is None:
            return None
        return self.expert_latency_ms / self.candidate_latency_ms


class CellAggregate(CanaryModel):
    """Two-replicate result for one Agent/task/target cell."""

    agent_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: CellStatus
    expected_replicates: int = 2
    observed_replicates: int = Field(ge=0, le=2)
    infrastructure_failures: int = Field(ge=0, le=2)
    qualified_at_first_count: int = Field(ge=0, le=2)
    qualified_at_final_count: int = Field(ge=0, le=2)
    performance_stable: bool
    median_candidate_latency_ms: float | None = Field(default=None, gt=0)
    median_efficiency: float | None = Field(default=None, gt=0)
    median_target_realization: float | None = Field(default=None, gt=0)
    median_calls: float | None = Field(default=None, ge=0, le=4)
    final_timing_cvs: tuple[float, ...] = ()


class CellOracle(CanaryModel):
    """Hindsight choice among stable-qualified targets for one Agent/task."""

    agent_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: OracleStatus
    stable_target_ids: tuple[str, ...]
    tied_target_ids: tuple[str, ...] = ()
    selected_target_id: str | None = None
    selected_efficiency: float | None = Field(default=None, gt=0)
    selected_calls: float | None = Field(default=None, ge=0, le=4)


class TargetHindsightScore(CanaryModel):
    """One target's lexicographic score for a fixed hindsight baseline."""

    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    stable_qualified_cells: int = Field(ge=0)
    covered_task_ids: tuple[str, ...]
    median_efficiency: float | None = Field(default=None, gt=0)
    median_calls: float | None = Field(default=None, ge=0, le=4)


class FixedHindsight(CanaryModel):
    """One global or per-Agent fixed-target hindsight baseline."""

    scope: Literal["global", "agent"]
    agent_id: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    selected_target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    tied_target_ids: tuple[str, ...]
    selected_score: TargetHindsightScore
    scores: tuple[TargetHindsightScore, ...]


class OracleComparison(CanaryModel):
    """Cell oracle compared with every tied Fixed-Agent-Hindsight choice."""

    agent_id: str = Field(pattern=IDENTIFIER_PATTERN)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    oracle_status: OracleStatus
    fixed_target_ids: tuple[str, ...]
    oracle_stable_qualified: bool
    fixed_stable_qualified: bool
    stable_coverage_gain: bool
    performance_regret: float | None = Field(default=None, ge=0)
    lower_effort_gain: bool
    actual_gain: bool


class StudyAnalysis(CanaryModel):
    """Complete deterministic R1 aggregate and terminal decision."""

    schema_version: Literal["abstrak-canary-analysis.v1"] = "abstrak-canary-analysis.v1"
    outcome: StudyOutcome
    rationale: tuple[str, ...]
    expected_trajectories: int = Field(ge=1)
    received_trajectories: int = Field(ge=0)
    qualified_at_first: int = Field(ge=0)
    qualified_at_final: int = Field(ge=0)
    cells: tuple[CellAggregate, ...]
    cell_oracles: tuple[CellOracle, ...]
    fixed_global_hindsight: FixedHindsight
    fixed_agent_hindsight: tuple[FixedHindsight, ...]
    oracle_comparisons: tuple[OracleComparison, ...]
    stable_coverage_gap_cells: int = Field(ge=0)
    stable_coverage_gap_task_ids: tuple[str, ...]
    median_performance_regret: float | None = Field(default=None, ge=0)
    non_tied_frontier_task_ids: tuple[str, ...]
    non_tied_winner_target_ids: tuple[str, ...]
    actual_gain_task_ids: tuple[str, ...]


def _ordered_unique(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if not values or len(values) != len(set(values)):
        raise AnalysisError(f"{name} must be non-empty and unique")
    return values


def _consistent_runtime(
    records: tuple[TrajectoryMeasurement, ...], attribute: str, label: str
) -> None:
    values = {getattr(record, attribute) for record in records}
    if len(values) > 1:
        raise AnalysisError(f"{label} runtime drifted across replicates")


def aggregate_cells(
    measurements: Iterable[TrajectoryMeasurement],
    *,
    agents: tuple[str, ...],
    tasks: tuple[str, ...],
    targets: tuple[str, ...],
    replicates: tuple[int, ...] = (1, 2),
    max_timing_cv: float = 0.05,
) -> tuple[CellAggregate, ...]:
    """Aggregate formal trajectories without silently treating missing data as failure."""

    agents = _ordered_unique("agents", agents)
    tasks = _ordered_unique("tasks", tasks)
    targets = _ordered_unique("targets", targets)
    if replicates != (1, 2):
        raise AnalysisError("R1 analysis requires replicates (1, 2)")
    if not math.isfinite(max_timing_cv) or not 0 < max_timing_cv <= 1:
        raise AnalysisError("max_timing_cv must be in (0, 1]")

    records = tuple(measurements)
    allowed = {
        (agent, task, target, replicate)
        for agent in agents
        for task in tasks
        for target in targets
        for replicate in replicates
    }
    by_key: dict[tuple[str, str, str, int], TrajectoryMeasurement] = {}
    for record in records:
        key = (record.agent_id, record.task_id, record.target_id, record.replicate)
        if key not in allowed:
            raise AnalysisError(f"measurement is outside the declared matrix: {key}")
        if key in by_key:
            raise AnalysisError(f"duplicate trajectory measurement: {key}")
        by_key[key] = record

    for task in tasks:
        task_records = tuple(record for record in records if record.task_id == task)
        if task_records:
            _consistent_runtime(task_records, "baseline_latency_ms", f"{task} baseline")
        for target in targets:
            target_records = tuple(record for record in task_records if record.target_id == target)
            if target_records:
                _consistent_runtime(
                    target_records,
                    "expert_latency_ms",
                    f"{task}/{target} expert oracle",
                )

    aggregates: list[CellAggregate] = []
    for agent in agents:
        for task in tasks:
            for target in targets:
                cell_records = tuple(
                    by_key[key]
                    for replicate in replicates
                    if (key := (agent, task, target, replicate)) in by_key
                )
                observed = tuple(
                    record for record in cell_records if not record.infrastructure_failure
                )
                infrastructure_failures = sum(
                    record.infrastructure_failure for record in cell_records
                )
                qualified = tuple(record for record in observed if record.qualified_at_final)
                if len(cell_records) != len(replicates) or infrastructure_failures:
                    status: CellStatus = "infrastructure_missing"
                elif len(qualified) == 2:
                    status = "stable_qualified"
                elif len(qualified) == 1:
                    status = "unstable"
                else:
                    status = "failed"

                final_cvs = tuple(record.timing_cvs[-1] for record in qualified)
                performance_stable = (
                    status == "stable_qualified"
                    and len(final_cvs) == 2
                    and all(cv <= max_timing_cv for cv in final_cvs)
                )
                aggregate = CellAggregate(
                    agent_id=agent,
                    task_id=task,
                    target_id=target,
                    status=status,
                    observed_replicates=len(observed),
                    infrastructure_failures=infrastructure_failures,
                    qualified_at_first_count=sum(record.qualified_at_first for record in observed),
                    qualified_at_final_count=len(qualified),
                    performance_stable=performance_stable,
                    median_candidate_latency_ms=(
                        statistics.median(
                            record.candidate_latency_ms
                            for record in qualified
                            if record.candidate_latency_ms is not None
                        )
                        if qualified
                        else None
                    ),
                    median_efficiency=(
                        statistics.median(
                            record.efficiency
                            for record in qualified
                            if record.efficiency is not None
                        )
                        if qualified
                        else None
                    ),
                    median_target_realization=(
                        statistics.median(
                            record.target_realization
                            for record in qualified
                            if record.target_realization is not None
                        )
                        if qualified
                        else None
                    ),
                    median_calls=(
                        statistics.median(record.calls for record in observed) if observed else None
                    ),
                    final_timing_cvs=final_cvs,
                )
                aggregates.append(aggregate)
    return tuple(aggregates)


def select_cell_oracles(
    cells: Iterable[CellAggregate],
    *,
    agents: tuple[str, ...],
    tasks: tuple[str, ...],
    targets: tuple[str, ...],
    tie_fraction: float = 0.05,
) -> tuple[CellOracle, ...]:
    """Select cell oracles using the registered 5% tie and effort rules."""

    if not math.isfinite(tie_fraction) or not 0 <= tie_fraction < 1:
        raise AnalysisError("tie_fraction must be in [0, 1)")
    by_key = {(cell.agent_id, cell.task_id, cell.target_id): cell for cell in cells}
    oracles: list[CellOracle] = []
    epsilon = 1e-12
    for agent in agents:
        for task in tasks:
            stable = tuple(
                by_key[(agent, task, target)]
                for target in targets
                if by_key[(agent, task, target)].status == "stable_qualified"
            )
            stable_ids = tuple(cell.target_id for cell in stable)
            if not stable:
                oracles.append(
                    CellOracle(
                        agent_id=agent,
                        task_id=task,
                        status="no_stable_target",
                        stable_target_ids=(),
                    )
                )
                continue
            if any(
                not cell.performance_stable or cell.median_efficiency is None for cell in stable
            ):
                oracles.append(
                    CellOracle(
                        agent_id=agent,
                        task_id=task,
                        status="timing_unstable",
                        stable_target_ids=stable_ids,
                    )
                )
                continue

            ranked = sorted(
                stable,
                key=lambda cell: (
                    -(cell.median_efficiency or 0.0),
                    targets.index(cell.target_id),
                ),
            )
            best_efficiency = ranked[0].median_efficiency
            assert best_efficiency is not None
            tied = tuple(
                cell
                for cell in ranked
                if (best_efficiency - (cell.median_efficiency or 0.0)) / best_efficiency
                <= tie_fraction + epsilon
            )
            if len(tied) == 1:
                selected = tied[0]
                oracles.append(
                    CellOracle(
                        agent_id=agent,
                        task_id=task,
                        status="selected",
                        stable_target_ids=stable_ids,
                        selected_target_id=selected.target_id,
                        selected_efficiency=selected.median_efficiency,
                        selected_calls=selected.median_calls,
                    )
                )
                continue

            effort_ranked = sorted(
                tied,
                key=lambda cell: (
                    cell.median_calls if cell.median_calls is not None else math.inf,
                    targets.index(cell.target_id),
                ),
            )
            lowest = effort_ranked[0]
            next_calls = effort_ranked[1].median_calls
            lowest_calls = lowest.median_calls
            has_effort_preference = (
                lowest_calls is not None
                and next_calls is not None
                and next_calls - lowest_calls >= 1 - epsilon
            )
            oracles.append(
                CellOracle(
                    agent_id=agent,
                    task_id=task,
                    status=(
                        "lower_effort_preference" if has_effort_preference else "performance_tie"
                    ),
                    stable_target_ids=stable_ids,
                    tied_target_ids=tuple(cell.target_id for cell in tied),
                    selected_target_id=lowest.target_id if has_effort_preference else None,
                    selected_efficiency=(
                        lowest.median_efficiency if has_effort_preference else None
                    ),
                    selected_calls=lowest_calls if has_effort_preference else None,
                )
            )
    return tuple(oracles)


def _fixed_hindsight(
    cells: tuple[CellAggregate, ...],
    *,
    targets: tuple[str, ...],
    scope: Literal["global", "agent"],
    agent_id: str | None = None,
) -> FixedHindsight:
    scoped = tuple(cell for cell in cells if agent_id is None or cell.agent_id == agent_id)
    scores: list[TargetHindsightScore] = []
    for target in targets:
        stable = tuple(
            cell
            for cell in scoped
            if cell.target_id == target and cell.status == "stable_qualified"
        )
        performance_complete = bool(stable) and all(
            cell.performance_stable and cell.median_efficiency is not None for cell in stable
        )
        calls_complete = bool(stable) and all(cell.median_calls is not None for cell in stable)
        scores.append(
            TargetHindsightScore(
                target_id=target,
                stable_qualified_cells=len(stable),
                covered_task_ids=tuple(dict.fromkeys(cell.task_id for cell in stable)),
                median_efficiency=(
                    statistics.median(
                        cell.median_efficiency
                        for cell in stable
                        if cell.median_efficiency is not None
                    )
                    if performance_complete
                    else None
                ),
                median_calls=(
                    statistics.median(
                        cell.median_calls for cell in stable if cell.median_calls is not None
                    )
                    if calls_complete
                    else None
                ),
            )
        )

    def ranking(score: TargetHindsightScore) -> tuple[int, float, float]:
        return (
            score.stable_qualified_cells,
            score.median_efficiency if score.median_efficiency is not None else -math.inf,
            -(score.median_calls if score.median_calls is not None else math.inf),
        )

    best_rank = max(ranking(score) for score in scores)
    tied = tuple(score for score in scores if ranking(score) == best_rank)
    selected = tied[0]
    return FixedHindsight(
        scope=scope,
        agent_id=agent_id,
        selected_target_id=selected.target_id,
        tied_target_ids=tuple(score.target_id for score in tied),
        selected_score=selected,
        scores=tuple(scores),
    )


def _oracle_comparisons(
    cells: tuple[CellAggregate, ...],
    oracles: tuple[CellOracle, ...],
    fixed_agents: tuple[FixedHindsight, ...],
    *,
    tie_fraction: float,
) -> tuple[OracleComparison, ...]:
    by_cell = {(cell.agent_id, cell.task_id, cell.target_id): cell for cell in cells}
    fixed_by_agent = {fixed.agent_id: fixed for fixed in fixed_agents}
    comparisons: list[OracleComparison] = []
    epsilon = 1e-12
    for oracle in oracles:
        fixed = fixed_by_agent[oracle.agent_id]
        fixed_ids = fixed.tied_target_ids
        fixed_cells = tuple(
            by_cell[(oracle.agent_id, oracle.task_id, target)] for target in fixed_ids
        )
        oracle_cells = tuple(
            by_cell[(oracle.agent_id, oracle.task_id, target)]
            for target in oracle.stable_target_ids
        )
        oracle_stable = bool(oracle_cells)
        fixed_stable_cells = tuple(
            cell for cell in fixed_cells if cell.status == "stable_qualified"
        )
        fixed_stable = bool(fixed_stable_cells)
        coverage_gain = oracle_stable and not fixed_stable
        regret: float | None = None
        if (
            oracle.status != "timing_unstable"
            and oracle_cells
            and fixed_stable_cells
            and all(cell.performance_stable for cell in fixed_stable_cells)
        ):
            oracle_efficiency = max(cell.median_efficiency or 0.0 for cell in oracle_cells)
            fixed_efficiency = max(cell.median_efficiency or 0.0 for cell in fixed_stable_cells)
            regret = max(0.0, (oracle_efficiency - fixed_efficiency) / oracle_efficiency)

        lower_effort_gain = False
        if (
            oracle.status == "lower_effort_preference"
            and oracle.selected_target_id is not None
            and oracle.selected_calls is not None
            and fixed_stable_cells
        ):
            selected = by_cell[(oracle.agent_id, oracle.task_id, oracle.selected_target_id)]
            lower_effort_gain = all(
                fixed_cell.target_id != selected.target_id
                and fixed_cell.target_id in oracle.tied_target_ids
                and fixed_cell.median_calls is not None
                and fixed_cell.median_calls - oracle.selected_calls >= 1 - epsilon
                for fixed_cell in fixed_stable_cells
            )
        performance_gain = regret is not None and regret > tie_fraction + epsilon
        comparisons.append(
            OracleComparison(
                agent_id=oracle.agent_id,
                task_id=oracle.task_id,
                oracle_status=oracle.status,
                fixed_target_ids=fixed_ids,
                oracle_stable_qualified=oracle_stable,
                fixed_stable_qualified=fixed_stable,
                stable_coverage_gain=coverage_gain,
                performance_regret=regret,
                lower_effort_gain=lower_effort_gain,
                actual_gain=coverage_gain or performance_gain or lower_effort_gain,
            )
        )
    return tuple(comparisons)


def _has_distinct_task_target_winners(oracles: tuple[CellOracle, ...]) -> bool:
    winners = tuple(
        (oracle.task_id, oracle.selected_target_id)
        for oracle in oracles
        if oracle.status == "selected" and oracle.selected_target_id is not None
    )
    return any(
        first_task != second_task and first_target != second_target
        for first_task, first_target in winners
        for second_task, second_target in winners
    )


def analyze_study(
    measurements: Iterable[TrajectoryMeasurement],
    *,
    agents: tuple[str, ...],
    tasks: tuple[str, ...],
    targets: tuple[str, ...],
    expert_oracle_complete: bool,
    shakeout_passed: bool,
    max_timing_cv: float = 0.05,
    tie_fraction: float = 0.05,
) -> StudyAnalysis:
    """Apply all registered R1 aggregation, hindsight, and terminal rules."""

    records = tuple(measurements)
    cells = aggregate_cells(
        records,
        agents=agents,
        tasks=tasks,
        targets=targets,
        max_timing_cv=max_timing_cv,
    )
    oracles = select_cell_oracles(
        cells,
        agents=agents,
        tasks=tasks,
        targets=targets,
        tie_fraction=tie_fraction,
    )
    fixed_global = _fixed_hindsight(cells, targets=targets, scope="global")
    fixed_agents = tuple(
        _fixed_hindsight(
            cells,
            targets=targets,
            scope="agent",
            agent_id=agent,
        )
        for agent in agents
    )
    comparisons = _oracle_comparisons(
        cells,
        oracles,
        fixed_agents,
        tie_fraction=tie_fraction,
    )

    frontier_tasks = tuple(
        dict.fromkeys(oracle.task_id for oracle in oracles if oracle.status == "selected")
    )
    winner_targets = tuple(
        dict.fromkeys(
            oracle.selected_target_id
            for oracle in oracles
            if oracle.status == "selected" and oracle.selected_target_id is not None
        )
    )
    gain_tasks = tuple(
        dict.fromkeys(comparison.task_id for comparison in comparisons if comparison.actual_gain)
    )
    positive = (
        len(frontier_tasks) >= 2
        and _has_distinct_task_target_winners(oracles)
        and len(gain_tasks) >= 2
    )
    has_infrastructure_gap = any(cell.status == "infrastructure_missing" for cell in cells)
    has_instability = any(
        cell.status == "unstable"
        or (cell.status == "stable_qualified" and not cell.performance_stable)
        for cell in cells
    )

    if not expert_oracle_complete or not shakeout_passed:
        outcome: StudyOutcome = "invalid_floor"
        rationale = (
            "expert oracle matrix is incomplete"
            if not expert_oracle_complete
            else "shakeout gate did not pass",
        )
    elif has_infrastructure_gap:
        outcome = "inconclusive_infrastructure"
        rationale = ("formal matrix contains missing or infrastructure-censored cells",)
    elif positive:
        outcome = "positive_signal"
        rationale = (
            "at least two tasks have non-tied frontiers with distinct target winners",
            "cell oracle improves on Fixed-Agent-Hindsight on at least two task IDs",
        )
    elif has_instability:
        outcome = "inconclusive_instability"
        rationale = ("unstable qualification or timing could change the negative conclusion",)
    else:
        outcome = "provisional_negative"
        rationale = ("stable complete matrix does not satisfy the positive-signal rules",)

    regrets = tuple(
        comparison.performance_regret
        for comparison in comparisons
        if comparison.performance_regret is not None
    )
    coverage_gap_tasks = tuple(
        dict.fromkeys(
            comparison.task_id for comparison in comparisons if comparison.stable_coverage_gain
        )
    )
    return StudyAnalysis(
        outcome=outcome,
        rationale=rationale,
        expected_trajectories=len(agents) * len(tasks) * len(targets) * 2,
        received_trajectories=len(records),
        qualified_at_first=sum(record.qualified_at_first for record in records),
        qualified_at_final=sum(record.qualified_at_final for record in records),
        cells=cells,
        cell_oracles=oracles,
        fixed_global_hindsight=fixed_global,
        fixed_agent_hindsight=fixed_agents,
        oracle_comparisons=comparisons,
        stable_coverage_gap_cells=sum(
            comparison.stable_coverage_gain for comparison in comparisons
        ),
        stable_coverage_gap_task_ids=coverage_gap_tasks,
        median_performance_regret=statistics.median(regrets) if regrets else None,
        non_tied_frontier_task_ids=frontier_tasks,
        non_tied_winner_target_ids=winner_targets,
        actual_gain_task_ids=gain_tasks,
    )
