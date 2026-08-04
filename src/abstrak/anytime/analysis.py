"""Artifact-only analysis for fixed-call anytime DSL studies.

The objects in this module are deliberately normalized projections of verified,
immutable artifacts.  Nothing here discovers runs, contacts a provider, opens a
worker connection, or executes candidate code.  The public entrypoint revalidates
every supplied model from JSON so ``model_copy(update=...)`` cannot bypass the
strict artifact contracts.

Agent trajectory replicates are the smallest statistical observations.  CUDA
timing trials only establish one clean-process latency and are never expanded into
pseudo-replicates.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import Field, field_validator, model_validator

from abstrak.anytime.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    AnytimeModel,
)
from abstrak.providers.contracts import sha256_json


class AnytimeAnalysisError(ValueError):
    """Raised when verified artifacts cannot support the requested analysis."""


FloorStatus = Literal[
    "valid",
    "invalid_floor",
    "unstable_timing",
    "infrastructure_missing",
]
TerminalStatus = Literal["complete", "early_resource_cap", "infrastructure_censored"]
CandidateStage = Literal[
    "provider_error",
    "parse_failure",
    "qualification_pending",
    "static_check_failure",
    "compile_failure",
    "wrong_result",
    "target_use_failure",
    "timing_unstable",
    "eligible",
]
MeasurementKind = Literal["none", "exploratory_dev", "formal_checkpoint"]
MatchMode = Literal["iteration", "wall_clock"]
EvidenceScope = Literal["formal_only", "formal_and_exploratory"]
MissingReason = Literal[
    "none",
    "invalid_floor",
    "unstable_floor_timing",
    "floor_infrastructure_missing",
    "trajectory_infrastructure_censored",
    "early_resource_cap",
    "not_reached",
    "no_eligible_incumbent",
    "no_measurement_at_or_before_budget",
]


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return 0.0 if value == 0.0 else value


def _ordered_unique(label: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{label} cannot be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    return values


class AnytimeArtifactTrust(AnytimeModel):
    """Evidence that an analysis row came from an audited immutable artifact."""

    schema_version: Literal["abstrak-anytime-analysis-artifact-trust.v1"] = (
        "abstrak-anytime-analysis-artifact-trust.v1"
    )
    source_kind: Literal["verified_immutable_artifact"] = "verified_immutable_artifact"
    artifact_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    checksum_audit_passed: Literal[True] = True
    immutable_terminal: Literal[True] = True


class AnytimeAgentReplicateAxis(AnytimeModel):
    schema_version: Literal["abstrak-anytime-agent-replicate-axis.v1"] = (
        "abstrak-anytime-agent-replicate-axis.v1"
    )
    agent_id: str = Field(pattern=IDENTIFIER_PATTERN)
    replicates: tuple[int, ...] = Field(min_length=1)

    @field_validator("replicates")
    @classmethod
    def replicates_are_ordered_unique(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 1 for value in values):
            raise ValueError("replicates must be positive")
        if values != tuple(sorted(set(values))):
            raise ValueError("replicates must be strictly increasing and unique")
        return values


class AnytimeWorkloadAxis(AnytimeModel):
    schema_version: Literal["abstrak-anytime-analysis-workload.v1"] = (
        "abstrak-anytime-analysis-workload.v1"
    )
    workload_id: str = Field(pattern=IDENTIFIER_PATTERN)
    semantic_family_id: str = Field(pattern=IDENTIFIER_PATTERN)


class AnytimeAnalysisSpec(AnytimeModel):
    """Frozen axes and decision rules for one analysis reconstruction."""

    schema_version: Literal["abstrak-anytime-analysis-spec.v1"] = "abstrak-anytime-analysis-spec.v1"
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    study_spec_sha256: str = Field(pattern=SHA256_PATTERN)
    study_stage: Literal["synthetic_fixture", "shakeout", "formal"]
    analysis_version: Literal["anytime-analysis.v1"] = "anytime-analysis.v1"
    agents: tuple[AnytimeAgentReplicateAxis, ...] = Field(min_length=1)
    workloads: tuple[AnytimeWorkloadAxis, ...] = Field(min_length=1)
    targets: tuple[str, ...] = Field(min_length=1)
    max_scientific_calls: int = Field(default=12, ge=1, le=12)
    formal_checkpoints: tuple[int, ...] = (1, 4, 8, 12)
    wall_clock_budgets_seconds: tuple[float, ...] = Field(min_length=1)
    winner_relative_tolerance: float = Field(default=0.05, ge=0, le=0.25)
    bstar_qualification_slack: float = Field(default=0.0, ge=0, le=0.25)
    bootstrap_seed: int = Field(default=20260803, ge=0)
    bootstrap_resamples: int = Field(default=1000, ge=100, le=10000)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)

    @field_validator("targets")
    @classmethod
    def targets_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        values = _ordered_unique("targets", values)
        for value in values:
            if not value or re.fullmatch(IDENTIFIER_PATTERN, value) is None:
                raise ValueError("target IDs must use the identifier grammar")
        return values

    @field_validator("formal_checkpoints")
    @classmethod
    def checkpoints_are_ordered_unique(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values or values != tuple(sorted(set(values))) or values[0] < 1:
            raise ValueError("formal checkpoints must be positive, unique, and ordered")
        return values

    @field_validator("wall_clock_budgets_seconds")
    @classmethod
    def wall_budgets_are_finite_ordered(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("wall-clock budgets must be finite and positive")
        if values != tuple(sorted(set(values))):
            raise ValueError("wall-clock budgets must be strictly increasing and unique")
        return values

    @model_validator(mode="after")
    def axes_are_coherent(self) -> AnytimeAnalysisSpec:
        agent_ids = tuple(axis.agent_id for axis in self.agents)
        _ordered_unique("agent IDs", agent_ids)
        workload_ids = tuple(axis.workload_id for axis in self.workloads)
        _ordered_unique("workload IDs", workload_ids)
        if self.formal_checkpoints[-1] != self.max_scientific_calls:
            raise ValueError("last formal checkpoint must equal max scientific calls")
        if any(call > self.max_scientific_calls for call in self.formal_checkpoints):
            raise ValueError("formal checkpoint exceeds max scientific calls")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeFloorArtifact(AnytimeModel):
    """One target floor plus common eager and B* references for a workload."""

    schema_version: Literal["abstrak-anytime-floor-analysis-input.v1"] = (
        "abstrak-anytime-floor-analysis-input.v1"
    )
    trust: AnytimeArtifactTrust
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    study_spec_sha256: str = Field(pattern=SHA256_PATTERN)
    workload_id: str = Field(pattern=IDENTIFIER_PATTERN)
    semantic_family_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: FloorStatus
    eager_latency_ms: float | None = Field(default=None, gt=0)
    bstar_latency_ms: float | None = Field(default=None, gt=0)
    target_expert_latency_ms: float | None = Field(default=None, gt=0)
    clean_process: bool = False
    timing_stable: bool = False
    independently_sealed: bool = False
    timing_trial_count: int = Field(default=0, ge=0)

    @field_validator("eager_latency_ms", "bstar_latency_ms", "target_expert_latency_ms")
    @classmethod
    def latencies_are_finite(cls, value: float | None) -> float | None:
        if value is not None:
            return _finite(value, "floor latency")
        return None

    @model_validator(mode="after")
    def status_matches_evidence(self) -> AnytimeFloorArtifact:
        latencies = (
            self.eager_latency_ms,
            self.bstar_latency_ms,
            self.target_expert_latency_ms,
        )
        if self.status == "valid":
            if any(value is None for value in latencies):
                raise ValueError("valid floor requires eager, B*, and target expert latencies")
            if not (self.clean_process and self.timing_stable and self.independently_sealed):
                raise ValueError("valid floor requires sealed stable clean-process timing")
            if self.timing_trial_count < 1:
                raise ValueError("valid floor requires timing trials")
        elif any(value is not None for value in latencies):
            raise ValueError("non-valid floor cannot expose performance latencies")
        return self


class AnytimeTurnArtifact(AnytimeModel):
    """Analysis projection of one consumed scientific call."""

    schema_version: Literal["abstrak-anytime-turn-analysis-input.v1"] = (
        "abstrak-anytime-turn-analysis-input.v1"
    )
    scientific_call_index: int = Field(ge=1, le=12)
    cumulative_wall_seconds: float = Field(ge=0)
    candidate_stage: CandidateStage
    incumbent_candidate_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    incumbent_latency_ms: float | None = Field(default=None, gt=0)
    measurement_kind: MeasurementKind = "none"
    clean_process_measurement: bool = False
    independently_retimed: bool = False
    timing_trial_count: int = Field(default=0, ge=0)

    @field_validator("cumulative_wall_seconds", "incumbent_latency_ms")
    @classmethod
    def numeric_values_are_finite(cls, value: float | None) -> float | None:
        if value is not None:
            return _finite(value, "turn measurement")
        return None

    @model_validator(mode="after")
    def incumbent_matches_measurement(self) -> AnytimeTurnArtifact:
        has_incumbent = self.incumbent_candidate_sha256 is not None
        if has_incumbent != (self.incumbent_latency_ms is not None):
            raise ValueError("incumbent hash and latency must be present together")
        if self.measurement_kind == "none":
            if has_incumbent or self.clean_process_measurement or self.independently_retimed:
                raise ValueError("unmeasured turn cannot contain incumbent measurement evidence")
            if self.timing_trial_count != 0:
                raise ValueError("unmeasured turn cannot contain timing trials")
        else:
            if not has_incumbent or self.timing_trial_count < 1:
                raise ValueError("measured turn requires an incumbent and timing trials")
            if self.measurement_kind == "formal_checkpoint" and not (
                self.clean_process_measurement and self.independently_retimed
            ):
                raise ValueError("formal checkpoint requires independent clean-process retiming")
        return self

    @property
    def compiled(self) -> bool:
        return self.candidate_stage in {
            "wrong_result",
            "target_use_failure",
            "timing_unstable",
            "eligible",
        }

    @property
    def correct(self) -> bool:
        return self.candidate_stage in {
            "target_use_failure",
            "timing_unstable",
            "eligible",
        }

    @property
    def qualified(self) -> bool:
        return self.candidate_stage in {"timing_unstable", "eligible"}

    @property
    def eligible(self) -> bool:
        return self.candidate_stage == "eligible"


class AnytimeTrajectoryArtifact(AnytimeModel):
    """One verified Agent trajectory; timing trials stay nested metadata."""

    schema_version: Literal["abstrak-anytime-trajectory-analysis-input.v1"] = (
        "abstrak-anytime-trajectory-analysis-input.v1"
    )
    trust: AnytimeArtifactTrust
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    study_spec_sha256: str = Field(pattern=SHA256_PATTERN)
    trajectory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    agent_id: str = Field(pattern=IDENTIFIER_PATTERN)
    workload_id: str = Field(pattern=IDENTIFIER_PATTERN)
    semantic_family_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    replicate: int = Field(ge=1)
    terminal_status: TerminalStatus
    turns: tuple[AnytimeTurnArtifact, ...] = ()

    @model_validator(mode="after")
    def turns_are_a_verified_prefix(self) -> AnytimeTrajectoryArtifact:
        calls = tuple(turn.scientific_call_index for turn in self.turns)
        if calls != tuple(range(1, len(self.turns) + 1)):
            raise ValueError("trajectory turns must be the exact one-based prefix")
        walls = tuple(turn.cumulative_wall_seconds for turn in self.turns)
        if any(current < previous for previous, current in zip(walls, walls[1:], strict=False)):
            raise ValueError("cumulative wall time cannot decrease")
        if self.terminal_status == "infrastructure_censored" and self.turns:
            # Persisted consumed calls may precede infrastructure censorship; retaining them is
            # useful for rates, but performance matching remains censored below.
            pass
        return self


class AnytimeAnalysisDataset(AnytimeModel):
    """Complete immutable input population for deterministic reconstruction."""

    schema_version: Literal["abstrak-anytime-analysis-dataset.v1"] = (
        "abstrak-anytime-analysis-dataset.v1"
    )
    spec: AnytimeAnalysisSpec
    floors: tuple[AnytimeFloorArtifact, ...]
    trajectories: tuple[AnytimeTrajectoryArtifact, ...]

    @model_validator(mode="after")
    def pending_qualification_is_synthetic_only(self) -> AnytimeAnalysisDataset:
        has_pending = any(
            turn.candidate_stage == "qualification_pending"
            for trajectory in self.trajectories
            for turn in trajectory.turns
        )
        if has_pending and self.spec.study_stage != "synthetic_fixture":
            raise ValueError("qualification-pending turns are restricted to synthetic fixtures")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeRateRow(AnytimeModel):
    schema_version: Literal["abstrak-anytime-rate-row.v1"] = "abstrak-anytime-rate-row.v1"
    agent_id: str
    scientific_call_index: int
    formal_checkpoint: bool
    expected_trajectories: int
    observed_turns: int
    infrastructure_censored: int
    early_resource_caps: int
    not_reached: int
    compiled_count: int
    correct_count: int
    qualified_count: int
    eligible_count: int
    compiled_rate: float
    correct_rate: float
    qualification_rate: float
    eligible_rate: float
    observed_compiled_rate: float | None
    observed_correct_rate: float | None
    observed_qualification_rate: float | None
    observed_eligible_rate: float | None


class AnytimeMatchedObservation(AnytimeModel):
    schema_version: Literal["abstrak-anytime-matched-observation.v1"] = (
        "abstrak-anytime-matched-observation.v1"
    )
    agent_id: str
    workload_id: str
    semantic_family_id: str
    target_id: str
    replicate: int
    match_mode: MatchMode
    iteration_budget: int | None = None
    wall_clock_budget_seconds: float | None = None
    evidence_scope: EvidenceScope
    selected_call_index: int | None = None
    selected_cumulative_wall_seconds: float | None = None
    measurement_kind: MeasurementKind = "none"
    latency_ms: float | None = None
    eager_speedup: float | None = None
    bstar_relative_performance: float | None = None
    bstar_qualified: bool = False
    timing_trial_count: int = 0
    early_resource_cap: bool = False
    missing_reason: MissingReason = "none"

    @model_validator(mode="after")
    def fields_match_availability(self) -> AnytimeMatchedObservation:
        available = self.missing_reason == "none"
        values = (self.latency_ms, self.eager_speedup, self.bstar_relative_performance)
        if available and (
            self.selected_call_index is None
            or self.measurement_kind == "none"
            or any(value is None for value in values)
        ):
            raise ValueError("available match requires a measured selected turn")
        if not available and any(value is not None for value in values):
            raise ValueError("missing match cannot expose performance")
        return self


WinnerStatus = Literal[
    "selected",
    "tie",
    "no_eligible_target",
    "invalid_floor",
    "unstable_floor_timing",
    "floor_infrastructure_missing",
    "infrastructure_censored",
    "early_resource_cap",
    "replicate_disagreement",
]


class AnytimeTargetScore(AnytimeModel):
    target_id: str
    expected_replicates: int
    observed_replicates: int
    median_eager_speedup: float | None = None
    median_bstar_relative_performance: float | None = None
    bstar_qualified_replicates: int


class AnytimeWinnerRow(AnytimeModel):
    schema_version: Literal["abstrak-anytime-winner-row.v1"] = "abstrak-anytime-winner-row.v1"
    agent_id: str
    workload_id: str
    semantic_family_id: str
    match_mode: MatchMode
    iteration_budget: int | None = None
    wall_clock_budget_seconds: float | None = None
    evidence_scope: EvidenceScope
    status: WinnerStatus
    winner_target_ids: tuple[str, ...]
    replicate_winner_sets: tuple[tuple[str, ...], ...]
    replicate_disagreement: bool
    scores: tuple[AnytimeTargetScore, ...]


class AnytimeMissingnessRow(AnytimeModel):
    schema_version: Literal["abstrak-anytime-missingness-row.v1"] = (
        "abstrak-anytime-missingness-row.v1"
    )
    agent_id: str
    match_mode: MatchMode
    iteration_budget: int | None = None
    wall_clock_budget_seconds: float | None = None
    evidence_scope: EvidenceScope
    expected_observations: int
    available_observations: int
    infrastructure_censored: int
    early_resource_caps: int
    invalid_floor: int
    unstable_floor_timing: int
    floor_infrastructure_missing: int
    no_eligible_incumbent: int
    not_reached: int


class AnytimeClusteredInterval(AnytimeModel):
    schema_version: Literal["abstrak-anytime-clustered-interval.v1"] = (
        "abstrak-anytime-clustered-interval.v1"
    )
    cluster_unit: Literal["semantic_family_x_agent_x_trajectory_replicate"] = (
        "semantic_family_x_agent_x_trajectory_replicate"
    )
    timing_trials_are_replicates: Literal[False] = False
    cluster_count: int = Field(ge=1)
    observation_count: int = Field(ge=1)
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    estimand: Literal[
        "geometric_mean_ratio",
        "workload_median_oracle_over_best_fixed_geomean_ratio",
    ] = "geometric_mean_ratio"
    method: Literal["deterministic_cluster_percentile_bootstrap_v2"] = (
        "deterministic_cluster_percentile_bootstrap_v2"
    )
    seed: int
    resamples: int


HindsightStatus = Literal[
    "complete",
    "invalid_floor",
    "infrastructure_censored",
    "early_resource_cap",
    "insufficient_coverage",
]


class AnytimeWorkloadOracleGain(AnytimeModel):
    workload_id: str
    semantic_family_id: str
    fixed_target_id: str
    oracle_target_ids: tuple[str, ...]
    fixed_speedup: float
    oracle_speedup: float
    oracle_over_fixed_gain: float


class AnytimeHindsightComparison(AnytimeModel):
    schema_version: Literal["abstrak-anytime-hindsight-comparison.v1"] = (
        "abstrak-anytime-hindsight-comparison.v1"
    )
    agent_id: str
    match_mode: MatchMode
    iteration_budget: int | None = None
    wall_clock_budget_seconds: float | None = None
    evidence_scope: EvidenceScope
    status: HindsightStatus
    fixed_target_ids: tuple[str, ...]
    best_fixed_target_id: str | None = None
    fixed_geometric_mean_speedup: float | None = None
    oracle_geometric_mean_speedup: float | None = None
    oracle_over_fixed_gain: float | None = None
    workload_gains: tuple[AnytimeWorkloadOracleGain, ...] = ()
    clustered_gain_interval: AnytimeClusteredInterval | None = None

    @model_validator(mode="after")
    def complete_rows_identify_the_numerical_fixed_target(
        self,
    ) -> AnytimeHindsightComparison:
        if self.status == "complete":
            if self.best_fixed_target_id is None:
                raise ValueError("complete hindsight comparison requires a best fixed target")
            if self.best_fixed_target_id not in self.fixed_target_ids:
                raise ValueError("best fixed target must belong to the equivalent target set")
        elif self.best_fixed_target_id is not None:
            raise ValueError("incomplete hindsight comparison cannot identify a fixed target")
        return self


class AnytimeModelRankingDisagreement(AnytimeModel):
    schema_version: Literal["abstrak-anytime-model-ranking-disagreement.v1"] = (
        "abstrak-anytime-model-ranking-disagreement.v1"
    )
    workload_id: str
    match_mode: MatchMode
    iteration_budget: int | None = None
    wall_clock_budget_seconds: float | None = None
    evidence_scope: EvidenceScope
    agent_winner_sets: tuple[tuple[str, tuple[str, ...]], ...]
    model_dependent: bool


class AnytimeAnalysisReport(AnytimeModel):
    schema_version: Literal["abstrak-anytime-analysis-report.v1"] = (
        "abstrak-anytime-analysis-report.v1"
    )
    analysis_spec_sha256: str = Field(pattern=SHA256_PATTERN)
    input_dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    rates: tuple[AnytimeRateRow, ...]
    matches: tuple[AnytimeMatchedObservation, ...]
    winners: tuple[AnytimeWinnerRow, ...]
    missingness: tuple[AnytimeMissingnessRow, ...]
    hindsight: tuple[AnytimeHindsightComparison, ...]
    model_ranking_disagreements: tuple[AnytimeModelRankingDisagreement, ...]

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeAnalysisTable(AnytimeModel):
    schema_version: Literal["abstrak-anytime-analysis-table.v1"] = (
        "abstrak-anytime-analysis-table.v1"
    )
    name: str = Field(pattern=IDENTIFIER_PATTERN)
    columns: tuple[str, ...] = Field(min_length=1)
    rows: tuple[tuple[str, ...], ...]

    @model_validator(mode="after")
    def rows_match_columns(self) -> AnytimeAnalysisTable:
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("table rows must match the declared columns")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _validate_dataset(dataset: AnytimeAnalysisDataset) -> AnytimeAnalysisDataset:
    """Revalidate untrusted object graphs and the full declared population."""

    try:
        checked = AnytimeAnalysisDataset.model_validate_json(dataset.model_dump_json())
    except Exception as exc:  # Pydantic errors are part of the fail-closed boundary.
        raise AnytimeAnalysisError(f"invalid analysis artifact projection: {exc}") from exc

    spec = checked.spec
    agents = {axis.agent_id: axis.replicates for axis in spec.agents}
    workloads = {axis.workload_id: axis.semantic_family_id for axis in spec.workloads}
    targets = set(spec.targets)

    expected_floor_keys = {(workload, target) for workload in workloads for target in targets}
    floors: dict[tuple[str, str], AnytimeFloorArtifact] = {}
    for floor in checked.floors:
        key = (floor.workload_id, floor.target_id)
        if floor.study_id != spec.study_id or floor.study_spec_sha256 != spec.study_spec_sha256:
            raise AnytimeAnalysisError("floor artifact is not bound to the analysis study")
        if floor.workload_id not in workloads or floor.target_id not in targets:
            raise AnytimeAnalysisError(f"floor artifact is outside declared axes: {key}")
        if floor.semantic_family_id != workloads[floor.workload_id]:
            raise AnytimeAnalysisError("floor semantic family differs from the workload axis")
        if key in floors:
            raise AnytimeAnalysisError(f"duplicate floor artifact: {key}")
        floors[key] = floor
    if set(floors) != expected_floor_keys:
        missing = sorted(expected_floor_keys - set(floors))
        extra = sorted(set(floors) - expected_floor_keys)
        raise AnytimeAnalysisError(
            f"floor population is incomplete: missing={missing}, extra={extra}"
        )
    expected_floor_order = tuple(
        (workload.workload_id, target) for workload in spec.workloads for target in spec.targets
    )
    observed_floor_order = tuple((floor.workload_id, floor.target_id) for floor in checked.floors)
    if observed_floor_order != expected_floor_order:
        raise AnytimeAnalysisError("floor artifacts are not in canonical workload-target order")

    # Eager and B* are common workload references, not target-specific opportunities to drift.
    for workload in workloads:
        valid = [
            floors[(workload, target)]
            for target in spec.targets
            if floors[(workload, target)].status == "valid"
        ]
        if valid:
            eager = {floor.eager_latency_ms for floor in valid}
            bstar = {floor.bstar_latency_ms for floor in valid}
            if len(eager) != 1 or len(bstar) != 1:
                raise AnytimeAnalysisError("eager or B* latency drifted across target floors")

    trajectory_keys: set[tuple[str, str, str, int]] = set()
    for trajectory in checked.trajectories:
        key = (
            trajectory.agent_id,
            trajectory.workload_id,
            trajectory.target_id,
            trajectory.replicate,
        )
        if (
            trajectory.study_id != spec.study_id
            or trajectory.study_spec_sha256 != spec.study_spec_sha256
        ):
            raise AnytimeAnalysisError("trajectory artifact is not bound to the analysis study")
        if (
            trajectory.agent_id not in agents
            or trajectory.workload_id not in workloads
            or trajectory.target_id not in targets
            or trajectory.replicate not in agents.get(trajectory.agent_id, ())
        ):
            raise AnytimeAnalysisError(f"trajectory artifact is outside declared axes: {key}")
        if trajectory.semantic_family_id != workloads[trajectory.workload_id]:
            raise AnytimeAnalysisError("trajectory semantic family differs from workload axis")
        if key in trajectory_keys:
            raise AnytimeAnalysisError(f"duplicate trajectory artifact: {key}")
        trajectory_keys.add(key)
        if len(trajectory.turns) > spec.max_scientific_calls:
            raise AnytimeAnalysisError("trajectory exceeds the scientific-call budget")
        if (
            trajectory.terminal_status == "complete"
            and len(trajectory.turns) != spec.max_scientific_calls
        ):
            raise AnytimeAnalysisError(
                "complete trajectory must contain the full fixed-call prefix"
            )
        if (
            trajectory.terminal_status == "early_resource_cap"
            and len(trajectory.turns) >= spec.max_scientific_calls
        ):
            raise AnytimeAnalysisError("early resource cap must stop before the fixed-call budget")
        for turn in trajectory.turns:
            is_checkpoint = turn.scientific_call_index in spec.formal_checkpoints
            if turn.measurement_kind == "formal_checkpoint" and not is_checkpoint:
                raise AnytimeAnalysisError("formal measurement occurs outside a checkpoint")
            if (
                is_checkpoint
                and turn.incumbent_candidate_sha256 is not None
                and turn.measurement_kind != "formal_checkpoint"
            ):
                raise AnytimeAnalysisError(
                    "checkpoint incumbent lacks a formal clean-process measurement"
                )

    expected_trajectory_keys = {
        (agent.agent_id, workload.workload_id, target, replicate)
        for agent in spec.agents
        for workload in spec.workloads
        for target in spec.targets
        for replicate in agent.replicates
    }
    if not trajectory_keys <= expected_trajectory_keys:
        raise AnytimeAnalysisError("trajectory set contains undeclared cells")
    declared_trajectory_order = tuple(
        (agent.agent_id, workload.workload_id, target, replicate)
        for agent in spec.agents
        for workload in spec.workloads
        for target in spec.targets
        for replicate in agent.replicates
    )
    canonical_trajectory_order = tuple(
        key for key in declared_trajectory_order if key in trajectory_keys
    )
    observed_trajectory_order = tuple(
        (
            trajectory.agent_id,
            trajectory.workload_id,
            trajectory.target_id,
            trajectory.replicate,
        )
        for trajectory in checked.trajectories
    )
    if observed_trajectory_order != canonical_trajectory_order:
        raise AnytimeAnalysisError("trajectory artifacts are not in canonical axis order")
    # Missing trajectories are represented by absent immutable artifacts and remain explicit in all
    # denominators; requiring a full set here would erase the distinction between absence and a
    # persisted infrastructure tombstone.
    return checked


def _trajectory_map(
    dataset: AnytimeAnalysisDataset,
) -> dict[tuple[str, str, str, int], AnytimeTrajectoryArtifact]:
    return {
        (row.agent_id, row.workload_id, row.target_id, row.replicate): row
        for row in dataset.trajectories
    }


def _floor_map(
    dataset: AnytimeAnalysisDataset,
) -> dict[tuple[str, str], AnytimeFloorArtifact]:
    return {(row.workload_id, row.target_id): row for row in dataset.floors}


def aggregate_anytime_rates(dataset: AnytimeAnalysisDataset) -> tuple[AnytimeRateRow, ...]:
    """Compute fixed-denominator stage rates at every turn and formal checkpoint."""

    dataset = _validate_dataset(dataset)
    trajectories = _trajectory_map(dataset)
    rows: list[AnytimeRateRow] = []
    for agent_axis in dataset.spec.agents:
        expected_keys = [
            (agent_axis.agent_id, workload.workload_id, target, replicate)
            for workload in dataset.spec.workloads
            for target in dataset.spec.targets
            for replicate in agent_axis.replicates
        ]
        denominator = len(expected_keys)
        for call in range(1, dataset.spec.max_scientific_calls + 1):
            observed: list[AnytimeTurnArtifact] = []
            infrastructure = 0
            early_caps = 0
            not_reached = 0
            for key in expected_keys:
                trajectory = trajectories.get(key)
                if trajectory is None:
                    infrastructure += 1
                    continue
                if (
                    trajectory.terminal_status == "infrastructure_censored"
                    and len(trajectory.turns) < call
                ):
                    infrastructure += 1
                    continue
                if len(trajectory.turns) < call:
                    if trajectory.terminal_status == "early_resource_cap":
                        early_caps += 1
                    else:
                        not_reached += 1
                    continue
                observed.append(trajectory.turns[call - 1])
            counts = {
                "compiled": sum(turn.compiled for turn in observed),
                "correct": sum(turn.correct for turn in observed),
                "qualified": sum(turn.qualified for turn in observed),
                "eligible": sum(turn.eligible for turn in observed),
            }

            def rate(count: int, base: int) -> float | None:
                return count / base if base else None

            rows.append(
                AnytimeRateRow(
                    agent_id=agent_axis.agent_id,
                    scientific_call_index=call,
                    formal_checkpoint=call in dataset.spec.formal_checkpoints,
                    expected_trajectories=denominator,
                    observed_turns=len(observed),
                    infrastructure_censored=infrastructure,
                    early_resource_caps=early_caps,
                    not_reached=not_reached,
                    compiled_count=counts["compiled"],
                    correct_count=counts["correct"],
                    qualified_count=counts["qualified"],
                    eligible_count=counts["eligible"],
                    compiled_rate=counts["compiled"] / denominator,
                    correct_rate=counts["correct"] / denominator,
                    qualification_rate=counts["qualified"] / denominator,
                    eligible_rate=counts["eligible"] / denominator,
                    observed_compiled_rate=rate(counts["compiled"], len(observed)),
                    observed_correct_rate=rate(counts["correct"], len(observed)),
                    observed_qualification_rate=rate(counts["qualified"], len(observed)),
                    observed_eligible_rate=rate(counts["eligible"], len(observed)),
                )
            )
    return tuple(rows)


def _floor_missing_reason(floor: AnytimeFloorArtifact) -> MissingReason:
    return {
        "invalid_floor": "invalid_floor",
        "unstable_timing": "unstable_floor_timing",
        "infrastructure_missing": "floor_infrastructure_missing",
    }.get(floor.status, "none")  # type: ignore[return-value]


def _match_one(
    trajectory: AnytimeTrajectoryArtifact | None,
    floor: AnytimeFloorArtifact,
    *,
    agent_id: str,
    workload_id: str,
    family_id: str,
    target_id: str,
    replicate: int,
    mode: MatchMode,
    iteration_budget: int | None,
    wall_budget: float | None,
    evidence_scope: EvidenceScope,
    bstar_slack: float,
) -> AnytimeMatchedObservation:
    common = dict(
        agent_id=agent_id,
        workload_id=workload_id,
        semantic_family_id=family_id,
        target_id=target_id,
        replicate=replicate,
        match_mode=mode,
        iteration_budget=iteration_budget,
        wall_clock_budget_seconds=wall_budget,
        evidence_scope=evidence_scope,
    )
    floor_reason = _floor_missing_reason(floor)
    if floor_reason != "none":
        return AnytimeMatchedObservation(**common, missing_reason=floor_reason)
    if trajectory is None or trajectory.terminal_status == "infrastructure_censored":
        return AnytimeMatchedObservation(
            **common, missing_reason="trajectory_infrastructure_censored"
        )

    candidates: list[AnytimeTurnArtifact]
    if mode == "iteration":
        assert iteration_budget is not None
        if len(trajectory.turns) < iteration_budget:
            reason: MissingReason = (
                "early_resource_cap"
                if trajectory.terminal_status == "early_resource_cap"
                else "not_reached"
            )
            return AnytimeMatchedObservation(
                **common,
                early_resource_cap=trajectory.terminal_status == "early_resource_cap",
                missing_reason=reason,
            )
        candidates = [trajectory.turns[iteration_budget - 1]]
    else:
        assert wall_budget is not None
        candidates = [
            turn for turn in trajectory.turns if turn.cumulative_wall_seconds <= wall_budget
        ]
        if evidence_scope == "formal_only":
            candidates = [
                turn for turn in candidates if turn.measurement_kind == "formal_checkpoint"
            ]
        if not candidates:
            exhausted_by_budget = trajectory.terminal_status == "early_resource_cap" and (
                not trajectory.turns or trajectory.turns[-1].cumulative_wall_seconds <= wall_budget
            )
            return AnytimeMatchedObservation(
                **common,
                early_resource_cap=exhausted_by_budget,
                missing_reason=(
                    "early_resource_cap"
                    if exhausted_by_budget
                    else "no_measurement_at_or_before_budget"
                ),
            )
        # The latest state available by the budget is the only honest anytime state.  We never
        # search later turns for a better measured value.
        candidates = [candidates[-1]]

    selected = candidates[0]
    allowed = (
        selected.measurement_kind == "formal_checkpoint"
        if evidence_scope == "formal_only"
        else selected.measurement_kind in {"formal_checkpoint", "exploratory_dev"}
    )
    if not allowed or selected.incumbent_latency_ms is None:
        return AnytimeMatchedObservation(
            **common,
            selected_call_index=selected.scientific_call_index,
            selected_cumulative_wall_seconds=selected.cumulative_wall_seconds,
            missing_reason=(
                "no_eligible_incumbent"
                if selected.incumbent_candidate_sha256 is None
                else "no_measurement_at_or_before_budget"
            ),
        )

    assert floor.eager_latency_ms is not None and floor.bstar_latency_ms is not None
    latency = selected.incumbent_latency_ms
    return AnytimeMatchedObservation(
        **common,
        selected_call_index=selected.scientific_call_index,
        selected_cumulative_wall_seconds=selected.cumulative_wall_seconds,
        measurement_kind=selected.measurement_kind,
        latency_ms=latency,
        eager_speedup=floor.eager_latency_ms / latency,
        bstar_relative_performance=floor.bstar_latency_ms / latency,
        bstar_qualified=latency <= floor.bstar_latency_ms * (1.0 + bstar_slack),
        timing_trial_count=selected.timing_trial_count,
        missing_reason="none",
    )


def match_anytime_observations(
    dataset: AnytimeAnalysisDataset,
) -> tuple[AnytimeMatchedObservation, ...]:
    """Produce iteration- and wall-matched rows without looking beyond either budget."""

    dataset = _validate_dataset(dataset)
    trajectories = _trajectory_map(dataset)
    floors = _floor_map(dataset)
    result: list[AnytimeMatchedObservation] = []
    iteration_requests = [
        (
            call,
            "formal_only" if call in dataset.spec.formal_checkpoints else "formal_and_exploratory",
        )
        for call in range(1, dataset.spec.max_scientific_calls + 1)
    ]
    for agent in dataset.spec.agents:
        for workload in dataset.spec.workloads:
            for target in dataset.spec.targets:
                floor = floors[(workload.workload_id, target)]
                for replicate in agent.replicates:
                    trajectory = trajectories.get(
                        (agent.agent_id, workload.workload_id, target, replicate)
                    )
                    for call, scope in iteration_requests:
                        result.append(
                            _match_one(
                                trajectory,
                                floor,
                                agent_id=agent.agent_id,
                                workload_id=workload.workload_id,
                                family_id=workload.semantic_family_id,
                                target_id=target,
                                replicate=replicate,
                                mode="iteration",
                                iteration_budget=call,
                                wall_budget=None,
                                evidence_scope=scope,  # type: ignore[arg-type]
                                bstar_slack=dataset.spec.bstar_qualification_slack,
                            )
                        )
                    for wall_budget in dataset.spec.wall_clock_budgets_seconds:
                        for scope in ("formal_only", "formal_and_exploratory"):
                            result.append(
                                _match_one(
                                    trajectory,
                                    floor,
                                    agent_id=agent.agent_id,
                                    workload_id=workload.workload_id,
                                    family_id=workload.semantic_family_id,
                                    target_id=target,
                                    replicate=replicate,
                                    mode="wall_clock",
                                    iteration_budget=None,
                                    wall_budget=wall_budget,
                                    evidence_scope=scope,
                                    bstar_slack=dataset.spec.bstar_qualification_slack,
                                )
                            )
    return tuple(result)


def _budget_key(row: AnytimeMatchedObservation | AnytimeWinnerRow) -> tuple[object, ...]:
    return (
        row.agent_id,
        row.match_mode,
        row.iteration_budget,
        row.wall_clock_budget_seconds,
        row.evidence_scope,
    )


def _relative_winners(
    scores: dict[str, float], target_order: tuple[str, ...], tolerance: float
) -> tuple[str, ...]:
    if not scores:
        return ()
    best = max(scores.values())
    cutoff = best * (1.0 - tolerance)
    return tuple(target for target in target_order if scores.get(target, -math.inf) >= cutoff)


def select_anytime_winners(
    matches: Iterable[AnytimeMatchedObservation],
    *,
    spec: AnytimeAnalysisSpec,
) -> tuple[AnytimeWinnerRow, ...]:
    """Select workload winners only when all declared trajectory replicates agree."""

    checked = tuple(
        AnytimeMatchedObservation.model_validate_json(row.model_dump_json()) for row in matches
    )
    replicate_axes = {axis.agent_id: axis.replicates for axis in spec.agents}
    grouped: dict[tuple[object, ...], list[AnytimeMatchedObservation]] = defaultdict(list)
    for row in checked:
        grouped[
            (
                row.agent_id,
                row.workload_id,
                row.semantic_family_id,
                row.match_mode,
                row.iteration_budget,
                row.wall_clock_budget_seconds,
                row.evidence_scope,
            )
        ].append(row)

    rows: list[AnytimeWinnerRow] = []
    for key in sorted(
        grouped,
        key=lambda item: tuple("" if value is None else str(value) for value in item),
    ):
        (
            agent,
            workload,
            family,
            mode,
            iteration,
            wall,
            scope,
        ) = key
        observations = grouped[key]
        expected_replicates = replicate_axes[str(agent)]
        by_target = {
            target: [row for row in observations if row.target_id == target]
            for target in spec.targets
        }
        scores: list[AnytimeTargetScore] = []
        aggregate: dict[str, float] = {}
        for target in spec.targets:
            available = [row for row in by_target[target] if row.missing_reason == "none"]
            if available:
                speedups = [row.eager_speedup for row in available]
                bstar = [row.bstar_relative_performance for row in available]
                assert all(value is not None for value in speedups + bstar)
                median_speedup = statistics.median(float(value) for value in speedups)
                median_bstar = statistics.median(float(value) for value in bstar)
                if len(available) == len(expected_replicates):
                    aggregate[target] = median_speedup
            else:
                median_speedup = None
                median_bstar = None
            scores.append(
                AnytimeTargetScore(
                    target_id=target,
                    expected_replicates=len(expected_replicates),
                    observed_replicates=len(available),
                    median_eager_speedup=median_speedup,
                    median_bstar_relative_performance=median_bstar,
                    bstar_qualified_replicates=sum(row.bstar_qualified for row in available),
                )
            )

        reasons = {row.missing_reason for row in observations if row.missing_reason != "none"}
        status: WinnerStatus
        winners: tuple[str, ...] = ()
        replicate_sets: list[tuple[str, ...]] = []
        disagreement = False
        precedence: tuple[tuple[MissingReason, WinnerStatus], ...] = (
            ("invalid_floor", "invalid_floor"),
            ("unstable_floor_timing", "unstable_floor_timing"),
            ("floor_infrastructure_missing", "floor_infrastructure_missing"),
            ("trajectory_infrastructure_censored", "infrastructure_censored"),
            ("early_resource_cap", "early_resource_cap"),
        )
        blocking = next((mapped for reason, mapped in precedence if reason in reasons), None)
        if blocking is None and any(row.early_resource_cap for row in observations):
            blocking = "early_resource_cap"
        if blocking is not None:
            status = blocking
        elif len(aggregate) != len(spec.targets):
            status = "no_eligible_target"
        else:
            winners = _relative_winners(aggregate, spec.targets, spec.winner_relative_tolerance)
            for replicate in expected_replicates:
                replicate_scores = {
                    target: float(
                        next(
                            row.eager_speedup
                            for row in by_target[target]
                            if row.replicate == replicate and row.eager_speedup is not None
                        )
                    )
                    for target in spec.targets
                }
                replicate_sets.append(
                    _relative_winners(
                        replicate_scores, spec.targets, spec.winner_relative_tolerance
                    )
                )
            disagreement = len(set(replicate_sets)) > 1
            if disagreement:
                status = "replicate_disagreement"
                winners = ()
            else:
                status = "tie" if len(winners) > 1 else "selected"
        rows.append(
            AnytimeWinnerRow(
                agent_id=str(agent),
                workload_id=str(workload),
                semantic_family_id=str(family),
                match_mode=mode,  # type: ignore[arg-type]
                iteration_budget=iteration,  # type: ignore[arg-type]
                wall_clock_budget_seconds=wall,  # type: ignore[arg-type]
                evidence_scope=scope,  # type: ignore[arg-type]
                status=status,
                winner_target_ids=winners,
                replicate_winner_sets=tuple(replicate_sets),
                replicate_disagreement=disagreement,
                scores=tuple(scores),
            )
        )
    return tuple(rows)


def summarize_anytime_missingness(
    matches: Iterable[AnytimeMatchedObservation],
) -> tuple[AnytimeMissingnessRow, ...]:
    grouped: dict[tuple[object, ...], list[AnytimeMatchedObservation]] = defaultdict(list)
    for row in matches:
        grouped[_budget_key(row)].append(row)
    result: list[AnytimeMissingnessRow] = []
    for key in sorted(
        grouped,
        key=lambda item: tuple("" if value is None else str(value) for value in item),
    ):
        agent, mode, iteration, wall, scope = key
        rows = grouped[key]
        counts = defaultdict(int)
        for row in rows:
            counts[row.missing_reason] += 1
        result.append(
            AnytimeMissingnessRow(
                agent_id=str(agent),
                match_mode=mode,  # type: ignore[arg-type]
                iteration_budget=iteration,  # type: ignore[arg-type]
                wall_clock_budget_seconds=wall,  # type: ignore[arg-type]
                evidence_scope=scope,  # type: ignore[arg-type]
                expected_observations=len(rows),
                available_observations=counts["none"],
                infrastructure_censored=counts["trajectory_infrastructure_censored"],
                early_resource_caps=sum(row.early_resource_cap for row in rows),
                invalid_floor=counts["invalid_floor"],
                unstable_floor_timing=counts["unstable_floor_timing"],
                floor_infrastructure_missing=counts["floor_infrastructure_missing"],
                no_eligible_incumbent=counts["no_eligible_incumbent"],
                not_reached=counts["not_reached"] + counts["no_measurement_at_or_before_budget"],
            )
        )
    return tuple(result)


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise AnytimeAnalysisError("geometric mean requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def clustered_interval(
    observations: Iterable[tuple[tuple[str, str, int], float]],
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> AnytimeClusteredInterval:
    """Bootstrap a geometric mean ratio over family×Agent×replicate clusters."""

    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    count = 0
    for cluster, value in observations:
        if not math.isfinite(value) or value <= 0:
            raise AnytimeAnalysisError("cluster ratio must be finite and positive")
        grouped[cluster].append(value)
        count += 1
    if not grouped:
        raise AnytimeAnalysisError("clustered interval requires observations")
    clusters = [grouped[key] for key in sorted(grouped)]

    def estimate(groups: Sequence[Sequence[float]]) -> float:
        logs = [math.log(value) for group in groups for value in group]
        return math.exp(math.fsum(logs) / len(logs))

    point = estimate(clusters)
    rng = random.Random(seed)
    bootstraps = sorted(
        estimate([rng.choice(clusters) for _ in clusters]) for _ in range(resamples)
    )
    alpha = (1.0 - confidence_level) / 2.0
    return AnytimeClusteredInterval(
        cluster_count=len(clusters),
        observation_count=count,
        point_estimate=point,
        lower=_percentile(bootstraps, alpha),
        upper=_percentile(bootstraps, 1.0 - alpha),
        confidence_level=confidence_level,
        seed=seed,
        resamples=resamples,
    )


def _hindsight_clustered_interval(
    observations: Sequence[AnytimeMatchedObservation],
    *,
    agent_id: str,
    spec: AnytimeAnalysisSpec,
) -> AnytimeClusteredInterval:
    """Bootstrap the exact workload-median hindsight estimand within semantic families."""

    replicates = next(axis.replicates for axis in spec.agents if axis.agent_id == agent_id)
    lookup = {
        (row.workload_id, row.target_id, row.replicate): float(row.eager_speedup)
        for row in observations
        if row.eager_speedup is not None
    }
    families = tuple(dict.fromkeys(workload.semantic_family_id for workload in spec.workloads))

    def estimate(draws: dict[str, tuple[int, ...]]) -> float:
        workload_target = {
            (workload.workload_id, target): statistics.median(
                lookup[(workload.workload_id, target, replicate)]
                for replicate in draws[workload.semantic_family_id]
            )
            for workload in spec.workloads
            for target in spec.targets
        }
        fixed_scores = {
            target: _geometric_mean(
                [workload_target[(workload.workload_id, target)] for workload in spec.workloads]
            )
            for target in spec.targets
        }
        oracle_score = _geometric_mean(
            [
                max(workload_target[(workload.workload_id, target)] for target in spec.targets)
                for workload in spec.workloads
            ]
        )
        return oracle_score / max(fixed_scores.values())

    original = {family: replicates for family in families}
    point = estimate(original)
    rng = random.Random(spec.bootstrap_seed)
    bootstraps = []
    for _ in range(spec.bootstrap_resamples):
        draws = {family: tuple(rng.choice(replicates) for _ in replicates) for family in families}
        bootstraps.append(estimate(draws))
    bootstraps.sort()
    alpha = (1.0 - spec.confidence_level) / 2.0
    return AnytimeClusteredInterval(
        cluster_count=len(families) * len(replicates),
        observation_count=len(spec.workloads) * len(replicates),
        point_estimate=point,
        lower=_percentile(bootstraps, alpha),
        upper=_percentile(bootstraps, 1.0 - alpha),
        confidence_level=spec.confidence_level,
        estimand="workload_median_oracle_over_best_fixed_geomean_ratio",
        seed=spec.bootstrap_seed,
        resamples=spec.bootstrap_resamples,
    )


def compare_anytime_hindsight(
    matches: Iterable[AnytimeMatchedObservation],
    *,
    spec: AnytimeAnalysisSpec,
) -> tuple[AnytimeHindsightComparison, ...]:
    """Compare best fixed target with a per-workload hindsight oracle."""

    rows = tuple(matches)
    grouped: dict[tuple[object, ...], list[AnytimeMatchedObservation]] = defaultdict(list)
    for row in rows:
        grouped[_budget_key(row)].append(row)
    results: list[AnytimeHindsightComparison] = []
    for key in sorted(
        grouped,
        key=lambda item: tuple("" if value is None else str(value) for value in item),
    ):
        agent, mode, iteration, wall, scope = key
        observations = grouped[key]
        reasons = {row.missing_reason for row in observations if row.missing_reason != "none"}
        status: HindsightStatus | None = None
        if reasons & {"invalid_floor", "unstable_floor_timing", "floor_infrastructure_missing"}:
            status = "invalid_floor"
        elif "trajectory_infrastructure_censored" in reasons:
            status = "infrastructure_censored"
        elif "early_resource_cap" in reasons:
            status = "early_resource_cap"
        elif any(row.early_resource_cap for row in observations):
            status = "early_resource_cap"
        elif reasons:
            status = "insufficient_coverage"
        common = dict(
            agent_id=str(agent),
            match_mode=mode,
            iteration_budget=iteration,
            wall_clock_budget_seconds=wall,
            evidence_scope=scope,
        )
        if status is not None:
            results.append(AnytimeHindsightComparison(**common, status=status, fixed_target_ids=()))
            continue

        workload_target: dict[tuple[str, str], float] = {}
        for workload in (axis.workload_id for axis in spec.workloads):
            for target in spec.targets:
                values = [
                    float(row.eager_speedup)
                    for row in observations
                    if row.workload_id == workload
                    and row.target_id == target
                    and row.eager_speedup is not None
                ]
                expected = next(
                    len(axis.replicates) for axis in spec.agents if axis.agent_id == agent
                )
                if len(values) != expected:
                    status = "insufficient_coverage"
                    break
                workload_target[(workload, target)] = statistics.median(values)
            if status is not None:
                break
        if status is not None:
            results.append(AnytimeHindsightComparison(**common, status=status, fixed_target_ids=()))
            continue

        fixed_scores = {
            target: _geometric_mean(
                [workload_target[(workload.workload_id, target)] for workload in spec.workloads]
            )
            for target in spec.targets
        }
        fixed_targets = tuple(
            sorted(
                _relative_winners(
                    fixed_scores,
                    spec.targets,
                    spec.winner_relative_tolerance,
                )
            )
        )
        best_fixed_score = max(fixed_scores.values())
        fixed = min(target for target, score in fixed_scores.items() if score == best_fixed_score)
        gains: list[AnytimeWorkloadOracleGain] = []
        oracle_values: list[float] = []
        fixed_values: list[float] = []
        for workload in spec.workloads:
            target_scores = {
                target: workload_target[(workload.workload_id, target)] for target in spec.targets
            }
            oracle_targets = _relative_winners(
                target_scores, spec.targets, spec.winner_relative_tolerance
            )
            oracle_value = max(target_scores.values())
            fixed_value = target_scores[fixed]
            oracle_values.append(oracle_value)
            fixed_values.append(fixed_value)
            gains.append(
                AnytimeWorkloadOracleGain(
                    workload_id=workload.workload_id,
                    semantic_family_id=workload.semantic_family_id,
                    fixed_target_id=fixed,
                    oracle_target_ids=oracle_targets,
                    fixed_speedup=fixed_value,
                    oracle_speedup=oracle_value,
                    oracle_over_fixed_gain=oracle_value / fixed_value,
                )
            )

        fixed_mean = _geometric_mean(fixed_values)
        oracle_mean = _geometric_mean(oracle_values)
        interval = _hindsight_clustered_interval(
            observations,
            agent_id=str(agent),
            spec=spec,
        )
        gain = oracle_mean / fixed_mean
        if not math.isclose(interval.point_estimate, gain, rel_tol=1e-12, abs_tol=1e-12):
            raise AnytimeAnalysisError("clustered interval does not match the hindsight estimand")
        results.append(
            AnytimeHindsightComparison(
                **common,
                status="complete",
                fixed_target_ids=fixed_targets,
                best_fixed_target_id=fixed,
                fixed_geometric_mean_speedup=fixed_mean,
                oracle_geometric_mean_speedup=oracle_mean,
                oracle_over_fixed_gain=gain,
                workload_gains=tuple(gains),
                clustered_gain_interval=interval,
            )
        )
    return tuple(results)


def compare_model_rankings(
    winners: Iterable[AnytimeWinnerRow],
) -> tuple[AnytimeModelRankingDisagreement, ...]:
    grouped: dict[tuple[object, ...], list[AnytimeWinnerRow]] = defaultdict(list)
    for row in winners:
        grouped[
            (
                row.workload_id,
                row.match_mode,
                row.iteration_budget,
                row.wall_clock_budget_seconds,
                row.evidence_scope,
            )
        ].append(row)
    result = []
    for key in sorted(
        grouped,
        key=lambda item: tuple("" if value is None else str(value) for value in item),
    ):
        workload, mode, iteration, wall, scope = key
        agent_sets = tuple(sorted((row.agent_id, row.winner_target_ids) for row in grouped[key]))
        sets = {targets for _agent, targets in agent_sets if targets}
        result.append(
            AnytimeModelRankingDisagreement(
                workload_id=str(workload),
                match_mode=mode,  # type: ignore[arg-type]
                iteration_budget=iteration,  # type: ignore[arg-type]
                wall_clock_budget_seconds=wall,  # type: ignore[arg-type]
                evidence_scope=scope,  # type: ignore[arg-type]
                agent_winner_sets=agent_sets,
                model_dependent=len(sets) > 1,
            )
        )
    return tuple(result)


def build_anytime_analysis(dataset: AnytimeAnalysisDataset) -> AnytimeAnalysisReport:
    """Reconstruct the complete generic report from immutable artifact projections."""

    dataset = _validate_dataset(dataset)
    rates = aggregate_anytime_rates(dataset)
    matches = match_anytime_observations(dataset)
    winners = select_anytime_winners(matches, spec=dataset.spec)
    missingness = summarize_anytime_missingness(matches)
    hindsight = compare_anytime_hindsight(matches, spec=dataset.spec)
    model_rankings = compare_model_rankings(winners)
    return AnytimeAnalysisReport(
        analysis_spec_sha256=dataset.spec.sha256,
        input_dataset_sha256=dataset.sha256,
        rates=rates,
        matches=matches,
        winners=winners,
        missingness=missingness,
        hindsight=hindsight,
        model_ranking_disagreements=model_rankings,
    )


def _number(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, tuple):
        return "|".join(str(item) for item in value)
    return str(value)


def anytime_analysis_tables(report: AnytimeAnalysisReport) -> tuple[AnytimeAnalysisTable, ...]:
    """Return hashable, deterministic table values without a dataframe dependency."""

    checked = AnytimeAnalysisReport.model_validate_json(report.model_dump_json())
    rate_columns = (
        "agent_id",
        "scientific_call_index",
        "formal_checkpoint",
        "expected_trajectories",
        "observed_turns",
        "compiled_rate",
        "correct_rate",
        "qualification_rate",
        "eligible_rate",
        "infrastructure_censored",
        "early_resource_caps",
    )
    winner_columns = (
        "agent_id",
        "workload_id",
        "match_mode",
        "iteration_budget",
        "wall_clock_budget_seconds",
        "evidence_scope",
        "status",
        "winner_target_ids",
        "replicate_disagreement",
    )
    hindsight_columns = (
        "agent_id",
        "match_mode",
        "iteration_budget",
        "wall_clock_budget_seconds",
        "evidence_scope",
        "status",
        "fixed_target_ids",
        "best_fixed_target_id",
        "fixed_geometric_mean_speedup",
        "oracle_geometric_mean_speedup",
        "oracle_over_fixed_gain",
    )
    missing_columns = (
        "agent_id",
        "match_mode",
        "iteration_budget",
        "wall_clock_budget_seconds",
        "evidence_scope",
        "expected_observations",
        "available_observations",
        "infrastructure_censored",
        "early_resource_caps",
        "invalid_floor",
        "unstable_floor_timing",
        "floor_infrastructure_missing",
        "no_eligible_incumbent",
        "not_reached",
    )
    ranking_columns = (
        "workload_id",
        "match_mode",
        "iteration_budget",
        "wall_clock_budget_seconds",
        "evidence_scope",
        "agent_winner_sets",
        "model_dependent",
    )

    def rows(
        values: Iterable[AnytimeModel], columns: tuple[str, ...]
    ) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(_number(getattr(value, column)) for column in columns) for value in values
        )

    return (
        AnytimeAnalysisTable(
            name="rates-by-turn",
            columns=rate_columns,
            rows=rows(checked.rates, rate_columns),
        ),
        AnytimeAnalysisTable(
            name="workload-winners",
            columns=winner_columns,
            rows=rows(checked.winners, winner_columns),
        ),
        AnytimeAnalysisTable(
            name="hindsight-oracle-gain",
            columns=hindsight_columns,
            rows=rows(checked.hindsight, hindsight_columns),
        ),
        AnytimeAnalysisTable(
            name="missingness",
            columns=missing_columns,
            rows=rows(checked.missingness, missing_columns),
        ),
        AnytimeAnalysisTable(
            name="model-ranking-disagreement",
            columns=ranking_columns,
            rows=rows(checked.model_ranking_disagreements, ranking_columns),
        ),
    )
