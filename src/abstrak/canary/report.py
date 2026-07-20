"""Artifact-backed construction of the preregistered A100 R1 analysis report."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, model_validator

from abstrak.canary.analysis import StudyAnalysis, TrajectoryMeasurement, analyze_study
from abstrak.canary.artifacts import (
    TrajectoryArtifactError,
    TrajectoryStore,
    verify_trajectory,
)
from abstrak.canary.baselines import BASELINE_VARIANTS
from abstrak.canary.contracts import CanaryModel, TimingSpec, TrajectoryOutcome, WorkerResult
from abstrak.canary.gates import GateRecord
from abstrak.canary.schedule import (
    R1_AGENTS,
    R1_TARGETS,
    R1_TASKS,
    R1StudySchedule,
    ScheduleCell,
    build_r1_schedule,
)
from abstrak.canary.timing import TimingProtocolSummary

DEFAULT_ARTIFACT_ROOT = "artifacts/r1-a100"
DEFAULT_FORMAL_STUDY_ID = "r1-a100-formal-v1"
DEFAULT_ORACLE_GATE_STUDY_ID = "r1-a100-oracle-gates-v1"
DEFAULT_BASELINE_GATE_STUDY_ID = "r1-a100-baseline-gates-v1"
DEFAULT_TIMING_STUDY_ID = "r1-a100-formal-timing-v1"
DEFAULT_SHAKEOUT_STUDY_ID = "r1-a100-canary"
DEFAULT_REPORT_STUDY_ID = "r1-a100-analysis-v1"

TimingDisposition = Literal[
    "stable",
    "unstable",
    "worker_failure",
    "correctness_failure",
    "missing",
]


class AnalysisReportError(ValueError):
    """Raised when source artifacts cannot form one auditable R1 report."""


class CandidateTimingRecordLike(Protocol):
    trajectory_id: str
    candidate_labels: tuple[Literal["first", "final"], ...]
    task_id: str
    target_id: str
    source_sha256: str
    summary: TimingProtocolSummary


class TrajectoryArtifact(CanaryModel):
    """One schedule cell bound to its verified terminal artifact."""

    cell: ScheduleCell
    outcome: TrajectoryOutcome
    artifact_directory: str

    @model_validator(mode="after")
    def identity_matches(self) -> TrajectoryArtifact:
        if self.cell.trajectory_id != self.outcome.trajectory_id:
            raise ValueError("schedule cell and trajectory outcome IDs differ")
        return self


class NamedCount(CanaryModel):
    """A deterministic count without relying on free-form JSON object ordering."""

    name: str = Field(min_length=1)
    count: int = Field(ge=0)


class RuntimeReference(CanaryModel):
    """One stable expert or selected B* runtime used in normalized measurements."""

    kind: Literal["oracle", "baseline"]
    task_id: str
    target_id: str
    variant: str | None = None
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    median_ms: float = Field(gt=0)


class GateCoverage(CanaryModel):
    """Completeness and stable references for the two preregistered gate matrices."""

    expected_oracles: int = 12
    received_oracles: int = Field(ge=0, le=12)
    stable_oracles: int = Field(ge=0, le=12)
    expert_oracle_complete: bool
    expected_baselines: int = 12
    received_baselines: int = Field(ge=0, le=12)
    stable_baselines: int = Field(ge=0, le=12)
    baseline_complete: bool
    expert_references: tuple[RuntimeReference, ...]
    selected_baselines: tuple[RuntimeReference, ...]


class ShakeoutCoverage(CanaryModel):
    """Frozen canary-floor gate derived from the sealed shakeout trajectories."""

    expected_trajectories: int = 12
    received_trajectories: int = Field(ge=0, le=12)
    qualified_at_final: int = Field(ge=0, le=12)
    supported_target_ids: tuple[str, ...]
    passed: bool


class FormalCoverage(CanaryModel):
    """Primary non-performance counts from the complete formal matrix."""

    expected_trajectories: int = 48
    received_trajectories: int = Field(ge=0, le=48)
    observed_trajectories: int = Field(ge=0, le=48)
    infrastructure_censored: int = Field(ge=0, le=48)
    qualified_at_first: int = Field(ge=0, le=48)
    qualified_at_final: int = Field(ge=0, le=48)
    total_calls: int = Field(ge=0, le=192)
    known_input_tokens: int = Field(ge=0)
    known_output_tokens: int = Field(ge=0)
    incomplete_usage_trajectories: int = Field(ge=0, le=48)
    total_wall_seconds: float = Field(ge=0)
    outcome_status_counts: tuple[NamedCount, ...]
    infrastructure_trajectory_ids: tuple[str, ...]


class CandidateTimingAudit(CanaryModel):
    """Timing disposition for one correctness-qualified first or final candidate."""

    trajectory_id: str
    candidate_label: Literal["first", "final"]
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: TimingDisposition
    median_ms: float | None = Field(default=None, gt=0)
    attempt_worst_cvs: tuple[float, ...] = Field(default=(), max_length=2)

    @model_validator(mode="after")
    def stable_is_the_only_publishable_latency(self) -> CandidateTimingAudit:
        if (self.median_ms is not None) != (self.status == "stable"):
            raise ValueError("only stable candidate timing may expose a median")
        if any(not math.isfinite(value) or value < 0 for value in self.attempt_worst_cvs):
            raise ValueError("timing CVs must be finite and non-negative")
        return self


class TimingCoverage(CanaryModel):
    """Completeness and disposition counts for qualified candidate timing."""

    expected_first_candidates: int = Field(ge=0, le=48)
    expected_final_candidates: int = Field(ge=0, le=48)
    first_status_counts: tuple[NamedCount, ...]
    final_status_counts: tuple[NamedCount, ...]
    audits: tuple[CandidateTimingAudit, ...]


class R1AnalysisReport(CanaryModel):
    """Complete source-linked preregistered analysis product."""

    schema_version: Literal["abstrak-canary-study-report.v1"] = (
        "abstrak-canary-study-report.v1"
    )
    formal_study_id: str
    oracle_gate_study_id: str
    baseline_gate_study_id: str
    timing_study_id: str
    shakeout_study_id: str
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gate_coverage: GateCoverage
    shakeout_coverage: ShakeoutCoverage
    formal_coverage: FormalCoverage
    timing_coverage: TimingCoverage
    measurements: tuple[TrajectoryMeasurement, ...]
    analysis: StudyAnalysis


def _qualified(result: WorkerResult | None) -> bool:
    return result is not None and result.status == "completed" and result.correct


def _attempt_worst_cvs(summary: TimingProtocolSummary) -> tuple[float, ...]:
    worst: list[float] = []
    for attempt in summary.attempts:
        values = list(attempt.process_cvs)
        if attempt.across_process_cv is not None:
            values.append(attempt.across_process_cv)
        if values:
            worst.append(max(values))
    return tuple(worst)


def _count_names(values: Iterable[str], order: tuple[str, ...]) -> tuple[NamedCount, ...]:
    counts = Counter(values)
    return tuple(NamedCount(name=name, count=counts[name]) for name in order)


def _gate_coverage(
    oracle_records: tuple[GateRecord, ...],
    baseline_records: tuple[GateRecord, ...],
) -> tuple[GateCoverage, dict[str, float], dict[tuple[str, str], float]]:
    expected_oracles = {(task, target) for task in R1_TASKS for target in R1_TARGETS}
    oracles: dict[tuple[str, str], GateRecord] = {}
    for record in oracle_records:
        key = (record.task_id, record.target_id)
        if record.kind != "oracle" or record.variant is not None or key not in expected_oracles:
            raise AnalysisReportError(f"unexpected expert-oracle gate record: {key}")
        if key in oracles:
            raise AnalysisReportError(f"duplicate expert-oracle gate record: {key}")
        oracles[key] = record

    expected_baselines = {
        (task, variant) for task in R1_TASKS for variant in BASELINE_VARIANTS
    }
    baselines: dict[tuple[str, str], GateRecord] = {}
    for record in baseline_records:
        key = (record.task_id, record.variant or "")
        if (
            record.kind != "baseline"
            or record.target_id != "triton-a100"
            or key not in expected_baselines
        ):
            raise AnalysisReportError(f"unexpected baseline gate record: {key}")
        if key in baselines:
            raise AnalysisReportError(f"duplicate baseline gate record: {key}")
        baselines[key] = record

    stable_oracles = tuple(
        record for record in oracles.values() if record.summary.stable
    )
    expert_complete = (
        set(oracles) == expected_oracles and len(stable_oracles) == len(expected_oracles)
    )
    expert_latency = {
        key: record.summary.median_ms
        for key, record in oracles.items()
        if record.summary.stable and record.summary.median_ms is not None
    }

    selected = {}
    for task in R1_TASKS:
        stable = tuple(
            record
            for (record_task, _), record in baselines.items()
            if record_task == task
            and record.summary.stable
            and record.summary.median_ms is not None
        )
        if stable:
            selected[task] = min(
                stable, key=lambda record: record.summary.median_ms or float("inf")
            )
    baseline_complete = set(baselines) == expected_baselines and set(selected) == set(R1_TASKS)
    baseline_latency = {
        task: record.summary.median_ms
        for task, record in selected.items()
        if record.summary.median_ms is not None
    }
    coverage = GateCoverage(
        received_oracles=len(oracles),
        stable_oracles=len(stable_oracles),
        expert_oracle_complete=expert_complete,
        received_baselines=len(baselines),
        stable_baselines=sum(record.summary.stable for record in baselines.values()),
        baseline_complete=baseline_complete,
        expert_references=tuple(
            RuntimeReference(
                kind="oracle",
                task_id=task,
                target_id=target,
                source_sha256=record.source_sha256,
                median_ms=record.summary.median_ms,
            )
            for task in R1_TASKS
            for target in R1_TARGETS
            if (record := oracles.get((task, target))) is not None
            and record.summary.stable
            and record.summary.median_ms is not None
        ),
        selected_baselines=tuple(
            RuntimeReference(
                kind="baseline",
                task_id=task,
                target_id=record.target_id,
                variant=record.variant,
                source_sha256=record.source_sha256,
                median_ms=record.summary.median_ms,
            )
            for task in R1_TASKS
            if (record := selected.get(task)) is not None
            and record.summary.median_ms is not None
        ),
    )
    return coverage, baseline_latency, expert_latency


def _shakeout_coverage(records: tuple[TrajectoryArtifact, ...]) -> ShakeoutCoverage:
    by_id = {record.cell.trajectory_id: record for record in records}
    qualified = tuple(
        record for record in records if _qualified(record.outcome.final_sealed_result)
    )
    supported = tuple(
        target
        for target in R1_TARGETS
        if any(record.cell.target_id == target for record in qualified)
    )
    return ShakeoutCoverage(
        received_trajectories=len(by_id),
        qualified_at_final=len(qualified),
        supported_target_ids=supported,
        passed=len(by_id) == 12 and supported == R1_TARGETS,
    )


def _timing_audits(
    formal_records: tuple[TrajectoryArtifact, ...],
    timing_records: tuple[CandidateTimingRecordLike, ...],
) -> tuple[CandidateTimingAudit, ...]:
    formal_by_id = {record.cell.trajectory_id: record for record in formal_records}
    timing_by_key: dict[tuple[str, str], CandidateTimingRecordLike] = {}
    for record in timing_records:
        trajectory = formal_by_id.get(record.trajectory_id)
        if trajectory is None:
            raise AnalysisReportError(
                f"timing record has no formal trajectory: {record.trajectory_id}"
            )
        if len(record.candidate_labels) != len(set(record.candidate_labels)):
            raise AnalysisReportError(
                f"timing record has duplicate labels: {record.trajectory_id}"
            )
        if (
            record.summary.job_kind != "sealed"
            or record.summary.task_id != record.task_id
            or record.summary.target_id != record.target_id
        ):
            raise AnalysisReportError(
                f"candidate timing summary identity mismatch: {record.trajectory_id}"
            )
        for label in record.candidate_labels:
            key = (record.trajectory_id, label)
            if key in timing_by_key:
                raise AnalysisReportError(f"duplicate candidate timing record: {key}")
            result = (
                trajectory.outcome.first_sealed_result
                if label == "first"
                else trajectory.outcome.final_sealed_result
            )
            expected_sha256 = (
                trajectory.outcome.first_candidate_sha256
                if label == "first"
                else trajectory.outcome.final_candidate_sha256
            )
            if not _qualified(result):
                raise AnalysisReportError(
                    f"timing record targets an unqualified candidate: {key}"
                )
            if (
                record.task_id != trajectory.cell.task_id
                or record.target_id != trajectory.cell.target_id
                or record.source_sha256 != expected_sha256
                or record.summary.candidate_sha256 != expected_sha256
            ):
                raise AnalysisReportError(f"candidate timing identity mismatch: {key}")
            timing_by_key[key] = record

    audits: list[CandidateTimingAudit] = []
    for trajectory in formal_records:
        outcome = trajectory.outcome
        for label, result, candidate_sha256 in (
            ("first", outcome.first_sealed_result, outcome.first_candidate_sha256),
            ("final", outcome.final_sealed_result, outcome.final_candidate_sha256),
        ):
            if not _qualified(result):
                continue
            assert candidate_sha256 is not None
            timing = timing_by_key.get((outcome.trajectory_id, label))
            audits.append(
                CandidateTimingAudit(
                    trajectory_id=outcome.trajectory_id,
                    candidate_label=label,
                    candidate_sha256=candidate_sha256,
                    status="missing" if timing is None else timing.summary.status,
                    median_ms=(
                        timing.summary.median_ms
                        if timing is not None and timing.summary.stable
                        else None
                    ),
                    attempt_worst_cvs=(
                        () if timing is None else _attempt_worst_cvs(timing.summary)
                    ),
                )
            )
    return tuple(audits)


def build_analysis_report(
    *,
    formal_records: Iterable[TrajectoryArtifact],
    oracle_records: Iterable[GateRecord],
    baseline_records: Iterable[GateRecord],
    timing_records: Iterable[CandidateTimingRecordLike],
    shakeout_records: Iterable[TrajectoryArtifact],
    formal_study_id: str = DEFAULT_FORMAL_STUDY_ID,
    oracle_gate_study_id: str = DEFAULT_ORACLE_GATE_STUDY_ID,
    baseline_gate_study_id: str = DEFAULT_BASELINE_GATE_STUDY_ID,
    timing_study_id: str = DEFAULT_TIMING_STUDY_ID,
    shakeout_study_id: str = DEFAULT_SHAKEOUT_STUDY_ID,
    schedule: R1StudySchedule | None = None,
) -> R1AnalysisReport:
    """Normalize verified artifacts and apply the frozen R1 decision rules."""

    frozen = schedule or build_r1_schedule()
    formal = tuple(formal_records)
    shakeout = tuple(shakeout_records)
    formal_ids = [record.cell.trajectory_id for record in formal]
    if len(formal_ids) != len(set(formal_ids)):
        raise AnalysisReportError("formal trajectories contain a duplicate schedule cell")
    expected_formal = {cell.trajectory_id for cell in frozen.formal}
    if any(identifier not in expected_formal for identifier in formal_ids):
        raise AnalysisReportError("formal trajectories contain a cell outside the frozen schedule")
    formal_by_id = {record.cell.trajectory_id: record for record in formal}
    formal = tuple(
        formal_by_id[cell.trajectory_id]
        for cell in frozen.formal
        if cell.trajectory_id in formal_by_id
    )
    expected_shakeout = {cell.trajectory_id for cell in frozen.shakeout}
    shakeout_ids = [record.cell.trajectory_id for record in shakeout]
    if len(shakeout_ids) != len(set(shakeout_ids)):
        raise AnalysisReportError("shakeout trajectories contain a duplicate schedule cell")
    if any(record.cell.trajectory_id not in expected_shakeout for record in shakeout):
        raise AnalysisReportError(
            "shakeout trajectories contain a cell outside the frozen schedule"
        )
    shakeout_by_id = {record.cell.trajectory_id: record for record in shakeout}
    shakeout = tuple(
        shakeout_by_id[cell.trajectory_id]
        for cell in frozen.shakeout
        if cell.trajectory_id in shakeout_by_id
    )

    gates, baseline_latency, expert_latency = _gate_coverage(
        tuple(oracle_records), tuple(baseline_records)
    )
    shakeout_coverage = _shakeout_coverage(shakeout)
    audits = _timing_audits(formal, tuple(timing_records))
    audit_by_key = {
        (audit.trajectory_id, audit.candidate_label): audit for audit in audits
    }

    measurements: list[TrajectoryMeasurement] = []
    infrastructure_ids = [
        cell.trajectory_id
        for cell in frozen.formal
        if cell.trajectory_id not in formal_by_id
    ]
    for record in formal:
        outcome = record.outcome
        if outcome.status == "provider_error":
            terminal_status = "provider_error"
        elif outcome.status == "worker_error":
            terminal_status = "worker_error"
        else:
            terminal_status = "observed"
        infrastructure = terminal_status != "observed"
        if infrastructure:
            infrastructure_ids.append(outcome.trajectory_id)
        first_qualified = not infrastructure and _qualified(outcome.first_sealed_result)
        final_qualified = not infrastructure and _qualified(outcome.final_sealed_result)
        final_timing = audit_by_key.get((outcome.trajectory_id, "final"))
        measurements.append(
            TrajectoryMeasurement(
                agent_id=record.cell.agent_id,
                task_id=record.cell.task_id,
                target_id=record.cell.target_id,
                replicate=record.cell.replicate,
                terminal_status=terminal_status,
                qualified_at_first=first_qualified,
                qualified_at_final=final_qualified,
                candidate_latency_ms=(
                    final_timing.median_ms
                    if final_timing is not None and final_timing.status == "stable"
                    else None
                ),
                baseline_latency_ms=baseline_latency.get(record.cell.task_id),
                expert_latency_ms=expert_latency.get(
                    (record.cell.task_id, record.cell.target_id)
                ),
                timing_cvs=(
                    final_timing.attempt_worst_cvs if final_timing is not None else ()
                ),
                calls=outcome.calls,
                input_tokens=outcome.known_input_tokens,
                output_tokens=outcome.known_output_tokens,
                wall_seconds=(outcome.finished_at_utc - outcome.started_at_utc).total_seconds(),
                gpu_seconds=None,
            )
        )

    analysis = analyze_study(
        measurements,
        agents=R1_AGENTS,
        tasks=R1_TASKS,
        targets=R1_TARGETS,
        expert_oracle_complete=gates.expert_oracle_complete,
        baseline_complete=gates.baseline_complete,
        shakeout_passed=shakeout_coverage.passed,
    )
    formal_status_order = (
        "finished",
        "call_limit",
        "budget_exhausted",
        "provider_error",
        "worker_error",
        "no_candidate",
    )
    formal_coverage = FormalCoverage(
        received_trajectories=len(formal),
        observed_trajectories=sum(
            measurement.terminal_status == "observed" for measurement in measurements
        ),
        infrastructure_censored=len(infrastructure_ids),
        qualified_at_first=sum(measurement.qualified_at_first for measurement in measurements),
        qualified_at_final=sum(measurement.qualified_at_final for measurement in measurements),
        total_calls=sum(record.outcome.calls for record in formal),
        known_input_tokens=sum(record.outcome.known_input_tokens for record in formal),
        known_output_tokens=sum(record.outcome.known_output_tokens for record in formal),
        incomplete_usage_trajectories=sum(
            not record.outcome.usage_complete for record in formal
        ),
        total_wall_seconds=sum(measurement.wall_seconds for measurement in measurements),
        outcome_status_counts=_count_names(
            (record.outcome.status for record in formal), formal_status_order
        ),
        infrastructure_trajectory_ids=tuple(infrastructure_ids),
    )
    timing_status_order = (
        "stable",
        "unstable",
        "worker_failure",
        "correctness_failure",
        "missing",
    )
    first_audits = tuple(audit for audit in audits if audit.candidate_label == "first")
    final_audits = tuple(audit for audit in audits if audit.candidate_label == "final")
    timing_coverage = TimingCoverage(
        expected_first_candidates=len(first_audits),
        expected_final_candidates=len(final_audits),
        first_status_counts=_count_names(
            (audit.status for audit in first_audits), timing_status_order
        ),
        final_status_counts=_count_names(
            (audit.status for audit in final_audits), timing_status_order
        ),
        audits=audits,
    )
    return R1AnalysisReport(
        formal_study_id=formal_study_id,
        oracle_gate_study_id=oracle_gate_study_id,
        baseline_gate_study_id=baseline_gate_study_id,
        timing_study_id=timing_study_id,
        shakeout_study_id=shakeout_study_id,
        schedule_sha256=frozen.sha256,
        gate_coverage=gates,
        shakeout_coverage=shakeout_coverage,
        formal_coverage=formal_coverage,
        timing_coverage=timing_coverage,
        measurements=tuple(measurements),
        analysis=analysis,
    )


def _load_json(directory: Path, filename: str, model: type[CanaryModel]) -> CanaryModel:
    try:
        verify_trajectory(directory)
        return model.model_validate_json((directory / filename).read_text(encoding="utf-8"))
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise AnalysisReportError(f"invalid sealed artifact {directory}: {error}") from error


def _load_trajectories(
    root: Path,
    study_id: str,
    cells: tuple[ScheduleCell, ...],
) -> tuple[TrajectoryArtifact, ...]:
    records: list[TrajectoryArtifact] = []
    for cell in cells:
        directory = root / study_id / cell.trajectory_id
        if not directory.exists():
            continue
        outcome = _load_json(directory, "outcome.json", TrajectoryOutcome)
        assert isinstance(outcome, TrajectoryOutcome)
        try:
            manifest = json.loads((directory / "run-manifest.json").read_text(encoding="utf-8"))
            if (
                manifest["task"]["id"] != cell.task_id
                or manifest["target"]["id"] != cell.target_id
                or manifest["resolved_provider"]["model"]["id"] != cell.agent_id
            ):
                raise AnalysisReportError(f"trajectory manifest identity mismatch: {directory}")
        except (KeyError, OSError, json.JSONDecodeError) as error:
            raise AnalysisReportError(f"invalid trajectory manifest: {directory}") from error
        records.append(
            TrajectoryArtifact(
                cell=cell,
                outcome=outcome,
                artifact_directory=str(directory),
            )
        )
    return tuple(records)


def _verify_formal_study_manifest(
    root: Path, study_id: str, schedule: R1StudySchedule
) -> None:
    directory = root / study_id / "study-manifest"
    try:
        verify_trajectory(directory)
        manifest = json.loads(
            (directory / "run-manifest.json").read_text(encoding="utf-8")
        )
        if (
            manifest["study_id"] != study_id
            or manifest["schedule_sha256"] != schedule.sha256
        ):
            raise AnalysisReportError("formal study manifest does not match the frozen schedule")
    except (KeyError, OSError, json.JSONDecodeError, TrajectoryArtifactError) as error:
        raise AnalysisReportError(f"invalid formal study manifest: {directory}") from error


def _load_gate_records(root: Path, study_id: str) -> tuple[GateRecord, ...]:
    directory = root / study_id
    if not directory.exists():
        return ()
    records: list[GateRecord] = []
    for path in sorted(directory.glob("*/gate-record.json")):
        record = _load_json(path.parent, path.name, GateRecord)
        assert isinstance(record, GateRecord)
        records.append(record)
    return tuple(records)


def _load_timing_records(
    root: Path,
    study_id: str,
    formal_study_id: str,
    schedule: R1StudySchedule,
) -> tuple[CandidateTimingRecordLike, ...]:
    directory = root / study_id
    if not directory.exists():
        return ()
    from abstrak.canary.postprocess_timing import CandidateTimingRecord

    manifest_directory = directory / "study-manifest"
    try:
        verify_trajectory(manifest_directory)
        study_manifest = json.loads(
            (manifest_directory / "run-manifest.json").read_text(encoding="utf-8")
        )
        formal_manifest_path = (
            root / formal_study_id / "study-manifest" / "run-manifest.json"
        )
        formal_manifest_sha256 = hashlib.sha256(formal_manifest_path.read_bytes()).hexdigest()
        if (
            study_manifest["schema_version"]
            != "abstrak-candidate-timing-study-manifest.v1"
            or study_manifest["formal_study_id"] != formal_study_id
            or study_manifest["timing_study_id"] != study_id
            or study_manifest["formal_schedule_sha256"] != schedule.sha256
            or study_manifest["formal_manifest_sha256"] != formal_manifest_sha256
            or study_manifest["timing"] != TimingSpec().model_dump(mode="json")
            or study_manifest["transport"]["device"] != "cuda:0"
        ):
            raise AnalysisReportError("candidate timing study manifest identity mismatch")
    except (KeyError, OSError, json.JSONDecodeError, TrajectoryArtifactError) as error:
        raise AnalysisReportError(
            f"invalid candidate timing study manifest: {manifest_directory}"
        ) from error

    records: list[CandidateTimingRecordLike] = []
    by_artifact_id: dict[str, CandidateTimingRecordLike] = {}
    for path in sorted(directory.glob("*/timing-record.json")):
        record = _load_json(path.parent, path.name, CandidateTimingRecord)
        try:
            run_manifest = json.loads(
                (path.parent / "run-manifest.json").read_text(encoding="utf-8")
            )
            if (
                run_manifest["formal_study_id"] != formal_study_id
                or run_manifest["timing_study_id"] != study_id
            ):
                raise AnalysisReportError(
                    f"candidate timing manifest study mismatch: {path.parent}"
                )
        except (KeyError, OSError, json.JSONDecodeError) as error:
            raise AnalysisReportError(
                f"invalid candidate timing manifest: {path.parent}"
            ) from error
        records.append(record)  # type: ignore[arg-type]
        by_artifact_id[path.parent.name] = record  # type: ignore[assignment]
    try:
        manifest_records = study_manifest["records"]
        if study_manifest["candidate_count"] != len(records) or len(
            manifest_records
        ) != len(records):
            raise AnalysisReportError("candidate timing study count mismatch")
        if len({item["artifact_id"] for item in manifest_records}) != len(manifest_records):
            raise AnalysisReportError("candidate timing study has duplicate artifact IDs")
        for item in manifest_records:
            record = by_artifact_id.get(item["artifact_id"])
            if record is None or (
                item["trajectory_id"],
                item["task_id"],
                item["target_id"],
                tuple(item["candidate_labels"]),
                item["source_sha256"],
            ) != (
                record.trajectory_id,
                record.task_id,
                record.target_id,
                record.candidate_labels,
                record.source_sha256,
            ):
                raise AnalysisReportError("candidate timing study record mismatch")
    except (KeyError, TypeError) as error:
        raise AnalysisReportError("invalid candidate timing study records") from error
    return tuple(records)


def load_analysis_report(
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    formal_study_id: str = DEFAULT_FORMAL_STUDY_ID,
    oracle_gate_study_id: str = DEFAULT_ORACLE_GATE_STUDY_ID,
    baseline_gate_study_id: str = DEFAULT_BASELINE_GATE_STUDY_ID,
    timing_study_id: str = DEFAULT_TIMING_STUDY_ID,
    shakeout_study_id: str = DEFAULT_SHAKEOUT_STUDY_ID,
) -> R1AnalysisReport:
    """Verify and load every source artifact before constructing the report."""

    root = Path(artifact_root).expanduser()
    schedule = build_r1_schedule()
    _verify_formal_study_manifest(root, formal_study_id, schedule)
    return build_analysis_report(
        formal_records=_load_trajectories(root, formal_study_id, schedule.formal),
        oracle_records=_load_gate_records(root, oracle_gate_study_id),
        baseline_records=_load_gate_records(root, baseline_gate_study_id),
        timing_records=_load_timing_records(
            root,
            timing_study_id,
            formal_study_id,
            schedule,
        ),
        shakeout_records=_load_trajectories(root, shakeout_study_id, schedule.shakeout),
        formal_study_id=formal_study_id,
        oracle_gate_study_id=oracle_gate_study_id,
        baseline_gate_study_id=baseline_gate_study_id,
        timing_study_id=timing_study_id,
        shakeout_study_id=shakeout_study_id,
        schedule=schedule,
    )


def write_analysis_report(
    report: R1AnalysisReport,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    report_study_id: str = DEFAULT_REPORT_STUDY_ID,
) -> tuple[Path, bool]:
    """Seal a deterministic report, or verify and resume an identical one."""

    directory = Path(artifact_root).expanduser() / report_study_id / "study-report"
    if directory.exists():
        existing = _load_json(directory, "analysis-report.json", R1AnalysisReport)
        if existing != report:
            raise AnalysisReportError("existing sealed analysis report differs from current inputs")
        return directory, True
    store = TrajectoryStore.create(artifact_root, report_study_id, "study-report")
    store.write_json(
        "run-manifest.json",
        {
            "schema_version": "abstrak-canary-study-report-manifest.v1",
            "report_study_id": report_study_id,
            "formal_study_id": report.formal_study_id,
            "oracle_gate_study_id": report.oracle_gate_study_id,
            "baseline_gate_study_id": report.baseline_gate_study_id,
            "timing_study_id": report.timing_study_id,
            "shakeout_study_id": report.shakeout_study_id,
            "schedule_sha256": report.schedule_sha256,
        },
    )
    store.write_json("analysis-report.json", report)
    store.seal()
    return store.run_directory, False
