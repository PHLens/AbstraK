"""Pure descriptive analysis for the exploratory KernelBench agent pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from abstrak.evaluation.agent_contracts import AgentAttemptRecord, KernelBenchAgentStudy
from abstrak.evaluation.agent_provider import AgentUsage, extract_agent_usage


class AgentAnalysisError(ValueError):
    """Raised when a pilot run cannot be analyzed."""


_CSV_FIELDS = (
    "record_type",
    "model_id",
    "task_ref",
    "task_name",
    "stratum",
    "target",
    "iteration",
    "token_budget",
    "budget_status",
    "attempts_completed",
    "generation_status",
    "evaluation_status",
    "compiled",
    "correct",
    "speedup",
    "best_speedup",
    "best_correct_runtime_ms",
    "utility",
    "deployment_utility",
    "correctness_coverage",
    "winner_status",
    "winner_target",
    "winner_speedup",
    "tied_targets",
    "best_fixed_target",
    "tied_best_fixed_targets",
    "target_geomean_utilities",
    "target_correctness_coverages",
    "best_fixed_geomean_utility",
    "best_fixed_correctness_coverage",
    "oracle_geomean_utility",
    "oracle_correctness_coverage",
    "oracle_gain_ratio",
    "oracle_gain_percent",
    "equal_split_per_target_budget",
    "equal_split_unspent_tokens",
    "equal_split_budget_status",
    "equal_split_geomean_utility",
    "equal_split_correctness_coverage",
    "equal_split_gain_ratio",
    "equal_split_gain_percent",
    "fixed_utility",
    "oracle_utility",
    "usage_status",
    "usage_source",
    "attempt_tokens",
    "exact_prefix_attempts",
    "exact_prefix_tokens",
    "known_token_lower_bound",
    "first_unknown_iteration",
    "censored",
    "attempt_count",
    "static_check_passes",
    "compiled_count",
    "correct_count",
    "reference_speedup",
    "reference_realization",
    "reference_label",
    "reference_winner_status",
    "reference_winner_targets",
)

_JSON_CSV_FIELDS = {
    "tied_targets",
    "tied_best_fixed_targets",
    "target_geomean_utilities",
    "target_correctness_coverages",
    "reference_winner_targets",
}


@dataclass(frozen=True)
class ResolvedUsage:
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    source: str

    @property
    def exact(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None

    @property
    def total(self) -> int | None:
        if not self.exact:
            return None
        return int(self.input_tokens) + int(self.output_tokens)

    @property
    def known_lower_bound(self) -> int:
        return (self.input_tokens or 0) + (self.output_tokens or 0)


@dataclass(frozen=True)
class CandidateState:
    attempts_completed: int
    correct: bool
    best_speedup: float | None
    best_correct_runtime_ms: float | None


@dataclass(frozen=True)
class TokenTrajectory:
    attempts: tuple[AgentAttemptRecord, ...]
    usages: tuple[ResolvedUsage, ...]
    exact_boundaries: tuple[int, ...]
    first_unknown_iteration: int | None
    configured_iterations: int

    @property
    def exact_prefix_tokens(self) -> int:
        return self.exact_boundaries[-1] if self.exact_boundaries else 0

    @property
    def known_token_lower_bound(self) -> int:
        return sum(usage.known_lower_bound for usage in self.usages)

    @property
    def incomplete_trajectory(self) -> bool:
        return len(self.attempts) < self.configured_iterations

    def at_budget(self, budget: int) -> tuple[str, CandidateState]:
        completed = bisect_right(self.exact_boundaries, budget)
        state = _candidate_state(self.attempts[:completed])
        beyond_prefix = budget > self.exact_prefix_tokens
        if beyond_prefix and (
            self.first_unknown_iteration is not None or self.incomplete_trajectory
        ):
            return "censored", state
        return "exact", state


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AgentAnalysisError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AgentAnalysisError(f"{path} must contain one JSON object")
    return payload


def _load_attempts(path: Path) -> tuple[AgentAttemptRecord, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AgentAnalysisError(f"cannot read {path}: {error}") from error

    attempts: list[AgentAttemptRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            attempts.append(AgentAttemptRecord.model_validate_json(line))
        except ValidationError as error:
            raise AgentAnalysisError(
                f"invalid attempt at {path}:{line_number}: {error.errors()[0]['msg']}"
            ) from None
    return tuple(attempts)


def _geometric_mean(values: Sequence[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _winner(
    values: Mapping[str, float | None], target_order: Sequence[str]
) -> tuple[str | None, float | None, list[str]]:
    valid = {target: value for target, value in values.items() if value is not None}
    if not valid:
        return None, None, []
    maximum = max(valid.values())
    tied = [
        target
        for target in target_order
        if target in valid and math.isclose(valid[target], maximum, rel_tol=1e-12, abs_tol=1e-12)
    ]
    return tied[0], maximum, tied


def _candidate_state(attempts: Sequence[AgentAttemptRecord]) -> CandidateState:
    correct = any(attempt.correct for attempt in attempts)
    measured = [
        attempt
        for attempt in attempts
        if attempt.correct and attempt.speedup is not None
    ]
    best = max(measured, key=lambda attempt: float(attempt.speedup)) if measured else None
    return CandidateState(
        attempts_completed=len(attempts),
        correct=correct,
        best_speedup=best.speedup if best is not None else None,
        best_correct_runtime_ms=(best.candidate_runtime_ms if best is not None else None),
    )


def _deployment_utility(state: CandidateState) -> float:
    return max(1.0, state.best_speedup) if state.best_speedup is not None else 1.0


def _trajectory_key(attempt: AgentAttemptRecord) -> tuple[str, str, str]:
    return attempt.model_id, attempt.task_ref, attempt.target


def _coalesce_usage_field(
    field: str,
    candidates: Sequence[tuple[str, int | None]],
    attempt: AgentAttemptRecord,
) -> tuple[int | None, tuple[str, ...]]:
    present = [(source, value) for source, value in candidates if value is not None]
    values = {value for _, value in present}
    if len(values) > 1:
        rendered = ", ".join(f"{source}={value}" for source, value in present)
        raise AgentAnalysisError(
            f"conflicting {field} for {attempt.trajectory_id} iteration {attempt.iteration}: "
            f"{rendered}"
        )
    value = present[0][1] if present else None
    return value, tuple(source for source, _ in present)


def _resolve_attempt_usage(run_path: Path, attempt: AgentAttemptRecord) -> ResolvedUsage:
    response_payload: Mapping[str, Any] = {}
    raw_usage = AgentUsage()
    if attempt.response_path:
        response_path = (run_path / attempt.response_path).resolve()
        if not response_path.is_relative_to(run_path):
            raise AgentAnalysisError(
                f"response path escapes run directory: {attempt.response_path}"
            )
        response_payload = _read_json_mapping(response_path)
        raw_response = response_payload.get("raw_response")
        raw_usage = extract_agent_usage(raw_response if isinstance(raw_response, Mapping) else None)

    field_candidates = {
        "input_tokens": (
            ("attempt", attempt.input_tokens),
            ("response", _nonnegative_int(response_payload.get("input_tokens"))),
            ("raw_response", raw_usage.input_tokens),
        ),
        "cached_input_tokens": (
            ("attempt", attempt.cached_input_tokens),
            ("response", _nonnegative_int(response_payload.get("cached_input_tokens"))),
            ("raw_response", raw_usage.cached_input_tokens),
        ),
        "output_tokens": (
            ("attempt", attempt.output_tokens),
            ("response", _nonnegative_int(response_payload.get("output_tokens"))),
            ("raw_response", raw_usage.output_tokens),
        ),
        "reasoning_tokens": (
            ("attempt", attempt.reasoning_tokens),
            ("response", _nonnegative_int(response_payload.get("reasoning_tokens"))),
            ("raw_response", raw_usage.reasoning_tokens),
        ),
    }
    resolved: dict[str, int | None] = {}
    sources: set[str] = set()
    for field, candidates in field_candidates.items():
        value, field_sources = _coalesce_usage_field(field, candidates, attempt)
        resolved[field] = value
        sources.update(field_sources)
    return ResolvedUsage(
        input_tokens=resolved["input_tokens"],
        cached_input_tokens=resolved["cached_input_tokens"],
        output_tokens=resolved["output_tokens"],
        reasoning_tokens=resolved["reasoning_tokens"],
        source="+".join(sorted(sources)) if sources else "unknown",
    )


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _build_token_trajectory(
    attempts: Sequence[AgentAttemptRecord],
    usages: Sequence[ResolvedUsage],
    *,
    configured_iterations: int,
) -> TokenTrajectory:
    boundaries: list[int] = []
    cumulative = 0
    first_unknown: int | None = None
    for attempt, usage in zip(attempts, usages, strict=True):
        if not usage.exact:
            first_unknown = attempt.iteration
            break
        cumulative += int(usage.total)
        boundaries.append(cumulative)
    return TokenTrajectory(
        attempts=tuple(attempts),
        usages=tuple(usages),
        exact_boundaries=tuple(boundaries),
        first_unknown_iteration=first_unknown,
        configured_iterations=configured_iterations,
    )


def _load_references(
    reference_file: str | Path | None,
    *,
    task_refs: Sequence[str],
    targets: Sequence[str],
) -> tuple[list[dict[str, Any]], str | None]:
    if reference_file is None:
        return [], None
    path = Path(reference_file).expanduser().resolve()
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise AgentAnalysisError(f"cannot read reference file {path}: {error}") from error
    try:
        rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    except (UnicodeDecodeError, csv.Error) as error:
        raise AgentAnalysisError(f"cannot parse reference file {path}: {error}") from error
    required = {"task_ref", "target", "speedup", "label"}
    if not rows:
        raise AgentAnalysisError(f"reference file {path} has no rows")
    if not required.issubset(rows[0]):
        raise AgentAnalysisError(
            f"reference file {path} requires columns: {', '.join(sorted(required))}"
        )

    parsed: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    task_set = set(task_refs)
    target_set = set(targets)
    for line_number, row in enumerate(rows, start=2):
        task_value = row.get("task_ref")
        target_value = row.get("target")
        label_value = row.get("label")
        task_ref = task_value.strip() if isinstance(task_value, str) else ""
        target = target_value.strip() if isinstance(target_value, str) else ""
        label = label_value.strip() if isinstance(label_value, str) else ""
        if task_ref not in task_set:
            raise AgentAnalysisError(f"unknown reference task at {path}:{line_number}: {task_ref}")
        if target not in target_set:
            raise AgentAnalysisError(f"unknown reference target at {path}:{line_number}: {target}")
        coordinate = task_ref, target
        if coordinate in seen:
            raise AgentAnalysisError(f"duplicate reference coordinate: {coordinate}")
        seen.add(coordinate)
        try:
            speedup = float(row.get("speedup"))
        except (TypeError, ValueError):
            raise AgentAnalysisError(
                f"invalid reference speedup at {path}:{line_number}"
            ) from None
        if not math.isfinite(speedup) or speedup <= 0:
            raise AgentAnalysisError(f"invalid reference speedup at {path}:{line_number}")
        parsed.append(
            {
                "task_ref": task_ref,
                "target": target,
                "speedup": speedup,
                "label": label,
            }
        )

    by_task = {
        task_ref: {row["target"]: row["speedup"] for row in parsed if row["task_ref"] == task_ref}
        for task_ref in task_refs
    }
    for row in parsed:
        values = by_task[row["task_ref"]]
        if set(values) != target_set:
            row["winner_status"] = "incomplete"
            row["winner_targets"] = []
            continue
        _, _, tied = _winner(values, targets)
        row["winner_status"] = "tie" if len(tied) > 1 else "unique"
        row["winner_targets"] = tied
    return parsed, hashlib.sha256(raw).hexdigest()


def _curve_row(
    attempt: AgentAttemptRecord,
    state: CandidateState,
    *,
    stratum: str,
    reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    reference_speedup = reference["speedup"] if reference is not None else None
    return {
        "model_id": attempt.model_id,
        "task_ref": attempt.task_ref,
        "task_name": attempt.task_name,
        "stratum": stratum,
        "target": attempt.target,
        "iteration": attempt.iteration,
        "generation_status": attempt.generation_status,
        "evaluation_status": attempt.evaluation_status,
        "compiled": attempt.compiled,
        "correct": attempt.correct,
        "speedup": attempt.speedup,
        "best_speedup": state.best_speedup,
        "utility": _deployment_utility(state),
        "reference_speedup": reference_speedup,
        "reference_realization": (
            state.best_speedup / reference_speedup
            if state.best_speedup is not None and reference_speedup is not None
            else None
        ),
    }


def _aggregate_state(
    *,
    model_id: str,
    budget_name: str,
    budget_value: int,
    states: Mapping[tuple[str, str], CandidateState],
    task_refs: Sequence[str],
    targets: Sequence[str],
) -> dict[str, Any]:
    utilities = {
        target: {
            task_ref: _deployment_utility(states[(task_ref, target)])
            for task_ref in task_refs
        }
        for target in targets
    }
    coverages = {
        target: sum(states[(task_ref, target)].correct for task_ref in task_refs) / len(task_refs)
        for target in targets
    }
    target_geomeans = {
        target: _geometric_mean([utilities[target][task_ref] for task_ref in task_refs])
        for target in targets
    }
    best_fixed_value = max(target_geomeans.values())
    tied_fixed = [
        target
        for target in targets
        if math.isclose(target_geomeans[target], best_fixed_value, rel_tol=1e-12, abs_tol=1e-12)
    ]
    best_fixed_target = max(tied_fixed, key=lambda target: coverages[target])
    oracle_by_task = {
        task_ref: max(utilities[target][task_ref] for target in targets)
        for task_ref in task_refs
    }
    oracle_geomean = _geometric_mean(list(oracle_by_task.values()))
    oracle_coverage = sum(
        any(states[(task_ref, target)].correct for target in targets) for task_ref in task_refs
    ) / len(task_refs)
    return {
        "model_id": model_id,
        budget_name: budget_value,
        "target_geomean_utilities": target_geomeans,
        "target_correctness_coverages": coverages,
        "best_fixed_target": best_fixed_target,
        "tied_best_fixed_targets": tied_fixed,
        "best_fixed_geomean_utility": best_fixed_value,
        "best_fixed_correctness_coverage": coverages[best_fixed_target],
        "oracle_geomean_utility": oracle_geomean,
        "oracle_correctness_coverage": oracle_coverage,
        "oracle_gain_ratio": oracle_geomean / best_fixed_value,
        "oracle_gain_percent": (oracle_geomean / best_fixed_value - 1.0) * 100.0,
        "oracle_by_task": oracle_by_task,
        "utilities": utilities,
    }


def _iteration_metrics(
    *,
    models: Sequence[str],
    targets: Sequence[str],
    task_refs: Sequence[str],
    task_names: Mapping[str, str],
    trajectories: Mapping[tuple[str, str, str], tuple[AgentAttemptRecord, ...]],
    iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    winners: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    task_oracle_gains: list[dict[str, Any]] = []
    for model_id in models:
        for task_ref in task_refs:
            for iteration in range(1, iterations + 1):
                states = {
                    target: _candidate_state(
                        [
                            attempt
                            for attempt in trajectories.get((model_id, task_ref, target), ())
                            if attempt.iteration <= iteration
                        ]
                    )
                    for target in targets
                }
                values = {target: states[target].best_speedup for target in targets}
                winner_target, winner_speedup, tied = _winner(values, targets)
                winners.append(
                    {
                        "model_id": model_id,
                        "task_ref": task_ref,
                        "task_name": task_names[task_ref],
                        "iteration": iteration,
                        "winner_target": winner_target,
                        "winner_speedup": winner_speedup,
                        "tied_targets": tied,
                        "winner_status": (
                            "no_correct"
                            if winner_target is None
                            else "tie" if len(tied) > 1 else "unique"
                        ),
                    }
                )

        for iteration in range(1, iterations + 1):
            states = {
                (task_ref, target): _candidate_state(
                    [
                        attempt
                        for attempt in trajectories.get((model_id, task_ref, target), ())
                        if attempt.iteration <= iteration
                    ]
                )
                for task_ref in task_refs
                for target in targets
            }
            aggregate = _aggregate_state(
                model_id=model_id,
                budget_name="iteration",
                budget_value=iteration,
                states=states,
                task_refs=task_refs,
                targets=targets,
            )
            utilities = aggregate.pop("utilities")
            oracle_by_task = aggregate.pop("oracle_by_task")
            aggregates.append(aggregate)
            best_fixed_target = aggregate["best_fixed_target"]
            for task_ref in task_refs:
                fixed_utility = utilities[best_fixed_target][task_ref]
                oracle_utility = oracle_by_task[task_ref]
                task_oracle_gains.append(
                    {
                        "model_id": model_id,
                        "task_ref": task_ref,
                        "task_name": task_names[task_ref],
                        "iteration": iteration,
                        "best_fixed_target": best_fixed_target,
                        "fixed_utility": fixed_utility,
                        "oracle_utility": oracle_utility,
                        "oracle_gain_ratio": oracle_utility / fixed_utility,
                        "oracle_gain_percent": (oracle_utility / fixed_utility - 1.0) * 100.0,
                    }
                )
    return winners, aggregates, task_oracle_gains


def _token_metrics(
    *,
    models: Sequence[str],
    targets: Sequence[str],
    task_refs: Sequence[str],
    task_names: Mapping[str, str],
    task_strata: Mapping[str, str],
    token_trajectories: Mapping[tuple[str, str, str], TokenTrajectory],
    references: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_budgets: set[int] = set()
    model_budgets: dict[str, list[int]] = {}
    for model_id in models:
        boundaries = {
            boundary
            for (candidate_model, _, _), trajectory in token_trajectories.items()
            if candidate_model == model_id
            for boundary in trajectory.exact_boundaries
        }
        model_budgets[model_id] = [0, *sorted(boundaries)] if boundaries else []
        all_budgets.update(model_budgets[model_id])

    curve_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for model_id in models:
        for budget in model_budgets[model_id]:
            budget_states: dict[tuple[str, str], CandidateState] = {}
            budget_statuses: dict[tuple[str, str], str] = {}
            for task_ref in task_refs:
                for target in targets:
                    coordinate = model_id, task_ref, target
                    trajectory = token_trajectories[coordinate]
                    status, state = trajectory.at_budget(budget)
                    budget_states[(task_ref, target)] = state
                    budget_statuses[(task_ref, target)] = status
                    reference = references.get((task_ref, target))
                    reference_speedup = reference["speedup"] if reference else None
                    curve_rows.append(
                        {
                            "model_id": model_id,
                            "task_ref": task_ref,
                            "task_name": task_names[task_ref],
                            "stratum": task_strata[task_ref],
                            "target": target,
                            "token_budget": budget,
                            "budget_status": status,
                            "attempts_completed": state.attempts_completed,
                            "correct": state.correct,
                            "best_correct_runtime_ms": state.best_correct_runtime_ms,
                            "best_speedup": state.best_speedup,
                            "deployment_utility": _deployment_utility(state),
                            "reference_speedup": reference_speedup,
                            "reference_realization": (
                                state.best_speedup / reference_speedup
                                if state.best_speedup is not None
                                and reference_speedup is not None
                                else None
                            ),
                        }
                    )

                task_statuses = [budget_statuses[(task_ref, target)] for target in targets]
                if any(status != "exact" for status in task_statuses):
                    winner_rows.append(
                        {
                            "model_id": model_id,
                            "task_ref": task_ref,
                            "task_name": task_names[task_ref],
                            "stratum": task_strata[task_ref],
                            "token_budget": budget,
                            "winner_status": "censored",
                            "winner_target": None,
                            "winner_speedup": None,
                            "tied_targets": [],
                        }
                    )
                else:
                    values = {
                        target: budget_states[(task_ref, target)].best_speedup
                        for target in targets
                    }
                    winner_target, winner_speedup, tied = _winner(values, targets)
                    winner_rows.append(
                        {
                            "model_id": model_id,
                            "task_ref": task_ref,
                            "task_name": task_names[task_ref],
                            "stratum": task_strata[task_ref],
                            "token_budget": budget,
                            "winner_status": (
                                "no_correct"
                                if winner_target is None
                                else "tie" if len(tied) > 1 else "unique"
                            ),
                            "winner_target": winner_target,
                            "winner_speedup": winner_speedup,
                            "tied_targets": tied,
                        }
                    )

            if any(status != "exact" for status in budget_statuses.values()):
                aggregate = {
                    "model_id": model_id,
                    "token_budget": budget,
                    "budget_status": "censored",
                }
            else:
                aggregate = _aggregate_state(
                    model_id=model_id,
                    budget_name="token_budget",
                    budget_value=budget,
                    states=budget_states,
                    task_refs=task_refs,
                    targets=targets,
                )
                aggregate.pop("utilities")
                aggregate.pop("oracle_by_task")
                aggregate["budget_status"] = "exact"

            per_target_budget = budget // len(targets)
            split_states: dict[tuple[str, str], CandidateState] = {}
            split_censored = False
            for task_ref in task_refs:
                for target in targets:
                    status, state = token_trajectories[
                        (model_id, task_ref, target)
                    ].at_budget(per_target_budget)
                    split_states[(task_ref, target)] = state
                    split_censored = split_censored or status != "exact"
            aggregate["equal_split_per_target_budget"] = per_target_budget
            aggregate["equal_split_unspent_tokens"] = budget - per_target_budget * len(targets)
            aggregate["equal_split_budget_status"] = (
                "censored" if split_censored else "exact"
            )
            if split_censored:
                aggregate.update(
                    {
                        "equal_split_geomean_utility": None,
                        "equal_split_correctness_coverage": None,
                        "equal_split_gain_ratio": None,
                        "equal_split_gain_percent": None,
                    }
                )
            else:
                split_utility_by_task = {
                    task_ref: max(
                        _deployment_utility(split_states[(task_ref, target)])
                        for target in targets
                    )
                    for task_ref in task_refs
                }
                split_utility = _geometric_mean(list(split_utility_by_task.values()))
                split_coverage = sum(
                    any(split_states[(task_ref, target)].correct for target in targets)
                    for task_ref in task_refs
                ) / len(task_refs)
                fixed_utility = aggregate.get("best_fixed_geomean_utility")
                aggregate.update(
                    {
                        "equal_split_geomean_utility": split_utility,
                        "equal_split_correctness_coverage": split_coverage,
                        "equal_split_gain_ratio": (
                            split_utility / fixed_utility
                            if fixed_utility is not None
                            else None
                        ),
                        "equal_split_gain_percent": (
                            (split_utility / fixed_utility - 1.0) * 100.0
                            if fixed_utility is not None
                            else None
                        ),
                    }
                )
            aggregate_rows.append(aggregate)
    return sorted(all_budgets), curve_rows, winner_rows, aggregate_rows


def _csv_row(record_type: str, row: Mapping[str, Any]) -> dict[str, Any]:
    rendered = {"record_type": record_type, **row}
    for field in _JSON_CSV_FIELDS:
        if field in rendered:
            rendered[field] = json.dumps(rendered[field], separators=(",", ":"), sort_keys=True)
    return rendered


def _csv_rows(metrics: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    record_sets = (
        ("curve", "curve_rows"),
        ("winner", "winners"),
        ("aggregate", "aggregates"),
        ("task_oracle_gain", "task_oracle_gains"),
        ("trajectory_usage", "trajectory_usage_rows"),
        ("token_curve", "token_curve_rows"),
        ("token_winner", "token_winners"),
        ("token_aggregate", "token_aggregates"),
        ("target_health", "target_health_rows"),
        ("reference", "reference_rows"),
    )
    for record_type, key in record_sets:
        for row in metrics[key]:
            yield _csv_row(record_type, row)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in _csv_rows(metrics):
            writer.writerow(row)


def analyze_agent_run(
    run_directory: str | Path,
    *,
    reference_file: str | Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Analyze raw pilot attempts and persist JSON plus a flat mixed-record CSV."""

    run_path = Path(run_directory).expanduser().resolve()
    raw_path = run_path / "raw"
    run_payload = _read_json_mapping(raw_path / "run.json")
    if run_payload.get("schema_version") != "kernelbench-agent-run.v1":
        raise AgentAnalysisError("raw/run.json is not a kernelbench-agent-run.v1 artifact")
    try:
        study = KernelBenchAgentStudy.model_validate(run_payload["study"])
    except (KeyError, ValidationError) as error:
        raise AgentAnalysisError(f"raw/run.json has an invalid study: {error}") from None

    attempts = _load_attempts(raw_path / "attempts.jsonl")
    run_id = str(run_payload.get("run_id", run_path.name))
    iterations = int(run_payload.get("iterations", study.iterations))
    models = [model.id for model in study.models]
    targets = list(study.targets)
    task_refs = [task.ref for task in study.tasks]
    task_strata = {task.ref: task.stratum for task in study.tasks}
    task_names = {
        task.ref: next(
            (attempt.task_name for attempt in attempts if attempt.task_ref == task.ref),
            task.ref,
        )
        for task in study.tasks
    }

    attempts_by_coordinate: dict[tuple[str, str, str, int], AgentAttemptRecord] = {}
    trajectories_mutable: dict[tuple[str, str, str], list[AgentAttemptRecord]] = {
        (model_id, task_ref, target): []
        for model_id in models
        for task_ref in task_refs
        for target in targets
    }
    for attempt in attempts:
        coordinate = (*_trajectory_key(attempt), attempt.iteration)
        if coordinate in attempts_by_coordinate:
            raise AgentAnalysisError(f"duplicate attempt coordinate: {coordinate}")
        if _trajectory_key(attempt) not in trajectories_mutable:
            raise AgentAnalysisError(f"attempt coordinate is outside the study: {coordinate}")
        attempts_by_coordinate[coordinate] = attempt
        trajectories_mutable[_trajectory_key(attempt)].append(attempt)
    trajectories = {
        key: tuple(sorted(values, key=lambda attempt: attempt.iteration))
        for key, values in trajectories_mutable.items()
    }

    reference_rows, reference_sha256 = _load_references(
        reference_file,
        task_refs=task_refs,
        targets=targets,
    )
    references = {(row["task_ref"], row["target"]): row for row in reference_rows}

    resolved_usages = {
        coordinate: tuple(_resolve_attempt_usage(run_path, attempt) for attempt in trajectory)
        for coordinate, trajectory in trajectories.items()
    }
    token_trajectories = {
        coordinate: _build_token_trajectory(
            trajectory,
            resolved_usages[coordinate],
            configured_iterations=iterations,
        )
        for coordinate, trajectory in trajectories.items()
    }

    curve_rows: list[dict[str, Any]] = []
    for trajectory in trajectories.values():
        for index, attempt in enumerate(trajectory, start=1):
            curve_rows.append(
                _curve_row(
                    attempt,
                    _candidate_state(trajectory[:index]),
                    stratum=task_strata[attempt.task_ref],
                    reference=references.get((attempt.task_ref, attempt.target)),
                )
            )
    winners, aggregates, task_oracle_gains = _iteration_metrics(
        models=models,
        targets=targets,
        task_refs=task_refs,
        task_names=task_names,
        trajectories=trajectories,
        iterations=iterations,
    )

    trajectory_usage_rows = []
    for (model_id, task_ref, target), trajectory in token_trajectories.items():
        exact_prefix_attempts = len(trajectory.exact_boundaries)
        trajectory_usage_rows.append(
            {
                "model_id": model_id,
                "task_ref": task_ref,
                "task_name": task_names[task_ref],
                "stratum": task_strata[task_ref],
                "target": target,
                "attempt_count": len(trajectory.attempts),
                "exact_prefix_attempts": exact_prefix_attempts,
                "exact_prefix_tokens": trajectory.exact_prefix_tokens,
                "known_token_lower_bound": trajectory.known_token_lower_bound,
                "first_unknown_iteration": trajectory.first_unknown_iteration,
                "censored": bool(
                    trajectory.first_unknown_iteration is not None
                    or trajectory.incomplete_trajectory
                ),
                "usage_status": (
                    "exact"
                    if exact_prefix_attempts == len(trajectory.attempts)
                    else "unknown"
                ),
                "usage_source": "+".join(
                    sorted({usage.source for usage in trajectory.usages})
                ),
            }
        )

    token_budgets, token_curve_rows, token_winners, token_aggregates = _token_metrics(
        models=models,
        targets=targets,
        task_refs=task_refs,
        task_names=task_names,
        task_strata=task_strata,
        token_trajectories=token_trajectories,
        references=references,
    )

    target_health_rows = []
    for model_id in models:
        for target in targets:
            target_attempts = [
                attempt
                for attempt in attempts
                if attempt.model_id == model_id and attempt.target == target
            ]
            target_health_rows.append(
                {
                    "model_id": model_id,
                    "target": target,
                    "attempt_count": len(target_attempts),
                    "static_check_passes": sum(
                        attempt.evaluation_status == "evaluated"
                        for attempt in target_attempts
                    ),
                    "compiled_count": sum(attempt.compiled for attempt in target_attempts),
                    "correct_count": sum(attempt.correct for attempt in target_attempts),
                }
            )

    normalized_reference_rows = [
        {
            "task_ref": row["task_ref"],
            "target": row["target"],
            "reference_speedup": row["speedup"],
            "reference_label": row["label"],
            "reference_winner_status": row["winner_status"],
            "reference_winner_targets": row["winner_targets"],
        }
        for row in reference_rows
    ]

    metrics: dict[str, Any] = {
        "schema_version": "kernelbench-agent-metrics.v1",
        "run_id": run_id,
        "study_id": study.id,
        "models": models,
        "targets": targets,
        "tasks": [
            {
                "ref": task_ref,
                "name": task_names[task_ref],
                "stratum": task_strata[task_ref],
            }
            for task_ref in task_refs
        ],
        "iterations": iterations,
        "token_budgets": token_budgets,
        "reference_file_sha256": reference_sha256,
        "curve_rows": curve_rows,
        "winners": winners,
        "aggregates": aggregates,
        "task_oracle_gains": task_oracle_gains,
        "trajectory_usage_rows": trajectory_usage_rows,
        "token_curve_rows": token_curve_rows,
        "token_winners": token_winners,
        "token_aggregates": token_aggregates,
        "target_health_rows": target_health_rows,
        "reference_rows": normalized_reference_rows,
        "interpretation": (
            "Descriptive single-replicate pilot only. Generation speedup and correctness are "
            "reported separately from deployment fallback utility. Token results stop at the "
            "first request with unknown usage for each trajectory."
        ),
    }
    analysis_path = run_path / "analysis"
    analysis_path.mkdir(parents=True, exist_ok=True)
    metrics_json = analysis_path / "metrics.json"
    metrics_csv = analysis_path / "metrics.csv"
    _write_json(metrics_json, metrics)
    _write_csv(metrics_csv, metrics)
    return metrics, metrics_json, metrics_csv


def load_agent_metrics(run_directory: str | Path) -> dict[str, Any]:
    """Load the derived metrics used by the pure plotting stage."""

    run_path = Path(run_directory).expanduser().resolve()
    payload = _read_json_mapping(run_path / "analysis" / "metrics.json")
    if payload.get("schema_version") != "kernelbench-agent-metrics.v1":
        raise AgentAnalysisError("analysis/metrics.json is not kernelbench-agent-metrics.v1")
    return payload
