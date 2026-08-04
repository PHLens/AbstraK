"""Pure descriptive analysis for the exploratory KernelBench agent pilot."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from abstrak.evaluation.agent_contracts import AgentAttemptRecord, KernelBenchAgentStudy


class AgentAnalysisError(ValueError):
    """Raised when a pilot run cannot be analyzed."""


_CSV_FIELDS = (
    "record_type",
    "model_id",
    "task_ref",
    "task_name",
    "target",
    "iteration",
    "generation_status",
    "evaluation_status",
    "compiled",
    "correct",
    "speedup",
    "best_speedup",
    "utility",
    "winner_target",
    "winner_speedup",
    "tied_targets",
    "best_fixed_target",
    "tied_best_fixed_targets",
    "target_geomean_utilities",
    "best_fixed_geomean_utility",
    "oracle_geomean_utility",
    "oracle_gain_ratio",
    "oracle_gain_percent",
    "fixed_utility",
    "oracle_utility",
)


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


def _curve_row(attempt: AgentAttemptRecord) -> dict[str, Any]:
    return {
        "model_id": attempt.model_id,
        "task_ref": attempt.task_ref,
        "task_name": attempt.task_name,
        "target": attempt.target,
        "iteration": attempt.iteration,
        "generation_status": attempt.generation_status,
        "evaluation_status": attempt.evaluation_status,
        "compiled": attempt.compiled,
        "correct": attempt.correct,
        "speedup": attempt.speedup,
        "best_speedup": attempt.best_speedup,
        "utility": attempt.best_speedup if attempt.best_speedup is not None else 1.0,
    }


def _task_names(
    study: KernelBenchAgentStudy, attempts: Sequence[AgentAttemptRecord]
) -> dict[str, str]:
    names = {attempt.task_ref: attempt.task_name for attempt in attempts}
    return {task.ref: names.get(task.ref, task.ref) for task in study.tasks}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv_rows(metrics: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for row in metrics["curve_rows"]:
        yield {"record_type": "curve", **row}
    for row in metrics["winners"]:
        yield {
            "record_type": "winner",
            **row,
            "tied_targets": json.dumps(row["tied_targets"], separators=(",", ":")),
        }
    for row in metrics["aggregates"]:
        yield {
            "record_type": "aggregate",
            **row,
            "tied_best_fixed_targets": json.dumps(
                row["tied_best_fixed_targets"], separators=(",", ":")
            ),
            "target_geomean_utilities": json.dumps(
                row["target_geomean_utilities"], separators=(",", ":")
            ),
        }
    for row in metrics["task_oracle_gains"]:
        yield {"record_type": "task_oracle_gain", **row}


def _write_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in _csv_rows(metrics):
            writer.writerow(row)


def analyze_agent_run(
    run_directory: str | Path,
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
    task_names = _task_names(study, attempts)

    attempts_by_coordinate: dict[tuple[str, str, str, int], AgentAttemptRecord] = {}
    for attempt in attempts:
        coordinate = (
            attempt.model_id,
            attempt.task_ref,
            attempt.target,
            attempt.iteration,
        )
        if coordinate in attempts_by_coordinate:
            raise AgentAnalysisError(f"duplicate attempt coordinate: {coordinate}")
        attempts_by_coordinate[coordinate] = attempt

    curve_rows = [_curve_row(attempt) for attempt in attempts]
    winners: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    task_oracle_gains: list[dict[str, Any]] = []

    for model_id in models:
        for task_ref in task_refs:
            for iteration in range(1, iterations + 1):
                values = {
                    target: (
                        attempts_by_coordinate[(model_id, task_ref, target, iteration)].best_speedup
                        if (model_id, task_ref, target, iteration) in attempts_by_coordinate
                        else None
                    )
                    for target in targets
                }
                winner_target, winner_speedup, tied_targets = _winner(values, targets)
                winners.append(
                    {
                        "model_id": model_id,
                        "task_ref": task_ref,
                        "task_name": task_names[task_ref],
                        "iteration": iteration,
                        "winner_target": winner_target,
                        "winner_speedup": winner_speedup,
                        "tied_targets": tied_targets,
                    }
                )

        for iteration in range(1, iterations + 1):
            utilities: dict[str, dict[str, float]] = {}
            for target in targets:
                utilities[target] = {}
                for task_ref in task_refs:
                    attempt = attempts_by_coordinate.get((model_id, task_ref, target, iteration))
                    utilities[target][task_ref] = (
                        attempt.best_speedup
                        if attempt is not None and attempt.best_speedup is not None
                        else 1.0
                    )

            target_geomeans = {
                target: _geometric_mean([utilities[target][task_ref] for task_ref in task_refs])
                for target in targets
            }
            best_fixed_value = max(target_geomeans.values())
            tied_fixed = [
                target
                for target in targets
                if math.isclose(
                    target_geomeans[target], best_fixed_value, rel_tol=1e-12, abs_tol=1e-12
                )
            ]
            best_fixed_target = tied_fixed[0]
            oracle_by_task = {
                task_ref: max(utilities[target][task_ref] for target in targets)
                for task_ref in task_refs
            }
            oracle_geomean = _geometric_mean(list(oracle_by_task.values()))
            gain_ratio = oracle_geomean / best_fixed_value
            aggregates.append(
                {
                    "model_id": model_id,
                    "iteration": iteration,
                    "target_geomean_utilities": target_geomeans,
                    "best_fixed_target": best_fixed_target,
                    "tied_best_fixed_targets": tied_fixed,
                    "best_fixed_geomean_utility": best_fixed_value,
                    "oracle_geomean_utility": oracle_geomean,
                    "oracle_gain_ratio": gain_ratio,
                    "oracle_gain_percent": (gain_ratio - 1.0) * 100.0,
                }
            )
            for task_ref in task_refs:
                fixed_utility = utilities[best_fixed_target][task_ref]
                oracle_utility = oracle_by_task[task_ref]
                task_gain_ratio = oracle_utility / fixed_utility
                task_oracle_gains.append(
                    {
                        "model_id": model_id,
                        "task_ref": task_ref,
                        "task_name": task_names[task_ref],
                        "iteration": iteration,
                        "best_fixed_target": best_fixed_target,
                        "fixed_utility": fixed_utility,
                        "oracle_utility": oracle_utility,
                        "oracle_gain_ratio": task_gain_ratio,
                        "oracle_gain_percent": (task_gain_ratio - 1.0) * 100.0,
                    }
                )

    metrics: dict[str, Any] = {
        "schema_version": "kernelbench-agent-metrics.v1",
        "run_id": run_id,
        "study_id": study.id,
        "models": models,
        "targets": targets,
        "tasks": [
            {"ref": task_ref, "name": task_names[task_ref]} for task_ref in task_refs
        ],
        "iterations": iterations,
        "curve_rows": curve_rows,
        "winners": winners,
        "aggregates": aggregates,
        "task_oracle_gains": task_oracle_gains,
        "interpretation": (
            "Descriptive single-replicate pilot only. Missing or not-yet-correct targets use "
            "1.0x utility for fixed-target and oracle aggregates; winner maps only use correct "
            "candidate speedups."
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
