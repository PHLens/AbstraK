"""Descriptive correctness and performance summaries for the naive screen."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from abstrak.evaluation.artifacts import verify_directory_checksums, write_derived_json
from abstrak.evaluation.contracts import CellSpec, EvaluationResult


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_run(run_directory: str | Path) -> tuple[dict[str, Any], Path]:
    run_path = Path(run_directory).expanduser().resolve()
    groups: dict[tuple[str, str], list[EvaluationResult]] = defaultdict(list)
    missing: list[str] = []
    evaluations_path = run_path / "evaluations"
    for cell_directory in sorted((run_path / "cells").iterdir()):
        if not cell_directory.is_dir():
            continue
        verify_directory_checksums(cell_directory, "generation.sha256sums")
        spec = CellSpec.model_validate(
            json.loads((cell_directory / "cell.json").read_text(encoding="utf-8"))
        )
        evaluation_path = evaluations_path / spec.cell_id / "evaluation.json"
        if not evaluation_path.is_file():
            missing.append(spec.cell_id)
            continue
        verify_directory_checksums(evaluation_path.parent, "evaluation.sha256sums")
        result = EvaluationResult.model_validate(
            json.loads(evaluation_path.read_text(encoding="utf-8"))
        )
        groups[(spec.profile, spec.target)].append(result)

    summaries: list[dict[str, Any]] = []
    for (profile, target), results in sorted(groups.items()):
        attempted = len(results)
        ratios = [
            result.performance_ratio for result in results if result.performance_ratio is not None
        ]
        summaries.append(
            {
                "profile": profile,
                "target": target,
                "attempted": attempted,
                "compiled": sum(result.compiled for result in results),
                "correct": sum(result.correctness for result in results),
                "compile_rate": _rate(sum(result.compiled for result in results), attempted),
                "correctness_rate": _rate(sum(result.correctness for result in results), attempted),
                "fast_1_rate": _rate(sum(result.fast_1 for result in results), attempted),
                "fast_2_rate": _rate(sum(result.fast_2 for result in results), attempted),
                "performance_coverage": _rate(len(ratios), attempted),
                "geomean_performance_ratio_correct": (
                    math.exp(sum(math.log(value) for value in ratios) / len(ratios))
                    if ratios
                    else None
                ),
                "median_performance_ratio_correct": (statistics.median(ratios) if ratios else None),
                "status_counts": {
                    status: sum(result.status == status for result in results)
                    for status in sorted({result.status for result in results})
                },
            }
        )
    payload = {
        "schema_version": "kernelbench-naive-metrics.v1",
        "run_id": run_path.name,
        "groups": summaries,
        "missing_evaluations": missing,
        "interpretation": (
            "Descriptive single-replicate screen only; absence of a difference is not an "
            "equivalence result."
        ),
    }
    path = write_derived_json(run_path / "metrics.json", payload)
    return payload, path
