from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.image as mpimg
import pytest

from abstrak.evaluation.agent_analysis import AgentAnalysisError, analyze_agent_run
from abstrak.evaluation.agent_contracts import AgentAttemptRecord
from abstrak.evaluation.agent_figures import plot_agent_run

MODELS = ("deepseek-v4-flash", "gpt-5.6-luna")
TARGETS = ("triton", "tilelang", "cute")
TASKS = (
    ("level1-problem1", "Square_Matrix_Multiplication"),
    ("level1-problem24", "LogSoftmax"),
    ("level2-problem1", "Conv2D_ReLU_BiasAdd"),
    ("level2-problem76", "Gemm_Add_ReLU"),
)


def _study_payload() -> dict[str, Any]:
    return {
        "schema_version": "kernelbench-agent-study.v1",
        "id": "synthetic-agent-pilot",
        "source": {
            "repository": "https://example.test/KernelBench.git",
            "commit": "a" * 40,
            "require_clean_checkout": True,
        },
        "models": [
            {
                "id": MODELS[0],
                "protocol": "chat_completions",
                "litellm_provider": "deepseek",
                "api_model": "deepseek/deepseek-v4-flash",
                "api_key_env": "DEEPSEEK_API_KEY",
            },
            {
                "id": MODELS[1],
                "protocol": "responses",
                "litellm_provider": "openai",
                "api_model": "openai/gpt-5.6-luna",
                "api_key_env": "OPENAI_API_KEY",
            },
        ],
        "targets": list(TARGETS),
        "tasks": [
            {"level": 1, "problem_id": 1, "stratum": "dense"},
            {"level": 1, "problem_id": 24, "stratum": "reduction"},
            {"level": 2, "problem_id": 1, "stratum": "conv-fusion"},
            {"level": 2, "problem_id": 76, "stratum": "gemm-fusion"},
        ],
        "precision": "fp16",
        "iterations": 3,
    }


def _specialized_speedup(model_id: str, task_index: int, target: str) -> float | None:
    if model_id == MODELS[0]:
        winners = ("triton", "tilelang", "cute", "triton")
        values = (4.0, 4.0, 4.0, 2.0)
    else:
        winners = ("cute", "triton", "tilelang", "cute")
        values = (3.0, 3.0, 3.0, 3.0)
    return values[task_index] if target == winners[task_index] else None


def _attempt(
    *,
    model_id: str,
    task_ref: str,
    task_name: str,
    target: str,
    iteration: int,
    speedup: float | None,
) -> AgentAttemptRecord:
    correct = speedup is not None
    return AgentAttemptRecord(
        run_id="synthetic-run",
        trajectory_id=f"{model_id}-{task_ref}-{target}",
        model_id=model_id,
        task_ref=task_ref,
        task_name=task_name,
        target=target,
        iteration=iteration,
        generation_status="generated",
        evaluation_status="evaluated",
        compiled=correct,
        correct=correct,
        candidate_runtime_ms=10.0 / speedup if speedup is not None else None,
        reference_runtime_ms=10.0 if speedup is not None else None,
        speedup=speedup,
        best_speedup=speedup,
        error=None if correct else "synthetic incorrect candidate",
    )


def _write_synthetic_run(tmp_path: Path) -> Path:
    run_path = tmp_path / "synthetic-run"
    raw_path = run_path / "raw"
    raw_path.mkdir(parents=True)
    run_payload = {
        "schema_version": "kernelbench-agent-run.v1",
        "run_id": "synthetic-run",
        "study": _study_payload(),
        "created_at_utc": "2026-08-04T00:00:00Z",
        "iterations": 3,
        "trajectory_count": len(MODELS) * len(TARGETS) * len(TASKS),
    }
    (raw_path / "run.json").write_text(
        json.dumps(run_payload, indent=2) + "\n", encoding="utf-8"
    )

    attempts: list[AgentAttemptRecord] = []
    for model_id in MODELS:
        for task_index, (task_ref, task_name) in enumerate(TASKS):
            for target in TARGETS:
                final_speedup = _specialized_speedup(model_id, task_index, target)
                for iteration in range(1, 4):
                    speedup = None
                    if final_speedup is not None and iteration >= 2:
                        speedup = final_speedup * (0.75 if iteration == 2 else 1.0)
                    attempts.append(
                        _attempt(
                            model_id=model_id,
                            task_ref=task_ref,
                            task_name=task_name,
                            target=target,
                            iteration=iteration,
                            speedup=speedup,
                        )
                    )
    (raw_path / "attempts.jsonl").write_text(
        "".join(attempt.model_dump_json() + "\n" for attempt in attempts),
        encoding="utf-8",
    )
    return run_path


def test_analyze_agent_run_writes_curves_winners_and_oracle_gain(tmp_path: Path) -> None:
    run_path = _write_synthetic_run(tmp_path)

    metrics, json_path, csv_path = analyze_agent_run(run_path)

    assert json_path == run_path / "analysis" / "metrics.json"
    assert csv_path == run_path / "analysis" / "metrics.csv"
    assert metrics["schema_version"] == "kernelbench-agent-metrics.v1"
    assert len(metrics["curve_rows"]) == 72
    assert len(metrics["winners"]) == 24
    assert len(metrics["aggregates"]) == 6
    assert len(metrics["task_oracle_gains"]) == 24

    first_iteration_winner = next(
        row
        for row in metrics["winners"]
        if row["model_id"] == MODELS[0]
        and row["task_ref"] == TASKS[0][0]
        and row["iteration"] == 1
    )
    assert first_iteration_winner["winner_target"] is None
    assert first_iteration_winner["winner_speedup"] is None

    final_winner = next(
        row
        for row in metrics["winners"]
        if row["model_id"] == MODELS[0]
        and row["task_ref"] == TASKS[1][0]
        and row["iteration"] == 3
    )
    assert final_winner["winner_target"] == "tilelang"
    assert final_winner["winner_speedup"] == 4.0

    initial_aggregate = next(
        row
        for row in metrics["aggregates"]
        if row["model_id"] == MODELS[0] and row["iteration"] == 1
    )
    assert initial_aggregate["target_geomean_utilities"] == {
        "triton": 1.0,
        "tilelang": 1.0,
        "cute": 1.0,
    }
    assert initial_aggregate["oracle_gain_percent"] == 0.0

    final_aggregate = next(
        row
        for row in metrics["aggregates"]
        if row["model_id"] == MODELS[0] and row["iteration"] == 3
    )
    assert final_aggregate["best_fixed_target"] == "triton"
    assert final_aggregate["best_fixed_geomean_utility"] == pytest.approx(8.0**0.25)
    assert final_aggregate["oracle_geomean_utility"] == pytest.approx(128.0**0.25)
    assert final_aggregate["oracle_gain_ratio"] == pytest.approx(2.0)
    assert final_aggregate["oracle_gain_percent"] == pytest.approx(100.0)

    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted == metrics
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 126
    assert {row["record_type"] for row in csv_rows} == {
        "curve",
        "winner",
        "aggregate",
        "task_oracle_gain",
    }


def test_analyze_agent_run_rejects_duplicate_attempt_coordinate(tmp_path: Path) -> None:
    run_path = _write_synthetic_run(tmp_path)
    attempts_path = run_path / "raw" / "attempts.jsonl"
    first_line = attempts_path.read_text(encoding="utf-8").splitlines()[0]
    with attempts_path.open("a", encoding="utf-8") as handle:
        handle.write(first_line + "\n")

    with pytest.raises(AgentAnalysisError, match="duplicate attempt coordinate"):
        analyze_agent_run(run_path)


def test_plot_agent_run_writes_exact_two_png_pdf_bases_without_global_style_leak(
    tmp_path: Path,
) -> None:
    run_path = _write_synthetic_run(tmp_path)
    analyze_agent_run(run_path)
    original_font_size = matplotlib.rcParams["font.size"]

    paths = plot_agent_run(run_path)

    assert [path.name for path in paths] == [
        "01_anytime_performance_profiles.png",
        "01_anytime_performance_profiles.pdf",
        "02_winner_map_and_oracle_gain.png",
        "02_winner_map_and_oracle_gain.pdf",
    ]
    assert sorted(path.name for path in (run_path / "figures").iterdir()) == sorted(
        path.name for path in paths
    )
    assert all(path.stat().st_size > 1_000 for path in paths)
    assert all(path.read_bytes().startswith(b"%PDF") for path in (paths[1], paths[3]))
    assert matplotlib.rcParams["font.size"] == original_font_size

    for path in (paths[0], paths[2]):
        pixels = mpimg.imread(path)
        assert pixels.shape[0] > 500
        assert pixels.shape[1] > 1_000
        assert not math.isclose(float(pixels.std()), 0.0, abs_tol=1e-6)
