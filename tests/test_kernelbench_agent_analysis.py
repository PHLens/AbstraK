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
    input_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    output_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    generation_status: str = "generated",
    evaluation_status: str | None = None,
    response_path: str | None = None,
) -> AgentAttemptRecord:
    correct = speedup is not None
    resolved_evaluation_status = evaluation_status or (
        "evaluated" if generation_status == "generated" else "not_run"
    )
    return AgentAttemptRecord(
        run_id="synthetic-run",
        trajectory_id=f"{model_id}-{task_ref}-{target}",
        model_id=model_id,
        task_ref=task_ref,
        task_name=task_name,
        target=target,
        iteration=iteration,
        generation_status=generation_status,
        evaluation_status=resolved_evaluation_status,
        compiled=correct,
        correct=correct,
        candidate_runtime_ms=10.0 / speedup if speedup is not None else None,
        reference_runtime_ms=10.0 if speedup is not None else None,
        speedup=speedup,
        best_speedup=speedup,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        response_path=response_path,
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
    assert metrics["token_budgets"] == []
    assert len(metrics["trajectory_usage_rows"]) == 24
    assert len(metrics["target_health_rows"]) == 6
    assert all(row["static_check_passes"] == 12 for row in metrics["target_health_rows"])
    assert len(csv_rows) == 156
    assert {row["record_type"] for row in csv_rows} == {
        "curve",
        "winner",
        "aggregate",
        "task_oracle_gain",
        "trajectory_usage",
        "target_health",
    }


def _write_token_run(
    tmp_path: Path,
    *,
    unknown_target: str | None = None,
    raw_usage_target: str | None = None,
) -> Path:
    run_path = tmp_path / f"token-run-{unknown_target or 'exact'}-{raw_usage_target or 'direct'}"
    raw_path = run_path / "raw"
    raw_path.mkdir(parents=True)
    study = {
        "schema_version": "kernelbench-agent-study.v1",
        "id": "token-agent-pilot",
        "source": {
            "repository": "https://example.test/KernelBench.git",
            "commit": "b" * 40,
            "require_clean_checkout": True,
        },
        "models": [
            {
                "id": "deepseek-v4-flash",
                "protocol": "chat_completions",
                "litellm_provider": "deepseek",
                "api_model": "deepseek/deepseek-v4-flash",
                "api_key_env": "DEEPSEEK_API_KEY",
            }
        ],
        "targets": list(TARGETS),
        "tasks": [{"level": 1, "problem_id": 1, "stratum": "control"}],
        "precision": "fp16",
        "iterations": 2,
    }
    (raw_path / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "kernelbench-agent-run.v1",
                "run_id": run_path.name,
                "study": study,
                "iterations": 2,
                "trajectory_count": 3,
                "completed_at_utc": "2026-08-11T00:00:00Z",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    first_speedups = {"triton": 0.8, "tilelang": 2.0, "cute": None}
    final_speedups = {"triton": 3.0, "tilelang": 2.5, "cute": 4.0}
    attempts: list[AgentAttemptRecord] = []
    for target in TARGETS:
        first_input = None if target in {unknown_target, raw_usage_target} else 1
        first_output = None if target in {unknown_target, raw_usage_target} else 2
        response_path = None
        generation_status = "generated"
        evaluation_status = "evaluated"
        if target == raw_usage_target:
            generation_status = "output_truncated"
            evaluation_status = "not_run"
            response_path = f"raw/responses/{target}/iteration-001.json"
            artifact = run_path / response_path
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                json.dumps(
                    {
                        "schema_version": "kernelbench-agent-provider-error.v1",
                        "raw_response": {
                            "usage": {"prompt_tokens": 1, "completion_tokens": 2}
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        attempts.append(
            _attempt(
                model_id="deepseek-v4-flash",
                task_ref="level1-problem1",
                task_name="Square_Matrix_Multiplication",
                target=target,
                iteration=1,
                speedup=(None if target == raw_usage_target else first_speedups[target]),
                input_tokens=first_input,
                cached_input_tokens=(1 if first_input is not None else None),
                output_tokens=first_output,
                reasoning_tokens=(2 if first_output is not None else None),
                generation_status=generation_status,
                evaluation_status=evaluation_status,
                response_path=response_path,
            )
        )
        attempts.append(
            _attempt(
                model_id="deepseek-v4-flash",
                task_ref="level1-problem1",
                task_name="Square_Matrix_Multiplication",
                target=target,
                iteration=2,
                speedup=final_speedups[target],
                input_tokens=3,
                output_tokens=4,
            )
        )
    (raw_path / "attempts.jsonl").write_text(
        "".join(attempt.model_dump_json() + "\n" for attempt in attempts),
        encoding="utf-8",
    )
    return run_path


def test_token_analysis_uses_exact_cost_and_equal_split_step_function(tmp_path: Path) -> None:
    run_path = _write_token_run(tmp_path)

    metrics, _, _ = analyze_agent_run(run_path)

    assert metrics["token_budgets"] == [0, 3, 10]
    triton_usage = next(
        row
        for row in metrics["trajectory_usage_rows"]
        if row["target"] == "triton"
    )
    assert triton_usage["exact_prefix_tokens"] == 10
    assert triton_usage["known_token_lower_bound"] == 10
    assert triton_usage["usage_status"] == "exact"

    triton_at_three = next(
        row
        for row in metrics["token_curve_rows"]
        if row["target"] == "triton" and row["token_budget"] == 3
    )
    assert triton_at_three["best_speedup"] == pytest.approx(0.8)
    assert triton_at_three["deployment_utility"] == 1.0
    aggregate = next(
        row for row in metrics["token_aggregates"] if row["token_budget"] == 10
    )
    assert aggregate["best_fixed_target"] == "cute"
    assert aggregate["best_fixed_geomean_utility"] == pytest.approx(4.0)
    assert aggregate["equal_split_per_target_budget"] == 3
    assert aggregate["equal_split_unspent_tokens"] == 1
    assert aggregate["equal_split_budget_status"] == "exact"
    assert aggregate["equal_split_geomean_utility"] == pytest.approx(2.0)
    assert aggregate["equal_split_gain_ratio"] == pytest.approx(0.5)


def test_token_analysis_recovers_truncation_usage_from_response_artifact(tmp_path: Path) -> None:
    run_path = _write_token_run(tmp_path, raw_usage_target="cute")

    metrics, _, _ = analyze_agent_run(run_path)

    cute_usage = next(
        row for row in metrics["trajectory_usage_rows"] if row["target"] == "cute"
    )
    assert cute_usage["exact_prefix_tokens"] == 10
    assert cute_usage["usage_status"] == "exact"
    assert "raw_response" in cute_usage["usage_source"]


def test_unknown_usage_censors_later_token_budget_without_hiding_iterations(
    tmp_path: Path,
) -> None:
    run_path = _write_token_run(tmp_path, unknown_target="tilelang")

    metrics, _, _ = analyze_agent_run(run_path)

    assert any(
        row["target"] == "tilelang" and row["iteration"] == 2
        for row in metrics["curve_rows"]
    )
    censored = next(
        row
        for row in metrics["token_curve_rows"]
        if row["target"] == "tilelang" and row["token_budget"] == 10
    )
    assert censored["budget_status"] == "censored"
    aggregate = next(
        row for row in metrics["token_aggregates"] if row["token_budget"] == 10
    )
    assert aggregate["model_id"] == "deepseek-v4-flash"
    assert aggregate["token_budget"] == 10
    assert aggregate["budget_status"] == "censored"
    assert aggregate["equal_split_budget_status"] == "censored"
    assert aggregate["equal_split_geomean_utility"] is None


def test_early_terminated_trajectory_does_not_carry_into_later_budget(
    tmp_path: Path,
) -> None:
    run_path = _write_token_run(tmp_path)
    attempts_path = run_path / "raw" / "attempts.jsonl"
    attempts = [
        AgentAttemptRecord.model_validate_json(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
    ]
    attempts = [
        attempt
        for attempt in attempts
        if not (attempt.target == "tilelang" and attempt.iteration == 2)
    ]
    attempts_path.write_text(
        "".join(attempt.model_dump_json() + "\n" for attempt in attempts),
        encoding="utf-8",
    )

    metrics, _, _ = analyze_agent_run(run_path)

    tilelang_at_ten = next(
        row
        for row in metrics["token_curve_rows"]
        if row["target"] == "tilelang" and row["token_budget"] == 10
    )
    assert tilelang_at_ten["budget_status"] == "censored"
    assert tilelang_at_ten["best_speedup"] == pytest.approx(2.0)
    aggregate = next(
        row for row in metrics["token_aggregates"] if row["token_budget"] == 10
    )
    assert aggregate["budget_status"] == "censored"
    assert aggregate["equal_split_budget_status"] == "exact"
    assert aggregate["equal_split_geomean_utility"] == pytest.approx(2.0)
    assert aggregate["equal_split_gain_ratio"] is None


def test_best_fixed_tie_uses_correctness_coverage_as_scalar_tiebreak(
    tmp_path: Path,
) -> None:
    run_path = _write_token_run(tmp_path)
    attempts_path = run_path / "raw" / "attempts.jsonl"
    attempts = [
        AgentAttemptRecord.model_validate_json(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
    ]
    attempts = [
        (
            attempt.model_copy(
                update={
                    "compiled": False,
                    "correct": False,
                    "candidate_runtime_ms": None,
                    "reference_runtime_ms": None,
                    "speedup": None,
                    "best_speedup": None,
                    "error": "synthetic incorrect candidate",
                }
            )
            if attempt.target == "tilelang" and attempt.iteration == 1
            else attempt
        )
        for attempt in attempts
    ]
    attempts_path.write_text(
        "".join(attempt.model_dump_json() + "\n" for attempt in attempts),
        encoding="utf-8",
    )

    metrics, _, _ = analyze_agent_run(run_path)

    aggregate = next(
        row for row in metrics["token_aggregates"] if row["token_budget"] == 3
    )
    assert aggregate["tied_best_fixed_targets"] == list(TARGETS)
    assert aggregate["best_fixed_target"] == "triton"
    assert aggregate["best_fixed_correctness_coverage"] == pytest.approx(1.0)


def test_optional_reference_overlay_is_partial_and_non_blocking(tmp_path: Path) -> None:
    run_path = _write_token_run(tmp_path)
    reference_file = tmp_path / "references.csv"
    reference_file.write_text(
        "task_ref,target,speedup,label\n"
        "level1-problem1,triton,3.0,expert-v1\n"
        "level1-problem1,tilelang,5.0,expert-v1\n"
        "level1-problem1,cute,4.0,expert-v1\n",
        encoding="utf-8",
    )

    metrics, _, _ = analyze_agent_run(run_path, reference_file=reference_file)

    assert metrics["reference_file_sha256"] is not None
    assert len(metrics["reference_rows"]) == 3
    assert all(
        row["reference_winner_targets"] == ["tilelang"]
        for row in metrics["reference_rows"]
    )
    cute_at_ten = next(
        row
        for row in metrics["token_curve_rows"]
        if row["target"] == "cute" and row["token_budget"] == 10
    )
    assert cute_at_ten["reference_realization"] == pytest.approx(1.0)

    partial_file = tmp_path / "partial.csv"
    partial_file.write_text(
        "task_ref,target,speedup,label\nlevel1-problem1,triton,3.0,expert-v1\n",
        encoding="utf-8",
    )
    partial_metrics, _, _ = analyze_agent_run(run_path, reference_file=partial_file)
    assert partial_metrics["reference_rows"][0]["reference_winner_status"] == "incomplete"


def test_malformed_reference_row_has_analysis_error(tmp_path: Path) -> None:
    run_path = _write_token_run(tmp_path)
    reference_file = tmp_path / "malformed.csv"
    reference_file.write_text(
        "task_ref,target,speedup,label\nlevel1-problem1,triton\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentAnalysisError, match="invalid reference speedup"):
        analyze_agent_run(run_path, reference_file=reference_file)


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
