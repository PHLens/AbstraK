from __future__ import annotations

import pytest

from abstrak.canary.analysis import (
    AnalysisError,
    TrajectoryMeasurement,
    aggregate_cells,
    analyze_study,
    select_cell_oracles,
)

AGENTS = ("flash", "pro")
TASKS = ("norm-a", "norm-b", "gemm-a", "gemm-b")
TARGETS = ("target-a", "target-b", "target-c")


def _measurement(
    agent: str,
    task: str,
    target: str,
    replicate: int,
    *,
    efficiency: float | None,
    calls: int = 2,
    cvs: tuple[float, ...] = (0.01,),
    terminal_status: str = "observed",
) -> TrajectoryMeasurement:
    qualified = efficiency is not None and terminal_status == "observed"
    return TrajectoryMeasurement(
        agent_id=agent,
        task_id=task,
        target_id=target,
        replicate=replicate,
        terminal_status=terminal_status,
        qualified_at_first=qualified,
        qualified_at_final=qualified,
        candidate_latency_ms=1.0 / efficiency if qualified else None,
        baseline_latency_ms=1.0,
        expert_latency_ms=0.8,
        timing_cvs=cvs if qualified else (),
        calls=calls,
    )


def _matrix(efficiencies: dict[str, tuple[float, float, float, float]]):
    records = []
    for agent in AGENTS:
        for task_index, task in enumerate(TASKS):
            for target in TARGETS:
                for replicate in (1, 2):
                    records.append(
                        _measurement(
                            agent,
                            task,
                            target,
                            replicate,
                            efficiency=efficiencies[target][task_index],
                        )
                    )
    return tuple(records)


def _analyze(records, **overrides):
    arguments = {
        "agents": AGENTS,
        "tasks": TASKS,
        "targets": TARGETS,
        "expert_oracle_complete": True,
        "shakeout_passed": True,
    }
    arguments.update(overrides)
    return analyze_study(records, **arguments)


def test_replicates_aggregate_to_stable_unstable_and_failed() -> None:
    records = (
        _measurement("flash", "task", "stable", 1, efficiency=1.0),
        _measurement("flash", "task", "stable", 2, efficiency=1.0, cvs=(0.10, 0.04)),
        _measurement("flash", "task", "unstable", 1, efficiency=1.0),
        _measurement("flash", "task", "unstable", 2, efficiency=None),
        _measurement("flash", "task", "failed", 1, efficiency=None),
        _measurement("flash", "task", "failed", 2, efficiency=None),
    )

    cells = aggregate_cells(
        records,
        agents=("flash",),
        tasks=("task",),
        targets=("stable", "unstable", "failed"),
    )

    assert [cell.status for cell in cells] == [
        "stable_qualified",
        "unstable",
        "failed",
    ]
    assert cells[0].performance_stable is True
    assert cells[0].final_timing_cvs == (0.01, 0.04)
    assert cells[1].qualified_at_final_count == 1


def test_missing_and_infrastructure_trajectories_are_not_scientific_failures() -> None:
    records = (
        _measurement("flash", "task", "target", 1, efficiency=1.0),
        _measurement(
            "flash",
            "task",
            "target",
            2,
            efficiency=None,
            terminal_status="provider_error",
        ),
    )
    cell = aggregate_cells(
        records,
        agents=("flash",),
        tasks=("task",),
        targets=("target",),
    )[0]

    assert cell.status == "infrastructure_missing"
    assert cell.observed_replicates == 1
    assert cell.infrastructure_failures == 1


def test_five_percent_is_a_tie_and_more_than_five_percent_selects_winner() -> None:
    def oracle_for(second_efficiency: float):
        records = tuple(
            _measurement("flash", "task", target, replicate, efficiency=efficiency)
            for target, efficiency in (
                ("target-a", 1.0),
                ("target-b", second_efficiency),
                ("target-c", 0.5),
            )
            for replicate in (1, 2)
        )
        cells = aggregate_cells(
            records,
            agents=("flash",),
            tasks=("task",),
            targets=TARGETS,
        )
        return select_cell_oracles(
            cells,
            agents=("flash",),
            tasks=("task",),
            targets=TARGETS,
        )[0]

    tied = oracle_for(0.95)
    selected = oracle_for(0.949)

    assert tied.status == "performance_tie"
    assert tied.tied_target_ids == ("target-a", "target-b")
    assert tied.selected_target_id is None
    assert selected.status == "selected"
    assert selected.selected_target_id == "target-a"


def test_performance_tie_prefers_target_with_at_least_one_fewer_call() -> None:
    records = tuple(
        _measurement(
            "flash",
            "task",
            target,
            replicate,
            efficiency=efficiency,
            calls=calls,
        )
        for target, efficiency, calls in (
            ("target-a", 1.0, 3),
            ("target-b", 0.98, 2),
            ("target-c", 0.5, 1),
        )
        for replicate in (1, 2)
    )
    cells = aggregate_cells(
        records,
        agents=("flash",),
        tasks=("task",),
        targets=TARGETS,
    )
    oracle = select_cell_oracles(
        cells,
        agents=("flash",),
        tasks=("task",),
        targets=TARGETS,
    )[0]

    assert oracle.status == "lower_effort_preference"
    assert oracle.selected_target_id == "target-b"
    assert oracle.selected_calls == 2.0


def test_hindsight_and_distinct_task_support_produce_positive_signal() -> None:
    records = _matrix(
        {
            "target-a": (2.2, 1.0, 2.2, 1.0),
            "target-b": (1.0, 2.0, 1.0, 1.9),
            "target-c": (0.5, 0.5, 0.5, 0.5),
        }
    )

    result = _analyze(records)

    assert result.outcome == "positive_signal"
    assert result.expected_trajectories == result.received_trajectories == 48
    assert result.qualified_at_first == result.qualified_at_final == 48
    assert result.fixed_global_hindsight.selected_target_id == "target-a"
    assert {fixed.selected_target_id for fixed in result.fixed_agent_hindsight} == {"target-a"}
    assert result.non_tied_frontier_task_ids == TASKS
    assert result.non_tied_winner_target_ids == ("target-a", "target-b")
    assert result.actual_gain_task_ids == ("norm-b", "gemm-b")


def test_complete_stable_single_target_frontier_is_provisional_negative() -> None:
    records = _matrix(
        {
            "target-a": (2.0, 2.0, 2.0, 2.0),
            "target-b": (1.0, 1.0, 1.0, 1.0),
            "target-c": (0.5, 0.5, 0.5, 0.5),
        }
    )

    result = _analyze(records)

    assert result.outcome == "provisional_negative"
    assert result.fixed_global_hindsight.selected_target_id == "target-a"
    assert result.actual_gain_task_ids == ()


def test_floor_infrastructure_and_instability_have_distinct_terminal_states() -> None:
    records = _matrix(
        {
            "target-a": (2.0, 2.0, 2.0, 2.0),
            "target-b": (1.0, 1.0, 1.0, 1.0),
            "target-c": (0.5, 0.5, 0.5, 0.5),
        }
    )

    assert _analyze(records, expert_oracle_complete=False).outcome == "invalid_floor"
    assert _analyze(records, shakeout_passed=False).outcome == "invalid_floor"
    assert _analyze(records[:-1]).outcome == "inconclusive_infrastructure"

    unstable = list(records)
    index = next(
        index
        for index, record in enumerate(unstable)
        if record.agent_id == "flash"
        and record.task_id == "norm-a"
        and record.target_id == "target-b"
        and record.replicate == 2
    )
    unstable[index] = _measurement("flash", "norm-a", "target-b", 2, efficiency=None)
    assert _analyze(unstable).outcome == "inconclusive_instability"


def test_timing_instability_remains_separate_from_qualification_stability() -> None:
    records = list(
        _matrix(
            {
                "target-a": (2.0, 2.0, 2.0, 2.0),
                "target-b": (1.0, 1.0, 1.0, 1.0),
                "target-c": (0.5, 0.5, 0.5, 0.5),
            }
        )
    )
    index = next(
        index
        for index, record in enumerate(records)
        if record.agent_id == "flash"
        and record.task_id == "norm-a"
        and record.target_id == "target-a"
        and record.replicate == 1
    )
    records[index] = _measurement(
        "flash", "norm-a", "target-a", 1, efficiency=2.0, cvs=(0.10, 0.08)
    )

    result = _analyze(records)
    cell = next(
        cell
        for cell in result.cells
        if (cell.agent_id, cell.task_id, cell.target_id) == ("flash", "norm-a", "target-a")
    )

    assert cell.status == "stable_qualified"
    assert cell.performance_stable is False
    assert result.outcome == "inconclusive_instability"


def test_analysis_rejects_duplicate_unknown_and_runtime_drift() -> None:
    record = _measurement("flash", "task", "target", 1, efficiency=1.0)
    with pytest.raises(AnalysisError, match="duplicate trajectory"):
        aggregate_cells(
            (record, record),
            agents=("flash",),
            tasks=("task",),
            targets=("target",),
        )

    with pytest.raises(AnalysisError, match="outside the declared matrix"):
        aggregate_cells(
            (record,),
            agents=("pro",),
            tasks=("task",),
            targets=("target",),
        )

    drifted = record.model_copy(update={"replicate": 2, "baseline_latency_ms": 2.0})
    with pytest.raises(AnalysisError, match="baseline.*drifted"):
        aggregate_cells(
            (record, drifted),
            agents=("flash",),
            tasks=("task",),
            targets=("target",),
        )

    expert_drifted = record.model_copy(update={"replicate": 2, "expert_latency_ms": 0.9})
    with pytest.raises(AnalysisError, match="expert oracle runtime drifted"):
        aggregate_cells(
            (record, expert_drifted),
            agents=("flash",),
            tasks=("task",),
            targets=("target",),
        )
