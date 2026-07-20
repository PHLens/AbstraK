from __future__ import annotations

import pytest

from abstrak.canary.schedule import (
    R1_REPLICATES,
    SCHEDULE_SEED,
    ScheduleError,
    build_r1_schedule,
)


def test_r1_schedule_has_exact_shakeout_and_formal_matrices() -> None:
    schedule = build_r1_schedule()

    assert schedule.seed == SCHEDULE_SEED
    assert len(schedule.shakeout) == 12
    assert len(schedule.formal) == 48
    assert len({cell.key for cell in schedule.shakeout}) == 12
    assert len({cell.key for cell in schedule.formal}) == 48
    assert len({cell.trajectory_id for cell in (*schedule.shakeout, *schedule.formal)}) == 60


def test_formal_schedule_randomizes_targets_within_every_fixed_block() -> None:
    schedule = build_r1_schedule()
    blocks: dict[tuple[str, str, int], list[str]] = {}
    for cell in schedule.formal:
        blocks.setdefault(cell.block_key, []).append(cell.target_id)

    assert len(blocks) == 4 * 2 * len(R1_REPLICATES)
    assert all(set(order) == set(schedule.targets) for order in blocks.values())
    assert all(len(order) == len(schedule.targets) for order in blocks.values())
    assert len({tuple(order) for order in blocks.values()}) > 1


def test_schedule_is_deterministic_and_seed_changes_only_formal_order() -> None:
    first = build_r1_schedule()
    second = build_r1_schedule()
    alternate = build_r1_schedule(seed=SCHEDULE_SEED + 1)

    assert first == second
    assert first.sha256 == second.sha256
    assert first.shakeout == alternate.shakeout
    assert tuple(cell.key for cell in first.formal) != tuple(cell.key for cell in alternate.formal)
    assert {cell.key for cell in first.formal} == {cell.key for cell in alternate.formal}


def test_schedule_rejects_axis_drift_and_duplicate_identifiers() -> None:
    with pytest.raises(ScheduleError, match="exactly 2 agents"):
        build_r1_schedule(agents=("flash",))
    with pytest.raises(ScheduleError, match="targets must be unique"):
        build_r1_schedule(targets=("triton", "triton", "cute"))
    with pytest.raises(ScheduleError, match="invalid identifiers"):
        build_r1_schedule(canaries=("valid", "not valid"))
    with pytest.raises(ScheduleError, match="seed must be an integer"):
        build_r1_schedule(seed=True)
