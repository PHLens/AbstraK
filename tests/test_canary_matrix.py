from __future__ import annotations

from collections import Counter

import pytest
from pydantic import ValidationError

from abstrak.canary.matrix import (
    CoreGateThresholds,
    FullGateThresholds,
    MatrixSpecError,
    MatrixStudySpec,
    PhaseSpec,
    PortfolioGateSpec,
    TaskGroupSpec,
    build_matrix_schedule,
)


def _phase(phase_id: str, tasks: tuple[str, ...], *, policy: str) -> PhaseSpec:
    return PhaseSpec(
        id=phase_id,
        task_ids=tasks,
        replicates=(1, 2, 3),
        order_policy=policy,
        max_calls_per_trajectory=3,
    )


def _spec(*, seed: int = 20260723, policy: str = "balanced_rotation") -> MatrixStudySpec:
    core = ("core-a", "core-b", "core-c", "core-d")
    reserve = ("reserve-a", "reserve-b", "reserve-c", "reserve-d")
    return MatrixStudySpec(
        study_id="matrix-test",
        seed=seed,
        agents=("agent",),
        targets=("pack-b", "pack-bs", "pack-bm", "pack-bsm"),
        task_groups=tuple(
            TaskGroupSpec(id=f"group-{index}", task_ids=(core[index], reserve[index]))
            for index in range(4)
        ),
        phases=(
            _phase("core", core, policy=policy),
            _phase("reserve", reserve, policy=policy),
        ),
        gate=PortfolioGateSpec(
            core_phase_id="core",
            reserve_phase_id="reserve",
            core=CoreGateThresholds(),
            full=FullGateThresholds(),
        ),
    )


def test_capability_sized_matrix_has_dynamic_counts_hashes_and_unique_ids() -> None:
    first = build_matrix_schedule(_spec())
    second = build_matrix_schedule(_spec())

    assert first == second
    assert first.sha256 == second.sha256
    assert first.spec.sha256 == first.spec_sha256
    assert first.expected_trajectories == first.spec.expected_trajectories == 96
    assert first.request_ceiling == first.spec.request_ceiling == 288
    assert len(first.cells_for_phase("core")) == 48
    assert len(first.cells_for_phase("reserve")) == 48
    assert first.phase_request_ceiling("core") == 144
    assert len({cell.key for cell in first.cells}) == 96
    assert len({cell.trajectory_id for cell in first.cells}) == 96
    assert tuple(cell.ordinal for cell in first.cells) == tuple(range(96))


def test_infrastructure_retry_does_not_inflate_scientific_request_ceiling() -> None:
    payload = _spec().model_dump()
    for phase in payload["phases"]:
        phase["infrastructure_retries"] = 1
    spec = MatrixStudySpec.model_validate(payload)

    assert all(phase.infrastructure_retries == 1 for phase in spec.phases)
    assert spec.request_ceiling == 288
    assert spec.phase_operational_request_ceiling("core") == 288
    assert spec.operational_request_ceiling == 576


@pytest.mark.parametrize(
    ("field", "value"),
    (("max_calls_per_trajectory", 5), ("infrastructure_retries", 2)),
)
def test_phase_policy_stays_within_v1_wire_limits(field: str, value: int) -> None:
    payload = _phase("core", ("task",), policy="fixed").model_dump()
    payload[field] = value

    with pytest.raises(ValidationError):
        PhaseSpec.model_validate(payload)


def test_balanced_rotation_balances_every_target_position() -> None:
    schedule = build_matrix_schedule(_spec())

    for phase_id in ("core", "reserve"):
        cells = schedule.cells_for_phase(phase_id)
        positions = Counter((cell.target_id, cell.target_order_index) for cell in cells)
        assert set(positions.values()) == {3}
        blocks: dict[tuple[str, str, str, int], list[str]] = {}
        for cell in cells:
            blocks.setdefault(cell.block_key, []).append(cell.target_id)
        assert len(blocks) == 12
        assert all(set(order) == set(schedule.spec.targets) for order in blocks.values())


def test_order_policies_are_seeded_without_changing_cell_identity() -> None:
    shuffled = build_matrix_schedule(_spec(policy="seeded_shuffle"))
    repeated = build_matrix_schedule(_spec(policy="seeded_shuffle"))
    alternate = build_matrix_schedule(_spec(seed=20260724, policy="seeded_shuffle"))
    fixed = build_matrix_schedule(_spec(policy="fixed"))
    fixed_alternate = build_matrix_schedule(_spec(seed=20260724, policy="fixed"))

    assert shuffled == repeated
    assert tuple(cell.key for cell in shuffled.cells) != tuple(cell.key for cell in alternate.cells)
    assert {cell.key for cell in shuffled.cells} == {cell.key for cell in alternate.cells}
    assert tuple(cell.key for cell in fixed.cells) == tuple(
        cell.key for cell in fixed_alternate.cells
    )
    for cell in fixed.cells:
        assert cell.target_id == fixed.spec.targets[cell.target_order_index]


@pytest.mark.parametrize(
    ("update", "message"),
    (
        ({"agents": ("agent", "agent")}, "agents must be unique"),
        ({"targets": ("pack", "pack")}, "targets must be unique"),
        (
            {
                "task_groups": (
                    TaskGroupSpec(id="same", task_ids=("core-a",)),
                    TaskGroupSpec(id="same", task_ids=("core-b",)),
                )
            },
            "task group IDs must be unique",
        ),
        (
            {
                "phases": (
                    _phase("same", ("core-a",), policy="fixed"),
                    _phase("same", ("core-b",), policy="fixed"),
                )
            },
            "phase IDs must be unique",
        ),
    ),
)
def test_spec_rejects_duplicate_axes_groups_and_phases(
    update: dict[str, object], message: str
) -> None:
    payload = _spec().model_dump()
    payload.update(update)
    with pytest.raises(ValidationError, match=message):
        MatrixStudySpec.model_validate(payload)


def test_spec_rejects_duplicate_group_assignment_and_inexact_group_coverage() -> None:
    payload = _spec().model_dump()
    groups = list(payload["task_groups"])
    groups[1]["task_ids"] = (*groups[1]["task_ids"], "core-a")
    payload["task_groups"] = tuple(groups)
    with pytest.raises(ValidationError, match="exactly one group"):
        MatrixStudySpec.model_validate(payload)

    payload = _spec().model_dump()
    groups = list(payload["task_groups"])
    groups[0]["task_ids"] = ("reserve-a",)
    payload["task_groups"] = tuple(groups)
    with pytest.raises(ValidationError, match="ungrouped phase tasks: core-a"):
        MatrixStudySpec.model_validate(payload)


def test_gate_references_and_thresholds_are_checked_against_axes() -> None:
    payload = _spec().model_dump()
    payload["gate"]["core_phase_id"] = "unknown"
    with pytest.raises(ValidationError, match="core phase is not declared"):
        MatrixStudySpec.model_validate(payload)

    payload = _spec().model_dump()
    payload["gate"]["full"]["min_unique_winner_targets"] = 5
    with pytest.raises(ValidationError, match="winner-target threshold"):
        MatrixStudySpec.model_validate(payload)

    with pytest.raises(ValidationError, match="reserve phase requires full"):
        PortfolioGateSpec(core_phase_id="core", reserve_phase_id="reserve")


def test_ambiguous_trajectory_ids_are_rejected() -> None:
    spec = MatrixStudySpec(
        study_id="ambiguous",
        seed=1,
        agents=("b-c", "c"),
        targets=("d",),
        task_groups=(TaskGroupSpec(id="group", task_ids=("a", "a-b")),),
        phases=(
            PhaseSpec(
                id="core",
                task_ids=("a", "a-b"),
                replicates=(1,),
                order_policy="fixed",
                max_calls_per_trajectory=1,
            ),
        ),
    )

    with pytest.raises(MatrixSpecError, match="ambiguous trajectory IDs"):
        build_matrix_schedule(spec)


def test_unknown_phase_lookup_fails_explicitly() -> None:
    schedule = build_matrix_schedule(_spec())

    with pytest.raises(MatrixSpecError, match="unknown phase"):
        schedule.cells_for_phase("missing")
