from __future__ import annotations

from collections import Counter

import pytest
from pydantic import ValidationError

from abstrak.anytime.contracts import (
    FORMAL_CHECKPOINT_CALLS,
    SHAKEOUT_CHECKPOINT_CALLS,
    AnytimeAgentSpec,
    AnytimeCheckpointPolicy,
    AnytimeCohortSpec,
    AnytimeGenerationSpec,
    AnytimeLoopPolicy,
    AnytimeReasoningSpec,
    AnytimeResourceBudget,
    AnytimeStudySpec,
)
from abstrak.anytime.schedule import AnytimeSchedule, build_anytime_schedule

CORE_TASKS = ("kb-l1-2", "kb-l1-8", "kb-l1-40", "kb-l1-93", "kb-l1-97", "kb-l2-2")
RESERVE_TASKS = (
    "kb-l1-24",
    "kb-l1-85",
    "kb-l2-14",
    "kb-l2-99",
    "kb-l2-1",
    "kb-l2-85",
)
TARGETS = ("triton-a100", "tilelang-a100", "cute-a100")


def _agent(identifier: str, protocol: str) -> AnytimeAgentSpec:
    return AnytimeAgentSpec(
        id=identifier,
        provider_id="deepseek" if protocol == "chat_completions" else "openai",
        model_ref=identifier,
        native_protocol=protocol,
        generation=AnytimeGenerationSpec(
            max_output_tokens=16384,
            reasoning=AnytimeReasoningSpec(
                requested_reasoning_effort="xhigh",
                conformance_requirement="literal_xhigh",
            ),
        ),
    )


def _loop(calls: int) -> AnytimeLoopPolicy:
    checkpoints = FORMAL_CHECKPOINT_CALLS if calls == 12 else SHAKEOUT_CHECKPOINT_CALLS
    return AnytimeLoopPolicy(
        budget=AnytimeResourceBudget(
            max_scientific_calls=calls,
            max_total_output_tokens=calls * 16384,
            max_compile_attempts=calls,
            max_evaluation_attempts=calls,
            max_gpu_seconds=calls * 600.0,
        ),
        checkpoints=AnytimeCheckpointPolicy(calls=checkpoints),
    )


def _formal_study(seed: int = 20260803) -> AnytimeStudySpec:
    deepseek = _agent("deepseek-v4-flash", "chat_completions")
    gpt = _agent("gpt-5.6-luna", "responses")
    loop = _loop(12)
    return AnytimeStudySpec(
        study_id="anytime-dsl-a100-formal",
        study_kind="formal",
        seed=seed,
        agents=(deepseek, gpt),
        cohorts=(
            AnytimeCohortSpec(
                id="primary-core",
                agent_id=deepseek.id,
                task_ids=CORE_TASKS,
                target_ids=TARGETS,
                replicates=(1, 2, 3, 4),
                scoring=True,
                loop=loop,
            ),
            AnytimeCohortSpec(
                id="robustness-core",
                agent_id=gpt.id,
                task_ids=CORE_TASKS,
                target_ids=TARGETS,
                replicates=(1, 2, 3),
                scoring=True,
                loop=loop,
            ),
            AnytimeCohortSpec(
                id="primary-reserve",
                agent_id=deepseek.id,
                task_ids=RESERVE_TASKS,
                target_ids=TARGETS,
                replicates=(1, 2, 3, 4),
                scoring=True,
                activation="core_gate",
                loop=loop,
            ),
        ),
    )


def _shakeout_study() -> AnytimeStudySpec:
    tasks = CORE_TASKS[:4]
    deepseek = _agent("deepseek-v4-flash", "chat_completions")
    gpt = _agent("gpt-5.6-luna", "responses")
    loop = _loop(4)
    cohorts = tuple(
        AnytimeCohortSpec(
            id=f"{agent.id}-shakeout",
            agent_id=agent.id,
            task_ids=tasks,
            target_ids=TARGETS,
            replicates=(1, 2),
            scoring=False,
            loop=loop,
        )
        for agent in (deepseek, gpt)
    )
    return AnytimeStudySpec(
        study_id="anytime-dsl-a100-shakeout",
        study_kind="shakeout",
        seed=20260803,
        agents=(deepseek, gpt),
        cohorts=cohorts,
    )


def test_formal_schedule_has_exact_three_cohort_shape_and_ceilings() -> None:
    schedule = build_anytime_schedule(_formal_study())

    assert tuple(len(schedule.cells_for_cohort(cohort_id)) for cohort_id in (
        "primary-core",
        "robustness-core",
        "primary-reserve",
    )) == (72, 54, 72)
    assert schedule.expected_trajectories == 198
    assert schedule.scientific_request_ceiling == 2376
    assert schedule.operational_request_ceiling == 4752
    assert tuple(cell.ordinal for cell in schedule.cells) == tuple(range(198))
    assert len({cell.trajectory_id for cell in schedule.cells}) == 198
    assert schedule.spec.sha256 == (
        "816f8c7a42412d86a503e4df1ad4679369ed49c4e6065ad5cb0526d9d934af9e"
    )
    assert schedule.sha256 == (
        "1bc18ed75d6ffa62671c801d20be9f38c00630c0954cc1af67c85a7a221585d4"
    )


def test_balanced_rotation_equalizes_every_target_position() -> None:
    schedule = build_anytime_schedule(_formal_study())
    expected_first_position_counts = {
        "primary-core": 8,
        "robustness-core": 6,
        "primary-reserve": 8,
    }

    for cohort_id, expected_count in expected_first_position_counts.items():
        first_targets = Counter(
            cell.target_id
            for cell in schedule.cells_for_cohort(cohort_id)
            if cell.target_order_index == 0
        )
        assert first_targets == {target: expected_count for target in TARGETS}


def test_schedule_is_deterministic_and_seed_is_hash_bound() -> None:
    first = build_anytime_schedule(_formal_study())
    replay = build_anytime_schedule(_formal_study())
    different_seed = build_anytime_schedule(_formal_study(seed=20260804))

    assert first == replay
    assert first.sha256 == replay.sha256
    assert first.sha256 != different_seed.sha256


def test_schedule_rejects_tampered_cell_order() -> None:
    schedule = build_anytime_schedule(_formal_study())
    cells = list(schedule.cells)
    cells[0], cells[1] = cells[1], cells[0]

    with pytest.raises(ValidationError, match="do not match"):
        AnytimeSchedule(spec=schedule.spec, spec_sha256=schedule.spec_sha256, cells=tuple(cells))


def test_independent_four_call_shakeout_has_48_trajectories_and_192_calls() -> None:
    schedule = build_anytime_schedule(_shakeout_study())

    assert tuple(len(schedule.cells_for_cohort(cohort.id)) for cohort in schedule.spec.cohorts) == (
        24,
        24,
    )
    assert schedule.expected_trajectories == 48
    assert schedule.scientific_request_ceiling == 192
    assert schedule.operational_request_ceiling == 384
    assert schedule.spec.sha256 == (
        "72d640a79252dc6cfd811545547f88ed91ffabebd2ba43e6bde482de134cdc12"
    )
    assert schedule.sha256 == (
        "a4ee21c1ba7ae9aff00ca84bbe92310ce156335ad159320aef97bf58d8208d8b"
    )
