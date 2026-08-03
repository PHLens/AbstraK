from __future__ import annotations

import pytest
from pydantic import ValidationError

from abstrak.anytime.contracts import (
    FORMAL_CHECKPOINT_CALLS,
    SHAKEOUT_CHECKPOINT_CALLS,
    AnytimeAgentSpec,
    AnytimeCheckpointIdentity,
    AnytimeCheckpointPolicy,
    AnytimeCohortSpec,
    AnytimeContextPolicy,
    AnytimeGenerationSpec,
    AnytimeInfrastructurePolicy,
    AnytimeLoopPolicy,
    AnytimeReasoningSpec,
    AnytimeResourceBudget,
    AnytimeResourceSnapshot,
    AnytimeStudySpec,
)

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
        provider_id="deepseek" if identifier.startswith("deepseek") else "openai",
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


def _formal_study() -> AnytimeStudySpec:
    deepseek = _agent("deepseek-v4-flash", "chat_completions")
    gpt = _agent("gpt-5.6-luna", "responses")
    formal_loop = _loop(12)
    return AnytimeStudySpec(
        study_id="anytime-dsl-a100-formal",
        study_kind="formal",
        seed=20260803,
        agents=(deepseek, gpt),
        cohorts=(
            AnytimeCohortSpec(
                id="primary-core",
                agent_id=deepseek.id,
                task_ids=CORE_TASKS,
                target_ids=TARGETS,
                replicates=(1, 2, 3, 4),
                scoring=True,
                loop=formal_loop,
            ),
            AnytimeCohortSpec(
                id="robustness-core",
                agent_id=gpt.id,
                task_ids=CORE_TASKS,
                target_ids=TARGETS,
                replicates=(1, 2, 3),
                scoring=True,
                loop=formal_loop,
            ),
            AnytimeCohortSpec(
                id="primary-reserve",
                agent_id=deepseek.id,
                task_ids=RESERVE_TASKS,
                target_ids=TARGETS,
                replicates=(1, 2, 3, 4),
                scoring=True,
                activation="core_gate",
                loop=formal_loop,
            ),
        ),
    )


def test_formal_three_cohort_cardinalities_and_request_ceilings() -> None:
    study = _formal_study()

    assert tuple(cohort.expected_trajectories for cohort in study.cohorts) == (72, 54, 72)
    assert study.expected_trajectories == 198
    assert study.scientific_request_ceiling == 2376
    assert study.operational_request_ceiling == 4752
    assert study.cohort("primary-reserve").activation == "core_gate"
    assert study.agent("gpt-5.6-luna").native_protocol == "responses"


def test_reasoning_contract_requires_literal_xhigh_without_claiming_an_effective_mode() -> None:
    reasoning = AnytimeReasoningSpec(
        requested_reasoning_effort="xhigh",
        conformance_requirement="literal_xhigh",
    )

    assert reasoning.model_dump(mode="json") == {
        "schema_version": "abstrak-anytime-reasoning.v1",
        "requested_reasoning_effort": "xhigh",
        "conformance_requirement": "literal_xhigh",
    }
    with pytest.raises(ValidationError):
        AnytimeReasoningSpec.model_validate(
            {
                "requested_reasoning_effort": "high",
                "conformance_requirement": "literal_xhigh",
            }
        )
    with pytest.raises(ValidationError):
        AnytimeReasoningSpec.model_validate(
            {
                "requested_reasoning_effort": "xhigh",
                "conformance_requirement": "thinking_enabled",
            }
        )


@pytest.mark.parametrize(
    "calls,checkpoints",
    ((12, FORMAL_CHECKPOINT_CALLS), (4, SHAKEOUT_CHECKPOINT_CALLS)),
)
def test_fixed_call_loop_supports_formal_and_shakeout(
    calls: int,
    checkpoints: tuple[int, ...],
) -> None:
    policy = _loop(calls)

    assert policy.stop_policy == "fixed_calls"
    assert policy.incumbent_selection == "best_eligible_latency"
    assert policy.budget.max_scientific_calls == calls
    assert policy.budget.max_total_output_tokens == calls * 16384
    assert policy.budget.max_compile_attempts == calls
    assert policy.budget.max_evaluation_attempts == calls
    assert policy.budget.max_gpu_seconds == calls * 600.0
    assert policy.checkpoints.calls == checkpoints
    assert policy.scientific_request_ceiling == calls
    assert policy.operational_request_ceiling == calls * 2
    assert policy.infrastructure.unsubmitted_failures_consume_scientific_calls is False
    assert policy.infrastructure.submitted_failures_consume_scientific_calls is True


@pytest.mark.parametrize(
    "calls,checkpoints,match",
    (
        (12, (4, 8, 12), "first scientific call"),
        (12, (1, 4, 8), "last checkpoint"),
        (4, (1, 4, 8), "last checkpoint"),
    ),
)
def test_loop_rejects_checkpoints_inconsistent_with_budget(
    calls: int,
    checkpoints: tuple[int, ...],
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        AnytimeLoopPolicy(
            budget=AnytimeResourceBudget(
                max_scientific_calls=calls,
                max_total_output_tokens=calls * 16384,
                max_compile_attempts=calls,
                max_evaluation_attempts=calls,
                max_gpu_seconds=calls * 600.0,
            ),
            checkpoints=AnytimeCheckpointPolicy(calls=checkpoints),
        )


@pytest.mark.parametrize("checkpoints", ((1, 4, 4, 12), (1, 8, 4, 12), (0, 4, 8, 12)))
def test_checkpoint_policy_rejects_duplicate_unsorted_or_zero_calls(
    checkpoints: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError):
        AnytimeCheckpointPolicy(calls=checkpoints)


def test_resource_and_infrastructure_bounds_fail_closed() -> None:
    with pytest.raises(ValidationError):
        AnytimeResourceBudget(
            max_scientific_calls=13,
            max_total_output_tokens=13 * 16384,
            max_compile_attempts=12,
            max_evaluation_attempts=12,
            max_gpu_seconds=12 * 600.0,
        )
    with pytest.raises(ValidationError):
        AnytimeInfrastructurePolicy(max_attempts_per_trajectory=3)
    with pytest.raises(ValidationError):
        AnytimeResourceBudget(
            max_scientific_calls=True,
            max_total_output_tokens=16384,
            max_compile_attempts=1,
            max_evaluation_attempts=1,
            max_gpu_seconds=600.0,
        )
    with pytest.raises(ValidationError, match="compile-attempt cap"):
        AnytimeResourceBudget(
            max_scientific_calls=4,
            max_total_output_tokens=4 * 16384,
            max_compile_attempts=5,
            max_evaluation_attempts=4,
            max_gpu_seconds=4 * 600.0,
        )
    with pytest.raises(ValidationError, match="GPU-seconds cap"):
        AnytimeResourceBudget(
            max_scientific_calls=4,
            max_total_output_tokens=4 * 16384,
            max_compile_attempts=4,
            max_evaluation_attempts=4,
            max_gpu_seconds=2401.0,
        )


def test_context_policy_freezes_component_order_and_forbids_provider_state() -> None:
    policy = AnytimeContextPolicy()

    assert policy.component_order == (
        "base_prompt",
        "incumbent",
        "previous_candidate",
        "previous_feedback",
    )
    assert policy.provider_session_state == "forbidden"
    assert policy.previous_response_id == "forbidden"
    assert policy.automatic_compaction == "forbidden"
    assert policy.max_candidate_source_characters == 262144
    assert policy.oversize_source_policy == "reject"
    with pytest.raises(ValidationError, match="canonical order"):
        AnytimeContextPolicy(
            component_order=(
                "base_prompt",
                "previous_candidate",
                "incumbent",
                "previous_feedback",
            )
        )


def test_resource_snapshot_and_nullable_checkpoint_identity_are_hash_bound() -> None:
    snapshot = AnytimeResourceSnapshot(
        scientific_calls_consumed=1,
        provider_requests_submitted=1,
        possibly_charged_requests=1,
        known_input_tokens=100,
        known_cached_input_tokens=20,
        known_output_tokens=50,
        known_reasoning_tokens=25,
        usage_complete=True,
        compile_attempts=0,
        evaluation_attempts=0,
        provider_seconds=2.0,
        wall_seconds=2.1,
    )
    checkpoint = AnytimeCheckpointIdentity(
        trajectory_id="trajectory-1",
        infrastructure_attempt_index=1,
        scientific_call_index=1,
        trajectory_execution_sha256="1" * 64,
        ledger_prefix_sha256="2" * 64,
        incumbent_candidate_sha256=None,
        resource_snapshot_sha256=snapshot.sha256,
    )

    assert len(snapshot.sha256) == 64
    assert len(checkpoint.sha256) == 64
    assert checkpoint.incumbent_candidate_sha256 is None
    assert checkpoint.infrastructure_attempt_index == 1
    with pytest.raises(ValidationError, match="cached input"):
        AnytimeResourceSnapshot(
            scientific_calls_consumed=1,
            provider_requests_submitted=1,
            possibly_charged_requests=1,
            known_input_tokens=10,
            known_cached_input_tokens=11,
            usage_complete=True,
        )
    with pytest.raises(ValidationError, match="must equal consumed"):
        AnytimeResourceSnapshot(
            scientific_calls_consumed=1,
            provider_requests_submitted=2,
            possibly_charged_requests=1,
            usage_complete=False,
        )
    checkpoint_payload = checkpoint.model_dump()
    checkpoint_payload["infrastructure_attempt_index"] = 0
    with pytest.raises(ValidationError):
        AnytimeCheckpointIdentity.model_validate(checkpoint_payload)


def test_study_rejects_unknown_fields_duplicate_axes_and_invalid_identifiers() -> None:
    study = _formal_study()
    payload = study.model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        AnytimeStudySpec.model_validate(payload)

    cohort = study.cohorts[0]
    invalid = cohort.model_dump()
    invalid["task_ids"] = ("kb-l1-2", "kb-l1-2")
    with pytest.raises(ValidationError, match="must be unique"):
        AnytimeCohortSpec.model_validate(invalid)
    invalid["task_ids"] = ("NOT SAFE",)
    with pytest.raises(ValidationError, match="invalid identifiers"):
        AnytimeCohortSpec.model_validate(invalid)


def test_study_rejects_cross_cohort_scientific_cell_collision() -> None:
    study = _formal_study()
    duplicate = study.cohorts[0].model_copy(update={"id": "duplicate-core"})

    with pytest.raises(ValidationError, match="duplicate scientific cell"):
        AnytimeStudySpec(
            study_id=study.study_id,
            study_kind=study.study_kind,
            seed=study.seed,
            agents=study.agents,
            cohorts=(*study.cohorts, duplicate),
        )


def test_formal_and_shakeout_semantics_cannot_be_mixed() -> None:
    formal = _formal_study()
    bad_loop = _loop(4)
    payload = formal.model_dump()
    payload["cohorts"][0]["loop"] = bad_loop.model_dump()

    with pytest.raises(ValidationError, match="formal cohorts require 12 calls"):
        AnytimeStudySpec.model_validate(payload)


def test_study_binds_total_output_cap_to_agent_generation_limit() -> None:
    study = _formal_study()
    payload = study.model_dump()
    payload["cohorts"][0]["loop"]["budget"]["max_total_output_tokens"] -= 1

    with pytest.raises(ValidationError, match="output-token cap"):
        AnytimeStudySpec.model_validate(payload)
