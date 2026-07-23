from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from abstrak.canary.contracts import (
    R1_AGENT_LOOP_POLICY,
    AgentBudget,
    AgentLoopPolicy,
    CaseResult,
    InputCaseSpec,
    TargetStackSpec,
    TaskPackSpec,
    TrajectoryOutcome,
    WorkerJob,
    WorkerResult,
)
from abstrak.providers.contracts import sha256_json


def _task() -> TaskPackSpec:
    return TaskPackSpec(
        id="row-reduction-scale",
        specification="Sum each row in FP32, scale by 0.5, and return FP16.",
        source_path="tasks/row_reduction_scale.py",
        source_sha256="1" * 64,
        dtype="fp16",
        input_shapes=((1024, 1024),),
        parameters=(("scale", 0.5),),
        atol=1e-2,
        rtol=1e-2,
        fallback_policy="forbid_framework_ops",
        dev_cases=(InputCaseSpec(id="dev-1", kind="random", seed=1),),
        sealed_cases=(InputCaseSpec(id="sealed-1", kind="constant", seed=2, value=0.25),),
    )


def _target() -> TargetStackSpec:
    return TargetStackSpec(
        id="triton-a100",
        backend="triton",
        version="3.7.1",
        card_path="targets/triton.md",
        card_sha256="2" * 64,
        adapter="kernelbench",
    )


def _job(*, kind: str = "dev", case_ids: tuple[str, ...] = ("dev-1",)) -> WorkerJob:
    source = "class ModelNew: pass\n"
    return WorkerJob(
        job_id="job-1",
        kind=kind,
        task=_task(),
        target=_target(),
        case_ids=case_ids,
        candidate_source=source,
        candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


def test_input_case_value_matches_kind() -> None:
    with pytest.raises(ValidationError, match="constant cases require"):
        InputCaseSpec(id="constant", kind="constant", seed=1)
    with pytest.raises(ValidationError, match="random cases cannot"):
        InputCaseSpec(id="random", kind="random", seed=1, value=0.0)

    with pytest.raises(ValidationError):
        InputCaseSpec(id="random", kind="random", seed="1")


def test_task_rejects_duplicate_case_ids_across_splits() -> None:
    task = _task().model_dump()
    task["sealed_cases"][0]["id"] = "dev-1"

    with pytest.raises(ValidationError, match="case IDs must be unique"):
        TaskPackSpec.model_validate(task)


def test_worker_job_checks_candidate_hash_and_case_split() -> None:
    payload = _job().model_dump()
    payload["candidate_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="candidate_source does not match"):
        WorkerJob.model_validate(payload)

    with pytest.raises(ValidationError, match="dev job cannot use"):
        _job(case_ids=("sealed-1",))


def test_worker_job_hashes_are_stable_and_job_id_scoped() -> None:
    first = _job()
    second = first.model_copy(update={"job_id": "job-2"})

    assert first.input_sha256 == second.input_sha256
    assert first.sha256 != second.sha256


def test_task_parameters_are_immutable_and_public_view_hides_cases() -> None:
    task = _task()
    public_payload = task.public_view().model_dump(mode="json")

    with pytest.raises(TypeError):
        task.parameters[0] = ("scale", 1.0)
    assert set(public_payload) == {
        "id",
        "specification",
        "dtype",
        "reference_precision",
        "input_shapes",
        "parameters",
        "init_args",
        "atol",
        "rtol",
    }
    assert "sealed" not in str(public_payload)
    assert "seed" not in str(public_payload)


def test_completed_result_requires_all_cases_to_pass() -> None:
    job = _job()
    passing = CaseResult(
        case_id="dev-1",
        status="pass",
        correct=True,
        max_abs_error=0.0,
        max_rel_error=0.0,
        output_finite=True,
        inputs_unchanged=True,
    )
    result = WorkerResult(
        job_id=job.job_id,
        job_sha256=job.sha256,
        input_sha256=job.input_sha256,
        candidate_sha256=job.candidate_sha256,
        status="completed",
        compiled=True,
        correct=True,
        cases=(passing,),
    )

    assert result.status == "completed"
    with pytest.raises(ValidationError, match="completed results require compiled"):
        WorkerResult.model_validate({**result.model_dump(), "correct": False})
    assert result.verify_for_job(job) is result

    other_job = job.model_copy(update={"job_id": "job-other"})
    with pytest.raises(ValueError, match="does not match job"):
        result.verify_for_job(other_job)


def test_wrong_result_requires_one_failing_case() -> None:
    job = _job()
    failing = CaseResult(
        case_id="dev-1",
        status="wrong_result",
        correct=False,
        max_abs_error=1.0,
        max_rel_error=1.0,
        output_finite=True,
        inputs_unchanged=True,
    )

    result = WorkerResult(
        job_id=job.job_id,
        job_sha256=job.sha256,
        input_sha256=job.input_sha256,
        candidate_sha256=job.candidate_sha256,
        status="wrong_result",
        compiled=True,
        cases=(failing,),
    )

    assert result.correct is False
    assert result.verify_for_job(job) is result


def test_target_oracle_reference_must_be_complete() -> None:
    with pytest.raises(ValidationError, match="must be supplied together"):
        TargetStackSpec(
            id="triton-a100",
            backend="triton",
            version="3.7.1",
            card_path="targets/triton.md",
            card_sha256="2" * 64,
            adapter="kernelbench",
            oracle_path="oracles/triton.py",
        )


def test_agent_budget_is_strict_and_fixed_to_four_calls() -> None:
    assert AgentBudget().max_calls == 4
    with pytest.raises(ValidationError):
        AgentBudget(max_calls="4")
    with pytest.raises(ValidationError):
        AgentBudget(max_calls=5)


def test_agent_loop_policy_defaults_to_the_frozen_r1_behavior() -> None:
    assert R1_AGENT_LOOP_POLICY == AgentLoopPolicy()
    assert R1_AGENT_LOOP_POLICY.model_dump(mode="json") == {
        "schema_version": "canary-agent-loop-policy.v1",
        "response_parser": "agent_marker",
        "stop_policy": "agent",
        "final_selection": "last",
        "latency_ceiling_ms": None,
    }
    assert R1_AGENT_LOOP_POLICY.sha256 == sha256_json(R1_AGENT_LOOP_POLICY)


def test_candidate_only_policy_requires_controller_latency_stop() -> None:
    policy = AgentLoopPolicy(
        response_parser="candidate_only",
        stop_policy="correct_latency",
        final_selection="best_correct_latency",
        latency_ceiling_ms=1.25,
    )

    assert policy.latency_ceiling_ms == 1.25
    with pytest.raises(ValidationError, match="supported pair"):
        AgentLoopPolicy(response_parser="candidate_only")
    with pytest.raises(ValidationError, match="finite positive"):
        AgentLoopPolicy(
            response_parser="candidate_only",
            stop_policy="correct_latency",
            latency_ceiling_ms=float("inf"),
        )
    with pytest.raises(ValidationError, match="cannot declare"):
        AgentLoopPolicy(latency_ceiling_ms=1.25)


def test_no_candidate_outcome_cannot_contain_candidate_or_sealed_results() -> None:
    now = datetime.now(timezone.utc)
    outcome = TrajectoryOutcome(
        trajectory_id="trajectory-1",
        status="no_candidate",
        calls=4,
        usage_complete=True,
        started_at_utc=now,
        finished_at_utc=now,
    )

    assert outcome.first_candidate_sha256 is None
    with pytest.raises(ValidationError, match="cannot contain candidate"):
        TrajectoryOutcome.model_validate(
            {**outcome.model_dump(), "first_candidate_sha256": "1" * 64,
             "final_candidate_sha256": "1" * 64}
        )
