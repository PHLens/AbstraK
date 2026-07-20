from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from abstrak.canary.contracts import CaseResult, WorkerJob, WorkerResult
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import get_task_pack, load_oracle_source
from abstrak.canary.worker import load_job_payload, run_worker_job


def _job() -> WorkerJob:
    task = get_task_pack("row-reduction-scale")
    source = load_oracle_source("row-reduction-scale", "triton")
    return WorkerJob(
        job_id="worker-round-trip",
        kind="dev",
        task=task,
        target=get_target_stack("triton-a100"),
        case_ids=tuple(case.id for case in task.dev_cases),
        candidate_source=source,
        candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


def test_worker_job_json_round_trip_and_result_verification() -> None:
    job = _job()

    def fake_evaluator(candidate_job: WorkerJob, _root: object, **_kwargs: object) -> WorkerResult:
        cases = tuple(
            CaseResult(
                case_id=case_id,
                status="pass",
                correct=True,
                max_abs_error=0.0,
                max_rel_error=0.0,
                output_finite=True,
                inputs_unchanged=True,
            )
            for case_id in candidate_job.case_ids
        )
        return WorkerResult(
            job_id=candidate_job.job_id,
            job_sha256=candidate_job.sha256,
            input_sha256=candidate_job.input_sha256,
            candidate_sha256=candidate_job.candidate_sha256,
            status="completed",
            compiled=True,
            correct=True,
            cases=cases,
        )

    parsed = load_job_payload(job.model_dump_json())
    result = run_worker_job(parsed, kernelbench_root="/unused", evaluator=fake_evaluator)

    assert parsed == job
    assert result.status == "completed"
    assert result.verify_for_job(job) is result


def test_worker_job_json_rejects_candidate_tampering() -> None:
    payload = _job().model_dump(mode="json")
    payload["candidate_source"] += "\n# changed"

    with pytest.raises(ValidationError, match="candidate_source does not match"):
        load_job_payload(json.dumps(payload))


def test_importing_worker_does_not_import_torch() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import abstrak.canary.worker; "
            "raise SystemExit(1 if 'torch' in sys.modules else 0)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
