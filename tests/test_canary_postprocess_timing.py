from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from abstrak.canary import postprocess_timing
from abstrak.canary.artifacts import verify_trajectory
from abstrak.canary.contracts import CaseResult, TimingSpec, WorkerJob, WorkerResult
from abstrak.canary.postprocess_timing import (
    PostprocessTimingError,
    run_formal_candidate_timing,
)


class FakeWorker:
    def __init__(self) -> None:
        self.jobs: list[WorkerJob] = []

    def execute(self, job: WorkerJob) -> WorkerResult:
        self.jobs.append(job)
        assert job.timing is not None
        return WorkerResult(
            job_id=job.job_id,
            job_sha256=job.sha256,
            input_sha256=job.input_sha256,
            candidate_sha256=job.candidate_sha256,
            status="completed",
            compiled=True,
            correct=True,
            cases=tuple(
                CaseResult(
                    case_id=case_id,
                    status="pass",
                    correct=True,
                    max_abs_error=0.0,
                    max_rel_error=0.0,
                    output_finite=True,
                    inputs_unchanged=True,
                )
                for case_id in job.case_ids
            ),
            timing_ms=tuple(1.0 for _ in range(job.timing.trial_runs)),
            timing_cv=0.0,
        )


def _candidate(tmp_path: Path):
    source = "class ModelNew:\n    pass\n"
    return (
        "formal-rmsnorm-static-deepseek-v4-flash-triton-a100-r1",
        "rmsnorm-static",
        "triton-a100",
        tmp_path / "formal-source",
        ("first", "final"),
        source,
        hashlib.sha256(source.encode()).hexdigest(),
    )


def test_timing_record_is_atomic_and_resumes_a_sealed_staging_directory(
    tmp_path: Path, monkeypatch
) -> None:
    candidate = _candidate(tmp_path)
    monkeypatch.setattr(
        postprocess_timing,
        "discover_qualified_candidates",
        lambda **_: (candidate,),
    )
    timing = TimingSpec(warmup_runs=1, trial_runs=2, repetitions=1)
    worker = FakeWorker()

    first = run_formal_candidate_timing(
        worker,
        root=tmp_path,
        timing_study_id="timing-study",
        timing=timing,
    )
    final_path = Path(first[0].artifact_directory)
    staging_path = final_path.with_name(f"{final_path.name}.incomplete")
    os.replace(final_path, staging_path)

    resumed = run_formal_candidate_timing(
        worker,
        root=tmp_path,
        timing_study_id="timing-study",
        timing=timing,
    )

    assert resumed == first
    assert len(worker.jobs) == 1
    assert final_path.is_dir()
    assert not staging_path.exists()
    verify_trajectory(final_path)


def test_unsealed_staging_directory_is_replaced(tmp_path: Path, monkeypatch) -> None:
    candidate = _candidate(tmp_path)
    monkeypatch.setattr(
        postprocess_timing,
        "discover_qualified_candidates",
        lambda **_: (candidate,),
    )
    timing_id = f"timing-{candidate[0]}-first-final"
    staging = tmp_path / "timing-study" / f"{timing_id}.incomplete"
    staging.mkdir(parents=True)
    (staging / "partial.json").write_text("{}", encoding="utf-8")
    worker = FakeWorker()

    records = run_formal_candidate_timing(
        worker,
        root=tmp_path,
        timing_study_id="timing-study",
        timing=TimingSpec(warmup_runs=1, trial_runs=2, repetitions=1),
    )

    assert len(records) == 1
    assert len(worker.jobs) == 1
    assert not staging.exists()


def test_resume_rejects_a_changed_run_manifest(tmp_path: Path, monkeypatch) -> None:
    candidate = _candidate(tmp_path)
    monkeypatch.setattr(
        postprocess_timing,
        "discover_qualified_candidates",
        lambda **_: (candidate,),
    )
    timing = TimingSpec(warmup_runs=1, trial_runs=2, repetitions=1)
    worker = FakeWorker()
    records = run_formal_candidate_timing(
        worker,
        root=tmp_path,
        timing_study_id="timing-study",
        timing=timing,
    )

    with pytest.raises(PostprocessTimingError, match="run manifest differs"):
        run_formal_candidate_timing(
            worker,
            root=tmp_path,
            timing_study_id="timing-study",
            timing=TimingSpec(warmup_runs=2, trial_runs=2, repetitions=1),
        )

    assert len(worker.jobs) == 1
    verify_trajectory(records[0].artifact_directory)
