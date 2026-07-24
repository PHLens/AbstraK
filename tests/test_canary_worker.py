from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import abstrak.canary.worker as worker_module
from abstrak.canary.contracts import CaseResult, WorkerJob, WorkerResult
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import get_task_pack, load_oracle_source
from abstrak.canary.worker import _read_worker_revision, load_job_payload, run_worker_job
from abstrak.canary.worker import main as worker_main


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


def test_worker_cli_rejects_device_override_that_differs_from_job(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    job = _job()
    monkeypatch.setattr(sys, "stdin", io.StringIO(job.model_dump_json()))

    status = worker_main(
        [
            "--job",
            "-",
            "--kernelbench-root",
            "/unused",
            "--device",
            "cuda:1",
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert "does not match job.device 'cuda:0'" in captured.err


def test_worker_health_check_emits_one_json_value_and_uses_default_device(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: list[str] = []

    def fake_health(device: str) -> dict[str, object]:
        observed.append(device)
        return {
            "schema_version": "canary-worker-health.v1",
            "status": "healthy",
            "device": device,
            "hardware": "Fake A100",
            "compute_capability": [8, 0],
            "python_version": "3.10.20",
            "torch_version": "2.13.0+cu126",
            "torch_cuda_version": "12.6",
            "triton_version": "3.7.1",
            "value": 2.0,
        }

    monkeypatch.setattr(worker_module, "gpu_health", fake_health)

    status = worker_main(["--health-check"])

    captured = capsys.readouterr()
    assert status == 0
    assert observed == ["cuda:0"]
    assert json.loads(captured.out) == {
        "schema_version": "canary-worker-health.v1",
        "status": "healthy",
        "device": "cuda:0",
        "hardware": "Fake A100",
        "compute_capability": [8, 0],
        "python_version": "3.10.20",
        "torch_version": "2.13.0+cu126",
        "torch_cuda_version": "12.6",
        "triton_version": "3.7.1",
        "value": 2.0,
    }
    assert captured.out.count("\n") == 1
    assert captured.err == ""


def test_worker_health_check_passes_worker_root_to_revision_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: list[tuple[str, str | None]] = []

    def fake_health(
        device: str,
        *,
        worker_root: str | None = None,
    ) -> dict[str, object]:
        observed.append((device, worker_root))
        return {
            "schema_version": "canary-worker-health.v1",
            "status": "healthy",
            "device": device,
            "worker_revision": "a" * 40,
        }

    monkeypatch.setattr(worker_module, "gpu_health", fake_health)

    status = worker_main(
        ["--health-check", "--device", "cuda:1", "--worker-root", "/srv/AbstraK"]
    )

    assert status == 0
    assert observed == [("cuda:1", "/srv/AbstraK")]
    assert json.loads(capsys.readouterr().out)["worker_revision"] == "a" * 40


def test_worker_revision_requires_a_clean_checkout(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "initial"], check=True)

    assert len(_read_worker_revision(tmp_path)) == 40

    (tmp_path / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be clean"):
        _read_worker_revision(tmp_path)


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


def test_worker_uses_ephemeral_scrubbed_environment_and_restores_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    original_directory = os.getcwd()
    monkeypatch.setenv("TEST_API_KEY", "must-not-reach-worker")
    monkeypatch.setenv("TEST_BASE_URL", "https://must-not-reach-worker.invalid/secret")

    def fake_evaluator(candidate_job: WorkerJob, _root: object, **_kwargs: object) -> WorkerResult:
        assert "TEST_API_KEY" not in os.environ
        assert "TEST_BASE_URL" not in os.environ
        assert os.environ["HOME"].startswith("/tmp/abstrak-")
        assert os.getcwd() == os.environ["HOME"]
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

    run_worker_job(job, kernelbench_root="/unused", evaluator=fake_evaluator)

    assert os.getcwd() == original_directory
    assert os.environ["TEST_API_KEY"] == "must-not-reach-worker"
    assert os.environ["TEST_BASE_URL"].endswith("/secret")
