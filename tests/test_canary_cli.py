from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abstrak.canary import cli
from abstrak.canary.artifacts import verify_trajectory
from abstrak.canary.contracts import CaseResult, WorkerJob, WorkerResult
from abstrak.canary.remote import WorkerExecutionError
from abstrak.providers.contracts import NormalizedResponse, NormalizedUsage
from abstrak.providers.manifests import ManifestBundle


def _worker_result(job: WorkerJob) -> WorkerResult:
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
        for case_id in job.case_ids
    )
    timing = tuple(1.0 for _ in range(job.timing.trial_runs)) if job.timing else ()
    return WorkerResult(
        job_id=job.job_id,
        job_sha256=job.sha256,
        input_sha256=job.input_sha256,
        candidate_sha256=job.candidate_sha256,
        status="completed",
        compiled=True,
        correct=True,
        cases=cases,
        timing_ms=timing,
        timing_cv=0.0 if timing else None,
        metadata={"post_job_gpu_health": {"status": "healthy", "device": job.device}},
    )


class FakeWorker:
    def __init__(self) -> None:
        self.jobs: list[WorkerJob] = []
        self.python_executable = "/fake/python"
        self.kernelbench_root = "/fake/KernelBench"
        self.asset_root = "/fake/assets"
        self.timeout_seconds = 300.0
        self.expected_hardware_substring = "A100"
        self.expected_compute_capability = (8, 0)
        self.expected_triton_version = "3.7.1"

    def execute(self, job: WorkerJob) -> WorkerResult:
        self.jobs.append(job)
        return _worker_result(job)


def _response(request: Any, text: str) -> NormalizedResponse:
    now = datetime.now(timezone.utc)
    return NormalizedResponse(
        request_id=request.request_id,
        attempt_id="attempt-1",
        provider_request_id="provider-1",
        provider_id="test-provider",
        model_id="test-model",
        provider_manifest_sha256="1" * 64,
        model_manifest_sha256="2" * 64,
        requested_model="test-model",
        returned_model="test-model",
        text=text,
        finish_reason="stop",
        provider_finish_reason="stop",
        usage=NormalizedUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_characters=100,
            output_characters=len(text),
            provider_reported=True,
            core_fields_complete=True,
        ),
        started_at_utc=now,
        finished_at_utc=now,
        elapsed_ms=1.0,
        logical_request_sha256="3" * 64,
        transport_request_sha256="4" * 64,
        transport_response_sha256="5" * 64,
        sanitized_transport_request={},
        raw_transport_response={},
    )


def test_validate_is_offline_and_reports_frozen_pair(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "ProviderClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider was created")),
    )

    exit_code = cli.main(["validate"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert output["status"] == "valid"
    assert output["tasks"] == ["matmul-bias", "row-reduction-scale"]
    assert output["targets"] == ["cute-a100", "tilelang-a100", "triton-a100"]
    assert len(output["trusted_pairs"]) == 6
    assert {pair["target_id"] for pair in output["trusted_pairs"]} == set(
        output["targets"]
    )


def test_run_cell_guards_precede_config_auth_artifacts_and_network(capsys, monkeypatch) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("side effect occurred before live guard")

    monkeypatch.setattr(cli, "load_app_config", unexpected)
    monkeypatch.setattr(cli, "TrajectoryStore", unexpected)
    monkeypatch.setattr(cli, "ProviderClient", unexpected)

    missing_live = cli.main(["run-cell", "--expected-max-requests", "4"])
    wrong_count = cli.main(
        ["run-cell", "--live", "--expected-max-requests", "3"]
    )

    error = capsys.readouterr().err
    assert missing_live == cli.EXIT_CONFIG
    assert wrong_count == cli.EXIT_CONFIG
    assert "requires --live" in error
    assert "must equal the fixed request ceiling (4)" in error


def test_worker_subcommand_delegates_exact_arguments(monkeypatch) -> None:
    received: list[str] = []

    def fake_worker(arguments: list[str]) -> int:
        received.extend(arguments)
        return 17

    monkeypatch.setattr(cli, "worker_main", fake_worker)

    assert cli.main(["worker", "--health-check", "--device", "cuda:1"]) == 17
    assert received == ["--health-check", "--device", "cuda:1"]


def test_ssh_worker_defaults_are_hashable_a100_execution_inputs() -> None:
    arguments = cli._parser().parse_args(
        [
            "run-trusted",
            "--ssh-host",
            "gpu.example",
            "--worker-root",
            "/srv/AbstraK",
        ]
    )

    worker = cli._worker_executor(arguments)
    record = cli._transport_record(worker)

    assert record["python_executable"] == "/tmp/abstrak-gpu-venv/bin/python"
    assert record["pythonpath"] == "/srv/AbstraK/src"
    assert record["kernelbench_root"] == "/srv/KernelBench"
    assert record["asset_root"] == "/srv/AbstraK/benchmarks/r1-a100"
    assert record["expected_hardware_substring"] == "A100"
    assert record["expected_compute_capability"] == (8, 0)
    assert record["expected_triton_version"] == "3.7.1"
    assert record["sandbox"] == "bubblewrap"
    assert record["network_isolated"] is True


def test_supervised_ssh_worker_records_missing_isolation() -> None:
    arguments = cli._parser().parse_args(
        [
            "run-trusted",
            "--ssh-host",
            "gpu.example",
            "--worker-root",
            "/srv/AbstraK",
            "--allow-supervised-worker",
        ]
    )

    worker = cli._worker_executor(arguments)
    record = cli._transport_record(worker)

    assert record["sandbox"] == "setpriv-supervised"
    assert record["sandbox_user"] == "nobody"
    assert record["network_isolated"] is False
    assert record["filesystem_read_only"] is False
    assert record["low_privilege"] is True


def test_run_cell_rejects_unsandboxed_local_worker_before_auth(capsys, monkeypatch) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("auth or provider was accessed")

    monkeypatch.setattr(cli, "load_app_config", unexpected)
    monkeypatch.setattr(cli, "ProviderClient", unexpected)

    exit_code = cli.main(
        ["run-cell", "--live", "--expected-max-requests", "4"]
    )

    assert exit_code == cli.EXIT_CONFIG
    assert "requires --ssh-host" in capsys.readouterr().err


def test_run_trusted_uses_target_backend_and_all_sealed_cases(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    worker = FakeWorker()
    observed_backend: list[str] = []
    original_loader = cli.load_oracle_source

    def load_oracle(task_id: str, target_backend: str, **kwargs: Any) -> str:
        observed_backend.append(target_backend)
        return original_loader(task_id, target_backend, **kwargs)

    monkeypatch.setattr(cli, "load_oracle_source", load_oracle)
    monkeypatch.setattr(cli, "_worker_executor", lambda arguments: worker)

    exit_code = cli.main(
        [
            "run-trusted",
            "--job-id",
            "trusted-test",
            "--artifact-root",
            str(tmp_path),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert observed_backend == ["triton"]
    assert output["status"] == "complete"
    assert worker.jobs[0].kind == "oracle"
    assert worker.jobs[0].case_ids == tuple(
        case.id for case in worker.jobs[0].task.sealed_cases
    )
    verify_trajectory(output["artifact_directory"])


def test_run_trusted_seals_transport_failure_and_health(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    health = {
        "schema_version": "canary-worker-health.v1",
        "status": "unhealthy",
        "device": "cuda:0",
        "error": "CUDA context failed",
    }

    class FailingExecutor(FakeWorker):
        def execute(self, job: WorkerJob) -> WorkerResult:
            raise WorkerExecutionError("health_unhealthy", "CUDA context failed", health=health)

    monkeypatch.setattr(cli, "_worker_executor", lambda arguments: FailingExecutor())

    exit_code = cli.main(
        [
            "run-trusted",
            "--job-id",
            "trusted-failure",
            "--artifact-root",
            str(tmp_path),
        ]
    )

    directory = tmp_path / "r1-a100-trusted" / "trusted-failure"
    error = json.loads((directory / "worker-error.json").read_text(encoding="utf-8"))
    assert exit_code == cli.EXIT_WORKER
    assert "artifact directory" in capsys.readouterr().err
    assert error["post_job_gpu_health"] == health
    verify_trajectory(directory)


def test_run_trusted_seals_unexpected_controller_failure(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    class CrashingExecutor(FakeWorker):
        def execute(self, job: WorkerJob) -> WorkerResult:
            raise RuntimeError("sensitive diagnostic must not be persisted")

    monkeypatch.setattr(cli, "_worker_executor", lambda arguments: CrashingExecutor())

    exit_code = cli.main(
        [
            "run-trusted",
            "--job-id",
            "trusted-crash",
            "--artifact-root",
            str(tmp_path),
        ]
    )

    directory = tmp_path / "r1-a100-trusted" / "trusted-crash"
    error = json.loads((directory / "controller-error.json").read_text(encoding="utf-8"))
    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_ARTIFACT
    assert error == {"error_type": "RuntimeError"}
    assert "sensitive diagnostic" not in captured.err
    verify_trajectory(directory)


def test_run_cell_wires_fixed_bundle_and_seals_private_artifacts(
    tmp_path: Path,
    capsys,
    monkeypatch,
    manifest_bundle: ManifestBundle,
    provider_environment: dict[str, str],
) -> None:
    worker = FakeWorker()
    clients: list[Any] = []

    class FakeConfig:
        def bundle(self, profile: str | None = None) -> ManifestBundle:
            assert profile is None
            return manifest_bundle

    class FakeClient:
        def __init__(self, bundle: ManifestBundle, *, environment: dict[str, str]) -> None:
            self.bundle = bundle
            self.environment = environment
            self.requests: list[Any] = []
            self.resolved_manifest_record = {
                "provider_id": bundle.provider.id,
                "model_id": bundle.model.id,
            }
            clients.append(self)

        def complete(self, request: Any) -> NormalizedResponse:
            self.requests.append(request)
            return _response(
                request,
                "```python\nclass ModelNew:\n    pass\n```\nFINISH\n",
            )

    monkeypatch.setattr(cli, "load_app_config", lambda path: FakeConfig())
    monkeypatch.setattr(cli, "load_auth_store", lambda path, missing_ok: object())
    monkeypatch.setattr(
        cli,
        "runtime_environment",
        lambda auth, environment: provider_environment,
    )
    monkeypatch.setattr(cli, "ProviderClient", FakeClient)
    monkeypatch.setattr(cli, "_worker_executor", lambda arguments: worker)

    exit_code = cli.main(
        [
            "run-cell",
            "--live",
            "--expected-max-requests",
            "4",
            "--trajectory-id",
            "cli-test",
            "--artifact-root",
            str(tmp_path),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    run_directory = Path(output["artifact_directory"])
    assert exit_code == cli.EXIT_OK
    assert output["status"] == "finished"
    assert len(clients) == 1
    assert clients[0].bundle.model.generation.max_completion_tokens == 8192
    assert clients[0].bundle.model.generation.temperature == 0
    assert clients[0].bundle.model.output_contract == "plain_text"
    assert clients[0].bundle.model.allow_live_probe is False
    assert manifest_bundle.model.generation.max_completion_tokens == 128
    assert [job.kind for job in worker.jobs] == ["dev", "sealed", "sealed"]
    assert "sealed-random" not in clients[0].requests[0].messages[1].content
    assert (run_directory / "run-manifest.json").is_file()
    verify_trajectory(run_directory)
    assert all(
        provider_environment["TEST_API_KEY"].encode() not in path.read_bytes()
        for path in run_directory.rglob("*")
        if path.is_file()
    )
