from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from abstrak.canary import cli
from abstrak.canary.artifacts import verify_trajectory
from abstrak.canary.contracts import CaseResult, WorkerJob, WorkerResult
from abstrak.canary.remote import WorkerExecutionError
from abstrak.providers.contracts import (
    ErrorCategory,
    NormalizedError,
    NormalizedResponse,
    NormalizedUsage,
    ProviderCallError,
)
from abstrak.providers.manifests import ManifestBundle

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_STUDY = REPOSITORY_ROOT / "benchmarks" / "capability-gate-a100" / "study.json"
CAPABILITY_STUDY_SHA256 = "876b18e75d86e77c6e2e4cd47038f60719ba6108943ddc754086ea82685ecd00"
CAPABILITY_SCHEDULE_SHA256 = "40c372285875337ebd62529d72b2dd5bc2f6d123cbb2940a93c7482d2537983e"


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
    assert output["tasks"] == [
        "gemm-bias-relu-static",
        "gemm-static",
        "layernorm-static",
        "matmul-bias",
        "rmsnorm-static",
        "row-reduction-scale",
    ]
    assert output["targets"] == ["cute-a100", "tilelang-a100", "triton-a100"]
    assert len(output["trusted_pairs"]) == 18
    assert {pair["target_id"] for pair in output["trusted_pairs"]} == set(output["targets"])


def test_inspect_study_is_offline_and_reports_dynamic_phase_ceilings(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "ProviderClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider was created")),
    )

    exit_code = cli.main(
        [
            "inspect-study",
            "--study-spec",
            str(CAPABILITY_STUDY),
            "--expected-study-sha256",
            CAPABILITY_STUDY_SHA256,
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert output["status"] == "structurally_valid"
    assert output["assets_validated"] is False
    assert output["study_id"] == "tilelang-capability-gate-a100-v1"
    assert output["study_spec_sha256"] == CAPABILITY_STUDY_SHA256
    assert output["schedule_sha256"] == CAPABILITY_SCHEDULE_SHA256
    assert output["expected_trajectories"] == 96
    assert output["request_ceiling"] == 288
    assert output["operational_request_ceiling"] == 576
    assert [phase["expected_trajectories"] for phase in output["phases"]] == [48, 48]
    assert [phase["request_ceiling"] for phase in output["phases"]] == [144, 144]
    assert [phase["operational_request_ceiling"] for phase in output["phases"]] == [
        288,
        288,
    ]


PREFLIGHT_MAX_WORKER_JOBS = 372


def _preflight_study_arguments(
    *extra: str,
    expected_max_jobs: int = PREFLIGHT_MAX_WORKER_JOBS,
) -> list[str]:
    return [
        "preflight-study",
        "--study-spec",
        str(CAPABILITY_STUDY),
        "--expected-study-sha256",
        CAPABILITY_STUDY_SHA256,
        "--asset-root",
        str(CAPABILITY_STUDY.parent),
        "--artifact-root",
        "/local/preflight-artifacts",
        "--expected-max-jobs",
        str(expected_max_jobs),
        "--ssh-host",
        "gpu.example",
        "--ssh-port",
        "30554",
        "--worker-root",
        "/mnt/lipenghui/AbstraK",
        "--worker-python",
        "/tmp/abstrak-gpu-venv/bin/python",
        "--worker-pythonpath",
        "/mnt/lipenghui/AbstraK/src",
        "--worker-kernelbench-root",
        "/mnt/lipenghui/KernelBench",
        "--worker-asset-root",
        "/mnt/lipenghui/AbstraK/benchmarks/capability-gate-a100",
        "--worker-timeout",
        "777",
        "--device",
        "cuda:1",
        "--expected-accelerator",
        "NVIDIA A100-SXM4-80GB",
        "--expected-compute-capability",
        "8.0",
        "--expected-python-version",
        "3.11.9",
        "--expected-tilelang-version",
        "0.1.7",
        "--expected-triton-version",
        "3.7.1",
        "--expected-torch-version",
        "2.8.0+cu128",
        "--expected-cuda-version",
        "12.8",
        "--expected-driver-version",
        "570.133.20",
        "--expected-kernelbench-revision",
        "b" * 40,
        *extra,
    ]


def test_preflight_study_live_guard_precedes_study_revision_and_remote_access(
    capsys,
    monkeypatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("preflight side effect occurred before --live guard")

    for name in (
        "load_study_spec",
        "build_capability_asset_manifest",
        "read_clean_controller_revision",
        "_preflight_transport",
        "run_matrix_preflight",
    ):
        monkeypatch.setattr(cli, name, unexpected)

    exit_code = cli.main(_preflight_study_arguments())

    assert exit_code == cli.EXIT_CONFIG
    assert "requires --live" in capsys.readouterr().err


def test_preflight_study_hash_guard_precedes_revision_and_remote_access(
    capsys,
    monkeypatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("preflight continued after study hash mismatch")

    for name in (
        "build_capability_asset_manifest",
        "read_clean_controller_revision",
        "_preflight_transport",
        "run_matrix_preflight",
    ):
        monkeypatch.setattr(cli, name, unexpected)
    arguments = _preflight_study_arguments("--live")
    arguments[arguments.index(CAPABILITY_STUDY_SHA256)] = "0" * 64

    exit_code = cli.main(arguments)

    assert exit_code == cli.EXIT_CONFIG
    assert "study manifest SHA-256 mismatch" in capsys.readouterr().err


def test_preflight_study_job_ceiling_guard_precedes_revision_and_remote_access(
    capsys,
    monkeypatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("preflight continued after worker-job ceiling mismatch")

    for name in (
        "read_clean_controller_revision",
        "_preflight_transport",
        "run_matrix_preflight",
    ):
        monkeypatch.setattr(cli, name, unexpected)

    exit_code = cli.main(
        _preflight_study_arguments(
            "--live",
            expected_max_jobs=PREFLIGHT_MAX_WORKER_JOBS - 1,
        )
    )

    assert exit_code == cli.EXIT_CONFIG
    assert (
        f"frozen preflight ceiling ({PREFLIGHT_MAX_WORKER_JOBS})"
        in capsys.readouterr().err
    )


def test_preflight_study_maps_parser_inputs_to_generic_runner_without_provider_access(
    capsys,
    monkeypatch,
) -> None:
    revision = "a" * 40
    hashes = {
        "assets": "1" * 64,
        "environment": "2" * 64,
        "floor": "3" * 64,
        "execution_context": "4" * 64,
        "receipt": "5" * 64,
        "bundle": "6" * 64,
        "contract": "7" * 64,
        "probe": "8" * 64,
    }
    protocols = tuple(
        SimpleNamespace(kind=kind)
        for kind in (
            "oracle",
            "oracle",
            "baseline",
            "capability",
            "capability",
            "capability",
            "launch",
        )
    )
    result = SimpleNamespace(
        resumed_ready_bundle=False,
        contract=SimpleNamespace(
            protocols=protocols,
            max_worker_jobs_per_invocation=PREFLIGHT_MAX_WORKER_JOBS,
            sha256=hashes["contract"],
        ),
        environment_probe=SimpleNamespace(sha256=hashes["probe"]),
        bundle=SimpleNamespace(
            assets=SimpleNamespace(sha256=hashes["assets"]),
            environment=SimpleNamespace(sha256=hashes["environment"]),
            floor=SimpleNamespace(sha256=hashes["floor"]),
            execution_context=SimpleNamespace(sha256=hashes["execution_context"]),
            receipt=SimpleNamespace(sha256=hashes["receipt"]),
            sha256=hashes["bundle"],
        ),
        preflight_directory=Path("/sealed/preflight-ready"),
    )
    observed: dict[str, Any] = {}
    original_asset_builder = cli.build_capability_asset_manifest

    def build_assets(
        pinned: object,
        schedule: object,
        *,
        asset_root: Path,
    ) -> object:
        observed["asset_builder"] = (pinned, schedule, asset_root)
        return original_asset_builder(pinned, schedule, asset_root=asset_root)

    def run_preflight(*args: Any, **kwargs: Any) -> object:
        observed["run"] = (args, kwargs)
        return result

    def read_revision(root: Path) -> str:
        observed["revision_root"] = root
        return revision

    monkeypatch.setattr(cli, "build_capability_asset_manifest", build_assets)
    monkeypatch.setattr(cli, "read_clean_controller_revision", read_revision)
    monkeypatch.setattr(cli, "run_matrix_preflight", run_preflight)
    for name in (
        "load_app_config",
        "load_auth_store",
        "runtime_environment",
        "ProviderClient",
    ):
        monkeypatch.setattr(
            cli,
            name,
            lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"preflight-study accessed provider path {_name}")
            ),
        )

    exit_code = cli.main(_preflight_study_arguments("--live"))

    output = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert capsys.readouterr().err == ""
    pinned, schedule, assets, pending_environment = observed["run"][0]
    run_options = observed["run"][1]
    assert observed["revision_root"] == cli.REPOSITORY_ROOT
    assert observed["asset_builder"] == (
        pinned,
        schedule,
        CAPABILITY_STUDY.parent.resolve(),
    )
    assert pending_environment.controller_revision == revision
    assert pending_environment.worker_revision == revision
    assert pending_environment.accelerator == "NVIDIA A100-SXM4-80GB"
    assert pending_environment.compute_capability == "8.0"
    assert pending_environment.python_version == "3.11.9"
    assert pending_environment.tilelang_version == "0.1.7"
    assert pending_environment.triton_version == "3.7.1"
    assert pending_environment.torch_version == "2.8.0+cu128"
    assert pending_environment.cuda_version == "12.8"
    assert pending_environment.driver_version == "570.133.20"
    assert pending_environment.kernelbench_revision == "b" * 40
    assert pending_environment.transport.model_dump(mode="json") == {
        "schema_version": "abstrak-matrix-transport-context.v1",
        "kind": "ssh",
        "host": "gpu.example",
        "port": 30554,
        "worker_root": "/mnt/lipenghui/AbstraK",
        "python_executable": "/tmp/abstrak-gpu-venv/bin/python",
        "pythonpath": "/mnt/lipenghui/AbstraK/src",
        "kernelbench_root": "/mnt/lipenghui/KernelBench",
        "asset_root": (
            "/mnt/lipenghui/AbstraK/benchmarks/capability-gate-a100"
        ),
        "sandbox": "bubblewrap",
        "device": "cuda:1",
        "timeout_seconds": 777.0,
        "network_isolated": True,
        "filesystem_read_only": True,
    }
    assert run_options == {
        "artifact_root": "/local/preflight-artifacts",
        "asset_root": CAPABILITY_STUDY.parent.resolve(),
        "baseline_target_id": assets.targets[0].target_id,
        "live": True,
        "expected_max_worker_jobs_per_invocation": PREFLIGHT_MAX_WORKER_JOBS,
    }
    assert output == {
        "status": "ready",
        "resumed_ready_bundle": False,
        "study_id": pinned.spec.study_id,
        "study_spec_sha256": CAPABILITY_STUDY_SHA256,
        "schedule_sha256": schedule.sha256,
        "asset_manifest_sha256": hashes["assets"],
        "environment_manifest_sha256": hashes["environment"],
        "floor_manifest_sha256": hashes["floor"],
        "execution_context_sha256": hashes["execution_context"],
        "preflight_receipt_sha256": hashes["receipt"],
        "preflight_bundle_sha256": hashes["bundle"],
        "preflight_contract_sha256": hashes["contract"],
        "environment_probe_artifact_sha256": hashes["probe"],
        "protocol_counts": {
            "oracle": 2,
            "baseline": 1,
            "capability": 3,
            "launch": 1,
        },
        "max_worker_jobs_per_invocation": PREFLIGHT_MAX_WORKER_JOBS,
        "preflight_directory": "/sealed/preflight-ready",
    }


@pytest.mark.parametrize(
    ("error", "expected_exit", "message"),
    [
        (
            cli.MatrixPreflightInfrastructureError("remote health failed"),
            cli.EXIT_WORKER,
            "preflight infrastructure error",
        ),
        (
            cli.MatrixPreflightInvalidFloorError("launch floor failed"),
            cli.EXIT_ARTIFACT,
            "preflight invalid floor",
        ),
        (
            cli.MatrixPreflightArtifactError("raw checksum mismatch"),
            cli.EXIT_ARTIFACT,
            "preflight artifact error",
        ),
    ],
)
def test_preflight_study_preserves_runner_failure_category(
    error: Exception,
    expected_exit: int,
    message: str,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "read_clean_controller_revision",
        lambda root: "a" * 40,
    )
    monkeypatch.setattr(
        cli,
        "run_matrix_preflight",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    exit_code = cli.main(_preflight_study_arguments("--live"))

    assert exit_code == expected_exit
    assert message in capsys.readouterr().err


def _run_study_arguments(*extra: str) -> list[str]:
    return [
        "run-study",
        "--study-spec",
        str(CAPABILITY_STUDY),
        "--expected-study-sha256",
        CAPABILITY_STUDY_SHA256,
        "--phase",
        "core",
        "--preflight-directory",
        "/sealed/preflight",
        "--expected-operational-request-ceiling",
        "288",
        *extra,
    ]


def test_run_study_guards_precede_preflight_config_auth_and_network(capsys, monkeypatch) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("side effect occurred before matrix live guard")

    for name in (
        "load_preflight_bundle",
        "load_app_config",
        "load_auth_store",
        "ProviderClient",
        "build_authorized_ssh_worker",
        "run_matrix_phase",
    ):
        monkeypatch.setattr(cli, name, unexpected)

    missing_live = cli.main(_run_study_arguments())
    wrong_ceiling = cli.main(
        [
            *_run_study_arguments("--live"),
            "--expected-operational-request-ceiling",
            "287",
        ]
    )

    error = capsys.readouterr().err
    assert missing_live == cli.EXIT_CONFIG
    assert wrong_ceiling == cli.EXIT_CONFIG
    assert "requires --live" in error
    assert "frozen full-phase ceiling (288)" in error


def test_run_study_rejects_reserve_before_preflight_access(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "load_preflight_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reserve accessed preflight before authorization")
        ),
    )
    arguments = _run_study_arguments("--live")
    arguments[arguments.index("core")] = "reserve"

    exit_code = cli.main(arguments)

    assert exit_code == cli.EXIT_CONFIG
    assert "sealed core analysis authorization" in capsys.readouterr().err


def test_run_study_rejects_wrong_study_hash_before_preflight_access(
    capsys, monkeypatch
) -> None:
    monkeypatch.setattr(
        cli,
        "load_preflight_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("preflight accessed after study hash mismatch")
        ),
    )
    arguments = _run_study_arguments("--live")
    arguments[arguments.index(CAPABILITY_STUDY_SHA256)] = "0" * 64

    exit_code = cli.main(arguments)

    assert exit_code == cli.EXIT_CONFIG
    assert "study manifest SHA-256 mismatch" in capsys.readouterr().err


def test_run_study_dispatches_generic_runtime_without_live_factory_access(
    capsys,
    monkeypatch,
    manifest_bundle: ManifestBundle,
) -> None:
    preflight = object()
    authorization = object()
    observed: dict[str, Any] = {}

    class FakeConfig:
        def bundle(self, profile: str | None = None) -> ManifestBundle:
            observed.setdefault("profiles", []).append(profile)
            return manifest_bundle

    class FakeRuntime:
        def __init__(self, **kwargs: Any) -> None:
            observed["runtime"] = kwargs

        def resolve_task(self, identifier: str) -> str:
            return identifier

        def resolve_target(self, identifier: str) -> str:
            return identifier

        def resolve_agent(self, identifier: str) -> str:
            return identifier

        def resolve_execution(self, cell: object) -> object:
            return cell

        def runtime_for(self, identity: object) -> object:
            raise AssertionError(f"full resume requested a live runtime: {identity}")

    def fake_run(pinned: object, phase: str, **kwargs: Any) -> SimpleNamespace:
        observed["run"] = (pinned, phase, kwargs)
        return SimpleNamespace(status="complete")

    monkeypatch.setattr(cli, "load_preflight_bundle", lambda *args: preflight)
    monkeypatch.setattr(
        cli,
        "runtime_authorization",
        lambda bundle: authorization if bundle is preflight else None,
    )
    monkeypatch.setattr(cli, "load_app_config", lambda path: FakeConfig())
    monkeypatch.setattr(cli, "MatrixStudyRuntime", FakeRuntime)
    monkeypatch.setattr(cli, "run_matrix_phase", fake_run)
    monkeypatch.setattr(cli, "_emit", lambda value: observed.setdefault("emitted", value))
    monkeypatch.setattr(
        cli,
        "load_auth_store",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("auth was loaded")),
    )
    monkeypatch.setattr(
        cli,
        "ProviderClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider was created")),
    )
    monkeypatch.setattr(
        cli,
        "build_authorized_ssh_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("SSH worker was created")),
    )

    exit_code = cli.main(_run_study_arguments("--live"))

    assert capsys.readouterr().err == ""
    assert exit_code == cli.EXIT_OK
    assert observed["profiles"] == ["deepseek-v4-pro"]
    runtime = observed["runtime"]
    assert runtime["authorization"] is authorization
    assert runtime["agent_bundles"] == {"deepseek-v4-pro": manifest_bundle}
    assert runtime["asset_root"] == CAPABILITY_STUDY.parent
    _, phase, run_options = observed["run"]
    assert phase == "core"
    assert run_options["expected_operational_request_ceiling"] == 288
    assert run_options["preflight_directory"] == "/sealed/preflight"
    assert observed["emitted"].status == "complete"


def test_matrix_client_factory_loads_auth_lazily_once(monkeypatch, manifest_bundle) -> None:
    calls: list[object] = []
    environment = {"TEST_API_KEY": "secret"}

    monkeypatch.setattr(
        cli,
        "load_auth_store",
        lambda path, missing_ok: calls.append((path, missing_ok)) or object(),
    )
    monkeypatch.setattr(
        cli,
        "runtime_environment",
        lambda auth, process: calls.append((auth, process)) or environment,
    )
    monkeypatch.setattr(
        cli,
        "ProviderClient",
        lambda bundle, *, environment: calls.append((bundle, environment)) or object(),
    )

    factory = cli._matrix_client_factory("/tmp/auth.json")
    assert calls == []

    factory("agent-a", manifest_bundle)
    factory("agent-b", manifest_bundle)

    assert calls[0] == (Path("/tmp/auth.json"), False)
    assert sum(item == (manifest_bundle, environment) for item in calls) == 2
    assert len(calls) == 4


def test_matrix_summary_exit_code_uses_cumulative_exhausted_provider_failure() -> None:
    summary = SimpleNamespace(
        status="incomplete_infrastructure",
        retry_exhausted_outcome_statuses=("provider_error",),
        records=(),
    )

    assert cli._matrix_summary_exit_code(summary) == cli.EXIT_PROVIDER


def _time_study_arguments(*extra: str, expected: int = 1) -> list[str]:
    return [
        "time-study",
        "--study-spec",
        str(CAPABILITY_STUDY),
        "--expected-study-sha256",
        CAPABILITY_STUDY_SHA256,
        "--phase",
        "core",
        "--preflight-directory",
        "/sealed/preflight",
        "--expected-qualified-candidates",
        str(expected),
        *extra,
    ]


def test_time_study_live_and_count_guards_precede_remote_side_effects(
    capsys,
    monkeypatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("time-study side effect occurred before its local guard")

    for name in (
        "load_study_spec",
        "load_preflight_bundle",
        "build_authorized_ssh_worker",
        "run_matrix_candidate_timing",
    ):
        monkeypatch.setattr(cli, name, unexpected)

    missing_live = cli.main(_time_study_arguments())
    negative_count = cli.main(_time_study_arguments("--live", expected=-1))

    assert missing_live == cli.EXIT_CONFIG
    assert negative_count == cli.EXIT_CONFIG
    error = capsys.readouterr().err
    assert "requires --live" in error
    assert "must be non-negative" in error


def test_time_study_count_guard_precedes_worker_creation(capsys, monkeypatch) -> None:
    preflight = object()
    monkeypatch.setattr(cli, "load_preflight_bundle", lambda *args: preflight)
    monkeypatch.setattr(
        cli,
        "discover_matrix_qualified_candidates",
        lambda **kwargs: (object(),),
    )
    monkeypatch.setattr(
        cli,
        "load_matrix_phase_contract",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("phase contract loaded after candidate count mismatch")
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_authorized_ssh_worker",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("worker created after candidate count mismatch")
        ),
    )

    exit_code = cli.main(_time_study_arguments("--live", expected=0))

    assert exit_code == cli.EXIT_CONFIG
    assert "discovered sealed count (1)" in capsys.readouterr().err


def test_time_study_uses_only_preflight_worker_and_frozen_protocol(
    capsys,
    monkeypatch,
) -> None:
    revision = "a" * 40
    preflight = SimpleNamespace(
        execution_context=SimpleNamespace(
            controller_revision=revision,
        )
    )
    candidate = object()
    contract = object()
    authorization = object()
    manifest = SimpleNamespace(
        timing_study_id="tilelang-capability-gate-a100-v1-core-timing",
        device="cuda:0",
        sha256="b" * 64,
    )
    observed: dict[str, Any] = {}

    class FakeWorker:
        def validate_environment(self, device: str) -> None:
            observed["validated_device"] = device

    record = SimpleNamespace(
        summary=SimpleNamespace(
            job_prefix="timing-cell",
            status="stable",
            stable=True,
            median_ms=1.0,
        )
    )
    monkeypatch.setattr(cli, "load_preflight_bundle", lambda *args: preflight)
    monkeypatch.setattr(
        cli,
        "discover_matrix_qualified_candidates",
        lambda **kwargs: (candidate,),
    )
    monkeypatch.setattr(
        cli,
        "load_matrix_phase_contract",
        lambda *args, **kwargs: contract,
    )
    monkeypatch.setattr(
        cli,
        "build_matrix_timing_study_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(cli, "read_clean_controller_revision", lambda root: revision)
    monkeypatch.setattr(
        cli,
        "runtime_authorization",
        lambda bundle: authorization if bundle is preflight else None,
    )
    monkeypatch.setattr(
        cli,
        "build_authorized_ssh_worker",
        lambda value: FakeWorker() if value is authorization else None,
    )
    monkeypatch.setattr(
        cli,
        "seal_matrix_timing_study_manifest",
        lambda root, value: observed.setdefault("sealed", (root, value)),
    )

    def run_timing(worker: object, **kwargs: Any) -> tuple[object, ...]:
        observed["run"] = (worker, kwargs)
        kwargs["progress"](1, 1, record, False)
        return (record,)

    monkeypatch.setattr(cli, "run_matrix_candidate_timing", run_timing)
    monkeypatch.setattr(cli, "_emit", lambda value: observed.setdefault("summary", value))
    for name in ("load_app_config", "load_auth_store", "ProviderClient"):
        monkeypatch.setattr(
            cli,
            name,
            lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"time-study accessed {_name}")
            ),
        )

    exit_code = cli.main(_time_study_arguments("--live"))

    assert exit_code == cli.EXIT_OK
    assert observed["validated_device"] == "cuda:0"
    assert observed["sealed"][1] is manifest
    assert observed["run"][1]["candidates"] == (candidate,)
    assert observed["summary"]["stable_count"] == 1
    assert "timing-cell" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("cause", "expected_exit"),
    [
        (cli.ConfigurationError("invalid auth"), cli.EXIT_CONFIG),
        (WorkerExecutionError("health_drift", "worker changed"), cli.EXIT_WORKER),
        (
            ProviderCallError(
                NormalizedError(
                    request_id="req-matrix-cli",
                    attempt_id="attempt-matrix-cli",
                    logical_request_sha256="0" * 64,
                    provider_id="fake-provider",
                    model_id="fake-model",
                    category=ErrorCategory.NETWORK,
                    provider_type="FakeNetworkError",
                    sanitized_message="provider unavailable",
                    retryable=True,
                    request_submitted=False,
                    possibly_charged=False,
                    started_at_utc=datetime.now(timezone.utc),
                    failed_at_utc=datetime.now(timezone.utc),
                    elapsed_ms=1.0,
                    sanitized_transport_request={},
                )
            ),
            cli.EXIT_PROVIDER,
        ),
        (RuntimeError("controller failure"), cli.EXIT_ARTIFACT),
    ],
)
def test_matrix_run_error_exit_code_preserves_wrapped_failure_category(
    cause: Exception,
    expected_exit: int,
) -> None:
    try:
        raise cause
    except Exception as error:
        try:
            raise cli.MatrixStudyRunError("wrapped matrix failure") from error
        except cli.MatrixStudyRunError as wrapped:
            assert cli._matrix_run_error_exit_code(wrapped) == expected_exit


def test_run_cell_guards_precede_config_auth_artifacts_and_network(capsys, monkeypatch) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("side effect occurred before live guard")

    monkeypatch.setattr(cli, "load_app_config", unexpected)
    monkeypatch.setattr(cli, "TrajectoryStore", unexpected)
    monkeypatch.setattr(cli, "ProviderClient", unexpected)

    missing_live = cli.main(["run-cell", "--expected-max-requests", "4"])
    wrong_count = cli.main(["run-cell", "--live", "--expected-max-requests", "3"])

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
            "--ssh-port",
            "30554",
            "--worker-root",
            "/srv/AbstraK",
        ]
    )

    worker = cli._worker_executor(arguments)
    record = cli._transport_record(worker)

    assert record["python_executable"] == "/tmp/abstrak-gpu-venv/bin/python"
    assert record["port"] == 30554
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

    exit_code = cli.main(["run-cell", "--live", "--expected-max-requests", "4"])

    assert exit_code == cli.EXIT_CONFIG
    assert "requires --ssh-host" in capsys.readouterr().err


def test_worker_rejects_ssh_port_without_host(capsys) -> None:
    exit_code = cli.main(["run-trusted", "--ssh-port", "30554"])

    assert exit_code == cli.EXIT_CONFIG
    assert "--ssh-port is only valid with --ssh-host" in capsys.readouterr().err


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
    assert worker.jobs[0].case_ids == tuple(case.id for case in worker.jobs[0].task.sealed_cases)
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
