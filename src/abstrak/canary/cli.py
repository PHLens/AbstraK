"""Controller CLI for reusable canary studies and the frozen A100 R1 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from abstrak.canary.artifacts import TrajectoryArtifactError, TrajectoryStore
from abstrak.canary.baselines import BaselineRegistryError
from abstrak.canary.capability_assets import build_capability_asset_manifest
from abstrak.canary.contracts import AgentBudget, TimingSpec, WorkerJob
from abstrak.canary.gates import GateError, run_baseline_gates, run_oracle_gates
from abstrak.canary.loop import CanaryAgentLoop
from abstrak.canary.manifests import PinnedStudySpec, StudyManifestError, load_study_spec
from abstrak.canary.matrix import MatrixSchedule, MatrixSpecError, build_matrix_schedule
from abstrak.canary.matrix_preflight import (
    MatrixPreflightError,
    build_pending_environment,
    load_preflight_bundle,
)
from abstrak.canary.matrix_preflight_runner import (
    MatrixPreflightArtifactError,
    MatrixPreflightInfrastructureError,
    MatrixPreflightInvalidFloorError,
    MatrixPreflightRunnerError,
    preflight_worker_job_ceiling,
    run_matrix_preflight,
)
from abstrak.canary.matrix_runner import (
    MatrixPhaseRunSummary,
    MatrixStudyRunError,
    MatrixTransportContext,
    load_matrix_phase_contract,
    run_matrix_phase,
    validate_matrix_phase_guards,
)
from abstrak.canary.matrix_runtime import (
    MatrixRuntimeError,
    MatrixStudyRuntime,
    build_authorized_ssh_worker,
    read_clean_controller_revision,
    runtime_authorization,
)
from abstrak.canary.matrix_timing import (
    MatrixCandidateTimingRecord,
    MatrixTimingError,
    build_matrix_timing_study_manifest,
    discover_matrix_qualified_candidates,
    run_matrix_candidate_timing,
    seal_matrix_timing_study_manifest,
)
from abstrak.canary.protocol import build_initial_messages
from abstrak.canary.remote import LocalWorkerExecutor, SshWorkerExecutor, WorkerExecutionError
from abstrak.canary.report import (
    DEFAULT_BASELINE_GATE_STUDY_ID,
    DEFAULT_FORMAL_STUDY_ID,
    DEFAULT_ORACLE_GATE_STUDY_ID,
    DEFAULT_REPORT_STUDY_ID,
    DEFAULT_SHAKEOUT_STUDY_ID,
    DEFAULT_TIMING_STUDY_ID,
    AnalysisReportError,
    load_analysis_report,
    write_analysis_report,
)
from abstrak.canary.schedule import R1_TARGETS, R1_TASKS
from abstrak.canary.targets import (
    TargetRegistryError,
    get_target_stack,
    list_target_ids,
    load_target_card,
    validate_target_registry,
)
from abstrak.canary.tasks import (
    DEFAULT_ASSET_ROOT,
    TaskRegistryError,
    get_task_pack,
    list_task_ids,
    load_oracle_source,
    validate_task_registry,
)
from abstrak.canary.worker import main as worker_main
from abstrak.config import (
    AUTH_ENV,
    CONFIG_ENV,
    ConfigurationError,
    default_auth_path,
    default_config_path,
    load_app_config,
    load_auth_store,
    resolve_path,
    runtime_environment,
)
from abstrak.providers.client import ProviderClient, ProviderConfigurationError
from abstrak.providers.contracts import ProviderCallError
from abstrak.providers.manifests import (
    ManifestBundle,
    MissingEnvironmentError,
    ModelManifest,
    required_environment,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "r1-a100-canary"
EXPECTED_MAX_REQUESTS = 4

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_WORKER = 3
EXIT_PROVIDER = 4
EXIT_ARTIFACT = 5


class CanaryCliError(ValueError):
    """Raised for invalid controller command combinations."""


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _emit(value: object) -> None:
    print(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def default_trajectory_id(prefix: str = "cell") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S.%fZ").lower()
    return f"{prefix}-{timestamp}-{uuid4().hex[:10]}"


def _canary_bundle(bundle: ManifestBundle, budget: AgentBudget) -> ManifestBundle:
    model_payload = bundle.model.model_dump(mode="json")
    generation = model_payload["generation"]
    generation.update(
        {
            "max_completion_tokens": budget.max_completion_tokens_per_call,
            "temperature": 0,
            "top_p": None,
            "api_seed": None,
            "stop": [],
            "reasoning_effort": None,
        }
    )
    model_payload.update(
        {
            "allow_live_probe": False,
            "output_contract": "plain_text",
            "generation": generation,
        }
    )
    return ManifestBundle(
        provider=bundle.provider,
        model=ModelManifest.model_validate(model_payload),
        pricing=bundle.pricing,
    )


def _add_registry_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", default="row-reduction-scale", choices=list_task_ids())
    parser.add_argument("--target", default="triton-a100", choices=list_target_ids())
    parser.add_argument(
        "--asset-root",
        default=str(DEFAULT_ASSET_ROOT),
        help="local frozen task, target-card, and oracle assets",
    )


def _add_worker_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ssh-host", help="non-interactive SSH destination for the GPU worker")
    parser.add_argument("--ssh-port", type=int, help="SSH port (default: OpenSSH configuration)")
    parser.add_argument(
        "--worker-root",
        help="AbstraK checkout on the worker; required for SSH and inferred locally",
    )
    parser.add_argument("--worker-python", help="worker Python executable")
    parser.add_argument("--worker-pythonpath", help="worker AbstraK src directory")
    parser.add_argument("--worker-kernelbench-root", help="KernelBench checkout on the worker")
    parser.add_argument("--worker-asset-root", help="frozen benchmark assets on the worker")
    parser.add_argument("--worker-timeout", type=float, default=300.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-supervised-worker",
        action="store_true",
        help=(
            "run SSH jobs as an unprivileged user without network isolation when the "
            "platform forbids bubblewrap"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate frozen canary assets offline")
    validate.add_argument("--asset-root", default=str(DEFAULT_ASSET_ROOT))

    inspect_study = subparsers.add_parser(
        "inspect-study",
        help="inspect and materialize one hash-pinned generic study definition",
    )
    inspect_study.add_argument("--study-spec", required=True)
    inspect_study.add_argument("--expected-study-sha256")

    preflight_study = subparsers.add_parser(
        "preflight-study",
        help="run or resume the trusted environment, floor, canary, and launch gates",
    )
    preflight_study.add_argument("--study-spec", required=True)
    preflight_study.add_argument("--expected-study-sha256", required=True)
    preflight_study.add_argument(
        "--asset-root",
        help="local frozen assets (default: directory containing --study-spec)",
    )
    preflight_study.add_argument(
        "--artifact-root",
        default="artifacts/capability-gate-a100",
    )
    preflight_study.add_argument(
        "--live",
        action="store_true",
        help="acknowledge execution of trusted GPU code under the frozen timing protocol",
    )
    preflight_study.add_argument(
        "--expected-max-jobs",
        type=int,
        required=True,
        help=(
            "must equal the dynamically derived preflight ceiling for each "
            "authorized invocation"
        ),
    )
    _add_worker_options(preflight_study)
    preflight_study.add_argument("--expected-accelerator", required=True)
    preflight_study.add_argument("--expected-compute-capability", required=True)
    preflight_study.add_argument("--expected-python-version", required=True)
    preflight_study.add_argument("--expected-tilelang-version", required=True)
    preflight_study.add_argument("--expected-triton-version", required=True)
    preflight_study.add_argument("--expected-torch-version", required=True)
    preflight_study.add_argument("--expected-cuda-version", required=True)
    preflight_study.add_argument("--expected-driver-version", required=True)
    preflight_study.add_argument("--expected-kernelbench-revision", required=True)

    run_study = subparsers.add_parser(
        "run-study",
        help="run or resume one preflight-authorized generic matrix phase",
    )
    run_study.add_argument("--study-spec", required=True)
    run_study.add_argument("--expected-study-sha256", required=True)
    run_study.add_argument("--phase", required=True)
    run_study.add_argument("--preflight-directory", required=True)
    run_study.add_argument("--asset-root")
    run_study.add_argument("--artifact-root", default="artifacts/capability-gate-a100")
    run_study.add_argument(
        "--config",
        help=f"user config YAML (default: ${CONFIG_ENV} or ~/.abstrak/config.yaml)",
    )
    run_study.add_argument(
        "--auth",
        help=f"credential JSON (default: ${AUTH_ENV} or ~/.abstrak/auth.json)",
    )
    run_study.add_argument(
        "--live",
        action="store_true",
        help="acknowledge billable requests and execution of generated GPU code",
    )
    run_study.add_argument(
        "--expected-operational-request-ceiling",
        type=int,
        required=True,
        help="must equal the frozen full-phase ceiling, including one infrastructure retry",
    )

    time_study = subparsers.add_parser(
        "time-study",
        help="run or resume preflight-authorized timing for one terminal matrix phase",
    )
    time_study.add_argument("--study-spec", required=True)
    time_study.add_argument("--expected-study-sha256", required=True)
    time_study.add_argument("--phase", required=True)
    time_study.add_argument("--preflight-directory", required=True)
    time_study.add_argument("--artifact-root", default="artifacts/capability-gate-a100")
    time_study.add_argument("--timing-study-id")
    time_study.add_argument("--expected-qualified-candidates", type=int, required=True)
    time_study.add_argument(
        "--live",
        action="store_true",
        help="acknowledge execution of generated GPU code under the frozen timing protocol",
    )

    subparsers.add_parser("worker", help="run one JSON worker job or GPU health check")

    trusted = subparsers.add_parser(
        "run-trusted", help="run a registered expert canary without a provider request"
    )
    _add_registry_options(trusted)
    _add_worker_options(trusted)
    trusted.add_argument("--job-id", help="safe immutable job identifier")
    trusted.add_argument(
        "--artifact-root",
        default="artifacts/r1-a100",
        help="ignored-by-Git root for the immutable trusted-run bundle",
    )
    trusted.add_argument(
        "--timing",
        action="store_true",
        help="also collect one process-local timing sample set",
    )

    run_cell = subparsers.add_parser(
        "run-cell", help="run one fixed four-call provider/worker canary trajectory"
    )
    _add_registry_options(run_cell)
    _add_worker_options(run_cell)
    run_cell.add_argument(
        "--config",
        help=f"user config YAML (default: ${CONFIG_ENV} or ~/.abstrak/config.yaml)",
    )
    run_cell.add_argument("--profile", help="model profile from the user config")
    run_cell.add_argument(
        "--auth",
        help=f"credential JSON (default: ${AUTH_ENV} or ~/.abstrak/auth.json)",
    )
    run_cell.add_argument("--trajectory-id", help="safe immutable trajectory identifier")
    run_cell.add_argument(
        "--study-id",
        default=STUDY_ID,
        help="safe immutable study directory identifier",
    )
    run_cell.add_argument(
        "--artifact-root",
        default="artifacts/r1-a100",
        help="ignored-by-Git root for private trajectory artifacts",
    )
    run_cell.add_argument(
        "--live",
        action="store_true",
        help="acknowledge billable requests and execution of generated GPU code",
    )
    run_cell.add_argument(
        "--expected-max-requests",
        type=int,
        required=True,
        help=f"must equal the fixed request ceiling ({EXPECTED_MAX_REQUESTS})",
    )

    gates = subparsers.add_parser(
        "run-gates", help="run or resume the formal expert-oracle or B* timing gates"
    )
    _add_worker_options(gates)
    gates.add_argument("--gate-kind", choices=("oracle", "baseline"), required=True)
    gates.add_argument("--artifact-root", default="artifacts/r1-a100")
    gates.add_argument("--study-id", help="sealed gate study directory")
    gates.add_argument(
        "--live",
        action="store_true",
        help="acknowledge execution of trusted GPU code and baseline code",
    )
    gates.add_argument(
        "--expected-max-jobs",
        type=int,
        required=True,
        help="must equal 72 (12 pairs x 3 processes x one complete retry)",
    )
    gates.add_argument("--asset-root", default=str(DEFAULT_ASSET_ROOT))
    gates.set_defaults(target="triton-a100")

    analyze = subparsers.add_parser(
        "analyze-study", help="build or resume the sealed preregistered R1 report"
    )
    analyze.add_argument("--artifact-root", default="artifacts/r1-a100")
    analyze.add_argument("--formal-study-id", default=DEFAULT_FORMAL_STUDY_ID)
    analyze.add_argument("--oracle-gate-study-id", default=DEFAULT_ORACLE_GATE_STUDY_ID)
    analyze.add_argument("--baseline-gate-study-id", default=DEFAULT_BASELINE_GATE_STUDY_ID)
    analyze.add_argument("--timing-study-id", default=DEFAULT_TIMING_STUDY_ID)
    analyze.add_argument("--shakeout-study-id", default=DEFAULT_SHAKEOUT_STUDY_ID)
    analyze.add_argument("--report-study-id", default=DEFAULT_REPORT_STUDY_ID)
    return parser


def _worker_executor(arguments: argparse.Namespace) -> LocalWorkerExecutor | SshWorkerExecutor:
    timeout = arguments.worker_timeout
    if timeout <= 0:
        raise CanaryCliError("--worker-timeout must be positive")

    target = get_target_stack(arguments.target)
    if arguments.ssh_host:
        if not arguments.worker_root:
            raise CanaryCliError("--worker-root is required with --ssh-host")
        root = PurePosixPath(arguments.worker_root)
        python_executable = arguments.worker_python or "/tmp/abstrak-gpu-venv/bin/python"
        pythonpath = arguments.worker_pythonpath or str(root / "src")
        kernelbench_root = arguments.worker_kernelbench_root or str(root.parent / "KernelBench")
        asset_root = arguments.worker_asset_root or str(root / "benchmarks" / "r1-a100")
        return SshWorkerExecutor(
            arguments.ssh_host,
            port=arguments.ssh_port,
            python_executable=python_executable,
            pythonpath=pythonpath,
            kernelbench_root=kernelbench_root,
            asset_root=asset_root,
            device=arguments.device,
            timeout_seconds=timeout,
            expected_hardware_substring="A100",
            expected_compute_capability=(8, 0),
            expected_triton_version=target.version if target.backend == "triton" else None,
            expected_tilelang_version=(
                target.version if target.adapter.startswith("tilelang-capability-") else None
            ),
            sandbox_mode=("setpriv" if arguments.allow_supervised_worker else "bubblewrap"),
        )

    if arguments.worker_pythonpath is not None:
        raise CanaryCliError("--worker-pythonpath is only valid with --ssh-host")
    if arguments.ssh_port is not None:
        raise CanaryCliError("--ssh-port is only valid with --ssh-host")
    if arguments.allow_supervised_worker:
        raise CanaryCliError("--allow-supervised-worker is only valid with --ssh-host")
    if arguments.command == "run-cell":
        raise CanaryCliError("run-cell requires --ssh-host for the remote bwrap sandbox")
    root = Path(arguments.worker_root).expanduser() if arguments.worker_root else REPOSITORY_ROOT
    kernelbench_root = (
        Path(arguments.worker_kernelbench_root).expanduser()
        if arguments.worker_kernelbench_root
        else Path(os.environ.get("KERNELBENCH_ROOT", root.parent / "KernelBench")).expanduser()
    )
    asset_root = (
        Path(arguments.worker_asset_root).expanduser()
        if arguments.worker_asset_root
        else Path(arguments.asset_root).expanduser()
    )
    return LocalWorkerExecutor(
        kernelbench_root,
        asset_root=asset_root,
        python_executable=arguments.worker_python or sys.executable,
        timeout_seconds=timeout,
        expected_hardware_substring="A100",
        expected_compute_capability=(8, 0),
        expected_triton_version=target.version if target.backend == "triton" else None,
        expected_tilelang_version=(
            target.version if target.adapter.startswith("tilelang-capability-") else None
        ),
    )


def _transport_record(
    worker: LocalWorkerExecutor | SshWorkerExecutor,
) -> dict[str, object]:
    if isinstance(worker, SshWorkerExecutor):
        supervised = worker.sandbox_mode == "setpriv"
        record: dict[str, object] = {
            "kind": "ssh",
            "host": worker.host,
            "ssh_executable": worker.ssh_executable,
            "remote_timeout_executable": worker.remote_timeout_executable,
            "sandbox": "setpriv-supervised" if supervised else "bubblewrap",
            "sandbox_executable": worker.sandbox_executable,
            "sandbox_user": worker.sandbox_user if supervised else None,
            "network_isolated": not supervised,
            "filesystem_read_only": not supervised,
            "low_privilege": supervised,
            "python_executable": worker.python_executable,
            "pythonpath": worker.pythonpath,
            "kernelbench_root": worker.kernelbench_root,
            "asset_root": worker.asset_root,
            "device": worker.device,
            "timeout_seconds": worker.timeout_seconds,
            "expected_hardware_substring": worker.expected_hardware_substring,
            "expected_compute_capability": worker.expected_compute_capability,
            "expected_triton_version": worker.expected_triton_version,
        }
        if worker.port is not None:
            record["port"] = worker.port
        tilelang_version = getattr(worker, "expected_tilelang_version", None)
        if tilelang_version is not None:
            record["expected_tilelang_version"] = tilelang_version
        return record
    record = {
        "kind": "local",
        "python_executable": worker.python_executable,
        "kernelbench_root": worker.kernelbench_root,
        "asset_root": worker.asset_root,
        "timeout_seconds": worker.timeout_seconds,
        "expected_hardware_substring": worker.expected_hardware_substring,
        "expected_compute_capability": worker.expected_compute_capability,
        "expected_triton_version": worker.expected_triton_version,
    }
    tilelang_version = getattr(worker, "expected_tilelang_version", None)
    if tilelang_version is not None:
        record["expected_tilelang_version"] = tilelang_version
    return record


def _validate(arguments: argparse.Namespace) -> int:
    validate_task_registry(asset_root=arguments.asset_root)
    validate_target_registry(asset_root=arguments.asset_root)
    pairs: list[dict[str, str]] = []
    for task_id in list_task_ids():
        for target_id in list_target_ids():
            target = get_target_stack(target_id)
            source = load_oracle_source(task_id, target.backend, asset_root=arguments.asset_root)
            pairs.append(
                {
                    "task_id": task_id,
                    "target_id": target_id,
                    "oracle_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                }
            )
    _emit(
        {
            "status": "valid",
            "study_id": STUDY_ID,
            "asset_root": str(Path(arguments.asset_root).expanduser().resolve()),
            "tasks": list_task_ids(),
            "targets": list_target_ids(),
            "trusted_pairs": pairs,
        }
    )
    return EXIT_OK


def _inspect_study(arguments: argparse.Namespace) -> int:
    pinned = load_study_spec(
        arguments.study_spec,
        expected_sha256=arguments.expected_study_sha256,
    )
    schedule = build_matrix_schedule(pinned.spec)
    _emit(
        {
            "status": "structurally_valid",
            "assets_validated": False,
            "study_id": pinned.spec.study_id,
            "study_spec_path": str(pinned.path),
            "study_spec_sha256": pinned.sha256,
            "schedule_sha256": schedule.sha256,
            "expected_trajectories": schedule.expected_trajectories,
            "request_ceiling": schedule.request_ceiling,
            "operational_request_ceiling": schedule.operational_request_ceiling,
            "agents": pinned.spec.agents,
            "targets": pinned.spec.targets,
            "phases": [
                {
                    "id": phase.id,
                    "task_ids": phase.task_ids,
                    "replicates": phase.replicates,
                    "expected_trajectories": pinned.spec.phase_trajectory_count(phase.id),
                    "request_ceiling": pinned.spec.phase_request_ceiling(phase.id),
                    "operational_request_ceiling": (
                        pinned.spec.phase_operational_request_ceiling(phase.id)
                    ),
                }
                for phase in pinned.spec.phases
            ],
        }
    )
    return EXIT_OK


def _preflight_transport(
    arguments: argparse.Namespace,
) -> MatrixTransportContext:
    if not arguments.ssh_host:
        raise CanaryCliError("preflight-study requires --ssh-host")
    if not arguments.worker_root:
        raise CanaryCliError("preflight-study requires --worker-root")
    if arguments.worker_timeout <= 0:
        raise CanaryCliError("--worker-timeout must be positive")
    worker_root = PurePosixPath(arguments.worker_root)
    if not worker_root.is_absolute():
        raise CanaryCliError("--worker-root must be an absolute remote path")
    python_executable = arguments.worker_python or "/tmp/abstrak-gpu-venv/bin/python"
    pythonpath = arguments.worker_pythonpath or str(worker_root / "src")
    if not PurePosixPath(python_executable).is_absolute():
        raise CanaryCliError("--worker-python must be an absolute remote path")
    pythonpath_value = PurePosixPath(pythonpath)
    if (
        not pythonpath_value.is_absolute()
        or pythonpath_value.parent != worker_root
    ):
        raise CanaryCliError(
            "--worker-pythonpath must be the worker checkout's direct src path"
        )
    kernelbench_root = arguments.worker_kernelbench_root or str(
        worker_root.parent / "KernelBench"
    )
    worker_asset_root = arguments.worker_asset_root or str(
        worker_root / "benchmarks" / "capability-gate-a100"
    )
    if not PurePosixPath(kernelbench_root).is_absolute():
        raise CanaryCliError(
            "--worker-kernelbench-root must be an absolute remote path"
        )
    if not PurePosixPath(worker_asset_root).is_absolute():
        raise CanaryCliError(
            "--worker-asset-root must be an absolute remote path"
        )
    supervised = arguments.allow_supervised_worker
    return MatrixTransportContext(
        host=arguments.ssh_host,
        port=arguments.ssh_port,
        worker_root=str(worker_root),
        python_executable=python_executable,
        pythonpath=str(pythonpath_value),
        kernelbench_root=kernelbench_root,
        asset_root=worker_asset_root,
        sandbox="setpriv-supervised" if supervised else "bubblewrap",
        device=arguments.device,
        timeout_seconds=arguments.worker_timeout,
        network_isolated=not supervised,
        filesystem_read_only=not supervised,
    )


def _run_preflight_study(arguments: argparse.Namespace) -> int:
    if arguments.live is not True:
        raise CanaryCliError(
            "preflight-study requires --live because it executes trusted GPU code"
        )
    pinned = load_study_spec(
        arguments.study_spec,
        expected_sha256=arguments.expected_study_sha256,
    )
    schedule = build_matrix_schedule(pinned.spec)
    asset_root = (
        Path(arguments.asset_root).expanduser().resolve()
        if arguments.asset_root is not None
        else pinned.path.parent.resolve()
    )
    assets = build_capability_asset_manifest(
        pinned,
        schedule,
        asset_root=asset_root,
    )
    baseline_target_id = assets.targets[0].target_id
    frozen_ceiling = preflight_worker_job_ceiling(
        assets,
        baseline_target_id=baseline_target_id,
    )
    if arguments.expected_max_jobs != frozen_ceiling:
        raise CanaryCliError(
            "--expected-max-jobs must equal the frozen preflight ceiling "
            f"({frozen_ceiling})"
        )
    controller_revision = read_clean_controller_revision(REPOSITORY_ROOT)
    pending_environment = build_pending_environment(
        pinned,
        schedule,
        controller_revision=controller_revision,
        worker_revision=controller_revision,
        transport=_preflight_transport(arguments),
        accelerator=arguments.expected_accelerator,
        compute_capability=arguments.expected_compute_capability,
        python_version=arguments.expected_python_version,
        tilelang_version=arguments.expected_tilelang_version,
        triton_version=arguments.expected_triton_version,
        torch_version=arguments.expected_torch_version,
        cuda_version=arguments.expected_cuda_version,
        driver_version=arguments.expected_driver_version,
        kernelbench_revision=arguments.expected_kernelbench_revision,
    )
    result = run_matrix_preflight(
        pinned,
        schedule,
        assets,
        pending_environment,
        artifact_root=arguments.artifact_root,
        asset_root=asset_root,
        baseline_target_id=baseline_target_id,
        live=arguments.live,
        expected_max_worker_jobs_per_invocation=arguments.expected_max_jobs,
    )
    protocol_counts = {
        kind: sum(
            item.kind == kind for item in result.contract.protocols
        )
        for kind in ("oracle", "baseline", "capability", "launch")
    }
    _emit(
        {
            "status": "ready",
            "resumed_ready_bundle": result.resumed_ready_bundle,
            "study_id": pinned.spec.study_id,
            "study_spec_sha256": pinned.sha256,
            "schedule_sha256": schedule.sha256,
            "asset_manifest_sha256": result.bundle.assets.sha256,
            "environment_manifest_sha256": result.bundle.environment.sha256,
            "floor_manifest_sha256": result.bundle.floor.sha256,
            "execution_context_sha256": result.bundle.execution_context.sha256,
            "preflight_receipt_sha256": result.bundle.receipt.sha256,
            "preflight_bundle_sha256": result.bundle.sha256,
            "preflight_contract_sha256": result.contract.sha256,
            "environment_probe_artifact_sha256": (
                result.environment_probe.sha256
            ),
            "protocol_counts": protocol_counts,
            "max_worker_jobs_per_invocation": (
                result.contract.max_worker_jobs_per_invocation
            ),
            "preflight_directory": str(result.preflight_directory),
        }
    )
    return EXIT_OK


def _guard_matrix_phase(
    arguments: argparse.Namespace,
    *,
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
) -> None:
    try:
        validate_matrix_phase_guards(
            pinned,
            arguments.phase,
            live=arguments.live,
            expected_operational_request_ceiling=(
                arguments.expected_operational_request_ceiling
            ),
            schedule=schedule,
        )
    except MatrixStudyRunError as error:
        raise CanaryCliError(str(error)) from error
    gate = pinned.spec.gate
    if gate is not None and arguments.phase == gate.reserve_phase_id:
        raise CanaryCliError(
            "reserve phase execution requires a sealed core analysis authorization"
        )


def _matrix_client_factory(auth_argument: str | None):
    environment: dict[str, str] | None = None
    environment_loaded = False

    def create(_agent_id: str, bundle: ManifestBundle) -> ProviderClient:
        nonlocal environment, environment_loaded
        if not environment_loaded:
            auth_path, configured = resolve_path(
                auth_argument,
                environment_name=AUTH_ENV,
                default=default_auth_path(),
            )
            auth = load_auth_store(auth_path, missing_ok=not configured)
            environment = runtime_environment(auth, os.environ)
            environment_loaded = True
        assert environment is not None
        return ProviderClient(bundle, environment=environment)

    return create


def _run_study(arguments: argparse.Namespace) -> int:
    if arguments.live is not True:
        raise CanaryCliError(
            "run-study requires --live because it performs billable requests and executes "
            "generated GPU code"
        )
    pinned = load_study_spec(
        arguments.study_spec,
        expected_sha256=arguments.expected_study_sha256,
    )
    schedule = build_matrix_schedule(pinned.spec)
    _guard_matrix_phase(arguments, pinned=pinned, schedule=schedule)

    bundle = load_preflight_bundle(arguments.preflight_directory, pinned, schedule)
    authorization = runtime_authorization(bundle)
    config = load_app_config(_config_path(arguments.config))
    agent_bundles = {
        agent_id: config.bundle(agent_id) for agent_id in pinned.spec.agents
    }
    asset_root = (
        Path(arguments.asset_root).expanduser()
        if arguments.asset_root is not None
        else pinned.path.parent
    )
    runtime = MatrixStudyRuntime(
        pinned=pinned,
        schedule=schedule,
        authorization=authorization,
        agent_bundles=agent_bundles,
        client_factory=_matrix_client_factory(arguments.auth),
        worker_factory=lambda: build_authorized_ssh_worker(authorization),
        controller_root=REPOSITORY_ROOT,
        asset_root=asset_root,
    )
    summary = run_matrix_phase(
        pinned,
        arguments.phase,
        artifact_root=arguments.artifact_root,
        preflight_directory=arguments.preflight_directory,
        live=arguments.live,
        expected_operational_request_ceiling=(
            arguments.expected_operational_request_ceiling
        ),
        resolve_task=runtime.resolve_task,
        resolve_target=runtime.resolve_target,
        resolve_agent=runtime.resolve_agent,
        resolve_execution=runtime.resolve_execution,
        runtime_factory=runtime.runtime_for,
        schedule=schedule,
    )
    _emit(summary)
    return _matrix_summary_exit_code(summary)


def _matrix_summary_exit_code(summary: MatrixPhaseRunSummary) -> int:
    """Return a stable category even when exhausted failures predate this resume."""

    if summary.status == "complete":
        return EXIT_OK
    if (
        summary.status == "incomplete_infrastructure"
        and "provider_error" in summary.retry_exhausted_outcome_statuses
    ):
        return EXIT_PROVIDER
    if summary.records and summary.records[-1].outcome_status == "provider_error":
        return EXIT_PROVIDER
    return EXIT_WORKER


def _matrix_run_error_exit_code(error: MatrixStudyRunError) -> int:
    """Preserve the actionable failure category hidden by runner context wrapping."""

    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, WorkerExecutionError):
            return EXIT_WORKER
        if isinstance(cause, ProviderCallError):
            return EXIT_PROVIDER
        if isinstance(
            cause,
            (
                CanaryCliError,
                ConfigurationError,
                MissingEnvironmentError,
                ProviderConfigurationError,
                MatrixRuntimeError,
            ),
        ):
            return EXIT_CONFIG
        cause = cause.__cause__
    return EXIT_ARTIFACT


def _run_time_study(arguments: argparse.Namespace) -> int:
    if arguments.live is not True:
        raise CanaryCliError(
            "time-study requires --live because it executes generated GPU code"
        )
    if arguments.expected_qualified_candidates < 0:
        raise CanaryCliError("--expected-qualified-candidates must be non-negative")
    pinned = load_study_spec(
        arguments.study_spec,
        expected_sha256=arguments.expected_study_sha256,
    )
    schedule = build_matrix_schedule(pinned.spec)
    try:
        schedule.cells_for_phase(arguments.phase)
    except ValueError as error:
        raise CanaryCliError(f"unknown matrix phase: {arguments.phase}") from error
    gate = pinned.spec.gate
    if gate is not None and arguments.phase == gate.reserve_phase_id:
        raise CanaryCliError(
            "reserve phase timing requires a sealed core analysis authorization"
        )
    timing_study_id = (
        arguments.timing_study_id
        or f"{pinned.spec.study_id}-{arguments.phase}-timing"
    )
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", timing_study_id) is None:
        raise CanaryCliError("--timing-study-id must be a normalized identifier")

    preflight = load_preflight_bundle(
        arguments.preflight_directory,
        pinned,
        schedule,
    )
    candidates = discover_matrix_qualified_candidates(
        artifact_root=arguments.artifact_root,
        pinned=pinned,
        phase_id=arguments.phase,
        preflight=preflight,
        schedule=schedule,
    )
    if arguments.expected_qualified_candidates != len(candidates):
        raise CanaryCliError(
            "--expected-qualified-candidates must equal the discovered sealed count "
            f"({len(candidates)})"
        )
    contract = load_matrix_phase_contract(
        arguments.artifact_root,
        pinned,
        arguments.phase,
        execution_context=preflight.execution_context,
        schedule=schedule,
    )
    manifest = build_matrix_timing_study_manifest(
        pinned,
        schedule,
        arguments.phase,
        preflight=preflight,
        contract=contract,
        candidates=candidates,
        timing_study_id=timing_study_id,
    )
    authorization = runtime_authorization(preflight)
    controller_revision = read_clean_controller_revision(REPOSITORY_ROOT)
    if controller_revision != preflight.execution_context.controller_revision:
        raise MatrixRuntimeError(
            "controller revision differs from the preflight environment"
        )
    seal_matrix_timing_study_manifest(arguments.artifact_root, manifest)
    worker = build_authorized_ssh_worker(authorization)
    worker.validate_environment(manifest.device)

    def progress(
        index: int,
        total: int,
        record: MatrixCandidateTimingRecord,
        resumed: bool,
    ) -> None:
        print(
            json.dumps(
                {
                    "progress": f"{index}/{total}",
                    "artifact_id": record.summary.job_prefix,
                    "status": record.summary.status,
                    "stable": record.summary.stable,
                    "median_ms": record.summary.median_ms,
                    "resumed": resumed,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    records = run_matrix_candidate_timing(
        worker,
        artifact_root=arguments.artifact_root,
        manifest=manifest,
        candidates=candidates,
        progress=progress,
    )
    _emit(
        {
            "status": "complete",
            "timing_study_id": timing_study_id,
            "timing_study_manifest_sha256": manifest.sha256,
            "candidate_count": len(records),
            "stable_count": sum(record.summary.stable for record in records),
            "missing_count": sum(not record.summary.stable for record in records),
        }
    )
    return EXIT_OK


def _run_trusted(arguments: argparse.Namespace) -> int:
    task = get_task_pack(arguments.task)
    target = get_target_stack(arguments.target)
    source = load_oracle_source(task.id, target.backend, asset_root=arguments.asset_root)
    timing = TimingSpec(repetitions=1) if arguments.timing else None
    job_id = arguments.job_id or default_trajectory_id("trusted")
    job = WorkerJob(
        job_id=job_id,
        kind="oracle",
        task=task,
        target=target,
        case_ids=tuple(case.id for case in task.sealed_cases),
        candidate_source=source,
        candidate_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        timing=timing,
        device=arguments.device,
    )
    worker = _worker_executor(arguments)
    store = TrajectoryStore.create(
        arguments.artifact_root,
        "r1-a100-trusted",
        job_id,
    )
    store.write_json(
        "run-manifest.json",
        {
            "schema_version": "canary-trusted-run-manifest.v1",
            "study_id": "r1-a100-trusted",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "job": job,
            "transport": _transport_record(worker),
            "timing_scope": "single_process" if timing is not None else None,
        },
    )
    try:
        result = worker.execute(job)
    except WorkerExecutionError as error:
        store.write_json(
            "worker-error.json",
            {
                "category": error.category,
                "returncode": error.returncode,
                "error": str(error),
                "post_job_gpu_health": error.health,
            },
        )
        store.seal()
        print(
            f"worker error: {error}; artifact directory: {store.run_directory}",
            file=sys.stderr,
        )
        return EXIT_WORKER
    except Exception as error:
        store.write_json(
            "controller-error.json",
            {"error_type": type(error).__name__},
        )
        store.seal()
        raise
    store.write_json("worker-result.json", result)
    store.seal()
    _emit(
        {
            "status": "complete" if result.status == "completed" else "failed",
            "job": job,
            "result": result,
            "transport": _transport_record(worker),
            "timing_scope": "single_process" if timing is not None else None,
            "artifact_directory": str(store.run_directory),
        }
    )
    return EXIT_OK if result.status == "completed" else EXIT_WORKER


def _config_path(explicit: str | None) -> Path:
    path, _ = resolve_path(explicit, environment_name=CONFIG_ENV, default=default_config_path())
    return path


def _run_cell(arguments: argparse.Namespace) -> int:
    # These acknowledgements intentionally precede config, auth, artifact, SSH, and API access.
    if not arguments.live:
        raise CanaryCliError(
            "run-cell requires --live because it performs billable requests and executes "
            "generated GPU code"
        )
    if arguments.expected_max_requests != EXPECTED_MAX_REQUESTS:
        raise CanaryCliError(
            f"--expected-max-requests must equal the fixed request ceiling "
            f"({EXPECTED_MAX_REQUESTS})"
        )

    budget = AgentBudget(max_calls=EXPECTED_MAX_REQUESTS)
    task = get_task_pack(arguments.task)
    target = get_target_stack(arguments.target)
    target_card = load_target_card(arguments.target, asset_root=arguments.asset_root)
    trajectory_id = arguments.trajectory_id or default_trajectory_id()
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", trajectory_id) is None:
        raise CanaryCliError("--trajectory-id must be one safe lowercase identifier")
    worker = _worker_executor(arguments)
    bundle = _canary_bundle(
        load_app_config(_config_path(arguments.config)).bundle(arguments.profile), budget
    )
    auth_path, configured = resolve_path(
        arguments.auth,
        environment_name=AUTH_ENV,
        default=default_auth_path(),
    )
    auth = load_auth_store(auth_path, missing_ok=not configured)
    environment = runtime_environment(auth, os.environ)
    client = ProviderClient(bundle, environment=environment)
    secret_values = tuple(
        sorted(
            {
                environment[name]
                for name in required_environment(bundle.provider)
                if environment.get(name)
            }
        )
    )
    store = TrajectoryStore.create(
        arguments.artifact_root,
        arguments.study_id,
        trajectory_id,
        secrets=secret_values,
    )
    messages = build_initial_messages(task, target_card)
    store.write_json(
        "run-manifest.json",
        {
            "schema_version": "canary-run-manifest.v1",
            "study_id": arguments.study_id,
            "trajectory_id": trajectory_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "target": target,
            "budget": budget,
            "device": arguments.device,
            "controller_asset_root": str(Path(arguments.asset_root).expanduser().resolve()),
            "transport": _transport_record(worker),
            "resolved_provider": client.resolved_manifest_record,
            "initial_messages": messages,
        },
    )
    try:
        outcome = CanaryAgentLoop(client=client, worker=worker, store=store).run(
            trajectory_id=trajectory_id,
            model_ref=bundle.model.id,
            initial_messages=messages,
            task=task,
            target=target,
            budget=budget,
            device=arguments.device,
        )
    except Exception as error:
        store.write_json(
            "controller-error.json",
            {"error_type": type(error).__name__},
        )
        store.seal()
        raise
    _emit(
        {
            "status": outcome.status,
            "trajectory_id": trajectory_id,
            "calls": outcome.calls,
            "artifact_directory": str(store.run_directory),
            "outcome": outcome,
        }
    )
    if outcome.status == "provider_error":
        return EXIT_PROVIDER
    if outcome.status == "worker_error":
        return EXIT_WORKER
    return EXIT_OK


def _run_gates(arguments: argparse.Namespace) -> int:
    if not arguments.live:
        raise CanaryCliError(
            "run-gates requires --live because it executes trusted or baseline GPU code"
        )
    if arguments.expected_max_jobs != 72:
        raise CanaryCliError("--expected-max-jobs must equal the fixed gate ceiling (72)")
    validate_task_registry(asset_root=arguments.asset_root)
    validate_target_registry(asset_root=arguments.asset_root)
    worker = _worker_executor(arguments)
    tasks = tuple(get_task_pack(task_id) for task_id in R1_TASKS)
    targets = tuple(get_target_stack(target_id) for target_id in R1_TARGETS)
    if arguments.gate_kind == "oracle":
        records = run_oracle_gates(
            worker,
            tasks=tasks,
            targets=targets,
            root=arguments.artifact_root,
            study_id=arguments.study_id or "r1-a100-oracle-gates",
            asset_root=arguments.asset_root,
            device=arguments.device,
        )
    else:
        records = run_baseline_gates(
            worker,
            tasks=tasks,
            target=get_target_stack("triton-a100"),
            root=arguments.artifact_root,
            study_id=arguments.study_id or "r1-a100-baseline-gates",
            device=arguments.device,
        )
    _emit(
        {
            "status": "complete",
            "kind": arguments.gate_kind,
            "records": [
                {
                    "task_id": record.task_id,
                    "target_id": record.target_id,
                    "variant": record.variant,
                    "timing_status": record.summary.status,
                    "stable": record.summary.stable,
                    "median_ms": record.summary.median_ms,
                    "artifact_directory": record.artifact_directory,
                }
                for record in records
            ],
            "transport": _transport_record(worker),
        }
    )
    return EXIT_OK


def _run_analysis(arguments: argparse.Namespace) -> int:
    report = load_analysis_report(
        artifact_root=arguments.artifact_root,
        formal_study_id=arguments.formal_study_id,
        oracle_gate_study_id=arguments.oracle_gate_study_id,
        baseline_gate_study_id=arguments.baseline_gate_study_id,
        timing_study_id=arguments.timing_study_id,
        shakeout_study_id=arguments.shakeout_study_id,
    )
    directory, resumed = write_analysis_report(
        report,
        artifact_root=arguments.artifact_root,
        report_study_id=arguments.report_study_id,
    )
    _emit(
        {
            "status": "complete",
            "outcome": report.analysis.outcome,
            "received_trajectories": report.formal_coverage.received_trajectories,
            "qualified_at_first": report.formal_coverage.qualified_at_first,
            "qualified_at_final": report.formal_coverage.qualified_at_final,
            "infrastructure_censored": report.formal_coverage.infrastructure_censored,
            "timing_final_status_counts": report.timing_coverage.final_status_counts,
            "artifact_directory": str(directory),
            "resumed": resumed,
        }
    )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "worker":
        return worker_main(values[1:])

    parser = _parser()
    arguments = parser.parse_args(values)
    try:
        if arguments.command == "validate":
            return _validate(arguments)
        if arguments.command == "inspect-study":
            return _inspect_study(arguments)
        if arguments.command == "preflight-study":
            return _run_preflight_study(arguments)
        if arguments.command == "run-study":
            return _run_study(arguments)
        if arguments.command == "time-study":
            return _run_time_study(arguments)
        if arguments.command == "run-trusted":
            return _run_trusted(arguments)
        if arguments.command == "run-gates":
            return _run_gates(arguments)
        if arguments.command == "analyze-study":
            return _run_analysis(arguments)
        return _run_cell(arguments)
    except MatrixPreflightInfrastructureError as error:
        print(f"preflight infrastructure error: {error}", file=sys.stderr)
        return EXIT_WORKER
    except MatrixPreflightArtifactError as error:
        print(f"preflight artifact error: {error}", file=sys.stderr)
        return EXIT_ARTIFACT
    except MatrixPreflightInvalidFloorError as error:
        print(f"preflight invalid floor: {error}", file=sys.stderr)
        return EXIT_ARTIFACT
    except (
        CanaryCliError,
        ConfigurationError,
        MissingEnvironmentError,
        ProviderConfigurationError,
        TargetRegistryError,
        TaskRegistryError,
        BaselineRegistryError,
        GateError,
        AnalysisReportError,
        MatrixPreflightError,
        MatrixRuntimeError,
        MatrixPreflightRunnerError,
        MatrixSpecError,
        StudyManifestError,
        ValidationError,
    ) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG
    except WorkerExecutionError as error:
        print(f"worker error: {error}", file=sys.stderr)
        return EXIT_WORKER
    except MatrixStudyRunError as error:
        print(f"matrix run error: {error}", file=sys.stderr)
        return _matrix_run_error_exit_code(error)
    except MatrixTimingError as error:
        print(f"matrix timing error: {error}", file=sys.stderr)
        return EXIT_ARTIFACT
    except TrajectoryArtifactError as error:
        print(f"artifact error: {error}", file=sys.stderr)
        return EXIT_ARTIFACT
    except OSError as error:
        print(f"controller error: {error}", file=sys.stderr)
        return EXIT_ARTIFACT
    except Exception as error:
        print(f"controller error: {type(error).__name__}", file=sys.stderr)
        return EXIT_ARTIFACT


if __name__ == "__main__":
    raise SystemExit(main())
