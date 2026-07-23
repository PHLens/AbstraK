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
from abstrak.canary.contracts import AgentBudget, TimingSpec, WorkerJob
from abstrak.canary.gates import GateError, run_baseline_gates, run_oracle_gates
from abstrak.canary.loop import CanaryAgentLoop
from abstrak.canary.manifests import StudyManifestError, load_study_spec
from abstrak.canary.matrix import MatrixSpecError, build_matrix_schedule
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
            python_executable=python_executable,
            pythonpath=pythonpath,
            kernelbench_root=kernelbench_root,
            asset_root=asset_root,
            device=arguments.device,
            timeout_seconds=timeout,
            expected_hardware_substring="A100",
            expected_compute_capability=(8, 0),
            expected_triton_version=target.version if target.backend == "triton" else None,
            sandbox_mode=("setpriv" if arguments.allow_supervised_worker else "bubblewrap"),
        )

    if arguments.worker_pythonpath is not None:
        raise CanaryCliError("--worker-pythonpath is only valid with --ssh-host")
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
    )


def _transport_record(
    worker: LocalWorkerExecutor | SshWorkerExecutor,
) -> dict[str, object]:
    if isinstance(worker, SshWorkerExecutor):
        supervised = worker.sandbox_mode == "setpriv"
        return {
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
    return {
        "kind": "local",
        "python_executable": worker.python_executable,
        "kernelbench_root": worker.kernelbench_root,
        "asset_root": worker.asset_root,
        "timeout_seconds": worker.timeout_seconds,
        "expected_hardware_substring": worker.expected_hardware_substring,
        "expected_compute_capability": worker.expected_compute_capability,
        "expected_triton_version": worker.expected_triton_version,
    }


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
        if arguments.command == "run-trusted":
            return _run_trusted(arguments)
        if arguments.command == "run-gates":
            return _run_gates(arguments)
        if arguments.command == "analyze-study":
            return _run_analysis(arguments)
        return _run_cell(arguments)
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
        MatrixSpecError,
        StudyManifestError,
        ValidationError,
    ) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG
    except WorkerExecutionError as error:
        print(f"worker error: {error}", file=sys.stderr)
        return EXIT_WORKER
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
