"""CLI for KernelBench screening and iterative DSL-target pilots."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

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
from abstrak.evaluation.agent_analysis import AgentAnalysisError, analyze_agent_run
from abstrak.evaluation.agent_contracts import (
    KernelBenchAgentStudy,
    load_agent_study,
)
from abstrak.evaluation.agent_figures import AgentFigureError, plot_agent_run
from abstrak.evaluation.agent_provider import AgentProviderError, PilotProviderClient
from abstrak.evaluation.agent_runner import (
    AgentCollectionError,
    AgentCollectionResult,
    AgentCollectionRunner,
    AgentEvaluationTransportError,
    SshAgentEvaluator,
)
from abstrak.evaluation.agent_worker import AgentEvaluationJob
from abstrak.evaluation.artifacts import EvaluationArtifactError
from abstrak.evaluation.contracts import (
    KernelBenchEvaluatorConfig,
    StudyError,
    load_study,
)
from abstrak.evaluation.evaluator import evaluate_run
from abstrak.evaluation.generation import NaiveGenerationRunner
from abstrak.evaluation.kernelbench import KernelBenchCheckout, prompt_sha256
from abstrak.evaluation.summary import summarize_run
from abstrak.providers.client import ProviderConfigurationError
from abstrak.providers.manifests import MissingEnvironmentError

KERNELBENCH_ROOT_ENV = "KERNELBENCH_ROOT"
EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_GENERATION = 3
EXIT_EVALUATION = 4
EXIT_AGENT = 5


def _emit(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _add_study_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study", required=True, help="naive study YAML")
    parser.add_argument(
        "--kernelbench-root",
        default=os.environ.get(KERNELBENCH_ROOT_ENV),
        help=f"pinned KernelBench checkout (default: ${KERNELBENCH_ROOT_ENV})",
    )
    parser.add_argument(
        "--config",
        help=f"user config YAML (default: ${CONFIG_ENV} or ~/.abstrak/config.yaml)",
    )


def _add_remote_worker_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ssh-host", required=True, help="SSH destination for the GPU worker")
    parser.add_argument("--ssh-port", type=int, help="SSH port (default: OpenSSH config)")
    parser.add_argument(
        "--worker-root",
        required=True,
        help="AbstraK repository path on the GPU worker",
    )
    parser.add_argument(
        "--worker-python",
        default="/tmp/abstrak-gpu-venv/bin/python",
        help="Python executable on the GPU worker",
    )
    parser.add_argument(
        "--worker-kernelbench-root",
        required=True,
        help="KernelBench checkout path on the GPU worker",
    )
    parser.add_argument("--device", default="cuda:0")


def _add_agent_collection_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study", required=True, help="agent pilot study YAML")
    parser.add_argument(
        "--kernelbench-root",
        default=os.environ.get(KERNELBENCH_ROOT_ENV),
        help=f"local pinned KernelBench checkout (default: ${KERNELBENCH_ROOT_ENV})",
    )
    parser.add_argument(
        "--auth",
        help=f"credential JSON (default: ${AUTH_ENV} or ~/.abstrak/auth.json)",
    )
    _add_remote_worker_inputs(parser)
    parser.add_argument("--iterations", type=int, help="override the study's agent turns")
    parser.add_argument("--run-id", help="run directory name")
    parser.add_argument(
        "--artifact-root",
        default="artifacts/kernelbench-agent",
        help="root for raw attempts, analysis, and figures",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="acknowledge model requests and execution of generated code over SSH",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate study, tasks, and prompts")
    _add_study_inputs(validate)

    generate = subparsers.add_parser(
        "generate", help="perform one billable model request per study cell"
    )
    _add_study_inputs(generate)
    generate.add_argument(
        "--auth",
        help=f"credential JSON (default: ${AUTH_ENV} or ~/.abstrak/auth.json)",
    )
    generate.add_argument(
        "--live",
        action="store_true",
        help="acknowledge that the matrix performs billable network requests",
    )
    generate.add_argument(
        "--expected-requests",
        required=True,
        type=int,
        help="must exactly match the frozen matrix size",
    )
    generate.add_argument("--run-id", help="safe immutable run directory name")
    generate.add_argument(
        "--artifact-root",
        default="artifacts/kernelbench-naive",
        help="ignored-by-Git root for private generated kernels",
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate a generated run in a KernelBench GPU environment"
    )
    evaluate.add_argument("--run", required=True, help="generated study run directory")
    evaluate.add_argument(
        "--kernelbench-root",
        default=os.environ.get(KERNELBENCH_ROOT_ENV),
        help=f"pinned KernelBench checkout (default: ${KERNELBENCH_ROOT_ENV})",
    )
    evaluate.add_argument("--python", default=sys.executable, help="GPU worker Python")
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument(
        "--execute-generated-code",
        action="store_true",
        help="acknowledge execution of untrusted model-generated code",
    )

    summarize = subparsers.add_parser(
        "summarize", help="aggregate correctness and speed ratios by profile and target"
    )
    summarize.add_argument("--run", required=True, help="evaluated study run directory")

    agent_eval = subparsers.add_parser(
        "agent-eval", help="evaluate one candidate on a remote KernelBench worker"
    )
    agent_eval.add_argument("--candidate", required=True, help="candidate Python source")
    agent_eval.add_argument(
        "--task",
        required=True,
        help="KernelBench task such as level1:1 or level2:76",
    )
    agent_eval.add_argument(
        "--target", required=True, choices=("triton", "tilelang", "cute")
    )
    agent_eval.add_argument(
        "--precision", default="fp16", choices=("fp16", "bf16", "fp32")
    )
    _add_remote_worker_inputs(agent_eval)
    agent_eval.add_argument("--num-correct-trials", type=int, default=5)
    agent_eval.add_argument("--num-perf-trials", type=int, default=100)
    agent_eval.add_argument(
        "--timing-method",
        default="cuda_event",
        choices=("cuda_event", "do_bench", "do_bench_impl", "host_time"),
    )
    agent_eval.add_argument("--timeout-seconds", type=int, default=300)
    agent_eval.add_argument("--excessive-speedup-threshold", type=float, default=10.0)
    agent_eval.add_argument("--no-static-check", action="store_true")

    agent_collect = subparsers.add_parser(
        "agent-collect", help="run model turns with immediate KernelBench feedback"
    )
    _add_agent_collection_inputs(agent_collect)

    agent_analyze = subparsers.add_parser(
        "agent-analyze", help="derive metrics from raw agent attempts"
    )
    agent_analyze.add_argument("--run", required=True, help="agent run directory")
    agent_analyze.add_argument(
        "--reference-file",
        help="optional task/target reference-speedup CSV",
    )

    agent_plot = subparsers.add_parser(
        "agent-plot", help="render pilot figures from derived metrics"
    )
    agent_plot.add_argument("--run", required=True, help="analyzed agent run directory")

    agent_pipeline = subparsers.add_parser(
        "agent-pipeline", help="collect, analyze, and plot in one command"
    )
    _add_agent_collection_inputs(agent_pipeline)
    agent_pipeline.add_argument(
        "--reference-file",
        help="optional task/target reference-speedup CSV",
    )
    return parser


def _config_path(explicit: str | None) -> Path:
    path, _ = resolve_path(explicit, environment_name=CONFIG_ENV, default=default_config_path())
    return path


def _require_checkout(value: str | None) -> str:
    if not value:
        raise StudyError(
            f"KernelBench checkout is required via --kernelbench-root or ${KERNELBENCH_ROOT_ENV}"
        )
    return value


def _validate(arguments: argparse.Namespace) -> int:
    study = load_study(arguments.study)
    config = load_app_config(_config_path(arguments.config))
    for profile in study.profiles:
        config.bundle(profile)
    checkout = KernelBenchCheckout(_require_checkout(arguments.kernelbench_root), study.source)
    tasks: list[dict[str, object]] = []
    for task in study.tasks:
        material = checkout.load_task(task)
        prompt_hashes = {
            target: prompt_sha256(checkout.zero_shot_prompt(material, target, study.precision))
            for target in study.targets
        }
        tasks.append(
            {
                "ref": task.ref,
                "name": material.name,
                "stratum": task.stratum,
                "source_sha256": material.source_sha256,
                "prompt_sha256": prompt_hashes,
            }
        )
    _emit(
        {
            "status": "valid",
            "study_id": study.id,
            "study_sha256": study.sha256,
            "matrix_size": study.matrix_size,
            "profiles": study.profiles,
            "targets": study.targets,
            "precision": study.precision,
            "single_turn": True,
            "memory": False,
            "workflow": False,
            "hardware_prompt": False,
            "tasks": tasks,
        }
    )
    return EXIT_OK


def _generate(arguments: argparse.Namespace) -> int:
    if not arguments.live:
        raise StudyError("generate requires --live because it performs billable requests")
    study = load_study(arguments.study)
    if arguments.expected_requests != study.matrix_size:
        raise StudyError(
            f"--expected-requests must equal the frozen matrix size ({study.matrix_size})"
        )
    config = load_app_config(_config_path(arguments.config))
    checkout = KernelBenchCheckout(_require_checkout(arguments.kernelbench_root), study.source)
    auth_path, configured = resolve_path(
        arguments.auth,
        environment_name=AUTH_ENV,
        default=default_auth_path(),
    )
    auth = load_auth_store(auth_path, missing_ok=not configured)
    runner = NaiveGenerationRunner(
        study=study,
        config=config,
        environment=runtime_environment(auth, os.environ),
        checkout=checkout,
        artifact_root=arguments.artifact_root,
        run_id=arguments.run_id,
    )
    run_directory, counts = runner.run()
    _emit(
        {
            "status": "complete",
            "study_id": study.id,
            "matrix_size": study.matrix_size,
            "status_counts": counts,
            "run_directory": str(run_directory),
        }
    )
    return EXIT_OK


def _evaluate(arguments: argparse.Namespace) -> int:
    if not arguments.execute_generated_code:
        raise StudyError("evaluate requires --execute-generated-code")
    counts, summary_path = evaluate_run(
        arguments.run,
        _require_checkout(arguments.kernelbench_root),
        python_executable=arguments.python,
        device=arguments.device,
    )
    _emit(
        {
            "status": "complete",
            "status_counts": counts,
            "evaluation_summary": str(summary_path),
        }
    )
    return EXIT_OK


def _summarize(arguments: argparse.Namespace) -> int:
    payload, path = summarize_run(arguments.run)
    _emit({**payload, "metrics_path": str(path)})
    return EXIT_OK


def _parse_agent_task(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(?:level)?(?P<level>[1-4])(?::|-problem)(?P<problem>[1-9][0-9]*)", value)
    if match is None:
        raise StudyError("--task must look like level1:1 or level2-problem76")
    return int(match.group("level")), int(match.group("problem"))


def _ssh_evaluator(arguments: argparse.Namespace) -> SshAgentEvaluator:
    return SshAgentEvaluator(
        host=arguments.ssh_host,
        port=arguments.ssh_port,
        worker_root=arguments.worker_root,
        worker_python=arguments.worker_python,
    )


def _agent_eval(arguments: argparse.Namespace) -> int:
    level, problem_id = _parse_agent_task(arguments.task)
    candidate = Path(arguments.candidate).expanduser().read_text(encoding="utf-8")
    evaluator_config = KernelBenchEvaluatorConfig(
        num_correct_trials=arguments.num_correct_trials,
        num_perf_trials=arguments.num_perf_trials,
        timing_method=arguments.timing_method,
        timeout_seconds=arguments.timeout_seconds,
        excessive_speedup_threshold=arguments.excessive_speedup_threshold,
        static_check=not arguments.no_static_check,
    )
    job = AgentEvaluationJob(
        cell_id=f"agent-eval--level{level}-problem{problem_id}--{arguments.target}",
        task_level=level,
        problem_id=problem_id,
        target=arguments.target,
        precision=arguments.precision,
        candidate_source=candidate,
        kernelbench_root=arguments.worker_kernelbench_root,
        device=arguments.device,
        evaluator=evaluator_config,
    )
    outcome = _ssh_evaluator(arguments).evaluate(job)
    _emit(
        {
            "status": "complete",
            "evaluation": outcome.result.model_dump(mode="json"),
            "worker_log": outcome.log,
        }
    )
    return EXIT_OK


def _collect_agent(
    arguments: argparse.Namespace,
) -> tuple[KernelBenchAgentStudy, AgentCollectionResult]:
    if not arguments.live:
        raise StudyError(
            f"{arguments.command} requires --live because it calls models and executes "
            "generated code"
        )
    study = load_agent_study(arguments.study)
    checkout = KernelBenchCheckout(_require_checkout(arguments.kernelbench_root), study.source)
    auth_path, configured = resolve_path(
        arguments.auth,
        environment_name=AUTH_ENV,
        default=default_auth_path(),
    )
    auth = load_auth_store(auth_path, missing_ok=not configured)
    environment = runtime_environment(auth, os.environ)
    runner = AgentCollectionRunner(
        study=study,
        checkout=checkout,
        provider_factory=lambda model: PilotProviderClient(
            model,
            study.generation,
            environment=environment,
        ),
        evaluator=_ssh_evaluator(arguments),
        worker_kernelbench_root=arguments.worker_kernelbench_root,
        artifact_root=arguments.artifact_root,
        run_id=arguments.run_id,
        iterations=arguments.iterations,
        device=arguments.device,
    )
    return study, runner.run()


def _agent_collect(arguments: argparse.Namespace) -> int:
    study, result = _collect_agent(arguments)
    _emit(
        {
            "status": "complete",
            "study_id": study.id,
            **result.model_dump(mode="json"),
        }
    )
    return EXIT_OK


def _agent_analyze(arguments: argparse.Namespace) -> int:
    metrics, metrics_json, metrics_csv = analyze_agent_run(
        arguments.run,
        reference_file=getattr(arguments, "reference_file", None),
    )
    _emit(
        {
            "status": "complete",
            "run_id": metrics["run_id"],
            "metrics_json": str(metrics_json),
            "metrics_csv": str(metrics_csv),
        }
    )
    return EXIT_OK


def _agent_plot(arguments: argparse.Namespace) -> int:
    paths = plot_agent_run(arguments.run)
    _emit({"status": "complete", "figures": [str(path) for path in paths]})
    return EXIT_OK


def _agent_pipeline(arguments: argparse.Namespace) -> int:
    study, collection = _collect_agent(arguments)
    metrics, metrics_json, metrics_csv = analyze_agent_run(
        collection.run_directory,
        reference_file=getattr(arguments, "reference_file", None),
    )
    figures = plot_agent_run(collection.run_directory)
    _emit(
        {
            "status": "complete",
            "study_id": study.id,
            "run_id": metrics["run_id"],
            "run_directory": str(collection.run_directory),
            "attempts": collection.attempts,
            "metrics_json": str(metrics_json),
            "metrics_csv": str(metrics_csv),
            "figures": [str(path) for path in figures],
        }
    )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate":
            return _validate(arguments)
        if arguments.command == "generate":
            return _generate(arguments)
        if arguments.command == "evaluate":
            return _evaluate(arguments)
        if arguments.command == "summarize":
            return _summarize(arguments)
        if arguments.command == "agent-eval":
            return _agent_eval(arguments)
        if arguments.command == "agent-collect":
            return _agent_collect(arguments)
        if arguments.command == "agent-analyze":
            return _agent_analyze(arguments)
        if arguments.command == "agent-plot":
            return _agent_plot(arguments)
        return _agent_pipeline(arguments)
    except (
        AgentAnalysisError,
        AgentCollectionError,
        AgentEvaluationTransportError,
        AgentFigureError,
        AgentProviderError,
    ) as error:
        print(f"agent error: {error}", file=sys.stderr)
        return EXIT_AGENT
    except (
        ConfigurationError,
        EvaluationArtifactError,
        MissingEnvironmentError,
        ProviderConfigurationError,
        StudyError,
        ValidationError,
    ) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG
    except OSError as error:
        print(f"generation error: {error}", file=sys.stderr)
        return EXIT_GENERATION
    except Exception as error:
        print(f"evaluation error: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_EVALUATION


if __name__ == "__main__":
    raise SystemExit(main())
