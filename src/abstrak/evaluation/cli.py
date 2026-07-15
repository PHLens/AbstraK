"""CLI for the naive single-turn KernelBench capability screen."""

from __future__ import annotations

import argparse
import json
import os
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
from abstrak.evaluation.artifacts import EvaluationArtifactError
from abstrak.evaluation.contracts import StudyError, load_study
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
        return _summarize(arguments)
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
