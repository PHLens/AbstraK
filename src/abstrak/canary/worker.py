"""JSON-in/JSON-out CLI for one canary GPU worker job."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from abstrak.canary.contracts import WorkerJob, WorkerResult
from abstrak.canary.evaluator import evaluate_job

JobEvaluator = Callable[..., WorkerResult]


def load_job_payload(value: str) -> WorkerJob:
    """Parse one strict canonical worker job from JSON text."""

    return WorkerJob.model_validate_json(value)


def run_worker_job(
    job: WorkerJob,
    *,
    kernelbench_root: str | Path,
    device: str | None = None,
    asset_root: str | Path | None = None,
    evaluator: JobEvaluator = evaluate_job,
) -> WorkerResult:
    result = evaluator(
        job,
        kernelbench_root,
        device=device,
        asset_root=asset_root,
    )
    return result.verify_for_job(job)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default="-", help="worker job JSON path, or - for stdin")
    parser.add_argument("--kernelbench-root", required=True)
    parser.add_argument("--asset-root")
    parser.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.job == "-":
            payload = sys.stdin.read()
        else:
            payload = Path(arguments.job).read_text(encoding="utf-8")
        job = load_job_payload(payload)
    except (OSError, ValidationError, ValueError) as error:
        print(f"invalid worker job: {error}", file=sys.stderr)
        return 2

    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = run_worker_job(
                job,
                kernelbench_root=arguments.kernelbench_root,
                device=arguments.device,
                asset_root=arguments.asset_root,
            )
    except Exception as error:
        print(f"worker failure: {type(error).__name__}: {error}", file=sys.stderr)
        return 3
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
