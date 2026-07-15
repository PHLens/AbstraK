"""Serial subprocess driver for KernelBench evaluation cells."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from abstrak.evaluation.artifacts import (
    EvaluationArtifactError,
    seal_directory,
    verify_directory_checksums,
    write_derived_json,
    write_derived_text,
)
from abstrak.evaluation.contracts import (
    CellSpec,
    EvaluationRequest,
    EvaluationResult,
    KernelBenchNaiveStudy,
)
from abstrak.evaluation.kernelbench import KernelBenchCheckout


def _scrub_worker_environment() -> dict[str, str]:
    sensitive_markers = ("API_KEY", "AUTH", "SECRET", "TOKEN")
    return {
        name: value
        for name, value in os.environ.items()
        if not any(marker in name.upper() for marker in sensitive_markers)
    }


def _terminal_result(
    spec: CellSpec,
    status: str,
    started_at: datetime,
    error: str,
) -> EvaluationResult:
    return EvaluationResult(
        cell_id=spec.cell_id,
        status=status,
        backend=spec.target,
        precision=spec.precision,
        error=error,
        started_at_utc=started_at,
        finished_at_utc=datetime.now(timezone.utc),
    )


def evaluate_run(
    run_directory: str | Path,
    kernelbench_root: str | Path,
    *,
    python_executable: str = sys.executable,
    device: str = "cuda:0",
) -> tuple[dict[str, int], Path]:
    run_path = Path(run_directory).expanduser().resolve()
    study = KernelBenchNaiveStudy.model_validate(
        json.loads((run_path / "study.json").read_text(encoding="utf-8"))
    )
    KernelBenchCheckout(kernelbench_root, study.source)
    counts: Counter[str] = Counter()
    evaluations_path = run_path / "evaluations"
    evaluations_path.mkdir(exist_ok=True, mode=0o700)
    for cell_directory in sorted((run_path / "cells").iterdir()):
        if not cell_directory.is_dir():
            continue
        generation_checksum_sha256 = verify_directory_checksums(
            cell_directory, "generation.sha256sums"
        )
        spec = CellSpec.model_validate(
            json.loads((cell_directory / "cell.json").read_text(encoding="utf-8"))
        )
        evaluation_directory = evaluations_path / spec.cell_id
        try:
            evaluation_directory.mkdir(exist_ok=False, mode=0o700)
        except FileExistsError:
            raise EvaluationArtifactError(
                f"evaluation bundle already exists for {spec.cell_id}"
            ) from None
        generation_ref = {
            "schema_version": "kernelbench-naive-generation-ref.v1",
            "cell_id": spec.cell_id,
            "generation_checksum_sha256": generation_checksum_sha256,
        }
        write_derived_json(evaluation_directory / "generation-ref.json", generation_ref)
        started_at = datetime.now(timezone.utc)
        request = EvaluationRequest(
            cell_id=spec.cell_id,
            kernelbench_commit=study.source.commit,
            python_executable=python_executable,
            device=device,
            evaluator=study.evaluator,
            requested_at_utc=started_at,
        )
        write_derived_json(evaluation_directory / "evaluation-request.json", request)
        if not (cell_directory / "candidate.py").is_file():
            result = _terminal_result(
                spec, "no_candidate", started_at, "generation produced no code block"
            )
            log = ""
        else:
            command = [
                python_executable,
                "-m",
                "abstrak.evaluation.worker",
                "--cell-directory",
                str(cell_directory),
                "--kernelbench-root",
                str(Path(kernelbench_root).resolve()),
                "--device",
                device,
                "--num-correct-trials",
                str(study.evaluator.num_correct_trials),
                "--num-perf-trials",
                str(study.evaluator.num_perf_trials),
                "--timing-method",
                study.evaluator.timing_method,
                "--excessive-speedup-threshold",
                str(study.evaluator.excessive_speedup_threshold),
            ]
            if not study.evaluator.static_check:
                command.append("--no-static-check")
            try:
                process = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=study.evaluator.timeout_seconds,
                    env=_scrub_worker_environment(),
                )
            except subprocess.TimeoutExpired as error:
                result = _terminal_result(
                    spec,
                    "timeout",
                    started_at,
                    f"worker exceeded {study.evaluator.timeout_seconds}s",
                )
                log = (error.stdout or "") + (error.stderr or "")
            else:
                log = process.stderr
                try:
                    if process.returncode != 0:
                        raise ValueError(f"worker exited with status {process.returncode}")
                    result = EvaluationResult.model_validate(json.loads(process.stdout))
                    if result.cell_id != spec.cell_id:
                        raise ValueError("worker result cell_id mismatch")
                except (ValueError, json.JSONDecodeError) as error:
                    result = _terminal_result(
                        spec,
                        "harness_error",
                        started_at,
                        f"invalid worker result: {error}",
                    )
                    log = f"{log}\nworker stdout:\n{process.stdout}"
        write_derived_json(evaluation_directory / "evaluation.json", result)
        write_derived_text(evaluation_directory / "evaluation.log", log)
        seal_directory(evaluation_directory, "evaluation.sha256sums")
        counts[result.status] += 1
    summary_path = write_derived_json(
        run_path / "evaluation-summary.json",
        {
            "schema_version": "kernelbench-naive-evaluation-summary.v1",
            "run_id": run_path.name,
            "status_counts": dict(sorted(counts.items())),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return dict(counts), summary_path
