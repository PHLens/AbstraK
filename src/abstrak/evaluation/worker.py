"""One-process-per-cell KernelBench evaluator for a GPU worker environment."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from abstrak.evaluation.contracts import (
    CellSpec,
    EvaluationResult,
    Precision,
    TargetName,
)

RUNTIME_DISTRIBUTIONS = (
    "torch",
    "triton",
    "tilelang",
    "nvidia-cutlass-dsl",
    "cuda-python",
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    return str(value)


def _runtime_metadata(torch: Any, device: str) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(getattr(torch.version, "cuda", None)),
        "device": device,
        "package_versions": packages,
    }
    try:
        metadata["hardware"] = torch.cuda.get_device_name(torch.device(device))
    except Exception as error:
        metadata["hardware_error"] = f"{type(error).__name__}: {error}"
    return metadata


def _result(
    *,
    cell_id: str,
    backend: TargetName,
    precision: Precision,
    status: str,
    started_at: datetime,
    compiled: bool = False,
    correctness: bool = False,
    kernel_runtime_ms: float | None = None,
    reference_runtime_ms: float | None = None,
    static_errors: tuple[str, ...] = (),
    static_warnings: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
    error: str | None = None,
) -> EvaluationResult:
    ratio = None
    if correctness and kernel_runtime_ms and reference_runtime_ms:
        ratio = reference_runtime_ms / kernel_runtime_ms
    return EvaluationResult(
        cell_id=cell_id,
        status=status,
        backend=backend,
        precision=precision,
        compiled=compiled,
        correctness=correctness,
        kernel_runtime_ms=kernel_runtime_ms,
        reference_runtime_ms=reference_runtime_ms,
        performance_ratio=ratio,
        fast_0=correctness,
        fast_1=bool(correctness and ratio is not None and ratio > 1.0),
        fast_2=bool(correctness and ratio is not None and ratio > 2.0),
        static_errors=static_errors,
        static_warnings=static_warnings,
        metadata=_json_safe(metadata or {}),
        error=error,
        started_at_utc=started_at,
        finished_at_utc=datetime.now(timezone.utc),
    )


def _load_kernelbench_runtime(
    kernelbench_root: str | Path,
) -> tuple[Any, Any, Any]:
    root = Path(kernelbench_root).expanduser().resolve()
    source_path = str(root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    import torch
    from kernelbench import eval as kernel_eval
    from kernelbench.kernel_static_checker import validate_kernel_static

    return torch, kernel_eval, validate_kernel_static


def load_kernelbench_reference(
    kernelbench_root: str | Path,
    *,
    task_level: int,
    problem_id: int,
) -> str:
    """Load one KernelBench reference by its stable level/problem identifier."""

    root = Path(kernelbench_root).expanduser().resolve()
    level_directory = root / "KernelBench" / f"level{task_level}"
    matches = sorted(level_directory.glob(f"{problem_id}_*.py"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one source for level{task_level}-problem{problem_id}, "
            f"found {len(matches)} in {level_directory}"
        )
    return matches[0].read_text(encoding="utf-8")


def evaluate_candidate_source(
    *,
    cell_id: str,
    target: TargetName,
    precision: Precision,
    reference_source: str,
    candidate_source: str,
    kernelbench_root: str | Path,
    device: str,
    num_correct_trials: int,
    num_perf_trials: int,
    timing_method: str,
    excessive_speedup_threshold: float,
    static_check: bool,
) -> EvaluationResult:
    """Evaluate one candidate while leaving task and artifact ownership to the caller."""

    started_at = datetime.now(timezone.utc)
    base_metadata = {"python_version": platform.python_version(), "device": device}
    try:
        torch, kernel_eval, validate_kernel_static = _load_kernelbench_runtime(kernelbench_root)
    except Exception as error:
        return _result(
            cell_id=cell_id,
            backend=target,
            precision=precision,
            status="environment_error",
            started_at=started_at,
            metadata=base_metadata,
            error=f"{type(error).__name__}: {error}",
        )

    runtime_metadata = _runtime_metadata(torch, device)
    if not torch.cuda.is_available():
        return _result(
            cell_id=cell_id,
            backend=target,
            precision=precision,
            status="environment_error",
            started_at=started_at,
            metadata={"runtime_environment": runtime_metadata},
            error="CUDA is not available",
        )

    static_errors: tuple[str, ...] = ()
    static_warnings: tuple[str, ...] = ()
    if static_check:
        try:
            valid, errors, warnings = validate_kernel_static(
                candidate_source,
                backend=target,
                precision=precision,
            )
            static_errors = tuple(str(item) for item in errors)
            static_warnings = tuple(str(item) for item in warnings)
        except Exception as error:
            return _result(
                cell_id=cell_id,
                backend=target,
                precision=precision,
                status="harness_error",
                started_at=started_at,
                metadata={"runtime_environment": runtime_metadata},
                error=f"static checker failed: {type(error).__name__}: {error}",
            )
        if not valid:
            return _result(
                cell_id=cell_id,
                backend=target,
                precision=precision,
                status="static_check_failed",
                started_at=started_at,
                static_errors=static_errors,
                static_warnings=static_warnings,
                metadata={"runtime_environment": runtime_metadata},
            )

    try:
        with contextlib.redirect_stdout(sys.stderr):
            execution = kernel_eval.eval_kernel_against_ref(
                original_model_src=reference_source,
                custom_model_src=candidate_source,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                measure_performance=True,
                timing_method=timing_method,
                verbose=False,
                device=torch.device(device),
                backend=target,
                precision=kernel_eval.get_torch_dtype_from_string(precision),
                check_for_excessive_speedup=True,
                excessive_speedup_threshold=excessive_speedup_threshold,
            )
    except Exception as error:
        return _result(
            cell_id=cell_id,
            backend=target,
            precision=precision,
            status="harness_error",
            started_at=started_at,
            static_errors=static_errors,
            static_warnings=static_warnings,
            metadata={"runtime_environment": runtime_metadata},
            error=f"{type(error).__name__}: {error}",
        )
    if execution is None:
        return _result(
            cell_id=cell_id,
            backend=target,
            precision=precision,
            status="harness_error",
            started_at=started_at,
            static_errors=static_errors,
            static_warnings=static_warnings,
            metadata={"runtime_environment": runtime_metadata},
            error="KernelBench returned no execution result",
        )
    kernel_runtime = execution.runtime if execution.runtime > 0 else None
    reference_runtime = execution.ref_runtime if execution.ref_runtime > 0 else None
    return _result(
        cell_id=cell_id,
        backend=target,
        precision=precision,
        status="evaluated",
        started_at=started_at,
        compiled=execution.compiled,
        correctness=execution.correctness,
        kernel_runtime_ms=kernel_runtime,
        reference_runtime_ms=reference_runtime,
        static_errors=static_errors,
        static_warnings=static_warnings,
        metadata={
            **_json_safe(execution.metadata),
            "runtime_environment": runtime_metadata,
        },
    )


def evaluate_kernelbench_task_candidate(
    *,
    cell_id: str,
    task_level: int,
    problem_id: int,
    target: TargetName,
    precision: Precision,
    candidate_source: str,
    kernelbench_root: str | Path,
    device: str,
    num_correct_trials: int,
    num_perf_trials: int,
    timing_method: str,
    excessive_speedup_threshold: float,
    static_check: bool,
) -> EvaluationResult:
    """Load a KernelBench task and evaluate a candidate against its reference."""

    started_at = datetime.now(timezone.utc)
    try:
        reference_source = load_kernelbench_reference(
            kernelbench_root,
            task_level=task_level,
            problem_id=problem_id,
        )
    except (OSError, ValueError) as error:
        return _result(
            cell_id=cell_id,
            backend=target,
            precision=precision,
            status="harness_error",
            started_at=started_at,
            metadata={"python_version": platform.python_version(), "device": device},
            error=f"cannot load KernelBench reference: {type(error).__name__}: {error}",
        )
    return evaluate_candidate_source(
        cell_id=cell_id,
        target=target,
        precision=precision,
        reference_source=reference_source,
        candidate_source=candidate_source,
        kernelbench_root=kernelbench_root,
        device=device,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        timing_method=timing_method,
        excessive_speedup_threshold=excessive_speedup_threshold,
        static_check=static_check,
    )


def evaluate_cell(
    cell_directory: str | Path,
    kernelbench_root: str | Path,
    *,
    device: str,
    num_correct_trials: int,
    num_perf_trials: int,
    timing_method: str,
    excessive_speedup_threshold: float,
    static_check: bool,
) -> EvaluationResult:
    directory = Path(cell_directory)
    spec = CellSpec.model_validate(json.loads((directory / "cell.json").read_text()))
    reference_source = (directory / "reference.py").read_text(encoding="utf-8")
    candidate_source = (directory / "candidate.py").read_text(encoding="utf-8")
    return evaluate_candidate_source(
        cell_id=spec.cell_id,
        target=spec.target,
        precision=spec.precision,
        reference_source=reference_source,
        candidate_source=candidate_source,
        kernelbench_root=kernelbench_root,
        device=device,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        timing_method=timing_method,
        excessive_speedup_threshold=excessive_speedup_threshold,
        static_check=static_check,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-directory", required=True)
    parser.add_argument("--kernelbench-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-correct-trials", required=True, type=int)
    parser.add_argument("--num-perf-trials", required=True, type=int)
    parser.add_argument("--timing-method", required=True)
    parser.add_argument("--excessive-speedup-threshold", required=True, type=float)
    parser.add_argument("--no-static-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):
        result = evaluate_cell(
            arguments.cell_directory,
            arguments.kernelbench_root,
            device=arguments.device,
            num_correct_trials=arguments.num_correct_trials,
            num_perf_trials=arguments.num_perf_trials,
            timing_method=arguments.timing_method,
            excessive_speedup_threshold=arguments.excessive_speedup_threshold,
            static_check=not arguments.no_static_check,
        )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
