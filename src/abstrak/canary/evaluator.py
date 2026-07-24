"""GPU-side evaluator for one explicit canary worker job."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import math
import os
import platform
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abstrak.canary.baselines import BaselineRegistryError, validate_baseline_source
from abstrak.canary.contracts import CaseResult, WorkerJob, WorkerResult
from abstrak.canary.target_adapters import (
    DEFAULT_TARGET_ADAPTER_REGISTRY,
    TargetAdapterRegistry,
    validate_target_source,
)
from abstrak.canary.targets import TargetRegistryError, get_target_stack
from abstrak.canary.tasks import (
    TaskRegistryError,
    get_task_pack,
    load_task_source,
)


@dataclass(frozen=True)
class EvaluationRuntime:
    torch: Any
    kernel_eval: Any
    timing: Any


RuntimeLoader = Callable[[str | Path], EvaluationRuntime]


def _generated_code_metadata(candidate_model: Any, job: WorkerJob) -> dict[str, Any]:
    """Capture final TileLang source for every non-baseline capability job."""

    if job.kind == "baseline" or not job.target.adapter.startswith("tilelang-capability-"):
        return {}
    kernel = getattr(candidate_model, "kernel", None)
    get_kernel_source = getattr(kernel, "get_kernel_source", None)
    if not callable(get_kernel_source):
        raise RuntimeError("capability candidate does not expose TileLang get_kernel_source()")
    source = get_kernel_source()
    if not isinstance(source, str) or not source:
        raise RuntimeError("TileLang generated kernel source is empty")
    encoded = source.encode("utf-8")
    return {
        "generated_code_capture": "tilelang.get_kernel_source.v1",
        "generated_code_sha256": hashlib.sha256(encoded).hexdigest(),
        "generated_code_size_bytes": len(encoded),
    }


def _load_runtime(kernelbench_root: str | Path) -> EvaluationRuntime:
    root = Path(kernelbench_root).expanduser().resolve()
    source_root = root / "src"
    if not source_root.is_dir():
        raise RuntimeError(f"invalid KernelBench checkout: {root}")
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    torch = importlib.import_module("torch")
    kernel_eval = importlib.import_module("kernelbench.eval")
    timing = importlib.import_module("kernelbench.timing")
    return EvaluationRuntime(torch=torch, kernel_eval=kernel_eval, timing=timing)


def _result(
    job: WorkerJob,
    status: str,
    *,
    compiled: bool = False,
    correct: bool = False,
    cases: tuple[CaseResult, ...] = (),
    timing_ms: tuple[float, ...] = (),
    timing_cv: float | None = None,
    static_errors: tuple[str, ...] = (),
    static_warnings: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
    error: str | None = None,
) -> WorkerResult:
    result = WorkerResult(
        job_id=job.job_id,
        job_sha256=job.sha256,
        input_sha256=job.input_sha256,
        candidate_sha256=job.candidate_sha256,
        status=status,
        compiled=compiled,
        correct=correct,
        cases=cases,
        timing_ms=timing_ms,
        timing_cv=timing_cv,
        static_errors=static_errors,
        static_warnings=static_warnings,
        metadata=metadata or {},
        error=error,
    )
    return result.verify_for_job(job)


def _runtime_metadata(
    runtime: EvaluationRuntime,
    device: str,
    *,
    include_isolation: bool = False,
) -> dict[str, Any]:
    torch = runtime.torch
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(getattr(torch.version, "cuda", None)),
        "device": device,
    }
    if include_isolation:
        workspace = Path.cwd().resolve()
        cache_names = ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME")
        cache_roots = {
            name: str(Path(os.environ.get(name, "")).resolve())
            for name in cache_names
            if os.environ.get(name)
        }
        metadata.update(
            {
                "worker_workspace": str(workspace),
                "worker_cache_roots": cache_roots,
                "per_job_cache_isolated": len(cache_roots) == len(cache_names)
                and all(Path(path).is_relative_to(workspace) for path in cache_roots.values()),
            }
        )
    try:
        metadata["hardware"] = str(torch.cuda.get_device_name(torch.device(device)))
    except Exception as error:
        metadata["hardware_error"] = f"{type(error).__name__}: {error}"
    return metadata


def _coefficient_of_variation(samples: tuple[float, ...]) -> float:
    if not samples:
        raise ValueError("cannot compute timing CV without samples")
    mean = statistics.fmean(samples)
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError("timing mean must be finite and positive")
    return statistics.pstdev(samples) / mean


def _load_reference(source: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "abstrak_canary_reference"}
    exec(compile(source, "<canary-reference>", "exec"), namespace)
    required = ("Model", "make_inputs", "get_init_inputs")
    missing = [name for name in required if not callable(namespace.get(name))]
    if missing:
        raise RuntimeError(f"task source is missing callables: {', '.join(missing)}")
    return namespace


def _to_device(value: Any, *, torch: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return value


def _clone_value(value: Any, *, torch: Any) -> Any:
    return value.clone() if isinstance(value, torch.Tensor) else value


def _inputs_unchanged(inputs: list[Any], snapshots: list[Any], *, torch: Any) -> bool:
    for value, snapshot in zip(inputs, snapshots, strict=True):
        if isinstance(value, torch.Tensor):
            if not isinstance(snapshot, torch.Tensor) or not bool(torch.equal(value, snapshot)):
                return False
        elif value != snapshot:
            return False
    return True


def _case_result(
    case_id: str,
    expected: Any,
    actual: Any,
    candidate_inputs: list[Any],
    input_snapshots: list[Any],
    *,
    torch: Any,
    atol: float,
    rtol: float,
) -> CaseResult:
    inputs_unchanged = _inputs_unchanged(candidate_inputs, input_snapshots, torch=torch)
    if not inputs_unchanged:
        return CaseResult(
            case_id=case_id,
            status="input_mutation",
            correct=False,
            output_finite=True,
            inputs_unchanged=False,
        )
    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        return CaseResult(
            case_id=case_id,
            status="wrong_result",
            correct=False,
            output_finite=True,
            inputs_unchanged=True,
            error="reference and candidate must each return one Tensor",
        )
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return CaseResult(
            case_id=case_id,
            status="wrong_result",
            correct=False,
            output_finite=True,
            inputs_unchanged=True,
            error=(
                f"expected shape/dtype {tuple(expected.shape)}/{expected.dtype}, "
                f"got {tuple(actual.shape)}/{actual.dtype}"
            ),
        )
    if expected.device != actual.device:
        return CaseResult(
            case_id=case_id,
            status="wrong_result",
            correct=False,
            output_finite=True,
            inputs_unchanged=True,
            error=f"expected output on {expected.device}, got {actual.device}",
        )
    output_finite = bool(torch.isfinite(actual).all().item())
    if not output_finite:
        return CaseResult(
            case_id=case_id,
            status="nonfinite_output",
            correct=False,
            output_finite=False,
            inputs_unchanged=True,
        )
    expected_fp32 = expected.to(torch.float32)
    actual_fp32 = actual.to(torch.float32)
    absolute = torch.abs(expected_fp32 - actual_fp32)
    relative = absolute / torch.clamp(torch.abs(expected_fp32), min=1e-12)
    max_abs_error = float(torch.max(absolute).item())
    max_rel_error = float(torch.max(relative).item())
    correct = bool(torch.allclose(expected, actual, atol=atol, rtol=rtol))
    return CaseResult(
        case_id=case_id,
        status="pass" if correct else "wrong_result",
        correct=correct,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        output_finite=True,
        inputs_unchanged=True,
    )


def _registry_error(job: WorkerJob) -> str | None:
    try:
        registered_task = get_task_pack(job.task.id)
        registered_target = get_target_stack(job.target.id)
    except (TaskRegistryError, TargetRegistryError) as error:
        return str(error)
    if registered_task != job.task:
        return f"job task contract differs from registry: {job.task.id}"
    if registered_target != job.target:
        return f"job target contract differs from registry: {job.target.id}"
    return None


def evaluate_job(
    job: WorkerJob,
    kernelbench_root: str | Path,
    *,
    device: str | None = None,
    asset_root: str | Path | None = None,
    runtime_loader: RuntimeLoader = _load_runtime,
    target_adapter_registry: TargetAdapterRegistry = DEFAULT_TARGET_ADAPTER_REGISTRY,
) -> WorkerResult:
    """Evaluate one hash-bound job; Torch is imported only after static preflight."""

    registry_error = _registry_error(job)
    if registry_error is not None:
        return _result(job, "worker_error", error=registry_error)

    static_warnings: tuple[str, ...] = ()
    validation_metadata: dict[str, Any] = {}
    if job.kind == "baseline":
        try:
            validate_baseline_source(
                job.task.id,
                job.candidate_source,
                source_sha256=job.candidate_sha256,
            )
        except BaselineRegistryError as error:
            return _result(job, "worker_error", error=str(error))
    else:
        static = validate_target_source(
            job.candidate_source,
            job.target,
            registry=target_adapter_registry,
        )
        static_errors = tuple(f"{issue.code}: {issue.message}" for issue in static.errors)
        static_warnings = tuple(f"{issue.code}: {issue.message}" for issue in static.warnings)
        validation_metadata = dict(static.metadata)
        if not static.valid:
            return _result(
                job,
                "static_check_failed",
                static_errors=static_errors,
                static_warnings=static_warnings,
                metadata=validation_metadata,
            )

    selected_device = device or job.device
    try:
        runtime = runtime_loader(kernelbench_root)
    except Exception as error:
        metadata = dict(validation_metadata)
        metadata["device"] = selected_device
        return _result(
            job,
            "environment_error",
            error=f"{type(error).__name__}: {error}",
            static_warnings=static_warnings,
            metadata=metadata,
        )
    torch = runtime.torch
    metadata = dict(validation_metadata)
    metadata.update(
        _runtime_metadata(
            runtime,
            selected_device,
            include_isolation=job.target.adapter.startswith("tilelang-capability-"),
        )
    )
    if not torch.cuda.is_available():
        return _result(
            job,
            "environment_error",
            error="CUDA is not available",
            static_warnings=static_warnings,
            metadata=metadata,
        )

    torch_device = torch.device(selected_device)
    torch.cuda.set_device(torch_device)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference_source = load_task_source(job.task.id, asset_root=asset_root)
        reference = _load_reference(reference_source)
    except Exception as error:
        return _result(
            job,
            "worker_error",
            error=f"cannot load task reference: {type(error).__name__}: {error}",
            static_warnings=static_warnings,
            metadata=metadata,
        )

    candidate_file: Any = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            candidate_class, candidate_file = runtime.kernel_eval.load_custom_model_with_tempfile(
                job.candidate_source,
                entry_point="ModelNew",
            )
            reference_model = reference["Model"](*job.task.init_args).to(device=torch_device)
            candidate_model = candidate_class(*job.task.init_args).to(device=torch_device)
            reference_model.eval()
            candidate_model.eval()
            metadata.update(_generated_code_metadata(candidate_model, job))
            torch.cuda.synchronize(device=torch_device)
    except Exception as error:
        if candidate_file is not None:
            candidate_path = Path(candidate_file.name)
            candidate_file.close()
            candidate_path.unlink(missing_ok=True)
        return _result(
            job,
            "compile_error",
            error=f"{type(error).__name__}: {error}",
            static_warnings=static_warnings,
            metadata=metadata,
        )

    case_map = job.task.cases_by_id()
    case_results: list[CaseResult] = []
    timing_inputs: list[Any] | None = None
    try:
        for case_id in job.case_ids:
            case = case_map[case_id]
            source_inputs = reference["make_inputs"](case.kind, case.seed, case.value)
            if not isinstance(source_inputs, list | tuple):
                raise RuntimeError("make_inputs must return a list or tuple")
            device_inputs = [
                _to_device(value, torch=torch, device=torch_device) for value in source_inputs
            ]
            reference_inputs = [_clone_value(value, torch=torch) for value in device_inputs]
            candidate_inputs = [_clone_value(value, torch=torch) for value in device_inputs]
            snapshots = [_clone_value(value, torch=torch) for value in candidate_inputs]
            try:
                with torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
                    expected = reference_model(*reference_inputs)
                    actual = candidate_model(*candidate_inputs)
                    torch.cuda.synchronize(device=torch_device)
            except Exception as error:
                case_results.append(
                    CaseResult(
                        case_id=case_id,
                        status="runtime_error",
                        correct=False,
                        output_finite=True,
                        inputs_unchanged=_inputs_unchanged(
                            candidate_inputs, snapshots, torch=torch
                        ),
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                return _result(
                    job,
                    "runtime_error",
                    compiled=True,
                    cases=tuple(case_results),
                    error=f"candidate failed on {case_id}: {type(error).__name__}: {error}",
                    static_warnings=static_warnings,
                    metadata=metadata,
                )
            result = _case_result(
                case_id,
                expected,
                actual,
                candidate_inputs,
                snapshots,
                torch=torch,
                atol=job.task.atol,
                rtol=job.task.rtol,
            )
            case_results.append(result)
            if timing_inputs is None and result.correct:
                timing_inputs = [_clone_value(value, torch=torch) for value in device_inputs]

        if any(not result.correct for result in case_results):
            return _result(
                job,
                "wrong_result",
                compiled=True,
                cases=tuple(case_results),
                static_warnings=static_warnings,
                metadata=metadata,
            )

        timing_samples: tuple[float, ...] = ()
        timing_cv: float | None = None
        if job.timing is not None:
            if timing_inputs is None:
                raise RuntimeError("no correct input is available for timing")
            with contextlib.redirect_stdout(sys.stderr):
                raw_samples = runtime.timing.time_execution_with_cuda_event(
                    candidate_model,
                    timing_inputs,
                    num_warmup=job.timing.warmup_runs,
                    num_trials=job.timing.trial_runs,
                    discard_first=job.timing.discard_first,
                    verbose=False,
                    device=torch_device,
                )
            timing_samples = tuple(float(value) for value in raw_samples)
            timing_cv = _coefficient_of_variation(timing_samples)
        return _result(
            job,
            "completed",
            compiled=True,
            correct=True,
            cases=tuple(case_results),
            timing_ms=timing_samples,
            timing_cv=timing_cv,
            static_warnings=static_warnings,
            metadata=metadata,
        )
    except Exception as error:
        return _result(
            job,
            "worker_error",
            compiled=True,
            cases=tuple(case_results),
            error=f"{type(error).__name__}: {error}",
            static_warnings=static_warnings,
            metadata=metadata,
        )
    finally:
        if candidate_file is not None:
            candidate_path = Path(candidate_file.name)
            candidate_file.close()
            candidate_path.unlink(missing_ok=True)
