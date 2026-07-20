from __future__ import annotations

import hashlib
import subprocess
import sys

from abstrak.canary.baselines import load_baseline_source
from abstrak.canary.contracts import WorkerJob
from abstrak.canary.evaluator import _coefficient_of_variation, evaluate_job
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import get_task_pack, load_oracle_source


def _job(source: str, *, kind: str = "dev", task_id: str = "row-reduction-scale") -> WorkerJob:
    task = get_task_pack(task_id)
    cases = task.dev_cases if kind == "dev" else task.sealed_cases
    return WorkerJob(
        job_id="row-reduction-triton-dev-1",
        kind=kind,  # type: ignore[arg-type]
        task=task,
        target=get_target_stack("triton-a100"),
        case_ids=tuple(case.id for case in cases),
        candidate_source=source,
        candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


def test_static_rejection_does_not_load_gpu_runtime() -> None:
    source = load_oracle_source("row-reduction-scale", "triton").replace(
        "return output", "return torch.sum(output)"
    )

    def forbidden_runtime_loader(_root: object) -> object:
        raise AssertionError("static rejection must happen before runtime loading")

    result = evaluate_job(
        _job(source),
        "/missing",
        runtime_loader=forbidden_runtime_loader,  # type: ignore[arg-type]
    )

    assert result.status == "static_check_failed"
    assert any("framework_compute_fallback" in error for error in result.static_errors)


def test_runtime_import_failure_is_a_terminal_environment_result() -> None:
    def missing_runtime(_root: object) -> object:
        raise ModuleNotFoundError("torch is not installed")

    job = _job(load_oracle_source("row-reduction-scale", "triton"))
    result = evaluate_job(
        job,
        "/missing",
        runtime_loader=missing_runtime,  # type: ignore[arg-type]
    )

    assert result.status == "environment_error"
    assert result.error == "ModuleNotFoundError: torch is not installed"
    assert result.verify_for_job(job) is result


def test_registered_baseline_skips_target_static_validation() -> None:
    def missing_runtime(_root: object) -> object:
        raise ModuleNotFoundError("runtime was reached")

    source = load_baseline_source("rmsnorm-static", "eager")
    job = _job(source, kind="baseline", task_id="rmsnorm-static")
    result = evaluate_job(
        job,
        "/missing",
        runtime_loader=missing_runtime,  # type: ignore[arg-type]
    )

    assert result.status == "environment_error"
    assert result.error == "ModuleNotFoundError: runtime was reached"


def test_agent_source_does_not_inherit_baseline_static_bypass() -> None:
    def forbidden_runtime_loader(_root: object) -> object:
        raise AssertionError("agent framework fallback must fail before runtime loading")

    source = load_baseline_source("rmsnorm-static", "eager")
    result = evaluate_job(
        _job(source, kind="dev", task_id="rmsnorm-static"),
        "/missing",
        runtime_loader=forbidden_runtime_loader,  # type: ignore[arg-type]
    )

    assert result.status == "static_check_failed"
    assert any("framework_compute_fallback" in error for error in result.static_errors)


def test_unregistered_baseline_does_not_receive_static_bypass() -> None:
    def forbidden_runtime_loader(_root: object) -> object:
        raise AssertionError("unregistered baseline must fail before runtime loading")

    source = load_baseline_source("rmsnorm-static", "eager") + "\n"
    result = evaluate_job(
        _job(source, kind="baseline", task_id="rmsnorm-static"),
        "/missing",
        runtime_loader=forbidden_runtime_loader,  # type: ignore[arg-type]
    )

    assert result.status == "worker_error"
    assert result.error == "unregistered baseline source for task: rmsnorm-static"


def test_coefficient_of_variation_uses_raw_samples() -> None:
    assert _coefficient_of_variation((1.0, 1.0, 1.0)) == 0.0
    assert _coefficient_of_variation((1.0, 2.0)) > 0.0


def test_importing_evaluator_does_not_import_torch() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import abstrak.canary.evaluator; "
            "raise SystemExit(1 if 'torch' in sys.modules else 0)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
