from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from abstrak.evaluation import agent_worker, worker
from abstrak.evaluation.contracts import EvaluationResult


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(device: object) -> str:
        return f"fake-a100:{device}"


class FakeTorch:
    __version__ = "test-torch"
    version = SimpleNamespace(cuda="test-cuda")
    cuda = FakeCuda()

    @staticmethod
    def device(value: str) -> str:
        return f"device:{value}"


class FakeKernelEval:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def get_torch_dtype_from_string(precision: str) -> str:
        return f"dtype:{precision}"

    def eval_kernel_against_ref(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            compiled=True,
            correctness=True,
            runtime=2.0,
            ref_runtime=5.0,
            metadata={"fake_execution": True},
        )


def _reference_checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "KernelBench"
    task = root / "KernelBench" / "level1" / "1_Add.py"
    task.parent.mkdir(parents=True)
    source = "class Model: pass\n"
    task.write_text(source, encoding="utf-8")
    return root, source


def _evaluation_result(cell_id: str = "attempt-1") -> EvaluationResult:
    now = datetime.now(timezone.utc)
    return EvaluationResult(
        cell_id=cell_id,
        status="evaluated",
        backend="triton",
        precision="fp16",
        compiled=True,
        correctness=True,
        kernel_runtime_ms=2.0,
        reference_runtime_ms=5.0,
        performance_ratio=2.5,
        fast_0=True,
        fast_1=True,
        fast_2=True,
        started_at_utc=now,
        finished_at_utc=now,
    )


def _job_payload(kernelbench_root: str = "/remote/KernelBench") -> dict[str, Any]:
    return {
        "schema_version": "kernelbench-agent-evaluation-job.v1",
        "cell_id": "attempt-1",
        "task_level": 1,
        "problem_id": 1,
        "target": "triton",
        "precision": "fp16",
        "candidate_source": "class ModelNew: pass\n",
        "kernelbench_root": kernelbench_root,
        "device": "cuda:3",
        "evaluator": {
            "num_correct_trials": 7,
            "num_perf_trials": 23,
            "timing_method": "cuda_event",
            "timeout_seconds": 91,
            "excessive_speedup_threshold": 8.0,
            "static_check": True,
        },
    }


def test_task_candidate_loads_reference_and_uses_shared_kernelbench_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, reference_source = _reference_checkout(tmp_path)
    kernel_eval = FakeKernelEval()
    static_calls: list[tuple[str, str, str]] = []

    def validate_static(source: str, *, backend: str, precision: str) -> object:
        static_calls.append((source, backend, precision))
        return True, [], ["fake warning"]

    monkeypatch.setattr(
        worker,
        "_load_kernelbench_runtime",
        lambda unused_root: (FakeTorch(), kernel_eval, validate_static),
    )

    result = worker.evaluate_kernelbench_task_candidate(
        cell_id="attempt-1",
        task_level=1,
        problem_id=1,
        target="triton",
        precision="fp16",
        candidate_source="class ModelNew: pass\n",
        kernelbench_root=root,
        device="cuda:3",
        num_correct_trials=7,
        num_perf_trials=23,
        timing_method="cuda_event",
        excessive_speedup_threshold=8.0,
        static_check=True,
    )

    assert result.status == "evaluated"
    assert result.compiled is True
    assert result.correctness is True
    assert result.performance_ratio == pytest.approx(2.5)
    assert result.static_warnings == ("fake warning",)
    assert static_calls == [("class ModelNew: pass\n", "triton", "fp16")]
    assert len(kernel_eval.calls) == 1
    call = kernel_eval.calls[0]
    assert call["original_model_src"] == reference_source
    assert call["custom_model_src"] == "class ModelNew: pass\n"
    assert call["num_correct_trials"] == 7
    assert call["num_perf_trials"] == 23
    assert call["device"] == "device:cuda:3"
    assert call["backend"] == "triton"
    assert call["precision"] == "dtype:fp16"


def test_static_failure_does_not_call_kernelbench_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _reference_checkout(tmp_path)
    kernel_eval = FakeKernelEval()
    monkeypatch.setattr(
        worker,
        "_load_kernelbench_runtime",
        lambda unused_root: (
            FakeTorch(),
            kernel_eval,
            lambda source, **kwargs: (False, ["target API missing"], []),
        ),
    )

    result = worker.evaluate_kernelbench_task_candidate(
        cell_id="attempt-static",
        task_level=1,
        problem_id=1,
        target="tilelang",
        precision="bf16",
        candidate_source="not executed",
        kernelbench_root=root,
        device="cuda:0",
        num_correct_trials=5,
        num_perf_trials=100,
        timing_method="cuda_event",
        excessive_speedup_threshold=10.0,
        static_check=True,
    )

    assert result.status == "static_check_failed"
    assert result.static_errors == ("target API missing",)
    assert kernel_eval.calls == []


def test_missing_task_returns_compatible_harness_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        worker,
        "_load_kernelbench_runtime",
        lambda unused_root: pytest.fail("runtime must not load without a reference"),
    )

    result = worker.evaluate_kernelbench_task_candidate(
        cell_id="attempt-missing",
        task_level=2,
        problem_id=76,
        target="cute",
        precision="fp16",
        candidate_source="not executed",
        kernelbench_root=tmp_path,
        device="cuda:0",
        num_correct_trials=5,
        num_perf_trials=100,
        timing_method="cuda_event",
        excessive_speedup_threshold=10.0,
        static_check=True,
    )

    assert result.status == "harness_error"
    assert result.cell_id == "attempt-missing"
    assert "cannot load KernelBench reference" in (result.error or "")


def test_existing_cell_worker_delegates_to_shared_source_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = tmp_path / "cell"
    cell.mkdir()
    (cell / "cell.json").write_text(
        json.dumps(
            {
                "schema_version": "kernelbench-naive-cell.v1",
                "cell_id": "naive-cell",
                "study_id": "study",
                "study_sha256": "study-sha",
                "profile": "model",
                "target": "cute",
                "precision": "bf16",
                "task": {"level": 1, "problem_id": 1, "stratum": "compute"},
                "task_name": "add",
                "task_source_sha256": "source-sha",
                "prompt_sha256": "prompt-sha",
                "replicate": 0,
            }
        ),
        encoding="utf-8",
    )
    (cell / "reference.py").write_text("reference source\n", encoding="utf-8")
    (cell / "candidate.py").write_text("candidate source\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_evaluate(**kwargs: Any) -> EvaluationResult:
        captured.update(kwargs)
        return _evaluation_result("naive-cell").model_copy(
            update={"backend": "cute", "precision": "bf16"}
        )

    monkeypatch.setattr(worker, "evaluate_candidate_source", fake_evaluate)

    result = worker.evaluate_cell(
        cell,
        "/remote/KernelBench",
        device="cuda:2",
        num_correct_trials=3,
        num_perf_trials=17,
        timing_method="host_time",
        excessive_speedup_threshold=9.0,
        static_check=False,
    )

    assert result.cell_id == "naive-cell"
    assert captured == {
        "cell_id": "naive-cell",
        "target": "cute",
        "precision": "bf16",
        "reference_source": "reference source\n",
        "candidate_source": "candidate source\n",
        "kernelbench_root": "/remote/KernelBench",
        "device": "cuda:2",
        "num_correct_trials": 3,
        "num_perf_trials": 17,
        "timing_method": "host_time",
        "excessive_speedup_threshold": 9.0,
        "static_check": False,
    }


def test_agent_job_maps_all_worker_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_evaluate(**kwargs: Any) -> EvaluationResult:
        captured.update(kwargs)
        return _evaluation_result(kwargs["cell_id"])

    monkeypatch.setattr(agent_worker, "evaluate_kernelbench_task_candidate", fake_evaluate)
    job = agent_worker.AgentEvaluationJob.model_validate(_job_payload())

    result = agent_worker.evaluate_agent_job(job)

    assert result.cell_id == "attempt-1"
    assert captured == {
        "cell_id": "attempt-1",
        "task_level": 1,
        "problem_id": 1,
        "target": "triton",
        "precision": "fp16",
        "candidate_source": "class ModelNew: pass\n",
        "kernelbench_root": "/remote/KernelBench",
        "device": "cuda:3",
        "num_correct_trials": 7,
        "num_perf_trials": 23,
        "timing_method": "cuda_event",
        "excessive_speedup_threshold": 8.0,
        "static_check": True,
    }


def test_agent_worker_reads_one_json_job_and_writes_one_json_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[agent_worker.AgentEvaluationJob] = []

    def fake_evaluate(job: agent_worker.AgentEvaluationJob) -> EvaluationResult:
        seen.append(job)
        return _evaluation_result(job.cell_id)

    monkeypatch.setattr(agent_worker, "evaluate_agent_job", fake_evaluate)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_job_payload())))

    status = agent_worker.main([])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 0
    assert captured.err == ""
    assert payload["cell_id"] == "attempt-1"
    assert payload["performance_ratio"] == 2.5
    assert len(seen) == 1
    assert seen[0].evaluator.timeout_seconds == 91


def test_agent_worker_rejects_invalid_json_without_evaluation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO('{"unexpected": true}'))
    monkeypatch.setattr(
        agent_worker,
        "evaluate_agent_job",
        lambda job: pytest.fail("invalid input must not reach evaluation"),
    )

    status = agent_worker.main([])

    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert "invalid agent evaluation job" in captured.err
