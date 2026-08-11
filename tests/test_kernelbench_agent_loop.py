from __future__ import annotations

import json
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from abstrak.evaluation.agent_analysis import analyze_agent_run
from abstrak.evaluation.agent_contracts import (
    AgentGenerationConfig,
    AgentModelSpec,
    KernelBenchAgentStudy,
    load_agent_study,
)
from abstrak.evaluation.agent_figures import plot_agent_run
from abstrak.evaluation.agent_provider import (
    AgentCompletion,
    AgentMessage,
    AgentOutputTruncated,
    AgentProviderError,
    PilotProviderClient,
)
from abstrak.evaluation.agent_runner import (
    AgentCollectionRunner,
    AgentEvaluationOutcome,
    SshAgentEvaluator,
    _evaluation_feedback,
    build_initial_messages,
    extract_runnable_candidate,
)
from abstrak.evaluation.agent_worker import AgentEvaluationJob
from abstrak.evaluation.contracts import (
    EvaluationResult,
    KernelBenchSource,
    KernelBenchTask,
    TargetName,
)
from abstrak.evaluation.kernelbench import TaskMaterial
from abstrak.providers.contracts import MessageRole

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _model(model_id: str = "test-model", protocol: str = "chat_completions") -> AgentModelSpec:
    return AgentModelSpec(
        id=model_id,
        protocol=protocol,
        litellm_provider="test-provider",
        api_model=f"test/{model_id}",
        api_key_env="TEST_API_KEY",
        base_url_env="TEST_BASE_URL",
    )


def _study(
    *,
    models: tuple[AgentModelSpec, ...] = (_model(),),
    targets: tuple[str, ...] = ("triton",),
    iterations: int = 3,
) -> KernelBenchAgentStudy:
    return KernelBenchAgentStudy(
        id="agent-test",
        source=KernelBenchSource(
            repository="https://example.invalid/KernelBench.git",
            commit="423217d9fda91e0c2d67e4a43bf62f96f6d104f1",
            require_clean_checkout=False,
        ),
        models=models,
        targets=targets,
        tasks=(KernelBenchTask(level=1, problem_id=1, stratum="compute"),),
        iterations=iterations,
    )


class FakeCheckout:
    def __init__(self) -> None:
        self.material = TaskMaterial(
            task=KernelBenchTask(level=1, problem_id=1, stratum="compute"),
            name="Add",
            relative_path="KernelBench/level1/1_Add.py",
            source="class Model: pass\n",
            source_sha256="source-sha",
        )

    def load_task(self, task: KernelBenchTask) -> TaskMaterial:
        assert task.ref == self.material.task.ref
        return self.material

    def zero_shot_prompt(self, material: TaskMaterial, target: str, precision: str) -> str:
        return f"Implement {material.name} for {target} at {precision}."


def _completion(
    text: str, request_id: str, *, reasoning_content: str | None = None
) -> AgentCompletion:
    return AgentCompletion(
        text=text,
        reasoning_content=reasoning_content,
        protocol="chat_completions",
        provider_request_id=request_id,
        returned_model="test-returned",
        input_tokens=10,
        output_tokens=20,
        elapsed_ms=1.5,
        sanitized_request={"model": "test/model"},
        raw_response={"id": request_id, "choices": [{"message": {"content": text}}]},
    )


def _candidate(label: str) -> str:
    return f"```python\nclass ModelNew:\n    label = {label!r}\n```"


def _result(cell_id: str, *, speedup: float | None, correct: bool = True) -> EvaluationResult:
    now = datetime.now(timezone.utc)
    return EvaluationResult(
        cell_id=cell_id,
        status="evaluated",
        backend="triton",
        precision="fp16",
        compiled=correct,
        correctness=correct,
        kernel_runtime_ms=(2.0 if correct else None),
        reference_runtime_ms=(4.0 if correct else None),
        performance_ratio=speedup,
        started_at_utc=now,
        finished_at_utc=now,
    )


class FakeClient:
    def __init__(self, completions: list[AgentCompletion | Exception]) -> None:
        self.completions = completions
        self.messages: list[tuple[AgentMessage, ...]] = []

    def complete(
        self,
        messages: list[AgentMessage],
        *,
        progress: Any = None,
    ) -> AgentCompletion:
        self.messages.append(tuple(messages))
        if progress is not None:
            progress("stream progress chunks=1 reasoning_chars=10 content_chars=0")
        item = self.completions.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeEvaluator:
    def __init__(self, speeds: list[float | None]) -> None:
        self.speeds = speeds
        self.jobs: list[AgentEvaluationJob] = []

    def evaluate(self, job: AgentEvaluationJob) -> AgentEvaluationOutcome:
        self.jobs.append(job)
        speed = self.speeds.pop(0)
        return AgentEvaluationOutcome(
            result=_result(job.cell_id, speedup=speed, correct=speed is not None)
        )


def test_extract_runnable_candidate_requires_one_complete_modelnew() -> None:
    extracted = extract_runnable_candidate(_candidate("ok"))
    assert extracted.error is None
    assert extracted.code == "class ModelNew:\n    label = 'ok'\n"
    assert extract_runnable_candidate("```python\nclass ModelNew: pass\n```\ntext").code is None
    assert extract_runnable_candidate("```python\nclass Other: pass\n```").code is None
    assert extract_runnable_candidate("no code").code is None


@pytest.mark.parametrize(
    ("target", "required_apis"),
    [
        ("triton", ("@triton.jit", "_vector_add_kernel[grid]")),
        ("tilelang", ("@T.prim_func", "tilelang.compile", "return self.kernel(x, y)")),
        (
            "cute",
            (
                "import cutlass.cute as cute",
                "@cute.kernel",
                "@cute.jit",
                ".launch(",
                "cute.compile",
                "from_dlpack",
            ),
        ),
    ],
)
def test_initial_messages_include_complete_target_card(
    target: TargetName, required_apis: tuple[str, ...]
) -> None:
    messages = build_initial_messages(
        FakeCheckout(),
        FakeCheckout().material,
        target,
        "fp16",  # type: ignore[arg-type]
    )

    assert len(messages) == 1
    assert "TARGET CONTRACT (must follow)" in messages[0].content
    assert all(required_api in messages[0].content for required_api in required_apis)
    assert "Reason concisely" in messages[0].content
    assert "## Model scaffold and launch example" in messages[0].content
    if target == "cute":
        assert "compile a custom CUDA extension" in " ".join(messages[0].content.split())


def test_pilot_study_is_the_small_24_trajectory_matrix() -> None:
    study = load_agent_study(
        REPOSITORY_ROOT / "configs" / "studies" / "kernelbench-agent-pilot.yaml"
    )
    assert study.trajectory_count == 24
    assert study.request_count == 96
    assert study.iterations == 4
    assert [(model.id, model.protocol) for model in study.models] == [
        ("deepseek-v4-flash", "chat_completions"),
        ("gpt-5.6-luna", "responses"),
    ]
    assert [task.ref for task in study.tasks] == [
        "level1-problem1",
        "level1-problem24",
        "level2-problem1",
        "level2-problem76",
    ]
    assert study.generation.reasoning_effort == "xhigh"
    assert study.generation.max_output_tokens == 65536
    assert [model.timeout_seconds for model in study.models] == [1200.0, 600.0]


def test_smoke_study_is_one_deepseek_triton_trajectory() -> None:
    study = load_agent_study(
        REPOSITORY_ROOT / "configs" / "studies" / "kernelbench-agent-smoke.yaml"
    )
    assert study.trajectory_count == 1
    assert study.request_count == 2
    assert [model.id for model in study.models] == ["deepseek-v4-flash"]
    assert study.targets == ("triton",)
    assert [task.ref for task in study.tasks] == ["level1-problem1"]
    assert study.generation.reasoning_effort == "xhigh"
    assert study.generation.max_output_tokens == 65536
    assert study.models[0].timeout_seconds == 1200.0


def test_deepseek_pilot_keeps_the_full_workload_target_matrix() -> None:
    study = load_agent_study(
        REPOSITORY_ROOT / "configs" / "studies" / "kernelbench-agent-deepseek-pilot.yaml"
    )
    assert [model.id for model in study.models] == ["deepseek-v4-flash"]
    assert study.targets == ("triton", "tilelang", "cute")
    assert [task.ref for task in study.tasks] == [
        "level1-problem1",
        "level1-problem24",
        "level2-problem1",
        "level2-problem76",
    ]
    assert study.iterations == 4
    assert study.trajectory_count == 12
    assert study.request_count == 48
    assert study.generation.max_output_tokens == 65536
    assert study.models[0].timeout_seconds == 1200.0


AFFINITY_HEADLINE_TASKS = (
    ("level1-problem36", "triton-affinity"),
    ("level1-problem47", "triton-affinity"),
    ("level1-problem41", "triton-affinity"),
    ("level1-problem3", "tilelang-affinity"),
    ("level2-problem76", "tilelang-affinity"),
    ("level2-problem99", "tilelang-affinity"),
    ("level1-problem8", "cute-affinity"),
    ("level1-problem16", "cute-affinity"),
    ("level1-problem17", "cute-affinity"),
    ("level1-problem1", "control"),
    ("level1-problem5", "control"),
    ("level1-problem7", "control"),
)


@pytest.mark.parametrize(
    ("filename", "model_ids", "tasks", "iterations", "trajectories", "requests"),
    [
        (
            "kernelbench-agent-affinity-qualification.yaml",
            ("deepseek-v4-flash",),
            (
                ("level1-problem36", "triton-affinity"),
                ("level1-problem3", "tilelang-affinity"),
                ("level1-problem17", "cute-affinity"),
                ("level1-problem1", "control"),
            ),
            2,
            12,
            24,
        ),
        (
            "kernelbench-agent-affinity-deepseek.yaml",
            ("deepseek-v4-flash",),
            AFFINITY_HEADLINE_TASKS,
            4,
            36,
            144,
        ),
        (
            "kernelbench-agent-affinity-full.yaml",
            ("deepseek-v4-flash", "gpt-5.6-luna"),
            AFFINITY_HEADLINE_TASKS,
            4,
            72,
            288,
        ),
        (
            "kernelbench-agent-affinity-stress.yaml",
            ("deepseek-v4-flash",),
            (
                ("level1-problem82", "convolution-stress"),
                ("level2-problem1", "convolution-stress"),
            ),
            4,
            6,
            24,
        ),
    ],
)
def test_affinity_study_configs_pin_the_minimal_matrices(
    filename: str,
    model_ids: tuple[str, ...],
    tasks: tuple[tuple[str, str], ...],
    iterations: int,
    trajectories: int,
    requests: int,
) -> None:
    study = load_agent_study(REPOSITORY_ROOT / "configs" / "studies" / filename)

    assert tuple(model.id for model in study.models) == model_ids
    assert study.targets == ("triton", "tilelang", "cute")
    assert tuple((task.ref, task.stratum) for task in study.tasks) == tasks
    assert study.iterations == iterations
    assert study.trajectory_count == trajectories
    assert study.request_count == requests
    assert study.generation.max_output_tokens == 65536
    assert study.generation.reasoning_effort == "xhigh"
    assert all(model.timeout_seconds == 1200.0 for model in study.models)
    assert study.evaluator.timeout_seconds == 900
    assert study.evaluator.num_correct_trials == 5
    assert study.evaluator.num_perf_trials == 100
    assert study.evaluator.timing_method == "cuda_event"
    assert study.evaluator.static_check is True


def test_runner_evaluates_each_generated_turn_and_feeds_feedback(tmp_path: Path) -> None:
    study = _study(iterations=3)
    client = FakeClient(
        [
            _completion(_candidate("first"), "r1", reasoning_content="first reasoning trace"),
            _completion("not a code block", "r2", reasoning_content="second reasoning trace"),
            _completion(_candidate("third"), "r3"),
        ]
    )
    evaluator = FakeEvaluator([1.2, 1.1])
    progress: list[str] = []
    runner = AgentCollectionRunner(
        study=study,
        checkout=FakeCheckout(),  # type: ignore[arg-type]
        provider_factory=lambda model: client,
        evaluator=evaluator,
        worker_kernelbench_root="/worker/KernelBench",
        artifact_root=tmp_path,
        run_id="run-1",
        progress=progress.append,
    )

    outcome = runner.run()

    assert outcome.attempts == 3
    assert len(evaluator.jobs) == 2
    assert evaluator.jobs[0].cell_id.endswith("i001")
    assert evaluator.jobs[1].cell_id.endswith("i003")
    assert len(client.messages) == 3
    assert [len(request) for request in client.messages] == [1, 3, 3]
    assert [message.role for message in client.messages[2]] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    assert client.messages[1][1].reasoning_content == "first reasoning trace"
    assert "speedup_vs_reference: 1.2" in client.messages[1][-1].content
    assert client.messages[2][1].content == "not a code block"
    assert client.messages[2][1].reasoning_content == "second reasoning trace"
    assert "Candidate extraction failed" in client.messages[2][-1].content
    assert "best_correct_speedup_so_far: 1.2" in client.messages[2][-1].content
    assert "label = 'first'" in client.messages[2][-1].content
    assert all(
        message.reasoning_content != "first reasoning trace" for message in client.messages[2]
    )

    attempts = [
        json.loads(line)
        for line in (outcome.run_directory / "raw" / "attempts.jsonl").read_text().splitlines()
    ]
    assert [item["generation_status"] for item in attempts] == [
        "generated",
        "parse_failure",
        "generated",
    ]
    assert [item["best_speedup"] for item in attempts] == [1.2, 1.2, 1.2]
    assert (outcome.run_directory / "raw" / "candidates").is_dir()
    assert (outcome.run_directory / "raw" / "responses").is_dir()
    assert any("iteration=1/3 provider request started" in line for line in progress)
    assert any("provider stream progress" in line for line in progress)
    assert any("iteration=1 SSH evaluation started" in line for line in progress)
    assert any("run=run-1 completed attempts=3" in line for line in progress)


def test_feedback_returns_structured_diagnostics_and_bounds_log_fallback() -> None:
    failed = _result("cell-1", speedup=None, correct=False).model_copy(
        update={
            "metadata": {
                "compilation_error_name": "TypeError",
                "compilation_error": "unexpected keyword argument 'cuda'",
                "runtime_environment": {"package_versions": {"torch": "secret-noise"}},
                "runtime_error_traceback": "unbounded traceback",
            }
        }
    )

    structured = _evaluation_feedback(failed, None, "worker log should not win")

    assert "unexpected keyword argument 'cuda'" in structured
    assert "runtime_environment" not in structured
    assert "unbounded traceback" not in structured
    assert "worker log should not win" not in structured

    log_only = "diagnostic head " + "x" * 5000 + " diagnostic tail"
    fallback = _evaluation_feedback(_result("cell-2", speedup=None, correct=False), None, log_only)
    assert "diagnostic head" in fallback
    assert "[diagnostic truncated]" in fallback
    assert "diagnostic tail" in fallback
    assert len(fallback) < 4500

    timeout = _result("cell-timeout", speedup=None, correct=False).model_copy(
        update={"status": "timeout", "error": "remote worker timed out"}
    )
    timeout_feedback = _evaluation_feedback(timeout, None, "partial compiler stderr")
    assert "remote worker timed out" in timeout_feedback
    assert "partial compiler stderr" in timeout_feedback

    successful = _evaluation_feedback(
        _result("cell-3", speedup=1.2), 1.2, "successful worker noise"
    )
    assert "successful worker noise" not in successful


def test_runner_passes_worker_diagnostic_and_retains_best_candidate(tmp_path: Path) -> None:
    study = _study(iterations=3)
    client = FakeClient(
        [
            _completion(_candidate("best"), "r1", reasoning_content="reasoning one"),
            _completion(_candidate("slower"), "r2", reasoning_content="reasoning two"),
            _completion(_candidate("final"), "r3"),
        ]
    )

    class DiagnosticEvaluator(FakeEvaluator):
        def evaluate(self, job: AgentEvaluationJob) -> AgentEvaluationOutcome:
            outcome = super().evaluate(job)
            if len(self.jobs) == 1:
                result = outcome.result.model_copy(
                    update={"metadata": {"artifact_only": "persisted diagnostic"}}
                )
                return outcome.model_copy(update={"result": result})
            if len(self.jobs) == 2:
                return outcome.model_copy(update={"log": "second attempt diagnostic"})
            return outcome

    evaluator = DiagnosticEvaluator([1.2, None, 1.1])
    AgentCollectionRunner(
        study=study,
        checkout=FakeCheckout(),  # type: ignore[arg-type]
        provider_factory=lambda model: client,
        evaluator=evaluator,
        worker_kernelbench_root="/worker/KernelBench",
        artifact_root=tmp_path,
        run_id="bounded-incumbent",
    ).run()

    third_request = client.messages[2]
    assert len(third_request) == 3
    assert third_request[1].content == _candidate("slower")
    assert third_request[1].reasoning_content == "reasoning two"
    assert "second attempt diagnostic" in third_request[2].content
    assert "best_correct_speedup_so_far: 1.2" in third_request[2].content
    assert "label = 'best'" in third_request[2].content
    assert "reasoning one" not in "\n".join(message.content for message in third_request)
    result_path = next(
        (tmp_path / "bounded-incumbent" / "raw" / "worker-logs").rglob("iteration-001.result.json")
    )
    persisted_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert persisted_result["metadata"] == {"artifact_only": "persisted diagnostic"}


def test_correct_candidate_without_timing_is_retained_as_incumbent(tmp_path: Path) -> None:
    study = _study(iterations=3)
    client = FakeClient(
        [
            _completion(_candidate("correct-unmeasured"), "r1"),
            _completion(_candidate("incorrect"), "r2"),
            _completion(_candidate("final"), "r3"),
        ]
    )

    class UnmeasuredEvaluator:
        def __init__(self) -> None:
            self.results = [
                _result("unused", speedup=None, correct=True).model_copy(
                    update={"kernel_runtime_ms": None, "reference_runtime_ms": None}
                ),
                _result("unused", speedup=None, correct=False),
                _result("unused", speedup=1.1),
            ]

        def evaluate(self, job: AgentEvaluationJob) -> AgentEvaluationOutcome:
            return AgentEvaluationOutcome(
                result=self.results.pop(0).model_copy(update={"cell_id": job.cell_id})
            )

    AgentCollectionRunner(
        study=study,
        checkout=FakeCheckout(),  # type: ignore[arg-type]
        provider_factory=lambda model: client,
        evaluator=UnmeasuredEvaluator(),
        worker_kernelbench_root="/worker/KernelBench",
        artifact_root=tmp_path,
        run_id="unmeasured-incumbent",
    ).run()

    third_feedback = client.messages[2][-1].content
    assert "label = 'correct-unmeasured'" in third_feedback
    assert "best_correct_speedup_so_far" not in third_feedback


def test_provider_error_stops_one_trajectory_but_matrix_continues(tmp_path: Path) -> None:
    study = _study(targets=("triton", "cute"), iterations=1)
    provider_error = AgentProviderError(
        "temporary failure",
        elapsed_ms=3.0,
        sanitized_request={"model": "test/model"},
        raw_response={
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 13,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
            "stream": {
                "chunk_count": 7,
                "reasoning_chars": 31,
                "content_chars": 0,
                "completed": False,
            }
        },
    )
    client = FakeClient(
        [
            provider_error,
            _completion(_candidate("cute"), "r2"),
        ]
    )
    evaluator = FakeEvaluator([1.5])
    result = AgentCollectionRunner(
        study=study,
        checkout=FakeCheckout(),  # type: ignore[arg-type]
        provider_factory=lambda model: client,
        evaluator=evaluator,
        worker_kernelbench_root="/worker/KernelBench",
        artifact_root=tmp_path,
        run_id="run-2",
    ).run()
    assert result.attempts == 2
    assert result.generation_status_counts == {"generated": 1, "provider_error": 1}
    assert len(evaluator.jobs) == 1
    attempts = [
        json.loads(line)
        for line in (result.run_directory / "raw" / "attempts.jsonl").read_text().splitlines()
    ]
    assert attempts[0]["provider_elapsed_ms"] == 3.0
    assert attempts[0]["input_tokens"] == 11
    assert attempts[0]["output_tokens"] == 13
    assert attempts[0]["reasoning_tokens"] == 7
    response = json.loads(
        (
            result.run_directory
            / "raw"
            / "responses"
            / "test-model--level1-problem1--triton"
            / "iteration-001.json"
        ).read_text()
    )
    assert response["sanitized_request"] == {"model": "test/model"}
    assert response["raw_response"]["stream"]["reasoning_chars"] == 31


def test_output_truncation_consumes_iteration_and_retries_without_empty_assistant(
    tmp_path: Path,
) -> None:
    study = _study(iterations=2)
    truncated = AgentOutputTruncated(
        "chat response exhausted max_tokens before emitting final text (finish_reason=length)",
        elapsed_ms=4.0,
        sanitized_request={"max_tokens": 65536},
        raw_response={
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": ""},
                }
            ],
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 65536,
                "completion_tokens_details": {"reasoning_tokens": 65536},
            },
        },
    )
    client = FakeClient([truncated, _completion(_candidate("recovered"), "r2")])
    result = AgentCollectionRunner(
        study=study,
        checkout=FakeCheckout(),  # type: ignore[arg-type]
        provider_factory=lambda model: client,
        evaluator=FakeEvaluator([1.2]),
        worker_kernelbench_root="/worker/KernelBench",
        artifact_root=tmp_path,
        run_id="truncated-retry",
    ).run()

    assert result.generation_status_counts == {"generated": 1, "output_truncated": 1}
    assert len(client.messages) == 2
    assert [message.role for message in client.messages[1]] == [MessageRole.USER]
    assert "TARGET CONTRACT (must follow)" in client.messages[1][0].content
    assert "exhausted its output budget" in client.messages[1][0].content
    assert "immediately return" in client.messages[1][0].content
    attempts = [
        json.loads(line)
        for line in (result.run_directory / "raw" / "attempts.jsonl").read_text().splitlines()
    ]
    assert [item["generation_status"] for item in attempts] == [
        "output_truncated",
        "generated",
    ]
    assert attempts[0]["input_tokens"] == 17
    assert attempts[0]["output_tokens"] == 65536
    assert attempts[0]["reasoning_tokens"] == 65536
    truncated_response = json.loads(
        (
            result.run_directory
            / "raw"
            / "responses"
            / "test-model--level1-problem1--triton"
            / "iteration-001.json"
        ).read_text()
    )
    assert truncated_response["error_kind"] == "output_truncated"
    assert truncated_response["raw_response"]["choices"][0]["finish_reason"] == "length"


def test_collector_artifacts_feed_real_analysis_and_plot(tmp_path: Path) -> None:
    study = _study(iterations=2)
    client = FakeClient(
        [
            _completion(_candidate("first"), "r1"),
            _completion(_candidate("second"), "r2"),
        ]
    )
    collection = AgentCollectionRunner(
        study=study,
        checkout=FakeCheckout(),  # type: ignore[arg-type]
        provider_factory=lambda model: client,
        evaluator=FakeEvaluator([1.1, 1.3]),
        worker_kernelbench_root="/worker/KernelBench",
        artifact_root=tmp_path,
        run_id="end-to-end",
    ).run()

    metrics, metrics_json, metrics_csv = analyze_agent_run(collection.run_directory)
    figures = plot_agent_run(collection.run_directory)
    run_payload = json.loads(
        (collection.run_directory / "raw" / "run.json").read_text(encoding="utf-8")
    )

    assert metrics["curve_rows"][-1]["best_speedup"] == 1.3
    assert run_payload["worker"] == {
        "kind": "injected",
        "kernelbench_root": "/worker/KernelBench",
        "device": "cuda:0",
    }
    assert metrics_json.is_file()
    assert metrics_csv.is_file()
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)


def test_ssh_evaluator_serializes_one_job(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}
    result = _result("cell-1", speedup=2.0).model_dump(mode="json")

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        observed["command"] = command
        observed["input"] = kwargs["input"]
        return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr="worker log\n")

    monkeypatch.setattr("abstrak.evaluation.agent_runner.subprocess.run", fake_run)
    evaluator = SshAgentEvaluator(
        host="a100-r1",
        port=2222,
        worker_root="/srv/AbstraK",
        worker_python="/srv/venv/bin/python",
    )
    job = AgentEvaluationJob(
        cell_id="cell-1",
        task_level=1,
        problem_id=1,
        target="triton",
        precision="fp16",
        candidate_source="class ModelNew: pass\n",
        kernelbench_root="/srv/KernelBench",
        device="cuda:3",
        evaluator=_study().evaluator,
    )
    outcome = evaluator.evaluate(job)
    assert outcome.result.performance_ratio == 2.0
    assert json.loads(observed["input"])["cell_id"] == "cell-1"
    assert observed["command"][:5] == ["ssh", "-o", "BatchMode=yes", "-p", "2222"]
    assert "abstrak.evaluation.agent_worker" in observed["command"][-1]
    assert evaluator.binding == {
        "kind": "ssh",
        "host": "a100-r1",
        "port": 2222,
        "worker_root": "/srv/AbstraK",
        "worker_python": "/srv/venv/bin/python",
    }


class FakeTransport:
    call_count = 0

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.call_count += 1
        return self.payload

    def responses(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.call_count += 1
        return self.payload


class FalsyFakeTransport(FakeTransport):
    def __bool__(self) -> bool:
        return False


@pytest.mark.parametrize(
    ("protocol", "payload"),
    [
        (
            "chat_completions",
            {
                "id": "chat-1",
                "model": "returned-chat",
                "choices": [
                    {
                        "message": {
                            "content": _candidate("chat"),
                            "reasoning_content": "chat reasoning trace",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            },
        ),
        (
            "responses",
            {
                "id": "response-1",
                "model": "returned-response",
                "output_text": _candidate("response"),
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 6,
                    "output_tokens_details": {"reasoning_tokens": 7},
                },
            },
        ),
    ],
)
def test_pilot_provider_uses_requested_protocol_and_xhigh(
    protocol: str, payload: dict[str, Any]
) -> None:
    transport = FakeTransport(payload)
    client = PilotProviderClient(
        _model(protocol=protocol),
        AgentGenerationConfig(),
        environment={"TEST_API_KEY": "secret", "TEST_BASE_URL": "https://provider.invalid"},
        transport=transport,
    )
    completion = client.complete([AgentMessage(role=MessageRole.USER, content="prompt")])
    request = transport.calls[0]
    assert completion.text.startswith("```python")
    token_key = "max_output_tokens" if protocol == "responses" else "max_completion_tokens"
    assert request[token_key] == 16384
    assert request["stream"] is (protocol == "chat_completions")
    if protocol == "responses":
        assert request["reasoning"] == {"effort": "xhigh"}
        assert completion.reasoning_content is None
    else:
        assert request["reasoning_effort"] == "xhigh"
        assert completion.reasoning_content == "chat reasoning trace"
    assert "reasoning_content" not in request.get("messages", request.get("input"))[0]
    assert request["api_key"] == "secret"
    assert completion.sanitized_request["api_key_env"] == "TEST_API_KEY"
    assert "secret" not in json.dumps(completion.sanitized_request)


def test_pilot_provider_uses_deepseek_wire_parameters_and_aggregates_stream() -> None:
    chunks = [
        {
            "id": "chat-stream-1",
            "created": 1,
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {"role": "assistant", "reasoning_content": "reason "},
                }
            ],
        },
        {
            "id": "chat-stream-1",
            "created": 1,
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {"reasoning_content": "trace"},
                }
            ],
        },
        {
            "id": "chat-stream-1",
            "created": 1,
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "delta": {"content": _candidate("streamed")},
                }
            ],
        },
        {
            "id": "chat-stream-1",
            "created": 1,
            "model": "deepseek-v4-flash",
            "choices": [],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 11,
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        },
    ]
    transport = FakeTransport(iter(chunks))
    deepseek = _model("deepseek-v4-flash").model_copy(
        update={
            "litellm_provider": "deepseek",
            "api_model": "deepseek/deepseek-v4-flash",
        }
    )
    progress: list[str] = []
    client = PilotProviderClient(
        deepseek,
        AgentGenerationConfig(),
        environment={"TEST_API_KEY": "secret", "TEST_BASE_URL": "https://provider.invalid"},
        transport=transport,
    )

    completion = client.complete(
        [AgentMessage(role=MessageRole.USER, content="prompt")],
        progress=progress.append,
    )

    request = transport.calls[0]
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert request["max_tokens"] == 16384
    assert "max_completion_tokens" not in request
    assert "reasoning_effort" not in request
    assert request["extra_body"] == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }
    assert completion.text == _candidate("streamed")
    assert completion.reasoning_content == "reason trace"
    assert completion.output_tokens == 11
    assert completion.reasoning_tokens == 5
    assert completion.raw_response["stream"] == {
        "chunk_count": 4,
        "reasoning_chars": len("reason trace"),
        "content_chars": len(_candidate("streamed")),
        "completed": True,
    }
    assert progress[0].startswith("stream progress")
    assert progress[-1].startswith("stream completed")
    assert "reason trace" not in "\n".join(progress)


def test_pilot_provider_preserves_partial_stream_metadata_on_error() -> None:
    def interrupted_stream():
        yield {
            "id": "chat-stream-partial",
            "created": 1,
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {"role": "assistant", "reasoning_content": "partial"},
                }
            ],
            "usage": {
                "prompt_tokens": 23,
                "completion_tokens": 29,
                "completion_tokens_details": {"reasoning_tokens": 19},
            },
        }
        raise TimeoutError("read stalled")

    transport = FakeTransport(interrupted_stream())
    deepseek = _model("deepseek-v4-flash").model_copy(
        update={
            "litellm_provider": "deepseek",
            "api_model": "deepseek/deepseek-v4-flash",
        }
    )
    client = PilotProviderClient(
        deepseek,
        AgentGenerationConfig(),
        environment={"TEST_API_KEY": "secret", "TEST_BASE_URL": "https://provider.invalid"},
        transport=transport,
    )

    with pytest.raises(AgentProviderError) as raised:
        client.complete([AgentMessage(role=MessageRole.USER, content="prompt")])

    error = raised.value
    assert "TimeoutError: read stalled" in str(error)
    assert error.raw_response is not None
    assert error.raw_response["stream"] == {
        "chunk_count": 1,
        "reasoning_chars": len("partial"),
        "content_chars": 0,
        "completed": False,
    }
    assert error.raw_response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "",
    }
    assert error.sanitized_request is not None
    assert error.usage.input_tokens == 23
    assert error.usage.output_tokens == 29
    assert error.usage.reasoning_tokens == 19


def test_pilot_provider_enforces_wall_clock_deadline() -> None:
    if (
        not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
        or signal.getitimer(signal.ITIMER_REAL)[0] > 0
    ):
        pytest.skip("wall-clock signal deadline is unavailable in this test process")

    class BlockingTransport(FakeTransport):
        def chat_completion(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            self.call_count += 1
            time.sleep(0.2)
            return self.payload

    transport = BlockingTransport({"choices": [{"message": {"content": "late"}}]})
    model = _model().model_copy(update={"timeout_seconds": 0.02})
    client = PilotProviderClient(
        model,
        AgentGenerationConfig(),
        environment={"TEST_API_KEY": "secret", "TEST_BASE_URL": "https://provider.invalid"},
        transport=transport,
    )

    with pytest.raises(AgentProviderError, match="wall-clock deadline exceeded 0.02s") as raised:
        client.complete([AgentMessage(role=MessageRole.USER, content="prompt")])

    assert raised.value.raw_response is None
    assert raised.value.sanitized_request is not None


def test_pilot_provider_classifies_reasoning_only_length_stream() -> None:
    chunks = [
        {
            "id": "chat-stream-truncated",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {"role": "assistant", "reasoning_content": "still thinking"},
                }
            ],
        },
        {
            "id": "chat-stream-truncated",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "delta": {},
                }
            ],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 65536,
                "completion_tokens_details": {"reasoning_tokens": 65536},
            },
        },
    ]
    transport = FakeTransport(iter(chunks))
    deepseek = _model("deepseek-v4-flash").model_copy(
        update={
            "litellm_provider": "deepseek",
            "api_model": "deepseek/deepseek-v4-flash",
        }
    )
    client = PilotProviderClient(
        deepseek,
        AgentGenerationConfig(),
        environment={"TEST_API_KEY": "secret", "TEST_BASE_URL": "https://provider.invalid"},
        transport=transport,
    )

    with pytest.raises(AgentOutputTruncated) as raised:
        client.complete([AgentMessage(role=MessageRole.USER, content="prompt")])

    error = raised.value
    assert "exhausted max_tokens" in str(error)
    assert error.raw_response is not None
    assert error.raw_response["choices"][0]["finish_reason"] == "length"
    assert error.raw_response["choices"][0]["message"]["content"] == ""
    assert error.raw_response["usage"]["completion_tokens"] == 65536
    assert error.sanitized_request is not None
    assert error.usage.input_tokens == 7
    assert error.usage.output_tokens == 65536
    assert error.usage.reasoning_tokens == 65536


def test_pilot_provider_classifies_partial_chat_text_as_truncated() -> None:
    payload = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "```python\nclass ModelNew:"},
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 31},
    }
    client = PilotProviderClient(
        _model(),
        AgentGenerationConfig(),
        environment={"TEST_API_KEY": "secret", "TEST_BASE_URL": "https://provider.invalid"},
        transport=FakeTransport(payload),
    )

    with pytest.raises(AgentOutputTruncated) as raised:
        client.complete([AgentMessage(role=MessageRole.USER, content="prompt")])

    assert raised.value.usage.input_tokens == 5
    assert raised.value.usage.output_tokens == 31


def test_pilot_provider_classifies_incomplete_responses_output_as_truncated() -> None:
    payload = {
        "id": "response-incomplete",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output_text": "```python\nclass ModelNew:",
        "usage": {"input_tokens": 37, "output_tokens": 41},
    }
    client = PilotProviderClient(
        _model(protocol="responses"),
        AgentGenerationConfig(),
        environment={"TEST_API_KEY": "secret", "TEST_BASE_URL": "https://provider.invalid"},
        transport=FakeTransport(payload),
    )

    with pytest.raises(AgentOutputTruncated) as raised:
        client.complete([AgentMessage(role=MessageRole.USER, content="prompt")])

    assert raised.value.usage.input_tokens == 37
    assert raised.value.usage.output_tokens == 41


def test_pilot_provider_keeps_an_explicit_falsy_transport() -> None:
    payload = {
        "choices": [{"message": {"content": _candidate("offline")}}],
        "usage": {},
    }
    transport = FalsyFakeTransport(payload)
    client = PilotProviderClient(
        _model(),
        AgentGenerationConfig(),
        environment={"TEST_API_KEY": "secret", "TEST_BASE_URL": "https://provider.invalid"},
        transport=transport,
    )
    client.complete([AgentMessage(role=MessageRole.USER, content="prompt")])
    assert transport.call_count == 1


def test_pilot_provider_replays_reasoning_content_for_chat_history() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": _candidate("next"),
                    "provider_specific_fields": {"reasoning_content": "next reasoning trace"},
                }
            }
        ],
        "usage": {},
    }
    transport = FakeTransport(payload)
    client = PilotProviderClient(
        _model(),
        AgentGenerationConfig(),
        environment={"TEST_API_KEY": "secret", "TEST_BASE_URL": "https://provider.invalid"},
        transport=transport,
    )

    completion = client.complete(
        [
            AgentMessage(role=MessageRole.USER, content="prompt"),
            AgentMessage(
                role=MessageRole.ASSISTANT,
                content=_candidate("previous"),
                reasoning_content="previous reasoning trace",
            ),
            AgentMessage(role=MessageRole.USER, content="evaluation feedback"),
        ]
    )

    assert completion.reasoning_content == "next reasoning trace"
    assert transport.calls[0]["messages"][1]["reasoning_content"] == ("previous reasoning trace")
