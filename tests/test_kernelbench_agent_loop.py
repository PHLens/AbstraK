from __future__ import annotations

import json
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
    AgentProviderError,
    PilotProviderClient,
)
from abstrak.evaluation.agent_runner import (
    AgentCollectionRunner,
    AgentEvaluationOutcome,
    SshAgentEvaluator,
    extract_runnable_candidate,
)
from abstrak.evaluation.agent_worker import AgentEvaluationJob
from abstrak.evaluation.contracts import EvaluationResult, KernelBenchSource, KernelBenchTask
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

    def complete(self, messages: list[AgentMessage]) -> AgentCompletion:
        self.messages.append(tuple(messages))
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
    assert study.generation.reasoning_effort == "xhigh"


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


def test_deepseek_pilot_keeps_the_full_workload_target_matrix() -> None:
    study = load_agent_study(
        REPOSITORY_ROOT / "configs" / "studies" / "kernelbench-agent-deepseek-pilot.yaml"
    )
    assert [model.id for model in study.models] == ["deepseek-v4-flash"]
    assert study.targets == ("triton", "tilelang", "cute")
    assert [task.ref for task in study.tasks] == [
        "level1-problem1",
        "level1-problem3",
        "level1-problem40",
        "level2-problem76",
    ]
    assert study.iterations == 4
    assert study.trajectory_count == 12
    assert study.request_count == 48


def test_runner_evaluates_each_generated_turn_and_feeds_feedback(tmp_path: Path) -> None:
    study = _study(iterations=3)
    client = FakeClient(
        [
            _completion(_candidate("first"), "r1", reasoning_content="first reasoning trace"),
            _completion("not a code block", "r2"),
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
    assert client.messages[1][-2].reasoning_content == "first reasoning trace"
    assert "speedup_vs_reference: 1.2" in client.messages[1][-1].content
    assert "Candidate extraction failed" in client.messages[2][-1].content
    assert "best_correct_speedup_so_far: 1.2" in client.messages[2][-1].content

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
    assert any("iteration=1 SSH evaluation started" in line for line in progress)
    assert any("run=run-1 completed attempts=3" in line for line in progress)


def test_provider_error_stops_one_trajectory_but_matrix_continues(tmp_path: Path) -> None:
    study = _study(targets=("triton", "cute"), iterations=1)
    client = FakeClient(
        [
            AgentProviderError("temporary failure", elapsed_ms=3.0),
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

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        self.call_count += 1
        return self.payload

    def responses(self, **kwargs: Any) -> dict[str, Any]:
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
