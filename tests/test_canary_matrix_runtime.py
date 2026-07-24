from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from abstrak.canary.contracts import TimingSpec
from abstrak.canary.manifests import PinnedStudySpec
from abstrak.canary.matrix import MatrixStudySpec, PhaseSpec, TaskGroupSpec, build_matrix_schedule
from abstrak.canary.matrix_runner import (
    MatrixExecutionContext,
    MatrixTransportContext,
    MatrixWorkerBinding,
    build_matrix_phase_contract,
)
from abstrak.canary.matrix_runtime import (
    MatrixRuntimeAuthorization,
    MatrixRuntimeError,
    MatrixStudyRuntime,
    RuntimeTaskAuthorization,
    build_authorized_ssh_worker,
    read_clean_controller_revision,
)
from abstrak.canary.tasks import CAPABILITY_GATE_ASSET_ROOT
from abstrak.providers.manifests import ManifestBundle, completion_client_identity


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _pinned(tmp_path: Path) -> PinnedStudySpec:
    spec = MatrixStudySpec(
        study_id="matrix-runtime-test",
        seed=20260724,
        agents=("fake-agent",),
        targets=("tilelang-a100-core",),
        task_groups=(TaskGroupSpec(id="base", task_ids=("gelu-static",)),),
        phases=(
            PhaseSpec(
                id="core",
                task_ids=("gelu-static",),
                replicates=(1,),
                order_policy="fixed",
                max_calls_per_trajectory=3,
                infrastructure_retries=1,
            ),
        ),
    )
    return PinnedStudySpec(
        path=tmp_path / "study.json",
        sha256=_digest("matrix-runtime-study"),
        spec=spec,
    )


def _context() -> MatrixExecutionContext:
    return MatrixExecutionContext(
        controller_revision="1" * 40,
        worker_revision="2" * 40,
        transport=MatrixTransportContext(
            host="root@gpu.example",
            port=30554,
            worker_root="/srv/AbstraK",
            python_executable="/srv/venv/bin/python",
            pythonpath="/srv/AbstraK/src",
            kernelbench_root="/srv/KernelBench",
            asset_root="/srv/AbstraK/benchmarks/capability-gate-a100",
            sandbox="setpriv-supervised",
            device="cuda:0",
            timeout_seconds=300.0,
            network_isolated=False,
            filesystem_read_only=False,
        ),
        asset_manifest_sha256=_digest("assets"),
        floor_manifest_sha256=_digest("floor"),
        environment_manifest_sha256=_digest("environment"),
    )


def _authorization(pinned: PinnedStudySpec) -> MatrixRuntimeAuthorization:
    schedule = build_matrix_schedule(pinned.spec)
    return MatrixRuntimeAuthorization(
        study_id=pinned.spec.study_id,
        raw_study_sha256=pinned.sha256,
        spec_sha256=pinned.spec.sha256,
        schedule_sha256=schedule.sha256,
        execution_context=_context(),
        accelerator="Fake A100",
        compute_capability="8.0",
        python_version="3.10.20",
        tilelang_version="0.1.12",
        triton_version="3.7.1",
        torch_version="2.13.0+cu126",
        cuda_version="12.6",
        driver_version="570.00",
        tasks=(
            RuntimeTaskAuthorization(
                task_id="gelu-static",
                latency_ceiling_ms=1.25,
            ),
        ),
    )


class _FakeClient:
    def __init__(self, bundle: ManifestBundle) -> None:
        self.completion_identity = completion_client_identity(bundle)
        self.artifact_secrets = ("provider-secret",)


class _FakeWorker:
    def __init__(self, context: MatrixExecutionContext) -> None:
        self.matrix_worker_binding = MatrixWorkerBinding(
            worker_revision=context.worker_revision,
            transport=context.transport,
        )
        self.validated_devices: list[str] = []

    def validate_environment(self, device: str) -> dict[str, object]:
        self.validated_devices.append(device)
        return {"status": "healthy", "device": device}


def _runtime(
    pinned: PinnedStudySpec,
    bundle: ManifestBundle,
):
    client_calls: list[tuple[str, ManifestBundle]] = []
    worker_calls: list[bool] = []

    def client_factory(agent_id: str, normalized: ManifestBundle):
        client_calls.append((agent_id, normalized))
        return _FakeClient(normalized)

    def worker_factory():
        worker_calls.append(True)
        return _FakeWorker(_context())

    runtime = MatrixStudyRuntime(
        pinned=pinned,
        schedule=build_matrix_schedule(pinned.spec),
        authorization=_authorization(pinned),
        agent_bundles={"fake-agent": bundle},
        client_factory=client_factory,
        worker_factory=worker_factory,
        controller_root=pinned.path.parent,
        controller_revision_reader=lambda _root: "1" * 40,
        asset_root=CAPABILITY_GATE_ASSET_ROOT,
        dev_timing=TimingSpec(warmup_runs=5, trial_runs=20, repetitions=1),
    )
    return runtime, client_calls, worker_calls


def test_runtime_resolvers_freeze_policy_prompt_and_agent_identity(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
) -> None:
    pinned = _pinned(tmp_path)
    runtime, client_calls, worker_calls = _runtime(pinned, manifest_bundle)
    cell = build_matrix_schedule(pinned.spec).cells_for_phase("core")[0]

    task = runtime.resolve_task(cell.task_id)
    target = runtime.resolve_target(cell.target_id)
    agent = runtime.resolve_agent(cell.agent_id)
    execution = runtime.resolve_execution(cell)

    assert task.id == "gelu-static"
    assert target.id == "tilelang-a100-core"
    assert agent.id == "fake-agent"
    assert execution.budget.max_calls == 3
    assert execution.budget.max_completion_tokens_per_call == 8192
    assert execution.budget.max_wall_seconds == 1200.0
    assert execution.policy.response_parser == "candidate_only"
    assert execution.policy.stop_policy == "correct_latency"
    assert execution.policy.final_selection == "best_correct_latency"
    assert execution.policy.latency_ceiling_ms == 1.25
    assert execution.dev_timing == TimingSpec(
        warmup_runs=5,
        trial_runs=20,
        repetitions=1,
    )
    assert execution.execution_context_sha256 == _context().sha256
    assert client_calls == []
    assert worker_calls == []


def test_runtime_factory_lazily_reuses_bound_client_and_worker(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
) -> None:
    pinned = _pinned(tmp_path)
    runtime, client_calls, worker_calls = _runtime(pinned, manifest_bundle)
    contract = build_matrix_phase_contract(
        pinned,
        "core",
        execution_context=_context(),
        resolve_task=runtime.resolve_task,
        resolve_target=runtime.resolve_target,
        resolve_agent=runtime.resolve_agent,
        resolve_execution=runtime.resolve_execution,
    )
    identity = contract.plan.cells[0].identity

    first = runtime.runtime_for(identity)
    second = runtime.runtime_for(identity)

    assert first.client is second.client
    assert first.worker is second.worker
    assert first.worker.validated_devices == ["cuda:0"]
    assert first.artifact_secrets == ("provider-secret",)
    assert first.agent_binding.completion == first.client.completion_identity
    assert first.execution == runtime.resolve_execution(identity.cell)
    assert "sealed-random" not in first.initial_messages[1].content
    assert len(client_calls) == 1
    assert client_calls[0][0] == "fake-agent"
    normalized = client_calls[0][1]
    assert normalized.model.generation.max_completion_tokens == 8192
    assert normalized.model.generation.temperature == 0
    assert normalized.model.allow_live_probe is False
    assert normalized.model.output_contract == "plain_text"
    assert worker_calls == [True]


def test_runtime_validates_worker_before_constructing_provider_client(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
) -> None:
    pinned = _pinned(tmp_path)
    client_calls: list[bool] = []

    class FailingWorker(_FakeWorker):
        def validate_environment(self, device: str) -> dict[str, object]:
            raise RuntimeError(f"environment drift on {device}")

    runtime = MatrixStudyRuntime(
        pinned=pinned,
        schedule=build_matrix_schedule(pinned.spec),
        authorization=_authorization(pinned),
        agent_bundles={"fake-agent": manifest_bundle},
        client_factory=lambda _agent, bundle: (
            client_calls.append(True) or _FakeClient(bundle)
        ),
        worker_factory=lambda: FailingWorker(_context()),
        controller_root=tmp_path,
        controller_revision_reader=lambda _root: "1" * 40,
        asset_root=CAPABILITY_GATE_ASSET_ROOT,
    )
    contract = build_matrix_phase_contract(
        pinned,
        "core",
        execution_context=_context(),
        resolve_task=runtime.resolve_task,
        resolve_target=runtime.resolve_target,
        resolve_agent=runtime.resolve_agent,
        resolve_execution=runtime.resolve_execution,
    )

    with pytest.raises(RuntimeError, match="environment drift"):
        runtime.runtime_for(contract.plan.cells[0].identity)
    assert client_calls == []


def test_runtime_rejects_worker_route_before_health_probe(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
) -> None:
    pinned = _pinned(tmp_path)
    worker = _FakeWorker(_context())
    worker.matrix_worker_binding = worker.matrix_worker_binding.model_copy(
        update={
            "transport": worker.matrix_worker_binding.transport.model_copy(
                update={"host": "wrong.example"}
            )
        }
    )
    runtime = MatrixStudyRuntime(
        pinned=pinned,
        schedule=build_matrix_schedule(pinned.spec),
        authorization=_authorization(pinned),
        agent_bundles={"fake-agent": manifest_bundle},
        client_factory=lambda _agent, bundle: _FakeClient(bundle),
        worker_factory=lambda: worker,
        controller_root=tmp_path,
        controller_revision_reader=lambda _root: "1" * 40,
        asset_root=CAPABILITY_GATE_ASSET_ROOT,
    )
    contract = build_matrix_phase_contract(
        pinned,
        "core",
        execution_context=_context(),
        resolve_task=runtime.resolve_task,
        resolve_target=runtime.resolve_target,
        resolve_agent=runtime.resolve_agent,
        resolve_execution=runtime.resolve_execution,
    )

    with pytest.raises(MatrixRuntimeError, match="worker differs"):
        runtime.runtime_for(contract.plan.cells[0].identity)
    assert worker.validated_devices == []


def test_runtime_rejects_study_floor_and_agent_coverage_drift(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
) -> None:
    pinned = _pinned(tmp_path)
    authorization = _authorization(pinned)
    common = {
        "pinned": pinned,
        "schedule": build_matrix_schedule(pinned.spec),
        "client_factory": lambda _agent, bundle: _FakeClient(bundle),
        "worker_factory": lambda: _FakeWorker(_context()),
        "controller_root": pinned.path.parent,
        "controller_revision_reader": lambda _root: "1" * 40,
        "asset_root": CAPABILITY_GATE_ASSET_ROOT,
    }

    with pytest.raises(MatrixRuntimeError, match="Agent bundles"):
        MatrixStudyRuntime(
            **common,
            authorization=authorization,
            agent_bundles={"other-agent": manifest_bundle},
        )

    missing_floor = authorization.model_copy(update={"tasks": ()})
    with pytest.raises(MatrixRuntimeError, match="floors do not exactly cover"):
        MatrixStudyRuntime(
            **common,
            authorization=missing_floor,
            agent_bundles={"fake-agent": manifest_bundle},
        )

    wrong_schedule = authorization.model_copy(update={"schedule_sha256": _digest("drift")})
    with pytest.raises(MatrixRuntimeError, match="differs from the pinned study"):
        MatrixStudyRuntime(
            **common,
            authorization=wrong_schedule,
            agent_bundles={"fake-agent": manifest_bundle},
        )


def test_authorized_ssh_worker_preserves_complete_preflight_route(tmp_path: Path) -> None:
    authorization = _authorization(_pinned(tmp_path))

    worker = build_authorized_ssh_worker(authorization)

    assert worker.host == "root@gpu.example"
    assert worker.port == 30554
    assert worker.device == "cuda:0"
    assert worker.sandbox_mode == "setpriv"
    assert worker.expected_worker_revision == "2" * 40
    assert worker.expected_hardware_substring == "Fake A100"
    assert worker.expected_compute_capability == (8, 0)
    assert worker.expected_python_version == "3.10.20"
    assert worker.expected_torch_version == "2.13.0+cu126"
    assert worker.expected_torch_cuda_version == "12.6"
    assert worker.expected_tilelang_version == "0.1.12"
    assert worker.expected_triton_version == "3.7.1"
    assert worker.expected_driver_version == "570.00"
    assert worker.expected_non_container_worker is True
    assert worker.matrix_worker_binding == MatrixWorkerBinding(
        worker_revision="2" * 40,
        transport=_context().transport,
    )


def test_runtime_rejects_controller_revision_drift(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
) -> None:
    pinned = _pinned(tmp_path)
    with pytest.raises(MatrixRuntimeError, match="controller revision"):
        MatrixStudyRuntime(
            pinned=pinned,
            schedule=build_matrix_schedule(pinned.spec),
            authorization=_authorization(pinned),
            agent_bundles={"fake-agent": manifest_bundle},
            client_factory=lambda _agent, bundle: _FakeClient(bundle),
            worker_factory=lambda: _FakeWorker(_context()),
            controller_root=tmp_path,
            controller_revision_reader=lambda _root: "9" * 40,
            asset_root=CAPABILITY_GATE_ASSET_ROOT,
        )


def test_clean_controller_revision_rejects_dirty_checkout(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "initial"], check=True)

    revision = read_clean_controller_revision(tmp_path)
    assert len(revision) == 40

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(MatrixRuntimeError, match="must be clean"):
        read_clean_controller_revision(tmp_path)
