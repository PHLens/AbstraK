"""Reusable runtime assembly for preflight-authorized matrix studies."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from abstrak.canary.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    AgentBudget,
    AgentLoopPolicy,
    CanaryModel,
    TimingSpec,
)
from abstrak.canary.loop import CompletionClient, WorkerExecutor
from abstrak.canary.manifests import PinnedStudySpec
from abstrak.canary.matrix import MatrixCell, MatrixSchedule
from abstrak.canary.matrix_preflight import PreflightBundle
from abstrak.canary.matrix_runner import (
    MatrixAgentBinding,
    MatrixAttemptRuntime,
    MatrixExecutionContext,
    MatrixWorkerBinding,
)
from abstrak.canary.matrix_study import (
    MatrixAxisIdentity,
    MatrixCellArtifactIdentity,
    MatrixCellExecutionSpec,
)
from abstrak.canary.protocol import build_initial_messages
from abstrak.canary.remote import SshWorkerExecutor
from abstrak.canary.targets import get_target_stack, load_target_card
from abstrak.canary.tasks import get_task_pack, load_task_source
from abstrak.providers.contracts import ChatMessage, sha256_json
from abstrak.providers.manifests import (
    ManifestBundle,
    ModelManifest,
    completion_client_identity,
)

ClientFactory = Callable[[str, ManifestBundle], CompletionClient]
WorkerFactory = Callable[[], WorkerExecutor]
ControllerRevisionReader = Callable[[str | Path], str]


class MatrixRuntimeError(ValueError):
    """Raised when runtime inputs differ from their preflight authorization."""


class RuntimeTaskAuthorization(CanaryModel):
    """The only per-task floor value needed while assembling Agent cells."""

    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    latency_ceiling_ms: float = Field(gt=0)


class MatrixRuntimeAuthorization(CanaryModel):
    """Narrow ready-only view derived from a verified preflight bundle."""

    schema_version: Literal["abstrak-matrix-runtime-authorization.v1"] = (
        "abstrak-matrix-runtime-authorization.v1"
    )
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    raw_study_sha256: str = Field(pattern=SHA256_PATTERN)
    spec_sha256: str = Field(pattern=SHA256_PATTERN)
    schedule_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_context: MatrixExecutionContext
    accelerator: str = Field(min_length=1)
    compute_capability: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    python_version: str = Field(min_length=1)
    tilelang_version: str = Field(min_length=1)
    triton_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    cuda_version: str = Field(min_length=1)
    driver_version: str = Field(min_length=1)
    non_container_worker: Literal[True] = True
    tasks: tuple[RuntimeTaskAuthorization, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def task_ids_are_unique(self) -> MatrixRuntimeAuthorization:
        task_ids = tuple(item.task_id for item in self.tasks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("runtime authorization task IDs must be unique")
        return self

    @property
    def task_map(self) -> dict[str, RuntimeTaskAuthorization]:
        return {item.task_id: item for item in self.tasks}


def runtime_authorization(bundle: PreflightBundle) -> MatrixRuntimeAuthorization:
    """Project a fully verified bundle into the narrow live-runtime contract."""

    tasks: list[RuntimeTaskAuthorization] = []
    for floor in bundle.floor.tasks:
        if floor.status != "valid" or floor.ceiling is None:
            raise MatrixRuntimeError("runtime authorization requires valid task floors")
        tasks.append(
            RuntimeTaskAuthorization(
                task_id=floor.task_id,
                latency_ceiling_ms=floor.ceiling.latency_ceiling_ms,
            )
        )
    return MatrixRuntimeAuthorization(
        study_id=bundle.assets.study_id,
        raw_study_sha256=bundle.assets.raw_study_sha256,
        spec_sha256=bundle.assets.spec_sha256,
        schedule_sha256=bundle.assets.schedule_sha256,
        execution_context=bundle.execution_context,
        accelerator=bundle.environment.accelerator,
        compute_capability=bundle.environment.compute_capability,
        python_version=bundle.environment.python_version,
        tilelang_version=bundle.environment.tilelang_version,
        triton_version=bundle.environment.triton_version,
        torch_version=bundle.environment.torch_version,
        cuda_version=bundle.environment.cuda_version,
        driver_version=bundle.environment.driver_version,
        non_container_worker=bundle.environment.non_container_worker,
        tasks=tuple(tasks),
    )


def read_clean_controller_revision(root: str | Path) -> str:
    """Return HEAD only when the controller checkout has no tracked or untracked changes."""

    repository = Path(root).expanduser().resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise MatrixRuntimeError("cannot inspect the controller checkout") from error
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise MatrixRuntimeError("controller checkout did not report a full Git revision")
    if status:
        raise MatrixRuntimeError("controller checkout must be clean for matrix execution")
    return revision


def build_authorized_ssh_worker(
    authorization: MatrixRuntimeAuthorization,
) -> SshWorkerExecutor:
    """Construct the actual SSH executor from the hash-bound preflight route."""

    transport = authorization.execution_context.transport
    major, minor = (int(value) for value in authorization.compute_capability.split("."))
    worker = SshWorkerExecutor(
        transport.host,
        port=transport.port,
        python_executable=transport.python_executable,
        pythonpath=transport.pythonpath,
        kernelbench_root=transport.kernelbench_root,
        asset_root=transport.asset_root,
        device=transport.device,
        sandbox_mode=("setpriv" if transport.sandbox == "setpriv-supervised" else "bubblewrap"),
        timeout_seconds=transport.timeout_seconds,
        expected_hardware_substring=authorization.accelerator,
        expected_compute_capability=(major, minor),
        expected_python_version=authorization.python_version,
        expected_torch_version=authorization.torch_version,
        expected_torch_cuda_version=authorization.cuda_version,
        expected_tilelang_version=authorization.tilelang_version,
        expected_triton_version=authorization.triton_version,
        expected_driver_version=authorization.driver_version,
        expected_non_container_worker=authorization.non_container_worker,
        expected_worker_revision=authorization.execution_context.worker_revision,
    )
    expected_binding = MatrixWorkerBinding(
        worker_revision=authorization.execution_context.worker_revision,
        transport=transport,
    )
    if worker.matrix_worker_binding != expected_binding:
        raise MatrixRuntimeError("SSH worker configuration differs from runtime authorization")
    return worker


def normalize_matrix_agent_bundle(
    bundle: ManifestBundle,
    budget: AgentBudget,
) -> ManifestBundle:
    """Freeze deterministic generation settings shared by every matrix Agent."""

    model_payload = bundle.model.model_dump(mode="json")
    generation = model_payload["generation"]
    generation.update(
        {
            "max_completion_tokens": budget.max_completion_tokens_per_call,
            "temperature": 0,
            "top_p": None,
            "api_seed": None,
            "stop": [],
            "reasoning_effort": None,
        }
    )
    model_payload.update(
        {
            "allow_live_probe": False,
            "output_contract": "plain_text",
            "generation": generation,
        }
    )
    return ManifestBundle(
        provider=bundle.provider,
        model=ModelManifest.model_validate(model_payload),
        pricing=bundle.pricing,
    )


@dataclass
class MatrixStudyRuntime:
    """Pure resolvers plus lazy live factories for one authorized matrix study."""

    pinned: PinnedStudySpec
    schedule: MatrixSchedule
    authorization: MatrixRuntimeAuthorization
    agent_bundles: Mapping[str, ManifestBundle]
    client_factory: ClientFactory
    worker_factory: WorkerFactory
    controller_root: str | Path
    asset_root: str | Path | None = None
    max_completion_tokens_per_call: int = 8192
    max_wall_seconds: float = 1200.0
    dev_timing: TimingSpec = TimingSpec(repetitions=1)
    controller_revision_reader: ControllerRevisionReader = read_clean_controller_revision
    _bindings: dict[str, MatrixAgentBinding] = field(init=False, default_factory=dict)
    _bundles: dict[str, ManifestBundle] = field(init=False, default_factory=dict)
    _clients: dict[str, CompletionClient] = field(init=False, default_factory=dict)
    _worker: WorkerExecutor | None = field(init=False, default=None)
    _executions: dict[tuple[str, str, str, str], MatrixCellExecutionSpec] = field(
        init=False,
        default_factory=dict,
    )
    _messages: dict[tuple[str, str, str, str], tuple[ChatMessage, ...]] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        expected_binding = (
            self.pinned.spec.study_id,
            self.pinned.sha256,
            self.pinned.spec.sha256,
            self.schedule.sha256,
        )
        actual_binding = (
            self.authorization.study_id,
            self.authorization.raw_study_sha256,
            self.authorization.spec_sha256,
            self.authorization.schedule_sha256,
        )
        if self.schedule.spec != self.pinned.spec or actual_binding != expected_binding:
            raise MatrixRuntimeError("runtime authorization differs from the pinned study")
        controller_revision = self.controller_revision_reader(self.controller_root)
        if controller_revision != self.authorization.execution_context.controller_revision:
            raise MatrixRuntimeError("controller revision differs from the preflight environment")
        if tuple(self.agent_bundles) != self.pinned.spec.agents:
            raise MatrixRuntimeError("Agent bundles do not exactly cover the study axis")
        if set(self.authorization.task_map) != {
            task_id for phase in self.pinned.spec.phases for task_id in phase.task_ids
        }:
            raise MatrixRuntimeError("runtime floors do not exactly cover the study tasks")

        maximum_calls = max(phase.max_calls_per_trajectory for phase in self.pinned.spec.phases)
        normalization_budget = AgentBudget(
            max_calls=maximum_calls,
            max_completion_tokens_per_call=self.max_completion_tokens_per_call,
            max_wall_seconds=self.max_wall_seconds,
        )
        for agent_id, bundle in self.agent_bundles.items():
            normalized = normalize_matrix_agent_bundle(bundle, normalization_budget)
            self._bundles[agent_id] = normalized
            self._bindings[agent_id] = MatrixAgentBinding(
                agent_id=agent_id,
                completion=completion_client_identity(normalized),
            )

    def resolve_task(self, task_id: str) -> MatrixAxisIdentity:
        task = get_task_pack(task_id)
        load_task_source(task_id, asset_root=self.asset_root)
        return MatrixAxisIdentity(kind="task", id=task_id, sha256=sha256_json(task))

    def resolve_target(self, target_id: str) -> MatrixAxisIdentity:
        target = get_target_stack(target_id)
        load_target_card(target_id, asset_root=self.asset_root)
        return MatrixAxisIdentity(kind="target", id=target_id, sha256=sha256_json(target))

    def resolve_agent(self, agent_id: str) -> MatrixAxisIdentity:
        try:
            return self._bindings[agent_id].axis_identity
        except KeyError:
            raise MatrixRuntimeError(f"unknown matrix Agent: {agent_id}") from None

    def _blueprint(
        self,
        cell: MatrixCell,
    ) -> tuple[MatrixCellExecutionSpec, tuple[ChatMessage, ...]]:
        key = (cell.phase_id, cell.task_id, cell.target_id, cell.agent_id)
        existing = self._executions.get(key)
        if existing is not None:
            return existing, self._messages[key]
        try:
            floor = self.authorization.task_map[cell.task_id]
            bundle = self._bundles[cell.agent_id]
        except KeyError as error:
            raise MatrixRuntimeError("matrix cell references an unauthorized axis value") from error
        phase = self.pinned.spec.phase(cell.phase_id)
        budget = AgentBudget(
            max_calls=phase.max_calls_per_trajectory,
            max_completion_tokens_per_call=self.max_completion_tokens_per_call,
            max_wall_seconds=self.max_wall_seconds,
        )
        policy = AgentLoopPolicy(
            response_parser="candidate_only",
            stop_policy="correct_latency",
            final_selection="best_correct_latency",
            latency_ceiling_ms=floor.latency_ceiling_ms,
        )
        task = get_task_pack(cell.task_id)
        card = load_target_card(cell.target_id, asset_root=self.asset_root)
        messages = build_initial_messages(task, card, policy=policy)
        execution = MatrixCellExecutionSpec(
            budget=budget,
            policy=policy,
            dev_timing=self.dev_timing,
            model_ref=bundle.model.id,
            initial_messages_sha256=sha256_json(
                [message.model_dump(mode="json") for message in messages]
            ),
            device=self.authorization.execution_context.transport.device,
            execution_context_sha256=self.authorization.execution_context.sha256,
        )
        self._executions[key] = execution
        self._messages[key] = messages
        return execution, messages

    def resolve_execution(self, cell: MatrixCell) -> MatrixCellExecutionSpec:
        return self._blueprint(cell)[0]

    def runtime_for(self, identity: MatrixCellArtifactIdentity) -> MatrixAttemptRuntime:
        execution, messages = self._blueprint(identity.cell)
        agent_id = identity.cell.agent_id
        if self._worker is None:
            worker = self.worker_factory()
            expected_worker = MatrixWorkerBinding(
                worker_revision=self.authorization.execution_context.worker_revision,
                transport=self.authorization.execution_context.transport,
            )
            if getattr(worker, "matrix_worker_binding", None) != expected_worker:
                raise MatrixRuntimeError("matrix worker differs from runtime authorization")
            validate_environment = getattr(worker, "validate_environment", None)
            if not callable(validate_environment):
                raise MatrixRuntimeError("matrix worker cannot validate its live environment")
            validate_environment(execution.device)
            self._worker = worker
        client = self._clients.get(agent_id)
        if client is None:
            client = self.client_factory(agent_id, self._bundles[agent_id])
            if getattr(client, "completion_identity", None) != self._bindings[agent_id].completion:
                raise MatrixRuntimeError("completion client differs from runtime authorization")
            self._clients[agent_id] = client
        secrets = getattr(client, "artifact_secrets", ())
        if not isinstance(secrets, tuple) or any(not isinstance(value, str) for value in secrets):
            raise MatrixRuntimeError("completion client artifact secrets must be a string tuple")
        return MatrixAttemptRuntime(
            task=get_task_pack(identity.cell.task_id),
            target=get_target_stack(identity.cell.target_id),
            agent_binding=self._bindings[agent_id],
            execution=execution,
            initial_messages=messages,
            client=client,
            worker=self._worker,
            artifact_secrets=secrets,
        )
