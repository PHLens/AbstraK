from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

import abstrak.canary.matrix_runner as matrix_runner
from abstrak.canary.contracts import AgentBudget, AgentLoopPolicy, TimingSpec
from abstrak.canary.manifests import PinnedStudySpec
from abstrak.canary.matrix import (
    CoreGateThresholds,
    MatrixStudySpec,
    PhaseSpec,
    PortfolioGateSpec,
    TaskGroupSpec,
    build_matrix_schedule,
)
from abstrak.canary.matrix_preflight import (
    FORMAL_FLOOR_TIMING,
    AssetManifest,
    BaselineAssetBinding,
    BaselineTimingEvidence,
    CanaryAssetBinding,
    CapabilityCanaryEvidence,
    CapabilityTargetEvidence,
    EnvironmentManifest,
    EnvironmentObservation,
    EnvironmentProbeEvidence,
    ExpertCorrectnessEvidence,
    FloorManifest,
    LatencyCeilingDerivation,
    LaunchFloorEvidence,
    LaunchTimingMeasurement,
    MatrixPreflightError,
    TargetAssetBinding,
    TargetCodegenEvidence,
    TaskAssetBinding,
    TaskFloorRecord,
    VerifiedTaskFloorEvidence,
    build_asset_manifest,
    build_pending_environment,
    build_pending_floor,
    build_preflight_receipt,
    load_preflight_bundle,
    seal_preflight_bundle,
)
from abstrak.canary.matrix_runner import (
    MatrixExecutionContext,
    MatrixTransportContext,
    run_matrix_phase,
)
from abstrak.canary.matrix_study import MatrixAxisIdentity, MatrixCellExecutionSpec


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _spec() -> MatrixStudySpec:
    return MatrixStudySpec(
        study_id="matrix-preflight-test",
        seed=20260724,
        agents=("fake-agent",),
        targets=("target-core", "target-full"),
        task_groups=(TaskGroupSpec(id="group", task_ids=("task-a", "task-b")),),
        phases=(
            PhaseSpec(
                id="core",
                task_ids=("task-a", "task-b"),
                replicates=(1,),
                order_policy="fixed",
                max_calls_per_trajectory=3,
            ),
        ),
        gate=PortfolioGateSpec(
            core_phase_id="core",
            reserve_phase_id=None,
            reserve_on_outcomes=(),
            core=CoreGateThresholds(
                min_stable_tasks=1,
                min_unique_winner_groups=1,
                min_competitive_gap_units=1,
                min_latency_gain_tasks=1,
            ),
        ),
    )


def _pinned(tmp_path: Path) -> PinnedStudySpec:
    spec = _spec()
    path = tmp_path / "study.json"
    rendered = json.dumps(spec.model_dump(mode="json"), sort_keys=True) + "\n"
    path.write_text(rendered, encoding="utf-8")
    return PinnedStudySpec(
        path=path,
        sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        spec=spec,
    )


def _task_asset(task_id: str) -> TaskAssetBinding:
    return TaskAssetBinding(
        task_id=task_id,
        task_pack_sha256=_digest(f"{task_id}:pack"),
        reference_source_sha256=_digest(f"{task_id}:reference"),
        expert_source_sha256=_digest(f"{task_id}:expert"),
        baselines=tuple(
            BaselineAssetBinding(
                variant=variant,
                source_sha256=_digest(f"{task_id}:baseline:{variant}"),
            )
            for variant in ("eager", "compile", "vendor")
        ),
    )


def _target_asset(target_id: str) -> TargetAssetBinding:
    return TargetAssetBinding(
        target_id=target_id,
        target_stack_sha256=_digest(f"{target_id}:stack"),
        card_sha256=_digest(f"{target_id}:card"),
    )


def _transport() -> MatrixTransportContext:
    return MatrixTransportContext(
        host="a100.example",
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
    )


def _assets(pinned: PinnedStudySpec):
    schedule = build_matrix_schedule(pinned.spec)
    return build_asset_manifest(
        pinned,
        schedule,
        tasks=tuple(_task_asset(task_id) for task_id in ("task-a", "task-b")),
        targets=tuple(_target_asset(target_id) for target_id in pinned.spec.targets),
        canaries=(
            CanaryAssetBinding(
                canary_id="schedule-canary",
                task_id="task-a",
                source_sha256=_digest("schedule-canary:source"),
                required_target_ids=("target-full",),
            ),
        ),
    )


def _pending_environment(pinned: PinnedStudySpec) -> EnvironmentManifest:
    return build_pending_environment(
        pinned,
        build_matrix_schedule(pinned.spec),
        controller_revision=_digest("controller-revision")[:40],
        worker_revision=_digest("worker-revision")[:40],
        transport=_transport(),
        accelerator="NVIDIA A100-SXM4-80GB",
        compute_capability="8.0",
        python_version="3.10.20",
        tilelang_version="0.1.12",
        triton_version="3.7.1",
        torch_version="2.13.0",
        cuda_version="12.6",
        driver_version="570.00",
    )


def _verified_environment(pinned: PinnedStudySpec) -> EnvironmentManifest:
    pending = _pending_environment(pinned)
    payload = pending.model_dump()
    payload["status"] = "verified"
    payload["verification_evidence"] = EnvironmentProbeEvidence(
        artifact_sha256=_digest("remote-environment-probe"),
        status="pass",
        observation=EnvironmentObservation(
            worker_revision=pending.worker_revision,
            transport=pending.transport,
            accelerator=pending.accelerator,
            compute_capability=pending.compute_capability,
            python_version=pending.python_version,
            tilelang_version=pending.tilelang_version,
            triton_version=pending.triton_version,
            torch_version=pending.torch_version,
            cuda_version=pending.cuda_version,
            driver_version=pending.driver_version,
        ),
    )
    return EnvironmentManifest.model_validate(payload)


def _valid_task_floor(
    task: TaskAssetBinding,
    targets: tuple[TargetAssetBinding, ...],
    *,
    factor: float = 1.25,
) -> TaskFloorRecord:
    medians = {"eager": 2.0, "compile": 1.0, "vendor": 1.25}
    timings = tuple(
        BaselineTimingEvidence(
            variant=baseline.variant,
            source_sha256=baseline.source_sha256,
            artifact_sha256=_digest(f"{task.task_id}:{baseline.variant}:artifact"),
            timing_summary_sha256=_digest(f"{task.task_id}:{baseline.variant}:summary"),
            status="stable",
            median_ms=medians[baseline.variant],
        )
        for baseline in task.baselines
    )
    selected = timings[1]
    evidence = VerifiedTaskFloorEvidence(
        task_id=task.task_id,
        expert_source_sha256=task.expert_source_sha256,
        expert_correctness=ExpertCorrectnessEvidence(
            artifact_sha256=_digest(f"{task.task_id}:expert-correctness"),
            task_id=task.task_id,
            task_pack_sha256=task.task_pack_sha256,
            expert_source_sha256=task.expert_source_sha256,
            status="pass",
            compiled=True,
            all_sealed_cases_passed=True,
            output_finite=True,
            inputs_unchanged=True,
            fallback_free=True,
        ),
        target_codegen=tuple(
            TargetCodegenEvidence(
                artifact_sha256=_digest(f"{task.task_id}:{target.target_id}:validation"),
                task_id=task.task_id,
                task_pack_sha256=task.task_pack_sha256,
                target_id=target.target_id,
                target_stack_sha256=target.target_stack_sha256,
                expert_source_sha256=task.expert_source_sha256,
                status="pass",
                compiled=True,
                correct=True,
                fallback_free=True,
                generated_code_sha256=_digest(f"{task.task_id}:generated-cuda"),
            )
            for target in targets
        ),
        baseline_timings=timings,
        selected_baseline_variant=selected.variant,
        selected_baseline_source_sha256=selected.source_sha256,
        selected_timing_summary_sha256=selected.timing_summary_sha256,
    )
    return TaskFloorRecord(
        task_id=task.task_id,
        status="valid",
        expert_source_sha256=task.expert_source_sha256,
        verified_evidence=evidence,
        ceiling=LatencyCeilingDerivation(
            l_star_ms=evidence.l_star_ms,
            competitive_factor=factor,
            latency_ceiling_ms=evidence.l_star_ms * factor,
        ),
    )


def _capability_evidence(
    canary: CanaryAssetBinding,
    targets: tuple[TargetAssetBinding, ...],
) -> CapabilityCanaryEvidence:
    target_by_id = {target.target_id: target for target in targets}
    return CapabilityCanaryEvidence(
        canary_id=canary.canary_id,
        task_id=canary.task_id,
        source_sha256=canary.source_sha256,
        status="pass",
        targets=tuple(
            CapabilityTargetEvidence(
                artifact_sha256=_digest(f"{canary.canary_id}:{target_id}:validation"),
                target_id=target_id,
                target_stack_sha256=target_by_id[target_id].target_stack_sha256,
                status="pass",
                compiled=True,
                correct=True,
                used_capabilities=(canary.canary_id,),
                generated_code_sha256=_digest(
                    f"{canary.canary_id}:{target_id}:capability-codegen"
                ),
                control_generated_code_sha256=_digest(
                    f"{canary.canary_id}:{target_id}:control-codegen"
                ),
            )
            for target_id in canary.required_target_ids
        ),
    )


def _launch_floor(assets: AssetManifest) -> LaunchFloorEvidence:
    workloads = tuple(("task", item.task_id) for item in assets.tasks) + tuple(
        ("canary", item.canary_id) for item in assets.canaries
    )
    return LaunchFloorEvidence(
        artifact_sha256=_digest("launch-floor-results"),
        status="pass",
        measurements=tuple(
            LaunchTimingMeasurement(
                workload_kind=kind,
                workload_id=identifier,
                launch_ms=0.01,
                task_ms=1.0,
            )
            for kind, identifier in workloads
        ),
    )


def _valid_floor(
    pinned: PinnedStudySpec,
    assets: AssetManifest,
    environment: EnvironmentManifest,
    *,
    factor: float = 1.25,
) -> FloorManifest:
    schedule = build_matrix_schedule(pinned.spec)
    fields = {
        "study_id": pinned.spec.study_id,
        "raw_study_sha256": pinned.sha256,
        "spec_sha256": pinned.spec.sha256,
        "schedule_sha256": schedule.sha256,
    }
    return FloorManifest(
        **fields,
        status="valid",
        asset_manifest_sha256=assets.sha256,
        environment_manifest_sha256=environment.sha256,
        timing=FORMAL_FLOOR_TIMING,
        tasks=tuple(
            _valid_task_floor(task, assets.targets, factor=factor) for task in assets.tasks
        ),
        capability_canaries=tuple(
            _capability_evidence(canary, assets.targets) for canary in assets.canaries
        ),
        launch_floor=_launch_floor(assets),
    )


def _context(
    assets: AssetManifest,
    floor: FloorManifest,
    environment: EnvironmentManifest,
) -> MatrixExecutionContext:
    return MatrixExecutionContext(
        controller_revision=environment.controller_revision,
        worker_revision=environment.worker_revision,
        transport=environment.transport,
        asset_manifest_sha256=assets.sha256,
        floor_manifest_sha256=floor.sha256,
        environment_manifest_sha256=environment.sha256,
    )


@dataclass(frozen=True)
class ReadyInputs:
    pinned: PinnedStudySpec
    assets: AssetManifest
    environment: EnvironmentManifest
    floor: FloorManifest
    context: MatrixExecutionContext

    @property
    def schedule(self):
        return build_matrix_schedule(self.pinned.spec)


@pytest.fixture
def ready(tmp_path: Path) -> ReadyInputs:
    pinned = _pinned(tmp_path)
    assets = _assets(pinned)
    environment = _verified_environment(pinned)
    floor = _valid_floor(pinned, assets, environment)
    return ReadyInputs(
        pinned=pinned,
        assets=assets,
        environment=environment,
        floor=floor,
        context=_context(assets, floor, environment),
    )


def test_local_builders_are_deterministic_and_leave_floor_pending(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)
    assets = _assets(pinned)
    environment = _pending_environment(pinned)
    floor = build_pending_floor(
        pinned,
        build_matrix_schedule(pinned.spec),
        assets=assets,
        environment=environment,
    )

    assert assets.sha256 == _assets(pinned).sha256
    assert environment.status == "pending"
    assert environment.python_version == "3.10.20"
    assert environment.triton_version == "3.7.1"
    assert floor.status == "pending"
    assert floor.timing == TimingSpec(
        warmup_runs=25,
        trial_runs=200,
        repetitions=3,
        max_cv=0.05,
    )
    assert {task.status for task in floor.tasks} == {"pending"}
    assert (
        floor.sha256
        == hashlib.sha256(
            json.dumps(
                floor.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )


def test_valid_floor_requires_explicit_verified_evidence_and_formal_timing(
    ready: ReadyInputs,
) -> None:
    task = ready.assets.tasks[0]
    with pytest.raises(ValidationError, match="explicit verified evidence"):
        TaskFloorRecord(
            task_id=task.task_id,
            status="valid",
            expert_source_sha256=task.expert_source_sha256,
        )

    payload = ready.floor.model_dump()
    payload["timing"] = TimingSpec()
    with pytest.raises(ValidationError, match="25/200/3"):
        FloorManifest.model_validate(payload)


def test_floor_status_is_derived_and_invalid_is_fail_closed(ready: ReadyInputs) -> None:
    payload = ready.floor.model_dump()
    payload["status"] = "pending"
    with pytest.raises(ValidationError, match="floor status must be valid"):
        FloorManifest.model_validate(payload)

    invalid = TaskFloorRecord(
        task_id=ready.assets.tasks[0].task_id,
        status="invalid",
        expert_source_sha256=ready.assets.tasks[0].expert_source_sha256,
        invalid_evidence_sha256=_digest("invalid-floor-evidence"),
        invalid_reason="capability is inert",
    )
    payload = ready.floor.model_dump()
    payload["tasks"] = (invalid, ready.floor.tasks[1])
    payload["status"] = "invalid"
    invalid_floor = FloorManifest.model_validate(payload)
    assert invalid_floor.status == "invalid"

    payload["status"] = "valid"
    with pytest.raises(ValidationError, match="floor status must be invalid"):
        FloorManifest.model_validate(payload)

    payload = ready.floor.model_dump()
    payload["launch_floor"] = None
    payload["status"] = "pending"
    assert FloorManifest.model_validate(payload).status == "pending"
    payload["status"] = "valid"
    with pytest.raises(ValidationError, match="floor status must be pending"):
        FloorManifest.model_validate(payload)


def test_verified_environment_rejects_opaque_or_drifted_probe_evidence(
    tmp_path: Path,
) -> None:
    pinned = _pinned(tmp_path)
    payload = _pending_environment(pinned).model_dump()
    payload["status"] = "verified"
    payload["verification_evidence_sha256"] = _digest("opaque-placeholder")
    with pytest.raises(ValidationError):
        EnvironmentManifest.model_validate(payload)

    payload = _verified_environment(pinned).model_dump()
    payload["verification_evidence"]["observation"]["tilelang_version"] = "drifted"
    with pytest.raises(ValidationError, match="differs from its probe observation"):
        EnvironmentManifest.model_validate(payload)


def test_passing_codegen_and_launch_evidence_requires_observed_results(
    ready: ReadyInputs,
) -> None:
    payload = ready.floor.model_dump()
    payload["tasks"][0]["verified_evidence"]["target_codegen"][0]["correct"] = False
    with pytest.raises(ValidationError, match="target codegen status must be fail"):
        FloorManifest.model_validate(payload)

    payload = ready.floor.model_dump()
    target = payload["capability_canaries"][0]["targets"][0]
    target["control_generated_code_sha256"] = target["generated_code_sha256"]
    with pytest.raises(ValidationError, match="capability target status must be fail"):
        FloorManifest.model_validate(payload)

    payload = ready.floor.model_dump()
    payload["launch_floor"]["measurements"] = ()
    with pytest.raises(ValidationError, match="passing launch floor requires measurements"):
        FloorManifest.model_validate(payload)


@pytest.mark.parametrize("coverage", ["canary", "launch"])
def test_ready_receipt_requires_exact_terminal_evidence_coverage(
    ready: ReadyInputs,
    coverage: str,
) -> None:
    payload = ready.floor.model_dump()
    if coverage == "canary":
        payload["capability_canaries"] = ()
        message = "exactly cover frozen canaries"
    else:
        payload["launch_floor"]["measurements"] = payload["launch_floor"]["measurements"][:-1]
        message = "exactly cover frozen workloads"
    incomplete = FloorManifest.model_validate(payload)

    with pytest.raises(MatrixPreflightError, match=message):
        build_preflight_receipt(
            ready.pinned,
            ready.schedule,
            assets=ready.assets,
            floor=incomplete,
            environment=ready.environment,
            execution_context=_context(ready.assets, incomplete, ready.environment),
        )


@pytest.mark.parametrize("axis", ["task", "target"])
def test_asset_manifest_requires_exact_ordered_study_coverage(
    tmp_path: Path,
    axis: str,
) -> None:
    pinned = _pinned(tmp_path)
    schedule = build_matrix_schedule(pinned.spec)
    tasks = tuple(_task_asset(task_id) for task_id in ("task-a", "task-b"))
    targets = tuple(_target_asset(target_id) for target_id in pinned.spec.targets)
    if axis == "task":
        tasks = tasks[:1]
    else:
        targets = tuple(reversed(targets))

    with pytest.raises(MatrixPreflightError, match=f"asset {axis}s do not exactly cover"):
        build_asset_manifest(pinned, schedule, tasks=tasks, targets=targets)


def test_canary_references_must_resolve_inside_asset_manifest(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)
    schedule = build_matrix_schedule(pinned.spec)
    fields = {
        "study_id": pinned.spec.study_id,
        "raw_study_sha256": pinned.sha256,
        "spec_sha256": pinned.spec.sha256,
        "schedule_sha256": schedule.sha256,
    }
    with pytest.raises(ValidationError, match="undeclared target"):
        AssetManifest(
            **fields,
            tasks=tuple(_task_asset(task_id) for task_id in ("task-a", "task-b")),
            targets=tuple(_target_asset(target_id) for target_id in pinned.spec.targets),
            canaries=(
                CanaryAssetBinding(
                    canary_id="bad-canary",
                    task_id="task-a",
                    source_sha256=_digest("bad-canary"),
                    required_target_ids=("target-missing",),
                ),
            ),
        )


def test_shared_study_binding_rejects_raw_manifest_drift(tmp_path: Path) -> None:
    pinned = _pinned(tmp_path)
    assets = _assets(pinned)
    environment = _pending_environment(pinned)
    payload = assets.model_dump()
    payload["raw_study_sha256"] = _digest("different-raw-study")
    drifted = AssetManifest.model_validate(payload)

    with pytest.raises(MatrixPreflightError, match="pinned study identity"):
        build_pending_floor(
            pinned,
            build_matrix_schedule(pinned.spec),
            assets=drifted,
            environment=environment,
        )


def test_preflight_receipt_binds_study_floor_environment_and_context(
    ready: ReadyInputs,
) -> None:
    receipt = build_preflight_receipt(
        ready.pinned,
        ready.schedule,
        assets=ready.assets,
        floor=ready.floor,
        environment=ready.environment,
        execution_context=ready.context,
    )

    assert receipt.status == "ready"
    assert receipt.raw_study_sha256 == ready.pinned.sha256
    assert receipt.schedule_sha256 == ready.schedule.sha256
    assert receipt.asset_manifest_sha256 == ready.assets.sha256
    assert receipt.floor_manifest_sha256 == ready.floor.sha256
    assert receipt.environment_manifest_sha256 == ready.environment.sha256
    assert receipt.execution_context_sha256 == ready.context.sha256
    assert receipt.evidence_closure_sha256
    assert receipt.task_ids == ("task-a", "task-b")
    assert receipt.target_ids == ("target-core", "target-full")
    assert (
        receipt.sha256
        == build_preflight_receipt(
            ready.pinned,
            ready.schedule,
            assets=ready.assets,
            floor=ready.floor,
            environment=ready.environment,
            execution_context=ready.context,
        ).sha256
    )


def test_preflight_rejects_pending_status_and_wrong_gate_factor(
    ready: ReadyInputs,
) -> None:
    pending_environment = _pending_environment(ready.pinned)
    pending_floor = build_pending_floor(
        ready.pinned,
        ready.schedule,
        assets=ready.assets,
        environment=pending_environment,
    )
    pending_context = _context(ready.assets, pending_floor, pending_environment)
    with pytest.raises(MatrixPreflightError, match="valid floor"):
        build_preflight_receipt(
            ready.pinned,
            ready.schedule,
            assets=ready.assets,
            floor=pending_floor,
            environment=pending_environment,
            execution_context=pending_context,
        )

    valid_floor_on_pending_environment = _valid_floor(
        ready.pinned,
        ready.assets,
        pending_environment,
    )
    with pytest.raises(MatrixPreflightError, match="verified environment"):
        build_preflight_receipt(
            ready.pinned,
            ready.schedule,
            assets=ready.assets,
            floor=valid_floor_on_pending_environment,
            environment=pending_environment,
            execution_context=_context(
                ready.assets,
                valid_floor_on_pending_environment,
                pending_environment,
            ),
        )

    wrong_factor = _valid_floor(
        ready.pinned,
        ready.assets,
        ready.environment,
        factor=1.5,
    )
    with pytest.raises(MatrixPreflightError, match="competitive factor"):
        build_preflight_receipt(
            ready.pinned,
            ready.schedule,
            assets=ready.assets,
            floor=wrong_factor,
            environment=ready.environment,
            execution_context=_context(ready.assets, wrong_factor, ready.environment),
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("controller_revision", "controller revision"),
        ("worker_revision", "worker revision"),
        ("asset_manifest_sha256", "manifest hashes"),
    ],
)
def test_preflight_rejects_execution_context_mismatch(
    ready: ReadyInputs,
    field: str,
    message: str,
) -> None:
    payload = ready.context.model_dump()
    payload[field] = _digest(f"drift:{field}")[:40] if "revision" in field else _digest(field)
    mismatched = MatrixExecutionContext.model_validate(payload)
    with pytest.raises(MatrixPreflightError, match=message):
        build_preflight_receipt(
            ready.pinned,
            ready.schedule,
            assets=ready.assets,
            floor=ready.floor,
            environment=ready.environment,
            execution_context=mismatched,
        )


def test_floor_evidence_must_cover_frozen_targets_and_baselines(ready: ReadyInputs) -> None:
    floor_payload = ready.floor.model_dump()
    floor_payload["tasks"][0]["verified_evidence"]["target_codegen"] = floor_payload["tasks"][0][
        "verified_evidence"
    ]["target_codegen"][:1]
    incomplete = FloorManifest.model_validate(floor_payload)
    with pytest.raises(MatrixPreflightError, match="every target"):
        build_preflight_receipt(
            ready.pinned,
            ready.schedule,
            assets=ready.assets,
            floor=incomplete,
            environment=ready.environment,
            execution_context=_context(ready.assets, incomplete, ready.environment),
        )

    floor_payload = ready.floor.model_dump()
    floor_payload["tasks"][0]["verified_evidence"]["baseline_timings"] = floor_payload["tasks"][0][
        "verified_evidence"
    ]["baseline_timings"][:2]
    incomplete = FloorManifest.model_validate(floor_payload)
    with pytest.raises(MatrixPreflightError, match="every frozen baseline"):
        build_preflight_receipt(
            ready.pinned,
            ready.schedule,
            assets=ready.assets,
            floor=incomplete,
            environment=ready.environment,
            execution_context=_context(ready.assets, incomplete, ready.environment),
        )


def test_sealed_preflight_bundle_round_trips_with_strict_typed_records(
    tmp_path: Path,
    ready: ReadyInputs,
) -> None:
    directory = seal_preflight_bundle(
        tmp_path / "artifacts",
        ready.pinned,
        ready.schedule,
        assets=ready.assets,
        floor=ready.floor,
        environment=ready.environment,
        execution_context=ready.context,
    )
    bundle = load_preflight_bundle(
        directory,
        ready.pinned,
        ready.schedule,
    )

    assert bundle.assets == ready.assets
    assert bundle.floor == ready.floor
    assert bundle.environment == ready.environment
    assert bundle.execution_context == ready.context
    assert bundle.receipt.execution_context_sha256 == ready.context.sha256
    assert bundle.receipt.evidence_closure_sha256
    assert (directory / "sha256sums.txt").is_file()


def test_public_runner_derives_context_assets_and_ceiling_from_sealed_preflight(
    tmp_path: Path,
    ready: ReadyInputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _sealed_directory(tmp_path, ready)
    cell = ready.schedule.cells_for_phase("core")[0]
    task_assets = {item.task_id: item for item in ready.assets.tasks}
    target_assets = {item.target_id: item for item in ready.assets.targets}
    floor_by_task = {item.task_id: item for item in ready.floor.tasks}
    sentinel = object()

    def resolve_task(identifier: str) -> MatrixAxisIdentity:
        return MatrixAxisIdentity(
            kind="task",
            id=identifier,
            sha256=task_assets[identifier].task_pack_sha256,
        )

    def resolve_target(identifier: str) -> MatrixAxisIdentity:
        return MatrixAxisIdentity(
            kind="target",
            id=identifier,
            sha256=target_assets[identifier].target_stack_sha256,
        )

    def resolve_execution(matrix_cell) -> MatrixCellExecutionSpec:
        floor = floor_by_task[matrix_cell.task_id]
        assert floor.ceiling is not None
        return MatrixCellExecutionSpec(
            budget=AgentBudget(max_calls=3),
            policy=AgentLoopPolicy(
                response_parser="candidate_only",
                stop_policy="correct_latency",
                final_selection="best_correct_latency",
                latency_ceiling_ms=floor.ceiling.latency_ceiling_ms,
            ),
            dev_timing=TimingSpec(trial_runs=1, repetitions=1),
            model_ref="fake-model",
            initial_messages_sha256=_digest("messages"),
            execution_context_sha256=ready.context.sha256,
        )

    def capture_authorized(*args, **kwargs):
        assert args == (ready.pinned, "core")
        assert kwargs["execution_context"] == ready.context
        assert kwargs["resolve_task"](cell.task_id).sha256 == (
            task_assets[cell.task_id].task_pack_sha256
        )
        assert kwargs["resolve_target"](cell.target_id).sha256 == (
            target_assets[cell.target_id].target_stack_sha256
        )
        assert kwargs["resolve_execution"](cell).policy.latency_ceiling_ms == (
            floor_by_task[cell.task_id].ceiling.latency_ceiling_ms
        )
        return sentinel

    monkeypatch.setattr(matrix_runner, "_run_authorized_matrix_phase", capture_authorized)
    result = run_matrix_phase(
        ready.pinned,
        "core",
        artifact_root=tmp_path / "runs",
        preflight_directory=directory,
        live=True,
        expected_operational_request_ceiling=(
            ready.schedule.phase_operational_request_ceiling("core")
        ),
        resolve_task=resolve_task,
        resolve_target=resolve_target,
        resolve_agent=lambda identifier: MatrixAxisIdentity(
            kind="agent",
            id=identifier,
            sha256=_digest(identifier),
        ),
        resolve_execution=resolve_execution,
        runtime_factory=lambda _identity: None,
        schedule=ready.schedule,
    )

    assert result is sentinel


def _sealed_directory(tmp_path: Path, ready: ReadyInputs) -> Path:
    return seal_preflight_bundle(
        tmp_path / "artifacts",
        ready.pinned,
        ready.schedule,
        assets=ready.assets,
        floor=ready.floor,
        environment=ready.environment,
        execution_context=ready.context,
    )


def test_load_rejects_checksum_tampering(tmp_path: Path, ready: ReadyInputs) -> None:
    directory = _sealed_directory(tmp_path, ready)
    path = directory / "floor-manifest.json"
    path.chmod(0o600)
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(MatrixPreflightError, match="sealed matrix preflight bundle is invalid"):
        load_preflight_bundle(
            directory,
            ready.pinned,
            ready.schedule,
            execution_context=ready.context,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_load_rejects_missing_or_extra_semantic_json(
    tmp_path: Path,
    ready: ReadyInputs,
    mutation: str,
) -> None:
    directory = _sealed_directory(tmp_path, ready)
    directory.chmod(0o700)
    if mutation == "missing":
        path = directory / "asset-manifest.json"
        path.chmod(0o600)
        path.unlink()
    else:
        (directory / "extra.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(MatrixPreflightError, match="missing or extra semantic files"):
        load_preflight_bundle(
            directory,
            ready.pinned,
            ready.schedule,
            execution_context=ready.context,
        )


def test_load_rejects_artifact_symlink(tmp_path: Path, ready: ReadyInputs) -> None:
    directory = _sealed_directory(tmp_path, ready)
    link = tmp_path / "preflight-link"
    link.symlink_to(directory, target_is_directory=True)

    with pytest.raises(MatrixPreflightError, match="symbolic link"):
        load_preflight_bundle(
            link,
            ready.pinned,
            ready.schedule,
            execution_context=ready.context,
        )


def test_load_rejects_expected_context_drift(tmp_path: Path, ready: ReadyInputs) -> None:
    directory = _sealed_directory(tmp_path, ready)
    payload = ready.context.model_dump()
    payload["worker_revision"] = _digest("different-worker")[:40]
    different = MatrixExecutionContext.model_validate(payload)

    with pytest.raises(MatrixPreflightError, match="differs from the expected context"):
        load_preflight_bundle(
            directory,
            ready.pinned,
            ready.schedule,
            execution_context=different,
        )


def test_load_rejects_unknown_fields_even_with_recomputed_checksums(
    tmp_path: Path,
    ready: ReadyInputs,
) -> None:
    directory = _sealed_directory(tmp_path, ready)
    directory.chmod(0o700)
    path = directory / "environment-manifest.json"
    path.chmod(0o600)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = "field"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum = directory / "sha256sums.txt"
    checksum.chmod(0o600)
    lines = []
    for artifact in sorted(directory.rglob("*")):
        if artifact.is_file() and artifact != checksum:
            relative = artifact.relative_to(directory).as_posix()
            lines.append(f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}  {relative}")
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(MatrixPreflightError, match="sealed matrix preflight bundle is invalid"):
        load_preflight_bundle(
            directory,
            ready.pinned,
            ready.schedule,
            execution_context=ready.context,
        )
