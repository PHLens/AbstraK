"""Resumable expert-oracle and common-baseline gate execution."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import Field

from abstrak.canary.artifacts import TrajectoryArtifactError, TrajectoryStore, verify_trajectory
from abstrak.canary.baselines import (
    BASELINE_VARIANTS,
    get_baseline_source,
)
from abstrak.canary.contracts import CanaryModel, TargetStackSpec, TaskPackSpec, TimingSpec
from abstrak.canary.loop import WorkerExecutor
from abstrak.canary.tasks import load_oracle_source
from abstrak.canary.timing import TimingProtocolSummary, run_timing_protocol

GateKind = Literal["oracle", "baseline"]
DEFAULT_GATE_TIMING = TimingSpec()


class GateError(ValueError):
    """Raised when a gate matrix is incomplete or its artifact is invalid."""


class GateRecord(CanaryModel):
    """One sealed gate summary and its content-addressed source."""

    schema_version: Literal["abstrak-canary-gate-record.v1"] = (
        "abstrak-canary-gate-record.v1"
    )
    kind: GateKind
    task_id: str
    target_id: str
    variant: str | None = None
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_directory: str
    summary: TimingProtocolSummary


def _gate_id(kind: GateKind, task_id: str, target_id: str, variant: str | None) -> str:
    suffix = "" if variant is None else f"-{variant}"
    return f"{kind}-{task_id}-{target_id}{suffix}"


def _load_existing(path: Path) -> GateRecord | None:
    if not path.exists():
        return None
    try:
        verify_trajectory(path)
        return GateRecord.model_validate_json((path / "gate-record.json").read_text())
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise GateError(f"existing gate artifact is invalid: {path}: {error}") from error


def _run_one(
    worker: WorkerExecutor,
    *,
    root: str | Path,
    study_id: str,
    kind: GateKind,
    task: TaskPackSpec,
    target: TargetStackSpec,
    source: str,
    variant: str | None,
    timing: TimingSpec,
    device: str,
) -> GateRecord:
    gate_id = _gate_id(kind, task.id, target.id, variant)
    path = Path(root).expanduser() / study_id / gate_id
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    existing = _load_existing(path)
    if existing is not None:
        identity = (
            existing.kind,
            existing.task_id,
            existing.target_id,
            existing.variant,
            existing.source_sha256,
            existing.summary.timing,
        )
        expected = (kind, task.id, target.id, variant, source_sha256, timing)
        if identity != expected:
            raise GateError(f"existing gate artifact does not match frozen inputs: {path}")
        return existing
    summary = run_timing_protocol(
        worker,
        task=task,
        target=target,
        source=source,
        job_prefix=gate_id,
        device=device,
        timing=timing,
        job_kind=kind,
    )
    store = TrajectoryStore.create(root, study_id, gate_id)
    record = GateRecord(
        kind=kind,
        task_id=task.id,
        target_id=target.id,
        variant=variant,
        source_sha256=source_sha256,
        artifact_directory=str(store.run_directory),
        summary=summary,
    )
    store.write_json(
        "run-manifest.json",
        {
            "schema_version": "abstrak-canary-gate-manifest.v1",
            "kind": kind,
            "task_id": task.id,
            "target_id": target.id,
            "variant": variant,
            "timing": timing,
        },
    )
    store.write_json("gate-record.json", record)
    store.seal()
    return record


def run_oracle_gates(
    worker: WorkerExecutor,
    *,
    tasks: Iterable[TaskPackSpec],
    targets: Iterable[TargetStackSpec],
    root: str | Path,
    study_id: str = "r1-a100-oracle-gates",
    timing: TimingSpec = DEFAULT_GATE_TIMING,
    asset_root: str | Path | None = None,
    device: str = "cuda:0",
) -> tuple[GateRecord, ...]:
    """Run or resume every task/target expert gate in stable registry order."""

    records: list[GateRecord] = []
    for task in tasks:
        for target in targets:
            source = load_oracle_source(task.id, target.backend, asset_root=asset_root)
            records.append(
                _run_one(
                    worker,
                    root=root,
                    study_id=study_id,
                    kind="oracle",
                    task=task,
                    target=target,
                    source=source,
                    variant=None,
                    timing=timing,
                    device=device,
                )
            )
    return tuple(records)


def run_baseline_gates(
    worker: WorkerExecutor,
    *,
    tasks: Iterable[TaskPackSpec],
    target: TargetStackSpec,
    root: str | Path,
    study_id: str = "r1-a100-baseline-gates",
    timing: TimingSpec = DEFAULT_GATE_TIMING,
    device: str = "cuda:0",
) -> tuple[GateRecord, ...]:
    """Run or resume eager/compile/vendor baselines for every formal task."""

    records: list[GateRecord] = []
    for task in tasks:
        for variant in BASELINE_VARIANTS:
            source = get_baseline_source(task.id, variant).source
            records.append(
                _run_one(
                    worker,
                    root=root,
                    study_id=study_id,
                    kind="baseline",
                    task=task,
                    target=target,
                    source=source,
                    variant=variant,
                    timing=timing,
                    device=device,
                )
            )
    return tuple(records)


def fastest_stable_baselines(records: Iterable[GateRecord]) -> dict[str, GateRecord]:
    """Select B* strictly from stable baseline records; never fall back silently."""

    by_task: dict[str, list[GateRecord]] = {}
    for record in records:
        if record.kind != "baseline":
            continue
        by_task.setdefault(record.task_id, []).append(record)
    selected: dict[str, GateRecord] = {}
    for task_id, candidates in by_task.items():
        stable = [record for record in candidates if record.summary.stable]
        if not stable:
            raise GateError(f"no stable B* baseline for task: {task_id}")
        selected[task_id] = min(stable, key=lambda record: record.summary.median_ms or float("inf"))
    return selected
