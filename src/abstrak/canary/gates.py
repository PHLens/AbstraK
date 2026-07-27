"""Resumable expert-oracle and common-baseline gate execution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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
_STORE_DIRECTORIES = frozenset({"events", "turns", "candidates", "sealed"})
_GATE_FILES = frozenset(
    {"run-manifest.json", "gate-record.json", "sha256sums.txt"}
)


class GateError(ValueError):
    """Raised when a gate matrix is incomplete or its artifact is invalid."""


class GateInfrastructureError(GateError):
    """Raised when infrastructure prevents a gate from reaching a scientific result."""


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


def _load_existing(
    path: Path,
    *,
    final_path: Path | None = None,
) -> GateRecord | None:
    if path.is_symlink():
        raise GateError(f"gate artifact cannot be a symbolic link: {path}")
    if not path.exists():
        return None
    if not path.is_dir():
        raise GateError(f"gate artifact is not a regular directory: {path}")
    try:
        entries = tuple(path.rglob("*"))
        if any(item.is_symlink() for item in entries):
            raise GateError(f"gate artifact contains a symbolic link: {path}")
        files = {
            item.relative_to(path).as_posix()
            for item in entries
            if item.is_file()
        }
        directories = {
            item.relative_to(path).as_posix()
            for item in entries
            if item.is_dir()
        }
        if files != _GATE_FILES or directories != _STORE_DIRECTORIES:
            raise GateError(f"gate artifact has an unexpected shape: {path}")
        verify_trajectory(path)
        record = GateRecord.model_validate_json(
            (path / "gate-record.json").read_text(encoding="utf-8")
        )
        expected_directory = path if final_path is None else final_path
        if (
            Path(record.artifact_directory).expanduser().resolve(strict=False)
            != expected_directory.resolve(strict=False)
        ):
            raise GateError(
                f"gate record does not identify its final artifact directory: {path}"
            )
        return record
    except GateError:
        raise
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise GateError(f"existing gate artifact is invalid: {path}: {error}") from error


def _require_frozen_inputs(
    record: GateRecord,
    *,
    path: Path,
    kind: GateKind,
    task: TaskPackSpec,
    target: TargetStackSpec,
    variant: str | None,
    source_sha256: str,
    timing: TimingSpec,
    device: str,
) -> None:
    gate_id = _gate_id(kind, task.id, target.id, variant)
    identity = (
        record.kind,
        record.task_id,
        record.target_id,
        record.variant,
        record.source_sha256,
        record.summary.task_id,
        record.summary.target_id,
        record.summary.candidate_sha256,
        record.summary.job_prefix,
        record.summary.job_kind,
        record.summary.device,
        record.summary.timing,
    )
    expected = (
        kind,
        task.id,
        target.id,
        variant,
        source_sha256,
        task.id,
        target.id,
        source_sha256,
        gate_id,
        kind,
        device,
        timing,
    )
    if identity != expected:
        raise GateError(f"existing gate artifact does not match frozen inputs: {path}")
    try:
        manifest = json.loads(
            (path / "run-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise GateError(f"existing gate run manifest is invalid: {path}") from error
    expected_manifest = {
        "schema_version": "abstrak-canary-gate-manifest.v1",
        "kind": kind,
        "task_id": task.id,
        "target_id": target.id,
        "variant": variant,
        "timing": timing.model_dump(mode="json"),
    }
    if manifest != expected_manifest:
        raise GateError(
            f"existing gate run manifest does not match frozen inputs: {path}"
        )


def _remove_unsealed_staging(path: Path) -> bool:
    if path.is_symlink():
        raise GateError(f"gate staging artifact cannot be a symbolic link: {path}")
    if not path.exists():
        return False
    if not path.is_dir():
        raise GateError(f"gate staging artifact is not a regular directory: {path}")
    if (path / "sha256sums.txt").exists():
        return False
    try:
        shutil.rmtree(path)
    except OSError as error:
        raise GateError(f"cannot discard unsealed gate staging artifact: {path}") from error
    return True


def _verify_gate_study_directory(
    root: str | Path,
    study_id: str,
    gate_ids: Iterable[str],
) -> None:
    study_root = Path(root).expanduser() / study_id
    if study_root.is_symlink():
        raise GateError(f"gate study cannot be a symbolic link: {study_root}")
    if not study_root.exists():
        return
    if not study_root.is_dir():
        raise GateError(f"gate study is not a regular directory: {study_root}")
    try:
        entries = tuple(study_root.iterdir())
    except OSError as error:
        raise GateError(f"cannot inspect gate study directory: {study_root}") from error
    if any(entry.is_symlink() for entry in entries):
        raise GateError(f"gate study cannot contain symbolic links: {study_root}")
    allowed = {
        name
        for gate_id in gate_ids
        for name in (gate_id, f"{gate_id}.incomplete")
    }
    unexpected = sorted(entry.name for entry in entries if entry.name not in allowed)
    if unexpected:
        raise GateError(
            f"gate study contains unexpected artifacts: {unexpected}"
        )


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
    staging = path.with_name(f"{gate_id}.incomplete")
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if path.is_symlink():
        raise GateError(f"gate artifact cannot be a symbolic link: {path}")
    if staging.is_symlink():
        raise GateError(f"gate staging artifact cannot be a symbolic link: {staging}")
    if path.exists() and staging.exists():
        raise GateError(f"final and staging gate artifacts both exist: {gate_id}")
    existing = _load_existing(path)
    if existing is not None:
        _require_frozen_inputs(
            existing,
            path=path,
            kind=kind,
            task=task,
            target=target,
            variant=variant,
            source_sha256=source_sha256,
            timing=timing,
            device=device,
        )
        return existing
    if staging.exists():
        removed = _remove_unsealed_staging(staging)
        if not removed:
            staged = _load_existing(staging, final_path=path)
            if staged is None:
                raise GateError(f"sealed gate staging artifact disappeared: {staging}")
            _require_frozen_inputs(
                staged,
                path=staging,
                kind=kind,
                task=task,
                target=target,
                variant=variant,
                source_sha256=source_sha256,
                timing=timing,
                device=device,
            )
            if path.exists() or path.is_symlink():
                raise GateError(f"gate artifact appeared during staging resume: {gate_id}")
            try:
                os.replace(staging, path)
            except OSError as error:
                raise GateError(
                    f"cannot promote sealed gate staging artifact: {staging}"
                ) from error
            persisted = _load_existing(path)
            if persisted is None:
                raise GateError(f"promoted gate artifact disappeared: {path}")
            return persisted
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
    if summary.status == "worker_failure":
        detail = summary.error or "timing worker failed without an error message"
        raise GateInfrastructureError(f"gate infrastructure failure: {gate_id}: {detail}")
    try:
        store = TrajectoryStore.create(root, study_id, f"{gate_id}.incomplete")
    except TrajectoryArtifactError as error:
        raise GateError(f"cannot create gate staging artifact: {staging}") from error
    record = GateRecord(
        kind=kind,
        task_id=task.id,
        target_id=target.id,
        variant=variant,
        source_sha256=source_sha256,
        artifact_directory=str(path.resolve(strict=False)),
        summary=summary,
    )
    try:
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
        if path.exists() or path.is_symlink():
            raise GateError(f"gate artifact appeared during atomic staging: {gate_id}")
        os.replace(store.run_directory, path)
    except GateError:
        raise
    except (OSError, TrajectoryArtifactError) as error:
        raise GateError(f"cannot seal gate artifact: {gate_id}") from error
    persisted = _load_existing(path)
    if persisted is None:
        raise GateError(f"sealed gate artifact disappeared: {path}")
    return persisted


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

    task_values = tuple(tasks)
    target_values = tuple(targets)
    gate_ids = tuple(
        _gate_id("oracle", task.id, target.id, None)
        for task in task_values
        for target in target_values
    )
    _verify_gate_study_directory(root, study_id, gate_ids)
    records: list[GateRecord] = []
    for task in task_values:
        for target in target_values:
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
    _verify_gate_study_directory(root, study_id, gate_ids)
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

    task_values = tuple(tasks)
    gate_ids = tuple(
        _gate_id("baseline", task.id, target.id, variant)
        for task in task_values
        for variant in BASELINE_VARIANTS
    )
    _verify_gate_study_directory(root, study_id, gate_ids)
    records: list[GateRecord] = []
    for task in task_values:
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
    _verify_gate_study_directory(root, study_id, gate_ids)
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
