"""Resumable serial runner for the frozen 48-cell A100 R1 formal matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from abstrak.canary.artifacts import TrajectoryArtifactError, TrajectoryStore, verify_trajectory
from abstrak.canary.contracts import CanaryModel, TrajectoryOutcome
from abstrak.canary.schedule import R1StudySchedule, ScheduleCell, build_r1_schedule

FORMAL_REQUEST_CEILING = 48 * 4
DEFAULT_STUDY_ID = "r1-a100-formal-v1"
DEFAULT_ARTIFACT_ROOT = "artifacts/r1-a100"
DEFAULT_WORKER_ROOT = "/workspace/volume/lipenghui/AbstraK"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class StudyExecutionError(RuntimeError):
    """Raised when a formal cell cannot produce or resume one sealed outcome."""


class FormalCellRecord(CanaryModel):
    """Compact controller record for one sealed trajectory outcome."""

    trajectory_id: str
    ordinal: int = Field(ge=0)
    agent_id: str
    task_id: str
    target_id: str
    replicate: int = Field(ge=1)
    resumed: bool
    outcome_status: str
    calls: int = Field(ge=0, le=4)
    final_qualified: bool
    artifact_directory: str


class FormalStudySummary(CanaryModel):
    """Terminal collection status for the exact frozen formal schedule."""

    schema_version: Literal["abstrak-canary-formal-run.v1"] = (
        "abstrak-canary-formal-run.v1"
    )
    study_id: str
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_cells: int = 48
    completed_cells: int = Field(ge=0, le=48)
    resumed_cells: int = Field(ge=0, le=48)
    qualified_final_cells: int = Field(ge=0, le=48)
    total_calls: int = Field(ge=0, le=192)
    records: tuple[FormalCellRecord, ...]


@dataclass(frozen=True)
class StudyRuntime:
    artifact_root: str
    study_id: str
    ssh_host: str
    worker_root: str
    worker_timeout: float
    allow_supervised_worker: bool
    revision: str
    config: str | None = None
    auth: str | None = None
    asset_root: str | None = None
    worker_python: str | None = None
    worker_pythonpath: str | None = None
    worker_kernelbench_root: str | None = None
    worker_asset_root: str | None = None


RunProcess = Callable[..., subprocess.CompletedProcess[str]]


def _manifest_payload(schedule: R1StudySchedule, runtime: StudyRuntime) -> dict[str, object]:
    return {
        "schema_version": "abstrak-canary-formal-manifest.v1",
        "study_id": runtime.study_id,
        "schedule": schedule,
        "schedule_sha256": schedule.sha256,
        "request_ceiling": FORMAL_REQUEST_CEILING,
        "controller_revision": runtime.revision,
        "worker_revision": runtime.revision,
        "transport": {
            "kind": "ssh",
            "host": runtime.ssh_host,
            "worker_root": runtime.worker_root,
            "worker_timeout": runtime.worker_timeout,
            "sandbox": (
                "setpriv-supervised" if runtime.allow_supervised_worker else "bubblewrap"
            ),
        },
    }


def _ensure_manifest(
    schedule: R1StudySchedule,
    runtime: StudyRuntime,
) -> None:
    directory = Path(runtime.artifact_root).expanduser() / runtime.study_id / "study-manifest"
    expected = _manifest_payload(schedule, runtime)
    if directory.exists():
        try:
            verify_trajectory(directory)
            actual = json.loads((directory / "run-manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TrajectoryArtifactError) as error:
            raise StudyExecutionError(f"formal study manifest is invalid: {error}") from error
        rendered_expected = json.loads(
            json.dumps(
                expected,
                default=lambda value: value.model_dump(mode="json"),
                sort_keys=True,
            )
        )
        if actual != rendered_expected:
            raise StudyExecutionError("formal study manifest differs from frozen runtime inputs")
        return
    store = TrajectoryStore.create(runtime.artifact_root, runtime.study_id, "study-manifest")
    store.write_json("run-manifest.json", expected)
    store.seal()


def _cell_command(cell: ScheduleCell, runtime: StudyRuntime) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "abstrak.canary.cli",
        "run-cell",
        "--task",
        cell.task_id,
        "--target",
        cell.target_id,
        "--profile",
        cell.agent_id,
        "--trajectory-id",
        cell.trajectory_id,
        "--study-id",
        runtime.study_id,
        "--artifact-root",
        runtime.artifact_root,
        "--live",
        "--expected-max-requests",
        "4",
        "--ssh-host",
        runtime.ssh_host,
        "--worker-root",
        runtime.worker_root,
        "--worker-timeout",
        f"{runtime.worker_timeout:g}",
    ]
    optional = (
        ("--config", runtime.config),
        ("--auth", runtime.auth),
        ("--asset-root", runtime.asset_root),
        ("--worker-python", runtime.worker_python),
        ("--worker-pythonpath", runtime.worker_pythonpath),
        ("--worker-kernelbench-root", runtime.worker_kernelbench_root),
        ("--worker-asset-root", runtime.worker_asset_root),
    )
    for flag, value in optional:
        if value is not None:
            command.extend((flag, value))
    if runtime.allow_supervised_worker:
        command.append("--allow-supervised-worker")
    return command


def _load_outcome(path: Path) -> TrajectoryOutcome:
    try:
        verify_trajectory(path)
        return TrajectoryOutcome.model_validate_json(
            (path / "outcome.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise StudyExecutionError(f"trajectory artifact is invalid: {path}: {error}") from error


def _record(
    cell: ScheduleCell,
    path: Path,
    outcome: TrajectoryOutcome,
    *,
    resumed: bool,
) -> FormalCellRecord:
    final = outcome.final_sealed_result
    qualified = final is not None and final.status == "completed" and final.correct
    return FormalCellRecord(
        trajectory_id=cell.trajectory_id,
        ordinal=cell.ordinal,
        agent_id=cell.agent_id,
        task_id=cell.task_id,
        target_id=cell.target_id,
        replicate=cell.replicate,
        resumed=resumed,
        outcome_status=outcome.status,
        calls=outcome.calls,
        final_qualified=qualified,
        artifact_directory=str(path),
    )


def run_formal_study(
    runtime: StudyRuntime,
    *,
    schedule: R1StudySchedule | None = None,
    run_process: RunProcess = subprocess.run,
) -> FormalStudySummary:
    """Run every pending formal cell once and resume only sealed terminal artifacts."""

    frozen = schedule or build_r1_schedule()
    _ensure_manifest(frozen, runtime)
    records: list[FormalCellRecord] = []
    for cell in frozen.formal:
        path = Path(runtime.artifact_root).expanduser() / runtime.study_id / cell.trajectory_id
        if path.exists():
            outcome = _load_outcome(path)
            record = _record(cell, path, outcome, resumed=True)
        else:
            try:
                process = run_process(
                    _cell_command(cell, runtime),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1800.0,
                )
            except subprocess.TimeoutExpired as error:
                raise StudyExecutionError(
                    f"controller subprocess timed out before sealing {cell.trajectory_id}"
                ) from error
            if not path.exists():
                diagnostic = process.stderr.strip()[-1000:]
                raise StudyExecutionError(
                    f"cell did not create an artifact: {cell.trajectory_id}: {diagnostic}"
                )
            outcome = _load_outcome(path)
            record = _record(cell, path, outcome, resumed=False)
        records.append(record)
        print(
            json.dumps(
                {
                    "progress": f"{len(records)}/{len(frozen.formal)}",
                    "trajectory_id": record.trajectory_id,
                    "status": record.outcome_status,
                    "calls": record.calls,
                    "final_qualified": record.final_qualified,
                    "resumed": record.resumed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return FormalStudySummary(
        study_id=runtime.study_id,
        schedule_sha256=frozen.sha256,
        completed_cells=len(records),
        resumed_cells=sum(record.resumed for record in records),
        qualified_final_cells=sum(record.final_qualified for record in records),
        total_calls=sum(record.calls for record in records),
        records=tuple(records),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--expected-max-requests", type=int, required=True)
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--study-id", default=DEFAULT_STUDY_ID)
    parser.add_argument("--ssh-host", default="a100-r1")
    parser.add_argument("--worker-root", default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--worker-timeout", type=float, default=1200.0)
    parser.add_argument("--allow-supervised-worker", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--auth")
    parser.add_argument("--asset-root")
    parser.add_argument("--worker-python")
    parser.add_argument("--worker-pythonpath")
    parser.add_argument("--worker-kernelbench-root")
    parser.add_argument("--worker-asset-root")
    return parser


def _repository_revision() -> str:
    try:
        process = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise StudyExecutionError(f"cannot resolve controller revision: {error}") from error
    revision = process.stdout.strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise StudyExecutionError("controller revision is not one full lowercase Git SHA")
    return revision


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.live:
        print("configuration error: formal study requires --live", file=sys.stderr)
        return 2
    if arguments.expected_max_requests != FORMAL_REQUEST_CEILING:
        print(
            f"configuration error: --expected-max-requests must equal "
            f"{FORMAL_REQUEST_CEILING}",
            file=sys.stderr,
        )
        return 2
    if arguments.worker_timeout <= 0:
        print("configuration error: --worker-timeout must be positive", file=sys.stderr)
        return 2
    try:
        runtime = StudyRuntime(
            artifact_root=arguments.artifact_root,
            study_id=arguments.study_id,
            ssh_host=arguments.ssh_host,
            worker_root=arguments.worker_root,
            worker_timeout=arguments.worker_timeout,
            allow_supervised_worker=arguments.allow_supervised_worker,
            revision=_repository_revision(),
            config=arguments.config,
            auth=arguments.auth,
            asset_root=arguments.asset_root,
            worker_python=arguments.worker_python,
            worker_pythonpath=arguments.worker_pythonpath,
            worker_kernelbench_root=arguments.worker_kernelbench_root,
            worker_asset_root=arguments.worker_asset_root,
        )
        summary = run_formal_study(runtime)
    except StudyExecutionError as error:
        print(f"study error: {error}", file=sys.stderr)
        return 5
    print(summary.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
