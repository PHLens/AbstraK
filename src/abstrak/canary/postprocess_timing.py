"""Resume clean-process timing for qualified formal R1 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from abstrak.canary.artifacts import (
    TrajectoryArtifactError,
    TrajectoryStore,
    verify_trajectory,
)
from abstrak.canary.contracts import (
    CanaryModel,
    TargetStackSpec,
    TaskPackSpec,
    TimingSpec,
    TrajectoryOutcome,
    WorkerJob,
    WorkerResult,
)
from abstrak.canary.remote import SshWorkerExecutor
from abstrak.canary.schedule import build_r1_schedule
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import get_task_pack
from abstrak.canary.timing import TimingProtocolSummary, run_timing_protocol

DEFAULT_ARTIFACT_ROOT = "artifacts/r1-a100"
DEFAULT_FORMAL_STUDY_ID = "r1-a100-formal-v1"
DEFAULT_TIMING_STUDY_ID = "r1-a100-formal-timing-v1"
DEFAULT_WORKER_ROOT = "/workspace/volume/lipenghui/AbstraK"
DEFAULT_TIMING = TimingSpec()
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class PostprocessTimingError(RuntimeError):
    """Raised when formal inputs or resumed timing artifacts are invalid."""


class CandidateTimingRecord(CanaryModel):
    """One sealed timing result shared by identical first/final candidates."""

    schema_version: Literal["abstrak-candidate-timing-record.v1"] = (
        "abstrak-candidate-timing-record.v1"
    )
    trajectory_id: str
    task_id: str
    target_id: str
    candidate_labels: tuple[Literal["first", "final"], ...] = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    formal_artifact_directory: str
    artifact_directory: str
    summary: TimingProtocolSummary

    @model_validator(mode="after")
    def identities_match(self) -> CandidateTimingRecord:
        if len(self.candidate_labels) != len(set(self.candidate_labels)):
            raise ValueError("candidate timing labels must be unique")
        expected_prefix = (
            f"timing-{self.trajectory_id}-{'-'.join(self.candidate_labels)}"
        )
        if (
            self.summary.task_id != self.task_id
            or self.summary.target_id != self.target_id
            or self.summary.candidate_sha256 != self.source_sha256
            or self.summary.job_kind != "sealed"
            or self.summary.job_prefix != expected_prefix
        ):
            raise ValueError("candidate timing record and summary identities differ")
        return self


def _load_outcome(directory: Path) -> TrajectoryOutcome:
    try:
        verify_trajectory(directory)
        return TrajectoryOutcome.model_validate_json(
            (directory / "outcome.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise PostprocessTimingError(
            f"formal trajectory artifact is invalid: {directory}: {error}"
        ) from error


def _qualified_sources(
    directory: Path,
    outcome: TrajectoryOutcome,
    *,
    task: TaskPackSpec,
    target: TargetStackSpec,
) -> tuple[tuple[tuple[Literal["first", "final"], ...], str, str], ...]:
    by_hash: dict[str, tuple[str, list[Literal["first", "final"]]]] = {}
    for label in ("first", "final"):
        result = getattr(outcome, f"{label}_sealed_result")
        if result is None or result.status != "completed" or not result.correct:
            continue
        declared_hash = getattr(outcome, f"{label}_candidate_sha256")
        if declared_hash is None:
            raise PostprocessTimingError(
                f"qualified {label} candidate has no declared hash: {directory}"
            )
        try:
            source_bytes = (directory / "candidates" / f"{label}.py").read_bytes()
            source = source_bytes.decode("utf-8")
            metadata = json.loads(
                (directory / "candidates" / f"{label}.json").read_text(encoding="utf-8")
            )
            job = WorkerJob.model_validate_json(
                (directory / "sealed" / label / "worker-job.json").read_text(
                    encoding="utf-8"
                )
            )
            sealed_result = WorkerResult.model_validate_json(
                (directory / "sealed" / label / "worker-result.json").read_text(
                    encoding="utf-8"
                )
            )
            sealed_result.verify_for_job(job)
        except (OSError, ValueError) as error:
            raise PostprocessTimingError(
                f"qualified {label} candidate is unreadable: {directory}: {error}"
            ) from error
        actual_hash = hashlib.sha256(source_bytes).hexdigest()
        if (
            actual_hash != declared_hash
            or metadata != {"label": label, "sha256": actual_hash}
            or result != sealed_result
            or job.kind != "sealed"
            or job.task != task
            or job.target != target
            or job.candidate_sha256 != actual_hash
            or job.case_ids != tuple(case.id for case in task.sealed_cases)
        ):
            raise PostprocessTimingError(
                f"qualified {label} candidate provenance mismatch: {directory}"
            )
        existing = by_hash.get(actual_hash)
        if existing is None:
            by_hash[actual_hash] = (source, [label])
        else:
            if existing[0] != source:
                raise PostprocessTimingError("one candidate hash resolved to different sources")
            existing[1].append(label)
    return tuple(
        (tuple(labels), source, source_hash)
        for source_hash, (source, labels) in by_hash.items()
    )


def discover_qualified_candidates(
    *,
    root: str | Path,
    formal_study_id: str,
) -> tuple[tuple[str, str, str, Path, tuple[str, ...], str, str], ...]:
    """Validate the frozen schedule and return unique qualified sources in its order."""

    schedule = build_r1_schedule()
    base = Path(root).expanduser() / formal_study_id
    manifest = base / "study-manifest"
    try:
        verify_trajectory(manifest)
        manifest_value = json.loads((manifest / "run-manifest.json").read_text())
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise PostprocessTimingError(f"formal study manifest is invalid: {error}") from error
    if manifest_value.get("schedule_sha256") != schedule.sha256:
        raise PostprocessTimingError("formal study schedule hash does not match the registry")

    candidates: list[tuple[str, str, str, Path, tuple[str, ...], str, str]] = []
    for cell in schedule.formal:
        directory = base / cell.trajectory_id
        outcome = _load_outcome(directory)
        if outcome.trajectory_id != cell.trajectory_id:
            raise PostprocessTimingError(f"trajectory identity mismatch: {directory}")
        task = get_task_pack(cell.task_id)
        target = get_target_stack(cell.target_id)
        try:
            run_manifest = json.loads(
                (directory / "run-manifest.json").read_text(encoding="utf-8")
            )
            manifest_task = run_manifest["task"]
            manifest_target = run_manifest["target"]
            manifest_agent = run_manifest["resolved_provider"]["model"]["id"]
        except (KeyError, OSError, ValueError) as error:
            raise PostprocessTimingError(
                f"formal trajectory manifest is invalid: {directory}: {error}"
            ) from error
        if (
            run_manifest.get("study_id") != formal_study_id
            or run_manifest.get("trajectory_id") != cell.trajectory_id
            or manifest_task != task.model_dump(mode="json")
            or manifest_target != target.model_dump(mode="json")
            or manifest_agent != cell.agent_id
        ):
            raise PostprocessTimingError(
                f"formal trajectory manifest identity mismatch: {directory}"
            )
        for labels, source, source_hash in _qualified_sources(
            directory,
            outcome,
            task=task,
            target=target,
        ):
            candidates.append(
                (
                    cell.trajectory_id,
                    cell.task_id,
                    cell.target_id,
                    directory,
                    labels,
                    source,
                    source_hash,
                )
            )
    return tuple(candidates)


def _timing_id(trajectory_id: str, labels: tuple[str, ...]) -> str:
    return f"timing-{trajectory_id}-{'-'.join(labels)}"


def _controller_revision() -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PostprocessTimingError(f"cannot resolve controller revision: {error}") from error
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise PostprocessTimingError("controller revision is not a full lowercase Git SHA")
    return revision, dirty


def _ensure_timing_study_manifest(
    worker: SshWorkerExecutor,
    *,
    root: str | Path,
    formal_study_id: str,
    timing_study_id: str,
    candidates: tuple[tuple[str, str, str, Path, tuple[str, ...], str, str], ...],
    timing: TimingSpec,
    device: str,
) -> None:
    root_path = Path(root).expanduser()
    formal_directory = root_path / formal_study_id / "study-manifest"
    try:
        verify_trajectory(formal_directory)
        formal_bytes = (formal_directory / "run-manifest.json").read_bytes()
        formal_manifest = json.loads(formal_bytes)
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise PostprocessTimingError(f"formal study manifest is invalid: {error}") from error
    records = sorted(
        [
        {
            "trajectory_id": trajectory_id,
            "task_id": task_id,
            "target_id": target_id,
            "candidate_labels": labels,
            "source_sha256": source_hash,
            "artifact_id": _timing_id(trajectory_id, labels),
        }
        for trajectory_id, task_id, target_id, _, labels, _, source_hash in candidates
        ],
        key=lambda record: record["artifact_id"],
    )
    supervised = worker.sandbox_mode == "setpriv"
    transport = {
        "kind": "ssh",
        "host": worker.host,
        "worker_root": str(PurePosixPath(worker.pythonpath).parent),
        "worker_python": worker.python_executable,
        "worker_pythonpath": worker.pythonpath,
        "worker_kernelbench_root": worker.kernelbench_root,
        "worker_asset_root": worker.asset_root,
        "sandbox": "setpriv-supervised" if supervised else "bubblewrap",
        "network_isolated": not supervised,
        "filesystem_read_only": not supervised,
        "low_privilege": supervised,
        "device": device,
    }
    frozen = {
        "schema_version": "abstrak-candidate-timing-study-manifest.v1",
        "formal_study_id": formal_study_id,
        "timing_study_id": timing_study_id,
        "formal_schedule_sha256": formal_manifest["schedule_sha256"],
        "formal_manifest_sha256": hashlib.sha256(formal_bytes).hexdigest(),
        "worker_revision": formal_manifest["worker_revision"],
        "transport": transport,
        "timing": timing.model_dump(mode="json"),
        "candidate_count": len(records),
        "records": records,
    }
    frozen = json.loads(json.dumps(frozen, sort_keys=True))
    directory = root_path / timing_study_id / "study-manifest"
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if directory.exists():
        try:
            verify_trajectory(directory)
            actual = json.loads(
                (directory / "run-manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TrajectoryArtifactError) as error:
            raise PostprocessTimingError(
                f"timing study manifest is invalid: {directory}: {error}"
            ) from error
        if any(actual.get(key) != value for key, value in frozen.items()):
            raise PostprocessTimingError("timing study manifest differs from frozen inputs")
        incomplete = any(
            not (root_path / timing_study_id / record["artifact_id"]).is_dir()
            for record in records
        )
        if incomplete and actual.get("execution_controller_source_sha256") != source_sha256:
            raise PostprocessTimingError(
                "cannot resume an incomplete timing study with changed controller source"
            )
        return

    revision, dirty = _controller_revision()
    store = TrajectoryStore.create(root_path, timing_study_id, "study-manifest")
    store.write_json(
        "run-manifest.json",
        {
            **frozen,
            "execution_controller_git_revision": revision,
            "execution_controller_source_sha256": source_sha256,
            "execution_controller_worktree_dirty": dirty,
            "manifest_created_after_timing_completion": False,
        },
    )
    store.seal()


def _load_existing(
    path: Path,
    *,
    expected_manifest: dict[str, object] | None = None,
) -> CandidateTimingRecord | None:
    if not path.exists():
        return None
    try:
        verify_trajectory(path)
        record = CandidateTimingRecord.model_validate_json(
            (path / "timing-record.json").read_text(encoding="utf-8")
        )
        if expected_manifest is not None:
            manifest = json.loads(
                (path / "run-manifest.json").read_text(encoding="utf-8")
            )
            normalized_expected = json.loads(
                json.dumps(
                    expected_manifest,
                    default=lambda value: value.model_dump(mode="json"),
                    sort_keys=True,
                )
            )
            if manifest != normalized_expected:
                raise ValueError("timing run manifest differs from frozen inputs")
        return record
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise PostprocessTimingError(
            f"existing timing artifact is invalid: {path}: {error}"
        ) from error


def run_formal_candidate_timing(
    worker: SshWorkerExecutor,
    *,
    root: str | Path,
    formal_study_id: str = DEFAULT_FORMAL_STUDY_ID,
    timing_study_id: str = DEFAULT_TIMING_STUDY_ID,
    timing: TimingSpec = DEFAULT_TIMING,
    device: str = "cuda:0",
) -> tuple[CandidateTimingRecord, ...]:
    """Run or resume every unique qualified first/final candidate."""

    candidates = discover_qualified_candidates(root=root, formal_study_id=formal_study_id)
    records: list[CandidateTimingRecord] = []
    for index, (
        trajectory_id,
        task_id,
        target_id,
        formal_directory,
        labels,
        source,
        source_hash,
    ) in enumerate(candidates, start=1):
        timing_id = _timing_id(trajectory_id, labels)
        path = Path(root).expanduser() / timing_study_id / timing_id
        expected_manifest = {
            "schema_version": "abstrak-candidate-timing-manifest.v1",
            "formal_study_id": formal_study_id,
            "timing_study_id": timing_study_id,
            "trajectory_id": trajectory_id,
            "task_id": task_id,
            "target_id": target_id,
            "candidate_labels": labels,
            "source_sha256": source_hash,
            "timing": timing,
            "device": device,
        }
        existing = _load_existing(path, expected_manifest=expected_manifest)
        if existing is not None:
            identity = (
                existing.trajectory_id,
                existing.task_id,
                existing.target_id,
                existing.candidate_labels,
                existing.source_sha256,
                existing.summary.timing,
                existing.summary.device,
            )
            expected = (
                trajectory_id,
                task_id,
                target_id,
                labels,
                source_hash,
                timing,
                device,
            )
            if identity != expected:
                raise PostprocessTimingError(
                    f"existing timing artifact does not match frozen inputs: {path}"
                )
            record = existing
            resumed = True
        else:
            staging_path = path.with_name(f"{timing_id}.incomplete")
            if staging_path.exists():
                try:
                    staged = _load_existing(
                        staging_path,
                        expected_manifest=expected_manifest,
                    )
                except PostprocessTimingError:
                    shutil.rmtree(staging_path)
                else:
                    assert staged is not None
                    if staged.artifact_directory != str(path):
                        raise PostprocessTimingError(
                            f"staged timing artifact has the wrong final path: {staging_path}"
                        )
                    os.replace(staging_path, path)
                    record = staged
                    records.append(record)
                    print(
                        json.dumps(
                            {
                                "progress": f"{index}/{len(candidates)}",
                                "trajectory_id": trajectory_id,
                                "labels": labels,
                                "status": record.summary.status,
                                "stable": record.summary.stable,
                                "median_ms": record.summary.median_ms,
                                "resumed": True,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
            summary = run_timing_protocol(
                worker,
                task=get_task_pack(task_id),
                target=get_target_stack(target_id),
                source=source,
                job_prefix=timing_id,
                device=device,
                timing=timing,
                job_kind="sealed",
            )
            staging_id = f"{timing_id}.incomplete"
            store = TrajectoryStore.create(root, timing_study_id, staging_id)
            record = CandidateTimingRecord(
                trajectory_id=trajectory_id,
                task_id=task_id,
                target_id=target_id,
                candidate_labels=labels,
                source_sha256=source_hash,
                formal_artifact_directory=str(formal_directory),
                artifact_directory=str(path),
                summary=summary,
            )
            store.write_json("run-manifest.json", expected_manifest)
            store.write_json("timing-record.json", record)
            store.seal()
            os.replace(store.run_directory, path)
            resumed = False
        records.append(record)
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(candidates)}",
                    "trajectory_id": trajectory_id,
                    "labels": labels,
                    "status": record.summary.status,
                    "stable": record.summary.stable,
                    "median_ms": record.summary.median_ms,
                    "resumed": resumed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return tuple(records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--expected-qualified-candidates", type=int, required=True)
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--formal-study-id", default=DEFAULT_FORMAL_STUDY_ID)
    parser.add_argument("--timing-study-id", default=DEFAULT_TIMING_STUDY_ID)
    parser.add_argument("--ssh-host", default="a100-r1")
    parser.add_argument("--worker-root", default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--worker-python")
    parser.add_argument("--worker-pythonpath")
    parser.add_argument("--worker-kernelbench-root")
    parser.add_argument("--worker-asset-root")
    parser.add_argument("--worker-timeout", type=float, default=1200.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-supervised-worker", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.live:
        print("configuration error: candidate timing requires --live", file=sys.stderr)
        return 2
    if arguments.worker_timeout <= 0:
        print("configuration error: --worker-timeout must be positive", file=sys.stderr)
        return 2
    try:
        candidates = discover_qualified_candidates(
            root=arguments.artifact_root,
            formal_study_id=arguments.formal_study_id,
        )
        if arguments.expected_qualified_candidates != len(candidates):
            raise PostprocessTimingError(
                "--expected-qualified-candidates must equal the discovered sealed count "
                f"({len(candidates)})"
            )
        root = PurePosixPath(arguments.worker_root)
        worker = SshWorkerExecutor(
            arguments.ssh_host,
            python_executable=(
                arguments.worker_python or "/tmp/abstrak-gpu-venv/bin/python"
            ),
            pythonpath=arguments.worker_pythonpath or str(root / "src"),
            kernelbench_root=(
                arguments.worker_kernelbench_root or str(root.parent / "KernelBench")
            ),
            asset_root=(
                arguments.worker_asset_root or str(root / "benchmarks" / "r1-a100")
            ),
            device=arguments.device,
            timeout_seconds=arguments.worker_timeout,
            expected_hardware_substring="A100",
            expected_compute_capability=(8, 0),
            expected_triton_version="3.7.1",
            sandbox_mode=(
                "setpriv" if arguments.allow_supervised_worker else "bubblewrap"
            ),
        )
        _ensure_timing_study_manifest(
            worker,
            root=arguments.artifact_root,
            formal_study_id=arguments.formal_study_id,
            timing_study_id=arguments.timing_study_id,
            candidates=candidates,
            timing=DEFAULT_TIMING,
            device=arguments.device,
        )
        records = run_formal_candidate_timing(
            worker,
            root=arguments.artifact_root,
            formal_study_id=arguments.formal_study_id,
            timing_study_id=arguments.timing_study_id,
            device=arguments.device,
        )
    except (
        OSError,
        PostprocessTimingError,
        TrajectoryArtifactError,
        ValueError,
    ) as error:
        print(f"candidate timing error: {error}", file=sys.stderr)
        return 5
    print(
        json.dumps(
            {
                "status": "complete",
                "candidate_count": len(records),
                "stable_count": sum(record.summary.stable for record in records),
                "timing_study_id": arguments.timing_study_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
