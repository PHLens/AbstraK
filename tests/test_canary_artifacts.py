from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from abstrak.canary.artifacts import (
    TrajectoryArtifactError,
    TrajectoryStore,
    verify_trajectory,
)


def test_trajectory_store_writes_turn_events_snapshots_and_seals(tmp_path: Path) -> None:
    store = TrajectoryStore.create(tmp_path, "study", "trajectory")
    candidate = "class ModelNew: pass\n"
    digest = hashlib.sha256(candidate.encode()).hexdigest()

    store.append_event(0, "request_started", 0, {"request_id": "request-1"})
    store.write_turn(
        0,
        request={"request_id": "request-1"},
        response={"text": "response"},
        candidate=candidate,
        worker_job={"job_id": "job-1"},
        worker_result={"status": "completed"},
    )
    store.snapshot_candidate("first", candidate, digest)
    store.snapshot_candidate("final", candidate, digest)
    store.write_sealed("first", {"job_id": "sealed-1"}, {"status": "completed"})
    store.write_sealed("final", {"job_id": "sealed-2"}, {"status": "completed"})
    checksum = store.seal()

    assert checksum.is_file()
    assert json.loads((store.run_directory / "events/0000.json").read_text())["sequence"] == 0
    verify_trajectory(store.run_directory)


def test_event_sequence_and_overwrite_fail_closed(tmp_path: Path) -> None:
    store = TrajectoryStore.create(tmp_path, "study", "trajectory")
    with pytest.raises(TrajectoryArtifactError, match="sequence must be 0"):
        store.append_event(1, "bad", None, {})
    store.write_json("value.json", {"value": 1})
    with pytest.raises(TrajectoryArtifactError, match="already exists"):
        store.write_json("value.json", {"value": 2})

    with pytest.raises(TrajectoryArtifactError, match="study_id"):
        TrajectoryStore.create(tmp_path, "..", "escaped")


def test_store_rejects_secrets_and_candidate_hash_mismatch(tmp_path: Path) -> None:
    store = TrajectoryStore.create(
        tmp_path,
        "study",
        "trajectory",
        secrets=("unit-test-secret",),
    )
    with pytest.raises(TrajectoryArtifactError, match="credential"):
        store.write_text("secret.txt", "unit-test-secret")
    with pytest.raises(TrajectoryArtifactError, match="hash mismatch"):
        store.snapshot_candidate("first", "source", "0" * 64)


def test_verify_detects_tampering(tmp_path: Path) -> None:
    store = TrajectoryStore.create(tmp_path, "study", "trajectory")
    path = store.write_text("artifact.txt", "original")
    store.seal()
    path.chmod(0o600)
    path.write_text("changed", encoding="utf-8")

    with pytest.raises(TrajectoryArtifactError, match="checksum mismatch"):
        verify_trajectory(store.run_directory)
