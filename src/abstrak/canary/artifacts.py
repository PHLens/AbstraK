"""Crash-resistant private artifacts for one multi-turn canary trajectory."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from abstrak.providers.contracts import sha256_json


class TrajectoryArtifactError(RuntimeError):
    pass


def _payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _payload(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _payload(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_payload(item) for item in value]
    return value


class TrajectoryStore:
    """One append-only trajectory directory with atomic individual artifacts."""

    def __init__(self, run_directory: Path, *, secrets: tuple[str, ...] = ()) -> None:
        self.run_directory = run_directory
        self._secret_bytes = tuple(value.encode() for value in secrets if value)
        self._next_event = 0
        self.run_directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        (self.run_directory / "events").mkdir(mode=0o700)
        (self.run_directory / "turns").mkdir(mode=0o700)
        (self.run_directory / "candidates").mkdir(mode=0o700)
        (self.run_directory / "sealed").mkdir(mode=0o700)

    @classmethod
    def create(
        cls,
        root: str | Path,
        study_id: str,
        trajectory_id: str,
        *,
        secrets: tuple[str, ...] = (),
    ) -> TrajectoryStore:
        for label, value in (("study_id", study_id), ("trajectory_id", trajectory_id)):
            if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value) is None:
                raise TrajectoryArtifactError(f"{label} must be one safe path component")
        return cls(Path(root).expanduser() / study_id / trajectory_id, secrets=secrets)

    def _resolve(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise TrajectoryArtifactError(f"unsafe artifact path: {relative}")
        destination = (self.run_directory / relative).resolve()
        if self.run_directory.resolve() not in destination.parents:
            raise TrajectoryArtifactError(f"artifact path escaped trajectory: {relative}")
        return destination

    def _reject_secrets(self, content: bytes) -> None:
        if any(secret in content for secret in self._secret_bytes):
            raise TrajectoryArtifactError("refusing to write credential material")

    def write_text(self, relative_path: str | Path, content: str) -> Path:
        destination = self._resolve(relative_path)
        encoded = content.encode("utf-8")
        self._reject_secrets(encoded)
        if destination.exists():
            raise TrajectoryArtifactError(f"artifact already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise TrajectoryArtifactError("artifact write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)
        destination.chmod(0o600)
        return destination

    def write_json(self, relative_path: str | Path, value: Any) -> Path:
        rendered = json.dumps(
            _payload(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        return self.write_text(relative_path, f"{rendered}\n")

    def append_event(
        self,
        sequence: int,
        kind: str,
        turn_index: int | None,
        payload: Any,
    ) -> Path:
        if sequence != self._next_event:
            raise TrajectoryArtifactError(
                f"event sequence must be {self._next_event}, received {sequence}"
            )
        normalized = _payload(payload)
        event = {
            "schema_version": "canary-trajectory-event.v1",
            "sequence": sequence,
            "kind": kind,
            "turn_index": turn_index,
            "payload": normalized,
            "payload_sha256": sha256_json(normalized),
        }
        path = self.write_json(Path("events") / f"{sequence:04d}.json", event)
        self._next_event += 1
        return path

    def write_turn(
        self,
        turn: int,
        *,
        request: Any,
        response: Any | None = None,
        error: Any | None = None,
        candidate: str | None = None,
        worker_job: Any | None = None,
        worker_result: Any | None = None,
    ) -> Path:
        if (response is None) == (error is None):
            raise TrajectoryArtifactError("turn requires exactly one response or error")
        if (worker_job is None) != (worker_result is None):
            raise TrajectoryArtifactError("worker job and result must be supplied together")
        directory = Path("turns") / f"{turn:04d}"
        self.write_json(directory / "request.json", request)
        terminal_name = "response.json" if response is not None else "error.json"
        terminal_payload = response if response is not None else error
        self.write_json(directory / terminal_name, terminal_payload)
        if candidate is not None:
            self.write_text(directory / "candidate.py", candidate)
        if worker_job is not None:
            self.write_json(directory / "worker-job.json", worker_job)
            self.write_json(directory / "worker-result.json", worker_result)
        return self._resolve(directory)

    def snapshot_candidate(self, label: str, source: str, expected_sha256: str) -> Path:
        if label not in {"first", "final"}:
            raise TrajectoryArtifactError("candidate label must be first or final")
        actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if actual != expected_sha256:
            raise TrajectoryArtifactError("candidate snapshot hash mismatch")
        path = self.write_text(Path("candidates") / f"{label}.py", source)
        self.write_json(
            Path("candidates") / f"{label}.json",
            {"label": label, "sha256": actual},
        )
        return path

    def write_sealed(self, label: str, job: Any, result: Any) -> Path:
        if label not in {"first", "final"}:
            raise TrajectoryArtifactError("sealed label must be first or final")
        directory = Path("sealed") / label
        self.write_json(directory / "worker-job.json", job)
        self.write_json(directory / "worker-result.json", result)
        return self._resolve(directory)

    def seal(self) -> Path:
        checksum_path = self.run_directory / "sha256sums.txt"
        if checksum_path.exists():
            raise TrajectoryArtifactError("trajectory is already sealed")
        lines: list[str] = []
        for path in sorted(self.run_directory.rglob("*")):
            if path.is_file():
                self._reject_secrets(path.read_bytes())
                relative = path.relative_to(self.run_directory).as_posix()
                lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
        self.write_text("sha256sums.txt", "\n".join(lines) + "\n")
        for path in sorted(self.run_directory.rglob("*"), reverse=True):
            path.chmod(0o400 if path.is_file() else 0o500)
        self.run_directory.chmod(0o500)
        return checksum_path


def verify_trajectory(directory: str | Path) -> None:
    root = Path(directory)
    checksum_path = root / "sha256sums.txt"
    if not checksum_path.is_file():
        raise TrajectoryArtifactError("trajectory checksum manifest is missing")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or relative in expected:
            raise TrajectoryArtifactError("invalid trajectory checksum entry")
        expected[relative] = digest
    actual = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if actual != expected:
        raise TrajectoryArtifactError("trajectory checksum mismatch")
