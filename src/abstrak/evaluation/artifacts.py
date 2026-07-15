"""Atomic, private artifacts for naive KernelBench studies."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel


class EvaluationArtifactError(RuntimeError):
    pass


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _payload(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


class StudyRunStore:
    def __init__(self, run_directory: Path, secrets: tuple[str, ...] = ()) -> None:
        self.run_directory = run_directory
        self.cells_directory = run_directory / "cells"
        self.evaluations_directory = run_directory / "evaluations"
        self._secret_bytes = tuple(value.encode() for value in secrets if value)
        self.run_directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.cells_directory.mkdir(mode=0o700)
        self.evaluations_directory.mkdir(mode=0o700)

    @classmethod
    def create(
        cls,
        root: str | Path,
        study_id: str,
        run_id: str,
        *,
        secrets: tuple[str, ...] = (),
    ) -> StudyRunStore:
        if not run_id or Path(run_id).name != run_id:
            raise EvaluationArtifactError("run_id must be one safe path component")
        return cls(Path(root).expanduser() / study_id / run_id, secrets)

    def create_cell(self, cell_id: str) -> Path:
        if not cell_id or Path(cell_id).name != cell_id:
            raise EvaluationArtifactError("cell_id must be one safe path component")
        path = self.cells_directory / cell_id
        path.mkdir(exist_ok=False, mode=0o700)
        return path

    def write_json(self, relative_path: str | Path, value: Any) -> Path:
        content = json.dumps(
            _payload(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        return self.write_text(relative_path, f"{content}\n")

    def write_text(self, relative_path: str | Path, content: str) -> Path:
        destination = self._resolve(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        encoded = content.encode()
        self._reject_secrets(encoded)
        if destination.exists():
            raise EvaluationArtifactError(f"artifact already exists: {destination}")
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise EvaluationArtifactError("artifact write made no progress")
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

    def seal_generation_cell(self, cell_directory: Path) -> Path:
        if cell_directory.parent != self.cells_directory:
            raise EvaluationArtifactError("cell directory does not belong to this run")
        lines: list[str] = []
        for path in sorted(cell_directory.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.name != "generation.sha256sums":
                self._reject_secrets(path.read_bytes())
                lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        checksum = self.write_text(
            cell_directory.relative_to(self.run_directory) / "generation.sha256sums",
            "\n".join(lines) + "\n",
        )
        for path in cell_directory.iterdir():
            if path.is_file():
                path.chmod(0o400)
        cell_directory.chmod(0o500)
        return checksum

    def verify_no_secrets(self) -> None:
        for path in self.run_directory.rglob("*"):
            if path.is_file():
                self._reject_secrets(path.read_bytes())

    def _resolve(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise EvaluationArtifactError(f"unsafe artifact path: {relative}")
        destination = (self.run_directory / relative).resolve()
        if self.run_directory.resolve() not in destination.parents:
            raise EvaluationArtifactError(f"artifact path escaped run directory: {relative}")
        return destination

    def _reject_secrets(self, content: bytes) -> None:
        if any(secret in content for secret in self._secret_bytes):
            raise EvaluationArtifactError("refusing to write credential material")


def write_derived_json(path: str | Path, value: Any) -> Path:
    """Atomically replace a reproducible derived result such as a summary."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(
        _payload(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.write_text(f"{payload}\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, destination)
    return destination


def write_derived_text(path: str | Path, content: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, destination)
    return destination


def seal_directory(directory: str | Path, checksum_name: str) -> Path:
    path = Path(directory)
    if not path.is_dir() or Path(checksum_name).name != checksum_name:
        raise EvaluationArtifactError("invalid directory sealing request")
    lines = [
        f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}"
        for item in sorted(path.iterdir(), key=lambda value: value.name)
        if item.is_file() and item.name != checksum_name
    ]
    checksum = write_derived_text(path / checksum_name, "\n".join(lines) + "\n")
    for item in path.iterdir():
        if item.is_file():
            item.chmod(0o400)
    path.chmod(0o500)
    return checksum


def verify_directory_checksums(directory: str | Path, checksum_name: str) -> str:
    """Verify a sealed flat bundle and return the checksum file's SHA-256."""

    path = Path(directory)
    checksum_path = path / checksum_name
    if (
        path.is_symlink()
        or not path.is_dir()
        or checksum_path.is_symlink()
        or not checksum_path.is_file()
    ):
        raise EvaluationArtifactError(f"missing checksum file: {checksum_path}")

    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or not SHA256_PATTERN.fullmatch(digest)
            or not name
            or Path(name).name != name
            or name == checksum_name
            or name in expected
        ):
            raise EvaluationArtifactError(f"invalid checksum entry in {checksum_path}: {line!r}")
        expected[name] = digest

    actual: dict[str, Path] = {}
    for item in path.iterdir():
        if item.name == checksum_name:
            continue
        if item.is_symlink() or not item.is_file():
            raise EvaluationArtifactError(f"unexpected bundle entry: {item}")
        actual[item.name] = item
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise EvaluationArtifactError(
            f"bundle contents do not match {checksum_path}: missing={missing}, extra={extra}"
        )
    for name, item in actual.items():
        observed = hashlib.sha256(item.read_bytes()).hexdigest()
        if not hmac.compare_digest(observed, expected[name]):
            raise EvaluationArtifactError(f"checksum mismatch: {item}")
    return hashlib.sha256(checksum_path.read_bytes()).hexdigest()
