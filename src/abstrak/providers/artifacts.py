"""Private, atomic artifact storage for provider conformance runs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel


class ArtifactError(RuntimeError):
    pass


def _json_payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


class ProviderArtifactStore:
    def __init__(self, run_directory: Path, secrets: tuple[str, ...] = ()) -> None:
        self.run_directory = run_directory
        self._secret_bytes = tuple(secret.encode() for secret in secrets if secret)
        self._finalized = False
        self.run_directory.mkdir(exist_ok=False, mode=0o700)
        self.run_directory.chmod(0o700)

    @classmethod
    def create(
        cls,
        root: str | Path,
        provider_id: str,
        model_id: str,
        *,
        secrets: tuple[str, ...] = (),
    ) -> ProviderArtifactStore:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_id = uuid4().hex[:12]
        name = f"{timestamp}-{provider_id}-{model_id}-{run_id}"
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        root_path.chmod(0o700)
        return cls(root_path / name, secrets)

    def write_json(self, name: str, value: Any) -> Path:
        self._ensure_open()
        if Path(name).name != name or not name.endswith(".json"):
            raise ArtifactError(f"invalid artifact filename: {name!r}")
        payload = json.dumps(
            _json_payload(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        return self._atomic_write(name, f"{payload}\n")

    def append_event(self, event: dict[str, Any]) -> None:
        self._ensure_open()
        path = self.run_directory / "events.jsonl"
        line = json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False)
        self._reject_secrets(line.encode())
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            self._write_all(descriptor, f"{line}\n".encode())
        finally:
            os.close(descriptor)

    def verify_no_secrets(self, secrets: tuple[str, ...]) -> None:
        secret_bytes = self._secret_bytes + tuple(secret.encode() for secret in secrets if secret)
        for path in self.run_directory.iterdir():
            if not path.is_file():
                continue
            content = path.read_bytes()
            if any(secret in content for secret in secret_bytes):
                raise ArtifactError(f"secret material detected in artifact {path.name}")

    def finalize(self) -> None:
        self._ensure_open()
        self.verify_no_secrets(())
        self._write_checksums()
        for path in self.run_directory.iterdir():
            if path.is_file():
                path.chmod(0o400)
        self.run_directory.chmod(0o500)
        self._finalized = True

    def verify_checksums(self) -> None:
        checksum_path = self.run_directory / "sha256sums.txt"
        if not checksum_path.is_file():
            raise ArtifactError("artifact has no checksum manifest")
        expected_names: set[str] = set()
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, separator, name = line.partition("  ")
            if not separator or Path(name).name != name:
                raise ArtifactError("malformed checksum manifest")
            expected_names.add(name)
            path = self.run_directory / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ArtifactError(f"checksum mismatch for {name}")
        actual_names = {
            path.name
            for path in self.run_directory.iterdir()
            if path.is_file() and path.name != "sha256sums.txt"
        }
        if actual_names != expected_names:
            raise ArtifactError("artifact file set does not match checksum manifest")

    def _write_checksums(self) -> Path:
        lines: list[str] = []
        for path in sorted(self.run_directory.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.name == "sha256sums.txt":
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.name}")
        return self._atomic_write("sha256sums.txt", "\n".join(lines) + "\n")

    def _atomic_write(self, name: str, content: str) -> Path:
        destination = self.run_directory / name
        if destination.exists():
            raise ArtifactError(f"artifact already exists: {name}")
        temporary = self.run_directory / f".{name}.{uuid4().hex}.tmp"
        encoded = content.encode()
        self._reject_secrets(encoded)
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            self._write_all(descriptor, encoded)
            os.fsync(descriptor)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)
        destination.chmod(0o600)
        return destination

    def _reject_secrets(self, content: bytes) -> None:
        if any(secret in content for secret in self._secret_bytes):
            raise ArtifactError("refusing to write credential material to artifact storage")

    def _ensure_open(self) -> None:
        if self._finalized:
            raise ArtifactError("artifact bundle is finalized")

    @staticmethod
    def _write_all(descriptor: int, content: bytes) -> None:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise ArtifactError("artifact write made no progress")
            remaining = remaining[written:]
