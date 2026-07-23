"""Pinned JSON loaders for generic canary study specifications."""

from __future__ import annotations

import hashlib
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from abstrak.canary.contracts import SHA256_PATTERN
from abstrak.canary.matrix import MatrixStudySpec

_SHA256 = re.compile(SHA256_PATTERN)


class StudyManifestError(ValueError):
    """Raised when a study manifest cannot be loaded and verified safely."""


@dataclass(frozen=True)
class PinnedStudySpec:
    """A validated study spec bound to the exact bytes loaded from disk."""

    path: Path
    sha256: str
    spec: MatrixStudySpec


def load_study_spec(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> PinnedStudySpec:
    """Load one regular UTF-8 JSON file after optional raw-byte SHA verification."""

    manifest_path = Path(path).expanduser()
    if expected_sha256 is not None and _SHA256.fullmatch(expected_sha256) is None:
        raise StudyManifestError("expected study manifest SHA-256 is invalid")

    try:
        metadata = manifest_path.stat()
    except FileNotFoundError:
        raise StudyManifestError(
            f"cannot read study manifest {manifest_path}: file does not exist"
        ) from None
    except OSError as error:
        raise StudyManifestError(
            f"cannot inspect study manifest {manifest_path}: {error}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise StudyManifestError(f"study manifest is not a regular file: {manifest_path}")

    try:
        payload = manifest_path.read_bytes()
    except OSError as error:
        raise StudyManifestError(f"cannot read study manifest {manifest_path}: {error}") from error
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise StudyManifestError(
            f"study manifest SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StudyManifestError(f"study manifest is not UTF-8: {manifest_path}") from error
    try:
        spec = MatrixStudySpec.model_validate_json(text)
    except ValidationError as error:
        raise StudyManifestError(f"invalid study manifest {manifest_path}: {error}") from error
    return PinnedStudySpec(
        path=manifest_path.resolve(),
        sha256=actual_sha256,
        spec=spec,
    )


def load_study_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> PinnedStudySpec:
    """Compatibility spelling for callers that refer to the JSON manifest."""

    return load_study_spec(path, expected_sha256=expected_sha256)
