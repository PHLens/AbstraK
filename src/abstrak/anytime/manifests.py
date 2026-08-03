"""Pinned JSON loader for version-one anytime DSL study specifications."""

from __future__ import annotations

import hashlib
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from abstrak.anytime.contracts import SHA256_PATTERN, AnytimeStudySpec

_SHA256 = re.compile(SHA256_PATTERN)


class AnytimeManifestError(ValueError):
    """Raised when an anytime study manifest cannot be loaded or verified safely."""


@dataclass(frozen=True)
class PinnedAnytimeStudySpec:
    """A validated anytime study bound to the exact bytes loaded from disk."""

    path: Path
    sha256: str
    spec: AnytimeStudySpec


def load_anytime_study_spec(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> PinnedAnytimeStudySpec:
    """Load one regular UTF-8 JSON file after optional raw-byte SHA verification."""

    manifest_path = Path(path).expanduser()
    if expected_sha256 is not None and _SHA256.fullmatch(expected_sha256) is None:
        raise AnytimeManifestError("expected anytime study SHA-256 is invalid")
    try:
        metadata = manifest_path.stat()
    except FileNotFoundError:
        raise AnytimeManifestError(
            f"cannot read anytime study manifest {manifest_path}: file does not exist"
        ) from None
    except OSError as error:
        raise AnytimeManifestError(
            f"cannot inspect anytime study manifest {manifest_path}: {error}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise AnytimeManifestError(
            f"anytime study manifest is not a regular file: {manifest_path}"
        )
    try:
        payload = manifest_path.read_bytes()
    except OSError as error:
        raise AnytimeManifestError(
            f"cannot read anytime study manifest {manifest_path}: {error}"
        ) from error
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise AnytimeManifestError(
            "anytime study manifest SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AnytimeManifestError(
            f"anytime study manifest is not UTF-8: {manifest_path}"
        ) from error
    try:
        spec = AnytimeStudySpec.model_validate_json(text)
    except ValidationError as error:
        raise AnytimeManifestError(
            f"invalid anytime study manifest {manifest_path}: {error}"
        ) from error
    return PinnedAnytimeStudySpec(
        path=manifest_path.resolve(),
        sha256=actual_sha256,
        spec=spec,
    )
