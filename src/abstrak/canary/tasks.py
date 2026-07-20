"""Pinned task-pack registry for the A100 R1 canary study.

This module deliberately does not import PyTorch. Task sources are loaded as
hash-verified text and are imported only inside an evaluator process.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from abstrak.canary.contracts import InputCaseSpec, TaskPackSpec

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_ASSET_ROOT = Path(__file__).resolve().parents[3] / "benchmarks" / "r1-a100"


class TaskRegistryError(ValueError):
    """Raised when a task ID or pinned task asset is invalid."""


@dataclass(frozen=True)
class PinnedAsset:
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class TaskAssets:
    source: PinnedAsset
    oracles: Mapping[str, PinnedAsset]


_ROW_REDUCTION_SOURCE = PinnedAsset(
    relative_path="tasks/row_reduction_scale.py",
    sha256="00057e63fd2b5be59044173fad8a0cc3b4d573021b8486880f089bf534c2f2cf",
)
_ROW_REDUCTION_TRITON_ORACLE = PinnedAsset(
    relative_path="oracles/triton/row_reduction_scale.py",
    sha256="037ec6d894daa9c59f01a43739fb18296a3c3aaf10d0946c5d07c04116f48fb8",
)
_ROW_REDUCTION_TILELANG_ORACLE = PinnedAsset(
    relative_path="oracles/tilelang/row_reduction_scale.py",
    sha256="f725e8dce6ccb192a0cc0f8c7b85a988807c9a7445ab2dcbfc92f92cd4978d2a",
)
_ROW_REDUCTION_CUTE_ORACLE = PinnedAsset(
    relative_path="oracles/cute/row_reduction_scale.py",
    sha256="b123d2ddee776b4a7dfd2cb4c375fb7acd29f4b9e79ea726bc6d51eb8e4fb390",
)
_MATMUL_BIAS_SOURCE = PinnedAsset(
    relative_path="tasks/matmul_bias.py",
    sha256="56c37bc614bbaca3c6e09d278960283c5c22726e8876026e7bedb6f063d561da",
)
_MATMUL_BIAS_TRITON_ORACLE = PinnedAsset(
    relative_path="oracles/triton/matmul_bias.py",
    sha256="ee2f018ab72ca51425d01762ba1a93008465225b6e4f81ea853cdc3281cf3aac",
)
_MATMUL_BIAS_TILELANG_ORACLE = PinnedAsset(
    relative_path="oracles/tilelang/matmul_bias.py",
    sha256="1f19050b78a01ea010b8dd488001b8d9924701d7f00d954a5b8d2352b620a16d",
)
_MATMUL_BIAS_CUTE_ORACLE = PinnedAsset(
    relative_path="oracles/cute/matmul_bias.py",
    sha256="948d6083215d7c472a77fbf42354b810f5b221a03cb24a9e8e8872c2baef685f",
)

_TASK_PACKS: Mapping[str, TaskPackSpec] = MappingProxyType(
    {
        "row-reduction-scale": TaskPackSpec(
            id="row-reduction-scale",
            specification=(
                "Given a contiguous FP16 tensor x with shape (1024, 1024), sum each row "
                "using FP32 accumulation, multiply every row sum by 0.5, and return a "
                "contiguous FP16 tensor with shape (1024,)."
            ),
            source_path=_ROW_REDUCTION_SOURCE.relative_path,
            source_sha256=_ROW_REDUCTION_SOURCE.sha256,
            dtype="fp16",
            reference_precision="fp32",
            input_shapes=((1024, 1024),),
            parameters=(
                ("rows", 1024),
                ("columns", 1024),
                ("scale", 0.5),
                ("output_dtype", "fp16"),
            ),
            atol=1e-2,
            rtol=1e-2,
            fallback_policy="forbid_framework_ops",
            dev_cases=(
                InputCaseSpec(id="dev-random-1", kind="random", seed=2026071701),
                InputCaseSpec(id="dev-random-2", kind="random", seed=2026071702),
            ),
            sealed_cases=(
                InputCaseSpec(id="sealed-random-1", kind="random", seed=2026071801),
                InputCaseSpec(id="sealed-random-2", kind="random", seed=2026071802),
                InputCaseSpec(id="sealed-random-3", kind="random", seed=2026071803),
                InputCaseSpec(id="sealed-random-4", kind="random", seed=2026071804),
                InputCaseSpec(
                    id="sealed-constant",
                    kind="constant",
                    seed=2026071805,
                    value=0.25,
                ),
            ),
        ),
        "matmul-bias": TaskPackSpec(
            id="matmul-bias",
            specification=(
                "Given contiguous FP16 tensors a, b, and bias with shapes (256, 256), "
                "(256, 256), and (256,), compute a @ b using FP32 accumulation, add "
                "the bias in FP32, and return a contiguous FP16 tensor with shape "
                "(256, 256)."
            ),
            source_path=_MATMUL_BIAS_SOURCE.relative_path,
            source_sha256=_MATMUL_BIAS_SOURCE.sha256,
            dtype="fp16",
            reference_precision="fp32",
            input_shapes=((256, 256), (256, 256), (256,)),
            parameters=(
                ("m", 256),
                ("n", 256),
                ("k", 256),
                ("epilogue", "bias"),
                ("output_dtype", "fp16"),
            ),
            atol=1e-2,
            rtol=1e-2,
            fallback_policy="forbid_framework_ops",
            dev_cases=(
                InputCaseSpec(id="dev-random-1", kind="random", seed=2026071901),
                InputCaseSpec(id="dev-random-2", kind="random", seed=2026071902),
            ),
            sealed_cases=(
                InputCaseSpec(id="sealed-random-1", kind="random", seed=2026072001),
                InputCaseSpec(id="sealed-random-2", kind="random", seed=2026072002),
                InputCaseSpec(id="sealed-random-3", kind="random", seed=2026072003),
                InputCaseSpec(id="sealed-random-4", kind="random", seed=2026072004),
                InputCaseSpec(
                    id="sealed-constant",
                    kind="constant",
                    seed=2026072005,
                    value=0.125,
                ),
            ),
        ),
    }
)

_TASK_ASSETS: Mapping[str, TaskAssets] = MappingProxyType(
    {
        "row-reduction-scale": TaskAssets(
            source=_ROW_REDUCTION_SOURCE,
            oracles=MappingProxyType(
                {
                    "triton": _ROW_REDUCTION_TRITON_ORACLE,
                    "tilelang": _ROW_REDUCTION_TILELANG_ORACLE,
                    "cute": _ROW_REDUCTION_CUTE_ORACLE,
                }
            ),
        ),
        "matmul-bias": TaskAssets(
            source=_MATMUL_BIAS_SOURCE,
            oracles=MappingProxyType(
                {
                    "triton": _MATMUL_BIAS_TRITON_ORACLE,
                    "tilelang": _MATMUL_BIAS_TILELANG_ORACLE,
                    "cute": _MATMUL_BIAS_CUTE_ORACLE,
                }
            ),
        ),
    }
)


def list_task_ids() -> tuple[str, ...]:
    """Return registered task IDs in stable order."""

    return tuple(sorted(_TASK_PACKS))


def get_task_pack(task_id: str) -> TaskPackSpec:
    """Return an isolated copy of a registered task-pack contract."""

    try:
        return _TASK_PACKS[task_id].model_copy(deep=True)
    except KeyError:
        raise TaskRegistryError(f"unknown task pack: {task_id}") from None


def get_task_assets(task_id: str) -> TaskAssets:
    """Return pinned source and oracle references for a task pack."""

    try:
        return _TASK_ASSETS[task_id]
    except KeyError:
        raise TaskRegistryError(f"unknown task pack: {task_id}") from None


def _resolve_asset_root(asset_root: str | Path | None) -> Path:
    configured_root = DEFAULT_ASSET_ROOT if asset_root is None else Path(asset_root).expanduser()
    try:
        resolved_root = configured_root.resolve(strict=True)
    except OSError as error:
        message = f"cannot resolve task asset root {configured_root}: {error}"
        raise TaskRegistryError(message) from error
    if not resolved_root.is_dir():
        raise TaskRegistryError(f"task asset root is not a directory: {resolved_root}")
    return resolved_root


def _resolve_asset_path(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        not relative.parts
        or relative.is_absolute()
        or "\\" in relative_path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise TaskRegistryError(f"unsafe task asset path: {relative_path!r}")
    try:
        resolved_path = root.joinpath(*relative.parts).resolve(strict=True)
        resolved_path.relative_to(root)
    except (OSError, ValueError) as error:
        raise TaskRegistryError(f"task asset escaped or is missing: {relative_path}") from error
    if not resolved_path.is_file():
        raise TaskRegistryError(f"task asset is not a regular file: {relative_path}")
    return resolved_path


def load_pinned_asset(
    asset: PinnedAsset,
    *,
    asset_root: str | Path | None = None,
) -> str:
    """Load UTF-8 source only after path containment and SHA-256 checks."""

    if _SHA256_PATTERN.fullmatch(asset.sha256) is None:
        raise TaskRegistryError(f"invalid SHA-256 for task asset: {asset.relative_path}")
    root = _resolve_asset_root(asset_root)
    path = _resolve_asset_path(root, asset.relative_path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise TaskRegistryError(f"cannot read task asset {asset.relative_path}: {error}") from error
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != asset.sha256:
        raise TaskRegistryError(
            f"task asset SHA-256 mismatch for {asset.relative_path}: "
            f"expected {asset.sha256}, found {actual_sha256}"
        )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TaskRegistryError(f"task asset is not UTF-8: {asset.relative_path}") from error


def load_task_source(task_id: str, *, asset_root: str | Path | None = None) -> str:
    """Load one registered public task fixture."""

    return load_pinned_asset(get_task_assets(task_id).source, asset_root=asset_root)


def load_oracle_source(
    task_id: str,
    target_id: str,
    *,
    asset_root: str | Path | None = None,
) -> str:
    """Load one registered trusted oracle fixture."""

    assets = get_task_assets(task_id)
    try:
        oracle = assets.oracles[target_id]
    except KeyError:
        raise TaskRegistryError(f"no {target_id} oracle registered for task {task_id}") from None
    return load_pinned_asset(oracle, asset_root=asset_root)


def validate_task_registry(*, asset_root: str | Path | None = None) -> None:
    """Validate task contracts, cross-references, paths, and content hashes."""

    if set(_TASK_PACKS) != set(_TASK_ASSETS):
        raise TaskRegistryError("task contracts and asset registry have different task IDs")
    for task_id in sorted(_TASK_PACKS):
        task_pack = _TASK_PACKS[task_id]
        assets = _TASK_ASSETS[task_id]
        if task_pack.id != task_id:
            raise TaskRegistryError(f"task registry key does not match contract ID: {task_id}")
        if (
            task_pack.source_path != assets.source.relative_path
            or task_pack.source_sha256 != assets.source.sha256
        ):
            raise TaskRegistryError(f"task source reference mismatch: {task_id}")
        load_pinned_asset(assets.source, asset_root=asset_root)
        if not assets.oracles:
            raise TaskRegistryError(f"task has no registered oracle: {task_id}")
        for oracle in assets.oracles.values():
            load_pinned_asset(oracle, asset_root=asset_root)
