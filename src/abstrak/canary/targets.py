"""Pinned target-stack registry for the A100 R1 canary study.

Target cards are inert, hash-verified text. Importing this module does not load
PyTorch, Triton, or any GPU runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from abstrak.canary.contracts import TargetStackSpec
from abstrak.canary.tasks import (
    PinnedAsset,
    TaskRegistryError,
    load_pinned_asset,
)


class TargetRegistryError(ValueError):
    """Raised when a target ID or pinned target card is invalid."""


@dataclass(frozen=True)
class TargetCardAsset:
    relative_path: str
    sha256: str


_TRITON_CARD = TargetCardAsset(
    relative_path="targets/triton.md",
    sha256="7cf01f1aeb862d6eb648a56eda6c47875596c325ca940a20cba2703533b45631",
)
_TILELANG_CARD = TargetCardAsset(
    relative_path="targets/tilelang.md",
    sha256="c4788211172520d35c3fa4695ddfc532f66221642006f4d9325bab4543f33586",
)
_CUTE_CARD = TargetCardAsset(
    relative_path="targets/cute.md",
    sha256="2419ee0373e8c86ec64581130b3c4cdcd1ba67f5468d734ea0c2eb160d9a8fed",
)

_TARGET_STACKS: Mapping[str, TargetStackSpec] = MappingProxyType(
    {
        "triton-a100": TargetStackSpec(
            id="triton-a100",
            backend="triton",
            version="3.7.1",
            card_path=_TRITON_CARD.relative_path,
            card_sha256=_TRITON_CARD.sha256,
            adapter="kernelbench",
            allowed_assets=(),
        ),
        "tilelang-a100": TargetStackSpec(
            id="tilelang-a100",
            backend="tilelang",
            version="0.1.12",
            card_path=_TILELANG_CARD.relative_path,
            card_sha256=_TILELANG_CARD.sha256,
            adapter="kernelbench",
            allowed_assets=(),
        ),
        "cute-a100": TargetStackSpec(
            id="cute-a100",
            backend="cute",
            version="4.6.1",
            card_path=_CUTE_CARD.relative_path,
            card_sha256=_CUTE_CARD.sha256,
            adapter="kernelbench",
            allowed_assets=(),
        ),
    }
)

_TARGET_CARDS: Mapping[str, TargetCardAsset] = MappingProxyType(
    {
        "triton-a100": _TRITON_CARD,
        "tilelang-a100": _TILELANG_CARD,
        "cute-a100": _CUTE_CARD,
    }
)


def list_target_ids() -> tuple[str, ...]:
    """Return registered target-stack IDs in stable order."""

    return tuple(sorted(_TARGET_STACKS))


def get_target_stack(target_id: str) -> TargetStackSpec:
    """Return an isolated copy of a registered target-stack contract."""

    try:
        return _TARGET_STACKS[target_id].model_copy(deep=True)
    except KeyError:
        raise TargetRegistryError(f"unknown target stack: {target_id}") from None


def get_target_card_asset(target_id: str) -> TargetCardAsset:
    """Return the pinned card reference for a target stack."""

    try:
        return _TARGET_CARDS[target_id]
    except KeyError:
        raise TargetRegistryError(f"unknown target stack: {target_id}") from None


def load_pinned_card(
    card: TargetCardAsset,
    *,
    asset_root: str | Path | None = None,
) -> str:
    """Load a UTF-8 target card after containment and SHA-256 checks."""

    try:
        return load_pinned_asset(
            PinnedAsset(relative_path=card.relative_path, sha256=card.sha256),
            asset_root=asset_root,
        )
    except TaskRegistryError as error:
        message = str(error).replace("task asset", "target card")
        raise TargetRegistryError(message) from error


def load_target_card(target_id: str, *, asset_root: str | Path | None = None) -> str:
    """Load one registered Agent-visible target card."""

    return load_pinned_card(get_target_card_asset(target_id), asset_root=asset_root)


def validate_target_registry(*, asset_root: str | Path | None = None) -> None:
    """Validate target contracts, card cross-references, paths, and hashes."""

    if set(_TARGET_STACKS) != set(_TARGET_CARDS):
        raise TargetRegistryError("target contracts and card registry have different target IDs")
    for target_id in sorted(_TARGET_STACKS):
        target = _TARGET_STACKS[target_id]
        card = _TARGET_CARDS[target_id]
        if target.id != target_id:
            message = f"target registry key does not match contract ID: {target_id}"
            raise TargetRegistryError(message)
        if target.card_path != card.relative_path or target.card_sha256 != card.sha256:
            raise TargetRegistryError(f"target card reference mismatch: {target_id}")
        load_pinned_card(card, asset_root=asset_root)
