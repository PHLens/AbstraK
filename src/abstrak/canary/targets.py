"""Pinned target-stack registry for the A100 R1 canary study.

Target cards are inert, hash-verified text. Importing this module does not load
PyTorch, Triton, or any GPU runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from abstrak.canary.capabilities import CORE_PACK, FULL_PACK, MAP_PACK, SCHED_PACK
from abstrak.canary.contracts import TargetStackSpec
from abstrak.canary.tasks import (
    CAPABILITY_GATE_ASSET_ROOT,
    DEFAULT_ASSET_ROOT,
    PinnedAsset,
    TaskRegistryError,
    load_pinned_asset,
)


class TargetRegistryError(ValueError):
    """Raised when a target ID or pinned target card is invalid."""


R1_SCOPE = "r1"
CAPABILITY_GATE_SCOPE = "capability-gate"


@dataclass(frozen=True)
class TargetCardAsset:
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class _TargetRegistryScope:
    stacks: Mapping[str, TargetStackSpec]
    cards: Mapping[str, TargetCardAsset]
    asset_root: Path = DEFAULT_ASSET_ROOT


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

_R1_TARGET_STACKS: Mapping[str, TargetStackSpec] = MappingProxyType(
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

_R1_TARGET_CARDS: Mapping[str, TargetCardAsset] = MappingProxyType(
    {
        "triton-a100": _TRITON_CARD,
        "tilelang-a100": _TILELANG_CARD,
        "cute-a100": _CUTE_CARD,
    }
)


_CAPABILITY_CARDS_BY_PACK_ID: Mapping[str, TargetCardAsset] = MappingProxyType(
    {
        CORE_PACK.id: TargetCardAsset(
            relative_path="targets/core.md",
            sha256="57d79d14cf585232508dcf72aee280b1151c5913159a40ba094a33fb6c509154",
        ),
        SCHED_PACK.id: TargetCardAsset(
            relative_path="targets/sched.md",
            sha256="58f2e39c69fc1cae3a1bb0dc1224111e6268be01366358716df2da8083416ba6",
        ),
        MAP_PACK.id: TargetCardAsset(
            relative_path="targets/map.md",
            sha256="07d7fbe1235c416e256b1a19819b23dd535d6bdb61aa678dd081a58c9b2bc386",
        ),
        FULL_PACK.id: TargetCardAsset(
            relative_path="targets/full.md",
            sha256="a9ed048414cd3062ee95e58dc75a6ba7273d2e82eda60d047664c4d5e09b123a",
        ),
    }
)
_CAPABILITY_GATE_TARGET_STACKS: Mapping[str, TargetStackSpec] = MappingProxyType(
    {
        pack.target_id: TargetStackSpec(
            id=pack.target_id,
            backend="tilelang",
            version="0.1.12",
            card_path=_CAPABILITY_CARDS_BY_PACK_ID[pack.id].relative_path,
            card_sha256=_CAPABILITY_CARDS_BY_PACK_ID[pack.id].sha256,
            adapter=pack.adapter_id,
            allowed_assets=(),
        )
        for pack in (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK)
    }
)
_CAPABILITY_GATE_TARGET_CARDS: Mapping[str, TargetCardAsset] = MappingProxyType(
    {
        pack.target_id: _CAPABILITY_CARDS_BY_PACK_ID[pack.id]
        for pack in (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK)
    }
)

_TARGET_REGISTRIES: Mapping[str, _TargetRegistryScope] = MappingProxyType(
    {
        R1_SCOPE: _TargetRegistryScope(
            stacks=_R1_TARGET_STACKS,
            cards=_R1_TARGET_CARDS,
        ),
        CAPABILITY_GATE_SCOPE: _TargetRegistryScope(
            stacks=_CAPABILITY_GATE_TARGET_STACKS,
            cards=_CAPABILITY_GATE_TARGET_CARDS,
            asset_root=CAPABILITY_GATE_ASSET_ROOT,
        ),
    }
)


def _build_global_indexes(
    registries: Mapping[str, _TargetRegistryScope],
) -> tuple[Mapping[str, TargetStackSpec], Mapping[str, TargetCardAsset]]:
    stacks: dict[str, TargetStackSpec] = {}
    cards: dict[str, TargetCardAsset] = {}
    owners: dict[str, str] = {}
    for scope, registry in registries.items():
        target_ids = set(registry.stacks) | set(registry.cards)
        for target_id in sorted(target_ids):
            previous_scope = owners.get(target_id)
            if previous_scope is not None:
                raise TargetRegistryError(
                    f"target ID is registered in multiple scopes: {target_id} "
                    f"({previous_scope}, {scope})"
                )
            owners[target_id] = scope
        stacks.update(registry.stacks)
        cards.update(registry.cards)
    return MappingProxyType(stacks), MappingProxyType(cards)


_TARGET_STACKS, _TARGET_CARDS = _build_global_indexes(_TARGET_REGISTRIES)


def _registry_for_scope(scope: str) -> _TargetRegistryScope:
    try:
        return _TARGET_REGISTRIES[scope]
    except KeyError:
        raise TargetRegistryError(f"unknown target registry scope: {scope}") from None


def _registry_for_target(target_id: str) -> _TargetRegistryScope:
    for registry in _TARGET_REGISTRIES.values():
        if target_id in registry.stacks:
            return registry
    raise TargetRegistryError(f"unknown target stack: {target_id}")


def list_target_ids(scope: str = R1_SCOPE) -> tuple[str, ...]:
    """Return registered target-stack IDs in stable order."""

    return tuple(sorted(_registry_for_scope(scope).stacks))


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

    resolved_root = _registry_for_target(target_id).asset_root if asset_root is None else asset_root
    return load_pinned_card(get_target_card_asset(target_id), asset_root=resolved_root)


def validate_target_registry(
    *,
    scope: str = R1_SCOPE,
    asset_root: str | Path | None = None,
) -> None:
    """Validate target contracts, card cross-references, paths, and hashes."""

    registry = _registry_for_scope(scope)
    resolved_root = registry.asset_root if asset_root is None else asset_root
    if set(registry.stacks) != set(registry.cards):
        raise TargetRegistryError("target contracts and card registry have different target IDs")
    for target_id in sorted(registry.stacks):
        target = registry.stacks[target_id]
        card = registry.cards[target_id]
        if target.id != target_id:
            message = f"target registry key does not match contract ID: {target_id}"
            raise TargetRegistryError(message)
        if target.card_path != card.relative_path or target.card_sha256 != card.sha256:
            raise TargetRegistryError(f"target card reference mismatch: {target_id}")
        load_pinned_card(card, asset_root=resolved_root)
