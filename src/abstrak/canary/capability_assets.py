"""Frozen local assets for the TileLang capability-gate matrix study."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import Field, model_validator

from abstrak.canary.baselines import (
    BASELINE_VARIANTS,
    CAPABILITY_GATE_SCOPE,
    get_baseline_source,
    validate_baseline_registry,
)
from abstrak.canary.capabilities import (
    CORE_PACK,
    FULL_PACK,
    MAP_PACK,
    SCHED_PACK,
    PackId,
    validate_tilelang_capability_source,
)
from abstrak.canary.contracts import IDENTIFIER_PATTERN, SHA256_PATTERN, CanaryModel
from abstrak.canary.manifests import PinnedStudySpec
from abstrak.canary.matrix import MatrixSchedule
from abstrak.canary.matrix_preflight import (
    AssetManifest,
    BaselineAssetBinding,
    CanaryAssetBinding,
    TargetAssetBinding,
    TaskAssetBinding,
    build_asset_manifest,
)
from abstrak.canary.targets import (
    get_target_stack,
    validate_target_registry,
)
from abstrak.canary.tasks import (
    CAPABILITY_GATE_ASSET_ROOT,
    CAPABILITY_GATE_TASK_IDS,
    PinnedAsset,
    get_task_assets,
    get_task_pack,
    load_oracle_source,
    load_pinned_asset,
    validate_task_registry,
)
from abstrak.providers.contracts import sha256_json

CanaryId = Literal["schedule", "mapping", "schedule-mapping"]
CAPABILITY_STUDY_ID = "tilelang-capability-gate-a100-v1"
CAPABILITY_STUDY_SHA256 = "876b18e75d86e77c6e2e4cd47038f60719ba6108943ddc754086ea82685ecd00"
CAPABILITY_SCHEDULE_SHA256 = "40c372285875337ebd62529d72b2dd5bc2f6d123cbb2940a93c7482d2537983e"
CAPABILITY_ASSET_MANIFEST_SHA256 = (
    "5309958850f2bc1b6dd9d20079ab88935965fc2b4632e3efacbbc4879ca862bb"
)


class CapabilityAssetError(ValueError):
    """Raised when a capability-gate source or registry binding is inconsistent."""


class CapabilityCanarySpec(CanaryModel):
    """One standalone correctness probe that must exercise a non-core pack."""

    schema_version: Literal["tilelang-capability-canary.v1"] = (
        "tilelang-capability-canary.v1"
    )
    id: CanaryId
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_path: str = Field(pattern=r"^canaries/[a-z0-9_]+\.py$")
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    minimum_pack_id: PackId
    required_target_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def targets_match_minimum_pack(self) -> CapabilityCanarySpec:
        expected = {
            "tileops-sched": (SCHED_PACK.target_id, FULL_PACK.target_id),
            "tileops-map": (MAP_PACK.target_id, FULL_PACK.target_id),
            "tileops-full": (FULL_PACK.target_id,),
        }
        if self.minimum_pack_id == CORE_PACK.id:
            raise ValueError("capability canaries must exercise a non-core pack")
        if self.required_target_ids != expected[self.minimum_pack_id]:
            raise ValueError("canary target acceptance does not match its minimum pack")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


_CANARIES: Mapping[str, CapabilityCanarySpec] = MappingProxyType(
    {
        "schedule": CapabilityCanarySpec(
            id="schedule",
            task_id="gemm-large-k-static",
            source_path="canaries/schedule.py",
            source_sha256="9d06f464541977c84acddebbdba46701451dedd47ef27d76a2fbadf3ae624855",
            minimum_pack_id=SCHED_PACK.id,
            required_target_ids=(SCHED_PACK.target_id, FULL_PACK.target_id),
        ),
        "mapping": CapabilityCanarySpec(
            id="mapping",
            task_id="row-sum-static",
            source_path="canaries/mapping.py",
            source_sha256="391e52d481e914875c8a0c85293f8b3381565ad7e54f835863d7d43693f0e016",
            minimum_pack_id=MAP_PACK.id,
            required_target_ids=(MAP_PACK.target_id, FULL_PACK.target_id),
        ),
        "schedule-mapping": CapabilityCanarySpec(
            id="schedule-mapping",
            task_id="gemm-large-k-static",
            source_path="canaries/schedule_mapping.py",
            source_sha256="e9f6431ba5de641530c848823368987b4180d2e148504e9543c6e684fa3df051",
            minimum_pack_id=FULL_PACK.id,
            required_target_ids=(FULL_PACK.target_id,),
        ),
    }
)


def list_capability_canary_ids() -> tuple[str, ...]:
    return tuple(_CANARIES)


def get_capability_canary(canary_id: str) -> CapabilityCanarySpec:
    try:
        return _CANARIES[canary_id].model_copy(deep=True)
    except KeyError:
        raise CapabilityAssetError(f"unknown capability canary: {canary_id}") from None


def load_capability_canary(
    canary_id: str,
    *,
    asset_root: str | Path = CAPABILITY_GATE_ASSET_ROOT,
) -> str:
    canary = get_capability_canary(canary_id)
    try:
        return load_pinned_asset(
            PinnedAsset(
                relative_path=canary.source_path,
                sha256=canary.source_sha256,
            ),
            asset_root=asset_root,
        )
    except ValueError as error:
        raise CapabilityAssetError(f"invalid capability canary {canary_id}: {error}") from error


def validate_capability_asset_registry(
    *,
    asset_root: str | Path = CAPABILITY_GATE_ASSET_ROOT,
) -> None:
    """Validate every local study asset without importing a GPU runtime."""

    validate_task_registry(scope=CAPABILITY_GATE_SCOPE, asset_root=asset_root)
    validate_target_registry(scope=CAPABILITY_GATE_SCOPE, asset_root=asset_root)
    validate_baseline_registry(scope=CAPABILITY_GATE_SCOPE)
    packs = (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK)
    for task_id in CAPABILITY_GATE_TASK_IDS:
        expert = load_oracle_source(task_id, "tilelang", asset_root=asset_root)
        results = tuple(validate_tilelang_capability_source(expert, pack) for pack in packs)
        if not all(result.valid for result in results) or any(
            result.minimum_pack_id != CORE_PACK.id for result in results
        ):
            raise CapabilityAssetError(f"expert is not B-legal under every pack: {task_id}")

    for canary_id in list_capability_canary_ids():
        canary = get_capability_canary(canary_id)
        if canary.id != canary_id or canary.task_id not in CAPABILITY_GATE_TASK_IDS:
            raise CapabilityAssetError(f"invalid capability canary registry entry: {canary_id}")
        source = load_capability_canary(canary_id, asset_root=asset_root)
        results = tuple(validate_tilelang_capability_source(source, pack) for pack in packs)
        accepted_targets = tuple(
            pack.target_id for pack, result in zip(packs, results, strict=True) if result.valid
        )
        if accepted_targets != canary.required_target_ids or any(
            result.minimum_pack_id != canary.minimum_pack_id for result in results
        ):
            raise CapabilityAssetError(
                f"capability canary acceptance matrix differs from its contract: {canary_id}"
            )


def build_capability_asset_manifest(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    *,
    asset_root: str | Path = CAPABILITY_GATE_ASSET_ROOT,
) -> AssetManifest:
    """Resolve real registries into the generic preflight asset contract."""

    if (
        pinned.spec.study_id != CAPABILITY_STUDY_ID
        or pinned.sha256 != CAPABILITY_STUDY_SHA256
        or schedule.sha256 != CAPABILITY_SCHEDULE_SHA256
    ):
        raise CapabilityAssetError("study identity differs from the frozen capability gate")
    validate_capability_asset_registry(asset_root=asset_root)
    ordered_task_ids = tuple(
        dict.fromkeys(task_id for phase in pinned.spec.phases for task_id in phase.task_ids)
    )
    tasks = []
    for task_id in ordered_task_ids:
        task = get_task_pack(task_id)
        assets = get_task_assets(task_id)
        try:
            expert = assets.oracles["tilelang"]
        except KeyError:
            raise CapabilityAssetError(f"task has no TileLang expert: {task_id}") from None
        tasks.append(
            TaskAssetBinding(
                task_id=task_id,
                task_pack_sha256=sha256_json(task),
                reference_source_sha256=assets.source.sha256,
                expert_source_sha256=expert.sha256,
                baselines=tuple(
                    BaselineAssetBinding(
                        variant=variant,
                        source_sha256=get_baseline_source(task_id, variant).source_sha256,
                    )
                    for variant in BASELINE_VARIANTS
                ),
            )
        )
    targets = tuple(
        TargetAssetBinding(
            target_id=target_id,
            target_stack_sha256=sha256_json(get_target_stack(target_id)),
            card_sha256=get_target_stack(target_id).card_sha256,
        )
        for target_id in pinned.spec.targets
    )
    canaries = tuple(
        CanaryAssetBinding(
            canary_id=canary.id,
            task_id=canary.task_id,
            source_sha256=canary.source_sha256,
            required_target_ids=canary.required_target_ids,
        )
        for canary in (get_capability_canary(canary_id) for canary_id in _CANARIES)
    )
    manifest = build_asset_manifest(
        pinned,
        schedule,
        tasks=tuple(tasks),
        targets=targets,
        canaries=canaries,
    )
    if manifest.sha256 != CAPABILITY_ASSET_MANIFEST_SHA256:
        raise CapabilityAssetError("resolved assets differ from the frozen capability manifest")
    return manifest
