from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

import abstrak.canary.targets as target_registry
from abstrak.canary.capabilities import (
    CORE_PACK,
    FULL_PACK,
    MAP_PACK,
    SCHED_PACK,
    render_tilelang_target_card,
)
from abstrak.canary.targets import (
    CAPABILITY_GATE_SCOPE,
    R1_SCOPE,
    TargetCardAsset,
    TargetRegistryError,
    get_target_card_asset,
    get_target_stack,
    list_target_ids,
    load_pinned_card,
    load_target_card,
    validate_target_registry,
)


def test_triton_target_stack_is_frozen_for_a100() -> None:
    target = get_target_stack("triton-a100")

    assert list_target_ids() == ("cute-a100", "tilelang-a100", "triton-a100")
    assert list_target_ids(scope=R1_SCOPE) == list_target_ids()
    assert list_target_ids(scope=CAPABILITY_GATE_SCOPE) == (
        "tilelang-a100-core",
        "tilelang-a100-full",
        "tilelang-a100-map",
        "tilelang-a100-sched",
    )
    assert target.backend == "triton"
    assert target.version == "3.7.1"
    assert target.adapter == "kernelbench"
    assert target.card_path == "targets/triton.md"
    assert target.allowed_assets == ()
    assert target.oracle_path is None
    assert target.oracle_sha256 is None


def test_triton_card_is_hash_verified_and_contains_one_unrelated_example() -> None:
    validate_target_registry()

    card = load_target_card("triton-a100")

    assert "NVIDIA A100" in card
    assert "@triton.jit" in card
    assert "class ModelNew" in card
    assert "kernel[grid]" in card
    assert card.count("This VectorAdd example") == 1
    assert "tl.sum" not in card
    assert "row reduction" not in card.lower()
    assert "row-reduction" not in card.lower()


def test_tilelang_and_cute_target_stacks_are_hash_verified() -> None:
    tilelang = get_target_stack("tilelang-a100")
    cute = get_target_stack("cute-a100")

    assert tilelang.backend == "tilelang"
    assert tilelang.version == "0.1.12"
    assert "@T.prim_func" in load_target_card(tilelang.id)
    assert cute.backend == "cute"
    assert cute.version == "4.6.1"
    assert "@cute.kernel" in load_target_card(cute.id)
    assert load_target_card(tilelang.id).count("This VectorAdd example") == 1
    assert load_target_card(cute.id).count("This VectorAdd example") == 1


def test_unknown_target_is_rejected() -> None:
    with pytest.raises(TargetRegistryError, match="unknown target stack"):
        get_target_stack("missing")
    with pytest.raises(TargetRegistryError, match="unknown target stack"):
        load_target_card("missing")


def test_unknown_target_registry_scope_is_rejected() -> None:
    with pytest.raises(TargetRegistryError, match="unknown target registry scope: missing"):
        list_target_ids(scope="missing")
    with pytest.raises(TargetRegistryError, match="unknown target registry scope: missing"):
        validate_target_registry(scope="missing")


def test_registry_validation_is_isolated_by_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_paths: list[str] = []
    loaded_roots: list[Path] = []

    def record_load(card: TargetCardAsset, *, asset_root: object = None) -> str:
        loaded_paths.append(card.relative_path)
        assert isinstance(asset_root, Path)
        loaded_roots.append(asset_root)
        return "card"

    monkeypatch.setattr(target_registry, "load_pinned_card", record_load)

    validate_target_registry(scope=CAPABILITY_GATE_SCOPE)
    assert loaded_paths == [
        "targets/core.md",
        "targets/full.md",
        "targets/map.md",
        "targets/sched.md",
    ]
    assert {root.name for root in loaded_roots} == {"capability-gate-a100"}

    loaded_paths.clear()
    loaded_roots.clear()
    validate_target_registry()
    assert loaded_paths == ["targets/cute.md", "targets/tilelang.md", "targets/triton.md"]
    assert {root.name for root in loaded_roots} == {"r1-a100"}


def test_capability_targets_bind_machine_rendered_cards_and_adapters() -> None:
    validate_target_registry(scope=CAPABILITY_GATE_SCOPE)
    for pack in (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK):
        target = get_target_stack(pack.target_id)
        card = load_target_card(pack.target_id)

        assert target.backend == "tilelang"
        assert target.version == "0.1.12"
        assert target.adapter == pack.adapter_id
        assert target.allowed_assets == ()
        assert card == render_tilelang_target_card(pack)


def test_global_lookup_indexes_targets_from_every_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = get_target_stack("triton-a100").model_copy(update={"id": "capability-only"})
    card = get_target_card_asset("triton-a100")
    stacks, cards = target_registry._build_global_indexes(
        {
            R1_SCOPE: target_registry._TargetRegistryScope(stacks={}, cards={}),
            CAPABILITY_GATE_SCOPE: target_registry._TargetRegistryScope(
                stacks={"capability-only": target},
                cards={"capability-only": card},
            ),
        }
    )
    monkeypatch.setattr(target_registry, "_TARGET_STACKS", stacks)
    monkeypatch.setattr(target_registry, "_TARGET_CARDS", cards)

    assert get_target_stack("capability-only").id == "capability-only"
    assert get_target_card_asset("capability-only") == card


def test_global_target_ids_must_be_unique_across_scopes() -> None:
    target = get_target_stack("triton-a100")
    card = get_target_card_asset("triton-a100")
    duplicate = target_registry._TargetRegistryScope(
        stacks={target.id: target},
        cards={target.id: card},
    )

    with pytest.raises(TargetRegistryError, match="registered in multiple scopes: triton-a100"):
        target_registry._build_global_indexes(
            {
                R1_SCOPE: duplicate,
                CAPABILITY_GATE_SCOPE: duplicate,
            }
        )


def test_pinned_card_rejects_parent_traversal(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()

    with pytest.raises(TargetRegistryError, match="unsafe target card path"):
        load_pinned_card(TargetCardAsset("../outside.md", digest), asset_root=root)


def test_target_registry_rejects_card_tampering(tmp_path: Path) -> None:
    target_directory = tmp_path / "targets"
    target_directory.mkdir()
    (target_directory / "cute.md").write_text(load_target_card("cute-a100"), encoding="utf-8")
    (target_directory / "tilelang.md").write_text(
        load_target_card("tilelang-a100"), encoding="utf-8"
    )
    (target_directory / "triton.md").write_text("changed", encoding="utf-8")

    with pytest.raises(TargetRegistryError, match="SHA-256 mismatch"):
        validate_target_registry(asset_root=tmp_path)


def test_importing_target_registry_does_not_import_gpu_libraries() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import abstrak.canary.targets; "
            "raise SystemExit(1 if {'torch', 'triton'} & set(sys.modules) else 0)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
