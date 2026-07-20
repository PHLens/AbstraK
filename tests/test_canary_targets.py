from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from abstrak.canary.targets import (
    TargetCardAsset,
    TargetRegistryError,
    get_target_stack,
    list_target_ids,
    load_pinned_card,
    load_target_card,
    validate_target_registry,
)


def test_triton_target_stack_is_frozen_for_a100() -> None:
    target = get_target_stack("triton-a100")

    assert list_target_ids() == ("triton-a100",)
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


def test_unknown_target_is_rejected() -> None:
    with pytest.raises(TargetRegistryError, match="unknown target stack"):
        get_target_stack("missing")
    with pytest.raises(TargetRegistryError, match="unknown target stack"):
        load_target_card("missing")


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
