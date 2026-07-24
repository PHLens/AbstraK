from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from abstrak.canary.capabilities import (
    CORE_PACK,
    FULL_PACK,
    MAP_PACK,
    SCHED_PACK,
    validate_tilelang_capability_source,
)
from abstrak.canary.capability_assets import (
    CAPABILITY_ASSET_MANIFEST_SHA256,
    CapabilityAssetError,
    build_capability_asset_manifest,
    get_capability_canary,
    list_capability_canary_ids,
    load_capability_canary,
    validate_capability_asset_registry,
)
from abstrak.canary.manifests import load_study_spec
from abstrak.canary.matrix import build_matrix_schedule
from abstrak.canary.tasks import (
    CAPABILITY_GATE_TASK_IDS,
    load_oracle_source,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_ROOT = REPOSITORY_ROOT / "benchmarks" / "capability-gate-a100"
CAPABILITY_STUDY = CAPABILITY_ROOT / "study.json"
CAPABILITY_STUDY_SHA256 = "876b18e75d86e77c6e2e4cd47038f60719ba6108943ddc754086ea82685ecd00"


def test_experts_are_b_legal_under_every_nested_pack() -> None:
    validate_capability_asset_registry()
    packs = (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK)

    for task_id in CAPABILITY_GATE_TASK_IDS:
        source = load_oracle_source(task_id, "tilelang")
        results = tuple(validate_tilelang_capability_source(source, pack) for pack in packs)
        assert [result.valid for result in results] == [True, True, True, True]
        assert {result.minimum_pack_id for result in results} == {CORE_PACK.id}


def test_canaries_have_exact_non_core_acceptance_matrix() -> None:
    assert list_capability_canary_ids() == ("schedule", "mapping", "schedule-mapping")
    packs = (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK)
    expected = {
        "schedule": [False, True, False, True],
        "mapping": [False, False, True, True],
        "schedule-mapping": [False, False, False, True],
    }
    for canary_id, acceptance in expected.items():
        canary = get_capability_canary(canary_id)
        source = load_capability_canary(canary_id)
        results = tuple(validate_tilelang_capability_source(source, pack) for pack in packs)
        assert [result.valid for result in results] == acceptance
        assert {result.minimum_pack_id for result in results} == {canary.minimum_pack_id}

    assert get_capability_canary("schedule").task_id == "gemm-large-k-static"
    assert get_capability_canary("mapping").task_id == "row-sum-static"
    assert get_capability_canary("schedule-mapping").task_id == "gemm-large-k-static"
    schedule_source = load_capability_canary("schedule")
    mapping_source = load_capability_canary("mapping")
    assert all(value in schedule_source for value in ("M = 1024", "N = 4096", "K = 4096"))
    assert all(value in mapping_source for value in ("ROWS = 16384", "COLUMNS = 4096"))
    assert "T.sync_threads()" in load_capability_canary("schedule-mapping")


def test_real_study_builds_one_deterministic_generic_asset_manifest() -> None:
    pinned = load_study_spec(CAPABILITY_STUDY, expected_sha256=CAPABILITY_STUDY_SHA256)
    schedule = build_matrix_schedule(pinned.spec)

    first = build_capability_asset_manifest(pinned, schedule)
    second = build_capability_asset_manifest(pinned, schedule)

    assert first == second
    assert first.sha256 == second.sha256 == CAPABILITY_ASSET_MANIFEST_SHA256
    assert tuple(task.task_id for task in first.tasks) == (
        "gelu-static",
        "gemm-large-k-static",
        "gemm-small-k-irregular-static",
        "row-softmax-static",
        "gated-silu-static",
        "gemm-bias-relu-mirror-static",
        "row-sum-static",
        "rmsnorm-wide-static",
    )
    assert tuple(target.target_id for target in first.targets) == pinned.spec.targets
    assert len(first.tasks) == 8
    assert sum(len(task.baselines) for task in first.tasks) == 24
    assert tuple(canary.canary_id for canary in first.canaries) == (
        "schedule",
        "mapping",
        "schedule-mapping",
    )


def test_canary_source_is_hash_verified(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    directory = root / "canaries"
    directory.mkdir(parents=True)
    (directory / "schedule.py").write_text("changed\n", encoding="utf-8")

    with pytest.raises(CapabilityAssetError, match="SHA-256 mismatch"):
        load_capability_canary("schedule", asset_root=root)
    with pytest.raises(CapabilityAssetError, match="unknown capability canary"):
        get_capability_canary("missing")


def test_importing_capability_assets_does_not_import_torch() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import abstrak.canary.capability_assets; "
            "raise SystemExit(1 if 'torch' in sys.modules else 0)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
