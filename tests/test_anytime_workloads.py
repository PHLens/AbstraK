from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from abstrak.anytime.workloads import (
    DEFAULT_INPUT_MANIFEST,
    ENVIRONMENT_ASSET_PATHS,
    PINNED_INPUT_MANIFEST_SHA256,
    TARGET_IDS,
    WORKLOAD_IDS,
    WORKLOAD_SOURCE_INPUTS,
    AnytimeExpertSourceInput,
    AnytimeWorkloadError,
    AnytimeWorkloadInputManifest,
    formal_readiness_issues,
    load_anytime_workload_inputs,
    require_formal_workload_readiness,
    validate_anytime_workload_inputs,
    validate_public_workload_leakage,
    verify_kernelbench_source_inputs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TARGET_CARD_PATHS = tuple(
    f"benchmarks/r1-a100/targets/{backend}.md" for backend in ("triton", "tilelang", "cute")
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _render(manifest: AnytimeWorkloadInputManifest) -> bytes:
    return (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_manifest(path: Path, manifest: AnytimeWorkloadInputManifest) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _render(manifest)
    path.write_bytes(payload)
    return _sha256(payload)


def _mirror_repository_inputs(root: Path) -> None:
    relative_paths = (*TARGET_CARD_PATHS, *ENVIRONMENT_ASSET_PATHS.values())
    for relative_path in relative_paths:
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / relative_path).read_bytes())


def test_default_workload_inputs_are_frozen_complete_and_offline_only() -> None:
    pinned = load_anytime_workload_inputs()
    trusted = validate_anytime_workload_inputs(pinned)
    manifest = trusted.manifest

    assert _sha256(DEFAULT_INPUT_MANIFEST.read_bytes()) == PINNED_INPUT_MANIFEST_SHA256
    assert tuple(item.id for item in manifest.workloads) == WORKLOAD_IDS
    assert tuple(item.target_id for item in manifest.target_cards) == TARGET_IDS
    assert len(manifest.experts) == 36
    assert len(manifest.baselines) == 36
    assert len({item.generator.sha256 for item in manifest.workloads}) == 12
    assert all(
        set(case.seed for case in workload.generator.dev_cases).isdisjoint(
            case.seed for case in workload.generator.sealed_cases
        )
        for workload in manifest.workloads
    )
    assert all(
        workload.state_transfer.reference_state_slot == workload.state_transfer.candidate_state_slot
        for workload in manifest.workloads
        if workload.state_transfer.parameterized
    )
    assert all(
        not workload.state_transfer.sequential_random_construction
        for workload in manifest.workloads
    )
    assert len(formal_readiness_issues(manifest)) == 85
    with pytest.raises(AnytimeWorkloadError, match="environment observation pending M9"):
        require_formal_workload_readiness(manifest)


def test_public_views_exclude_lineage_seeds_sealed_cases_and_source_slots() -> None:
    manifest = load_anytime_workload_inputs().manifest
    validate_public_workload_leakage(manifest)

    workload = manifest.workloads[0]
    sealed_seed = str(workload.generator.sealed_cases[0].seed)
    tampered = workload.model_copy(
        update={"specification": workload.specification + f" private seed {sealed_seed}"}
    )
    forged = manifest.model_copy(update={"workloads": (tampered, *manifest.workloads[1:])})
    with pytest.raises(AnytimeWorkloadError, match="leaks private material"):
        validate_public_workload_leakage(forged)


def test_pinned_manifest_detects_raw_byte_changes_after_load(tmp_path: Path) -> None:
    path = tmp_path / "inputs.json"
    payload = DEFAULT_INPUT_MANIFEST.read_bytes()
    path.write_bytes(payload)
    pinned = load_anytime_workload_inputs(path, expected_sha256=_sha256(payload))
    path.write_bytes(payload + b" \n")

    with pytest.raises(AnytimeWorkloadError, match="bytes changed after load"):
        validate_anytime_workload_inputs(pinned)


def test_target_card_and_environment_assets_are_content_bound(tmp_path: Path) -> None:
    _mirror_repository_inputs(tmp_path)
    pinned = load_anytime_workload_inputs()
    validate_anytime_workload_inputs(pinned, repository_root=tmp_path)

    card = tmp_path / TARGET_CARD_PATHS[0]
    card.write_bytes(card.read_bytes() + b"\nchanged\n")
    with pytest.raises(AnytimeWorkloadError, match="target card SHA-256 mismatch"):
        validate_anytime_workload_inputs(pinned, repository_root=tmp_path)

    _mirror_repository_inputs(tmp_path)
    lock = tmp_path / ENVIRONMENT_ASSET_PATHS["lock_sha256"]
    lock.write_bytes(lock.read_bytes() + b"\nchanged\n")
    with pytest.raises(AnytimeWorkloadError, match="environment input lock_sha256"):
        validate_anytime_workload_inputs(pinned, repository_root=tmp_path)


def test_materialized_source_slots_are_hash_checked_without_execution(tmp_path: Path) -> None:
    _mirror_repository_inputs(tmp_path)
    default = load_anytime_workload_inputs().manifest
    original = default.experts[0]
    source_bytes = b"# synthetic source-binding fixture; never imported\n"
    materialized = AnytimeExpertSourceInput(
        task_id=original.task_id,
        target_id=original.target_id,
        source_path=original.source_path,
        status="materialized_pending_m9",
        expected_source_sha256=_sha256(source_bytes),
    )
    manifest = AnytimeWorkloadInputManifest(
        workloads=default.workloads,
        target_cards=default.target_cards,
        experts=(materialized, *default.experts[1:]),
        baselines=default.baselines,
        environment=default.environment,
        floor_policy=default.floor_policy,
    )
    source_path = tmp_path / materialized.source_path
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_bytes)
    manifest_path = tmp_path / "manifests" / "materialized.json"
    expected = _write_manifest(manifest_path, manifest)
    pinned = load_anytime_workload_inputs(manifest_path, expected_sha256=expected)
    validate_anytime_workload_inputs(pinned, repository_root=tmp_path)

    source_path.write_bytes(source_bytes + b"# tampered\n")
    with pytest.raises(AnytimeWorkloadError, match="materialized source"):
        validate_anytime_workload_inputs(pinned, repository_root=tmp_path)


def test_source_state_machine_rejects_unearned_live_claims() -> None:
    with pytest.raises(ValidationError, match="cannot claim source bytes"):
        AnytimeExpertSourceInput(
            task_id=WORKLOAD_IDS[0],
            target_id=TARGET_IDS[0],
            source_path=(
                f"benchmarks/anytime-dsl-a100/experts/{WORKLOAD_IDS[0]}/{TARGET_IDS[0]}.py"
            ),
            expected_source_sha256="a" * 64,
        )
    with pytest.raises(ValidationError, match="all M9 observations"):
        AnytimeExpertSourceInput(
            task_id=WORKLOAD_IDS[0],
            target_id=TARGET_IDS[0],
            source_path=(
                f"benchmarks/anytime-dsl-a100/experts/{WORKLOAD_IDS[0]}/{TARGET_IDS[0]}.py"
            ),
            status="validated_m9",
            expected_source_sha256="a" * 64,
            formal_ready=True,
        )


def test_kernelbench_source_verifier_fails_closed_on_wrong_bytes(tmp_path: Path) -> None:
    first_source = WORKLOAD_SOURCE_INPUTS[0][3]
    path = tmp_path / first_source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# wrong frozen source\n")

    with pytest.raises(AnytimeWorkloadError, match="KernelBench source SHA-256 mismatch"):
        verify_kernelbench_source_inputs(load_anytime_workload_inputs(), tmp_path)
