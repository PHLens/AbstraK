from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from abstrak.canary.manifests import (
    StudyManifestError,
    load_study_manifest,
    load_study_spec,
)
from abstrak.canary.matrix import build_matrix_schedule

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_STUDY = REPOSITORY_ROOT / "benchmarks" / "capability-gate-a100" / "study.json"
CAPABILITY_STUDY_SHA256 = "876b18e75d86e77c6e2e4cd47038f60719ba6108943ddc754086ea82685ecd00"
CAPABILITY_SCHEDULE_SHA256 = "40c372285875337ebd62529d72b2dd5bc2f6d123cbb2940a93c7482d2537983e"


def test_capability_study_manifest_is_pinned_and_matches_the_frozen_axes() -> None:
    pinned = load_study_spec(
        CAPABILITY_STUDY,
        expected_sha256=CAPABILITY_STUDY_SHA256,
    )
    spec = pinned.spec

    assert pinned.path == CAPABILITY_STUDY.resolve()
    assert pinned.sha256 == hashlib.sha256(CAPABILITY_STUDY.read_bytes()).hexdigest()
    assert pinned.sha256 == CAPABILITY_STUDY_SHA256
    assert spec.study_id == "tilelang-capability-gate-a100-v1"
    assert spec.seed == 20260723
    assert spec.agents == ("deepseek-v4-pro",)
    assert spec.targets == (
        "tilelang-a100-core",
        "tilelang-a100-sched",
        "tilelang-a100-map",
        "tilelang-a100-full",
    )
    assert tuple(group.id for group in spec.task_groups) == (
        "base-simple",
        "schedule",
        "mapping",
        "schedule-mapping-interaction",
    )
    assert all(phase.replicates == (1, 2, 3) for phase in spec.phases)
    assert all(phase.order_policy == "balanced_rotation" for phase in spec.phases)
    assert all(phase.max_calls_per_trajectory == 3 for phase in spec.phases)
    assert all(phase.infrastructure_retries == 1 for phase in spec.phases)


def test_capability_manifest_materializes_96_cells_and_dynamic_ceilings() -> None:
    pinned = load_study_manifest(
        CAPABILITY_STUDY,
        expected_sha256=CAPABILITY_STUDY_SHA256,
    )
    schedule = build_matrix_schedule(pinned.spec)

    assert len(schedule.cells_for_phase("core")) == 48
    assert len(schedule.cells_for_phase("reserve")) == 48
    assert schedule.expected_trajectories == 96
    assert schedule.phase_request_ceiling("core") == 144
    assert schedule.phase_request_ceiling("reserve") == 144
    assert schedule.request_ceiling == 288
    assert schedule.phase_operational_request_ceiling("core") == 288
    assert schedule.phase_operational_request_ceiling("reserve") == 288
    assert schedule.operational_request_ceiling == 576
    assert schedule.sha256 == CAPABILITY_SCHEDULE_SHA256
    assert len({cell.trajectory_id for cell in schedule.cells}) == 96


def test_capability_manifest_freezes_current_gate_thresholds() -> None:
    gate = load_study_spec(CAPABILITY_STUDY).spec.gate

    assert gate is not None
    assert gate.core_phase_id == "core"
    assert gate.reserve_phase_id == "reserve"
    assert gate.metrics.competitive_latency_factor == 1.25
    assert gate.metrics.latency_tie_fraction == 0.10
    assert gate.metrics.max_timing_cv == 0.05
    assert gate.core.min_stable_tasks == 3
    assert gate.core.min_competitive_gap_units == 3
    assert gate.core.no_go_max_competitive_gap_units == 1
    assert gate.full is not None
    assert gate.full.min_stable_tasks == 6
    assert gate.full.min_competitive_gap_units == 3
    assert gate.full.dominant_winner_min_tasks == 7
    assert gate.reserve_on_outcomes == ("provisional_go", "inconclusive")


def test_expected_sha_rejects_tampering_before_schema_validation(tmp_path: Path) -> None:
    tampered = tmp_path / "study.json"
    tampered.write_bytes(CAPABILITY_STUDY.read_bytes().replace(b"20260723", b"20260724"))

    with pytest.raises(StudyManifestError, match="SHA-256 mismatch"):
        load_study_spec(tampered, expected_sha256=CAPABILITY_STUDY_SHA256)

    unpinned = load_study_spec(tampered)
    assert unpinned.sha256 != CAPABILITY_STUDY_SHA256
    assert unpinned.spec.seed == 20260724


def test_loader_rejects_invalid_expected_sha_before_file_access() -> None:
    with pytest.raises(StudyManifestError, match="expected.*SHA-256 is invalid"):
        load_study_spec("does-not-exist.json", expected_sha256="not-a-sha")


def test_loader_requires_a_regular_utf8_json_file(tmp_path: Path) -> None:
    with pytest.raises(StudyManifestError, match="file does not exist"):
        load_study_spec(tmp_path / "missing.json")
    with pytest.raises(StudyManifestError, match="not a regular file"):
        load_study_spec(tmp_path)

    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe")
    with pytest.raises(StudyManifestError, match="not UTF-8"):
        load_study_spec(binary)

    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema_version":"abstrak-matrix-study-spec.v1"}', encoding="utf-8")
    with pytest.raises(StudyManifestError, match="invalid study manifest"):
        load_study_spec(invalid)


def test_loader_rejects_unknown_manifest_fields(tmp_path: Path) -> None:
    drifted = tmp_path / "drifted.json"
    payload = CAPABILITY_STUDY.read_text(encoding="utf-8")
    drifted.write_text(payload.replace("{", '{\n  "unknown": true,', 1), encoding="utf-8")

    with pytest.raises(StudyManifestError, match="invalid study manifest"):
        load_study_spec(drifted)
