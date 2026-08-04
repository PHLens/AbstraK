from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from abstrak.anytime.freeze import (
    CORE_SOURCE_ASSET_PATHS,
    CORE_WORKLOAD_IDS,
    FORMAL_STUDY_FILENAME,
    M9_BLOCKERS,
    OFFLINE_FREEZE_FILENAME,
    RESERVE_WORKLOAD_IDS,
    SHAKEOUT_STUDY_FILENAME,
    SHAKEOUT_WORKLOAD_IDS,
    AnytimeEvaluationPolicy,
    AnytimeFreezeError,
    AnytimeOfflineFreezeManifest,
    AnytimeTaskGroups,
    AnytimeTimingPolicy,
    AnytimeTimingProtocol,
    PinnedAnytimeOfflineFreeze,
    build_anytime_formal_study,
    build_anytime_offline_freeze,
    build_anytime_shakeout_study,
    check_anytime_freeze_manifests,
    frozen_request_ceilings,
    load_anytime_offline_freeze,
    verify_anytime_offline_freeze,
    write_anytime_freeze_manifests,
)
from abstrak.anytime.schedule import build_anytime_schedule
from abstrak.anytime.workloads import TARGET_IDS, WORKLOAD_IDS


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_studies_have_exact_axes_budgets_and_hashes() -> None:
    formal = build_anytime_formal_study()
    shakeout = build_anytime_shakeout_study()
    formal_schedule = build_anytime_schedule(formal)
    shakeout_schedule = build_anytime_schedule(shakeout)

    assert tuple(agent.id for agent in formal.agents) == (
        "deepseek-v4-flash",
        "gpt-5.6-luna",
    )
    assert all(
        agent.generation.reasoning.requested_reasoning_effort == "xhigh"
        for agent in formal.agents
    )
    assert all(
        agent.generation.reasoning.conformance_requirement == "literal_xhigh"
        for agent in formal.agents
    )
    assert all(agent.generation.max_output_tokens == 16384 for agent in formal.agents)
    assert all(agent.generation.temperature is None for agent in formal.agents)
    assert all(agent.generation.top_p is None for agent in formal.agents)
    assert tuple(cohort.id for cohort in formal.cohorts) == (
        "primary-core",
        "robustness-core",
        "primary-reserve",
    )
    assert tuple(cohort.expected_trajectories for cohort in formal.cohorts) == (72, 54, 72)
    assert formal.cohorts[0].task_ids == CORE_WORKLOAD_IDS
    assert formal.cohorts[1].task_ids == CORE_WORKLOAD_IDS
    assert formal.cohorts[2].task_ids == RESERVE_WORKLOAD_IDS
    assert formal.cohorts[2].activation == "core_gate"
    assert formal_schedule.expected_trajectories == 198
    assert len(formal_schedule.executable_cells()) == 126
    assert formal_schedule.scientific_request_ceiling == 2376
    assert formal_schedule.operational_request_ceiling == 4752
    assert formal.sha256 == "e5534fecf63d83cb89708e8c3c7b0b7c9a2163d63cc93d9a42c8d030d4a015ef"
    assert formal_schedule.sha256 == (
        "2582bd909cf7f9bfb43960e7443062155b0efea07c5cff06754978a877f056e9"
    )

    assert shakeout.study_kind == "shakeout"
    assert all(cohort.task_ids == SHAKEOUT_WORKLOAD_IDS for cohort in shakeout.cohorts)
    assert all(cohort.replicates == (1, 2) for cohort in shakeout.cohorts)
    assert all(cohort.scoring is False for cohort in shakeout.cohorts)
    assert tuple(cohort.expected_trajectories for cohort in shakeout.cohorts) == (24, 24)
    assert shakeout_schedule.expected_trajectories == 48
    assert len(shakeout_schedule.executable_cells()) == 48
    assert shakeout_schedule.scientific_request_ceiling == 192
    assert shakeout_schedule.operational_request_ceiling == 384
    assert shakeout.sha256 == "427639a328c9bfbdf1f63e8ca64f7dcc69327b6da94d76a30c7827d6218cb069"
    assert shakeout_schedule.sha256 == (
        "bc99f932581445505a9e9ec6aa7259af00545696242307e6c3bcd7b1d3883515"
    )
    assert frozen_request_ceilings() == (
        ("formal", 198, 2376, 4752),
        ("shakeout", 48, 192, 384),
    )


def test_task_timing_winner_and_gate_policies_are_frozen() -> None:
    manifest = build_anytime_offline_freeze()

    assert manifest.task_groups == AnytimeTaskGroups(
        core=CORE_WORKLOAD_IDS,
        reserve=RESERVE_WORKLOAD_IDS,
        shakeout=SHAKEOUT_WORKLOAD_IDS,
    )
    assert set((*manifest.task_groups.core, *manifest.task_groups.reserve)) == set(WORKLOAD_IDS)
    assert manifest.targets == TARGET_IDS
    assert manifest.randomization.seed == 20260803
    assert manifest.randomization.target_order == "deterministic-balanced-rotation"
    assert manifest.base_prompt.renderer_version == "anytime-base-prompt-renderer.v1"
    assert manifest.base_prompt.workload_disclosure == "public-whitelist-view-only"
    assert manifest.base_prompt.target_card_disclosure == "selected-target-card-only"
    assert manifest.base_prompt.sealed_cases_disclosed is False
    assert manifest.base_prompt.trusted_expert_source_disclosed is False
    assert manifest.base_prompt.other_target_results_disclosed is False

    timing = manifest.evaluation.timing
    assert (
        timing.search.warmup_runs,
        timing.search.timed_trials,
        timing.search.discard_initial_trials,
        timing.search.independent_clean_process_blocks,
        timing.search.evidence_scope,
    ) == (5, 100, 1, 1, "exploratory-dev")
    assert (
        timing.formal_checkpoint.warmup_runs,
        timing.formal_checkpoint.timed_trials,
        timing.formal_checkpoint.discard_initial_trials,
        timing.formal_checkpoint.independent_clean_process_blocks,
        timing.formal_checkpoint.evidence_scope,
    ) == (25, 200, 1, 3, "formal-sealed")
    assert manifest.evaluation.winner.practical_equivalence_relative_tolerance == 0.05
    assert manifest.evaluation.winner.p_best_role == "descriptive-only"

    continuation = manifest.evaluation.continuation
    assert continuation.minimum_eligible_workloads_per_target_full == 8
    assert continuation.minimum_eligible_workloads_per_target_core == 4
    assert continuation.minimum_distinct_stable_winner_targets == 2
    assert continuation.minimum_stable_winner_workloads == 3
    assert continuation.minimum_stable_winner_families == 2
    assert continuation.iteration_endpoint_oracle_ratio == 1.05
    assert continuation.common_wall_clock_oracle_ratio == 1.03
    assert continuation.common_wall_clock_bootstrap_lower_ratio == 1.0
    assert continuation.robustness_winner_agreement_workloads == 4
    assert continuation.robustness_alternative_oracle_ratio == 1.03

    shakeout = manifest.evaluation.shakeout
    assert shakeout.minimum_stable_correct_workloads_per_target == 2
    assert shakeout.minimum_distinct_families_per_target == 2
    assert shakeout.maximum_infrastructure_censoring_rate == 0.05
    assert shakeout.maximum_uniform_target_card_revisions == 1
    analysis = manifest.evaluation.analysis
    assert analysis.wall_clock_grid == (
        "derive-common-support-from-verified-checkpoint-artifacts"
    )
    assert analysis.bootstrap_cluster_unit == "semantic-workload-family"
    assert analysis.bootstrap_seed == 20260803
    assert analysis.bootstrap_resamples == 1000
    assert analysis.missing_cells_retained_in_denominator is True


def test_policy_models_reject_threshold_or_timing_drift() -> None:
    search = AnytimeTimingProtocol(
        role="search-selection",
        warmup_runs=5,
        timed_trials=100,
        discard_initial_trials=1,
        independent_clean_process_blocks=1,
        evidence_scope="exploratory-dev",
    )
    formal = AnytimeTimingProtocol(
        role="formal-checkpoint",
        warmup_runs=25,
        timed_trials=200,
        discard_initial_trials=1,
        independent_clean_process_blocks=3,
        evidence_scope="formal-sealed",
    )
    evaluation = AnytimeEvaluationPolicy(
        timing=AnytimeTimingPolicy(search=search, formal_checkpoint=formal)
    )
    payload = evaluation.model_dump(mode="python")
    payload["timing"]["formal_checkpoint"]["timed_trials"] = 199
    with pytest.raises(ValidationError, match="timing protocols differ"):
        AnytimeEvaluationPolicy.model_validate(payload)

    payload = evaluation.model_dump(mode="python")
    payload["winner"]["practical_equivalence_relative_tolerance"] = 0.03
    with pytest.raises(ValidationError):
        AnytimeEvaluationPolicy.model_validate(payload)

    with pytest.raises(ValidationError, match="core workload group differs"):
        AnytimeTaskGroups(
            core=tuple(reversed(CORE_WORKLOAD_IDS)),
            reserve=RESERVE_WORKLOAD_IDS,
            shakeout=SHAKEOUT_WORKLOAD_IDS,
        )


def test_freeze_binds_m6_isolation_static_policies_dependencies_and_non_live_state() -> None:
    manifest = build_anytime_offline_freeze()

    assert len(manifest.workload_inputs.raw_sha256) == 64
    assert len(manifest.workload_inputs.canonical_manifest_sha256) == 64
    assert len(manifest.workload_inputs.environment_contract_sha256) == 64
    assert len(manifest.workload_inputs.floor_policy_sha256) == 64
    assert len(manifest.workload_inputs.isolation_contract_sha256) == 64
    assert len(set(manifest.workload_inputs.target_static_policy_sha256)) == 3
    assert tuple(binding.agent_id for binding in manifest.provider_dependencies) == (
        "deepseek-v4-flash",
        "gpt-5.6-luna",
    )
    deepseek, gpt = manifest.provider_dependencies
    assert deepseek.conformance.status == "fail"
    assert deepseek.conformance.reasoning.fidelity == "collapsed"
    assert deepseek.conformance.study_readiness == "dependency_blocked"
    assert gpt.conformance.status == "pass"
    assert gpt.conformance.reasoning.fidelity == "literal"
    assert gpt.conformance.study_readiness == "pending_endpoint_conformance"
    assert all(
        binding.conformance.study_ready is False
        for binding in manifest.provider_dependencies
    )

    assert manifest.authorization_policy == "no-live-action-authorized"
    assert manifest.live_ready is False
    assert manifest.provider_requests_performed is False
    assert manifest.credentials_read is False
    assert manifest.ssh_connections_performed is False
    assert manifest.gpu_code_executed is False
    assert manifest.candidate_code_executed is False
    assert manifest.generated_code_created is False
    assert manifest.live_environment_observed is False
    assert manifest.live_floor_constructed is False
    assert manifest.m9_blockers == M9_BLOCKERS
    asset_paths = tuple(item.relative_path for item in manifest.source_assets)
    assert asset_paths == tuple(sorted(CORE_SOURCE_ASSET_PATHS))
    assert "src/abstrak/anytime/freeze_pins.py" not in asset_paths
    assert "scripts/freeze_anytime_offline.py" in asset_paths
    assert "src/abstrak/anytime/prompts.py" in asset_paths
    assert "src/abstrak/anytime/rehearsal.py" in asset_paths
    assert "src/abstrak/providers/contracts.py" in asset_paths
    assert "src/abstrak/providers/__init__.py" in asset_paths
    assert "benchmarks/r1-a100/targets/triton.md" in asset_paths
    assert not any(path.endswith("study.json") for path in asset_paths)
    assert OFFLINE_FREEZE_FILENAME not in asset_paths


def test_write_load_check_and_verify_round_trip(tmp_path: Path) -> None:
    result = write_anytime_freeze_manifests(tmp_path)

    assert result.formal_raw_sha256 == _digest(tmp_path / FORMAL_STUDY_FILENAME)
    assert result.shakeout_raw_sha256 == _digest(tmp_path / SHAKEOUT_STUDY_FILENAME)
    assert result.freeze_raw_sha256 == _digest(tmp_path / OFFLINE_FREEZE_FILENAME)
    pinned = load_anytime_offline_freeze(
        tmp_path / OFFLINE_FREEZE_FILENAME,
        expected_sha256=result.freeze_raw_sha256,
    )
    assert verify_anytime_offline_freeze(pinned) == result.manifest
    assert check_anytime_freeze_manifests(
        tmp_path,
        expected_formal_sha256=result.formal_raw_sha256,
        expected_shakeout_sha256=result.shakeout_raw_sha256,
        expected_freeze_sha256=result.freeze_raw_sha256,
    ) == result.manifest

    for name, model in (
        (FORMAL_STUDY_FILENAME, build_anytime_formal_study()),
        (SHAKEOUT_STUDY_FILENAME, build_anytime_shakeout_study()),
    ):
        expected = (
            json.dumps(
                model.model_dump(mode="json"),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        assert (tmp_path / name).read_bytes() == expected


def test_tampered_study_freeze_and_forged_object_fail_closed(tmp_path: Path) -> None:
    result = write_anytime_freeze_manifests(tmp_path)
    freeze_path = tmp_path / OFFLINE_FREEZE_FILENAME
    pinned = load_anytime_offline_freeze(
        freeze_path,
        expected_sha256=result.freeze_raw_sha256,
    )

    forged_manifest = pinned.manifest.model_copy(update={"live_ready": True})
    forged = PinnedAnytimeOfflineFreeze(
        path=pinned.path,
        raw_sha256=pinned.raw_sha256,
        manifest=forged_manifest,
    )
    with pytest.raises(AnytimeFreezeError, match="invalid pinned offline-freeze object"):
        verify_anytime_offline_freeze(forged)

    formal_path = tmp_path / FORMAL_STUDY_FILENAME
    formal_payload = json.loads(formal_path.read_text(encoding="utf-8"))
    formal_payload["seed"] += 1
    formal_path.write_text(json.dumps(formal_payload), encoding="utf-8")
    with pytest.raises(AnytimeFreezeError, match="formal study raw SHA-256 mismatch"):
        verify_anytime_offline_freeze(pinned)

    write_anytime_freeze_manifests(tmp_path)
    freeze_path.write_bytes(freeze_path.read_bytes() + b" ")
    with pytest.raises(AnytimeFreezeError, match="offline-freeze SHA-256 mismatch"):
        load_anytime_offline_freeze(
            freeze_path,
            expected_sha256=result.freeze_raw_sha256,
        )


def test_semantically_equal_noncanonical_freeze_is_rejected(tmp_path: Path) -> None:
    write_anytime_freeze_manifests(tmp_path)
    freeze_path = tmp_path / OFFLINE_FREEZE_FILENAME
    parsed = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_path.write_text(json.dumps(parsed), encoding="utf-8")
    noncanonical_hash = _digest(freeze_path)
    pinned = load_anytime_offline_freeze(
        freeze_path,
        expected_sha256=noncanonical_hash,
    )

    with pytest.raises(AnytimeFreezeError, match="not canonical JSON"):
        verify_anytime_offline_freeze(pinned)


def test_freeze_rejects_a_repository_mirror_with_different_loaded_code(tmp_path: Path) -> None:
    with pytest.raises(AnytimeFreezeError, match="differs from the checkout"):
        write_anytime_freeze_manifests(tmp_path / "output", repository_root=tmp_path)


def test_freeze_contract_rejects_recursive_assets_unknown_fields_and_changed_blockers() -> None:
    manifest = build_anytime_offline_freeze()
    payload = manifest.model_dump(mode="python")
    payload["unknown"] = True
    with pytest.raises(ValidationError):
        AnytimeOfflineFreezeManifest.model_validate(payload)

    payload = manifest.model_dump(mode="python")
    payload["m9_blockers"] = payload["m9_blockers"][:-1]
    with pytest.raises(ValidationError, match="M9 blockers differ"):
        AnytimeOfflineFreezeManifest.model_validate(payload)

    payload = manifest.model_dump(mode="python")
    assets = list(payload["source_assets"])
    recursive = dict(assets[-1])
    recursive["relative_path"] = "src/abstrak/anytime/freeze_pins.py"
    assets.append(recursive)
    assets.sort(key=lambda item: item["relative_path"])
    payload["source_assets"] = tuple(assets)
    payload["source_asset_bundle_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="freeze pins must be excluded"):
        AnytimeOfflineFreezeManifest.model_validate(payload)
