from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image
from pydantic import BaseModel, ValidationError

from abstrak.anytime.analysis import (
    AnytimeAgentReplicateAxis,
    AnytimeAnalysisDataset,
    AnytimeAnalysisError,
    AnytimeAnalysisReport,
    AnytimeAnalysisSpec,
    AnytimeArtifactTrust,
    AnytimeFloorArtifact,
    AnytimeTrajectoryArtifact,
    AnytimeTurnArtifact,
    AnytimeWorkloadAxis,
    aggregate_anytime_rates,
    anytime_analysis_tables,
    build_anytime_analysis,
    clustered_interval,
)
from abstrak.anytime.figures import (
    AnytimeAnalysisBundleManifest,
    AnytimeFigureError,
    AnytimeFigureManifest,
    verify_anytime_analysis_bundle,
    write_anytime_analysis_bundle,
)

HASH = "a" * 64
SPEC_HASH = "b" * 64
AGENTS = ("agent-a", "agent-b")
WORKLOADS = (
    AnytimeWorkloadAxis(workload_id="reduce", semantic_family_id="reduction"),
    AnytimeWorkloadAxis(workload_id="gemm", semantic_family_id="dense"),
    AnytimeWorkloadAxis(workload_id="fused", semantic_family_id="dense"),
)
TARGETS = ("triton", "tilelang", "cute")


def _trust(suffix: str = "a") -> AnytimeArtifactTrust:
    return AnytimeArtifactTrust(artifact_manifest_sha256=suffix * 64)


def _spec() -> AnytimeAnalysisSpec:
    return AnytimeAnalysisSpec(
        study_id="synthetic-study",
        study_spec_sha256=SPEC_HASH,
        study_stage="synthetic_fixture",
        agents=tuple(
            AnytimeAgentReplicateAxis(agent_id=agent, replicates=(1, 2)) for agent in AGENTS
        ),
        workloads=WORKLOADS,
        targets=TARGETS,
        max_scientific_calls=4,
        formal_checkpoints=(1, 4),
        wall_clock_budgets_seconds=(15.0, 25.0, 45.0),
        bootstrap_resamples=200,
    )


def _floor(
    workload: AnytimeWorkloadAxis,
    target: str,
    *,
    status: str = "valid",
) -> AnytimeFloorArtifact:
    common = dict(
        trust=_trust("c"),
        study_id="synthetic-study",
        study_spec_sha256=SPEC_HASH,
        workload_id=workload.workload_id,
        semantic_family_id=workload.semantic_family_id,
        target_id=target,
        status=status,
    )
    if status == "valid":
        return AnytimeFloorArtifact(
            **common,
            eager_latency_ms=10.0,
            bstar_latency_ms=5.0,
            target_expert_latency_ms=3.0,
            clean_process=True,
            timing_stable=True,
            independently_sealed=True,
            timing_trial_count=100,
        )
    return AnytimeFloorArtifact(**common)


def _latencies(agent: str, workload: str, target: str) -> tuple[float, ...]:
    curves = {
        # Crossover on reduce: TileLang leads at call 1, Triton at call 4.
        ("agent-a", "reduce", "triton"): (8.0, 5.0, 3.0, 2.0),
        ("agent-a", "reduce", "tilelang"): (6.0, 4.0, 4.0, 4.0),
        ("agent-a", "reduce", "cute"): (9.0, 8.0, 7.0, 6.0),
        # One-target dominance on GEMM.
        ("agent-a", "gemm", "triton"): (6.0, 5.0, 4.5, 4.0),
        ("agent-a", "gemm", "tilelang"): (5.0, 4.0, 3.0, 2.0),
        ("agent-a", "gemm", "cute"): (8.0, 7.0, 6.5, 6.0),
        # Formal five-percent tie at the final checkpoint.
        ("agent-a", "fused", "triton"): (6.0, 4.0, 3.0, 2.0),
        ("agent-a", "fused", "tilelang"): (6.2, 4.2, 3.1, 2.08),
        ("agent-a", "fused", "cute"): (8.0, 7.0, 6.0, 5.0),
        # The second model reverses the reduce ranking.
        ("agent-b", "reduce", "triton"): (7.0, 5.0, 3.0, 2.2),
        ("agent-b", "reduce", "tilelang"): (6.0, 4.0, 3.0, 1.8),
        ("agent-b", "reduce", "cute"): (9.0, 8.0, 7.0, 6.0),
        ("agent-b", "gemm", "triton"): (6.0, 5.0, 4.5, 4.0),
        ("agent-b", "gemm", "tilelang"): (5.0, 4.0, 3.0, 2.0),
        ("agent-b", "gemm", "cute"): (8.0, 7.0, 6.5, 6.0),
        ("agent-b", "fused", "triton"): (6.0, 4.0, 3.0, 2.0),
        ("agent-b", "fused", "tilelang"): (6.1, 4.1, 3.0, 2.0),
        ("agent-b", "fused", "cute"): (8.0, 7.0, 6.0, 5.0),
    }
    return curves[(agent, workload, target)]


def _trajectory(
    agent: str,
    workload: AnytimeWorkloadAxis,
    target: str,
    replicate: int,
    *,
    terminal_status: str = "complete",
    turn_count: int = 4,
    latency_transform: Callable[[int, float], float] | None = None,
) -> AnytimeTrajectoryArtifact:
    turns = []
    for index, raw_latency in enumerate(_latencies(agent, workload.workload_id, target), start=1):
        if index > turn_count:
            break
        latency = (
            raw_latency if latency_transform is None else latency_transform(index, raw_latency)
        )
        turns.append(
            AnytimeTurnArtifact(
                scientific_call_index=index,
                cumulative_wall_seconds=float(index * 10),
                candidate_stage="eligible",
                incumbent_candidate_sha256=f"{index:x}" * 64,
                incumbent_latency_ms=latency,
                measurement_kind=("formal_checkpoint" if index in (1, 4) else "exploratory_dev"),
                clean_process_measurement=index in (1, 4),
                independently_retimed=index in (1, 4),
                timing_trial_count=50,
            )
        )
    return AnytimeTrajectoryArtifact(
        trust=_trust("d"),
        study_id="synthetic-study",
        study_spec_sha256=SPEC_HASH,
        trajectory_id=f"{agent}-{workload.workload_id}-{target}-r{replicate}",
        agent_id=agent,
        workload_id=workload.workload_id,
        semantic_family_id=workload.semantic_family_id,
        target_id=target,
        replicate=replicate,
        terminal_status=terminal_status,
        turns=tuple(turns),
    )


def _dataset() -> AnytimeAnalysisDataset:
    return AnytimeAnalysisDataset(
        spec=_spec(),
        floors=tuple(_floor(workload, target) for workload in WORKLOADS for target in TARGETS),
        trajectories=tuple(
            _trajectory(agent, workload, target, replicate)
            for agent in AGENTS
            for workload in WORKLOADS
            for target in TARGETS
            for replicate in (1, 2)
        ),
    )


def _replace_dataset(dataset: AnytimeAnalysisDataset, **updates: object) -> AnytimeAnalysisDataset:
    payload = dataset.model_dump(mode="json")
    payload.update(updates)
    return AnytimeAnalysisDataset.model_validate_json(
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    )


def _canonical_model_bytes(value: BaseModel) -> bytes:
    return (
        json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def test_artifact_schemas_are_strict_and_revalidated_at_analysis_boundary() -> None:
    with pytest.raises(ValidationError, match="extra"):
        AnytimeArtifactTrust.model_validate({**_trust().model_dump(), "raw_path": "/unverified"})
    with pytest.raises(ValidationError, match="finite|greater than or equal"):
        AnytimeTurnArtifact(
            scientific_call_index=1,
            cumulative_wall_seconds=float("nan"),
            candidate_stage="parse_failure",
        )

    tampered = _dataset().model_copy(
        update={"spec": _spec().model_copy(update={"max_scientific_calls": 99})}
    )
    with pytest.raises(AnytimeAnalysisError, match="invalid analysis artifact projection"):
        build_anytime_analysis(tampered)


def test_rates_use_planned_denominator_and_keep_checkpoint_identity() -> None:
    dataset = _dataset()
    trajectories = list(dataset.trajectories)
    original = trajectories[0]
    stage_turns = list(original.turns)
    stage_turns[1] = AnytimeTurnArtifact(
        scientific_call_index=2,
        cumulative_wall_seconds=20.0,
        candidate_stage="compile_failure",
    )
    trajectories[0] = original.model_copy(update={"turns": tuple(stage_turns)})
    # One absent artifact is infrastructure missing, not a scientific failure silently relabelled.
    dataset = _replace_dataset(
        dataset,
        trajectories=[row.model_dump(mode="json") for row in trajectories[:-1]],
    )

    rows = aggregate_anytime_rates(dataset)
    agent_a_call2 = next(
        row for row in rows if row.agent_id == "agent-a" and row.scientific_call_index == 2
    )
    agent_b_call4 = next(
        row for row in rows if row.agent_id == "agent-b" and row.scientific_call_index == 4
    )

    assert agent_a_call2.expected_trajectories == 18
    assert agent_a_call2.observed_turns == 18
    assert agent_a_call2.compiled_count == 17
    assert agent_a_call2.compiled_rate == 17 / 18
    assert agent_a_call2.observed_compiled_rate == 17 / 18
    assert agent_a_call2.formal_checkpoint is False
    assert agent_b_call4.infrastructure_censored == 1
    assert agent_b_call4.formal_checkpoint is True


def test_offline_qualification_pending_is_not_compiled_or_correct() -> None:
    turn = AnytimeTurnArtifact(
        scientific_call_index=1,
        cumulative_wall_seconds=0.02,
        candidate_stage="qualification_pending",
    )

    assert turn.compiled is False
    assert turn.correct is False
    assert turn.qualified is False
    assert turn.eligible is False


def test_pending_qualification_cannot_enter_non_synthetic_analysis() -> None:
    trajectory = _trajectory("agent-a", WORKLOADS[0], TARGETS[0], 1)
    turns = list(trajectory.turns)
    turns[0] = AnytimeTurnArtifact(
        scientific_call_index=1,
        cumulative_wall_seconds=10.0,
        candidate_stage="qualification_pending",
    )
    pending = trajectory.model_copy(update={"turns": tuple(turns)})
    with pytest.raises(ValueError, match="synthetic fixtures"):
        AnytimeAnalysisDataset(
            spec=_spec().model_copy(update={"study_stage": "formal"}),
            floors=_dataset().floors,
            trajectories=(pending,),
        )


def test_iteration_winners_cover_crossover_dominance_tie_and_model_dependence() -> None:
    report = build_anytime_analysis(_dataset())

    def winner(agent: str, workload: str, call: int):
        return next(
            row
            for row in report.winners
            if row.agent_id == agent
            and row.workload_id == workload
            and row.match_mode == "iteration"
            and row.iteration_budget == call
        )

    assert winner("agent-a", "reduce", 1).winner_target_ids == ("tilelang",)
    assert winner("agent-a", "reduce", 4).winner_target_ids == ("triton",)
    assert winner("agent-a", "gemm", 1).winner_target_ids == ("tilelang",)
    assert winner("agent-a", "gemm", 4).winner_target_ids == ("tilelang",)
    assert winner("agent-a", "fused", 4).status == "tie"
    assert winner("agent-a", "fused", 4).winner_target_ids == ("triton", "tilelang")
    assert winner("agent-b", "reduce", 4).winner_target_ids == ("tilelang",)

    disagreement = next(
        row
        for row in report.model_ranking_disagreements
        if row.workload_id == "reduce"
        and row.match_mode == "iteration"
        and row.iteration_budget == 4
    )
    assert disagreement.model_dependent is True


def test_matching_never_uses_future_turns_and_labels_exploratory_evidence() -> None:
    report = build_anytime_analysis(_dataset())
    exploratory = next(
        row
        for row in report.matches
        if row.agent_id == "agent-a"
        and row.workload_id == "reduce"
        and row.target_id == "triton"
        and row.replicate == 1
        and row.match_mode == "wall_clock"
        and row.wall_clock_budget_seconds == 25.0
        and row.evidence_scope == "formal_and_exploratory"
    )
    formal = next(
        row
        for row in report.matches
        if row.agent_id == "agent-a"
        and row.workload_id == "reduce"
        and row.target_id == "triton"
        and row.replicate == 1
        and row.match_mode == "wall_clock"
        and row.wall_clock_budget_seconds == 25.0
        and row.evidence_scope == "formal_only"
    )

    assert exploratory.selected_call_index == 2
    assert exploratory.selected_cumulative_wall_seconds == 20.0
    assert exploratory.measurement_kind == "exploratory_dev"
    assert formal.selected_call_index == 1
    assert formal.selected_cumulative_wall_seconds == 10.0
    assert formal.measurement_kind == "formal_checkpoint"
    assert formal.latency_ms == 8.0  # The faster call-3/call-4 values are future information.


def test_eager_speedup_bstar_qualification_and_hindsight_gain() -> None:
    report = build_anytime_analysis(_dataset())
    matched = next(
        row
        for row in report.matches
        if row.agent_id == "agent-a"
        and row.workload_id == "reduce"
        and row.target_id == "triton"
        and row.replicate == 1
        and row.match_mode == "iteration"
        and row.iteration_budget == 4
    )
    comparison = next(
        row
        for row in report.hindsight
        if row.agent_id == "agent-a" and row.match_mode == "iteration" and row.iteration_budget == 4
    )

    assert matched.eager_speedup == 5.0
    assert matched.bstar_relative_performance == 2.5
    assert matched.bstar_qualified is True
    assert comparison.status == "complete"
    assert comparison.oracle_over_fixed_gain is not None
    assert comparison.oracle_over_fixed_gain > 1.0
    assert comparison.clustered_gain_interval is not None
    interval = comparison.clustered_gain_interval
    assert interval.cluster_count == 4  # dense workloads share family×Agent×replicate clusters.
    assert interval.observation_count == 6
    assert interval.timing_trials_are_replicates is False
    assert interval.estimand == "workload_median_oracle_over_best_fixed_geomean_ratio"
    assert interval.point_estimate == pytest.approx(comparison.oracle_over_fixed_gain)


def test_hindsight_best_fixed_target_is_invariant_to_target_axis_order() -> None:
    original = _dataset()
    target_order = ("tilelang", "triton", "cute")
    spec = original.spec.model_copy(update={"targets": target_order})
    floors = tuple(
        next(
            floor
            for floor in original.floors
            if floor.workload_id == workload.workload_id and floor.target_id == target
        )
        for workload in spec.workloads
        for target in target_order
    )
    trajectories = tuple(
        next(
            trajectory
            for trajectory in original.trajectories
            if trajectory.agent_id == agent.agent_id
            and trajectory.workload_id == workload.workload_id
            and trajectory.target_id == target
            and trajectory.replicate == replicate
        )
        for agent in spec.agents
        for workload in spec.workloads
        for target in target_order
        for replicate in agent.replicates
    )
    reordered = AnytimeAnalysisDataset(
        spec=spec,
        floors=floors,
        trajectories=trajectories,
    )

    def final(report: AnytimeAnalysisReport):
        return next(
            row
            for row in report.hindsight
            if row.agent_id == "agent-a"
            and row.match_mode == "iteration"
            and row.iteration_budget == 4
        )

    before = final(build_anytime_analysis(original))
    after = final(build_anytime_analysis(reordered))
    assert after.best_fixed_target_id == before.best_fixed_target_id
    assert after.fixed_target_ids == before.fixed_target_ids
    assert after.oracle_over_fixed_gain == pytest.approx(before.oracle_over_fixed_gain)


def test_hindsight_cluster_interval_matches_point_with_heterogeneous_replicates() -> None:
    dataset = _dataset()
    trajectories = list(dataset.trajectories)
    index = next(
        index
        for index, row in enumerate(trajectories)
        if (row.agent_id, row.workload_id, row.target_id, row.replicate)
        == ("agent-a", "reduce", "triton", 2)
    )
    trajectories[index] = _trajectory(
        "agent-a",
        WORKLOADS[0],
        "triton",
        2,
        latency_transform=lambda call, latency: 1.6 if call == 4 else latency,
    )
    report = build_anytime_analysis(
        _replace_dataset(
            dataset,
            trajectories=[row.model_dump(mode="json") for row in trajectories],
        )
    )
    comparison = next(
        row
        for row in report.hindsight
        if row.agent_id == "agent-a" and row.match_mode == "iteration" and row.iteration_budget == 4
    )

    assert comparison.clustered_gain_interval is not None
    assert comparison.clustered_gain_interval.point_estimate == pytest.approx(
        comparison.oracle_over_fixed_gain
    )


def test_model_ranking_rows_retain_evidence_scope_in_their_unique_key() -> None:
    report = build_anytime_analysis(_dataset())
    keys = tuple(
        (
            row.workload_id,
            row.match_mode,
            row.iteration_budget,
            row.wall_clock_budget_seconds,
            row.evidence_scope,
        )
        for row in report.model_ranking_disagreements
    )
    wall_scopes = {
        row.evidence_scope
        for row in report.model_ranking_disagreements
        if row.workload_id == "reduce"
        and row.match_mode == "wall_clock"
        and row.wall_clock_budget_seconds == 25.0
    }

    assert len(keys) == len(set(keys))
    assert wall_scopes == {"formal_only", "formal_and_exploratory"}


@pytest.mark.parametrize(
    "floor_status,expected_winner_status",
    (
        ("invalid_floor", "invalid_floor"),
        ("unstable_timing", "unstable_floor_timing"),
        ("infrastructure_missing", "floor_infrastructure_missing"),
    ),
)
def test_floor_outcomes_fail_closed(floor_status: str, expected_winner_status: str) -> None:
    dataset = _dataset()
    floors = list(dataset.floors)
    floors[0] = _floor(WORKLOADS[0], TARGETS[0], status=floor_status)
    dataset = _replace_dataset(dataset, floors=[row.model_dump(mode="json") for row in floors])
    report = build_anytime_analysis(dataset)
    winner = next(
        row
        for row in report.winners
        if row.agent_id == "agent-a"
        and row.workload_id == "reduce"
        and row.match_mode == "iteration"
        and row.iteration_budget == 4
    )
    assert winner.status == expected_winner_status
    assert winner.winner_target_ids == ()


def test_infrastructure_missing_early_cap_and_replicate_disagreement_are_distinct() -> None:
    dataset = _dataset()
    trajectories = list(dataset.trajectories)
    infrastructure_key = ("agent-a", "reduce", "cute", 2)
    trajectories = [
        row
        for row in trajectories
        if (row.agent_id, row.workload_id, row.target_id, row.replicate) != infrastructure_key
    ]
    cap_index = next(
        index
        for index, row in enumerate(trajectories)
        if (row.agent_id, row.workload_id, row.target_id, row.replicate)
        == ("agent-a", "gemm", "cute", 2)
    )
    trajectories[cap_index] = _trajectory(
        "agent-a", WORKLOADS[1], "cute", 2, terminal_status="early_resource_cap", turn_count=2
    )
    # Replicate 2 reverses the final reduce winner while remaining a complete artifact.
    disagreement_index = next(
        index
        for index, row in enumerate(trajectories)
        if (row.agent_id, row.workload_id, row.target_id, row.replicate)
        == ("agent-b", "reduce", "triton", 2)
    )
    trajectories[disagreement_index] = _trajectory(
        "agent-b",
        WORKLOADS[0],
        "triton",
        2,
        latency_transform=lambda call, latency: 1.5 if call == 4 else latency,
    )
    dataset = _replace_dataset(
        dataset,
        trajectories=[row.model_dump(mode="json") for row in trajectories],
    )
    report = build_anytime_analysis(dataset)

    statuses = {
        (row.agent_id, row.workload_id): row.status
        for row in report.winners
        if row.match_mode == "iteration" and row.iteration_budget == 4
    }
    assert statuses[("agent-a", "reduce")] == "infrastructure_censored"
    assert statuses[("agent-a", "gemm")] == "early_resource_cap"
    assert statuses[("agent-b", "reduce")] == "replicate_disagreement"


def test_early_resource_cap_does_not_retroactively_censor_prior_budgets() -> None:
    dataset = _dataset()
    trajectories = list(dataset.trajectories)
    index = next(
        index
        for index, row in enumerate(trajectories)
        if (row.agent_id, row.workload_id, row.target_id, row.replicate)
        == ("agent-a", "gemm", "cute", 2)
    )
    trajectories[index] = _trajectory(
        "agent-a",
        WORKLOADS[1],
        "cute",
        2,
        terminal_status="early_resource_cap",
        turn_count=2,
    )
    report = build_anytime_analysis(
        _replace_dataset(
            dataset,
            trajectories=[row.model_dump(mode="json") for row in trajectories],
        )
    )

    call_one = next(
        row
        for row in report.winners
        if row.agent_id == "agent-a"
        and row.workload_id == "gemm"
        and row.match_mode == "iteration"
        and row.iteration_budget == 1
    )
    call_four = next(
        row
        for row in report.winners
        if row.agent_id == "agent-a"
        and row.workload_id == "gemm"
        and row.match_mode == "iteration"
        and row.iteration_budget == 4
    )
    assert call_one.status == "selected"
    assert call_four.status == "early_resource_cap"


def test_clustered_interval_uses_a_geometric_ratio_estimand() -> None:
    interval = clustered_interval(
        (
            (("dense", "agent-a", 1), 1.0),
            (("dense", "agent-a", 1), 4.0),
            (("reduction", "agent-a", 1), 1.0),
        ),
        seed=7,
        resamples=200,
        confidence_level=0.95,
    )

    assert interval.estimand == "geometric_mean_ratio"
    assert interval.point_estimate == pytest.approx(math.pow(4.0, 1.0 / 3.0))
    assert interval.lower <= interval.point_estimate <= interval.upper


def test_analysis_rejects_noncanonical_artifact_order() -> None:
    dataset = _dataset()
    forged = dataset.model_copy(update={"floors": tuple(reversed(dataset.floors))})
    with pytest.raises(AnytimeAnalysisError, match="canonical workload-target order"):
        build_anytime_analysis(forged)


def test_tables_are_deterministic_and_hash_bound() -> None:
    report = build_anytime_analysis(_dataset())
    first = anytime_analysis_tables(report)
    second = anytime_analysis_tables(report)

    assert first == second
    assert [table.name for table in first] == [
        "rates-by-turn",
        "workload-winners",
        "hindsight-oracle-gain",
        "missingness",
        "model-ranking-disagreement",
    ]
    ranking_table = next(table for table in first if table.name == "model-ranking-disagreement")
    assert "evidence_scope" in ranking_table.columns
    assert [table.sha256 for table in first] == [table.sha256 for table in second]


def test_figure_bundle_is_complete_labelled_and_visually_nonblank(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    bundle = write_anytime_analysis_bundle(_dataset(), output)
    verified = verify_anytime_analysis_bundle(output)
    figure_manifest = AnytimeFigureManifest.model_validate_json(
        (output / "figure-manifest.json").read_bytes()
    )

    assert verified == bundle
    assert len(figure_manifest.figures) == 7  # Two views per agent plus three summaries.
    assert len(bundle.files) == 33  # Report + 10 tables + figure manifest + 21 figures.
    assert figure_manifest.study_stage == "synthetic_fixture"
    assert (
        figure_manifest.renderer_environment.uv_lock_sha256
        == hashlib.sha256((Path(__file__).parents[1] / "uv.lock").read_bytes()).hexdigest()
    )
    assert len({style.color for style in figure_manifest.target_styles}) == len(TARGETS)
    assert len({style.marker for style in figure_manifest.target_styles}) == len(TARGETS)
    assert all("Synthetic fixture only" in figure.caption for figure in figure_manifest.figures)
    assert all("SYNTHETIC FIXTURE" in figure.title for figure in figure_manifest.figures)

    for artifact in figure_manifest.figures:
        png = output / f"figures/{artifact.figure_id}.png"
        svg = output / f"figures/{artifact.figure_id}.svg"
        pdf = output / f"figures/{artifact.figure_id}.pdf"
        with Image.open(png) as image:
            assert image.width >= 800
            assert image.height >= 600
            colors = image.convert("RGB").getcolors(maxcolors=image.width * image.height)
            assert colors is not None and len(colors) > 10
        svg_text = svg.read_text(encoding="utf-8")
        assert "<svg" in svg_text
        assert "SYNTHETIC FIXTURE" in svg_text
        assert pdf.read_bytes().startswith(b"%PDF")


def test_figure_bundle_is_byte_deterministic_and_exploratory_view_is_optional(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    formal_only = tmp_path / "formal-only"
    first_manifest = write_anytime_analysis_bundle(_dataset(), first)
    second_manifest = write_anytime_analysis_bundle(_dataset(), second)
    formal_manifest = write_anytime_analysis_bundle(
        _dataset(), formal_only, include_exploratory=False
    )

    assert first_manifest == second_manifest
    assert all(
        (first / item.relative_path).read_bytes() == (second / item.relative_path).read_bytes()
        for item in first_manifest.files
    )
    formal_figures = AnytimeFigureManifest.model_validate_json(
        (formal_only / "figure-manifest.json").read_bytes()
    ).figures
    assert len(formal_figures) == 5
    assert not any(figure.kind.startswith("exploratory") for figure in formal_figures)
    assert len(formal_manifest.files) == 27


def test_figure_bundle_refuses_overwrite_and_detects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    write_anytime_analysis_bundle(_dataset(), output)

    with pytest.raises(AnytimeFigureError, match="already exists"):
        write_anytime_analysis_bundle(_dataset(), output)

    image_path = output / "figures/formal-anytime-agent-a.png"
    image_path.write_bytes(image_path.read_bytes() + b"tampered")
    with pytest.raises(AnytimeFigureError, match="checksum mismatch"):
        verify_anytime_analysis_bundle(output)


def test_bundle_verifier_rejects_logical_rebinding_even_with_fresh_checksums(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    write_anytime_analysis_bundle(_dataset(), output)
    report_path = output / "analysis-report.json"
    bundle_path = output / "bundle-manifest.json"
    report = AnytimeAnalysisReport.model_validate_json(report_path.read_bytes()).model_copy(
        update={"input_dataset_sha256": "e" * 64}
    )
    report_bytes = _canonical_model_bytes(report)
    report_path.write_bytes(report_bytes)
    bundle = AnytimeAnalysisBundleManifest.model_validate_json(bundle_path.read_bytes())
    files = tuple(
        item.model_copy(
            update={
                "sha256": hashlib.sha256(report_bytes).hexdigest(),
                "size_bytes": len(report_bytes),
            }
        )
        if item.relative_path == "analysis-report.json"
        else item
        for item in bundle.files
    )
    forged = bundle.model_copy(
        update={
            "input_dataset_sha256": "e" * 64,
            "analysis_report_sha256": report.sha256,
            "files": files,
        }
    )
    bundle_path.write_bytes(_canonical_model_bytes(forged))

    with pytest.raises(AnytimeFigureError, match="bindings disagree|bound to another"):
        verify_anytime_analysis_bundle(output)


def test_hindsight_figure_caption_does_not_claim_an_observation_when_none_exists(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    floors = tuple(
        _floor(workload, target, status="invalid_floor")
        for workload in WORKLOADS
        for target in TARGETS
    )
    output = tmp_path / "missing"
    write_anytime_analysis_bundle(
        _replace_dataset(
            dataset,
            floors=[floor.model_dump(mode="json") for floor in floors],
        ),
        output,
    )
    manifest = AnytimeFigureManifest.model_validate_json(
        (output / "figure-manifest.json").read_bytes()
    )
    hindsight = next(
        figure for figure in manifest.figures if figure.kind == "formal_hindsight_gain"
    )

    assert "No formal checkpoint has complete hindsight evidence" in hindsight.caption
    assert "largest observed" not in hindsight.caption


def test_writer_preserves_callers_existing_matplotlib_figures(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    existing = plt.figure()
    try:
        write_anytime_analysis_bundle(_dataset(), tmp_path / "bundle")
        assert plt.fignum_exists(existing.number)
    finally:
        plt.close(existing)
