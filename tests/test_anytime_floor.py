from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from abstrak.anytime.floor import (
    AnytimeBaselineFloorEvidence,
    AnytimeExpertFloorEvidence,
    AnytimeFloorEvidenceBundle,
    AnytimeInvalidFloorError,
    AnytimeObservedEnvironment,
    AnytimeRawTimingEvidence,
    build_anytime_b_star,
    validate_anytime_floor_evidence,
)
from abstrak.anytime.workloads import (
    ENVIRONMENT_ASSET_PATHS,
    AnytimeBaselineSourceInput,
    AnytimeExpertSourceInput,
    AnytimeWorkloadInputManifest,
    PinnedAnytimeWorkloadInputs,
    load_anytime_workload_inputs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TARGET_CARD_PATHS = tuple(
    f"benchmarks/r1-a100/targets/{backend}.md" for backend in ("triton", "tilelang", "cute")
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _mirror_repository_inputs(root: Path) -> None:
    for relative_path in (*TARGET_CARD_PATHS, *ENVIRONMENT_ASSET_PATHS.values()):
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / relative_path).read_bytes())


def _write_pinned_manifest(
    root: Path,
    manifest: AnytimeWorkloadInputManifest,
) -> PinnedAnytimeWorkloadInputs:
    payload = (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    path = root / "manifests" / "synthetic-formal-inputs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return load_anytime_workload_inputs(
        path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _synthetic_formal_inputs(tmp_path: Path) -> PinnedAnytimeWorkloadInputs:
    """Materialize static fixture bytes only; no source is imported or executed."""

    _mirror_repository_inputs(tmp_path)
    default = load_anytime_workload_inputs().manifest
    experts: list[AnytimeExpertSourceInput] = []
    for item in default.experts:
        payload = f"# synthetic expert fixture: {item.task_id}/{item.target_id}\n".encode()
        path = tmp_path / item.source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        experts.append(
            AnytimeExpertSourceInput(
                task_id=item.task_id,
                target_id=item.target_id,
                source_path=item.source_path,
                status="validated_m9",
                expected_source_sha256=hashlib.sha256(payload).hexdigest(),
                correctness_observation="passed-m9",
                target_launch_observation="passed-m9",
                timing_observation="stable-m9",
                formal_ready=True,
            )
        )
    baselines: list[AnytimeBaselineSourceInput] = []
    for item in default.baselines:
        payload = f"# synthetic baseline fixture: {item.task_id}/{item.variant}\n".encode()
        path = tmp_path / item.source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        baselines.append(
            AnytimeBaselineSourceInput(
                task_id=item.task_id,
                variant=item.variant,
                source_path=item.source_path,
                status="validated_m9",
                expected_source_sha256=hashlib.sha256(payload).hexdigest(),
                applicability_observation="passed-m9",
                correctness_observation="passed-m9",
                timing_observation="stable-m9",
                formal_ready=True,
            )
        )
    workloads = tuple(
        item.model_copy(
            update={
                "resource_feasibility_status": "validated-m9",
                "resource_feasibility_artifact_sha256": _digest(f"resource-feasibility/{item.id}"),
            }
        )
        for item in default.workloads
    )
    manifest = AnytimeWorkloadInputManifest(
        workloads=workloads,
        target_cards=default.target_cards,
        experts=tuple(experts),
        baselines=tuple(baselines),
        environment=default.environment,
        floor_policy=default.floor_policy,
    )
    return _write_pinned_manifest(tmp_path, manifest)


def _environment(manifest: AnytimeWorkloadInputManifest) -> AnytimeObservedEnvironment:
    expected = manifest.environment
    return AnytimeObservedEnvironment(
        environment_contract_sha256=expected.sha256,
        observation_artifact_sha256=_digest("environment-observation"),
        controller_revision="1" * 40,
        worker_revision="1" * 40,
        accelerator=expected.accelerator,
        compute_capability=expected.compute_capability,
        python_version=expected.python_version,
        torch_version=expected.torch_version,
        cuda_runtime_version=expected.cuda_runtime_version,
        driver_version=expected.minimum_driver_version,
        triton_version=expected.triton_version,
        tilelang_version=expected.tilelang_version,
        cute_cutlass_dsl_version=expected.cute_cutlass_dsl_version,
        cuda_python_version=expected.cuda_python_version,
        cuda_bindings_version=expected.cuda_bindings_version,
        kernelbench_revision=expected.kernelbench_revision,
        worker_bootstrap_sha256=expected.worker_bootstrap_sha256,
        worker_update_sha256=expected.worker_update_sha256,
        isolation_mode=expected.isolation_mode,
        isolation_contract_sha256=expected.isolation_contract_sha256,
        lock_sha256=expected.lock_sha256,
        wheelhouse_archive_sha256=expected.wheelhouse_archive_sha256,
    )


def _stable_timing(median_ms: float) -> AnytimeRawTimingEvidence:
    return AnytimeRawTimingEvidence(
        block_medians_ms=(median_ms * 0.99, median_ms, median_ms * 1.01)
    )


def _complete_bundle(
    pinned: PinnedAnytimeWorkloadInputs,
    *,
    vendor_applicable: bool = True,
) -> AnytimeFloorEvidenceBundle:
    manifest = pinned.manifest
    environment = _environment(manifest)
    experts: list[AnytimeExpertFloorEvidence] = []
    for task_index, workload in enumerate(manifest.workloads):
        for target_index, target_id in enumerate(("triton-a100", "tilelang-a100", "cute-a100")):
            source = manifest.expert(workload.id, target_id)
            experts.append(
                AnytimeExpertFloorEvidence(
                    task_id=workload.id,
                    target_id=target_id,
                    input_manifest_sha256=manifest.sha256,
                    workload_pack_sha256=workload.sha256,
                    target_card_input_sha256=manifest.target_card(target_id).sha256,
                    expert_source_input_sha256=source.sha256,
                    observed_expert_source_sha256=(
                        source.expected_source_sha256
                        or _digest(f"pending-expert/{workload.id}/{target_id}")
                    ),
                    environment_contract_sha256=manifest.environment.sha256,
                    environment_observation_sha256=environment.sha256,
                    correctness_artifact_sha256=_digest(
                        f"expert-correctness/{workload.id}/{target_id}"
                    ),
                    launch_artifact_sha256=_digest(f"expert-launch/{workload.id}/{target_id}"),
                    timing_artifact_sha256=_digest(f"expert-timing/{workload.id}/{target_id}"),
                    compiled=True,
                    all_sealed_cases_passed=True,
                    output_finite=True,
                    inputs_unchanged=True,
                    fallback_free=True,
                    target_launch_verified=True,
                    timing=_stable_timing(10.0 + task_index + target_index),
                )
            )
    baselines: list[AnytimeBaselineFloorEvidence] = []
    variant_latency = {"eager": 8.0, "inductor": 6.0, "vendor": 7.0}
    for task_index, workload in enumerate(manifest.workloads):
        for variant in ("eager", "inductor", "vendor"):
            source = manifest.baseline(workload.id, variant)
            applicable = variant != "vendor" or vendor_applicable
            common = dict(
                task_id=workload.id,
                variant=variant,
                input_manifest_sha256=manifest.sha256,
                workload_pack_sha256=workload.sha256,
                baseline_source_input_sha256=source.sha256,
                observed_baseline_source_sha256=(
                    source.expected_source_sha256
                    or _digest(f"pending-baseline/{workload.id}/{variant}")
                ),
                environment_contract_sha256=manifest.environment.sha256,
                environment_observation_sha256=environment.sha256,
                applicability_artifact_sha256=_digest(
                    f"baseline-applicability/{workload.id}/{variant}"
                ),
                applicable=applicable,
            )
            if applicable:
                baselines.append(
                    AnytimeBaselineFloorEvidence(
                        **common,
                        correctness_artifact_sha256=_digest(
                            f"baseline-correctness/{workload.id}/{variant}"
                        ),
                        timing_artifact_sha256=_digest(f"baseline-timing/{workload.id}/{variant}"),
                        correct=True,
                        output_finite=True,
                        inputs_unchanged=True,
                        timing=_stable_timing(variant_latency[variant] + task_index),
                    )
                )
            else:
                baselines.append(AnytimeBaselineFloorEvidence(**common))
    return AnytimeFloorEvidenceBundle(
        input_manifest_sha256=manifest.sha256,
        environment=environment,
        experts=tuple(experts),
        baselines=tuple(baselines),
    )


def test_missing_floor_evidence_fails_closed() -> None:
    pinned = load_anytime_workload_inputs()
    bundle = AnytimeFloorEvidenceBundle(input_manifest_sha256=pinned.manifest.sha256)

    validation = validate_anytime_floor_evidence(pinned, bundle)

    assert validation.status == "invalid_floor"
    assert "missing environment observation" in validation.reasons
    assert any("expert evidence coverage mismatch" in reason for reason in validation.reasons)
    with pytest.raises(AnytimeInvalidFloorError, match="invalid_floor"):
        build_anytime_b_star(pinned, bundle)


def test_complete_looking_evidence_cannot_activate_pending_source_slots() -> None:
    pinned = load_anytime_workload_inputs()
    validation = validate_anytime_floor_evidence(pinned, _complete_bundle(pinned))

    assert validation.status == "invalid_floor"
    assert sum("source is not M9-validated" in reason for reason in validation.reasons) == 72


def test_synthetic_formal_floor_builds_b_star_and_excludes_inapplicable_vendor(
    tmp_path: Path,
) -> None:
    pinned = _synthetic_formal_inputs(tmp_path)
    bundle = _complete_bundle(pinned, vendor_applicable=False)

    validation = validate_anytime_floor_evidence(
        pinned,
        bundle,
        repository_root=str(tmp_path),
    )
    floor = build_anytime_b_star(
        pinned,
        bundle,
        repository_root=str(tmp_path),
    )

    assert validation.status == "valid_floor"
    assert not validation.reasons
    assert len(floor.target_expert_floors) == 36
    assert len(floor.b_star) == 12
    assert {item.selected_variant for item in floor.b_star} == {"inductor"}


def test_unstable_or_hash_mismatched_evidence_is_invalid_floor(tmp_path: Path) -> None:
    pinned = _synthetic_formal_inputs(tmp_path)
    bundle = _complete_bundle(pinned)
    unstable_expert = bundle.experts[0].model_copy(
        update={"timing": AnytimeRawTimingEvidence(block_medians_ms=(1.0, 2.0, 3.0))}
    )
    unstable = bundle.model_copy(update={"experts": (unstable_expert, *bundle.experts[1:])})
    validation = validate_anytime_floor_evidence(
        pinned,
        unstable,
        repository_root=str(tmp_path),
    )
    assert validation.status == "invalid_floor"
    assert any("unstable timing" in reason for reason in validation.reasons)

    mismatched_expert = bundle.experts[0].model_copy(update={"workload_pack_sha256": "0" * 64})
    mismatched = bundle.model_copy(update={"experts": (mismatched_expert, *bundle.experts[1:])})
    validation = validate_anytime_floor_evidence(
        pinned,
        mismatched,
        repository_root=str(tmp_path),
    )
    assert validation.status == "invalid_floor"
    assert any("expert hash mismatch" in reason for reason in validation.reasons)


def test_duplicate_artifact_digests_and_no_applicable_baseline_are_rejected(
    tmp_path: Path,
) -> None:
    pinned = _synthetic_formal_inputs(tmp_path)
    bundle = _complete_bundle(pinned)
    duplicate = bundle.baselines[0].model_copy(
        update={"applicability_artifact_sha256": bundle.experts[0].correctness_artifact_sha256}
    )
    duplicated = bundle.model_copy(update={"baselines": (duplicate, *bundle.baselines[1:])})
    validation = validate_anytime_floor_evidence(
        pinned,
        duplicated,
        repository_root=str(tmp_path),
    )
    assert validation.status == "invalid_floor"
    assert "floor gate artifact digests must be globally unique" in validation.reasons

    first_task = pinned.manifest.workloads[0].id
    no_baseline = tuple(
        item.model_copy(
            update={
                "applicable": False,
                "correctness_artifact_sha256": None,
                "timing_artifact_sha256": None,
                "correct": None,
                "output_finite": None,
                "inputs_unchanged": None,
                "timing": None,
            }
        )
        if item.task_id == first_task
        else item
        for item in bundle.baselines
    )
    validation = validate_anytime_floor_evidence(
        pinned,
        bundle.model_copy(update={"baselines": no_baseline}),
        repository_root=str(tmp_path),
    )
    assert validation.status == "invalid_floor"
    assert f"no applicable common baseline: {first_task}" in validation.reasons


def test_baseline_applicability_contract_rejects_partial_results() -> None:
    common = dict(
        task_id="l1-2-standard-matmul",
        variant="vendor",
        input_manifest_sha256="1" * 64,
        workload_pack_sha256="2" * 64,
        baseline_source_input_sha256="3" * 64,
        observed_baseline_source_sha256="4" * 64,
        environment_contract_sha256="5" * 64,
        environment_observation_sha256="6" * 64,
        applicability_artifact_sha256="7" * 64,
    )
    with pytest.raises(ValidationError, match="applicable baseline evidence is incomplete"):
        AnytimeBaselineFloorEvidence(**common, applicable=True)
    with pytest.raises(ValidationError, match="only applicability evidence"):
        AnytimeBaselineFloorEvidence(
            **common,
            applicable=False,
            correct=True,
        )


def test_malformed_nonfinite_bundle_returns_invalid_floor() -> None:
    pinned = load_anytime_workload_inputs()
    validation = validate_anytime_floor_evidence(
        pinned,
        {
            "schema_version": "abstrak-anytime-floor-evidence-bundle.v1",
            "input_manifest_sha256": pinned.manifest.sha256,
            "experts": [{"timing": {"block_medians_ms": [float("nan")]}}],
            "baselines": [],
        },
    )
    assert validation.status == "invalid_floor"
    assert "malformed evidence" in validation.reasons[0]
