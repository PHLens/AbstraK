from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from abstrak.anytime.artifacts import AnytimeArtifactError
from abstrak.anytime.figures import AnytimeFigureManifest
from abstrak.anytime.freeze import (
    DEFAULT_FREEZE_DIRECTORY,
    DEFAULT_REPOSITORY_ROOT,
    load_anytime_offline_freeze,
    write_anytime_freeze_manifests,
)
from abstrak.anytime.prompts import (
    AnytimeBasePromptPolicy,
    build_anytime_base_prompt_policy,
    render_anytime_base_prompt,
)
from abstrak.anytime.rehearsal import (
    FAKE_PROVIDER_WARNING,
    REHEARSAL_MANIFEST_FILENAME,
    AnytimeOfflineQualificationFixture,
    AnytimeOfflineRehearsalFile,
    AnytimeOfflineRehearsalManifest,
    AnytimeRehearsalError,
    run_anytime_offline_rehearsal,
    verify_anytime_offline_rehearsal,
)
from abstrak.anytime.workloads import (
    PINNED_INPUT_MANIFEST_SHA256,
    load_anytime_workload_inputs,
)
from abstrak.providers.contracts import sha256_json
from abstrak.providers.native_contracts import NativeNormalizedResponse


def _canonical_bytes(value: object) -> bytes:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )


def _sequence_sha256(values: tuple[object, ...]) -> str:
    payload = tuple(value.model_dump(mode="json") for value in values)
    return sha256_json(payload)


@pytest.fixture(scope="module")
def rehearsal_bundle(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("anytime-offline-rehearsal")
    frozen = write_anytime_freeze_manifests(root / "freeze")
    pinned = load_anytime_offline_freeze(
        root / "freeze" / "offline-freeze.json",
        expected_sha256=frozen.freeze_raw_sha256,
    )
    result = run_anytime_offline_rehearsal(
        root / "rehearsal",
        pinned_freeze=pinned,
        repository_root=DEFAULT_REPOSITORY_ROOT,
    )
    return root, pinned, result


def test_base_prompt_is_deterministic_and_uses_only_frozen_public_inputs() -> None:
    inputs = load_anytime_workload_inputs(
        expected_sha256=PINNED_INPUT_MANIFEST_SHA256
    )
    policy = build_anytime_base_prompt_policy()
    first = render_anytime_base_prompt(
        inputs,
        task_id="l1-2-standard-matmul",
        target_id="triton-a100",
        policy=policy,
    )
    second = render_anytime_base_prompt(
        inputs,
        task_id="l1-2-standard-matmul",
        target_id="triton-a100",
        policy=policy,
    )

    assert first == second
    assert tuple(message.role.value for message in first) == ("system", "user")
    assert "PUBLIC WORKLOAD JSON" in first[1].content
    assert "SELECTED TARGET CARD: triton-a100" in first[1].content
    assert "one fenced Python source block" in first[1].content
    workload = inputs.manifest.workload("l1-2-standard-matmul")
    assert workload.lineage.source_path not in first[1].content
    assert all(case.id not in first[1].content for case in workload.generator.sealed_cases)
    assert all(
        expert.source_path not in first[1].content
        for expert in inputs.manifest.experts
        if expert.task_id == workload.id
    )

    with pytest.raises(ValidationError, match="study boundary"):
        AnytimeBasePromptPolicy(system_message="Optimize one kernel.")


def test_full_scripted_shakeout_rehearsal_is_closed_and_explicitly_non_live(
    rehearsal_bundle,
) -> None:
    _, pinned, result = rehearsal_bundle
    receipt = result.receipt

    assert receipt.planned_trajectories == 48
    assert receipt.scientific_model_call_ceiling == 192
    assert receipt.operational_provider_request_ceiling == 384
    assert receipt.completed_trajectories == 48
    assert receipt.attempt_count == 49
    assert receipt.infrastructure_retry_count == 1
    assert receipt.scripted_provider_response_count == 192
    assert receipt.scripted_worker_artifact_count == 192
    assert receipt.checkpoint_count == 96
    assert receipt.provider_protocol_counts == (
        ("chat_completions", 96),
        ("responses", 96),
    )
    assert receipt.candidate_outcome_counts == (("ineligible", 192),)
    assert receipt.terminal_counts == (("infrastructure_failure", 1), ("success", 48))
    assert sum(phase.attempt_count == 3 for phase in receipt.phase_receipts) == 1
    assert receipt.floor_gate == "invalid_floor"
    assert receipt.formal_readiness == "blocked"
    assert receipt.environment_status == "pending-m9"
    assert receipt.next_gate == "m9-trusted-gpu-preflight"
    side_effects = receipt.side_effects.model_dump(mode="python", exclude={"schema_version"})
    assert not any(side_effects.values())
    assert (
        verify_anytime_offline_rehearsal(
            result.directory,
            pinned_freeze=pinned,
            repository_root=DEFAULT_REPOSITORY_ROOT,
        )
        == result.manifest
    )

    qualifications = sorted(result.directory.glob("qualifications/*/call-*.json"))
    assert len(qualifications) == 12
    assert all(
        AnytimeOfflineQualificationFixture.model_validate_json(path.read_bytes()).decision.status
        == "pending-m9"
        for path in qualifications
    )
    responses = sorted(result.directory.glob("source/**/provider-native-response.json"))
    assert len(responses) == 192
    assert all(
        NativeNormalizedResponse.model_validate_json(path.read_bytes()).warnings
        == (FAKE_PROVIDER_WARNING,)
        for path in responses
    )
    figures = AnytimeFigureManifest.model_validate_json(
        (result.directory / "analysis" / "figure-manifest.json").read_bytes()
    )
    assert len(figures.figures) == 7
    assert all("Synthetic fixture only" in figure.caption for figure in figures.figures)


def test_rehearsal_rejects_inner_attempt_tamper_even_after_outer_rehash(
    rehearsal_bundle,
    tmp_path: Path,
) -> None:
    _, pinned, result = rehearsal_bundle
    tampered = tmp_path / "tampered"
    shutil.copytree(result.directory, tampered)
    response_path = next(tampered.glob("source/**/provider-native-response.json"))
    response_path.chmod(0o600)
    payload = json.loads(response_path.read_text())
    payload["raw_transport_response"]["origin"] = "forged-live-observation"
    response_payload = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    response_path.write_bytes(response_payload)

    manifest_path = tampered / REHEARSAL_MANIFEST_FILENAME
    manifest_path.chmod(0o600)
    manifest = AnytimeOfflineRehearsalManifest.model_validate_json(manifest_path.read_bytes())
    relative = response_path.relative_to(tampered).as_posix()
    files = tuple(
        AnytimeOfflineRehearsalFile(
            relative_path=item.relative_path,
            role=item.role,
            raw_sha256=(
                hashlib.sha256(response_payload).hexdigest()
                if item.relative_path == relative
                else item.raw_sha256
            ),
            size_bytes=(
                len(response_payload)
                if item.relative_path == relative
                else item.size_bytes
            ),
        )
        for item in manifest.files
    )
    rehashed = AnytimeOfflineRehearsalManifest(
        receipt_raw_sha256=manifest.receipt_raw_sha256,
        receipt_sha256=manifest.receipt_sha256,
        files=files,
        file_bundle_sha256=_sequence_sha256(files),
    )
    manifest_path.write_bytes(_canonical_bytes(rehashed))

    with pytest.raises((AnytimeRehearsalError, AnytimeArtifactError)):
        verify_anytime_offline_rehearsal(
            tampered,
            pinned_freeze=pinned,
            repository_root=DEFAULT_REPOSITORY_ROOT,
        )


def test_offline_cli_prints_ceilings_before_non_authorization_and_has_no_live_flags() -> None:
    script = DEFAULT_REPOSITORY_ROOT / "scripts" / "freeze_anytime_offline.py"
    help_result = subprocess.run(
        (sys.executable, str(script), "--help"),
        cwd=DEFAULT_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for forbidden in ("--api-key", "--endpoint", "--ssh", "--gpu"):
        assert forbidden not in help_result.stdout

    result = subprocess.run(
        (
            sys.executable,
            str(script),
            "--check",
            "--output-directory",
            str(DEFAULT_FREEZE_DIRECTORY),
        ),
        cwd=DEFAULT_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    assert "formal_scientific_model_call_ceiling=2376" in lines
    assert "formal_operational_provider_request_ceiling=4752" in lines
    assert "shakeout_scientific_model_call_ceiling=192" in lines
    assert "shakeout_operational_provider_request_ceiling=384" in lines
    assert lines.index("formal_scientific_model_call_ceiling=2376") < lines.index(
        "authorization_emitted=false"
    )
    assert "formal_authorized=false" in lines
    assert "freeze_status=verified" in lines
    assert "live_worker_revision=pending_m9" in lines
