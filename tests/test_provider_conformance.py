from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import ScriptedTransport
from pydantic import ValidationError

from abstrak.providers.artifacts import ArtifactError, ProviderArtifactStore
from abstrak.providers.cli import EXIT_CONFIG, EXIT_OK, main
from abstrak.providers.client import ProviderClient
from abstrak.providers.conformance import (
    LiveProbeConfigurationError,
    build_probe_request,
    run_live_probe,
)
from abstrak.providers.contracts import ConformanceReport
from abstrak.providers.manifests import ManifestBundle, ModelManifest


def test_probe_nonce_is_visible_only_through_system_message() -> None:
    request, nonce = build_probe_request("model", nonce="system-only-nonce")

    assert nonce in request.messages[0].content
    assert nonce not in request.messages[1].content


def test_live_probe_writes_complete_private_artifact(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
    provider_environment: dict[str, str],
    valid_response: dict[str, Any],
) -> None:
    transport = ScriptedTransport(response=valid_response)
    client = ProviderClient(manifest_bundle, transport=transport, environment=provider_environment)

    report, store = run_live_probe(client, artifact_root=str(tmp_path), nonce="fixed-nonce")

    assert report.status == "pass"
    assert store is not None
    store.verify_no_secrets(client.artifact_secrets)
    assert stat.S_IMODE(store.run_directory.stat().st_mode) == 0o500
    assert {path.name for path in store.run_directory.iterdir()} == {
        "events.jsonl",
        "manifest.resolved.json",
        "request.logical.json",
        "request.transport.json",
        "response.normalized.json",
        "response.sdk.json",
        "result.json",
        "sha256sums.txt",
    }
    artifact_bytes = b"".join(
        path.read_bytes() for path in store.run_directory.iterdir() if path.is_file()
    )
    assert provider_environment["TEST_API_KEY"].encode() not in artifact_bytes
    assert provider_environment["TEST_BASE_URL"].encode() not in artifact_bytes
    store.verify_checksums()
    with pytest.raises(ArtifactError, match="finalized"):
        store.write_json("late.json", {})


def test_invalid_action_is_separate_from_transport_readiness(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
    provider_environment: dict[str, str],
    valid_response: dict[str, Any],
) -> None:
    invalid = json.loads(json.dumps(valid_response))
    invalid["choices"][0]["message"]["content"] = '{"action":"finish","nonce":"wrong"}'
    client = ProviderClient(
        manifest_bundle,
        transport=ScriptedTransport(response=invalid),
        environment=provider_environment,
    )

    report, _ = run_live_probe(client, artifact_root=str(tmp_path), nonce="fixed-nonce")

    checks = {check.name: check.status for check in report.checks}
    assert report.status == "fail"
    assert checks["nonempty_text"] == "pass"
    assert checks["plain_json_action"] == "pass"
    assert checks["system_nonce_fidelity"] == "fail"


def test_validate_cli_is_offline(capsys) -> None:
    exit_code = main(
        [
            "validate",
            "--provider",
            "configs/examples/provider.openai-compatible.example.yaml",
            "--model",
            "configs/examples/model.openai-compatible.example.yaml",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == EXIT_OK
    assert output["status"] == "valid"
    assert output["required_environment"] == [
        "ABSTRAK_EXAMPLE_API_KEY",
        "ABSTRAK_EXAMPLE_BASE_URL",
    ]


def test_example_manifest_cannot_run_live(capsys) -> None:
    exit_code = main(
        [
            "smoke",
            "--live",
            "--provider",
            "configs/examples/provider.openai-compatible.example.yaml",
            "--model",
            "configs/examples/model.openai-compatible.example.yaml",
        ]
    )

    assert exit_code == EXIT_CONFIG
    assert "allow_live_probe" in capsys.readouterr().err


def test_live_smoke_missing_secrets_is_configuration_failure(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.delenv("ABSTRAK_EXAMPLE_API_KEY", raising=False)
    monkeypatch.delenv("ABSTRAK_EXAMPLE_BASE_URL", raising=False)
    model_payload = yaml.safe_load(
        Path("configs/examples/model.openai-compatible.example.yaml").read_text()
    )
    model_payload["allow_live_probe"] = True
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(model_payload), encoding="utf-8")

    exit_code = main(
        [
            "smoke",
            "--live",
            "--provider",
            "configs/examples/provider.openai-compatible.example.yaml",
            "--model",
            str(model_path),
        ]
    )

    assert exit_code == EXIT_CONFIG
    assert "missing required environment variables" in capsys.readouterr().err


@pytest.mark.parametrize("missing_field", ["prompt_tokens", "completion_tokens", "total_tokens"])
def test_partial_core_usage_is_not_conformant(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
    provider_environment: dict[str, str],
    valid_response: dict[str, Any],
    missing_field: str,
) -> None:
    partial = json.loads(json.dumps(valid_response))
    del partial["usage"][missing_field]
    client = ProviderClient(
        manifest_bundle,
        transport=ScriptedTransport(response=partial),
        environment=provider_environment,
    )

    report, _ = run_live_probe(client, artifact_root=str(tmp_path), nonce="fixed-nonce")

    checks = {check.name: check.status for check in report.checks}
    assert report.status == "fail"
    assert report.transport_ready is False
    assert checks["usage_reporting"] == "fail"


def test_mutable_alias_can_pass_endpoint_check_but_not_pilot_readiness(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
    provider_environment: dict[str, str],
    valid_response: dict[str, Any],
) -> None:
    model_payload = manifest_bundle.model.model_dump(mode="json")
    model_payload["model_id_policy"] = "mutable_alias"
    model_payload["expected_returned_model"] = None
    bundle = ManifestBundle(
        provider=manifest_bundle.provider,
        model=ModelManifest.model_validate(model_payload),
    )
    aliased = json.loads(json.dumps(valid_response))
    aliased["model"] = "rolling-model-revision"
    client = ProviderClient(
        bundle,
        transport=ScriptedTransport(response=aliased),
        environment=provider_environment,
    )

    report, _ = run_live_probe(client, artifact_root=str(tmp_path), nonce="fixed-nonce")

    assert report.status == "pass"
    assert report.transport_ready is True
    assert report.action_protocol_ready is True
    assert report.pilot_ready is False


def test_exact_model_identity_mismatch_fails_conformance(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
    provider_environment: dict[str, str],
    valid_response: dict[str, Any],
) -> None:
    mismatched = json.loads(json.dumps(valid_response))
    mismatched["model"] = "unexpected-model"
    client = ProviderClient(
        manifest_bundle,
        transport=ScriptedTransport(response=mismatched),
        environment=provider_environment,
    )

    report, _ = run_live_probe(client, artifact_root=str(tmp_path), nonce="fixed-nonce")

    checks = {check.name: check.status for check in report.checks}
    assert report.status == "fail"
    assert report.pilot_ready is False
    assert checks["returned_model_exact"] == "fail"


def test_library_live_probe_enforces_manifest_guard(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
    provider_environment: dict[str, str],
    valid_response: dict[str, Any],
) -> None:
    model_payload = manifest_bundle.model.model_dump(mode="json")
    model_payload["allow_live_probe"] = False
    bundle = ManifestBundle(
        provider=manifest_bundle.provider,
        model=ModelManifest.model_validate(model_payload),
    )
    transport = ScriptedTransport(response=valid_response)
    client = ProviderClient(bundle, transport=transport, environment=provider_environment)

    with pytest.raises(LiveProbeConfigurationError, match="allow_live_probe"):
        run_live_probe(client, artifact_root=str(tmp_path), nonce="fixed-nonce")

    assert transport.call_count == 0
    assert not any(tmp_path.iterdir())


def test_conformance_report_rejects_pass_with_failed_check(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
    provider_environment: dict[str, str],
    valid_response: dict[str, Any],
) -> None:
    client = ProviderClient(
        manifest_bundle,
        transport=ScriptedTransport(response=valid_response),
        environment=provider_environment,
    )
    report, _ = run_live_probe(client, artifact_root=str(tmp_path), nonce="fixed-nonce")
    payload = report.model_dump(mode="json")
    payload["checks"][0]["status"] = "fail"

    with pytest.raises(ValidationError, match="passing report"):
        ConformanceReport.model_validate(payload)


def test_invalid_environment_base_url_uses_configuration_exit(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    model_payload = yaml.safe_load(
        Path("configs/examples/model.openai-compatible.example.yaml").read_text()
    )
    model_payload["allow_live_probe"] = True
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(model_payload), encoding="utf-8")
    monkeypatch.setenv("ABSTRAK_EXAMPLE_API_KEY", "test-secret")
    monkeypatch.setenv("ABSTRAK_EXAMPLE_BASE_URL", "http://remote.example/v1")

    exit_code = main(
        [
            "smoke",
            "--live",
            "--provider",
            "configs/examples/provider.openai-compatible.example.yaml",
            "--model",
            str(model_path),
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ]
    )

    assert exit_code == EXIT_CONFIG
    assert "must use HTTPS" in capsys.readouterr().err


class AuthenticationError(Exception):
    pass


def test_failed_probe_also_produces_a_sealed_terminal_bundle(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
    provider_environment: dict[str, str],
) -> None:
    client = ProviderClient(
        manifest_bundle,
        transport=ScriptedTransport(error=AuthenticationError("denied")),
        environment=provider_environment,
    )

    report, store = run_live_probe(client, artifact_root=str(tmp_path), nonce="fixed-nonce")

    assert report.status == "fail"
    assert report.error is not None
    assert stat.S_IMODE(store.run_directory.stat().st_mode) == 0o500
    assert "response.error.json" in {path.name for path in store.run_directory.iterdir()}
    store.verify_checksums()


def test_artifact_store_rejects_secrets_before_writing(tmp_path: Path) -> None:
    store = ProviderArtifactStore.create(tmp_path, "provider", "model", secrets=("do-not-write",))

    with pytest.raises(ArtifactError, match="credential material"):
        store.write_json("unsafe.json", {"value": "do-not-write"})

    assert not (store.run_directory / "unsafe.json").exists()


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    ],
)
def test_zero_usage_is_preserved_but_fails_plausibility(
    tmp_path: Path,
    manifest_bundle: ManifestBundle,
    provider_environment: dict[str, str],
    valid_response: dict[str, Any],
    usage: dict[str, int],
) -> None:
    zero_response = json.loads(json.dumps(valid_response))
    zero_response["usage"] = usage
    client = ProviderClient(
        manifest_bundle,
        transport=ScriptedTransport(response=zero_response),
        environment=provider_environment,
    )

    report, store = run_live_probe(client, artifact_root=str(tmp_path), nonce="fixed-nonce")

    checks = {check.name: check.status for check in report.checks}
    assert report.status == "fail"
    assert report.response is not None
    assert report.response.usage.output_tokens == 0
    assert report.response.usage.provider_reported is True
    assert report.response.usage.core_fields_complete is True
    assert checks["usage_plausibility"] == "fail"
    store.verify_checksums()


def test_artifact_store_writes_all_bytes_after_short_writes(tmp_path: Path, monkeypatch) -> None:
    real_write = os.write

    def short_write(descriptor: int, content: bytes | memoryview) -> int:
        return real_write(descriptor, bytes(content[:3]))

    monkeypatch.setattr(os, "write", short_write)
    store = ProviderArtifactStore.create(tmp_path, "provider", "model")
    store.write_json("record.json", {"value": "complete-record"})
    store.append_event({"event": "complete-event"})
    store.finalize()

    assert json.loads((store.run_directory / "record.json").read_text())["value"] == (
        "complete-record"
    )
    assert json.loads((store.run_directory / "events.jsonl").read_text())["event"] == (
        "complete-event"
    )
    store.verify_checksums()
