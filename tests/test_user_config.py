from __future__ import annotations

import json
import traceback
from pathlib import Path

import pytest
import yaml

from abstrak.config import (
    AuthStore,
    ConfigurationError,
    load_app_config,
    load_auth_store,
    runtime_environment,
)
from abstrak.providers.cli import EXIT_CONFIG, EXIT_OK, main
from abstrak.providers.manifests import ModelManifest, ProviderManifest


def _write_config(
    home: Path,
    provider: ProviderManifest,
    models: tuple[ModelManifest, ...],
) -> Path:
    config_directory = home / ".abstrak"
    config_directory.mkdir(mode=0o700, exist_ok=True)
    config_path = config_directory / "config.yaml"
    profiles = {
        model.id: {
            "provider": provider.model_dump(mode="json"),
            "model": model.model_dump(mode="json"),
        }
        for model in models
    }
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "config.v1",
                "default_profile": models[0].id,
                "profiles": profiles,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _write_auth(path: Path, key: str, base_url: str = "https://provider.example/v1") -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "auth.v1",
                "environment": {
                    "TEST_API_KEY": key,
                    "TEST_BASE_URL": base_url,
                },
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


def test_cli_uses_default_home_config_without_auth_for_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider_manifest: ProviderManifest,
    model_manifest: ModelManifest,
) -> None:
    _write_config(tmp_path, provider_manifest, (model_manifest,))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ABSTRAK_CONFIG", raising=False)
    monkeypatch.delenv("ABSTRAK_AUTH", raising=False)

    exit_code = main(["validate"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == EXIT_OK
    assert output["provider_id"] == provider_manifest.id
    assert output["model_id"] == model_manifest.id
    assert not (tmp_path / ".abstrak" / "auth.json").exists()


def test_cli_profile_overrides_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider_manifest: ProviderManifest,
    model_manifest: ModelManifest,
) -> None:
    alternate_payload = model_manifest.model_dump(mode="json")
    alternate_payload.update(
        {
            "id": "alternate-model",
            "api_model": "openai/alternate-model-snapshot",
            "expected_returned_model": "alternate-model-snapshot",
        }
    )
    alternate = ModelManifest.model_validate(alternate_payload)
    config_path = _write_config(tmp_path, provider_manifest, (model_manifest, alternate))
    monkeypatch.delenv("ABSTRAK_CONFIG", raising=False)

    exit_code = main(
        ["validate", "--config", str(config_path), "--profile", alternate.id]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == EXIT_OK
    assert output["model_id"] == alternate.id


def test_cli_rejects_unpaired_legacy_manifest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "validate",
            "--provider",
            "configs/examples/provider.openai-compatible.example.yaml",
        ]
    )

    assert exit_code == EXIT_CONFIG
    assert "must be supplied together" in capsys.readouterr().err


def test_config_rejects_profile_with_mismatched_provider(
    tmp_path: Path,
    provider_manifest: ProviderManifest,
    model_manifest: ModelManifest,
) -> None:
    model_payload = model_manifest.model_dump(mode="json")
    model_payload["provider"] = "different-provider"
    config_path = _write_config(
        tmp_path,
        provider_manifest,
        (ModelManifest.model_validate(model_payload),),
    )

    with pytest.raises(ConfigurationError, match="does not match"):
        load_app_config(config_path)


def test_auth_file_requires_private_permissions(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "schema_version": "auth.v1",
                "environment": {"TEST_API_KEY": "test-secret"},
            }
        ),
        encoding="utf-8",
    )
    auth_path.chmod(0o644)

    with pytest.raises(ConfigurationError, match="permissions 0600"):
        load_auth_store(auth_path)


def test_auth_values_are_redacted_and_process_environment_wins(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "schema_version": "auth.v1",
                "environment": {
                    "TEST_API_KEY": "disk-secret",
                    "TEST_BASE_URL": "https://disk.example/v1",
                },
            }
        ),
        encoding="utf-8",
    )
    auth_path.chmod(0o600)

    auth = load_auth_store(auth_path)
    merged = runtime_environment(
        auth,
        {
            "TEST_API_KEY": "process-secret",
            "TEST_BASE_URL": "https://process.example/v1",
        },
    )

    assert "disk-secret" not in repr(auth)
    assert merged["TEST_API_KEY"] == "process-secret"
    assert merged["TEST_BASE_URL"] == "https://process.example/v1"


def test_invalid_auth_error_does_not_echo_secret(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    secret = "must-not-appear-in-errors"
    auth_path.write_text(
        json.dumps(
            {
                "schema_version": "auth.v1",
                "environment": {"invalid-name": secret},
            }
        ),
        encoding="utf-8",
    )
    auth_path.chmod(0o600)

    with pytest.raises(ConfigurationError) as captured:
        load_auth_store(auth_path)

    rendered = "".join(traceback.format_exception(captured.value))
    assert secret not in str(captured.value)
    assert secret not in rendered


def test_invalid_config_error_does_not_echo_accidental_secret(tmp_path: Path) -> None:
    config_payload = yaml.safe_load(Path("configs/examples/config.example.yaml").read_text())
    secret = "must-not-appear-in-config-errors"
    config_payload["profiles"]["example-model"]["provider"]["api_key"] = secret
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    with pytest.raises(ConfigurationError) as captured:
        load_app_config(config_path)

    rendered = "".join(traceback.format_exception(captured.value))
    assert "api_key" in str(captured.value)
    assert secret not in str(captured.value)
    assert secret not in rendered


@pytest.mark.parametrize(
    ("auth_source", "expected_key"),
    [
        ("default", "default-secret"),
        ("environment", "environment-secret"),
        ("explicit", "explicit-secret"),
    ],
)
def test_smoke_cli_auth_path_precedence_and_process_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider_manifest: ProviderManifest,
    model_manifest: ModelManifest,
    auth_source: str,
    expected_key: str,
) -> None:
    config_path = _write_config(tmp_path, provider_manifest, (model_manifest,))
    default_auth = tmp_path / ".abstrak" / "auth.json"
    environment_auth = tmp_path / "environment-auth.json"
    explicit_auth = tmp_path / "explicit-auth.json"
    _write_auth(default_auth, "default-secret")
    _write_auth(environment_auth, "environment-secret")
    _write_auth(explicit_auth, "explicit-secret")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TEST_BASE_URL", "https://process.example/v1")
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    monkeypatch.delenv("ABSTRAK_AUTH", raising=False)
    arguments = ["smoke", "--live", "--config", str(config_path)]
    if auth_source in {"environment", "explicit"}:
        monkeypatch.setenv("ABSTRAK_AUTH", str(environment_auth))
    if auth_source == "explicit":
        arguments.extend(["--auth", str(explicit_auth)])

    captured_environment: dict[str, str] = {}

    def fake_smoke(
        _bundle: object, _artifact_root: str, environment: dict[str, str] | None = None
    ) -> int:
        captured_environment.update(environment or {})
        return EXIT_OK

    monkeypatch.setattr("abstrak.providers.cli._smoke", fake_smoke)

    assert main(arguments) == EXIT_OK
    output = capsys.readouterr()
    assert captured_environment["TEST_API_KEY"] == expected_key
    assert captured_environment["TEST_BASE_URL"] == "https://process.example/v1"
    assert "secret" not in output.out
    assert "secret" not in output.err


def test_missing_default_auth_is_an_empty_store(tmp_path: Path) -> None:
    auth = load_auth_store(tmp_path / "missing.json", missing_ok=True)

    assert auth == AuthStore(environment={})


def test_versioned_user_configuration_examples_match_the_schemas() -> None:
    config = load_app_config("configs/examples/config.example.yaml")
    auth_payload = json.loads(Path("configs/examples/auth.example.json").read_text())

    assert config.bundle().model.id == "example-model"
    assert AuthStore.model_validate(auth_payload).environment
