from __future__ import annotations

from pathlib import Path

import pytest

from abstrak.providers.manifests import (
    GenerationConfig,
    ManifestBundle,
    ManifestLoadError,
    ModelManifest,
    ProviderManifest,
    load_manifest,
    manifest_sha256,
)


def test_example_manifests_validate_offline() -> None:
    provider = load_manifest(
        "configs/examples/provider.openai-compatible.example.yaml", ProviderManifest
    )
    model = load_manifest("configs/examples/model.openai-compatible.example.yaml", ModelManifest)

    bundle = ManifestBundle(provider=provider, model=model)

    assert bundle.provider.retry.max_attempts == 1
    assert bundle.provider.transport.allow_fallback is False
    assert bundle.model.generation.transport_parameters() == {"max_completion_tokens": 128}


def test_unknown_or_secret_fields_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text(
        """
schema_version: provider.v1
id: unsafe
api_key_env: SAFE_KEY_ENV
api_key: literal-secret
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ManifestLoadError, match="api_key"):
        load_manifest(path, ProviderManifest)


def test_implicit_retry_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text(
        """
schema_version: provider.v1
id: retrying
api_key_env: TEST_KEY
retry:
  max_attempts: 2
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ManifestLoadError, match="max_attempts"):
        load_manifest(path, ProviderManifest)


def test_exact_model_requires_expected_returned_model(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """
schema_version: model.v1
id: exact-model
provider: provider
api_model: exact-api-model
model_id_policy: exact
generation:
  max_completion_tokens: 32
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ManifestLoadError, match="expected_returned_model"):
        load_manifest(path, ModelManifest)


def test_manifest_hash_is_stable(provider_manifest: ProviderManifest) -> None:
    reconstructed = ProviderManifest.model_validate(
        dict(reversed(list(provider_manifest.model_dump(mode="json").items())))
    )

    assert manifest_sha256(provider_manifest) == manifest_sha256(reconstructed)


def test_live_probe_requires_strict_common_capabilities() -> None:
    with pytest.raises(ValueError, match="requires usage"):
        ModelManifest(
            id="weak-model",
            provider="provider",
            api_model="weak-model",
            model_id_policy="mutable_alias",
            allow_live_probe=True,
            generation=GenerationConfig(max_completion_tokens=32),
            output_contract="plain_json",
            capabilities={"usage_reporting": "optional"},
        )
