from __future__ import annotations

from typing import Any

import pytest

from abstrak.providers.manifests import (
    GenerationConfig,
    ManifestBundle,
    ModelManifest,
    ProviderManifest,
)


class ScriptedTransport:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.call_count = 0
        self.requests: list[dict[str, Any]] = []

    def completion(self, **kwargs: Any) -> Any:
        self.call_count += 1
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def provider_manifest() -> ProviderManifest:
    return ProviderManifest(
        id="test-provider",
        litellm_provider="openai",
        base_url_env="TEST_BASE_URL",
        api_key_env="TEST_API_KEY",
        timeout_seconds=30,
    )


@pytest.fixture
def model_manifest() -> ModelManifest:
    return ModelManifest(
        id="test-model",
        provider="test-provider",
        api_model="openai/test-model-snapshot",
        model_id_policy="exact",
        expected_returned_model="test-model-snapshot",
        allow_live_probe=True,
        generation=GenerationConfig(max_completion_tokens=128),
        output_contract="plain_json",
    )


@pytest.fixture
def manifest_bundle(
    provider_manifest: ProviderManifest, model_manifest: ModelManifest
) -> ManifestBundle:
    return ManifestBundle(provider=provider_manifest, model=model_manifest)


@pytest.fixture
def provider_environment() -> dict[str, str]:
    return {
        "TEST_API_KEY": "unit-test-secret-value",
        "TEST_BASE_URL": "https://provider.example/v1",
    }


@pytest.fixture
def valid_response() -> dict[str, Any]:
    return {
        "id": "response-1",
        "model": "test-model-snapshot",
        "system_fingerprint": "system-fingerprint-1",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"action":"finish","nonce":"fixed-nonce"}',
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }
