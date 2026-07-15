from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import ScriptedTransport

from abstrak.providers.client import ProviderClient, ProviderConfigurationError, _git_state
from abstrak.providers.contracts import (
    ChatMessage,
    ErrorCategory,
    LogicalRequest,
    MessageRole,
    ProviderCallError,
)
from abstrak.providers.manifests import ManifestBundle


def _request() -> LogicalRequest:
    return LogicalRequest(
        model_ref="test-model",
        messages=(
            ChatMessage(role=MessageRole.SYSTEM, content="Keep whitespace exactly.  "),
            ChatMessage(role=MessageRole.USER, content="Return JSON."),
        ),
        local_trajectory_seed=17,
    )


def test_single_call_and_response_normalization(
    manifest_bundle: ManifestBundle,
    provider_environment: dict[str, str],
    valid_response: dict[str, Any],
) -> None:
    transport = ScriptedTransport(response=valid_response)
    client = ProviderClient(manifest_bundle, transport=transport, environment=provider_environment)

    response = client.complete(_request())

    assert transport.call_count == 1
    sent = transport.requests[0]
    assert sent["api_key"] == provider_environment["TEST_API_KEY"]
    assert sent["base_url"] == provider_environment["TEST_BASE_URL"]
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][0]["content"].endswith("  ")
    assert sent["stream"] is False
    assert sent["n"] == 1
    assert sent["num_retries"] == 0
    assert sent["max_retries"] == 0
    assert sent["caching"] is False
    assert "temperature" not in sent
    assert response.returned_model == "test-model-snapshot"
    assert response.usage.input_tokens == 10
    assert response.usage.cached_input_tokens == 2
    assert response.usage.output_tokens == 5
    assert response.usage.reasoning_tokens == 1
    assert response.usage.provider_reported is True
    assert response.usage.core_fields_complete is True
    artifact_text = str(response.model_dump(mode="json"))
    assert provider_environment["TEST_API_KEY"] not in artifact_text
    assert provider_environment["TEST_BASE_URL"] not in artifact_text
    assert str(MessageRole.USER) == "user"


def test_missing_usage_remains_unknown(
    manifest_bundle: ManifestBundle, provider_environment: dict[str, str]
) -> None:
    response_without_usage = {
        "id": "response-1",
        "model": "test-model-snapshot",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "{}"},
            }
        ],
    }
    client = ProviderClient(
        manifest_bundle,
        transport=ScriptedTransport(response=response_without_usage),
        environment=provider_environment,
    )

    response = client.complete(_request())

    assert response.usage.provider_reported is False
    assert response.usage.core_fields_complete is False
    assert response.usage.input_tokens is None
    assert response.usage.output_tokens is None
    assert response.usage.total_tokens is None


class AuthenticationError(Exception):
    status_code = 401
    code = "invalid_api_key"


def test_error_is_classified_redacted_and_not_retried(
    manifest_bundle: ManifestBundle, provider_environment: dict[str, str]
) -> None:
    secret = provider_environment["TEST_API_KEY"]
    transport = ScriptedTransport(error=AuthenticationError(f"bad credential {secret}"))
    client = ProviderClient(manifest_bundle, transport=transport, environment=provider_environment)

    with pytest.raises(ProviderCallError) as captured:
        client.complete(_request())

    assert transport.call_count == 1
    assert captured.value.record.category == ErrorCategory.AUTHENTICATION
    assert captured.value.record.http_status == 401
    assert secret not in captured.value.record.sanitized_message
    assert captured.value.record.retryable is False


@pytest.mark.parametrize(
    ("exception_type", "expected"),
    [
        (type("RateLimitError", (Exception,), {}), ErrorCategory.RATE_LIMIT),
        (type("Timeout", (Exception,), {}), ErrorCategory.TIMEOUT),
        (type("APIConnectionError", (Exception,), {}), ErrorCategory.NETWORK),
        (type("ContextWindowExceededError", (Exception,), {}), ErrorCategory.CONTEXT_LENGTH),
        (type("InternalServerError", (Exception,), {}), ErrorCategory.SERVER_ERROR),
    ],
)
def test_error_taxonomy(
    manifest_bundle: ManifestBundle,
    provider_environment: dict[str, str],
    exception_type: type[Exception],
    expected: ErrorCategory,
) -> None:
    client = ProviderClient(
        manifest_bundle,
        transport=ScriptedTransport(error=exception_type("failure")),
        environment=provider_environment,
    )

    with pytest.raises(ProviderCallError) as captured:
        client.complete(_request())

    assert captured.value.record.category == expected


def test_malformed_response_is_a_terminal_error(
    manifest_bundle: ManifestBundle, provider_environment: dict[str, str]
) -> None:
    transport = ScriptedTransport(response={"choices": []})
    client = ProviderClient(manifest_bundle, transport=transport, environment=provider_environment)

    with pytest.raises(ProviderCallError) as captured:
        client.complete(_request())

    assert transport.call_count == 1
    assert captured.value.record.category == ErrorCategory.MALFORMED_RESPONSE
    assert captured.value.record.possibly_charged is True


@pytest.mark.parametrize(
    ("base_url", "message"),
    [
        ("http://provider.example/v1", "must use HTTPS"),
        ("https://user:secret@provider.example/v1", "cannot contain userinfo"),
        ("https://provider.example/v1?token=secret", "cannot contain userinfo"),
        ("https://provider.example:invalid/v1", "invalid port"),
    ],
)
def test_unsafe_base_urls_are_configuration_errors(
    manifest_bundle: ManifestBundle, base_url: str, message: str
) -> None:
    with pytest.raises(ProviderConfigurationError, match=message):
        ProviderClient(
            manifest_bundle,
            transport=ScriptedTransport(),
            environment={"TEST_API_KEY": "secret", "TEST_BASE_URL": base_url},
        )


def test_source_provenance_includes_untracked_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=AbstraK Test",
            "-c",
            "user.email=abstrak@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    clean = _git_state(tmp_path)
    (tmp_path / "untracked.txt").write_text("untracked contents", encoding="utf-8")
    dirty = _git_state(tmp_path)

    assert clean["worktree_dirty"] is False
    assert dirty["worktree_dirty"] is True
    assert clean["source_state_sha256"] != dirty["source_state_sha256"]
