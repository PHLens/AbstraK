from __future__ import annotations

import litellm
import pytest

from abstrak.providers.client import ProviderClient
from abstrak.providers.contracts import (
    ChatMessage,
    ErrorCategory,
    LogicalRequest,
    MessageRole,
    ProviderCallError,
)
from abstrak.providers.manifests import ManifestBundle
from abstrak.providers.transport import LiteLLMTransport


def test_global_fallback_is_rejected_before_transport_call(
    monkeypatch,
    manifest_bundle: ManifestBundle,
    provider_environment: dict[str, str],
) -> None:
    monkeypatch.setattr(litellm, "model_fallbacks", [{"model": "forbidden-fallback"}])
    completion_called = False

    def completion_fn(**kwargs) -> None:
        nonlocal completion_called
        completion_called = True

    transport = LiteLLMTransport(completion_fn=completion_fn)
    client = ProviderClient(manifest_bundle, transport=transport, environment=provider_environment)
    request = LogicalRequest(
        model_ref="test-model",
        messages=(ChatMessage(role=MessageRole.USER, content="test"),),
    )

    with pytest.raises(ProviderCallError) as captured:
        client.complete(request)

    assert completion_called is False
    assert transport.call_count == 0
    assert captured.value.record.category == ErrorCategory.INVALID_REQUEST
    assert captured.value.record.request_submitted is False
    assert captured.value.record.possibly_charged is False
