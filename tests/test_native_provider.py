from __future__ import annotations

import json
from typing import Any

import litellm
import pytest
from pydantic import ValidationError

from abstrak.anytime.contracts import (
    AnytimeAgentSpec,
    AnytimeGenerationSpec,
    AnytimeReasoningSpec,
)
from abstrak.providers.contracts import (
    ChatMessage,
    ErrorCategory,
    LogicalRequest,
    MessageRole,
    sha256_json,
)
from abstrak.providers.native_client import NativeProviderClient
from abstrak.providers.native_conformance import evaluate_native_dependency_conformance
from abstrak.providers.native_contracts import (
    NativeAgentBindingError,
    NativeClientIdentity,
    NativeConformanceCheck,
    NativeDependencyConformance,
    NativeFormalReadinessError,
    NativeManifestBundle,
    NativeModelManifest,
    NativeNormalizedError,
    NativeProviderCallError,
    NativeProviderManifest,
    NativeReasoningRecord,
    NativeResolvedProviderBinding,
    validate_anytime_agent_binding,
)
from abstrak.providers.native_transport import (
    LiteLLMNativeTransport,
    NativeUnsafeTransportState,
)


class ScriptedNativeTransport:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.call_count = 0
        self.call_protocols: list[str] = []
        self.requests: list[dict[str, Any]] = []

    def _call(self, protocol: str, kwargs: dict[str, Any]) -> Any:
        self.call_count += 1
        self.call_protocols.append(protocol)
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    def chat_completion(self, **kwargs: Any) -> Any:
        return self._call("chat_completions", kwargs)

    def responses(self, **kwargs: Any) -> Any:
        return self._call("responses", kwargs)


class SDKResponse(dict):
    def __init__(self, payload: dict[str, Any], *, hidden_secret: str | None = None) -> None:
        super().__init__(payload)
        self._hidden_params = {"debug": hidden_secret} if hidden_secret else {}
        self._response_headers = {
            "x-request-id": "header-request-id",
            "authorization": hidden_secret or "not-recorded",
        }


def _responses_bundle(
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> NativeManifestBundle:
    provider = NativeProviderManifest(
        id="openai",
        protocol="responses",
        litellm_provider="openai",
        base_url_env="NATIVE_BASE_URL",
        api_key_env="NATIVE_API_KEY",
        timeout_seconds=30.0,
    )
    model = NativeModelManifest(
        id="gpt-5.6-luna",
        provider=provider.id,
        api_model="openai/gpt-5.6-luna",
        protocol="responses",
        max_output_tokens=16384,
        temperature=temperature,
        top_p=top_p,
        model_id_policy="exact",
        expected_returned_model="gpt-5.6-luna-2026-08-01",
    )
    return NativeManifestBundle(provider=provider, model=model)


def _deepseek_bundle() -> NativeManifestBundle:
    provider = NativeProviderManifest(
        id="deepseek",
        protocol="chat_completions",
        litellm_provider="deepseek",
        base_url_env="NATIVE_BASE_URL",
        api_key_env="NATIVE_API_KEY",
        timeout_seconds=30.0,
    )
    model = NativeModelManifest(
        id="deepseek-v4-flash",
        provider=provider.id,
        api_model="deepseek/deepseek-v4-flash",
        protocol="chat_completions",
        max_output_tokens=16384,
    )
    return NativeManifestBundle(provider=provider, model=model)


def _environment() -> dict[str, str]:
    return {
        "NATIVE_API_KEY": "native-unit-test-secret",
        "NATIVE_BASE_URL": "https://provider.example/private-path-segment-123456789/v1",
    }


def _request(model_ref: str) -> LogicalRequest:
    return LogicalRequest(
        request_id="trajectory-001-attempt-1-call-01",
        model_ref=model_ref,
        messages=(
            ChatMessage(role=MessageRole.SYSTEM, content="Preserve trailing space.  "),
            ChatMessage(role=MessageRole.USER, content="First input"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Prior candidate"),
            ChatMessage(role=MessageRole.USER, content="Improve it"),
        ),
        trajectory_id="trajectory-001",
        turn_index=0,
    )


def _responses_payload() -> dict[str, Any]:
    return {
        "id": "response-001",
        "model": "gpt-5.6-luna-2026-08-01",
        "status": "completed",
        "output": [
            {"type": "reasoning", "id": "reasoning-001", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "candidate ", "annotations": []},
                    {"type": "output_text", "text": "code", "annotations": []},
                ],
            },
        ],
        "usage": {
            "input_tokens": 101,
            "input_tokens_details": {"cached_tokens": 11},
            "output_tokens": 29,
            "output_tokens_details": {"reasoning_tokens": 17},
            "total_tokens": 130,
        },
    }


def _anytime_agent(bundle: NativeManifestBundle) -> AnytimeAgentSpec:
    return AnytimeAgentSpec(
        id=bundle.model.id,
        provider_id=bundle.provider.id,
        model_ref=bundle.model.id,
        native_protocol=bundle.provider.protocol,
        generation=AnytimeGenerationSpec(
            max_output_tokens=bundle.model.max_output_tokens,
            reasoning=AnytimeReasoningSpec(
                requested_reasoning_effort="xhigh",
                conformance_requirement="literal_xhigh",
            ),
            temperature=bundle.model.temperature,
            top_p=bundle.model.top_p,
        ),
    )


def test_anytime_native_manifests_are_separate_and_protocol_strict() -> None:
    bundle = _responses_bundle()

    assert bundle.provider.schema_version == "abstrak-anytime-provider.v1"
    assert bundle.model.schema_version == "abstrak-anytime-model.v1"
    assert bundle.model.requested_reasoning_effort == "xhigh"
    assert bundle.model.temperature is None
    assert bundle.model.top_p is None
    assert len(bundle.provider_sha256) == 64
    assert len(bundle.model_sha256) == 64

    with pytest.raises(ValidationError, match="native route"):
        NativeProviderManifest(
            id="bad-deepseek",
            protocol="responses",
            litellm_provider="deepseek",
            api_key_env="KEY",
        )
    with pytest.raises(ValidationError):
        NativeModelManifest(
            id="too-small",
            provider="provider",
            api_model="model",
            protocol="responses",
            max_output_tokens=255,
        )


def test_dependency_conformance_preserves_gpt_xhigh_and_blocks_deepseek_collapse() -> None:
    gpt = evaluate_native_dependency_conformance(_responses_bundle())
    deepseek = evaluate_native_dependency_conformance(_deepseek_bundle())

    assert gpt.status == "pass"
    assert gpt.formal_ready is True
    assert gpt.provider_manifest_sha256 == _responses_bundle().provider_sha256
    assert gpt.model_manifest_sha256 == _responses_bundle().model_sha256
    assert gpt.reasoning.effective_mode == "literal_xhigh"
    assert gpt.reasoning.fidelity == "literal"
    assert {check.name for check in gpt.checks} >= {
        "litellm_version",
        "native_responses_entrypoint",
        "native_responses_config",
        "generation_parameters_preserved",
        "literal_xhigh_preserved",
    }

    assert deepseek.status == "fail"
    assert deepseek.formal_ready is False
    assert deepseek.reasoning.requested_effort == "xhigh"
    assert deepseek.reasoning.effective_mode == "thinking_enabled"
    assert deepseek.reasoning.fidelity == "collapsed"
    fidelity = next(check for check in deepseek.checks if check.name == "literal_xhigh_preserved")
    assert fidelity.status == "fail"
    assert "formal blocked" in fidelity.detail


def test_m1_agent_binding_matches_every_runtime_generation_field() -> None:
    bundle = _responses_bundle()
    agent = _anytime_agent(bundle)

    validate_anytime_agent_binding(agent, bundle)

    with pytest.raises(NativeAgentBindingError, match="model_ref"):
        validate_anytime_agent_binding(
            agent.model_copy(update={"model_ref": "different-model"}),
            bundle,
        )
    with pytest.raises(NativeAgentBindingError, match="temperature"):
        validate_anytime_agent_binding(agent, _responses_bundle(temperature=0.25))

    transport = ScriptedNativeTransport(response=_responses_payload())
    with pytest.raises(NativeAgentBindingError, match="model_ref"):
        NativeProviderClient(
            bundle,
            agent=agent.model_copy(update={"model_ref": "different-model"}),
            transport=transport,
            environment=_environment(),
        )
    assert transport.call_count == 0


def test_responses_request_is_exact_single_call_and_normalizes_usage() -> None:
    secret = _environment()["NATIVE_API_KEY"]
    payload = _responses_payload()
    payload["usage"]["provider_debug"] = {"credential": secret}
    sdk_response = SDKResponse(payload, hidden_secret=secret)
    transport = ScriptedNativeTransport(response=sdk_response)
    bundle = _responses_bundle()
    client = NativeProviderClient(
        bundle,
        agent=_anytime_agent(bundle),
        transport=transport,
        environment=_environment(),
    )
    request = _request(bundle.model.id)

    response = client.complete(request)

    assert client.resolved_binding.schema_version == "abstrak-anytime-provider-binding.v1"
    assert client.resolved_binding.agent_sha256 == _anytime_agent(bundle).sha256
    assert client.resolved_binding.provider_manifest_sha256 == bundle.provider_sha256
    assert client.resolved_binding.dependency_conformance.formal_ready is True
    assert transport.call_count == 1
    assert transport.call_protocols == ["responses"]
    sent = transport.requests[0]
    assert sent["input"] == [message.model_dump(mode="json") for message in request.messages]
    assert sent["input"][0]["content"].endswith("  ")
    assert sent["reasoning"] == {"effort": "xhigh"}
    assert sent["max_output_tokens"] == 16384
    assert sent["store"] is False
    assert sent["truncation"] == "disabled"
    assert sent["stream"] is False
    assert sent["num_retries"] == 0
    assert sent["max_retries"] == 0
    assert sent["retry_policy"] is None
    assert sent["context_window_fallback_dict"] == {}
    assert sent["caching"] is False
    assert sent["drop_params"] is False
    assert not {
        "messages",
        "n",
        "tools",
        "previous_response_id",
        "temperature",
        "top_p",
        "max_completion_tokens",
    }.intersection(sent)

    assert response.request_id == request.request_id
    assert response.text == "candidate code"
    assert response.finish_reason == "stop"
    assert response.provider_finish_reason == "completed"
    assert response.usage.input_tokens == 101
    assert response.usage.cached_input_tokens == 11
    assert response.usage.output_tokens == 29
    assert response.usage.reasoning_tokens == 17
    assert response.usage.total_tokens == 130
    assert response.usage.raw_usage is not None
    assert response.usage.raw_usage["provider_debug"]["credential"] == "<redacted>"
    assert response.reasoning.effective_mode == "literal_xhigh"
    artifact_text = json.dumps(response.model_dump(mode="json"), ensure_ascii=False)
    assert secret not in artifact_text
    assert _environment()["NATIVE_BASE_URL"] not in artifact_text
    assert "<redacted>" in artifact_text
    assert "authorization" not in response.raw_transport_response["response_headers_allowlist"]


def test_optional_sampling_values_are_forwarded_but_none_is_omitted() -> None:
    explicit = _responses_bundle(temperature=0.25, top_p=0.9)
    client = NativeProviderClient(
        explicit,
        agent=_anytime_agent(explicit),
        transport=ScriptedNativeTransport(response=_responses_payload()),
        environment=_environment(),
    )
    actual, sanitized = client._transport_requests(_request(explicit.model.id))

    assert actual["temperature"] == 0.25
    assert actual["top_p"] == 0.9
    assert sanitized["temperature"] == 0.25
    assert sanitized["top_p"] == 0.9

    omitted = _responses_bundle()
    omitted_client = NativeProviderClient(
        omitted,
        agent=_anytime_agent(omitted),
        transport=ScriptedNativeTransport(response=_responses_payload()),
        environment=_environment(),
    )
    omitted_actual, _ = omitted_client._transport_requests(_request(omitted.model.id))
    assert "temperature" not in omitted_actual
    assert "top_p" not in omitted_actual


def test_unsupported_sampling_is_formal_blocked_before_transport() -> None:
    bundle = _responses_bundle(temperature=0.25)
    transport = ScriptedNativeTransport(response=_responses_payload())
    client = NativeProviderClient(
        bundle,
        agent=_anytime_agent(bundle),
        transport=transport,
        environment=_environment(),
    )

    assert client.dependency_conformance.status == "fail"
    assert client.dependency_conformance.formal_ready is False
    parameter_check = next(
        check
        for check in client.dependency_conformance.checks
        if check.name == "generation_parameters_preserved"
    )
    assert parameter_check.status == "fail"
    assert "UnsupportedParamsError" in parameter_check.detail
    with pytest.raises(NativeFormalReadinessError):
        client.complete(_request(bundle.model.id))
    assert transport.call_count == 0


def test_deepseek_formal_call_fails_closed_before_transport() -> None:
    bundle = _deepseek_bundle()
    transport = ScriptedNativeTransport(response={})
    client = NativeProviderClient(
        bundle,
        agent=_anytime_agent(bundle),
        transport=transport,
        environment=_environment(),
    )
    request = _request(bundle.model.id)

    with pytest.raises(NativeFormalReadinessError, match="formal blocked"):
        client.complete(request)

    assert client.formal_ready is False
    assert transport.call_count == 0
    actual, sanitized = client._transport_requests(request)
    assert actual["messages"] == [
        message.model_dump(mode="json") for message in request.messages
    ]
    assert actual["reasoning_effort"] == "xhigh"
    assert actual["max_completion_tokens"] == 16384
    assert actual["stream"] is False
    assert "temperature" not in actual
    assert "tools" not in actual
    assert sanitized["reasoning_effort"] == "xhigh"


def test_chat_client_dispatch_and_normalization_with_literal_conformance_fixture(
    monkeypatch,
) -> None:
    """Exercise the Chat path without weakening the production DeepSeek gate."""

    bundle = _deepseek_bundle()
    reasoning = NativeReasoningRecord(
        submitted_parameter="reasoning_effort",
        submitted_value="xhigh",
        effective_mode="literal_xhigh",
        fidelity="literal",
        evidence="scripted future adapter preserves literal xhigh",
    )
    conformance = NativeDependencyConformance(
        status="pass",
        formal_ready=True,
        protocol="chat_completions",
        provider_manifest_sha256=bundle.provider_sha256,
        model_manifest_sha256=bundle.model_sha256,
        litellm_version="1.92.0",
        reasoning=reasoning,
        checks=(
            NativeConformanceCheck(
                name="scripted_literal_xhigh",
                status="pass",
                detail="offline fixture only",
            ),
        ),
    )
    monkeypatch.setattr(
        "abstrak.providers.native_client.evaluate_native_dependency_conformance",
        lambda resolved_bundle: conformance,
    )
    payload = {
        "id": "chat-001",
        "model": "deepseek-v4-flash-2026-08-01",
        "choices": [
            {
                "message": {"role": "assistant", "content": "chat candidate"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 80,
            "prompt_tokens_details": {"cached_tokens": 7},
            "completion_tokens": 25,
            "completion_tokens_details": {"reasoning_tokens": 13},
            "total_tokens": 105,
        },
    }
    transport = ScriptedNativeTransport(response=payload)
    client = NativeProviderClient(
        bundle,
        agent=_anytime_agent(bundle),
        transport=transport,
        environment=_environment(),
    )

    response = client.complete(_request(bundle.model.id))

    assert transport.call_count == 1
    assert transport.call_protocols == ["chat_completions"]
    sent = transport.requests[0]
    assert sent["messages"] == [
        message.model_dump(mode="json") for message in _request(bundle.model.id).messages
    ]
    assert sent["n"] == 1
    assert sent["reasoning_effort"] == "xhigh"
    assert sent["max_completion_tokens"] == 16384
    assert "input" not in sent
    assert "max_output_tokens" not in sent
    assert response.text == "chat candidate"
    assert response.usage.input_tokens == 80
    assert response.usage.cached_input_tokens == 7
    assert response.usage.output_tokens == 25
    assert response.usage.reasoning_tokens == 13
    assert response.usage.total_tokens == 105


@pytest.mark.parametrize(
    ("reason", "finish_reason", "provider_finish_reason"),
    [
        ("max_output_tokens", "length", "incomplete:max_output_tokens"),
        ("content_filter", "content_filter", "incomplete:content_filter"),
    ],
)
def test_responses_incomplete_status_is_normalized(
    reason: str,
    finish_reason: str,
    provider_finish_reason: str,
) -> None:
    payload = _responses_payload()
    payload["status"] = "incomplete"
    payload["incomplete_details"] = {"reason": reason}
    bundle = _responses_bundle()
    client = NativeProviderClient(
        bundle,
        agent=_anytime_agent(bundle),
        transport=ScriptedNativeTransport(response=payload),
        environment=_environment(),
    )

    response = client.complete(_request(bundle.model.id))

    assert response.finish_reason == finish_reason
    assert response.provider_finish_reason == provider_finish_reason


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(status="mystery"),
        lambda payload: payload.update(status="failed"),
        lambda payload: payload.update(error={"message": "provider failed"}),
        lambda payload: payload.update(output=[]),
        lambda payload: payload.update(model="wrong-model"),
    ],
)
def test_responses_endpoint_conformance_failures_are_terminal(mutation) -> None:
    payload = _responses_payload()
    mutation(payload)
    secret = _environment()["NATIVE_API_KEY"]
    payload["usage"]["provider_debug"] = {"credential": secret}
    bundle = _responses_bundle()
    transport = ScriptedNativeTransport(response=payload)
    client = NativeProviderClient(
        bundle,
        agent=_anytime_agent(bundle),
        transport=transport,
        environment=_environment(),
    )

    with pytest.raises(NativeProviderCallError) as captured:
        client.complete(_request(bundle.model.id))

    assert transport.call_count == 1
    assert captured.value.record.category == ErrorCategory.MALFORMED_RESPONSE
    assert captured.value.record.request_submitted is True
    assert captured.value.record.possibly_charged is True
    assert captured.value.record.partial_usage is not None
    assert captured.value.record.partial_usage.total_tokens == 130
    assert captured.value.record.partial_usage.raw_usage is not None
    assert (
        captured.value.record.partial_usage.raw_usage["provider_debug"]["credential"]
        == "<redacted>"
    )
    if payload.get("model") == "wrong-model":
        assert "endpoint conformance failure" in captured.value.record.sanitized_message


class AuthenticationError(Exception):
    status_code = True
    code = None


def test_provider_error_is_redacted_and_never_retried() -> None:
    secret = _environment()["NATIVE_API_KEY"]
    transport = ScriptedNativeTransport(error=AuthenticationError(f"bad credential {secret}"))
    bundle = _responses_bundle()
    client = NativeProviderClient(
        bundle,
        agent=_anytime_agent(bundle),
        transport=transport,
        environment=_environment(),
    )

    with pytest.raises(NativeProviderCallError) as captured:
        client.complete(_request(bundle.model.id))

    assert transport.call_count == 1
    assert captured.value.record.category == ErrorCategory.AUTHENTICATION
    assert captured.value.record.request_submitted is True
    assert captured.value.record.possibly_charged is False
    assert captured.value.record.partial_usage is None
    assert captured.value.record.http_status is None
    assert captured.value.record.provider_code is None
    assert secret not in captured.value.record.sanitized_message
    assert "<redacted>" in captured.value.record.sanitized_message

    invalid_charge = captured.value.record.model_dump(mode="python")
    invalid_charge["request_submitted"] = False
    invalid_charge["possibly_charged"] = True
    with pytest.raises(ValidationError, match="cannot be possibly charged"):
        NativeNormalizedError.model_validate(invalid_charge)


def test_reasoning_identity_and_resolved_binding_validators_fail_closed() -> None:
    literal = NativeReasoningRecord(
        submitted_parameter="reasoning",
        submitted_value={"effort": "xhigh"},
        effective_mode="literal_xhigh",
        fidelity="literal",
        evidence="fixture preserves literal xhigh",
    )
    identity = NativeClientIdentity(
        provider_id="openai",
        model_id="gpt-5.6-luna",
        protocol="responses",
        provider_manifest_sha256="a" * 64,
        model_manifest_sha256="b" * 64,
        requested_model="openai/gpt-5.6-luna",
        reasoning=literal,
        formal_ready=False,
    )
    assert identity.formal_ready is False

    with pytest.raises(ValidationError, match="submit exactly"):
        NativeReasoningRecord(
            submitted_parameter="reasoning",
            submitted_value="xhigh",
            effective_mode="literal_xhigh",
            fidelity="literal",
            evidence="invalid fixture",
        )

    bundle = _responses_bundle()
    mismatched_model = bundle.model.model_copy(update={"provider": "different-provider"})
    with pytest.raises(ValidationError, match="model provider"):
        NativeResolvedProviderBinding(
            provider=bundle.provider,
            model=mismatched_model,
            agent=_anytime_agent(bundle),
            agent_sha256=_anytime_agent(bundle).sha256,
            provider_manifest_sha256=bundle.provider_sha256,
            model_manifest_sha256=sha256_json(mismatched_model),
            dependency_conformance=evaluate_native_dependency_conformance(bundle),
        )

    other_bundle = _responses_bundle(top_p=0.9)
    with pytest.raises(ValidationError, match="different model manifest"):
        NativeResolvedProviderBinding(
            provider=other_bundle.provider,
            model=other_bundle.model,
            agent=_anytime_agent(other_bundle),
            agent_sha256=_anytime_agent(other_bundle).sha256,
            provider_manifest_sha256=other_bundle.provider_sha256,
            model_manifest_sha256=other_bundle.model_sha256,
            dependency_conformance=evaluate_native_dependency_conformance(bundle),
        )


def test_controlled_transport_rejects_global_fallback_before_call(monkeypatch) -> None:
    called = False

    def responses_fn(**kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(litellm, "model_fallbacks", [{"model": "forbidden"}])
    transport = LiteLLMNativeTransport(responses_fn=responses_fn)

    with pytest.raises(NativeUnsafeTransportState, match="global state"):
        transport.responses(model="openai/model", input="test")

    assert called is False
    assert transport.call_count == 0
    assert transport.call_protocols == []


def test_native_transport_dispatches_each_protocol_exactly_once() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def completion_fn(**kwargs: Any) -> dict[str, str]:
        calls.append(("chat", kwargs))
        return {"kind": "chat"}

    def responses_fn(**kwargs: Any) -> dict[str, str]:
        calls.append(("responses", kwargs))
        return {"kind": "responses"}

    transport = LiteLLMNativeTransport(
        completion_fn=completion_fn,
        responses_fn=responses_fn,
    )

    assert transport.chat_completion(model="deepseek/model", messages=[]) == {"kind": "chat"}
    assert transport.responses(model="openai/model", input=[]) == {"kind": "responses"}
    assert transport.call_count == 2
    assert transport.call_protocols == ["chat_completions", "responses"]
    assert [kind for kind, _ in calls] == ["chat", "responses"]
