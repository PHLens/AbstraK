"""Protocol-native provider client for the anytime study infrastructure."""

from __future__ import annotations

import ipaddress
import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from pydantic import BaseModel

from abstrak.anytime.contracts import AnytimeAgentSpec
from abstrak.providers.contracts import ErrorCategory, LogicalRequest, NormalizedUsage, sha256_json
from abstrak.providers.native_conformance import evaluate_native_dependency_conformance
from abstrak.providers.native_contracts import (
    NativeDependencyReadinessError,
    NativeMalformedProviderResponse,
    NativeManifestBundle,
    NativeNormalizedError,
    NativeNormalizedResponse,
    NativeProviderCallError,
    NativeResolvedProviderBinding,
    native_client_identity,
    validate_anytime_agent_binding,
    validate_native_request,
)
from abstrak.providers.native_transport import (
    LiteLLMNativeTransport,
    NativeTransport,
    NativeUnsafeTransportState,
)

RESPONSE_HEADER_ALLOWLIST = {
    "date",
    "request-id",
    "x-request-id",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
}


class NativeProviderConfigurationError(ValueError):
    pass


class NativeArtifactSecretError(NativeProviderConfigurationError):
    """Raised before an unsafe request or artifact can cross the client boundary."""


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise NativeMalformedProviderResponse(
            f"expected mapping-like response, received {type(value).__name__}"
        )
    try:
        return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise NativeMalformedProviderResponse(
            f"response is not JSON serializable: {error}"
        ) from error


def _nested_int(mapping: Mapping[str, Any], *path: str) -> int | None:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _validate_base_url(base_url: str | None) -> None:
    if base_url is None:
        return
    parts = urlsplit(base_url)
    try:
        _ = parts.port
    except ValueError as error:
        raise NativeProviderConfigurationError("provider base URL has an invalid port") from error
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise NativeProviderConfigurationError(
            "provider base URL must be an absolute HTTP(S) URL"
        )
    if parts.username or parts.password or parts.query or parts.fragment:
        raise NativeProviderConfigurationError(
            "provider base URL cannot contain userinfo, query parameters, or fragments"
        )
    is_loopback = parts.hostname == "localhost"
    try:
        is_loopback = is_loopback or ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        pass
    if parts.scheme != "https" and not is_loopback:
        raise NativeProviderConfigurationError("remote provider base URLs must use HTTPS")


def _sanitized_base_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    parts = urlsplit(base_url)
    hostname = parts.hostname or ""
    if ":" in hostname:
        hostname = f"[{hostname}]"
    if parts.port is not None:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, "", "", ""))


def _redact_text(text: str, secrets: tuple[str, ...]) -> str:
    return _replace_secrets(text, secrets)[:4000]


def _replace_secrets(text: str, secrets: tuple[str, ...]) -> str:
    sanitized = text
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, "<redacted>")
    return sanitized


def _redact_json(value: Any, secrets: tuple[str, ...]) -> Any:
    normalized = json.loads(json.dumps(value, ensure_ascii=False, default=str))

    def redact(item: Any) -> Any:
        if isinstance(item, str):
            return _replace_secrets(item, secrets)
        if isinstance(item, list):
            return [redact(member) for member in item]
        if isinstance(item, dict):
            return {
                _replace_secrets(str(key), secrets): redact(member)
                for key, member in item.items()
            }
        return item

    return redact(normalized)


def _contains_artifact_secret(value: Any, secrets: tuple[str, ...]) -> bool:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        return any(secret and secret in value for secret in secrets)
    if isinstance(value, Mapping):
        return any(
            _contains_artifact_secret(str(key), secrets)
            or _contains_artifact_secret(member, secrets)
            for key, member in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_artifact_secret(member, secrets) for member in value)
    return False


def _redact_provider_scalar(value: Any, secrets: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    rendered = str(value)
    if not rendered:
        return None
    return _redact_text(rendered, secrets)


def _error_category(error: Exception) -> tuple[ErrorCategory, bool]:
    name = type(error).__name__.lower()
    message = str(error).lower()
    if "authentication" in name or "unauthorized" in message:
        return ErrorCategory.AUTHENTICATION, False
    if "permission" in name or "forbidden" in message:
        return ErrorCategory.PERMISSION, False
    if "ratelimit" in name or "rate limit" in message:
        return ErrorCategory.RATE_LIMIT, True
    if "timeout" in name or "timed out" in message:
        return ErrorCategory.TIMEOUT, True
    if "connection" in name or "network" in message:
        return ErrorCategory.NETWORK, True
    if "contextwindow" in name or "context length" in message:
        return ErrorCategory.CONTEXT_LENGTH, False
    if "unsupportedparam" in name or "unsupported parameter" in message:
        return ErrorCategory.UNSUPPORTED_PARAMETER, False
    if "contentpolicy" in name or "content filter" in message:
        return ErrorCategory.CONTENT_FILTER, False
    if "badrequest" in name or "invalidrequest" in name:
        return ErrorCategory.INVALID_REQUEST, False
    if any(marker in name for marker in ("server", "gateway", "serviceunavailable")):
        return ErrorCategory.SERVER_ERROR, True
    return ErrorCategory.UNKNOWN_PROVIDER_ERROR, False


def _usage(
    request: LogicalRequest,
    content: str,
    raw_usage: Any,
    *,
    protocol: str,
    secrets: tuple[str, ...],
) -> NormalizedUsage:
    usage_mapping = raw_usage if isinstance(raw_usage, Mapping) else None
    if protocol == "chat_completions":
        input_tokens = _nested_int(usage_mapping or {}, "prompt_tokens")
        output_tokens = _nested_int(usage_mapping or {}, "completion_tokens")
        cached_tokens = _nested_int(
            usage_mapping or {},
            "prompt_tokens_details",
            "cached_tokens",
        )
        reasoning_tokens = _nested_int(
            usage_mapping or {},
            "completion_tokens_details",
            "reasoning_tokens",
        )
    else:
        input_tokens = _nested_int(usage_mapping or {}, "input_tokens")
        output_tokens = _nested_int(usage_mapping or {}, "output_tokens")
        cached_tokens = _nested_int(
            usage_mapping or {},
            "input_tokens_details",
            "cached_tokens",
        )
        reasoning_tokens = _nested_int(
            usage_mapping or {},
            "output_tokens_details",
            "reasoning_tokens",
        )
    total_tokens = _nested_int(usage_mapping or {}, "total_tokens")
    core_fields = (input_tokens, output_tokens, total_tokens)
    return NormalizedUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        input_characters=sum(len(message.content) for message in request.messages),
        output_characters=len(content),
        provider_reported=any(value is not None for value in core_fields),
        core_fields_complete=all(value is not None for value in core_fields),
        raw_usage=(
            _redact_json(dict(usage_mapping), secrets)
            if usage_mapping is not None
            else None
        ),
    )


def _partial_usage(
    request: LogicalRequest,
    response: Any,
    *,
    protocol: str,
    secrets: tuple[str, ...],
) -> NormalizedUsage | None:
    try:
        payload = _json_mapping(response)
        usage = _usage(
            request,
            "",
            payload.get("usage"),
            protocol=protocol,
            secrets=secrets,
        )
    except NativeMalformedProviderResponse:
        return None
    return usage if usage.provider_reported else None


def _chat_output(payload: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise NativeMalformedProviderResponse("chat response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise NativeMalformedProviderResponse("chat response choice must be a mapping")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise NativeMalformedProviderResponse("chat response choice has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise NativeMalformedProviderResponse("chat response content must be non-empty text")
    raw_finish = choice.get("finish_reason")
    finish = str(raw_finish) if raw_finish is not None else None
    return content, finish, finish


def _responses_output(payload: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    if payload.get("error") is not None:
        raise NativeMalformedProviderResponse(
            "Responses endpoint conformance failure: payload contains an error"
        )
    status = payload.get("status")
    if status in {"failed", "cancelled"}:
        raise NativeMalformedProviderResponse(
            f"Responses endpoint conformance failure: request ended with status {status}"
        )
    if status not in {"completed", "incomplete"}:
        raise NativeMalformedProviderResponse(
            f"Responses endpoint conformance failure: unknown status {status!r}"
        )
    output = payload.get("output")
    if not isinstance(output, list):
        raise NativeMalformedProviderResponse("Responses payload has no output list")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "output_text":
                continue
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
    joined = "".join(texts)
    if not joined:
        raise NativeMalformedProviderResponse("Responses payload has no output_text")

    incomplete = payload.get("incomplete_details")
    reason = incomplete.get("reason") if isinstance(incomplete, Mapping) else None
    if status == "completed":
        finish = "stop"
    elif status == "incomplete" and reason == "max_output_tokens":
        finish = "length"
    elif status == "incomplete" and reason == "content_filter":
        finish = "content_filter"
    else:
        raise NativeMalformedProviderResponse(
            f"Responses endpoint conformance failure: unknown incomplete reason {reason!r}"
        )
    provider_finish = str(status)
    if reason is not None:
        provider_finish = f"{provider_finish}:{reason}"
    return joined, finish, provider_finish


class NativeProviderClient:
    """Execute one dependency-ready native call for infrastructure validation.

    This client does not authorize a formal study.  A formal runner must also
    require the endpoint-bound conformance receipt introduced in M9.
    """

    def __init__(
        self,
        bundle: NativeManifestBundle,
        *,
        agent: AnytimeAgentSpec,
        transport: NativeTransport | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        validate_anytime_agent_binding(agent, bundle)
        self.bundle = bundle
        self.agent = agent
        self.transport = (
            transport if transport is not None else LiteLLMNativeTransport()
        )
        values = environment or {}
        try:
            self._api_key = values[bundle.provider.api_key_env]
        except KeyError as error:
            raise NativeProviderConfigurationError(
                f"missing required environment variable: {bundle.provider.api_key_env}"
            ) from error
        if not self._api_key:
            raise NativeProviderConfigurationError("provider API key cannot be empty")
        if bundle.provider.base_url_env is None:
            self._base_url = None
        else:
            try:
                self._base_url = values[bundle.provider.base_url_env]
            except KeyError as error:
                raise NativeProviderConfigurationError(
                    f"missing required environment variable: {bundle.provider.base_url_env}"
                ) from error
            if not self._base_url:
                raise NativeProviderConfigurationError("provider base URL cannot be empty")
        _validate_base_url(self._base_url)
        self.dependency_conformance = evaluate_native_dependency_conformance(bundle)
        self._assert_artifact_secret_free(
            self.dependency_conformance,
            context="native dependency conformance",
        )
        self.native_identity = native_client_identity(bundle, self.dependency_conformance)
        self._assert_artifact_secret_free(
            self.native_identity,
            context="native client identity",
        )

    @property
    def dependency_ready(self) -> bool:
        return self.dependency_conformance.dependency_ready

    @property
    def study_ready(self) -> bool:
        """Always false until a later formal runner validates an M9 receipt."""

        return False

    @property
    def resolved_binding(self) -> NativeResolvedProviderBinding:
        binding = NativeResolvedProviderBinding(
            provider=self.bundle.provider,
            model=self.bundle.model,
            agent=self.agent,
            agent_sha256=self.agent.sha256,
            provider_manifest_sha256=self.bundle.provider_sha256,
            model_manifest_sha256=self.bundle.model_sha256,
            dependency_conformance=self.dependency_conformance,
        )
        self._assert_artifact_secret_free(binding, context="resolved provider binding")
        return binding

    @property
    def artifact_secrets(self) -> tuple[str, ...]:
        path_secrets: tuple[str, ...] = ()
        if self._base_url:
            path_secrets = tuple(
                segment
                for segment in urlsplit(self._base_url).path.split("/")
                if len(segment) >= 16
            )
        return self._api_key, self._base_url or "", *path_secrets

    def _assert_artifact_secret_free(self, value: Any, *, context: str) -> None:
        if _contains_artifact_secret(value, self.artifact_secrets):
            raise NativeArtifactSecretError(
                f"{context} contains provider credential material"
            )

    def _reject_unsafe_logical_request(self, request: LogicalRequest) -> None:
        if _contains_artifact_secret(request, self.artifact_secrets):
            raise NativeArtifactSecretError(
                "logical request contains provider credential material"
            )

    def _transport_requests(
        self,
        request: LogicalRequest,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        validate_native_request(request, self.bundle)
        self._reject_unsafe_logical_request(request)
        provider = self.bundle.provider
        model = self.bundle.model
        controlled = {
            "stream": False,
            "timeout": provider.timeout_seconds,
            "num_retries": 0,
            "max_retries": 0,
            "retry_policy": None,
            "context_window_fallback_dict": {},
            "caching": False,
            "drop_params": False,
        }
        if provider.protocol == "chat_completions":
            common: dict[str, Any] = {
                "model": model.api_model,
                "messages": [message.model_dump(mode="json") for message in request.messages],
                "n": 1,
                "max_completion_tokens": model.max_output_tokens,
                "reasoning_effort": model.requested_reasoning_effort,
                **controlled,
            }
        else:
            common = {
                "model": model.api_model,
                "input": [message.model_dump(mode="json") for message in request.messages],
                "max_output_tokens": model.max_output_tokens,
                "reasoning": {"effort": model.requested_reasoning_effort},
                "store": False,
                "truncation": "disabled",
                **controlled,
            }
        if model.temperature is not None:
            common["temperature"] = model.temperature
        if model.top_p is not None:
            common["top_p"] = model.top_p
        common["custom_llm_provider"] = provider.litellm_provider

        actual = {**common, "api_key": self._api_key}
        if self._base_url is not None:
            actual["base_url"] = self._base_url
        sanitized = {
            **_redact_json(common, self.artifact_secrets),
            "api_key_env": provider.api_key_env,
            "base_url_origin": _redact_provider_scalar(
                _sanitized_base_url(self._base_url),
                self.artifact_secrets,
            ),
            "base_url_sha256": sha256_json(self._base_url) if self._base_url else None,
        }
        self._assert_artifact_secret_free(
            sanitized,
            context="sanitized transport request",
        )
        return actual, sanitized

    def complete(
        self,
        request: LogicalRequest,
    ) -> NativeNormalizedResponse:
        validate_native_request(request, self.bundle)
        if not self.dependency_ready:
            failures = tuple(
                f"{check.name}: {check.detail}"
                for check in self.dependency_conformance.checks
                if check.status == "fail"
            )
            raise NativeDependencyReadinessError(
                "dependency blocked: " + "; ".join(failures)
            )

        actual_request, sanitized_request = self._transport_requests(request)
        attempt_id = uuid4().hex
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        try:
            if self.bundle.provider.protocol == "chat_completions":
                response = self.transport.chat_completion(**actual_request)
            else:
                response = self.transport.responses(**actual_request)
        except Exception as error:
            failed_at = datetime.now(timezone.utc)
            request_submitted = not isinstance(error, NativeUnsafeTransportState)
            if request_submitted:
                category, retryable = _error_category(error)
            else:
                category, retryable = ErrorCategory.INVALID_REQUEST, False
            status_code = getattr(error, "status_code", None)
            provider_code = getattr(error, "code", None)
            record = NativeNormalizedError(
                request_id=request.request_id,
                attempt_id=attempt_id,
                provider_id=self.bundle.provider.id,
                model_id=self.bundle.model.id,
                protocol=self.bundle.provider.protocol,
                category=category,
                http_status=(
                    status_code
                    if isinstance(status_code, int) and not isinstance(status_code, bool)
                    else None
                ),
                provider_code=(
                    _redact_provider_scalar(provider_code, self.artifact_secrets)
                ),
                provider_type=(
                    _redact_provider_scalar(
                        type(error).__name__,
                        self.artifact_secrets,
                    )
                    or "provider_error"
                ),
                sanitized_message=_redact_text(str(error), self.artifact_secrets),
                retryable=retryable,
                request_submitted=request_submitted,
                possibly_charged=request_submitted
                and category
                not in {
                    ErrorCategory.AUTHENTICATION,
                    ErrorCategory.PERMISSION,
                    ErrorCategory.INVALID_REQUEST,
                    ErrorCategory.UNSUPPORTED_PARAMETER,
                },
                reasoning=self.dependency_conformance.reasoning,
                started_at_utc=started_at,
                failed_at_utc=failed_at,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                logical_request_sha256=sha256_json(request),
                sanitized_transport_request=sanitized_request,
            )
            self._assert_artifact_secret_free(record, context="normalized provider error")
            raise NativeProviderCallError(record) from error

        try:
            return self._normalize_response(
                request=request,
                response=response,
                sanitized_request=sanitized_request,
                attempt_id=attempt_id,
                started_at=started_at,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except NativeMalformedProviderResponse as error:
            failed_at = datetime.now(timezone.utc)
            record = NativeNormalizedError(
                request_id=request.request_id,
                attempt_id=attempt_id,
                provider_id=self.bundle.provider.id,
                model_id=self.bundle.model.id,
                protocol=self.bundle.provider.protocol,
                category=ErrorCategory.MALFORMED_RESPONSE,
                provider_type=type(error).__name__,
                sanitized_message=_redact_text(str(error), self.artifact_secrets),
                retryable=False,
                request_submitted=True,
                possibly_charged=True,
                partial_usage=_partial_usage(
                    request,
                    response,
                    protocol=self.bundle.provider.protocol,
                    secrets=self.artifact_secrets,
                ),
                reasoning=self.dependency_conformance.reasoning,
                started_at_utc=started_at,
                failed_at_utc=failed_at,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                logical_request_sha256=sha256_json(request),
                sanitized_transport_request=sanitized_request,
            )
            self._assert_artifact_secret_free(record, context="normalized provider error")
            raise NativeProviderCallError(record) from error

    def _normalize_response(
        self,
        *,
        request: LogicalRequest,
        response: Any,
        sanitized_request: dict[str, Any],
        attempt_id: str,
        started_at: datetime,
        elapsed_ms: float,
    ) -> NativeNormalizedResponse:
        payload = _json_mapping(response)
        if self.bundle.provider.protocol == "chat_completions":
            content, finish_reason, provider_finish_reason = _chat_output(payload)
        else:
            content, finish_reason, provider_finish_reason = _responses_output(payload)
        if any(secret and secret in content for secret in self.artifact_secrets):
            raise NativeMalformedProviderResponse("response included credential material")

        usage = _usage(
            request,
            content,
            payload.get("usage"),
            protocol=self.bundle.provider.protocol,
            secrets=self.artifact_secrets,
        )
        hidden = getattr(response, "_hidden_params", None)
        hidden_mapping = dict(hidden) if isinstance(hidden, Mapping) else {}
        response_headers = getattr(response, "_response_headers", None)
        allowed_headers = {
            str(key).lower(): str(value)
            for key, value in (
                response_headers.items() if isinstance(response_headers, Mapping) else ()
            )
            if str(key).lower() in RESPONSE_HEADER_ALLOWLIST
        }
        raw_sdk_record = {
            "capture_fidelity": "sdk_object",
            "payload": _redact_json(payload, self.artifact_secrets),
            "litellm_hidden_params": _redact_json(hidden_mapping, self.artifact_secrets),
            "response_headers_allowlist": _redact_json(
                allowed_headers,
                self.artifact_secrets,
            ),
        }

        returned_model = payload.get("model")
        returned_model = returned_model if isinstance(returned_model, str) else None
        if (
            self.bundle.model.model_id_policy == "exact"
            and returned_model != self.bundle.model.expected_returned_model
        ):
            raise NativeMalformedProviderResponse(
                "endpoint conformance failure: returned model differs from exact manifest"
            )
        warnings: list[str] = []
        if not usage.provider_reported:
            warnings.append("provider usage was absent")
        if returned_model is None:
            warnings.append("provider did not report a model identifier")
        provider_request_id = payload.get("id")
        if not provider_request_id:
            provider_request_id = allowed_headers.get("x-request-id") or allowed_headers.get(
                "request-id"
            )
        system_fingerprint = payload.get("system_fingerprint")
        record = NativeNormalizedResponse(
            request_id=request.request_id,
            attempt_id=attempt_id,
            provider_request_id=_redact_provider_scalar(
                provider_request_id,
                self.artifact_secrets,
            ),
            provider_id=self.bundle.provider.id,
            model_id=self.bundle.model.id,
            protocol=self.bundle.provider.protocol,
            provider_manifest_sha256=self.bundle.provider_sha256,
            model_manifest_sha256=self.bundle.model_sha256,
            requested_model=self.bundle.model.api_model,
            returned_model=_redact_provider_scalar(
                returned_model,
                self.artifact_secrets,
            ),
            system_fingerprint=_redact_provider_scalar(
                system_fingerprint,
                self.artifact_secrets,
            ),
            text=content,
            finish_reason=_redact_provider_scalar(
                finish_reason,
                self.artifact_secrets,
            ),
            provider_finish_reason=_redact_provider_scalar(
                provider_finish_reason,
                self.artifact_secrets,
            ),
            usage=usage,
            resource_usage_complete=all(
                value is not None
                for value in (
                    usage.input_tokens,
                    usage.cached_input_tokens,
                    usage.output_tokens,
                    usage.reasoning_tokens,
                )
            ),
            reasoning=self.dependency_conformance.reasoning,
            started_at_utc=started_at,
            finished_at_utc=datetime.now(timezone.utc),
            elapsed_ms=elapsed_ms,
            logical_request_sha256=sha256_json(request),
            transport_request_sha256=sha256_json(sanitized_request),
            transport_response_sha256=sha256_json(raw_sdk_record),
            sanitized_transport_request=sanitized_request,
            raw_transport_response=raw_sdk_record,
            warnings=tuple(warnings),
        )
        self._assert_artifact_secret_free(record, context="normalized provider response")
        return record
