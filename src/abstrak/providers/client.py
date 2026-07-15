"""Single-attempt provider client and LiteLLM response normalization."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import platform
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from pydantic import BaseModel

from abstrak.providers.contracts import (
    ErrorCategory,
    LogicalRequest,
    MalformedProviderResponse,
    NormalizedCost,
    NormalizedError,
    NormalizedResponse,
    NormalizedUsage,
    ProviderCallError,
    sha256_json,
)
from abstrak.providers.manifests import (
    ManifestBundle,
    manifest_sha256,
    resolve_environment,
)
from abstrak.providers.transport import (
    CompletionTransport,
    LiteLLMTransport,
    UnsafeTransportState,
)

RESPONSE_HEADER_ALLOWLIST = {
    "date",
    "request-id",
    "x-request-id",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class ProviderConfigurationError(ValueError):
    """Raised when resolved provider environment values are unsafe or invalid."""


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise MalformedProviderResponse(
            f"expected mapping-like response, received {type(value).__name__}"
        )
    try:
        return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise MalformedProviderResponse(f"response is not JSON serializable: {error}") from error


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


def _validate_base_url(base_url: str | None) -> None:
    if base_url is None:
        return
    parts = urlsplit(base_url)
    try:
        _ = parts.port
    except ValueError as error:
        raise ProviderConfigurationError("provider base URL has an invalid port") from error
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ProviderConfigurationError("provider base URL must be an absolute HTTP(S) URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ProviderConfigurationError(
            "provider base URL cannot contain userinfo, query parameters, or fragments"
        )
    is_loopback = parts.hostname == "localhost"
    try:
        is_loopback = is_loopback or ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        pass
    if parts.scheme != "https" and not is_loopback:
        raise ProviderConfigurationError("remote provider base URLs must use HTTPS")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_state_bytes(root: Path, status: bytes, diff: bytes) -> bytes:
    state = bytearray(status)
    state.extend(diff)
    untracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
        timeout=5,
    ).stdout.split(b"\0")
    for encoded_path in sorted(path for path in untracked if path):
        state.extend(encoded_path)
        path = root / encoded_path.decode(errors="surrogateescape")
        if path.is_file():
            state.extend(path.read_bytes())
    return bytes(state)


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            timeout=5,
        ).stdout
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
            timeout=5,
        ).stdout
        source_state = _source_state_bytes(root, status, diff)
        lock_path = root / "uv.lock"
        lock_sha256 = _sha256_bytes(lock_path.read_bytes()) if lock_path.is_file() else None
        dirty = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return {
            "commit": None,
            "worktree_dirty": None,
            "source_state_sha256": None,
            "uv_lock_sha256": None,
        }
    return {
        "commit": commit,
        "worktree_dirty": dirty,
        "source_state_sha256": _sha256_bytes(source_state),
        "uv_lock_sha256": lock_sha256,
    }


def _redact_text(text: str, secrets: tuple[str, ...]) -> str:
    sanitized = text
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, "<redacted>")
    return sanitized[:4000]


def _redact_json(value: Any, secrets: tuple[str, ...]) -> Any:
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    for secret in secrets:
        if secret:
            serialized = serialized.replace(secret, "<redacted>")
    return json.loads(serialized)


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


class ProviderClient:
    """Execute one logical request as exactly one transport call."""

    def __init__(
        self,
        bundle: ManifestBundle,
        *,
        transport: CompletionTransport | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.bundle = bundle
        self.transport = transport or LiteLLMTransport()
        self._api_key, self._base_url = resolve_environment(bundle.provider, environment)
        _validate_base_url(self._base_url)
        self.provider_manifest_sha256 = manifest_sha256(bundle.provider)
        self.model_manifest_sha256 = manifest_sha256(bundle.model)

    @property
    def artifact_secrets(self) -> tuple[str, ...]:
        path_secrets: tuple[str, ...] = ()
        if self._base_url:
            segments = tuple(
                segment
                for segment in urlsplit(self._base_url).path.split("/")
                if len(segment) >= 16
            )
            path_secrets = segments
        return self._api_key, self._base_url or "", *path_secrets

    @property
    def resolved_manifest_record(self) -> dict[str, Any]:
        return {
            "provider": self.bundle.provider.model_dump(mode="json"),
            "model": self.bundle.model.model_dump(mode="json"),
            "provider_manifest_sha256": self.provider_manifest_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "endpoint_origin": _sanitized_base_url(self._base_url),
            "endpoint_sha256": sha256_json(self._base_url) if self._base_url else None,
            "capture_fidelity": "sdk_object",
            "runtime": {
                "python": platform.python_version(),
                "abstrak": _package_version("abstrak"),
                "litellm": _package_version("litellm"),
                "openai": _package_version("openai"),
                "httpx": _package_version("httpx"),
                "pydantic": _package_version("pydantic"),
                "git": _git_state(REPOSITORY_ROOT),
            },
        }

    def _transport_requests(self, request: LogicalRequest) -> tuple[dict[str, Any], dict[str, Any]]:
        if request.model_ref != self.bundle.model.id:
            raise ValueError(
                f"request model_ref {request.model_ref!r} does not match {self.bundle.model.id!r}"
            )
        provider = self.bundle.provider
        model = self.bundle.model
        messages = [message.model_dump(mode="json") for message in request.messages]
        common: dict[str, Any] = {
            "model": model.api_model,
            "messages": messages,
            "stream": False,
            "n": 1,
            "timeout": provider.timeout_seconds,
            "num_retries": 0,
            "max_retries": 0,
            "retry_policy": None,
            "context_window_fallback_dict": {},
            "caching": False,
            **model.generation.transport_parameters(),
        }
        if provider.litellm_provider is not None:
            common["custom_llm_provider"] = provider.litellm_provider

        actual = {**common, "api_key": self._api_key}
        if self._base_url is not None:
            actual["base_url"] = self._base_url

        sanitized = {
            **common,
            "api_key_env": provider.api_key_env,
            "base_url_origin": _sanitized_base_url(self._base_url),
            "base_url_sha256": sha256_json(self._base_url) if self._base_url else None,
        }
        return actual, sanitized

    def complete(self, request: LogicalRequest) -> NormalizedResponse:
        actual_request, sanitized_request = self._transport_requests(request)
        attempt_id = uuid4().hex
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        try:
            response = self.transport.completion(**actual_request)
        except Exception as error:
            failed_at = datetime.now(timezone.utc)
            category, retryable = _error_category(error)
            status_code = getattr(error, "status_code", None)
            request_submitted = not isinstance(error, UnsafeTransportState)
            if not request_submitted:
                category, retryable = ErrorCategory.INVALID_REQUEST, False
            record = NormalizedError(
                request_id=request.request_id,
                attempt_id=attempt_id,
                provider_id=self.bundle.provider.id,
                model_id=self.bundle.model.id,
                category=category,
                http_status=status_code if isinstance(status_code, int) else None,
                provider_code=str(getattr(error, "code", "")) or None,
                provider_type=type(error).__name__,
                sanitized_message=_redact_text(str(error), (self._api_key, self._base_url or "")),
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
                started_at_utc=started_at,
                failed_at_utc=failed_at,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                logical_request_sha256=sha256_json(request),
                sanitized_transport_request=sanitized_request,
            )
            raise ProviderCallError(record) from error

        try:
            return self._normalize_response(
                request=request,
                response=response,
                sanitized_request=sanitized_request,
                attempt_id=attempt_id,
                started_at=started_at,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except MalformedProviderResponse as error:
            failed_at = datetime.now(timezone.utc)
            record = NormalizedError(
                request_id=request.request_id,
                attempt_id=attempt_id,
                provider_id=self.bundle.provider.id,
                model_id=self.bundle.model.id,
                category=ErrorCategory.MALFORMED_RESPONSE,
                provider_type=type(error).__name__,
                sanitized_message=str(error),
                retryable=False,
                request_submitted=True,
                possibly_charged=True,
                started_at_utc=started_at,
                failed_at_utc=failed_at,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                logical_request_sha256=sha256_json(request),
                sanitized_transport_request=sanitized_request,
            )
            raise ProviderCallError(record) from error

    def _normalize_response(
        self,
        *,
        request: LogicalRequest,
        response: Any,
        sanitized_request: dict[str, Any],
        attempt_id: str,
        started_at: datetime,
        elapsed_ms: float,
    ) -> NormalizedResponse:
        payload = _json_mapping(response)
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise MalformedProviderResponse("response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise MalformedProviderResponse("response choice must be a mapping")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise MalformedProviderResponse("response choice has no message")
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise MalformedProviderResponse("response message content must be non-empty text")
        if any(secret and secret in content for secret in self.artifact_secrets):
            raise MalformedProviderResponse("response included credential material")

        raw_usage = payload.get("usage")
        usage_mapping = raw_usage if isinstance(raw_usage, Mapping) else None
        input_tokens = _nested_int(usage_mapping or {}, "prompt_tokens")
        output_tokens = _nested_int(usage_mapping or {}, "completion_tokens")
        total_tokens = _nested_int(usage_mapping or {}, "total_tokens")
        cached_tokens = _nested_int(usage_mapping or {}, "prompt_tokens_details", "cached_tokens")
        reasoning_tokens = _nested_int(
            usage_mapping or {}, "completion_tokens_details", "reasoning_tokens"
        )
        provider_reported = any(
            value is not None for value in (input_tokens, output_tokens, total_tokens)
        )
        core_fields_complete = all(
            value is not None for value in (input_tokens, output_tokens, total_tokens)
        )
        usage = NormalizedUsage(
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            input_characters=sum(len(message.content) for message in request.messages),
            output_characters=len(content),
            provider_reported=provider_reported,
            core_fields_complete=core_fields_complete,
            raw_usage=dict(usage_mapping) if usage_mapping is not None else None,
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
            "response_headers_allowlist": _redact_json(allowed_headers, self.artifact_secrets),
        }

        returned_model = payload.get("model")
        returned_model = returned_model if isinstance(returned_model, str) else None
        warnings: list[str] = []
        if not provider_reported:
            warnings.append("provider usage was absent or indistinguishable from all-zero usage")
        if returned_model is None:
            warnings.append("provider did not report a model identifier")

        finished_at = datetime.now(timezone.utc)
        provider_request_id = payload.get("id")
        if not provider_request_id:
            provider_request_id = allowed_headers.get("x-request-id") or allowed_headers.get(
                "request-id"
            )
        return NormalizedResponse(
            request_id=request.request_id,
            attempt_id=attempt_id,
            provider_request_id=str(provider_request_id) if provider_request_id else None,
            provider_id=self.bundle.provider.id,
            model_id=self.bundle.model.id,
            provider_manifest_sha256=self.provider_manifest_sha256,
            model_manifest_sha256=self.model_manifest_sha256,
            requested_model=self.bundle.model.api_model,
            returned_model=returned_model,
            system_fingerprint=(
                str(payload["system_fingerprint"])
                if payload.get("system_fingerprint") is not None
                else None
            ),
            text=content,
            finish_reason=(
                str(choice["finish_reason"]) if choice.get("finish_reason") is not None else None
            ),
            provider_finish_reason=(
                str(choice["finish_reason"]) if choice.get("finish_reason") is not None else None
            ),
            usage=usage,
            cost=NormalizedCost(),
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            elapsed_ms=elapsed_ms,
            logical_request_sha256=sha256_json(request),
            transport_request_sha256=sha256_json(sanitized_request),
            transport_response_sha256=sha256_json(raw_sdk_record),
            sanitized_transport_request=sanitized_request,
            raw_transport_response=raw_sdk_record,
            warnings=tuple(warnings),
        )
