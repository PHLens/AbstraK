"""Minimal native provider calls for the exploratory KernelBench agent loop."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from abstrak.evaluation.agent_contracts import AgentGenerationConfig, AgentModelSpec
from abstrak.providers.contracts import ChatMessage
from abstrak.providers.native_transport import LiteLLMNativeTransport, NativeTransport


class AgentProviderError(RuntimeError):
    def __init__(self, message: str, *, elapsed_ms: float) -> None:
        super().__init__(message)
        self.elapsed_ms = elapsed_ms


class AgentCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    protocol: str
    provider_request_id: str | None = None
    returned_model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    elapsed_ms: float = Field(ge=0)
    sanitized_request: dict[str, Any]
    raw_response: dict[str, Any]


class AgentCompletionClient(Protocol):
    def complete(self, messages: Sequence[ChatMessage]) -> AgentCompletion: ...


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        payload = dict(value)
    elif hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")
    else:
        raise ValueError(f"provider response is not mapping-like: {type(value).__name__}")
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str))


def _optional_int(mapping: Mapping[str, Any], *paths: tuple[str, ...]) -> int | None:
    for path in paths:
        value: Any = mapping
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _chat_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("chat response has no choice")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ValueError("chat response has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("chat response has no text")
    return content


def _responses_text(payload: Mapping[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    output = payload.get("output")
    if not isinstance(output, list):
        raise ValueError("Responses payload has no output")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "output_text":
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
    joined = "".join(texts)
    if not joined:
        raise ValueError("Responses payload has no output text")
    return joined


class PilotProviderClient:
    """Make one native call without formal-study conformance machinery."""

    def __init__(
        self,
        model: AgentModelSpec,
        generation: AgentGenerationConfig,
        *,
        environment: Mapping[str, str],
        transport: NativeTransport | None = None,
    ) -> None:
        self.model = model
        self.generation = generation
        self.transport = transport if transport is not None else LiteLLMNativeTransport()
        try:
            self._api_key = environment[model.api_key_env]
        except KeyError as error:
            raise ValueError(f"missing environment variable: {model.api_key_env}") from error
        if not self._api_key:
            raise ValueError(f"environment variable is empty: {model.api_key_env}")
        if model.base_url_env is None:
            self._base_url = None
        else:
            try:
                self._base_url = environment[model.base_url_env]
            except KeyError as error:
                raise ValueError(f"missing environment variable: {model.base_url_env}") from error
            if not self._base_url:
                raise ValueError(f"environment variable is empty: {model.base_url_env}")

    @property
    def artifact_secrets(self) -> tuple[str, ...]:
        return self._api_key, self._base_url or ""

    def _request(self, messages: Sequence[ChatMessage]) -> tuple[dict[str, Any], dict[str, Any]]:
        rendered = [message.model_dump(mode="json") for message in messages]
        common: dict[str, Any] = {
            "model": self.model.api_model,
            "stream": False,
            "timeout": self.model.timeout_seconds,
            "num_retries": 0,
            "max_retries": 0,
            "custom_llm_provider": self.model.litellm_provider,
        }
        if self.model.protocol == "chat_completions":
            common.update(
                {
                    "messages": rendered,
                    "n": 1,
                    "max_completion_tokens": self.generation.max_output_tokens,
                    "reasoning_effort": self.generation.reasoning_effort,
                }
            )
        else:
            common.update(
                {
                    "input": rendered,
                    "max_output_tokens": self.generation.max_output_tokens,
                    "reasoning": {"effort": self.generation.reasoning_effort},
                    "store": False,
                }
            )
        actual = {**common, "api_key": self._api_key}
        if self._base_url is not None:
            actual["base_url"] = self._base_url
        sanitized = {
            **common,
            "api_key_env": self.model.api_key_env,
            "base_url_env": self.model.base_url_env,
        }
        return actual, sanitized

    def complete(self, messages: Sequence[ChatMessage]) -> AgentCompletion:
        actual, sanitized = self._request(messages)
        started = time.perf_counter()
        try:
            if self.model.protocol == "chat_completions":
                response = self.transport.chat_completion(**actual)
            else:
                response = self.transport.responses(**actual)
            elapsed_ms = (time.perf_counter() - started) * 1000
            payload = _mapping(response)
            text = (
                _chat_text(payload)
                if self.model.protocol == "chat_completions"
                else _responses_text(payload)
            )
        except Exception as error:
            elapsed_ms = (time.perf_counter() - started) * 1000
            message = str(error).replace(self._api_key, "<redacted>")
            if self._base_url:
                message = message.replace(self._base_url, "<redacted>")
            raise AgentProviderError(
                f"{type(error).__name__}: {message[:2000]}", elapsed_ms=elapsed_ms
            ) from error

        usage = payload.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        return AgentCompletion(
            text=text,
            protocol=self.model.protocol,
            provider_request_id=(
                str(payload["id"]) if payload.get("id") is not None else None
            ),
            returned_model=(
                str(payload["model"]) if payload.get("model") is not None else None
            ),
            input_tokens=_optional_int(usage, ("input_tokens",), ("prompt_tokens",)),
            cached_input_tokens=_optional_int(
                usage,
                ("input_tokens_details", "cached_tokens"),
                ("prompt_tokens_details", "cached_tokens"),
            ),
            output_tokens=_optional_int(usage, ("output_tokens",), ("completion_tokens",)),
            reasoning_tokens=_optional_int(
                usage,
                ("output_tokens_details", "reasoning_tokens"),
                ("completion_tokens_details", "reasoning_tokens"),
            ),
            elapsed_ms=elapsed_ms,
            sanitized_request=sanitized,
            raw_response=payload,
        )
