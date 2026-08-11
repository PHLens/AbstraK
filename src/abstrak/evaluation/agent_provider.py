"""Minimal native provider calls for the exploratory KernelBench agent loop."""

from __future__ import annotations

import contextlib
import json
import signal
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from abstrak.evaluation.agent_contracts import AgentGenerationConfig, AgentModelSpec
from abstrak.providers.contracts import MessageRole
from abstrak.providers.native_transport import LiteLLMNativeTransport, NativeTransport


@dataclass(frozen=True)
class AgentUsage:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None


class AgentProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        elapsed_ms: float,
        raw_response: dict[str, Any] | None = None,
        sanitized_request: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.elapsed_ms = elapsed_ms
        self.raw_response = raw_response
        self.sanitized_request = sanitized_request
        self.usage = extract_agent_usage(raw_response)


class AgentOutputTruncated(AgentProviderError):
    """The model used its output budget before emitting a final answer."""


class AgentMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: MessageRole
    content: str = Field(min_length=1)
    reasoning_content: str | None = Field(default=None, min_length=1)


class AgentCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    reasoning_content: str | None = Field(default=None, min_length=1)
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
    def complete(
        self,
        messages: Sequence[AgentMessage],
        *,
        progress: Callable[[str], None] | None = None,
    ) -> AgentCompletion: ...


class _ChatOutputTruncated(ValueError):
    pass


class _ProviderWallClockTimeout(BaseException):
    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(f"provider wall-clock deadline exceeded {timeout_seconds:g}s")


class _ChatStreamInterrupted(Exception):
    def __init__(self, cause: BaseException, partial_response: dict[str, Any]) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.partial_response = partial_response


@contextlib.contextmanager
def _provider_wall_clock_deadline(timeout_seconds: float) -> Iterator[None]:
    """Interrupt a stuck native request instead of relying on its read timeout alone."""

    if (
        not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
        or not hasattr(signal, "getitimer")
        or not hasattr(signal, "ITIMER_REAL")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    if previous_delay > 0 or previous_interval > 0:
        yield
        return

    def expire(_signum: int, _frame: Any) -> None:
        raise _ProviderWallClockTimeout(timeout_seconds)

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


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


def extract_agent_usage(payload: Mapping[str, Any] | None) -> AgentUsage:
    if payload is None:
        return AgentUsage()
    usage = payload.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    return AgentUsage(
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
    )


def _chat_output(payload: Mapping[str, Any]) -> tuple[str, str | None]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("chat response has no choice")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ValueError("chat response has no message")
    finish_reason = choices[0].get("finish_reason")
    if finish_reason == "length":
        raise _ChatOutputTruncated(
            "chat response exhausted max_tokens before completing final text "
            "(finish_reason=length)"
        )
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError(f"chat response has no text (finish_reason={finish_reason})")
    reasoning_content = message.get("reasoning_content")
    if not isinstance(reasoning_content, str) or not reasoning_content:
        provider_fields = message.get("provider_specific_fields")
        reasoning_content = (
            provider_fields.get("reasoning_content")
            if isinstance(provider_fields, Mapping)
            else None
        )
    if not isinstance(reasoning_content, str) or not reasoning_content:
        reasoning_content = None
    return content, reasoning_content


def _stream_text(delta: Mapping[str, Any], name: str) -> str:
    value = delta.get(name)
    if isinstance(value, str):
        return value
    provider_fields = delta.get("provider_specific_fields")
    if isinstance(provider_fields, Mapping):
        value = provider_fields.get(name)
        if isinstance(value, str):
            return value
    return ""


def _aggregate_chat_stream(
    response: Any,
    *,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    # Test transports may return a completed response even when the request asks for a stream.
    if isinstance(response, (BaseModel, Mapping)) or hasattr(response, "model_dump"):
        return _mapping(response)
    try:
        chunks = iter(response)
    except TypeError as error:
        raise ValueError("chat streaming response is not iterable") from error

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    response_id: str | None = None
    returned_model: str | None = None
    created: Any = None
    system_fingerprint: Any = None
    finish_reason: str | None = None
    chunk_count = 0
    last_progress_at: float | None = None

    def stream_metadata(*, completed: bool) -> dict[str, Any]:
        return {
            "chunk_count": chunk_count,
            "reasoning_chars": sum(map(len, reasoning_parts)),
            "content_chars": sum(map(len, content_parts)),
            "completed": completed,
        }

    def partial_response() -> dict[str, Any]:
        return {
            "id": response_id,
            "created": created,
            "model": returned_model,
            "object": "chat.completion.stream.aggregate",
            "system_fingerprint": system_fingerprint,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": ""},
                }
            ],
            "usage": usage,
            "stream": stream_metadata(completed=False),
        }

    def close_stream() -> None:
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def report(*, final: bool = False) -> None:
        nonlocal last_progress_at
        if progress is None:
            return
        now = time.monotonic()
        if not final and last_progress_at is not None and now - last_progress_at < 10.0:
            return
        state = "completed" if final else "progress"
        progress(
            f"stream {state} chunks={chunk_count} "
            f"reasoning_chars={sum(map(len, reasoning_parts))} "
            f"content_chars={sum(map(len, content_parts))}"
        )
        last_progress_at = now

    try:
        for chunk in chunks:
            chunk_count += 1
            payload = _mapping(chunk)
            if response_id is None and payload.get("id") is not None:
                response_id = str(payload["id"])
            if returned_model is None and payload.get("model") is not None:
                returned_model = str(payload["model"])
            if created is None:
                created = payload.get("created")
            if system_fingerprint is None:
                system_fingerprint = payload.get("system_fingerprint")
            chunk_usage = payload.get("usage")
            if isinstance(chunk_usage, Mapping):
                usage = dict(chunk_usage)

            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
                continue
            choice = choices[0]
            if isinstance(choice.get("finish_reason"), str):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            reasoning_delta = _stream_text(delta, "reasoning_content")
            content_delta = _stream_text(delta, "content")
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
            if content_delta:
                content_parts.append(content_delta)
            if reasoning_delta or content_delta:
                report()
    except _ProviderWallClockTimeout as error:
        close_stream()
        raise _ChatStreamInterrupted(error, partial_response()) from error
    except Exception as error:
        close_stream()
        raise _ChatStreamInterrupted(error, partial_response()) from error

    if chunk_count == 0:
        raise ValueError("chat stream returned no chunks")
    report(final=True)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    reasoning_content = "".join(reasoning_parts)
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    return {
        "id": response_id,
        "created": created,
        "model": returned_model,
        "object": "chat.completion.stream.aggregate",
        "system_fingerprint": system_fingerprint,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": message,
            }
        ],
        "usage": usage,
        "stream": stream_metadata(completed=True),
    }


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


def _check_responses_completion(payload: Mapping[str, Any]) -> None:
    status = payload.get("status")
    if status != "incomplete":
        return
    details = payload.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, Mapping) else None
    if reason in {"max_output_tokens", "max_tokens"}:
        raise _ChatOutputTruncated(
            f"Responses output exhausted its budget before completion (reason={reason})"
        )
    raise ValueError(f"Responses output is incomplete (reason={reason})")


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

    def _request(self, messages: Sequence[AgentMessage]) -> tuple[dict[str, Any], dict[str, Any]]:
        rendered = [message.model_dump(mode="json", exclude_none=True) for message in messages]
        stream_chat = self.model.protocol == "chat_completions"
        common: dict[str, Any] = {
            "model": self.model.api_model,
            "stream": stream_chat,
            "timeout": self.model.timeout_seconds,
            "num_retries": 0,
            "max_retries": 0,
            "custom_llm_provider": self.model.litellm_provider,
        }
        if self.model.protocol == "chat_completions":
            common.update({"messages": rendered, "n": 1, "stream_options": {"include_usage": True}})
            if self.model.litellm_provider == "deepseek":
                # LiteLLM 1.92 collapses top-level effort to binary thinking mode, so retain
                # DeepSeek's native effort through extra_body.
                common.update(
                    {
                        "max_tokens": self.generation.max_output_tokens,
                        "extra_body": {
                            "thinking": {"type": "enabled"},
                            "reasoning_effort": "max",
                        },
                    }
                )
            else:
                common.update(
                    {
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

    def complete(
        self,
        messages: Sequence[AgentMessage],
        *,
        progress: Callable[[str], None] | None = None,
    ) -> AgentCompletion:
        actual, sanitized = self._request(messages)
        started = time.perf_counter()
        payload: dict[str, Any] | None = None
        try:
            with _provider_wall_clock_deadline(self.model.timeout_seconds):
                if self.model.protocol == "chat_completions":
                    response = self.transport.chat_completion(**actual)
                    payload = _aggregate_chat_stream(response, progress=progress)
                else:
                    response = self.transport.responses(**actual)
                    payload = _mapping(response)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if self.model.protocol == "chat_completions":
                text, reasoning_content = _chat_output(payload)
            else:
                _check_responses_completion(payload)
                text = _responses_text(payload)
                reasoning_content = None
        except _ChatStreamInterrupted as error:
            elapsed_ms = (time.perf_counter() - started) * 1000
            cause = error.cause
            message = str(cause).replace(self._api_key, "<redacted>")
            if self._base_url:
                message = message.replace(self._base_url, "<redacted>")
            raise AgentProviderError(
                f"{type(cause).__name__}: {message[:2000]}",
                elapsed_ms=elapsed_ms,
                raw_response=error.partial_response,
                sanitized_request=sanitized,
            ) from error
        except _ProviderWallClockTimeout as error:
            elapsed_ms = (time.perf_counter() - started) * 1000
            raise AgentProviderError(
                str(error),
                elapsed_ms=elapsed_ms,
                raw_response=payload,
                sanitized_request=sanitized,
            ) from error
        except _ChatOutputTruncated as error:
            elapsed_ms = (time.perf_counter() - started) * 1000
            raise AgentOutputTruncated(
                str(error),
                elapsed_ms=elapsed_ms,
                raw_response=payload,
                sanitized_request=sanitized,
            ) from error
        except Exception as error:
            elapsed_ms = (time.perf_counter() - started) * 1000
            message = str(error).replace(self._api_key, "<redacted>")
            if self._base_url:
                message = message.replace(self._base_url, "<redacted>")
            raise AgentProviderError(
                f"{type(error).__name__}: {message[:2000]}",
                elapsed_ms=elapsed_ms,
                raw_response=payload,
                sanitized_request=sanitized,
            ) from error

        usage = extract_agent_usage(payload)
        return AgentCompletion(
            text=text,
            reasoning_content=reasoning_content,
            protocol=self.model.protocol,
            provider_request_id=(str(payload["id"]) if payload.get("id") is not None else None),
            returned_model=(str(payload["model"]) if payload.get("model") is not None else None),
            input_tokens=usage.input_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            elapsed_ms=elapsed_ms,
            sanitized_request=sanitized,
            raw_response=payload,
        )
