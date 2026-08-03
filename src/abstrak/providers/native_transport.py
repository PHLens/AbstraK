"""Controlled LiteLLM transport with explicit Chat and Responses entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class NativeTransport(Protocol):
    call_count: int

    def chat_completion(self, **kwargs: Any) -> Any: ...

    def responses(self, **kwargs: Any) -> Any: ...


class NativeUnsafeTransportState(RuntimeError):
    pass


class LiteLLMNativeTransport:
    """Make exactly one call through the manifest-selected native protocol."""

    def __init__(
        self,
        *,
        completion_fn: Callable[..., Any] | None = None,
        responses_fn: Callable[..., Any] | None = None,
    ) -> None:
        import litellm

        self._litellm = litellm
        self._completion_fn = (
            completion_fn if completion_fn is not None else litellm.completion
        )
        self._responses_fn = responses_fn if responses_fn is not None else litellm.responses
        self.call_count = 0
        self.call_protocols: list[str] = []

    def _assert_controlled_globals(self) -> None:
        checks = {
            "model_fallbacks": getattr(self._litellm, "model_fallbacks", None),
            "model_alias_map": getattr(self._litellm, "model_alias_map", None),
            "context_window_fallbacks": getattr(
                self._litellm,
                "context_window_fallbacks",
                None,
            ),
            "cache": getattr(self._litellm, "cache", None),
            "callbacks": getattr(self._litellm, "callbacks", None),
            "success_callback": getattr(self._litellm, "success_callback", None),
            "failure_callback": getattr(self._litellm, "failure_callback", None),
            "input_callback": getattr(self._litellm, "input_callback", None),
            "async_input_callback": getattr(self._litellm, "_async_input_callback", None),
            "async_success_callback": getattr(self._litellm, "_async_success_callback", None),
            "async_failure_callback": getattr(self._litellm, "_async_failure_callback", None),
            "service_callback": getattr(self._litellm, "service_callback", None),
            "audit_log_callbacks": getattr(self._litellm, "audit_log_callbacks", None),
            "callback_settings": getattr(self._litellm, "callback_settings", None),
            "pre_call_rules": getattr(self._litellm, "pre_call_rules", None),
            "post_call_rules": getattr(self._litellm, "post_call_rules", None),
            "proxy_auth": getattr(self._litellm, "proxy_auth", None),
        }
        configured = {name: value for name, value in checks.items() if value}
        if configured:
            names = ", ".join(sorted(configured))
            raise NativeUnsafeTransportState(
                f"controlled native transport forbids global state: {names}"
            )
        if getattr(self._litellm, "drop_params", False):
            raise NativeUnsafeTransportState("controlled transport forbids drop_params=True")
        if getattr(self._litellm, "num_retries", None) not in (None, 0):
            raise NativeUnsafeTransportState("controlled transport forbids global retries")

    def _call(self, protocol: str, function: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        self._assert_controlled_globals()
        self.call_count += 1
        self.call_protocols.append(protocol)
        return function(**kwargs)

    def chat_completion(self, **kwargs: Any) -> Any:
        return self._call("chat_completions", self._completion_fn, kwargs)

    def responses(self, **kwargs: Any) -> Any:
        return self._call("responses", self._responses_fn, kwargs)
