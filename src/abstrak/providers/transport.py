"""Injectable completion transports with controlled LiteLLM behavior."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class CompletionTransport(Protocol):
    call_count: int

    def completion(self, **kwargs: Any) -> Any: ...


class UnsafeTransportState(RuntimeError):
    pass


class LiteLLMTransport:
    """Use LiteLLM as a single-call transport, never as an agent or router."""

    def __init__(self, completion_fn: Callable[..., Any] | None = None) -> None:
        import litellm

        self._litellm = litellm
        self._completion_fn = completion_fn or litellm.completion
        self.call_count = 0

    def _assert_controlled_globals(self) -> None:
        checks = {
            "model_fallbacks": getattr(self._litellm, "model_fallbacks", None),
            "model_alias_map": getattr(self._litellm, "model_alias_map", None),
            "context_window_fallbacks": getattr(self._litellm, "context_window_fallbacks", None),
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
            raise UnsafeTransportState(f"controlled transport forbids global state: {names}")
        if getattr(self._litellm, "drop_params", False):
            raise UnsafeTransportState("controlled transport forbids drop_params=True")
        if getattr(self._litellm, "num_retries", None) not in (None, 0):
            raise UnsafeTransportState("controlled transport forbids global retries")

    def completion(self, **kwargs: Any) -> Any:
        self._assert_controlled_globals()
        self.call_count += 1
        return self._completion_fn(**kwargs)
