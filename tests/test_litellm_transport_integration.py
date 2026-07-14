from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from abstrak.providers.client import ProviderClient
from abstrak.providers.contracts import (
    ChatMessage,
    ErrorCategory,
    LogicalRequest,
    MessageRole,
    ProviderCallError,
)
from abstrak.providers.manifests import (
    GenerationConfig,
    ManifestBundle,
    ModelManifest,
    ProviderManifest,
)


class OpenAIStubHandler(BaseHTTPRequestHandler):
    response_status = 200
    request_count = 0
    request_path = ""
    request_headers: dict[str, str] = {}
    request_body: dict[str, Any] = {}

    def do_POST(self) -> None:  # noqa: N802
        type(self).request_count += 1
        type(self).request_path = self.path
        type(self).request_headers = {key.lower(): value for key, value in self.headers.items()}
        content_length = int(self.headers.get("content-length", "0"))
        type(self).request_body = json.loads(self.rfile.read(content_length))

        if type(self).response_status == 200:
            payload = {
                "id": "stub-response-1",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model-snapshot",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"finish","nonce":"integration"}',
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 5,
                    "total_tokens": 13,
                },
            }
        else:
            payload = {
                "error": {
                    "message": "stub provider failure",
                    "type": "server_error",
                    "code": "server_error",
                }
            }
        body = json.dumps(payload).encode()
        self.send_response(type(self).response_status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.send_header("x-request-id", "stub-http-request-1")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def openai_stub(status: int) -> Iterator[tuple[ThreadingHTTPServer, type[OpenAIStubHandler]]]:
    class Handler(OpenAIStubHandler):
        response_status = status
        request_count = 0
        request_path = ""
        request_headers = {}
        request_body = {}

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _bundle() -> ManifestBundle:
    provider = ProviderManifest(
        id="stub-provider",
        litellm_provider="openai",
        api_key_env="STUB_API_KEY",
        base_url_env="STUB_BASE_URL",
        timeout_seconds=5,
    )
    model = ModelManifest(
        id="stub-model",
        provider="stub-provider",
        api_model="openai/test-model-snapshot",
        model_id_policy="exact",
        expected_returned_model="test-model-snapshot",
        generation=GenerationConfig(max_completion_tokens=32),
    )
    return ManifestBundle(provider=provider, model=model)


def _request() -> LogicalRequest:
    return LogicalRequest(
        model_ref="stub-model",
        messages=(ChatMessage(role=MessageRole.USER, content="Return one JSON object."),),
    )


def test_real_litellm_transport_preserves_single_request() -> None:
    with openai_stub(200) as (server, handler):
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        client = ProviderClient(
            _bundle(),
            environment={"STUB_API_KEY": "stub-secret", "STUB_BASE_URL": base_url},
        )

        response = client.complete(_request())

    assert handler.request_count == 1
    assert handler.request_path == "/v1/chat/completions"
    assert handler.request_headers["authorization"] == "Bearer stub-secret"
    assert handler.request_body["model"] == "test-model-snapshot"
    assert handler.request_body.get("stream", False) is False
    assert response.returned_model == "test-model-snapshot"
    assert response.usage.total_tokens == 13
    assert client.transport.call_count == 1
    recorded_response = json.dumps(response.raw_transport_response)
    assert "stub-secret" not in recorded_response
    assert base_url not in recorded_response


def test_real_litellm_transport_does_not_retry_server_error() -> None:
    with openai_stub(500) as (server, handler):
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        client = ProviderClient(
            _bundle(),
            environment={"STUB_API_KEY": "stub-secret", "STUB_BASE_URL": base_url},
        )

        with pytest.raises(ProviderCallError) as captured:
            client.complete(_request())

    assert handler.request_count == 1
    assert client.transport.call_count == 1
    assert captured.value.record.category == ErrorCategory.SERVER_ERROR
