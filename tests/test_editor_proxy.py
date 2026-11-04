"""ASGI/HTTP transport tests with a simulated upstream; no live model claims."""

from __future__ import annotations

import copy
import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from iwibench.editor_proxy import UPSTREAM_URL, create_app


@asynccontextmanager
async def proxy(handler):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        app = create_app(upstream)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                yield client


def example_request(model="gpt-5.1"):
    return {
        "model": model,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": "Use the supplied tools."},
            {"role": "user", "content": "Inspect the tensor."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_weights", "arguments": '{"tensor":"embedding"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"shape":[10,2]}'},
        ],
        "tools": [{"type": "function", "function": {"name": "read_weights", "description": "Read a tensor", "parameters": {"type": "object", "properties": {"tensor": {"type": "string"}}, "required": ["tensor"], "additionalProperties": False}, "strict": True}}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "result", "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}, "strict": True}},
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "reasoning_effort": "low",
        "stream": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-5.1", "gpt-6.1-terra"])
async def test_maps_4096_and_preserves_messages_tools_schema_auth(model):
    original = example_request(model)
    expected = copy.deepcopy(original)
    expected["max_completion_tokens"] = expected.pop("max_tokens")
    requests = []

    def upstream(request):
        requests.append(request)
        assert str(request.url) == UPSTREAM_URL
        assert request.method == "POST"
        assert request.headers["authorization"] == "Bearer transport-test-only"
        assert json.loads(request.content) == expected
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}, headers={"x-request-id": "test-request"})

    async with proxy(upstream) as client:
        response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer transport-test-only"}, json=original)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert response.headers["x-request-id"] == "test-request"
    assert original["max_tokens"] == 4096
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4.1", "other/gpt-5.1", "o3"])
async def test_other_model_payload_unchanged(model):
    payload = example_request(model)

    def upstream(request):
        assert json.loads(request.content) == payload
        return httpx.Response(200, json={"choices": []})

    async with proxy(upstream) as client:
        response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer test"}, json=payload)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_equal_caps_are_merged_and_existing_new_cap_is_preserved():
    seen = []

    def upstream(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": []})

    async with proxy(upstream) as client:
        for payload in [
            {"model": "gpt-5", "max_tokens": 4096, "max_completion_tokens": 4096},
            {"model": "gpt-5", "max_completion_tokens": 4096},
        ]:
            response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer test"}, json=payload)
            assert response.status_code == 200
    assert seen == [{"model": "gpt-5", "max_completion_tokens": 4096}] * 2


@pytest.mark.asyncio
async def test_conflicting_caps_rejected_without_upstream_request():
    def upstream(request):
        raise AssertionError("Conflicting caps must not reach the upstream")

    async with proxy(upstream) as client:
        response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer test"}, json={"model": "gpt-5", "max_tokens": 4096, "max_completion_tokens": 1024})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_missing_auth_and_other_paths_do_not_forward():
    def upstream(request):
        raise AssertionError("Unauthorised or unknown routes must not reach upstream")

    async with proxy(upstream) as client:
        assert (await client.post("/v1/chat/completions", json=example_request())).status_code == 401
        for path in ("/v1/responses", "/v1/models", "/docs", "/openapi.json", "/health"):
            assert (await client.get(path)).status_code == 404
            assert (await client.post(path, json={})).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
async def test_upstream_errors_preserve_status_and_body(status):
    error = {"error": {"message": "simulated upstream rejection", "type": "test_error"}}

    def upstream(request):
        return httpx.Response(status, json=error, headers={"retry-after": "2"})

    async with proxy(upstream) as client:
        response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer test"}, json=example_request())
    assert response.status_code == status
    assert response.json() == error
    assert response.headers["retry-after"] == "2"
    assert "reward" not in response.json()


class Chunks(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False
        self.yields = 0

    async def __aiter__(self):
        for chunk in [b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n', b"data: [DONE]\n\n"]:
            self.yields += 1
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_streaming_sse_is_forwarded_and_response_closed():
    chunks = Chunks()

    def upstream(request):
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["max_completion_tokens"] == 4096
        return httpx.Response(200, stream=chunks, headers={"content-type": "text/event-stream", "cache-control": "no-cache"})

    payload = example_request()
    payload.update(stream=True, stream_options={"include_usage": True})
    async with proxy(upstream) as client:
        response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer test"}, json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content == b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
    assert chunks.yields == 2
    assert chunks.closed


@pytest.mark.asyncio
async def test_first_sse_chunk_reaches_asgi_before_upstream_finishes():
    first_sent = asyncio.Event()

    class WaitingChunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: first\n\n"
            # A proxy that reads the entire upstream body before sending the
            # response cannot satisfy this condition.
            await asyncio.wait_for(first_sent.wait(), timeout=1)
            yield b"data: [DONE]\n\n"

    def upstream(request):
        return httpx.Response(200, stream=WaitingChunks(), headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(upstream_client)

        async def observed_app(scope, receive, send):
            async def observed_send(message):
                if message["type"] == "http.response.body" and message.get("body") == b"data: first\n\n":
                    first_sent.set()
                await send(message)
            await app(scope, receive, observed_send)

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=observed_app), base_url="http://test") as client:
                response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer test"}, json={"model": "gpt-5", "max_tokens": 4096, "stream": True})
    assert first_sent.is_set()
    assert response.content == b"data: first\n\ndata: [DONE]\n\n"


@pytest.mark.asyncio
async def test_connection_error_returns_execution_error_without_secret():
    def upstream(request):
        raise httpx.ConnectError("simulated transport failure", request=request)

    async with proxy(upstream) as client:
        response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer never-return-this-value"}, json=example_request())
    assert response.status_code == 502
    assert "never-return-this-value" not in response.text
    assert "reward" not in response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [[], {"stream": "true"}])
async def test_invalid_json_shape_or_stream_flag_rejected(payload):
    def upstream(request):
        raise AssertionError("Invalid requests must not reach upstream")

    async with proxy(upstream) as client:
        response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer test"}, json=payload)
    assert response.status_code == 400
