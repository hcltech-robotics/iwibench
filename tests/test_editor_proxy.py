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


def _sol_request_for_test(stream=False):
    return {"model": "gpt-6.1-sol", "max_tokens": 4096, "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "Read the tensor."}],
            "tools": [{"type": "function", "function": {"name": "read_weights", "description": "Read weights",
                       "parameters": {"type": "object", "properties": {}, "additionalProperties": False}, "strict": True}}],
            "tool_choice": "auto", "stream": stream, "stream_options": {"include_usage": True}}


def _sol_result(output, status="completed"):
    return {"id": "resp_1", "model": "gpt-6.1-sol", "created_at": 100, "status": status,
            "output": output, "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                                          "output_tokens_details": {"reasoning_tokens": 2}}}


@pytest.mark.asyncio
async def test_sol_stream_and_native_reasoning_tool_round_trip():
    payload = _sol_request_for_test(True)
    requests = []
    reasoning = {"type": "reasoning", "id": "rs_1", "encrypted_content": "private-replay-data", "summary": []}
    function = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_weights", "arguments": "{}", "status": "completed"}
    def handler(request):
        assert str(request.url).endswith("/v1/responses")
        wire = json.loads(request.content)
        requests.append(wire)
        assert wire["model"] == "gpt-6.1-sol"
        assert wire["store"] is False and wire["include"] == ["reasoning.encrypted_content"]
        assert wire["reasoning"] == {"effort": "low"} and wire["max_output_tokens"] == 4096
        assert wire["tools"] == [{"type": "function", **payload["tools"][0]["function"]}]
        if len(requests) == 1:
            return httpx.Response(200, json=_sol_result([reasoning, function]))
        assert wire["input"][1:] == [reasoning, function, {"type": "function_call_output", "call_id": "call_1", "output": "tensor result"}]
        return httpx.Response(200, json=_sol_result([{"type": "message", "content": [{"type": "output_text", "text": "Finished."}]}]))
    headers = {"Authorization": "Bearer test", "x-session-id": "session-1"}
    async with proxy(handler) as client:
        first = await client.post("/v1/chat/completions", headers=headers, json=payload)
        assert first.status_code == 200
        assert "private-replay-data" not in first.text
        events = [json.loads(line[6:]) for line in first.text.splitlines() if line.startswith("data: {")]
        delta = events[0]["choices"][0]["delta"]
        assert events[1]["choices"][0]["finish_reason"] == "tool_calls"
        assert events[2]["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2
        assert first.text.endswith("data: [DONE]\n\n")
        call = {k: v for k, v in delta["tool_calls"][0].items() if k != "index"}
        payload["messages"] += [{"role": "assistant", "content": None, "tool_calls": [call]},
                                {"role": "tool", "tool_call_id": "call_1", "content": "tensor result"}]
        second = await client.post("/v1/chat/completions", headers=headers, json=payload)
        assert second.status_code == 200 and "Finished." in second.text
        assert "private-replay-data" not in second.text


@pytest.mark.asyncio
async def test_sol_cache_rejects_changed_prefix_session_auth_and_arguments():
    payload = _sol_request_for_test()
    function = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_weights", "arguments": "{}"}
    called = []
    def handler(request):
        called.append(request)
        return httpx.Response(200, json=_sol_result([function]))
    headers = {"Authorization": "Bearer a", "x-session-id": "one"}
    async with proxy(handler) as client:
        first = await client.post("/v1/chat/completions", headers=headers, json=payload)
        message = first.json()["choices"][0]["message"]
        continuation = copy.deepcopy(payload)
        continuation["messages"] += [message, {"role": "tool", "tool_call_id": "call_1", "content": "ok"}]
        cases = []
        changed = copy.deepcopy(continuation); changed["messages"][0]["content"] = "Different prompt."; cases.append((headers, changed))
        changed = copy.deepcopy(continuation); changed["messages"][1]["tool_calls"][0]["function"]["arguments"] = "{\"x\":1}"; cases.append((headers, changed))
        cases += [({"Authorization": "Bearer b", "x-session-id": "one"}, continuation),
                  ({"Authorization": "Bearer a", "x-session-id": "two"}, continuation),
                  ({"Authorization": "Bearer a", "session_id": "one"}, continuation),
                  ({"Authorization": "Bearer a"}, continuation)]
        for request_headers, data in cases:
            response = await client.post("/v1/chat/completions", headers=request_headers, json=data)
            assert response.status_code == 400
    assert len(called) == 1


@pytest.mark.asyncio
async def test_sol_rejects_unreplayable_reasoning_and_surfaces_upstream_error():
    def handler(request):
        return httpx.Response(200, json=_sol_result([{"type": "reasoning", "id": "rs_1", "summary": []}]))
    async with proxy(handler) as client:
        response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer test", "x-session-id": "one"}, json=_sol_request_for_test())
    assert response.status_code == 502
    def rejected(request):
        return httpx.Response(400, json={"error": {"message": "reasoning rejected"}})
    async with proxy(rejected) as client:
        response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer test", "x-session-id": "one"}, json=_sol_request_for_test())
    assert response.status_code == 400 and response.json()["error"]["message"] == "reasoning rejected"


@pytest.mark.asyncio
async def test_sol_rejects_empty_completion_and_does_not_invent_reasoning_usage():
    def empty(request):
        return httpx.Response(200, json=_sol_result([]))
    headers = {"Authorization": "Bearer test", "x-session-id": "one"}
    async with proxy(empty) as client:
        response = await client.post("/v1/chat/completions", headers=headers, json=_sol_request_for_test())
    assert response.status_code == 502

    def no_reasoning_detail(request):
        result = _sol_result([{"type": "message", "content": [{"type": "output_text", "text": "Done."}]}])
        del result["usage"]["output_tokens_details"]
        return httpx.Response(200, json=result)
    async with proxy(no_reasoning_detail) as client:
        response = await client.post("/v1/chat/completions", headers=headers, json=_sol_request_for_test())
    assert response.status_code == 200
    assert "completion_tokens_details" not in response.json()["usage"]
