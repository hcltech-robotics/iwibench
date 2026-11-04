"""Loopback editor transport for OpenAI editor requests.

The pinned Verifiers client sends ``max_tokens``. GPT-5 and GPT-6 Chat Completions
expect ``max_completion_tokens``. Supported models use the Responses API, with native
reasoning items retained only in bounded process memory for tool continuations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

UPSTREAM_URL = "https://api.openai.com/v1/chat/completions"
RESPONSES_URL = "https://api.openai.com/v1/responses"
RESPONSES_MODELS = {"gpt-6.1-sol", "gpt-6-astra"}
MAX_HISTORY_ENTRIES = 512
MAX_HISTORY_BYTES = 32 * 1024 * 1024
_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
    # httpx.aiter_bytes() decodes the response body before forwarding it.
    "content-encoding",
}


def _payload(body: Any) -> dict:
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    if "stream" in body and not isinstance(body["stream"], bool):
        raise ValueError("stream must be a boolean")
    if "max_tokens" in body and "max_completion_tokens" in body and body["max_tokens"] != body["max_completion_tokens"]:
        raise ValueError("Conflicting max_tokens and max_completion_tokens")
    model = body.get("model", "")
    if isinstance(model, str) and model.startswith(("gpt-5", "gpt-6")) and "max_tokens" in body:
        body = dict(body)
        body["max_completion_tokens"] = body.pop("max_tokens")
    return body


def _history_key(authorization: str, session_id: str, body: dict, prefix: list) -> str:
    material = {"auth": hashlib.sha256(authorization.encode()).hexdigest(), "session": session_id,
                "model": body["model"], "tools": body.get("tools", []), "messages": prefix}
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _responses_request(body: dict, history: OrderedDict, authorization: str, session_id: str) -> dict:
    """Translate the null harness transcript, retaining original reasoning items."""
    allowed = {"model", "messages", "tools", "tool_choice", "parallel_tool_calls",
               "max_completion_tokens", "stream", "stream_options", "reasoning_effort"}
    if set(body) - allowed:
        raise ValueError("unsupported Responses request parameter")
    if body.get("reasoning_effort", "low") not in {"low", "medium", "high", "xhigh"}:
        raise ValueError("unsupported reasoning effort")
    inputs = []
    if not session_id or len(session_id) > 256:
        raise ValueError("a bounded session_id is required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages are required")
    pending: set[str] = set()
    for index, msg in enumerate(messages):
        role = msg.get("role")
        if pending and role != "tool":
            raise ValueError("missing tool outputs")
        if role in {"system", "developer", "user"}:
            if not isinstance(msg.get("content"), str):
                raise ValueError("unsupported message content")
            inputs.append({"role": role, "content": msg["content"]})
        elif role == "tool":
            if msg["tool_call_id"] not in pending or not isinstance(msg.get("content"), str):
                raise ValueError("unexpected tool output")
            pending.remove(msg["tool_call_id"])
            inputs.append({"type": "function_call_output", "call_id": msg["tool_call_id"], "output": msg["content"]})
        elif role == "assistant" and msg.get("tool_calls"):
            if pending:
                raise ValueError("missing tool outputs")
            calls = msg["tool_calls"]
            ids = [call["id"] for call in calls]
            key = _history_key(authorization, session_id, body, messages[:index])
            record = history.get(key)
            if record is None:
                raise ValueError("missing original Responses tool-call history")
            original, expected_calls, expected_content, _ = record
            history.move_to_end(key)
            original_ids = [item["call_id"] for item in original if item.get("type") == "function_call"]
            actual = [(call["id"], call["function"]["name"], call["function"]["arguments"]) for call in calls]
            if ids != original_ids or actual != expected_calls or msg.get("content") != expected_content:
                raise ValueError("Responses tool-call history mismatch")
            pending = set(ids)
            if len(pending) != len(ids):
                raise ValueError("duplicate tool calls")
            inputs.extend(original)
        elif role == "assistant" and isinstance(msg.get("content"), str):
            inputs.append({"role": "assistant", "content": msg["content"]})
        else:
            raise ValueError("unsupported transcript message")
    if pending:
        raise ValueError("missing tool outputs")
    tools = []
    for tool in body.get("tools", []):
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("only caller-supplied function tools are supported")
        tools.append({"type": "function", **tool["function"]})
    payload = {"model": body["model"], "input": inputs, "tools": tools,
               "store": False, "include": ["reasoning.encrypted_content"],
               "reasoning": {"effort": body.get("reasoning_effort", "low")}}
    if "max_completion_tokens" in body:
        payload["max_output_tokens"] = body["max_completion_tokens"]
    for key in ("tool_choice", "parallel_tool_calls"):
        if key in body:
            payload[key] = body[key]
    return payload


def _responses_completion(result: dict, history: OrderedDict, key: str) -> dict:
    if result.get("status") not in {"completed", "incomplete"}:
        raise ValueError("Responses did not complete")
    if result["status"] == "incomplete" and (result.get("incomplete_details") or {}).get("reason") != "max_output_tokens":
        raise ValueError("unsupported incomplete response")
    output = result.get("output", [])
    calls = []
    content = []
    for item in output:
        if item.get("type") == "function_call":
            calls.append({"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": item["arguments"]}})
        elif item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    content.append(part["text"])
                else:
                    raise ValueError("unsupported Responses output content")
        elif item.get("type") == "reasoning":
            if not item.get("encrypted_content"):
                raise ValueError("reasoning item is not replayable")
        else:
            raise ValueError("unsupported Responses output item")
    if calls and result["status"] != "completed":
        raise ValueError("incomplete function call")
    if not calls and not content and result["status"] == "completed":
        raise ValueError("empty completed response")
    usage = result.get("usage")
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int for k in ("input_tokens", "output_tokens", "total_tokens")):
        raise ValueError("missing usage")
    message = {"role": "assistant", "content": "".join(content) if content else None}
    if calls:
        message["tool_calls"] = calls
        size = len(json.dumps(output, ensure_ascii=False).encode())
        if size > MAX_HISTORY_BYTES:
            raise ValueError("Responses history exceeds cache capacity")
        record = (output, [(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in calls], message["content"], size)
        history[key] = record
        history.move_to_end(key)
        while len(history) > MAX_HISTORY_ENTRIES or sum(v[3] for v in history.values()) > MAX_HISTORY_BYTES:
            history.popitem(last=False)
    chat_usage = {"prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"],
                  "total_tokens": usage["total_tokens"]}
    input_detail = usage.get("input_tokens_details") or {}
    if "cached_tokens" in input_detail:
        chat_usage["prompt_tokens_details"] = {"cached_tokens": input_detail["cached_tokens"]}
    detail = usage.get("output_tokens_details") or {}
    if "reasoning_tokens" in detail:
        chat_usage["completion_tokens_details"] = {"reasoning_tokens": detail["reasoning_tokens"]}
    return {"id": result["id"], "object": "chat.completion", "created": result.get("created_at", int(time.time())),
            "model": result["model"], "choices": [{"index": 0, "message": message,
            "finish_reason": "length" if result["status"] == "incomplete" else "tool_calls" if calls else "stop"}],
            "usage": chat_usage}


def _responses_stream(completion: dict) -> bytes:
    common = {"id": completion["id"], "object": "chat.completion.chunk", "created": completion["created"], "model": completion["model"]}
    message = completion["choices"][0]["message"]
    delta = {"role": "assistant"}
    if message["content"] is not None:
        delta["content"] = message["content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]
    chunks = [common | {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
              common | {"choices": [{"index": 0, "delta": {}, "finish_reason": completion["choices"][0]["finish_reason"]}]},
              common | {"choices": [], "usage": completion["usage"]}]
    return b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks) + b"data: [DONE]\n\n"



def _buffered_chat_stream(completion: Any) -> bytes:
    """Emit a complete Chat response as SSE without changing message content."""
    if not isinstance(completion, dict) or not isinstance(completion.get("id"), str) or not isinstance(completion.get("model"), str) or type(completion.get("created")) is not int:
        raise ValueError("invalid completion metadata")
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing choices")
    common = {k: completion[k] for k in ("id", "model", "created")}
    common["object"] = "chat.completion.chunk"
    for key in ("system_fingerprint", "service_tier"):
        if key in completion:
            common[key] = completion[key]
    deltas, endings = [], []
    indices = set()
    for choice in choices:
        if not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] in indices:
            raise ValueError("invalid choice")
        indices.add(choice["index"])
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant" or set(message) - {"role", "content", "tool_calls", "reasoning", "reasoning_content", "reasoning_details"}:
            raise ValueError("unsupported message")
        for key in ("content", "reasoning", "reasoning_content"):
            if message.get(key) is not None and not isinstance(message[key], str):
                raise ValueError("unsupported message content")
        details = message.get("reasoning_details")
        if details is not None and (not isinstance(details, list) or not all(isinstance(item, dict) for item in details)):
            raise ValueError("unsupported reasoning details")
        delta = dict(message)
        calls = message.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                raise ValueError("invalid tool calls")
            for call in calls:
                if (not isinstance(call, dict) or call.get("type") != "function" or not isinstance(call.get("id"), str)
                    or not isinstance(call.get("function"), dict) or not isinstance(call["function"].get("name"), str)
                    or not isinstance(call["function"].get("arguments"), str)):
                    raise ValueError("unsupported tool call")
            delta["tool_calls"] = [{**call, "index": i} for i, call in enumerate(calls)]
        if not isinstance(choice.get("finish_reason"), str) or choice.get("logprobs") is not None:
            raise ValueError("unsupported finish reason or logprobs")
        deltas.append({"index": choice["index"], "delta": delta, "finish_reason": None})
        endings.append({"index": choice["index"], "delta": {}, "finish_reason": choice["finish_reason"]})
    chunks = [common | {"choices": deltas}, common | {"choices": endings}]
    if completion.get("usage") is not None:
        if not isinstance(completion["usage"], dict):
            raise ValueError("invalid usage")
        chunks.append(common | {"choices": [], "usage": completion["usage"]})
    return b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks) + b"data: [DONE]\n\n"


def create_app(upstream_client: httpx.AsyncClient | None = None, *, upstream_url: str = UPSTREAM_URL,
               buffer_chat: bool = False) -> FastAPI:
    """Create the proxy; an injected client supports transport-level tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if upstream_client is not None:
            app.state.upstream = upstream_client
            yield
        else:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=30.0),
                follow_redirects=False,
                trust_env=False,
            ) as client:
                app.state.upstream = client
                yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    responses_history: OrderedDict = OrderedDict()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        authorization = request.headers.get("authorization", "")
        if not authorization.strip():
            return JSONResponse({"error": {"message": "Authorization header required"}}, status_code=401)
        try:
            payload = _payload(await request.json())
        except (ValueError, json.JSONDecodeError):
            # Return the validation reason, never the request body or headers.
            return JSONResponse({"error": {"message": "Invalid JSON body, stream flag or conflicting token caps"}}, status_code=400)
        client = request.app.state.upstream
        if payload.get("model") in RESPONSES_MODELS:
            try:
                # Verifiers' EvalClient routes each trace through x-session-id.
                session_id = request.headers.get("x-session-id", "")
                responses_payload = _responses_request(payload, responses_history, authorization, session_id)
                cache_key = _history_key(authorization, session_id, payload, payload["messages"])
            except (ValueError, KeyError, TypeError):
                return JSONResponse({"error": {"message": "Unsupported Responses transcript or request parameter"}}, status_code=400)
            try:
                upstream = await client.post(RESPONSES_URL, headers={"Authorization": authorization}, json=responses_payload)
                if upstream.status_code >= 400:
                    return Response(upstream.content, status_code=upstream.status_code, media_type="application/json")
                completion = _responses_completion(upstream.json(), responses_history, cache_key)
            except (ValueError, KeyError, TypeError, httpx.RequestError):
                return JSONResponse({"error": {"message": "Responses transport failed"}}, status_code=502)
            if payload.get("stream", False):
                return Response(_responses_stream(completion), media_type="text/event-stream")
            return JSONResponse(completion)
        if buffer_chat and payload.get("stream", False):
            buffered = dict(payload)
            buffered["stream"] = False
            buffered.pop("stream_options", None)
            try:
                upstream = await client.post(upstream_url, headers={"Authorization": authorization}, json=buffered)
                if upstream.status_code >= 400:
                    return Response(upstream.content, status_code=upstream.status_code, media_type="application/json")
                return Response(_buffered_chat_stream(upstream.json()), media_type="text/event-stream")
            except (ValueError, KeyError, TypeError, httpx.RequestError):
                return JSONResponse({"error": {"message": "Buffered Chat transport failed"}}, status_code=502)
        forwarded = client.build_request(
            "POST", upstream_url,
            headers={"Authorization": authorization, "Content-Type": "application/json"},
            json=payload,
        )
        try:
            upstream = await client.send(forwarded, stream=True, follow_redirects=False)
        except httpx.RequestError:
            return JSONResponse({"error": {"message": "OpenAI upstream connection failed"}}, status_code=502)
        connection_headers = {name.strip().lower() for name in upstream.headers.get("connection", "").split(",")}
        headers = {key: value for key, value in upstream.headers.items() if key.lower() not in _HOP_HEADERS | connection_headers}
        if payload.get("stream", False):
            async def chunks():
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                finally:
                    await upstream.aclose()

            return StreamingResponse(
                chunks(), status_code=upstream.status_code, headers=headers,
                background=BackgroundTask(upstream.aclose),
            )
        try:
            body = await upstream.aread()
            return Response(body, status_code=upstream.status_code, headers=headers)
        except httpx.RequestError:
            return JSONResponse({"error": {"message": "OpenAI upstream response failed"}}, status_code=502)
        finally:
            await upstream.aclose()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--upstream-url", default=UPSTREAM_URL)
    parser.add_argument("--buffer-chat", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    url = httpx.URL(args.upstream_url)
    if url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.fragment:
        parser.error("upstream URL must be an HTTP(S) URL without credentials or fragment")
    import uvicorn
    uvicorn.run(create_app(upstream_url=args.upstream_url, buffer_chat=args.buffer_chat), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
