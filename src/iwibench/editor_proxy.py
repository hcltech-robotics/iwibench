"""Loopback editor transport for OpenAI Chat Completions.

The pinned Verifiers client sends ``max_tokens``. GPT-5 and GPT-6 Chat Completions
expects ``max_completion_tokens``. This proxy changes that field only and
forwards the editor's request to the fixed OpenAI endpoint.
"""

from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

UPSTREAM_URL = "https://api.openai.com/v1/chat/completions"
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


def create_app(upstream_client: httpx.AsyncClient | None = None) -> FastAPI:
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
        forwarded = client.build_request(
            "POST", UPSTREAM_URL,
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
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
