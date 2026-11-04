"""Authenticated calls to the trusted GPU service."""

from __future__ import annotations

import os
from urllib.parse import quote

import httpx


class ExecutionError(RuntimeError):
    """The service did not complete the requested operation."""


class ServiceClient:
    def __init__(self, service_url: str = "", *, admin: bool = False, timeout: float = 1800):
        self.service_url = (service_url or os.environ.get("IWIBENCH_SERVICE_URL", "http://127.0.0.1:8768")).rstrip("/")
        self.token_var = "IWIBENCH_ADMIN_TOKEN" if admin else "IWIBENCH_EDITOR_TOKEN"
        self.admin = admin
        self.timeout = timeout

    async def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        token = os.environ.get(self.token_var)
        if not token:
            raise ExecutionError(f"Set {self.token_var} for the GPU service")
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
                response = await client.request(
                    method,
                    self.service_url + path,
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if not self.admin and "/tools/" in path and exc.response.status_code == 400:
                try:
                    detail = exc.response.json().get("detail")
                except (ValueError, AttributeError):
                    detail = None
                if isinstance(detail, str):
                    raise ExecutionError(f"Invalid tool request: {detail[:512]}") from exc
            # Deliberately omit response bodies: privileged endpoints may contain
            # grader details that must not become a tool error message.
            raise ExecutionError(f"GPU service returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise ExecutionError(f"GPU service request failed: {type(exc).__name__}") from exc
        try:
            result = response.json()
        except ValueError as exc:
            raise ExecutionError("GPU service returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise ExecutionError("GPU service response must be an object")
        return result

    async def create(self, **episode) -> dict:
        return await self.request("POST", "/sessions", episode)

    async def tool(self, session_id: str, name: str, request: dict) -> dict:
        if name not in {"read_weights", "write_weights", "run_target", "inspect_target"}:
            raise ValueError("Unknown IWIBench tool")
        return await self.request("POST", f"/sessions/{quote(session_id, safe='')}/tools/{name}", request)

    async def submit(self, session_id: str) -> dict:
        return await self.request("POST", f"/sessions/{quote(session_id, safe='')}/submit", {})

    async def grade(self, session_id: str) -> dict:
        return await self.request("POST", f"/sessions/{quote(session_id, safe='')}/grade", {})

    async def metadata(self, session_id: str) -> dict:
        return await self.request("GET", f"/sessions/{quote(session_id, safe='')}")
