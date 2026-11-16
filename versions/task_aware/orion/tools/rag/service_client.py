"""共享的根项目本地 KG 服务客户端。"""
from __future__ import annotations

import os
from typing import Any

import httpx


class RootKgServiceClient:
    """同步调用仅监听本机回环地址的根 KG 服务。"""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float = 90.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("ORION_RAG_SERVICE_URL", "http://127.0.0.1:8799")
        ).rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def _decode_response(self, response: httpx.Response) -> Any:
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"local KG service error ({response.status_code}): {detail}")
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("local KG service returned invalid JSON") from exc

    def call(self, path: str, payload: dict[str, Any]) -> Any:
        """向服务的固定路径提交 JSON 请求并规范化错误。"""
        if not path.startswith("/"):
            raise ValueError("root KG service path must start with '/'")
        try:
            response = self._client.post(f"{self.base_url}{path}", json=payload)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"local KG service unavailable: {exc}") from exc
        return self._decode_response(response)

    def get(self, path: str) -> Any:
        """读取根服务固定只读端点。"""
        if not path.startswith("/"):
            raise ValueError("root KG service path must start with '/'")
        try:
            response = self._client.get(f"{self.base_url}{path}")
        except httpx.HTTPError as exc:
            raise RuntimeError(f"local KG service unavailable: {exc}") from exc
        return self._decode_response(response)

    def acquire_runtime_resource(self, resource_id: str) -> dict[str, Any]:
        """请求根服务建立一个 allowlisted tool resource。"""
        payload = self.call(f"/v1/runtime/resources/{resource_id}/acquire", {})
        return dict(payload.get("resource") or {})

    def release_runtime_resource(self, resource_id: str) -> dict[str, Any]:
        """请求根服务释放一个 allowlisted tool resource。"""
        payload = self.call(f"/v1/runtime/resources/{resource_id}/release", {})
        return dict(payload.get("resource") or {})

    def runtime_resources(self) -> dict[str, dict[str, Any]]:
        """读取根服务当前 tool-resource residency。"""
        payload = self.get("/v1/runtime/resources")
        resources = payload.get("resources") if isinstance(payload, dict) else None
        if not isinstance(resources, dict):
            raise RuntimeError("local KG service returned invalid runtime resource snapshot")
        return {str(key): dict(value) for key, value in resources.items()}

    def search(self, operation: str, payload: dict[str, Any]) -> Any:
        """调用服务 allowlist 中的 RAG operation。"""
        return self.call(f"/v1/search/{operation}", payload)


_root_kg_service_client: RootKgServiceClient | None = None


def get_root_kg_service_client() -> RootKgServiceClient:
    """获取进程级共享客户端；客户端不加载模型或 Neo4j driver。"""
    global _root_kg_service_client
    if _root_kg_service_client is None:
        _root_kg_service_client = RootKgServiceClient()
    return _root_kg_service_client
