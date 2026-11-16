"""Root KG service 中可重建工具资源的远程 residency adapter。"""



from __future__ import annotations



from typing import Any, Protocol

from dataclasses import asdict



from kg_rag.resource_accounting import account_resource_snapshot



from ..measurement_batch import runtime_snapshots

from ..model import ResidencyLevel, ResourceVector

from .base import BaseCapabilityAdapter





class RuntimeResourceClient(Protocol):

    def acquire_runtime_resource(self, resource_id: str) -> dict[str, Any]: ...

    def release_runtime_resource(self, resource_id: str) -> dict[str, Any]: ...

    def runtime_resources(self) -> dict[str, dict[str, Any]]: ...





class RootServiceResourceAdapter(BaseCapabilityAdapter):

    """把根 RAG 服务拥有的 Model/Session 映射到统一 tool-resource lifecycle。"""



    def __init__(

        self,

        *,

        client: RuntimeResourceClient,

        remote_resource_id: str,

        declared_cost: ResourceVector,

    ) -> None:

        if not remote_resource_id.strip():

            raise ValueError("remote_resource_id is required")

        self.client = client

        self.remote_resource_id = remote_resource_id

        self._declared_cost = declared_cost

        self._last_snapshot: dict[str, Any] = {}

        self.last_cost_basis: dict[str, str] = {}

        self.last_startup_latency_s: float | None = None



    @property

    def resident(self) -> bool:

        return bool(self._last_snapshot.get("resident"))



    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:

        if level < ResidencyLevel.BOUND:

            return ResourceVector()

        return self._declared_cost



    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:

        if level < ResidencyLevel.BOUND:

            self.last_cost_basis = {}

            return ResourceVector()

        try:

            snapshot = runtime_snapshots(self.client).get(self.remote_resource_id, {})

            if snapshot:

                self._last_snapshot = dict(snapshot)

        except Exception:

            # 通信失败时不能把过期观测当作当前实测，也不能据此记零。

            snapshot = {}

        costs, self.last_cost_basis = account_resource_snapshot(

            snapshot, asdict(self._declared_cost))

        return ResourceVector(**costs)



    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:

        if current < ResidencyLevel.BOUND <= target:

            snapshot = self.client.acquire_runtime_resource(self.remote_resource_id)

            self._last_snapshot = dict(snapshot)

            latency = snapshot.get("last_open_seconds") or snapshot.get("last_load_seconds")

            self.last_startup_latency_s = float(latency) if latency is not None else None



    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:

        if current >= ResidencyLevel.BOUND > target:

            self._last_snapshot = dict(

                self.client.release_runtime_resource(self.remote_resource_id)

            )



    def close(self) -> None:

        if not self._last_snapshot.get("resident"):

            return

        try:

            self._last_snapshot = dict(

                self.client.release_runtime_resource(self.remote_resource_id)

            )

        except Exception:

            # 根服务可能先于 Agent 退出；关闭失败不能阻塞本地进程收尾。

            pass

