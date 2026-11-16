"""可承载任意 LangGraph 科学逻辑的 stateful capability plugin。

与 V0 的 `EvolvableLangGraphPlugin` 不同，本类不把执行语义固定为 signal->action 规则表。
调用方提供 graph factory；插件只负责统一 identity、私有 learned-state namespace、
materialization/unmount 和可恢复执行实例。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from .evolvable_plugin import (
    CapabilityLearnedState,
    JsonCapabilityLearnedStateStore,
)


GraphFactory = Callable[[JsonCapabilityLearnedStateStore], CompiledStateGraph]


class StatefulLangGraphCapabilityPlugin:
    """执行图可卸载、learned state 独立持久的通用 LangGraph capability。"""

    def __init__(
        self,
        *,
        plugin_id: str,
        version: str,
        state_root: str | Path,
        graph_factory: GraphFactory,
        resident_mb: float = 0.0,
    ) -> None:
        if not version.strip():
            raise ValueError("version is required")
        if resident_mb < 0:
            raise ValueError("resident_mb must be non-negative")
        self.plugin_id = plugin_id
        self.version = version.strip()
        self.state_store = JsonCapabilityLearnedStateStore(state_root, plugin_id)
        self.graph_factory = graph_factory
        self.resident_mb = float(resident_mb)
        self._graph: CompiledStateGraph | None = None
        self._resident_buffer: bytearray | None = None

    @property
    def is_mounted(self) -> bool:
        return self._graph is not None

    @property
    def resident_bytes(self) -> int:
        return len(self._resident_buffer) if self._resident_buffer is not None else 0

    @property
    def learned_state(self) -> CapabilityLearnedState:
        return self.state_store.load()

    def mount(self) -> None:
        if self._graph is not None:
            return
        self._graph = self.graph_factory(self.state_store)
        allocation_bytes = int(round(self.resident_mb * 1024 * 1024))
        if allocation_bytes:
            resident = bytearray(allocation_bytes)
            for index in range(0, allocation_bytes, 4096):
                resident[index] = 1
            self._resident_buffer = resident
        else:
            self._resident_buffer = None

    def unmount(self) -> None:
        self._graph = None
        self._resident_buffer = None

    def invoke(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._graph is None:
            raise RuntimeError(f"plugin is not resident: {self.plugin_id}")
        return dict(self._graph.invoke(dict(payload)))

    def learn_rule(self, key: str, value: str) -> CapabilityLearnedState:
        if self._graph is None:
            raise RuntimeError(f"plugin is not resident: {self.plugin_id}")
        return self.state_store.learn_rule(key, value)
