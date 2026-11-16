"""可演化 LangGraph capability plugin 的最小原型。

V0 只验证一个核心不变量：插件的 learned state 生命周期独立于当前 LangGraph
执行实例的 residency 生命周期。持久状态写入插件私有 namespace；执行图可以被卸载、
重新构造并恢复此前学到的行为。

本模块不接管 ORION 主 Agent，不调用 LLM/KG/RAG，也不实现完整的生产插件系统。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph


_PLUGIN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STATE_SCHEMA_VERSION = "orion-capability-learned-state-v1"


class PluginExecutionState(TypedDict, total=False):
    """V0 LangGraph 插件的最小执行状态。"""

    signal: str
    action: str
    generation: int
    used_learned_state: bool


@dataclass(frozen=True)
class CapabilityLearnedState:
    """插件私有、可跨 runtime instance 恢复的 learned state。"""

    plugin_id: str
    generation: int = 0
    rules: dict[str, str] = field(default_factory=dict)


class JsonCapabilityLearnedStateStore:
    """按 plugin identity 隔离的 JSON learned-state store。

    V0 使用文件存储是为了让状态寿命天然越过 Python 对象和单次 LangGraph 编译实例。
    后续可以替换为数据库/对象存储，但 namespace 语义不应改变。
    """

    def __init__(self, root: str | Path, plugin_id: str) -> None:
        if not _PLUGIN_ID_RE.fullmatch(plugin_id):
            raise ValueError(f"invalid plugin_id: {plugin_id!r}")
        self.root = Path(root)
        self.plugin_id = plugin_id
        self.namespace_dir = self.root / plugin_id
        self.path = self.namespace_dir / "learned_state.json"

    def load(self) -> CapabilityLearnedState:
        if not self.path.exists():
            return CapabilityLearnedState(plugin_id=self.plugin_id)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != _STATE_SCHEMA_VERSION:
            raise ValueError("unsupported learned-state schema")
        if payload.get("plugin_id") != self.plugin_id:
            raise ValueError("learned-state plugin identity mismatch")
        generation = int(payload.get("generation", 0))
        raw_rules = payload.get("rules") or {}
        if generation < 0:
            raise ValueError("learned-state generation must be non-negative")
        if not isinstance(raw_rules, dict):
            raise ValueError("learned-state rules must be an object")
        rules = {str(key): str(value) for key, value in raw_rules.items()}
        return CapabilityLearnedState(
            plugin_id=self.plugin_id,
            generation=generation,
            rules=rules,
        )

    def save(self, state: CapabilityLearnedState) -> None:
        if state.plugin_id != self.plugin_id:
            raise ValueError("cannot save learned state for another plugin")
        self.namespace_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "plugin_id": state.plugin_id,
            "generation": state.generation,
            "rules": dict(sorted(state.rules.items())),
        }
        temp_path = self.path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.path)

    def learn_rule(self, signal: str, action: str) -> CapabilityLearnedState:
        signal_key = signal.strip()
        action_value = action.strip()
        if not signal_key:
            raise ValueError("signal is required")
        if not action_value:
            raise ValueError("action is required")
        current = self.load()
        updated_rules = dict(current.rules)
        updated_rules[signal_key] = action_value
        updated = CapabilityLearnedState(
            plugin_id=self.plugin_id,
            generation=current.generation + 1,
            rules=updated_rules,
        )
        self.save(updated)
        return updated


class EvolvableLangGraphPlugin:
    """执行实例可卸载、learned state 可持久存在的 LangGraph capability plugin。"""

    def __init__(
        self,
        *,
        plugin_id: str,
        version: str,
        state_root: str | Path,
        fallback_action: str = "unknown",
        resident_mb: float = 0.0,
        memory_namespace: str | None = None,
    ) -> None:
        if not version.strip():
            raise ValueError("version is required")
        if not fallback_action.strip():
            raise ValueError("fallback_action is required")
        if resident_mb < 0:
            raise ValueError("resident_mb must be non-negative")
        self.plugin_id = plugin_id
        self.version = version.strip()
        self.fallback_action = fallback_action.strip()
        self.resident_mb = float(resident_mb)
        self.memory_namespace = memory_namespace or plugin_id
        self.state_store = JsonCapabilityLearnedStateStore(
            state_root,
            self.memory_namespace,
        )
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
        """构造执行图并 materialize V0 的插件私有 resident payload。"""

        if self._graph is not None:
            return
        self._graph = self._build_graph()
        allocation_bytes = int(round(self.resident_mb * 1024 * 1024))
        if allocation_bytes:
            resident = bytearray(allocation_bytes)
            # 触碰页面，避免只保留逻辑长度而未实际 materialize。
            for index in range(0, allocation_bytes, 4096):
                resident[index] = 1
            self._resident_buffer = resident
        else:
            self._resident_buffer = None

    def unmount(self) -> None:
        """释放瞬态执行实体；不删除 learned state。"""

        self._graph = None
        self._resident_buffer = None

    def invoke(self, signal: str) -> PluginExecutionState:
        if self._graph is None:
            raise RuntimeError(f"plugin is not resident: {self.plugin_id}")
        normalized = signal.strip()
        if not normalized:
            raise ValueError("signal is required")
        result = self._graph.invoke({"signal": normalized})
        return PluginExecutionState(
            signal=str(result["signal"]),
            action=str(result["action"]),
            generation=int(result["generation"]),
            used_learned_state=bool(result["used_learned_state"]),
        )

    def learn(self, signal: str, action: str) -> CapabilityLearnedState:
        """在插件驻留期间更新其私有 learned state。"""

        if self._graph is None:
            raise RuntimeError(f"plugin is not resident: {self.plugin_id}")
        return self.state_store.learn_rule(signal, action)

    def _build_graph(self) -> CompiledStateGraph:
        def apply_learned_state(state: PluginExecutionState) -> PluginExecutionState:
            signal = str(state.get("signal") or "").strip()
            learned = self.state_store.load()
            learned_action = learned.rules.get(signal)
            return {
                "signal": signal,
                "action": learned_action or self.fallback_action,
                "generation": learned.generation,
                "used_learned_state": learned_action is not None,
            }

        graph = StateGraph(PluginExecutionState)
        graph.add_node("apply_learned_state", apply_learned_state)
        graph.set_entry_point("apply_learned_state")
        graph.set_finish_point("apply_learned_state")
        return graph.compile()
