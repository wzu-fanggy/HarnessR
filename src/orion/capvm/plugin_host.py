"""在稳定 LangGraph host 上挂载可演化 capability plugin 的最小运行时。

V2 的目标不是造新 Agent 框架，而是验证：host graph 在编译后保持不变，能力通过
registry 动态挂载/卸载，并能在不同 host topology 之间迁移。

V10/V11 将 gateway 泛化为任意 stateful capability executable：旧的 signal->action API
继续兼容，同时真实 scientific LangGraph capability 可以通过结构化 payload 走同一个稳定
Host graph，而不需要为每种能力重新编译主图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from .evolvable_plugin import CapabilityLearnedState


class StatefulCapabilityExecutable(Protocol):
    """稳定 Host registry 需要的最小 stateful capability contract。"""

    plugin_id: str
    version: str

    @property
    def learned_state(self) -> CapabilityLearnedState: ...

    @property
    def is_mounted(self) -> bool: ...

    def mount(self) -> None: ...

    def unmount(self) -> None: ...

    def invoke(self, payload: Any) -> Any: ...


class HostExecutionState(TypedDict, total=False):
    capability_id: str
    signal: str
    normalized_signal: str
    action: str
    capability_payload: dict[str, object]
    plugin_output: dict[str, object]
    host_id: str
    audit: list[str]


@dataclass(frozen=True)
class PluginHotSwapResult:
    """一次 state-preserving capability hot swap 的审计结果。"""

    plugin_id: str
    previous_version: str
    active_version: str
    generation_before: int
    generation_after: int
    learned_state_preserved: bool
    previous_unmounted: bool


@dataclass
class PluginRegistry:
    """Host-independent plugin registry。"""

    _plugins: dict[str, StatefulCapabilityExecutable] = field(default_factory=dict)

    def mount(self, plugin: StatefulCapabilityExecutable) -> None:
        existing = self._plugins.get(plugin.plugin_id)
        if existing is not None and existing is not plugin:
            raise ValueError(f"plugin already mounted: {plugin.plugin_id}")
        plugin.mount()
        self._plugins[plugin.plugin_id] = plugin

    def unmount(self, plugin_id: str) -> StatefulCapabilityExecutable:
        try:
            plugin = self._plugins.pop(plugin_id)
        except KeyError as exc:
            raise KeyError(f"plugin is not mounted: {plugin_id}") from exc
        plugin.unmount()
        return plugin

    def get(self, plugin_id: str) -> StatefulCapabilityExecutable:
        try:
            return self._plugins[plugin_id]
        except KeyError as exc:
            raise KeyError(f"plugin is not mounted: {plugin_id}") from exc

    def mounted_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def hot_swap(
        self,
        plugin: StatefulCapabilityExecutable,
        *,
        validator: Callable[[StatefulCapabilityExecutable], None] | None = None,
        require_state_preservation: bool = True,
    ) -> PluginHotSwapResult:
        """先 stage/validate 新实例，再原子替换 registry 指针。

        该操作只替换 capability execution instance，不修改 Host graph。默认要求候选
        完整继承当前 learned state，避免把代码热更新误写成一次隐式遗忘/行为重置。
        validator 在切换前运行；任何异常都会卸载候选并让旧实例继续服务。
        """

        plugin_id = plugin.plugin_id
        try:
            previous = self._plugins[plugin_id]
        except KeyError as exc:
            raise KeyError(f"plugin is not mounted: {plugin_id}") from exc
        if previous is plugin:
            raise ValueError("hot-swap candidate must be a different plugin instance")

        previous_state = previous.learned_state
        candidate_state = plugin.learned_state
        if require_state_preservation and candidate_state != previous_state:
            raise ValueError(
                "hot-swap candidate must preserve the current learned state exactly"
            )

        plugin.mount()
        try:
            if validator is not None:
                validator(plugin)
        except Exception:
            plugin.unmount()
            raise

        # dict assignment is the single registry commit point. Before this line every Host
        # invocation resolves to the previous instance; after it, new invocations resolve to
        # the candidate. Host graph identity/topology is unchanged.
        self._plugins[plugin_id] = plugin
        try:
            previous.unmount()
        except Exception:
            # 若旧实例释放失败，不留下半切换状态：恢复 registry，再卸载候选。
            self._plugins[plugin_id] = previous
            plugin.unmount()
            raise

        active_state = plugin.learned_state
        return PluginHotSwapResult(
            plugin_id=plugin_id,
            previous_version=previous.version,
            active_version=plugin.version,
            generation_before=previous_state.generation,
            generation_after=active_state.generation,
            learned_state_preserved=active_state == previous_state,
            previous_unmounted=not previous.is_mounted,
        )


class StableLangGraphPluginHost:
    """主图一次编译，插件挂载不会重写或重新编译 host graph。"""

    def __init__(
        self,
        *,
        host_id: str,
        topology: str,
        declared_tools: tuple[str, ...] = (),
    ) -> None:
        if topology not in {"direct", "audited"}:
            raise ValueError("topology must be 'direct' or 'audited'")
        self.host_id = host_id
        self.topology = topology
        self.declared_tools = tuple(declared_tools)
        self.registry = PluginRegistry()
        self.graph = self._build_graph()

    @property
    def graph_identity(self) -> int:
        return id(self.graph)

    def graph_node_names(self) -> tuple[str, ...]:
        rendered = self.graph.get_graph()
        return tuple(sorted(rendered.nodes))

    def mount(self, plugin: StatefulCapabilityExecutable) -> None:
        self.registry.mount(plugin)

    def unmount(self, plugin_id: str) -> StatefulCapabilityExecutable:
        return self.registry.unmount(plugin_id)

    def hot_swap(
        self,
        plugin: StatefulCapabilityExecutable,
        *,
        validator: Callable[[StatefulCapabilityExecutable], None] | None = None,
        require_state_preservation: bool = True,
    ) -> PluginHotSwapResult:
        """在不重编译 Host graph 的前提下替换同 identity capability 实例。"""

        return self.registry.hot_swap(
            plugin,
            validator=validator,
            require_state_preservation=require_state_preservation,
        )

    def invoke(self, *, capability_id: str, signal: str) -> HostExecutionState:
        """兼容 V2/V10 的简单 signal -> action capability 调用。"""

        result = self.graph.invoke(
            {
                "capability_id": capability_id,
                "signal": signal,
                "host_id": self.host_id,
                "audit": [],
            }
        )
        return HostExecutionState(
            capability_id=str(result["capability_id"]),
            signal=str(result["signal"]),
            normalized_signal=str(result["normalized_signal"]),
            action=str(result["action"]),
            host_id=str(result["host_id"]),
            audit=list(result.get("audit") or []),
        )

    def invoke_payload(
        self,
        *,
        capability_id: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        """通过同一个稳定 gateway 调用结构化 scientific/plugin payload。"""

        result = self.graph.invoke(
            {
                "capability_id": capability_id,
                "signal": "",
                "capability_payload": dict(payload),
                "host_id": self.host_id,
                "audit": [],
            }
        )
        output = result.get("plugin_output")
        if not isinstance(output, dict):
            raise TypeError("structured capability did not return a mapping output")
        return dict(output)

    def _build_graph(self) -> CompiledStateGraph:
        def normalize(state: HostExecutionState) -> HostExecutionState:
            signal = str(state.get("signal") or "").strip().lower()
            return {"normalized_signal": signal}

        def audit(state: HostExecutionState) -> HostExecutionState:
            existing = list(state.get("audit") or [])
            existing.append(f"{self.host_id}:pre-plugin-audit")
            return {"audit": existing}

        def capability_gateway(state: HostExecutionState) -> HostExecutionState:
            capability_id = str(state.get("capability_id") or "").strip()
            plugin = self.registry.get(capability_id)
            existing = list(state.get("audit") or [])
            existing.append(f"{self.host_id}:plugin:{capability_id}")

            structured_payload = state.get("capability_payload")
            if structured_payload is not None:
                result = plugin.invoke(dict(structured_payload))
                if not isinstance(result, Mapping):
                    raise TypeError("structured capability output must be a mapping")
                return {
                    "plugin_output": dict(result),
                    "audit": existing,
                }

            signal = str(state.get("normalized_signal") or "").strip()
            result = plugin.invoke(signal)
            if not isinstance(result, Mapping):
                raise TypeError("signal capability output must be a mapping")
            return {
                "action": str(result["action"]),
                "audit": existing,
            }

        def finish(state: HostExecutionState) -> HostExecutionState:
            existing = list(state.get("audit") or [])
            existing.append(f"{self.host_id}:finish")
            return {"audit": existing}

        graph = StateGraph(HostExecutionState)
        graph.add_node("normalize", normalize)
        if self.topology == "audited":
            graph.add_node("audit", audit)
        graph.add_node("capability_gateway", capability_gateway)
        graph.add_node("finish", finish)
        graph.set_entry_point("normalize")
        if self.topology == "audited":
            graph.add_edge("normalize", "audit")
            graph.add_edge("audit", "capability_gateway")
        else:
            graph.add_edge("normalize", "capability_gateway")
        graph.add_edge("capability_gateway", "finish")
        graph.set_finish_point("finish")
        return graph.compile()
