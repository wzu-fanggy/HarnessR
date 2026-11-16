"""按工具资源预算执行 LLM 已提出的并行调用，不改变调用选择与参数。"""
from __future__ import annotations

from collections.abc import Mapping
from orion.capvm import ResidencyLevel, ResourceVector
from orion.core.langgraph_residency import ToolCallResidencyBridge


class ResourceBudgetedToolBridge(ToolCallResidencyBridge):
    """并行批次无法同时容纳、但各调用可独立容纳时，按原顺序执行。

    不依据工具名字或科研内容调度。原批次可容纳时，仍使用原来的并行 ToolNode。
    单个调用也无法容纳时，在执行任何工具之前保留原有拒绝语义。
    """

    def _subset(self, state, calls):
        messages = list(state["messages"])
        latest = messages[-1]
        if isinstance(latest, Mapping):
            latest = {**latest, "tool_calls": calls}
        else:
            latest = latest.model_copy(update={"tool_calls": calls})
        return {**state, "messages": [*messages[:-1], latest]}

    def _cost(self, state):
        required = dict(self.requirements(state))
        for resource in self.governor.catalog.all():
            if resource.pinned_min_level > ResidencyLevel.CATALOG:
                required[resource.capability_id] = max(
                    required.get(resource.capability_id, ResidencyLevel.CATALOG), resource.pinned_min_level)
        pending = list(required)
        while pending:
            resource_id = pending.pop()
            resource = self.governor.catalog.get(resource_id)
            for dependency in resource.dependencies:
                previous = required.get(dependency.capability_id, ResidencyLevel.CATALOG)
                if dependency.required_level > previous:
                    required[dependency.capability_id] = dependency.required_level
                    pending.append(dependency.capability_id)
        total = ResourceVector()
        for resource_id, level in required.items():
            resource = self.governor.catalog.get(resource_id)
            total += resource.adapter.declared_cost(max(level, resource.pinned_min_level))
        return total

    def execution_states(self, state):
        """所有调用都通过单项预计检查后，才生成拆分方案。"""
        if not isinstance(state, Mapping) or not state.get("messages"):
            return [state]
        latest = state["messages"][-1]
        calls = latest.get("tool_calls", []) if isinstance(latest, Mapping) else latest.tool_calls
        if len(calls) < 2 or self._cost(state).fits(self.governor.budget):
            return [state]
        separate = [self._subset(state, [call]) for call in calls]
        if any(not self._cost(part).fits(self.governor.budget) for part in separate):
            # 交回原 bridge 作正式拒绝，不先运行批次中其他工具。
            return [state]
        return separate

    @staticmethod
    def _merge(outputs):
        if len(outputs) == 1:
            return outputs[0]
        if any(not isinstance(output, Mapping) or set(output) - {"messages"} for output in outputs):
            raise TypeError("Budgeted ToolNode requires ordinary message-only outputs")
        return {"messages": [message for output in outputs for message in output.get("messages", [])]}

    def invoke(self, tool_node, state, config=None, *, request_id="langgraph-toolnode"):
        parts = self.execution_states(state)
        outputs = [super(ResourceBudgetedToolBridge, self).invoke(
            tool_node, part, config, request_id=f"{request_id}/part-{index}")
            for index, part in enumerate(parts)]
        return self._merge(outputs)

    async def ainvoke(self, tool_node, state, config=None, *, request_id="langgraph-toolnode"):
        parts = self.execution_states(state)
        outputs = []
        for index, part in enumerate(parts):
            outputs.append(await super().ainvoke(tool_node, part, config,
                                                  request_id=f"{request_id}/part-{index}"))
        return self._merge(outputs)
