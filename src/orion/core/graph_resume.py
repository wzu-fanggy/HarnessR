"""为旧挂起节点保留恢复入口；新回合使用三节点图，不迁移 checkpoint。"""
from langgraph.types import Command


class ResumeCompatibleGraph:
    def __init__(self, primary, legacy_factory):
        self.primary = primary
        self._legacy_factory = legacy_factory
        self._legacy = None

    def __getattr__(self, name):
        return getattr(self.primary, name)

    def _legacy_graph(self):
        if self._legacy is None:
            self._legacy = self._legacy_factory()
        return self._legacy

    @staticmethod
    def _needs_legacy(snapshot):
        names = set(snapshot.next or ()) | {task.name for task in snapshot.tasks}
        return bool(names & {"pre_tool_gate", "review", "grounding"})

    def _select(self, value, config):
        if config and self._needs_legacy(self.get_state(config)):
            if value is not None and not isinstance(value, Command):
                raise ValueError("旧回合仍挂起，请先恢复或明确取消，不能用新输入覆盖")
            return self._legacy_graph()
        return self.primary

    async def _aselect(self, value, config):
        if config and self._needs_legacy(await self.aget_state(config)):
            if value is not None and not isinstance(value, Command):
                raise ValueError("旧回合仍挂起，请先恢复或明确取消，不能用新输入覆盖")
            return self._legacy_graph()
        return self.primary

    def get_state(self, config, **kwargs):
        # 仅用新图读取会忽略旧节点的待执行 channel，错误地显示为已完成。
        return self._legacy_graph().get_state(config, **kwargs)

    async def aget_state(self, config, **kwargs):
        return await self._legacy_graph().aget_state(config, **kwargs)

    def update_state(self, config, values, **kwargs):
        graph = self._legacy_graph() if self._needs_legacy(self.get_state(config)) else self.primary
        return graph.update_state(config, values, **kwargs)

    async def aupdate_state(self, config, values, **kwargs):
        graph = self._legacy_graph() if self._needs_legacy(await self.aget_state(config)) else self.primary
        return await graph.aupdate_state(config, values, **kwargs)

    def invoke(self, input, config=None, **kwargs):
        return self._select(input, config).invoke(input, config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        graph = await self._aselect(input, config)
        return await graph.ainvoke(input, config, **kwargs)

    def stream(self, input, config=None, **kwargs):
        yield from self._select(input, config).stream(input, config, **kwargs)

    async def astream(self, input, config=None, **kwargs):
        graph = await self._aselect(input, config)
        async for item in graph.astream(input, config, **kwargs):
            yield item

    async def astream_events(self, input, config=None, **kwargs):
        graph = await self._aselect(input, config)
        async for item in graph.astream_events(input, config, **kwargs):
            yield item
