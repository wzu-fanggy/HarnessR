"""在异步图调用边界建立任务作用域；不把会话当任务，不持久化预测。"""
from collections.abc import Mapping
from contextlib import asynccontextmanager, aclosing
import hashlib
import json
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, convert_to_messages
from .task_scope import current_user_message, current_task_messages


def bind_resource_task_graph(graph, bridge, *, role):
    shared = getattr(bridge, 'shared_runtime', None)
    coordinator = getattr(shared, 'demand_coordinator', None)
    if coordinator is None:
        return graph
    if isinstance(graph, ResourceTaskGraph):
        graph = graph.graph
    return ResourceTaskGraph(graph, coordinator, role)


class ResourceTaskGraph:
    """支持 ainvoke/astream/astream_events；状态读取保持原图行为。"""
    def __init__(self, graph, coordinator, role):
        self.graph, self.coordinator, self.role = graph, coordinator, role

    def __getattr__(self, name):
        # 这些执行组合器会返回未包裹的底层图，不能静默绕过任务作用域。
        if name in {'batch', 'abatch', 'batch_as_completed', 'abatch_as_completed',
                    'with_retry', 'with_fallbacks', 'bind', 'pipe', 'astream_log'}:
            raise NotImplementedError('协调模式请使用 ainvoke/astream/astream_events')
        return getattr(self.graph, name)

    def with_config(self, config=None, **kwargs):
        # 原图负责合并配置，包装器保留供身份识别使用的同一配置。
        merged = {**getattr(self, '_bound_config', {}), **(config or {}), **kwargs}
        result = ResourceTaskGraph(self.graph, self.coordinator, self.role)
        result._bound_config = merged
        return result

    def _config(self, config):
        base = getattr(self, '_bound_config', {})
        return {**base, **(config or {}), 'configurable': {
            **base.get('configurable', {}), **(config or {}).get('configurable', {})}}

    async def _input(self, value, config):
        if isinstance(value, Mapping) and 'messages' in value:
            messages = convert_to_messages(value['messages'])
            user_index = next((i for i in range(len(messages)-1, -1, -1)
                if isinstance(messages[i], HumanMessage)
                and not messages[i].additional_kwargs.get('orion_internal')), None)
            if user_index is None:
                raise ValueError('协调任务入口缺少真实用户输入')
            user = messages[user_index]
            if not user.id:
                user = user.model_copy(update={'id': str(uuid4())})
                messages[user_index] = user
            return {**value, 'messages': messages}, user, False, [], []
        # None/Command 恢复必须读取已保存的任务边界，不能只靠 thread_id 推断目标。
        snapshot = await self.graph.aget_state(config)
        user = current_user_message(snapshot.values)
        if user is None or not user.id:
            raise ValueError('恢复任务缺少可验证的用户消息边界')
        return (value, user, True, snapshot.values.get('runtime_board_task_ids', []),
                current_task_messages(snapshot.values))

    def _restore_observations(self, messages):
        """从检查点恢复最近完整工具批次；未返回或不配对的调用不冒充已执行。"""
        batches = []
        for index, message in enumerate(messages):
            if not isinstance(message, AIMessage) or not message.tool_calls:
                continue
            calls = message.tool_calls
            ids = {call['id'] for call in calls}
            results = messages[index + 1:index + 1 + len(calls)]
            if (len(ids) != len(calls) or len(results) != len(calls)
                    or not all(isinstance(result, ToolMessage) for result in results)
                    or {result.tool_call_id for result in results} != ids):
                continue
            batches.append((calls, results))
        token = self.coordinator.token()
        for calls, results in batches[-4:]:
            self.coordinator.select_tool_calls(token, calls)
            self.coordinator.record_tool_results(token, [call['name'] for call in calls],
                                                 {'messages': results})

    @asynccontextmanager
    async def _scope(self, value, config):
        value, user, resumed, restored_ids, restored_messages = await self._input(value, config)
        configurable = config.get('configurable', {})
        identity = [configurable.get('thread_id'), configurable.get('checkpoint_ns'), user.id]
        task_id = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        parent = self.coordinator.token()
        metadata = {'role': self.role, 'run_id': str(uuid4()), 'message_id': user.id,
                    'parent_token': parent, 'resumed': resumed}
        intents = getattr(self.coordinator, 'task_intents', None)
        if intents is not None:
            from .task_board_intent import board_task_ids
            inherited = intents.ids(parent) if parent is not None else []
            ids = board_task_ids(configurable.get('resource_board_task_ids',
                restored_ids if resumed else inherited))
            metadata['board_task_ids'] = ids
            if not resumed:
                value = {**value, 'runtime_board_task_ids': ids}
        async with self.coordinator.task(task_id, goal=str(user.content), metadata=metadata):
            if resumed and intents is not None:
                try:
                    self._restore_observations(restored_messages)
                except Exception as error:
                    self.coordinator.events.append({'event':'context_restore_failed',
                                                    'error':type(error).__name__})
            yield value

    async def ainvoke(self, input, config=None, **kwargs):
        config = self._config(config)
        async with self._scope(input, config) as value:
            return await self.graph.ainvoke(value, config, **kwargs)

    async def astream(self, input, config=None, **kwargs):
        config = self._config(config)
        async with self._scope(input, config) as value:
            async with aclosing(self.graph.astream(value, config, **kwargs)) as stream:
                async for item in stream:
                    yield item

    async def astream_events(self, input, config=None, **kwargs):
        config = self._config(config)
        async with self._scope(input, config) as value:
            async with aclosing(self.graph.astream_events(value, config, **kwargs)) as stream:
                async for item in stream:
                    yield item

    def invoke(self, *args, **kwargs):
        raise ValueError('共享协调模式需要异步图调用')

    def stream(self, *args, **kwargs):
        raise ValueError('共享协调模式需要异步图调用')
