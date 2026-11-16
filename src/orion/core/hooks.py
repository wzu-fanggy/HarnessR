"""薄 hook 层：让横切逻辑挂在节点外围，而不写进节点/循环本身。



对齐博客《Learn Claude Code》s04 的哲学（"Hang on the Loop, Don't Write

into It"），但落在 LangGraph 的图范式上：



- 每个节点通过 ``hookify("<name>")`` 包装后，自动获得 ``before_<name>`` 与

  ``after_<name>`` 两个挂载点；

- 行为通过 ``register_hook(event, callback)`` 注册，加行为 = 加一行注册，

  不修改节点函数本身；

- hook 返回语义只有三种：

    ``None``     → 放行，不干预（纯观察，如记日志）

    ``dict``     → 更新局部视图；persist_before=True 时随节点增量持久化

    ``Command``  → 改路由（拦截，如 goto 到其他节点）



注意：这一层只管"横切逻辑的组织"，不替代 LangGraph 的 ``interrupt()`` /

checkpoint（那些是硬能力）。after 回调读取节点更新后的状态，只有 Command 会短路。

"""

from __future__ import annotations



import inspect

from typing import Any, Callable



from langgraph.types import Command, RunnableConfig

from langgraph.graph.message import add_messages



# 事件名 → 回调列表。事件名约定为 before_<node> / after_<node>。

_HOOKS: dict[str, list[Callable]] = {}





def register_hook(event: str, callback: Callable) -> None:

    """给某个事件注册一个 hook。事件名形如 ``before_llm`` / ``after_tools``。"""

    _HOOKS.setdefault(event, []).append(callback)









def collect_hooks(event: str, state: Any, config: Any) -> list[Any]:

    """收集型 hook：遍历所有回调，汇总非 None 的返回值（list 展平一层）。



    用于每个回调贡献内容的场景，例如新回合开始时注入任务合同与任务板。

    此接口返回内容列表，不承担节点状态更新或路由。

    """

    out: list[Any] = []

    for callback in _HOOKS.get(event, []):

        result = callback(state, config)

        if result is None:

            continue

        if isinstance(result, (list, tuple)):

            out.extend(result)

        else:

            out.append(result)

    return out













def _merge_updates(*updates):

    """合并增量；messages 保留操作顺序，交给图的 reducer 最终应用。"""

    merged = {}

    for update in updates:

        for key, value in (update or {}).items():

            if key == "messages":

                merged[key] = [*merged.get(key, []), *value]

            else:

                merged[key] = value

    return merged





def _state_with_update(state, update):

    result = {**state, **(update or {})}

    if update and "messages" in update:

        result["messages"] = add_messages(state.get("messages", []), update["messages"])

    return result





def _with_command_update(command, update):

    return Command(graph=command.graph, goto=command.goto, resume=command.resume,

                   update=_merge_updates(update, command.update))





def hookify(node_name: str, *, before=(), after=(), persist_before=False):

    """把节点包一层：调用前后各触发一次 before/after hook。



    局部回调随图构造，避免多次 build 时污染全局注册表。

    before 默认只改变模型输入视图；审批等需要持久化的增量显式启用 persist_before。

    普通更新继续后续回调，只有 Command 才短路；after 读取节点更新后的完整状态。

    """



    def deco(node_fn):

        if inspect.iscoroutinefunction(node_fn):



            async def wrapped_async(state, config: RunnableConfig | None = None):

                original, pending = state, {}

                for callback in [*_HOOKS.get(f"before_{node_name}", []), *before]:

                    update = callback(state, config)

                    if inspect.isawaitable(update):

                        update = await update

                    if isinstance(update, Command):

                        return _with_command_update(update, pending)

                    if update:

                        state = (_state_with_update(state, update) if persist_before

                                 else {**state, **update})

                        if persist_before:

                            pending = _merge_updates(pending, update)

                result = await node_fn(state, config)

                if isinstance(result, Command):

                    return _with_command_update(result, pending)

                pending = _merge_updates(pending, result)

                state = _state_with_update(original, pending)

                for callback in [*_HOOKS.get(f"after_{node_name}", []), *after]:

                    update = callback(state, config)

                    if inspect.isawaitable(update):

                        update = await update

                    if isinstance(update, Command):

                        return _with_command_update(update, pending)

                    if update:

                        pending = _merge_updates(pending, update)

                        state = _state_with_update(state, update)

                return pending



            wrapped_async.__name__ = node_fn.__name__

            wrapped_async.__annotations__["config"] = RunnableConfig

            return wrapped_async



        def wrapped_sync(state, config: RunnableConfig | None = None):

            original, pending = state, {}

            for callback in [*_HOOKS.get(f"before_{node_name}", []), *before]:

                update = callback(state, config)

                if inspect.isawaitable(update):

                    getattr(update, "close", lambda: None)()

                    raise TypeError("同步节点不能执行异步 Hook；请使用异步图")

                if isinstance(update, Command):

                    return _with_command_update(update, pending)

                if update:

                    state = (_state_with_update(state, update) if persist_before

                             else {**state, **update})

                    if persist_before:

                        pending = _merge_updates(pending, update)

            result = node_fn(state, config)

            if isinstance(result, Command):

                return _with_command_update(result, pending)

            pending = _merge_updates(pending, result)

            state = _state_with_update(original, pending)

            for callback in [*_HOOKS.get(f"after_{node_name}", []), *after]:

                update = callback(state, config)

                if inspect.isawaitable(update):

                    getattr(update, "close", lambda: None)()

                    raise TypeError("同步节点不能执行异步 Hook；请使用异步图")

                if isinstance(update, Command):

                    return _with_command_update(update, pending)

                if update:

                    pending = _merge_updates(pending, update)

                    state = _state_with_update(state, update)

            return pending



        wrapped_sync.__name__ = node_fn.__name__

        wrapped_sync.__annotations__["config"] = RunnableConfig

        return wrapped_sync



    return deco

