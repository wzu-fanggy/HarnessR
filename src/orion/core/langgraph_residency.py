"""Bridge LangGraph ToolNode execution to the capability residency governor.



The bridge does not choose tools. It observes tool calls that an upstream LangGraph

agent has already emitted, materializes the registered runtime capabilities and their

dependencies, keeps them active for the ToolNode invocation, and leaves the resulting

idle objects available for later reuse or pressure-driven release.

"""



from __future__ import annotations



import asyncio

import json

from collections.abc import Mapping

from contextlib import contextmanager

from contextvars import ContextVar

from dataclasses import dataclass

from typing import Any, Iterator



from orion.capvm import CapabilityResidencyGovernor, ResidencyLevel

from orion.capvm.host_memory import HostMemoryUnavailable

from orion.capvm.model import MeasuredResidencyBudgetExceeded, ResidencyBudgetExceeded





_ADMISSION_ERRORS = (ResidencyBudgetExceeded, MeasuredResidencyBudgetExceeded,

                     HostMemoryUnavailable, TimeoutError)





_active_runtime_resources: ContextVar[Mapping[str, Any]] = ContextVar(

    "orion_active_runtime_resources",

    default={},

)





_managed_tool_execution: ContextVar[bool] = ContextVar("orion_managed_tool_execution", default=False)





def in_managed_tool_execution() -> bool:

    """当前工具是否位于共享运行时内；线程工具继承这一上下文。"""

    return _managed_tool_execution.get()





def get_active_runtime_resource(resource_id: str) -> Any | None:

    """Return a resource adapter only while its ToolNode lease is active."""



    return _active_runtime_resources.get().get(resource_id)





@contextmanager

def isolated_runtime_resources():

    """子图只使用本次取得的资源，不继承父协程可能已归还的资源引用。"""

    token = _active_runtime_resources.set({})

    try:

        yield

    finally:

        _active_runtime_resources.reset(token)





@dataclass(frozen=True)

class RuntimeResourceRequirement:

    """One statically declared runtime resource required by a tool call."""



    resource_id: str

    required_level: ResidencyLevel = ResidencyLevel.BOUND



    def __post_init__(self) -> None:

        if not self.resource_id.strip():

            raise ValueError("resource_id is required")

        if self.required_level == ResidencyLevel.ACTIVE:

            raise ValueError("ACTIVE residency must be entered through governor.acquire()")





@dataclass(frozen=True)

class ToolResidencyBinding:

    """Static resource requirements for one LangGraph tool call."""



    capability_id: str | None = None

    required_level: ResidencyLevel = ResidencyLevel.INTERFACE

    resources: tuple[RuntimeResourceRequirement, ...] = ()

    no_runtime_resources: bool = False



    def __post_init__(self) -> None:

        if not isinstance(self.no_runtime_resources, bool):

            raise ValueError("no_runtime_resources must be boolean")

        if self.no_runtime_resources and (self.capability_id is not None or self.resources):

            raise ValueError("zero-backend declaration cannot contain resource requirements")

        if not self.no_runtime_resources and self.capability_id is None and not self.resources:

            raise ValueError("binding requires capability_id or resources")

        if self.capability_id is not None and not self.capability_id.strip():

            raise ValueError("capability_id cannot be blank")

        if self.required_level == ResidencyLevel.ACTIVE:

            raise ValueError("ACTIVE residency must be entered through governor.acquire()")



    def requirements(self) -> tuple[tuple[str, ResidencyLevel], ...]:

        values: list[tuple[str, ResidencyLevel]] = []

        if self.capability_id is not None:

            values.append((self.capability_id, self.required_level))

        values.extend((item.resource_id, item.required_level) for item in self.resources)

        return tuple(values)





class ToolCallResidencyBridge:

    """Hold runtime dependencies resident while a LangGraph ToolNode executes.



    Multiple tool calls in the same ToolNode batch are acquired together so a parallel

    batch cannot evict one of its own active dependencies. Unregistered tools may either

    pass through unchanged or fail fast, controlled by ``strict``.

    """



    def __init__(

        self,

        *,

        governor: CapabilityResidencyGovernor,

        bindings: Mapping[str, ToolResidencyBinding],

        strict: bool = False,

        reconcile_measured: bool = True,

        observation_recorder: Any | None = None,

        shared_runtime: Any | None = None,

        admission_timeout: float | None = None,

        admission_error_feedback: bool = False,

    ) -> None:

        self.governor = governor

        self.bindings = dict(bindings)

        self.strict = bool(strict)

        self.reconcile_measured = bool(reconcile_measured)

        self.observation_recorder = observation_recorder

        if shared_runtime is not None and shared_runtime.governor is not governor:

            raise ValueError("bridge and shared runtime must use the same governor")

        self.shared_runtime = shared_runtime

        self.admission_timeout = admission_timeout

        self.admission_error_feedback = bool(admission_error_feedback)



    def _admission_feedback(self, state, error, request_id):

        """只为尚未执行的完整批次返回事实，不替 Agent 决定下一行动。"""

        if not self.admission_error_feedback:

            return None

        messages = state.get("messages", []) if isinstance(state, Mapping) else getattr(state, "messages", [])

        message = messages[-1] if messages else None

        calls = message.get("tool_calls", ()) if isinstance(message, Mapping) else getattr(message, "tool_calls", ())

        paired = []

        for call in calls or ():

            if not isinstance(call, Mapping):

                return None

            function = call.get("function")

            name = call.get("name") or (function.get("name") if isinstance(function, Mapping) else None)

            call_id = call.get("id")

            if not isinstance(name, str) or not name.strip() or not isinstance(call_id, str) or not call_id.strip():

                return None

            paired.append((name, call_id))

        # 不制造调用 ID；无法完整配对时保留原异常。

        if not paired or len({call_id for _, call_id in paired}) != len(paired):

            return None

        if isinstance(error, ResidencyBudgetExceeded):

            reason = "declared_capacity_exceeded"

        elif isinstance(error, MeasuredResidencyBudgetExceeded):

            reason = "measured_capacity_exceeded"

        elif isinstance(error, HostMemoryUnavailable):

            reason = "host_memory_unavailable"

        else:

            reason = "admission_timeout"

        payload = {

            "kind": "resource_admission_failure",

            "execution_status": "not_executed",

            "reason": reason,

            "request_id": request_id,

            "required_resources": [r for r, _ in self.requirements(state)],

            "violated_dimensions": list(getattr(error, "violated_dimensions", ())),

            "message": "本批次未通过资源准入，所有工具函数均未执行。这不是科研结果或任务完成证明。",

        }

        from langchain_core.messages import ToolMessage

        return {"messages": [ToolMessage(content=json.dumps(payload, ensure_ascii=False),

            name=name, tool_call_id=call_id, status="error") for name, call_id in paired]}



    def tool_names(self, state: Any) -> tuple[str, ...]:

        """Return tool names from the latest LangGraph message in deterministic order."""



        if isinstance(state, Mapping):

            messages = state.get("messages") or []

        else:

            messages = getattr(state, "messages", []) or []

        if not messages:

            return ()

        message = messages[-1]

        calls = getattr(message, "tool_calls", None)

        if calls is None and isinstance(message, Mapping):

            calls = message.get("tool_calls")

        names: list[str] = []

        for call in calls or ():

            if isinstance(call, Mapping):

                name = call.get("name")

                if not name:

                    function = call.get("function")

                    if isinstance(function, Mapping):

                        name = function.get("name")

            else:

                name = getattr(call, "name", None)

            text = str(name or "").strip()

            if text:

                names.append(text)

        return tuple(names)



    def requirements(self, state: Any) -> tuple[tuple[str, ResidencyLevel], ...]:

        """Resolve the latest ToolNode batch to a merged logical working set."""



        merged: dict[str, ResidencyLevel] = {}

        missing: list[str] = []

        for tool_name in self.tool_names(state):

            binding = self.bindings.get(tool_name)

            if binding is None:

                missing.append(tool_name)

                continue

            for resource_id, required_level in binding.requirements():

                previous = merged.get(resource_id, ResidencyLevel.CATALOG)

                merged[resource_id] = max(previous, required_level)

        if missing and self.strict:

            raise KeyError(

                "missing residency binding for ToolNode call(s): " + ", ".join(sorted(set(missing)))

            )

        return tuple(sorted(merged.items(), key=lambda item: item[0]))



    @contextmanager

    def acquire_for_state(

        self,

        state: Any,

        *,

        request_id: str = "langgraph-toolnode",

    ) -> Iterator[tuple[tuple[str, ResidencyLevel], ...]]:

        """Materialize and pin the current ToolNode batch for the duration of execution."""



        if self.shared_runtime is not None:

            raise RuntimeError("shared runtime requires ainvoke on its owning event loop")

        requirements = self.requirements(state)

        if not requirements:

            yield requirements

            return

        entered = False

        try:

            with self.governor.acquire_many(

                requirements, request_id=request_id,

                reconcile_before_active=self.reconcile_measured,

            ):

                entered = True

                with self._bind_resources(state, requirements, request_id):

                    yield requirements

        finally:

            if entered:

                self._record_observation("lease_released", tool_names=self.tool_names(state),

                    requirements=requirements, request_id=request_id)



    @contextmanager

    def _bind_resources(self, state, requirements, request_id):

        """只绑定已取得租约的完整依赖；同步与异步准入共用该上下文。"""

        tool_names = self.tool_names(state)

        self._record_observation("lease_acquired", tool_names=tool_names,

            requirements=requirements, request_id=request_id)

        previous = dict(_active_runtime_resources.get())

        leased = {

            resource_id: self.governor.catalog.get(resource_id).adapter

            for resource_id, _ in self.governor.expand_requirements(requirements)

        }

        token = _active_runtime_resources.set({**previous, **leased})

        managed = _managed_tool_execution.set(self.shared_runtime is not None)

        try:

            yield requirements

        finally:

            _managed_tool_execution.reset(managed)

            _active_runtime_resources.reset(token)

            observer = getattr(self.governor.policy, "observe_tool_batch", None)

            if observer is not None:

                try:

                    observer(tool_names, tuple(item for item, _ in requirements),

                             catalog=self.governor.catalog)

                except Exception:

                    # 预测状态采集失败不能覆盖工具执行结果或阻止租约归还。

                    pass



    def _record_observation(

        self,

        event_type: str,

        *,

        tool_names: tuple[str, ...],

        requirements: tuple[tuple[str, ResidencyLevel], ...],

        request_id: str,

    ) -> None:

        recorder = self.observation_recorder

        if recorder is None:

            return

        try:

            recorder.record(

                event_type,

                governor=self.governor,

                tool_names=tool_names,

                requirements=requirements,

                request_id=request_id,

            )

        except Exception:

            # 观测失败不能改变工具结果、引用计数或资源回收语义。

            pass



    def invoke(

        self,

        tool_node: Any,

        state: Any,

        config: Any = None,

        *,

        request_id: str = "langgraph-toolnode",

    ) -> Any:

        """Invoke a synchronous ToolNode while its capabilities are protected."""



        if self.shared_runtime is not None:

            raise RuntimeError("shared runtime requires ainvoke on its owning event loop")

        entered = False

        try:

            with self.acquire_for_state(state, request_id=request_id):

                entered = True

                return tool_node.invoke(state, config)

        except _ADMISSION_ERRORS as error:

            feedback = None if entered else self._admission_feedback(state, error, request_id)

            if feedback is None:

                raise

            return feedback



    async def ainvoke(

        self,

        tool_node: Any,

        state: Any,

        config: Any = None,

        *,

        request_id: str = "langgraph-toolnode",

    ) -> Any:

        """Invoke an asynchronous ToolNode while its capabilities are protected."""



        if self.shared_runtime is None:

            entered = False

            try:

                with self.acquire_for_state(state, request_id=request_id):

                    entered = True

                    from .resource_tool_node import LeaseSafeToolNode

                    if isinstance(tool_node, LeaseSafeToolNode):

                        return await self._invoke_until_finished(tool_node, state, config)

                    return await tool_node.ainvoke(state, config)

            except _ADMISSION_ERRORS as error:

                feedback = None if entered else self._admission_feedback(state, error, request_id)

                if feedback is None:

                    raise

                return feedback

        if len(self.tool_names(state)) > 1:

            from .resource_tool_node import LeaseSafeToolNode

            if not isinstance(tool_node, LeaseSafeToolNode):

                raise TypeError("shared multi-tool batches require LeaseSafeToolNode")

        requirements = self.requirements(state)

        coordinator = self.shared_runtime.demand_coordinator

        if coordinator is not None:

            try:

                messages = state.get('messages', []) if isinstance(state, Mapping) else getattr(state, 'messages', [])

                message = messages[-1] if messages else None

                calls = message.get('tool_calls', ()) if isinstance(message, Mapping) else getattr(message, 'tool_calls', ())

                coordinator.select_tool_calls(coordinator.token(), calls)

            except Exception as error:

                coordinator.events.append({'event':'context_capture_failed', 'error':type(error).__name__})

        entered = False

        try:

            async with self.shared_runtime.acquire(

                requirements, request_id=request_id, timeout=self.admission_timeout,

                reconcile_measured=self.reconcile_measured,

            ):

                entered = True

                with self._bind_resources(state, requirements, request_id):

                    result = await self._invoke_until_finished(tool_node, state, config)

                    coordinator = self.shared_runtime.demand_coordinator

                    if coordinator is not None:

                        try:

                            coordinator.record_tool_results(coordinator.token(), self.tool_names(state), result)

                        except Exception as error:

                            # 观察失败可见，但不得覆盖科学结果或阻止租约释放。

                            coordinator.events.append({'event':'context_capture_failed', 'error':type(error).__name__})

                    if (coordinator is not None and getattr(coordinator, 'task_intents', None) is not None

                            and isinstance(result, dict)):

                        # 只持久化显式任务卡绑定，不向主模型注入资源遥测。

                        result = {**result, 'runtime_board_task_ids':

                                  coordinator.task_intents.ids(coordinator.token())}

                    return result

        except _ADMISSION_ERRORS as error:

            feedback = None if entered else self._admission_feedback(state, error, request_id)

            if feedback is None:

                raise

            if coordinator is not None:

                try:

                    # 清除本批次成功执行的条件假设；错误状态不能确认科研覆盖。

                    coordinator.record_tool_results(coordinator.token(), self.tool_names(state), feedback)

                except Exception as observation_error:

                    coordinator.events.append({'event': 'context_capture_failed',

                                               'error': type(observation_error).__name__})

            return feedback

        finally:

            if entered:

                self._record_observation("lease_released", tool_names=self.tool_names(state),

                    requirements=requirements, request_id=request_id)



    async def _invoke_until_finished(self, tool_node, state, config):

        """安全收尾模式等待底层执行真正结束后再处理取消。"""

        # ToolNode 可能在线程中执行同步工具。取消等待者不能提前释放该线程的资源。

        running = asyncio.ensure_future(tool_node.ainvoke(state, config))

        cancelled = None

        while not running.done():

            try:

                await asyncio.shield(running)

            except asyncio.CancelledError as error:

                cancelled = error

            except BaseException:

                break

        if cancelled is not None:

            # 读取异常避免后台任务警告，并保留取消作为调用者可见的终态。

            if not running.cancelled():

                running.exception()

            raise cancelled

        return running.result()

