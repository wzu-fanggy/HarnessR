"""共享运行时合同：等待、复用、公平、取消、失败与排空，不使用 LLM/KG。"""
import asyncio
import threading

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from orion.core.resource_tool_node import LeaseSafeToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.runnables import RunnableConfig

from orion.capvm import (
    CapabilityCatalog, CapabilityKind, CapabilityResidencyGovernor,
    ResidencyBudgetExceeded, ResidencyLevel, ResourceVector, VirtualCapability,
)
from orion.capvm.model import CapabilityDependency, MeasuredResidencyBudgetExceeded
from orion.capvm.adapters.base import BaseCapabilityAdapter
from orion.capvm.shared_runtime import SharedResourceRuntime
from orion.core.langgraph_residency import (
    ToolCallResidencyBridge, ToolResidencyBinding, RuntimeResourceRequirement,
    get_active_runtime_resource,
)

BOUND = ResidencyLevel.BOUND


class Resource(BaseCapabilityAdapter):
    def __init__(self, cost=1, *, fail=False, measured=None):
        self.cost, self.fail, self.measured = cost, fail, measured
        self.starts = 0
        self.live = False

    def declared_cost(self, level):
        return ResourceVector(process_slots=self.cost if level >= BOUND else 0)

    def measured_cost(self, level):
        return ResourceVector(process_slots=(self.measured if self.measured is not None else self.cost)
                              if self.live else 0)

    def promote(self, current, target):
        if target >= BOUND:
            self.live = True
            self.starts += 1
            if self.fail:
                raise RuntimeError('establishment failed')

    def demote(self, current, target):
        if target < BOUND:
            self.live = False


def make_runtime(costs=(1, 1, 1), budget=1):
    resources = [Resource(cost) for cost in costs]
    governor = CapabilityResidencyGovernor(catalog=CapabilityCatalog([
        VirtualCapability(str(i), CapabilityKind.PROCESS, '测试资源', resource)
        for i, resource in enumerate(resources)
    ]), budget=ResourceVector(process_slots=budget))
    return SharedResourceRuntime(governor), resources


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0)


async def invoke_graph(bridge, tools, state):
    node = LeaseSafeToolNode(tools, handle_tool_errors=False)
    async def execute(state, config: RunnableConfig):
        return await bridge.ainvoke(node, state, config)
    graph = StateGraph(MessagesState)
    graph.add_node('tools', execute)
    graph.add_edge(START, 'tools')
    graph.add_edge('tools', END)
    return await graph.compile().ainvoke(state)


async def test_shared_resource_is_loaded_once_for_overlapping_requests():
    runtime, resources = make_runtime()
    ready, done = asyncio.Event(), asyncio.Event()
    async def first():
        async with runtime.acquire([('0', BOUND)]):
            ready.set()
            await done.wait()
    task = asyncio.create_task(first())
    await ready.wait()
    async with runtime.acquire([('0', BOUND)]):
        assert runtime.governor.state('0').active_count == 2
        assert resources[0].starts == 1
    done.set()
    await task
    await runtime.aclose()


async def test_capacity_conflict_waits_and_fifo_prevents_overtaking():
    runtime, resources = make_runtime()
    entered, release = asyncio.Event(), asyncio.Event()
    order = []
    async def use(index, hold=False):
        async with runtime.acquire([(str(index), BOUND)], request_id=str(index)):
            order.append(index)
            if hold:
                entered.set()
                await release.wait()
    first = asyncio.create_task(use(0, True))
    await entered.wait()
    second = asyncio.create_task(use(1))
    await until(lambda: len(runtime.snapshot()['waiting']) == 1)
    third = asyncio.create_task(use(2))
    await until(lambda: len(runtime.snapshot()['waiting']) == 2)
    assert order == [0] and resources[0].live
    release.set()
    await asyncio.gather(first, second, third)
    assert order == [0, 1, 2]
    assert not runtime.snapshot()['active']
    await runtime.aclose()


async def test_nested_request_rejected_instead_of_deadlocking():
    runtime, resources = make_runtime(costs=(1, 2))
    async with runtime.acquire([('0', BOUND)]):
        with pytest.raises(RuntimeError, match='nested admission'):
            async with runtime.acquire([('1', BOUND)]):
                pass
    # 同一任务的嵌套租约先拒绝死锁；永久超限在独立请求中核查。
    await runtime.aclose()


async def test_intrinsic_oversize_and_timeout_zero_are_distinct():
    runtime, resources = make_runtime(costs=(2,))
    with pytest.raises(ResidencyBudgetExceeded):
        async with runtime.acquire([('0', BOUND)], timeout=0):
            pass
    assert resources[0].starts == 0
    assert runtime.snapshot()['events'][-1]['detail'] == 'intrinsic_capacity'
    await runtime.aclose()


@pytest.mark.parametrize('cancel', [True, False])
async def test_waiter_cancel_or_timeout_leaves_no_queue_entry(cancel):
    runtime, _ = make_runtime()
    async def waiter():
        async with runtime.acquire([('1', BOUND)], timeout=None if cancel else 0):
            pytest.fail('must wait')
    async with runtime.acquire([('0', BOUND)]):
        task = asyncio.create_task(waiter())
        if cancel:
            await until(lambda: bool(runtime.snapshot()['waiting']))
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
        assert not runtime.snapshot()['waiting']
        assert runtime.governor.state('0').active_count == 1
    await runtime.aclose()


async def test_failure_cleanup_does_not_poison_next_request():
    runtime, resources = make_runtime()
    resources[0].fail = True
    with pytest.raises(RuntimeError, match='establishment failed'):
        async with runtime.acquire([('0', BOUND)]):
            pass
    assert not resources[0].live
    assert not runtime.snapshot()['active']
    async with runtime.acquire([('1', BOUND)]):
        assert resources[1].live
    await runtime.aclose()


async def test_measured_overrun_cleans_preparation_without_retry_loop():
    runtime, resources = make_runtime()
    resources[0].measured = 2
    with pytest.raises(MeasuredResidencyBudgetExceeded):
        async with runtime.acquire([('0', BOUND)]):
            pass
    assert not resources[0].live
    assert resources[0].starts == 1
    await runtime.aclose()


async def test_shutdown_drains_active_and_rejects_waiter():
    runtime, _ = make_runtime()
    release, entered = asyncio.Event(), asyncio.Event()
    async def active():
        async with runtime.acquire([('0', BOUND)]):
            entered.set()
            await release.wait()
    async def queued():
        async with runtime.acquire([('1', BOUND)]):
            pytest.fail('closed queue must not execute')
    task = asyncio.create_task(active())
    await entered.wait()
    waiting = asyncio.create_task(queued())
    await until(lambda: bool(runtime.snapshot()['waiting']))
    closing = asyncio.create_task(runtime.aclose())
    with pytest.raises(RuntimeError, match='closed'):
        await waiting
    assert not closing.done()
    release.set()
    await asyncio.gather(task, closing)
    assert not runtime.snapshot()['active']


async def test_sync_close_cannot_destroy_an_active_lease():
    runtime, resources = make_runtime()
    async with runtime.acquire([('0', BOUND)]):
        with pytest.raises(RuntimeError, match='pending work'):
            runtime.close()
        assert resources[0].live
    runtime.close()
    with pytest.raises(RuntimeError, match='closed'):
        async with runtime.acquire([('0', BOUND)]):
            pass


async def test_default_factory_enables_queue_without_loading_resources(monkeypatch):
    from orion.core.default_resource_runtime import build_default_tool_resource_runtime
    monkeypatch.setenv('ORION_RESOURCE_POOL_ENABLED', '1')
    monkeypatch.setenv('ORION_RESOURCE_ADMISSION_QUEUE', '1')
    monkeypatch.delenv('ORION_RESOURCE_EVENTS_PATH', raising=False)
    runtime = build_default_tool_resource_runtime()
    assert isinstance(runtime.bridge.shared_runtime, SharedResourceRuntime)
    assert not any(row['resident'] for row in runtime.snapshot())
    await runtime.aclose()
    runtime.close()


async def test_dependency_closure_is_injected_into_tool_context():
    parent, dependency = Resource(0), Resource(1)
    governor = CapabilityResidencyGovernor(catalog=CapabilityCatalog([
        VirtualCapability('parent', CapabilityKind.PROCESS, '工具资源', parent,
            dependencies=(CapabilityDependency('dependency', BOUND),)),
        VirtualCapability('dependency', CapabilityKind.MODEL, '共享模型', dependency),
    ]), budget=ResourceVector(process_slots=1))
    runtime = SharedResourceRuntime(governor)
    @tool
    async def check() -> str:
        """验证完整依赖注入。"""
        assert get_active_runtime_resource('dependency') is dependency
        assert governor.state('dependency').active_count == 1
        return 'ok'
    bridge = ToolCallResidencyBridge(governor=governor,
        bindings={'check': ToolResidencyBinding(resources=(RuntimeResourceRequirement('parent'),))},
        shared_runtime=runtime)
    state = {'messages': [AIMessage(content='', tool_calls=[
        {'name': 'check', 'args': {}, 'id': 'check-1', 'type': 'tool_call'}])]}
    result = await invoke_graph(bridge, [check], state)
    assert result['messages'][-1].content == 'ok'
    assert get_active_runtime_resource('dependency') is None
    await runtime.aclose()


async def test_toolnode_cancellation_keeps_lease_until_sync_worker_exits():
    runtime, resources = make_runtime()
    started, finish = threading.Event(), threading.Event()
    @tool
    def slow() -> str:
        """在线程中等待外部事件。"""
        started.set()
        assert finish.wait(5)
        assert resources[0].live
        return 'done'
    bridge = ToolCallResidencyBridge(governor=runtime.governor,
        bindings={'slow': ToolResidencyBinding(resources=(RuntimeResourceRequirement('0'),))},
        shared_runtime=runtime)
    state = {'messages': [AIMessage(content='', tool_calls=[
        {'name': 'slow', 'args': {}, 'id': 'slow-1', 'type': 'tool_call'}])]}
    task = asyncio.create_task(invoke_graph(bridge, [slow], state))
    try:
        await until(started.is_set)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert runtime.governor.state('0').active_count == 1
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.governor.state('0').active_count == 0
    await runtime.aclose()


async def test_batch_failure_waits_for_sibling_before_releasing_resources():
    runtime, resources = make_runtime()
    entered, finish = asyncio.Event(), asyncio.Event()
    @tool
    async def fails() -> str:
        """等同批工具启动后抛出异常。"""
        await entered.wait()
        raise ValueError('tool failed')
    @tool
    async def sibling() -> str:
        """失败传播前仍在使用共享资源。"""
        entered.set()
        await finish.wait()
        assert resources[0].live
        return 'done'
    binding = ToolResidencyBinding(resources=(RuntimeResourceRequirement('0'),))
    bridge = ToolCallResidencyBridge(governor=runtime.governor,
        bindings={'fails': binding, 'sibling': binding}, shared_runtime=runtime)
    state = {'messages': [AIMessage(content='', tool_calls=[
        {'name': name, 'args': {}, 'id': name, 'type': 'tool_call'}
        for name in ('fails', 'sibling')])]}
    task = asyncio.create_task(invoke_graph(bridge, [fails, sibling], state))
    await entered.wait()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not task.done()
    assert runtime.governor.state('0').active_count == 1
    finish.set()
    with pytest.raises(ValueError, match='tool failed'):
        await task
    assert runtime.governor.state('0').active_count == 0
    await runtime.aclose()
