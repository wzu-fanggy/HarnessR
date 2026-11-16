"""同一事件循环中共享的资源准入层，不依赖 Agent 或 LangGraph。



所有并发调用须通过同一实例获取租约；适配器的建立和释放仍在控制线程串行执行。

FIFO 控制准入顺序，已准入的工具可并发运行。不提供跨进程或跨线程资源仲裁。

"""

from __future__ import annotations



import asyncio

import math

from collections import deque

from contextlib import asynccontextmanager

from dataclasses import asdict, dataclass

from time import monotonic



from .host_memory import HostMemoryUnavailable

from .model import ResidencyBudgetExceeded, ResidencyLevel, ResourceVector





@dataclass(frozen=True)

class AdmissionEvent:

    """有界的请求审计记录；ticket 区分相同业务 request_id。"""



    ticket: int

    request_id: str

    status: str

    elapsed_s: float

    resources: tuple[str, ...]

    detail: str = ""





class SharedResourceRuntime:

    """在共享 governor 上区分永久容量不足与活动租约造成的暂时竞争。"""



    def __init__(self, governor, *, event_limit=1024, memory_budget=None,

                 host_pressure_reclaim_ids=(), host_pressure_wait_s=5):

        if isinstance(event_limit, bool) or not isinstance(event_limit, int) or event_limit < 1:

            raise ValueError("event_limit must be a positive integer")

        if memory_budget is not None and governor.preparation_guard is None:

            raise ValueError("automatic memory requires a preparation guard")

        if not math.isfinite(host_pressure_wait_s) or host_pressure_wait_s<=0:raise ValueError('invalid host pressure wait')

        self.host_pressure_reclaim_ids=frozenset(host_pressure_reclaim_ids)

        self.host_pressure_wait_s=host_pressure_wait_s

        self.memory_budget = memory_budget

        self.governor = governor

        self.demand_coordinator = None

        self._events = deque(maxlen=event_limit)

        self._queue = deque()

        self._active = {}

        self._changed = asyncio.Event()

        self._loop = None

        self._sequence = 0

        self._closed = False



    def _check_loop(self):

        loop = asyncio.get_running_loop()

        if self._loop is None:

            self._loop = loop

        elif self._loop is not loop:

            raise RuntimeError("SharedResourceRuntime requires one event loop")



    def snapshot(self):

        """返回请求层状态，不扫描尚未驻留的资源目录。"""

        return {

            "closed": self._closed,

            "waiting": tuple(self._queue),

            "active": tuple(self._active),

            "events": tuple(asdict(event) for event in self._events),

        }



    def _record(self, ticket, request_id, status, started, required, detail=""):

        self._events.append(AdmissionEvent(

            ticket, request_id, status, monotonic() - started, tuple(required), detail))



    @asynccontextmanager

    async def acquire(self, requests, *, request_id="", timeout=None, reconcile_measured=True):

        """获取完整依赖的租约；timeout 只限制等待，不中断正在建立或使用的资源。"""

        self._check_loop()

        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):

            raise ValueError("timeout must be finite and non-negative")

        if self._closed:

            raise RuntimeError("runtime admission is closed")

        owner = asyncio.current_task()

        if owner in self._active.values():

            raise RuntimeError("nested admission may deadlock; request the complete working set")

        coordinator = self.demand_coordinator

        demand_token = coordinator.token() if coordinator is not None else None

        requests = tuple(requests)

        required = dict(self.governor.expand_requirements(requests))

        self._sequence += 1

        ticket, started = self._sequence, monotonic()

        if self.memory_budget is not None:

            self.memory_budget.refresh(self.governor)

        # 不看当前活动租约，先检验请求自身和固定驻留依赖的最低占用。

        minimum = dict(self.governor.expand_requirements(

            (*requests, *((item, self.governor.catalog.get(item).pinned_min_level)

                          for item in self.governor.pinned_ids))))

        cost = ResourceVector()

        for item, level in minimum.items():

            cost += self.governor.catalog.get(item).adapter.declared_cost(level)

        violations = cost.violated_dimensions(self.governor.budget)

        if self.host_pressure_reclaim_ids and self.memory_budget is not None:

            # 当前系统余量不是请求永久容量；用户明确设置的上限仍是硬限制。

            limit=self.memory_budget.user_limit_mb

            if limit is None or cost.host_ram_mb<=limit:

                violations=tuple(d for d in violations if d!='host_ram_mb')

        if violations:

            self._record(ticket, request_id, "rejected", started, required, "intrinsic_capacity")

            raise ResidencyBudgetExceeded("__request__", max(required.values(), default=ResidencyLevel.CATALOG), violations)

        self._queue.append(ticket)

        self._record(ticket, request_id, "queued", started, required)

        lease = None

        entered = False

        advice_attempted = False

        prediction_attempted = False

        reclaimed=set();pressure_deadline=None

        try:

            while True:

                if self._closed:

                    raise RuntimeError("runtime admission is closed")

                if self._queue[0] == ticket:

                    if self.memory_budget is not None:

                        self.memory_budget.refresh(self.governor)

                    if coordinator is not None:

                        coordinator.profiler.refresh()

                    if coordinator is not None and not prediction_attempted:

                        prediction_attempted = True

                        remaining = None if timeout is None else max(0, timeout - (monotonic() - started))

                        if remaining is None:

                            await coordinator.prepare_prediction(requests, demand_token)

                        elif remaining > 0:

                            await asyncio.wait_for(coordinator.prepare_prediction(requests, demand_token),remaining)

                        if self._closed:

                            raise RuntimeError("runtime admission is closed")

                    advisor = self.governor.plan_advisor

                    if advisor is not None and not advice_attempted:

                        advice_attempted = True

                        remaining = None if timeout is None else max(0, timeout - (monotonic() - started))

                        if remaining is None:

                            await advisor.prepare(self.governor, requests)

                        elif remaining > 0:

                            await asyncio.wait_for(advisor.prepare(self.governor, requests), remaining)

                        # 等待模型期间其他活动请求可能结束，关闭请求也可能到达。

                        if self._closed:

                            raise RuntimeError("runtime admission is closed")

                    lease = self.governor.acquire_many(requests, request_id=request_id,

                        reconcile_before_active=reconcile_measured)

                    try:

                        capabilities = lease.__enter__()

                    except (ResidencyBudgetExceeded,HostMemoryUnavailable) as error:

                        lease = None

                        host_pressure=(isinstance(error,HostMemoryUnavailable) or

                            getattr(error,'violated_dimensions',())==('host_ram_mb',))

                        if isinstance(error,HostMemoryUnavailable) and not self.host_pressure_reclaim_ids:raise

                        if host_pressure and self.host_pressure_reclaim_ids:

                            guard=self.governor.preparation_guard

                            if isinstance(error,HostMemoryUnavailable) and guard.events[-1]['status']=='unavailable':raise

                            if pressure_deadline is None:pressure_deadline=monotonic()+self.host_pressure_wait_s

                            if monotonic()>=pressure_deadline:raise

                            victim=self._reclaim_idle(required,reclaimed)

                            if victim is not None:

                                reclaimed.add(victim)

                                self._record(ticket,request_id,'host_pressure_reclaimed',started,required,victim)

                                # 回收量不可由声明值替代；下一次循环重新读取真实整机余量。

                                advice_attempted=False

                                continue

                            if self._active:

                                self._record(ticket,request_id,'host_pressure_waiting',started,required,type(error).__name__)

                        # 没有自有活动工作就没有可等待的释放事件，不能无限等待其他软件。

                        if not self._active:raise

                    else:

                        entered = True

                        self._queue.popleft()

                        self._active[ticket] = owner

                        self._changed.set()

                        self._record(ticket, request_id, "admitted", started, required)

                        break

                remaining = None if timeout is None else timeout - (monotonic() - started)

                if pressure_deadline is not None:

                    pressure_remaining=max(0.,pressure_deadline-monotonic())

                    remaining=pressure_remaining if remaining is None else min(remaining,pressure_remaining)

                if remaining is not None and remaining <= 0:

                    raise TimeoutError("resource admission timed out")

                self._changed.clear()

                await asyncio.wait_for(self._changed.wait(), timeout=remaining)

        except BaseException as error:

            if ticket in self._queue:

                self._queue.remove(ticket)

            self._changed.set()

            status = "cancelled" if isinstance(error, asyncio.CancelledError) else (

                "timed_out" if isinstance(error, TimeoutError) else "rejected")

            self._record(ticket, request_id, status, started, required, type(error).__name__)

            raise

        try:

            yield capabilities

        except BaseException as error:

            self._record(ticket, request_id, "execution_failed", started, required, type(error).__name__)

            raise

        finally:

            if entered:

                try:

                    lease.__exit__(None, None, None)

                finally:

                    if coordinator is not None:

                        coordinator.completed(demand_token, tuple(required))

                    self._active.pop(ticket, None)

                    self._changed.set()

                    self._record(ticket, request_id, "released", started, required)



    def _reclaim_idle(self, required, already_reclaimed):

        """仅回收显式授权的自有资源；当前请求、活动租约及固定依赖均排除。"""

        protected=set(required)|self.governor._active_protected_ids()

        protected.update(dict(self.governor.expand_requirements(

            (r,self.governor.catalog.get(r).pinned_min_level) for r in self.governor.pinned_ids)))

        candidates=[]

        for state in self.governor.states():

            resource=state.capability_id

            if resource not in self.host_pressure_reclaim_ids or resource in protected or resource in already_reclaimed or state.active_count:continue

            cap=self.governor.catalog.get(resource);target=cap.pinned_min_level

            released=cap.adapter.declared_cost(state.current_level)-cap.adapter.declared_cost(target)

            if state.current_level>target and released.host_ram_mb>0:

                candidates.append((state.last_used_tick,resource,target))

        if not candidates:return None

        _,resource,target=min(candidates)

        self.governor.demote(resource,target)

        return resource



    async def aclose(self):

        """停止新请求、唤醒等待者，待已准入调用归还后关闭底层资源。"""

        self._check_loop()

        if asyncio.current_task() in self._active.values():

            raise RuntimeError("cannot drain runtime from inside its own lease")

        self._closed = True

        self._changed.set()

        while self._active or self._queue:

            self._changed.clear()

            await self._changed.wait()

        self.governor.close()



    def close(self):

        """仅空闲时同步关闭；存在请求时要求调用者等待 aclose。"""

        if self._active or self._queue:

            raise RuntimeError("runtime has pending work; await aclose()")

        self._closed = True

        self.governor.close()

