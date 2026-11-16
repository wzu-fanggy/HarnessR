"""Capability Virtual Memory 的统一 residency governor。"""



from __future__ import annotations



from collections.abc import Callable, Iterable, Iterator

from contextlib import contextmanager

from dataclasses import replace

from time import monotonic, perf_counter



from .adapters.base import estimate_prompt_tokens

from .catalog import CapabilityCatalog

from .measurement_batch import measurement_batch

from .model import (

    CapabilityFault,

    CapabilityRuntimeState,

    MeasuredResidencyBudgetExceeded,

    ResidencyBudgetExceeded,

    ResidencyEvent,

    ResidencyLevel,

    ResourceVector,

    VirtualCapability,

)

from .policies.lru_cost import LRUCostEvictionPolicy





class CapabilityResidencyGovernor:

    """在逻辑 capability 空间上服务 fault，并在 hard budgets 下管理 working set。"""



    def __init__(

        self,

        *,

        catalog: CapabilityCatalog,

        budget: ResourceVector,

        policy: LRUCostEvictionPolicy | None = None,

        clock: Callable[[], float] = monotonic,

        preparation_guard=None,

        plan_advisor=None,

        lifecycle_profiler=None,

        profile_clock=perf_counter,

    ) -> None:

        self.lifecycle_profiler = lifecycle_profiler

        self._profile_clock = profile_clock

        self.plan_advisor = plan_advisor

        self.preparation_guard = preparation_guard

        self.catalog = catalog

        self.budget = budget

        self.policy = policy or LRUCostEvictionPolicy()

        self._clock = clock

        self._states: dict[str, CapabilityRuntimeState] = {

            capability.capability_id: CapabilityRuntimeState(

                capability_id=capability.capability_id

            )

            for capability in catalog.all()

        }

        self._events: list[ResidencyEvent] = []

        # Physical accounting must scale with the resident working set, not with the

        # logical capability address space.  Keep an explicit resident index so hot-path

        # cost/telemetry aggregation never scans thousands of CATALOG-only entries.

        self._resident_ids: set[str] = set()

        # Active protection is another working-set property. Keep an explicit index so

        # resident hits do not scan every CATALOG-only state just to discover that most

        # capabilities are inactive.

        self._active_ids: set[str] = set()

        # Monotonic lifecycle telemetry is deliberately separate from deterministic

        # benchmark ticks/events so existing replay hashes stay unchanged.

        self._idle_since_s: dict[str, float] = {}

        self._last_idle_gap_s: dict[str, float] = {}

        self._reuse_after_idle_count: dict[str, int] = {}

        self._last_acquire_requested_s: dict[str, float] = {}

        self._last_active_started_s: dict[str, float] = {}

        self._last_active_ended_s: dict[str, float] = {}

        self._tick = 0

        self._closed = False

        self._pinned_ids = {

            item.capability_id for item in catalog.all()

            if item.pinned_min_level > ResidencyLevel.CATALOG

        }

        self._initialize_pins()



    @property

    def events(self) -> tuple[ResidencyEvent, ...]:

        return tuple(self._events)



    @property

    def pinned_ids(self) -> tuple[str, ...]:

        """返回固定驻留资源标识，不扫描完整目录。"""

        return tuple(sorted(self._pinned_ids))



    def expand_requirements(self, requests) -> tuple[tuple[str, ResidencyLevel], ...]:

        """只读解析完整依赖，供准入层和框架适配层共享同一资源定义。"""

        self._assert_open()

        return tuple(self._expanded_requirements(requests).items())



    def state(self, capability_id: str) -> CapabilityRuntimeState:

        self._assert_open()

        self.catalog.get(capability_id)

        return self._states[capability_id]



    def states(self) -> tuple[CapabilityRuntimeState, ...]:

        self._assert_open()

        return tuple(self._states[item] for item in sorted(self._states))



    def resource_observation(self, capability_id: str) -> dict[str, object]:

        """Return a read-only lifecycle snapshot for one managed resource."""



        self._assert_open()

        capability = self.catalog.get(capability_id)

        state = self._states[capability_id]

        resident = state.current_level > ResidencyLevel.CATALOG

        idle_since = self._idle_since_s.get(capability_id)

        idle_seconds = None

        if resident and state.active_count == 0 and idle_since is not None:

            idle_seconds = max(0.0, self._clock() - idle_since)

        declared = capability.adapter.declared_cost(state.current_level)

        measurement_error: str | None = None

        try:

            measured = capability.adapter.measured_cost(state.current_level)

        except Exception as exc:

            measured = None

            measurement_error = type(exc).__name__

        cost_basis = dict(getattr(capability.adapter, "last_cost_basis", {}))

        accounted = measured

        if "declared_fallback" in cost_basis.values():

            measured = None

        adapter_pid = getattr(capability.adapter, "pid", None)

        prepare_seconds = getattr(capability.adapter, "last_startup_latency_s", None)

        return {

            "resource_id": capability_id,

            "resource_type": capability.kind.value,

            "current_level": state.current_level.name,

            "resident": resident,

            "active_count": state.active_count,

            "idle_since_monotonic_s": idle_since,

            "idle_seconds": round(idle_seconds, 6) if idle_seconds is not None else None,

            "last_idle_gap_seconds": self._last_idle_gap_s.get(capability_id),

            "reuse_after_idle_count": self._reuse_after_idle_count.get(capability_id, 0),

            "last_acquire_requested_monotonic_s": self._last_acquire_requested_s.get(capability_id),

            "last_active_started_monotonic_s": self._last_active_started_s.get(capability_id),

            "last_active_ended_monotonic_s": self._last_active_ended_s.get(capability_id),

            "promotions": state.promotions,

            "demotions": state.demotions,

            "declared_cost": {

                "host_ram_mb": declared.host_ram_mb,

                "gpu_vram_mb": declared.gpu_vram_mb,

                "process_slots": declared.process_slots,

                "connection_slots": declared.connection_slots,

            },

            "measured_cost": ({

                "host_ram_mb": measured.host_ram_mb,

                "gpu_vram_mb": measured.gpu_vram_mb,

                "process_slots": measured.process_slots,

                "connection_slots": measured.connection_slots,

            } if measured is not None else None),

            "accounted_cost": ({

                "host_ram_mb": accounted.host_ram_mb,

                "gpu_vram_mb": accounted.gpu_vram_mb,

                "process_slots": accounted.process_slots,

                "connection_slots": accounted.connection_slots,

            } if accounted is not None else None),

            "cost_basis": cost_basis,

            "measurement_error": measurement_error,

            "instance_pid": int(adapter_pid) if adapter_pid is not None else None,

            "last_prepare_seconds": (

                float(prepare_seconds) if prepare_seconds is not None else None

            ),

        }



    def resource_observations(self) -> tuple[dict[str, object], ...]:

        """Return lifecycle snapshots for all catalog entries in deterministic order."""



        with measurement_batch():

            return tuple(self.resource_observation(item) for item in self.catalog.ids())



    def resident_model_context_blocks(self) -> tuple[tuple[str, str], ...]:

        """返回当前 resident capabilities 真正可暴露给模型的文本块。



        capability adapter 可选实现 ``model_context_blocks(level)``。未实现该扩展或没有

        模型侧文本的 Process/Session 等对象不会产生 block。返回值保持 capability ID，

        便于上层 prompt/tool-schema assembler 做 provider-specific formatting。

        """



        self._assert_open()

        blocks: list[tuple[str, str]] = []

        for capability_id in sorted(self._resident_ids):

            state = self._states[capability_id]

            capability = self.catalog.get(capability_id)

            provider = getattr(capability.adapter, "model_context_blocks", None)

            if provider is None:

                continue

            for block in provider(state.current_level):

                text = str(block).strip()

                if text:

                    blocks.append((capability_id, text))

        return tuple(blocks)



    def partition_model_context(

        self, required_ids: Iterable[str],

    ) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:

        """区分本轮必需信息和闲置保留信息；可选信息按最久未使用优先排列。"""

        required: set[str] = set()

        for item in required_ids:

            required.update(self._dependency_closure(item))

        blocks = self.resident_model_context_blocks()

        mandatory = tuple(block for block in blocks if block[0] in required)

        optional = tuple(sorted(

            (block for block in blocks if block[0] not in required),

            key=lambda block: (self._states[block[0]].last_used_tick, block[0]),

        ))

        return mandatory, optional



    def resident_model_context_estimated_tokens(self) -> int:

        """按与 Tool/Skill/Plugin adapter 相同的估计器统计当前 model-context blocks。"""



        return sum(

            estimate_prompt_tokens(text)

            for _capability_id, text in self.resident_model_context_blocks()

        )



    def total_declared_cost(self) -> ResourceVector:

        self._assert_open()

        total = ResourceVector()

        for capability_id in self._resident_ids:

            state = self._states[capability_id]

            capability = self.catalog.get(capability_id)

            total = total + capability.adapter.declared_cost(state.current_level)

        return total



    def total_measured_cost(self) -> ResourceVector:

        self._assert_open()

        total = ResourceVector()

        with measurement_batch():

            for capability_id in self._resident_ids:

                state = self._states[capability_id]

                capability = self.catalog.get(capability_id)

                total = total + capability.adapter.measured_cost(state.current_level)

        return total



    def measured_budget_violations(self) -> tuple[str, ...]:

        """返回建立后记账相对预算的超限项；适配器可标注预计值回退。"""



        self._assert_open()

        return self.total_measured_cost().violated_dimensions(self.budget)



    def reconcile_measured_budget(

        self,

        *,

        protected_ids: Iterable[str] = (),

    ) -> ResourceVector:

        """page-in 后按占用观测二次检查 working set。



        远程观测未知时，RootServiceResourceAdapter 使用登记预计值并标注来源；

        预计值不能保证工具资源的物理占用低于预算。



        declared-cost admission 只能在 materialization 前预测资源。真实 process/session/plugin

        可能存在解释器、连接或运行时开销，因此完成 page-in 后需要再次检查 measured cost。

        本方法只驱逐非 active、非显式 protected 的 capability；若活动 working set 本身已经

        超预算，则显式抛出 ``MeasuredResidencyBudgetExceeded``，不能假装 admission 成功。

        """



        self._assert_open()

        explicit_protected = set(protected_ids)

        while True:

            measured = self.total_measured_cost()

            violations = measured.violated_dimensions(self.budget)

            if not violations:

                return measured

            protected = explicit_protected | self._active_protected_ids()

            eviction = self.policy.choose(

                states={item: self._states[item] for item in self._resident_ids},

                catalog=self.catalog,

                violated_dimensions=violations,

                protected_ids=protected,

                cost_basis="measured",

            )

            if eviction is None:

                self._record(

                    event_type="measured_budget_overrun",

                    capability_id="__runtime__",

                    from_level=ResidencyLevel.CATALOG,

                    to_level=ResidencyLevel.CATALOG,

                    detail=",".join(violations),

                )

                raise MeasuredResidencyBudgetExceeded(

                    measured_total=measured,

                    budget=self.budget,

                    violated_dimensions=violations,

                )

            victim_id, victim_target = eviction

            self._demote(

                victim_id,

                victim_target,

                detail="measured_budget_eviction",

            )



    def access(

        self,

        capability_id: str,

        required_level: ResidencyLevel,

        *,

        request_id: str = "",

    ) -> VirtualCapability:

        """访问逻辑 capability；若 nonresident，则自动服务 CapabilityFault。"""



        self._assert_open()

        if required_level == ResidencyLevel.ACTIVE:

            raise ValueError("ACTIVE residency must be entered through acquire()")

        return self.ensure_many(

            ((capability_id, required_level),), request_id=request_id

        )[0]



    def handle_fault(self, fault: CapabilityFault) -> None:

        """把 nonresident fault 解析为预算内的 materialization。"""



        self._assert_open()

        self.ensure_many(

            ((fault.capability_id, fault.required_level),),

            request_id=fault.request_id or fault.reason,

        )



    def ensure_many(

        self,

        requests: Iterable[tuple[str, ResidencyLevel]],

        *,

        request_id: str = "",

    ) -> tuple[VirtualCapability, ...]:

        """先核算完整依赖，再准备本次调用。



        在修改任何驻留状态前检查完整需求，并在临时状态上规划可行的回收集合。

        准备异常时清理本次新建或提升的对象；已释放的旧缓存不承诺重新加载。

        """



        self._assert_open()

        merged: dict[str, ResidencyLevel] = {}

        for capability_id, required_level in requests:

            if required_level == ResidencyLevel.ACTIVE:

                raise ValueError("ACTIVE is not a materialization target")

            previous = merged.get(capability_id, ResidencyLevel.CATALOG)

            merged[capability_id] = max(previous, required_level)

        if not merged:

            return ()



        required = self._expanded_requirements(merged.items())

        protected = set(required) | self._active_protected_ids()

        # 类型局部策略也使用同一全局预检，只限制回收候选的类型。

        begin_request = getattr(self.policy, "begin_request", None)

        if begin_request is not None:

            begin_request(catalog=self.catalog, states=self._states, required=required)

        plan = self._plan_complete_request(required, protected)

        before = {item: self._states[item].current_level for item in required}

        results: list[VirtualCapability] = []

        try:

            for victim_id, target in plan:

                self._demote(victim_id, target, detail="budget_eviction")

            for capability_id in sorted(merged):

                required_level = merged[capability_id]

                capability = self.catalog.get(capability_id)

                state = self._states[capability_id]

                if state.current_level < required_level:

                    if state.highest_level_ever >= required_level:

                        state.refaults += 1

                        event_type = "refault"

                    else:

                        state.compulsory_faults += 1

                        event_type = "compulsory_fault"

                    self._record(

                        event_type=event_type, capability_id=capability_id,

                        from_level=state.current_level, to_level=required_level,

                        detail=request_id,

                    )

                self._ensure_with_dependencies(

                    capability_id, required_level,

                    protected_ids=protected, stack=(),

                )

                self._tick += 1

                self._mark_dependency_closure_used(capability_id, self._tick)

                results.append(capability)

            return tuple(results)

        except BaseException as exc:

            self._cleanup_new_preparation(before, exc)

            raise



    def _expanded_requirements(

        self, requests: Iterable[tuple[str, ResidencyLevel]],

    ) -> dict[str, ResidencyLevel]:

        """去重共享依赖，并在准备前检查所有标识、层级与依赖环。"""

        required: dict[str, ResidencyLevel] = {}

        visiting: set[str] = set()



        def visit(item: str, level: ResidencyLevel) -> None:

            cap = self.catalog.get(item)

            if level == ResidencyLevel.ACTIVE or level > cap.maximum_level:

                raise ValueError(f"{item} cannot reach residency {level.name}")

            if item in visiting:

                raise ValueError(f"capability dependency cycle involving {item}")

            if item in required and required[item] >= level:

                return

            visiting.add(item)

            for dep in cap.dependencies:

                visit(dep.capability_id, dep.required_level)

            visiting.remove(item)

            required[item] = max(level, required.get(item, ResidencyLevel.CATALOG))



        for item, level in requests:

            visit(item, level)

        return required



    def _plan_complete_request(

        self, required: dict[str, ResidencyLevel], protected: set[str],

    ) -> list[tuple[str, ResidencyLevel]]:

        """规划完整请求的回收动作；失败时不改变已有对象。"""

        mandatory = dict(required)

        for item in self._active_ids:

            mandatory[item] = max(mandatory.get(item, ResidencyLevel.CATALOG),

                                  self._states[item].current_level)

        pins = self._expanded_requirements(

            (item, self.catalog.get(item).pinned_min_level)

            for item in sorted(self._pinned_ids)

        )

        for item, level in pins.items():

            mandatory[item] = max(level, mandatory.get(item, ResidencyLevel.CATALOG))

        minimum = ResourceVector()

        for item, level in mandatory.items():

            minimum += self.catalog.get(item).adapter.declared_cost(level)

        violations = minimum.violated_dimensions(self.budget)

        if violations:

            raise ResidencyBudgetExceeded("__request__", max(required.values()), violations)



        simulated = {item: replace(self._states[item]) for item in self._resident_ids}

        plan: list[tuple[str, ResidencyLevel]] = []

        while True:

            projected = ResourceVector()

            for item in set(simulated) | set(required):

                current = simulated[item].current_level if item in simulated else ResidencyLevel.CATALOG

                level = max(current, required.get(item, ResidencyLevel.CATALOG))

                projected += self.catalog.get(item).adapter.declared_cost(level)

            violations = projected.violated_dimensions(self.budget)

            if not violations:

                if self.plan_advisor is not None:

                    return self.plan_advisor.select(self, required, plan)

                return plan

            eviction = self.policy.choose(

                states=simulated, catalog=self.catalog,

                violated_dimensions=violations,

                protected_ids=protected | set(pins),

            )

            if eviction is None:

                raise ResidencyBudgetExceeded("__request__", max(required.values()), violations)

            item, target = eviction

            state = simulated[item]

            if target >= state.current_level or item in protected or state.active_count:

                raise RuntimeError("eviction policy returned an invalid candidate")

            plan.append((item, target))

            state.current_level = target

            if target == ResidencyLevel.CATALOG:

                del simulated[item]



    def _cleanup_new_preparation(

        self, before: dict[str, ResidencyLevel], error: BaseException,

    ) -> None:

        """按依赖的逆顺序清理此次新建对象，不重新加载被回收的旧缓存。"""

        for item, old_level in reversed(tuple(before.items())):

            state = self._states[item]

            if state.current_level > old_level and not state.active_count:

                try:

                    self._demote(item, old_level, detail="failed_preparation_cleanup")

                except Exception as cleanup_error:

                    error.add_note(f"cleanup failed for {item}: {cleanup_error!r}")



    @contextmanager

    def acquire_many(

        self,

        requests: Iterable[tuple[str, ResidencyLevel]],

        *,

        request_id: str = "",

        reconcile_before_active: bool = False,

    ) -> Iterator[tuple[VirtualCapability, ...]]:

        """建立并在 context 生命周期内 pin 住一个同时驻留的 working set。"""



        acquire_requested_s = self._clock()

        request_tuple = tuple(requests)

        required = self._expanded_requirements(request_tuple)

        prior_idle_since = {

            item: self._idle_since_s.get(item)

            for item in required

            if self._states[item].current_level > ResidencyLevel.CATALOG

            and self._states[item].active_count == 0

        }

        before = {item: self._states[item].current_level for item in required}

        capabilities = self.ensure_many(request_tuple, request_id=request_id)

        active_ids = set(required)

        if reconcile_before_active:

            try:

                self.reconcile_measured_budget(protected_ids=active_ids)

            except BaseException as exc:

                self._cleanup_new_preparation(before, exc)

                raise

        active_started_s = self._clock()

        for item in active_ids:

            state = self._states[item]

            if state.active_count == 0:

                previous_idle = prior_idle_since.get(item)

                if previous_idle is not None:

                    self._last_idle_gap_s[item] = round(

                        max(0.0, acquire_requested_s - previous_idle), 6

                    )

                    self._reuse_after_idle_count[item] = (

                        self._reuse_after_idle_count.get(item, 0) + 1

                    )

                self._idle_since_s.pop(item, None)

                self._last_acquire_requested_s[item] = acquire_requested_s

                self._last_active_started_s[item] = active_started_s

            state.active_count += 1

            self._active_ids.add(item)

        for capability_id, _required_level in request_tuple:

            self._record(

                event_type="active_enter",

                capability_id=capability_id,

                from_level=self._states[capability_id].current_level,

                to_level=ResidencyLevel.ACTIVE,

                detail=request_id,

            )

        try:

            yield capabilities

        finally:

            for item in active_ids:

                self._profile_sample(self.catalog.get(item),"execution_end")

            active_ended_s = self._clock()

            for item in active_ids:

                self._states[item].active_count -= 1

                if self._states[item].active_count == 0:

                    self._active_ids.discard(item)

                    self._last_active_ended_s[item] = active_ended_s

                    if self._states[item].current_level > ResidencyLevel.CATALOG:

                        self._idle_since_s[item] = active_ended_s

            for capability_id, _required_level in request_tuple:

                self._record(

                    event_type="active_exit",

                    capability_id=capability_id,

                    from_level=ResidencyLevel.ACTIVE,

                    to_level=self._states[capability_id].current_level,

                    detail=request_id,

                )



    @contextmanager

    def acquire(

        self,

        capability_id: str,

        required_level: ResidencyLevel = ResidencyLevel.BOUND,

        *,

        request_id: str = "",

    ) -> Iterator[VirtualCapability]:

        """在一次 in-flight use 期间 pin 住 capability 及其依赖，防止被 eviction。"""



        with self.acquire_many(

            ((capability_id, required_level),),

            request_id=request_id,

        ) as capabilities:

            yield capabilities[0]



    def demote(

        self,

        capability_id: str,

        target_level: ResidencyLevel = ResidencyLevel.CATALOG,

    ) -> None:

        """显式降驻留；不能越过 pin 或破坏 active dependency。"""



        self._assert_open()

        capability = self.catalog.get(capability_id)

        state = self._states[capability_id]

        if state.active_count > 0:

            raise RuntimeError(f"cannot demote active capability: {capability_id}")

        if target_level < capability.pinned_min_level:

            raise ValueError(

                f"target {target_level.name} is below pinned minimum "

                f"{capability.pinned_min_level.name}"

            )

        if target_level >= state.current_level:

            return

        self._demote(capability_id, target_level, detail="explicit")



    def close(self) -> None:

        if self._closed:

            return

        for capability_id in sorted(

            self._states,

            key=lambda item: self._states[item].current_level,

            reverse=True,

        ):

            capability = self.catalog.get(capability_id)

            state = self._states[capability_id]

            if state.active_count:

                raise RuntimeError(f"cannot close active capability: {capability_id}")

            try:

                if state.current_level > ResidencyLevel.CATALOG:

                    self._profile_sample(capability,"release_start")

                    capability.adapter.demote(

                        state.current_level,

                        ResidencyLevel.CATALOG,

                    )

                capability.adapter.close()

            finally:

                state.current_level = ResidencyLevel.CATALOG

                self._resident_ids.discard(capability_id)

        self._resident_ids.clear()

        self._active_ids.clear()

        self._idle_since_s.clear()

        self._closed = True



    def _initialize_pins(self) -> None:

        self.ensure_many(

            (item, self.catalog.get(item).pinned_min_level)

            for item in sorted(self._pinned_ids)

        )



    def _ensure_with_dependencies(

        self,

        capability_id: str,

        target_level: ResidencyLevel,

        *,

        protected_ids: set[str],

        stack: tuple[str, ...],

    ) -> None:

        if capability_id in stack:

            cycle = " -> ".join((*stack, capability_id))

            raise ValueError(f"capability dependency cycle: {cycle}")

        capability = self.catalog.get(capability_id)

        if target_level > capability.maximum_level:

            raise ValueError(

                f"{capability_id} cannot reach residency {target_level.name}"

            )

        next_stack = (*stack, capability_id)

        for dependency in capability.dependencies:

            self._ensure_with_dependencies(

                dependency.capability_id,

                dependency.required_level,

                protected_ids=protected_ids,

                stack=next_stack,

            )

        state = self._states[capability_id]

        if state.current_level >= target_level:

            return

        self._ensure_budget_for(

            capability_id,

            target_level,

            protected_ids=protected_ids,

        )

        self._promote(capability_id, target_level)



    def _ensure_budget_for(

        self,

        capability_id: str,

        target_level: ResidencyLevel,

        *,

        protected_ids: set[str],

    ) -> None:

        capability = self.catalog.get(capability_id)

        state = self._states[capability_id]

        while True:

            current_total = self.total_declared_cost()

            projected = (

                current_total

                - capability.adapter.declared_cost(state.current_level)

                + capability.adapter.declared_cost(target_level)

            )

            violations = projected.violated_dimensions(self.budget)

            if not violations:

                return

            protected = protected_ids | self._active_protected_ids() | {capability_id}

            eviction = self.policy.choose(

                states={item: self._states[item] for item in self._resident_ids},

                catalog=self.catalog,

                violated_dimensions=violations,

                protected_ids=protected,

            )

            if eviction is None:

                raise ResidencyBudgetExceeded(

                    capability_id,

                    target_level,

                    violations,

                )

            victim_id, victim_target = eviction

            self._demote(victim_id, victim_target, detail="budget_eviction")



    def _promote(self, capability_id: str, target_level: ResidencyLevel) -> None:

        capability = self.catalog.get(capability_id)

        state = self._states[capability_id]

        current = state.current_level

        # 在适配器产生副作用前检查整机余量；拒绝不调用适配器清理。

        if self.preparation_guard is not None:

            self.preparation_guard.check(capability, current, target_level)

        if self.lifecycle_profiler is not None and current < ResidencyLevel.BOUND <= target_level:

            begin=getattr(self.lifecycle_profiler,'begin',None)

            if begin is not None:begin(capability)

        started = self._profile_clock()

        try:

            capability.adapter.promote(current, target_level)

            if self.preparation_guard is not None:

                self.preparation_guard.check(capability, target_level, target_level, phase="prepared")

        except BaseException as exc:

            self._profile_transition(capability, current, target_level, started, False)

            # 适配器可能在创建部分资源后失败，状态尚未提交时也要清理。

            try:

                capability.adapter.demote(target_level, current)

            except Exception as cleanup_error:

                exc.add_note(f"adapter cleanup failed for {capability_id}: {cleanup_error!r}")

            raise

        self._profile_transition(capability, current, target_level, started, True)

        state.current_level = target_level

        if target_level > ResidencyLevel.CATALOG:

            self._resident_ids.add(capability_id)

            if state.active_count == 0 and capability_id not in self._idle_since_s:

                self._idle_since_s[capability_id] = self._clock()

        state.promotions += 1

        state.ever_resident = target_level > ResidencyLevel.CATALOG

        state.highest_level_ever = max(state.highest_level_ever, target_level)

        self._record(

            event_type="promote",

            capability_id=capability_id,

            from_level=current,

            to_level=target_level,

        )



    def _profile_sample(self, capability, phase):

        if self.lifecycle_profiler is not None:

            sample=getattr(self.lifecycle_profiler,'sample',None)

            if sample is not None:sample(capability,phase)



    def _profile_transition(self, capability, current, target, started, success):

        # 画像写入失败不能破坏工具语义，也不能吞掉原始建立异常。

        if self.lifecycle_profiler is not None and current < ResidencyLevel.BOUND <= target:

            self.lifecycle_profiler.record(capability, max(0, self._profile_clock()-started), success)



    def _demote(

        self,

        capability_id: str,

        target_level: ResidencyLevel,

        *,

        detail: str,

    ) -> None:

        capability = self.catalog.get(capability_id)

        state = self._states[capability_id]

        current = state.current_level

        if target_level >= current:

            return

        if state.active_count > 0:

            raise RuntimeError(f"cannot evict active capability: {capability_id}")

        self._profile_sample(capability,"release_start")

        capability.adapter.demote(current, target_level)

        state.current_level = target_level

        if target_level == ResidencyLevel.CATALOG:

            self._resident_ids.discard(capability_id)

            self._idle_since_s.pop(capability_id, None)

        else:

            self._resident_ids.add(capability_id)

            if state.active_count == 0 and capability_id not in self._idle_since_s:

                self._idle_since_s[capability_id] = self._clock()

        state.demotions += 1

        self._record(

            event_type="demote",

            capability_id=capability_id,

            from_level=current,

            to_level=target_level,

            detail=detail,

        )



    def _dependency_closure(self, capability_id: str) -> set[str]:

        closure: set[str] = set()

        visiting: set[str] = set()



        def visit(item_id: str) -> None:

            if item_id in visiting:

                raise ValueError(f"capability dependency cycle involving {item_id}")

            if item_id in closure:

                return

            visiting.add(item_id)

            capability = self.catalog.get(item_id)

            for dependency in capability.dependencies:

                visit(dependency.capability_id)

            visiting.remove(item_id)

            closure.add(item_id)



        visit(capability_id)

        return closure



    def _active_protected_ids(self) -> set[str]:

        # acquire_many() indexes the complete dependency closure in _active_ids, so the

        # protected set can be returned in O(active working set) rather than O(catalog).

        return set(self._active_ids)



    def _mark_dependency_closure_used(self, capability_id: str, tick: int) -> None:

        """Refresh recency for the capability and every dependency used by the call."""

        for item_id in self._dependency_closure(capability_id):

            self._states[item_id].last_used_tick = tick



    def _record(

        self,

        *,

        event_type: str,

        capability_id: str,

        from_level: ResidencyLevel,

        to_level: ResidencyLevel,

        detail: str = "",

    ) -> None:

        self._tick += 1

        self._events.append(

            ResidencyEvent(

                tick=self._tick,

                event_type=event_type,

                capability_id=capability_id,

                from_level=from_level,

                to_level=to_level,

                detail=detail,

            )

        )



    def _assert_open(self) -> None:

        if self._closed:

            raise RuntimeError("capability residency governor is closed")

