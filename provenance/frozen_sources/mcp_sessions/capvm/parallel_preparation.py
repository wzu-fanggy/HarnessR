"""在完整容量检查后并行建立独立根服务资源；实验模式，默认串行。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .adapters.root_service_resource import RootServiceResourceAdapter
from .governor import CapabilityResidencyGovernor
from .model import ResidencyLevel


class ParallelPreparationGovernor(CapabilityResidencyGovernor):
    """由单一控制线程持有；工作线程只建立不同的远程资源。

    整批需求和正在使用的资源仍由原 governor 保护。进程、插件和未知适配器
    沿用串行准备，不能据此推断任意适配器可以在线程之间共享。
    """

    def __init__(self, *, prepare_workers=1, **kwargs):
        if isinstance(prepare_workers, bool) or not isinstance(prepare_workers, int) or not 1 <= prepare_workers <= 8:
            raise ValueError("prepare_workers must be an integer in [1, 8]")
        self.prepare_workers = prepare_workers
        super().__init__(**kwargs)

    def ensure_many(self, requests, *, request_id=""):
        requests = tuple(requests)
        if self.prepare_workers == 1:
            return super().ensure_many(requests, request_id=request_id)
        self._assert_open()
        merged = {}
        for item, level in requests:
            if level == ResidencyLevel.ACTIVE:
                raise ValueError("ACTIVE is not a materialization target")
            merged[item] = max(level, merged.get(item, ResidencyLevel.CATALOG))
        if not merged:
            return ()
        required = self._expanded_requirements(merged.items())
        protected = set(required) | self._active_protected_ids()
        begin = getattr(self.policy, "begin_request", None)
        if begin is not None:
            begin(catalog=self.catalog, states=self._states, required=required)
        plan = self._plan_complete_request(required, protected)
        before = {item: self._states[item].current_level for item in required}
        try:
            for victim, target in plan:
                self._demote(victim, target, detail="budget_eviction")
            for item in sorted(merged):
                state = self._states[item]
                if state.current_level < merged[item]:
                    refault = state.highest_level_ever >= merged[item]
                    if refault:
                        state.refaults += 1
                    else:
                        state.compulsory_faults += 1
                    self._record(event_type="refault" if refault else "compulsory_fault",
                                 capability_id=item, from_level=state.current_level,
                                 to_level=merged[item], detail=request_id)
            pending = {item for item in required if self._states[item].current_level < required[item]}
            while pending:
                ready = sorted(item for item in pending if all(
                    self._states[dep.capability_id].current_level >= dep.required_level
                    for dep in self.catalog.get(item).dependencies))
                if not ready:
                    raise RuntimeError("No prepared dependency frontier")
                remote = [item for item in ready if isinstance(
                    self.catalog.get(item).adapter, RootServiceResourceAdapter)
                    and required[item] >= ResidencyLevel.BOUND]
                if len(remote) >= 2:
                    self._prepare_remote_group(remote, required)
                    pending.difference_update(remote)
                for item in ready:
                    if item in pending:
                        self._ensure_with_dependencies(item, required[item],
                                                       protected_ids=protected, stack=())
                        pending.remove(item)
            result = []
            for item in sorted(merged):
                self._tick += 1
                self._mark_dependency_closure_used(item, self._tick)
                result.append(self.catalog.get(item))
            return tuple(result)
        except BaseException as exc:
            self._cleanup_new_preparation(before, exc)
            raise

    def _prepare_remote_group(self, items, required):
        old = {item: self._states[item].current_level for item in items}

        def establish(item):
            try:
                self.catalog.get(item).adapter.promote(old[item], required[item])
                return None
            except BaseException as exc:
                return exc

        futures = {}
        interrupted = None
        try:
            with ThreadPoolExecutor(max_workers=self.prepare_workers) as pool:
                futures = {item: pool.submit(establish, item) for item in items}
                # 不因首个失败提前退出。executor 退出前等待所有物理建立结束。
                for future in futures.values():
                    future.result()
        except BaseException as exc:
            interrupted = exc
        errors = [interrupted] if interrupted is not None else []
        for item in items:
            if item not in futures:
                continue
            error = futures[item].result()
            if error is not None:
                errors.append(error)
                try:
                    self.catalog.get(item).adapter.demote(required[item], old[item])
                except BaseException as cleanup:
                    error.add_note(f"adapter cleanup failed for {item}: {cleanup!r}")
                continue
            # 物理建立完成后，由控制线程提交状态；不并发修改记账、事件和索引。
            state = self._states[item]
            state.current_level = required[item]
            self._resident_ids.add(item)
            if not state.active_count and item not in self._idle_since_s:
                self._idle_since_s[item] = self._clock()
            state.promotions += 1
            state.ever_resident = True
            state.highest_level_ever = max(state.highest_level_ever, required[item])
            self._record(event_type="promote", capability_id=item,
                         from_level=old[item], to_level=required[item])
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f"parallel preparation also failed: {error!r}")
            raise errors[0]
