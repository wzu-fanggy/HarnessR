"""V0 冻结的 deterministic multi-resource LRU-cost eviction policy。"""

from __future__ import annotations

from collections.abc import Mapping

from ..catalog import CapabilityCatalog
from ..model import CapabilityRuntimeState, ResidencyLevel


class LRUCostEvictionPolicy:
    """优先淘汰最久未使用且能释放当前瓶颈资源的 capability。

    该策略只用于验证 CVM abstraction，不声称 replacement-policy novelty。
    """

    def choose(
        self,
        *,
        states: Mapping[str, CapabilityRuntimeState],
        catalog: CapabilityCatalog,
        violated_dimensions: tuple[str, ...],
        protected_ids: set[str],
        cost_basis: str = "declared",
    ) -> tuple[str, ResidencyLevel] | None:
        if cost_basis not in {"declared", "measured"}:
            raise ValueError("cost_basis must be 'declared' or 'measured'")
        candidates: list[tuple[int, str, ResidencyLevel]] = []
        for capability_id, state in states.items():
            if capability_id in protected_ids or state.active_count > 0:
                continue
            capability = catalog.get(capability_id)
            target = capability.pinned_min_level
            if state.current_level <= target:
                continue
            cost_fn = (
                capability.adapter.measured_cost
                if cost_basis == "measured"
                else capability.adapter.declared_cost
            )
            current_cost = cost_fn(state.current_level)
            target_cost = cost_fn(target)
            released = current_cost - target_cost
            relieves_violation = any(
                released.value(dimension) > 0
                for dimension in violated_dimensions
            )
            if not relieves_violation:
                continue
            # 不跨资源维度比较释放量：先按最近使用时间，再按稳定 ID。
            candidates.append((state.last_used_tick, capability_id, target))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, capability_id, target = candidates[0]
        return capability_id, target


class TypeLocalLRUEvictionPolicy(LRUCostEvictionPolicy):
    """使用同一全局容量检查，仅允许回收本次缺失依赖所属类型的对象。"""

    def __init__(self) -> None:
        self._request_kinds = frozenset()

    def begin_request(self, *, catalog, states, required) -> None:
        self._request_kinds = frozenset(
            catalog.get(item).kind
            for item, target in required.items()
            if states[item].current_level < target
        )

    def choose(self, *, states, catalog, violated_dimensions, protected_ids,
               cost_basis="declared"):
        eligible = {
            item: state for item, state in states.items()
            if catalog.get(item).kind in self._request_kinds
        }
        return super().choose(
            states=eligible, catalog=catalog,
            violated_dimensions=violated_dimensions,
            protected_ids=protected_ids, cost_basis=cost_basis,
        )
