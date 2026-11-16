"""以已经观测的建立耗时和当前容量限制选择空闲回收对象。

成本感知换出已有研究先例。本模块是实验策略，不改变默认 LRU；不预测未来
调用，也不把不同资源维度的原始数值直接相加。unknown 成本回退到原 LRU。
"""
from __future__ import annotations

import math

from .lru_cost import LRUCostEvictionPolicy
from ..model import ResidencyLevel, ResourceVector


class ObservedCostEvictionPolicy(LRUCostEvictionPolicy):
    """使用过去观测；成本须由适配器已完成的建立操作提供。

mode=cost_first 为普通低成本优先基线；mode=bottleneck_ratio 按当前超限项
的归一化释放量与建立耗时之比排序。它不是 mCache 或 RAGCache 的复现。
建立耗时只代表适配器记录的边界，未必包括按需加载和工具执行。
"""

    def __init__(self, *, budget: ResourceVector, mode="bottleneck_ratio"):
        if mode not in {"cost_first", "bottleneck_ratio"}:
            raise ValueError("Unknown cost policy mode")
        self.budget = budget
        self.mode = mode
        self.observed_seconds: dict[str, float] = {}

    def record_prepare(self, resource_id: str, seconds: float) -> None:
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Observed preparation time must be finite and positive")
        self.observed_seconds[resource_id] = float(seconds)

    def observe_tool_batch(self, tool_names, resource_ids, *, catalog):
        """原 bridge 在租约结束前调用此钩子；只读已完成资源的成本。"""
        pending, seen = list(resource_ids), set()
        while pending:
            resource_id = pending.pop()
            if resource_id in seen:
                continue
            seen.add(resource_id)
            resource = catalog.get(resource_id)
            pending.extend(dep.capability_id for dep in resource.dependencies)
            for attribute in ("last_startup_latency_s", "last_load_latency_s"):
                seconds = getattr(resource.adapter, attribute, None)
                if isinstance(seconds, (float, int)) and not isinstance(seconds, bool):
                    if math.isfinite(seconds) and seconds > 0:
                        self.record_prepare(resource_id, seconds)
                        break

    def select(self, candidates, violated_dimensions):
        """仅接收已筛选的合法候选，不接收未来复用或实验评分字段。

候选为 (resource_id, target_level, last_used_tick, released_vector)。
"""
        candidates = list(candidates)
        if not candidates:
            return None
        fallback = min(candidates, key=lambda row: (row[2], row[0]))
        if any(row[0] not in self.observed_seconds for row in candidates):
            return fallback[:2]
        if self.mode == "bottleneck_ratio" and any(
            not math.isfinite(self.budget.value(d)) or self.budget.value(d) <= 0
            for d in violated_dimensions
        ):
            return fallback[:2]

        def priority(row):
            resource_id, _, tick, released = row
            benefit = 1.0
            if self.mode == "bottleneck_ratio":
                benefit = sum(max(0.0, released.value(d)) / self.budget.value(d)
                              for d in violated_dimensions)
                if benefit <= 0:
                    return (math.inf, tick, resource_id)
            return (self.observed_seconds[resource_id] / benefit, tick, resource_id)

        selected = min(candidates, key=priority)
        return selected[:2]

    def choose(self, *, states, catalog, violated_dimensions, protected_ids,
               cost_basis="declared"):
        if cost_basis not in {"declared", "measured"}:
            raise ValueError("Unknown resource cost basis")
        candidates = []
        for resource_id, state in states.items():
            resource = catalog.get(resource_id)
            target = resource.pinned_min_level
            if resource_id in protected_ids or state.active_count or state.current_level <= target:
                continue
            cost_fn = (resource.adapter.measured_cost if cost_basis == "measured"
                       else resource.adapter.declared_cost)
            released = cost_fn(state.current_level) - cost_fn(target)
            if any(released.value(d) > 0 for d in violated_dimensions):
                # 已记录的完整建立时间不能充当部分层级调整的耗时。
                if not (state.current_level >= ResidencyLevel.BOUND > target):
                    return super().choose(states=states, catalog=catalog,
                        violated_dimensions=violated_dimensions,
                        protected_ids=protected_ids, cost_basis=cost_basis)
                candidates.append((resource_id, target, state.last_used_tick, released))
        return self.select(candidates, violated_dimensions)
