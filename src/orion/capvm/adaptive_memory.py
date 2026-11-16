"""共享运行时的自动主存预算；保留原建立峰值保护，不承诺硬隔离。"""

from collections import deque

from dataclasses import replace,asdict

import math

from .host_memory import windows_host_memory,HostMemoryUnavailable



class AdaptiveMemoryBudget:

    """用当前驻留声明 + 新增可用余量计算总预算，避免重复扣除驻留量。"""

    def __init__(self, *, sampler=windows_host_memory, physical_floor_mb=2048,

                 commit_floor_mb=4096, user_limit_mb=None, reserve_policy=None):

        for value in (physical_floor_mb,commit_floor_mb):

            if not math.isfinite(value) or value<0:raise ValueError('invalid reserve')

        if user_limit_mb is not None and (not math.isfinite(user_limit_mb) or user_limit_mb<0):raise ValueError('invalid user cap')

        self.sampler=sampler;self.physical_floor_mb=physical_floor_mb;self.commit_floor_mb=commit_floor_mb;self.user_limit_mb=user_limit_mb

        self.reserve_policy=reserve_policy

        self.events=deque(maxlen=128)



    def refresh(self, governor):

        try:

            sample=self.sampler()

            if any(not math.isfinite(v) or v<0 for v in asdict(sample).values()):raise ValueError('invalid sample')

        except Exception as error:

            raise HostMemoryUnavailable('automatic memory budget telemetry unavailable') from error

        # 两个维度均要满足；当前实现只把较紧余量投影到统一 RAM 声明预算。

        floors=(self.physical_floor_mb,self.commit_floor_mb)

        if self.reserve_policy is not None:floors=self.reserve_policy.floors(sample)

        extra=min(sample.physical_available_mb-floors[0],sample.commit_available_mb-floors[1])

        resident=governor.total_declared_cost().host_ram_mb

        budget=max(0,resident+extra)

        if self.user_limit_mb is not None:budget=min(budget,self.user_limit_mb)

        governor.budget=replace(governor.budget,host_ram_mb=budget)

        # 依赖预算归一化的成本策略必须同步，避免继续使用旧预算。

        if hasattr(governor.policy,'budget'):governor.policy.budget=governor.budget

        self.events.append({'available':asdict(sample),'floors_mb':floors,'resident_declared_mb':resident,'extra_headroom_mb':extra,'budget_mb':budget})

        return governor.budget

