"""容量相关保底与观测波动余量；策略初值不冒充测得的最低需求。"""
from collections import deque
from dataclasses import dataclass,asdict
import math
from time import monotonic


@dataclass(frozen=True)
class HostMemoryCapacity:
    physical_total_mb: float
    commit_limit_mb: float


class AdaptiveReserve:
    """即时增大、缓慢收缩；已知建立动作的前后差值不重复算成外部波动。"""
    def __init__(self, capacity, *, fraction=.05, minimum_mb=512, half_life_s=60, clock=monotonic):
        if any(not math.isfinite(v) or v<=0 for v in asdict(capacity).values()):raise ValueError('invalid capacity')
        if not math.isfinite(fraction) or not 0<fraction<1:raise ValueError('invalid risk fraction')
        if not math.isfinite(minimum_mb) or minimum_mb<0 or not math.isfinite(half_life_s) or half_life_s<=0:raise ValueError('invalid reserve policy')
        self.base=(max(minimum_mb,capacity.physical_total_mb*fraction),max(minimum_mb,capacity.commit_limit_mb*fraction))
        self.half_life_s=half_life_s;self.clock=clock;self.events=deque(maxlen=256)
        self._previous=None;self._at=None;self._burst=[0.,0.]

    def floors(self, sample, *, learn=True):
        current=(sample.physical_available_mb,sample.commit_available_mb)
        if any(not math.isfinite(v) or v<0 for v in current):raise ValueError('invalid memory sample')
        now=self.clock();dt=max(0.,now-self._at) if self._at is not None else 0.
        decay=2**(-dt/self.half_life_s)
        for i,value in enumerate(current):
            drop=max(0.,self._previous[i]-value) if learn and self._previous is not None else 0.
            self._burst[i]=max(drop,self._burst[i]*decay)
        self._previous=current;self._at=now
        floors=tuple(self.base[i]+self._burst[i] for i in range(2))
        self.events.append({'base_mb':self.base,'observed_drop_envelope_mb':tuple(self._burst),
            'floors_mb':floors,'learning':learn,'basis':'policy_base_plus_decaying_observed_drop'})
        return floors
