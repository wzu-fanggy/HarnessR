"""整机建立余量检查；这是准入策略，不是操作系统内存隔离。"""

from collections import deque

from dataclasses import asdict, dataclass

import math

import os

from time import time

from .model import ResidencyLevel





@dataclass(frozen=True)

class HostMemorySnapshot:

    physical_available_mb: float

    commit_available_mb: float





@dataclass(frozen=True)

class PreparationReservation:

    """跨入 BOUND 时额外需要的系统余量；独立于稳态驻留成本。"""

    physical_mb: float

    commit_mb: float



    def __post_init__(self):

        if any(not math.isfinite(v) or v < 0 for v in (self.physical_mb, self.commit_mb)):

            raise ValueError('preparation reservation must be finite and non-negative')





class HostMemoryUnavailable(RuntimeError):

    """整机余量不足或无法测量；调用方须显式处理拒绝。"""





def windows_host_memory(*, include_capacity=False):

    """读取全系统物理余量和 CommitLimit-CommitTotal，单位 MiB。"""

    if os.name != 'nt':

        raise OSError('host memory sampler currently requires Windows')

    import ctypes

    from ctypes import wintypes



    class PerformanceInformation(ctypes.Structure):

        _fields_ = [('cb', wintypes.DWORD)] + [

            (name, ctypes.c_size_t) for name in (

                'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal',

                'PhysicalAvailable', 'SystemCache', 'KernelTotal', 'KernelPaged',

                'KernelNonpaged', 'PageSize')] + [

            (name, wintypes.DWORD) for name in ('HandleCount', 'ProcessCount', 'ThreadCount')]



    dll = ctypes.WinDLL('psapi', use_last_error=True)

    dll.GetPerformanceInfo.argtypes = [ctypes.POINTER(PerformanceInformation), wintypes.DWORD]

    dll.GetPerformanceInfo.restype = wintypes.BOOL

    info = PerformanceInformation()

    info.cb = ctypes.sizeof(info)

    if not dll.GetPerformanceInfo(ctypes.byref(info), info.cb):

        raise ctypes.WinError(ctypes.get_last_error())

    factor = info.PageSize / (1024 * 1024)

    sample=HostMemorySnapshot(info.PhysicalAvailable * factor,

        max(0, info.CommitLimit - info.CommitTotal) * factor)

    if include_capacity:

        from .adaptive_reserve import HostMemoryCapacity

        return sample,HostMemoryCapacity(info.PhysicalTotal*factor,info.CommitLimit*factor)

    return sample





class HostMemoryGuard:

    """逐次建立检查，保留有界审计；预留倍数不是实测峰值。"""

    def __init__(self, *, physical_floor_mb=2048, commit_floor_mb=4096,

                 preparation_multiplier=2, sampler=None, reservations=None, reserve_policy=None):

        for value in (physical_floor_mb, commit_floor_mb, preparation_multiplier):

            if not math.isfinite(value) or value < 0:

                raise ValueError('headroom configuration must be finite and non-negative')

        self.physical_floor_mb = physical_floor_mb

        self.commit_floor_mb = commit_floor_mb

        self.preparation_multiplier = preparation_multiplier

        self.reserve_policy=reserve_policy

        self.reservations = dict(reservations or {})

        if any(not isinstance(v, PreparationReservation) for v in self.reservations.values()):

            raise TypeError('reservations must contain PreparationReservation values')

        self.sampler = sampler or windows_host_memory

        self.events = deque(maxlen=1024)



    def check(self, capability, current, target, *, phase='prepare'):

        adapter = capability.adapter

        delta = max(0, adapter.declared_cost(target).host_ram_mb

                    - adapter.declared_cost(current).host_ram_mb)

        reservation = delta * self.preparation_multiplier

        physical_reservation = commit_reservation = reservation

        profile = self.reservations.get(capability.capability_id)

        uses_profile = (phase == 'prepare' and current < ResidencyLevel.BOUND <= target

                        and profile is not None)

        if uses_profile:

            # 配置是保守下限，不能降低原有声明增量保护。

            physical_reservation = max(reservation, profile.physical_mb)

            commit_reservation = max(reservation, profile.commit_mb)

        event = dict(time=time(), capability_id=capability.capability_id, phase=phase,

                     reservation_mb=reservation,

                     physical_reservation_mb=physical_reservation,

                     commit_reservation_mb=commit_reservation,

                     reservation_basis='resource_profile' if uses_profile else 'declared_ram_multiplier',

                     physical_floor_mb=self.physical_floor_mb,

                     commit_floor_mb=self.commit_floor_mb)

        try:

            sample = self.sampler()

            event.update(asdict(sample))

            if any(not math.isfinite(v) or v < 0 for v in asdict(sample).values()):

                raise ValueError('invalid host memory sample')

        except Exception as error:

            event.update(status='unavailable', error=type(error).__name__)

            self.events.append(event)

            raise HostMemoryUnavailable(f'host memory telemetry unavailable: {capability.capability_id}') from error

        physical_floor,commit_floor=self.physical_floor_mb,self.commit_floor_mb

        if self.reserve_policy is not None:

            physical_floor,commit_floor=self.reserve_policy.floors(sample,learn=phase!='prepared')

            event.update(physical_floor_mb=physical_floor,commit_floor_mb=commit_floor,

                floor_basis='adaptive_policy',reserve_policy=self.reserve_policy.events[-1])

        else:event['floor_basis']='fixed_policy'

        ok = (sample.physical_available_mb >= physical_floor + physical_reservation

              and sample.commit_available_mb >= commit_floor + commit_reservation)

        event['reservation_shortfall_mb']={

            'physical':max(0.,physical_reservation-sample.physical_available_mb),

            'commit':max(0.,commit_reservation-sample.commit_available_mb)}

        event['rejection_kind']=None if ok else ('operation_estimate_shortfall'

            if any(event['reservation_shortfall_mb'].values()) else 'policy_headroom')

        event['status'] = 'passed' if ok else 'rejected'

        self.events.append(event)

        if not ok:

            raise HostMemoryUnavailable(

                f'{capability.capability_id}: {phase}: physical_available_mb='

                f'{sample.physical_available_mb:.1f}, commit_available_mb='

                f'{sample.commit_available_mb:.1f}, physical_reservation_mb={physical_reservation:.1f}, '

                f'commit_reservation_mb={commit_reservation:.1f}, '

                f'physical_floor_mb={physical_floor}, commit_floor_mb={commit_floor}')

