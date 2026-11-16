"""可 checkpoint 的 session/environment capability adapter。"""

from __future__ import annotations

import socket
from copy import deepcopy
from time import monotonic

from ..model import ResidencyLevel, ResourceVector
from .base import BaseCapabilityAdapter


class SessionCapabilityAdapter(BaseCapabilityAdapter):
    """用真实 socket pair 表示一个有连接资源的可恢复 session。

    R3+ 时 session 为 live；降到 R2- 时先 checkpoint 状态再关闭 sockets；重新
    materialize 后从 checkpoint 恢复，借此验证 eviction 不等于 logical deletion。
    """

    def __init__(
        self,
        *,
        startup_delay_s: float = 0.0,
        resident_mb: float = 0.0,
    ) -> None:
        if startup_delay_s < 0:
            raise ValueError("startup_delay_s must be non-negative")
        if resident_mb < 0:
            raise ValueError("resident_mb must be non-negative")
        self.startup_delay_s = float(startup_delay_s)
        self.resident_mb = float(resident_mb)
        self._sockets: tuple[socket.socket, socket.socket] | None = None
        self._resident_buffer: bytearray | None = None
        self._checkpoint: dict[str, object] = {}
        self._live_state: dict[str, object] | None = None
        self.last_startup_latency_s = 0.0
        self.last_shutdown_latency_s = 0.0

    @property
    def is_connected(self) -> bool:
        return self._sockets is not None

    @property
    def checkpoint(self) -> dict[str, object]:
        return deepcopy(self._checkpoint)

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND:
            return ResourceVector()
        return ResourceVector(
            host_ram_mb=self.resident_mb,
            connection_slots=1.0,
        )

    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND or not self.is_connected:
            return ResourceVector()
        allocated_mb = (
            len(self._resident_buffer) / (1024.0 * 1024.0)
            if self._resident_buffer is not None
            else 0.0
        )
        return ResourceVector(
            host_ram_mb=allocated_mb,
            connection_slots=1.0,
        )

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current < ResidencyLevel.BOUND <= target and not self.is_connected:
            self._open_session()

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current >= ResidencyLevel.BOUND > target:
            self._checkpoint_and_close()

    def write_state(self, key: str, value: object) -> None:
        if self._live_state is None:
            raise RuntimeError("session capability is not resident")
        self._live_state[key] = deepcopy(value)

    def read_state(self, key: str, default: object = None) -> object:
        if self._live_state is None:
            raise RuntimeError("session capability is not resident")
        return deepcopy(self._live_state.get(key, default))

    def close(self) -> None:
        self._checkpoint_and_close()

    def _open_session(self) -> None:
        started = monotonic()
        first, second = socket.socketpair()
        first.setblocking(False)
        second.setblocking(False)
        self._sockets = (first, second)
        allocation_bytes = int(round(self.resident_mb * 1024 * 1024))
        if allocation_bytes:
            buffer = bytearray(allocation_bytes)
            for index in range(0, allocation_bytes, 4096):
                buffer[index] = 1
            self._resident_buffer = buffer
        else:
            self._resident_buffer = None
        self._live_state = deepcopy(self._checkpoint)
        if self.startup_delay_s:
            import time

            time.sleep(self.startup_delay_s)
        self.last_startup_latency_s = monotonic() - started

    def _checkpoint_and_close(self) -> None:
        if self._sockets is None and self._live_state is None:
            return
        started = monotonic()
        if self._live_state is not None:
            self._checkpoint = deepcopy(self._live_state)
        if self._sockets is not None:
            for item in self._sockets:
                try:
                    item.close()
                except OSError:
                    pass
        self._sockets = None
        self._resident_buffer = None
        self._live_state = None
        self.last_shutdown_latency_s = monotonic() - started
