"""根 RAG 进程内资源生命周期的只读计量；不管理或卸载实例。"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def process_rss_mib() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 3)
    except (ImportError, OSError):
        return None


def host_ram_pressure() -> float | None:
    """返回系统主机内存使用比例；不可用时返回 None。"""
    try:
        import psutil

        pressure = float(psutil.virtual_memory().percent) / 100.0
        return round(pressure, 6) if 0.0 <= pressure <= 1.0 else None
    except (ImportError, OSError, AttributeError, TypeError, ValueError):
        return None


class ResourceTelemetry:
    """记录进程拥有的资源加载、使用和空闲；事件不含查询文本。"""

    def __init__(self, *, event_path: Path | None = None, clock=time.monotonic) -> None:
        self.event_path = event_path
        self.clock = clock
        self._lock = threading.Lock()
        self._states: dict[str, dict[str, Any]] = {}
        self.write_errors = 0

    def loaded(
        self,
        resource_key: str,
        *,
        load_seconds: float,
        rss_before_mib: float | None = None,
        gpu_before_mib: float | None = None,
        gpu_allocated_mib: float | None,
        gpu_vram_pressure: float | None = None,
    ) -> None:
        now = self.clock()
        with self._lock:
            state = self._states.setdefault(resource_key, self._new_state(resource_key))
            state["resident"] = True
            state["load_count"] += 1
            state["last_load_seconds"] = round(load_seconds, 6)
            state["idle_since"] = now
            state["rss_mib_at_last_load"] = process_rss_mib()
            state["rss_delta_mib_last_load"] = (
                round(state["rss_mib_at_last_load"] - rss_before_mib, 3)
                if state["rss_mib_at_last_load"] is not None and rss_before_mib is not None
                else None
            )
            state["gpu_allocated_mib_at_last_load"] = gpu_allocated_mib
            state["gpu_delta_mib_last_load"] = (
                round(gpu_allocated_mib - gpu_before_mib, 3)
                if gpu_allocated_mib is not None and gpu_before_mib is not None else None
            )
            state["last_host_ram_pressure"] = host_ram_pressure()
            state["last_gpu_vram_pressure"] = gpu_vram_pressure
            self._write({"event": "load", "resource_key": resource_key,
                         "load_seconds": round(load_seconds, 6),
                         "rss_mib": state["rss_mib_at_last_load"],
                         "rss_delta_mib": state["rss_delta_mib_last_load"],
                         "gpu_allocated_mib": gpu_allocated_mib,
                         "gpu_delta_mib": state["gpu_delta_mib_last_load"],
                         "host_ram_pressure": state["last_host_ram_pressure"],
                         "gpu_vram_pressure": state["last_gpu_vram_pressure"]})

    def begin_use(self, resource_key: str, *, gpu_vram_pressure: float | None = None) -> float:
        now = self.clock()
        with self._lock:
            state = self._states.setdefault(resource_key, self._new_state(resource_key))
            state["resident"] = True
            gap = now - state["idle_since"] if state["active_count"] == 0 and state["idle_since"] is not None else None
            state["active_count"] += 1
            state["idle_since"] = None
            state["last_idle_gap_seconds"] = round(max(0.0, gap), 6) if gap is not None else None
            state["last_host_ram_pressure"] = host_ram_pressure()
            state["last_gpu_vram_pressure"] = gpu_vram_pressure
            self._write({"event": "use_start", "resource_key": resource_key,
                         "idle_gap_seconds": state["last_idle_gap_seconds"],
                         "active_count": state["active_count"],
                         "host_ram_pressure": state["last_host_ram_pressure"],
                         "gpu_vram_pressure": state["last_gpu_vram_pressure"]})
        return now

    def end_use(
        self,
        resource_key: str,
        started: float,
        *,
        success: bool,
        gpu_vram_pressure: float | None = None,
    ) -> None:
        now = self.clock()
        with self._lock:
            state = self._states[resource_key]
            state["active_count"] -= 1
            if state["active_count"] < 0:
                raise RuntimeError("resource telemetry active count underflow")
            state["use_count"] += 1
            state["failure_count"] += int(not success)
            state["last_use_seconds"] = round(max(0.0, now - started), 6)
            state["last_host_ram_pressure"] = host_ram_pressure()
            state["last_gpu_vram_pressure"] = gpu_vram_pressure
            if state["active_count"] == 0:
                state["idle_since"] = now
            self._write({"event": "use_end", "resource_key": resource_key,
                         "duration_seconds": state["last_use_seconds"],
                         "success": success, "active_count": state["active_count"],
                         "host_ram_pressure": state["last_host_ram_pressure"],
                         "gpu_vram_pressure": state["last_gpu_vram_pressure"]})

    def snapshot(self) -> dict[str, Any]:
        now = self.clock()
        with self._lock:
            resources = []
            for state in self._states.values():
                row = {key: value for key, value in state.items() if key != "idle_since"}
                idle_since = state["idle_since"]
                row["idle_seconds"] = (
                    round(max(0.0, now - idle_since), 6)
                    if state["resident"] and state["active_count"] == 0 and idle_since is not None
                    else None
                )
                resources.append(row)
            return {"schema_version": "resource-residency-v1", "pid": os.getpid(),
                    "process_rss_mib": process_rss_mib(),
                    "host_ram_pressure": host_ram_pressure(),
                    "resources": sorted(resources, key=lambda row: row["resource_key"]),
                    "telemetry_write_errors": self.write_errors}

    @staticmethod
    def _new_state(resource_key: str) -> dict[str, Any]:
        return {
            "resource_key": resource_key, "resident": False, "active_count": 0,
            "load_count": 0, "use_count": 0, "failure_count": 0,
            "last_load_seconds": None, "last_use_seconds": None,
            "last_idle_gap_seconds": None, "rss_mib_at_last_load": None,
            "rss_delta_mib_last_load": None,
            "gpu_allocated_mib_at_last_load": None,
            "gpu_delta_mib_last_load": None,
            "last_host_ram_pressure": None,
            "last_gpu_vram_pressure": None,
            "idle_since": None,
        }

    def _write(self, event: dict[str, Any]) -> None:
        if self.event_path is None:
            return
        try:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            with self.event_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"schema_version": "resource-event-v1", "pid": os.getpid(),
                                         "wall_time_unix_s": time.time(), **event},
                                        ensure_ascii=False, sort_keys=True) + "\n")
        except OSError:
            # 遥测故障不能改变检索结果；snapshot 暴露失败计数。
            self.write_errors += 1


_recorder = ResourceTelemetry(
    event_path=Path(os.environ["KG_RAG_RESOURCE_EVENTS_PATH"])
    if os.environ.get("KG_RAG_RESOURCE_EVENTS_PATH") else None
)


def recorder() -> ResourceTelemetry:
    return _recorder


def snapshot() -> dict[str, Any]:
    return _recorder.snapshot()
