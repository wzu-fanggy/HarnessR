"""Read-only persistence for managed runtime-resource lifecycle observations."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from .governor import CapabilityResidencyGovernor
from .measurement_batch import measurement_batch
from .model import ResidencyLevel
from .telemetry import host_ram_pressure


class ResourceObservationRecorder:
    """Record lease-boundary snapshots without changing residency decisions."""

    def __init__(
        self,
        event_path: Path | None = None,
        *,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        recent_limit: int = 256,
    ) -> None:
        if recent_limit <= 0:
            raise ValueError("recent_limit must be positive")
        self.event_path = event_path.resolve() if event_path is not None else None
        self.wall_clock = wall_clock
        self.monotonic_clock = monotonic_clock
        self._records: deque[dict[str, Any]] = deque(maxlen=recent_limit)
        self._lock = threading.Lock()
        self.write_errors = 0

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._records)

    def record(
        self,
        event_type: str,
        *,
        governor: CapabilityResidencyGovernor,
        tool_names: Sequence[str],
        requirements: Iterable[tuple[str, ResidencyLevel]],
        request_id: str,
    ) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        timestamp = round(float(self.wall_clock()), 6)
        monotonic_timestamp = round(float(self.monotonic_clock()), 6)
        names = [str(name) for name in tool_names]
        pool_error: str | None = None
        requirement_items = tuple(requirements)
        with measurement_batch():
            try:
                pool_total = governor.total_measured_cost()
            except Exception as exc:
                pool_total = None
                pool_error = type(exc).__name__
            resource_states = [
                governor.resource_observation(resource_id)
                for resource_id, _ in requirement_items
            ]

        def utilization(value: float, limit: float) -> float | None:
            if not math.isfinite(limit) or limit <= 0:
                return None
            return round(max(0.0, float(value)) / limit, 6)

        pool_metrics = {
            "host_ram_pressure": host_ram_pressure(),
            "pool_host_ram_utilization": (
                utilization(pool_total.host_ram_mb, governor.budget.host_ram_mb)
                if pool_total is not None else None
            ),
            "pool_gpu_vram_utilization": (
                utilization(pool_total.gpu_vram_mb, governor.budget.gpu_vram_mb)
                if pool_total is not None else None
            ),
            "pool_process_slot_utilization": (
                utilization(pool_total.process_slots, governor.budget.process_slots)
                if pool_total is not None else None
            ),
            "pool_connection_slot_utilization": (
                utilization(pool_total.connection_slots, governor.budget.connection_slots)
                if pool_total is not None else None
            ),
            "pool_measurement_error": pool_error,
        }
        for (resource_id, required_level), observation in zip(
            requirement_items, resource_states
        ):
            row = {
                "schema_version": "orion-resource-observation-v1",
                "event": str(event_type),
                "wall_time_s": timestamp,
                "monotonic_time_s": monotonic_timestamp,
                "request_id": str(request_id),
                "tool_names": names,
                "required_level": required_level.name,
                **pool_metrics,
                **observation,
            }
            self._records.append(row)
            rows.append(row)
        self._append(rows)
        return tuple(rows)

    def record_all(
        self,
        event_type: str,
        *,
        governor: CapabilityResidencyGovernor,
        request_id: str = "",
    ) -> tuple[dict[str, Any], ...]:
        requirements = tuple(
            (resource_id, governor.state(resource_id).current_level)
            for resource_id in governor.catalog.ids()
        )
        return self.record(
            event_type,
            governor=governor,
            tool_names=(),
            requirements=requirements,
            request_id=request_id,
        )

    def _append(self, rows: Sequence[dict[str, Any]]) -> None:
        if self.event_path is None or not rows:
            return
        try:
            with self._lock:
                self.event_path.parent.mkdir(parents=True, exist_ok=True)
                with self.event_path.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(
                            json.dumps(
                                row,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
        except OSError:
            self.write_errors += 1
