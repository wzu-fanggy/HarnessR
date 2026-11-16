"""将工具资源读数与预算记账分开，未知读数不得视为零占用。"""
from __future__ import annotations

import math
from collections.abc import Mapping


def account_resource_snapshot(snapshot: Mapping, declared: Mapping) -> tuple[dict, dict]:
    """返回用于预算检查的数值及逐项来源。

    正的加载前后 RSS 差值是服务进程的增量观测，并非模型独占内存。
    读数缺失或非正、但该项已登记预计占用时，使用预计值并显式标记。
    预计值不是物理占用上界。明确未驻留的资源可以记为零。
    """
    fields = {"host_ram_mb": "host_ram_mib", "gpu_vram_mb": "gpu_allocated_mib",
              "process_slots": "process_slots", "connection_slots": "connection_slots"}
    costs, basis = {}, {}
    for dimension, field in fields.items():
        expected = float(declared.get(dimension, 0.0))
        if not math.isfinite(expected) or expected < 0:
            raise ValueError("Invalid declared cost: " + dimension)
        if snapshot.get("resident") is False:
            costs[dimension], basis[dimension] = 0.0, "not_resident"
            continue
        raw = snapshot.get(field)
        value = None if raw is None else float(raw)
        valid = value is not None and math.isfinite(value) and value >= 0
        if dimension == "host_ram_mb" and snapshot.get("host_ram_measurement") in {
            "unavailable", "nonpositive_load_delta", "not_observed"}:
            valid = False
        if valid and (value > 0 or expected == 0):
            costs[dimension] = value
            basis[dimension] = ("load_rss_increment" if dimension == "host_ram_mb"
                                and snapshot.get("host_ram_measurement") == "load_rss_increment"
                                else "observed")
        elif expected > 0:
            costs[dimension], basis[dimension] = expected, "declared_fallback"
        elif raw is None and dimension != "host_ram_mb":
            costs[dimension], basis[dimension] = 0.0, "not_budgeted"
        elif raw is None and snapshot.get("host_ram_measurement") is None:
            # 无此占用项目的旧接口，例如只按连接数量统计的 Driver。
            costs[dimension], basis[dimension] = 0.0, "not_budgeted"
        else:
            raise ValueError("Unknown resource cost without declaration: " + dimension)
    return costs, basis
