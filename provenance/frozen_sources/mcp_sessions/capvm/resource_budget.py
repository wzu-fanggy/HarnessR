"""只对工具执行侧物理资源设置容量约束。"""

from __future__ import annotations

import math

from .model import ResourceVector


def physical_resource_budget(
    *,
    host_ram_mb: float,
    gpu_vram_mb: float,
    process_slots: float,
    connection_slots: float,
    reserved_policy_ram_mb: float = 0,
    reserved_policy_vram_mb: float = 0,
) -> ResourceVector:
    """构造物理预算，并为策略模型预留自身占用。"""

    values = (
        host_ram_mb, gpu_vram_mb, process_slots, connection_slots,
        reserved_policy_ram_mb, reserved_policy_vram_mb,
    )
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("physical resource limits must be finite and non-negative")
    if reserved_policy_ram_mb > host_ram_mb or reserved_policy_vram_mb > gpu_vram_mb:
        raise ValueError("policy reservation cannot exceed physical budget")
    return ResourceVector(
        prompt_tokens=math.inf,
        host_ram_mb=host_ram_mb - reserved_policy_ram_mb,
        gpu_vram_mb=gpu_vram_mb - reserved_policy_vram_mb,
        process_slots=process_slots,
        connection_slots=connection_slots,
    )
