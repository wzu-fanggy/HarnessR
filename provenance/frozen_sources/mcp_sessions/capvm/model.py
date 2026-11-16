"""Capability Virtual Memory 的核心数据模型。

本模块只定义逻辑 capability、resident level、资源向量和 fault，不依赖 ORION
现有 Agent 图。V0 的关键不变量是：逻辑可寻址性与物理驻留状态分离。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Protocol, runtime_checkable


class CapabilityKind(str, Enum):
    """V0 明确要求覆盖的异构 capability 类型。"""

    TOOL = "tool"
    SKILL = "skill"
    PLUGIN = "plugin"
    PROCESS = "process"
    SESSION = "session"
    MODEL = "model"


class ResidencyLevel(IntEnum):
    """Capability 的统一驻留层级；数字越大表示物理驻留程度越高。"""

    CATALOG = 0
    INTERFACE = 1
    REASONING = 2
    BOUND = 3
    WARM = 4
    ACTIVE = 5


RESOURCE_DIMENSIONS: tuple[str, ...] = (
    "prompt_tokens",
    "host_ram_mb",
    "gpu_vram_mb",
    "process_slots",
    "connection_slots",
)


@dataclass(frozen=True)
class ResourceVector:
    """统一资源向量。

    Governor 使用 declared cost 做 admission；adapter 可以另外报告 measured cost。
    向量允许临时出现负值，以便表示 target-current 的资源增量。
    """

    prompt_tokens: float = 0.0
    host_ram_mb: float = 0.0
    gpu_vram_mb: float = 0.0
    process_slots: float = 0.0
    connection_slots: float = 0.0

    def __add__(self, other: "ResourceVector") -> "ResourceVector":
        return ResourceVector(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            host_ram_mb=self.host_ram_mb + other.host_ram_mb,
            gpu_vram_mb=self.gpu_vram_mb + other.gpu_vram_mb,
            process_slots=self.process_slots + other.process_slots,
            connection_slots=self.connection_slots + other.connection_slots,
        )

    def __sub__(self, other: "ResourceVector") -> "ResourceVector":
        return ResourceVector(
            prompt_tokens=self.prompt_tokens - other.prompt_tokens,
            host_ram_mb=self.host_ram_mb - other.host_ram_mb,
            gpu_vram_mb=self.gpu_vram_mb - other.gpu_vram_mb,
            process_slots=self.process_slots - other.process_slots,
            connection_slots=self.connection_slots - other.connection_slots,
        )

    def value(self, dimension: str) -> float:
        if dimension not in RESOURCE_DIMENSIONS:
            raise KeyError(f"unknown resource dimension: {dimension}")
        return float(getattr(self, dimension))

    def fits(self, budget: "ResourceVector", *, tolerance: float = 1e-9) -> bool:
        return all(
            self.value(dimension) <= budget.value(dimension) + tolerance
            for dimension in RESOURCE_DIMENSIONS
        )

    def violated_dimensions(
        self,
        budget: "ResourceVector",
        *,
        tolerance: float = 1e-9,
    ) -> tuple[str, ...]:
        return tuple(
            dimension
            for dimension in RESOURCE_DIMENSIONS
            if self.value(dimension) > budget.value(dimension) + tolerance
        )

    def positive_part(self) -> "ResourceVector":
        return ResourceVector(
            prompt_tokens=max(0.0, self.prompt_tokens),
            host_ram_mb=max(0.0, self.host_ram_mb),
            gpu_vram_mb=max(0.0, self.gpu_vram_mb),
            process_slots=max(0.0, self.process_slots),
            connection_slots=max(0.0, self.connection_slots),
        )


@dataclass(frozen=True)
class CapabilityDependency:
    """一个 capability 在使用前必须驻留的逻辑依赖。"""

    capability_id: str
    required_level: ResidencyLevel = ResidencyLevel.INTERFACE

    def __post_init__(self) -> None:
        if not self.capability_id:
            raise ValueError("dependency capability_id is required")
        if self.required_level == ResidencyLevel.ACTIVE:
            raise ValueError("dependencies cannot require ACTIVE residency directly")


@runtime_checkable
class CapabilityAdapter(Protocol):
    """不同 capability 类型共享的驻留转换接口。"""

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        """返回 admission 使用的声明资源成本。"""

    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:
        """返回当前可测量的实际资源成本。"""

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        """把 backing resource 提升到 target level。"""

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        """释放或 checkpoint backing resource 到 target level。"""

    def close(self) -> None:
        """最终释放 adapter 持有的外部资源。"""


@dataclass(frozen=True)
class VirtualCapability:
    """Residency-layer logical capability descriptor.

    即使 backing state 不驻留也保持逻辑可寻址。这里的 ``VirtualCapability``
    只统一 identity/dependency/residency/resource 语义；Tool/Skill/Process 被包装到
    本层并不自动意味着它已经具备论文所定义的完整 long-lived evolvable lifecycle。
    完整 I/V/G/Q/R/A 视图由 ``CapabilityLifecycleCoordinates`` 在相应 executable、
    qualification 与 governor state 都存在时组装和校验。
    """

    capability_id: str
    kind: CapabilityKind
    summary: str
    adapter: CapabilityAdapter
    dependencies: tuple[CapabilityDependency, ...] = ()
    pinned_min_level: ResidencyLevel = ResidencyLevel.CATALOG
    maximum_level: ResidencyLevel = ResidencyLevel.WARM
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.capability_id:
            raise ValueError("capability_id is required")
        if not self.summary:
            raise ValueError("summary is required")
        if self.pinned_min_level == ResidencyLevel.ACTIVE:
            raise ValueError("ACTIVE cannot be a pinned minimum level")
        if self.maximum_level == ResidencyLevel.ACTIVE:
            raise ValueError("ACTIVE is a transient use state, not a resident maximum")
        if self.pinned_min_level > self.maximum_level:
            raise ValueError("pinned_min_level cannot exceed maximum_level")
        dependency_ids = [item.capability_id for item in self.dependencies]
        if len(dependency_ids) != len(set(dependency_ids)):
            raise ValueError("dependencies must be unique")
        if self.capability_id in dependency_ids:
            raise ValueError("capability cannot depend on itself")


@dataclass
class CapabilityRuntimeState:
    """Governor 持有的可变驻留状态。"""

    capability_id: str
    current_level: ResidencyLevel = ResidencyLevel.CATALOG
    last_used_tick: int = 0
    active_count: int = 0
    promotions: int = 0
    demotions: int = 0
    compulsory_faults: int = 0
    refaults: int = 0
    ever_resident: bool = False
    highest_level_ever: ResidencyLevel = ResidencyLevel.CATALOG

    @property
    def effective_level(self) -> ResidencyLevel:
        if self.active_count > 0:
            return ResidencyLevel.ACTIVE
        return self.current_level


class CapabilityFault(RuntimeError):
    """逻辑 capability 存在，但当前物理驻留层级不足。"""

    def __init__(
        self,
        capability_id: str,
        current_level: ResidencyLevel,
        required_level: ResidencyLevel,
        request_id: str = "",
        reason: str = "nonresident_capability",
    ) -> None:
        super().__init__(
            f"{capability_id} requires {required_level.name}, "
            f"current={current_level.name}"
        )
        self.capability_id = capability_id
        self.current_level = current_level
        self.required_level = required_level
        self.request_id = request_id
        self.reason = reason


class ResidencyBudgetExceeded(RuntimeError):
    """在不违反 pin/active/dependency 不变量时无法满足 hard residency budget。"""

    def __init__(
        self,
        capability_id: str,
        target_level: ResidencyLevel,
        violated_dimensions: tuple[str, ...],
    ) -> None:
        super().__init__(
            "working_set_exceeds_budget: "
            f"capability={capability_id}, target={target_level.name}, "
            f"dimensions={','.join(violated_dimensions)}"
        )
        self.capability_id = capability_id
        self.target_level = target_level
        self.violated_dimensions = violated_dimensions


class MeasuredResidencyBudgetExceeded(RuntimeError):
    """page-in 后真实 telemetry 仍超出 hard budget，且没有可安全驱逐的非活动资源。"""

    def __init__(
        self,
        *,
        measured_total: ResourceVector,
        budget: ResourceVector,
        violated_dimensions: tuple[str, ...],
    ) -> None:
        super().__init__(
            "measured_working_set_exceeds_budget: "
            f"dimensions={','.join(violated_dimensions)}"
        )
        self.measured_total = measured_total
        self.budget = budget
        self.violated_dimensions = violated_dimensions


@dataclass(frozen=True)
class ResidencyEvent:
    """用于 benchmark 的确定性驻留事件。"""

    tick: int
    event_type: str
    capability_id: str
    from_level: ResidencyLevel
    to_level: ResidencyLevel
    detail: str = ""
