"""可演化 LangGraph plugin 的 capability-residency adapter。"""

from __future__ import annotations

from typing import Protocol

from ..model import ResidencyLevel, ResourceVector
from .base import BaseCapabilityAdapter, estimate_prompt_tokens


class ResidentPlugin(Protocol):
    """Residency adapter 所需的最小 plugin contract。"""

    plugin_id: str
    resident_mb: float

    @property
    def is_mounted(self) -> bool: ...

    @property
    def resident_bytes(self) -> int: ...

    def mount(self) -> None: ...

    def unmount(self) -> None: ...


class EvolvablePluginCapabilityAdapter(BaseCapabilityAdapter):
    """把插件执行图映射到统一 residency lifecycle。

    INTERFACE/REASONING 只计入模型可见描述成本；BOUND 及以上表示执行图已 materialize。
    插件 learned state 不属于 resident cost，因为它是持久状态而不是当前执行 working set。
    """

    def __init__(
        self,
        *,
        plugin: ResidentPlugin,
        interface_text: str,
        reasoning_text: str = "",
    ) -> None:
        if not interface_text.strip():
            raise ValueError("interface_text is required")
        self.plugin = plugin
        self.interface_text = interface_text
        self.reasoning_text = reasoning_text

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        prompt_tokens = 0.0
        if level >= ResidencyLevel.INTERFACE:
            prompt_tokens += estimate_prompt_tokens(self.interface_text)
        if level >= ResidencyLevel.REASONING:
            prompt_tokens += estimate_prompt_tokens(self.reasoning_text)
        host_ram_mb = self.plugin.resident_mb if level >= ResidencyLevel.BOUND else 0.0
        return ResourceVector(
            prompt_tokens=prompt_tokens,
            host_ram_mb=host_ram_mb,
        )

    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:
        prompt_tokens = 0.0
        if level >= ResidencyLevel.INTERFACE:
            prompt_tokens += estimate_prompt_tokens(self.interface_text)
        if level >= ResidencyLevel.REASONING:
            prompt_tokens += estimate_prompt_tokens(self.reasoning_text)
        host_ram_mb = 0.0
        if level >= ResidencyLevel.BOUND and self.plugin.is_mounted:
            host_ram_mb = self.plugin.resident_bytes / (1024.0 * 1024.0)
        return ResourceVector(
            prompt_tokens=prompt_tokens,
            host_ram_mb=host_ram_mb,
        )

    def model_context_blocks(self, level: ResidencyLevel) -> tuple[str, ...]:
        if level <= ResidencyLevel.CATALOG:
            return ()
        blocks = [self.interface_text]
        if level >= ResidencyLevel.REASONING and self.reasoning_text.strip():
            blocks.append(self.reasoning_text)
        return tuple(blocks)

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current < ResidencyLevel.BOUND <= target:
            self.plugin.mount()

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current >= ResidencyLevel.BOUND > target:
            self.plugin.unmount()

    def close(self) -> None:
        self.plugin.unmount()
