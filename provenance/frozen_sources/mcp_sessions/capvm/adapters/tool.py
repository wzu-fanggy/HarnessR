"""Tool capability 的 prompt/interface residency adapter。"""

from __future__ import annotations

from ..model import ResidencyLevel, ResourceVector
from .base import BaseCapabilityAdapter, estimate_prompt_tokens


class ToolCapabilityAdapter(BaseCapabilityAdapter):
    """把 tool catalog、schema 与完整说明映射到统一驻留层级。"""

    def __init__(
        self,
        *,
        interface_text: str,
        reasoning_text: str = "",
    ) -> None:
        self.interface_text = interface_text
        self.reasoning_text = reasoning_text

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level <= ResidencyLevel.CATALOG:
            return ResourceVector()
        interface_tokens = estimate_prompt_tokens(self.interface_text)
        if level == ResidencyLevel.INTERFACE:
            return ResourceVector(prompt_tokens=interface_tokens)
        reasoning_tokens = estimate_prompt_tokens(self.reasoning_text)
        return ResourceVector(prompt_tokens=interface_tokens + reasoning_tokens)

    def model_context_blocks(self, level: ResidencyLevel) -> tuple[str, ...]:
        if level <= ResidencyLevel.CATALOG:
            return ()
        blocks = [self.interface_text]
        if level >= ResidencyLevel.REASONING and self.reasoning_text.strip():
            blocks.append(self.reasoning_text)
        return tuple(blocks)

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        del current, target

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        del current, target
