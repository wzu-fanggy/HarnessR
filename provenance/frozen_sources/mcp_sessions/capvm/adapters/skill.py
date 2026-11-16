"""Skill capability 的 progressive-disclosure residency adapter。"""

from __future__ import annotations

from ..model import ResidencyLevel, ResourceVector
from .base import BaseCapabilityAdapter, estimate_prompt_tokens


class SkillCapabilityAdapter(BaseCapabilityAdapter):
    """Catalog 中保留 Skill 摘要，R2 才加载完整执行指引。"""

    def __init__(
        self,
        *,
        index_text: str,
        instruction_text: str,
    ) -> None:
        self.index_text = index_text
        self.instruction_text = instruction_text

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level <= ResidencyLevel.CATALOG:
            return ResourceVector()
        index_tokens = estimate_prompt_tokens(self.index_text)
        if level == ResidencyLevel.INTERFACE:
            return ResourceVector(prompt_tokens=index_tokens)
        return ResourceVector(
            prompt_tokens=index_tokens + estimate_prompt_tokens(self.instruction_text)
        )

    def model_context_blocks(self, level: ResidencyLevel) -> tuple[str, ...]:
        if level <= ResidencyLevel.CATALOG:
            return ()
        blocks = [self.index_text]
        if level >= ResidencyLevel.REASONING and self.instruction_text.strip():
            blocks.append(self.instruction_text)
        return tuple(blocks)

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        del current, target

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        del current, target
