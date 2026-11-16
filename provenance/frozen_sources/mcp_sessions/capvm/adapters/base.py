"""Capability adapter 的共享辅助实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..model import ResidencyLevel, ResourceVector


class BaseCapabilityAdapter(ABC):
    """把不同 backing resource 映射到统一 residency lifecycle。"""

    @abstractmethod
    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        ...

    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:
        """默认以 declared cost 作为测量回退；真实资源 adapter 应覆盖。"""

        return self.declared_cost(level)

    def model_context_blocks(self, level: ResidencyLevel) -> tuple[str, ...]:
        """返回该驻留层级实际应暴露给模型的 capability 文本块。

        外部进程/会话等没有模型侧文本的 adapter 默认返回空。调用方可以把这些块
        组合进具体 provider 的 prompt/tool-schema surface；固定格式开销由调用方承担。
        """

        del level
        return ()

    @abstractmethod
    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        ...

    @abstractmethod
    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        ...

    def close(self) -> None:
        """无外部资源的 adapter 默认无需清理。"""


def estimate_prompt_tokens(text: str) -> int:
    """V0 的确定性 token 近似，仅用于本地系统 benchmark 的 residency cost。"""

    normalized = text.strip()
    if not normalized:
        return 0
    # 不依赖具体 provider tokenizer；正式 benchmark 同时保存真实字节数。
    return max(1, (len(normalized) + 3) // 4)
