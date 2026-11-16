from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class EvalResult(BaseModel):
    """评估结果模型，包含总体评分、维度评分、推理说明与是否通过。"""

    score: float = Field(ge=0.0, le=1.0)
    dimensions: dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""
    passed: bool = False


class Evaluator(ABC):
    """评估器抽象基类，接收运行状态并返回评估结果。"""

    @abstractmethod
    async def evaluate(self, run_state: dict[str, Any]) -> EvalResult:
        """评估给定的运行状态并返回 ``EvalResult``。"""
        ...
