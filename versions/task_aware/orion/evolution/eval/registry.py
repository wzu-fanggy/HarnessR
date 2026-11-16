from __future__ import annotations

from orion.evolution.eval.base import Evaluator

#: 评估器名称到评估器实例的全局注册表。
evaluator_registry: dict[str, Evaluator] = {}


def register_evaluator(name: str, evaluator: Evaluator) -> None:
    """将评估器实例注册到全局注册表。"""
    evaluator_registry[name] = evaluator


def get_evaluator(name: str) -> Evaluator | None:
    """从全局注册表获取评估器实例；不存在时返回 ``None``。

    首次调用时会惰性初始化默认评估器。
    """
    init_evaluators()
    return evaluator_registry.get(name)


def init_evaluators() -> None:
    """显式初始化并注册默认评估器；按需调用以避免模块导入时初始化 LLM。"""
    if "electrolyte_design" in evaluator_registry:
        return
    from orion.evolution.eval.design_evaluator import ElectrolyteDesignEvaluator  # noqa: E402

    register_evaluator("electrolyte_design", ElectrolyteDesignEvaluator())
