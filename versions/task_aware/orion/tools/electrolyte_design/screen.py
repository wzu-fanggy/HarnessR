"""候选配方约束评估的兼容工具入口。"""

from __future__ import annotations

from typing import Annotated
from langgraph.prebuilt import InjectedState
from orion.core.retrieved_evidence import restore_candidate_source_text

from langchain_core.tools import tool

from tools.kg.models import AgentScreeningRequest, ScreeningRequest

from .evaluator import evaluate_formulation_constraints


def _get_engine():
    """已移除的内部测试缝；screen_formulations 不得再调用它。"""

    raise RuntimeError("screen_formulations no longer accesses KgQueryEngine")


def get_root_kg_service_client():
    """已移除的内部测试缝；证据补全已迁到 enrichment 工具。"""

    raise RuntimeError("use enrich_formulation_evidence before screening")


@tool
def screen_formulations(request: AgentScreeningRequest, state: Annotated[dict | None, InjectedState] = None) -> list[dict]:
    """确定性评估候选配方，不访问 KG、RAG 或 LLM。

    参数 request 包含 candidates 与结构化 constraint_records；criteria 和
    target_conditions 仅兼容旧调用方。候选证据必须已经带有原文；只有 chunk_id
    的紧凑证据应先显式调用 ``enrich_formulation_evidence``。
    返回 passed、score、matched_properties、criteria_evaluations 和 gaps；
    recommendation 固定为空；这些是局部约束检查结果，主 Agent 结合原文决定
    如何继续和回答。历史结构化模板所需的 DecisionState 由显式兼容调用处理。
    """

    request = ScreeningRequest.model_validate(restore_candidate_source_text(
        request.model_dump(mode="json", exclude_unset=True), state))
    return [
        result.model_dump(mode="json")
        for result in evaluate_formulation_constraints(request)
    ]
