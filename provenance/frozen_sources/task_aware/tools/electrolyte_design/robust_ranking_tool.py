"""Agent tool for robust multi-temperature ranking of electrolyte candidates.

The tool is advisory in the same sense as the single-point conductivity
predictor: it reports model-robustness statistics and never converts a model
estimate into literature ``evidence`` or into an experimentally established fact.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from .robust_ranking import RobustRankingError, rank_formulations, validate_ranking_request


ROBUST_RANKING_PROCESS_RESOURCE_ID = "process:robust-ranking-worker"


def _invalid_request(exc: RobustRankingError) -> dict[str, Any]:
    return {
        "status": "invalid_request",
        "code": "invalid_ranking_request",
        "recoverable": True,
        "reason": str(exc),
        "error": str(exc),
        "guidance": "仅在任务已有正的最低电导率要求时调用本工具；无阈值比较可使用点预测。不要补造阈值。",
    }


@tool
def rank_formulations_under_uncertainty(request: dict[str, Any]) -> dict[str, Any]:
    """在多个目标温度下对一组域内电解液候选做稳健排序。

    request 需要 ``candidates``（每项给出 candidate_id、salt、solvents、additives，
    以及 composition_inputs 中的 x_NaSalt/x_EC/x_PC、component_mmol 或 molar_ratios）、
    ``temperature_c``（1–6 个互不相同的温度）与 ``min_conductivity_ms_cm``（最低电导率
    约束，必须为正有限数且来自任务要求）。无最低阈值的任务不要补造门槛来调用本工具，
    可比较适用的点预测。可选 ``ensemble_budget`` 限制参与统计的集成成员数。

    返回每个候选满足约束的联合概率、逐温度中位数与分位、跨温度最差表现、Pareto 层、
    排名稳定度、域外/组成不足的拒绝原因，以及该候选自身配方是否在场内训练重采样中的
    诊断。结果属于模型稳健性分析：它是概率与不确定性报告，不是实验事实，也不能替代
    文献证据或单点测量。模型域限制为 NaPF6 的 EC/EMC/PC 体系且不含添加剂。
    """

    try:
        validate_ranking_request(request)
    except RobustRankingError as exc:
        return _invalid_request(exc)

    from orion.core.langgraph_residency import get_active_runtime_resource

    resource = get_active_runtime_resource(ROBUST_RANKING_PROCESS_RESOURCE_ID)
    if resource is None:
        # Compatibility path for direct calls and tests only; the production Agent
        # ToolNode acquires the managed worker before invoking the tool.
        from .robust_ranking import in_process_ensemble

        ensemble = in_process_ensemble()
        try:
            return rank_formulations(request, ensemble=ensemble)
        except RobustRankingError as exc:
            return _invalid_request(exc)

    from .robust_ranking_process import RobustRankingProcessAdapter

    if not isinstance(resource, RobustRankingProcessAdapter):
        raise RuntimeError(
            "leased robust-ranking resource has unexpected adapter type: "
            f"{type(resource).__name__}"
        )
    try:
        return resource.rank_formulations(request)
    except RobustRankingError as exc:
        return _invalid_request(exc)
