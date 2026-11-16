from __future__ import annotations

from typing import Any
from langchain_core.tools import tool

from tools.kg.models import KgQueryResult
from tools.kg.query_engine import KgQueryEngine


_engine: KgQueryEngine | None = None


def _get_engine() -> KgQueryEngine:
    global _engine
    if _engine is None:
        _engine = KgQueryEngine()
    return _engine


@tool
def kg_query(query_type: str, params: dict[str, Any]) -> list[dict] | dict[str, Any]:
    """对 Neo4j 知识图谱执行结构化查询。

    Agent 按当前问题与证据缺口选择本工具，领域方法可按需加载 Skill。
    返回的图谱关系是核验线索；涉及具体配方、测试条件与性能的结论仍需对应原文支持。

    query_type 可选：
    - "find_formulations": 按盐/溶剂/添加剂找配方。params: {"salt": "NaPF6", "solvent": "DEGDME", "additive": "FEC", "limit": 10}
    - "what_improves": 什么改善某目标。params: {"target": "low temperature performance", "limit": 10}
    - "properties_of": 某材料的属性。params: {"material": "NaPF6", "limit": 10}
    - "compare_materials": 对比两种材料。params: {"material_a": "FEC", "material_b": "VC", "limit": 10}
    - "find_co_solvents": 某盐可溶于哪些溶剂。params: {"salt": "NaPF6", "limit": 10}
    - "find_materials_for_condition": 某条件下测试过的材料。params: {"condition": "-40°C", "limit": 10, "material_types": ["SodiumSalt", "Electrolyte"]}
      material_types 可选，限定返回的材料类别（Electrolyte/Anode/Cathode/Component/Additive/SodiumSalt/Solvent/Precursor，默认全部）。
      查"哪些盐/电解液在目标温度下测试过"时传 ["SodiumSalt", "Electrolyte"]，避免被负极/正极类材料淹没。

    返回每条结果的 start_entity, relation, end_entity, confidence_mean, entity_labels, chunk_ids, kg_paths。
    """
    engine = _get_engine()
    try:
        results = engine.run_query(query_type, params or {})
    except (ValueError, RuntimeError) as exc:
        return {
            "error": "kg_query_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "query_type": query_type,
            "params": params or {},
            "recoverable": isinstance(exc, ValueError),
        }
    return [r.model_dump() for r in results]
