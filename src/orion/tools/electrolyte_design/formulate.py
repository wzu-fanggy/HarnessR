from __future__ import annotations

from typing import Any, Annotated
from langgraph.prebuilt import InjectedState
from orion.core.retrieved_evidence import normalize_retrieved_references
from langchain_core.tools import tool

from tools.electrolyte_design.models import FormulationCandidate as DesignCandidate
from tools.kg.models import FormulationSpaceRequest
from tools.kg.query_engine import KgQueryEngine
from tools.rag.service_client import get_root_kg_service_client

from .formulation_evidence import assemble_same_paper_candidates


_engine: KgQueryEngine | None = None


def _get_engine() -> KgQueryEngine:
    global _engine
    if _engine is None:
        _engine = KgQueryEngine()
    return _engine


def _to_formulation_request(raw: Any) -> FormulationSpaceRequest:
    if isinstance(raw, FormulationSpaceRequest):
        return raw
    if isinstance(raw, dict):
        return FormulationSpaceRequest.model_validate(raw)
    raise TypeError(f"Unsupported request type: {type(raw)}")


def _hydrate_evidence_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """用 chunk_id 补齐本轮 Agent 传入的截断原文，失败时保留原输入。"""
    chunk_ids = list(dict.fromkeys(
        str(chunk.get("chunk_id") or "") for chunk in chunks if isinstance(chunk, dict)
    ))[:20]
    if not chunk_ids:
        return chunks
    try:
        payload = get_root_kg_service_client().call(
            "/v1/evidence/chunks", {"chunk_ids": chunk_ids}
        )
        by_id = {
            str(item.get("chunk_id")): item
            for item in payload.get("chunks", [])
            if isinstance(item, dict) and item.get("chunk_id")
        }
    except Exception:
        return chunks
    hydrated = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            hydrated.append(chunk)
            continue
        full = by_id.get(str(chunk.get("chunk_id") or ""))
        # 调用方可显式声明这是一个必须保持边界的逐字 observation excerpt。
        # 这用于受控暴露、人工裁剪的原文证据等场景；普通 compact RAG chunk
        # 未设置该标志时仍可按 chunk_id hydrate 到完整 child text。
        if chunk.get("preserve_excerpt") is True:
            hydrated.append(chunk)
            continue
        if not full or not full.get("text"):
            hydrated.append(chunk)
            continue
        merged = dict(chunk)
        merged["text"] = full["text"]
        merged.pop("content", None)
        merged.setdefault("paper_title", full.get("paper_title", ""))
        hydrated.append(merged)
    return hydrated



def _source_evidence(req: FormulationSpaceRequest) -> list[dict[str, Any]]:
    """返回阅读材料，不从文字推断组成或将同论文结果分配给候选。"""
    limit = min(20, max(8, req.max_candidates * 4))
    if req.evidence_chunks:
        chunks = req.evidence_chunks[:20]
        if any(not (c.get("content") or c.get("text") or c.get("excerpt")) for c in chunks):
            chunks = _hydrate_evidence_chunks(chunks)
    else:
        entities = list(dict.fromkeys([
            *(req.preferred_salts or []), *(req.preferred_solvents or []),
            *(req.preferred_additives or []),
        ]))[:10]
        if entities:
            chunks = _get_engine().chunks_for_entities(entities, top_k=limit)
        elif req.target_conditions:
            payload = get_root_kg_service_client().search("literature_chunks", {
                "query": " ".join(req.target_conditions), "top_k": limit,
                "mode": "hybrid", "include_parent": True,
            })
            chunks = payload.get("chunks") if isinstance(payload, dict) else payload
        else:
            return []
    if not isinstance(chunks, list):
        raise ValueError("配方来源检索未返回原文列表")
    records = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("配方来源检索返回了无效原文记录")
        content = chunk.get("content") or chunk.get("text") or chunk.get("excerpt")
        if not chunk.get("chunk_id") or not isinstance(content, str) or not content.strip():
            raise ValueError("配方来源缺少 chunk_id 或原文，请取得正文后再阅读")
        records.append({**chunk, "content": content,
                        "record_type": "source_evidence", "interpretation_status": "not_assessed"})
    return records


@tool
def query_formulation_space(request: Any, state: Annotated[dict | None, InjectedState] = None) -> list[dict]:
    """检索配方相关原文，由 Agent 阅读并确认组成和实验分支。

    参数 request 包含 task_type, target_conditions, preferred_salts, preferred_solvents,
    preferred_additives, excluded_components, evidence_chunks, max_candidates。

    默认 result_mode=evidence，返回带 chunk_id、原文及元数据的 source_evidence 列表，
    interpretation_status=not_assessed；不提取完整配方、不绑定性能、不制造占位候选。
    已传入原文则直接保留；否则按实体偏好或目标条件检索。max_candidates 在默认模式
    仅用于控制检索数量，不表示已找到多少候选；空列表不证明文献不存在配方。
    Agent 可直接使用其他检索工具读原文并构造候选，无需调用本工具。
    仅旧调用方显式指定 result_mode=legacy_candidates 时返回规则组装的候选草稿，
    沿用旧输出结构；草稿的组成与性能归属仍须 Agent 核验，不能当作已验证结论。
    """
    raw = request.model_dump(mode='json') if isinstance(request, FormulationSpaceRequest) else request
    if isinstance(raw, dict):
        raw = normalize_retrieved_references(raw, state)
    req = _to_formulation_request(raw)
    if req.result_mode == "evidence":
        try:
            return _source_evidence(req)
        except (RuntimeError, ValueError) as exc:
            return [{"record_type": "retrieval_error", "error": "formulation_source_unavailable",
                     "message": str(exc), "recoverable": True}]
    engine = _get_engine()

    salts = set(req.preferred_salts or [])
    solvents = set(req.preferred_solvents or [])

    # If no preferred salt/solvent, use target_conditions to infer. Natural-language
    # design briefs are allowed here; KG query validators may reject a condition that is
    # too broad to map to a structured query. Such a condition is a retrieval miss, not
    # a reason to abort the whole Agent episode.
    if not salts or not solvents:
        for condition in req.target_conditions:
            try:
                improve_results = engine.run_query(
                    "what_improves", {"target": condition, "limit": 10}
                )
            except ValueError:
                improve_results = []
            for result in improve_results:
                entity = result.start_entity
                labels = result.entity_labels or []
                if "SodiumSalt" in labels:
                    salts.add(entity)
                elif "Solvent" in labels or "Additive" in labels:
                    solvents.add(entity)
            try:
                condition_results = engine.run_query(
                    "find_materials_for_condition",
                    {
                        "condition": condition,
                        "limit": 10,
                        "material_types": ["SodiumSalt", "Electrolyte"],
                    },
                )
            except ValueError:
                condition_results = []
            for result in condition_results:
                entity = result.start_entity
                labels = result.entity_labels or []
                if "SodiumSalt" in labels:
                    salts.add(entity)
                elif "Solvent" in labels:
                    solvents.add(entity)

    excluded = {e.lower() for e in req.excluded_components}

    candidates: list[DesignCandidate] = []
    requested_additives = [
        additive for additive in (req.preferred_additives or [])
        if additive.lower() not in excluded
    ]

    # 配方证据主路径：从本轮检索到的原文 chunk 文本抽取显式配方，并按同一论文组装。
    # 命中 child、读原文、抽配方；不从跨论文聚合的 KG 实体边（CONTAINS）推断配方。
    if req.evidence_chunks:
        hydrated_chunks = _hydrate_evidence_chunks(req.evidence_chunks[:20])
        evidence_candidates = assemble_same_paper_candidates(
            hydrated_chunks,
            preferred_salts=list(salts),
            preferred_solvents=list(solvents),
            preferred_additives=requested_additives,
            target_conditions=req.target_conditions,
            max_candidates=req.max_candidates,
        )
        candidates.extend(evidence_candidates)
        if len(candidates) >= req.max_candidates:
            return [candidate.model_dump() for candidate in candidates[: req.max_candidates]]

    # 兜底路径：本轮未携带检索证据、或检索证据不足以凑满候选时，按盐/溶剂/添加剂
    # 实体名召回原文 chunk（resolver 走同义闭包），再从这些 chunk 文本抽显式配方，
    # 仍限定同一论文组装。
    if len(candidates) < req.max_candidates and salts:
        chunks: list[dict[str, Any]] = []
        entity_names = list(
            dict.fromkeys([*salts, *solvents, *requested_additives])
        )[:10]
        try:
            chunks = engine.chunks_for_entities(
                entity_names, top_k=min(15, max(8, req.max_candidates * 4))
            )
        except Exception:
            chunks = []
        assembled = assemble_same_paper_candidates(
            chunks,
            preferred_salts=list(salts),
            preferred_solvents=list(solvents),
            preferred_additives=requested_additives,
            target_conditions=req.target_conditions,
            max_candidates=req.max_candidates,
        )
        remaining = max(0, req.max_candidates - len(candidates))
        candidates.extend(assembled[:remaining])

    if not candidates:
        return [
            DesignCandidate(
                candidate_id="candidate-1",
                salt=None,
                target_rationale=[
                    "检索到的原文 chunk 中未抽到含显式溶剂比例的配方候选，"
                    "请补充检索或放宽约束。"
                ],
                evidence=[],
                evidence_level="weak",
            ).model_dump()
        ]

    return [candidate.model_dump() for candidate in candidates]


# Backwards-compatible alias for callers still importing formulate_candidates.
formulate_candidates = query_formulation_space
