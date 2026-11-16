"""KG+RAG hybrid search tools for orion integration."""

from __future__ import annotations



import hashlib

import os

from dataclasses import asdict, is_dataclass

from typing import Any



import httpx



from langchain_core.tools import tool

from tools.rag.service_client import RootKgServiceClient, get_root_kg_service_client



from tools.kg.models import EvidenceRecord, KgPath



try:

    from pydantic import BaseModel

except Exception:  # pragma: no cover - pydantic is available in production

    BaseModel = None  # type: ignore[misc,assignment]





# 检索硬上限：防止 agent/skill 传过大 top_k 导致笔记本内存/上下文爆掉。

# 可通过环境变量覆盖；默认保持与 skill 推荐值兼容（top_k=10, per_query=8）。

_RETRIEVAL_MAX_TOP_K = int(os.environ.get("ORION_RETRIEVAL_MAX_TOP_K", "12"))

_RETRIEVAL_MAX_PER_QUERY_TOP_K = int(os.environ.get("ORION_RETRIEVAL_MAX_PER_QUERY_TOP_K", "10"))

_RETRIEVAL_MAX_QUERIES = int(os.environ.get("ORION_RETRIEVAL_MAX_QUERIES", "5"))





def _json_safe(value: Any) -> Any:

    """Recursively serialize Pydantic models / dataclasses to plain JSON-safe types."""

    if BaseModel is not None and isinstance(value, BaseModel):

        return _json_safe(value.model_dump())

    if is_dataclass(value) and not isinstance(value, type):

        return _json_safe(asdict(value))

    if isinstance(value, (list, tuple)):

        return [_json_safe(item) for item in value]

    if isinstance(value, dict):

        return {str(k): _json_safe(v) for k, v in value.items()}

    return value





def _cap_retrieval_top_k(top_k: int, max_val: int = _RETRIEVAL_MAX_TOP_K) -> int:

    """Clamp top_k to the global hard cap and log if clamped."""

    # 消融实验在运行时设置该变量；必须在调用时读取，不能只使用模块导入时的常量。

    short_context = os.environ.get("ORION_ABLATION_SHORT_CTX")

    if short_context:

        try:

            max_val = min(max_val, max(1, int(short_context)))

        except ValueError:

            pass

    if top_k > max_val:

        print(

            f"[search] clamp top_k {top_k} -> {max_val} (ORION_RETRIEVAL_MAX_TOP_K)",

            flush=True,

        )

        return max_val

    return top_k





def _cap_per_query_top_k(per_query_top_k: int) -> int:

    """Clamp per_query_top_k to the global hard cap."""

    if per_query_top_k > _RETRIEVAL_MAX_PER_QUERY_TOP_K:

        print(

            f"[search] clamp per_query_top_k {per_query_top_k} -> "

            f"{_RETRIEVAL_MAX_PER_QUERY_TOP_K} (ORION_RETRIEVAL_MAX_PER_QUERY_TOP_K)",

            flush=True,

        )

        return _RETRIEVAL_MAX_PER_QUERY_TOP_K

    return per_query_top_k





def _error_evidence(message: str) -> list[dict]:

    """Return a standard error evidence list for kg_rag failures."""

    return [

        {

            "error": True,

            "content_type": "error",

            "content": f"kg_rag search unavailable: {message}",

            "source": "kg_rag",

        }

    ]





def _is_error_response(parsed: Any) -> bool:

    """Detect kg_rag backend error payloads produced by _error_evidence/_error_dict."""

    if isinstance(parsed, dict) and parsed.get("error"):

        return True

    if isinstance(parsed, list) and parsed and parsed[0].get("error"):

        return True

    return False





def _error_dict(message: str) -> dict:

    """Return a standard error dict for kg_rag failures that expect a single object."""

    return {

        "error": True,

        "content_type": "error",

        "content": f"kg_rag search unavailable: {message}",

        "source": "kg_rag",

    }





class _RagServiceClient(RootKgServiceClient):

    """兼容旧 RAG operation 调用；新增功能请用 RootKgServiceClient。"""



    def call(self, operation: str, payload: dict[str, Any]) -> Any:

        try:

            return RootKgServiceClient.call(self, f"/v1/search/{operation}", payload)

        except RuntimeError as exc:

            raise RuntimeError(str(exc).replace("local KG service", "local RAG service")) from exc





_rag_service_client: _RagServiceClient | None = None





def _get_rag_service_client() -> _RagServiceClient:

    global _rag_service_client

    if _rag_service_client is None:

        _rag_service_client = _RagServiceClient()

    return _rag_service_client





def _format_chunk(chunk: dict, score: float, source: str = "vector") -> dict:

    """Normalize a raw chunk dict into the evidence format expected by agents.



    Source anchor fields are passed through with safe defaults so that missing

    Neo4j properties do not cause downstream serialization warnings.

    """

    text = chunk.get("text", "") or ""

    triples = chunk.get("triples", []) or []

    source_anchor = chunk.get("source_anchor") or {}

    if not isinstance(source_anchor, dict):

        source_anchor = _json_safe(source_anchor)

        if not isinstance(source_anchor, dict):

            source_anchor = {}



    def _anchor(field: str, default: Any = None) -> Any:

        return source_anchor.get(field, chunk.get(field, default))



    kg_paths = [

        KgPath(start_entity=t.get("s"), relation=t.get("r"), end_entity=t.get("t"))

        for t in triples[:5]

    ]



    # kg_rag 后端有时不返回 paper_id，但 paper_title 一定存在；用 title 兜底

    # 保证下游按 paper 去重/截断（如 max_per_paper）不会把所有 chunk 当成同一篇。

    paper_title = chunk.get("paper_title", "")

    paper_id = chunk.get("paper_id") or paper_title

    chunk_id = chunk.get("chunk_id", "")

    parent_chunk_id = chunk.get("parent_id")

    payload = {

        "content_type": "kg_rag",

        # 单篇论文检索保留较长预览，避免图片 Markdown 占满前缀而截掉其后的图注。

        "content": text[:2400] if source == "paper_vector" else text[:1000],

        "paper_title": paper_title,

        "paper_id": paper_id,

        "section_title": chunk.get("section", ""),

        "chunk_id": chunk_id,

        "parent_chunk_id": parent_chunk_id,

        "score": score,

        "entities": list(chunk.get("entities", []) or []),

        "triples": triples[:10],

        "kg_paths": [p.model_dump() for p in kg_paths],

        "source": source,

        "page_idx": _anchor("page_idx"),

        "paragraph_idx": _anchor("paragraph_idx"),

        "bbox": _anchor("bbox"),

        "char_start": _anchor("char_start"),

        "char_end": _anchor("char_end"),

        "text_hash": _anchor("text_hash", ""),

    }

    if isinstance(paper_id, str) and paper_id and isinstance(chunk_id, str) and chunk_id and text:

        quote = text[:240]

        payload["runtime_source_span"] = {

            "source_kind": "vector_child",

            "paper_id": paper_id,

            "parent_chunk_id": parent_chunk_id,

            "child_chunk_id": chunk_id,

            "chunk_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),

            "char_start": 0,

            "char_end": len(quote),

            "quote": quote,

        }

    return _json_safe(payload)





def _rrf_merge_weighted(

    results: list[list[dict]], weights: list[float], k: int = 60

) -> list[dict]:

    """Reciprocal rank fusion across multiple ranked chunk lists with per-query weights."""

    scores: dict[str, float] = {}

    chunks: dict[str, dict] = {}

    for weight, result in zip(weights, results):

        for rank, chunk in enumerate(result):

            cid = chunk.get("chunk_id")

            if not cid:

                continue

            chunks[cid] = chunk

            # Standard RRF uses 1-indexed rank.

            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    return [chunks[cid] for cid, _ in ranked]





def _apply_per_paper_cap(ranked: list[dict], max_per_paper: int) -> list[dict]:

    """Enforce a maximum number of returned chunks per paper."""

    if max_per_paper is None or max_per_paper <= 0:

        return ranked

    counts: dict[str, int] = {}

    out: list[dict] = []

    for chunk in ranked:

        # paper_id 缺失时回退到 paper_title，避免空 key 把所有 chunk 当同一篇。

        pid = chunk.get("paper_id") or chunk.get("paper_title") or ""

        if counts.get(pid, 0) >= max_per_paper:

            continue

        out.append(chunk)

        counts[pid] = counts.get(pid, 0) + 1

    return out





@tool

def search_literature_chunks(

    query: str,

    top_k: int = 8,

    mode: str = "hybrid",

    sections: list[str] | None = None,

    paper_ids: list[str] | None = None,

    entities: list[str] | None = None,

    exclude_low_value: bool = True,

    expand_hops: int = 0,

    include_parent: bool = True,

) -> list[dict]:

    """基于向量+KG混合检索，支持按章节、论文ID、实体集合过滤。



    适用场景：需要从钠离子电池文献中获取结构化证据的问题，例如配方成分、

    机理关系、性能属性及对应的三元组证据。



    Args:

        query: 检索查询（英文效果更好）。

        top_k: 返回结果数量，默认 10（设计类任务常需更大召回，可在 skill 中设 15+）。

        mode: 检索模式，可选 "hybrid"/"vector"/"graph"。

        sections: 限定章节标题列表（可选）。

        paper_ids: 限定论文ID列表（可选）。

        entities: 需要命中的实体名称列表（可选）。

        exclude_low_value: 是否过滤 References/Acknowledgements 等低价值章节。

        expand_hops: 是否通过 KG 实体邻居扩展结果（>0 启用）。

        include_parent: 是否补充父块上下文。



    Returns:

        相关 chunk 列表，每项为包含文本、元数据、实体、三元组和评分的证据字典。

    """

    try:

        top_k = _cap_retrieval_top_k(top_k)

        print(

            f"[search_literature_chunks] query={query!r} top_k={top_k} mode={mode!r}",

            flush=True,

        )

        parsed = _get_rag_service_client().call(

            "literature_chunks",

            {

                "query": query,

                "top_k": top_k,

                "mode": mode,

                "sections": sections,

                "paper_ids": paper_ids,

                "entities": entities,

                "exclude_low_value": exclude_low_value,

                "expand_hops": expand_hops,

                "include_parent": include_parent,

            },

        )

        if _is_error_response(parsed):

            return parsed if isinstance(parsed, list) else [parsed]

        chunks = parsed.get("chunks", []) if isinstance(parsed, dict) else parsed

        result = [

            _format_chunk(c, c.get("score", 0.0), source=c.get("source", "vector"))

            for c in chunks

        ]



        seen: dict[str, dict] = {}

        for c in result:

            cid = c["chunk_id"]

            if cid not in seen or c["score"] > seen[cid]["score"]:

                seen[cid] = c

        result = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]



        print(

            f"[search_literature_chunks] returned {len(result)} chunks; "

            f"titles={sorted({c['paper_title'] for c in result})}",

            flush=True,

        )

        for i, chunk in enumerate(result, start=1):

            chunk["index"] = i

        return result

    except Exception as exc:  # noqa: BLE001

        return _error_evidence(str(exc))





@tool

def resolve_literature_papers(query: str, top_k: int = 5) -> dict:

    """按作者、题名、DOI 或具体实验标识定位本地论文的 paper_id。



    也可查询用户给出的电极缩写或数值与单位，返回正文中的字面命中。

    建议使用少量有辨识度的词，而非整个问题。命中只是线索，仍需查看

    原文核对电池类型、配方与性能是否符合任务。该工具不调用向量模型，不访问

    外部文献 API，也不写入 KG。解析成功后，应将返回的 paper_id 传给

    search_paper_chunks.paper_id 做论文内聚焦检索。

    """

    try:

        parsed = _get_rag_service_client().call(

            "resolve_papers",

            {"query": query, "top_k": max(1, min(top_k, 10))},

        )

        if _is_error_response(parsed):

            return parsed if isinstance(parsed, dict) else _error_dict(str(parsed))

        if not isinstance(parsed, dict):

            return _error_dict("unexpected resolve_papers result format")

        return _json_safe(parsed)

    except Exception as exc:  # noqa: BLE001

        return _error_dict(str(exc))





@tool

def search_paper_chunks(

    paper_id: str,

    query: str,

    top_k: int = 8,

    sections: list[str] | None = None,

) -> list[dict]:

    """在已解析的单篇论文内部检索相关 chunk，避免全库召回后再过滤。



    用于 ``resolve_literature_papers`` 已确认 ``paper_id`` 的 named-study

    问题，或通用检索已经返回明确 ``paper_id``、但仍缺少配方/目标条件证据

    的设计问题。query 应使用英文技术词概括缺失事实，不得包含作者名，也

    不得用本工具访问任意文件路径。

    """

    try:

        top_k = _cap_retrieval_top_k(top_k)

        parsed = _get_rag_service_client().call(

            "paper_search",

            {

                "paper_id": paper_id,

                "query": query,

                "sections": sections,

                "top_k": top_k,

            },

        )

        if _is_error_response(parsed):

            return parsed if isinstance(parsed, list) else [parsed]

        chunks = parsed.get("chunks", []) if isinstance(parsed, dict) else parsed

        result = [

            _format_chunk(c, c.get("score", 0.0), source="paper_vector")

            for c in chunks

        ]

        for index, chunk in enumerate(result[:top_k], start=1):

            chunk["index"] = index

        return result[:top_k]

    except Exception as exc:  # noqa: BLE001

        return _error_evidence(str(exc))





@tool

def search_literature_multiview(

    queries: list[str],

    top_k: int = 10,

    per_query_top_k: int = 10,

    weights: list[float] | None = None,

    mode: str = "hybrid",

    sections: list[str] | None = None,

    paper_ids: list[str] | None = None,

    entities: list[str] | None = None,

    max_per_paper: int = 2,

    rrf_k: int = 60,

) -> list[dict]:

    """对多个查询分别执行混合检索，用加权 RRF 融合去重后返回综合排序结果。



    适用于需要从不同角度查询同一目标的问题，例如配方设计时分别检索不同

    盐/溶剂/添加剂组合，再合并证据。



    Args:

        queries: 多个检索查询（英文效果更好）。

        top_k: 最终返回结果数量。

        per_query_top_k: 每个查询返回的结果数量。

        weights: 每个查询的 RRF 权重，长度需与 queries 一致；默认等权重 1.0。

        mode: 检索模式，可选 "hybrid"/"vector"/"graph"。

        sections: 限定章节标题列表（可选）。

        paper_ids: 限定论文ID列表（可选）。

        entities: 需要命中的实体名称列表（可选）。

        max_per_paper: 每篇论文最多返回的 chunk 数，<=0 表示不限制。

        rrf_k: RRF 融合参数 k，默认 60。



    Returns:

        合并后的相关 chunk 列表，每项包含文本、元数据、实体、三元组和评分。

    """

    try:

        n_queries = len(queries)

        if n_queries > _RETRIEVAL_MAX_QUERIES:

            print(

                f"[search_literature_multiview] clamp queries {n_queries} -> "

                f"{_RETRIEVAL_MAX_QUERIES} (ORION_RETRIEVAL_MAX_QUERIES)",

                flush=True,

            )

            queries = queries[:_RETRIEVAL_MAX_QUERIES]

            if weights:

                weights = weights[:_RETRIEVAL_MAX_QUERIES]

            n_queries = _RETRIEVAL_MAX_QUERIES



        if weights and len(weights) == n_queries:

            query_weights = [float(w) for w in weights]

        else:

            query_weights = [1.0] * n_queries



        top_k = _cap_retrieval_top_k(top_k)

        per_query_top_k = _cap_per_query_top_k(per_query_top_k)



        all_results: list[list[dict]] = []

        for query, weight in zip(queries, query_weights):

            print(

                f"[search_literature_multiview] query={query!r} "

                f"per_query_top_k={per_query_top_k} weight={weight}",

                flush=True,

            )

            parsed = _get_rag_service_client().call(

                "hybrid_search_filtered",

                {

                    "query": query,

                    "top_k": per_query_top_k,

                    "mode": mode,

                    "sections": sections,

                    "paper_ids": paper_ids,

                    "entities": entities,

                },

            )

            if _is_error_response(parsed):

                return parsed if isinstance(parsed, list) else [parsed]

            chunks = parsed.get("chunks", []) if isinstance(parsed, dict) else parsed

            all_results.append([_format_chunk(c, c.get("score", 0.0)) for c in chunks])



        merged = _rrf_merge_weighted(all_results, query_weights, rrf_k)

        merged = _apply_per_paper_cap(merged, max_per_paper)

        print(f"[search_literature_multiview] merged {len(merged)} unique chunks", flush=True)

        for i, chunk in enumerate(merged[:top_k], start=1):

            chunk["index"] = i

        return merged[:top_k]

    except Exception as exc:  # noqa: BLE001

        return _error_evidence(str(exc))





@tool

def explore_entity_neighborhood(

    entities: list[str],

    depth: int = 2,

    top_k: int = 10,

    with_chunks: bool = True,

) -> dict:

    """以指定实体为中心，在知识图谱中检索其邻居子图与相关 chunk。



    Args:

        entities: 种子实体名称列表。

        depth: 子图展开深度，默认 2。

        top_k: 返回的 chunk / 关系数量上限。

        with_chunks: 是否同时返回关联 chunk。



    Returns:

        包含 nodes、edges 和 chunks 的子图结果字典。

    """

    try:

        print(

            f"[explore_entity_neighborhood] entities={entities!r} depth={depth} "

            f"top_k={top_k} with_chunks={with_chunks}",

            flush=True,

        )

        parsed = _get_rag_service_client().call(

            "subgraph_search",

            {"entities": entities, "depth": depth, "top_k": top_k, "with_chunks": with_chunks},

        )

        if not isinstance(parsed, dict):

            return _error_dict("unexpected subgraph_search result format")



        parsed = _json_safe(parsed)

        if parsed.get("chunks") is not None:

            parsed["chunks"] = [

                _format_chunk(c, c.get("score", 0.0))

                for c in parsed.get("chunks", [])

            ]

        return parsed

    except Exception as exc:  # noqa: BLE001

        return _error_dict(str(exc))





@tool

def aggregate_evidence_by_entities(

    entity_groups: list[list[str]],

    mode: str = "any",

    top_k: int = 10,

    min_cooccurrence: int = 1,

) -> list[dict]:

    """按实体组聚合证据，返回共现统计与代表性 chunk。



    Args:

        entity_groups: 实体组列表，每组为一个实体名称列表。

        mode: "any" 表示命中任意实体即可，"all" 表示需命中全部实体。

        top_k: 每组返回的 chunk 数量上限。

        min_cooccurrence: 最少共现实体数（仅对 "any" 模式生效）。



    Returns:

        每组对应的聚合结果列表。

    """

    try:

        print(

            f"[aggregate_evidence_by_entities] groups={entity_groups!r} mode={mode!r} "

            f"top_k={top_k} min_cooccurrence={min_cooccurrence}",

            flush=True,

        )

        parsed = _get_rag_service_client().call(

            "aggregate_evidence",

            {

                "entity_groups": entity_groups,

                "mode": mode,

                "top_k": top_k,

                "min_cooccurrence": min_cooccurrence,

            },

        )

        groups = parsed.get("groups", []) if isinstance(parsed, dict) else parsed

        for group in groups:

            group["chunks"] = [

                _format_chunk(c, c.get("score", 0.0))

                for c in group.get("chunks", [])

            ]

        return _json_safe(groups)

    except Exception as exc:  # noqa: BLE001

        return _error_evidence(str(exc))





# Backward-compatible aliases so existing callers keep working.

search_kg_rag_hybrid = search_literature_chunks

search_kg_rag_multi = search_literature_multiview

