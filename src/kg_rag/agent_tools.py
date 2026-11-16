"""LangChain/LangGraph compatible tools for agent integration."""
from __future__ import annotations

import json
import threading
from neo4j import Driver, GraphDatabase

from kg_rag.config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from kg_rag.embedder import Bgem3Embedder
from kg_rag.entity_linker import FlashTextLinker
from kg_rag.retriever import HybridRetriever
from kg_rag.types import HybridSearchRequest, HybridSearchResult, SearchFilters


def _hybrid_result_to_json(result: HybridSearchResult) -> str:
    """Serialize a HybridSearchResult using the canonical chunk shape."""
    return json.dumps(
        {
            "query": result.query,
            "entities": result.entities,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text[:800],
                    "paper_title": c.paper_title,
                    "section": c.section,
                    "score": c.score,
                    "entities": c.entities,
                    "triples": [
                        {"s": t.source, "r": t.relation, "t": t.target}
                        for t in c.triples[:5]
                    ],
                    "page_idx": c.source_anchor.page_idx,
                    "paragraph_idx": c.source_anchor.paragraph_idx,
                    "bbox": c.source_anchor.bbox,
                    "char_start": c.source_anchor.char_start,
                    "char_end": c.source_anchor.char_end,
                    "text_hash": c.source_anchor.text_hash,
                }
                for c in result.chunks
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


class _LazyFlashTextLinker:
    """Delay KG entity-name loading until a retrieval path actually needs linking."""

    def __init__(self, driver: Driver) -> None:
        self.driver = driver
        self._delegate: FlashTextLinker | None = None
        self._lock = threading.RLock()

    def _get_delegate(self) -> FlashTextLinker:
        if self._delegate is not None:
            return self._delegate
        with self._lock:
            if self._delegate is None:
                from kg_rag.vector_store import Neo4jVectorStore

                names = Neo4jVectorStore(
                    self.driver,
                    database=NEO4J_DATABASE,
                ).load_entity_names()
                self._delegate = FlashTextLinker(names)
        return self._delegate

    def link(self, text: str):
        return self._get_delegate().link(text)


def make_tools(
    driver: Driver | None = None,
    embedder: Bgem3Embedder | None = None,
):
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    if embedder is None:
        embedder = Bgem3Embedder()
    linker = _LazyFlashTextLinker(driver)
    retriever = HybridRetriever(driver, embedder, linker)

    def hybrid_search(query: str, top_k: int = 10) -> str:
        result = retriever.search(HybridSearchRequest(query=query, top_k=top_k))
        return _hybrid_result_to_json(result)

    def hybrid_search_filtered(
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        sections: list[str] | None = None,
        paper_ids: list[str] | None = None,
        entities: list[str] | None = None,
        exclude_low_value: bool = True,
    ) -> str:
        request = HybridSearchRequest(
            query=query, mode=mode, top_k=top_k,
            filters=SearchFilters(paper_ids=paper_ids, sections=sections),
        )
        result = retriever.search_filtered(
            request, entity_names=entities, exclude_low_value=exclude_low_value
        )
        return _hybrid_result_to_json(result)

    def subgraph_search(
        entities: list[str],
        depth: int = 2,
        top_k: int = 10,
        with_chunks: bool = True,
    ) -> str:
        result = retriever.search_subgraph(entities, depth, top_k, with_chunks)
        return json.dumps(result, ensure_ascii=False)

    def path_search(
        from_entity: str,
        to_entity: str,
        max_depth: int = 4,
        top_k: int = 5,
    ) -> str:
        result = retriever.search_path(from_entity, to_entity, max_depth, top_k)
        return json.dumps(result, ensure_ascii=False)

    def paper_search(
        paper_id: str,
        query: str | None = None,
        sections: list[str] | None = None,
        top_k: int = 10,
    ) -> str:
        result = retriever.search_paper(paper_id, query, sections, top_k)
        return json.dumps(result, ensure_ascii=False)

    def resolve_papers(query: str, top_k: int = 5) -> str:
        result = retriever.resolve_papers(query, top_k)
        return json.dumps(result, ensure_ascii=False)

    def expand_entities(
        text: str,
        top_k: int = 10,
        fuzzy: bool = False,
    ) -> str:
        result = retriever.expand_entities(text, top_k, fuzzy)
        return json.dumps(result, ensure_ascii=False)

    def aggregate_evidence(
        entity_groups: list[list[str]],
        mode: str = "any",
        top_k: int = 10,
        min_cooccurrence: int = 1,
    ) -> str:
        result = retriever.aggregate_by_entities(
            entity_groups, mode, top_k, min_cooccurrence
        )
        return json.dumps(result, ensure_ascii=False)

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "hybrid_search",
                "description": "基于向量+KG混合检索，返回相关论文chunk和三元组证据",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hybrid_search_filtered",
                "description": "带显式过滤条件的混合检索，支持按章节、论文ID、实体集合过滤",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                        "mode": {"type": "string", "default": "hybrid"},
                        "sections": {"type": "array", "items": {"type": "string"}},
                        "paper_ids": {"type": "array", "items": {"type": "string"}},
                        "entities": {"type": "array", "items": {"type": "string"}},
                        "exclude_low_value": {"type": "boolean", "default": True},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "subgraph_search",
                "description": "以指定实体为中心，在KG中检索其邻居子图",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {"type": "array", "items": {"type": "string"}},
                        "depth": {"type": "integer", "default": 2},
                        "top_k": {"type": "integer", "default": 10},
                        "with_chunks": {"type": "boolean", "default": True},
                    },
                    "required": ["entities"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "path_search",
                "description": "在KG中搜索两个实体之间的路径",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_entity": {"type": "string"},
                        "to_entity": {"type": "string"},
                        "max_depth": {"type": "integer", "default": 4},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["from_entity", "to_entity"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "paper_search",
                "description": "在指定论文内进行检索，可限定章节",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string"},
                        "query": {"type": "string"},
                        "sections": {"type": "array", "items": {"type": "string"}},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["paper_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "resolve_papers",
                "description": "按作者、论文题名或 DOI 的字面信息解析论文 ID，不使用语义向量检索",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "expand_entities",
                "description": "从输入文本中识别并扩展KG实体",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                        "fuzzy": {"type": "boolean", "default": False},
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "aggregate_evidence",
                "description": "按实体组聚合证据，返回共现统计与代表性chunk",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_groups": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "string"}},
                        },
                        "mode": {"type": "string", "default": "any"},
                        "top_k": {"type": "integer", "default": 10},
                        "min_cooccurrence": {"type": "integer", "default": 1},
                    },
                    "required": ["entity_groups"],
                },
            },
        },
    ]

    callables = {
        "hybrid_search": hybrid_search,
        "hybrid_search_filtered": hybrid_search_filtered,
        "subgraph_search": subgraph_search,
        "path_search": path_search,
        "paper_search": paper_search,
        "resolve_papers": resolve_papers,
        "expand_entities": expand_entities,
        "aggregate_evidence": aggregate_evidence,
    }

    return schemas, callables, driver
