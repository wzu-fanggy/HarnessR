"""Shared data types for kg_rag."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class SearchFilters:
    paper_ids: list[str] | None = None
    sections: list[str] | None = None


@dataclass
class HybridSearchRequest:
    query: str
    mode: Literal["vector", "graph", "hybrid"] = "hybrid"
    top_k: int = 10
    filters: SearchFilters = field(default_factory=SearchFilters)


@dataclass
class KgTriple:
    source: str
    relation: str
    target: str
    confidence_mean: Optional[float] = None
    direction_status: Optional[str] = None
    sources: Optional[list[str]] = None


@dataclass
class SourceAnchor:
    page_idx: Optional[int] = None
    paragraph_idx: Optional[int] = None
    bbox: Optional[list[float]] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    text_hash: str = ""


@dataclass
class EvidenceChunk:
    chunk_id: str
    text: str
    paper_id: str
    paper_title: str
    section: str
    score: float = 0.0
    entities: list[str] = field(default_factory=list)
    triples: list[KgTriple] = field(default_factory=list)
    source_path: str = ""
    source_anchor: SourceAnchor = field(default_factory=SourceAnchor)


@dataclass
class HybridSearchResult:
    query: str
    chunks: list[EvidenceChunk] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    ranked_by: Literal["rrf", "reranker"] = "rrf"
    total_candidates: int = 0


@dataclass
class RetrievalCandidate:
    chunk_id: str
    text: str
    paper_id: str
    paper_title: str
    section: str
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    entities: list[str] = field(default_factory=list)
    triples: list[KgTriple] = field(default_factory=list)
    source_path: str = ""
    source_anchor: SourceAnchor = field(default_factory=SourceAnchor)
