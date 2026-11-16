"""Reranker interface and implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod

from kg_rag.types import RetrievalCandidate


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        raise NotImplementedError

    def rerank_with_scores(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[tuple[RetrievalCandidate, float]]:
        """Return candidates paired with their ranking scores.

        The default implementation delegates to :meth:`rerank` and assigns
        descending integer scores, which preserves ordering but loses real
        magnitudes. Concrete rerankers should override this to expose their
        actual scores.
        """
        ranked = self.rerank(query, candidates)
        return [(c, float(len(ranked) - i)) for i, c in enumerate(ranked)]


class RRFReranker(Reranker):
    def __init__(self, k: int = 60):
        self.k = k

    def rerank(self, query: str, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        return [c for c, _ in self.rerank_with_scores(query, candidates)]

    def rerank_with_scores(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[tuple[RetrievalCandidate, float]]:
        if not candidates:
            return []
        # 相同评分按稳定的 chunk_id 排序，避免数据库行顺序改变 RRF 的名次。
        vector_ranked = sorted(candidates, key=lambda c: (-(c.vector_score or 0.0), c.chunk_id))
        graph_ranked = sorted(candidates, key=lambda c: (-(c.graph_score or 0.0), c.chunk_id))
        ranks: dict[str, dict[str, int]] = {}
        for i, c in enumerate(vector_ranked):
            ranks.setdefault(c.chunk_id, {})["v"] = i + 1
        for i, c in enumerate(graph_ranked):
            ranks.setdefault(c.chunk_id, {})["g"] = i + 1
        scored = []
        for c in candidates:
            r = ranks.get(c.chunk_id, {})
            score = 1.0 / (self.k + r.get("v", 999)) + 1.0 / (self.k + r.get("g", 999))
            scored.append((c, score))
        scored.sort(key=lambda x: (-x[1], x[0].chunk_id))
        return scored
