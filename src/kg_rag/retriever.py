"""Hybrid retriever: vector + graph + reranking."""
from __future__ import annotations

import re

from neo4j import Driver

from kg_rag.config import NEO4J_DATABASE, VECTOR_INDEX_NAME
from kg_rag.embedder import Bgem3Embedder
from kg_rag.entity_linker import EntityLinker
from kg_rag.query_variants import generate_variants
from kg_rag.reranker import Reranker, RRFReranker
from kg_rag.types import (
    EvidenceChunk,
    HybridSearchRequest,
    HybridSearchResult,
    KgTriple,
    RetrievalCandidate,
    SearchFilters,
    SourceAnchor,
)

try:
    from neo4j_graphrag.exceptions import (
        RetrieverInitializationError,
        SearchValidationError,
    )
    from neo4j_graphrag.types import RetrieverResultItem

    _RETRIEVER_ERRORS = (RetrieverInitializationError, SearchValidationError)
except ImportError:
    _RETRIEVER_ERRORS = ()

_RETRIEVAL_QUERY = """
WITH node AS chunk, score
WHERE coalesce(chunk.chunk_type, 'child') = 'child'
OPTIONAL MATCH (chunk)<-[:MENTIONED_IN]-(e:Entity)
WITH chunk, score, e ORDER BY e.name, elementId(e)
WITH chunk, score, [e IN collect(DISTINCT e) WHERE e IS NOT NULL][0..20] AS entities
CALL (entities) {
  UNWIND entities AS e
  OPTIONAL MATCH (e)-[rel]-(neighbor:Entity)
  WITH e, rel ORDER BY type(rel), startNode(rel).name, endNode(rel).name, elementId(rel)
  WITH e, collect(rel)[0..3] AS rels
  UNWIND rels AS rel
  WITH rel WHERE rel IS NOT NULL
  RETURN collect(DISTINCT {
    source: startNode(rel).name,
    relation: type(rel),
    target: endNode(rel).name,
    confidence_mean: rel.confidence_mean,
    direction_status: rel.direction_status
  }) AS triples
}
RETURN chunk.chunk_id AS chunk_id,
       chunk.text AS text,
       chunk.paper_id AS paper_id,
       chunk.paper_title AS paper_title,
       chunk.section AS section,
       chunk.source_path AS source_path,
       chunk.page_idx AS page_idx,
       chunk.paragraph_idx AS paragraph_idx,
       chunk.bbox AS bbox,
       chunk.char_start AS char_start,
       chunk.char_end AS char_end,
       chunk.text_hash AS text_hash,
       score,
       [e IN entities | e.name] AS entities,
       triples
"""


_PROPERTY_KEYWORDS = {
    "melting point", "熔点", "viscosity", "粘度", "dielectric constant", "介电常数",
    "desolvation energy", "去溶剂化能", "conductivity", "电导率", "capacity", "容量",
}


def _looks_like_numerical_table(text: str) -> bool:
    if "<table>" in text.lower():
        return True
    # digit density >= 3%
    digit_chars = sum(1 for ch in text if ch.isdigit())
    if len(text) > 0 and digit_chars / len(text) >= 0.03:
        return True
    return False


def paper_lookup_quantities(query: str) -> list[dict[str, str]]:
    """显式数值与单位一起匹配正文，避免把文件名的数字误当实验电流等读数。"""
    quantities = []
    for match in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(mAh|mA|Ah|mV|V|A)\b", query):
        number, unit = match.groups()
        term = f"{number} {unit.lower()}"
        if any(item['term'] == term for item in quantities):
            continue
        gap = r"(?:\s|[$_{}\\]|mathrm|text|operatorname)*"
        pattern = r"(?is).*\b" + re.escape(number) + gap + re.escape(unit) + r"\b.*"
        quantities.append({"term": term, "pattern": pattern, "query_text": match.group()})
    # 负温度或明确的摄氏度符号不与倍率 C 混用，保留正负号和完整数值。
    for match in re.finditer(r'(?<![\w.])([−-]?\d+(?:\.\d+)?)\s*(°\s*)?C\b', query):
        number, degree = match.groups()
        if not degree and not number.startswith(('-', '−')):
            continue
        number = number.replace('−', '-')
        term = f'{number} °c'
        if any(item['term'] == term for item in quantities):
            continue
        sign = '[-−]' if number.startswith('-') else '(?<![-−])'
        gap = r'(?:\s|[$_{}\\^°]|mathrm|text|operatorname|circ)*'
        pattern = r'(?is).*(?<![\w.])' + sign + re.escape(number.lstrip('-')) + gap + r'C\b.*'
        quantities.append({'term': term, 'pattern': pattern, 'query_text': match.group()})
    return quantities


def paper_lookup_terms(query: str) -> list[str]:
    """提取用于书目解析的有界字面检索词。"""
    stopwords = {
        "about", "after", "and", "battery", "electrolyte", "for", "from",
        "ion", "paper", "research", "sodium", "study", "the", "with",
    }
    quantities = paper_lookup_quantities(query)
    terms: list[str] = [item['term'] for item in quantities]
    for item in quantities:
        query = query.replace(item['query_text'], ' ')
    pattern = r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9._/-]*"
    for token in re.findall(pattern, query):
        normalized = token.strip("._/-").casefold()
        if len(normalized) < 3 or normalized in stopwords:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms[:8]


def _lookup_evidence_excerpt(text: str, quantities: list[dict[str, str]]) -> str:
    """展示命中实验数值附近的连续原文，不把不同段落拼成同次实验。"""
    hits = []
    for quantity in quantities:
        body = quantity['pattern'].removeprefix('(?is).*').removesuffix('.*')
        match = re.search(body, text, re.I | re.S)
        if match:
            hits.append(match.start())
    start = max(0, min(hits) - 160) if hits else 0
    return text[start:start + 700]


def resolve_papers_by_metadata(
    driver: Driver,
    query: str,
    top_k: int = 5,
    *,
    database: str = NEO4J_DATABASE,
) -> dict:
    """用书目信息或正文的具体实验标识定位论文，不初始化 embedding 或实体链接器。"""
    terms = paper_lookup_terms(query)
    quantities = paper_lookup_quantities(query)
    quantity_terms = [item['term'] for item in quantities]
    identifier_terms = [term for term in terms if term not in quantity_terms
                        and re.search(r'[a-z]', term) and re.search(r'\d', term)]
    if not terms:
        return {"query": query, "terms": [], "papers": []}

    cypher = """
    MATCH (paper:Paper)-[:HAS_CHUNK]->(ch:Chunk)
    WITH paper, ch,
         [term IN $terms
          WHERE (NOT term IN $quantity_terms AND (
                  toLower(coalesce(paper.paper_id, '')) CONTAINS term
             OR toLower(coalesce(paper.title, '')) CONTAINS term
             OR toLower(coalesce(ch.text, '')) CONTAINS term))
             OR any(q IN $quantities WHERE q.term = term
                    AND coalesce(ch.text, '') =~ q.pattern)] AS matched_terms,
         [term IN $terms
          WHERE NOT term IN $quantity_terms AND (
                toLower(coalesce(paper.paper_id, '')) CONTAINS term
             OR toLower(coalesce(paper.title, '')) CONTAINS term)] AS paper_terms
    WHERE size(matched_terms) > 0
    WITH paper, ch, matched_terms,
         10000 * size([q IN $quantities WHERE coalesce(ch.text, '') =~ q.pattern])
         + 100 * size(paper_terms)
         + 10 * size(matched_terms)
         + 1000 * size([term IN matched_terms WHERE term IN $identifier_terms
                                                 OR term IN $quantity_terms])
         + CASE
             WHEN ch.chunk_id CONTAINS '::sec0::'
               OR ch.chunk_id CONTAINS '::sec1::' THEN 20
             ELSE 0
           END
         - CASE
             WHEN toLower(coalesce(ch.section, '')) CONTAINS 'reference'
               OR toLower(coalesce(ch.section, '')) CONTAINS '参考文献'
               OR toLower(coalesce(ch.section, '')) CONTAINS 'bibliograph' THEN 20
             ELSE 0
           END AS match_score
    ORDER BY match_score DESC, coalesce(ch.char_len, size(ch.text)) ASC
    WITH paper,
         max(match_score) AS score,
         reduce(all_terms = [], terms IN collect(matched_terms) |
                all_terms + [term IN terms WHERE NOT term IN all_terms]) AS matched_terms,
         collect({
             chunk_id: ch.chunk_id,
             section: ch.section,
             text: coalesce(ch.text, ''),
             matched_terms: matched_terms
         })[0] AS metadata_match
    RETURN paper.paper_id AS paper_id,
           paper.title AS paper_title,
           paper.source_path AS source_path,
           score,
           matched_terms,
           metadata_match
    ORDER BY score DESC, paper_id
    LIMIT $top_k
    """
    papers: list[dict] = []
    try:
        with driver.session(database=database) as session:
            result = session.run(
                cypher,
                terms=terms,
                quantities=quantities,
                quantity_terms=quantity_terms,
                identifier_terms=identifier_terms,
                top_k=max(1, min(top_k, 10)),
            )
            for record in result:
                metadata_match = dict(record.get('metadata_match') or {})
                metadata_match['text'] = _lookup_evidence_excerpt(
                    str(metadata_match.get('text') or ''), quantities)
                papers.append(
                    {
                        "paper_id": record.get("paper_id", ""),
                        "paper_title": record.get("paper_title", ""),
                        "source_path": record.get("source_path", ""),
                        "score": int(record.get("score", 0) or 0),
                        "matched_terms": list(record.get("matched_terms") or []),
                        "metadata_match": metadata_match,
                    }
                )
    except Exception:
        papers = []
    return {"query": query, "terms": terms, "papers": papers}


class HybridRetriever:
    _LOW_VALUE_SECTIONS = {
        "references",
        "acknowledgements",
        "keywords",
        "article info",
        "front matter",
        "accepted manuscript",
    }
    _HIGH_VALUE_SECTIONS = {
        "results",
        "discussion",
        "experimental",
        "methods",
        "electrochemical performance",
        "electrolyte",
    }

    def __init__(
        self,
        driver: Driver,
        embedder: Bgem3Embedder,
        linker: EntityLinker,
        reranker: Reranker | None = None,
        database: str = NEO4J_DATABASE,
        index_name: str = VECTOR_INDEX_NAME,
    ):
        self.driver = driver
        self.embedder = embedder
        self.linker = linker
        self.reranker = reranker or RRFReranker()
        self.database = database
        self.index_name = index_name

    @classmethod
    def _is_low_value_section(cls, section: str) -> bool:
        s = section.lower()
        return any(low in s for low in cls._LOW_VALUE_SECTIONS)

    @classmethod
    def _is_high_value_section(cls, section: str) -> bool:
        s = section.lower()
        return any(high in s for high in cls._HIGH_VALUE_SECTIONS)

    def _apply_section_quality(self, candidates: list[RetrievalCandidate], query: str = "") -> list[RetrievalCandidate]:
        """根据 section 质量调整用于排序的分数：低价值 section 降权，高价值 section 加分。"""
        query_lower = query.lower()
        is_property_query = any(kw in query_lower for kw in _PROPERTY_KEYWORDS)
        for c in candidates:
            if self._is_low_value_section(c.section):
                if c.vector_score is not None:
                    c.vector_score = max(0.0, c.vector_score - 0.25)
                if c.graph_score is not None:
                    c.graph_score = max(0.0, c.graph_score - 3.0)
            elif self._is_high_value_section(c.section):
                if c.vector_score is not None:
                    c.vector_score = min(1.0, c.vector_score + 0.05)
                if c.graph_score is not None:
                    c.graph_score += 0.5
            # boost table/numerical chunks for property queries
            if is_property_query and c.vector_score is not None and _looks_like_numerical_table(c.text):
                c.vector_score = min(1.0, c.vector_score + 0.04)
        return candidates

    def _filter_low_value_sections(self, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        """默认过滤掉 References、Acknowledgements、Keywords 等低价值章节。"""
        return [c for c in candidates if not self._is_low_value_section(c.section)]

    def _vector_search(self, query_embedding: list[float], top_k: int) -> list[RetrievalCandidate]:
        try:
            from neo4j_graphrag.retrievers import VectorCypherRetriever

            def _vector_result_formatter(record):
                return RetrieverResultItem(content=record.get("text"), metadata=record.data())

            retriever = VectorCypherRetriever(
                driver=self.driver,
                index_name=self.index_name,
                embedder=self.embedder,
                retrieval_query=_RETRIEVAL_QUERY,
                neo4j_database=self.database,
                result_formatter=_vector_result_formatter,
            )
            result = retriever.search(query_text="", query_vector=query_embedding, top_k=top_k)
            return [self._record_to_candidate(item.metadata, "vector") for item in result.items]
        except (ImportError,) + _RETRIEVER_ERRORS:
            records = self._vector_search_fallback(query_embedding, top_k)
            return [self._record_to_candidate(rec, "vector") for rec in records]

    def _vector_search_fallback(self, query_embedding: list[float], top_k: int) -> list[dict]:
        query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node AS chunk, score
        MATCH (chunk)<-[:MENTIONED_IN]-(e:Entity)
        WITH chunk, score, e ORDER BY e.name, elementId(e)
        WITH chunk, score, collect(DISTINCT e)[0..20] AS entities
        UNWIND entities AS e
        CALL (e) {
          MATCH (e)-[rel]-(neighbor:Entity)
          RETURN rel ORDER BY type(rel), startNode(rel).name, endNode(rel).name, elementId(rel)
          LIMIT 3
        }
        WITH chunk, score, e, rel
        RETURN chunk.chunk_id AS chunk_id,
               chunk.text AS text,
               chunk.paper_id AS paper_id,
               chunk.paper_title AS paper_title,
               chunk.section AS section,
               chunk.source_path AS source_path,
               chunk.page_idx AS page_idx,
               chunk.paragraph_idx AS paragraph_idx,
               chunk.bbox AS bbox,
               chunk.char_start AS char_start,
               chunk.char_end AS char_end,
               chunk.text_hash AS text_hash,
               score,
               collect(DISTINCT e.name) AS entities,
               collect(DISTINCT {
                 source: startNode(rel).name,
                 relation: type(rel),
                 target: endNode(rel).name,
                 confidence_mean: rel.confidence_mean,
                 direction_status: rel.direction_status
               }) AS triples
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                index_name=self.index_name,
                top_k=top_k,
                embedding=query_embedding,
            )
            return [record.data() for record in result]

    def _graph_search(self, entities: list[str], top_k: int) -> list[RetrievalCandidate]:
        if not entities:
            return []
        query = """
        UNWIND $entities AS entity_name
        MATCH (e:Entity {name: entity_name})-[:MENTIONED_IN]->(ch:Chunk)
        WITH ch, count(DISTINCT e) AS hit_count
        ORDER BY hit_count DESC, ch.chunk_id
        LIMIT $top_k
        WITH ch, hit_count
        MATCH (e:Entity)-[:MENTIONED_IN]->(ch)
        WITH ch, hit_count, e ORDER BY e.name, elementId(e)
        WITH ch, hit_count, collect(DISTINCT e)[0..20] AS entities
        UNWIND entities AS e
        CALL (e) {
          MATCH (e)-[rel]-(neighbor:Entity)
          RETURN rel ORDER BY type(rel), startNode(rel).name, endNode(rel).name, elementId(rel)
          LIMIT 3
        }
        WITH ch, hit_count, e, rel
        RETURN ch.chunk_id AS chunk_id,
               ch.text AS text,
               ch.paper_id AS paper_id,
               ch.paper_title AS paper_title,
               ch.section AS section,
               ch.source_path AS source_path,
               ch.page_idx AS page_idx,
               ch.paragraph_idx AS paragraph_idx,
               ch.bbox AS bbox,
               ch.char_start AS char_start,
               ch.char_end AS char_end,
               ch.text_hash AS text_hash,
               hit_count,
               collect(DISTINCT e.name) AS entities,
               collect(DISTINCT {
                 source: startNode(rel).name,
                 relation: type(rel),
                 target: endNode(rel).name,
                 confidence_mean: rel.confidence_mean,
                 direction_status: rel.direction_status
               }) AS triples
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, entities=entities, top_k=top_k)
            return [self._record_to_candidate(record.data(), "graph") for record in result]

    def _record_to_candidate(self, record: dict, source: str) -> RetrievalCandidate:
        score = record.get("score", 0.0) if source == "vector" else float(record.get("hit_count", 0))
        triples = [
            KgTriple(
                source=t.get("source", ""),
                relation=t.get("relation", ""),
                target=t.get("target", ""),
                confidence_mean=t.get("confidence_mean"),
                direction_status=t.get("direction_status"),
            )
            for t in (record.get("triples") or [])
            if t and t.get("source") and t.get("target")
        ]
        return RetrievalCandidate(
            chunk_id=record.get("chunk_id", ""),
            text=record.get("text", ""),
            paper_id=record.get("paper_id", ""),
            paper_title=record.get("paper_title", ""),
            section=record.get("section", ""),
            vector_score=score if source == "vector" else None,
            graph_score=score if source == "graph" else None,
            entities=sorted(set(record.get("entities") or [])),
            triples=sorted(triples, key=lambda t: (t.source, t.relation, t.target,
                          str(t.confidence_mean), str(t.direction_status))),
            source_path=record.get("source_path", ""),
            source_anchor=SourceAnchor(
                page_idx=record.get("page_idx"),
                paragraph_idx=record.get("paragraph_idx"),
                bbox=record.get("bbox"),
                char_start=record.get("char_start"),
                char_end=record.get("char_end"),
                text_hash=record.get("text_hash", ""),
            ),
        )

    def _merge_candidates(self, vector_cands: list[RetrievalCandidate], graph_cands: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        merged: dict[str, RetrievalCandidate] = {}
        for c in vector_cands:
            merged[c.chunk_id] = c
        for c in graph_cands:
            if c.chunk_id in merged:
                existing = merged[c.chunk_id]
                existing.graph_score = c.graph_score
                existing.entities = sorted(set(existing.entities + c.entities))
                existing.triples = existing.triples + [t for t in c.triples if t not in existing.triples]
            else:
                merged[c.chunk_id] = c
        return list(merged.values())

    def _apply_filters(
        self,
        candidates: list[RetrievalCandidate],
        filters: SearchFilters,
    ) -> list[RetrievalCandidate]:
        allowed_papers = set(filters.paper_ids) if filters.paper_ids else None
        allowed_sections = [s.lower() for s in filters.sections] if filters.sections else None
        result = candidates
        if allowed_papers is not None:
            result = [c for c in result if c.paper_id in allowed_papers]
        if allowed_sections is not None:
            result = [
                c
                for c in result
                if any(keyword in c.section.lower() for keyword in allowed_sections)
            ]
        return result

    def _vector_search_variants(
        self,
        query: str,
        fetch_k: int,
    ) -> list[RetrievalCandidate]:
        """对查询的多个格式变体分别做向量检索，按 chunk_id 去重保留最高分。"""
        all_cands: list[RetrievalCandidate] = []
        for variant in generate_variants(query):
            emb = self.embedder.embed_query(variant)
            all_cands.extend(self._vector_search(emb, fetch_k))

        best: dict[str, RetrievalCandidate] = {}
        for c in all_cands:
            existing = best.get(c.chunk_id)
            if existing is None or (c.vector_score or 0.0) > (existing.vector_score or 0.0):
                best[c.chunk_id] = c
        return list(best.values())

    def _quantity_search(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        """为原问题中的明确数值单位补充字面召回，不推测实验条件或配方。"""
        quantities = paper_lookup_quantities(query)
        if not quantities:
            return []
        cypher = """
        MATCH (ch:Chunk)
        WHERE coalesce(ch.chunk_type, 'child') = 'child'
          AND any(q IN $quantities WHERE coalesce(ch.text, '') =~ q.pattern)
        WITH ch, size([q IN $quantities WHERE coalesce(ch.text, '') =~ q.pattern]) AS exact_hits,
             size([t IN $terms WHERE toLower(coalesce(ch.text, '')) CONTAINS t]) AS term_hits
        RETURN ch.chunk_id AS chunk_id, ch.text AS text, ch.paper_id AS paper_id,
               ch.paper_title AS paper_title, ch.section AS section,
               ch.source_path AS source_path, ch.page_idx AS page_idx,
               ch.bbox AS bbox, ch.char_start AS char_start, ch.char_end AS char_end,
               ch.text_hash AS text_hash
        ORDER BY exact_hits DESC, term_hits DESC, chunk_id
        LIMIT $top_k
        """
        with self.driver.session(database=self.database) as session:
            records = session.run(cypher, quantities=quantities,
                                 terms=paper_lookup_terms(query), top_k=top_k)
            candidates = [self._record_to_candidate(record.data(), 'vector') for record in records]
        for candidate in candidates:
            candidate.vector_score = None
        return candidates

    def _fuse_quantity_results(self, reranked, request: HybridSearchRequest, *, exclude_low_value=True):
        literal = self._quantity_search(request.query, request.top_k * 5) if request.mode == 'hybrid' else []
        if exclude_low_value:
            literal = self._filter_low_value_sections(literal)
        literal = self._apply_filters(literal, request.filters)
        if not literal:
            return reranked
        # 将语义检索列表与字面列表按同一 RRF 常数融合，不改写向量/图谱分数。
        rank_constant = getattr(self.reranker, 'k', 60)
        fused = {c.chunk_id: (c, 1.0 / (rank_constant + rank))
                 for rank, (c, _) in enumerate(reranked, start=1)}
        for rank, c in enumerate(literal, start=1):
            prior, score = fused.get(c.chunk_id, (c, 0.0))
            fused[c.chunk_id] = (prior, score + 1.0 / (rank_constant + rank))
        return sorted(fused.values(), key=lambda item: (-item[1], item[0].chunk_id))

    def search(self, request: HybridSearchRequest) -> HybridSearchResult:
        """Run hybrid vector/graph search.

        By default, this method filters out low-value sections (e.g. References,
        Acknowledgements, Keywords) and applies section-quality scoring that
        penalizes low-value sections and boosts high-value sections such as
        Results, Experimental, and Methods before reranking.
        """
        query_entities = self.linker.link(request.query)

        vector_cands: list[RetrievalCandidate] = []
        graph_cands: list[RetrievalCandidate] = []

        # 放大 fetch_k，因为过滤低价值 section 后会损失一部分候选。
        fetch_k = request.top_k * 5
        if request.mode in ("vector", "hybrid"):
            vector_cands = self._vector_search_variants(request.query, fetch_k)
        if request.mode in ("graph", "hybrid"):
            graph_cands = self._graph_search(query_entities, fetch_k)

        merged = self._merge_candidates(vector_cands, graph_cands)
        # 默认过滤掉 References / Acknowledgements / Keywords 等低价值章节
        merged = self._filter_low_value_sections(merged)
        # section 质量加权（Results / Experimental / Methods 等加分）
        merged = self._apply_section_quality(merged, request.query)
        filtered = self._apply_filters(merged, request.filters)
        reranked = self.reranker.rerank_with_scores(request.query, filtered)
        reranked = self._fuse_quantity_results(reranked, request)
        top = reranked[: request.top_k]

        evidence = [
            EvidenceChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                paper_id=c.paper_id,
                paper_title=c.paper_title,
                section=c.section,
                score=score,
                entities=c.entities,
                triples=c.triples,
                source_path=c.source_path,
                source_anchor=c.source_anchor,
            )
            for c, score in top
        ]
        return HybridSearchResult(
            query=request.query,
            chunks=evidence,
            entities=query_entities,
            ranked_by="rrf",
            total_candidates=len(reranked),
        )

    def search_filtered(
        self,
        request: HybridSearchRequest,
        entity_names: list[str] | None = None,
        exclude_low_value: bool = True,
    ) -> HybridSearchResult:
        """Hybrid search with explicit filters, entity boost, and low-value toggle."""
        query_entities = self.linker.link(request.query)

        vector_cands: list[RetrievalCandidate] = []
        graph_cands: list[RetrievalCandidate] = []

        fetch_k = request.top_k * 5
        if request.mode in ("vector", "hybrid"):
            vector_cands = self._vector_search_variants(request.query, fetch_k)
        if request.mode in ("graph", "hybrid"):
            graph_cands = self._graph_search(query_entities, fetch_k)

        merged = self._merge_candidates(vector_cands, graph_cands)
        if exclude_low_value:
            merged = self._filter_low_value_sections(merged)
        merged = self._apply_section_quality(merged, request.query)
        filtered = self._apply_filters(merged, request.filters)

        if entity_names:
            allowed = {e.strip().lower() for e in entity_names if e.strip()}
            for c in filtered:
                overlap = {e.lower() for e in c.entities} & allowed
                if overlap:
                    boost = min(len(overlap) * 1.0, 3.0)
                    c.graph_score = (c.graph_score or 0.0) + boost

        reranked = self.reranker.rerank_with_scores(request.query, filtered)
        reranked = self._fuse_quantity_results(reranked, request, exclude_low_value=exclude_low_value)
        top = reranked[: request.top_k]

        evidence = [
            EvidenceChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                paper_id=c.paper_id,
                paper_title=c.paper_title,
                section=c.section,
                score=score,
                entities=c.entities,
                triples=c.triples,
                source_path=c.source_path,
                source_anchor=c.source_anchor,
            )
            for c, score in top
        ]
        return HybridSearchResult(
            query=request.query,
            chunks=evidence,
            entities=query_entities,
            ranked_by="rrf",
            total_candidates=len(reranked),
        )

    def search_subgraph(
        self,
        entities: list[str],
        depth: int,
        top_k: int,
        with_chunks: bool,
    ) -> dict:
        """Retrieve a centered neighborhood subgraph for the given entities."""
        if not entities:
            return {"entities": [], "nodes": [], "edges": [], "chunks": [] if with_chunks else None}

        max_rels_per_entity = 20
        max_total_edges = 100
        depth = max(1, min(depth, 4))

        query = """
        UNWIND $entities AS entity_name
        MATCH (seed:Entity {name: entity_name})
        WITH collect(DISTINCT seed) AS seeds
        UNWIND seeds AS seed
        CALL (seed) {
          WITH seed
          MATCH path = (seed)-[rel*1..$depth]-(neighbor:Entity)
          UNWIND rel AS r
          RETURN {
            source: startNode(r).name,
            relation: type(r),
            target: endNode(r).name,
            confidence_mean: r.confidence_mean,
            direction_status: r.direction_status
          } AS edge
          LIMIT $max_rels_per_entity
        }
        WITH seeds, collect(edge) AS edges
        RETURN seeds, edges
        """
        nodes: list[dict] = []
        edges: list[dict] = []
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    entities=list(entities),
                    depth=depth,
                    max_rels_per_entity=max_rels_per_entity,
                )
                record = result.single()
                if record:
                    seed_nodes = record.get("seeds") or []
                    raw_edges = record.get("edges") or []
                    seen_edges = set()
                    for edge in raw_edges[:max_total_edges]:
                        if not edge:
                            continue
                        key = (edge.get("source"), edge.get("relation"), edge.get("target"))
                        if key in seen_edges:
                            continue
                        seen_edges.add(key)
                        edges.append(
                            {
                                "source": edge.get("source"),
                                "relation": edge.get("relation"),
                                "target": edge.get("target"),
                                "confidence_mean": edge.get("confidence_mean"),
                                "direction_status": edge.get("direction_status"),
                            }
                        )
                    seed_names = {n.get("name") for n in seed_nodes if n}
                    node_names = set(seed_names)
                    for e in edges:
                        node_names.add(e.get("source"))
                        node_names.add(e.get("target"))
                    nodes = [
                        {"name": name, "is_seed": name in seed_names}
                        for name in sorted(node_names)
                        if name
                    ]
        except Exception:
            # Return empty structure on driver errors (e.g. mocked or missing DB).
            pass

        chunks: list[dict] = []
        if with_chunks and nodes:
            try:
                chunk_query = """
                UNWIND $entities AS entity_name
                MATCH (e:Entity {name: entity_name})-[:MENTIONED_IN]->(ch:Chunk)
                RETURN ch.chunk_id AS chunk_id,
                       ch.text AS text,
                       ch.paper_id AS paper_id,
                       ch.paper_title AS paper_title,
                       ch.section AS section,
                       ch.chunk_index AS chunk_index
                ORDER BY ch.chunk_index
                LIMIT $chunk_limit
                """
                with self.driver.session(database=self.database) as session:
                    result = session.run(
                        chunk_query,
                        entities=list(entities),
                        chunk_limit=top_k * 3,
                    )
                    chunks = [self._chunk_record_to_dict(rec) for rec in result]
            except Exception:
                pass

        return {
            "entities": list(entities),
            "nodes": nodes,
            "edges": edges,
            "chunks": chunks if with_chunks else None,
        }

    def search_path(
        self,
        from_entity: str,
        to_entity: str,
        max_depth: int,
        top_k: int,
    ) -> dict:
        """Find paths between two entities in the knowledge graph."""
        max_depth = max(1, min(max_depth, 6))
        query = """
        MATCH path = (a:Entity {name: $from_entity})-[rel*1..$max_depth]-(b:Entity {name: $to_entity})
        WITH path, length(path) AS d
        ORDER BY d ASC
        LIMIT $top_k
        RETURN [n IN nodes(path) | n.name] AS node_names,
               [r IN rels(path) | {
                 source: startNode(r).name,
                 relation: type(r),
                 target: endNode(r).name,
                 confidence_mean: r.confidence_mean,
                 direction_status: r.direction_status
               }] AS edges,
               d AS depth
        """
        paths: list[dict] = []
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    from_entity=from_entity,
                    to_entity=to_entity,
                    max_depth=max_depth,
                    top_k=top_k,
                )
                for record in result:
                    paths.append(
                        {
                            "nodes": record.get("node_names") or [],
                            "edges": record.get("edges") or [],
                            "depth": record.get("depth", 0),
                        }
                    )
        except Exception:
            pass

        return {
            "from_entity": from_entity,
            "to_entity": to_entity,
            "max_depth": max_depth,
            "top_k": top_k,
            "paths": paths,
        }

    def search_paper(
        self,
        paper_id: str,
        query: str | None,
        sections: list[str] | None,
        top_k: int,
    ) -> dict:
        """Search within a single paper, optionally by vector query and section."""
        query_text = query.strip() if query else None
        allowed_sections = set(sections) if sections else None
        chunks: list[dict] = []

        try:
            if query_text and self.embedder is not None:
                embedding = self.embedder.embed_query(query_text)
                vector_query = """
                MATCH (paper:Paper {paper_id: $paper_id})-[:HAS_CHUNK]->(ch:Chunk)
                WHERE ch.embedding IS NOT NULL
                  AND (
                    $sections IS NULL
                    OR any(section IN $sections
                           WHERE toLower(coalesce(ch.section, '')) CONTAINS section)
                  )
                WITH ch, vector.similarity.cosine(ch.embedding, $embedding) AS score
                RETURN ch, score
                ORDER BY score DESC
                LIMIT $top_k
                """
                with self.driver.session(database=self.database) as session:
                    result = session.run(
                        vector_query,
                        embedding=embedding,
                        paper_id=paper_id,
                        sections=sorted(s.lower() for s in allowed_sections)
                        if allowed_sections
                        else None,
                        top_k=top_k,
                    )
                    for record in result:
                        chunk = self._chunk_record_to_dict(record.get("ch"), score=record.get("score"))
                        chunks.append(chunk)
                # 部分历史导入缺少 Paper-[:HAS_CHUNK] 边，但 Chunk 自身保留
                # paper_id。关系路径为空时按该字段严格回退，绝不扩大到其它论文。
                if not chunks:
                    fallback_query = vector_query.replace(
                        "MATCH (paper:Paper {paper_id: $paper_id})-[:HAS_CHUNK]->(ch:Chunk)",
                        "MATCH (ch:Chunk {paper_id: $paper_id})",
                    )
                    with self.driver.session(database=self.database) as session:
                        result = session.run(
                            fallback_query,
                            embedding=embedding,
                            paper_id=paper_id,
                            sections=sorted(s.lower() for s in allowed_sections)
                            if allowed_sections
                            else None,
                            top_k=top_k,
                        )
                        for record in result:
                            chunk = self._chunk_record_to_dict(record.get("ch"), score=record.get("score"))
                            chunks.append(chunk)
            else:
                paper_query = """
                MATCH (paper:Paper {paper_id: $paper_id})-[:HAS_CHUNK]->(ch:Chunk)
                WHERE (
                  $sections IS NULL
                  OR any(section IN $sections
                         WHERE toLower(coalesce(ch.section, '')) CONTAINS section)
                )
                RETURN ch
                ORDER BY ch.section_index, ch.chunk_index
                LIMIT $top_k
                """
                with self.driver.session(database=self.database) as session:
                    result = session.run(
                        paper_query,
                        paper_id=paper_id,
                        sections=sorted(s.lower() for s in allowed_sections)
                        if allowed_sections
                        else None,
                        top_k=top_k,
                    )
                    for record in result:
                        chunk = self._chunk_record_to_dict(record.get("ch"))
                        chunks.append(chunk)
                if not chunks:
                    fallback_query = paper_query.replace(
                        "MATCH (paper:Paper {paper_id: $paper_id})-[:HAS_CHUNK]->(ch:Chunk)",
                        "MATCH (ch:Chunk {paper_id: $paper_id})",
                    )
                    with self.driver.session(database=self.database) as session:
                        result = session.run(
                            fallback_query,
                            paper_id=paper_id,
                            sections=sorted(s.lower() for s in allowed_sections)
                            if allowed_sections
                            else None,
                            top_k=top_k,
                        )
                        for record in result:
                            chunk = self._chunk_record_to_dict(record.get("ch"))
                            chunks.append(chunk)
        except ImportError:
            # 本地 embedding 不可用时无法执行带 query 的论文内排序；保持空结果，
            # 由上层明确视为检索不可用，而不是跨论文回退。
            return {
                "paper_id": paper_id,
                "query": query_text,
                "sections": list(allowed_sections) if allowed_sections else None,
                "top_k": top_k,
                "chunks": [],
            }
        except Exception as exc:
            # 绝不能把 Neo4j/序列化错误伪装成“该论文没有证据”。调用方需要
            # 得到显式错误，才能与真实 evidence gap 区分开。
            raise RuntimeError(f"paper search failed for {paper_id}: {exc}") from exc

        if allowed_sections is not None:
            chunks = [c for c in chunks if c.get("section") in allowed_sections]
        chunks = chunks[:top_k]

        return {
            "paper_id": paper_id,
            "query": query_text,
            "sections": list(allowed_sections) if allowed_sections else None,
            "top_k": top_k,
            "chunks": chunks,
        }

    @staticmethod
    def _paper_lookup_terms(query: str) -> list[str]:
        """提取用于书目解析的有界字面检索词。"""
        return paper_lookup_terms(query)

    def resolve_papers(self, query: str, top_k: int = 5) -> dict:
        """按作者、题名或 DOI 的字面元数据解析 paper_id。"""
        return resolve_papers_by_metadata(
            self.driver,
            query,
            top_k,
            database=self.database,
        )

    def expand_entities(
        self,
        text: str,
        top_k: int,
        fuzzy: bool,
    ) -> dict:
        """Link entities from text and optionally expand via vector similarity."""
        linked = self.linker.link(text) if self.linker else []
        expanded: list[dict] = []

        if fuzzy and text.strip() and self.embedder is not None:
            try:
                embedding = self.embedder.embed_query(text)
                query = """
                CALL db.index.vector.queryNodes($index_name, $fetch_k, $embedding)
                YIELD node AS ch, score
                MATCH (e:Entity)-[:MENTIONED_IN]->(ch)
                WHERE NOT e.name IN $linked
                RETURN e.name AS name, count(*) AS freq, avg(score) AS avg_score,
                       collect(DISTINCT e.type)[0] AS entity_type
                ORDER BY freq DESC, avg_score DESC
                LIMIT $top_k
                """
                with self.driver.session(database=self.database) as session:
                    result = session.run(
                        query,
                        index_name=self.index_name,
                        fetch_k=top_k * 5,
                        embedding=embedding,
                        linked=linked,
                        top_k=top_k,
                    )
                    for record in result:
                        expanded.append(
                            {
                                "name": record.get("name"),
                                "type": record.get("entity_type"),
                                "frequency": record.get("freq", 0),
                                "score": record.get("avg_score", 0.0),
                            }
                        )
            except (ImportError, Exception):
                # Embedder or vector index unavailable in test/minimal env.
                pass

        return {
            "text": text,
            "top_k": top_k,
            "fuzzy": fuzzy,
            "entities": linked,
            "expanded": expanded,
        }

    def aggregate_by_entities(
        self,
        entity_groups: list[list[str]],
        mode: str,
        top_k: int,
        min_cooccurrence: int,
    ) -> list[dict]:
        """Aggregate evidence chunks for each entity group."""
        mode = mode.lower()
        results: list[dict] = []

        for group in entity_groups:
            if not group:
                results.append({"entities": [], "count": 0, "chunks": []})
                continue

            if mode == "all":
                min_overlap = len(group)
            else:
                min_overlap = max(1, min_cooccurrence)

            query = """
            MATCH (e:Entity)-[:MENTIONED_IN]->(ch:Chunk)
            WHERE e.name IN $group
            WITH ch, count(DISTINCT e) AS overlap
            WHERE overlap >= $min_overlap
            RETURN ch, overlap
            ORDER BY overlap DESC, ch.chunk_index ASC
            LIMIT $top_k
            """
            chunks: list[dict] = []
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(
                        query,
                        group=list(group),
                        min_overlap=min_overlap,
                        top_k=top_k,
                    )
                    for record in result:
                        chunk = self._chunk_record_to_dict(record.get("ch"), overlap=record.get("overlap"))
                        chunks.append(chunk)
            except Exception:
                pass

            results.append(
                {
                    "entities": list(group),
                    "count": len(chunks),
                    "chunks": chunks,
                }
            )

        return results

    def _chunk_record_to_dict(
        self,
        record: dict | None,
        score: float | None = None,
        overlap: int | None = None,
    ) -> dict:
        """Normalize a chunk record (dict or Neo4j Node) to a plain dict."""
        if record is None:
            record = {}
        # Handle neo4j.graph.Node-like objects: they expose properties via __getitem__.
        data: dict = {}
        if hasattr(record, "items"):
            data = dict(record.items())  # type: ignore[arg-type]
        else:
            data = dict(record)
        result = {
            "chunk_id": data.get("chunk_id", ""),
            "text": data.get("text", ""),
            "paper_id": data.get("paper_id", ""),
            "paper_title": data.get("paper_title", ""),
            "section": data.get("section", ""),
            "chunk_index": data.get("chunk_index"),
            "source_path": data.get("source_path", ""),
            "page_idx": data.get("page_idx"),
            "paragraph_idx": data.get("paragraph_idx"),
            "bbox": data.get("bbox"),
            "char_start": data.get("char_start"),
            "char_end": data.get("char_end"),
            "text_hash": data.get("text_hash", ""),
        }
        if score is not None:
            result["score"] = score
        if overlap is not None:
            result["overlap"] = overlap
        return result
