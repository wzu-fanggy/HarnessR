"""Neo4j vector index and chunk/paper CRUD."""
from __future__ import annotations

import json
import re
from neo4j import Driver

from kg_rag.config import NEO4J_DATABASE, VECTOR_INDEX_NAME

# English stopwords plus generic tokens that should never be treated as entities.
_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "can't", "cannot", "could", "couldn't", "did",
    "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few",
    "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
    "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
}
_JUNK_RE = re.compile(r"^[\W\d_]+$", re.UNICODE)


def _is_linkable_name(name: str) -> bool:
    """Return False for stopwords, pure numbers/punctuation, or empty names."""
    if not name:
        return False
    stripped = name.strip()
    if not stripped:
        return False
    if len(stripped) == 1:
        return False
    if stripped.lower() in _STOPWORDS:
        return False
    if _JUNK_RE.match(stripped):
        return False
    return True


class Neo4jVectorStore:
    def __init__(self, driver: Driver, database: str = NEO4J_DATABASE):
        self.driver = driver
        self.database = database

    def get_version(self) -> tuple[int, int, int]:
        query = "CALL dbms.components() YIELD versions UNWIND versions AS v RETURN v LIMIT 1"
        with self.driver.session(database=self.database) as session:
            record = session.run(query).single()
            version_str = record["v"] if record else "5.11.0"
        if not isinstance(version_str, str):
            version_str = "5.11.0"
        parts = version_str.split(".")
        return tuple(int(p) for p in parts[:3])

    def _quantization_config(self) -> str:
        major, minor, patch = self.get_version()
        if major > 2026 or (major == 2026 and minor >= 6):
            return "`vector.quantization.type`: 'SCALAR'"
        return "`vector.quantization.enabled`: true"

    def ensure_indexes(self, dim: int) -> None:
        quantization = self._quantization_config()
        constraint_chunk = """
        CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
        FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE
        """
        constraint_paper = """
        CREATE CONSTRAINT paper_id_unique IF NOT EXISTS
        FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE
        """
        vector_index = f"""
        CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: $dim,
            `vector.similarity_function`: 'cosine',
            `vector.hnsw.m`: 16,
            `vector.hnsw.ef_construction`: 100,
            {quantization}
          }}
        }}
        """
        with self.driver.session(database=self.database) as session:
            session.run(constraint_chunk)
            session.run(constraint_paper)
            session.run(vector_index, dim=dim)

    def load_entity_names(self, min_len: int = 2) -> set[str]:
        query = "MATCH (n:Entity) RETURN n.name AS name"
        names = set()
        with self.driver.session(database=self.database) as session:
            for record in session.run(query):
                name = record["name"]
                if name and len(name) >= min_len and _is_linkable_name(name):
                    names.add(name)
        return names

    def create_paper_nodes(self, papers: list[dict]) -> None:
        query = """
        UNWIND $papers AS p
        MERGE (paper:Paper {paper_id: p.paper_id})
        ON CREATE SET paper.title = p.title, paper.source_path = p.source_path
        ON MATCH SET paper.title = p.title, paper.source_path = p.source_path
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, papers=papers)

    def create_chunks(self, chunks: list[dict], batch_size: int = 64) -> None:
        query = """
        UNWIND $chunks AS c
        MERGE (ch:Chunk {chunk_id: c.chunk_id})
        SET ch.text = c.text,
            ch.embedding = c.embedding,
            ch.paper_id = c.paper_id,
            ch.paper_title = c.paper_title,
            ch.section = c.section,
            ch.chunk_type = c.chunk_type,
            ch.chunk_index = c.chunk_index,
            ch.parent_id = c.parent_id,
            ch.source_path = c.source_path,
            ch.mentions = c.mentions,
            ch.char_len = c.char_len,
            ch.page_idx = c.page_idx,
            ch.paragraph_idx = c.paragraph_idx,
            ch.bbox = c.bbox,
            ch.char_start = c.char_start,
            ch.char_end = c.char_end,
            ch.text_hash = c.text_hash
        WITH ch, c
        MATCH (paper:Paper {paper_id: c.paper_id})
        MERGE (paper)-[:HAS_CHUNK]->(ch)
        """
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(chunks), batch_size):
                session.run(query, chunks=chunks[i:i + batch_size])

    def create_mention_rels(self, rels: list[dict], batch_size: int = 200) -> None:
        query = """
        UNWIND $rels AS r
        MATCH (ch:Chunk {chunk_id: r.chunk_id})
        MATCH (e:Entity {name: r.entity_name})
        MERGE (e)-[:MENTIONED_IN]->(ch)
        """
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rels), batch_size):
                session.run(query, rels=rels[i:i + batch_size])
