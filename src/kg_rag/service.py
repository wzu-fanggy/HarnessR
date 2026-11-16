"""Loopback-only HTTP boundary for the GPU-backed root RAG process."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from hashlib import sha256
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


_SUPPORTED_OPERATIONS = {
    "hybrid_search",
    "hybrid_search_filtered",
    "subgraph_search",
    "path_search",
    "paper_search",
    "resolve_papers",
    "expand_entities",
    "aggregate_evidence",
    "literature_chunks",
}

_RUNTIME_SOURCE_FILES = (
    "kg_rag/service.py",
    "kg_rag/config.py",
    "kg_rag/service_kg_query.py",
    "kg_rag/agent_tools.py",
    "kg_rag/retriever.py",
    "kg_rag/entity_resolver.py",
    "kg_rag/entity_linker.py",
    "kg_rag/query_variants.py",
    "kg_rag/reranker.py",
    "kg_rag/vector_store.py",
    "kg_rag/embedder.py",
    "kg_rag/resource_telemetry.py",
    "kg_rag/runtime_resources.py",
)


def _compute_runtime_code_fingerprint() -> str:
    """Hash the retrieval service source at process startup.

    The value is deliberately computed once during import. If files change
    while an old worker is still serving requests, its metadata continues to
    expose the code snapshot that process actually loaded.
    """
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for relative_path in _RUNTIME_SOURCE_FILES:
        path = root / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


_RUNTIME_CODE_FINGERPRINT = _compute_runtime_code_fingerprint()


class EmbedRequest(BaseModel):
    """受限批量 embedding 请求，避免单次 Agent 调用耗尽内存。"""

    texts: list[str] = Field(min_length=1, max_length=16)

    def validated_texts(self) -> list[str]:
        if any(not text.strip() or len(text) > 8000 for text in self.texts):
            raise ValueError("texts must be non-blank and at most 8000 characters")
        return self.texts


class EvidenceRequest(BaseModel):
    """受限证据读取请求；只允许已知 chunk ID。"""

    chunk_ids: list[str] = Field(min_length=1, max_length=20)

    def validated_chunk_ids(self) -> list[str]:
        if any(not chunk_id.strip() or len(chunk_id) > 512 for chunk_id in self.chunk_ids):
            raise ValueError("chunk_ids must be non-blank and at most 512 characters")
        return list(dict.fromkeys(self.chunk_ids))


class KgQueryRequest(BaseModel):
    """受限固定 KG 查询请求，不接收 Cypher 文本。"""

    query_type: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    salt: str | None = None
    solvents: list[str] | None = Field(default=None, max_length=10)
    additives: list[str] | None = Field(default=None, max_length=10)
    entity_names: list[str] | None = Field(default=None, max_length=10)
    child_chunk_ids: list[str] | None = Field(default=None, max_length=20)
    limit: int | None = Field(default=None, ge=1, le=15)
    top_k: int | None = Field(default=None, ge=1, le=15)


class SearchRequest(BaseModel):
    """Pass-through arguments for one allowlisted kg_rag callable."""

    query: str | None = None
    top_k: int | None = None
    mode: str | None = None
    sections: list[str] | None = None
    paper_ids: list[str] | None = None
    entities: list[str] | None = None
    exclude_low_value: bool | None = None
    depth: int | None = None
    with_chunks: bool | None = None
    from_entity: str | None = None
    to_entity: str | None = None
    max_depth: int | None = None
    paper_id: str | None = None
    text: str | None = None
    fuzzy: bool | None = None
    entity_groups: list[list[str]] | None = None
    min_cooccurrence: int | None = None
    expand_hops: int | None = None
    include_parent: bool | None = None


def _default_load_state():
    """Construct a stable callable surface over independently reloadable resources."""
    from kg_rag.agent_tools import make_tools
    from kg_rag.embedder import Bgem3Embedder
    from kg_rag.runtime_resources import ReloadableNeo4jDriver

    driver = ReloadableNeo4jDriver()
    embedder = Bgem3Embedder()
    _, callables, _ = make_tools(driver=driver, embedder=embedder)
    return type(
        "RagState",
        (),
        {"callables": callables, "driver": driver, "embedder": embedder},
    )()


def _default_load_metadata_driver():
    """Create only a Neo4j driver for the fixed metadata snapshot endpoint."""
    from neo4j import GraphDatabase

    from kg_rag.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _metadata_snapshot(driver: Any) -> dict[str, Any]:
    """Collect a fixed, credential-free snapshot without accepting Cypher from callers."""
    from kg_rag.config import NEO4J_DATABASE, VECTOR_INDEX_NAME

    queries = {
        "server": """
            CALL dbms.components() YIELD versions
            UNWIND versions AS version
            RETURN version
            LIMIT 1
        """,
        "nodes": "MATCH (n) RETURN count(n) AS count",
        "relationships": "MATCH ()-[r]->() RETURN count(r) AS count",
        "labels": """
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label, count(n) AS count
            ORDER BY label
        """,
        "relationship_types": """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS count
            ORDER BY type
        """,
        "index": """
            SHOW INDEXES YIELD name, type, state, entityType, labelsOrTypes, properties, options
            WHERE name = $index_name
            RETURN name, type, state, entityType, labelsOrTypes, properties, options
        """,
    }
    with driver.session(database=NEO4J_DATABASE) as session:
        results = {
            name: session.run(query, {"index_name": VECTOR_INDEX_NAME} if name == "index" else {}).data()
            for name, query in queries.items()
        }

    index = results["index"][0] if results["index"] else None
    options = index.get("options") if isinstance(index, dict) else None
    index_config = options.get("indexConfig") if isinstance(options, dict) else {}
    index_payload = {
        "name": VECTOR_INDEX_NAME,
        "present": index is not None,
        "type": index.get("type") if index else None,
        "state": index.get("state") if index else None,
        "entity_type": index.get("entityType") if index else None,
        "labels_or_types": index.get("labelsOrTypes") if index else [],
        "properties": index.get("properties") if index else [],
        "dimensions": index_config.get("vector.dimensions"),
        "similarity_function": index_config.get("vector.similarity_function"),
    }
    stable = {
        "schema_version": "1.0",
        "service": "kg-rag",
        "database": NEO4J_DATABASE,
        "neo4j_version": results["server"][0].get("version") if results["server"] else None,
        "graph": {
            "node_count": results["nodes"][0]["count"],
            "relationship_count": results["relationships"][0]["count"],
            "nodes_by_label": {row["label"]: row["count"] for row in results["labels"]},
            "relationships_by_type": {row["type"]: row["count"] for row in results["relationship_types"]},
        },
        "vector_index": index_payload,
    }
    fingerprint = sha256(
        json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **stable,
        "runtime_code": {
            "fingerprint": _RUNTIME_CODE_FINGERPRINT,
            "source_files": list(_RUNTIME_SOURCE_FILES),
        },
        "snapshot_fingerprint": fingerprint,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def _expand_entity_neighbors(state: Any, result: Any) -> Any:
    """Add a bounded, deduplicated one-hop chunk expansion within the root service."""
    if not isinstance(result, dict) or not isinstance(result.get("chunks"), list):
        return result
    entities = list(
        dict.fromkeys(
            entity
            for chunk in result["chunks"]
            if isinstance(chunk, dict)
            for entity in (chunk.get("entities") or [])[:5]
        )
    )[:10]
    if not entities:
        return result
    from kg_rag.config import NEO4J_DATABASE

    query = """
        UNWIND $entities AS entity_name
        MATCH (e:Entity {name: entity_name})-[:MENTIONED_IN]->(ch:Chunk)
        RETURN ch.chunk_id AS chunk_id, ch.text AS text, ch.paper_title AS paper_title,
               ch.section AS section, ch.mentions AS entities
        LIMIT $top_k
    """
    try:
        with state.driver.session(database=NEO4J_DATABASE) as session:
            neighbors = session.run(query, {"entities": entities, "top_k": 10}).data()
    except Exception:
        return result
    known_ids = {chunk.get("chunk_id") for chunk in result["chunks"] if isinstance(chunk, dict)}
    for neighbor in neighbors:
        if neighbor.get("chunk_id") not in known_ids:
            result["chunks"].append({**neighbor, "source": "kg_expansion"})
            known_ids.add(neighbor.get("chunk_id"))
    return result


def _attach_parent_context(state: Any, result: Any) -> Any:
    """Attach parent text inside the root process that owns the Neo4j driver."""
    if not isinstance(result, dict) or not isinstance(result.get("chunks"), list):
        return result
    child_ids = [
        chunk.get("chunk_id")
        for chunk in result["chunks"]
        if isinstance(chunk, dict) and chunk.get("chunk_id")
    ]
    if not child_ids:
        return result
    from kg_rag.config import NEO4J_DATABASE

    query = """
        UNWIND $child_ids AS child_id
        MATCH (child:Chunk {chunk_id: child_id})
        OPTIONAL MATCH (parent:Chunk {chunk_id: child.parent_id})
        RETURN child.chunk_id AS child_id, parent.chunk_id AS parent_chunk_id,
               parent.text AS parent_text
    """
    try:
        with state.driver.session(database=NEO4J_DATABASE) as session:
            parents = {row["child_id"]: row for row in session.run(query, {"child_ids": child_ids}).data()}
    except Exception:
        return result
    for chunk in result["chunks"]:
        if not isinstance(chunk, dict):
            continue
        parent = parents.get(chunk.get("chunk_id"))
        if parent:
            chunk["parent_chunk_id"] = parent.get("parent_chunk_id")
            chunk["parent_text"] = parent.get("parent_text")
    return result


def create_app(
    load_state: Callable[[], Any] = _default_load_state,
    load_metadata_driver: Callable[[], Any] = _default_load_metadata_driver,
) -> FastAPI:
    """Create a local-only service app without loading the embedding model eagerly."""
    app = FastAPI(title="KG RAG Local Service")
    state: Any | None = None
    metadata_driver: Any | None = None
    state_lock = threading.RLock()

    def get_state() -> Any:
        nonlocal state
        with state_lock:
            if state is None:
                state = load_state()
            return state

    def get_metadata_driver() -> Any:
        nonlocal metadata_driver
        with state_lock:
            if state is not None:
                return state.driver
            if metadata_driver is None:
                metadata_driver = load_metadata_driver()
            return metadata_driver

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "kg-rag"}

    @app.get("/v1/resource-residency")
    def resource_residency() -> dict[str, Any]:
        """只读查看本进程资源状态；不会触发模型加载。"""
        from kg_rag.resource_telemetry import snapshot

        return snapshot()

    def runtime_resource_snapshot() -> dict[str, dict[str, object]]:
        # A read-only inventory request must not import torch or initialize the
        # retrieval callables. Before the first acquire/search, both resources
        # are known to be absent and can be reported without constructing state.
        with state_lock:
            current = state
        if current is None:
            return {
                "bge-m3-model": {
                    "resource_id": "bge-m3-model", "resident": False,
                    "host_ram_mib": 0.0, "host_ram_measurement": "not_resident",
                    "gpu_allocated_mib": 0.0, "gpu_vram_pressure": None,
                    "load_count": 0, "release_count": 0, "last_load_seconds": 0.0,
                },
                "neo4j-session": {
                    "resource_id": "neo4j-session", "resident": False,
                    "connection_slots": 0.0, "open_count": 0,
                    "close_count": 0, "last_open_seconds": 0.0,
                },
            }
        embedder = getattr(current, "embedder", None)
        driver = getattr(current, "driver", None)
        model_snapshot = (
            embedder.runtime_snapshot()
            if embedder is not None and hasattr(embedder, "runtime_snapshot")
            else {"resource_id": "bge-m3-model", "resident": embedder is not None}
        )
        session_snapshot = (
            driver.runtime_snapshot()
            if driver is not None and hasattr(driver, "runtime_snapshot")
            else {"resource_id": "neo4j-session", "resident": driver is not None}
        )
        return {
            "bge-m3-model": dict(model_snapshot),
            "neo4j-session": dict(session_snapshot),
        }

    @app.get("/v1/runtime/resources")
    def runtime_resources() -> dict[str, Any]:
        """返回可由上层 Agent runtime 管理的物理资源状态。"""

        return {"resources": runtime_resource_snapshot()}

    @app.post("/v1/runtime/resources/{resource_id}/acquire")
    def acquire_runtime_resource(resource_id: str) -> dict[str, Any]:
        """显式建立一个可重建资源；该接口只暴露固定 allowlist。"""

        current = get_state()
        try:
            if resource_id == "bge-m3-model":
                embedder = getattr(current, "embedder", None)
                ensure_loaded = getattr(embedder, "ensure_loaded", None)
                if not callable(ensure_loaded):
                    raise RuntimeError("embedding resource is not lifecycle-managed")
                ensure_loaded()
            elif resource_id == "neo4j-session":
                driver = getattr(current, "driver", None)
                ensure_open = getattr(driver, "ensure_open", None)
                if not callable(ensure_open):
                    raise RuntimeError("Neo4j resource is not lifecycle-managed")
                ensure_open()
            else:
                raise HTTPException(status_code=404, detail="unsupported runtime resource")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"runtime resource acquire failed: {exc}",
            ) from exc
        return {"resource": runtime_resource_snapshot()[resource_id]}

    @app.post("/v1/runtime/resources/{resource_id}/release")
    def release_runtime_resource(resource_id: str) -> dict[str, Any]:
        """释放一个空闲可重建资源，同时保留服务 callable 与逻辑接口。"""

        current = get_state()
        try:
            if resource_id == "bge-m3-model":
                embedder = getattr(current, "embedder", None)
                release = getattr(embedder, "release", None)
                if not callable(release):
                    raise RuntimeError("embedding resource is not lifecycle-managed")
                release()
            elif resource_id == "neo4j-session":
                driver = getattr(current, "driver", None)
                close = getattr(driver, "close", None)
                if not callable(close):
                    raise RuntimeError("Neo4j resource is not lifecycle-managed")
                close()
            else:
                raise HTTPException(status_code=404, detail="unsupported runtime resource")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"runtime resource release failed: {exc}",
            ) from exc
        return {"resource": runtime_resource_snapshot()[resource_id]}

    @app.get("/v1/metadata")
    def metadata() -> dict[str, Any]:
        try:
            return _metadata_snapshot(get_metadata_driver())
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"metadata query failed: {exc}") from exc

    @app.post("/v1/embed")
    def embed(request: EmbedRequest) -> dict[str, Any]:
        try:
            texts = request.validated_texts()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        embedder = getattr(get_state(), "embedder", None)
        if embedder is None:
            raise HTTPException(status_code=503, detail="embedding model unavailable")
        try:
            vectors = embedder.embed_documents(texts)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"embedding failed: {exc}") from exc
        dimension = len(vectors[0]) if vectors else 0
        return {"dimension": dimension, "vectors": vectors}

    @app.post("/v1/evidence/chunks")
    def evidence_chunks(request: EvidenceRequest) -> dict[str, list[dict[str, Any]]]:
        try:
            chunk_ids = request.validated_chunk_ids()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        from kg_rag.config import NEO4J_DATABASE

        query = """
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {chunk_id: chunk_id})
            OPTIONAL MATCH (c)-[:MENTIONED_IN]->(e:Entity)
            OPTIONAL MATCH (c)-[:HAS_TRIPLE]->(t:Triple)
            RETURN c.chunk_id AS chunk_id, c.text AS text,
                   c.paper_title AS paper_title, c.paper_id AS paper_id,
                   c.section AS section, c.page_idx AS page_idx,
                   c.paragraph_idx AS paragraph_idx, c.bbox AS bbox,
                   c.char_start AS char_start, c.char_end AS char_end,
                   c.text_hash AS text_hash,
                   collect(DISTINCT e.name) AS entities,
                   [x IN collect(DISTINCT {
                       s: t.subject, p: t.predicate, o: t.object
                   }) WHERE x.s IS NOT NULL][0..10] AS triples
        """
        try:
            with get_state().driver.session(database=NEO4J_DATABASE) as session:
                chunks = session.run(query, {"chunk_ids": chunk_ids}).data()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"evidence query failed: {exc}") from exc
        return {"chunks": chunks}

    @app.post("/v1/kg/query/{operation}")
    def kg_query(operation: str, request: KgQueryRequest) -> dict[str, Any]:
        supported = {
            "run_query",
            "find_grounded_formulations",
            "chunks_for_entities",
            "parent_chunks",
        }
        if operation not in supported:
            raise HTTPException(status_code=404, detail="unsupported KG query operation")

        state = get_state()
        service = getattr(state, "kg_query_service", None)
        if service is None:
            from kg_rag.service_kg_query import KgQueryService

            service = KgQueryService(driver=state.driver)
            state.kg_query_service = service
        try:
            if operation == "run_query":
                if not request.query_type:
                    raise HTTPException(status_code=422, detail="query_type is required")
                return {"results": service.run_query(request.query_type, request.params)}
            if operation == "find_grounded_formulations":
                return {"results": service.find_grounded_formulations(
                    salt=request.salt,
                    solvents=request.solvents,
                    additives=request.additives,
                    limit=request.limit or 10,
                )}
            if operation == "chunks_for_entities":
                if request.entity_names is None:
                    raise HTTPException(status_code=422, detail="entity_names is required")
                return {"results": service.chunks_for_entities(
                    request.entity_names, top_k=request.top_k or 10
                )}
            if request.child_chunk_ids is None:
                raise HTTPException(status_code=422, detail="child_chunk_ids is required")
            return {"results": service.parent_chunks(request.child_chunk_ids)}
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"KG query failed: {exc}") from exc

    @app.post("/v1/search/{operation}")
    def search(operation: str, request: SearchRequest) -> Any:
        if operation not in _SUPPORTED_OPERATIONS:
            raise HTTPException(status_code=404, detail="unsupported RAG operation")

        kwargs = request.model_dump(exclude_none=True)
        include_parent = bool(kwargs.pop("include_parent", False))
        expand_hops = int(kwargs.pop("expand_hops", 0))
        if operation == "resolve_papers":
            query = str(kwargs.get("query") or "").strip()
            if not query:
                raise HTTPException(status_code=422, detail="query is required")
            from kg_rag.retriever import resolve_papers_by_metadata

            return resolve_papers_by_metadata(
                get_metadata_driver(),
                query,
                int(kwargs.get("top_k", 5)),
            )

        state = get_state()
        if operation == "literature_chunks":
            # 这是 Orion 的稳定检索契约：上下文选项不泄漏到通用检索 callable。
            callable_ = getattr(state, "callables", {}).get("hybrid_search_filtered")
        else:
            callable_ = getattr(state, "callables", {}).get(operation)
        if callable_ is None:
            raise HTTPException(status_code=503, detail="RAG operation unavailable")

        try:
            raw_result = callable_(**kwargs)
            result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
            if operation == "literature_chunks" and expand_hops > 0:
                result = _expand_entity_neighbors(state, result)
            return _attach_parent_context(state, result) if operation == "literature_chunks" and include_parent else result
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=502, detail="RAG service returned invalid JSON") from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"RAG operation failed: {exc}") from exc

    return app


app = create_app()
