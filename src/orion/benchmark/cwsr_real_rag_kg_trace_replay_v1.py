"""Real BGE-M3 + Neo4j replay of eight preselected scientific-agent traces.

The eight traces are selected before policy results are inspected:
arm == G_evidence_governed and repeat_index == 1, one per Study4C
research-task family. Only resource-bearing query calls are replayed:

- search_literature_chunks / search_literature_multiview / search_paper_chunks:
  real BGE-M3 embeddings in the root CUDA environment plus the real Neo4j
  chunk vector index;
- kg_query: the frozen production read-only Cypher template on the real Neo4j
  database, without BGE-M3.

Local deterministic screening/report calls are intentionally excluded. This makes
the measured wall clock a runtime-resource/query-path comparison rather than an
LLM or chemistry-quality benchmark.

Two policies execute the exact same frozen operations:
- cwsr: idle BGE and Neo4j backings remain available until pressure requires
  release;
- release_all: every idle backing is demoted after each resource-bearing call.

BGE-M3 remains in the root kg_create CUDA environment and is controlled as an
external MODEL backing process. Neo4j is a separate SESSION backing in Orion.
This preserves the deployed environment boundary while allowing their lifecycles
to be managed independently.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from threading import RLock
from time import monotonic
from typing import Any, Callable

from kg_rag.config import (
    EMBEDDING_MODEL,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    VECTOR_INDEX_NAME,
)
from kg_rag.service_kg_query import _QUERIES
from orion.capvm import (
    CapabilityCatalog,
    CapabilityDependency,
    CapabilityKind,
    CapabilityResidencyGovernor,
    ResidencyLevel,
    ResourceVector,
    VirtualCapability,
)
from orion.capvm.adapters import Neo4jSessionCapabilityAdapter, ToolCapabilityAdapter
from orion.capvm.adapters.base import BaseCapabilityAdapter
from orion.capvm.telemetry import process_rss_mb


ORION_ROOT = Path(__file__).resolve().parents[1]
KG_ROOT = ORION_ROOT.parent
TRACE_ROOT = (
    ORION_ROOT
    / "benchmark"
    / "self_evolution_study4c_v1_20260822"
    / "formal_run_v1"
    / "raw"
)
WORKER_SCRIPT = ORION_ROOT / "benchmark" / "cwsr_bgem3_external_worker_v1.py"
ROOT_PYTHON = Path(sys.executable)

MODEL_ID = "model:bge-m3-external"
SESSION_ID = "session:neo4j-kg-v1"
RAG_ID = "tool:real-vector-retrieval"
KG_ID = "tool:real-structured-kg-query"

RAG_TOOLS = {
    "search_literature_chunks",
    "search_literature_multiview",
    "search_paper_chunks",
}
RESOURCE_TOOLS = {*RAG_TOOLS, "kg_query"}
STRATEGIES = ("cwsr", "release_all")

DECLARED_BGE_GPU_MIB = 2300.0
HOST_RAM_BUDGET_MIB = 4096.0
GPU_BUDGET_MIB = 3000.0

VECTOR_QUERY = """
CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
YIELD node AS chunk, score
RETURN chunk.chunk_id AS chunk_id,
       chunk.paper_id AS paper_id,
       chunk.paper_title AS paper_title,
       chunk.section AS section,
       score
ORDER BY score DESC, chunk.chunk_id ASC
LIMIT $top_k
"""

PAPER_VECTOR_QUERY = """
MATCH (paper:Paper {paper_id: $paper_id})-[:HAS_CHUNK]->(ch:Chunk)
WHERE ch.embedding IS NOT NULL
  AND (
    $sections IS NULL
    OR any(section IN $sections
           WHERE toLower(coalesce(ch.section, '')) CONTAINS section)
  )
WITH ch, vector.similarity.cosine(ch.embedding, $embedding) AS score
RETURN ch.chunk_id AS chunk_id,
       ch.paper_id AS paper_id,
       ch.paper_title AS paper_title,
       ch.section AS section,
       score
ORDER BY score DESC, ch.chunk_id ASC
LIMIT $top_k
"""

PAPER_VECTOR_FALLBACK_QUERY = PAPER_VECTOR_QUERY.replace(
    "MATCH (paper:Paper {paper_id: $paper_id})-[:HAS_CHUNK]->(ch:Chunk)",
    "MATCH (ch:Chunk {paper_id: $paper_id})",
)


class ExternalBgeModelAdapter(BaseCapabilityAdapter):
    """Manage a CUDA BGE-M3 worker running in the root project environment."""

    def __init__(
        self,
        *,
        python_exe: Path,
        worker_script: Path,
        model_path: Path,
        declared_gpu_mib: float = DECLARED_BGE_GPU_MIB,
        shutdown_timeout_s: float = 10.0,
    ) -> None:
        self.python_exe = python_exe.resolve()
        self.worker_script = worker_script.resolve()
        self.model_path = model_path.resolve()
        self.declared_gpu_mib = float(declared_gpu_mib)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self._process: subprocess.Popen[str] | None = None
        self._worker_pid: int | None = None
        self.transport_ready_metadata: dict[str, Any] | None = None
        self.ready_metadata: dict[str, Any] | None = None
        self.latest_gpu_telemetry: dict[str, float] = {}
        self.query_gpu_telemetry: list[dict[str, float]] = []
        self.transport_baseline_rss_mb = 0.0
        self.transport_startup_latency_s = 0.0
        self.last_startup_latency_s = 0.0
        self.last_shutdown_latency_s = 0.0
        self.load_count = 0
        self.embed_count = 0
        self.embed_elapsed_s = 0.0
        self.startup_latencies_s: list[float] = []
        self.shutdown_latencies_s: list[float] = []
        self.worker_cleanup_latencies_s: list[float] = []
        self.worker_cleanup_allocated_mib: list[float] = []
        self.worker_cleanup_reserved_mib: list[float] = []
        self._protocol_lock = RLock()

    @property
    def launcher_pid(self) -> int | None:
        process = self._process
        if process is None or process.poll() is not None:
            return None
        return int(process.pid)

    @property
    def pid(self) -> int | None:
        return self._worker_pid if self.is_transport_open else None

    @property
    def is_transport_open(self) -> bool:
        return (
            self._process is not None
            and self._process.poll() is None
            and self._worker_pid is not None
            and self.transport_ready_metadata is not None
        )

    @property
    def is_loaded(self) -> bool:
        return self.is_transport_open and self.ready_metadata is not None

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND:
            return ResourceVector()
        return ResourceVector(gpu_vram_mb=self.declared_gpu_mib)

    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND or self.pid is None:
            return ResourceVector()
        ready = self.ready_metadata or {}
        current_rss = process_rss_mb(self.pid)
        baseline = float((self.transport_ready_metadata or {}).get("torch_allocated_mib") or 0.0)
        current_gpu = self.latest_gpu_telemetry.get("torch_allocated_mib")
        allocated_delta = (
            max(0.0, current_gpu - baseline) if current_gpu is not None
            else float(ready.get("torch_allocated_delta_mib") or 0.0)
        )
        return ResourceVector(
            host_ram_mb=max(0.0, current_rss - self.transport_baseline_rss_mb),
            gpu_vram_mb=allocated_delta,
        )

    def _record_gpu_telemetry(self, payload: dict[str, Any], *, query: bool) -> None:
        keys = ("torch_allocated_mib", "torch_reserved_mib",
                "torch_peak_allocated_mib", "torch_peak_reserved_mib")
        values = {key: float(payload[key]) for key in keys if key in payload}
        self.latest_gpu_telemetry = values
        if query:
            # 旧 worker 的缺失字段不以加载快照替代，也不计作已测得的查询峰值。
            self.query_gpu_telemetry.append(values)

    def prepare_transport(self) -> None:
        self._ensure_transport()

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current < ResidencyLevel.BOUND <= target and not self.is_loaded:
            self._ensure_transport()
            self._load_model()

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current >= ResidencyLevel.BOUND > target:
            self._unload_model()

    def embed_query(self, text: str, *, request_id: str) -> list[float]:
        with self._protocol_lock:
            if not self.is_loaded:
                raise RuntimeError("external BGE model is not resident")
            process = self._process
            assert process is not None and process.stdin is not None
            process.stdin.write(
                json.dumps(
                    {
                        "action": "embed",
                        "request_id": request_id,
                        "text": str(text),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            process.stdin.flush()
            payload = self._read_message("RESULT")
            self.embed_count += 1
            self.embed_elapsed_s += float(payload.get("embed_s") or 0.0)
            self._record_gpu_telemetry(payload, query=True)
            vector = payload.get("vector")
            if not isinstance(vector, list):
                raise RuntimeError("external BGE worker returned no vector")
            return [float(item) for item in vector]

    def close(self) -> None:
        self._unload_model()
        self._stop_transport()

    def _ensure_transport(self) -> None:
        if self.is_transport_open:
            return
        for path, label in (
            (self.python_exe, "root Python"),
            (self.worker_script, "BGE worker"),
            (self.model_path, "BGE model"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        started = monotonic()
        process = subprocess.Popen(
            [
                str(self.python_exe),
                "-u",
                str(self.worker_script),
                "--model",
                str(self.model_path),
                "--device",
                "cuda",
            ],
            cwd=str(KG_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self._process = process
        try:
            ready = self._read_message("TRANSPORT_READY")
        except Exception:
            process.kill()
            process.wait(timeout=self.shutdown_timeout_s)
            self._process = None
            raise
        worker_pid = ready.get("pid")
        if worker_pid is None:
            raise RuntimeError("external BGE transport READY message has no pid")
        self._worker_pid = int(worker_pid)
        self.transport_ready_metadata = ready
        self.transport_startup_latency_s = monotonic() - started
        self.transport_baseline_rss_mb = process_rss_mb(self._worker_pid)

    def _load_model(self) -> None:
        with self._protocol_lock:
            if self.is_loaded:
                return
            self._ensure_transport()
            process = self._process
            assert process is not None and process.stdin is not None
            started = monotonic()
            process.stdin.write(json.dumps({"action": "load"}) + "\n")
            process.stdin.flush()
            ready = self._read_message("MODEL_READY")
            self.ready_metadata = ready
            self._record_gpu_telemetry(ready, query=False)
            self.last_startup_latency_s = monotonic() - started
            self.startup_latencies_s.append(self.last_startup_latency_s)
            self.load_count += 1

    def _unload_model(self) -> None:
        with self._protocol_lock:
            if not self.is_loaded:
                return
            process = self._process
            assert process is not None and process.stdin is not None
            started = monotonic()
            process.stdin.write(json.dumps({"action": "unload"}) + "\n")
            process.stdin.flush()
            stopped = self._read_message("MODEL_UNLOADED")
            self.last_shutdown_latency_s = monotonic() - started
            self.shutdown_latencies_s.append(self.last_shutdown_latency_s)
            self.worker_cleanup_latencies_s.append(
                float(stopped.get("cleanup_s") or 0.0)
            )
            self.worker_cleanup_allocated_mib.append(
                float(stopped.get("torch_allocated_after_cleanup_mib") or 0.0)
            )
            self.worker_cleanup_reserved_mib.append(
                float(stopped.get("torch_reserved_after_cleanup_mib") or 0.0)
            )
            self.ready_metadata = None
            self.latest_gpu_telemetry = {}

    def _read_message(self, expected_status: str) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("external BGE worker is not running")
        log_tail: list[str] = []
        while True:
            raw = process.stdout.readline()
            if raw == "":
                stderr = process.stderr.read().strip() if process.stderr is not None else ""
                raise RuntimeError(
                    f"external BGE worker ended before {expected_status}: {stderr}"
                )
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                log_tail.append(line)
                continue
            status = str(payload.get("status") or "")
            if status == "ERROR":
                raise RuntimeError(f"external BGE worker error: {payload}")
            if status == expected_status:
                payload["worker_log_tail"] = log_tail[-20:]
                return payload
            log_tail.append(line)

    def _stop_transport(self) -> None:
        process = self._process
        if process is None:
            return
        launcher_pid = int(process.pid)
        worker_pid = self._worker_pid
        force_tree_kill = False
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(json.dumps({"action": "stop"}) + "\n")
                    process.stdin.flush()
                self._read_message("STOPPED")
                process.wait(timeout=self.shutdown_timeout_s)
            except Exception:
                force_tree_kill = True
        if process.poll() is None:
            force_tree_kill = True
        if force_tree_kill:
            subprocess.run(
                ["taskkill", "/PID", str(launcher_pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            try:
                process.wait(timeout=self.shutdown_timeout_s)
            except Exception:
                pass
        for pid in (worker_pid, launcher_pid):
            if pid is None:
                continue
            subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        self._process = None
        self._worker_pid = None
        self.transport_ready_metadata = None
        self.ready_metadata = None
        self.transport_baseline_rss_mb = 0.0


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_vector_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": row.get("chunk_id"),
            "paper_id": row.get("paper_id"),
            "paper_title": row.get("paper_title"),
            "section": row.get("section"),
        }
        for row in rows
    ]


def _selected_records() -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for path in sorted(TRACE_ROOT.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("arm") == "G_evidence_governed"
            and int(record.get("repeat_index") or 0) == 1
        ):
            selected.append({**record, "_trace_path": str(path)})
    selected.sort(key=lambda row: str(row.get("family_id") or ""))
    if len(selected) != 8:
        raise RuntimeError(f"expected 8 preselected traces, found {len(selected)}")
    return selected


def _operations(record: dict[str, Any]) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for index, item in enumerate(record.get("tool_calls") or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if name not in RESOURCE_TOOLS:
            continue
        operations.append(
            {
                "source_index": index,
                "tool": name,
                "args": dict(item.get("args") or {}),
            }
        )
    return operations


def _new_runtime(
    neo4j_adapter_factory: Callable[..., Neo4jSessionCapabilityAdapter] = Neo4jSessionCapabilityAdapter,
) -> tuple[
    CapabilityResidencyGovernor,
    ExternalBgeModelAdapter,
    Neo4jSessionCapabilityAdapter,
]:
    model = ExternalBgeModelAdapter(
        python_exe=ROOT_PYTHON,
        worker_script=WORKER_SCRIPT,
        model_path=Path(EMBEDDING_MODEL),
    )
    session = neo4j_adapter_factory(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )
    rag_tool = ToolCapabilityAdapter(
        interface_text="real vector-backed electrolyte literature retrieval"
    )
    kg_tool = ToolCapabilityAdapter(
        interface_text="real structured Neo4j knowledge-graph query"
    )
    governor = CapabilityResidencyGovernor(
        catalog=CapabilityCatalog(
            (
                VirtualCapability(
                    capability_id=MODEL_ID,
                    kind=CapabilityKind.MODEL,
                    summary="external CUDA BGE-M3 embedding model",
                    adapter=model,
                    maximum_level=ResidencyLevel.BOUND,
                ),
                VirtualCapability(
                    capability_id=SESSION_ID,
                    kind=CapabilityKind.SESSION,
                    summary="real Neo4j kg-v1 session",
                    adapter=session,
                    maximum_level=ResidencyLevel.BOUND,
                ),
                VirtualCapability(
                    capability_id=RAG_ID,
                    kind=CapabilityKind.TOOL,
                    summary="vector-backed literature retrieval",
                    adapter=rag_tool,
                    dependencies=(
                        CapabilityDependency(MODEL_ID, ResidencyLevel.BOUND),
                        CapabilityDependency(SESSION_ID, ResidencyLevel.BOUND),
                    ),
                    maximum_level=ResidencyLevel.INTERFACE,
                ),
                VirtualCapability(
                    capability_id=KG_ID,
                    kind=CapabilityKind.TOOL,
                    summary="structured KG query",
                    adapter=kg_tool,
                    dependencies=(
                        CapabilityDependency(SESSION_ID, ResidencyLevel.BOUND),
                    ),
                    maximum_level=ResidencyLevel.INTERFACE,
                ),
            )
        ),
        budget=ResourceVector(
            prompt_tokens=256.0,
            host_ram_mb=HOST_RAM_BUDGET_MIB,
            gpu_vram_mb=GPU_BUDGET_MIB,
            process_slots=1.0,
            connection_slots=1.0,
        ),
    )
    return governor, model, session


def _vector_search(
    model: ExternalBgeModelAdapter,
    session: Neo4jSessionCapabilityAdapter,
    *,
    query: str,
    top_k: int,
    request_id: str,
) -> dict[str, Any]:
    vector = model.embed_query(query, request_id=request_id)
    rows = session.run_read(
        VECTOR_QUERY,
        index_name=VECTOR_INDEX_NAME,
        top_k=max(1, min(int(top_k), 10)),
        embedding=vector,
    )
    stable = _stable_vector_rows(rows)
    return {
        "kind": "vector",
        "query": query,
        "rows": stable,
        "signature": _canonical_sha(stable),
    }


def _paper_search(
    model: ExternalBgeModelAdapter,
    session: Neo4jSessionCapabilityAdapter,
    *,
    paper_id: str,
    query: str,
    sections: list[str] | None,
    top_k: int,
    request_id: str,
) -> dict[str, Any]:
    vector = model.embed_query(query, request_id=request_id)
    params = {
        "paper_id": paper_id,
        "embedding": vector,
        "sections": [str(value).lower() for value in sections] if sections else None,
        "top_k": max(1, min(int(top_k), 10)),
    }
    rows = session.run_read(PAPER_VECTOR_QUERY, **params)
    if not rows:
        rows = session.run_read(PAPER_VECTOR_FALLBACK_QUERY, **params)
    stable = _stable_vector_rows(rows)
    return {
        "kind": "paper_vector",
        "paper_id": paper_id,
        "query": query,
        "rows": stable,
        "signature": _canonical_sha(stable),
    }


def _rag_operation(
    operation: dict[str, Any],
    *,
    model: ExternalBgeModelAdapter,
    session: Neo4jSessionCapabilityAdapter,
    request_prefix: str,
) -> dict[str, Any]:
    name = operation["tool"]
    args = operation["args"]
    if name == "search_literature_chunks":
        return _vector_search(
            model,
            session,
            query=str(args.get("query") or ""),
            top_k=int(args.get("top_k") or 8),
            request_id=request_prefix,
        )
    if name == "search_paper_chunks":
        return _paper_search(
            model,
            session,
            paper_id=str(args.get("paper_id") or ""),
            query=str(args.get("query") or ""),
            sections=list(args.get("sections") or []) or None,
            top_k=int(args.get("top_k") or 8),
            request_id=request_prefix,
        )
    if name == "search_literature_multiview":
        queries = [str(value) for value in (args.get("queries") or [])]
        per_query_top_k = int(args.get("per_query_top_k") or 8)
        outputs = [
            _vector_search(
                model,
                session,
                query=query,
                top_k=per_query_top_k,
                request_id=f"{request_prefix}:{index}",
            )
            for index, query in enumerate(queries)
        ]
        signatures = [row["signature"] for row in outputs]
        return {
            "kind": "multiview_vector",
            "query_count": len(outputs),
            "signatures": signatures,
            "signature": _canonical_sha(signatures),
        }
    raise RuntimeError(f"unsupported RAG tool: {name}")


def _kg_operation(
    operation: dict[str, Any],
    *,
    session: Neo4jSessionCapabilityAdapter,
) -> dict[str, Any]:
    args = operation["args"]
    query_type = str(args.get("query_type") or "")
    if query_type not in _QUERIES:
        raise RuntimeError(f"unsupported frozen KG query type: {query_type}")
    params = dict(args.get("params") or {})
    params["limit"] = max(1, min(int(params.get("limit") or 10), 15))
    for key in (
        "salt",
        "solvent",
        "additive",
        "target",
        "material",
        "material_a",
        "material_b",
        "condition",
    ):
        params.setdefault(key, None)
    rows = session.run_read(_QUERIES[query_type], **params)
    stable = json.loads(json.dumps(rows, ensure_ascii=False, default=str))
    return {
        "kind": "structured_kg",
        "query_type": query_type,
        "row_count": len(stable),
        "signature": _canonical_sha(stable),
    }


def _release_all(governor: CapabilityResidencyGovernor) -> int:
    count = 0
    for state in governor.states():
        if state.current_level <= ResidencyLevel.CATALOG:
            continue
        governor.demote(state.capability_id, ResidencyLevel.CATALOG)
        count += 1
    return count


def _run_trace(
    record: dict[str, Any],
    *,
    strategy: str,
    operation_start: int = 0,
    operation_stop: int | None = None,
    neo4j_adapter_factory: Callable[..., Neo4jSessionCapabilityAdapter] = Neo4jSessionCapabilityAdapter,
) -> dict[str, Any]:
    all_operations = _operations(record)
    operations = all_operations[operation_start:operation_stop]
    governor, model, session = _new_runtime(neo4j_adapter_factory)
    # The persistent worker represents an already-running local RAG service.
    # Its Python/framework startup is infrastructure and is excluded from the
    # capability residency timing window.
    model.prepare_transport()
    started = monotonic()
    rows: list[dict[str, Any]] = []
    explicit_releases = 0
    peak = ResourceVector()
    last_result_elapsed_s: float | None = None
    try:
        for op_index, operation in enumerate(operations):
            is_rag = operation["tool"] in RAG_TOOLS
            capability_id = RAG_ID if is_rag else KG_ID
            op_started = monotonic()
            with governor.acquire_many(
                ((capability_id, ResidencyLevel.INTERFACE),),
                request_id=(
                    f"{record['family_id']}:{strategy}:{op_index}:{operation['tool']}"
                ),
                reconcile_before_active=True,
            ):
                materialized_at = monotonic()
                result = (
                    _rag_operation(
                        operation,
                        model=model,
                        session=session,
                        request_prefix=(
                            f"{record['family_id']}:{strategy}:{op_index}"
                        ),
                    )
                    if is_rag
                    else _kg_operation(operation, session=session)
                )
                result_ready = monotonic()
                last_result_elapsed_s = result_ready - started
                measured = governor.total_measured_cost()
                peak = ResourceVector(
                    prompt_tokens=max(peak.prompt_tokens, measured.prompt_tokens),
                    host_ram_mb=max(peak.host_ram_mb, measured.host_ram_mb),
                    gpu_vram_mb=max(peak.gpu_vram_mb, measured.gpu_vram_mb),
                    process_slots=max(peak.process_slots, measured.process_slots),
                    connection_slots=max(
                        peak.connection_slots, measured.connection_slots
                    ),
                )
                rows.append(
                    {
                        "source_index": operation["source_index"],
                        "tool": operation["tool"],
                        "materialization_s": materialized_at - op_started,
                        "query_execution_s": result_ready - materialized_at,
                        "elapsed_s": result_ready - op_started,
                        "signature": result["signature"],
                        "result_kind": result["kind"],
                        "measured": asdict(measured),
                    }
                )
            if strategy == "release_all":
                explicit_releases += _release_all(governor)

        policy_cycle_elapsed_s = monotonic() - started
        model_state = governor.state(MODEL_ID)
        session_state = governor.state(SESSION_ID)
        events = Counter(event.event_type for event in governor.events)
        return {
            "family_id": record["family_id"],
            "trace_path": record["_trace_path"],
            "strategy": strategy,
            "operation_count": len(operations),
            "operation_start": operation_start,
            "operation_stop": operation_stop,
            "full_operation_count": len(all_operations),
            "rag_operation_count": sum(op["tool"] in RAG_TOOLS for op in operations),
            "kg_operation_count": sum(op["tool"] == "kg_query" for op in operations),
            "complete": len(rows) == len(operations),
            "elapsed_to_last_result_s": (
                float(last_result_elapsed_s)
                if last_result_elapsed_s is not None
                else 0.0
            ),
            "policy_cycle_elapsed_s": policy_cycle_elapsed_s,
            "rows": rows,
            "output_signature": _canonical_sha(
                [row["signature"] for row in rows]
            ),
            "transport_startup_s_excluded": model.transport_startup_latency_s,
            "model_load_count": model.load_count,
            "model_load_total_s": sum(model.startup_latencies_s),
            "model_load_latencies_s": list(model.startup_latencies_s),
            "model_shutdown_total_s": sum(model.shutdown_latencies_s),
            "model_shutdown_latencies_s": list(model.shutdown_latencies_s),
            "worker_cleanup_total_s": sum(model.worker_cleanup_latencies_s),
            "worker_cleanup_latencies_s": list(model.worker_cleanup_latencies_s),
            "worker_cleanup_allocated_mib": list(model.worker_cleanup_allocated_mib),
            "worker_cleanup_reserved_mib": list(model.worker_cleanup_reserved_mib),
            "model_embed_count": model.embed_count,
            "model_embed_total_s": model.embed_elapsed_s,
            "query_gpu_telemetry": list(model.query_gpu_telemetry),
            "query_peak_allocated_mib": max(
                (item["torch_peak_allocated_mib"] for item in model.query_gpu_telemetry
                 if "torch_peak_allocated_mib" in item), default=None,
            ),
            "query_peak_reserved_mib": max(
                (item["torch_peak_reserved_mib"] for item in model.query_gpu_telemetry
                 if "torch_peak_reserved_mib" in item), default=None,
            ),
            "session_promotions": int(session_state.promotions),
            "session_demotions": int(session_state.demotions),
            "session_refaults": int(session_state.refaults),
            "session_query_count": int(session.query_count),
            "neo4j_driver_open_count": int(getattr(session, "driver_open_count", 0)),
            "neo4j_session_creation_count": int(getattr(session, "session_creation_count", 0)),
            "model_promotions": int(model_state.promotions),
            "model_demotions": int(model_state.demotions),
            "model_refaults": int(model_state.refaults),
            "explicit_release_count": explicit_releases,
            "peak_measured": asdict(peak),
            "event_counts": dict(sorted(events.items())),
        }
    finally:
        governor.close()


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("summary requires values")
    if len(values) == 1:
        value = float(values[0])
        return {
            "median": value,
            "q1": value,
            "q3": value,
            "min": value,
            "max": value,
        }
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "median": float(statistics.median(values)),
        "q1": float(quartiles[0]),
        "q3": float(quartiles[2]),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def run(
    *,
    family: str | None = None,
    neo4j_adapter_factory: Callable[..., Neo4jSessionCapabilityAdapter] = Neo4jSessionCapabilityAdapter,
) -> dict[str, Any]:
    records = _selected_records()
    if family is not None:
        records = [row for row in records if row["family_id"] == family]
        if len(records) != 1:
            raise KeyError(f"preselected family not found: {family}")

    paired: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        order = STRATEGIES if index % 2 == 0 else tuple(reversed(STRATEGIES))
        outputs: dict[str, dict[str, Any]] = {}
        for strategy in order:
            outputs[strategy] = _run_trace(
                record,
                strategy=strategy,
                neo4j_adapter_factory=neo4j_adapter_factory,
            )
        cwsr = outputs["cwsr"]
        release = outputs["release_all"]
        paired.append(
            {
                "family_id": record["family_id"],
                "order": list(order),
                "cwsr": cwsr,
                "release_all": release,
                "scientific_outputs_identical": (
                    cwsr["output_signature"] == release["output_signature"]
                ),
                "seconds_saved_release_minus_cwsr": (
                    float(release["elapsed_to_last_result_s"])
                    - float(cwsr["elapsed_to_last_result_s"])
                ),
                "model_loads_saved": (
                    int(release["model_load_count"])
                    - int(cwsr["model_load_count"])
                ),
                "session_promotions_saved": (
                    int(release["session_promotions"])
                    - int(cwsr["session_promotions"])
                ),
            }
        )

    savings = [float(row["seconds_saved_release_minus_cwsr"]) for row in paired]
    model_saved = [int(row["model_loads_saved"]) for row in paired]
    session_saved = [int(row["session_promotions_saved"]) for row in paired]
    all_outputs_equal = all(
        bool(row["scientific_outputs_identical"]) for row in paired
    )
    all_complete = all(
        row[strategy]["complete"]
        for row in paired
        for strategy in STRATEGIES
    )
    return {
        "schema_version": "cwsr-real-rag-kg-trace-replay-v1",
        "status": "PASS" if all_outputs_equal and all_complete else "FAIL",
        "selection_rule": {
            "arm": "G_evidence_governed",
            "repeat_index": 1,
            "one_trace_per_research_task_family": True,
            "selected_trace_count": len(records),
            "policy_result_blind_selection": True,
        },
        "replayed_operation_types": sorted(RESOURCE_TOOLS),
        "resource_mapping": {
            "rag": ["external CUDA BGE-M3 MODEL", "Neo4j SESSION"],
            "kg_query": ["Neo4j SESSION"],
        },
        "budget": {
            "host_ram_mib": HOST_RAM_BUDGET_MIB,
            "gpu_vram_mib": GPU_BUDGET_MIB,
            "process_slots": 1,
            "connection_slots": 1,
        },
        "paired": paired,
        "summary": {
            "task_family_count": len(paired),
            "seconds_saved_release_minus_cwsr": _summary(savings),
            "model_loads_saved": _summary([float(v) for v in model_saved]),
            "session_promotions_saved": _summary(
                [float(v) for v in session_saved]
            ),
            "all_scientific_outputs_identical": all_outputs_equal,
            "all_policy_runs_complete": all_complete,
            "positive_time_saving_family_count": sum(v > 0 for v in savings),
        },
        "interpretation_guard": (
            "The replay uses real BGE-M3 embeddings, the real Neo4j kg-v1 vector index, "
            "and frozen production KG Cypher templates with query arguments taken from "
            "preselected real agent traces. search_literature_multiview is replayed as "
            "its vector-backed query set and does not reconstruct the production RRF/graph "
            "merge; therefore this experiment evaluates runtime backing-resource behavior "
            "and query-path timing, not retrieval-quality equivalence to the full production "
            "hybrid retriever. Local screening/report calls and LLM time are excluded."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family")
    parser.add_argument("--strategy", choices=STRATEGIES)
    parser.add_argument("--operation-start", type=int, default=0)
    parser.add_argument("--operation-stop", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.strategy is None:
        result = run(family=args.family)
    else:
        records = _selected_records()
        if args.family is not None:
            records = [row for row in records if row["family_id"] == args.family]
        if len(records) != 1:
            raise RuntimeError(
                "single-strategy mode requires exactly one selected family"
            )
        trace_result = _run_trace(
            records[0],
            strategy=args.strategy,
            operation_start=args.operation_start,
            operation_stop=args.operation_stop,
        )
        result = {
            "schema_version": "cwsr-real-rag-kg-trace-replay-single-v1",
            "status": "PASS" if trace_result["complete"] else "FAIL",
            "family_id": records[0]["family_id"],
            "strategy": args.strategy,
            "trace_result": trace_result,
        }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
