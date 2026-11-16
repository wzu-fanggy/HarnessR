"""把现有只读 KG 查询结果投影到 Change Domain scientific layer。

本模块不连接 Neo4j。调用方只能传入既有 ``KgQueryService`` / ORION 白名单接口
返回的结构化结果，因此 V0 不引入自由 Cypher 或新的生产 KG 写路径。

重要边界：投影出的 KG edge 只表示 scientific coupling / retrieval structure；它本身
不等价于可修改 Harness 的 authority。后续若研究 KG mutation 驱动 evolution，必须另设
provenance/admission gate。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from .models import CouplingEdge, CouplingLayer, HarnessArtifact, MultiplexHarnessGraph


SAFE_EVIDENCE_SCOPE_FIELDS: tuple[str, ...] = (
    "electrode",
    "cell_configuration",
)


def scientific_node_slug(value: str) -> str:
    raw = value.strip().casefold()
    # 科学条件中的符号具有语义，至少保留数值正负号，避免 -20 °C 与 20 °C
    # 被规约成同一个图节点。普通化学名中的连字符仍按分隔符处理。
    if re.match(r"^-\s*\d", raw):
        raw = "minus_" + raw[1:].lstrip()
    elif re.match(r"^\+\s*\d", raw):
        raw = "plus_" + raw[1:].lstrip()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_")
    return text or "unnamed"


def evidence_chunk_artifact_id(chunk_id: str) -> str:
    """Return the stable graph id used for one evidence chunk."""

    return f"scientific_evidence_chunk:{scientific_node_slug(chunk_id)}"


def scientific_relation_artifact_id(subject: str, relation: str, object_: str) -> str:
    """Return a stable identity for a typed scientific relation instance."""

    return (
        "scientific_relation:"
        f"{scientific_node_slug(subject)}:{scientific_node_slug(relation)}:"
        f"{scientific_node_slug(object_)}"
    )


def add_kg_relation_provenance_projection(
    graph: MultiplexHarnessGraph,
    *,
    subject: str,
    relation: str,
    object_: str,
    source_paper_id: str,
    evidence_chunk_ids: Iterable[str],
    confidence: float | None = None,
    provenance: str = "kg-v1:relation-provenance",
) -> str:
    """Project one admitted KG relation and its exact source chunks.

    This is deliberately stricter than entity co-mention. Callers must provide a
    relation source paper and chunks that are both in that source and support the
    relation. The relation remains read-only; the returned node is only an impact seed.
    """

    relation_node = scientific_relation_artifact_id(subject, relation, object_)
    graph.add_artifact(
        HarnessArtifact(
            artifact_id=relation_node,
            kind="scientific_relation",
            label=f"{subject} -[{relation}]-> {object_}",
            metadata={
                "source_paper_id": source_paper_id,
                "authority": "impact_seed_only",
                "confidence": "" if confidence is None else str(confidence),
            },
        )
    )
    for chunk_id in dict.fromkeys(str(item).strip() for item in evidence_chunk_ids if str(item).strip()):
        chunk_node = evidence_chunk_artifact_id(chunk_id)
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=chunk_node,
                kind="scientific_evidence_chunk",
                label=chunk_id,
                metadata={"authority": "retrieval_only", "paper_id": source_paper_id},
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=relation_node,
                target=chunk_node,
                layer=CouplingLayer.SCIENTIFIC,
                weight=1.25,
                directed=True,
                relation="supported_by_exact_chunk",
                provenance=provenance,
            )
        )
    return relation_node


def add_kg_query_results_projection(
    graph: MultiplexHarnessGraph,
    *,
    query_id: str,
    results: Iterable[Mapping[str, Any]],
    context_field: str | None = None,
    default_weight: float = 0.8,
    provenance: str = "kg_rag.service_kg_query.KgQueryService",
) -> None:
    """将白名单 KG 查询结果加入 scientific coupling layer。

    ``context_field`` 用于把任务发出的结构化查询锚定到同一个规范科学字段，例如
    `find_materials_for_condition(-20 °C)` 可显式声明 ``temperature_c``。这是查询合同
    元数据，不是从结果文本猜出来的。
    """

    if not query_id.strip():
        raise ValueError("query_id is required")
    if default_weight <= 0:
        raise ValueError("default_weight must be positive")

    query_node = f"kg_query:{query_id}"
    graph.add_artifact(
        HarnessArtifact(
            artifact_id=query_node,
            kind="scientific_kg_query",
            label=query_id,
            metadata={"authority": "retrieval_only"},
        )
    )
    if context_field:
        field_id = f"scientific_context_field:{context_field}"
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=field_id,
                kind="scientific_context_field",
                label=context_field,
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=query_node,
                target=field_id,
                layer=CouplingLayer.SCIENTIFIC,
                weight=1.0,
                directed=True,
                relation="queries_context_field",
                provenance=provenance,
            )
        )

    for index, result in enumerate(results):
        start = str(result.get("start_entity") or "").strip()
        end = str(result.get("end_entity") or "").strip()
        relation = str(result.get("relation") or "RELATED_TO").strip() or "RELATED_TO"
        if not start or not end:
            continue
        start_id = f"scientific_entity:{scientific_node_slug(start)}"
        end_id = f"scientific_entity:{scientific_node_slug(end)}"
        labels = result.get("entity_labels") or []
        label_text = ",".join(str(item) for item in labels)
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=start_id,
                kind="scientific_entity",
                label=start,
                metadata={"kg_labels": label_text, "authority": "retrieval_only"},
            )
        )
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=end_id,
                kind="scientific_entity_or_condition",
                label=end,
                metadata={"authority": "retrieval_only"},
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=query_node,
                target=start_id,
                layer=CouplingLayer.SCIENTIFIC,
                weight=default_weight,
                relation="returned_entity",
                provenance=f"{provenance}:{query_id}:{index}",
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=start_id,
                target=end_id,
                layer=CouplingLayer.SCIENTIFIC,
                weight=default_weight,
                directed=True,
                relation=relation,
                provenance=f"{provenance}:{query_id}:{index}",
            )
        )
        for chunk_id in result.get("chunk_ids") or []:
            chunk_text = str(chunk_id).strip()
            if not chunk_text:
                continue
            chunk_node = evidence_chunk_artifact_id(chunk_text)
            graph.add_artifact(
                HarnessArtifact(
                    artifact_id=chunk_node,
                    kind="scientific_evidence_chunk",
                    label=chunk_text,
                    metadata={"authority": "retrieval_only"},
                )
            )
            graph.add_edge(
                CouplingEdge(
                    source=start_id,
                    target=chunk_node,
                    layer=CouplingLayer.SCIENTIFIC,
                    weight=0.7 * default_weight,
                    relation="supported_by_chunk",
                    provenance=f"{provenance}:{query_id}:{index}",
                )
            )


def add_kg_evidence_scope_projection(
    graph: MultiplexHarnessGraph,
    *,
    task_id: str,
    evidence_id: str,
    scope: Mapping[str, Any] | Any,
    admitted_fields: Iterable[str] = SAFE_EVIDENCE_SCOPE_FIELDS,
    provenance: str = "orion.task_evidence.condition_extraction.extract_condition_scope",
) -> tuple[str, ...]:
    """把 KG 回链证据中的高精度条件观察投影到 scientific layer。

    V0 不把任意 regex 抽取结果都升级为 Harness coupling。当前只允许预声明的
    categorical fields（默认 ``electrode`` / ``cell_configuration``）进入强 scientific
    path；temperature 由结构化 KG query contract 提供，rate/cycle/measurement 等更易受
    PDF figure labels 或局部文本歧义影响的字段暂不自动晋升。

    返回实际新增的规范字段名，便于 benchmark 审计。
    """

    task_node = f"task:{task_id}"
    if task_node not in graph.artifacts:
        raise ValueError(f"task projection must exist before KG evidence projection: {task_id}")
    if not evidence_id.strip():
        raise ValueError("evidence_id is required")

    if hasattr(scope, "model_dump"):
        payload = dict(scope.model_dump())
    elif isinstance(scope, Mapping):
        payload = dict(scope)
    else:
        raise TypeError("scope must be a mapping or expose model_dump()")

    allowed = {str(field).strip() for field in admitted_fields if str(field).strip()}
    if not allowed:
        return ()

    chunk_node = evidence_chunk_artifact_id(evidence_id)
    graph.add_artifact(
        HarnessArtifact(
            artifact_id=chunk_node,
            kind="scientific_evidence_chunk",
            label=evidence_id,
            metadata={"authority": "retrieval_only"},
        )
    )
    graph.add_edge(
        CouplingEdge(
            source=task_node,
            target=chunk_node,
            layer=CouplingLayer.SCIENTIFIC,
            weight=0.9,
            directed=True,
            relation="kg_evidence_supports_change",
            provenance=provenance,
        )
    )

    added: list[str] = []
    for field_name in sorted(allowed):
        value = payload.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        field_node = f"scientific_context_field:{field_name}"
        observation_node = (
            f"scientific_context_observation:{scientific_node_slug(evidence_id)}:"
            f"{scientific_node_slug(field_name)}:{scientific_node_slug(str(value))}"
        )
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=field_node,
                kind="scientific_context_field",
                label=field_name,
            )
        )
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=observation_node,
                kind="scientific_context_observation",
                label=f"{field_name}={value}",
                metadata={
                    "field": field_name,
                    "value": str(value),
                    "authority": "retrieval_only",
                },
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=chunk_node,
                target=observation_node,
                layer=CouplingLayer.SCIENTIFIC,
                weight=1.0,
                directed=True,
                relation="extracts_context_observation",
                provenance=provenance,
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=observation_node,
                target=field_node,
                layer=CouplingLayer.SCIENTIFIC,
                weight=1.0,
                directed=True,
                relation="instantiates_context_field",
                provenance=provenance,
            )
        )
        added.append(field_name)
    return tuple(added)
