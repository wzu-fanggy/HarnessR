"""Change Domain V0 的只读结构提取器。

这些提取器只把 ORION 已有结构投影为 ``MultiplexHarnessGraph``；不调用 LLM、
不写生产 KG。V0 允许逻辑上统一为一张图，但物理上继续读取不同数据源。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from orion.capvm.model import VirtualCapability
from orion.core.design_decision import TaskContract

from .models import CouplingEdge, CouplingLayer, HarnessArtifact, MultiplexHarnessGraph


def _normalized_source_file(value: str) -> str:
    path = value.replace("\\\\", "/").replace("\\", "/").lstrip("./")
    return path[6:] if path.startswith("orion/") else path


def link_source_symbol_views(graph: MultiplexHarnessGraph) -> int:
    """Link extractor-specific nodes that denote the exact same source symbol.

    Graphify, state AST and scientific-field AST intentionally keep their own node IDs.  This
    bridge uses only normalized source-file identity plus the exact source start line; it does not
    use label similarity or embeddings.  The returned count is useful for development audits.
    """

    by_location: dict[tuple[str, str], list[str]] = {}
    for artifact_id, artifact in graph.artifacts.items():
        source_file = str(artifact.metadata.get("source_file") or "")
        source_location = str(artifact.metadata.get("source_location") or "")
        if not source_file or not source_location.startswith("L"):
            continue
        key = (_normalized_source_file(source_file), source_location)
        by_location.setdefault(key, []).append(artifact_id)

    added = 0
    for (source_file, source_location), ids in by_location.items():
        graphify_ids = [item for item in ids if item.startswith("graphify:")]
        ast_ids = [item for item in ids if item.startswith("pyfunc:")]
        for graphify_id in graphify_ids:
            for ast_id in ast_ids:
                edge = CouplingEdge(
                    source=graphify_id,
                    target=ast_id,
                    layer=CouplingLayer.BEHAVIOR,
                    weight=1.5,
                    directed=False,
                    relation="same_source_symbol",
                    provenance=f"exact-source-location:{source_file}:{source_location}",
                )
                before = len(graph.edges)
                graph.add_edge(edge)
                if len(graph.edges) > before:
                    added += 1
    return added


def add_capability_dependencies(
    graph: MultiplexHarnessGraph,
    capabilities: Iterable[VirtualCapability],
) -> None:
    """从 ``VirtualCapability.dependencies`` 自动加入 capability/service 图。"""

    items = list(capabilities)
    for capability in items:
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=f"capability:{capability.capability_id}",
                kind=f"capability:{capability.kind.value}",
                label=capability.summary,
                metadata={
                    "pinned_min_level": capability.pinned_min_level.name,
                    "maximum_level": capability.maximum_level.name,
                    "dependency_levels": json.dumps(
                        {
                            dependency.capability_id: dependency.required_level.name
                            for dependency in capability.dependencies
                        },
                        sort_keys=True,
                    ),
                },
            )
        )
    known = {item.capability_id for item in items}
    for capability in items:
        for dependency in capability.dependencies:
            # 允许依赖尚未出现在本次投影中；创建占位逻辑 capability，而不是静默丢边。
            if dependency.capability_id not in known:
                graph.add_artifact(
                    HarnessArtifact(
                        artifact_id=f"capability:{dependency.capability_id}",
                        kind="capability:external_or_unprojected",
                        label=dependency.capability_id,
                    )
                )
            graph.add_edge(
                CouplingEdge(
                    source=f"capability:{capability.capability_id}",
                    target=f"capability:{dependency.capability_id}",
                    layer=CouplingLayer.SERVICE,
                    weight=1.0,
                    directed=True,
                    relation="depends_on",
                    provenance="capvm.VirtualCapability.dependencies",
                )
            )


def add_task_contract_projection(
    graph: MultiplexHarnessGraph,
    *,
    task_id: str,
    contract: TaskContract,
) -> None:
    """把结构化任务合同投影为 task/constraint/context-field 图。"""

    task_node = f"task:{task_id}"
    graph.add_artifact(
        HarnessArtifact(
            artifact_id=task_node,
            kind="task_contract",
            label=f"{contract.task_type}:{task_id}",
        )
    )
    for constraint in contract.hard_constraints:
        constraint_node = f"constraint:{task_id}:{constraint.constraint_id}"
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=constraint_node,
                kind="scientific_constraint",
                label=constraint.description,
                metadata={"evidence_scope": constraint.required_evidence_scope},
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=task_node,
                target=constraint_node,
                layer=CouplingLayer.BEHAVIOR,
                weight=1.0,
                directed=True,
                relation="requires_constraint",
                provenance="core.design_decision.TaskContract",
            )
        )
        for evidence_kind in constraint.required_evidence_kinds:
            kind_node = f"evidence_kind:{evidence_kind}"
            graph.add_artifact(
                HarnessArtifact(
                    artifact_id=kind_node,
                    kind="evidence_kind",
                    label=evidence_kind,
                )
            )
            graph.add_edge(
                CouplingEdge(
                    source=constraint_node,
                    target=kind_node,
                    layer=CouplingLayer.SCIENTIFIC,
                    weight=0.9,
                    directed=True,
                    relation="requires_evidence_kind",
                    provenance="ConstraintSpec.required_evidence_kinds",
                )
            )

    for requirement in contract.joint_context_requirements:
        group_node = f"context_group:{task_id}:{requirement.group_id}"
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=group_node,
                kind="joint_context_requirement",
                label=requirement.group_id,
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=task_node,
                target=group_node,
                layer=CouplingLayer.BEHAVIOR,
                weight=1.0,
                directed=True,
                relation="requires_joint_context",
                provenance="core.design_decision.JointContextRequirement",
            )
        )
        for constraint_id in requirement.constraint_ids:
            constraint_node = f"constraint:{task_id}:{constraint_id}"
            if constraint_node in graph.artifacts:
                graph.add_edge(
                    CouplingEdge(
                        source=group_node,
                        target=constraint_node,
                        layer=CouplingLayer.STATE,
                        weight=1.0,
                        relation="joins_constraint",
                        provenance="JointContextRequirement.constraint_ids",
                    )
                )
        for field in requirement.context_fields:
            field_node = f"scientific_context_field:{field}"
            graph.add_artifact(
                HarnessArtifact(
                    artifact_id=field_node,
                    kind="scientific_context_field",
                    label=field,
                )
            )
            graph.add_edge(
                CouplingEdge(
                    source=group_node,
                    target=field_node,
                    layer=CouplingLayer.SCIENTIFIC,
                    weight=1.1,
                    directed=True,
                    relation="requires_context_field",
                    provenance="JointContextRequirement.context_fields",
                )
            )


def load_graphify_static_projection(
    graph: MultiplexHarnessGraph,
    *,
    graph_json: str | Path,
    source_prefixes: Sequence[str] = ("orion/",),
    code_only: bool = True,
) -> None:
    """从现有 ``graphify-out/graph.json`` 导入 AST 静态结构边。

    只消费 graphify 已抽取的关系，不做语义搜索。未知 relation 会被忽略，避免把
    模糊相似性误当作 evolution coupling。
    """

    payload = json.loads(Path(graph_json).read_text(encoding="utf-8"))
    nodes = {str(node["id"]): node for node in payload.get("nodes", [])}

    def in_scope(node: dict[str, object]) -> bool:
        source_file = str(node.get("source_file") or "").replace("\\", "/")
        if code_only and str(node.get("file_type") or "") != "code":
            return False
        return any(source_file.startswith(prefix) for prefix in source_prefixes)

    selected = {node_id: node for node_id, node in nodes.items() if in_scope(node)}
    for node_id, node in selected.items():
        source_file = str(node.get("source_file") or "")
        label = str(node.get("label") or node_id)
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=f"graphify:{node_id}",
                kind="source_artifact",
                label=label,
                metadata={
                    "source_file": source_file,
                    "source_location": str(node.get("source_location") or ""),
                },
            )
        )

    relation_layer = {
        "calls": CouplingLayer.CONTROL,
        "call": CouplingLayer.CONTROL,
        "imports": CouplingLayer.SERVICE,
        "import": CouplingLayer.SERVICE,
        "reads": CouplingLayer.STATE,
        "read": CouplingLayer.STATE,
        "writes": CouplingLayer.STATE,
        "write": CouplingLayer.STATE,
        "references": CouplingLayer.BEHAVIOR,
        "reference": CouplingLayer.BEHAVIOR,
    }
    for link in payload.get("links", []):
        source_id = str(link.get("source") or "")
        target_id = str(link.get("target") or "")
        if source_id not in selected or target_id not in selected:
            continue
        relation = str(link.get("relation") or "").casefold()
        layer = relation_layer.get(relation)
        if layer is None:
            continue
        source_file = str(selected[source_id].get("source_file") or "").replace("\\", "/")
        target_file = str(selected[target_id].get("source_file") or "").replace("\\", "/")
        if "/tests/" in f"/{source_file}" or source_file.startswith("orion/tests/"):
            layer = CouplingLayer.REGRESSION
        elif "/tests/" in f"/{target_file}" or target_file.startswith("orion/tests/"):
            layer = CouplingLayer.REGRESSION
        graph.add_edge(
            CouplingEdge(
                source=f"graphify:{source_id}",
                target=f"graphify:{target_id}",
                layer=layer,
                weight=float(link.get("weight") or 1.0),
                directed=relation in {"calls", "call", "imports", "import", "reads", "read", "writes", "write"},
                relation=relation,
                provenance=(
                    f"graphify:{link.get('source_file', '')}:{link.get('source_location', '')}"
                ),
            )
        )
