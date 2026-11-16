"""Structured scientific-task projection for the graph-native Harness view.

This module turns an already structured task scope into graph objects. It does not infer
scientific meaning from free text and does not query the production KG. The projection is the
bridge used by composition experiments: one task can point to the scientific context dimensions
that its Harness execution must cover, while the same context-field nodes are independently
connected to source/runtime responsibilities by the other extractors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .kg_projection import scientific_node_slug
from .models import CouplingEdge, CouplingLayer, HarnessArtifact, MultiplexHarnessGraph


@dataclass(frozen=True)
class ScientificTaskProjection:
    task_id: str
    task_node_id: str
    obligation_node_ids: tuple[str, ...]
    field_node_ids: tuple[str, ...]
    value_node_ids: tuple[str, ...]


@dataclass(frozen=True)
class TaskResponsibilityProjection:
    """Typed task->field->Harness responsibility projection.

    Runtime responsibilities and regression responsibilities are kept separate even when both are
    present in the same unified graph.  This lets task composition stay small while evolution and
    validation queries can still traverse the tests that guard the same scientific dimensions.
    """

    task_id: str
    strong_by_field: Mapping[str, tuple[str, ...]]
    weak_by_field: Mapping[str, tuple[str, ...]]
    regression_strong_by_field: Mapping[str, tuple[str, ...]]
    regression_weak_by_field: Mapping[str, tuple[str, ...]]
    selected_strong_ids: tuple[str, ...]
    explanation_paths: Mapping[str, tuple[str, ...]]

    @property
    def fields_with_strong_responsibility(self) -> tuple[str, ...]:
        return tuple(sorted(field for field, ids in self.strong_by_field.items() if ids))

    @property
    def fields_without_strong_responsibility(self) -> tuple[str, ...]:
        return tuple(sorted(field for field, ids in self.strong_by_field.items() if not ids))


def _render_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " | ".join(str(item) for item in value)
    return str(value)


def add_scientific_task_projection(
    graph: MultiplexHarnessGraph,
    *,
    task_id: str,
    task_text: str,
    target_scope: Mapping[str, Any],
    relevant_fields: Sequence[str],
) -> ScientificTaskProjection:
    """Project a structured task into per-field scientific obligations.

    The task-to-field path is deliberately typed::

        Task -> ScientificObligation -> ScientificContextField

    A concrete value is represented as a task-local value node for explanation, rather than being
    stored on the global context-field node. This prevents two tasks with different temperatures,
    cell configurations, etc. from collapsing into one value identity.
    """

    if not task_id.strip():
        raise ValueError("task_id is required")
    if not task_text.strip():
        raise ValueError("task_text is required")
    fields = tuple(dict.fromkeys(str(field).strip() for field in relevant_fields if str(field).strip()))
    if not fields:
        raise ValueError("relevant_fields must not be empty")

    task_slug = scientific_node_slug(task_id)
    task_node = f"task:{task_slug}"
    graph.add_artifact(
        HarnessArtifact(
            artifact_id=task_node,
            kind="scientific_task",
            label=task_text,
            metadata={"task_id": task_id},
        )
    )

    obligation_nodes: list[str] = []
    field_nodes: list[str] = []
    value_nodes: list[str] = []
    for field_name in fields:
        field_node = f"scientific_context_field:{field_name}"
        obligation_node = f"scientific_obligation:{task_slug}:{field_name}"
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=field_node,
                kind="scientific_context_field",
                label=field_name,
            )
        )
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=obligation_node,
                kind="scientific_obligation",
                label=f"{task_id}:{field_name}",
                metadata={"field": field_name},
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=task_node,
                target=obligation_node,
                layer=CouplingLayer.BEHAVIOR,
                weight=1.0,
                directed=True,
                relation="requires_scientific_obligation",
                provenance="structured-task-scope",
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=obligation_node,
                target=field_node,
                layer=CouplingLayer.SCIENTIFIC,
                weight=1.25,
                directed=True,
                relation="requires_context_field",
                provenance="structured-task-scope:relevant_fields",
            )
        )
        obligation_nodes.append(obligation_node)
        field_nodes.append(field_node)

        value = target_scope.get(field_name)
        if value is None:
            continue
        value_node = f"scientific_context_value:{task_slug}:{field_name}"
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=value_node,
                kind="scientific_context_value",
                label=_render_value(value),
                metadata={"field": field_name, "value": _render_value(value)},
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=obligation_node,
                target=value_node,
                layer=CouplingLayer.SCIENTIFIC,
                weight=0.15,
                directed=True,
                relation="binds_context_value",
                provenance="structured-task-scope:target_scope",
            )
        )
        value_nodes.append(value_node)

    return ScientificTaskProjection(
        task_id=task_id,
        task_node_id=task_node,
        obligation_node_ids=tuple(obligation_nodes),
        field_node_ids=tuple(field_nodes),
        value_node_ids=tuple(value_nodes),
    )


def project_task_responsibilities(
    graph: MultiplexHarnessGraph,
    projection: ScientificTaskProjection,
) -> TaskResponsibilityProjection:
    """Follow the legal task meta-path to Harness scientific responsibilities.

    Only edges emitted by the deterministic scientific-field extractor are considered here.
    Strong responsibility declarations are kept separate from ordinary field references; weak
    references are reported as diagnostic context and are not silently promoted into the selected
    execution/evolution surface.
    """

    strong: dict[str, tuple[str, ...]] = {}
    weak: dict[str, tuple[str, ...]] = {}
    regression_strong: dict[str, tuple[str, ...]] = {}
    regression_weak: dict[str, tuple[str, ...]] = {}
    paths: dict[str, tuple[str, ...]] = {}
    obligation_by_field = {
        node.rsplit(":", 1)[-1]: node for node in projection.obligation_node_ids
    }

    def is_regression_artifact(artifact_id: str) -> bool:
        artifact = graph.artifacts.get(artifact_id)
        if artifact is None:
            return False
        source_file = str(artifact.metadata.get("source_file") or "").replace("\\", "/")
        return source_file.startswith("tests/") or "/tests/" in f"/{source_file}"

    for field_node in projection.field_node_ids:
        field_name = field_node.split(":", 1)[1]
        strong_all = sorted(
            {
                edge.target
                for edge in graph.edges
                if edge.source == field_node
                and edge.layer is CouplingLayer.SCIENTIFIC
                and edge.relation == "declares_scientific_responsibility"
            }
        )
        weak_all = sorted(
            {
                edge.target
                for edge in graph.edges
                if edge.source == field_node
                and edge.layer is CouplingLayer.SCIENTIFIC
                and edge.relation == "references_scientific_field"
            }
        )
        strong_ids = [item for item in strong_all if not is_regression_artifact(item)]
        weak_ids = [item for item in weak_all if not is_regression_artifact(item)]
        regression_strong_ids = [item for item in strong_all if is_regression_artifact(item)]
        regression_weak_ids = [item for item in weak_all if is_regression_artifact(item)]
        strong[field_name] = tuple(strong_ids)
        weak[field_name] = tuple(weak_ids)
        regression_strong[field_name] = tuple(regression_strong_ids)
        regression_weak[field_name] = tuple(regression_weak_ids)
        obligation = obligation_by_field[field_name]
        for artifact_id in strong_ids:
            paths[artifact_id] = (
                projection.task_node_id,
                obligation,
                field_node,
                artifact_id,
            )

    selected = tuple(sorted({item for ids in strong.values() for item in ids}))
    return TaskResponsibilityProjection(
        task_id=projection.task_id,
        strong_by_field=strong,
        weak_by_field=weak,
        regression_strong_by_field=regression_strong,
        regression_weak_by_field=regression_weak,
        selected_strong_ids=selected,
        explanation_paths=paths,
    )
