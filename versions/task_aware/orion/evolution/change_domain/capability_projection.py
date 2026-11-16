"""Capability-contract projection for the graph-native Scientific Harness.

Runtime composition should not infer plugin semantics from source-code token occurrence.  A
capability explicitly declares the task roles and scientific context dimensions it supports; the
multiplex graph then links those contracts to task obligations and to the automatically recovered
program/runtime/evolution structure.

The contract representation reuses ``VirtualCapability.metadata`` so the same objects can be handed
to the existing capvm catalog/governor.  This module adds no new routing model and calls no LLM.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from orion.capvm.model import VirtualCapability

from .extractors import add_capability_dependencies
from .kg_projection import scientific_node_slug
from .models import CouplingEdge, CouplingLayer, HarnessArtifact, MultiplexHarnessGraph


@dataclass(frozen=True)
class CapabilityContractProjection:
    capability_ids: tuple[str, ...]
    role_node_ids: tuple[str, ...]
    field_node_ids: tuple[str, ...]
    implementation_links: Mapping[str, tuple[str, ...]]
    tool_provider_links: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class CapabilityWorkingSetProjection:
    task_id: str
    selected_capability_ids: tuple[str, ...]
    dependency_closure_ids: tuple[str, ...]
    explanation_paths: Mapping[str, tuple[str, ...]]
    field_coverage: Mapping[str, tuple[str, ...]]

    @property
    def materialization_ids(self) -> tuple[str, ...]:
        return self.dependency_closure_ids


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return (text,) if text else ()


def role_node_id(role: str) -> str:
    return f"task_role:{scientific_node_slug(role)}"


def _implementation_refs(capability: VirtualCapability) -> tuple[str, ...]:
    return _string_tuple(capability.metadata.get("implementation_refs"))


def _match_graphify_implementation(
    graph: MultiplexHarnessGraph,
    ref: str,
) -> str:
    """Resolve ``source_file::symbol_label`` against the exact Graphify projection."""

    if "::" not in ref:
        raise ValueError(f"implementation ref must be source_file::symbol_label: {ref}")
    source_file, symbol_label = (part.strip().replace("\\", "/") for part in ref.split("::", 1))
    matches = [
        artifact_id
        for artifact_id, artifact in graph.artifacts.items()
        if artifact_id.startswith("graphify:")
        and str(artifact.metadata.get("source_file") or "").replace("\\", "/") == source_file
        and artifact.label == symbol_label
    ]
    if len(matches) != 1:
        raise ValueError(f"implementation ref must resolve exactly once: {ref}: {matches}")
    return matches[0]


def add_capability_contract_projection(
    graph: MultiplexHarnessGraph,
    capabilities: Iterable[VirtualCapability],
) -> CapabilityContractProjection:
    """Project explicit runtime capability contracts into the multiplex graph.

    Supported metadata keys:

    ``task_roles``
        Stable role identifiers used by task composition, e.g. ``evidence_retrieval``.
    ``scientific_fields``
        Canonical scientific-context fields the capability can consume/validate.
    """

    items = tuple(capabilities)
    add_capability_dependencies(graph, items)
    role_nodes: set[str] = set()
    field_nodes: set[str] = set()
    implementation_links: dict[str, tuple[str, ...]] = {}
    tool_provider_links: dict[str, tuple[str, ...]] = {}

    for capability in items:
        capability_node = f"capability:{capability.capability_id}"
        roles = _string_tuple(capability.metadata.get("task_roles"))
        fields = _string_tuple(capability.metadata.get("scientific_fields"))

        for role in roles:
            role_node = role_node_id(role)
            role_nodes.add(role_node)
            graph.add_artifact(
                HarnessArtifact(
                    artifact_id=role_node,
                    kind="task_role",
                    label=role,
                )
            )
            graph.add_edge(
                CouplingEdge(
                    source=role_node,
                    target=capability_node,
                    layer=CouplingLayer.BEHAVIOR,
                    weight=1.0,
                    directed=True,
                    relation="provided_by_capability",
                    provenance="VirtualCapability.metadata.task_roles",
                )
            )

        implementation_nodes: list[str] = []
        for ref in _implementation_refs(capability):
            implementation_node = _match_graphify_implementation(graph, ref)
            implementation_nodes.append(implementation_node)
            graph.add_edge(
                CouplingEdge(
                    source=capability_node,
                    target=implementation_node,
                    layer=CouplingLayer.SERVICE,
                    weight=1.4,
                    directed=True,
                    relation="implemented_by",
                    provenance=f"VirtualCapability.metadata.implementation_refs:{ref}",
                )
            )
        implementation_links[capability.capability_id] = tuple(sorted(implementation_nodes))
        tool_nodes = sorted(
            {
                edge.source
                for edge in graph.edges
                if edge.relation == "wraps_tool_symbol"
                and edge.target in implementation_nodes
                and edge.source.startswith("tool_identity:")
            }
        )
        for tool_node in tool_nodes:
            graph.add_edge(
                CouplingEdge(
                    source=capability_node,
                    target=tool_node,
                    layer=CouplingLayer.SERVICE,
                    weight=1.5,
                    directed=True,
                    relation="implemented_by_tool",
                    provenance="capability-implementation-ref+tool-identity",
                )
            )
        tool_provider_links[capability.capability_id] = tuple(tool_nodes)

        for field_name in fields:
            field_node = f"scientific_context_field:{field_name}"
            field_nodes.add(field_node)
            graph.add_artifact(
                HarnessArtifact(
                    artifact_id=field_node,
                    kind="scientific_context_field",
                    label=field_name,
                )
            )
            graph.add_edge(
                CouplingEdge(
                    source=field_node,
                    target=capability_node,
                    layer=CouplingLayer.SCIENTIFIC,
                    weight=0.9,
                    directed=True,
                    relation="supported_by_capability",
                    provenance="VirtualCapability.metadata.scientific_fields",
                )
            )

    return CapabilityContractProjection(
        capability_ids=tuple(sorted(item.capability_id for item in items)),
        role_node_ids=tuple(sorted(role_nodes)),
        field_node_ids=tuple(sorted(field_nodes)),
        implementation_links=implementation_links,
        tool_provider_links=tool_provider_links,
    )


def add_task_role_requirements(
    graph: MultiplexHarnessGraph,
    *,
    task_node_id: str,
    required_roles: Iterable[str],
) -> tuple[str, ...]:
    """Attach explicit task-role requirements to an existing task node."""

    if task_node_id not in graph.artifacts:
        raise ValueError(f"unknown task node: {task_node_id}")
    nodes: list[str] = []
    for role in dict.fromkeys(str(item).strip() for item in required_roles if str(item).strip()):
        role_node = role_node_id(role)
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=role_node,
                kind="task_role",
                label=role,
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=task_node_id,
                target=role_node,
                layer=CouplingLayer.BEHAVIOR,
                weight=1.2,
                directed=True,
                relation="requires_task_role",
                provenance="structured-task-role-contract",
            )
        )
        nodes.append(role_node)
    return tuple(nodes)


def _dependency_closure_from_selected(
    graph: MultiplexHarnessGraph,
    *,
    selected: set[str],
    paths: dict[str, tuple[str, ...]],
    task_node_id: str,
) -> set[str]:
    closure = set(selected)
    queue: deque[str] = deque(sorted(selected))
    while queue:
        capability_node = queue.popleft()
        for edge in graph.edges:
            if (
                edge.source == capability_node
                and edge.relation == "depends_on"
                and edge.target.startswith("capability:")
                and edge.target not in closure
            ):
                closure.add(edge.target)
                queue.append(edge.target)
                parent_path = paths.get(capability_node, (task_node_id, capability_node))
                paths[edge.target] = parent_path + (edge.target,)
    return closure


def _field_coverage_for_closure(
    graph: MultiplexHarnessGraph,
    *,
    scientific_field_node_ids: Iterable[str],
    closure: set[str],
) -> dict[str, tuple[str, ...]]:
    field_coverage: dict[str, tuple[str, ...]] = {}
    for field_node in scientific_field_node_ids:
        field_name = field_node.split(":", 1)[1]
        providers = tuple(
            sorted(
                {
                    edge.target
                    for edge in graph.edges
                    if edge.source == field_node
                    and edge.relation == "supported_by_capability"
                    and edge.target in closure
                }
            )
        )
        field_coverage[field_name] = providers
    return field_coverage


def project_tool_capability_working_set(
    graph: MultiplexHarnessGraph,
    *,
    task_id: str,
    task_node_id: str,
    tool_node_ids: Iterable[str],
    scientific_field_node_ids: Iterable[str] = (),
) -> CapabilityWorkingSetProjection:
    """Project capabilities from an activated Skill ToolIdentity set.

    Each active tool must have exactly one capability provider in V0.  Tool provider competition is
    a separate routing/admissibility concern and must not be resolved by arbitrary graph order.
    """

    if task_node_id not in graph.artifacts:
        raise ValueError(f"unknown task node: {task_node_id}")
    selected: set[str] = set()
    paths: dict[str, tuple[str, ...]] = {}
    for tool_node in tool_node_ids:
        providers = sorted(
            {
                edge.source
                for edge in graph.edges
                if edge.target == tool_node
                and edge.relation == "implemented_by_tool"
                and edge.source.startswith("capability:")
            }
        )
        if len(providers) != 1:
            raise ValueError(
                f"active skill tool must have exactly one capability provider: {tool_node}: {providers}"
            )
        provider = providers[0]
        selected.add(provider)
        paths[provider] = (task_node_id, tool_node, provider)

    closure = _dependency_closure_from_selected(
        graph,
        selected=selected,
        paths=paths,
        task_node_id=task_node_id,
    )
    field_coverage = _field_coverage_for_closure(
        graph,
        scientific_field_node_ids=scientific_field_node_ids,
        closure=closure,
    )
    return CapabilityWorkingSetProjection(
        task_id=task_id,
        selected_capability_ids=tuple(sorted(item.removeprefix("capability:") for item in selected)),
        dependency_closure_ids=tuple(sorted(item.removeprefix("capability:") for item in closure)),
        explanation_paths={
            node.removeprefix("capability:"): path for node, path in paths.items()
        },
        field_coverage=field_coverage,
    )


def project_capability_working_set(
    graph: MultiplexHarnessGraph,
    *,
    task_id: str,
    task_node_id: str,
    required_role_node_ids: Iterable[str],
    scientific_field_node_ids: Iterable[str] = (),
) -> CapabilityWorkingSetProjection:
    """Project providers for required task roles and close over capability dependencies.

    V0 deliberately requires a unique provider per role.  Ambiguous provider choice belongs to the
    existing routing/admissibility layer; silently picking one here would hide a routing problem.
    """

    if task_node_id not in graph.artifacts:
        raise ValueError(f"unknown task node: {task_node_id}")

    selected: set[str] = set()
    paths: dict[str, tuple[str, ...]] = {}
    for role_node in required_role_node_ids:
        providers = sorted(
            {
                edge.target
                for edge in graph.edges
                if edge.source == role_node
                and edge.relation == "provided_by_capability"
                and edge.target.startswith("capability:")
            }
        )
        if len(providers) != 1:
            raise ValueError(
                f"task role must have exactly one provider in V0: {role_node}: {providers}"
            )
        provider = providers[0]
        selected.add(provider)
        paths[provider] = (task_node_id, role_node, provider)

    closure = _dependency_closure_from_selected(
        graph,
        selected=selected,
        paths=paths,
        task_node_id=task_node_id,
    )
    field_coverage = _field_coverage_for_closure(
        graph,
        scientific_field_node_ids=scientific_field_node_ids,
        closure=closure,
    )

    return CapabilityWorkingSetProjection(
        task_id=task_id,
        selected_capability_ids=tuple(sorted(item.removeprefix("capability:") for item in selected)),
        dependency_closure_ids=tuple(sorted(item.removeprefix("capability:") for item in closure)),
        explanation_paths={
            node.removeprefix("capability:"): path for node, path in paths.items()
        },
        field_coverage=field_coverage,
    )
