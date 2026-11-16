"""Compile a typed semantic Harness graph into an executable LangGraph.

This module deliberately keeps LangGraph as the execution engine.  It does not replace LangGraph's
state/checkpoint/runtime semantics; it resolves the higher-level Harness object model that already
exists in the multiplex graph:

    Skill step -> ToolIdentity -> Capability -> dependency closure

The resolved semantic slice is compiled into a concrete ordered LangGraph program.  Transactions are
copy-on-write and fail closed: invalid provider cardinality, duplicate Skill order, or capability
dependency cycles never mutate the caller's graph.
"""

from __future__ import annotations

import copy
import hashlib
import json
import operator
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Mapping, TypedDict

from langgraph.graph import END, START, StateGraph

from orion.capvm.catalog import CapabilityCatalog
from orion.capvm.model import (
    CapabilityDependency,
    ResidencyLevel,
    ResourceVector,
    VirtualCapability,
)
from orion.evolution.change_domain.models import (
    CouplingEdge,
    CouplingLayer,
    HarnessArtifact,
    MultiplexHarnessGraph,
)


class SemanticHarnessExecutionState(TypedDict, total=False):
    trace: Annotated[list[str], operator.add]
    capability_trace: Annotated[list[str], operator.add]
    payload: dict[str, Any]


StepExecutor = Callable[[str, Mapping[str, Any]], Mapping[str, Any] | None]
BeforeStepHook = Callable[["CompiledHarnessStep", Mapping[str, Any]], None]
ProfileToolExecutor = Callable[["CompiledToolBinding", Mapping[str, Any]], Mapping[str, Any] | None]


@dataclass(frozen=True)
class CompiledHarnessStep:
    step_id: str
    order: int
    tool_name: str
    tool_node_id: str
    capability_id: str
    dependency_closure_ids: tuple[str, ...]
    residency_requirements: tuple[tuple[str, str], ...]
    required: bool


@dataclass(frozen=True)
class CompiledHarnessSpec:
    skill_name: str
    graph_digest: str
    active_optional_step_ids: tuple[str, ...]
    steps: tuple[CompiledHarnessStep, ...]

    @property
    def tool_sequence(self) -> tuple[str, ...]:
        return tuple(step.tool_name for step in self.steps)

    @property
    def capability_union(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    capability_id
                    for step in self.steps
                    for capability_id in step.dependency_closure_ids
                }
            )
        )

    @property
    def residency_union(self) -> tuple[tuple[str, str], ...]:
        levels: dict[str, ResidencyLevel] = {}
        for step in self.steps:
            for capability_id, level_name in step.residency_requirements:
                level = ResidencyLevel[level_name]
                levels[capability_id] = max(levels.get(capability_id, ResidencyLevel.CATALOG), level)
        return tuple(sorted((capability_id, level.name) for capability_id, level in levels.items()))


@dataclass(frozen=True)
class CompiledToolBinding:
    step_id: str
    order: int
    tool_name: str
    tool_node_id: str
    capability_id: str
    residency_requirements: tuple[tuple[str, str], ...]
    required: bool

    @property
    def dependency_closure_ids(self) -> tuple[str, ...]:
        return tuple(capability_id for capability_id, _ in self.residency_requirements)


@dataclass(frozen=True)
class CompiledHarnessProfile:
    skill_name: str
    graph_digest: str
    enabled_optional_step_ids: tuple[str, ...]
    bindings: tuple[CompiledToolBinding, ...]

    @property
    def allowed_tool_names(self) -> tuple[str, ...]:
        return tuple(binding.tool_name for binding in self.bindings)

    @property
    def required_tool_names(self) -> tuple[str, ...]:
        return tuple(binding.tool_name for binding in self.bindings if binding.required)

    @property
    def capability_union(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    capability_id
                    for binding in self.bindings
                    for capability_id in binding.dependency_closure_ids
                }
            )
        )

    def binding_for_tool(self, tool_name: str) -> CompiledToolBinding:
        matches = tuple(binding for binding in self.bindings if binding.tool_name == tool_name)
        if len(matches) != 1:
            raise KeyError(f"tool must have exactly one binding in profile: {tool_name}: {len(matches)}")
        return matches[0]


class SemanticAgentLoopState(TypedDict, total=False):
    script: list[str]
    cursor: int
    pending_tool: str
    trace: Annotated[list[str], operator.add]
    capability_trace: Annotated[list[str], operator.add]
    denied_tools: Annotated[list[str], operator.add]
    payload: dict[str, Any]


@dataclass(frozen=True)
class SemanticHarnessValidation:
    skill_step_count: int
    tool_provider_count: int
    capability_dependency_edge_count: int
    skill_names: tuple[str, ...]


class SemanticHarnessBudgetExceeded(RuntimeError):
    def __init__(
        self,
        *,
        step_id: str,
        cost: ResourceVector,
        budget: ResourceVector,
        violated_dimensions: tuple[str, ...],
    ) -> None:
        super().__init__(
            "compiled_step_exceeds_budget: "
            f"step={step_id}, dimensions={','.join(violated_dimensions)}"
        )
        self.step_id = step_id
        self.cost = cost
        self.budget = budget
        self.violated_dimensions = violated_dimensions


def _residency_requirements_declared_cost(
    residency_requirements: tuple[tuple[str, str], ...],
    catalog: CapabilityCatalog,
) -> ResourceVector:
    total = ResourceVector()
    for capability_id, level_name in residency_requirements:
        capability = catalog.get(capability_id)
        total = total + capability.adapter.declared_cost(ResidencyLevel[level_name])
    return total


def compiled_step_declared_cost(
    step: CompiledHarnessStep,
    catalog: CapabilityCatalog,
) -> ResourceVector:
    return _residency_requirements_declared_cost(step.residency_requirements, catalog)


def compiled_tool_binding_declared_cost(
    binding: CompiledToolBinding,
    catalog: CapabilityCatalog,
) -> ResourceVector:
    return _residency_requirements_declared_cost(binding.residency_requirements, catalog)


def compiled_spec_peak_declared_cost(
    spec: CompiledHarnessSpec,
    catalog: CapabilityCatalog,
) -> ResourceVector:
    peak = {dimension: 0.0 for dimension in (
        "prompt_tokens",
        "host_ram_mb",
        "process_slots",
        "connection_slots",
    )}
    for step in spec.steps:
        cost = compiled_step_declared_cost(step, catalog)
        for dimension in peak:
            peak[dimension] = max(peak[dimension], cost.value(dimension))
    return ResourceVector(**peak)


def compiled_profile_peak_declared_cost(
    profile: CompiledHarnessProfile,
    catalog: CapabilityCatalog,
) -> ResourceVector:
    peak = {dimension: 0.0 for dimension in (
        "prompt_tokens",
        "host_ram_mb",
        "process_slots",
        "connection_slots",
    )}
    for binding in profile.bindings:
        cost = compiled_tool_binding_declared_cost(binding, catalog)
        for dimension in peak:
            peak[dimension] = max(peak[dimension], cost.value(dimension))
    return ResourceVector(**peak)


def validate_compiled_profile_budget(
    profile: CompiledHarnessProfile,
    catalog: CapabilityCatalog,
    budget: ResourceVector,
) -> ResourceVector:
    """Fail closed if any allowed agent-loop tool closure exceeds the hard budget."""

    for binding in profile.bindings:
        cost = compiled_tool_binding_declared_cost(binding, catalog)
        violations = cost.violated_dimensions(budget)
        if violations:
            raise SemanticHarnessBudgetExceeded(
                step_id=binding.step_id,
                cost=cost,
                budget=budget,
                violated_dimensions=violations,
            )
    return compiled_profile_peak_declared_cost(profile, catalog)


def validate_compiled_spec_budget(
    spec: CompiledHarnessSpec,
    catalog: CapabilityCatalog,
    budget: ResourceVector,
) -> ResourceVector:
    """Fail closed if any sequential step's active dependency closure exceeds the hard budget."""

    for step in spec.steps:
        cost = compiled_step_declared_cost(step, catalog)
        violations = cost.violated_dimensions(budget)
        if violations:
            raise SemanticHarnessBudgetExceeded(
                step_id=step.step_id,
                cost=cost,
                budget=budget,
                violated_dimensions=violations,
            )
    return compiled_spec_peak_declared_cost(spec, catalog)


def compile_capability_catalog_view(
    graph: MultiplexHarnessGraph,
    base_catalog: CapabilityCatalog,
) -> CapabilityCatalog:
    """Rebuild CVM dependency descriptors from the semantic graph.

    Adapter/resource implementations remain the same physical providers from ``base_catalog``;
    dependency identity and required residency levels come from the graph.  This removes the
    otherwise dangerous split-brain condition where a committed semantic transaction changes the
    graph but CVM continues to follow stale descriptor dependencies.
    """

    validate_semantic_harness_graph(graph)
    rebuilt: list[VirtualCapability] = []
    for capability in base_catalog.all():
        node = f"capability:{capability.capability_id}"
        if node not in graph.artifacts:
            raise ValueError(f"base capability missing from semantic graph: {capability.capability_id}")
        level_map = _dependency_level_map(graph, node)
        rebuilt.append(
            VirtualCapability(
                capability_id=capability.capability_id,
                kind=capability.kind,
                summary=capability.summary,
                adapter=capability.adapter,
                dependencies=tuple(
                    CapabilityDependency(dependency_id, required_level)
                    for dependency_id, required_level in sorted(level_map.items())
                ),
                pinned_min_level=capability.pinned_min_level,
                maximum_level=capability.maximum_level,
                metadata=dict(capability.metadata),
            )
        )
    return CapabilityCatalog(rebuilt)


def semantic_graph_digest(graph: MultiplexHarnessGraph) -> str:
    artifacts = [
        {
            "id": artifact_id,
            "kind": artifact.kind,
            "label": artifact.label,
            "metadata": sorted((str(key), str(value)) for key, value in artifact.metadata.items()),
            "roles": list(artifact.roles),
        }
        for artifact_id, artifact in sorted(graph.artifacts.items())
    ]
    edges = [
        {
            "source": edge.source,
            "target": edge.target,
            "layer": edge.layer.value,
            "weight": float(edge.weight),
            "directed": bool(edge.directed),
            "relation": edge.relation,
            "provenance": edge.provenance,
        }
        for edge in sorted(
            graph.edges,
            key=lambda item: (
                item.source,
                item.target,
                item.layer.value,
                item.relation,
                float(item.weight),
                item.provenance,
            ),
        )
    ]
    raw = json.dumps(
        {"artifacts": artifacts, "edges": edges},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _bool_metadata(value: object, *, key: str, node_id: str) -> bool:
    text = str(value).strip().casefold()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{node_id} requires boolean metadata {key}=true|false")


def _skill_step_nodes(graph: MultiplexHarnessGraph) -> tuple[str, ...]:
    return tuple(
        sorted(
            artifact_id
            for artifact_id, artifact in graph.artifacts.items()
            if artifact.kind == "skill_workflow_step"
            or artifact.has_role("skill_workflow_step")
        )
    )


def _capability_nodes(graph: MultiplexHarnessGraph) -> tuple[str, ...]:
    return tuple(sorted(node for node in graph.artifacts if node.startswith("capability:")))


def _outgoing(
    graph: MultiplexHarnessGraph,
    source: str,
    relation: str,
) -> tuple[CouplingEdge, ...]:
    return tuple(edge for edge in graph.edges if edge.source == source and edge.relation == relation)


def _incoming(
    graph: MultiplexHarnessGraph,
    target: str,
    relation: str,
) -> tuple[CouplingEdge, ...]:
    return tuple(edge for edge in graph.edges if edge.target == target and edge.relation == relation)


def _dependency_level_map(
    graph: MultiplexHarnessGraph,
    capability_node: str,
) -> dict[str, ResidencyLevel]:
    artifact = graph.artifacts.get(capability_node)
    if artifact is None:
        raise ValueError(f"unknown capability node: {capability_node}")
    raw = str(artifact.metadata.get("dependency_levels") or "{}").strip() or "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid dependency_levels metadata: {capability_node}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"dependency_levels must be an object: {capability_node}")
    levels: dict[str, ResidencyLevel] = {}
    for dependency_id, level_name in payload.items():
        try:
            levels[str(dependency_id)] = ResidencyLevel[str(level_name)]
        except KeyError as exc:
            raise ValueError(
                f"invalid dependency residency level: {capability_node}:{dependency_id}:{level_name}"
            ) from exc
    return levels


def _capability_residency_requirements(
    graph: MultiplexHarnessGraph,
    capability_node: str,
    *,
    root_level: ResidencyLevel = ResidencyLevel.BOUND,
) -> tuple[tuple[str, str], ...]:
    requirements: dict[str, ResidencyLevel] = {}
    visiting: list[str] = []

    def visit(node: str, required_level: ResidencyLevel) -> None:
        if node in visiting:
            cycle = " -> ".join([*visiting, node])
            raise ValueError(f"capability dependency cycle: {cycle}")
        capability_id = node.removeprefix("capability:")
        requirements[capability_id] = max(
            requirements.get(capability_id, ResidencyLevel.CATALOG), required_level
        )
        visiting.append(node)
        level_map = _dependency_level_map(graph, node)
        dependency_edges = _outgoing(graph, node, "depends_on")
        edge_ids = {
            edge.target.removeprefix("capability:")
            for edge in dependency_edges
            if edge.target.startswith("capability:")
        }
        if edge_ids != set(level_map):
            raise ValueError(
                f"dependency edge/level mismatch: {node}: edges={sorted(edge_ids)} "
                f"levels={sorted(level_map)}"
            )
        for edge in dependency_edges:
            if not edge.target.startswith("capability:"):
                raise ValueError(f"capability dependency must target capability node: {edge}")
            dependency_id = edge.target.removeprefix("capability:")
            visit(edge.target, level_map[dependency_id])
        visiting.pop()

    visit(capability_node, root_level)
    return tuple(sorted((capability_id, level.name) for capability_id, level in requirements.items()))


def _capability_dependency_closure(
    graph: MultiplexHarnessGraph,
    capability_node: str,
) -> tuple[str, ...]:
    return tuple(
        capability_id
        for capability_id, _level_name in _capability_residency_requirements(
            graph, capability_node
        )
    )


def validate_semantic_harness_graph(
    graph: MultiplexHarnessGraph,
) -> SemanticHarnessValidation:
    step_nodes = _skill_step_nodes(graph)
    skill_orders: dict[str, set[int]] = {}
    provider_nodes: set[str] = set()

    for step_node in step_nodes:
        artifact = graph.artifacts[step_node]
        skill_name = str(artifact.metadata.get("skill") or "").strip()
        if not skill_name:
            raise ValueError(f"skill step missing skill metadata: {step_node}")
        try:
            order = int(str(artifact.metadata.get("order") or ""))
        except ValueError as exc:
            raise ValueError(f"skill step has invalid order: {step_node}") from exc
        orders = skill_orders.setdefault(skill_name, set())
        if order in orders:
            raise ValueError(f"duplicate skill step order: {skill_name}:{order}")
        orders.add(order)
        _bool_metadata(artifact.metadata.get("required"), key="required", node_id=step_node)

        tool_edges = _outgoing(graph, step_node, "uses_tool")
        if len(tool_edges) != 1:
            raise ValueError(
                f"skill step must use exactly one ToolIdentity: {step_node}: {len(tool_edges)}"
            )
        tool_node = tool_edges[0].target
        if not tool_node.startswith("tool_identity:"):
            raise ValueError(f"skill step uses non-tool identity: {step_node}: {tool_node}")
        providers = _incoming(graph, tool_node, "implemented_by_tool")
        providers = tuple(edge for edge in providers if edge.source.startswith("capability:"))
        if len(providers) != 1:
            raise ValueError(
                f"ToolIdentity must have exactly one capability provider: {tool_node}: {len(providers)}"
            )
        provider_nodes.add(providers[0].source)

    dependency_edges = tuple(
        edge
        for edge in graph.edges
        if edge.relation == "depends_on" and edge.source.startswith("capability:")
    )
    for capability_node in _capability_nodes(graph):
        _capability_dependency_closure(graph, capability_node)

    return SemanticHarnessValidation(
        skill_step_count=len(step_nodes),
        tool_provider_count=len(provider_nodes),
        capability_dependency_edge_count=len(dependency_edges),
        skill_names=tuple(sorted(skill_orders)),
    )


def resolve_compiled_skill_spec(
    graph: MultiplexHarnessGraph,
    *,
    skill_name: str,
    active_optional_step_ids: tuple[str, ...] = (),
) -> CompiledHarnessSpec:
    validate_semantic_harness_graph(graph)
    skill_node = f"skill:{skill_name}"
    if skill_node not in graph.artifacts:
        raise ValueError(f"unknown skill: {skill_name}")

    contained = {
        edge.target
        for edge in _outgoing(graph, skill_node, "contains_workflow_step")
        if edge.target in graph.artifacts
    }
    if not contained:
        raise ValueError(f"skill has no workflow steps: {skill_name}")

    rows: list[tuple[int, str, bool]] = []
    optional_ids: set[str] = set()
    for step_node in contained:
        artifact = graph.artifacts[step_node]
        step_id = artifact.label or step_node.rsplit(":", 1)[-1]
        required = _bool_metadata(
            artifact.metadata.get("required"), key="required", node_id=step_node
        )
        order = int(str(artifact.metadata.get("order") or "0"))
        rows.append((order, step_node, required))
        if not required:
            optional_ids.add(step_id)

    requested_optional = set(active_optional_step_ids)
    unknown_optional = requested_optional - optional_ids
    if unknown_optional:
        raise ValueError(
            f"unknown or non-optional step ids for {skill_name}: {sorted(unknown_optional)}"
        )

    steps: list[CompiledHarnessStep] = []
    for order, step_node, required in sorted(rows):
        artifact = graph.artifacts[step_node]
        step_id = artifact.label or step_node.rsplit(":", 1)[-1]
        if not required and step_id not in requested_optional:
            continue
        tool_edge = _outgoing(graph, step_node, "uses_tool")[0]
        tool_node = tool_edge.target
        provider_edge = tuple(
            edge
            for edge in _incoming(graph, tool_node, "implemented_by_tool")
            if edge.source.startswith("capability:")
        )[0]
        capability_node = provider_edge.source
        steps.append(
            CompiledHarnessStep(
                step_id=step_id,
                order=order,
                tool_name=tool_node.removeprefix("tool_identity:"),
                tool_node_id=tool_node,
                capability_id=capability_node.removeprefix("capability:"),
                dependency_closure_ids=_capability_dependency_closure(graph, capability_node),
                residency_requirements=_capability_residency_requirements(
                    graph, capability_node
                ),
                required=required,
            )
        )

    if not steps:
        raise ValueError(f"compiled skill contains no active steps: {skill_name}")
    return CompiledHarnessSpec(
        skill_name=skill_name,
        graph_digest=semantic_graph_digest(graph),
        active_optional_step_ids=tuple(sorted(requested_optional)),
        steps=tuple(steps),
    )


def resolve_skill_runtime_profile(
    graph: MultiplexHarnessGraph,
    *,
    skill_name: str,
    enabled_optional_step_ids: tuple[str, ...] | None = None,
) -> CompiledHarnessProfile:
    """Compile a Skill into an agent-loop tool/capability aperture rather than a one-pass DAG.

    Required steps are always present.  ``enabled_optional_step_ids=None`` means all declared optional
    steps are logically available; an explicit tuple restricts the aperture to that optional subset.
    Runtime order/repetition remains the responsibility of the stable agent loop and its policy.
    """

    validate_semantic_harness_graph(graph)
    skill_node = f"skill:{skill_name}"
    if skill_node not in graph.artifacts:
        raise ValueError(f"unknown skill: {skill_name}")
    contained = {
        edge.target
        for edge in _outgoing(graph, skill_node, "contains_workflow_step")
        if edge.target in graph.artifacts
    }
    if not contained:
        raise ValueError(f"skill has no workflow steps: {skill_name}")

    rows: list[tuple[int, str, str, bool]] = []
    optional_ids: set[str] = set()
    for step_node in contained:
        artifact = graph.artifacts[step_node]
        step_id = artifact.label or step_node.rsplit(":", 1)[-1]
        required = _bool_metadata(
            artifact.metadata.get("required"), key="required", node_id=step_node
        )
        order = int(str(artifact.metadata.get("order") or "0"))
        rows.append((order, step_id, step_node, required))
        if not required:
            optional_ids.add(step_id)

    if enabled_optional_step_ids is None:
        enabled = set(optional_ids)
    else:
        enabled = set(enabled_optional_step_ids)
        unknown = enabled - optional_ids
        if unknown:
            raise ValueError(
                f"unknown or non-optional step ids for {skill_name}: {sorted(unknown)}"
            )

    bindings: list[CompiledToolBinding] = []
    for order, step_id, step_node, required in sorted(rows):
        if not required and step_id not in enabled:
            continue
        tool_node = _outgoing(graph, step_node, "uses_tool")[0].target
        provider = tuple(
            edge
            for edge in _incoming(graph, tool_node, "implemented_by_tool")
            if edge.source.startswith("capability:")
        )[0].source
        bindings.append(
            CompiledToolBinding(
                step_id=step_id,
                order=order,
                tool_name=tool_node.removeprefix("tool_identity:"),
                tool_node_id=tool_node,
                capability_id=provider.removeprefix("capability:"),
                residency_requirements=_capability_residency_requirements(graph, provider),
                required=required,
            )
        )
    return CompiledHarnessProfile(
        skill_name=skill_name,
        graph_digest=semantic_graph_digest(graph),
        enabled_optional_step_ids=tuple(sorted(enabled)),
        bindings=tuple(bindings),
    )


def compile_agent_loop_from_profile(
    profile: CompiledHarnessProfile,
    *,
    executor: ProfileToolExecutor | None = None,
):
    """Compile the stable LangGraph kernel from an already-resolved Harness profile.

    Keeping this backend independent of graph resolution is intentional: a strong typed-config
    baseline can produce the same profile and use the exact same execution kernel.  Any measured
    difference must therefore come from the control-plane representation/resolution semantics, not
    from giving the graph arm a better LangGraph implementation.
    """

    bindings = {binding.tool_name: binding for binding in profile.bindings}
    workflow = StateGraph(SemanticAgentLoopState)

    def planner(state: SemanticAgentLoopState) -> dict[str, Any]:
        script = list(state.get("script") or [])
        cursor = int(state.get("cursor") or 0)
        return {"pending_tool": script[cursor] if cursor < len(script) else ""}

    def route(state: SemanticAgentLoopState) -> str:
        script = list(state.get("script") or [])
        cursor = int(state.get("cursor") or 0)
        if cursor >= len(script):
            return "end"
        pending = str(state.get("pending_tool") or "")
        return "tools" if pending in bindings else "deny"

    def run_tool(state: SemanticAgentLoopState) -> dict[str, Any]:
        tool_name = str(state.get("pending_tool") or "")
        binding = bindings[tool_name]
        update: dict[str, Any] = {
            "cursor": int(state.get("cursor") or 0) + 1,
            "pending_tool": "",
            "trace": [tool_name],
            "capability_trace": list(binding.dependency_closure_ids),
        }
        if executor is not None:
            extra = executor(binding, state)
            if extra:
                forbidden = {
                    "cursor",
                    "pending_tool",
                    "trace",
                    "capability_trace",
                    "denied_tools",
                } & set(extra)
                if forbidden:
                    raise ValueError(
                        f"executor may not overwrite agent-loop control fields: {sorted(forbidden)}"
                    )
                update.update(dict(extra))
        return update

    def deny_tool(state: SemanticAgentLoopState) -> dict[str, Any]:
        tool_name = str(state.get("pending_tool") or "")
        return {
            "cursor": int(state.get("cursor") or 0) + 1,
            "pending_tool": "",
            "denied_tools": [tool_name],
        }

    workflow.add_node("planner", planner)
    workflow.add_node("tools", run_tool)
    workflow.add_node("deny", deny_tool)
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        route,
        {"tools": "tools", "deny": "deny", "end": END},
    )
    workflow.add_edge("tools", "planner")
    workflow.add_edge("deny", END)
    return workflow.compile()


def compile_semantic_skill_agent_loop(
    graph: MultiplexHarnessGraph,
    *,
    skill_name: str,
    enabled_optional_step_ids: tuple[str, ...] | None = None,
    executor: ProfileToolExecutor | None = None,
):
    """Resolve a semantic graph profile and attach it to the stable LangGraph agent loop."""

    profile = resolve_skill_runtime_profile(
        graph,
        skill_name=skill_name,
        enabled_optional_step_ids=enabled_optional_step_ids,
    )
    return compile_agent_loop_from_profile(profile, executor=executor), profile


def compile_semantic_skill_to_langgraph(
    graph: MultiplexHarnessGraph,
    *,
    skill_name: str,
    active_optional_step_ids: tuple[str, ...] = (),
    executor: StepExecutor | None = None,
    before_step: BeforeStepHook | None = None,
):
    """Resolve one Skill semantic slice and compile it into a sequential LangGraph program.

    The compiler is intentionally execution-engine agnostic above the final LangGraph step.  The
    optional ``executor`` supplies actual step behavior; when omitted the program remains executable
    and records the resolved ToolIdentity/capability trace, which is useful for architecture tests.
    """

    spec = resolve_compiled_skill_spec(
        graph,
        skill_name=skill_name,
        active_optional_step_ids=active_optional_step_ids,
    )
    workflow = StateGraph(SemanticHarnessExecutionState)

    for step in spec.steps:
        def run_step(
            state: SemanticHarnessExecutionState,
            *,
            _step: CompiledHarnessStep = step,
        ) -> dict[str, Any]:
            if before_step is not None:
                before_step(_step, state)
            update: dict[str, Any] = {
                "trace": [_step.tool_name],
                "capability_trace": list(_step.dependency_closure_ids),
            }
            if executor is not None:
                extra = executor(_step.tool_name, state)
                if extra:
                    forbidden = {"trace", "capability_trace"} & set(extra)
                    if forbidden:
                        raise ValueError(
                            f"executor may not overwrite compiler trace fields: {sorted(forbidden)}"
                        )
                    update.update(dict(extra))
            return update

        workflow.add_node(step.step_id, run_step)

    first = spec.steps[0].step_id
    workflow.add_edge(START, first)
    for left, right in zip(spec.steps, spec.steps[1:]):
        workflow.add_edge(left.step_id, right.step_id)
    workflow.add_edge(spec.steps[-1].step_id, END)
    return workflow.compile(), spec


class SemanticHarnessTransaction:
    """Copy-on-write semantic Harness graph transaction.

    The original graph is never modified.  ``commit`` validates the staged graph and returns it;
    discarding the transaction is rollback.
    """

    def __init__(self, graph: MultiplexHarnessGraph) -> None:
        self._base_graph = graph
        self._base_digest = semantic_graph_digest(graph)
        self._staged = copy.deepcopy(graph)
        self._committed = False

    @property
    def base_digest(self) -> str:
        return self._base_digest

    @property
    def staged_graph(self) -> MultiplexHarnessGraph:
        return self._staged

    def set_skill_step_required(
        self,
        *,
        skill_name: str,
        step_id: str,
        required: bool,
    ) -> None:
        node_id = f"skill_step:{skill_name}:{step_id}"
        artifact = self._staged.artifacts.get(node_id)
        if artifact is None:
            raise ValueError(f"unknown skill step: {node_id}")
        metadata = dict(artifact.metadata)
        metadata["required"] = "true" if required else "false"
        self._staged.artifacts[node_id] = HarnessArtifact(
            artifact_id=artifact.artifact_id,
            kind=artifact.kind,
            label=artifact.label,
            metadata=metadata,
            roles=artifact.roles,
        )

    def add_capability_dependency(
        self,
        *,
        capability_id: str,
        dependency_id: str,
        required_level: ResidencyLevel = ResidencyLevel.INTERFACE,
        weight: float = 1.0,
    ) -> None:
        source = f"capability:{capability_id}"
        target = f"capability:{dependency_id}"
        if source not in self._staged.artifacts or target not in self._staged.artifacts:
            raise ValueError(f"unknown capability dependency: {source} -> {target}")
        source_artifact = self._staged.artifacts[source]
        levels = _dependency_level_map(self._staged, source)
        levels[dependency_id] = required_level
        metadata = dict(source_artifact.metadata)
        metadata["dependency_levels"] = json.dumps(
            {key: value.name for key, value in sorted(levels.items())},
            sort_keys=True,
        )
        self._staged.artifacts[source] = HarnessArtifact(
            artifact_id=source_artifact.artifact_id,
            kind=source_artifact.kind,
            label=source_artifact.label,
            metadata=metadata,
            roles=source_artifact.roles,
        )
        exists = any(
            edge.source == source
            and edge.target == target
            and edge.relation == "depends_on"
            for edge in self._staged.edges
        )
        if not exists:
            self._staged.add_edge(
                CouplingEdge(
                    source=source,
                    target=target,
                    layer=CouplingLayer.SERVICE,
                    weight=weight,
                    directed=True,
                    relation="depends_on",
                    provenance="semantic-harness-transaction",
                )
            )

    def remove_capability_dependency(
        self,
        *,
        capability_id: str,
        dependency_id: str,
    ) -> None:
        source = f"capability:{capability_id}"
        target = f"capability:{dependency_id}"
        source_artifact = self._staged.artifacts.get(source)
        if source_artifact is None:
            raise ValueError(f"unknown capability: {source}")
        self._staged.edges = [
            edge
            for edge in self._staged.edges
            if not (
                edge.source == source
                and edge.target == target
                and edge.relation == "depends_on"
            )
        ]
        levels = _dependency_level_map(self._staged, source)
        levels.pop(dependency_id, None)
        metadata = dict(source_artifact.metadata)
        metadata["dependency_levels"] = json.dumps(
            {key: value.name for key, value in sorted(levels.items())},
            sort_keys=True,
        )
        self._staged.artifacts[source] = HarnessArtifact(
            artifact_id=source_artifact.artifact_id,
            kind=source_artifact.kind,
            label=source_artifact.label,
            metadata=metadata,
            roles=source_artifact.roles,
        )

    def commit(
        self,
        *,
        preflight: Callable[[MultiplexHarnessGraph], None] | None = None,
    ) -> MultiplexHarnessGraph:
        if self._committed:
            raise RuntimeError("transaction already committed")
        validate_semantic_harness_graph(self._staged)
        if preflight is not None:
            preflight(self._staged)
        self._committed = True
        return self._staged

    def rollback(self) -> MultiplexHarnessGraph:
        return self._base_graph
