"""Development-only end-to-end composition probe for the R6 graph-native Scientific Harness.

The probe connects existing ORION layers rather than testing an isolated primitive:

* explicit capability contracts (DeepSeek/Pi-style composition semantics);
* a typed task/scientific graph;
* the current multiplex program/state/regression/scientific graph;
* the existing capvm catalog/governor for bounded residency.

Inputs combine eight electrolyte tasks frozen for Study 4C on 2026-08-22 with three diverse task
families reused from the earlier V9 routing development bank plus an unsupported-task family.  The
experiment compares eager exposure, a size-matched metadata router, and typed contract-graph
projection.  It is a development gate, not confirmatory evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from orion.benchmark.evolvable_plugin_capability_routing_v9_dev import build_tasks as build_v9_tasks
from orion.benchmark.graph_native_scientific_composition_v0 import build_composed_harness_graph
from orion.benchmark.self_evolution_study4c_v1 import source_bank, study4c_cases
from orion.capvm.catalog import CapabilityCatalog
from orion.capvm.governor import CapabilityResidencyGovernor
from orion.capvm.model import (
    CapabilityDependency,
    CapabilityKind,
    ResidencyLevel,
    ResourceVector,
    VirtualCapability,
)
from orion.capvm.routing import CapabilityDescriptor, MetadataDiscoveryBackend, RoutingRequest
from orion.evolution.change_domain import (
    HarnessArtifact,
    add_capability_contract_projection,
    add_scientific_task_projection,
    add_task_role_requirements,
    project_capability_working_set,
)


INVENTORY_SIZE = 256
STUDY4C_ROLES = (
    "evidence_retrieval",
    "context_extraction",
    "applicability_evaluation",
    "design_screening",
    "decision_synthesis",
)
V9_FAMILY_ROLE = {
    "plain_match": "kg_query",
    "authority_fallback": "report_drafting",
    "resource_fallback": "simulation",
    "no_match_recovery": "quantum_transition_search",
}
SUPPORTED_V9_FAMILIES = frozenset(V9_FAMILY_ROLE)


class ContractAdapter:
    """Deterministic bounded-cost adapter for composition/residency integration."""

    def __init__(self, *, prompt_tokens: float = 24.0, host_ram_mb: float = 0.5) -> None:
        self.prompt_tokens = float(prompt_tokens)
        self.host_ram_mb = float(host_ram_mb)
        self.promotions = 0
        self.demotions = 0

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level <= ResidencyLevel.CATALOG:
            return ResourceVector()
        if level < ResidencyLevel.BOUND:
            return ResourceVector(prompt_tokens=self.prompt_tokens)
        return ResourceVector(prompt_tokens=self.prompt_tokens, host_ram_mb=self.host_ram_mb)

    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:
        return self.declared_cost(level)

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        del current, target
        self.promotions += 1

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        del current, target
        self.demotions += 1

    def close(self) -> None:
        return None


def _capability(
    capability_id: str,
    summary: str,
    *,
    roles: tuple[str, ...],
    scientific_fields: tuple[str, ...] = (),
    terms: tuple[str, ...] = (),
    dependencies: tuple[CapabilityDependency, ...] = (),
    implementation_refs: tuple[str, ...] = (),
    prompt_tokens: float = 24.0,
    host_ram_mb: float = 0.5,
) -> VirtualCapability:
    return VirtualCapability(
        capability_id=capability_id,
        kind=CapabilityKind.PLUGIN,
        summary=summary,
        adapter=ContractAdapter(prompt_tokens=prompt_tokens, host_ram_mb=host_ram_mb),
        dependencies=dependencies,
        maximum_level=ResidencyLevel.BOUND,
        metadata={
            "task_roles": roles,
            "scientific_fields": scientific_fields,
            "routing_terms": terms,
            "implementation_refs": implementation_refs,
            "routing_required_level": "BOUND",
            "version": "graph-native-composition-v1-dev",
        },
    )


def build_capability_catalog() -> CapabilityCatalog:
    condition_fields = (
        "temperature_c",
        "electrode",
        "cell_configuration",
        "voltage_window_v",
        "rate",
        "cycle_count",
        "measurement_definition",
        "salt",
        "solvent",
        "additive",
        "composition",
    )
    capabilities: list[VirtualCapability] = [
        _capability(
            "kg-structured-query",
            "retrieve structured sodium battery knowledge graph and literature evidence",
            roles=("kg_query", "evidence_retrieval"),
            scientific_fields=condition_fields,
            terms=("knowledge", "graph", "structured", "query", "evidence", "sodium", "battery", "electrolyte"),
            implementation_refs=(
                "tools/rag/kg_rag_tools.py::search_literature_multiview()",
                "tools/rag/kg_rag_tools.py::search_literature_chunks()",
                "tools/rag/kg_rag_tools.py::search_paper_chunks()",
                "tools/rag/kg_rag_tools.py::resolve_literature_papers()",
                "tools/kg_query.py::kg_query()",
            ),
        ),
        _capability(
            "condition-observation",
            "extract source-bound experimental conditions and measurement context",
            roles=("context_extraction",),
            scientific_fields=condition_fields,
            terms=("extract", "experimental", "condition", "measurement", "temperature", "cell", "electrode"),
            implementation_refs=(
                "task_evidence/condition_extraction.py::extract_condition_scope()",
            ),
        ),
        _capability(
            "strict-applicability",
            "evaluate scientific evidence applicability across structured source and target contexts",
            roles=("applicability_evaluation",),
            scientific_fields=condition_fields,
            terms=("strict", "provenance", "scientific", "evidence", "applicability", "context"),
            dependencies=(
                CapabilityDependency("condition-observation", ResidencyLevel.INTERFACE),
            ),
            implementation_refs=(
                "task_evidence/evidence_applicability.py::evaluate_applicability_input()",
            ),
        ),
        _capability(
            "numeric-property-extraction",
            "extract source-bound numeric electrolyte properties from retrieved evidence",
            roles=("numeric_extraction",),
            scientific_fields=(
                "temperature_c",
                "voltage_window_v",
                "rate",
                "cycle_count",
                "measurement_definition",
            ),
            terms=("extract", "numeric", "property", "temperature", "voltage", "rate", "cycle", "evidence"),
            implementation_refs=(
                "tools/electrolyte_design/properties.py::extract_numeric_properties()",
            ),
        ),
        _capability(
            "formulation-query",
            "query candidate sodium-ion electrolyte formulation space",
            roles=("formulation_query",),
            scientific_fields=("salt", "solvent", "additive", "composition"),
            terms=("query", "formulation", "candidate", "electrolyte", "salt", "solvent", "additive"),
            implementation_refs=(
                "tools/electrolyte_design/formulate.py::query_formulation_space()",
            ),
        ),
        _capability(
            "evidence-enrichment",
            "enrich candidate formulation evidence with exact source text and metadata",
            roles=("evidence_enrichment",),
            scientific_fields=condition_fields,
            terms=("enrich", "candidate", "formulation", "evidence", "source", "metadata"),
            implementation_refs=(
                "tools/electrolyte_design/enrichment.py::enrich_formulation_evidence()",
            ),
        ),
        _capability(
            "formulation-screening",
            "screen sodium-ion electrolyte formulations against structured scientific constraints",
            roles=("design_screening",),
            scientific_fields=(
                "temperature_c",
                "electrode",
                "cell_configuration",
                "voltage_window_v",
                "rate",
                "cycle_count",
                "salt",
                "solvent",
                "additive",
                "composition",
            ),
            terms=("electrolyte", "formulation", "screen", "design", "constraint", "temperature", "voltage"),
            dependencies=(
                CapabilityDependency("formulation-query", ResidencyLevel.INTERFACE),
            ),
            implementation_refs=(
                "tools/electrolyte_design/screen.py::screen_formulations()",
            ),
        ),
        _capability(
            "formulation-optimization",
            "translate structured electrolyte screening decisions into bounded optimization actions",
            roles=("formulation_optimization",),
            scientific_fields=("temperature_c", "voltage_window_v", "composition"),
            terms=("optimize", "formulation", "decision", "action", "electrolyte"),
            dependencies=(
                CapabilityDependency("formulation-screening", ResidencyLevel.INTERFACE),
            ),
            implementation_refs=(
                "tools/electrolyte_design/optimize.py::optimize_formulations()",
            ),
        ),
        _capability(
            "grounded-synthesis",
            "compile an evidence-grounded scientific design decision and report artifact",
            roles=("decision_synthesis",),
            scientific_fields=(
                "measurement_definition",
                "temperature_c",
                "electrode",
                "cell_configuration",
            ),
            terms=("grounded", "scientific", "decision", "report", "evidence", "design"),
            dependencies=(
                CapabilityDependency("strict-applicability", ResidencyLevel.INTERFACE),
                CapabilityDependency("formulation-screening", ResidencyLevel.INTERFACE),
            ),
            implementation_refs=(
                "tools/electrolyte_design/synthesis.py::synthesize_design_report()",
            ),
        ),
        _capability(
            "report-drafter",
            "draft scientific report content without filesystem side effects",
            roles=("report_drafting",),
            terms=("draft", "scientific", "report", "write", "content"),
        ),
        _capability(
            "light-molecular-simulation",
            "lightweight molecular simulation surrogate for electrolyte behavior",
            roles=("simulation",),
            scientific_fields=("temperature_c", "voltage_window_v", "composition"),
            terms=("lightweight", "surrogate", "molecular", "simulation", "electrolyte", "behavior", "accurate"),
            prompt_tokens=32.0,
            host_ram_mb=1.0,
        ),
    ]

    distractor_count = INVENTORY_SIZE - len(capabilities)
    for index in range(distractor_count):
        capabilities.append(
            _capability(
                f"archive-helper-{index:03d}",
                "generic archive formatting conversion helper",
                roles=(f"archive_helper_{index:03d}",),
                terms=("archive", "format", "conversion", f"helper{index}"),
                prompt_tokens=20.0,
                host_ram_mb=0.25,
            )
        )
    return CapabilityCatalog(capabilities)


@dataclass(frozen=True)
class CompositionTask:
    task_id: str
    family: str
    text: str
    required_roles: tuple[str, ...]
    scientific_fields: tuple[str, ...] = ()
    target_scope: dict[str, object] | None = None
    expected_supported: bool = True


def build_composition_tasks() -> tuple[CompositionTask, ...]:
    bank = source_bank()
    families = dict(bank["families"])
    tasks: list[CompositionTask] = []
    for case in study4c_cases():
        raw = dict(families[case.family_id])
        tasks.append(
            CompositionTask(
                task_id=case.case_id,
                family="study4c_electrolyte_workflow",
                text=case.prompt,
                required_roles=STUDY4C_ROLES,
                scientific_fields=tuple(str(item) for item in raw.get("relevant_fields_override") or ()),
                target_scope=dict(raw.get("target_scope_override") or {}),
                expected_supported=True,
            )
        )

    for task in build_v9_tasks():
        if task.family not in SUPPORTED_V9_FAMILIES:
            continue
        role = V9_FAMILY_ROLE[task.family]
        tasks.append(
            CompositionTask(
                task_id=f"v9::{task.task_id}",
                family=f"v9_{task.family}",
                text=task.text,
                required_roles=(role,),
                expected_supported=(task.family != "no_match_recovery"),
            )
        )
    return tuple(tasks)


def _generic_task_node(graph, task: CompositionTask) -> str:
    task_node = "task:" + task.task_id.replace(":", "_").replace(" ", "_").casefold()
    graph.add_artifact(
        HarnessArtifact(
            artifact_id=task_node,
            kind="task",
            label=task.text,
            metadata={"task_id": task.task_id},
        )
    )
    return task_node


def _roles_for_capability(capability: VirtualCapability) -> set[str]:
    raw = capability.metadata.get("task_roles") or ()
    if isinstance(raw, str):
        return {raw}
    return {str(item) for item in raw}


def _fields_for_capability(capability: VirtualCapability) -> set[str]:
    raw = capability.metadata.get("scientific_fields") or ()
    if isinstance(raw, str):
        return {raw}
    return {str(item) for item in raw}


def _coverage(required: Iterable[str], selected: Iterable[VirtualCapability], *, attr: str) -> float:
    required_set = {str(item) for item in required}
    if not required_set:
        return 1.0
    covered: set[str] = set()
    for capability in selected:
        covered.update(_roles_for_capability(capability) if attr == "role" else _fields_for_capability(capability))
    return len(required_set & covered) / len(required_set)


def _manual_profile_projection(
    task: CompositionTask,
    capabilities: tuple[VirtualCapability, ...],
) -> tuple[str, ...]:
    """Strong manual-profile baseline using the same explicit role contracts but no graph.

    This intentionally receives the same structured role requirements as the graph arm.  If it
    matches graph selection, that is expected: runtime role->provider selection is not claimed as a
    graph novelty.  The graph's extra value must come from cross-layer structure/evolution, not from
    hiding a weaker baseline.
    """

    by_role: dict[str, list[VirtualCapability]] = {}
    by_id = {item.capability_id: item for item in capabilities}
    for capability in capabilities:
        for role in _roles_for_capability(capability):
            by_role.setdefault(role, []).append(capability)
    selected: set[str] = set()
    for role in task.required_roles:
        providers = sorted(by_role.get(role, ()), key=lambda item: item.capability_id)
        if len(providers) != 1:
            raise ValueError(f"manual profile role provider ambiguity: {role}: {[p.capability_id for p in providers]}")
        selected.add(providers[0].capability_id)
    queue = list(sorted(selected))
    while queue:
        capability_id = queue.pop(0)
        for dependency in by_id[capability_id].dependencies:
            if dependency.capability_id not in selected:
                selected.add(dependency.capability_id)
                queue.append(dependency.capability_id)
    return tuple(sorted(selected))


def _resident_ids(governor: CapabilityResidencyGovernor) -> tuple[str, ...]:
    return tuple(
        sorted(
            state.capability_id
            for state in governor.states()
            if state.current_level > ResidencyLevel.CATALOG
        )
    )


def _resource_dict(vector: ResourceVector) -> dict[str, float]:
    return {
        "prompt_tokens": vector.prompt_tokens,
        "host_ram_mb": vector.host_ram_mb,
        "process_slots": vector.process_slots,
        "connection_slots": vector.connection_slots,
    }


def run_capability_composition_probe() -> dict[str, object]:
    catalog = build_capability_catalog()
    capabilities = tuple(catalog.all())
    descriptors = tuple(CapabilityDescriptor.from_capability(item) for item in capabilities)
    by_id = {item.capability_id: item for item in capabilities}
    graph, graph_diagnostics = build_composed_harness_graph()
    add_capability_contract_projection(graph, capabilities)

    budget = ResourceVector(prompt_tokens=160.0, host_ram_mb=4.0)
    governor = CapabilityResidencyGovernor(catalog=catalog, budget=budget)
    metadata_backend = MetadataDiscoveryBackend()
    tasks = build_composition_tasks()
    rows: list[dict[str, object]] = []
    peak_resident = 0
    peak_prompt = 0.0
    peak_ram = 0.0

    eager_cost = ResourceVector()
    for capability in capabilities:
        eager_cost = eager_cost + capability.adapter.declared_cost(ResidencyLevel.BOUND)

    try:
        for task in tasks:
            if task.scientific_fields:
                scientific_task = add_scientific_task_projection(
                    graph,
                    task_id=task.task_id,
                    task_text=task.text,
                    target_scope=task.target_scope or {},
                    relevant_fields=task.scientific_fields,
                )
                task_node = scientific_task.task_node_id
                field_nodes = scientific_task.field_node_ids
            else:
                task_node = _generic_task_node(graph, task)
                field_nodes = ()

            role_nodes = add_task_role_requirements(
                graph,
                task_node_id=task_node,
                required_roles=task.required_roles,
            )

            manual_supported = True
            manual_error = ""
            manual_ids: tuple[str, ...] = ()
            try:
                manual_ids = _manual_profile_projection(task, capabilities)
            except ValueError as exc:
                manual_supported = False
                manual_error = str(exc)

            graph_supported = True
            graph_error = ""
            graph_ids: tuple[str, ...] = ()
            graph_role_coverage = 0.0
            graph_field_coverage = 1.0 if not task.scientific_fields else 0.0
            graph_explanation_count = 0
            resident_inside: tuple[str, ...] = ()
            resident_cost = ResourceVector()
            try:
                working_set = project_capability_working_set(
                    graph,
                    task_id=task.task_id,
                    task_node_id=task_node,
                    required_role_node_ids=role_nodes,
                    scientific_field_node_ids=field_nodes,
                )
                graph_ids = working_set.materialization_ids
                selected_caps = tuple(by_id[item] for item in graph_ids)
                graph_role_coverage = _coverage(task.required_roles, selected_caps, attr="role")
                if task.scientific_fields:
                    graph_field_coverage = sum(
                        1 for field in task.scientific_fields if working_set.field_coverage.get(field)
                    ) / len(task.scientific_fields)
                graph_explanation_count = len(working_set.explanation_paths)
                requests = tuple((item, ResidencyLevel.BOUND) for item in graph_ids)
                resident_before = set(_resident_ids(governor))
                with governor.acquire_many(requests, request_id=f"composition::{task.task_id}"):
                    resident_inside = _resident_ids(governor)
                    resident_inside_set = set(resident_inside)
                    newly_materialized = tuple(sorted(resident_inside_set - resident_before))
                    cached_residents = tuple(sorted(resident_inside_set - set(graph_ids)))
                    resident_cost = governor.total_declared_cost()
                    peak_resident = max(peak_resident, len(resident_inside))
                    peak_prompt = max(peak_prompt, resident_cost.prompt_tokens)
                    peak_ram = max(peak_ram, resident_cost.host_ram_mb)
            except ValueError as exc:
                graph_supported = False
                graph_error = str(exc)
                newly_materialized = ()
                cached_residents = ()

            metadata_candidates = metadata_backend.discover(
                RoutingRequest(
                    task_text=task.text,
                    max_candidates=max(1, len(task.required_roles)),
                ),
                descriptors,
            )
            metadata_ids = tuple(item.capability_id for item in metadata_candidates)
            metadata_caps = tuple(by_id[item] for item in metadata_ids)
            metadata_role_coverage = _coverage(task.required_roles, metadata_caps, attr="role")
            metadata_field_coverage = _coverage(task.scientific_fields, metadata_caps, attr="field")

            rows.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "expected_supported": task.expected_supported,
                    "required_roles": list(task.required_roles),
                    "scientific_fields": list(task.scientific_fields),
                    "manual_supported": manual_supported,
                    "manual_error": manual_error,
                    "manual_profile_ids": list(manual_ids),
                    "manual_working_set_size": len(manual_ids),
                    "manual_equals_graph_working_set": set(manual_ids) == set(graph_ids),
                    "graph_supported": graph_supported,
                    "graph_error": graph_error,
                    "graph_working_set_ids": list(graph_ids),
                    "graph_working_set_size": len(graph_ids),
                    "graph_role_coverage": graph_role_coverage,
                    "graph_field_coverage": graph_field_coverage,
                    "graph_explanation_count": graph_explanation_count,
                    "resident_ids_inside": list(resident_inside),
                    "newly_materialized_ids": list(newly_materialized),
                    "cached_nonrequested_resident_ids": list(cached_residents),
                    "resident_declared_cost": _resource_dict(resident_cost),
                    "metadata_ids": list(metadata_ids),
                    "metadata_size": len(metadata_ids),
                    "metadata_role_coverage": metadata_role_coverage,
                    "metadata_field_coverage": metadata_field_coverage,
                    "eager_size": len(capabilities),
                    "graph_exposure_fraction": len(graph_ids) / len(capabilities),
                    "metadata_exposure_fraction": len(metadata_ids) / len(capabilities),
                    "correct_graph_support_decision": graph_supported == task.expected_supported,
                    "correct_manual_support_decision": manual_supported == task.expected_supported,
                }
            )
    finally:
        governor.close()

    supported_rows = [row for row in rows if row["expected_supported"]]
    unsupported_rows = [row for row in rows if not row["expected_supported"]]
    study_rows = [row for row in rows if row["family"] == "study4c_electrolyte_workflow"]

    def mean(rows_: list[dict[str, object]], key: str) -> float:
        return sum(float(row[key]) for row in rows_) / len(rows_) if rows_ else 0.0

    return {
        "schema_version": "graph-native-capability-composition-v1.0",
        "status": "development_only",
        "inventory_size": len(capabilities),
        "task_count": len(rows),
        "graph_base": {
            "artifact_count_before_task_projection": graph_diagnostics.artifact_count,
            "edge_count_before_task_projection": graph_diagnostics.edge_count,
            "layer_edge_counts": graph_diagnostics.layer_edge_counts,
            "same_symbol_links": graph_diagnostics.same_symbol_links,
        },
        "budget": _resource_dict(budget),
        "eager_declared_if_all_bound": _resource_dict(eager_cost),
        "aggregate": {
            "supported_task_count": len(supported_rows),
            "unsupported_task_count": len(unsupported_rows),
            "graph_support_decision_accuracy": mean(rows, "correct_graph_support_decision"),
            "manual_support_decision_accuracy": mean(rows, "correct_manual_support_decision"),
            "manual_graph_working_set_equivalence_rate": mean(rows, "manual_equals_graph_working_set"),
            "graph_mean_role_coverage_supported": mean(supported_rows, "graph_role_coverage"),
            "metadata_mean_role_coverage_supported": mean(supported_rows, "metadata_role_coverage"),
            "graph_mean_field_coverage_study4c": mean(study_rows, "graph_field_coverage"),
            "metadata_mean_field_coverage_study4c": mean(study_rows, "metadata_field_coverage"),
            "graph_mean_working_set_size_supported": mean(supported_rows, "graph_working_set_size"),
            "manual_mean_working_set_size_supported": mean(supported_rows, "manual_working_set_size"),
            "metadata_mean_candidate_size_supported": mean(supported_rows, "metadata_size"),
            "graph_mean_exposure_fraction_supported": mean(supported_rows, "graph_exposure_fraction"),
            "metadata_mean_exposure_fraction_supported": mean(supported_rows, "metadata_exposure_fraction"),
            "peak_resident_capability_count": peak_resident,
            "peak_resident_prompt_tokens": peak_prompt,
            "peak_resident_host_ram_mb": peak_ram,
        },
        "tasks": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run_capability_composition_probe(), ensure_ascii=False, indent=2))
