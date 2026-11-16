"""Development-only graph-native scientific Harness composition probe.

This probe evaluates a *combination property*, not the novelty of any individual component.
It composes four already-existing ORION surfaces:

1. a graphified program/control view (Graphify);
2. deterministic LangGraph-style shared-state extraction;
3. deterministic scientific-field responsibility extraction;
4. eight electrolyte task families that were frozen for Study 4C on 2026-08-22.

The question is whether one typed Harness graph can project a task-specific responsibility surface
from structured scientific obligations while keeping the complete system graph available for
cross-layer inspection.  This is not a paper result and it does not claim that graph traversal,
LangGraph, knowledge graphs, or task projection are individually novel.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from orion.benchmark.self_evolution_study4c_v1 import source_bank, study4c_cases
from orion.evolution.change_domain import (
    CouplingLayer,
    MultiplexHarnessGraph,
    add_scientific_task_projection,
    link_source_symbol_views,
    load_graphify_static_projection,
    load_python_state_access_projection,
    load_scientific_field_anchor_projection,
    load_tool_identity_projection,
    project_task_responsibilities,
)


ORION_ROOT = Path(__file__).resolve().parents[1]
GRAPHIFY_JSON = ORION_ROOT / "graphify-out" / "graph.json"
SOURCE_ROOTS = (
    ORION_ROOT / "core",
    ORION_ROOT / "task_evidence",
    ORION_ROOT / "tools",
    ORION_ROOT / "capvm",
    ORION_ROOT / "tests",
)


@dataclass(frozen=True)
class GraphBuildDiagnostics:
    parsed_python_files: int
    skipped_python_files: tuple[str, ...]
    artifact_count: int
    edge_count: int
    same_symbol_links: int
    layer_edge_counts: dict[str, int]
    multi_role_artifact_count: int


def _parseable_python_files() -> tuple[tuple[Path, ...], tuple[str, ...]]:
    files: list[Path] = []
    skipped: list[str] = []
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeError):
                skipped.append(path.relative_to(ORION_ROOT).as_posix())
                continue
            files.append(path)
    return tuple(files), tuple(skipped)


def build_composed_harness_graph() -> tuple[MultiplexHarnessGraph, GraphBuildDiagnostics]:
    """Build the read-only multiplex graph from current ORION assets."""

    source_files, skipped = _parseable_python_files()
    graph = MultiplexHarnessGraph()
    load_scientific_field_anchor_projection(
        graph,
        source_files=source_files,
        project_root=ORION_ROOT,
    )
    load_python_state_access_projection(
        graph,
        source_files=source_files,
        project_root=ORION_ROOT,
    )
    if GRAPHIFY_JSON.exists():
        load_graphify_static_projection(
            graph,
            graph_json=GRAPHIFY_JSON,
            source_prefixes=(
                "core/",
                "task_evidence/",
                "tools/",
                "capvm/",
                "main.py",
                "tests/",
            ),
            code_only=True,
        )
    load_tool_identity_projection(
        graph,
        source_files=source_files,
        project_root=ORION_ROOT,
    )
    same_symbol_links = link_source_symbol_views(graph)
    layer_counts = {
        layer.value: sum(1 for edge in graph.edges if edge.layer is layer)
        for layer in CouplingLayer
    }
    multi_role = sum(1 for artifact in graph.artifacts.values() if len(artifact.roles) > 1)
    return graph, GraphBuildDiagnostics(
        parsed_python_files=len(source_files),
        skipped_python_files=skipped,
        artifact_count=len(graph.artifacts),
        edge_count=len(graph.edges),
        same_symbol_links=same_symbol_links,
        layer_edge_counts=layer_counts,
        multi_role_artifact_count=multi_role,
    )


def _is_regression_artifact(graph: MultiplexHarnessGraph, artifact_id: str) -> bool:
    artifact = graph.artifacts.get(artifact_id)
    if artifact is None:
        return False
    source_file = str(artifact.metadata.get("source_file") or "").replace("\\", "/")
    return source_file.startswith("tests/") or "/tests/" in f"/{source_file}"


def _responsibility_ids(
    graph: MultiplexHarnessGraph,
    *,
    relation: str,
    regression: bool,
) -> set[str]:
    return {
        edge.target
        for edge in graph.edges
        if edge.layer is CouplingLayer.SCIENTIFIC
        and edge.relation == relation
        and _is_regression_artifact(graph, edge.target) is regression
    }


def _execution_support_surface(
    graph: MultiplexHarnessGraph,
    responsibility_ids: Iterable[str],
) -> set[str]:
    """Return a bounded, auditable one-hop execution-support surface.

    The projection is intentionally conservative.  It includes the selected scientific responsibility
    functions, their exact Graphify symbol view, directly shared state fields, and one Graphify
    control/service/regression neighbor.  It does *not* run unrestricted transitive closure, which
    would turn task projection into a monolithic blast radius.
    """

    responsibilities = set(responsibility_ids)
    surface = set(responsibilities)
    graphify_views: set[str] = set()
    for edge in graph.edges:
        if edge.relation == "same_source_symbol":
            if edge.target in responsibilities:
                graphify_views.add(edge.source)
            elif edge.source in responsibilities:
                graphify_views.add(edge.target)
        if edge.layer is CouplingLayer.STATE:
            if edge.source in responsibilities:
                surface.add(edge.target)
            if edge.target in responsibilities:
                surface.add(edge.source)
    surface.update(graphify_views)

    allowed_layers = {
        CouplingLayer.CONTROL,
        CouplingLayer.SERVICE,
        CouplingLayer.STATE,
        CouplingLayer.REGRESSION,
    }
    for edge in graph.edges:
        if edge.layer not in allowed_layers:
            continue
        if edge.source in graphify_views:
            surface.add(edge.target)
        if edge.target in graphify_views:
            surface.add(edge.source)
    return surface


def run_composition_probe() -> dict[str, object]:
    graph, diagnostics = build_composed_harness_graph()
    bank = source_bank()
    families = dict(bank["families"])
    all_strong = _responsibility_ids(
        graph,
        relation="declares_scientific_responsibility",
        regression=False,
    )
    all_weak = _responsibility_ids(
        graph,
        relation="references_scientific_field",
        regression=False,
    )
    all_regression_strong = _responsibility_ids(
        graph,
        relation="declares_scientific_responsibility",
        regression=True,
    )
    all_regression_weak = _responsibility_ids(
        graph,
        relation="references_scientific_field",
        regression=True,
    )
    eager_support = _execution_support_surface(graph, all_strong)

    rows: list[dict[str, object]] = []
    selected_sets: set[tuple[str, ...]] = set()
    for case in study4c_cases():
        raw = dict(families[case.family_id])
        target_scope = dict(raw.get("target_scope_override") or {})
        relevant_fields = tuple(
            str(item)
            for item in (raw.get("relevant_fields_override") or ())
        )
        projection = add_scientific_task_projection(
            graph,
            task_id=case.case_id,
            task_text=case.prompt,
            target_scope=target_scope,
            relevant_fields=relevant_fields,
        )
        responsibilities = project_task_responsibilities(graph, projection)
        selected = set(responsibilities.selected_strong_ids)
        selected_sets.add(tuple(sorted(selected)))
        task_support = _execution_support_surface(graph, selected)

        selected_files = sorted(
            {
                str(graph.artifacts[node_id].metadata.get("source_file") or "")
                for node_id in selected
                if node_id in graph.artifacts
                and graph.artifacts[node_id].metadata.get("source_file")
            }
        )
        rows.append(
            {
                "family_id": case.family_id,
                "case_id": case.case_id,
                "endpoint_role": case.endpoint_role,
                "relevant_fields": list(relevant_fields),
                "fields_with_strong_responsibility": list(
                    responsibilities.fields_with_strong_responsibility
                ),
                "fields_without_strong_responsibility": list(
                    responsibilities.fields_without_strong_responsibility
                ),
                "selected_strong_responsibility_count": len(selected),
                "selected_regression_strong_count": len(
                    {
                        node_id
                        for ids in responsibilities.regression_strong_by_field.values()
                        for node_id in ids
                    }
                ),
                "selected_source_file_count": len(selected_files),
                "selected_source_files": selected_files,
                "task_support_surface_count": len(task_support),
                "eager_strong_responsibility_count": len(all_strong),
                "eager_support_surface_count": len(eager_support),
                "responsibility_exposure_fraction": (
                    len(selected) / len(all_strong) if all_strong else 0.0
                ),
                "support_surface_fraction": (
                    len(task_support) / len(eager_support) if eager_support else 0.0
                ),
                "explanation_path_count": len(responsibilities.explanation_paths),
                "all_selected_have_typed_explanation": (
                    len(responsibilities.explanation_paths) == len(selected)
                ),
            }
        )

    covered_fields = sum(
        len(row["fields_with_strong_responsibility"])  # type: ignore[arg-type]
        for row in rows
    )
    total_fields = sum(len(row["relevant_fields"]) for row in rows)  # type: ignore[arg-type]
    mean_exposure = (
        sum(float(row["responsibility_exposure_fraction"]) for row in rows) / len(rows)
        if rows
        else 0.0
    )
    mean_support = (
        sum(float(row["support_surface_fraction"]) for row in rows) / len(rows)
        if rows
        else 0.0
    )
    return {
        "schema_version": "graph-native-scientific-composition-v0.1",
        "status": "development_only",
        "study4c_freeze_date": str(bank.get("date") or ""),
        "case_count": len(rows),
        "graph": {
            "parsed_python_files": diagnostics.parsed_python_files,
            "skipped_python_files": list(diagnostics.skipped_python_files),
            "artifact_count": diagnostics.artifact_count,
            "edge_count": diagnostics.edge_count,
            "same_symbol_links": diagnostics.same_symbol_links,
            "layer_edge_counts": diagnostics.layer_edge_counts,
            "multi_role_artifact_count": diagnostics.multi_role_artifact_count,
        },
        "global_surfaces": {
            "runtime_strong_scientific_responsibility_count": len(all_strong),
            "runtime_weak_scientific_reference_count": len(all_weak),
            "regression_strong_scientific_responsibility_count": len(all_regression_strong),
            "regression_weak_scientific_reference_count": len(all_regression_weak),
            "eager_support_surface_count": len(eager_support),
        },
        "aggregate": {
            "required_field_occurrences": total_fields,
            "strong_responsibility_field_occurrences": covered_fields,
            "strong_field_coverage": covered_fields / total_fields if total_fields else 0.0,
            "distinct_task_responsibility_sets": len(selected_sets),
            "mean_task_responsibility_exposure_fraction": mean_exposure,
            "mean_task_support_surface_fraction": mean_support,
            "all_selected_responsibilities_have_typed_explanations": all(
                bool(row["all_selected_have_typed_explanation"]) for row in rows
            ),
        },
        "cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run_composition_probe(), ensure_ascii=False, indent=2))
