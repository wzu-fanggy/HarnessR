"""Context-bound scientific coupling for Change Domain V0.2.

A raw scientific KG path is retrieval structure, not execution/evolution authority. This module
projects an already structured source/target applicability comparison into the multiplex Harness
graph. Only condition dimensions that are mismatched or missing become revalidation obligations;
matched dimensions stay visible for explanation but do not seed unrelated Harness changes.

The comparison reuses ORION's deterministic evidence-applicability gate. No LLM semantic judgment
is added here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from orion.task_evidence.evidence_applicability import (
    ApplicabilityAssessment,
    ConditionField,
    assess_evidence_applicability,
)

from .kg_projection import scientific_node_slug
from .models import CouplingEdge, CouplingLayer, HarnessArtifact, MultiplexHarnessGraph


@dataclass(frozen=True)
class ContextualBridgeProjection:
    bridge_id: str
    source_node_id: str
    target_node_id: str
    assessment: ApplicabilityAssessment
    activated_field_ids: tuple[str, ...]


def _scope_metadata(scope: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key): "" if value is None else str(value)
        for key, value in scope.items()
        if value is not None
    }


def add_contextual_scientific_bridge(
    graph: MultiplexHarnessGraph,
    *,
    bridge_id: str,
    source_scope: Mapping[str, Any],
    target_scope: Mapping[str, Any],
    relevant_fields: Sequence[ConditionField],
    source_label: str = "source scientific observation",
    target_label: str = "target scientific obligation",
) -> ContextualBridgeProjection:
    """Project one source->target scientific context comparison.

    ``mismatched_fields`` and ``missing_fields`` are the only dimensions that produce
    ``requires_revalidation`` edges. This is the key guard against the common error where a
    multi-hop KG path activates every nearby scientific dimension simply because the objects are
    connected in the raw graph.
    """

    if not bridge_id.strip():
        raise ValueError("bridge_id is required")

    assessment = assess_evidence_applicability(
        source_scope=source_scope,
        target_scope=target_scope,
        relevant_fields=relevant_fields,
    )
    slug = scientific_node_slug(bridge_id)
    source_node = f"scientific_observation:{slug}:source"
    target_node = f"scientific_obligation:{slug}:target"
    bridge_node = f"scientific_context_bridge:{slug}"

    graph.add_artifact(
        HarnessArtifact(
            artifact_id=source_node,
            kind="scientific_observation",
            label=source_label,
            metadata=_scope_metadata(source_scope),
        )
    )
    graph.add_artifact(
        HarnessArtifact(
            artifact_id=target_node,
            kind="scientific_obligation",
            label=target_label,
            metadata=_scope_metadata(target_scope),
        )
    )
    graph.add_artifact(
        HarnessArtifact(
            artifact_id=bridge_node,
            kind="scientific_context_bridge",
            label=bridge_id,
            metadata={"applicability_status": assessment.status},
        )
    )
    graph.add_edge(
        CouplingEdge(
            source=target_node,
            target=bridge_node,
            layer=CouplingLayer.SCIENTIFIC,
            weight=1.0,
            directed=True,
            relation="evaluates_source_applicability",
            provenance="task_evidence.assess_evidence_applicability",
        )
    )
    graph.add_edge(
        CouplingEdge(
            source=bridge_node,
            target=source_node,
            layer=CouplingLayer.SCIENTIFIC,
            weight=0.2 if assessment.status != "direct" else 0.9,
            directed=True,
            relation=(
                "blocked_source_context"
                if assessment.status != "direct"
                else "direct_source_context"
            ),
            provenance="task_evidence.assess_evidence_applicability",
        )
    )

    activated: list[str] = []
    mismatch_set = set(assessment.mismatched_fields) | set(assessment.missing_fields)
    matched_set = set(assessment.matched_fields)
    for field_name in relevant_fields:
        field_node = f"scientific_context_field:{field_name}"
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=field_node,
                kind="scientific_context_field",
                label=field_name,
            )
        )
        if field_name in mismatch_set:
            activated.append(field_node)
            graph.add_edge(
                CouplingEdge(
                    source=target_node,
                    target=field_node,
                    layer=CouplingLayer.SCIENTIFIC,
                    weight=1.3,
                    directed=True,
                    relation="requires_revalidation",
                    provenance="task_evidence.applicability:mismatch_or_missing",
                )
            )
        elif field_name in matched_set:
            # Keep matched context available for explanation without making it a strong closure edge.
            graph.add_edge(
                CouplingEdge(
                    source=target_node,
                    target=field_node,
                    layer=CouplingLayer.SCIENTIFIC,
                    weight=0.08,
                    directed=True,
                    relation="context_already_matched",
                    provenance="task_evidence.applicability:matched",
                )
            )

    return ContextualBridgeProjection(
        bridge_id=bridge_id,
        source_node_id=source_node,
        target_node_id=target_node,
        assessment=assessment,
        activated_field_ids=tuple(activated),
    )
