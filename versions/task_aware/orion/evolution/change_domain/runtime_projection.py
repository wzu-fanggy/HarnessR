"""Read-only projection of persisted TaskArtifact provenance into the Harness graph.

This module does not reinterpret scientific evidence. It only exposes identities that ORION
already persisted: evidence ledger entries, claim contracts, evidence-use decisions, candidates,
and their exact chunk references. The purpose is to let a scientific-relation delta enter the
existing provenance graph without inventing a second provenance subsystem.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .kg_projection import evidence_chunk_artifact_id, scientific_node_slug
from .models import CouplingEdge, CouplingLayer, HarnessArtifact, MultiplexHarnessGraph


def _add_run(graph: MultiplexHarnessGraph, run_id: str, task_artifact: Mapping[str, Any]) -> str:
    run_node = f"runtime_run:{scientific_node_slug(run_id)}"
    graph.add_artifact(
        HarnessArtifact(
            artifact_id=run_node,
            kind="runtime_run",
            label=run_id,
            metadata={
                "task_type": str(task_artifact.get("task_type") or ""),
                "task_status": str(task_artifact.get("task_status") or ""),
            },
        )
    )
    return run_node


def _ensure_chunk(graph: MultiplexHarnessGraph, chunk_id: str) -> str:
    chunk_node = evidence_chunk_artifact_id(chunk_id)
    # A chunk may already exist because the KG relation projection introduced it.
    # Identity is the chunk_id; provenance views must converge on the same node rather
    # than duplicating it merely because their metadata/authority labels differ.
    if chunk_node not in graph.artifacts:
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=chunk_node,
                kind="scientific_evidence_chunk",
                label=chunk_id,
                metadata={"authority": "persisted_provenance"},
            )
        )
    return chunk_node


def add_task_artifact_runtime_projection(
    graph: MultiplexHarnessGraph,
    *,
    run_id: str,
    task_artifact: Mapping[str, Any],
) -> None:
    """Project exact persisted evidence consumers from one TaskArtifact."""

    if not run_id.strip():
        raise ValueError("run_id is required")
    run_node = _add_run(graph, run_id, task_artifact)

    for item in task_artifact.get("evidence_ledger") or []:
        if not isinstance(item, Mapping):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        evidence_id = str(item.get("evidence_id") or chunk_id).strip()
        if not chunk_id or not evidence_id:
            continue
        chunk_node = _ensure_chunk(graph, chunk_id)
        evidence_node = f"runtime_evidence:{scientific_node_slug(run_id)}:{scientific_node_slug(evidence_id)}"
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=evidence_node,
                kind="runtime_evidence_record",
                label=evidence_id,
                metadata={
                    "candidate_id": str(item.get("candidate_id") or ""),
                    "paper_id": str(item.get("paper_id") or ""),
                },
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=chunk_node,
                target=evidence_node,
                layer=CouplingLayer.SCIENTIFIC,
                weight=1.15,
                directed=True,
                relation="recorded_as_runtime_evidence",
                provenance="TaskArtifact.evidence_ledger",
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=run_node,
                target=evidence_node,
                layer=CouplingLayer.STATE,
                weight=0.9,
                relation="contains_runtime_evidence",
                provenance="TaskArtifact.evidence_ledger",
            )
        )
        candidate_id = str(item.get("candidate_id") or "").strip()
        if candidate_id:
            candidate_node = f"runtime_candidate:{scientific_node_slug(run_id)}:{scientific_node_slug(candidate_id)}"
            graph.add_artifact(
                HarnessArtifact(
                    artifact_id=candidate_node,
                    kind="runtime_candidate",
                    label=candidate_id,
                )
            )
            graph.add_edge(
                CouplingEdge(
                    source=evidence_node,
                    target=candidate_node,
                    layer=CouplingLayer.BEHAVIOR,
                    weight=1.0,
                    directed=True,
                    relation="supports_candidate",
                    provenance="TaskArtifact.evidence_ledger.candidate_id",
                )
            )

    for contract in task_artifact.get("claim_contracts") or []:
        if not isinstance(contract, Mapping):
            continue
        claim_id = str(contract.get("claim_id") or "").strip()
        if not claim_id:
            continue
        claim_node = f"runtime_claim:{scientific_node_slug(run_id)}:{scientific_node_slug(claim_id)}"
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=claim_node,
                kind="runtime_claim_contract",
                label=claim_id,
                metadata={
                    "claim_type": str(contract.get("claim_type") or ""),
                    "claim_text": str(contract.get("claim_text") or "")[:240],
                },
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=run_node,
                target=claim_node,
                layer=CouplingLayer.STATE,
                weight=0.9,
                relation="contains_claim_contract",
                provenance="TaskArtifact.claim_contracts",
            )
        )
        for source_ref in contract.get("source_refs") or []:
            if not isinstance(source_ref, Mapping):
                continue
            chunk_id = str(source_ref.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            graph.add_edge(
                CouplingEdge(
                    source=_ensure_chunk(graph, chunk_id),
                    target=claim_node,
                    layer=CouplingLayer.SCIENTIFIC,
                    weight=1.15,
                    directed=True,
                    relation="supports_runtime_claim",
                    provenance="TaskArtifact.claim_contracts.source_refs",
                )
            )

    for decision in task_artifact.get("evidence_use_decisions") or []:
        if not isinstance(decision, Mapping):
            continue
        decision_id = str(decision.get("decision_id") or "").strip()
        if not decision_id:
            continue
        decision_node = f"runtime_use:{scientific_node_slug(run_id)}:{scientific_node_slug(decision_id)}"
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=decision_node,
                kind="runtime_evidence_use_decision",
                label=decision_id,
                metadata={
                    "use_kind": str(decision.get("use_kind") or ""),
                    "disposition": str(decision.get("disposition") or ""),
                    "claim_id": str(decision.get("claim_id") or ""),
                },
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=run_node,
                target=decision_node,
                layer=CouplingLayer.STATE,
                weight=0.9,
                relation="contains_evidence_use_decision",
                provenance="TaskArtifact.evidence_use_decisions",
            )
        )
        for evidence_key in decision.get("evidence_keys") or []:
            chunk_id = str(evidence_key or "").strip()
            if not chunk_id or chunk_id.startswith("computed:"):
                continue
            graph.add_edge(
                CouplingEdge(
                    source=_ensure_chunk(graph, chunk_id),
                    target=decision_node,
                    layer=CouplingLayer.SCIENTIFIC,
                    weight=1.2,
                    directed=True,
                    relation="authorized_runtime_use",
                    provenance="TaskArtifact.evidence_use_decisions.evidence_keys",
                )
            )
        claim_id = str(decision.get("claim_id") or "").strip()
        if claim_id:
            claim_node = f"runtime_claim:{scientific_node_slug(run_id)}:{scientific_node_slug(claim_id)}"
            if claim_node in graph.artifacts:
                graph.add_edge(
                    CouplingEdge(
                        source=claim_node,
                        target=decision_node,
                        layer=CouplingLayer.BEHAVIOR,
                        weight=1.0,
                        directed=True,
                        relation="governed_by_use_decision",
                        provenance="TaskArtifact.evidence_use_decisions.claim_id",
                    )
                )
