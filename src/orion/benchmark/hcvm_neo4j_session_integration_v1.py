"""HCVM V1: real sodium-electrolyte Neo4j session page-out/refault integration.

This is a narrow, read-only domain integration check.  It does not modify the
production KG, call an LLM, or claim chemistry accuracy.  The benchmark asks a
runtime question only: can the real electrolyte-KG connection be represented as
one SESSION-backed logical capability, physically closed on page-out, reopened
on refault, and return the same frozen read-only relation query?
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from orion.capvm import (
    CapabilityCatalog,
    CapabilityKind,
    CapabilityResidencyGovernor,
    ResidencyLevel,
    ResourceVector,
    VirtualCapability,
)
from orion.capvm.adapters import Neo4jSessionCapabilityAdapter
from orion.tools.db_config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


CAPABILITY_ID = "sodium-electrolyte-kg-session"
BUDGET = ResourceVector(connection_slots=1.0, host_ram_mb=64.0)

# Frozen, parameterized and read-only.  It intentionally does not use a free-form
# Cypher generator.  The target relation was already observed in the production
# electrolyte KG before this HCVM integration check was designed.
READ_QUERY = """
MATCH (a)-[r]->(b)
WHERE toLower(coalesce(a.name, '')) CONTAINS $material
  AND toLower(coalesce(b.name, '')) CONTAINS $condition
RETURN a.name AS source,
       type(r) AS relation,
       b.name AS target,
       coalesce(r.confidence, 0.0) AS confidence
ORDER BY confidence DESC, source, target
LIMIT 10
""".strip()
PARAMETERS = {"material": "hard carbon", "condition": "-20"}


def _normalized_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "source": str(row.get("source", "")),
                "relation": str(row.get("relation", "")),
                "target": str(row.get("target", "")),
                "confidence": round(float(row.get("confidence", 0.0)), 8),
            }
        )
    return normalized


def _contains_expected_relation(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        source = str(row.get("source", "")).lower()
        relation = str(row.get("relation", "")).upper()
        target = str(row.get("target", "")).lower()
        if "hard carbon" in source and relation == "TESTED_UNDER" and "-20" in target:
            return True
    return False


def run(
    adapter_factory: Callable[..., Neo4jSessionCapabilityAdapter] = Neo4jSessionCapabilityAdapter,
) -> dict[str, Any]:
    adapter = adapter_factory(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )
    capability = VirtualCapability(
        capability_id=CAPABILITY_ID,
        kind=CapabilityKind.SESSION,
        summary="Read-only session to the production sodium-electrolyte Neo4j knowledge graph",
        adapter=adapter,
        maximum_level=ResidencyLevel.BOUND,
        metadata={"domain": "sodium-ion electrolyte KG", "mode": "read-only benchmark"},
    )
    governor = CapabilityResidencyGovernor(
        catalog=CapabilityCatalog([capability]),
        budget=BUDGET,
    )

    before_rows: list[dict[str, Any]] = []
    after_rows: list[dict[str, Any]] = []
    during_first = ResourceVector()
    after_pageout = ResourceVector()
    during_refault = ResourceVector()
    connected_during_first = False
    connected_after_pageout = False
    connected_during_refault = False

    try:
        with governor.acquire_many(
            ((CAPABILITY_ID, ResidencyLevel.BOUND),),
            request_id="kg-first-query",
            reconcile_before_active=True,
        ):
            connected_during_first = adapter.is_connected
            during_first = governor.total_measured_cost()
            before_rows = _normalized_rows(adapter.run_read(READ_QUERY, **PARAMETERS))

        governor.demote(CAPABILITY_ID, ResidencyLevel.CATALOG)
        connected_after_pageout = adapter.is_connected
        after_pageout = governor.total_measured_cost()

        with governor.acquire_many(
            ((CAPABILITY_ID, ResidencyLevel.BOUND),),
            request_id="kg-refault-query",
            reconcile_before_active=True,
        ):
            connected_during_refault = adapter.is_connected
            during_refault = governor.total_measured_cost()
            after_rows = _normalized_rows(adapter.run_read(READ_QUERY, **PARAMETERS))

        state = governor.state(CAPABILITY_ID)
        expected_before = _contains_expected_relation(before_rows)
        expected_after = _contains_expected_relation(after_rows)
        rows_invariant = before_rows == after_rows
        connection_lifecycle_ok = (
            connected_during_first
            and not connected_after_pageout
            and connected_during_refault
            and during_first.connection_slots == 1.0
            and after_pageout.connection_slots == 0.0
            and during_refault.connection_slots == 1.0
        )
        fault_protocol_ok = state.compulsory_faults == 1 and state.refaults == 1
        success = (
            bool(before_rows)
            and expected_before
            and expected_after
            and rows_invariant
            and connection_lifecycle_ok
            and fault_protocol_ok
            and adapter.query_count == 2
        )

        before_sha256 = hashlib.sha256(
            json.dumps(before_rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        after_sha256 = hashlib.sha256(
            json.dumps(after_rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return {
            "schema_version": "hcvm-neo4j-session-integration-v1.0",
            "status": "PASS_REAL_ELECTROLYTE_KG_SESSION_REFAULT"
            if success
            else "FAIL_REAL_ELECTROLYTE_KG_SESSION_REFAULT",
            "success": success,
            "capability_id": CAPABILITY_ID,
            "kg_uri": NEO4J_URI,
            "query_parameters": dict(PARAMETERS),
            "read_only_query": READ_QUERY,
            "row_count_before": len(before_rows),
            "row_count_after": len(after_rows),
            "rows_sha256_before": before_sha256,
            "rows_sha256_after": after_sha256,
            "rows_exactly_invariant": rows_invariant,
            "expected_tested_under_minus20_relation_before": expected_before,
            "expected_tested_under_minus20_relation_after": expected_after,
            "connection_lifecycle": {
                "connected_during_first": connected_during_first,
                "measured_connection_slots_during_first": during_first.connection_slots,
                "connected_after_pageout": connected_after_pageout,
                "measured_connection_slots_after_pageout": after_pageout.connection_slots,
                "connected_during_refault": connected_during_refault,
                "measured_connection_slots_during_refault": during_refault.connection_slots,
            },
            "fault_protocol": {
                "compulsory_faults": state.compulsory_faults,
                "refaults": state.refaults,
                "promotions": state.promotions,
                "demotions": state.demotions,
            },
            "adapter": {
                "query_count": adapter.query_count,
                "driver_open_count": getattr(adapter, "driver_open_count", None),
                "session_creation_count": getattr(adapter, "session_creation_count", None),
                "last_startup_latency_ms": adapter.last_startup_latency_s * 1000.0,
                "last_shutdown_latency_ms": adapter.last_shutdown_latency_s * 1000.0,
            },
            "interpretation_guard": (
                "This is a read-only runtime integration check.  It establishes that the real local "
                "sodium-electrolyte Neo4j connection can participate in HCVM SESSION residency and "
                "refault while preserving a frozen structured query result.  It is not a chemistry "
                "accuracy benchmark, KG novelty claim, remote-MCP benchmark, or relation-provenance study."
            ),
        }
    finally:
        governor.close()


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
