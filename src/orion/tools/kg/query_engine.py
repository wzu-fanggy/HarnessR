"""Orion-side adapter for the root process's fixed, read-only KG queries."""
from __future__ import annotations

import re
from typing import Any

from orion.tools.rag.service_client import get_root_kg_service_client

from .models import KgQueryResult


_SUPPORTED_QUERY_TYPES = {
    "find_formulations",
    "what_improves",
    "properties_of",
    "compare_materials",
    "find_co_solvents",
    "find_materials_for_condition",
}

_REQUIRED_QUERY_PARAMS: dict[str, tuple[str, ...]] = {
    "what_improves": ("target",),
    "properties_of": ("material",),
    "compare_materials": ("material_a", "material_b"),
    "find_co_solvents": ("salt",),
    "find_materials_for_condition": ("condition",),
}


class KgQueryEngine:
    """Keep the existing tool API while delegating all KG reads to root service.

    The optional ``driver`` argument is accepted only for backwards-compatible
    construction. Runtime code must not use it: the loopback root service owns
    Neo4j connections, entity resolution, and all fixed Cypher templates.
    """

    def __init__(self, driver: Any | None = None):
        del driver

    @staticmethod
    def _results(payload: Any) -> list[Any]:
        if not isinstance(payload, dict):
            raise RuntimeError("local KG service returned invalid query payload")
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise RuntimeError("local KG service returned invalid query results")
        return results

    def run_query(self, query_type: str, params: dict[str, Any]) -> list[KgQueryResult]:
        if query_type not in _SUPPORTED_QUERY_TYPES:
            raise ValueError(f"Unknown query_type: {query_type}")
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        required = _REQUIRED_QUERY_PARAMS.get(query_type, ())
        missing = [
            name
            for name in required
            if name not in params
            or params[name] is None
            or (isinstance(params[name], str) and not params[name].strip())
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"query_type {query_type!r} requires non-empty params: {joined}"
            )
        if query_type == "find_materials_for_condition":
            condition = str(params["condition"]).strip()
            if re.search(r"[A-Za-z°℃%]", condition) is None:
                raise ValueError(
                    "find_materials_for_condition requires a unit or descriptive "
                    "condition (for example '-20 °C' or 'low temperature')"
                )
        payload = get_root_kg_service_client().call(
            "/v1/kg/query/run_query",
            {"query_type": query_type, "params": params},
        )
        return [KgQueryResult.model_validate(item) for item in self._results(payload)]

    def find_grounded_formulations(
        self,
        *,
        salt: str | None,
        solvents: list[str] | None = None,
        additives: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        payload = get_root_kg_service_client().call(
            "/v1/kg/query/find_grounded_formulations",
            {
                "salt": salt,
                "solvents": solvents,
                "additives": additives,
                "limit": limit,
            },
        )
        return [item for item in self._results(payload) if isinstance(item, dict)]

    def chunks_for_entities(
        self, entity_names: list[str], top_k: int = 10
    ) -> list[dict[str, Any]]:
        payload = get_root_kg_service_client().call(
            "/v1/kg/query/chunks_for_entities",
            {"entity_names": entity_names, "top_k": top_k},
        )
        return [item for item in self._results(payload) if isinstance(item, dict)]

    def parent_chunks(self, child_chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        payload = get_root_kg_service_client().call(
            "/v1/kg/query/parent_chunks",
            {"child_chunk_ids": child_chunk_ids},
        )
        results = payload.get("results", {}) if isinstance(payload, dict) else {}
        if not isinstance(results, dict) or not all(
            isinstance(key, str) and isinstance(value, dict)
            for key, value in results.items()
        ):
            raise RuntimeError("local KG service returned invalid parent-chunk payload")
        return results
