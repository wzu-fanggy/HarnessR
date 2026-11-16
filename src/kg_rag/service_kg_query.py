"""Fixed, read-only KG query adapter owned by the root RAG service."""
from __future__ import annotations

import re
from typing import Any

from neo4j import Driver, GraphDatabase

from kg_rag.config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from kg_rag.entity_linker import FlashTextLinker
from kg_rag.entity_resolver import EntityResolver
from kg_rag.vector_store import Neo4jVectorStore


_PARAM_LABELS: dict[str, list[str] | None] = {
    "salt": ["SodiumSalt"], "solvent": ["Solvent"], "additive": ["Additive"],
    "target": None, "material": None, "material_a": None, "material_b": None,
}

# find_materials_for_condition 默认返回的材料类别。调用方可用 material_types 覆盖，
# 以便在"探索盐/电解液空间"时不被负极/正极类材料淹没。
_DEFAULT_MATERIAL_TYPES: list[str] = [
    "Electrolyte", "Anode", "Cathode", "Component",
    "Additive", "SodiumSalt", "Solvent", "Precursor",
]


def _temperature_variants(condition: str) -> list[str]:
    """为温度类条件生成精确匹配的拼写变体（'-20 °C' vs '-20°C' vs '-20℃' 等）。

    纯描述性条件（不含数字）返回空列表，此时只走 CONTAINS 兜底。
    """
    s = condition.strip()
    if not re.search(r"\d", s):
        return []
    variants = {s}
    norm = s.replace("℃", "°C")
    variants.add(norm)
    variants.add(re.sub(r"\s*°\s*C", "°C", norm))
    variants.add(norm.replace("°", ""))
    variants.add(norm.replace(" ", ""))
    variants.discard("")
    return sorted(variants)


_QUERIES = {
    "find_formulations": """
        MATCH (s:SodiumSalt)-[:DISSOLVES]->(sol:Solvent)
        WHERE ($salt IS NULL OR s.name = $salt) AND ($solvent IS NULL OR sol.name = $solvent)
        OPTIONAL MATCH (e:Electrolyte)-[:CONTAINS]->(s), (e)-[:CONTAINS]->(sol)
        OPTIONAL MATCH (e)-[:CONTAINS]->(add:Additive)
        WITH s, sol, collect(DISTINCT e.name) AS electrolytes, collect(DISTINCT add.name) AS additives
        WHERE $additive IS NULL OR $additive IN additives
        RETURN s.name AS salt, sol.name AS solvent, additives[0] AS additive,
               electrolytes[0] AS electrolyte, size(electrolytes) AS support_count
        ORDER BY support_count DESC LIMIT $limit
    """,
    "what_improves": """
        MATCH (improver)-[rel:IMPROVES]->(target)
        WHERE ($target IS NULL OR target.name = $target) OR ($target IS NULL OR target.text CONTAINS $target)
        RETURN improver.name AS improver, labels(improver) AS improver_labels, type(rel) AS relation,
               target.name AS target, target.text AS target_text, rel.confidence_mean AS confidence
        ORDER BY rel.confidence_mean DESC LIMIT $limit
    """,
    "properties_of": """
        MATCH (material)-[rel:HAS_PROPERTY]->(prop:Property) WHERE material.name = $material
        RETURN material.name AS material, prop.name AS property_name, prop.text AS property_text,
               rel.confidence_mean AS confidence
        ORDER BY rel.confidence_mean DESC LIMIT $limit
    """,
    "compare_materials": """
        MATCH (a)-[r:HAS_PROPERTY|TESTED_UNDER|IMPROVES]->(x) WHERE a.name IN [$material_a, $material_b]
        RETURN a.name AS material, type(r) AS relation, x.name AS target, x.text AS target_text,
               r.confidence_mean AS confidence
        ORDER BY a.name, relation LIMIT $limit
    """,
    "find_co_solvents": """
        MATCH (s:SodiumSalt)-[:DISSOLVES]->(sol:Solvent) WHERE s.name = $salt
        RETURN s.name AS salt, sol.name AS solvent, count(*) AS support_count
        ORDER BY support_count DESC LIMIT $limit
    """,
    "find_materials_for_condition": """
        MATCH (material)-[rel:TESTED_UNDER]->(cond:Condition)
        WHERE any(label IN labels(material) WHERE label IN $material_types)
          AND (
            cond.name = $condition
            OR cond.name IN $condition_variants
            OR cond.text CONTAINS $condition
          )
        OPTIONAL MATCH (material)-[:MENTIONED_IN]->(ch:Chunk)<-[:MENTIONED_IN]-(cond)
        RETURN material.name AS material, labels(material) AS material_labels, cond.name AS condition,
               cond.text AS condition_text, rel.confidence_mean AS confidence,
               [chunk_id IN collect(DISTINCT ch.chunk_id) WHERE chunk_id IS NOT NULL][0..5] AS chunk_ids
        ORDER BY rel.confidence_mean IS NULL, rel.confidence_mean DESC LIMIT $limit
    """,
}

_REQUIRED_QUERY_PARAMS: dict[str, tuple[str, ...]] = {
    "what_improves": ("target",),
    "properties_of": ("material",),
    "compare_materials": ("material_a", "material_b"),
    "find_co_solvents": ("salt",),
    "find_materials_for_condition": ("condition",),
}

_DEFAULT_RELATIONS = {
    "find_formulations": "DISSOLVES",
    "what_improves": "IMPROVES",
    "properties_of": "HAS_PROPERTY",
    "compare_materials": "HAS_PROPERTY",
    "find_co_solvents": "DISSOLVES",
    "find_materials_for_condition": "TESTED_UNDER",
}


class KgQueryService:
    """Only exposes audited, parameterized Cypher templates; never arbitrary Cypher."""

    def __init__(
        self, driver: Driver | None = None, database: str = NEO4J_DATABASE,
        resolver: EntityResolver | None = None, linker: FlashTextLinker | None = None,
    ) -> None:
        self.driver = driver or GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.database = database
        self._resolver = resolver
        self._linker = linker

    @property
    def resolver(self) -> EntityResolver:
        if self._resolver is None:
            self._resolver = EntityResolver(self.driver, database=self.database)
        return self._resolver

    @property
    def linker(self) -> FlashTextLinker:
        if self._linker is None:
            names = Neo4jVectorStore(self.driver, database=self.database).load_entity_names()
            self._linker = FlashTextLinker(names)
        return self._linker

    def entity_link(self, text: str) -> str | None:
        matches = self.linker.link(text)
        return matches[0] if matches else None

    @staticmethod
    def _limit(value: Any, default: int = 10) -> int:
        try:
            return min(max(1, int(value)), 15)
        except (TypeError, ValueError):
            return default

    def run_query(self, query_type: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if query_type not in _QUERIES:
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
            raise ValueError(
                f"query_type {query_type!r} requires non-empty params: "
                + ", ".join(missing)
            )
        if query_type == "find_materials_for_condition":
            condition = str(params["condition"]).strip()
            if re.search(r"[A-Za-z°℃%]", condition) is None:
                raise ValueError(
                    "find_materials_for_condition requires a unit or descriptive "
                    "condition (for example '-20 °C' or 'low temperature')"
                )
        query_params = dict(params)
        if query_type == "find_materials_for_condition":
            query_params["condition_variants"] = _temperature_variants(
                str(params["condition"]).strip()
            )
            query_params["material_types"] = list(
                params.get("material_types") or _DEFAULT_MATERIAL_TYPES
            )
        for key, labels in _PARAM_LABELS.items():
            if query_params.get(key):
                resolved = self.resolver.resolve_name(query_params[key], type_labels=labels)
                if resolved:
                    query_params[key] = resolved
        query_params["limit"] = self._limit(query_params.get("limit"))
        for name in ("salt", "solvent", "additive", "material_a", "material_b"):
            query_params.setdefault(name, None)
        with self.driver.session(database=self.database) as session:
            records = session.run(_QUERIES[query_type], query_params).data()
        return self._records_to_results(query_type, records)

    @staticmethod
    def _records_to_results(query_type: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for record in records:
            start = (
                record.get("salt")
                or record.get("improver")
                or record.get("material")
                or record.get("source_entity")
                or ""
            )
            relation = record.get("relation") or _DEFAULT_RELATIONS[query_type]
            end = (
                record.get("solvent")
                or record.get("target")
                or record.get("target_text")
                or record.get("additive")
                or record.get("property_name")
                or record.get("property_text")
                or record.get("condition")
                or record.get("condition_text")
                or ""
            )
            confidence = record.get("confidence")
            if confidence is None:
                confidence = record.get("support_count")
            path = {"start_entity": start, "relation": relation, "end_entity": end, "confidence_mean": confidence}
            results.append({
                "query_type": query_type, "start_entity": start, "relation": relation,
                "end_entity": end, "confidence_mean": confidence,
                "entity_labels": record.get("improver_labels") or record.get("material_labels") or [],
                "chunk_ids": record.get("chunk_ids") or [], "kg_paths": [path],
            })
        return results

    def find_grounded_formulations(
        self, *, salt: str | None, solvents: list[str] | None = None,
        additives: list[str] | None = None, limit: int = 10,
    ) -> list[dict[str, Any]]:
        resolved_salt = self.resolver.resolve_name(salt, type_labels=["SodiumSalt"]) if salt else None
        resolved_solvents = [self.resolver.resolve_name(n, type_labels=["Solvent"]) or n for n in (solvents or [])]
        resolved_additives = [self.resolver.resolve_name(n, type_labels=["Additive"]) or n for n in (additives or [])]
        query = """
            MATCH (f:Electrolyte)-[:CONTAINS]->(s:SodiumSalt), (f)-[:CONTAINS]->(sol:Solvent),
                  (f)-[:MENTIONED_IN]->(ch:Chunk {chunk_type: 'child'}),
                  (s)-[:MENTIONED_IN]->(ch), (sol)-[:MENTIONED_IN]->(ch)
            WHERE ($salt IS NULL OR s.name = $salt) AND ($solvents = [] OR sol.name IN $solvents)
            WITH f, s, ch, collect(DISTINCT sol.name) AS formulation_solvents
            OPTIONAL MATCH (f)-[:CONTAINS]->(add:Additive) WHERE NOT 'Solvent' IN labels(add)
            OPTIONAL MATCH (add)-[:MENTIONED_IN]->(ch)
            WITH f, s, ch, formulation_solvents,
                 [name IN collect(DISTINCT add.name) WHERE name IS NOT NULL] AS formulation_additives
            WHERE ($additives = [] OR all(name IN $additives WHERE name IN formulation_additives))
            RETURN f.name AS electrolyte, s.name AS salt, formulation_solvents AS solvents,
                   formulation_additives AS additives, ch.chunk_id AS chunk_id,
                   ch.parent_id AS parent_chunk_id, ch.text AS text, ch.paper_title AS paper_title
            ORDER BY electrolyte, chunk_id LIMIT $limit
        """
        params = {"salt": resolved_salt or salt, "solvents": resolved_solvents,
                  "additives": resolved_additives, "limit": self._limit(limit)}
        with self.driver.session(database=self.database) as session:
            records = session.run(query, params).data()
        return [{**record, "binding_level": "same_child"} for record in records]

    def chunks_for_entities(self, entity_names: list[str], top_k: int = 10) -> list[dict[str, Any]]:
        if len(entity_names) > 10:
            raise ValueError("at most 10 entity names are allowed")
        resolved = [self.resolver.resolve_name(name) or name for name in entity_names]
        query = """
            UNWIND $entities AS entity_name
            MATCH (e:Entity {name: entity_name})-[:MENTIONED_IN]->(ch:Chunk)
            WITH ch, collect(DISTINCT entity_name) AS matched_entities
            RETURN ch.chunk_id AS chunk_id, ch.text AS text, ch.paper_title AS paper_title,
                   ch.section AS section, ch.parent_id AS parent_chunk_id,
                   ch.chunk_type AS chunk_type, matched_entities,
                   matched_entities[0] AS entity_name
            ORDER BY size(matched_entities) DESC, ch.chunk_id
            LIMIT $top_k
        """
        with self.driver.session(database=self.database) as session:
            return session.run(query, {"entities": resolved, "top_k": self._limit(top_k)}).data()

    def parent_chunks(self, child_chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not child_chunk_ids:
            return {}
        if len(child_chunk_ids) > 20:
            raise ValueError("at most 20 child chunk IDs are allowed")
        query = """
            UNWIND $child_ids AS child_id
            MATCH (child:Chunk {chunk_id: child_id})
            OPTIONAL MATCH (parent:Chunk {chunk_id: child.parent_id})
            RETURN child.chunk_id AS child_id, parent.text AS parent_text, parent.chunk_id AS parent_chunk_id
        """
        with self.driver.session(database=self.database) as session:
            records = session.run(query, {"child_ids": child_chunk_ids}).data()
        return {record["child_id"]: record for record in records}
