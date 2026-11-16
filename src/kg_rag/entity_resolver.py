"""Entity resolution: map agent/user-supplied names to real KG entity names.



The KG stores the same substance under heterogeneous names — full names

(``Sodium hexafluorophosphate``), ``full name (ABBR)`` forms

(``ethylene carbonate (EC)``), and subscript-corroded stubs (``NaPF``) —

while the agent side works with canonical short names (``NaPF6``, ``FEC``).

This module resolves names with a deterministic cascade; no models involved.



Cascade (all levels are collected, then ranked by strategy priority and

``appearance_count``):



1. ``exact``   — ``e.name = $name`` after unicode-subscript normalization

2. ``ci``      — case-insensitive exact match

3. ``synonym`` — curated abbreviation -> full-name candidates (sourced from

                 ``orion/skills/kg-retrieval/SKILL.md``); every candidate is

                 verified against the KG before being accepted

4. ``paren``   — ``name (ABBR)`` suffix pattern, e.g. ``diethyl carbonate (DEC)``

5. ``stem``    — trailing-digit stripped stem match for corroded stubs

                 (``NaPF6`` -> ``NaPF``); lowest confidence, dirty-data fallback

"""

from __future__ import annotations



import re

import unicodedata

from dataclasses import dataclass, field

from typing import Any



from neo4j import Driver, GraphDatabase



from .config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER



_SUBSCRIPT_MAP = {c: str(i) for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")}

_SUBSCRIPT_MAP.update({"ₓ": "x", "₋": "-", "₊": "+"})



# Abbreviation -> full-name candidates. Keys are matched case-insensitively.

# Candidates that do not exist in the KG are simply skipped, so unvalidated

# entries are harmless; keep the table in sync with kg-retrieval/SKILL.md.

SYNONYMS: dict[str, list[str]] = {

    # salts

    "NaPF6": ["sodium hexafluorophosphate"],

    "NaFSI": ["sodium bis(fluorosulfonyl)imide"],

    "NaTFSI": [

        "sodium bis(trifluoromethylsulfonyl)imide",

        "sodium bis(trifluoromethanesulfonyl)imide",

    ],

    "NaClO4": ["sodium perchlorate"],

    "NaDFP": ["sodium difluoro(phosphato)borate"],

    # solvents

    "PC": ["propylene carbonate"],

    "EC": ["ethylene carbonate"],

    "EMC": ["ethyl methyl carbonate"],

    "DEC": ["diethyl carbonate"],

    "DMC": ["dimethyl carbonate"],

    "DME": ["1,2-dimethoxyethane", "dimethoxyethane"],

    "DEGDME": ["diethylene glycol dimethyl ether", "diglyme"],

    "SL": ["sulfolane"],

    # solvent aliases

    "diglyme": ["diethylene glycol dimethyl ether"],

    "monoglyme": ["1,2-dimethoxyethane", "dimethoxyethane"],

    # additives

    "FEC": ["fluoroethylene carbonate"],

    "VC": ["vinylene carbonate"],

    "PS": ["propylene sulfite"],

    "DTD": ["ethylene sulfate"],

    "PFPN": ["pentafluorophenyl nitrile"],

    "TMSP": ["tris(trimethylsilyl) phosphite"],

    # electrodes

    "HC": ["hard carbon"],

    "NFM": ["NaFe0.5Mn0.5O2", "NaFeMnO2"],

    # interphases

    "SEI": ["solid electrolyte interphase"],

    "CEI": ["cathode electrolyte interphase"],

}

_SYNONYM_INDEX = {k.lower(): v for k, v in SYNONYMS.items()}



_STRATEGY_PRIORITY = {"exact": 0, "ci": 1, "synonym": 2, "paren": 3, "stem": 4}

_STRATEGY_CONFIDENCE = {

    "exact": 1.0,

    "ci": 0.95,

    "synonym": 0.9,

    "paren": 0.85,

    "stem": 0.5,

}



_FETCH = (

    "MATCH (e:Entity) WHERE {where} "

    "RETURN e.name AS name, labels(e) AS labels, e.appearance_count AS ac "

    "ORDER BY ac DESC LIMIT $limit"

)





@dataclass

class ResolvedEntity:

    """One KG entity matched for an input name."""



    input: str

    name: str

    labels: list[str] = field(default_factory=list)

    appearance_count: int = 0

    strategy: str = "exact"

    confidence: float = 1.0





def normalize_name(name: str) -> str:

    """Normalize unicode subscripts/specials and whitespace (NaPF₆ -> NaPF6)."""

    out = "".join(_SUBSCRIPT_MAP.get(ch, ch) for ch in name)

    return unicodedata.normalize("NFKC", out).strip()





def _strip_trailing_digits(name: str) -> str:

    return re.sub(r"[\d.\s/]+$", "", name)





class EntityResolver:

    """Resolve free-form entity names to KG entity names via the cascade."""



    def __init__(self, driver: Driver | None = None, database: str | None = None):

        self.driver = driver or GraphDatabase.driver(

            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)

        )

        self.database = database or NEO4J_DATABASE



    def resolve(

        self,

        name: str,

        type_labels: list[str] | None = None,

        limit: int = 3,

    ) -> list[ResolvedEntity]:

        """Return ranked candidates; empty list when nothing matches."""

        norm = normalize_name(name)

        if not norm:

            return []

        hits: list[ResolvedEntity] = []

        for strategy, fetcher in (

            ("exact", self._exact),

            ("ci", self._ci),

            ("synonym", self._synonym),

            ("paren", self._paren),

            ("stem", self._stem),

        ):

            for row in fetcher(norm, limit):

                hits.append(self._to_entity(norm, row, strategy))

        # dedup by KG name, keep highest-priority occurrence

        best: dict[str, ResolvedEntity] = {}

        for h in hits:

            cur = best.get(h.name)

            if cur is None or _STRATEGY_PRIORITY[h.strategy] < _STRATEGY_PRIORITY[cur.strategy]:

                best[h.name] = h

        ranked = sorted(

            best.values(),

            key=lambda h: (_STRATEGY_PRIORITY[h.strategy], -h.appearance_count),

        )

        if type_labels:

            wanted = set(type_labels)

            ranked.sort(

                key=lambda h: 0 if wanted.intersection(h.labels) else 1

            )

        return ranked[:limit]



    def resolve_name(self, name: str, type_labels: list[str] | None = None) -> str | None:

        """Top-1 KG name or None — the common case for query parameters."""

        hits = self.resolve(name, type_labels=type_labels, limit=1)

        return hits[0].name if hits else None



    # -- cascade levels -----------------------------------------------------



    def _run(self, where: str, params: dict[str, Any]) -> list[dict]:

        with self.driver.session(database=self.database) as session:

            return session.run(_FETCH.format(where=where), params).data()



    def _exact(self, name: str, limit: int) -> list[dict]:

        return self._run("e.name = $name", {"name": name, "limit": limit})



    def _ci(self, name: str, limit: int) -> list[dict]:

        return self._run(

            "toLower(e.name) = toLower($name)", {"name": name, "limit": limit}

        )



    def _synonym(self, name: str, limit: int) -> list[dict]:

        candidates = _SYNONYM_INDEX.get(name.lower(), [])

        if not candidates:

            return []

        return self._run(

            "toLower(e.name) IN [c IN $cands | toLower(c)]",

            {"cands": candidates, "limit": limit},

        )



    def _paren(self, name: str, limit: int) -> list[dict]:

        if len(name) > 12:

            return []

        pattern = f"(?i).*\\({re.escape(name)}\\)$"

        rows = self._run("e.name =~ $pat", {"pat": pattern, "limit": limit * 3})

        # Drop long formula-description entities ("1 M NaPF6-EC/DEC ... (NaClO4)");

        # the parenthesized form of a substance name is always short.

        return [r for r in rows if len(r["name"]) <= 60][:limit]



    def _stem(self, name: str, limit: int) -> list[dict]:

        stem = _strip_trailing_digits(name)

        if not stem or stem == name:

            return []

        rows = self._run("e.name STARTS WITH $stem", {"stem": stem, "limit": limit * 3})

        return [

            r

            for r in rows

            if re.fullmatch(r"[\d.\s/]*", r["name"][len(stem):])

        ][:limit]



    # -- helpers ------------------------------------------------------------



    @staticmethod

    def _to_entity(source: str, row: dict, strategy: str) -> ResolvedEntity:

        return ResolvedEntity(

            input=source,

            name=row["name"],

            labels=[lb for lb in row.get("labels", []) if lb != "Entity"],

            appearance_count=row.get("ac") or 0,

            strategy=strategy,

            confidence=_STRATEGY_CONFIDENCE[strategy],

        )

