from __future__ import annotations



from .models import EvidenceChunk, EvidenceLevel, EvidenceRef





_COVERAGE_KEYS = ("formulation", "property", "condition", "mechanism")





def _kinds_for_chunk(chunk: EvidenceChunk) -> set[str]:

    text = chunk.content.lower()

    triple_text = " ".join(f"{triple.r} {triple.t}".lower() for triple in chunk.triples)

    joined = f"{text} {triple_text}"

    kinds = set()

    if any(token in joined for token in ("used_in", "formulation", "electrolyte", " in pc", " in dme")):

        kinds.add("formulation")

    if any(token in joined for token in ("ms cm", "s cm", "viscos", "capacity", "cycle", "efficiency", "stability window")):

        kinds.add("property")

    if any(token in joined for token in ("°c", "℃", " at -", " at 25", "hard carbon", "evaluated")):

        kinds.add("condition")

    if any(token in joined for token in ("sei", "cei", "mechanism", "promotes", "interphase")):

        kinds.add("mechanism")

    return kinds or {"risk"}





def normalize_evidence_chunks(chunks: list[EvidenceChunk | dict]) -> list[EvidenceRef]:

    refs = []

    seen: set[tuple[str, str]] = set()

    for raw_chunk in chunks:

        chunk = raw_chunk if isinstance(raw_chunk, EvidenceChunk) else EvidenceChunk.model_validate(raw_chunk)

        for kind in sorted(_kinds_for_chunk(chunk)):

            key = (chunk.chunk_id, kind)

            if key in seen:

                continue

            seen.add(key)

            refs.append(EvidenceRef(

                chunk_id=chunk.chunk_id,

                paper_title=chunk.paper_title,

                section_title=chunk.section_title,

                score=chunk.score,

                excerpt=chunk.content[:500],

                evidence_kind=kind,

            ))

    return refs





def evidence_coverage(evidence: list[EvidenceRef]) -> dict[str, bool]:

    kinds = {item.evidence_kind for item in evidence}

    return {key: key in kinds for key in _COVERAGE_KEYS}





def evidence_level(coverage: dict[str, bool]) -> EvidenceLevel:

    if coverage["formulation"] and coverage["property"] and sum(coverage.values()) >= 3:

        return "strong"

    if coverage["formulation"] and (coverage["condition"] or coverage["mechanism"]):

        return "moderate"

    return "weak"

