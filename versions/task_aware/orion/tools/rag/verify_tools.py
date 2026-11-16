from __future__ import annotations

import re

from langchain_core.tools import tool

from tools.rag.kg_rag_tools import _format_chunk, _get_rag_service_client

# 常见化学实体（保持与 kg_rag/query_variants.py 一致）
_CHEM_RE = re.compile(
    r"\b(NaPF6|NaFSI|NaTFSI|NaBF4|NaClO4|NaDFP|NaDFOB|NaBOB|NaOTf|"
    r"FEC|VC|PC|EC|DMC|DEC|EMC|DME|DEGDME|TEGDME|G1|G2|G3|G4|THF|"
    r"PFPN|TMP|TMSPi|BTFE|DMSO|DMA|NMP|DMF|AN)\b",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"-?\d+\.?\d*")

# 属性关键词，用于生成 narrow queries
_PROPERTY_NAMES = [
    "melting point", "熔点",
    "viscosity", "粘度",
    "dielectric constant", "介电常数",
    "desolvation energy", "去溶剂化能",
    "conductivity", "电导率",
    "capacity", "容量",
    "cycle", "循环",
]


def _normalize_subscripts(text: str) -> str:
    sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    return text.translate(sub_map)


def _extract_entities_and_numbers(claim: str) -> tuple[list[str], list[str]]:
    norm = _normalize_subscripts(claim)
    entities = sorted(set(_CHEM_RE.findall(norm)), key=len, reverse=True)
    numbers = []
    for m in _NUM_RE.findall(norm):
        try:
            v = float(m)
            if not (1900 <= v <= 2100):  # skip years
                numbers.append(m)
        except ValueError:
            pass
    return entities, numbers


def _build_queries(entities: list[str], numbers: list[str], claim: str, max_queries: int = 6) -> list[str]:
    queries = []
    # entity + number narrow query (strongest signal)
    for e in entities[:2]:
        for n in numbers[:2]:
            queries.append(f"{e} {n}")
    # entity + property keyword（命中属性词时最多两条）
    matched_props = [p for p in _PROPERTY_NAMES if p.lower() in claim.lower()]
    for e in entities[:2]:
        if matched_props:
            queries.append(f"{e} {matched_props[0]}")
    # original claim plain text
    queries.append(claim)
    # number only as fallback
    if numbers:
        queries.append(" ".join(numbers[:2]))
    # deduplicate while preserving order，并限制总数
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
        if len(out) >= max_queries:
            break
    return out


def _local_match(text: str, entities: list[str], numbers: list[str], window: int = 400) -> bool:
    """Check that at least one entity and one number co-occur within window chars.

    A parameter claim must mention both a chemical entity and a numeric value.
    If either is missing we cannot confirm the claim locally, so we return False.
    """
    if not entities or not numbers:
        return False
    norm = text.lower()
    positions = []
    for term in entities + numbers:
        t = _normalize_subscripts(term).lower()
        for m in re.finditer(re.escape(t), norm):
            positions.append((m.start(), term in entities))
    if not positions:
        return False
    entity_positions = [p for p, is_ent in positions if is_ent]
    number_positions = [p for p, is_ent in positions if not is_ent]
    if not entity_positions or not number_positions:
        return False
    for ep in entity_positions:
        for np_ in number_positions:
            if abs(ep - np_) <= window:
                return True
    return False


@tool
def verify_parameter_claim(claim: str, top_k: int = 10) -> dict:
    """验证一条参数/事实论断是否能在文献 chunk 中找到直接证据。

    适用于引用具体数值、物性、性能数据前的主动校验。会针对论断中的
    化学实体和数值生成多个 narrow query，召回相关 chunk 后再做局部共现
    匹配。

    Args:
        claim: 待验证的论断，例如 "EC 的熔点为 36.4°C"。
        top_k: 每个 narrow query 召回的 chunk 数，默认 10。

    Returns:
        {
          "claim": "...",
          "verdict": "supported|unsupported|uncertain",
          "confidence": "high|medium|low",
          "sources": [...],
          "suggested_replacement": null  # reserved for future use
        }
    """
    try:
        client = _get_rag_service_client()
        entities, numbers = _extract_entities_and_numbers(claim)
        queries = _build_queries(entities, numbers, claim)

        matched_sources: list[dict] = []
        seen_chunk_ids: set[str] = set()
        for q in queries:
            try:
                parsed = client.call(
                    "hybrid_search_filtered",
                    {
                        "query": q,
                        "top_k": top_k,
                        "mode": "vector",
                        "exclude_low_value": True,
                    },
                )
                chunks = parsed.get("chunks", []) if isinstance(parsed, dict) else parsed
                for c in chunks:
                    text = c.get("text", "") or ""
                    cid = c.get("chunk_id", "")
                    if cid in seen_chunk_ids:
                        continue
                    if _local_match(text, entities, numbers):
                        matched_sources.append(_format_chunk(c, c.get("score", 0.0)))
                        seen_chunk_ids.add(cid)
            except Exception:
                continue

        if not matched_sources:
            return {
                "claim": claim,
                "verdict": "unsupported" if (entities or numbers) else "uncertain",
                "confidence": "medium",
                "sources": [],
                "suggested_replacement": None,
            }

        return {
            "claim": claim,
            "verdict": "supported",
            "confidence": "high" if len(matched_sources) >= 2 else "medium",
            "sources": matched_sources[:5],
            "suggested_replacement": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "claim": claim,
            "verdict": "uncertain",
            "confidence": "low",
            "sources": [],
            "suggested_replacement": None,
            "error": str(exc),
        }
