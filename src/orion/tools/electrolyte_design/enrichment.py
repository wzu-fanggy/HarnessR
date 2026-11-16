"""候选证据的显式、只读 enrichment 工具。"""

from __future__ import annotations

from typing import Any, Annotated
from langgraph.prebuilt import InjectedState
from orion.core.retrieved_evidence import normalize_retrieved_references
import re

from langchain_core.tools import tool

from orion.tools.rag.service_client import get_root_kg_service_client
from tools.kg.models import FormulationCandidate, EvidenceRecord

from .evaluator import coerce_screening_request
from .object_identity import resolve_candidate_payload_object_identity
from .figure_context import figure_references, caption_contexts


def hydrate_formulation_evidence(
    candidates: list[FormulationCandidate],
) -> list[FormulationCandidate]:
    """按精确 chunk_id 补齐缺失原文；服务失败时保留空证据供门禁拒绝。"""

    missing_ids = list(
        dict.fromkeys(
            record.chunk_id
            for candidate in candidates
            for record in candidate.evidence
            if record.chunk_id and not record.text
        )
    )
    if not missing_ids:
        return [
            FormulationCandidate.model_validate(
                resolve_candidate_payload_object_identity(
                    candidate.model_dump(mode="python")
                )
            )
            for candidate in candidates
        ]
    try:
        payload = get_root_kg_service_client().call(
            "/v1/evidence/chunks",
            {"chunk_ids": missing_ids[:20]},
        )
        chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
        by_id = {
            chunk.get("chunk_id"): chunk
            for chunk in chunks
            if isinstance(chunk, dict)
        }
    except Exception:
        return candidates

    hydrated: list[FormulationCandidate] = []
    for candidate in candidates:
        evidence = []
        for record in candidate.evidence:
            chunk = by_id.get(record.chunk_id, {})
            if not record.text and chunk.get("text"):
                record = record.model_copy(
                    update={
                        "text": chunk["text"],
                        "paper_id": (
                            record.paper_id or chunk.get("paper_id", "")
                        ),
                        "paper_title": (
                            record.paper_title or chunk.get("paper_title", "")
                        ),
                    }
                )
            evidence.append(record)
        paper_ids = {
            record.paper_id or record.paper_title
            for record in evidence
            if record.paper_id or record.paper_title
        }
        evidence_scope = candidate.evidence_scope
        if evidence_scope == "unknown" and len(paper_ids) == 1:
            evidence_scope = "same_paper"
        updated = candidate.model_copy(
            update={
                "evidence": evidence,
                "evidence_scope": evidence_scope,
            }
        )
        hydrated.append(
            FormulationCandidate.model_validate(
                resolve_candidate_payload_object_identity(
                    updated.model_dump(mode="python")
                )
            )
        )
    return hydrated


def complete_figure_context(candidates: list[FormulationCandidate]) -> list[FormulationCandidate]:
    """补取已引用结果段及相邻段，只附加图号、子图相符的图注原文。"""
    client = get_root_kg_service_client()
    output = []
    for candidate in candidates:
        wanted: dict[str, set[str]] = {}
        ids = []
        for record in candidate.evidence:
            if not record.chunk_id or not figure_references(record.text):
                continue
            paper = record.chunk_id.split('::', 1)[0] if '::' in record.chunk_id else record.paper_id
            for number, panels in figure_references(record.text):
                if panels and paper:
                    wanted.setdefault(paper, set()).add(number)
            ids.append(record.chunk_id)
            match = re.fullmatch(r'(.+::child)(\d+)', record.chunk_id)
            if match:
                ids.extend(match[1] + str(n) for n in (int(match[2])-1, int(match[2])+1) if n >= 0)
        if not wanted:
            output.append(candidate)
            continue
        rows = []
        try:
            payload = client.call('/v1/evidence/chunks', {'chunk_ids': list(dict.fromkeys(ids))[:20]})
            rows = payload.get('chunks', []) if isinstance(payload, dict) else []
        except RuntimeError:
            pass
        by_id = {row.get('chunk_id'): row for row in rows if isinstance(row, dict)}
        evidence = []
        for record in candidate.evidence:
            source = by_id.get(record.chunk_id, {})
            text = str(source.get('text') or '')
            # 已知段落的完整原文取代调用方预览；不把另一段落文本写到该编号上。
            if text:
                record = record.model_copy(update={'text': text, 'excerpt': text})
            evidence.append(record)
        found = []
        for paper, numbers in list(wanted.items())[:2]:
            def collect(raw_rows):
                for raw in raw_rows:
                    if not isinstance(raw, dict) or not raw.get('chunk_id') or not raw.get('text'):
                        continue
                    record = EvidenceRecord(chunk_id=raw['chunk_id'], paper_id=raw.get('paper_id') or paper,
                                            paper_title=raw.get('paper_title') or '', text=raw['text'])
                    for context in caption_contexts(record):
                        if context['paper_id'] == paper and context['figure'] in numbers:
                            found.append(record.model_copy(update={
                                'evidence_kind': 'background',
                            }))
            collect(rows)
            missing = numbers - {c['figure'] for record in found for c in caption_contexts(record)
                                 if c['paper_id'] == paper}
            if missing:
                try:
                    payload = client.call('/v1/search/paper_search', {
                        'paper_id': paper, 'query': 'Figure ' + ' '.join(sorted(missing)) +
                        ' caption electrode pouch coin cell configuration', 'top_k': 8,
                    })
                    collect(payload.get('chunks', []) if isinstance(payload, dict) else [])
                except RuntimeError:
                    pass
        # 保持已有同编号原文，避免将图注摘录覆盖完整段落。
        present = {record.chunk_id for record in evidence}
        for record in found:
            if record.chunk_id not in present:
                evidence.append(record)
                present.add(record.chunk_id)
        output.append(candidate.model_copy(update={'evidence': evidence}))
    return output


@tool
def enrich_formulation_evidence(request: Any, state: Annotated[dict | None, InjectedState] = None) -> list[dict]:
    """按候选已有的 chunk_id 只读补齐原文，不做筛选、评分或推荐。

    参数 request 至少包含 candidates；include_figure_context=true 时，沿结果中的
    显式图号补取对应图注，保留独立原文，不直接批准推荐。criteria/target_conditions 会被忽略。返回可直接
    传给 ``screen_formulations`` 的候选列表。服务不可用或 chunk 不存在时保留原记录，
    后续纯 evaluator 会产生证据缺口。
    """

    if isinstance(request, dict):
        request = normalize_retrieved_references(request, state)
    req = coerce_screening_request(request)
    candidates = hydrate_formulation_evidence(req.candidates)
    if isinstance(request, dict) and request.get('include_figure_context') is True:
        candidates = complete_figure_context(candidates)
    return [
        candidate.model_dump(mode="json")
        for candidate in candidates
    ]
