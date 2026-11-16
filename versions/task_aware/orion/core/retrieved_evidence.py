"""用本轮真实检索结果还原候选所引用的原文，不补选候选或实验条件。"""

from __future__ import annotations

from orion.core.task_scope import current_task_messages

from copy import deepcopy
import re
from types import SimpleNamespace
from typing import Any, Mapping

from langchain_core.messages import ToolMessage
from orion.core.evidence_extract import parse_chunk_payload


def _retrieved_sources(state: Mapping[str, Any] | None) -> dict[str, dict]:
    sources = {}
    def remember(row):
        chunk_id = row.get("chunk_id")
        content = row.get("content") or row.get("text") or row.get("excerpt")
        if not chunk_id or not content:
            return
        previous = sources.get(chunk_id)
        if previous:
            # 同一个 child 的检索预览可能只有前 1000 字，配方工具已按真实 ID
            # 取回完整文本。只在两份原文的前缀一致时保留更长版本。
            canonical = lambda value: re.sub(r"\s+", " ", str(value).replace("\\\\", "\\")).strip()
            old = canonical(previous["content"])
            new = canonical(content)
            if not (len(new) > len(old) and new.startswith(old)):
                return
        sources[chunk_id] = {**row, "content": content}
    for message in current_task_messages(state):
        if not isinstance(message, ToolMessage) or not str(message.name or "").startswith(
            ("search_literature", "search_paper_chunks", "explore_entity_neighborhood", "aggregate_evidence",
             "enrich_formulation_evidence", "query_formulation_space")
        ):
            continue
        for row in parse_chunk_payload(str(message.content)) or []:
            if isinstance(row, dict) and row.get("chunk_id") and row.get("content"):
                remember(row)
            elif isinstance(row, dict) and message.name in {"enrich_formulation_evidence", "query_formulation_space"}:
                for evidence in row.get("evidence") or []:
                    text = evidence.get("text") or evidence.get("excerpt")
                    if evidence.get("chunk_id") and text:
                        remember({**evidence, "content": text})
    return sources


def normalize_retrieved_references(request: dict[str, Any], state: Mapping[str, Any] | None) -> dict[str, Any]:
    """仅将同章节内唯一逐字引文定位到已检索 child；不猜测错误的 child ID。"""
    from orion.task_evidence.scientific_text import normalize_numeric_text
    sources = _retrieved_sources(state)
    restored = deepcopy(request)
    records = list(restored.get('evidence_chunks') or [])
    for candidate in restored.get('candidates') or []:
        records.extend(candidate.get('evidence') or [])
    canonical = lambda text: re.sub(r'\s+', '', normalize_numeric_text(str(text)))
    for record in records:
        prefix = str(record.get('chunk_id') or '')
        if prefix in sources:
            # 论文内检索也会返回真实 parent。已知的章节编号不是丢失 child 的引用。
            continue
        if not re.fullmatch(r'.+::sec\d+', prefix):
            continue
        quote = canonical(record.get('text') or record.get('content') or record.get('excerpt') or '')
        if len(quote) < 40:
            continue
        matches = [source for key, source in sources.items()
                   if re.fullmatch(re.escape(prefix) + r'::child\d+', key)
                   and quote in canonical(source['content'])]
        if len(matches) == 1:
            record.update(chunk_id=matches[0]['chunk_id'], parent_chunk_id=prefix)
    return restored


def restore_candidate_source_text(request: dict[str, Any], state: Mapping[str, Any] | None) -> dict[str, Any]:
    """还原已选原文，并附上本轮已检索的对应子图图注；不检索、不改约束。"""
    sources = _retrieved_sources(state)
    if not sources:
        return request
    restored = normalize_retrieved_references(request, state)
    prior_roles = {}
    for item in (state or {}).get('evidence_ledger') or []:
        row = item.model_dump(mode='python') if hasattr(item, 'model_dump') else item
        if isinstance(row, dict):
            key = (row.get('candidate_id'), row.get('chunk_id'))
            prior_roles.setdefault(key, set()).add(row.get('evidence_kind') or 'background')
    for candidate in restored.get("candidates") or []:
        for evidence in candidate.get("evidence") or []:
            chunk_id = evidence.get("chunk_id")
            source = sources.get(chunk_id)
            if source is None:
                # 有本轮检索结果时，不能用模型自写原文或未检索 ID 冒充证据。
                evidence.update(text="", excerpt="", condition_scope=None)
                continue
            if not evidence.get('evidence_kind'):
                roles = prior_roles.get((candidate.get('candidate_id'), chunk_id), set())
                if len(roles) == 1:
                    evidence['evidence_kind'] = next(iter(roles))
            evidence.update(text=source["content"], excerpt=source["content"],
                            paper_id=source.get("paper_id") or source.get("paper_title") or "",
                            paper_title=source.get("paper_title") or "", condition_scope=None,
                            experiment_id=source.get("experiment_id"),
                            scientific_object_id=source.get("scientific_object_id"))
        # 性能段引用具体子图时，保留本轮已经取到的对应图注。
        # 只补原文，不把同篇论文的其它性能段加入候选，不填模型猜测的条件。
        from orion.tools.electrolyte_design.figure_context import figure_references, caption_contexts
        references = []
        present = set()
        for evidence in candidate.get('evidence') or []:
            chunk_id = str(evidence.get('chunk_id') or '')
            present.add(chunk_id)
            paper = chunk_id.split('::', 1)[0] if '::' in chunk_id else evidence.get('paper_id')
            references.extend((paper, figure, panels)
                              for figure, panels in figure_references(evidence.get('text') or '') if panels)
        for chunk_id, source in sources.items():
            if chunk_id in present:
                continue
            record = SimpleNamespace(chunk_id=chunk_id, paper_id=source.get('paper_id') or '', text=source['content'])
            contexts = caption_contexts(record)
            if any(context['paper_id'] == paper and context['figure'] == figure
                   and panels.issubset(context['panels'])
                   for context in contexts for paper, figure, panels in references):
                candidate.setdefault('evidence', []).append({
                    'chunk_id': chunk_id, 'paper_id': source.get('paper_id') or chunk_id.split('::', 1)[0],
                    'paper_title': source.get('paper_title') or '', 'text': source['content'],
                    'excerpt': source['content'], 'evidence_kind': 'background',
                })
    return restored
