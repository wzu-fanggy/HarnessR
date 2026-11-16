"""按原文明确的图号和子图，把结果句与图注中的电池信息对应起来。"""

from __future__ import annotations

import re
from typing import Any

from orion.task_evidence.condition_extraction import extract_condition_scope
from orion.task_evidence.scientific_text import normalize_numeric_text, numeric_excerpt

_REFERENCE = re.compile(
    r"\b(?:fig(?:ure)?s?\.?)[ \t]*(?P<number>S?\d+)\s*"
    r"(?P<panels>\(?[a-z](?:\s*[-–]\s*[a-z])?\)?"
    r"(?:\s*(?:and|,)\s*[a-z])?)?(?![A-Za-z0-9])", re.I,
)
_CAPTION = re.compile(r"\b(?:Fig(?:ure)?\.?)\s*(S?\d+)\s*\.?\s*(?=\([a-z])", re.I)
_PANEL = re.compile(r"\(([a-z])(?:\s*[-–]\s*([a-z]))?\)", re.I)


def _panels(value: str) -> set[str]:
    value = value.casefold()
    span = re.search(r"([a-z])\s*[-–]\s*([a-z])", value)
    if span:
        if span[1] > span[2]:
            return set()
        return {chr(code) for code in range(ord(span[1]), ord(span[2]) + 1)}
    return set(re.findall(r"\b([a-z])\b", value))


def figure_references(text: str) -> list[tuple[str, set[str]]]:
    """只识别显式图号；无子图的引用不授权继承子图条件。"""
    return [(m['number'].upper(), _panels(m['panels'] or ''))
            for m in _REFERENCE.finditer(normalize_numeric_text(text))]


def _paper(record: Any) -> str:
    chunk_id = record.chunk_id
    return chunk_id.split('::', 1)[0] if '::' in chunk_id else record.paper_id


def caption_contexts(record: Any) -> list[dict]:
    """保留图注的逐句范围，避免把另一子图的条件搬到当前结果。"""
    contexts = []
    for head in _CAPTION.finditer(record.text):
        tail = record.text[head.end():]
        # 首句可共同描述多个子图。只取这一句的电极和电池类型，不继承温度、倍率。
        quote = numeric_excerpt(tail, 0, 1)
        panel_matches = list(_PANEL.finditer(quote))
        if not panel_matches:
            continue
        normalized = normalize_numeric_text(quote)
        configurations = set(re.findall(r'\b(pouch|coin|half-cell|full-cell)\b', normalized, re.I))
        # 同一句列出多个不同电池类型或电极对时，不据位置猜测哪个子图属于哪一种。
        pairs = set(re.findall(r'\b[A-Za-z][A-Za-z0-9]*\s*(?:/|\|\||//)\s*[A-Za-z][A-Za-z0-9]*\b', normalized))
        if len({x.casefold() for x in configurations}) > 1 or len(pairs) > 1:
            continue
        scope = extract_condition_scope(quote)
        values = {key: getattr(scope, key) for key in ('electrode', 'cell_configuration')
                  if scope is not None and getattr(scope, key) is not None}
        if not values:
            continue
        panels = set().union(*(_panels(m[0]) for m in panel_matches))
        contexts.append({'figure': head[1].upper(), 'panels': panels,
                         'chunk_id': record.chunk_id, 'paper_id': _paper(record),
                         'excerpt': record.text[head.start():head.end()] + quote,
                         'condition_scope': values})
    return contexts


def link_figure_context(candidate: Any, source: Any, excerpt: str, scope: dict) -> tuple[dict, list[dict]]:
    """只补同论文、同图同子图的电极/电池类型，保留两处引文；冲突时不合并。"""
    references = [(number, panels) for number, panels in figure_references(excerpt) if panels]
    if len(references) != 1 or not _paper(source):
        return dict(scope), []
    number, panels = references[0]
    found = []
    for record in candidate.evidence:
        if _paper(record) != _paper(source):
            continue
        for context in caption_contexts(record):
            if context['figure'] == number and panels.issubset(context['panels']):
                found.append(context)
    values = dict(scope)
    for context in found:
        for key, value in context['condition_scope'].items():
            if key in values and str(values[key]).casefold() != str(value).casefold():
                return dict(scope), []
            values[key] = value
    links = [{'chunk_id': c['chunk_id'], 'excerpt': c['excerpt'],
              'figure': number, 'panels': sorted(panels),
              'condition_scope': c['condition_scope'], 'relation': 'explicit_figure_panel'} for c in found]
    return values, links


def is_caption(text: str) -> bool:
    """图注不单独作为数值性能观测，避免混用其中不同子图的温度/倍率。"""
    return bool(_CAPTION.match(text.strip()))


def caption_numeric_ranges(text: str) -> list[tuple[int, int]]:
    """标记含明确子图标签的图注句，避免将其数值当成独立结果句。"""
    ranges = []
    for head in _CAPTION.finditer(text):
        quote = numeric_excerpt(text[head.end():], 0, 1)
        ranges.append((head.start(), head.end() + len(quote)))
    return ranges
