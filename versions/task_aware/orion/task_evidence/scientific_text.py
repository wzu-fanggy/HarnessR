"""为数值检查移除论文排版标记；不修改保存的原文证据。"""

from __future__ import annotations

import re


def normalize_numeric_text(text: str) -> str:
    """保留数值、正负号与单位，兼容 MinerU 的普通及双转义 LaTeX。"""
    for _ in range(3):
        text = text.replace("\\\\", "\\")
    text = text.replace("−", "-").replace("$", "").replace(r"\%", "%")
    text = re.sub(r"\\(?:mathrm|text|textrm)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}|\^\s*\\circ", "°", text)
    # LaTeX 的显式空格可位于数值和单位之间（例如 -20\ ^{\circ}C）。
    # 只去掉排版空格命令，不删除负号、指数或未知命令。
    text = re.sub(r"\\\s+|\\[,;!]|\\quad\b|\\qquad\b", " ", text)
    return text


def numeric_excerpt(text: str, start: int, end: int) -> str:
    """保留数值所在句，不把小数点或 Fig./Eq. 等缩写当作句界。"""
    boundaries = []
    abbreviation = re.compile(r"\b(?:figs?|eqs?|refs?|vs|e\.g|i\.e|et\s+al)$", re.I)
    for match in re.finditer(r"\.(?!\d)|[;。\n]", text):
        point = match.start()
        # e.g. / i.e. 的第一处点也属于缩写，不能把保持率列表切掉前半句。
        if match.group() == "." and re.search(r"\b(?:e\.g\.|i\.e\.)", text[max(0, point - 3):point + 4], re.I):
            continue
        if match.group() == "." and abbreviation.search(text[max(0, point - 16):point]):
            continue
        boundaries.append(point)
    left = max((point + 1 for point in boundaries if point < start), default=0)
    right = min((point + 1 for point in boundaries if point >= end), default=len(text))
    # 长句必须保留被检查的数值，不能固定取前 500 字而丢掉句尾观测。
    if right - left > 1200:
        left = max(left, start - 600)
        right = min(right, max(end, left + 1200))
    return text[left:right].strip()


def design_result_excerpt(text: str) -> str:
    """优先展示数值结果所在的完整原文句，保持连续引文，不生成性能解释。"""
    numeric_results = list(re.finditer(
        r"\d+(?:\.\d+)?\s*(?:%|mAh|mS\b|cycles\b)", text, re.I,
    ))
    if numeric_results:
        excerpts = [numeric_excerpt(text, match.start(), match.end()) for match in numeric_results]
        return max(excerpts, key=lambda quote: len(re.findall(
            r"\d+(?:\.\d+)?\s*(?:%|mAh|mS\b|cycles\b)", quote, re.I,
        )))
    for match in re.finditer(r"\S", text):
        quote = numeric_excerpt(text, match.start(), match.end())
        if len(quote) >= 30:
            return quote
    return text.strip()
