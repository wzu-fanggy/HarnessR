"""查询变体生成：把普通科学文本查询扩展为 mineru LaTeX 风格变体，
以匹配 mineru 解析出的 chunk 文本（chunk 中常保留 LaTeX 命令如
`$-10\\ ^{\\circ}C$`、`$NaPF_{6}$` 等）。

向量检索的 chunk embedding 是基于原始 LaTeX 文本计算的，
因此把查询也转成 mineru 风格后再 embedding，能显著提升温度、单位、
化学式等数值型内容的召回。
"""

from __future__ import annotations

import re


# mineru 常见温度表达：$-10\ ^{\circ}C$、$60\ ^{\circ}C$
# 注意空格：mineru 在部分输出中使用 "\\ "（反斜杠空格）分隔数字和 °C
_TEMPERATURE_RE = re.compile(
    r"(?<!\$)(-?\d+(?:\.\d+)?)\s*(?:°|℃|◦|\\circ\b|degrees?)?\s*C\b",
    re.IGNORECASE,
)

# 常见单位：mS cm-1、mAh g-1、Wh kg-1 等
_UNITS_RE = re.compile(
    r"\b(mS|mAh|Wh|mV|mA|mg|μA|cm|nm|mm)\s*(cm|g|kg|h|V|A|m)-1\b",
    re.IGNORECASE,
)

# 已知常见化学式：元素符号 + 数字；这里只覆盖已知无机盐/溶剂缩写，
# 避免把普通单词（如 H2O）误转。可以在 retriever 配置中扩展。
_KNOWN_FORMULAS = {
    "NaPF6": "$NaPF_{6}$",
    "NaFSI": "$NaFSI$",
    "NaTFSI": "$NaTFSI$",
    "NaClO4": "$NaClO_{4}$",
    "NaDFP": "$NaDFP$",
    "NaDFOB": "$NaDFOB$",
    "NaTFPB": "$NaTFPB$",
    "NaBOB": "$NaBOB$",
    "FEC": "$FEC$",
    "VC": "$VC$",
    "PC": "$PC$",
    "EC": "$EC$",
    "DMC": "$DMC$",
    "EMC": "$EMC$",
    "DEC": "$DEC$",
    "DME": "$DME$",
    "DEGDME": "$DEGDME$",
    "THF": "$THF$",
    "DOL": "$DOL$",
}

# 按长度降序，避免短缩写覆盖在长缩写里（如 EC 覆盖在 DEC 里）
_FORMULA_KEYS = sorted(_KNOWN_FORMULAS, key=len, reverse=True)


def _latex_temperature(match: re.Match) -> str:
    val = match.group(1)
    return f"${val}\\ ^{{\\circ}}C$"


def _latex_unit(match: re.Match) -> str:
    base, denom = match.group(1), match.group(2)
    return f"${base}\\ {denom}^{{-1}}$"


def to_mineru_latex(query: str) -> str:
    """把查询中的温度、单位、已知化学式转成 mineru 风格 LaTeX。"""
    out = query
    # 温度（带数字的 °C）
    out = _TEMPERATURE_RE.sub(_latex_temperature, out)
    # 单位
    out = _UNITS_RE.sub(_latex_unit, out)
    # 化学式：按词边界替换，避免 Na 被替换到 Nano 里
    for key in _FORMULA_KEYS:
        pattern = rf"\b{re.escape(key)}\b"
        out = re.sub(pattern, _KNOWN_FORMULAS[key], out, flags=re.IGNORECASE)
    return out


def to_plain_text(query: str) -> str:
    """把查询中已有的 LaTeX 命令剥成纯文本（用户直接贴 LaTeX 查询时使用）。"""
    out = query
    # $...$ 去掉外壳
    out = re.sub(r"\$([^$]+)\$", r"\1", out)
    # \^{\\circ}C → °C
    out = re.sub(r"\\?\^\{\\circ\}\s*C", "°C", out, flags=re.IGNORECASE)
    # \, \ 等空格命令
    out = re.sub(r"\\([ ,])", r"\1", out)
    # _{...}、^{...} 去掉花括号
    out = re.sub(r"_\{([^}]+)\}", r"\1", out)
    out = re.sub(r"\^\{([^}]+)\}", r"\1", out)
    # 残余反斜杠命令（简单兜底）
    out = re.sub(r"\\([a-zA-Z]+)\b", "", out)
    # 清理多余空格
    out = re.sub(r"\s+", " ", out).strip()
    return out


def generate_variants(query: str) -> list[str]:
    """生成查询变体列表（去重）。

    当前策略：
    1. 原始查询
    2. mineru LaTeX 风格查询
    3. 纯文本清洗查询（处理用户直接输入 LaTeX 的情况）
    """
    variants = [query, to_mineru_latex(query), to_plain_text(query)]
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out
