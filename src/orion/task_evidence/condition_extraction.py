"""从证据块自由文本中机器可审计地抽取 ComparisonConditions。

本模块不调用 LLM、RAG 或 KG；只使用确定性规则从 chunk 原文中识别实验条件，
用于在 RAG 后端未返回结构化条件时，将 ``EvidenceChunk`` 补充为可参与
condition-bound 校验的记录。
"""

from __future__ import annotations

import re
from typing import Any

from orion.task_evidence.task_gates import ComparisonConditions
from orion.task_evidence.scientific_text import normalize_numeric_text


# ---------- 常量正则 ----------

_NUMBER = r"[-+−–]?(?:\d+(?:\.\d+)?|\.\d+)"

# 温度：-20 °C, 25°C, 25 ℃, room temperature, RT
_TEMPERATURE_RE = re.compile(
    rf"(?P<temp>{_NUMBER})\s*(?:°\s*[CFK]|℃|°C|°F|°K)"
    rf"|(?P<signed_plain_temp>[-+−–]\d+(?:\.\d+)?)\s*[CFK]\b"
    rf"|(?:room\s+temperature|\.rt\.\b|\.room\s+temp\.\b)",
    re.IGNORECASE,
)

# 数值温度与 room temperature 分开匹配：当句中同时出现两者时，room temperature
# 常是容量保持率的参照点而非测试温度（如 "maintain 83% of room temperature
# capacity at -30°C"），必须优先取显式数值温度，不能把参照温度误当成测试温度。
_NUMERIC_TEMPERATURE_RE = re.compile(
    rf"(?P<temp>{_NUMBER})\s*(?:°\s*[CFK]|℃|°C|°F|°K)"
    rf"|(?P<signed_plain_temp>[-+−–]\d+(?:\.\d+)?)\s*[CFK]\b",
    re.IGNORECASE,
)
_ROOM_TEMP_RE = re.compile(
    r"(?:room\s+temperature|\.rt\.\b|\.room\s+temp\.\b)",
    re.IGNORECASE,
)

# 电压窗口：0.003 V to 2 V, 0.003–2 V, 0.01-2.0 V, between 0.003 and 2 V
_VOLTAGE_WINDOW_RE = re.compile(
    rf"(?:between\s+)?(?P<low>{_NUMBER})\s*(?:V|v)?\s*"
    rf"(?:to|and|[-–—])\s*(?P<high>{_NUMBER})\s*(?:V|v)"
    rf"|cycled\s+from\s+(?P<low2>{_NUMBER})\s*(?:V|v)\s*"
    rf"(?:to|[-–—])\s*(?P<high2>{_NUMBER})\s*(?:V|v)",
    re.IGNORECASE,
)

# 倍率：0.1 C, C/10, C/20, 2C, 1 C
_RATE_RE = re.compile(
    rf"(?P<rate>(?:C/\d+(?:\.\d+)?|"
    rf"(?<![-+−–\d.])(?:\d+(?:\.\d+)?\s*)?C\b))",
    re.IGNORECASE,
)

# 循环数：135 cycles, 1000 cycles, 1000 圈, 1000 cycles
_CYCLE_COUNT_RE = re.compile(
    rf"(?P<cycles>\d+)\s*(?:cycles?|圈|次循环|cycles)"
    rf"|(?:after|for)\s+(?P<cycles2>\d+)\s*(?:cycles?|圈)",
    re.IGNORECASE,
)

# 电极 / 电池构型：HC//Na, HC || Na, half-cell, full cell, hard carbon
_ELECTRODE_RE = re.compile(
    r"(?:"
    r"HC\s*(?://|\\|/|\\|\|\\|\|\|)\s*Na"
    r"|hard\s+carbon\s*(?:anode|electrode)?"
    r"|HC\s+(?:anode|electrode|half-cell)"
    r"|Na\s*metal\s*(?:counter|reference)?"
    r"|half[\s-]?cell"
    r"|full[\s-]?cell"
    r"|coin\s+cell"
    r"|three[\s-]?electrode"
    r"|two[\s-]?electrode"
    r"|Na/Na"
    r"|HC-based\s+electrodes"
    r")",
    re.IGNORECASE,
)

_CELL_PAIR_RE = re.compile(
    r"\b(?P<left>[A-Za-z][A-Za-z0-9]*)\s*(?://|\|\||/)\s*"
    r"(?P<right>[A-Za-z][A-Za-z0-9]*|hard[ -]+carbon)\b"
    r"\s+(?:pouch\s+)?(?:full[ -]+)?cells?\b",
    re.IGNORECASE,
)

# 测量定义：GCPL, galvanostatic, CV, EIS, charge-discharge, capacity retention, cycling
_MEASUREMENT_RE = re.compile(
    r"(?:"
    r"galvanostatic\s*(?:charge[\s/-]?discharge|mode|cycling|with\s+potential\s+limitation)?"
    r"|GCPL"
    r"|cyclic\s+voltammetry"
    r"|CV\s+(?:curves?|measurements?)"
    r"|electrochemical\s+impedance\s+spectroscopy"
    r"|EIS"
    r"|charge[\s/-]?discharge"
    r"|capacity\s+retention"
    r"|capacity\s+evolution"
    r"|rate\s+capability"
    r"|cycling\s+stability"
    r"|\b(?:were\s+)?cycled\b"
    r"|\b(?:charge-discharge\s+)?cycling\b"
    r")",
    re.IGNORECASE,
)

# 容量保持率相对参照的写法：maintain/retain/preserve/keep X% of ... capacity。
# 参照可为 room temperature / its / the / initial / original，且可带 25 °C 等数值。
_RETENTION_RELATIVE_RE = re.compile(
    r"\b(?:maintain|retain|preserv|keep|deliver)(?:s|ed|es|e)?\s+"
    r"\d+(?:\.\d+)?\s*%\s+of\s+"
    r"(?:(?:the|its)\s+)?"
    r"(?:(?:room[ -]temperature|original|initial|nominal|25\s*(?:°\s*C|℃))\s+)?"
    r"capacity",
    re.IGNORECASE,
)


def _canonical_number(value: str) -> float:
    """把中英文破折号统一为负号再转 float。"""
    cleaned = value.replace("−", "-").replace("–", "-").replace("—", "-")
    return float(cleaned)


def _explicit_temperature_c(text: str) -> float | None:
    """优先返回显式数值温度；无数值温度时才把 room temperature / RT 解析为 25.0。

    不触发电化学测试的室温默认推断（该推断只属于 complete-record 抽取器）。
    """
    numeric = _NUMERIC_TEMPERATURE_RE.search(text)
    if numeric:
        temp_str = numeric.group("temp") or numeric.group("signed_plain_temp")
        if temp_str is not None:
            return round(_canonical_number(temp_str), 6)
    if _ROOM_TEMP_RE.search(text):
        return 25.0
    return None


def _extract_temperature_c(text: str) -> float | None:
    """抽取温度（摄氏度）。

    优先取显式数值温度；无数值温度时才把 room temperature / RT 解析为 25.0。
    若文本描述电化学测试但未提及温度，则保守推断为室温 25.0 °C
    （电化学文献常见默认）。
    """
    explicit = _explicit_temperature_c(text)
    if explicit is not None:
        return explicit
    # 若文本包含电压/倍率/循环等测试要素，默认推断室温
    if re.search(r"(?i)cycled|voltage|capacity|galvanostatic|charge-discharge", text):
        return 25.0
    return None


def _extract_voltage_window_v(text: str) -> tuple[float, float] | None:
    """抽取电压窗口 [low, high]。"""
    match = _VOLTAGE_WINDOW_RE.search(text)
    if not match:
        return None
    low = match.group("low") or match.group("low2")
    high = match.group("high") or match.group("high2")
    if not low or not high:
        return None
    low_v = round(_canonical_number(low), 6)
    high_v = round(_canonical_number(high), 6)
    if low_v >= high_v:
        return None
    return (low_v, high_v)


def _extract_rate(text: str) -> str | None:
    """抽取倍率，优先 C/xx 形式，再匹配 N C 形式；跳过孤立的 'C'。"""
    # 优先 C/xx，更精确
    slash = re.search(r"(?i)C/\d+(?:\.\d+)?", text)
    if slash:
        return slash.group(0).strip()
    for match in _RATE_RE.finditer(text):
        rate = match.group("rate").strip()
        # 过滤掉纯 "C" 或空（常见于 EC:DMC 等化学式）
        if not rate or re.fullmatch(r"(?i)\s*C\s*", rate):
            continue
        return rate
    return None


def _extract_cycle_count(text: str) -> int | None:
    """抽取循环数。"""
    match = _CYCLE_COUNT_RE.search(text)
    if not match:
        return None
    cycles = match.group("cycles") or match.group("cycles2")
    if not cycles:
        return None
    return int(cycles)


def _extract_electrode(text: str) -> str | None:
    """抽取电极/电池构型描述。"""
    legacy = _ELECTRODE_RE.search(text)
    if legacy and legacy.group(0).lower().startswith("hc") and "na" in legacy.group(0).lower():
        return " ".join(legacy.group(0).split())
    # 从同一个明确的正负极叙述保留两种材料，不把“全电池”或单个电极当完整电极对。
    material = r"[A-Za-z][A-Za-z0-9().]*(?:[ -]+[A-Za-z0-9().]+){0,2}"
    roles = re.search(
        rf"(?:consisting\s+of|based\s+on|composed\s+of|using|with)\s+"
        rf"(?:an?\s+|the\s+)?(?P<anode>{material})\s+anodes?\s+and\s+"
        rf"(?:an?\s+|the\s+)?(?P<cathode>{material})\s+cathodes?\b", text, re.I,
    )
    if roles:
        return f"{roles.group('anode')}/{roles.group('cathode')}"
    compact = re.search(
        r"\b(?P<left>[A-Z][A-Za-z0-9().]*(?:\s+[a-z]+)?)\s*[–—-]\s*"
        r"(?P<right>[A-Za-z][A-Za-z0-9().]*(?:\s+[a-z]+)?)\s+full[ -]cells?\b", text,
    )
    if compact:
        return f"{compact.group('left')}/{compact.group('right')}"
    pair = _CELL_PAIR_RE.search(text)
    # 单斜线的电极对只在明确的电池/电极叙述中识别，避免把 EC/EMC 当成电极。
    if pair and ("/" not in pair.group(0).replace("//", "").replace("||", "")
                 or re.search(r"\b(?:cells?|cathodes?|anodes?|batter(?:y|ies))\b", text, re.I)):
        return f"{pair.group('left')}/{pair.group('right')}"
    match = _ELECTRODE_RE.search(text)
    if not match:
        return None
    return " ".join(match.group(0).split())


def _extract_cell_configuration(text: str) -> str | None:
    """只从原句显式表述中抽取可审计的 cell configuration。"""

    if re.search(r"\bpouch(?:[\s-]+[A-Za-z0-9|/]+){0,3}[\s-]+(?:cells?|batteries)\b", text, re.IGNORECASE):
        return "pouch"
    if re.search(r"\bcoin\s+cells?\b", text, re.IGNORECASE):
        return "coin"
    if re.search(r"\bhalf[\s-]?cells?\b", text, re.IGNORECASE):
        return "half-cell"
    if re.search(r"\bfull[\s-]?cells?\b", text, re.IGNORECASE):
        return "full-cell"
    pair = _CELL_PAIR_RE.search(text)
    if pair:
        return f"{pair.group('left')}||{pair.group('right')} cell"
    # MinerU/PDF OCR 常把 ``||`` 识别为两个小写 l，例如 NallHC / NallNVP。
    # 只接受 Na + ll + 简单电极 token + cell 的明确模式，避免一般单词被误改。
    ocr_pair = re.search(
        r"\bNa(?:ll)(?P<right>[A-Za-z][A-Za-z0-9]*)\s+cells?\b",
        text,
        re.IGNORECASE,
    )
    if ocr_pair:
        return f"Na||{ocr_pair.group('right')} cell"
    return None


def _extract_measurement_definition(text: str) -> str | None:
    """抽取测量定义。"""
    # "maintain/retain/preserve X% of (room temperature/its/the/initial/original)
    # capacity" 是"相对于室温容量的容量保持率"的另一种写法，优先于泛化的 cycling。
    if _RETENTION_RELATIVE_RE.search(text):
        return "capacity retention"
    match = _MEASUREMENT_RE.search(text)
    if match:
        return " ".join(match.group(0).split())
    return None


def extract_condition_scope(text: str) -> ComparisonConditions | None:
    """Extract every condition field stated explicitly in the source text.

    Unlike :func:`extract_comparison_conditions`, this function preserves a
    partial scope. It never fills an omitted temperature with a room-temperature
    default. The result is suitable for provenance metadata, but a partial scope
    must not by itself authorize a ``condition_bound`` claim or comparability.
    """

    if not text:
        return None
    text = normalize_numeric_text(text)
    temperature_c = _explicit_temperature_c(text)
    scope = ComparisonConditions(
        temperature_c=temperature_c,
        voltage_window_v=_extract_voltage_window_v(text),
        rate=_extract_rate(text),
        cycle_count=_extract_cycle_count(text),
        electrode=_extract_electrode(text),
        cell_configuration=_extract_cell_configuration(text),
        measurement_definition=_extract_measurement_definition(text),
    )
    if all(getattr(scope, field) is None for field in (
        "temperature_c",
        "voltage_window_v",
        "rate",
        "cycle_count",
        "electrode",
        "cell_configuration",
        "measurement_definition",
    )):
        return None
    return scope


def extract_comparison_conditions(text: str) -> ComparisonConditions | None:
    """从文本中抽取完整实验条件；缺失任何字段均返回 None（fail-closed）。"""
    scope = extract_condition_scope(text)
    if scope is None:
        return None
    temperature_c = scope.temperature_c
    # Preserve the historical room-temperature convention only for the strict
    # complete-record extractor. Partial provenance never invents this field.
    if temperature_c is None:
        temperature_c = _extract_temperature_c(text)
    voltage_window_v = scope.voltage_window_v
    rate = scope.rate
    cycle_count = scope.cycle_count
    electrode = scope.electrode
    measurement_definition = scope.measurement_definition
    if any(
        field is None
        for field in (
            temperature_c,
            voltage_window_v,
            rate,
            cycle_count,
            electrode,
            measurement_definition,
        )
    ):
        return None
    return ComparisonConditions(
        temperature_c=temperature_c,
        voltage_window_v=voltage_window_v,
        rate=rate,
        cycle_count=cycle_count,
        electrode=electrode,
        cell_configuration=scope.cell_configuration,
        measurement_definition=measurement_definition,
    )


def infer_experiment_id(
    *,
    chunk_id: str,
    paper_id: str,
    conditions: ComparisonConditions | dict[str, Any] | None,
) -> str:
    """为没有 experiment_id 的 chunk 生成稳定标识。

    优先使用 ``paper_id:chunk_id``；若条件可解析，则附加条件哈希，使同一 chunk
    内因不同条件集合生成的记录也能区分。
    """
    base = f"{paper_id or 'unknown'}:{chunk_id or 'unknown'}"
    if not conditions:
        return base
    if isinstance(conditions, dict):
        from orion.task_evidence.task_gates import ComparisonConditions
        scope = ComparisonConditions.model_validate(conditions)
    else:
        scope = conditions
    parts = [
        str(scope.temperature_c),
        str(scope.voltage_window_v),
        str(scope.rate),
        str(scope.cycle_count),
        str(scope.electrode),
        str(scope.cell_configuration),
        str(scope.measurement_definition),
    ]
    import hashlib

    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{base}:{digest}"


def infer_condition_quote(text: str, max_length: int = 240) -> str:
    """生成条件引文：直接返回前 max_length 字符作为可匹配的唯一子串。"""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[:max_length]
