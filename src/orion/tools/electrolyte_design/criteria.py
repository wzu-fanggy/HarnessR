from __future__ import annotations



import re

import math

from dataclasses import dataclass

from typing import Literal



from tools.kg.models import (

    CriterionEvaluation,

    FormulationCandidate,

    ScreeningConstraint,

)

from orion.task_evidence.condition_extraction import extract_condition_scope

from .scientific_numeric_text import normalize_numeric_text, numeric_excerpt

from .figure_context import link_figure_context, is_caption, caption_numeric_ranges

from .numeric_conditions import retention_temperature_pairs, same_cell_voltage_protocol, same_paragraph_cell_context





CriterionKind = Literal[

    "minimum_voltage",

    "c_rate",

    "minimum_cycles",

    "minimum_retention",

    "minimum_conductivity",

    "maximum_viscosity",

    "temperature",

    "required_component",

    "excluded_component",

    "mechanism_keyword",

    "evidence_claim",

]





@dataclass(frozen=True)

class ParsedCriterion:

    raw: str

    kind: CriterionKind

    operator: Literal[">=", "<=", "==", "contains", "absent"] | None = None

    value: float | None = None

    unit: str | None = None

    component: str | None = None





_NUMBER = r"(?<![\d.])(?P<value>-?\d+(?:\.\d+)?)"

_VOLTAGE = re.compile(rf"{_NUMBER}\s*V\b", re.I)

_VOLTAGE_WINDOW = re.compile(

    r"(?<![\d.])(?P<lower>-?\d+(?:\.\d+)?)\s*(?:[-–—]|to)\s*"

    r"(?P<upper>-?\d+(?:\.\d+)?)\s*V\b", re.I,

)

_C_RATE = re.compile(

    r"(?<![\w.°℃-])(?P<value>\d+(?:\.\d+)?)\s*C\b(?!\s*/\s*\d)"

    r"|\bC\s*/\s*(?P<inverse_rate>\d+(?:\.\d+)?)", re.I,

)

_CYCLES = re.compile(rf"{_NUMBER}\s*cycles?\b", re.I)

_RETENTION = re.compile(

    r"(?:capacity\s+retention|容量保持率)[^\d]{0,20}(?P<retention_after>-?\d+(?:\.\d+)?)\s*%"

    r"|(?P<retention_before>-?\d+(?:\.\d+)?)\s*%\s*(?:capacity\s+retention|容量保持率|capacity)"

    r"|(?:retain(?:s|ed)?|保持)[^\d]{0,20}(?P<retention_retained>-?\d+(?:\.\d+)?)\s*%"

    r"|maintain(?:s|ed)?\s+(?P<retention_maintained>\d+(?:\.\d+)?)\s*%(?=[^.;]{0,70}\bcapacity\b)",

    re.I,

)

_CONDUCTIVITY = re.compile(rf"{_NUMBER}\s*(?:mS\s*(?:/|cm[-−]?1)|S\s*cm[-−]?1)", re.I)

_VISCOSITY = re.compile(rf"{_NUMBER}\s*(?:mPa\s*·?\s*s|cP)\b", re.I)

# A bare ``C`` is ambiguous with a charge rate (for example ``1 C``).

_TEMPERATURE = re.compile(rf"{_NUMBER}\s*(?:°\s*C|℃)", re.I)





def _criterion_number(text: str) -> float | None:

    match = re.search(r"-?\d+(?:\.\d+)?", text)

    return float(match.group(0)) if match else None





def _threshold_after(text: str, marker: str) -> float | None:

    # Do not let the separator consume the minus sign in ``temperature -20°C``.

    match = re.search(rf"{marker}[^\d-]{{0,24}}(?:≥|>=|at\s+least\s+)?(-?\d+(?:\.\d+)?)", text, re.I)

    return float(match.group(1)) if match else None





def parse_criterion(raw: str) -> list[ParsedCriterion]:

    """将一条自然语言约束拆为可独立审计的原子条件。"""

    text = raw.strip()

    lowered = text.lower()

    parsed: list[ParsedCriterion] = []



    # These are compositional/application constraints, not numeric properties.

    # Interpret them explicitly rather than reporting them as unsupported.

    if "carbonate" in lowered or "碳酸酯" in text:

        parsed.append(ParsedCriterion(text, "required_component", "contains", component="__carbonate__"))

    primary_salt = re.search(r"([A-Za-z][A-Za-z0-9]*)\s+as\s+(?:the\s+)?primary\s+salt", text, re.I)

    if primary_salt:

        parsed.append(ParsedCriterion(text, "required_component", "contains", component=primary_salt.group(1)))

    if "sodium-ion" in lowered or "sodium ion" in lowered or "钠离子" in text:

        parsed.append(ParsedCriterion(text, "mechanism_keyword", "contains", component="sodium"))



    voltage = _threshold_after(text, r"(?:voltage|电压|高压)")

    if voltage is None and re.search(r"-?\d+(?:\.\d+)?\s*V\b", text, re.I):

        voltage = _criterion_number(text)

    if voltage is not None:

        parsed.append(ParsedCriterion(text, "minimum_voltage", ">=", voltage, "V"))

    # A temperature qualifier in e.g. ``capacity retention at -20°C`` is not

    # a retention threshold.  Only parse this criterion when a percentage is

    # actually requested.

    retention = _threshold_after(text, r"(?:capacity\s+retention|容量保持率)") if "%" in text else None

    if retention is not None:

        parsed.append(ParsedCriterion(text, "minimum_retention", ">=", retention, "%"))

    cycles = None

    if "cycle" in lowered or "循环" in text:

        cycle_matches = re.findall(r"(?:≥|>=|at\s+least\s+)?(\d+(?:\.\d+)?)\s*cycles?", text, re.I)

        cycles = float(cycle_matches[-1]) if cycle_matches else None

    if cycles is not None:

        parsed.append(ParsedCriterion(text, "minimum_cycles", ">=", cycles, "cycles"))

    conductivity = _threshold_after(text, r"(?:conductivity|电导)")

    if conductivity is not None:

        parsed.append(ParsedCriterion(text, "minimum_conductivity", ">=", conductivity, "mS/cm"))

    viscosity = _threshold_after(text, r"(?:viscosity|粘度)")

    if viscosity is not None:

        parsed.append(ParsedCriterion(text, "maximum_viscosity", "<=", viscosity, "mPa s"))

    temperature = _threshold_after(text, r"(?:temperature|温度)")

    if temperature is None and ("°c" in lowered or "℃" in text):

        temperature = _criterion_number(text)

    if temperature is not None and ("°c" in lowered or "℃" in text or "temperature" in lowered or "温度" in text):

        # A low-temperature target requires demonstrated operation at a

        # temperature at least as cold as the requested value.

        operator = "<=" if temperature < 0 or "low temperature" in lowered else ">="

        parsed.append(ParsedCriterion(text, "temperature", operator, temperature, "°C"))



    required = re.search(r"(?:contains?|must\s+(?:contain|include)|需要包含|必须包含)\s+(.+)$", text, re.I)

    if required:

        component = re.split(r"\s+as\s+|\s+additive", required.group(1), maxsplit=1, flags=re.I)[0].strip()

        parsed.append(ParsedCriterion(text, "required_component", "contains", component=component))

    excluded = re.search(r"(?:exclude|without|不得含有|排除)\s+(.+)$", text, re.I)

    if excluded:

        parsed.append(ParsedCriterion(text, "excluded_component", "absent", component=excluded.group(1).strip()))

    if "film-forming" not in lowered and "成膜添加剂" not in text and (any(keyword in lowered for keyword in ("cei", "sei", "oxidation", "antioxidation")) or any(

        keyword in text for keyword in ("界面", "抗氧化", "抗氧化分解")

    )):

        keyword = next((item for item in ("cei", "sei", "oxidation") if item in lowered), None)

        parsed.append(ParsedCriterion(text, "mechanism_keyword", "contains", component=keyword or ("cei" if "界面" in text else "oxidation")))

    return parsed





def _excerpt(text: str, start: int, end: int) -> str:

    return numeric_excerpt(text, start, end)





def _observations(candidate: FormulationCandidate, pattern: re.Pattern, unit: str) -> list[dict]:

    output = []

    for record in candidate.evidence:

        if is_caption(record.text):

            continue

        # MinerU and paper chunks may encode a temperature as LaTeX

        # (``-20^{\\circ}C``). Keep the source record unchanged for provenance.

        text = normalize_numeric_text(record.text)

        caption_ranges = caption_numeric_ranges(text)

        for match in pattern.finditer(text):

            if any(start <= match.start() < end for start, end in caption_ranges):

                continue

            value = next((group for group in match.groupdict().values() if group is not None), None)

            if value is None:

                continue

            value = float(value)

            if pattern is _C_RATE and match.groupdict().get("inverse_rate") is not None:

                if value <= 0:

                    continue

                value = 1.0 / value

            excerpt = _excerpt(text, match.start(), match.end())

            scope = extract_condition_scope(excerpt)

            scope_payload = scope.model_dump(mode="json", exclude_none=True) if scope is not None else {}

            if pattern is _C_RATE:

                # 列表段落可能同时含多个倍率；该观测的条件必须对应实际命中的倍率。

                scope_payload["rate"] = f"{value:g} C"

            if pattern is _TEMPERATURE:

                scope_payload['temperature_c'] = value

            elif pattern is _RETENTION and match.groupdict().get('retention_maintained') is not None:

                # maintain 的容量比例只由后续明确的容量—温度关系补入温度，

                # 不借用同句其他行为（例如低温启动用电器）的温度。

                scope_payload.pop('temperature_c', None)

            elif len({float(m['value']) for m in _TEMPERATURE.finditer(excerpt)}) > 1:

                # 多温度句中的其他指标不能默认对应首个温度。

                scope_payload.pop('temperature_c', None)

            scope_payload, cell_links = same_paragraph_cell_context(text, excerpt, scope_payload)

            scope_payload, protocol_links = same_cell_voltage_protocol(text, excerpt, scope_payload)

            protocol_links = cell_links + protocol_links

            for link in protocol_links:

                link['chunk_id'] = record.chunk_id

            scope_payload, context_evidence = link_figure_context(candidate, record, excerpt, scope_payload)

            context_evidence = protocol_links + context_evidence

            observation = {

                "value": value,

                "unit": unit,

                "chunk_id": record.chunk_id,

                "excerpt": excerpt,

            }

            if scope_payload:

                observation["condition_scope"] = scope_payload

            if context_evidence:

                observation["context_evidence"] = context_evidence

            # 同一次低温倍率测试可报告多个倍率。为温度观测分别保存明确的倍率，

            # 不把列表首项当成唯一倍率，也不把不同温度的测试拼成联合观测。

            rates = list(_C_RATE.finditer(excerpt))

            temperatures = {float(m["value"]) for m in _TEMPERATURE.finditer(excerpt)}

            if pattern is _TEMPERATURE and len(temperatures) == 1 and rates:

                for rate_match in rates:

                    if rate_match["inverse_rate"] and float(rate_match["inverse_rate"]) <= 0:

                        continue

                    rate = (float(rate_match["value"]) if rate_match["value"] else

                            1 / float(rate_match["inverse_rate"]))

                    output.append({**observation, "condition_scope": {

                        **scope_payload, "rate": f"{rate:g} C",

                    }})

            else:

                output.append(observation)

        if pattern is _RETENTION:

            pairs = retention_temperature_pairs(text)

            # 明确成对列表替换同句的首项匹配，避免留下温度未定的重复观测。

            pair_quotes = {p['excerpt'] for p in pairs}

            output = [o for o in output if not (o['chunk_id'] == record.chunk_id and o['excerpt'] in pair_quotes)]

            for pair in pairs:

                quote = pair['excerpt']

                scope = extract_condition_scope(quote)

                payload = scope.model_dump(mode='json', exclude_none=True) if scope else {}

                payload['temperature_c'] = pair['temperature_c']

                payload, links = link_figure_context(candidate, record, quote, payload)

                observation = {'value': pair['value'], 'unit': unit, 'chunk_id': record.chunk_id,

                               'excerpt': quote, 'condition_scope': payload}

                if links:

                    observation['context_evidence'] = links

                output.append(observation)

    return output





def _numeric_evaluation(parsed: ParsedCriterion, candidate: FormulationCandidate, pattern: re.Pattern) -> CriterionEvaluation:

    observations = _observations(candidate, pattern, parsed.unit or "")

    expected = {"operator": parsed.operator, "value": parsed.value, "unit": parsed.unit}

    if not observations:

        return CriterionEvaluation(

            criterion=parsed.raw,

            kind=parsed.kind,

            status="insufficient_evidence",

            expected=expected,

            reason=f"缺少可追溯的候选证据来验证“{parsed.raw}”。",

        )

    def _satisfies(item: dict) -> bool:

        value = float(item["value"])

        if parsed.operator == "==":

            return math.isclose(value, parsed.value, rel_tol=1e-9, abs_tol=1e-9)

        return value >= parsed.value if parsed.operator == ">=" else value <= parsed.value



    satisfying_observations = [item for item in observations if _satisfies(item)]

    if satisfying_observations:

        # ``observed`` 在 status=satisfied 时同时承担下游 support binding。

        # 只能暴露真正满足阈值的 records；否则 joint-context solver 可能把

        # “上下文兼容但数值不合格”的记录误当成 alternative support。

        return CriterionEvaluation(

            criterion=parsed.raw,

            kind=parsed.kind,

            status="satisfied",

            expected=expected,

            observed=satisfying_observations,

            reason=f"候选原文证据中存在满足“{parsed.raw}”的观测值。",

        )

    return CriterionEvaluation(

        criterion=parsed.raw,

        kind=parsed.kind,

        status="failed",

        expected=expected,

        observed=observations,

        reason=f"候选原文证据中的可比较观测值不满足“{parsed.raw}”。",

    )





def _component_evaluation(parsed: ParsedCriterion, candidate: FormulationCandidate) -> CriterionEvaluation:

    components = {item.lower() for item in [candidate.salt, *candidate.salts, *candidate.solvents, *candidate.additives] if item}

    component_text = (parsed.component or "").lower()

    carbonate_solvents = {"ec", "pc", "emc", "dmc", "dec", "fec", "vc"}

    film_formers = {"fec", "vc", "dfec", "vad", "na dfp", "nadfp"}

    if component_text == "__carbonate__":

        present = bool(components & carbonate_solvents)

    elif "film-forming" in component_text or "成膜" in component_text:

        present = bool(components & film_formers)

    elif ("/" in component_text or "+" in component_text) and not re.search(

        r"\s+(?:and/or|or|and)\s+", component_text

    ):

        # 结构化约束里 ``G2/DOL``、``EC+DMC`` 表示一个明确的混合组分合同，

        # 不是新的化学实体名。只按字面分隔符拆分并要求各组件均存在；不做

        # 同义词或化学名称猜测，实体归一化仍由上游负责。

        expected_components = [

            part.strip().lower()

            for part in re.split(r"\s*(?:/|\+)\s*", parsed.component or "")

            if part.strip()

        ]

        present = bool(expected_components) and all(

            item in components for item in expected_components

        )

    else:

        expected_components = [part.strip().lower() for part in re.split(r"\s+(?:and/or|or|and)\s+", parsed.component or "") if part.strip()]

        present = bool(expected_components) and any(item in components for item in expected_components)

    satisfied = present if parsed.operator == "contains" else not present

    return CriterionEvaluation(

        criterion=parsed.raw,

        kind=parsed.kind,

        status="satisfied" if satisfied else "failed",

        expected={"operator": parsed.operator, "component": parsed.component},

        observed=[{"component": item} for item in sorted(components)],

        reason=("候选组分满足条件。" if satisfied else "候选组分不满足条件。"),

    )





def _keyword_evaluation(parsed: ParsedCriterion, candidate: FormulationCandidate) -> CriterionEvaluation:

    keyword = (parsed.component or "").lower()

    observations = [

        {"chunk_id": record.chunk_id, "excerpt": record.text[:500]}

        for record in candidate.evidence

        if keyword in record.text.lower()

    ]

    return CriterionEvaluation(

        criterion=parsed.raw,

        kind=parsed.kind,

        status="satisfied" if observations else "insufficient_evidence",

        expected={"keyword": parsed.component},

        observed=observations,

        reason=("候选原文证据直接提及该机理关键词。" if observations else "缺少直接提及该机理关键词的候选原文证据。"),

    )





def evaluate_criteria(candidate: FormulationCandidate, criteria: list[str]) -> list[CriterionEvaluation]:

    results = []

    has_explicit_temperature = any(

        re.search(r"-?\d+(?:\.\d+)?\s*(?:°\s*C|℃)", raw, re.I)

        for raw in criteria

    )

    for raw in criteria:

        # A generic low-temperature phrase adds no independently auditable

        # requirement when the same request already specifies (for example)

        # -20°C.  Keeping both would double-count a single evidence gap.

        if has_explicit_temperature and raw.strip().lower() in {

            "low temperature", "low-temperature", "low temperature operation",

        }:

            continue

        parsed_items = parse_criterion(raw)

        if not parsed_items:

            results.append(CriterionEvaluation(

                criterion=raw,

                kind="unsupported",

                status="unsupported",

                reason=f"当前受控筛选器无法解析条件“{raw}”。",

            ))

            continue

        for parsed in parsed_items:

            if parsed.kind == "minimum_voltage":

                results.append(_numeric_evaluation(parsed, candidate, _VOLTAGE))

            elif parsed.kind == "minimum_cycles":

                results.append(_numeric_evaluation(parsed, candidate, _CYCLES))

            elif parsed.kind == "minimum_retention":

                results.append(_numeric_evaluation(parsed, candidate, _RETENTION))

            elif parsed.kind == "minimum_conductivity":

                results.append(_numeric_evaluation(parsed, candidate, _CONDUCTIVITY))

            elif parsed.kind == "maximum_viscosity":

                results.append(_numeric_evaluation(parsed, candidate, _VISCOSITY))

            elif parsed.kind == "temperature":

                results.append(_numeric_evaluation(parsed, candidate, _TEMPERATURE))

            elif parsed.kind in ("required_component", "excluded_component"):

                results.append(_component_evaluation(parsed, candidate))

            else:

                results.append(_keyword_evaluation(parsed, candidate))

    return results





_NUMERIC_PATTERNS = {

    "minimum_voltage": _VOLTAGE,

    "c_rate": _C_RATE,

    "minimum_cycles": _CYCLES,

    "minimum_retention": _RETENTION,

    "minimum_conductivity": _CONDUCTIVITY,

    "maximum_viscosity": _VISCOSITY,

    "temperature": _TEMPERATURE,

}





def _voltage_window_evaluation(

    constraint: ScreeningConstraint, candidate: FormulationCandidate,

) -> CriterionEvaluation:

    """要求原文中出现完整且相同的电压范围，不用单端点替代。



    value 已是 [lower, upper] 两个有限数且 operator 为 ==（由

    ScreeningConstraint._validate_structure 在入口保证），此处只做证据匹配。

    """

    target = constraint.value

    observed = []

    for record in candidate.evidence:

        if not record.chunk_id:

            continue

        text = normalize_numeric_text(record.text)

        for match in _VOLTAGE_WINDOW.finditer(text):

            lower, upper = float(match["lower"]), float(match["upper"])

            if lower > upper:

                continue

            observed.append({

                "value": [lower, upper], "unit": "V", "chunk_id": record.chunk_id,

                "excerpt": _excerpt(text, match.start(), match.end()),

                "condition_scope": {"voltage_window_v": [lower, upper]},

            })

    # 当放电句明确沿用同段同电极对的全体充放电程序时，保留程序与该测试句

    # 两处引文，使窗口约束也能对应到该温度/倍率，而非只对应一个裸范围。

    for item in _observations(candidate, _TEMPERATURE, '°C'):

        window = item.get('condition_scope', {}).get('voltage_window_v')

        if window and any(link.get('relation') == 'same_cell_charge_discharge_protocol'

                          for link in item.get('context_evidence', [])):

            observed.append({**item, 'value': window, 'unit': 'V'})

    matched = [item for item in observed if all(

        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)

        for a, b in zip(item["value"], target)

    )]

    status = "satisfied" if matched else "failed" if observed else "insufficient_evidence"

    return CriterionEvaluation(

        constraint_id=constraint.constraint_id, criterion=constraint.description,

        kind=constraint.kind, status=status,

        expected={"operator": "==", "value": target, "unit": "V"},

        observed=matched or observed,

        reason=("原文包含要求的完整电压范围。" if matched else

                "原文未提供相同的完整电压范围。"),

    )





def _with_constraint_id(

    evaluation: CriterionEvaluation,

    constraint_id: str,

) -> CriterionEvaluation:

    return evaluation.model_copy(update={"constraint_id": constraint_id})





def _formulation_observations(candidate: FormulationCandidate) -> list[dict]:

    return [

        {

            "chunk_id": record.chunk_id,

            "excerpt": record.text[:500],

        }

        for record in candidate.evidence

        if record.evidence_kind == "formulation" and record.chunk_id and record.text

    ]





def _field_presence_evaluation(

    constraint: ScreeningConstraint,

    candidate: FormulationCandidate,

) -> CriterionEvaluation:

    field = constraint.field or ""

    value = getattr(candidate, field, None) if field else None

    present = value is not None and value != "" and value != [] and value != {}

    observations = _formulation_observations(candidate)

    status = "satisfied" if present and observations else (

        "insufficient_evidence" if present else "failed"

    )

    reason = (

        f"候选字段 {field} 已填写且有同候选配方证据。"

        if status == "satisfied"

        else (

            f"候选字段 {field} 已填写，但缺少可追溯配方证据。"

            if present

            else f"候选缺少必填字段 {field}。"

        )

    )

    return CriterionEvaluation(

        constraint_id=constraint.constraint_id,

        criterion=constraint.description,

        kind=constraint.kind,

        status=status,

        expected={"operator": "present", "field": field},

        observed=observations,

        reason=reason,

    )





def _typed_component_evaluation(

    constraint: ScreeningConstraint,

    candidate: FormulationCandidate,

) -> CriterionEvaluation:

    component = str(constraint.value or "").strip()

    if constraint.kind == "required_component" and component.casefold() in {"carbonate", "carbonate-based", "碳酸酯"}:

        component = "__carbonate__"

    parsed = ParsedCriterion(

        raw=constraint.description,

        kind=constraint.kind,

        operator=constraint.operator or (

            "absent" if constraint.kind == "excluded_component" else "contains"

        ),

        component=component,

    )

    result = _component_evaluation(parsed, candidate)

    observations = _formulation_observations(candidate)

    if result.status == "satisfied" and not observations:

        result = result.model_copy(

            update={

                "status": "insufficient_evidence",

                "reason": "候选组成满足结构化条件，但缺少可追溯配方证据。",

            }

        )

    return _with_constraint_id(

        result.model_copy(update={"observed": observations or result.observed}),

        constraint.constraint_id,

    )





def _typed_evidence_term_evaluation(

    constraint: ScreeningConstraint,

    candidate: FormulationCandidate,

) -> CriterionEvaluation:

    terms = [

        term.strip().casefold()

        for term in constraint.evidence_terms

        if term.strip()

    ]

    observations = [

        {"chunk_id": record.chunk_id, "excerpt": record.text[:500]}

        for record in candidate.evidence

        if record.chunk_id

        and record.text

        and any(term in record.text.casefold() for term in terms)

    ]

    if constraint.kind in {"electrode_material", "cell_configuration"}:

        field = 'electrode' if constraint.kind == 'electrode_material' else 'cell_configuration'

        for observation in _observations(candidate, _TEMPERATURE, '°C'):

            value = str(observation.get('condition_scope', {}).get(field) or '').casefold()

            if observation.get('context_evidence') and any(term in value for term in terms):

                observations.append(observation)

    if observations:

        status = "satisfied"

        reason = "候选原文证据直接命中调用方提供的术语。"

    else:

        status = "insufficient_evidence"

        reason = "候选原文证据未直接命中调用方提供的术语。"

    return CriterionEvaluation(

        constraint_id=constraint.constraint_id,

        criterion=constraint.description,

        kind=constraint.kind,

        status=status,

        expected={"operator": "contains", "evidence_terms": constraint.evidence_terms},

        observed=observations,

        reason=reason,

    )





def _normalize_quote(text: str) -> str:

    """比较引文排版，保留数值、正负号、指数和文字内容。



    复用数值检查已有的 LaTeX 排版处理，不做同义词、否定或化学语义替换。

    原始引文与来源正文仍分别保存，不能把排版等价标成逐字一致。

    """



    rendered = normalize_numeric_text(text or "")

    rendered = re.sub(r"\s*°\s*", "°", rendered)

    return re.sub(r"\s+", " ", rendered).strip()





def _record_searchable_texts(record) -> list[str]:

    """候选证据中模型真正读到的正文，含命中 chunk 与其父节文本。"""



    texts = [record.text, record.excerpt, record.parent_text]

    return [text for text in texts if text]





def _evidence_claim_evaluation(

    constraint: ScreeningConstraint,

    candidate: FormulationCandidate,

) -> CriterionEvaluation:

    """核验模型对断言约束的语义判定，并验证其引文结构。



    语义判断（这段原文是否为该候选建立或推翻该断言）由模型完成；程序只做

    两件可机械核验的事：判定必须落到本次请求的候选，引文必须逐字存在于该

    候选的证据正文。程序不解析句式、不做词表匹配，因此对措辞、否定、同义

    称呼的变化不敏感——这正是 ``mechanism_keyword`` 做不到的。

    """



    claim_text = str(constraint.claim or "").strip()

    verdict = next(

        (

            item

            for item in constraint.claim_verdicts

            if item.candidate_id == candidate.candidate_id

        ),

        None,

    )

    if verdict is None:

        return CriterionEvaluation(

            constraint_id=constraint.constraint_id,

            criterion=constraint.description,

            kind=constraint.kind,

            status="insufficient_evidence",

            expected={"claim": claim_text},

            observed=[],

            reason=(

                f"未对候选 {candidate.candidate_id} 给出断言「{claim_text}」的判定；"

                "须逐候选给出 supported/refuted/not_found。"

            ),

        )



    if verdict.verdict == "not_found":

        return CriterionEvaluation(

            constraint_id=constraint.constraint_id,

            criterion=constraint.description,

            kind=constraint.kind,

            status="insufficient_evidence",

            expected={"claim": claim_text},

            observed=[],

            reason=(

                f"检索到的证据未直接建立断言「{claim_text}」"

                f"（依据：{verdict.basis or '未说明'}）；需补充检索或证据。"

            ),

        )



    # supported / refuted 须落到候选自带的证据正文，且引文逐字存在。

    quote = _normalize_quote(verdict.quote)

    matched_record = next(

        (

            record

            for record in candidate.evidence

            if record.chunk_id == verdict.chunk_id

            and quote

            and any(

                quote in _normalize_quote(text)

                for text in _record_searchable_texts(record)

            )

        ),

        None,

    )

    if matched_record is None:

        # 参数转写错误与缺少科学证据是不同的问题。把已经随候选提交的真实

        # 来源交还给规划者，供其重新选择连续引文；不替模型改引文或判为通过。

        citation_repair = [

            {"chunk_id": record.chunk_id, "source_text": text}

            for record in candidate.evidence

            if record.chunk_id == verdict.chunk_id

            for text in _record_searchable_texts(record)[:1]

        ]

        return CriterionEvaluation(

            constraint_id=constraint.constraint_id,

            criterion=constraint.description,

            kind=constraint.kind,

            status="insufficient_evidence",

            expected={"claim": claim_text, "citation_repair": citation_repair},

            observed=[],

            reason=(

                f"判定为 {verdict.verdict} 的引文无法在候选证据中逐字核验"

                f"（chunk_id={verdict.chunk_id!r}）；引文结构不成立，不予采信。"

                "请从 expected.citation_repair 中的来源原文重新截取连续引文，"

                "保留原约束和候选归属后再评估；不能改写原文或删除要求。"

            ),

        )



    observation = {

        "chunk_id": matched_record.chunk_id,

        "excerpt": matched_record.text[:500],

        "quote": verdict.quote,

        "basis": verdict.basis,

        "quote_match_mode": (

            "exact"

            if any(re.sub(r"\s+", " ", verdict.quote).strip()

                   in re.sub(r"\s+", " ", text).strip()

                   for text in _record_searchable_texts(matched_record))

            else "presentation_equivalent"

        ),

    }

    if verdict.verdict == "supported":

        status = "satisfied"

        reason = f"原文引文建立断言「{claim_text}」；依据：{verdict.basis or '见引文'}。"

    else:

        status = "failed"

        reason = f"原文引文反驳断言「{claim_text}」；依据：{verdict.basis or '见引文'}。"

    return CriterionEvaluation(

        constraint_id=constraint.constraint_id,

        criterion=constraint.description,

        kind=constraint.kind,

        status=status,

        expected={"claim": claim_text},

        observed=[observation],

        reason=reason,

    )





def evaluate_constraint_records(

    candidate: FormulationCandidate,

    constraints: list[ScreeningConstraint],

) -> list[CriterionEvaluation]:

    """执行显式约束记录；不解析或补写任何自然语言规则。"""



    results: list[CriterionEvaluation] = []

    for constraint in constraints:

        if constraint.kind == "voltage_window":

            results.append(_voltage_window_evaluation(constraint, candidate))

        elif constraint.kind in _NUMERIC_PATTERNS:

            # 结构合法性（数值 value、unit×kind 匹配、c_rate 只用 ==）已由

            # ScreeningConstraint._validate_structure 在入口拒绝，此处不再补正。

            operator = constraint.operator

            if constraint.kind == "c_rate":

                operator = "=="

            elif operator not in {">=", "<=", "=="}:

                operator = "<=" if constraint.kind in {

                    "maximum_viscosity",

                    "temperature",

                } else ">="

            parsed = ParsedCriterion(

                raw=constraint.description,

                kind=constraint.kind,

                operator=operator,

                value=float(constraint.value),

                unit=constraint.unit,

            )

            results.append(

                _with_constraint_id(

                    _numeric_evaluation(

                        parsed,

                        candidate,

                        _NUMERIC_PATTERNS[constraint.kind],

                    ),

                    constraint.constraint_id,

                )

            )

        elif constraint.kind in {"required_component", "excluded_component"}:

            results.append(_typed_component_evaluation(constraint, candidate))

        elif constraint.kind == "field_presence":

            results.append(_field_presence_evaluation(constraint, candidate))

        elif constraint.kind == "evidence_claim":

            results.append(_evidence_claim_evaluation(constraint, candidate))

        else:

            results.append(_typed_evidence_term_evaluation(constraint, candidate))

    return results

