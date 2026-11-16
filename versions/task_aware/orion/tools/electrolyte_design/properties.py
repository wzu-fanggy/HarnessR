from __future__ import annotations



import re



from langchain_core.tools import tool



from .evidence import normalize_evidence_chunks

from .models import EvidenceChunk, EvidenceRef, PropertyObservation





_CONDUCTIVITY = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mS\s*cm[-−]?1|S\s*cm[-−]?1)", re.I)

_TEMPERATURE = re.compile(

    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?:°\s*C|℃)"

    r"|(?:temperature|temp\.?)\s*(?:of|at|=|:)?\s*"

    r"(?P<context_value>-?\d+(?:\.\d+)?)\s*C\b",

    re.I,

)

_VISCOSITY = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mPa\s*s|cP)\b", re.I)

_CYCLES = re.compile(r"(?P<value>\d+)\s*cycles?", re.I)

_CAPACITY_RETENTION = re.compile(

    r"(?:capacity\s+retention|retained\s+capacity|retention\s+rate)\D{0,40}?"

    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%)",

    re.I,

)

_CAPACITY_RETENTION_SERIES = re.compile(

    r"(?P<retentions>\d+(?:\.\d+)?\s*%"

    r"(?:\s*,\s*\d+(?:\.\d+)?\s*%)*"

    r"(?:\s*(?:and|or)\s*\d+(?:\.\d+)?\s*%)?)"

    r"\s+capacity\s+retention\b"

    r"(?P<middle>.{0,160}?)"

    r"\bafter\s+(?P<cycles>\d+)\s+cycles?\b"

    r"(?P<tail>.{0,160}?)"

    r"\bwith\s+(?P<concentrations>\d+(?:\.\d+)?\s*%"

    r"(?:\s*,\s*\d+(?:\.\d+)?\s*%)*"

    r"(?:\s*(?:and|or)\s*\d+(?:\.\d+)?\s*%)?)"

    r"\s+(?P<component>[A-Za-z][A-Za-z0-9+-]*)"

    r"\s*,?\s*respectively",

    re.I,

)

_SPECIFIC_CAPACITY = re.compile(

    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mA\s*h\s*g\s*(?:[-−]?1|\^{-1\}|\^\{-1\}))",

    re.I,

)





def _local_temperature(text: str, start: int, end: int) -> float | None:

    sentence_start = max(text.rfind(".", 0, start), text.rfind(";", 0, start)) + 1

    sentence_end_candidates = [position for position in (text.find(".", end), text.find(";", end)) if position != -1]

    sentence_end = min(sentence_end_candidates) if sentence_end_candidates else len(text)

    matches = list(_TEMPERATURE.finditer(text[sentence_start:sentence_end]))

    if len(matches) != 1:

        return None

    value = matches[0].group("value") or matches[0].group("context_value")

    return float(value)





def _ref(chunk: EvidenceChunk, kind: str) -> EvidenceRef:

    refs = normalize_evidence_chunks([chunk])

    return next((item for item in refs if item.evidence_kind == kind), refs[0])





def _percent_values(text: str) -> list[float]:

    return [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)]





def _capacity_retention_series(

    chunk: EvidenceChunk,

    requested: set[str],

) -> tuple[list[PropertyObservation], list[tuple[int, int]]]:

    """抽取由 respectively 明确对齐的浓度—容量保持率序列。"""

    if requested and "capacity_retention" not in requested:

        return [], []

    observations: list[PropertyObservation] = []

    spans: list[tuple[int, int]] = []

    for match in _CAPACITY_RETENTION_SERIES.finditer(chunk.content):

        retentions = _percent_values(match.group("retentions"))

        concentrations = _percent_values(match.group("concentrations"))

        if not retentions or len(retentions) != len(concentrations):

            continue

        cycles = int(match.group("cycles"))

        component = match.group("component")

        spans.append(match.span())

        for retention, concentration in zip(retentions, concentrations):

            observations.append(

                PropertyObservation(

                    property_name="capacity_retention",

                    numeric_value=retention,

                    unit="%",

                    raw_value=f"{retention:g}% capacity retention",

                    test_condition=(

                        f"{component}={concentration:g}%; cycles={cycles}"

                    ),

                    evidence=_ref(chunk, "property"),

                    extraction_confidence="high",

                )

            )

    return observations, spans





@tool

def extract_numeric_properties(

    evidence_chunks: list[EvidenceChunk],

    property_names: list[str] | None = None,

) -> list[PropertyObservation]:

    """从证据 chunk 中用可审计正则抽取数值、单位、温度和来源，不猜测模糊数字。



    适用场景：检索到的证据涉及性能数值（电导率、容量、循环次数、倍率、库仑效率、

    粘度等）时，**必须**调用本工具抽取数值证据后再作答，不要仅凭文本印象总结。



    参数：evidence_chunks 为 KG/RAG 证据块，property_names 可限制抽取的属性名。

    返回：包含原始匹配值、测试温度与 EvidenceRef 来源的数值属性观察列表。

    """

    requested = {name.lower() for name in property_names or []}

    output = []

    for chunk in evidence_chunks:

        text = chunk.content

        series_observations, series_spans = _capacity_retention_series(chunk, requested)

        output.extend(series_observations)

        for match in _CONDUCTIVITY.finditer(text):

            if requested and "ionic_conductivity" not in requested:

                continue

            temperature = _local_temperature(text, match.start(), match.end())

            output.append(PropertyObservation(

                property_name="ionic_conductivity",

                numeric_value=float(match.group("value")),

                unit=re.sub(r"\s+", " ", match.group("unit")).replace("−", "-"),

                raw_value=match.group(0),

                temperature_c=temperature,

                test_condition=f"temperature={temperature} °C" if temperature is not None else "",

                evidence=_ref(chunk, "property"),

                extraction_confidence="high" if temperature is not None else "medium",

            ))

        for match in _VISCOSITY.finditer(text):

            if requested and "viscosity" not in requested:

                continue

            temperature = _local_temperature(text, match.start(), match.end())

            output.append(PropertyObservation(

                property_name="viscosity",

                numeric_value=float(match.group("value")),

                unit=match.group("unit"),

                raw_value=match.group(0),

                temperature_c=temperature,

                test_condition=f"temperature={temperature} °C" if temperature is not None else "",

                evidence=_ref(chunk, "property"),

                extraction_confidence="high" if temperature is not None else "medium",

            ))

        for match in _CYCLES.finditer(text):

            if requested and "cycle_count" not in requested:

                continue

            temperature = _local_temperature(text, match.start(), match.end())

            output.append(PropertyObservation(

                property_name="cycle_count",

                numeric_value=float(match.group("value")),

                unit="cycles",

                raw_value=match.group(0),

                temperature_c=temperature,

                test_condition=f"temperature={temperature} °C" if temperature is not None else "",

                evidence=_ref(chunk, "property"),

                extraction_confidence="medium" if temperature is not None else "low",

            ))

        for match in _CAPACITY_RETENTION.finditer(text):

            if requested and "capacity_retention" not in requested:

                continue

            if any(start <= match.start() < end for start, end in series_spans):

                continue

            temperature = _local_temperature(text, match.start(), match.end())

            output.append(PropertyObservation(

                property_name="capacity_retention",

                numeric_value=float(match.group("value")),

                unit=match.group("unit"),

                raw_value=match.group(0),

                temperature_c=temperature,

                test_condition=f"temperature={temperature} °C" if temperature is not None else "",

                evidence=_ref(chunk, "property"),

                extraction_confidence="high",

            ))

        for match in _SPECIFIC_CAPACITY.finditer(text):

            if requested and "specific_capacity" not in requested:

                continue

            temperature = _local_temperature(text, match.start(), match.end())

            output.append(PropertyObservation(

                property_name="specific_capacity",

                numeric_value=float(match.group("value")),

                unit=re.sub(r"\s+", " ", match.group("unit")).replace("−", "-"),

                raw_value=match.group(0),

                temperature_c=temperature,

                test_condition=f"temperature={temperature} °C" if temperature is not None else "",

                evidence=_ref(chunk, "property"),

                extraction_confidence="high" if temperature is not None else "medium",

            ))

    return output

