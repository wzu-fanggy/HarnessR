"""从同一论文的证据块组装可审计的配方候选。

这里只接受正文中显式出现的配方比例；不会根据常识推导缺失比例。
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from orion.task_evidence.condition_extraction import extract_condition_scope
from tools.electrolyte_design.models import EvidenceRef, FormulationCandidate
from tools.electrolyte_design.object_identity import (
    resolve_shared_explicit_object,
    text_explicitly_names_object,
)


SOLVENT_ALIASES = {
    "ethylene carbonate": "EC",
    "propylene carbonate": "PC",
    "ethyl methyl carbonate": "EMC",
    "dimethyl carbonate": "DMC",
    "diethyl carbonate": "DEC",
    "tetrahydrofuran": "THF",
    "diethylene glycol dimethyl ether": "G2",
    "diglyme": "G2",
    "2-methyloxolane": "MO",
    "1,2-diethoxyethane": "DEE",
    "diethoxyethane": "DEE",
    "cyclopentyl methyl ether": "CPME",
    "g2": "G2",
    "mo": "MO",
    "dee": "DEE",
    "cpme": "CPME",
    "ec": "EC",
    "pc": "PC",
    "emc": "EMC",
    "dmc": "DMC",
    "dec": "DEC",
    "thf": "THF",
}
ADDITIVE_ALIASES = {
    "fluoroethylene carbonate": "FEC",
    "fluoroethylene-carbonate": "FEC",
    "fec": "FEC",
    "nadfp": "NaDFP",
    "na dfp": "NaDFP",
    "pfpn": "PFPN",
    "lidfob": "LiDFOB",
}


def _canonical(text: str, aliases: dict[str, str]) -> str | None:
    lowered = text.lower().strip()
    for alias, value in sorted(aliases.items(), key=lambda item: -len(item[0])):
        if alias in lowered:
            return value
    return None


def extract_explicit_formulation(text: str, *, preferred_salts: list[str] | None = None,
                                preferred_solvents: list[str] | None = None) -> dict[str, Any] | None:
    """Extract only literal salt/concentration/solvent-ratio/additive claims."""
    if not text:
        return None
    normalized_text = re.sub(
        r"NaPF\s*(?:\$\s*)?(?:_\s*)?(?:\{\s*6\s*\}|6)(?:\s*\$)?",
        "NaPF6",
        text,
        flags=re.I,
    )
    normalized_text = re.sub(
        r"NaBF\s*(?:\$\s*)?(?:_\s*)?(?:\{\s*4\s*\}|4)(?:\s*\$)?",
        "NaBF4",
        normalized_text,
        flags=re.I,
    )
    salts = preferred_salts or []
    salt = next((item for item in salts if item.lower() in normalized_text.lower()), None)
    if salt is None:
        if salts:
            return None
        salt_match = re.search(r"\bNa[A-Z][A-Za-z0-9]*\b", normalized_text)
        salt = salt_match.group(0) if salt_match else None
    if not salt:
        return None

    # 比较段可同时写出目标单溶剂配方与另一种盐的混合溶剂参照。
    # 只从同一明确的“浓度—盐—in—溶剂”短语提取目标配方，不借用参照的溶剂比例。
    written_solvents = {value for alias, value in SOLVENT_ALIASES.items()
                        if re.search(rf'\b{re.escape(alias)}\b', normalized_text, re.I)}
    written_solvents.update(SOLVENT_ALIASES.get(name.casefold(), name)
                            for name in (preferred_solvents or [])
                            if re.search(rf'\b{re.escape(name)}\b', normalized_text, re.I))
    if len(written_solvents) > 1:
        solvent_pattern = '|'.join(re.escape(name) for name in sorted(written_solvents, key=len, reverse=True))
        recipe_pattern = re.compile(
            r'(?<![\d.])\d+(?:\.\d+)?\s*M\s+' + re.escape(salt)
            + r'\s+in\s+(?:pure\s+)?(?:' + solvent_pattern + r')\b'
            + r'(?!\s*(?:[:/]|and\b|\+\s*[A-Za-z]))'
            + r'(?:\s*(?:with\s+|\+\s*)\d+(?:\.\d+)?\s*(?:wt|vol|mol)\s*%\s+[A-Za-z][A-Za-z0-9-]*)?', re.I,
        )
        recipes = {match.group(0).strip() for match in recipe_pattern.finditer(normalized_text)}
        if len(recipes) == 1:
            focused = next(iter(recipes))
            remaining = normalized_text[normalized_text.find(focused)+len(focused):]
            if re.match(r'\s*(?:and|\+|with)\s*\d+(?:\.\d+)?\s*(?:wt|vol|mol)\s*%', remaining, re.I):
                return None
            # 聚焦后必须缩小输入；即使调用方名称重复，也不能递归处理同一短语。
            if focused != normalized_text.strip() and not re.search(r'\bwith\b', remaining[:12], re.I):
                return extract_explicit_formulation(focused, preferred_salts=preferred_salts,
                                                    preferred_solvents=preferred_solvents)
        elif len(recipes) > 1:
            return None

    # 一段综述可同时介绍多个浓度和溶剂比例。不能把段首浓度与另一配方的
    # 比例拼成一份候选。不同的显式比例必须由调用方先拆成各自的原文证据。
    ratio_signatures = set()
    for match in re.finditer(
        r"(?P<names>(?:EC|PC|EMC|DMC|DEC)(?:\s*[:/]\s*(?:EC|PC|EMC|DMC|DEC))+?)"
        r"\s*(?:=|\(|\s)\s*(?P<values>\d+(?:\.\d+)?(?:\s*[:/]\s*\d+(?:\.\d+)?)+)",
        normalized_text, re.I,
    ):
        names = tuple(re.split(r"\s*[:/]\s*", match['names'].upper()))
        values = tuple(float(v) for v in re.split(r"\s*[:/]\s*", match['values']))
        if len(names) == len(values) and sum(values) > 0:
            ratio_signatures.add((names, tuple(v / sum(values) for v in values)))
    if len(ratio_signatures) > 1:
        return None

    salt_pattern = re.escape(salt)
    concentration_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:M|mol\s*L\^-?1|mol\s*dm\^-?3)\s*(?:of\s+)?" + salt_pattern + r"\b", normalized_text, re.I)
    concentration = float(concentration_match.group(1)) if concentration_match else None
    if concentration is None:
        # NaPF6 (1 mol/L) 与 1 M NaPF6 均为明确组成，不能因语序而丢失。
        after_salt = re.search(
            r"\b" + salt_pattern + r"\s*\(\s*(\d+(?:\.\d+)?)\s*mol\s*/\s*L\s*\)",
            normalized_text, re.I,
        )
        if after_salt:
            concentration = float(after_salt.group(1))
    if concentration_match and re.search(r"(?:\bto|[-–])\s*$", normalized_text[max(0, concentration_match.start()-12):concentration_match.start()]):
        # 浓度系列的上端点不是当前候选的确定浓度。
        concentration = None
    salt_amounts: dict[str, set[float]] = defaultdict(set)
    for match in re.finditer(
        r"(?<![\d.])(?P<amount>\d+(?:\.\d+)?)\s*M\s*(?P<salt>Na(?:PF6|FSI|TFSI|ClO4|DFP|BF4))\b",
        normalized_text, re.I,
    ):
        if not re.search(r"(?:\bto|[-–])\s*$", normalized_text[max(0,match.start()-12):match.start()]):
            salt_amounts[match['salt']].add(float(match['amount']))
    component_concentrations = {name: next(iter(values)) for name, values in salt_amounts.items()
                               if len(values) == 1}
    formulation_salts = [salt]
    if len(salt_amounts) > 1 and all(len(values) == 1 for values in salt_amounts.values()):
        # 浓度唯一并不代表混合使用。必须同时出现各盐相连、溶于同一溶剂的
        # 配方写法，避免把两种单盐对照拼成双盐。
        joint_salts = re.search(
            r"\d+(?:\.\d+)?\s*M\s*Na(?:PF6|FSI|TFSI|ClO4|DFP|BF4)\s*"
            r"(?:and|\+)\s*\d+(?:\.\d+)?\s*M\s*Na(?:PF6|FSI|TFSI|ClO4|DFP|BF4)\s+in\b",
            normalized_text, re.I,
        )
        if not joint_salts:
            return None
        formulation_salts = list(component_concentrations)
        salt = " + ".join(formulation_salts)
        concentration = sum(component_concentrations.values())
    elif len(salt_amounts.get(salt, set())) > 1:
        concentration = None
        component_concentrations = {}
    if re.search(r"\b(?:from|between)\s+\d+(?:\.\d+)?\s*M?\s*(?:to|and|[-–])\s*\d+(?:\.\d+)?\s*M\b", normalized_text, re.I):
        concentration = None
        component_concentrations = {}

    ratio_match = re.search(
        r"(?P<names>(?:EC|PC|EMC|DMC|DEC)(?:\s*[:/]\s*(?:EC|PC|EMC|DMC|DEC))+?)"
        r"\s*(?:=|\(|\s)\s*(?P<values>\d+(?:\.\d+)?(?:\s*[:/]\s*\d+(?:\.\d+)?)+)",
        normalized_text,
        re.I,
    )
    ratios: dict[str, float] = {}
    ratio_basis = "unknown"
    solvents: list[str] = []
    if ratio_match:
        names = [SOLVENT_ALIASES[name.lower()] for name in re.split(r"\s*[:/]\s*", ratio_match.group("names"))]
        values = [float(item) for item in re.split(r"\s*[:/]\s*", ratio_match.group("values"))]
        if len(names) == len(values) and sum(values) > 0:
            total = sum(values)
            ratios = {name: value / total for name, value in zip(names, values, strict=True)}
            solvents = list(dict.fromkeys(names))
            ratio_context = normalized_text[max(0, ratio_match.start()-30):ratio_match.end()+40]
            if re.search(r"(?:weight|mass)\s+ratio|w/w", ratio_context, re.I):
                ratio_basis = "mass"
            elif re.search(r"volume\s+ratio|v/v|vol\s*%", ratio_context, re.I):
                ratio_basis = "volume"
            elif re.search(r"molar\s+ratio|mol\s*%", ratio_context, re.I):
                ratio_basis = "molar"
    if not ratios:
        prose_ratio = re.search(
            r"\b(?P<first>EC|PC|EMC|DMC|DEC)\s+and\s+(?P<second>EC|PC|EMC|DMC|DEC)\s*"
            r"\(\s*(?P<basis>weight|mass|volume|molar)\s+ratio\s*(?P<a>\d+(?:\.\d+)?):(?P<b>\d+(?:\.\d+)?)\s*\)",
            normalized_text, re.I,
        )
        if prose_ratio:
            first, second = prose_ratio['first'].upper(), prose_ratio['second'].upper()
            a, b = float(prose_ratio['a']), float(prose_ratio['b'])
            if a + b > 0:
                ratios = {first: a / (a+b), second: b / (a+b)}
                solvents = [first, second]
                ratio_basis = {"weight": "mass", "mass": "mass", "volume": "volume", "molar": "molar"}[prose_ratio['basis'].lower()]
    # 允许正文直接给出非碳酸酯缩写比例（例如 ``G2:MO = 7:3 vol%``）。
    # 只接受已在本文本中逐字出现的简单 token，且 token/value 数量严格对应；
    # 已知钠盐不能被误当作溶剂。规范化仅使用本模块的字面别名，不调用 KG。
    if not solvents:
        generic_ratio_match = re.search(
            r"(?P<names>[A-Za-z][A-Za-z0-9-]*(?:\s*[:/]\s*[A-Za-z][A-Za-z0-9-]*)+)"
            r"\s*(?:=|\()\s*(?P<values>\d+(?:\.\d+)?(?:\s*[:/]\s*\d+(?:\.\d+)?)+)"
            r"\s*(?:(?:vol|wt|mol)\s*%|in\s+volume\s+ratio)?\s*\)?",
            normalized_text,
            re.I,
        )
        if generic_ratio_match:
            raw_names = [
                item.strip()
                for item in re.split(r"\s*[:/]\s*", generic_ratio_match.group("names"))
                if item.strip()
            ]
            salt_tokens = {"napf6", "nafsi", "natfsi", "naclo4", "nadfp", "nabf4"}
            if raw_names and not any(item.casefold() in salt_tokens for item in raw_names):
                names = [
                    SOLVENT_ALIASES.get(item.casefold(), item)
                    for item in raw_names
                ]
                values = [
                    float(item)
                    for item in re.split(r"\s*[:/]\s*", generic_ratio_match.group("values"))
                ]
                if len(names) == len(values) and sum(values) > 0:
                    total = sum(values)
                    ratios = {
                        name: value / total
                        for name, value in zip(names, values, strict=True)
                    }
                    solvents = list(dict.fromkeys(names))
    # 论文方法部分常把盐也写进摩尔比，例如
    # ``NaPF6:PC:DEC = 2:15:3.4``。盐的第一个数不是溶剂比例，
    # 但后续 PC:DEC 是明确的配方信息，不能因表达形式不同而丢掉。
    if not solvents:
        salt_ratio_match = re.search(
            r"Na(?:PF6|FSI|TFSI|ClO4|DFP|BF4)\s*:\s*"
            r"(?P<names>(?:EC|PC|EMC|DMC|DEC)(?:\s*:\s*(?:EC|PC|EMC|DMC|DEC))+?)"
            r"\s*=\s*(?P<values>\d+(?:\.\d+)?(?:\s*:\s*\d+(?:\.\d+)?)+)",
            normalized_text,
            re.I,
        )
        if salt_ratio_match:
            names = [
                SOLVENT_ALIASES[name.lower()]
                for name in re.split(r"\s*:\s*", salt_ratio_match.group("names"))
            ]
            raw_values = [
                float(item) for item in re.split(r"\s*:\s*", salt_ratio_match.group("values"))
            ]
            values = raw_values[1:]
            if len(names) == len(values) and sum(values) > 0:
                total = sum(values)
                ratios = {name: value / total for name, value in zip(names, values, strict=True)}
                solvents = list(dict.fromkeys(names))
    if not solvents:
        solvents = list(dict.fromkeys(
            value for alias, value in SOLVENT_ALIASES.items() if re.search(rf"\b{re.escape(alias)}\b", text, re.I)
        ))
        # 调用方已指定的名称只按原文字面匹配，不新增化学实体别名映射。
        solvents.extend(SOLVENT_ALIASES.get(name.casefold(), name)
                        for name in (preferred_solvents or [])
                        if SOLVENT_ALIASES.get(name.casefold(), name) not in solvents
                        and re.search(rf"\b{re.escape(name)}\b", text, re.I))
        solvents = list(dict.fromkeys(solvents))
    if not solvents:
        return None
    if len(solvents) == 1 and not ratios:
        # 单一溶剂本身就是完整组成，不应因没有 ``A:B`` 比例而被证据门禁拦截。
        # 但 SEI 中的盐分解产物与附近的溶剂名称，不等于电解液组成。
        solvent = '(?:' + '|'.join(re.escape(name) for name in
                                  [solvents[0], *[alias for alias, canonical in SOLVENT_ALIASES.items()
                                                if canonical == solvents[0]]]) + ')'
        direct_solution = re.search(
            re.escape(salt) + r"(?:\s*\([^)]*\))?\s+(?:(?:was|were|is|are)\s+dissolved\s+)?(?:in|into)\s+(?:pure\s+)?" + solvent + r"\b",
            normalized_text, re.I,
        )
        pure_solution = re.search(
            r"\bpure\s+" + solvent + r"\b[^.]{0,100}\b" + re.escape(salt) + r"\b",
            normalized_text, re.I,
        )
        if not (direct_solution or pure_solution):
            return None
        ratios = {solvents[0]: 1.0}

    additives: list[str] = []
    additive_amount_candidates: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for alias, value in ADDITIVE_ALIASES.items():
        alias_matches = list(re.finditer(rf"\b{re.escape(alias)}\b", text, re.I))
        if not alias_matches:
            continue
        additives.append(value)
        for alias_match in alias_matches:
            boundaries = [
                match.start()
                for match in re.finditer(r"(?<!\d)[.;]|[.;](?!\d)", text)
            ]
            before = [index for index in boundaries if index < alias_match.start()]
            after = [index for index in boundaries if index >= alias_match.end()]
            sentence_start = (max(before) + 1) if before else 0
            sentence_end = min(after) if after else len(text)
            sentence = text[sentence_start:sentence_end]
            sentence_amounts = {
                (match.group(1), match.group(2).replace(" ", "").lower())
                for match in re.finditer(
                    r"(\d+(?:\.\d+)?)\s*((?:vol|wt|mol)\s*%)",
                    sentence,
                    re.I,
                )
            }
            # Bare percentages next to a reagent commonly describe purity
            # (for example ``PFPN, >98%``), not its amount in the electrolyte.
            # Accept only explicit composition units. A sentence listing several
            # concentrations describes a formulation series, so leave the exact
            # variant unresolved until condition evidence identifies it.
            if len(sentence_amounts) > 1:
                continue
            amount = re.search(
                rf"(\d+(?:\.\d+)?)\s*((?:vol|wt|mol)\s*%)\s*{re.escape(alias)}\b",
                sentence,
                re.I,
            )
            if amount is None:
                amount = re.search(
                    rf"{re.escape(alias)}\s*(?:\([^)]{{0,12}}\)\s*)?"
                    rf"(?:was\s+)?(?:added|introduced)?\s*(?:at\s*)?"
                    rf"(\d+(?:\.\d+)?)\s*((?:vol|wt|mol)\s*%)",
                    sentence,
                    re.I,
                )
            if amount:
                additive_amount_candidates[value].add(
                    (amount.group(1), amount.group(2).replace(" ", ""))
                )
    additive_amounts = {
        value: f"{next(iter(amounts))[0]} {next(iter(amounts))[1]}"
        for value, amounts in additive_amount_candidates.items()
        if len(amounts) == 1
    }
    # 接受明确的“用量 + 化合物全名 + was added”，不把附近的百分比当作添加剂用量。
    for match in re.finditer(
        r"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>wt|vol|mol)\s*%\s*"
        r"(?P<name>[A-Za-z0-9][A-Za-z0-9 -]{2,90}?)(?:\s*\([A-Za-z0-9-]+\))?\s+was\s+added\b",
        normalized_text, re.I,
    ):
        name = match['name'].strip()
        if name not in additives:
            additives.append(name)
        additive_amounts[name] = f"{match['amount']} {match['unit'].lower()}%"

    # 明确紧接单溶剂配方的“with 用量 添加剂缩写”保留原文字面名称。
    # 不建立新的名称别名表，也不把附近的纯度百分比当作用量。
    for match in re.finditer(r'(?:\bwith\s+|\+\s*)(?P<amount>\d+(?:\.\d+)?)\s*'
                             r'(?P<unit>wt|vol|mol)\s*%\s*(?P<name>[A-Za-z][A-Za-z0-9-]*)\b',
                             normalized_text, re.I):
        name = match['name']
        if name not in additives:
            additives.append(name)
        additive_amounts[name] = f"{match['amount']} {match['unit'].lower()}%"

    return {
        "salt": salt,
        "salts": formulation_salts,
        "salt_concentrations_mol_l": component_concentrations,
        "concentration_mol_l": concentration,
        "solvents": solvents,
        "solvent_ratios": ratios,
        "solvent_ratio_basis": ratio_basis,
        "additives": list(dict.fromkeys(additives)),
        "additive_amounts": additive_amounts,
    }


_RETRIEVED_EVIDENCE_KINDS = {"formulation", "property", "condition", "mechanism", "risk"}
_PROPERTY_EVIDENCE_RE = re.compile(
    r"(?:capacity\s+retention|retain(?:s|ed)?\s+\d+(?:\.\d+)?\s*%|"
    r"\d+(?:\.\d+)?\s*%\s+capacity|after\s+\d+\s+cycles?|"
    r"ionic\s+conductivity|coulombic\s+efficiency|oxidation\s+potential|"
    r"specific\s+capacity)",
    re.I,
)


def _retrieved_evidence_kind(chunk: dict[str, Any], text: str) -> str | None:
    """为同论文、调用方已检索到的 source chunk 赋保守语义角色。"""

    explicit = str(chunk.get("evidence_kind") or "").strip()
    if explicit in _RETRIEVED_EVIDENCE_KINDS:
        return explicit
    if _PROPERTY_EVIDENCE_RE.search(text):
        return "property"
    if extract_condition_scope(text) is not None:
        return "condition"
    return None


def assemble_same_paper_candidates(
    chunks: list[dict[str, Any]],
    *,
    preferred_salts: list[str],
    preferred_solvents: list[str] | None = None,
    preferred_additives: list[str] | None = None,
    target_conditions: list[str],
    max_candidates: int,
) -> list[FormulationCandidate]:
    """Use formulation anchors plus same-paper condition chunks only."""
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        paper = str(chunk.get("paper_title") or chunk.get("paper_id") or "").strip()
        if paper:
            by_paper[paper].append(chunk)
    def normalized_condition_pattern(condition: str) -> str:
        temperature = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*°\s*C\s*", condition, re.I)
        if temperature:
            return rf"{re.escape(temperature.group(1))}\s*°\s*C"
        escaped = re.escape(condition).replace(r"\ ", r"\s*")
        return escaped.replace(r"°", r"°\s*")

    target_pattern = re.compile(
        "|".join(normalized_condition_pattern(condition) for condition in target_conditions if condition), re.I
    ) if target_conditions else None

    output: list[FormulationCandidate] = []
    for paper, paper_chunks in by_paper.items():
        for anchor in paper_chunks:
            anchor_text = str(anchor.get("text") or anchor.get("content") or "")
            formulation = extract_explicit_formulation(
                anchor_text,
                preferred_salts=preferred_salts,
                preferred_solvents=preferred_solvents,
            )
            if not formulation or not formulation["solvent_ratios"]:
                continue
            requested_solvents = {
                (_canonical(item, SOLVENT_ALIASES) or item).lower()
                for item in (preferred_solvents or [])
            }
            formulation_solvents = {
                item.lower() for item in formulation.get("solvents", [])
            }
            if requested_solvents and not (
                requested_solvents & formulation_solvents
            ):
                continue
            # ``preferred_additives`` 是可选的设计杠杆，不是每个候选必须
            # 含有的硬约束。否则 1 M NaPF6/THF 这类有直接文献依据的无添加剂
            # 体系会被错误排除，并迫使上层 Agent 手工拼装不受控候选。
            evidence = [EvidenceRef(
                chunk_id=str(anchor.get("chunk_id") or ""),
                paper_title=paper,
                excerpt=anchor_text,
                evidence_kind="formulation",
            )]
            # 配方叙述常把“盐/溶剂”和“浓度”分置于同一论文的相邻块。
            # 仅在同盐且共享至少一种已选溶剂时补入显式浓度，并同时保留该块，
            # 不跨论文或凭常识推断浓度。
            if formulation.get("concentration_mol_l") is None:
                for supplemental in paper_chunks:
                    if supplemental is anchor:
                        continue
                    supplemental_text = str(
                        supplemental.get("text") or supplemental.get("content") or ""
                    )
                    parsed = extract_explicit_formulation(
                        supplemental_text, preferred_salts=preferred_salts,
                        preferred_solvents=preferred_solvents,
                    )
                    if not parsed or parsed.get("concentration_mol_l") is None:
                        continue
                    if formulation_solvents != {
                        item.lower() for item in parsed.get("solvents", [])
                    }:
                        continue
                    if set(formulation.get("salts", [])) != set(parsed.get("salts", [])):
                        continue
                    original_ratios = formulation.get("solvent_ratios") or {}
                    supplemental_ratios = parsed.get("solvent_ratios") or {}
                    if supplemental_ratios and original_ratios != supplemental_ratios:
                        continue
                    if (formulation.get("solvent_ratio_basis") != "unknown"
                            and parsed.get("solvent_ratio_basis") != "unknown"
                            and formulation.get("solvent_ratio_basis") != parsed.get("solvent_ratio_basis")):
                        continue
                    formulation["concentration_mol_l"] = parsed["concentration_mol_l"]
                    evidence.append(EvidenceRef(
                        chunk_id=str(supplemental.get("chunk_id") or ""),
                        paper_title=paper,
                        excerpt=supplemental_text,
                        evidence_kind="formulation",
                    ))
                    break
            condition_chunks = [
                chunk for chunk in paper_chunks
                if chunk is not anchor and target_pattern and target_pattern.search(str(chunk.get("text") or chunk.get("content") or ""))
            ]
            # 系列配方常在方法块中一次列出多个 NaDFP 浓度，而性能块用
            # ``BE2F+0.5DFP`` 指明真正接受测试的变体。只有同一论文、目标条件
            # 证据中出现唯一变体值时才回填，避免从 0.2/0.5/1.0 wt% 中猜测。
            dfp_variant_amounts = {
                match.group(1)
                for chunk in condition_chunks
                for match in re.finditer(
                    r"\b(?:BE2F\s*\+\s*)?(\d+(?:\.\d+)?)\s*DFP\b",
                    str(chunk.get("text") or chunk.get("content") or ""),
                    re.I,
                )
            }
            if (
                "NaDFP" in formulation.get("additives", [])
                and "NaDFP" not in formulation.get("additive_amounts", {})
                and len(dfp_variant_amounts) == 1
            ):
                formulation["additive_amounts"]["NaDFP"] = (
                    f"{next(iter(dfp_variant_amounts))} wt%"
                )

            # 保留本轮调用方已经检索到的同论文 observation evidence，而不是只
            # 保留恰好命中 ``target_conditions`` 字面串的块。joint-context gate 会
            # 再判断这些 observation 是否真正适用于目标条件；这里不做 target 补全。
            support_chunks: list[tuple[dict[str, Any], str, str]] = []
            for chunk in paper_chunks:
                if chunk is anchor:
                    continue
                support_text = str(chunk.get("text") or chunk.get("content") or "")
                support_kind = _retrieved_evidence_kind(chunk, support_text)
                if support_kind is not None:
                    support_chunks.append((chunk, support_text, support_kind))
            condition_texts = [text for _, text, _ in support_chunks]
            anchor_chunk_id = str(anchor.get("chunk_id") or "")
            paper_identity = str(anchor.get("paper_id") or "").strip()
            if not paper_identity and "::" in anchor_chunk_id:
                paper_identity = anchor_chunk_id.split("::", 1)[0]
            paper_identity = paper_identity or paper
            object_id, object_label = resolve_shared_explicit_object(
                paper_id=paper_identity,
                formulation_text=anchor_text,
                evidence_texts=condition_texts,
            )
            if object_id and object_label:
                evidence = [
                    item.model_copy(update={"scientific_object_id": object_id})
                    if text_explicitly_names_object(item.excerpt, object_label)
                    else item
                    for item in evidence
                ]
            extracted_condition_metadata = False
            for chunk, support_text, support_kind in support_chunks:
                support_object_id = (
                    object_id
                    if object_label
                    and text_explicitly_names_object(support_text, object_label)
                    else None
                )
                support_chunk_id = str(chunk.get("chunk_id") or "")
                condition_scope = extract_condition_scope(support_text)
                condition_scope_payload = (
                    condition_scope.model_dump(mode="json", exclude_none=True)
                    if condition_scope is not None
                    else None
                )
                if condition_scope_payload:
                    extracted_condition_metadata = True
                # experiment_id 只透传上游显式提供的实验身份。这里从 source text
                # 抽取 condition_scope，但不把 paper/chunk/condition 的稳定记录键
                # 冒充 scientific experiment identity。
                experiment_id = str(chunk.get("experiment_id") or "").strip() or None
                evidence.append(EvidenceRef(
                    chunk_id=support_chunk_id,
                    paper_title=paper,
                    excerpt=support_text,
                    evidence_kind=support_kind,
                    scientific_object_id=support_object_id,
                    experiment_id=experiment_id,
                    condition_scope=condition_scope_payload,
                ))
            gaps: list[str] = []
            if not support_chunks:
                gaps.append("同一论文中未找到可保留的实验 observation 证据")
            inferred_fields = (
                ["retrieved_same_paper_evidence_preserved_v2"]
                if support_chunks
                else []
            )
            if object_id:
                inferred_fields.append("scientific_object_id_from_shared_explicit_label_v1")
            if extracted_condition_metadata:
                inferred_fields.append("condition_scope_extracted_from_source_text_v1")
            output.append(FormulationCandidate(
                candidate_id=f"same-paper-{len(output) + 1}",
                scientific_object_id=object_id,
                **formulation,
                target_rationale=["配方比例来自同一论文的显式配方文本；性能条件证据单独绑定到同论文 child chunk。"],
                evidence=evidence,
                evidence_level="moderate",
                inferred_fields=inferred_fields,
                unresolved_gaps=gaps,
                evidence_scope="same_paper",
            ))
            if len(output) >= max_candidates:
                return output
    return output
