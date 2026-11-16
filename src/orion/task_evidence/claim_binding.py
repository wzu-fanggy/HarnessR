"""声明与逐字引文之间的确定性最低限度绑定检查。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from orion.task_evidence.task_gates import (
    DEFAULT_COMPARABILITY_FIELDS,
    ComparisonConditions,
)


_UNIT_VALUE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[-+−–]?\d+(?:\.\d+)?\s*"
    r"(?:%|℃|°?\s*[CFK]\b|mA\s*h(?:\s*g\s*(?:−|-)?\s*1)?|A\s*h(?:\s*g\s*(?:−|-)?\s*1)?|"
    r"mA(?:\s*g\s*(?:−|-)?\s*1)?|mS(?:\s*cm\s*(?:−|-)?\s*1)?\b|S(?:\s*cm\s*(?:−|-)?\s*1)?\b|"
    r"mol(?:\s*L(?:−|-)?1)?|M\b|V\b|cycles?\b|hours?\b|h\b)",
    flags=re.IGNORECASE,
)


def _canonical_unit_value(value: str) -> str:
    return (
        value.replace("−", "-")
        .replace("–", "-")
        .replace("℃", "°C")
        .replace(" ", "")
        .lower()
    )


def unit_values(text: str) -> set[str]:
    """提取会改变实验结论含义的带单位数值，忽略化学式中的数字。"""

    # MinerU/PDF 常把单位写成 ``$0.8\mathrm{M}$``；先只展开单位样式命令，
    # 不改写数值本身，避免把排版差异误报成 numeric drift。
    normalized = re.sub(
        r"\\(?:mathrm|text|operatorname)\s*\{([^{}]*)\}",
        r"\1",
        str(text or ""),
    )
    normalized = re.sub(r"\^\{\s*\\circ\s*\}", "°", normalized)
    normalized = re.sub(r"\^\{([^{}]*)\}", r"\1", normalized)
    normalized = (
        normalized.replace("$", "")
        .replace("^", "")
        .replace(r"\%", "%")
        .replace(r"\,", " ")
        .replace("\\ ", " ")
    )
    return {
        _canonical_unit_value(match.group(0))
        for match in _UNIT_VALUE.finditer(normalized)
    }


@dataclass(frozen=True)
class ClaimBindingCheck:
    accepted: bool
    level: str
    checks: tuple[str, ...]
    reason: str = ""
    unmatched_unit_values: tuple[str, ...] = ()
    missing_condition_fields: tuple[str, ...] = ()
    mismatched_condition_fields: tuple[str, ...] = ()


def check_quote_binding(*, claim_text: str, evidence_text: str) -> ClaimBindingCheck:
    """检查声明是否引入权威证据块中不存在的带单位数值。

    该函数不判断语义蕴含，也不声称实验条件已经完整绑定。
    """

    claim_values = unit_values(claim_text)
    evidence_values = unit_values(evidence_text)
    unmatched = tuple(sorted(claim_values - evidence_values))
    if unmatched:
        return ClaimBindingCheck(
            accepted=False,
            level="unverified",
            checks=("verbatim_quote_identity", "unit_value_consistency"),
            reason="unit_value_not_in_authoritative_evidence",
            unmatched_unit_values=unmatched,
        )
    return ClaimBindingCheck(
        accepted=True,
        level="quote_bound",
        checks=("verbatim_quote_identity", "unit_value_consistency"),
    )


def _condition_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field == "temperature_c":
        return round(float(value), 6)
    if field == "voltage_window_v":
        return tuple(round(float(item), 6) for item in value)
    if field == "cycle_count":
        return int(value)
    normalized = " ".join(str(value).split()).casefold()
    return normalized.replace(" ", "") if field == "rate" else normalized


def check_condition_binding(
    *,
    claim_experiment_id: str,
    evidence_experiment_id: str,
    claim_scope: ComparisonConditions | dict[str, Any] | None,
    evidence_scope: ComparisonConditions | dict[str, Any] | None,
    condition_quote: str,
    evidence_text: str,
) -> ClaimBindingCheck:
    """只在同一实验记录且六个关键条件完整一致时返回 condition_bound。"""

    checks = (
        "experiment_identity",
        "temperature_c",
        "electrode",
        "voltage_window_v",
        "rate",
        "cycle_count",
        "measurement_definition",
        "condition_quote_identity",
    )
    if not claim_experiment_id or not evidence_experiment_id:
        return ClaimBindingCheck(False, "quote_bound", checks, "missing_experiment_id")
    if claim_experiment_id != evidence_experiment_id:
        return ClaimBindingCheck(False, "quote_bound", checks, "cross_experiment_evidence")
    if claim_scope is None or evidence_scope is None:
        return ClaimBindingCheck(False, "quote_bound", checks, "missing_condition_scope")
    if not condition_quote:
        return ClaimBindingCheck(False, "quote_bound", checks, "missing_condition_quote")
    if evidence_text.count(condition_quote) != 1:
        return ClaimBindingCheck(
            False,
            "quote_bound",
            checks,
            "condition_quote_not_unique_verbatim",
        )
    claim = (
        claim_scope
        if isinstance(claim_scope, ComparisonConditions)
        else ComparisonConditions.model_validate(claim_scope)
    )
    evidence = (
        evidence_scope
        if isinstance(evidence_scope, ComparisonConditions)
        else ComparisonConditions.model_validate(evidence_scope)
    )
    missing = tuple(
        field
        for field in DEFAULT_COMPARABILITY_FIELDS
        if getattr(claim, field) is None or getattr(evidence, field) is None
    )
    if missing:
        return ClaimBindingCheck(
            False,
            "quote_bound",
            checks,
            "incomplete_condition_scope",
            missing_condition_fields=missing,
        )
    mismatched = tuple(
        field
        for field in DEFAULT_COMPARABILITY_FIELDS
        if _condition_value(field, getattr(claim, field))
        != _condition_value(field, getattr(evidence, field))
    )
    if mismatched:
        return ClaimBindingCheck(
            False,
            "quote_bound",
            checks,
            "condition_scope_mismatch",
            mismatched_condition_fields=mismatched,
        )
    return ClaimBindingCheck(True, "condition_bound", checks)
