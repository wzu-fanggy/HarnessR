"""无外部副作用的候选配方约束评估器。

本模块只消费调用方已经提供的候选、证据和约束。它不访问 KG、RAG、LLM 或
文件系统，也不生成推荐文本；因此同一输入可以在测试、历史 trace 复放和在线
LangGraph 中得到相同结果。
"""

from __future__ import annotations

import re
from typing import Any

from tools.kg.models import (
    CriterionEvaluation,
    FormulationCandidate,
    ScreeningRequest,
    ScreeningResult,
)

from .criteria import evaluate_constraint_records, evaluate_criteria
from .object_identity import resolve_candidate_payload_object_identity


def _normalize_candidate_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """把设计候选的 EvidenceRef 兼容为筛选模型的 EvidenceRecord。"""

    candidate = dict(raw)
    normalized_evidence = []
    for evidence in candidate.get("evidence") or []:
        if not isinstance(evidence, dict):
            normalized_evidence.append(evidence)
            continue
        record = dict(evidence)
        if not record.get("text") and record.get("excerpt"):
            record["text"] = record["excerpt"]
        if record.get("score") is None:
            record["score"] = 0.0
        normalized_evidence.append(record)
    if candidate.get("salt") is None:
        candidate["salt"] = ""
    candidate["evidence"] = normalized_evidence
    return resolve_candidate_payload_object_identity(candidate)


def coerce_screening_request(raw: Any) -> ScreeningRequest:
    """宽容解析旧工具调用，但不补查任何外部数据。"""

    if isinstance(raw, ScreeningRequest):
        return raw
    if isinstance(raw, dict):
        normalized = dict(raw)
        normalized["candidates"] = [
            _normalize_candidate_payload(candidate)
            if isinstance(candidate, dict)
            else candidate
            for candidate in raw.get("candidates") or []
        ]
        return ScreeningRequest.model_validate(normalized)
    raise TypeError(f"Unsupported request type: {type(raw)}")


def extract_properties_from_evidence(candidate: FormulationCandidate) -> list[dict]:
    """只从候选自带的可追溯原文提取数值属性。"""

    properties = []
    candidate_terms = {
        term.lower()
        for term in [candidate.salt, *candidate.solvents, *candidate.additives]
        if term
    }
    for record in candidate.evidence:
        text = record.text.lower()
        material = candidate.salt
        for term in candidate_terms:
            if term in text:
                material = term
                break
        for match in re.finditer(
            r"(\d+\.?\d*)\s*"
            r"(ms?\s*cm\^-?1|mpa\s*s|cycles?|mah\s*g\^-?1|ms?\s*/\s*cm)",
            text,
        ):
            properties.append(
                {
                    "material": material,
                    "property": match.group(0),
                    "confidence": 0.8,
                }
            )
    return properties


def _infer_evidence_level(
    candidate: FormulationCandidate,
    matched_properties: list[dict],
) -> str:
    if not candidate.evidence:
        return "weak"
    if not matched_properties:
        return "moderate"
    candidate_terms = {
        term.lower()
        for term in [candidate.salt, *candidate.solvents, *candidate.additives]
        if term
    }
    if any(
        any(term in record.text.lower() for term in candidate_terms)
        for record in candidate.evidence
    ):
        return "strong"
    return "moderate"


def _is_advisory_criterion(criterion: str) -> bool:
    """未量化的质量偏好保留为 gap，但不冒充硬约束。"""

    lowered = criterion.lower()
    return (
        ("conductivity" in lowered or "电导" in criterion)
        and any(
            word in lowered
            for word in ("adequate", "sufficient", "acceptable", "good")
        )
    )


def _candidate_evidence_gate(
    candidate: FormulationCandidate,
) -> list[CriterionEvaluation]:
    checks: list[CriterionEvaluation] = []
    if (
        candidate.evidence_scope in {"same_child", "same_paper"}
        and not candidate.solvent_ratios
    ):
        checks.append(
            CriterionEvaluation(
                criterion="explicit solvent ratio in formulation evidence",
                kind="formulation",
                status="failed",
                expected={"required": True},
                reason="配方候选缺少正文中的显式溶剂比例，不能通过硬约束评估。",
            )
        )
    for gap in candidate.unresolved_gaps:
        checks.append(
            CriterionEvaluation(
                criterion=gap,
                kind="evidence_gate",
                status="insufficient_evidence",
                expected={"required": True},
                observed=[],
                reason=gap,
            )
        )
    return checks


def evaluate_formulation_constraints(
    request: ScreeningRequest | dict[str, Any],
) -> list[ScreeningResult]:
    """确定性评估候选的全部约束，不访问外部服务。

    ``score`` 只保留为旧调用方的覆盖度诊断字段；``passed`` 只由证据等级和
    constraint status 决定。``recommendation`` 固定为空，最终动作由
    ``DecisionState`` 独立裁决。
    """

    req = coerce_screening_request(request)
    criteria = list(dict.fromkeys([*req.criteria, *req.target_conditions]))
    results: list[ScreeningResult] = []

    for candidate in req.candidates:
        matched_properties = extract_properties_from_evidence(candidate)
        effective_level = (
            candidate.evidence_level
            if candidate.evidence_level != "weak"
            else _infer_evidence_level(candidate, matched_properties)
        )
        evaluations = (
            evaluate_constraint_records(candidate, req.constraint_records)
            if req.constraint_records
            else evaluate_criteria(candidate, criteria)
        )
        evaluations.extend(_candidate_evidence_gate(candidate))
        non_satisfied = [
            item for item in evaluations if item.status != "satisfied"
        ]
        blocking = [
            item
            for item in non_satisfied
            if item.status != "unsupported"
            or not _is_advisory_criterion(item.criterion)
        ]

        evidence_scores = {"strong": 90.0, "moderate": 60.0, "weak": 30.0}
        score = evidence_scores.get(effective_level, 0.0)
        score += min(10.0, len(matched_properties) * 2.0)
        score += sum(3.0 for item in evaluations if item.status == "satisfied")
        score -= sum(20.0 for item in evaluations if item.status == "failed")
        score -= sum(
            12.0
            for item in evaluations
            if item.status == "insufficient_evidence"
            or (
                item.status == "unsupported"
                and not _is_advisory_criterion(item.criterion)
            )
        )

        gaps: list[str] = []
        if effective_level == "weak":
            gaps.append("候选证据等级不足")
        if not matched_properties:
            gaps.append("缺少性能/条件属性证据")
        gaps.extend(item.reason for item in non_satisfied)

        results.append(
            ScreeningResult(
                candidate_id=candidate.candidate_id,
                passed=effective_level != "weak" and not blocking,
                score=round(max(0.0, min(100.0, score)), 2),
                evidence_level=effective_level,
                matched_properties=matched_properties,
                model_predictions=candidate.model_predictions,
                criteria_evaluations=evaluations,
                gaps=list(dict.fromkeys(gap for gap in gaps if gap)),
                recommendation="",
            )
        )

    return results
