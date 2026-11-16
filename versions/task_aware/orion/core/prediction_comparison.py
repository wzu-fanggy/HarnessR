"""汇总本轮已执行的性质预测，不把模型结果当作文献实验事实。"""
from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class PredictionComparisonRequest(BaseModel):
    candidate_ids: list[str] = Field(min_length=2)
    temperatures_c: list[float] = Field(min_length=1)
    objective: Literal["maximise_minimum_conductivity"]
    priority_count: int = Field(default=1, ge=1, description="按预测初筛指标选取的优先候选数；并列末名一并保留")
    reference_temperature_c: float | None = None

    @model_validator(mode="after")
    def validate_scope(self):
        if any(not item.strip() for item in self.candidate_ids):
            raise ValueError("candidate_ids must be non-empty")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be distinct")
        if (not all(math.isfinite(t) for t in self.temperatures_c)
                or len(set(self.temperatures_c)) != len(self.temperatures_c)):
            raise ValueError("temperatures must be finite and distinct")
        if self.reference_temperature_c is not None and not math.isfinite(self.reference_temperature_c):
            raise ValueError("reference temperature must be finite")
        if self.priority_count > len(self.candidate_ids):
            raise ValueError("priority_count must not exceed candidate count")
        return self


class PredictionComparisonAssessment(BaseModel):
    status: Literal["model_assessed", "insufficient_data", "inconsistent_inputs"]
    request: PredictionComparisonRequest
    model_id: str = ""
    unit: str = "mS/cm"
    minimum_by_candidate: dict[str, float] = Field(default_factory=dict)
    priority_candidate_ids: list[str] = Field(default_factory=list)
    reference_by_candidate: dict[str, float] = Field(default_factory=dict)
    reference_priority_candidate_ids: list[str] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


def assess_prediction_comparison(
    records: list[dict[str, Any]], request: PredictionComparisonRequest,
) -> PredictionComparisonAssessment:
    """同模型、同配方、同温度集合的预测才能按显式指标作比较。

    records 由 recorded_predictions 从 AIMessage/ToolMessage 配对读取。这里
    不接受报告草稿中的数值，不使用自然语言关键词决定推荐，也不填充电池工况。
    """
    selected = []
    points: dict[tuple[str, float], float] = {}
    signatures: dict[str, set[str]] = {}
    model_ids: set[str] = set()
    issues = []
    inconsistent = False
    temperatures = list(request.temperatures_c)
    if request.reference_temperature_c is not None and request.reference_temperature_c not in temperatures:
        temperatures.append(request.reference_temperature_c)
    for record in records:
        cid = record.get("candidate_id")
        p = record.get("prediction") or {}
        t = p.get("temperature_c")
        if cid not in request.candidate_ids or t not in temperatures:
            continue
        value = p.get("numeric_value")
        if (p.get("status") != "predicted" or p.get("property_name") != "ionic_conductivity"
                or p.get("unit") != "mS/cm" or not p.get("model_id")
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            issues.append(f"invalid_prediction:{cid}:{t}")
            inconsistent = True
            continue
        if not record.get("tool_call_id"):
            issues.append(f"missing_tool_call:{cid}:{t}")
            inconsistent = True
            continue
        features = p.get("features") or {}
        candidate = record.get("candidate_input") or {}
        composition = record.get("composition_input") or {}
        # 优先采用模型实际接受的三个组成特征，温度是比较轴，不是配方身份。
        if all(k in features for k in ("x_NaSalt", "x_EC", "x_PC")):
            identity = {k: features[k] for k in ("x_NaSalt", "x_EC", "x_PC")}
        else:
            identity = composition or candidate.get("composition_inputs") or candidate.get("conductivity_features")
        if not identity:
            issues.append(f"missing_composition:{cid}:{t}")
            inconsistent = True
            continue
        signatures.setdefault(cid, set()).add(json.dumps(identity, sort_keys=True))
        model_ids.add(p["model_id"])
        key = (cid, float(t))
        if key in points and points[key] != value:
            issues.append(f"conflicting_prediction:{cid}:{t}")
            inconsistent = True
        points[key] = float(value)
        selected.append(record)
    if len(model_ids) > 1:
        issues.append("different_models")
        inconsistent = True
    if any(len(values) > 1 for values in signatures.values()):
        issues.append("candidate_composition_changed")
        inconsistent = True
    for cid in request.candidate_ids:
        for t in temperatures:
            if (cid, float(t)) not in points:
                issues.append(f"missing_prediction:{cid}:{t}")
    if inconsistent or issues:
        return PredictionComparisonAssessment(
            status="inconsistent_inputs" if inconsistent else "insufficient_data",
            request=request, records=selected, issues=list(dict.fromkeys(issues)),
        )
    minima = {cid: min(points[cid, float(t)] for t in request.temperatures_c)
              for cid in request.candidate_ids}
    ordered = sorted(request.candidate_ids, key=lambda cid: -minima[cid])
    boundary = minima[ordered[request.priority_count - 1]]
    priority = [cid for cid in ordered if minima[cid] > boundary
                or math.isclose(minima[cid], boundary, rel_tol=1e-12, abs_tol=1e-12)]
    reference = ({cid: points[cid, float(request.reference_temperature_c)] for cid in request.candidate_ids}
                 if request.reference_temperature_c is not None else {})
    return PredictionComparisonAssessment(
        status="model_assessed", request=request, model_id=next(iter(model_ids)),
        minimum_by_candidate=minima,
        priority_candidate_ids=priority,
        reference_by_candidate=reference,
        reference_priority_candidate_ids=[cid for cid, value in reference.items()
                                          if math.isclose(value, max(reference.values()), rel_tol=1e-12, abs_tol=1e-12)],
        records=selected,
    )


def comparison_text(assessment: PredictionComparisonAssessment) -> str:
    """只陈述已执行计算的取舍，不推断实验电池性能。"""
    values = "、".join(f"{cid}={value:.6g}" for cid, value in assessment.minimum_by_candidate.items())
    temperatures = "、".join(f"{t:g}" for t in assessment.request.temperatures_c)
    text = (f"在 {temperatures} °C 的 {assessment.model_id} 模型预测中，"
            f"各候选最低电导率为 {values} {assessment.unit}。")
    if assessment.request.priority_count == 1:
        text += (f"按最低预测电导率最大这一初筛指标，{'、'.join(assessment.priority_candidate_ids)}"
                 "的数值最高，可优先开展低温输运实验验证。")
    else:
        chosen = assessment.priority_candidate_ids
        values = "、".join(f"{cid}（{assessment.minimum_by_candidate[cid]:.4f} {assessment.unit}）"
                          for cid in chosen)
        tie_note = "（含末位并列）" if len(chosen) > assessment.request.priority_count else ""
        text += (f"按最低预测电导率最大这一初筛指标，优先开展后续实验的前"
                 f"{assessment.request.priority_count}种候选{tie_note}依次为{values}。")
    text += "这一排序来自模型计算，不等于电池容量或循环性能的实验排名。"
    if assessment.reference_by_candidate:
        values = "、".join(f"{cid}={value:.6g}" for cid, value in assessment.reference_by_candidate.items())
        text += (f"在参照温度 {assessment.request.reference_temperature_c:g} °C，预测值为 {values} {assessment.unit}，"
                 f"数值最高的候选为 {'、'.join(assessment.reference_priority_candidate_ids)}。")
    return text
