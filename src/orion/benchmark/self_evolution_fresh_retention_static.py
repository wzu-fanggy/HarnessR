"""D2 retention Static baseline 的确定性输入/输出合同。

本模块不调用模型。它负责冻结 retention 任务如何嵌入当前证据、验证 Static
ORION 返回的 JSON 结构，并调用既有 deterministic retention scorer。开发执行
runner 只负责产生原始 agent trace；解析与评分必须发生在 raw 落盘之后。
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.benchmark.self_evolution_fresh_retention import RetentionScore, score_retention_output


_TASK_TYPE_BY_CLASS = {
    "source_discipline": "evidence_synthesis",
    "scientific_scope_discipline": "design",
    "design_constraint_handling": "design",
    "measurement_definition_discipline": "evidence_synthesis",
}


class RetentionStaticOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["answer", "hold"]
    direct_facts: list[str]
    missing_facts: list[str]
    evidence_refs_used: list[str]
    unsupported_claims: list[str]


class RetentionStaticTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["orion_self_evolution_retention_static_trace_v1"]
    task_id: str
    task_class: str
    task_type: str
    answer_raw: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    ok: bool
    llm_request_max_retries: int | None = None
    llm_transport_retry_count: int = 0
    blind_context_audit: dict[str, Any]
    resolved_runtime: dict[str, Any]
    native_json_response_format: dict[str, str]
    suppressed_runtime_stdout_sha256: str
    development_only_not_formal: bool


class RetentionStaticEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    semantic_valid: bool
    output: RetentionStaticOutput | None = None
    score: RetentionScore | None = None
    reason_codes: list[str] = Field(default_factory=list)


def retention_task_type(task: dict[str, Any]) -> str:
    task_class = str(task.get("task_class") or "")
    try:
        return _TASK_TYPE_BY_CLASS[task_class]
    except KeyError as exc:
        raise ValueError(f"unmapped retention task_class: {task_class!r}") from exc


def render_retention_static_question(
    *,
    bank: dict[str, Any],
    task: dict[str, Any],
) -> str:
    evidence = task.get("evidence") or []
    if not evidence:
        raise ValueError("retention task requires current-task evidence")
    example = {
        "decision": "hold",
        "direct_facts": ["fact tied to supplied record"],
        "missing_facts": ["condition not established by supplied record"],
        "evidence_refs_used": [str(evidence[0]["evidence_id"])],
        "unsupported_claims": [],
    }
    packet = {
        "current_task_evidence": evidence,
        "user_task": task["user_task"],
        "fixed_output_contract": bank["fixed_output_contract"],
        "json_output_example_shape_only": example,
    }
    return (
        "This is a frozen development-only Static ORION retention preflight. "
        "Use only the current-task evidence in the JSON packet below as scientific evidence. "
        "Do not retrieve or infer missing scientific conditions. No persistent candidate is mounted. "
        "Return JSON only, with exactly the five keys shown by the fixed output contract and example.\n\n"
        + json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2)
    )


def evaluate_retention_static_trace(
    *,
    task: dict[str, Any],
    trace: RetentionStaticTrace | dict[str, Any],
) -> RetentionStaticEvaluation:
    if not isinstance(trace, RetentionStaticTrace):
        trace = RetentionStaticTrace.model_validate(trace)
    reasons: list[str] = []
    if trace.task_id != str(task.get("task_id") or ""):
        reasons.append("retention_trace_task_id_mismatch")
    if not trace.ok:
        reasons.append("retention_static_agent_failed")
    if not bool(trace.blind_context_audit.get("isolation_passed")):
        reasons.append("retention_static_context_isolation_failed")
    if trace.native_json_response_format != {"type": "json_object"}:
        reasons.append("retention_static_native_json_not_bound")

    output: RetentionStaticOutput | None = None
    if not reasons:
        try:
            output = RetentionStaticOutput.model_validate_json(trace.answer_raw)
        except Exception:
            reasons.append("retention_static_output_schema_invalid")

    score: RetentionScore | None = None
    if output is not None and not reasons:
        score = score_retention_output(
            task=task,
            output=output.model_dump(mode="json"),
            persistent_candidate_aliases=[],
        )
        if not score.passed:
            reasons.extend(score.reason_codes)

    return RetentionStaticEvaluation(
        task_id=str(task.get("task_id") or ""),
        semantic_valid=output is not None and trace.ok and bool(trace.blind_context_audit.get("isolation_passed")),
        output=output,
        score=score,
        reason_codes=reasons or ["retention_static_baseline_passed"],
    )


__all__ = [
    "RetentionStaticEvaluation",
    "RetentionStaticOutput",
    "RetentionStaticTrace",
    "evaluate_retention_static_trace",
    "render_retention_static_question",
    "retention_task_type",
]
