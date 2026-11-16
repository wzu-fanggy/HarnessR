"""结构化、无模型的电解液任务报告编译器。

这是历史结构化输出的兼容接口，不是主 Agent 正常作答的前置步骤。
显式调用中可复用既有纯规则补算 ``DecisionState``，不调用 LLM、不检索 KG。
旧格式的证据与准入合同继续保留；通用循环不执行该接口的科学裁决。
"""

from __future__ import annotations

from orion.core.task_scope import current_task_messages, current_user_message

from collections import defaultdict
import hashlib
import json
import re
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field, field_validator, model_validator

from orion.core.design_decision import (
    CandidateEvaluation,
    CandidateState,
    DecisionState,
    EvidenceGap,
    EvidenceLedgerItem,
    TaskContract,
)
from orion.core.claim_contract import (
    ClaimContract,
    ClaimProvenance,
    SourceRef,
    VerificationResult,
)
from orion.core.evidence_extract import parse_chunk_payload
from orion.core.prediction_comparison import (
    PredictionComparisonRequest, assess_prediction_comparison, comparison_text,
)
from orion.task_evidence.contracts_v2 import (
    ArtifactClaim,
    ArtifactEvidence,
    ArtifactGap,
    CandidateEvaluationArtifact,
    CandidateScreeningArtifact,
    DesignTaskArtifact,
    EvidenceSynthesisArtifact,
    InverseDesignTaskArtifact,
    TaskType,
    canonical_task_type,
    default_task_contract,
    task_type_for_skill,
)
from orion.task_evidence.task_gates import (
    CandidateComparisonRecord,
    ComparisonConditions,
    DiscriminatingTest,
    HypothesisRecord,
    evaluate_candidate_comparability,
    evaluate_inverse_hypotheses,
)
from orion.task_evidence.claim_binding import (
    check_condition_binding,
    unit_values,
)
from orion.task_evidence.semantic_support import (
    SEMANTIC_SUPPORT_POLICY_VERSION,
    enforce_quote_identity_v2,
)
from orion.task_evidence.evidence_applicability import (
    ApplicabilityInput,
    ApplicabilityRecord,
    evaluate_applicability_input,
)
from orion.task_evidence.evidence_use import compile_evidence_use_decisions
from orion.task_evidence.condition_extraction import (
    extract_condition_scope,
    extract_comparison_conditions,
    infer_condition_quote,
    infer_experiment_id,
)
from tools.kg.models import ScreeningResult


CLAIM_VALIDATION_SCOPE = (
    "quote_identity_v2_canonical_output_and_unit_value_consistency; "
    "condition_bound_requires_same_complete_experiment_scope; "
    "free_text_paraphrase_not_promoted_to_verified_fact"
)


class EvidenceChunk(BaseModel):
    chunk_id: str
    content: str
    paper_id: str = ""
    paper_title: str = ""
    section_title: str = ""
    source: str = ""
    experiment_id: str = ""
    condition_scope: ComparisonConditions | None = None
    condition_quote: str = Field(default="", max_length=600)

    @model_validator(mode="before")
    @classmethod
    def _alias_text_to_content(cls, value: Any):
        """兼容 Agent 从候选证据复制时使用的 `text` 字段名。"""
        if isinstance(value, dict) and "text" in value and "content" not in value:
            value = dict(value)
            value["content"] = value.pop("text")
        return value


class CandidateScheme(BaseModel):
    """旧公开参数的兼容模型；design 报告以 GraphState candidate 为准。"""

    scheme_id: str = Field(default="", description="候选方案标识，如 C1/A/方案一")
    candidate_id: str = Field(
        default="",
        description="screen_formulations 输出的 candidate_id",
    )
    salt: str | None = None
    solvents: list[str] = Field(default_factory=list)
    additives: list[str] = Field(default_factory=list)
    concentration: str | None = None
    rationale: str = ""
    evidence_chunk_ids: list[str] = Field(default_factory=list)


SynthesisClaim = ClaimContract


class SynthesisRequest(BaseModel):
    task_type: TaskType
    finalization_mode: Literal["assessed", "limited"] = Field(
        default="assessed", description="design 可显式选择 limited：不推荐的受限交付，不改变原评估。")
    question: str = ""
    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)
    candidates: list[CandidateScheme] = Field(default_factory=list)
    screening_results: list[ScreeningResult] | None = None
    claims: list[SynthesisClaim] = Field(default_factory=list)
    comparison_axes: list[str] = Field(default_factory=list)
    model_comparison: PredictionComparisonRequest | None = None
    comparison_records: list[CandidateComparisonRecord] = Field(
        default_factory=list
    )
    condition_comparable: bool | None = None
    symptom_boundary: dict[str, Any] = Field(default_factory=dict)
    hypotheses: list[HypothesisRecord] = Field(default_factory=list)
    confounders: list[str] = Field(default_factory=list)
    discriminating_tests: list[DiscriminatingTest] = Field(default_factory=list)
    conditional_adjustments: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    draft_report: str = ""
    direct_answer: str = ""
    applicability_inputs: list[ApplicabilityInput] = Field(default_factory=list)
    applicability_retrieval_allowed: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_task_type(cls, value: Any):
        """兼容旧工具调用，但进入编译器后只保留四类 canonical task type。"""

        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload["task_type"] = canonical_task_type(
            payload.get("task_type"),
            question=str(payload.get("question") or ""),
        )
        return payload

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_inverse_structures(cls, value: Any):
        """兼容 Agent 常见的 inverse-design 旧字段名，避免终端报告因纯 schema 差异失败。

        这里只做字段别名归一化，不把自由文本支持证据提升为 evidence id，也不改变
        后续 ``evaluate_inverse_hypotheses`` 的严格准入语义。
        """

        if not isinstance(value, dict):
            return value
        payload = dict(value)

        hypotheses = payload.get("hypotheses")
        if isinstance(hypotheses, list):
            normalized_hypotheses = []
            for item in hypotheses:
                if not isinstance(item, dict):
                    normalized_hypotheses.append(item)
                    continue
                row = dict(item)
                if not row.get("hypothesis_id") and row.get("id"):
                    row["hypothesis_id"] = row.get("id")
                if not row.get("text"):
                    row["text"] = (
                        row.get("title")
                        or row.get("label")
                        or row.get("description")
                        or ""
                    )
                if "counter_evidence_ids" not in row and isinstance(row.get("counter_evidence"), list):
                    row["counter_evidence_ids"] = row.get("counter_evidence")
                normalized_hypotheses.append(row)
            payload["hypotheses"] = normalized_hypotheses

        tests = payload.get("discriminating_tests")
        if isinstance(tests, list):
            normalized_tests = []
            for index, item in enumerate(tests, start=1):
                if isinstance(item, str):
                    normalized_tests.append(
                        {"test_id": f"legacy-test-{index}", "description": item}
                    )
                    continue
                if not isinstance(item, dict):
                    normalized_tests.append(item)
                    continue
                row = dict(item)
                if not row.get("test_id"):
                    row["test_id"] = row.get("id") or f"legacy-test-{index}"
                if not row.get("description"):
                    row["description"] = row.get("test") or row.get("text") or ""
                if (
                    "distinguishes_hypothesis_ids" not in row
                    and isinstance(row.get("hypotheses"), list)
                ):
                    row["distinguishes_hypothesis_ids"] = row.get("hypotheses")
                controlled_variables = row.get("controlled_variables")
                if isinstance(controlled_variables, str):
                    row["controlled_variables"] = [controlled_variables]
                if "expected_outcomes" not in row:
                    outcomes = {
                        str(key)[len("expected_") :]: str(val)
                        for key, val in row.items()
                        if str(key).startswith("expected_") and val is not None
                    }
                    if outcomes:
                        row["expected_outcomes"] = outcomes
                normalized_tests.append(row)
            payload["discriminating_tests"] = normalized_tests
        return payload

    @field_validator("question", "draft_report", "direct_answer", mode="before")
    @classmethod
    def normalize_empty_text(cls, value: Any) -> str:
        """保留旧参数兼容性；未校验的自由文本草稿不会进入最终报告。"""

        return "" if value is None else str(value)


_AUTHORITATIVE_EVIDENCE_TOOLS = frozenset({
    "search_literature_multiview",
    "search_literature_chunks",
    "search_paper_chunks",
})


def _referenced_chunk_ids(value: Any) -> set[str]:
    """收集请求中的原文引用ID；这里只选择已检索证据，不验证或补造引用。"""
    ids: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "chunk_id" and isinstance(item, str) and item.strip():
                ids.add(item.strip())
            elif key in {"chunk_ids", "evidence_ids"} and isinstance(item, list):
                ids.update(entry.strip() for entry in item if isinstance(entry, str) and entry.strip())
            ids.update(_referenced_chunk_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.update(_referenced_chunk_ids(item))
    return ids


def _authoritative_evidence_from_state(
    state: dict[str, Any] | None,
    *,
    limit: int = 16,
    referenced_chunk_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """从真实检索消息恢复原文，优先保留本次报告正在引用的证据。

    limit控制额外上下文，不能因后续检索而删除已取得且正在引用的原文。
    同ID的预览与完整文本前缀一致时保留完整版本；引用与数值校验照常执行。
    """
    chunks: dict[str, dict[str, Any]] = {}
    messages = list(current_task_messages(state))
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        if str(getattr(message, "name", "") or "") not in _AUTHORITATIVE_EVIDENCE_TOOLS:
            continue
        payload = parse_chunk_payload(str(getattr(message, "content", "") or ""))
        for item in payload or []:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id") or "").strip()
            content = str(item.get("content") or "").strip()
            if not chunk_id or not content:
                continue
            previous = chunks.get(chunk_id)
            if previous:
                old = re.sub(r"\s+", " ", previous["content"]).strip()
                current = re.sub(r"\s+", " ", content).strip()
                if not (len(current) > len(old) and current.startswith(old)):
                    continue
            chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "content": content,
                "paper_id": str(item.get("paper_id") or ""),
                "paper_title": str(item.get("paper_title") or ""),
                "section_title": str(item.get("section_title") or ""),
                "source": str(item.get("source") or ""),
            }
    referenced = referenced_chunk_ids or set()
    selected = [row for cid, row in chunks.items() if cid in referenced]
    extras = [row for cid, row in chunks.items() if cid not in referenced]
    selected.extend(extras[:max(0, limit - len(selected))])
    return selected


def _latest_user_question_from_state(state: dict[str, Any] | None) -> str:
    """报告问题固定为本轮用户输入，内部提醒不能覆盖它。"""
    user = current_user_message(state or {})
    return str(user.content or "").strip() if user else ""


def _loaded_skill_task_type_from_state(
    state: dict[str, Any] | None,
) -> str | None:
    """Resolve the latest successfully loaded canonical Skill without keywords."""

    messages = list(current_task_messages(state))
    successful_call_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "status", None) == "error":
            continue
        call_id = str(getattr(message, "tool_call_id", "") or "")
        if not call_id:
            continue
        content = str(getattr(message, "content", "") or "").strip()
        failed = content.startswith(("Error", "错误", "Traceback", "Exception"))
        if not failed:
            try:
                parsed = json.loads(content)
                failed = isinstance(parsed, dict) and bool(parsed.get("error"))
            except Exception:
                pass
        if not failed:
            successful_call_ids.add(call_id)

    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", []) or []:
            if str(call.get("name") or "") != "load_skill":
                continue
            call_id = str(call.get("id") or call.get("tool_call_id") or "")
            if call_id not in successful_call_ids:
                continue
            args = call.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if isinstance(args, dict):
                task_type = task_type_for_skill(args.get("name"))
                if task_type:
                    return task_type
    return None


def _hydrate_request_from_state(request: Any, state: dict[str, Any] | None) -> Any:
    """在报告工具边界补齐通用字段，并强制事实型任务使用真实检索原文。

    只处理 evidence_synthesis / candidate_screening：design / inverse_design 的
    结构化候选和 ledger 由 DecisionState 路径负责，不能在这里隐式改写。
    """

    if hasattr(request, "model_dump"):
        payload = request.model_dump(mode="json")
    elif isinstance(request, dict):
        payload = dict(request)
    else:
        return request

    requested_task_type = canonical_task_type(
        payload.get("task_type"),
        question=str(payload.get("question") or ""),
    )
    if requested_task_type not in {"evidence_synthesis", "candidate_screening"}:
        return payload

    if not str(payload.get("question") or "").strip():
        user_question = _latest_user_question_from_state(state)
        if user_question:
            payload["question"] = user_question

    # An external structured contract is authoritative. Without one, the
    # successfully loaded canonical Skill is authoritative so routing mistakes
    # remain auditable rather than being silently rewritten at terminalization.
    # Question-text classification is retained only for legacy/untyped direct
    # compiler calls that have neither control signal.
    question = str(payload.get("question") or "").strip()
    configured_task_type = canonical_task_type(
        (state or {}).get("configured_task_type")
    )
    loaded_skill_task_type = _loaded_skill_task_type_from_state(state)
    authoritative_task_type = (
        configured_task_type
        if configured_task_type in {"evidence_synthesis", "candidate_screening"}
        else loaded_skill_task_type
    )
    task_type = authoritative_task_type or (
        canonical_task_type("screening", question=question)
        if question
        else requested_task_type
    )
    payload["task_type"] = task_type

    authoritative = _authoritative_evidence_from_state(
        state, referenced_chunk_ids=_referenced_chunk_ids(payload)
    )
    if authoritative:
        # 一旦本轮存在真实检索结果，完全以 ToolMessage 为准；这会主动覆盖模型
        # 复制时产生的 LaTeX/单位规范化版本，并让 quote validation 对真实原文执行。
        payload["evidence_chunks"] = authoritative
    return payload


def _model_list(values: list[Any] | None, model_type):
    return [
        value if isinstance(value, model_type) else model_type.model_validate(value)
        for value in values or []
    ]


def _citation(chunk_ids: list[str]) -> str:
    unique = list(dict.fromkeys(item for item in chunk_ids if item))
    return "[" + "][".join(unique) + "]" if unique else ""


def _composition_text(candidate: CandidateState) -> str:
    composition = candidate.composition
    salt = composition.get("salt") or "未明确盐"
    concentration = composition.get("concentration_mol_l")
    solvents = ", ".join(str(item) for item in composition.get("solvents") or [])
    ratios = composition.get("solvent_ratios") or {}
    additives = ", ".join(str(item) for item in composition.get("additives") or [])
    amounts = composition.get("additive_amounts") or {}
    model_predictions = [
        item
        for item in composition.get("model_predictions") or []
        if isinstance(item, dict) and item.get("status") == "predicted"
    ]

    parts = [f"盐：{salt}"]
    if concentration is not None:
        concentration_label = "总盐浓度" if len(composition.get("salts") or []) > 1 else "浓度"
        parts.append(f"{concentration_label}：{concentration:g} mol L⁻¹" if isinstance(concentration, (int, float)) else f"浓度：{concentration}")
    for name, value in (composition.get("salt_concentrations_mol_l") or {}).items():
        parts.append(f"{name}：{value:g} mol L⁻¹")
    parts.append(f"溶剂：{solvents or '未明确'}")
    if ratios:
        parts.append(
            "溶剂比例（" + {"mass": "质量", "volume": "体积", "molar": "摩尔"}.get(
                composition.get("solvent_ratio_basis"), "基准待核对"
            ) + "）：" + ", ".join(f"{key}={value}" for key, value in ratios.items())
        )
    if additives:
        parts.append(f"添加剂：{additives}")
    if amounts:
        parts.append(
            "添加剂用量：" + ", ".join(f"{key}={value}" for key, value in amounts.items())
        )
    for prediction in model_predictions:
        value = prediction.get("numeric_value")
        temperature = prediction.get("temperature_c")
        factor = prediction.get("estimated_factor_1sigma")
        if not isinstance(value, (int, float)):
            continue
        label = f"模型估计离子电导率：{value:.3g} mS/cm"
        if isinstance(temperature, (int, float)):
            label += f"（{temperature:g} °C"
            if isinstance(factor, (int, float)):
                label += f"，树间离散度对应约 ×{factor:.2f}"
            label += "；非实验观测）"
        else:
            label += "（非实验观测）"
        parts.append(label)
    return "；".join(parts) + "。"


def _enrich_evidence_chunk(chunk: EvidenceChunk) -> EvidenceChunk:
    """为 RAG 后端未提供结构化条件的 chunk 推断 experiment_id/scope/quote。"""
    content = chunk.content or ""
    condition_scope = chunk.condition_scope
    if condition_scope is None:
        condition_scope = (
            extract_comparison_conditions(content)
            or extract_condition_scope(content)
        )
    condition_quote = chunk.condition_quote
    if not condition_quote.strip():
        condition_quote = infer_condition_quote(content)
    experiment_id = chunk.experiment_id
    if not experiment_id:
        experiment_id = infer_experiment_id(
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            conditions=condition_scope,
        )
    if (
        condition_scope == chunk.condition_scope
        and condition_quote == chunk.condition_quote
        and experiment_id == chunk.experiment_id
    ):
        return chunk
    return EvidenceChunk(
        chunk_id=chunk.chunk_id,
        content=content,
        paper_id=chunk.paper_id,
        paper_title=chunk.paper_title,
        section_title=chunk.section_title,
        source=chunk.source,
        experiment_id=experiment_id,
        condition_scope=condition_scope,
        condition_quote=condition_quote,
    )


def _scope_dict(scope: ComparisonConditions | dict[str, Any] | None) -> dict[str, Any] | None:
    if scope is None:
        return None
    if isinstance(scope, ComparisonConditions):
        return scope.model_dump(mode="json", exclude_none=True)
    return {
        key: value
        for key, value in dict(scope).items()
        if value is not None and value != ""
    } or None


def _attach_quote_condition_scope(
    claim: SynthesisClaim,
    *,
    paper_id: str = "",
) -> SynthesisClaim:
    """Attach only condition fields visible in the claim's authoritative quote."""

    explicit_scope = extract_condition_scope(claim.evidence_quote)
    if explicit_scope is None:
        # Still normalize any stronger, caller-supplied scope so model_copy does
        # not leave a Pydantic object in a dict-typed ClaimContract field.
        normalized = _scope_dict(claim.condition_scope)
        if normalized == claim.condition_scope:
            return claim
        return claim.model_copy(update={"condition_scope": normalized})
    scope = _scope_dict(explicit_scope)
    experiment_id = claim.experiment_id or infer_experiment_id(
        chunk_id=claim.chunk_id or "",
        paper_id=paper_id,
        conditions=scope,
    )
    return claim.model_copy(
        update={
            "condition_scope": scope,
            "experiment_id": experiment_id,
            "reflection_tags": list(
                dict.fromkeys(
                    [*claim.reflection_tags, "source_quote_condition_scope_v1"]
                )
            ),
        }
    )


def _validated_claims(
    claims: list[SynthesisClaim],
    chunks: dict[str, str],
    evidence_records: dict[str, EvidenceChunk] | None = None,
) -> tuple[list[SynthesisClaim], list[dict[str, str]]]:
    accepted: list[SynthesisClaim] = []
    rejected: list[dict[str, str]] = []
    # 先 enrich evidence records，保证后续 condition binding 能拿到结构化条件。
    enriched_records = {
        cid: _enrich_evidence_chunk(chunk)
        for cid, chunk in (evidence_records or {}).items()
    }
    for claim in claims:
        content = chunks.get(claim.chunk_id)
        if content is None:
            rejected.append(
                {
                    "text": claim.text,
                    "chunk_id": claim.chunk_id,
                    "reason": "chunk_not_in_authoritative_evidence",
                }
            )
            continue
        if content.count(claim.evidence_quote) != 1:
            rejected.append(
                {
                    "text": claim.text,
                    "chunk_id": claim.chunk_id,
                    "reason": "quote_not_unique_verbatim",
                }
            )
            continue
        semantic = enforce_quote_identity_v2(
            claim_text=claim.text,
            evidence_quote=claim.evidence_quote,
            claim_type=claim.claim_type,
        )
        if not semantic.accepted:
            rejected.append(
                {
                    "text": claim.text,
                    "chunk_id": claim.chunk_id,
                    "reason": semantic.reason,
                    "unmatched_unit_values": ",".join(
                        semantic.unmatched_unit_values
                    ),
                }
            )
            continue
        semantic_result = VerificationResult(
            verifier=SEMANTIC_SUPPORT_POLICY_VERSION,
            tier="high",
            status="passed",
            rule_id=semantic.action,
            data_used={
                "proposed_claim_text": claim.text,
                "canonical_claim_text": semantic.canonical_claim_text,
                "evidence_quote": claim.evidence_quote,
            },
        )
        effective_claim = claim.model_copy(
            update={
                "claim_text": semantic.canonical_claim_text,
                "verification_results": [
                    *claim.verification_results,
                    semantic_result,
                ],
                "reflection_tags": list(
                    dict.fromkeys(
                        [
                            *claim.reflection_tags,
                            "compiler_quote_identity_v2",
                            semantic.action,
                        ]
                    )
                ),
            }
        )
        # 绑定级别依据模型原始声明是否显式携带条件；canonical quote 只负责安全输出，
        # 不能凭引文本身含有数字而替模型补出 condition-bound 声明。
        if claim.binding_requirement == "quote_bound" and unit_values(claim.text):
            evidence = enriched_records.get(claim.chunk_id)
            if evidence and evidence.experiment_id and evidence.condition_scope:
                # 事实声明从 chunk 继承完整实验条件；比较/排序/推荐声明需自身携带条件。
                if claim.claim_type == "fact":
                    condition_binding = check_condition_binding(
                        claim_experiment_id=evidence.experiment_id,
                        evidence_experiment_id=evidence.experiment_id,
                        claim_scope=evidence.condition_scope,
                        evidence_scope=evidence.condition_scope,
                        condition_quote=evidence.condition_quote,
                        evidence_text=content,
                    )
                    if condition_binding.accepted:
                        effective_claim = effective_claim.model_copy(
                            update={
                                "evidence_binding": "condition_bound",
                                "experiment_id": evidence.experiment_id,
                                "condition_scope": _scope_dict(evidence.condition_scope),
                            }
                        )
                else:
                    claim_scope = extract_comparison_conditions(claim.text)
                    if claim_scope is not None:
                        condition_binding = check_condition_binding(
                            claim_experiment_id=evidence.experiment_id,
                            evidence_experiment_id=evidence.experiment_id,
                            claim_scope=claim_scope,
                            evidence_scope=evidence.condition_scope,
                            condition_quote=evidence.condition_quote,
                            evidence_text=content,
                        )
                        if condition_binding.accepted:
                            effective_claim = effective_claim.model_copy(
                                update={
                                    "evidence_binding": "condition_bound",
                                    "experiment_id": evidence.experiment_id,
                                    "condition_scope": _scope_dict(evidence.condition_scope),
                                }
                            )
        if effective_claim.binding_requirement == "condition_bound":
            evidence = enriched_records.get(effective_claim.chunk_id)
            condition_binding = check_condition_binding(
                claim_experiment_id=effective_claim.experiment_id,
                evidence_experiment_id=evidence.experiment_id if evidence else "",
                claim_scope=effective_claim.condition_scope,
                evidence_scope=evidence.condition_scope if evidence else None,
                condition_quote=evidence.condition_quote if evidence else "",
                evidence_text=content,
            )
            if not condition_binding.accepted:
                rejected.append(
                    {
                        "text": effective_claim.text,
                        "chunk_id": effective_claim.chunk_id,
                        "reason": condition_binding.reason,
                        "missing_condition_fields": ",".join(
                            condition_binding.missing_condition_fields
                        ),
                        "mismatched_condition_fields": ",".join(
                            condition_binding.mismatched_condition_fields
                        ),
                    }
                )
                continue
        evidence = enriched_records.get(effective_claim.chunk_id)
        accepted.append(
            _attach_quote_condition_scope(
                effective_claim,
                paper_id=evidence.paper_id if evidence else "",
            )
        )
    return accepted, rejected


def _meaningful_candidate_aliases(request: SynthesisRequest) -> list[str]:
    """提取 screening 候选在原文中可望出现的稳定标识，跳过 C1/cand-1 等内部编号。"""

    candidate_ids = [
        *[
            item.candidate_id or item.scheme_id
            for item in request.candidates
            if item.candidate_id or item.scheme_id
        ],
        *[item.candidate_id for item in request.comparison_records if item.candidate_id],
    ]
    aliases: list[str] = []

    def _add_alias(alias: str) -> None:
        alias = alias.strip()
        if len(alias) < 2:
            return
        if re.fullmatch(r"(?:c|cand|candidate|paper|same)\d*", alias, re.I):
            return
        folded = alias.casefold()
        if folded not in {item.casefold() for item in aliases}:
            aliases.append(alias)

    for candidate_id in dict.fromkeys(candidate_ids):
        parts = re.findall(r"[A-Za-z][A-Za-z0-9]*", str(candidate_id))
        meaningful = [
            part
            for part in parts
            if not re.fullmatch(r"(?:c|cand|candidate|paper|same)\d*", part, re.I)
        ]
        if meaningful:
            _add_alias(meaningful[0])

    # 模型偶尔会省略 candidates / comparison_records，但用户问题本身已经明确
    # 写出比较双方（例如 ``DMCF 与 FEC`` / ``NaPF6 vs NaFSI``）。此时仍应
    # 启用 comparison quote 覆盖门禁，不能因为结构化候选 id 缺失而放行过界声明。
    question = str(request.question or "")
    pair_pattern = re.compile(
        r"([A-Za-z][A-Za-z0-9+.-]{1,31})\s*(?:与|和|vs\.?|versus)\s*"
        r"([A-Za-z][A-Za-z0-9+.-]{1,31})",
        re.I,
    )
    for left, right in pair_pattern.findall(question):
        _add_alias(left)
        _add_alias(right)
    return aliases


def _standalone_alias_in_text(alias: str, text: str) -> bool:
    """候选缩写必须作为独立 token 出现，避免 FEC 误匹配 ``effect``。"""

    pattern = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
    return re.search(pattern, text, re.I) is not None


def _filter_screening_comparison_claims(
    claims: list[SynthesisClaim],
    request: SynthesisRequest,
) -> tuple[list[SynthesisClaim], list[dict[str, str]]]:
    """比较声明必须让其逐字证据覆盖至少两个被比较候选标识。

    仅当请求能提取出至少两个非内部编号候选别名时启用；C1/C2 等抽象编号
    不会触发该规则。这样可以拒绝“quote 只证明 FEC，却声称优于 DMCF”一类
    syntactically-bound but semantically incomplete comparison claim。
    """

    aliases = _meaningful_candidate_aliases(request)
    if len(aliases) < 2:
        return claims, []
    accepted: list[SynthesisClaim] = []
    rejected: list[dict[str, str]] = []

    def _proposed_text(candidate: SynthesisClaim) -> str:
        # ``_validated_claims`` safely canonicalizes the public claim text to
        # the exact quote. Comparison scope checks must nevertheless inspect
        # the model's original proposition; otherwise a missing formula or
        # condition can disappear during canonicalization and bypass this gate.
        for result in reversed(candidate.verification_results):
            proposed = result.data_used.get("proposed_claim_text")
            if proposed:
                return str(proposed)
        return candidate.text

    for claim in claims:
        if claim.claim_type not in {"comparison", "ranking"}:
            accepted.append(claim)
            continue
        quote = claim.evidence_quote
        covered = [alias for alias in aliases if _standalone_alias_in_text(alias, quote)]
        claim_without_citations = re.sub(
            r"\[[^\]]+\]", " ", _proposed_text(claim)
        )
        # 条件化比较里出现的化学式（如 NaPF6）也必须在逐字 quote 中出现；
        # 否则 quote 只能证明候选差异，不能证明该差异属于 claim 声明的体系。
        formula_tokens = {
            token
            for token in re.findall(r"\b[A-Za-z][A-Za-z0-9]{2,}\b", claim_without_citations)
            if any(ch.isdigit() for ch in token)
        }
        missing_formula = [
            token
            for token in formula_tokens
            if re.search(
                rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
                quote,
                re.I,
            )
            is None
        ]
        if len(covered) >= 2 and not missing_formula:
            accepted.append(claim)
            continue
        rejected.append(
            {
                "text": claim.text,
                "chunk_id": claim.chunk_id or "",
                "reason": (
                    "comparison_conditions_not_in_quote"
                    if missing_formula
                    else "comparison_entities_not_in_quote"
                ),
            }
        )
    return accepted, rejected


def _compiler_quote_fallback_claims(
    evidence_chunks: list[EvidenceChunk],
    *,
    task_type: TaskType,
    comparison_aliases: list[str] | None = None,
    limit: int = 3,
) -> list[SynthesisClaim]:
    """当模型声明全部未通过逐字绑定时，生成最保守的原文 quote contracts。

    该 fallback 不改写、不总结、不补实验条件：claim_text 与 evidence_quote
    完全相同，均为 authoritative evidence 中的连续原文子串。screening 会优先
    截取比较词附近、且覆盖双方候选标识的原文窗口；只有窗口本身包含比较语义
    时才标为 comparison，避免“整段后半有比较词、前 360 字符却没有”的假合同。
    """

    markers = (
        " compared ", " than ", " whereas ", " better ", " worse ",
        " superior ", " detrimental ", " improves ", " beneficial ",
        " in contrast ", " by contrast ",
    )
    aliases = list(comparison_aliases or [])

    def _has_marker(text: str) -> bool:
        padded = f" {text.casefold()} "
        return any(marker in padded for marker in markers)

    def _comparison_window(content: str) -> tuple[str, int]:
        if task_type != "candidate_screening":
            from orion.task_evidence.scientific_text import design_result_excerpt
            quote = design_result_excerpt(content) if task_type == "design" else content[:360]
            return quote, 0
        padded = f" {content.casefold()} "
        positions: list[int] = []
        for marker in markers:
            start = 0
            while True:
                idx = padded.find(marker, start)
                if idx < 0:
                    break
                positions.append(max(0, idx - 1))
                start = idx + max(1, len(marker))
        if not positions:
            quote = content[:360]
            alias_hits = sum(_standalone_alias_in_text(alias, quote) for alias in aliases)
            return quote, alias_hits

        best_quote = content[:360]
        best_score = -1
        for idx in positions:
            window_start = max(0, idx - 170)
            if window_start + 360 > len(content):
                window_start = max(0, len(content) - 360)
            quote = content[window_start : window_start + 360].strip()
            alias_hits = sum(_standalone_alias_in_text(alias, quote) for alias in aliases)
            # 候选覆盖优先，其次保证窗口本身仍含比较词。
            score = alias_hits * 10 + int(_has_marker(quote))
            if score > best_score:
                best_quote = quote
                best_score = score
        return best_quote, max(0, best_score // 10)

    prepared: list[tuple[int, bool, EvidenceChunk, str]] = []
    for chunk in evidence_chunks:
        content = str(chunk.content or "").strip()
        if len(content) < 12 or not chunk.chunk_id:
            continue
        quote, alias_hits = _comparison_window(content)
        is_comparison = task_type == "candidate_screening" and _has_marker(quote)
        prepared.append((alias_hits, is_comparison, chunk, quote))

    # screening 优先真正覆盖双方候选且窗口内存在比较语义的原文；其余按原始顺序稳定保留。
    prepared.sort(key=lambda item: (item[0] >= 2 and item[1], item[0], item[1]), reverse=True)

    fallback: list[SynthesisClaim] = []
    for _, is_comparison, chunk, quote in prepared:
        fallback.append(
            SynthesisClaim.model_validate(
                {
                    "claim_type": "comparison" if is_comparison else "fact",
                    "text": quote,
                    "chunk_id": chunk.chunk_id,
                    "evidence_quote": quote,
                    "binding_requirement": "quote_bound",
                    "reflection_tags": ["compiler_quote_fallback"],
                }
            )
        )
        if len(fallback) >= limit:
            break
    return fallback


def _verbatim_evidence_fact_contracts(
    claims: list[SynthesisClaim],
) -> list[SynthesisClaim]:
    """事实综合的 artifact contracts 使用逐字 quote，正文仍保留模型的中文表述。"""

    normalized: list[SynthesisClaim] = []
    for claim in claims:
        if claim.claim_type == "fact" and claim.evidence_quote.strip():
            tags = list(
                dict.fromkeys([*claim.reflection_tags, "compiler_verbatim_evidence_fact"])
            )
            normalized.append(
                claim.model_copy(
                    update={
                        "claim_text": claim.evidence_quote.strip(),
                        "reflection_tags": tags,
                    }
                )
            )
        else:
            normalized.append(claim)
    return normalized


def _verbatim_inverse_fact_contracts(
    claims: list[SynthesisClaim],
) -> list[SynthesisClaim]:
    """逆向诊断的 source-bound fact contract 使用绑定原文作为 claim_text。

    ``direct_answer`` 与 hypothesis records 负责中文诊断综合；fact contracts 的职责
    是提供可独立复核的原子证据。模型常把英文 quote 截掉前置体系限定，却在中文
    claim 中保留该限定，造成 quote-bound 通过而语义校验失败。这里不补写事实，
    只把已通过逐字绑定的 quote 原样提升为 contract 文本，使声明与证据边界一致。
    """

    normalized: list[SynthesisClaim] = []
    for claim in claims:
        if claim.claim_type == "fact" and claim.evidence_quote.strip():
            tags = list(dict.fromkeys([*claim.reflection_tags, "compiler_verbatim_inverse_fact"]))
            normalized.append(
                claim.model_copy(
                    update={
                        "claim_text": claim.evidence_quote.strip(),
                        "reflection_tags": tags,
                    }
                )
            )
        else:
            normalized.append(claim)
    return normalized


def _complete_design_fact_quotes(claims: list[SynthesisClaim], source_texts: dict[str, str]) -> list[SynthesisClaim]:
    """补全已逐字绑定的事实引文句界，避免报告在数值中间截断。

    不添加新的来源、实验关联或推荐结论；重新校验扩大后的连续原文引文。
    """
    from orion.task_evidence.scientific_text import numeric_excerpt
    completed = []
    for claim in claims:
        source = source_texts.get(claim.chunk_id, "")
        quote = claim.evidence_quote.strip()
        if (claim.claim_type != "fact" or claim.evidence_binding != "quote_bound"
                or not quote or source.count(quote) != 1):
            completed.append(claim)
            continue
        start = source.index(quote)
        first = numeric_excerpt(source, start, start)
        last = numeric_excerpt(source, start + len(quote) - 1, start + len(quote) - 1)
        left = source.find(first, max(0, start - 1200))
        right = source.find(last, max(0, start + len(quote) - 1200))
        if left < 0 or right < 0:
            completed.append(claim)
            continue
        full_quote = source[left:right + len(last)].strip()
        if len(full_quote) > 1600 or quote not in full_quote or source.count(full_quote) != 1:
            completed.append(claim)
            continue
        replacement = SynthesisClaim.model_validate({
            "claim_type": "fact", "text": full_quote, "chunk_id": claim.chunk_id,
            "evidence_quote": full_quote, "binding_requirement": "quote_bound",
            "reflection_tags": [*claim.reflection_tags, "compiler_complete_design_quote"],
        })
        accepted, _ = _validated_claims([replacement], {claim.chunk_id: source})
        completed.append(accepted[0] if accepted else claim)
    return completed


def _ledger_quote_contracts(
    ledger: list[EvidenceLedgerItem],
    *,
    candidate_ids: set[str] | None = None,
    limit: int = 4,
) -> list[SynthesisClaim]:
    """从结构化候选证据账本生成最保守的逐字 fact contracts。

    仅复制已经进入 DecisionState 的 ledger excerpt，不做改写或跨证据拼接。
    recommend 时调用方传入 approved candidate ids，确保未获批候选不会通过
    fallback 混入最终 artifact；hold 时可保留少量原子证据用于审计。
    """

    contracts: list[SynthesisClaim] = []
    seen: set[tuple[str, str]] = set()
    for item in ledger:
        if candidate_ids is not None and item.candidate_id not in candidate_ids:
            continue
        quote = " ".join(str(item.excerpt or "").split()).strip()
        if len(quote) < 12 or not item.chunk_id:
            continue
        key = (item.chunk_id, quote)
        if key in seen:
            continue
        seen.add(key)
        contracts.append(
            _attach_quote_condition_scope(
                SynthesisClaim.model_validate(
                {
                    "claim_type": "fact",
                    "text": quote,
                    "chunk_id": item.chunk_id,
                    "evidence_quote": quote,
                    "binding_requirement": "quote_bound",
                    "condition_scope": item.condition_scope,
                    "experiment_id": item.experiment_id,
                    "scientific_object_id": item.scientific_object_id,
                    "reflection_tags": ["compiler_ledger_quote_fallback"],
                }
                ),
                paper_id=item.paper_id,
            )
        )
        if len(contracts) >= limit:
            break
    return contracts


def _render_direct_answer(text: str) -> list[str]:
    if not text:
        return []
    return ["## 直接回答", text, ""]


def _render_claims(claims: list[SynthesisClaim]) -> list[str]:
    if not claims:
        return []
    lines = ["## 声明绑定状态（语义蕴含仍未自动验证）"]
    for claim in claims:
        source_suffix = f" [{claim.chunk_id}]" if claim.chunk_id else ""
        lines.append(
            f"- {claim.text}{source_suffix}"
            f"（绑定级别：{claim.binding_requirement}）"
        )
        if claim.evidence_quote:
            lines.append(f"  - 原文：{claim.evidence_quote}")
    return lines


def _dedupe_claim_contracts(
    claims: list[SynthesisClaim],
) -> list[SynthesisClaim]:
    """Return a stable terminal contract population without duplicate claims."""

    selected: list[SynthesisClaim] = []
    seen: set[tuple[str, str, str]] = set()
    for claim in claims:
        key = (
            claim.claim_type,
            " ".join(claim.claim_text.split()).casefold(),
            str(claim.chunk_id or ""),
        )
        if not key[1] or key in seen:
            continue
        seen.add(key)
        selected.append(claim)
    return selected


def _computation_claim_contract(
    *,
    claim_text: str,
    claim_type: Literal["fact", "comparison", "recommendation", "diagnosis"],
    rule_id: str,
    data_used: dict[str, Any],
    evidence: list[ArtifactEvidence],
) -> SynthesisClaim:
    """Bind a deterministic gate decision to its computation and input evidence.

    A decision or diagnosis status is not a quotation from one paper.  Treating
    it as quote-bound caused the terminal narrative to outrun the contract
    population.  The contract therefore records the deterministic gate as the
    verifier and retains the exact input evidence separately for audit.
    """

    normalized_text = " ".join(str(claim_text or "").split()).strip()
    digest = hashlib.sha256(
        json.dumps(
            {
                "claim_text": normalized_text,
                "rule_id": rule_id,
                "data_used": data_used,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:12]
    refs: list[SourceRef] = []
    input_condition_scopes: list[dict[str, Any]] = []
    seen_refs: set[tuple[str, str]] = set()
    seen_scopes: set[tuple[str, str, str]] = set()
    for item in evidence:
        key = (str(item.chunk_id or ""), str(item.quote or ""))
        if not key[0] or not key[1] or key in seen_refs:
            continue
        seen_refs.add(key)
        refs.append(
            SourceRef(
                chunk_id=item.chunk_id,
                paper_id=item.paper_id,
                raw_quote=item.quote,
            )
        )
        explicit_scope = _scope_dict(item.condition_scope) or _scope_dict(
            extract_comparison_conditions(item.quote)
            or extract_condition_scope(item.quote)
        )
        scope_key = (
            str(item.chunk_id or ""),
            str(item.paper_id or ""),
            json.dumps(explicit_scope, ensure_ascii=False, sort_keys=True)
            if explicit_scope
            else "",
        )
        if explicit_scope and scope_key not in seen_scopes:
            seen_scopes.add(scope_key)
            input_condition_scopes.append(
                {
                    "chunk_id": str(item.chunk_id or ""),
                    "paper_id": str(item.paper_id or ""),
                    "scope": explicit_scope,
                }
            )
        if len(refs) >= 12 and rule_id != "model_semantic_review_with_source_verification_v1":
            break
    return SynthesisClaim(
        claim_id=f"claim-computation-{digest}",
        claim_text=normalized_text,
        claim_type=claim_type,
        provenance=ClaimProvenance(
            tool_name="synthesize_design_report",
            source_kind="computation",
        ),
        source_refs=refs,
        condition_scope=(
            {"input_evidence_scopes": input_condition_scopes}
            if input_condition_scopes
            else None
        ),
        evidence_binding="computation_bound",
        verification_status="passed",
        verification_results=[
            VerificationResult(
                verifier=("orion_semantic_review_source_verifier"
                    if rule_id == "model_semantic_review_with_source_verification_v1" else "orion_structured_task_gate"),
                tier=("medium" if rule_id == "model_semantic_review_with_source_verification_v1" else "high"),
                status="passed",
                rule_id=rule_id,
                data_used=data_used,
            )
        ],
        reflection_tags=["terminal_computation_contract_v1", rule_id],
    )


def _render_contract_only_terminal(
    *,
    task_type: TaskType,
    task_status: str,
    claims: list[SynthesisClaim],
) -> str:
    """Project the user-visible terminal solely from committed contracts.

    Structured hypotheses, evidence gaps, candidate tables and validation plans
    remain in the TaskArtifact.  They are intentionally not expanded as free
    prose because doing so creates material terminal claims that are invisible
    to ClaimContract auditing.
    """

    lines = [
        "## 受合同约束的任务结论",
        f"- 任务类型：{task_type}",
        f"- 结构化终态：{task_status}",
        "- 以下动态科学或决策内容仅来自本次 TaskArtifact 的 ClaimContracts。",
    ]
    if claims:
        lines.extend(["", *_render_claims(claims)])
    else:
        lines.extend(
            [
                "",
                "## 证据边界",
                "当前没有可进入终端正文的 ClaimContract；系统不输出材料性科学结论。",
            ]
        )
    lines.extend(
        [
            "",
            "## 审计边界",
            "候选、假设、证据缺口与验证计划保留在结构化 TaskArtifact 中；"
            "它们未被提升为本终端正文中的已成立科学主张。",
        ]
    )
    return "\n".join(lines).strip()


def _render_design_terminal_details(artifact: dict[str, Any]) -> str:
    """将已保存的设计结果直接展示给使用者，不从草稿生成新的科学结论。

    候选组成是待核对的候选资料，获准推荐与否单独标注。文献性能仍只来自
    ClaimContract；缺口和下一步来自本次决策及计划。冻结的 v1 投影保持原样。
    """
    decision = artifact.get("formulation_decision") or {}
    approved = set(decision.get("approved_candidate_ids") or [])
    recommended = artifact.get("task_status") == "recommended"
    semantic_reviews = decision.get("semantic_evidence_reviews") or []
    lines = ["## 配方筛选结果", (
        ("获得文献支持的候选：" if semantic_reviews else "通过当前筛选条件的候选：") + "、".join(sorted(approved)) + "。"
        if recommended else "已整理候选资料，当前暂缓推荐。未满足的条件列在下方。"
    )]
    ledger = {e["evidence_id"]: e for e in artifact.get("evidence_ledger") or []}
    for raw in artifact.get("candidates") or []:
        candidate = CandidateState.model_validate(raw)
        if recommended and candidate.candidate_id not in approved:
            continue
        sources = [ledger[e] for e in candidate.formulation_evidence_ids if e in ledger]
        sources = [s for s in sources if s.get("quote", "").strip()]
        if not sources:
            continue
        label = "通过筛选" if candidate.candidate_id in approved and recommended else "尚未获推荐"
        lines.extend(["", f"### 候选资料：{candidate.candidate_id}（{label}）",
                      "以下为候选记录中的组成，配制时须核对所引原文的比例基准与用量。",
                      _composition_text(candidate.model_copy(update={"composition": {
                          **candidate.composition, "model_predictions": [],
                      }}))])
        for source in sources:
            lines.append(f"- 配方原文 [{source['chunk_id']}]：{source['quote']}")
    if semantic_reviews:
        lines.extend(["", "## 候选评估"])
        for review in semantic_reviews:
            lines.append(f"### {review['candidate_id']}")
            if review.get('recommendation_reason'):
                lines.append("评估建议：" + review['recommendation_reason'])
            lines.append("任务要求与已有依据（模型语义判断，程序核验出处）：")
            for item in review['requirements']:
                lines.append(f"- {item['requirement']}：{item['reasoning']} "
                             + _citation([quote['chunk_id'] for quote in item['citations']]))
            lines.append("文献与候选的关联：" + review['experimental_association'] + " "
                         + _citation([quote['chunk_id'] for quote in review['association_citations']]))
    claims = [SynthesisClaim.model_validate(c) for c in artifact.get("claim_contracts") or []]
    # 决策状态已在开头展示。其余事实/预测依然按原有来源合同渲染。
    visible = [c for c in claims if not any(
        v.rule_id in {"structured_design_decision_v1", "model_semantic_review_with_source_verification_v1"} for v in c.verification_results
    )]
    if visible:
        lines.extend(["", "## 文献依据与已记录的计算结果"])
        for claim in visible:
            source = f" [{claim.chunk_id}]" if claim.chunk_id else ""
            lines.append(f"- {claim.text}{source}")
            if claim.evidence_quote and claim.evidence_quote != claim.text:
                lines.append(f"  - 原文：{claim.evidence_quote}")
    gaps = artifact.get("gaps") or []
    if gaps:
        lines.extend(["", "## 尚未满足的条件"])
        for gap in gaps:
            prefix = f"{gap['candidate_id']}：" if gap.get("candidate_id") else ""
            lines.append(f"- {prefix}{gap['description']}")
    plan = artifact.get("validation_plan") or []
    if plan:
        lines.extend(["", "## 下一步核查与验证"])
        lines.extend(f"- {step}" for step in plan)
    return "\n".join(lines).strip()


def _artifact_claims(claims: list[SynthesisClaim]) -> list[ArtifactClaim]:
    return [
        ArtifactClaim(
            claim_id=f"claim-{index}",
            claim_type=claim.claim_type,
            text=claim.text,
            # 一项决定可以引用多篇文献，不能用旧单来源属性丢掉后续依据。
            evidence_ids=list(dict.fromkeys(
                ref.chunk_id for ref in claim.source_refs if ref.chunk_id
            )),
        )
        for index, claim in enumerate(claims, start=1)
    ]


def _claim_binding_levels(claims: list[SynthesisClaim]) -> dict[str, str]:
    """在不改写冻结 artifact schema 的前提下记录每条声明的校验强度。"""

    return {
        f"claim-{index}": claim.binding_requirement
        for index, claim in enumerate(claims, start=1)
    }


def _artifact_evidence_from_chunks(
    chunks: list[EvidenceChunk],
) -> list[ArtifactEvidence]:
    artifacts: list[ArtifactEvidence] = []
    for chunk in chunks:
        content = chunk.content or ""
        condition_scope = chunk.condition_scope
        condition_quote = chunk.condition_quote
        # 如果后端未提供结构化条件，从 chunk 原文中确定性抽取。
        if condition_scope is None:
            condition_scope = (
                extract_comparison_conditions(content)
                or extract_condition_scope(content)
            )
        if condition_quote is None or not condition_quote.strip():
            condition_quote = infer_condition_quote(content)
        experiment_id = chunk.experiment_id
        if not experiment_id:
            experiment_id = infer_experiment_id(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                conditions=condition_scope,
            )
        artifacts.append(
            ArtifactEvidence(
                evidence_id=chunk.chunk_id,
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                quote=chunk.content,
                experiment_id=experiment_id,
                condition_scope=_scope_dict(condition_scope),
                condition_quote=condition_quote,
            )
        )
    return artifacts


def _evaluate_request_applicability(
    request: SynthesisRequest,
    *,
    extra_evidence_text: dict[str, str] | None = None,
) -> list[ApplicabilityRecord]:
    """把显式 scope/role 输入编译为统一 runtime 状态，不从报告文本猜测。"""

    evidence_text = {
        item.chunk_id: item.content
        for item in request.evidence_chunks
        if item.chunk_id
    }
    evidence_text.update(extra_evidence_text or {})
    return [
        evaluate_applicability_input(
            item,
            evidence_text_by_chunk=evidence_text,
            retrieval_allowed=request.applicability_retrieval_allowed,
        )
        for item in request.applicability_inputs
    ]


def _blocked_applicability_claim_ids(
    records: list[ApplicabilityRecord],
) -> tuple[set[str], bool]:
    blocked = {
        item.claim_id
        for item in records
        if item.claim_id
        and item.decision in {"search", "hold", "reject_claim"}
    }
    global_block = any(
        not item.claim_id
        and item.decision in {"search", "hold", "reject_claim"}
        for item in records
    )
    return blocked, global_block


def _filter_claims_by_applicability(
    claims: list[SynthesisClaim],
    records: list[ApplicabilityRecord],
) -> tuple[list[SynthesisClaim], list[dict[str, Any]], list[ApplicabilityRecord]]:
    """阻止 insufficient/conflicting claim 进入终端正文与 ClaimContract 列表。"""

    blocked_ids, global_block = _blocked_applicability_claim_ids(records)
    if not blocked_ids and not global_block:
        return (
            claims,
            [],
            [item.model_copy(update={"terminal_disposition": "emitted"}) for item in records],
        )
    accepted: list[SynthesisClaim] = []
    rejected: list[dict[str, Any]] = []
    decision_by_claim = {item.claim_id: item for item in records if item.claim_id}
    for index, claim in enumerate(claims, start=1):
        claim_id = claim.claim_id or f"claim-{index}"
        aliases = {claim_id, f"claim-{index}"}
        record = decision_by_claim.get(claim_id) or decision_by_claim.get(f"claim-{index}")
        if global_block or aliases.intersection(blocked_ids):
            rejected.append(
                {
                    "text": claim.text,
                    "chunk_id": claim.chunk_id,
                    "reason": "applicability_action_blocked",
                    "claim_id": claim_id,
                    "applicability": (
                        record.assessment.status if record else "task_global"
                    ),
                }
            )
            continue
        accepted.append(claim)
    updated_records = [
        item.model_copy(
            update={
                "terminal_disposition": (
                    "blocked"
                    if global_block or item.claim_id in blocked_ids
                    else "emitted"
                )
            }
        )
        for item in records
    ]
    return accepted, rejected, updated_records


def _gate_gaps(
    issues: list[Any],
    *,
    bind_item_as_candidate: bool = True,
) -> list[ArtifactGap]:
    return [
        ArtifactGap(
            code=issue.code,
            description=issue.description,
            candidate_id=(
                issue.item_id if bind_item_as_candidate else None
            ),
        )
        for issue in issues
    ]


def _render_ledger_evidence(
    ledger: list[EvidenceLedgerItem],
    *,
    candidate_ids: set[str] | None = None,
) -> list[str]:
    selected = [
        item
        for item in ledger
        if candidate_ids is None or item.candidate_id in candidate_ids
    ]
    if not selected:
        return []
    lines = ["## 可追溯证据"]
    for item in selected:
        scope = f"{item.candidate_id} / {item.evidence_kind}" if item.candidate_id else item.evidence_kind
        excerpt = " ".join(item.excerpt.split())
        if len(excerpt) > 400:
            excerpt = excerpt[:397].rstrip() + "..."
        lines.append(f"- {scope}：{excerpt} [{item.chunk_id}]")
    return lines


def _application_goal_followups(question: str, supplied: list[str]) -> list[str]:
    """将明确的成本与近似温度目标保留为验证计划，不据此批准或拒绝候选。"""
    plans = []
    cost = re.compile(r'\blow[- ]cost\b|\bcost[- ]effective\b|低成本|成本较低', re.I)
    if cost.search(question) and not any(re.search(r'cost|price|成本|价格', step, re.I) for step in supplied):
        plans.append('按候选配方的原料用量、采购价格和参照配方核算成本，核查原任务中的低成本目标。')
    for match in re.finditer(r'(?:around|about|approximately|约|大约)\s*'
                             r'(?P<number>[-+−]?\d+(?:\.\d+)?)\s*(?:°\s*C|℃|C\b)', question, re.I):
        target = match.group('number')
        if any(re.search(re.escape(target) + r'\s*(?:°\s*C|℃|C\b)', step, re.I) for step in supplied):
            continue
        plans.append(f'在原任务要求的温度（{match.group()}）下验证候选的性能，保留文献实际测试温度作为已有依据。')
    return plans


def _limited_design_report(request: SynthesisRequest, state: dict[str, Any]):
    """Agent 显式选择不推荐的收尾；保留原评估，不制造新的科学判定。"""
    ledger = _model_list(state.get("evidence_ledger"), EvidenceLedgerItem)
    chunks = {item.chunk_id: item.excerpt for item in ledger}
    supplied = [claim for claim in request.claims if claim.claim_type == "fact"
                and claim.evidence_binding in {"quote_bound", "condition_bound"}
                and claim.provenance.source_kind == "chunk"]
    claims, rejected = _validated_claims(supplied or _ledger_quote_contracts(ledger), chunks)
    records = _evaluate_request_applicability(request, extra_evidence_text=chunks)
    claims, blocked, records = _filter_claims_by_applicability(claims, records)
    raw = state.get("design_decision")
    decision = (raw.model_dump(mode="json") if isinstance(raw, DecisionState)
                else dict(raw) if raw else None)
    gaps = [ArtifactGap(code="agent_limited_finalization",
                        description="主 Agent 选择受限收尾；本报告不作候选推荐，也不声称未完成的评估已通过。")]
    gaps.extend(ArtifactGap(code=item["code"], description=item["description"],
                            candidate_id=item.get("candidate_id"), search_hint=item.get("search_hint", ""))
                for item in (decision or {}).get("evidence_gaps", []))
    evidence = [ArtifactEvidence(evidence_id=item.evidence_id, chunk_id=item.chunk_id,
        paper_id=item.paper_id, quote=item.excerpt, evidence_kind=item.evidence_kind,
        candidate_id=item.candidate_id, scientific_object_id=item.scientific_object_id,
        experiment_id=item.experiment_id or "", condition_scope=item.condition_scope) for item in ledger]
    artifact = DesignTaskArtifact(contract=default_task_contract("design"), task_status="held",
        candidates=[item.model_dump(mode="json") for item in _model_list(state.get("design_candidates"), CandidateState)],
        claims=_artifact_claims(claims), claim_contracts=claims, evidence_ledger=evidence,
        candidate_evaluations=[CandidateEvaluationArtifact(candidate_id=item.candidate_id,
            criteria=[criterion.model_dump(mode="json") for criterion in item.constraints])
            for item in _model_list(state.get("candidate_evaluations"), CandidateEvaluation)],
        applicability_records=records, evidence_use_decisions=compile_evidence_use_decisions(
            task_type="design", claims=claims, applicability_records=records),
        gaps=gaps, formulation_decision=decision, validation_plan=request.next_steps,
        runtime_provenance={"renderer": "synthesize_design_report", "finalization_mode": "limited",
                            "rejected_claims": [*rejected, *blocked]})
    lines = ["## 受限结论", "当前不推荐候选。以下仅保留已有证据及未解决缺口，未完成的评估不视为已验证。",
             "", "## 证据与限制", *[f"- {gap.description}" for gap in gaps],
             "", *_render_claims(claims), "", *_render_ledger_evidence(ledger)]
    if request.next_steps:
        lines.extend(["", "## 后续核查或验证计划", *[f"- {step}" for step in request.next_steps]])
    return "\n".join(lines), artifact.model_dump(mode="json")


def _design_report(
    request: SynthesisRequest,
    state: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    from orion.core.recorded_predictions import recorded_predictions
    if request.finalization_mode == "limited":
        return _limited_design_report(request, state)
    decision_raw = state.get("design_decision")
    if not decision_raw:
        return (
            "错误：缺少结构化 design_decision；报告工具不能替代 decision gate 做候选判断。",
            {"status": "error", "reason": "missing_design_decision", "recoverable": True},
        )
    decision = (
        decision_raw
        if isinstance(decision_raw, DecisionState)
        else DecisionState.model_validate(decision_raw)
    )
    if decision.semantic_evidence_reviews:
        from orion.core.semantic_evidence_review import validate_review
        from orion.core.recorded_predictions import recorded_predictions
        computations = recorded_predictions(state)
        sources = state.get("design_semantic_sources") or {}
        for evidence_review in decision.semantic_evidence_reviews:
            validate_review(evidence_review, evidence_review.candidate_id, sources, computations)
        supported_ids = {item.candidate_id for item in decision.semantic_evidence_reviews if item.recommendation_supported}
        if set(decision.approved_candidate_ids) != supported_ids:
            raise ValueError("推荐决定与语义审查不一致")
    if decision.status == "search":
        return (
            "错误：当前评估尚不支持推荐。请自行决定继续核验，或显式选择 finalization_mode='limited' 收尾。",
            {
                "status": "error", "reason": "recommendation_not_authorized", "recoverable": True,
                "decision": decision.model_dump(mode="json"),
            },
        )

    candidates = _model_list(state.get("design_candidates"), CandidateState)
    ledger = _model_list(state.get("evidence_ledger"), EvidenceLedgerItem)
    evaluations = _model_list(
        state.get("candidate_evaluations"),
        CandidateEvaluation,
    )
    contract_raw = state.get("task_contract")
    contract = (
        contract_raw
        if isinstance(contract_raw, TaskContract)
        else TaskContract.model_validate(contract_raw)
        if contract_raw
        else None
    )
    candidate_by_id = {item.candidate_id: item for item in candidates}
    evaluation_by_id = {item.candidate_id: item for item in evaluations}
    ledger_by_id = {item.evidence_id: item for item in ledger}
    applicability_records = _evaluate_request_applicability(
        request,
        extra_evidence_text={item.chunk_id: item.excerpt for item in ledger},
    )
    decision_blockers = [
        item
        for item in applicability_records
        if item.decision in {"search", "hold", "reject_claim"}
        and item.claim_id in {"", "formulation_decision"}
    ]
    if decision.status == "recommend" and decision_blockers:
        return (
            "错误：条件适用性检查不支持当前推荐。请根据缺口继续判断，或显式选择受限报告。",
            {"status": "error", "reason": "applicability_blocks_recommendation", "recoverable": True,
             "decision": decision.model_dump(mode="json"),
             "applicability_records": [item.model_dump(mode="json") for item in applicability_records]},
        )
    authorized_candidate_ids = (
        set(decision.approved_candidate_ids)
        if decision.status == "recommend"
        else None
    )
    ledger_chunks = {
        item.chunk_id: item.excerpt
        for item in ledger
        if authorized_candidate_ids is None
        or item.candidate_id in authorized_candidate_ids
    }
    claim_chunks = ledger_chunks
    if request.task_type == "inverse_design":
        # inverse 的诊断证据账本独立于可选 formulation 子合同：根因/反证 claim
        # 可以引用本轮 request.evidence_chunks，而候选配方证据仍来自结构化 ledger。
        # 普通 design 继续只允许 candidate ledger，不能借此绕过候选准入。
        claim_chunks = {
            **{
                item.chunk_id: item.content
                for item in request.evidence_chunks
                if item.chunk_id and item.content
            },
            **ledger_chunks,
        }
    accepted_claims, rejected_claims = _validated_claims(
        request.claims,
        claim_chunks,
        {item.chunk_id: item for item in request.evidence_chunks},
    )
    if request.task_type == "inverse_design":
        accepted_claims = _verbatim_inverse_fact_contracts(accepted_claims)
    elif request.task_type == "design":
        accepted_claims = _complete_design_fact_quotes(accepted_claims, claim_chunks)
        # Agent 有时只摘性能段的开头，报告因此丢掉筛选已使用的实际数值。
        # 从获准候选、已满足条件所用的来源补入完整结果句，再走同一引文校验。
        if authorized_candidate_ids and accepted_claims:
            from orion.task_evidence.scientific_text import design_result_excerpt
            used_ids = {eid for evaluation in evaluations
                        if evaluation.candidate_id in authorized_candidate_ids
                        for criterion in evaluation.constraints if criterion.status == "satisfied"
                        for eid in criterion.evidence_ids}
            supplements = []
            for item in ledger:
                if item.candidate_id not in authorized_candidate_ids or item.evidence_id not in used_ids:
                    continue
                quote = design_result_excerpt(item.excerpt)
                if (not re.search(r"\d+(?:\.\d+)?\s*(?:%|mAh|mS\b|cycles\b)", quote, re.I)
                        or not re.search(r"capacity|retention|cycl|conductivity|coulombic|discharge", quote, re.I)):
                    continue
                if any(claim.chunk_id == item.chunk_id and quote in claim.evidence_quote
                       for claim in [*accepted_claims, *supplements]):
                    continue
                supplements.append(SynthesisClaim.model_validate({
                    "claim_type": "fact", "text": quote, "chunk_id": item.chunk_id,
                    "evidence_quote": quote, "binding_requirement": "quote_bound",
                    "condition_scope": item.condition_scope,
                    "scientific_object_id": item.scientific_object_id,
                    "experiment_id": item.experiment_id,
                    "reflection_tags": ["compiler_screened_result_quote"],
                }))
            verified, _ = _validated_claims(supplements, claim_chunks)
            accepted_claims.extend(verified)

    lines: list[str] = []
    if decision.status == "recommend":
        missing = [
            candidate_id
            for candidate_id in decision.approved_candidate_ids
            if candidate_id not in candidate_by_id
        ]
        if missing:
            return (
                "错误：approved candidate 不在结构化 candidate 状态中：" + ", ".join(missing),
                {
                    "status": "error",
                    "reason": "approved_candidate_missing",
                    "candidate_ids": missing,
                },
            )
        approved = set(decision.approved_candidate_ids)
        lines.extend(
            [
                "## 推荐结论",
                ("以下候选得到当前任务文献支持：" if decision.semantic_evidence_reviews
                 else "仅以下候选通过全部结构化硬约束：")
                + "、".join(decision.approved_candidate_ids)
                + "。",
                "",
                "## 获批候选",
            ]
        )
        constraint_by_id = (
            {item.constraint_id: item for item in contract.hard_constraints}
            if contract
            else {}
        )
        for candidate_id in decision.approved_candidate_ids:
            candidate = candidate_by_id[candidate_id]
            lines.extend([f"### {candidate_id}", _composition_text(candidate)])
            evaluation = evaluation_by_id.get(candidate_id)
            if decision.semantic_evidence_reviews:
                evidence_review = next(item for item in decision.semantic_evidence_reviews if item.candidate_id == candidate_id)
                lines.append("任务要求与原文依据（模型语义判断）：")
                for item in evidence_review.requirements:
                    lines.append(f"- {item.requirement}：{item.reasoning} " + _citation([quote.chunk_id for quote in item.citations]))
                lines.append("实验关联：" + evidence_review.experimental_association
                             + " " + _citation([quote.chunk_id for quote in evidence_review.association_citations]))
            elif evaluation:
                lines.append("硬约束：")
                for result in evaluation.constraints:
                    description = constraint_by_id.get(result.constraint_id)
                    label = description.description if description else result.constraint_id
                    evidence_chunks = [
                        ledger_by_id[evidence_id].chunk_id
                        for evidence_id in result.evidence_ids
                        if evidence_id in ledger_by_id
                    ]
                    lines.append(
                        f"- {label}：{result.status} {_citation(evidence_chunks)}".rstrip()
                    )
            lines.append("")
        lines.extend(_render_ledger_evidence(ledger, candidate_ids=approved))
    else:
        lines.extend(
            [
                "## 证据结论",
                ("当前原文证据不足以推荐候选，具体缺口如下。" if decision.semantic_evidence_reviews
                 else "当前状态：hold。无候选满足全部结构化硬约束；approved candidates 为空，当前不能给出候选推荐或排序。"),
                "",
                "## 证据缺口",
            ]
        )
        seen_gaps: set[tuple[str | None, str, str]] = set()
        for gap in decision.evidence_gaps:
            gap_key = (gap.candidate_id, gap.code, gap.description)
            if gap_key in seen_gaps:
                continue
            seen_gaps.add(gap_key)
            scope = gap.candidate_id or "全部候选"
            lines.append(f"- {scope}：{gap.description}（{gap.code}）")
        lines.extend(["", *_render_ledger_evidence(ledger)])

    rendered_claims = _render_claims(accepted_claims)
    if rendered_claims:
        lines.extend(["", *rendered_claims])

    next_steps = list(
        dict.fromkeys(
            [
                *request.next_steps,
                *_application_goal_followups(
                    _latest_user_question_from_state(state) or request.question,
                    request.next_steps),
                *[
                    gap.search_hint
                    for gap in decision.evidence_gaps
                    if gap.search_hint
                ],
            ]
        )
    )
    if next_steps:
        lines.extend(["", "## 下一步验证"])
        lines.extend(f"- {item}" for item in next_steps)

    artifact_evidence = [
        ArtifactEvidence(
            evidence_id=item.evidence_id,
            chunk_id=item.chunk_id,
            paper_id=item.paper_id,
            quote=item.excerpt,
            evidence_kind=item.evidence_kind,
            candidate_id=item.candidate_id,
            scientific_object_id=item.scientific_object_id,
            experiment_id=item.experiment_id or "",
            condition_scope=item.condition_scope,
        )
        for item in ledger
    ]
    artifact_gaps = [
        ArtifactGap(
            code=gap.code,
            description=gap.description,
            candidate_id=gap.candidate_id,
            search_hint=gap.search_hint,
        )
        for gap in decision.evidence_gaps
    ]
    artifact_evaluations = [
        CandidateEvaluationArtifact(
            candidate_id=item.candidate_id,
            criteria=[
                criterion.model_dump(mode="json")
                for criterion in item.constraints
            ],
        )
        for item in evaluations
    ]
    contract_claims = accepted_claims
    if request.task_type == "design" and not contract_claims:
        contract_claims = _ledger_quote_contracts(
            ledger,
            candidate_ids=(
                set(decision.approved_candidate_ids)
                if decision.status == "recommend"
                else None
            ),
        )
    if request.task_type == "inverse_design":
        merged_evidence: list[ArtifactEvidence] = []
        seen_evidence: set[str] = set()
        for item in [
            *_artifact_evidence_from_chunks(request.evidence_chunks),
            *artifact_evidence,
        ]:
            key = str(item.evidence_id or item.chunk_id or "")
            if not key or key in seen_evidence:
                continue
            seen_evidence.add(key)
            merged_evidence.append(item)
        artifact_evidence = merged_evidence

    if request.task_type == "design":
        approved_ids = list(decision.approved_candidate_ids)
        decision_text = (
            "The structured hard-constraint gate recommends only candidate(s): "
            + ", ".join(approved_ids)
            + "."
            if decision.status == "recommend"
            else (
                "The structured hard-constraint gate holds the design decision; "
                "no candidate is currently approved."
            )
        )
        if decision.semantic_evidence_reviews:
            decision_text = ("Source-verified model review recommends candidate(s): " + ", ".join(approved_ids)
                             if decision.status == "recommend" else "Source-verified model review does not support a recommendation.")
        decision_contract = _computation_claim_contract(
            claim_text=decision_text,
            claim_type="recommendation",
            rule_id=("model_semantic_review_with_source_verification_v1" if decision.semantic_evidence_reviews
                     else "structured_design_decision_v1"),
            data_used={
                "decision": decision.model_dump(mode="json"),
                "candidate_evaluations": [
                    item.model_dump(mode="json") for item in evaluations
                ],
                "recorded_tool_predictions": recorded_predictions(state),
            },
            evidence=artifact_evidence,
        )
        model_prediction_contracts: list[SynthesisClaim] = []
        if decision.status == "recommend":
            for candidate_id in approved_ids:
                candidate = candidate_by_id.get(candidate_id)
                if candidate is None:
                    continue
                for prediction in candidate.composition.get("model_predictions") or []:
                    if not isinstance(prediction, dict) or prediction.get("status") != "predicted":
                        continue
                    value = prediction.get("numeric_value")
                    temperature = prediction.get("temperature_c")
                    if not isinstance(value, (int, float)):
                        continue
                    temperature_text = (
                        f" at {float(temperature):g} °C"
                        if isinstance(temperature, (int, float))
                        else ""
                    )
                    model_id = str(prediction.get("model_id") or "scientific model")
                    model_prediction_contracts.append(
                        _computation_claim_contract(
                            claim_text=(
                                f"候选 {candidate_id} 的 {model_id} 模型估计离子电导率为 "
                                f"{float(value):.3g} mS/cm{temperature_text}；"
                                "该数值是模型估计，属于非实验观测。"
                            ),
                            claim_type="fact",
                            rule_id="na_conductivity_model_prediction_v1",
                            data_used={
                                "candidate_id": candidate_id,
                                "prediction": prediction,
                                "non_experimental": True,
                            },
                            evidence=[],
                        )
                    )
        contract_claims = _dedupe_claim_contracts(
            [decision_contract, *model_prediction_contracts, *contract_claims]
        )

    from orion.core.recorded_predictions import recorded_predictions, prediction_text
    recorded_prediction_contracts = [
        _computation_claim_contract(claim_text=prediction_text(record), claim_type="fact",
            rule_id="recorded_model_prediction_v1", data_used={**record, "non_experimental": True}, evidence=[])
        for record in recorded_predictions(state)
    ]
    contract_claims = _dedupe_claim_contracts([*contract_claims, *recorded_prediction_contracts])

    contract_claims, applicability_rejected, applicability_records = (
        _filter_claims_by_applicability(contract_claims, applicability_records)
    )
    rejected_claims.extend(applicability_rejected)

    common = {
        "contract": default_task_contract(request.task_type),
        "claims": _artifact_claims(contract_claims),
        "claim_contracts": contract_claims,
        "evidence_ledger": artifact_evidence,
        "applicability_records": applicability_records,
        "evidence_use_decisions": compile_evidence_use_decisions(
            task_type=request.task_type,
            claims=contract_claims,
            applicability_records=applicability_records,
        ),
        "gaps": artifact_gaps,
        "candidate_evaluations": artifact_evaluations,
        "formulation_decision": decision.model_dump(mode="json"),
        "validation_plan": next_steps,
        "runtime_provenance": {
            "renderer": "synthesize_design_report",
            "semantic_review_runs": state.get("design_semantic_results") if decision.semantic_evidence_reviews else None,
            "rejected_claims": rejected_claims,
            "claim_validation_scope": CLAIM_VALIDATION_SCOPE,
            "claim_binding_levels": _claim_binding_levels(contract_claims),
        },
    }
    if request.task_type == "design":
        task_artifact = DesignTaskArtifact(
            task_status=(
                "recommended" if decision.status == "recommend" else "held"
            ),
            candidates=[
                item.model_dump(mode="json")
                for item in candidates
            ],
            **common,
        )
    else:
        hypothesis_assessment = evaluate_inverse_hypotheses(
            request.hypotheses,
            confounders=request.confounders,
            tests=request.discriminating_tests,
            available_evidence_ids={
                item.evidence_id for item in artifact_evidence
            },
        )
        common["gaps"] = [
            *artifact_gaps,
            *_gate_gaps(
                hypothesis_assessment.issues,
                bind_item_as_candidate=False,
            ),
        ]
        supported_ids = list(hypothesis_assessment.supported_hypothesis_ids)
        if hypothesis_assessment.status == "diagnosis_supported":
            diagnosis_text = (
                "The structured hypothesis gate supports hypothesis ID(s) "
                + ", ".join(supported_ids)
                + " relative to the recorded alternatives under the current evidence and controls."
            )
        elif hypothesis_assessment.status == "diagnosis_hypothesis_only":
            diagnosis_text = (
                "The structured hypothesis gate retains competing hypotheses as unverified; "
                "no root cause is established."
            )
        else:
            diagnosis_text = (
                "The structured hypothesis gate finds the available evidence insufficient "
                "to establish a root cause."
            )
        diagnosis_contract = _computation_claim_contract(
            claim_text=diagnosis_text,
            claim_type="diagnosis",
            rule_id="structured_inverse_hypothesis_gate_v1",
            data_used={
                "assessment": hypothesis_assessment.model_dump(mode="json"),
                "hypothesis_ids": [
                    item.hypothesis_id for item in request.hypotheses
                ],
                "test_ids": [
                    item.test_id for item in request.discriminating_tests
                ],
            },
            evidence=artifact_evidence,
        )
        inverse_contracts = _dedupe_claim_contracts(
            [diagnosis_contract, *contract_claims]
        )
        common["claim_contracts"] = inverse_contracts
        common["claims"] = _artifact_claims(inverse_contracts)
        common["evidence_use_decisions"] = compile_evidence_use_decisions(
            task_type=request.task_type,
            claims=inverse_contracts,
            applicability_records=applicability_records,
        )
        common["runtime_provenance"]["claim_binding_levels"] = (
            _claim_binding_levels(inverse_contracts)
        )
        common["runtime_provenance"]["terminal_projection"] = (
            "claim_contract_only_v1"
        )
        diagnosis_lines = [
            "## 诊断结论",
            {
                "diagnosis_supported": (
                    "结构化证据当前支持一个主假设，并对竞争假设记录了反证；"
                    "仍须遵守给出的区分实验和受控变量。"
                ),
                "diagnosis_hypothesis_only": (
                    "已形成可区分的竞争假设和验证计划，但尚不能把任一假设表述为已证实根因。"
                ),
                "insufficient_evidence": (
                    "竞争假设、证据账本、混杂因素或区分实验仍不完整；"
                    "以下配方结果仅是可选子决策，不能证明故障根因。"
                ),
            }[hypothesis_assessment.status],
        ]
        for hypothesis in request.hypotheses:
            diagnosis_lines.append(
                f"- {hypothesis.hypothesis_id}：{hypothesis.text}"
            )
        if request.discriminating_tests:
            diagnosis_lines.extend(["", "## 区分实验"])
            diagnosis_lines.extend(
                f"- {item.test_id}：{item.description}"
                for item in request.discriminating_tests
            )
        lines = [*diagnosis_lines, "", *lines]
        task_artifact = InverseDesignTaskArtifact(
            task_status=hypothesis_assessment.status,
            symptom_boundary=request.symptom_boundary,
            hypotheses=request.hypotheses,
            confounders=request.confounders,
            discriminating_tests=request.discriminating_tests,
            conditional_adjustments=request.conditional_adjustments,
            hypothesis_assessment=hypothesis_assessment,
            **common,
        )
    artifact_payload = task_artifact.model_dump(mode="json")
    if request.task_type == "design":
        artifact_payload.setdefault("runtime_provenance", {})[
            "terminal_projection"
        ] = "structured_design_details_v2"
        return _render_design_terminal_details(artifact_payload), artifact_payload
    artifact_payload.setdefault("runtime_provenance", {})[
        "terminal_projection"
    ] = "claim_contract_only_v1"
    return (
        _render_contract_only_terminal(
            task_type=request.task_type,
            task_status=str(artifact_payload.get("task_status") or ""),
            claims=[
                SynthesisClaim.model_validate(item)
                for item in artifact_payload.get("claim_contracts") or []
            ],
        ),
        artifact_payload,
    )


def _inverse_report(
    request: SynthesisRequest,
    state: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """编译逆向诊断；完整配方 ``DecisionState`` 是可选子合同。"""

    if state.get("design_decision"):
        return _design_report(request, state)

    chunks = {
        item.chunk_id: item.content
        for item in request.evidence_chunks
        if item.chunk_id and item.content
    }
    accepted, rejected = (
        _validated_claims(
            request.claims,
            chunks,
            {item.chunk_id: item for item in request.evidence_chunks},
        )
        if chunks
        else ([], [])
    )
    accepted = _verbatim_inverse_fact_contracts(accepted)
    evidence_ledger = _artifact_evidence_from_chunks(request.evidence_chunks)
    assessment = evaluate_inverse_hypotheses(
        request.hypotheses,
        confounders=request.confounders,
        tests=request.discriminating_tests,
        available_evidence_ids={
            item.evidence_id for item in evidence_ledger
        },
    )
    applicability_records = _evaluate_request_applicability(request)
    diagnosis_blocked = any(
        item.decision in {"search", "hold", "reject_claim"}
        and item.claim_id in {"", "diagnosis_decision"}
        for item in applicability_records
    )
    if assessment.status == "diagnosis_supported" and diagnosis_blocked:
        assessment = assessment.model_copy(
            update={"status": "insufficient_evidence", "supported_hypothesis_ids": []}
        )
    supported_ids = list(assessment.supported_hypothesis_ids)
    if assessment.status == "diagnosis_supported":
        diagnosis_text = (
            "The structured hypothesis gate supports hypothesis ID(s) "
            + ", ".join(supported_ids)
            + " relative to the recorded alternatives under the current evidence and controls."
        )
    elif assessment.status == "diagnosis_hypothesis_only":
        diagnosis_text = (
            "The structured hypothesis gate retains competing hypotheses as unverified; "
            "no root cause is established."
        )
    else:
        diagnosis_text = (
            "The structured hypothesis gate finds the available evidence insufficient "
            "to establish a root cause."
        )
    diagnosis_contract = _computation_claim_contract(
        claim_text=diagnosis_text,
        claim_type="diagnosis",
        rule_id="structured_inverse_hypothesis_gate_v1",
        data_used={
            "assessment": assessment.model_dump(mode="json"),
            "hypothesis_ids": [item.hypothesis_id for item in request.hypotheses],
            "test_ids": [item.test_id for item in request.discriminating_tests],
        },
        evidence=evidence_ledger,
    )
    terminal_contracts = _dedupe_claim_contracts(
        [diagnosis_contract, *accepted]
    )
    terminal_contracts, applicability_rejected, applicability_records = (
        _filter_claims_by_applicability(terminal_contracts, applicability_records)
    )
    rejected.extend(applicability_rejected)
    gaps = [
        *[
            ArtifactGap(
                code=str(item.get("reason") or "claim_rejected"),
                description=f"声明未通过引文绑定校验：{item.get('text', '')}",
            )
            for item in rejected
        ],
        *_gate_gaps(
            assessment.issues,
            bind_item_as_candidate=False,
        ),
    ]
    lines = [
        "## 诊断结论",
        {
            "diagnosis_supported": (
                "结构化证据支持一个主假设，并对竞争假设记录了反证。"
            ),
            "diagnosis_hypothesis_only": (
                "已形成可区分的竞争假设和验证计划，当前仍属于待验证诊断。"
            ),
            "insufficient_evidence": (
                "竞争假设、证据账本、混杂因素或区分实验仍不完整。"
            ),
        }[assessment.status],
    ]
    lines.extend(
        f"- {item.hypothesis_id}：{item.text}"
        for item in request.hypotheses
    )
    if request.discriminating_tests:
        lines.extend(["", "## 区分实验"])
        lines.extend(
            f"- {item.test_id}：{item.description}"
            for item in request.discriminating_tests
        )
    rendered_claims = _render_claims(accepted)
    if rendered_claims:
        lines.extend(["", *rendered_claims])

    validation_plan = list(
        dict.fromkeys(
            [
                *request.next_steps,
                *[item.description for item in request.discriminating_tests],
            ]
        )
    )
    artifact = InverseDesignTaskArtifact(
        contract=default_task_contract("inverse_design"),
        task_status=assessment.status,
        claims=_artifact_claims(terminal_contracts),
        claim_contracts=terminal_contracts,
        evidence_ledger=evidence_ledger,
        applicability_records=applicability_records,
        evidence_use_decisions=compile_evidence_use_decisions(
            task_type=request.task_type,
            claims=terminal_contracts,
            applicability_records=applicability_records,
        ),
        gaps=gaps,
        formulation_decision=None,
        validation_plan=validation_plan,
        runtime_provenance={
            "renderer": "synthesize_design_report",
            "rejected_claims": rejected,
            "claim_validation_scope": CLAIM_VALIDATION_SCOPE,
            "claim_binding_levels": _claim_binding_levels(terminal_contracts),
            "formulation_subtask": "not_requested",
            "terminal_projection": "claim_contract_only_v1",
        },
        symptom_boundary=request.symptom_boundary,
        hypotheses=request.hypotheses,
        confounders=request.confounders,
        discriminating_tests=request.discriminating_tests,
        conditional_adjustments=request.conditional_adjustments,
        hypothesis_assessment=assessment,
    )
    return (
        _render_contract_only_terminal(
            task_type="inverse_design",
            task_status=assessment.status,
            claims=terminal_contracts,
        ),
        artifact.model_dump(mode="json"),
    )


def _model_screening_validation_plan(
    request: SynthesisRequest, assessment: PredictionComparisonAssessment,
) -> tuple[list[str], str]:
    """模型初筛缺省的实验复测建议；只陈述待做事项，不伪装成研究结果。"""
    submitted = list(dict.fromkeys(step.strip() for step in request.next_steps if step.strip()))
    if submitted:
        return submitted, "agent_submitted"
    if assessment.status != "model_assessed":
        return [], "none"
    candidates = "、".join(assessment.priority_candidate_ids)
    temperatures = list(assessment.request.temperatures_c)
    if (assessment.request.reference_temperature_c is not None
            and assessment.request.reference_temperature_c not in temperatures):
        temperatures.append(assessment.request.reference_temperature_c)
    condition = "、".join(f"{temperature:g}" for temperature in temperatures)
    plan = [
        f"以报告所列组成为依据，对{candidates}等已有配方在{condition} °C进行重复电导率测量，"
        "统一配制、温度平衡和测量条件，报告测量误差并与模型排序核对。",
        "电导率预测不能代替实际电池性能。若需评价低温电池适用性，应在相同电极、"
        "电芯构型、倍率和电压窗口下开展低温容量、循环保持率及阻抗的对照测试。",
    ]
    return plan, "compiler_scope_template_unexecuted"


def _render_model_screening_terminal(artifact: CandidateScreeningArtifact) -> str:
    """展示计算、输入及未执行的计划，不把计划写成已验证事实。"""
    assessment = artifact.model_comparison
    assert assessment is not None
    lines = ["## 电解液配方筛选", "", "### 候选组成"]
    for item in artifact.normalized_candidates:
        candidate = item.get("prediction_input") or {}
        composition = item.get("composition_input") or candidate.get("composition_inputs") or {}
        descriptions = []
        for key, label in [("molar_ratios", "全组分摩尔比例"), ("component_mmol", "组分用量（mmol）")]:
            values = composition.get(key) or {}
            if values:
                descriptions.append(label + "为 " + "，".join(f"{name}={value:.6g}" for name, value in values.items()))
        lines.append(f"- {item['candidate_id']}：" + ("；".join(descriptions) or "组成未记录"))
    lines.extend(["", "### 电导率预测", f"模型：{assessment.model_id or '未完成'}。数值单位为 mS/cm。", ""])
    temperatures = list(assessment.request.temperatures_c)
    reference = assessment.request.reference_temperature_c
    if reference is not None and reference not in temperatures:
        temperatures.append(reference)
    values = {(r['candidate_id'], r['prediction']['temperature_c']): r['prediction']['numeric_value']
              for r in assessment.records}
    lines += ["| 候选 | " + " | ".join(f"{t:g} °C" for t in temperatures) + " |",
              "| --- | " + " | ".join("---" for _ in temperatures) + " |"]
    for cid in assessment.request.candidate_ids:
        row = [f"{values[cid, t]:.4f}" if (cid, t) in values else "未计算" for t in temperatures]
        lines.append("| " + cid + " | " + " | ".join(row) + " |")
    lines.extend(["", "### 初筛结果"])
    if assessment.status == "model_assessed":
        lines.append(comparison_text(assessment))
    else:
        lines.append("所需计算尚未齐全或输入不一致，当前不生成候选优先级。")
        lines.extend("- " + issue for issue in assessment.issues)
    literature = [c for c in artifact.claim_contracts if c.provenance.source_kind != "computation"]
    if literature:
        lines.extend(["", "### 文献依据", *_render_claims(literature)])
    if artifact.validation_plan:
        plan_origin = artifact.runtime_provenance.get("validation_plan_source")
        plan_title = ("### 后续实验建议（尚未执行；依据模型筛选范围自动列出的通用复测项目）"
                      if plan_origin == "compiler_scope_template_unexecuted"
                      else "### 后续实验建议（尚未执行）")
        lines.extend(["", plan_title])
        lines.extend("- " + step for step in artifact.validation_plan)
    return "\n".join(lines)


def _model_screening_report(request: SynthesisRequest, state: dict[str, Any]):
    """模型初筛单独绑定已执行预测；原有实验工况比较规则保持不变。"""
    from orion.core.recorded_predictions import recorded_predictions, prediction_text
    assert request.model_comparison is not None
    assessment = assess_prediction_comparison(recorded_predictions(state), request.model_comparison)
    supplied_ids = [c.candidate_id or c.scheme_id for c in request.candidates]
    if supplied_ids and set(supplied_ids) != set(request.model_comparison.candidate_ids):
        return ("错误：模型比较的候选与本次候选清单不一致。",
                {"status": "error", "reason": "model_comparison_candidate_mismatch"})
    chunks = {c.chunk_id: c.content for c in request.evidence_chunks if c.chunk_id and c.content}
    evidence_records = {c.chunk_id: c for c in request.evidence_chunks}
    # 此路径中的文献承担背景事实。实验比较仍须走 comparison_records 完整工况路径。
    facts = [c for c in request.claims if c.claim_type == "fact"]
    accepted, rejected = _validated_claims(facts, chunks, evidence_records)
    applicability = _evaluate_request_applicability(request)
    accepted, extra_rejected, applicability = _filter_claims_by_applicability(accepted, applicability)
    rejected.extend(extra_rejected)
    contracts = list(accepted)
    for record in assessment.records:
        contract = _computation_claim_contract(
            claim_text=prediction_text(record), claim_type="fact",
            rule_id="recorded_model_prediction_v1", data_used={"prediction_record": record}, evidence=[],
        )
        contract.provenance = ClaimProvenance(
            tool_name=record["tool_name"], tool_call_id=record["tool_call_id"], source_kind="computation",
        )
        contracts.append(contract)
    if assessment.status == "model_assessed":
        contracts.append(_computation_claim_contract(
            claim_text=comparison_text(assessment), claim_type="comparison",
            rule_id="recorded_prediction_comparison_v1",
            data_used={"assessment": assessment.model_dump(mode="json")}, evidence=[],
        ))
    contracts = _dedupe_claim_contracts(contracts)
    validation_plan, plan_origin = _model_screening_validation_plan(request, assessment)
    normalized = []
    for cid in request.model_comparison.candidate_ids:
        record = next((r for r in assessment.records if r["candidate_id"] == cid), None)
        normalized.append({"candidate_id": cid, "prediction_input": record["candidate_input"] if record else None,
                           "composition_input": record["composition_input"] if record else None})
    artifact = CandidateScreeningArtifact(
        contract=default_task_contract("candidate_screening"),
        task_status="model_assessed" if assessment.status == "model_assessed" else "insufficient_evidence",
        normalized_candidates=normalized,
        comparison_axes=[request.model_comparison.objective], model_comparison=assessment,
        claims=_artifact_claims(contracts), claim_contracts=contracts,
        evidence_ledger=_artifact_evidence_from_chunks(request.evidence_chunks),
        applicability_records=applicability,
        evidence_use_decisions=compile_evidence_use_decisions(
            task_type="candidate_screening", claims=contracts, applicability_records=applicability,
        ),
        gaps=[ArtifactGap(code="incomplete_model_comparison", description=i) for i in assessment.issues]
             + [ArtifactGap(code=str(r.get("reason") or "claim_rejected"),
                            description=str(r.get("text") or "文献声明未通过校验")) for r in rejected],
        validation_plan=validation_plan,
        runtime_provenance={"renderer": "synthesize_design_report", "comparison_basis": "recorded_model_predictions",
                            "validation_plan_source": plan_origin,
                            "ignored_nonfact_literature_claims": sum(c.claim_type != "fact" for c in request.claims),
                            "rejected_claims": rejected, "claim_binding_levels": _claim_binding_levels(contracts)},
    )
    return _render_model_screening_terminal(artifact), artifact.model_dump(mode="json")


def _evidence_or_screening_report(
    request: SynthesisRequest,
) -> tuple[str, dict[str, Any]]:
    chunks = {
        item.chunk_id: item.content
        for item in request.evidence_chunks
        if item.chunk_id and item.content
    }
    if not chunks:
        return (
            "错误：未提供任何 evidence_chunks，无法生成带来源绑定的报告。",
            {"status": "error", "reason": "missing_evidence"},
        )
    evidence_records = {item.chunk_id: item for item in request.evidence_chunks}
    accepted, rejected = _validated_claims(
        request.claims,
        chunks,
        evidence_records,
    )
    if request.task_type == "candidate_screening":
        accepted, comparison_rejected = _filter_screening_comparison_claims(
            accepted, request
        )
        rejected.extend(comparison_rejected)
    # Evidence/screening agents sometimes retrieve authoritative chunks but
    # omit the optional ``claims`` argument entirely.  That must not produce a
    # formally successful artifact with zero ClaimContracts.  The compiler's
    # fallback is deliberately conservative: it promotes only exact verbatim
    # excerpts, then sends them through the same validation and screening
    # comparability gates as model-authored claims.
    if (
        not accepted
        and request.task_type in {"evidence_synthesis", "candidate_screening"}
    ):
        fallback_claims = _compiler_quote_fallback_claims(
            request.evidence_chunks,
            task_type=request.task_type,
            comparison_aliases=(
                _meaningful_candidate_aliases(request)
                if request.task_type == "candidate_screening"
                else None
            ),
        )
        fallback_accepted, fallback_rejected = _validated_claims(
            fallback_claims,
            chunks,
            evidence_records,
        )
        if request.task_type == "candidate_screening":
            fallback_accepted, fallback_comparison_rejected = (
                _filter_screening_comparison_claims(fallback_accepted, request)
            )
            fallback_rejected.extend(fallback_comparison_rejected)
        accepted.extend(fallback_accepted)
        rejected.extend(fallback_rejected)
    evidence_ledger = _artifact_evidence_from_chunks(request.evidence_chunks)
    comparability_assessment = None
    if request.task_type == "candidate_screening":
        expected_candidate_ids = [
            item.candidate_id or item.scheme_id
            for item in request.candidates
            if item.candidate_id or item.scheme_id
        ]
        comparability_assessment = evaluate_candidate_comparability(
            request.comparison_records,
            expected_candidate_ids=expected_candidate_ids or None,
            available_evidence_ids={item.evidence_id for item in evidence_ledger},
        )
        if comparability_assessment.status != "comparable":
            allowed_claims: list[SynthesisClaim] = []
            for claim in accepted:
                if claim.claim_type == "ranking":
                    rejected.append(
                        {
                            "text": claim.text,
                            "chunk_id": claim.chunk_id,
                            "reason": "ranking_without_comparability",
                        }
                    )
                else:
                    allowed_claims.append(claim)
            accepted = allowed_claims

    applicability_records = _evaluate_request_applicability(request)
    accepted, applicability_rejected, applicability_records = (
        _filter_claims_by_applicability(accepted, applicability_records)
    )
    rejected.extend(applicability_rejected)

    lines = ["## 证据结论"]
    if accepted:
        lines.append(
            "以下声明均通过 chunk_id、逐字引文和带单位数值一致性校验；"
            "标为 condition_bound 的声明还通过同一完整实验条件记录匹配。"
            "系统仍未自动验证语义蕴含。"
        )
        lines.extend(["", *_render_claims(accepted)])
    else:
        lines.append(
            "当前没有通过引文绑定校验的综合声明；以下仅列出可审计原文证据，不作额外推断。"
        )
        lines.extend(["", "## 原文证据"])
        for chunk in request.evidence_chunks[:12]:
            excerpt = " ".join(chunk.content.split())[:300]
            lines.append(f"- {excerpt} [{chunk.chunk_id}]")
    if request.next_steps:
        lines.extend(["", "## 下一步验证"])
        lines.extend(f"- {item}" for item in dict.fromkeys(request.next_steps))
    contract_claims = (
        _verbatim_evidence_fact_contracts(accepted)
        if request.task_type == "evidence_synthesis"
        else accepted
    )
    contract_claims = _dedupe_claim_contracts(contract_claims)
    claims = _artifact_claims(contract_claims)
    gaps = [
        ArtifactGap(
            code=str(item.get("reason") or "claim_rejected"),
            description=f"声明未通过引文绑定校验：{item.get('text', '')}",
        )
        for item in rejected
    ]
    common = {
        "claims": claims,
        "claim_contracts": contract_claims,
        "evidence_ledger": evidence_ledger,
        "applicability_records": applicability_records,
        "evidence_use_decisions": compile_evidence_use_decisions(
            task_type=request.task_type,
            claims=contract_claims,
            applicability_records=applicability_records,
        ),
        "gaps": gaps,
        "validation_plan": request.next_steps,
        "runtime_provenance": {
            "renderer": "synthesize_design_report",
            "rejected_claims": rejected,
            "claim_validation_scope": CLAIM_VALIDATION_SCOPE,
            "claim_binding_levels": _claim_binding_levels(contract_claims),
            "legacy_condition_comparable": request.condition_comparable,
            "terminal_projection": "claim_contract_only_v1",
        },
    }
    if request.task_type == "evidence_synthesis":
        artifact = EvidenceSynthesisArtifact(
            contract=default_task_contract("evidence_synthesis"),
            task_status=(
                "answered_grounded" if accepted else "insufficient_evidence"
            ),
            answer="\n".join(lines),
            **common,
        )
    else:
        assert comparability_assessment is not None
        gaps.extend(_gate_gaps(comparability_assessment.issues))
        grounded_comparison_claims = [
            claim
            for claim in claims
            if claim.claim_type in {"comparison", "ranking"}
        ]
        if comparability_assessment.status == "not_comparable":
            status = "not_comparable"
        elif (
            comparability_assessment.status == "comparable"
            and request.comparison_axes
            and grounded_comparison_claims
        ):
            status = "compared"
        else:
            status = "insufficient_evidence"
            if (
                comparability_assessment.status == "comparable"
                and not request.comparison_axes
            ):
                gaps.append(
                    ArtifactGap(
                        code="missing_comparison_axes",
                        description="候选工况可比，但尚未声明比较指标。",
                    )
                )
            if (
                comparability_assessment.status == "comparable"
                and not grounded_comparison_claims
            ):
                gaps.append(
                    ArtifactGap(
                        code="missing_grounded_comparison_claim",
                        description="候选工况可比，但尚无通过原文校验的比较声明。",
                    )
                )
            if (
                request.condition_comparable is not None
                and not request.comparison_records
            ):
                gaps.append(
                    ArtifactGap(
                        code="legacy_comparability_flag_ignored",
                        description=(
                            "单独的 condition_comparable 布尔值不具有准入权限；"
                            "必须提交结构化 comparison_records。"
                        ),
                    )
                )
        normalized_candidates = [
            candidate.model_dump(mode="json")
            for candidate in request.candidates
        ]
        if not normalized_candidates:
            normalized_candidates = [
                {"candidate_id": candidate_id}
                for candidate_id in comparability_assessment.candidate_ids
            ]
        artifact = CandidateScreeningArtifact(
            contract=default_task_contract("candidate_screening"),
            task_status=status,
            normalized_candidates=normalized_candidates,
            comparison_axes=request.comparison_axes,
            condition_comparability=comparability_assessment.status,
            comparability_assessment=comparability_assessment,
            **common,
        )
    artifact_payload = artifact.model_dump(mode="json")
    return (
        _render_contract_only_terminal(
            task_type=request.task_type,
            task_status=str(artifact_payload.get("task_status") or ""),
            claims=[
                SynthesisClaim.model_validate(item)
                for item in artifact_payload.get("claim_contracts") or []
            ],
        ),
        artifact_payload,
    )


@tool(response_format="content_and_artifact")
def synthesize_design_report(
    request: SynthesisRequest,
    state: Annotated[dict, InjectedState],
) -> tuple[str, dict[str, Any]]:
    """历史结构化模板兼容工具；Agent 的普通最终回答不需要调用。

    仅显式请求旧格式 artifact 时使用；保留旧科学合同，不调用模型、检索或 KG。

    design 的 finalization_mode="limited" 只生成不推荐的受限结果，不改变原评估；
    默认 assessed 路径和 inverse_design 的候选推荐必须经过结构化 decision gate，只允许渲染
    ``recommend`` 或 ``hold``。candidate_screening 还必须提交逐候选的
    ``comparison_records``，由纯 comparability gate 比较六类实验条件；
    若比较依据为模型性质预测，可传 ``model_comparison``：candidate_ids、
    temperatures_c、objective="maximise_minimum_conductivity"。编译器从本轮真实
    predict_formulation_conductivity 消息取得数值与组成；齐全且一致时返回
    model_assessed，不把该状态解释为实验工况比较或电池性能推荐。
    inverse_design 必须提交结构化竞争假设与区分实验。文献声明必须提供 chunk_id
    与唯一逐字引文。依赖实验条件的声明还必须提交 ``applicability_inputs``；
    编译器将 direct/transferable/insufficient/conflicting 映射为确定性动作权限，
    并阻止不适用或冲突声明进入终端正文。
    旧的 candidates、screening_results、draft_report 参数仅用于调用兼容，不再具有
    候选准入或自由文本直出的权限。

    request.next_steps 应覆盖原问题中尚未由文献测试证明的应用目标，包括近似
    工况的验证与未量化的成本目标。推荐候选不等于这些目标已经全部验证；
    在 claims 中保留实际测试条件，不把相近条件改写为用户目标条件。
    """

    req = SynthesisRequest.model_validate(_hydrate_request_from_state(request, state))
    # 旧模板需要的确定性准入仅在显式工具调用中计算，不挂在通用工具后 Hook。
    # 不注入行动指令，不修改 checkpoint，也不替主 Agent 选择是否继续研究。
    if (req.task_type in {"design", "inverse_design"} and not state.get("design_decision")
            and state.get("task_contract") and state.get("candidate_evaluations") is not None):
        from orion.core.design_runtime import scientific_design_policy
        assessment = scientific_design_policy(state).update or {}
        state = {**state, **{key: value for key, value in assessment.items() if key != "messages"}}
    if req.task_type == "inverse_design" and not req.direct_answer.strip():
        return (
            "错误：inverse_design 报告必须提供 `request.direct_answer`（1–2 句话直接回答用户问题）。"
            "请在 `request.direct_answer` 中写入直接回答后再调用本工具。",
            {"status": "error", "reason": "missing_direct_answer"},
        )
    if req.task_type == "design":
        return _design_report(req, state)
    if req.task_type == "inverse_design":
        return _inverse_report(req, state)
    if req.task_type == "candidate_screening" and req.model_comparison is not None:
        return _model_screening_report(req, state)
    return _evidence_or_screening_report(req)
