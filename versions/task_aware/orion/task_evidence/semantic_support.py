"""ClaimContract v2 的确定性语义支持下界。

该模块不尝试用词法启发式证明任意改写与引文等价。对只具备逐字引文的声明，
唯一可确定验证的输出就是引文本身；模型改写仅作为审计输入保留。这样可以在不调用
额外 LLM 的情况下阻断实体替换、因果升级、范围泛化和条件漂移。
"""

from __future__ import annotations

from dataclasses import dataclass

from orion.task_evidence.claim_binding import check_quote_binding


SEMANTIC_SUPPORT_POLICY_VERSION = "quote_identity_v2"
_EVIDENTIAL_CLAIM_TYPES = frozenset({"fact", "comparison", "ranking"})


@dataclass(frozen=True)
class SemanticSupportDecision:
    """确定性门禁结果；canonical 字段才允许进入 verified 输出。"""

    accepted: bool
    canonical_claim_text: str = ""
    canonical_condition: str = ""
    action: str = "rejected"
    reason: str = ""
    checks: tuple[str, ...] = ()
    unmatched_unit_values: tuple[str, ...] = ()


def enforce_quote_identity_v2(
    *,
    claim_text: str,
    evidence_quote: str,
    claim_type: str = "fact",
    condition: str = "",
    condition_quote: str = "",
) -> SemanticSupportDecision:
    """把 quote-bound 声明收缩到可确定验证的逐字边界。

    - fact/comparison/ranking 可以被规范化为逐字引文；
    - recommendation/diagnosis 等推断需要 computation/human binding，不能仅凭 quote
      被标为 verified；
    - 声明或条件若引入引文中不存在的带单位数值，直接拒绝而不是静默改写。
    """

    proposed = str(claim_text or "").strip()
    quote = str(evidence_quote or "").strip()
    proposed_condition = str(condition or "").strip()
    quote_condition = str(condition_quote or "").strip()
    checks = (
        "claim_type_is_evidential",
        "authoritative_quote_present",
        "claim_unit_value_consistency",
        "condition_unit_value_consistency",
        "canonical_claim_equals_quote",
    )

    if str(claim_type or "fact").strip() not in _EVIDENTIAL_CLAIM_TYPES:
        return SemanticSupportDecision(
            accepted=False,
            reason="inference_claim_requires_stronger_binding",
            checks=checks,
        )
    if not proposed:
        return SemanticSupportDecision(
            accepted=False,
            reason="missing_claim_text",
            checks=checks,
        )
    if not quote:
        return SemanticSupportDecision(
            accepted=False,
            reason="missing_authoritative_quote",
            checks=checks,
        )
    if "\ufffd" in quote or "\ufffd" in quote_condition:
        return SemanticSupportDecision(
            accepted=False,
            reason="authoritative_quote_encoding_error",
            checks=checks,
        )

    authoritative_text = "\n".join(
        part for part in (quote, quote_condition) if part
    )
    claim_binding = check_quote_binding(
        claim_text=proposed,
        evidence_text=authoritative_text,
    )
    if not claim_binding.accepted:
        return SemanticSupportDecision(
            accepted=False,
            reason=claim_binding.reason,
            checks=checks,
            unmatched_unit_values=claim_binding.unmatched_unit_values,
        )

    condition_binding = check_quote_binding(
        claim_text=proposed_condition,
        evidence_text=authoritative_text,
    )
    if not condition_binding.accepted:
        return SemanticSupportDecision(
            accepted=False,
            reason="condition_unit_value_not_in_authoritative_evidence",
            checks=checks,
            unmatched_unit_values=condition_binding.unmatched_unit_values,
        )

    canonical_condition = quote_condition or proposed_condition
    exact_claim = proposed == quote
    exact_condition = not proposed_condition or proposed_condition == canonical_condition
    action = (
        "accepted_verbatim"
        if exact_claim and exact_condition
        else "canonicalized_to_quote"
    )
    return SemanticSupportDecision(
        accepted=True,
        canonical_claim_text=quote,
        canonical_condition=canonical_condition,
        action=action,
        checks=checks,
    )
