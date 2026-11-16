"""科学声明证据适用性的确定性最低门禁。

第一阶段只自动产生 direct、insufficient 和 conflicting。bridge evidence 只记录，
在 TransferContract 校验器落地前不得自动升级为 transferable。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

ApplicabilityStatus: TypeAlias = Literal[
    "direct", "transferable", "insufficient", "conflicting"
]
ApplicabilityDecision: TypeAlias = Literal[
    "answer", "qualified_answer", "search", "hold", "reject_claim"
]
TransferReviewStatus: TypeAlias = Literal["draft", "reviewed", "rejected"]
TransferUseKind: TypeAlias = Literal[
    "report", "compare", "recommend", "diagnose", "adjust"
]
TransferKind: TypeAlias = Literal[
    "temperature_range",
    "rate_range",
    "configuration_bridge",
    "composition_bridge",
    "mechanistic_bridge",
    "other",
]
ConditionField: TypeAlias = Literal[
    "composition", "salt", "solvent", "additive", "temperature_c",
    "electrode", "cell_configuration", "voltage_window_v", "rate",
    "cycle_count", "measurement_definition",
]
CONDITION_FIELDS: tuple[ConditionField, ...] = (
    "composition", "salt", "solvent", "additive", "temperature_c",
    "electrode", "cell_configuration", "voltage_window_v", "rate",
    "cycle_count", "measurement_definition",
)


class ApplicabilityConditionScope(BaseModel):
    """证据来源或目标实验条件。"""

    model_config = ConfigDict(extra="forbid")
    composition: str | None = None
    salt: str | None = None
    solvent: str | None = None
    additive: str | None = None
    temperature_c: float | None = None
    electrode: str | None = None
    cell_configuration: str | None = None
    voltage_window_v: tuple[float, float] | None = None
    rate: str | None = None
    cycle_count: int | None = Field(default=None, ge=0)
    measurement_definition: str | None = None

    @model_validator(mode="after")
    def _validate_voltage_window(self) -> "ApplicabilityConditionScope":
        if self.voltage_window_v is not None and self.voltage_window_v[0] >= self.voltage_window_v[1]:
            raise ValueError("voltage_window_v must be ordered as [lower, upper]")
        return self


class ApplicabilityAssessment(BaseModel):
    """claim-evidence-target 元组的确定性适用性结果。"""

    model_config = ConfigDict(extra="forbid")
    status: ApplicabilityStatus
    relevant_fields: list[ConditionField] = Field(default_factory=list)
    matched_fields: list[ConditionField] = Field(default_factory=list)
    mismatched_fields: list[ConditionField] = Field(default_factory=list)
    missing_fields: list[ConditionField] = Field(default_factory=list)
    validated_conflict_evidence_ids: list[str] = Field(default_factory=list)
    pending_bridge_evidence_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class ApplicabilityInput(BaseModel):
    """主运行时提交的 claim-evidence-target 适用性输入。"""

    model_config = ConfigDict(extra="forbid")
    assessment_id: str = Field(min_length=1)
    claim_id: str = ""
    source_scope: ApplicabilityConditionScope
    target_scope: ApplicabilityConditionScope
    relevant_fields: list[ConditionField] = Field(min_length=1)
    validated_conflict_evidence_ids: list[str] = Field(default_factory=list)
    bridge_evidence_ids: list[str] = Field(default_factory=list)
    transfer_contract: "TransferContract | None" = None


class ApplicabilityRecord(BaseModel):
    """写入 GraphState/TaskArtifact 的可审计适用性状态与动作权限。"""

    model_config = ConfigDict(extra="forbid")
    assessment_id: str = Field(min_length=1)
    claim_id: str = ""
    source_scope: ApplicabilityConditionScope
    target_scope: ApplicabilityConditionScope
    assessment: ApplicabilityAssessment
    decision: ApplicabilityDecision
    retrieval_allowed: bool = False
    transfer_contract_id: str | None = None
    transfer_contract_check: "TransferContractCheck | None" = None
    transfer_authorized_uses: list[TransferUseKind] = Field(default_factory=list)
    terminal_disposition: Literal["pending", "emitted", "blocked"] = "pending"

    @model_validator(mode="after")
    def _default_legacy_transfer_scope(self) -> "ApplicabilityRecord":
        # Older transferable records predate use-scoped transfer. Preserve their
        # least-privilege meaning as report-only rather than granting every
        # downstream scientific use by default.
        if self.assessment.status == "transferable" and not self.transfer_authorized_uses:
            self.transfer_authorized_uses = ["report"]
        return self


class TransferEvidenceRef(BaseModel):
    """TransferContract 中必须逐字回链的单条证据引用。"""

    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    raw_quote: str = Field(min_length=12)


class TransferReview(BaseModel):
    """迁移语义的人工审阅状态；结构校验不能替代该审阅。"""

    model_config = ConfigDict(extra="forbid")
    status: TransferReviewStatus = "draft"
    reviewer: str | None = None
    reviewed_at: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def _reviewed_requires_identity(self) -> "TransferReview":
        if self.status == "reviewed" and (not self.reviewer or not self.reviewed_at):
            raise ValueError("reviewed transfer contract requires reviewer and reviewed_at")
        return self


class TransferContract(BaseModel):
    """证明 source 条件到 target 条件可限定迁移的独立证据合同。"""

    model_config = ConfigDict(extra="forbid")
    transfer_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    source_ref: TransferEvidenceRef
    bridge_ref: TransferEvidenceRef
    source_scope: ApplicabilityConditionScope
    target_scope: ApplicabilityConditionScope
    relevant_fields: list[ConditionField] = Field(min_length=1)
    changed_fields: list[ConditionField] = Field(min_length=1)
    supported_dimensions: list[ConditionField] = Field(min_length=1)
    transfer_kind: TransferKind
    limitations: list[str] = Field(min_length=1)
    authorized_uses: list[TransferUseKind] = Field(default_factory=lambda: ["report"])
    quantitative_transfer_allowed: bool = False
    review: TransferReview = Field(default_factory=TransferReview)

    @model_validator(mode="after")
    def _validate_transfer_shape(self) -> "TransferContract":
        for name, values in (
            ("relevant_fields", self.relevant_fields),
            ("changed_fields", self.changed_fields),
            ("supported_dimensions", self.supported_dimensions),
            ("authorized_uses", self.authorized_uses),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if not self.authorized_uses:
            raise ValueError("authorized_uses must not be empty")
        if not set(self.changed_fields).issubset(set(self.relevant_fields)):
            raise ValueError("changed_fields must be a subset of relevant_fields")
        if not set(self.changed_fields).issubset(set(self.supported_dimensions)):
            raise ValueError("bridge evidence must cover every changed field")
        if (
            self.source_ref.paper_id == self.bridge_ref.paper_id
            and self.source_ref.chunk_id == self.bridge_ref.chunk_id
            and self.source_ref.raw_quote == self.bridge_ref.raw_quote
        ):
            raise ValueError("bridge evidence must be distinct from source evidence")
        return self


class TransferContractCheck(BaseModel):
    """TransferContract 的结构/逐字绑定检查结果。"""

    model_config = ConfigDict(extra="forbid")
    binding_valid: bool
    review_approved: bool
    eligible_for_transfer: bool
    actual_changed_fields: list[ConditionField] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


def _parse_scope(value: ApplicabilityConditionScope | Mapping[str, Any]) -> ApplicabilityConditionScope:
    return value if isinstance(value, ApplicabilityConditionScope) else ApplicabilityConditionScope.model_validate(dict(value))


def _normalize(field: ConditionField, value: Any) -> Any:
    if value is None:
        return None
    if field == "temperature_c":
        return round(float(value), 6)
    if field == "voltage_window_v":
        return tuple(round(float(item), 6) for item in value)
    if field == "cycle_count":
        return int(value)
    text = " ".join(str(value).split()).casefold()
    return text.replace(" ", "") if field == "rate" else text


def _ids(values: Iterable[str] | None) -> list[str]:
    return list(dict.fromkeys(v.strip() for v in (values or []) if v.strip()))


def _actual_changed_fields(
    source: ApplicabilityConditionScope,
    target: ApplicabilityConditionScope,
    fields: Sequence[ConditionField],
) -> list[ConditionField]:
    changed: list[ConditionField] = []
    for field in fields:
        source_value, target_value = getattr(source, field), getattr(target, field)
        if source_value is None or target_value is None:
            changed.append(field)
        elif _normalize(field, source_value) != _normalize(field, target_value):
            changed.append(field)
    return changed


def check_transfer_contract_binding(
    contract: TransferContract | Mapping[str, Any],
    *,
    source_evidence_text: str,
    bridge_evidence_text: str,
) -> TransferContractCheck:
    """校验 TransferContract 的逐字绑定和人工审批门禁。

    这里只能证明合同结构和 quote identity 成立；bridge evidence 是否在科学上足以
    支撑迁移，必须由 ``review.status=reviewed`` 显式确认。draft 永远不能放行。
    """

    parsed = contract if isinstance(contract, TransferContract) else TransferContract.model_validate(dict(contract))
    issues: list[str] = []
    if source_evidence_text.count(parsed.source_ref.raw_quote) != 1:
        issues.append("source_quote_not_unique_verbatim")
    if bridge_evidence_text.count(parsed.bridge_ref.raw_quote) != 1:
        issues.append("bridge_quote_not_unique_verbatim")

    actual_changed = _actual_changed_fields(
        parsed.source_scope, parsed.target_scope, parsed.relevant_fields
    )
    if set(actual_changed) != set(parsed.changed_fields):
        issues.append("changed_fields_mismatch")
    if not set(actual_changed).issubset(set(parsed.supported_dimensions)):
        issues.append("bridge_dimensions_do_not_cover_actual_change")

    review_approved = parsed.review.status == "reviewed"
    if parsed.review.status == "draft":
        issues.append("transfer_semantic_review_pending")
    elif parsed.review.status == "rejected":
        issues.append("transfer_semantic_review_rejected")

    structural_issues = [
        issue
        for issue in issues
        if issue not in {"transfer_semantic_review_pending", "transfer_semantic_review_rejected"}
    ]
    binding_valid = not structural_issues
    return TransferContractCheck(
        binding_valid=binding_valid,
        review_approved=review_approved,
        eligible_for_transfer=binding_valid and review_approved,
        actual_changed_fields=actual_changed,
        issues=issues,
    )


def assess_evidence_applicability(
    *,
    source_scope: ApplicabilityConditionScope | Mapping[str, Any],
    target_scope: ApplicabilityConditionScope | Mapping[str, Any],
    relevant_fields: Sequence[ConditionField],
    validated_conflict_evidence_ids: Iterable[str] | None = None,
    bridge_evidence_ids: Iterable[str] | None = None,
) -> ApplicabilityAssessment:
    """不调用 LLM 的适用性最低门禁；不自行验证 bridge/conflict 证据语义。"""

    fields = list(relevant_fields)
    if not fields:
        raise ValueError("relevant_fields must not be empty")
    if len(set(fields)) != len(fields):
        raise ValueError("relevant_fields must not contain duplicates")
    unknown = [field for field in fields if field not in CONDITION_FIELDS]
    if unknown:
        raise ValueError(f"unknown applicability condition fields: {unknown}")

    source, target = _parse_scope(source_scope), _parse_scope(target_scope)
    matched: list[ConditionField] = []
    mismatched: list[ConditionField] = []
    missing: list[ConditionField] = []
    for field in fields:
        sv, tv = getattr(source, field), getattr(target, field)
        if sv is None or tv is None:
            missing.append(field)
        elif _normalize(field, sv) == _normalize(field, tv):
            matched.append(field)
        else:
            mismatched.append(field)

    conflicts, bridges = _ids(validated_conflict_evidence_ids), _ids(bridge_evidence_ids)
    reasons: list[str] = []
    if conflicts:
        status: ApplicabilityStatus = "conflicting"
        reasons.append("validated_conflict_evidence")
    elif missing or mismatched:
        status = "insufficient"
        if missing:
            reasons.append("missing_target_or_source_condition")
        if mismatched:
            reasons.append("relevant_condition_mismatch")
        if bridges:
            reasons.append("bridge_evidence_requires_transfer_contract_validation")
    else:
        status = "direct"
        reasons.append("all_relevant_conditions_match")

    return ApplicabilityAssessment(
        status=status,
        relevant_fields=fields,
        matched_fields=matched,
        mismatched_fields=mismatched,
        missing_fields=missing,
        validated_conflict_evidence_ids=conflicts,
        pending_bridge_evidence_ids=bridges,
        reason_codes=reasons,
    )


def preferred_applicability_decision(
    status: ApplicabilityStatus,
    *,
    retrieval_allowed: bool,
) -> ApplicabilityDecision:
    """把适用性状态确定性映射为唯一首选动作。"""

    if status == "direct":
        return "answer"
    if status == "transferable":
        return "qualified_answer"
    if status == "insufficient":
        return "search" if retrieval_allowed else "hold"
    return "reject_claim"


def evaluate_applicability_input(
    value: ApplicabilityInput | Mapping[str, Any],
    *,
    evidence_text_by_chunk: Mapping[str, str] | None = None,
    retrieval_allowed: bool = False,
) -> ApplicabilityRecord:
    """计算主运行时状态；只有已复核且逐字绑定的合同可升级为 transferable。"""

    item = value if isinstance(value, ApplicabilityInput) else ApplicabilityInput.model_validate(dict(value))
    assessment = assess_evidence_applicability(
        source_scope=item.source_scope,
        target_scope=item.target_scope,
        relevant_fields=item.relevant_fields,
        validated_conflict_evidence_ids=item.validated_conflict_evidence_ids,
        bridge_evidence_ids=item.bridge_evidence_ids,
    )
    transfer_check: TransferContractCheck | None = None
    transfer_id: str | None = None
    transfer_authorized_uses: list[TransferUseKind] = []
    if item.transfer_contract is not None and assessment.status != "conflicting":
        contract = item.transfer_contract
        transfer_id = contract.transfer_id
        transfer_authorized_uses = list(contract.authorized_uses)
        texts = dict(evidence_text_by_chunk or {})
        contract_matches = (
            (not item.claim_id or contract.claim_id == item.claim_id)
            and contract.source_scope == item.source_scope
            and contract.target_scope == item.target_scope
            and set(contract.relevant_fields) == set(item.relevant_fields)
            and contract.bridge_ref.chunk_id in set(item.bridge_evidence_ids)
        )
        if contract_matches:
            transfer_check = check_transfer_contract_binding(
                contract,
                source_evidence_text=texts.get(contract.source_ref.chunk_id, ""),
                bridge_evidence_text=texts.get(contract.bridge_ref.chunk_id, ""),
            )
        else:
            transfer_check = TransferContractCheck(
                binding_valid=False,
                review_approved=contract.review.status == "reviewed",
                eligible_for_transfer=False,
                issues=["transfer_contract_runtime_input_mismatch"],
            )
        if transfer_check.eligible_for_transfer:
            assessment = assessment.model_copy(
                update={
                    "status": "transferable",
                    "reason_codes": ["reviewed_transfer_contract"],
                }
            )
        else:
            assessment = assessment.model_copy(
                update={
                    "reason_codes": list(
                        dict.fromkeys(
                            [*assessment.reason_codes, *transfer_check.issues]
                        )
                    )
                }
            )

    return ApplicabilityRecord(
        assessment_id=item.assessment_id,
        claim_id=item.claim_id,
        source_scope=item.source_scope,
        target_scope=item.target_scope,
        assessment=assessment,
        decision=preferred_applicability_decision(
            assessment.status,
            retrieval_allowed=retrieval_allowed,
        ),
        retrieval_allowed=retrieval_allowed,
        transfer_contract_id=transfer_id,
        transfer_contract_check=transfer_check,
        transfer_authorized_uses=transfer_authorized_uses,
    )
