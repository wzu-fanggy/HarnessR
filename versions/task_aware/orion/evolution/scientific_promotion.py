from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.task_evidence.contracts_v2 import TaskType
from orion.task_evidence.evidence_applicability import (
    ApplicabilityConditionScope,
    ConditionField,
)


PromotionDecision = Literal["promote", "reject", "quarantine"]
PromotionUpdateKind = Literal["scientific_guard", "design_rule", "retrieval_skill"]
VerificationStatus = Literal["passed", "failed", "pending", "not_applicable"]


class PromotionEvidenceRef(BaseModel):
    """Immutable scientific evidence reference used to justify a persistent update."""

    model_config = ConfigDict(extra="forbid")

    source_run_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    claim_id: str = ""
    source_scope: ApplicabilityConditionScope


class ScientificPromotionScope(BaseModel):
    """Future-use scope claimed or admitted for one persistent update."""

    model_config = ConfigDict(extra="forbid")

    task_types: list[TaskType] = Field(min_length=1)
    scientific_object_id: str | None = None
    conditions: ApplicabilityConditionScope = Field(
        default_factory=ApplicabilityConditionScope
    )

    @model_validator(mode="after")
    def _validate_unique_task_types(self) -> "ScientificPromotionScope":
        if len(self.task_types) != len(set(self.task_types)):
            raise ValueError("task_types must not contain duplicates")
        return self


class PromotionVerifierResult(BaseModel):
    """Result from one verifier that is independent of the proposing update text."""

    model_config = ConfigDict(extra="forbid")

    verifier: str = Field(min_length=1)
    status: VerificationStatus
    reason_codes: list[str] = Field(default_factory=list)


class ScientificPromotionCandidate(BaseModel):
    """A proposed persistent update that may influence future scientific decisions.

    The proposing model may supply ``claimed_scope``.  The admission path separately
    supplies ``admitted_scope`` and verifier results.  Promotion is intentionally
    conservative: a field declared relevant for future use must be explicit in both
    scopes before it can be admitted.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    promotion_id: str = Field(min_length=1)
    update_kind: PromotionUpdateKind
    target_asset_id: str = Field(min_length=1)
    parent_version: str = Field(min_length=1)
    proposed_content: str = Field(min_length=1)

    sources: list[PromotionEvidenceRef] = Field(min_length=1)
    relevant_fields: list[ConditionField] = Field(default_factory=list)
    claimed_scope: ScientificPromotionScope
    admitted_scope: ScientificPromotionScope

    required_verifiers: list[str] = Field(
        default_factory=lambda: ["provenance", "evidence_use", "applicability"]
    )
    verifier_results: list[PromotionVerifierResult] = Field(default_factory=list)

    isolated_holdout_passed: bool | None = None
    capability_retention_passed: bool | None = None
    human_approval_required: Literal[True] = True
    human_approved: bool = False

    @model_validator(mode="after")
    def _validate_unique_contract_items(self) -> "ScientificPromotionCandidate":
        for name, values in (
            ("relevant_fields", self.relevant_fields),
            ("required_verifiers", self.required_verifiers),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")

        verifier_names = [item.verifier for item in self.verifier_results]
        if len(verifier_names) != len(set(verifier_names)):
            raise ValueError("verifier_results must contain at most one result per verifier")
        if not self.required_verifiers:
            raise ValueError("required_verifiers must not be empty")
        return self


class ScientificPromotionDecision(BaseModel):
    """Deterministic pre-commit decision for one persistent scientific update."""

    model_config = ConfigDict(extra="forbid")

    promotion_id: str = Field(min_length=1)
    decision: PromotionDecision
    reason_codes: list[str] = Field(default_factory=list)
    mismatched_scope_fields: list[ConditionField] = Field(default_factory=list)
    missing_scope_fields: list[ConditionField] = Field(default_factory=list)
    task_scope_mismatch: bool = False
    scientific_object_scope_mismatch: bool = False
    pending_verifiers: list[str] = Field(default_factory=list)
    failed_verifiers: list[str] = Field(default_factory=list)



def _scope_value(scope: ScientificPromotionScope, field: ConditionField) -> object:
    return getattr(scope.conditions, field)



def assess_scientific_promotion(
    candidate: ScientificPromotionCandidate,
) -> ScientificPromotionDecision:
    """Fail closed before a scientific update can become persistent behavior.

    This v1 gate deliberately uses exact agreement on task-declared relevant fields.
    It does not infer synonyms, bridge missing fields, or authorize a broader scope from
    a local observation.  Transfer beyond exact admitted scope belongs to an explicit
    reviewed transfer contract, not to this admission helper.
    """

    reasons: list[str] = []
    mismatched_scope_fields: list[ConditionField] = []
    missing_scope_fields: list[ConditionField] = []
    task_scope_mismatch = (
        set(candidate.claimed_scope.task_types) != set(candidate.admitted_scope.task_types)
    )
    scientific_object_scope_mismatch = (
        candidate.claimed_scope.scientific_object_id
        != candidate.admitted_scope.scientific_object_id
    )

    for field in candidate.relevant_fields:
        claimed = _scope_value(candidate.claimed_scope, field)
        admitted = _scope_value(candidate.admitted_scope, field)
        if claimed is None or admitted is None:
            missing_scope_fields.append(field)
            continue
        if claimed != admitted:
            mismatched_scope_fields.append(field)

    verifier_by_name = {item.verifier: item for item in candidate.verifier_results}
    pending_verifiers: list[str] = []
    failed_verifiers: list[str] = []
    for verifier in candidate.required_verifiers:
        result = verifier_by_name.get(verifier)
        if result is None or result.status == "pending":
            pending_verifiers.append(verifier)
        elif result.status == "failed":
            failed_verifiers.append(verifier)
        elif result.status == "not_applicable":
            # A verifier explicitly declared required cannot silently become N/A.
            pending_verifiers.append(verifier)

    if mismatched_scope_fields:
        reasons.append("claimed_scope_exceeds_or_conflicts_with_admitted_scope")
    if task_scope_mismatch:
        reasons.append("claimed_task_scope_not_admitted")
    if scientific_object_scope_mismatch:
        reasons.append("claimed_scientific_object_scope_not_admitted")
    if failed_verifiers:
        reasons.append("required_verifier_failed")
    if candidate.isolated_holdout_passed is False:
        reasons.append("isolated_holdout_failed")
    if candidate.capability_retention_passed is False:
        reasons.append("capability_retention_failed")

    if reasons:
        return ScientificPromotionDecision(
            promotion_id=candidate.promotion_id,
            decision="reject",
            reason_codes=reasons,
            mismatched_scope_fields=mismatched_scope_fields,
            missing_scope_fields=missing_scope_fields,
            task_scope_mismatch=task_scope_mismatch,
            scientific_object_scope_mismatch=scientific_object_scope_mismatch,
            pending_verifiers=pending_verifiers,
            failed_verifiers=failed_verifiers,
        )

    quarantine_reasons: list[str] = []
    if missing_scope_fields:
        quarantine_reasons.append("relevant_scope_not_fully_admitted")
    if pending_verifiers:
        quarantine_reasons.append("required_verifier_pending")
    if candidate.isolated_holdout_passed is None:
        quarantine_reasons.append("isolated_holdout_missing")
    if candidate.capability_retention_passed is None:
        quarantine_reasons.append("capability_retention_check_missing")
    if candidate.human_approval_required and not candidate.human_approved:
        quarantine_reasons.append("human_approval_pending")

    if quarantine_reasons:
        return ScientificPromotionDecision(
            promotion_id=candidate.promotion_id,
            decision="quarantine",
            reason_codes=quarantine_reasons,
            mismatched_scope_fields=mismatched_scope_fields,
            missing_scope_fields=missing_scope_fields,
            task_scope_mismatch=task_scope_mismatch,
            scientific_object_scope_mismatch=scientific_object_scope_mismatch,
            pending_verifiers=pending_verifiers,
            failed_verifiers=failed_verifiers,
        )

    return ScientificPromotionDecision(
        promotion_id=candidate.promotion_id,
        decision="promote",
        reason_codes=["all_promotion_gates_passed"],
    )
