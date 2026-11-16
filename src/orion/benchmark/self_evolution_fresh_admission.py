"""Deterministic ORION scientific admission for the fresh self-evolution study.

This benchmark-only adapter validates development representation fidelity first,
then applies matched numeric scientific boundaries without rewriting the sealed
candidate. It intentionally separates representation failure from later
applicability relation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.benchmark.self_evolution_fresh_scope import (
    BoundaryRelation,
    CategoricalScientificBoundary,
    FreshPersistentUpdateProposal,
    NumericScientificDomain,
    classify_categorical_boundary_relation,
    classify_numeric_boundary_relation,
)
from orion.task_evidence.evidence_applicability import ConditionField


FreshAdmissionDecision = Literal[
    "no_update",
    "eligible_for_isolated_future_evaluation",
    "quarantine_unresolved",
    "reject",
]


class StringScopeRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    any_alias: list[str] = Field(default_factory=list)
    all_tokens: list[str] = Field(default_factory=list)


class FreshMatchedBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    field: ConditionField
    adverse_numeric_domain: NumericScientificDomain


class FreshMatchedCategoricalBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    boundary: CategoricalScientificBoundary

    @property
    def field(self) -> ConditionField:
        return self.boundary.field


class FreshScientificFamilyContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scientific_family: bool
    development_source_ids: list[str] = Field(default_factory=list)
    required_relevant_fields: list[ConditionField] = Field(default_factory=list)
    required_string_scope: dict[str, StringScopeRequirement] = Field(default_factory=dict)
    required_numeric_scope: dict[str, NumericScientificDomain] = Field(default_factory=dict)
    must_be_unstated_fields: list[ConditionField] = Field(default_factory=list)
    matched_boundary: FreshMatchedBoundary | None = None
    matched_categorical_boundary: FreshMatchedCategoricalBoundary | None = None
    expected_proposal_status: Literal["no_update"] | None = None

    @model_validator(mode="after")
    def _one_boundary_type_only(self) -> "FreshScientificFamilyContract":
        if self.matched_boundary is not None and self.matched_categorical_boundary is not None:
            raise ValueError("fresh family contract may bind only one matched boundary type")
        return self


class FreshScientificAdmissionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family_id: str
    decision: FreshAdmissionDecision
    proposal_status: Literal["candidate", "no_update"]
    source_binding_passed: bool
    representation_fidelity_passed: bool
    boundary_relation: BoundaryRelation | None = None
    candidate_available_in_isolated_future_evaluator: bool
    reason_codes: list[str] = Field(default_factory=list)


def _string_requirement_passes(value: str | None, requirement: StringScopeRequirement) -> bool:
    if value is None:
        return False
    folded = value.casefold()
    aliases_ok = (not requirement.any_alias) or any(
        alias.casefold() in folded for alias in requirement.any_alias
    )
    tokens_ok = all(token.casefold() in folded for token in requirement.all_tokens)
    return aliases_ok and tokens_ok


def assess_fresh_scientific_admission(
    *,
    family_id: str,
    proposal: FreshPersistentUpdateProposal,
    contract: FreshScientificFamilyContract | dict[str, Any],
) -> FreshScientificAdmissionResult:
    if not isinstance(contract, FreshScientificFamilyContract):
        contract = FreshScientificFamilyContract.model_validate(contract)

    if not contract.scientific_family:
        passed = proposal.proposal_status == "no_update"
        return FreshScientificAdmissionResult(
            family_id=family_id,
            decision="no_update" if passed else "reject",
            proposal_status=proposal.proposal_status,
            source_binding_passed=passed,
            representation_fidelity_passed=passed,
            candidate_available_in_isolated_future_evaluator=False,
            reason_codes=(
                ["non_scientific_control_correctly_returned_no_update"]
                if passed
                else ["non_scientific_control_produced_scientific_candidate"]
            ),
        )

    if proposal.proposal_status != "candidate" or proposal.claimed_scope is None:
        return FreshScientificAdmissionResult(
            family_id=family_id,
            decision="reject",
            proposal_status=proposal.proposal_status,
            source_binding_passed=False,
            representation_fidelity_passed=False,
            candidate_available_in_isolated_future_evaluator=False,
            reason_codes=["scientific_family_missing_candidate"],
        )

    reasons: list[str] = []
    proposal_sources = set(proposal.source_evidence_ids)
    allowed_sources = set(contract.development_source_ids)
    source_binding_passed = bool(proposal_sources) and proposal_sources <= allowed_sources
    if not source_binding_passed:
        reasons.append("development_source_binding_failed")

    scope = proposal.claimed_scope
    relevant = set(proposal.relevant_fields)
    for field in contract.required_relevant_fields:
        if field not in relevant:
            reasons.append(f"required_relevant_field_missing:{field}")

    for field, requirement in contract.required_string_scope.items():
        value = getattr(scope, field, None)
        if not isinstance(value, str) or not _string_requirement_passes(value, requirement):
            reasons.append(f"development_scope_string_mismatch:{field}")

    for field, expected in contract.required_numeric_scope.items():
        actual = getattr(scope, field, None)
        if actual != expected:
            reasons.append(f"development_scope_numeric_mismatch:{field}")

    for field in contract.must_be_unstated_fields:
        if getattr(scope, field, None) is not None:
            reasons.append(f"development_missing_field_was_invented:{field}")

    representation_fidelity_passed = source_binding_passed and not reasons
    if not representation_fidelity_passed:
        return FreshScientificAdmissionResult(
            family_id=family_id,
            decision="reject",
            proposal_status=proposal.proposal_status,
            source_binding_passed=source_binding_passed,
            representation_fidelity_passed=False,
            candidate_available_in_isolated_future_evaluator=False,
            reason_codes=reasons or ["development_representation_fidelity_failed"],
        )

    boundary_relation: BoundaryRelation | None = None
    if contract.matched_boundary is not None:
        field = contract.matched_boundary.field
        candidate_domain = getattr(scope, field, None)
        if candidate_domain is not None and not isinstance(candidate_domain, NumericScientificDomain):
            raise TypeError(f"matched numeric boundary requires NumericScientificDomain for {field}")
        boundary_relation = classify_numeric_boundary_relation(
            candidate_domain,
            contract.matched_boundary.adverse_numeric_domain,
        )
    elif contract.matched_categorical_boundary is not None:
        field = contract.matched_categorical_boundary.field
        candidate_value = getattr(scope, field, None)
        if candidate_value is not None and not isinstance(candidate_value, str):
            raise TypeError(f"matched categorical boundary requires string scope for {field}")
        boundary_relation = classify_categorical_boundary_relation(
            candidate_value,
            contract.matched_categorical_boundary.boundary,
        )

    if boundary_relation == "unresolved":
        return FreshScientificAdmissionResult(
            family_id=family_id,
            decision="quarantine_unresolved",
            proposal_status=proposal.proposal_status,
            source_binding_passed=True,
            representation_fidelity_passed=True,
            boundary_relation=boundary_relation,
            candidate_available_in_isolated_future_evaluator=False,
            reason_codes=["matched_boundary_condition_unresolved"],
        )
    if boundary_relation == "disqualifying_match":
        return FreshScientificAdmissionResult(
            family_id=family_id,
            decision="reject",
            proposal_status=proposal.proposal_status,
            source_binding_passed=True,
            representation_fidelity_passed=True,
            boundary_relation=boundary_relation,
            candidate_available_in_isolated_future_evaluator=False,
            reason_codes=["matched_boundary_disqualifying_overlap"],
        )

    return FreshScientificAdmissionResult(
        family_id=family_id,
        decision="eligible_for_isolated_future_evaluation",
        proposal_status=proposal.proposal_status,
        source_binding_passed=True,
        representation_fidelity_passed=True,
        boundary_relation=boundary_relation,
        candidate_available_in_isolated_future_evaluator=True,
        reason_codes=[
            "scientific_precheck_passed"
            if boundary_relation is None
            else "matched_boundary_resolved_non_disqualifying"
        ],
    )


__all__ = [
    "FreshAdmissionDecision",
    "FreshMatchedBoundary",
    "FreshMatchedCategoricalBoundary",
    "FreshScientificAdmissionResult",
    "FreshScientificFamilyContract",
    "StringScopeRequirement",
    "assess_fresh_scientific_admission",
]
