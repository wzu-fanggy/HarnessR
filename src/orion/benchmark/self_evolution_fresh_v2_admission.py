"""Fresh v2 scientific admission adapter.

先执行 v2 typed representation fidelity，再复用开放世界三态 boundary semantics：
``unresolved``、``resolved_non_disqualifying``、``disqualifying_match``。
缺失条件永远不解释成 safely excluded。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.benchmark.self_evolution_fresh_scope import (
    BoundaryRelation,
    CategoricalScientificBoundary,
    NumericScientificDomain,
    classify_categorical_boundary_relation,
    classify_numeric_boundary_relation,
)
from orion.benchmark.self_evolution_fresh_v2_scope import (
    FreshPersistentUpdateProposalV2,
    FreshV2RepresentationContract,
    assess_v2_representation,
)
from orion.task_evidence.evidence_applicability import ConditionField


class FreshV2MatchedNumericBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    field: Literal["temperature_c", "voltage_window_v"]
    adverse_numeric_domain: NumericScientificDomain


class FreshV2MatchedCategoricalBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    boundary: CategoricalScientificBoundary

    @property
    def field(self) -> ConditionField:
        return self.boundary.field


class FreshV2ScientificFamilyContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    representation: FreshV2RepresentationContract
    matched_boundary: FreshV2MatchedNumericBoundary | None = None
    matched_categorical_boundary: FreshV2MatchedCategoricalBoundary | None = None

    @model_validator(mode="after")
    def _one_boundary_only(self) -> "FreshV2ScientificFamilyContract":
        if self.matched_boundary is not None and self.matched_categorical_boundary is not None:
            raise ValueError("v2 scientific contract may bind only one boundary type")
        return self


class FreshV2ScientificAdmissionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family_id: str
    decision: Literal[
        "eligible_for_isolated_future_evaluation",
        "quarantine_unresolved",
        "reject",
    ]
    source_binding_passed: bool
    representation_fidelity_passed: bool
    boundary_relation: BoundaryRelation | None = None
    candidate_available_in_isolated_future_evaluator: bool
    reason_codes: list[str] = Field(default_factory=list)


def assess_v2_scientific_admission(
    *,
    family_id: str,
    proposal: FreshPersistentUpdateProposalV2,
    contract: FreshV2ScientificFamilyContract,
) -> FreshV2ScientificAdmissionResult:
    """对 sealed v2 candidate 执行 representation + boundary admission。"""

    representation = assess_v2_representation(
        proposal=proposal,
        contract=contract.representation,
    )
    if not representation.representation_fidelity_passed:
        return FreshV2ScientificAdmissionResult(
            family_id=family_id,
            decision="reject",
            source_binding_passed=representation.source_binding_passed,
            representation_fidelity_passed=False,
            candidate_available_in_isolated_future_evaluator=False,
            reason_codes=list(representation.reason_codes),
        )

    assert proposal.claimed_scope is not None
    relation: BoundaryRelation | None = None
    if contract.matched_boundary is not None:
        field = contract.matched_boundary.field
        candidate_domain = getattr(proposal.claimed_scope, field)
        relation = classify_numeric_boundary_relation(
            candidate_domain,
            contract.matched_boundary.adverse_numeric_domain,
        )
    elif contract.matched_categorical_boundary is not None:
        field = contract.matched_categorical_boundary.field
        candidate_value = getattr(proposal.claimed_scope, field)
        relation = classify_categorical_boundary_relation(
            candidate_value,
            contract.matched_categorical_boundary.boundary,
        )

    if relation == "unresolved":
        return FreshV2ScientificAdmissionResult(
            family_id=family_id,
            decision="quarantine_unresolved",
            source_binding_passed=True,
            representation_fidelity_passed=True,
            boundary_relation=relation,
            candidate_available_in_isolated_future_evaluator=False,
            reason_codes=["v2_matched_boundary_condition_unresolved"],
        )
    if relation == "disqualifying_match":
        return FreshV2ScientificAdmissionResult(
            family_id=family_id,
            decision="reject",
            source_binding_passed=True,
            representation_fidelity_passed=True,
            boundary_relation=relation,
            candidate_available_in_isolated_future_evaluator=False,
            reason_codes=["v2_matched_boundary_disqualifying_overlap"],
        )

    return FreshV2ScientificAdmissionResult(
        family_id=family_id,
        decision="eligible_for_isolated_future_evaluation",
        source_binding_passed=True,
        representation_fidelity_passed=True,
        boundary_relation=relation,
        candidate_available_in_isolated_future_evaluator=True,
        reason_codes=[
            "v2_scientific_precheck_passed"
            if relation is None
            else "v2_matched_boundary_resolved_non_disqualifying"
        ],
    )


__all__ = [
    "FreshV2MatchedCategoricalBoundary",
    "FreshV2MatchedNumericBoundary",
    "FreshV2ScientificAdmissionResult",
    "FreshV2ScientificFamilyContract",
    "assess_v2_scientific_admission",
]
