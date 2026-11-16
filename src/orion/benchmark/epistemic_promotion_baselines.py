"""Executable admission contracts for strong Arm B and Arm C baselines.

These baselines intentionally exclude Orion-specific scientific transfer
semantics.  They are meant to be competitive comparators after the 2026
closest-work audit, not strawman controls.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BaselineAdmissionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    apply_update: bool
    reason_codes: list[str] = Field(default_factory=list)


class ScientificMemoryAdmissionRecord(BaseModel):
    """Strong provenance-linked memory contract for Arm B."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    development_trajectory_successful: bool
    local_outcome_validated: bool
    provenance_bound: bool
    reusable_memory_serialized: bool
    failure_notes_or_guardrails_preserved: bool


def assess_scientific_memory_baseline(
    record: ScientificMemoryAdmissionRecord,
) -> BaselineAdmissionDecision:
    checks = {
        "development_trajectory_not_successful": record.development_trajectory_successful,
        "local_outcome_not_validated": record.local_outcome_validated,
        "provenance_not_bound": record.provenance_bound,
        "memory_not_serializable": record.reusable_memory_serialized,
        "failure_notes_or_guardrails_not_preserved": record.failure_notes_or_guardrails_preserved,
    }
    failures = [code for code, passed in checks.items() if not passed]
    return BaselineAdmissionDecision(
        candidate_id=record.candidate_id,
        apply_update=not failures,
        reason_codes=failures or ["strong_scientific_memory_contract_passed"],
    )


class PairedGenericQualityJudgment(BaseModel):
    """Fixed-format output of the generic Arm-C quality judge."""

    model_config = ConfigDict(extra="forbid")

    baseline_quality_score: float = Field(ge=0.0, le=1.0)
    candidate_quality_score: float = Field(ge=0.0, le=1.0)
    candidate_materially_improves_task_quality: bool
    candidate_introduces_generic_task_regression: bool
    rationale: str = Field(min_length=1)

    @property
    def numeric_improvement(self) -> bool:
        return self.candidate_quality_score > self.baseline_quality_score


class GenericAuditedAdmissionRecord(BaseModel):
    """Strong generic audited-skill contract for Arm C."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    development_trajectory_successful: bool
    paired_quality_judgment: PairedGenericQualityJudgment
    fixed_contract_verifier_passed: bool
    isolated_generic_holdout_passed: bool
    capability_retention_passed: bool
    rollback_regression_check_passed: bool


def assess_generic_audited_baseline(
    record: GenericAuditedAdmissionRecord,
) -> BaselineAdmissionDecision:
    judgment = record.paired_quality_judgment
    checks = {
        "development_trajectory_not_successful": record.development_trajectory_successful,
        "paired_quality_score_not_improved": judgment.numeric_improvement,
        "paired_quality_judge_did_not_confirm_material_improvement": judgment.candidate_materially_improves_task_quality,
        "paired_quality_judge_found_generic_regression": not judgment.candidate_introduces_generic_task_regression,
        "fixed_generic_contract_verifier_failed": record.fixed_contract_verifier_passed,
        "isolated_generic_holdout_failed": record.isolated_generic_holdout_passed,
        "prior_capability_retention_failed": record.capability_retention_passed,
        "rollback_regression_check_failed": record.rollback_regression_check_passed,
    }
    failures = [code for code, passed in checks.items() if not passed]
    return BaselineAdmissionDecision(
        candidate_id=record.candidate_id,
        apply_update=not failures,
        reason_codes=failures or ["strong_generic_audited_evolution_contract_passed"],
    )


def reject_all_control(candidate_id: str) -> BaselineAdmissionDecision:
    return BaselineAdmissionDecision(
        candidate_id=candidate_id,
        apply_update=False,
        reason_codes=["reject_all_control"],
    )
