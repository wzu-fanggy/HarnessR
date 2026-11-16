"""Typed contract for the common pre-arm persistent-update proposer.

The proposer sees development material only and produces either one reusable
candidate or an explicit no-update result.  It never decides admission.  The
same sealed candidate is later consumed by Arms B/C/D.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.evolution.scientific_promotion import PromotionUpdateKind, ScientificPromotionScope
from orion.task_evidence.evidence_applicability import ConditionField


ProposalStatus = Literal["candidate", "no_update"]


class ProposalEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_run_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    claim_id: str = ""


class PersistentUpdateProposal(BaseModel):
    """Common proposer output before B/C/D admission semantics diverge."""

    model_config = ConfigDict(extra="forbid")

    proposal_status: ProposalStatus
    update_kind: PromotionUpdateKind | None = None
    target_asset_id: Literal["electrolyte_design.skill"] | None = None
    proposed_content: str = ""
    claimed_scope: ScientificPromotionScope
    relevant_fields: list[ConditionField] = Field(default_factory=list)
    source_evidence_refs: list[ProposalEvidenceRef] = Field(default_factory=list)
    development_rationale: str = ""
    limitations_visible_from_development: list[str] = Field(default_factory=list)
    no_update_reason: str = ""

    @model_validator(mode="after")
    def _validate_candidate_or_no_update(self) -> "PersistentUpdateProposal":
        if len(self.relevant_fields) != len(set(self.relevant_fields)):
            raise ValueError("relevant_fields must not contain duplicates")

        if self.proposal_status == "candidate":
            if self.update_kind is None:
                raise ValueError("candidate requires update_kind")
            if self.target_asset_id != "electrolyte_design.skill":
                raise ValueError("candidate must target electrolyte_design.skill")
            if not self.proposed_content.strip():
                raise ValueError("candidate requires proposed_content")
            if not self.source_evidence_refs:
                raise ValueError("candidate requires development source evidence refs")
            if self.no_update_reason.strip():
                raise ValueError("candidate must not contain no_update_reason")
        else:
            if self.update_kind is not None:
                raise ValueError("no_update must not contain update_kind")
            if self.target_asset_id is not None:
                raise ValueError("no_update must not target a persistent scientific asset")
            if self.proposed_content.strip():
                raise ValueError("no_update must not contain proposed_content")
            if self.relevant_fields:
                raise ValueError("no_update must not declare scientific relevant_fields")
            if self.source_evidence_refs:
                raise ValueError("no_update must not claim scientific evidence refs")
            if not self.no_update_reason.strip():
                raise ValueError("no_update requires a reason")
        return self
