"""Fresh v2 lossless scientific-scope representation.

v1 的自由文本 scope gate 会把 ``hard-carbon`` 与 ``hard carbon``、以及
``composition=1 M NaPF6/diglyme`` + ``salt=NaPF6`` 这类等价表达误判为不同。
v2 把关键 modifier 拆成 typed fields，并只对自然语言类别做保守 canonicalization。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.benchmark.self_evolution_fresh_scope import NumericScientificDomain
from orion.task_evidence.evidence_applicability import ConditionField


class FreshV2SaltScope(BaseModel):
    """盐身份与浓度分离，禁止把浓度藏在 salt identity 的字符串约定里。"""

    model_config = ConfigDict(extra="forbid")

    identity: str = Field(min_length=1)
    concentration_mol_l: float | None = Field(default=None, gt=0)


class FreshV2ElectrodePairScope(BaseModel):
    """把正负极类别分开表达；证据未给出的那一侧必须保持 ``None``。"""

    model_config = ConfigDict(extra="forbid")

    positive_category: str | None = Field(default=None, min_length=1)
    negative_category: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _at_least_one_side_is_explicit(self) -> "FreshV2ElectrodePairScope":
        if self.positive_category is None and self.negative_category is None:
            raise ValueError("electrode_pair requires at least one explicit electrode side")
        return self


class FreshScientificScopeV2(BaseModel):
    """v2 benchmark-only scientific scope；缺失条件必须显式保持 ``None``。"""

    model_config = ConfigDict(extra="forbid")

    task_types: list[str] = Field(min_length=1)
    scientific_object_id: str | None = None
    composition: str | None = None
    salt: FreshV2SaltScope | None = None
    solvent: str | None = None
    additive: str | None = None
    temperature_c: NumericScientificDomain | None = None
    electrode_pair: FreshV2ElectrodePairScope | None = None
    cell_configuration: str | None = None
    voltage_window_v: NumericScientificDomain | None = None
    rate: str | None = None
    cycle_count: int | None = Field(default=None, ge=0)
    measurement_definition: str | None = None

    @model_validator(mode="after")
    def _validate_scope(self) -> "FreshScientificScopeV2":
        if len(self.task_types) != len(set(self.task_types)):
            raise ValueError("task_types must be unique")
        if self.temperature_c is not None and self.temperature_c.unit != "C":
            raise ValueError("temperature_c unit must be C")
        if self.voltage_window_v is not None and self.voltage_window_v.unit != "V":
            raise ValueError("voltage_window_v unit must be V")
        return self


class FreshPersistentUpdateProposalV2(BaseModel):
    """v2 proposer schema；非法 ConditionField 在解析阶段直接失败，不做 repair。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["orion_self_evolution_fresh_proposal_v2"] = (
        "orion_self_evolution_fresh_proposal_v2"
    )
    proposal_status: Literal["candidate", "no_update"]
    proposed_content: str = ""
    claimed_scope: FreshScientificScopeV2 | None = None
    relevant_fields: list[ConditionField] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)
    development_rationale: str = ""
    limitations_visible_from_development: list[str] = Field(default_factory=list)
    no_update_reason: str = ""

    @model_validator(mode="after")
    def _candidate_or_no_update(self) -> "FreshPersistentUpdateProposalV2":
        if len(self.relevant_fields) != len(set(self.relevant_fields)):
            raise ValueError("relevant_fields must be unique")
        if self.proposal_status == "candidate":
            if self.claimed_scope is None:
                raise ValueError("candidate requires claimed_scope")
            if not self.proposed_content.strip():
                raise ValueError("candidate requires proposed_content")
            if not self.source_evidence_ids:
                raise ValueError("candidate requires source_evidence_ids")
            if self.no_update_reason.strip():
                raise ValueError("candidate must not contain no_update_reason")
        else:
            if self.claimed_scope is not None:
                raise ValueError("no_update must not claim scientific scope")
            if self.proposed_content.strip() or self.relevant_fields or self.source_evidence_ids:
                raise ValueError("no_update must not contain persistent scientific content")
            if not self.no_update_reason.strip():
                raise ValueError("no_update requires reason")
        return self


class FreshV2TextRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    any_alias: list[str] = Field(default_factory=list)
    all_tokens: list[str] = Field(default_factory=list)


class FreshV2SaltRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity_aliases: list[str] = Field(min_length=1)
    concentration_mol_l: float | None = Field(default=None, gt=0)


class FreshV2ElectrodePairRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive_any_alias: list[str] = Field(default_factory=list)
    negative_any_alias: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _at_least_one_side_is_required(self) -> "FreshV2ElectrodePairRequirement":
        if not self.positive_any_alias and not self.negative_any_alias:
            raise ValueError("electrode requirement must constrain at least one side")
        return self


class FreshV2RepresentationContract(BaseModel):
    """只约束 development representation；boundary applicability 在后续独立处理。"""

    model_config = ConfigDict(extra="forbid")

    development_source_ids: list[str] = Field(min_length=1)
    required_relevant_fields: list[ConditionField] = Field(default_factory=list)
    required_text_scope: dict[str, FreshV2TextRequirement] = Field(default_factory=dict)
    required_salt: FreshV2SaltRequirement | None = None
    required_electrode_pair: FreshV2ElectrodePairRequirement | None = None
    required_numeric_scope: dict[str, NumericScientificDomain] = Field(default_factory=dict)
    must_be_unstated_fields: list[str] = Field(default_factory=list)


class FreshV2RepresentationAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_binding_passed: bool
    representation_fidelity_passed: bool
    reason_codes: list[str] = Field(default_factory=list)


def normalize_scientific_text(value: str) -> str:
    """仅消除良性正字法差异，不做化学实体推断或同义词猜测。"""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("−", "-").replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[^0-9a-z]+", " ", normalized)
    return " ".join(normalized.split())


def _contains_normalized(value: str, needle: str) -> bool:
    haystack = f" {normalize_scientific_text(value)} "
    token = normalize_scientific_text(needle)
    return bool(token) and f" {token} " in haystack


def _text_requirement_passes(value: str | None, requirement: FreshV2TextRequirement) -> bool:
    if value is None:
        return False
    aliases_ok = (not requirement.any_alias) or any(
        _contains_normalized(value, alias) for alias in requirement.any_alias
    )
    tokens_ok = all(_contains_normalized(value, token) for token in requirement.all_tokens)
    return aliases_ok and tokens_ok


def _salt_requirement_passes(
    value: FreshV2SaltScope | None,
    requirement: FreshV2SaltRequirement,
) -> bool:
    if value is None:
        return False
    identity_ok = any(
        _contains_normalized(value.identity, alias) for alias in requirement.identity_aliases
    )
    if not identity_ok:
        return False
    expected = requirement.concentration_mol_l
    if expected is None:
        return True
    if value.concentration_mol_l is None:
        return False
    return abs(value.concentration_mol_l - expected) <= 1e-9


def _electrode_requirement_passes(
    value: FreshV2ElectrodePairScope | None,
    requirement: FreshV2ElectrodePairRequirement,
) -> bool:
    if value is None:
        return False
    positive_ok = not requirement.positive_any_alias or (
        value.positive_category is not None
        and any(
            _contains_normalized(value.positive_category, alias)
            for alias in requirement.positive_any_alias
        )
    )
    negative_ok = not requirement.negative_any_alias or (
        value.negative_category is not None
        and any(
            _contains_normalized(value.negative_category, alias)
            for alias in requirement.negative_any_alias
        )
    )
    return positive_ok and negative_ok


def assess_v2_representation(
    *,
    proposal: FreshPersistentUpdateProposalV2,
    contract: FreshV2RepresentationContract | dict[str, Any],
) -> FreshV2RepresentationAssessment:
    """对 sealed proposal 做 lossless typed representation fidelity 检查。"""

    if not isinstance(contract, FreshV2RepresentationContract):
        contract = FreshV2RepresentationContract.model_validate(contract)
    if proposal.proposal_status != "candidate" or proposal.claimed_scope is None:
        return FreshV2RepresentationAssessment(
            source_binding_passed=False,
            representation_fidelity_passed=False,
            reason_codes=["v2_scientific_candidate_missing"],
        )

    reasons: list[str] = []
    proposal_sources = set(proposal.source_evidence_ids)
    allowed_sources = set(contract.development_source_ids)
    source_ok = bool(proposal_sources) and proposal_sources <= allowed_sources
    if not source_ok:
        reasons.append("v2_development_source_binding_failed")

    relevant = set(proposal.relevant_fields)
    for field in contract.required_relevant_fields:
        if field not in relevant:
            reasons.append(f"v2_required_relevant_field_missing:{field}")

    scope = proposal.claimed_scope
    for field, requirement in contract.required_text_scope.items():
        value = getattr(scope, field, None)
        if not isinstance(value, str) or not _text_requirement_passes(value, requirement):
            reasons.append(f"v2_text_scope_mismatch:{field}")

    if contract.required_salt is not None and not _salt_requirement_passes(
        scope.salt, contract.required_salt
    ):
        reasons.append("v2_salt_scope_mismatch")

    if contract.required_electrode_pair is not None and not _electrode_requirement_passes(
        scope.electrode_pair, contract.required_electrode_pair
    ):
        reasons.append("v2_electrode_pair_scope_mismatch")

    for field, expected in contract.required_numeric_scope.items():
        if getattr(scope, field, None) != expected:
            reasons.append(f"v2_numeric_scope_mismatch:{field}")

    for field in contract.must_be_unstated_fields:
        if getattr(scope, field, None) is not None:
            reasons.append(f"v2_missing_field_was_invented:{field}")

    passed = source_ok and not reasons
    return FreshV2RepresentationAssessment(
        source_binding_passed=source_ok,
        representation_fidelity_passed=passed,
        reason_codes=["v2_representation_fidelity_passed"] if passed else reasons,
    )


__all__ = [
    "FreshPersistentUpdateProposalV2",
    "FreshScientificScopeV2",
    "FreshV2ElectrodePairRequirement",
    "FreshV2ElectrodePairScope",
    "FreshV2RepresentationAssessment",
    "FreshV2RepresentationContract",
    "FreshV2SaltRequirement",
    "FreshV2SaltScope",
    "FreshV2TextRequirement",
    "assess_v2_representation",
    "normalize_scientific_text",
]
