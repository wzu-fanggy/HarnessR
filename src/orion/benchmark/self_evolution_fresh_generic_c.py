"""Mechanical contracts for the fresh strong generic audited-evolution baseline.

Arm C is intentionally non-strawman.  Its paired baseline/candidate trajectories
receive byte-equivalent scientific evidence and the same generic task contract;
the exact sealed candidate is the only treatment difference.  Arm D inherits
this generic safeguard before its additional typed scientific admission step.

This module performs no LLM, RAG, KG, or network calls.  It validates source-role
walls, renders generic audit packets without ORION-only typed metadata, and
checks the fixed trajectory-output contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.benchmark.self_evolution_fresh_scope import FreshPersistentUpdateProposal


GenericCondition = Literal["baseline", "candidate"]

_FORBIDDEN_GENERIC_TEXT = (
    "resolved_non_disqualifying",
    "disqualifying_match",
    "quarantine_unresolved",
    "missing != excluded",
    "scientific relevance != scope invalidation",
    "orion relation",
    "orion applicability",
    "d expected",
    "desired arm outcome",
)

_ALLOWED_SOURCE_ROLES = {"development", "matched_admission_boundary"}


class FreshGenericCFamilyTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    # Paired generic audit is a burned proxy-transfer task. Development-source
    # validity is checked upstream; re-feeding the development paper to both
    # baseline/candidate would make the persistent candidate structurally
    # redundant and bias C toward reject-all. Plain evidence here is optional.
    evidence_source_ids: list[str] = Field(default_factory=list)
    matched_boundary_source_ids: list[str] = Field(default_factory=list)
    user_task: str = Field(min_length=1)
    generic_task_contract: list[str] = Field(min_length=1)


class FreshGenericCTrajectoryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_completion: str = Field(min_length=1)
    evidence_refs_used: list[str] = Field(default_factory=list)
    explicit_boundaries: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(min_length=1)
    unsupported_claims: list[str] = Field(default_factory=list)


class FreshGenericCFixedContractScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    evidence_refs_pass: bool
    unsupported_claims_pass: bool
    reason_codes: list[str] = Field(default_factory=list)


class FreshGenericCHoldoutScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    fixed_contract_passed: bool
    required_conditions_passed: bool
    insufficiency_judgment_passed: bool
    reason_codes: list[str] = Field(default_factory=list)


class FreshGenericCAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "orion_self_evolution_fresh_generic_c_audit_v1"
    passed: bool
    family_count: int
    matched_boundary_families: list[str]
    source_role_violations: list[str] = Field(default_factory=list)
    forbidden_text_violations: list[str] = Field(default_factory=list)
    packet_parity_violations: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


def _source_index(source_bundle: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    index: dict[str, tuple[str, dict[str, Any]]] = {}
    for family_id, family in source_bundle.get("families", {}).items():
        for source in family.get("sources", []):
            source_id = str(source.get("source_id") or "")
            if not source_id:
                continue
            if source_id in index:
                raise ValueError(f"duplicate source_id in source bundle: {source_id}")
            index[source_id] = (family_id, source)
    return index


def _plain_evidence(source: dict[str, Any]) -> dict[str, Any]:
    """Expose scientific prose but remove ORION-only typed boundary/scope metadata."""

    payload: dict[str, Any] = {
        "source_id": str(source["source_id"]),
        "source_identity": str(source.get("source_identity") or ""),
        "title": str(source.get("title") or ""),
        "facts": list(source.get("facts") or []),
    }
    explicit_absences = list(source.get("explicit_absences") or [])
    if explicit_absences:
        payload["explicit_absences"] = explicit_absences
    return payload


def _family_task(spec: dict[str, Any], family_id: str) -> FreshGenericCFamilyTask:
    raw = spec.get("family_tasks", {}).get(family_id)
    if not isinstance(raw, dict):
        raise KeyError(f"missing generic C family task: {family_id}")
    return FreshGenericCFamilyTask.model_validate(raw)


def render_generic_c_trajectory_packet(
    *,
    spec: dict[str, Any],
    source_bundle: dict[str, Any],
    family_id: str,
    condition: GenericCondition,
    candidate: FreshPersistentUpdateProposal | dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = _family_task(spec, family_id)
    sources = _source_index(source_bundle)

    evidence: list[dict[str, Any]] = []
    for source_id in task.evidence_source_ids + task.matched_boundary_source_ids:
        if source_id not in sources:
            raise KeyError(f"generic C task references unknown source_id: {source_id}")
        source_family, source = sources[source_id]
        if source_family != family_id:
            raise ValueError(f"generic C source belongs to another family: {source_id}")
        evidence.append(_plain_evidence(source))

    packet: dict[str, Any] = {
        "family_id": family_id,
        "task_id": task.task_id,
        "condition": condition,
        "user_task": task.user_task,
        "generic_task_contract": task.generic_task_contract,
        "evidence": evidence,
        "sealed_candidate": None,
    }
    if condition == "candidate":
        if candidate is None:
            raise ValueError("candidate condition requires exact sealed candidate")
        if not isinstance(candidate, FreshPersistentUpdateProposal):
            candidate = FreshPersistentUpdateProposal.model_validate(candidate)
        packet["sealed_candidate"] = candidate.model_dump(mode="json")
    elif candidate is not None:
        raise ValueError("baseline condition must not receive candidate")
    return packet


def _packet_without_treatment_fields(packet: dict[str, Any]) -> dict[str, Any]:
    result = dict(packet)
    result.pop("condition", None)
    result.pop("sealed_candidate", None)
    return result


def _generic_text(task: FreshGenericCFamilyTask) -> str:
    return "\n".join([task.user_task, *task.generic_task_contract]).casefold()


def audit_generic_c_spec(
    *,
    spec: dict[str, Any],
    source_bundle: dict[str, Any],
    root: str | Path = ".",
) -> FreshGenericCAuditReport:
    source_roles: list[str] = []
    forbidden: list[str] = []
    parity: list[str] = []
    source_index = _source_index(source_bundle)
    family_tasks = spec.get("family_tasks") or {}
    if not isinstance(family_tasks, dict) or not family_tasks:
        raise ValueError("generic C spec requires family_tasks")

    matched_families: list[str] = []
    for family_id, raw_task in family_tasks.items():
        task = FreshGenericCFamilyTask.model_validate(raw_task)
        visible_text = _generic_text(task)
        for token in _FORBIDDEN_GENERIC_TEXT:
            if token.casefold() in visible_text:
                forbidden.append(f"{family_id}:{token}")

        for source_id in task.evidence_source_ids:
            item = source_index.get(source_id)
            if item is None:
                source_roles.append(f"{family_id}:{source_id}:unknown")
                continue
            source_family, source = item
            role = str(source.get("role") or "")
            if source_family != family_id or role != "development":
                source_roles.append(f"{family_id}:{source_id}:{role}")

        if task.matched_boundary_source_ids:
            matched_families.append(family_id)
        for source_id in task.matched_boundary_source_ids:
            item = source_index.get(source_id)
            if item is None:
                source_roles.append(f"{family_id}:{source_id}:unknown")
                continue
            source_family, source = item
            role = str(source.get("role") or "")
            if source_family != family_id or role != "matched_admission_boundary":
                source_roles.append(f"{family_id}:{source_id}:{role}")

        # Render dummy sealed candidate only to verify paired packet parity.  It
        # never leaves this deterministic audit and carries no family science.
        development_source_ids = [
            source_id
            for source_id in task.evidence_source_ids
            if source_id in source_index
        ]
        dummy_candidate = FreshPersistentUpdateProposal(
            proposal_status="candidate",
            proposed_content="deterministic parity audit candidate",
            claimed_scope={"task_types": ["design"]},
            relevant_fields=[],
            source_evidence_ids=development_source_ids[:1] or ["PARITY_ONLY"],
            development_rationale="deterministic packet parity audit",
        )
        baseline = render_generic_c_trajectory_packet(
            spec=spec,
            source_bundle=source_bundle,
            family_id=family_id,
            condition="baseline",
        )
        candidate_packet = render_generic_c_trajectory_packet(
            spec=spec,
            source_bundle=source_bundle,
            family_id=family_id,
            condition="candidate",
            candidate=dummy_candidate,
        )
        if _packet_without_treatment_fields(baseline) != _packet_without_treatment_fields(candidate_packet):
            parity.append(family_id)

        rendered = json.dumps(candidate_packet["evidence"], ensure_ascii=False).casefold()
        if "typed_boundary" in rendered or "typed_scope_facts" in rendered:
            parity.append(f"{family_id}:typed_metadata_visible")

    judge_path = Path(root) / str(spec.get("judge_prompt_path") or "")
    if not judge_path.is_file():
        forbidden.append("judge_prompt_missing")
    else:
        judge_text = judge_path.read_text(encoding="utf-8").casefold()
        for token in _FORBIDDEN_GENERIC_TEXT:
            if token.casefold() in judge_text:
                forbidden.append(f"judge_prompt:{token}")
        if "orion" in judge_text:
            forbidden.append("judge_prompt:orion_name")

    visibility = spec.get("visibility_policy") or {}
    required_false = (
        "candidate_may_be_rewritten",
        "typed_boundary_metadata_visible",
        "typed_scientific_relation_labels_visible",
        "future_evaluator_task_or_source_visible",
        "future_evaluator_result_visible",
        "desired_arm_outcome_visible",
    )
    for key in required_false:
        if visibility.get(key) is not False:
            parity.append(f"visibility_policy_must_be_false:{key}")
    if visibility.get("exact_sealed_candidate_visible_only_in_candidate_condition") is not True:
        parity.append("candidate_visibility_policy_not_strict")
    if spec.get("shared_between_C_and_D") is not True:
        parity.append("generic_safeguard_not_shared_between_c_and_d")

    reasons: list[str] = []
    if source_roles:
        reasons.append("generic_c_source_role_violation")
    if forbidden:
        reasons.append("generic_c_hidden_semantics_or_prompt_leak")
    if parity:
        reasons.append("generic_c_paired_packet_or_visibility_violation")

    return FreshGenericCAuditReport(
        passed=not reasons,
        family_count=len(family_tasks),
        matched_boundary_families=sorted(matched_families),
        source_role_violations=sorted(set(source_roles)),
        forbidden_text_violations=sorted(set(forbidden)),
        packet_parity_violations=sorted(set(parity)),
        reason_codes=reasons or ["fresh_generic_c_spec_passed"],
    )


def score_generic_c_fixed_contract(
    *,
    output: FreshGenericCTrajectoryOutput | dict[str, Any],
    allowed_evidence_refs: list[str],
) -> FreshGenericCFixedContractScore:
    if not isinstance(output, FreshGenericCTrajectoryOutput):
        output = FreshGenericCTrajectoryOutput.model_validate(output)
    allowed = set(allowed_evidence_refs)
    refs = set(output.evidence_refs_used)
    evidence_refs_pass = refs <= allowed and (bool(refs) if allowed else not refs)
    unsupported_claims_pass = not output.unsupported_claims
    reasons: list[str] = []
    if not evidence_refs_pass:
        reasons.append("generic_c_evidence_reference_violation")
    if not unsupported_claims_pass:
        reasons.append("generic_c_unsupported_claims_present")
    return FreshGenericCFixedContractScore(
        passed=evidence_refs_pass and unsupported_claims_pass,
        evidence_refs_pass=evidence_refs_pass,
        unsupported_claims_pass=unsupported_claims_pass,
        reason_codes=reasons or ["generic_c_fixed_contract_passed"],
    )


def score_generic_c_holdout(
    *,
    output: FreshGenericCTrajectoryOutput | dict[str, Any],
    allowed_evidence_refs: list[str],
) -> FreshGenericCHoldoutScore:
    """Score the frozen generic holdout semantically, not only structurally.

    The holdout asks whether a record at 25 C / 0.5 C validates continuous
    operation at **50 C and 2 C**.  A strong generic safeguard must explicitly
    preserve both missing conditions and an insufficiency/unsupported judgment.
    """

    if not isinstance(output, FreshGenericCTrajectoryOutput):
        output = FreshGenericCTrajectoryOutput.model_validate(output)
    fixed = score_generic_c_fixed_contract(
        output=output,
        allowed_evidence_refs=allowed_evidence_refs,
    )
    rendered = " ".join(
        [
            output.task_completion,
            *output.explicit_boundaries,
            *output.next_actions,
        ]
    ).casefold()
    has_50c = re.search(r"(?<!\d)50\s*(?:°\s*)?c(?![a-z])", rendered) is not None
    has_2c = re.search(r"(?<![\d.])2\s*(?:°\s*)?c(?![a-z])", rendered) is not None
    required_conditions_passed = has_50c and has_2c
    insufficiency_tokens = (
        "not validated",
        "not supported",
        "unsupported",
        "insufficient",
        "missing",
        "no result",
        "no evidence",
        "cannot conclude",
        "cannot establish",
    )
    insufficiency_passed = any(token in rendered for token in insufficiency_tokens)

    reasons: list[str] = []
    if not fixed.passed:
        reasons.extend(fixed.reason_codes)
    if not required_conditions_passed:
        reasons.append("generic_holdout_missing_50c_or_2c_condition")
    if not insufficiency_passed:
        reasons.append("generic_holdout_missing_insufficiency_judgment")
    passed = fixed.passed and required_conditions_passed and insufficiency_passed
    return FreshGenericCHoldoutScore(
        passed=passed,
        fixed_contract_passed=fixed.passed,
        required_conditions_passed=required_conditions_passed,
        insufficiency_judgment_passed=insufficiency_passed,
        reason_codes=reasons or ["generic_c_holdout_passed"],
    )


__all__ = [
    "FreshGenericCAuditReport",
    "FreshGenericCFamilyTask",
    "FreshGenericCFixedContractScore",
    "FreshGenericCHoldoutScore",
    "FreshGenericCTrajectoryOutput",
    "audit_generic_c_spec",
    "render_generic_c_trajectory_packet",
    "score_generic_c_fixed_contract",
    "score_generic_c_holdout",
]
