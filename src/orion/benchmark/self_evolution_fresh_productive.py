"""Deterministic productive-task scoring for fresh ORION self-evolution.

The pre-evidence candidate-activation scorer answers whether persistent learning
changed behavior.  This module adds a second question: did the trace contain a
predeclared, task-relevant scientific action rather than succeeding merely by
mentioning (or avoiding) a candidate?

The scorer is intentionally structural. It does not call an LLM and does not
judge open-ended chemistry correctness. Each formal task must bind its own
productive-action contract before semantic execution.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.benchmark.epistemic_promotion_p1_behavior import PilotArm
from orion.benchmark.self_evolution_fresh_behavior import (
    FreshBehaviorScore,
    FreshBehaviorTaskSpec,
    score_fresh_behavior,
)


CandidateActionPolicy = Literal["must_pre_activate", "must_not_pre_activate"]


class FreshProductiveActionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["orion_self_evolution_fresh_productive_action_contract_v1"]
    task_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    candidate_policy: CandidateActionPolicy
    scientific_action_tools: list[str] = Field(min_length=1)
    productive_action_tools: list[str] = Field(min_length=1)
    required_first_action_term_groups: list[list[str]] = Field(default_factory=list)
    require_nonempty_first_scientific_action: bool = True
    notes: str | None = None


class FreshProductiveActionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["orion_self_evolution_fresh_productive_action_score_v1"] = (
        "orion_self_evolution_fresh_productive_action_score_v1"
    )
    task_id: str
    family_id: str
    arm: PilotArm | None = None
    trace_valid: bool
    behavior_score: FreshBehaviorScore
    first_scientific_action_index: int | None = None
    first_scientific_action_tool: str | None = None
    first_scientific_action_nonempty: bool
    first_scientific_action_is_productive_tool: bool
    required_term_groups_passed: bool
    candidate_policy_passed: bool
    productive_task_success: bool
    reason_codes: list[str] = Field(default_factory=list)


def _flatten_strings(payload: Any) -> list[str]:
    result: list[str] = []
    if isinstance(payload, str):
        result.append(payload)
    elif isinstance(payload, dict):
        for key, value in payload.items():
            result.append(str(key))
            result.extend(_flatten_strings(value))
    elif isinstance(payload, (list, tuple, set)):
        for item in payload:
            result.extend(_flatten_strings(item))
    elif payload is not None:
        result.append(str(payload))
    return result


def _normalized_payload_text(payload: Any) -> str:
    text = " ".join(_flatten_strings(payload)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("tool_calls")
    if not isinstance(raw, list):
        return []
    calls: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            calls.append(item)
        elif hasattr(item, "model_dump"):
            dumped = item.model_dump(mode="json")
            if isinstance(dumped, dict):
                calls.append(dumped)
    return calls


def _first_scientific_action(
    payload: dict[str, Any],
    scientific_action_tools: set[str],
) -> tuple[int | None, str | None, Any]:
    for index, call in enumerate(_tool_calls(payload)):
        name = str(call.get("name") or call.get("tool") or call.get("tool_name") or "")
        if name in scientific_action_tools:
            return index, name, call.get("args", {})
    return None, None, None


def _term_groups_pass(text: str, groups: list[list[str]]) -> bool:
    for group in groups:
        if not group:
            return False
        if not any(term.strip().casefold() in text for term in group if term.strip()):
            return False
    return True


def score_fresh_productive_action(
    task: FreshBehaviorTaskSpec,
    contract: FreshProductiveActionContract | dict[str, Any],
    evaluation_payload: dict[str, Any],
    *,
    arm: PilotArm | None = None,
) -> FreshProductiveActionScore:
    if not isinstance(contract, FreshProductiveActionContract):
        contract = FreshProductiveActionContract.model_validate(contract)

    if contract.task_id != task.task_id or contract.family_id != task.family_id:
        raise ValueError("productive-action contract must bind the exact task_id and family_id")

    behavior = score_fresh_behavior(task, evaluation_payload, arm=arm)
    index, tool_name, args = _first_scientific_action(
        evaluation_payload,
        set(contract.scientific_action_tools),
    )
    action_text = _normalized_payload_text(args)
    nonempty = bool(action_text)
    productive_tool = tool_name in set(contract.productive_action_tools) if tool_name else False
    term_groups_passed = _term_groups_pass(action_text, contract.required_first_action_term_groups)

    if contract.candidate_policy == "must_pre_activate":
        candidate_policy_passed = behavior.pre_evidence_candidate_activation
    else:
        candidate_policy_passed = not behavior.pre_evidence_candidate_activation

    reasons: list[str] = []
    if not behavior.trace_valid:
        reasons.append("behavior_trace_invalid")
    if index is None:
        reasons.append("missing_scientific_action")
    if contract.require_nonempty_first_scientific_action and not nonempty:
        reasons.append("first_scientific_action_empty")
    if not productive_tool:
        reasons.append("first_scientific_action_not_productive_tool")
    if not term_groups_passed:
        reasons.append("first_scientific_action_missing_required_target_terms")
    if not candidate_policy_passed:
        reasons.append("candidate_action_policy_failed")
    if not behavior.behavior_success:
        reasons.append("pre_evidence_behavior_contract_failed")

    success = (
        behavior.trace_valid
        and index is not None
        and (nonempty or not contract.require_nonempty_first_scientific_action)
        and productive_tool
        and term_groups_passed
        and candidate_policy_passed
        and behavior.behavior_success
    )
    if success:
        reasons.append("productive_task_contract_passed")

    return FreshProductiveActionScore(
        task_id=task.task_id,
        family_id=task.family_id,
        arm=arm,
        trace_valid=behavior.trace_valid,
        behavior_score=behavior,
        first_scientific_action_index=index,
        first_scientific_action_tool=tool_name,
        first_scientific_action_nonempty=nonempty,
        first_scientific_action_is_productive_tool=productive_tool,
        required_term_groups_passed=term_groups_passed,
        candidate_policy_passed=candidate_policy_passed,
        productive_task_success=success,
        reason_codes=reasons,
    )


__all__ = [
    "FreshProductiveActionContract",
    "FreshProductiveActionScore",
    "score_fresh_productive_action",
]
