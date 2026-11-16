"""Deterministic future-behavior scoring for the fresh ORION self-evolution study.

The scientific study's primary causal endpoint is a *pre-evidence action*: did a
persistent learned asset change an allowed agent-generated scientific tool
argument before the current task itself exposed the target component/strategy?

This module deliberately reuses the already-tested temporal semantics from the
historical P1 scorer while providing fresh-study naming and allowing a task to
be scored independently (a negative-transfer family does not need a synthetic
positive partner just to satisfy an old file schema).

No LLM or retrieval call occurs here.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.benchmark.epistemic_promotion_p1_behavior import PilotArm


FreshBehaviorRole = Literal["positive_transfer", "out_of_scope_control"]
FreshExpectation = Literal["activate", "do_not_pre_activate"]


class FreshBehaviorTaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["orion_self_evolution_fresh_behavior_task_v1"]
    task_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    role: FreshBehaviorRole
    task_type: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    candidate_key: str = Field(min_length=1)
    candidate_aliases: list[str] = Field(min_length=1)
    expectation: FreshExpectation
    allowed_activation_tools: list[str] = Field(min_length=1)
    scope_fields: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class FreshTaskAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    reason_codes: list[str] = Field(default_factory=list)


class FreshBehaviorScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["orion_self_evolution_fresh_behavior_score_v1"] = (
        "orion_self_evolution_fresh_behavior_score_v1"
    )
    task_id: str
    family_id: str
    role: FreshBehaviorRole
    arm: PilotArm | None = None
    trace_valid: bool
    prompt_matches_frozen_task: bool
    candidate_alias_in_prompt: bool
    candidate_seen_in_initial_context: bool
    pre_evidence_candidate_activation: bool
    activation_tool_call_index: int | None = None
    activation_tool_name: str | None = None
    first_candidate_evidence_call_index: int | None = None
    candidate_seen_in_answer: bool = False
    answer_only_or_post_evidence_mention: bool = False
    behavior_success: bool
    reason_codes: list[str] = Field(default_factory=list)


def _role_to_legacy(role: FreshBehaviorRole) -> Literal["positive_holdout", "out_of_scope_control"]:
    return "positive_holdout" if role == "positive_transfer" else "out_of_scope_control"


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _flatten_strings(payload: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(payload, str):
        strings.append(payload)
    elif isinstance(payload, dict):
        for key, value in payload.items():
            strings.append(str(key))
            strings.extend(_flatten_strings(value))
    elif isinstance(payload, (list, tuple, set)):
        for item in payload:
            strings.extend(_flatten_strings(item))
    elif payload is not None:
        strings.append(str(payload))
    return strings


def _contains_alias(payload: Any, aliases: list[str]) -> bool:
    """Fresh-study alias matching with token boundaries.

    Historical P1 intentionally remains unchanged for frozen replay semantics.
    The fresh study must not treat short chemical abbreviations such as ``FEC``
    as substrings inside ordinary words such as ``effect``.  Exact multi-word
    aliases still match naturally, and abbreviations match when bounded by
    non-alphanumeric characters (spaces, punctuation, hyphens, JSON syntax).
    """

    haystack = "\n".join(_flatten_strings(payload)).casefold()
    for raw_alias in aliases:
        alias = raw_alias.strip().casefold()
        if not alias:
            continue
        pattern = rf"(?<![0-9a-z]){re.escape(alias)}(?![0-9a-z])"
        if re.search(pattern, haystack):
            return True
    return False


def _payload_prompt(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    sample = payload.get("sample")
    if isinstance(sample, dict) and isinstance(sample.get("question"), str):
        return str(sample["question"])
    question = payload.get("question")
    return str(question) if isinstance(question, str) else ""


def _payload_contexts(payload: Any) -> list[Any]:
    if not isinstance(payload, dict):
        return []
    contexts = payload.get("contexts")
    return list(contexts) if isinstance(contexts, list) else []


def _payload_tool_calls(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
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


def _payload_answer(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    answer = payload.get("answer")
    return str(answer) if isinstance(answer, str) else ""


def audit_fresh_behavior_task(task: FreshBehaviorTaskSpec) -> FreshTaskAudit:
    leaked = [alias for alias in task.candidate_aliases if _contains_alias(task.prompt, [alias])]
    reasons: list[str] = []
    if leaked:
        reasons.append("candidate_alias_leaked_in_frozen_prompt")
    if task.role == "positive_transfer" and task.expectation != "activate":
        reasons.append("positive_task_must_expect_activation")
    if task.role == "out_of_scope_control" and task.expectation != "do_not_pre_activate":
        reasons.append("control_task_must_expect_no_pre_activation")
    return FreshTaskAudit(
        passed=not reasons,
        reason_codes=reasons or ["fresh_behavior_task_passed_preflight"],
    )


def score_fresh_behavior(
    task: FreshBehaviorTaskSpec,
    evaluation_payload: dict[str, Any],
    *,
    arm: PilotArm | None = None,
) -> FreshBehaviorScore:
    """Score fresh pre-evidence behavior without historical substring alias bugs."""

    audit = audit_fresh_behavior_task(task)
    if not audit.passed:
        return FreshBehaviorScore(
            task_id=task.task_id,
            family_id=task.family_id,
            role=task.role,
            arm=arm,
            trace_valid=False,
            prompt_matches_frozen_task=False,
            candidate_alias_in_prompt=True,
            candidate_seen_in_initial_context=False,
            pre_evidence_candidate_activation=False,
            behavior_success=False,
            reason_codes=audit.reason_codes,
        )

    aliases = task.candidate_aliases
    actual_prompt = _payload_prompt(evaluation_payload)
    prompt_matches = _normalize_space(actual_prompt) == _normalize_space(task.prompt)
    prompt_leak = _contains_alias(task.prompt, aliases) or _contains_alias(actual_prompt, aliases)

    initial_contexts = _payload_contexts(evaluation_payload)
    candidate_in_initial_context = _contains_alias(initial_contexts, aliases)
    evidence_seen = candidate_in_initial_context

    activation_index: int | None = None
    activation_name: str | None = None
    first_evidence_index: int | None = None
    allowed_tools = set(task.allowed_activation_tools)

    for index, call in enumerate(_payload_tool_calls(evaluation_payload)):
        name = str(call.get("name") or call.get("tool") or call.get("tool_name") or "")
        args = call.get("args", {})
        result = call.get("result")

        if (
            activation_index is None
            and not evidence_seen
            and name in allowed_tools
            and _contains_alias(args, aliases)
        ):
            activation_index = index
            activation_name = name

        # ``load_skill`` exposes persistent guidance; it is not current-task
        # scientific evidence and therefore does not close the pre-evidence window.
        if name != "load_skill" and _contains_alias(result, aliases):
            if first_evidence_index is None:
                first_evidence_index = index
            evidence_seen = True

    pre_activation = activation_index is not None
    answer_seen = _contains_alias(_payload_answer(evaluation_payload), aliases)
    answer_only_or_post = answer_seen and not pre_activation

    reasons: list[str] = []
    if not prompt_matches:
        reasons.append("frozen_prompt_mismatch")
    if prompt_leak:
        reasons.append("candidate_alias_leaked_in_prompt")
    if candidate_in_initial_context:
        reasons.append("candidate_present_in_initial_context")

    trace_valid = prompt_matches and not prompt_leak and not candidate_in_initial_context
    if task.expectation == "activate":
        behavior_success = trace_valid and pre_activation
        if trace_valid and not pre_activation:
            reasons.append("required_pre_evidence_activation_missing")
    else:
        behavior_success = trace_valid and not pre_activation
        if trace_valid and pre_activation:
            reasons.append("out_of_scope_pre_evidence_activation")

    if answer_only_or_post:
        reasons.append("answer_only_or_post_evidence_mention_no_adaptation_credit")

    if behavior_success and not reasons:
        reasons.append("fresh_behavior_contract_passed")
    elif behavior_success:
        reasons.append("fresh_behavior_contract_passed_with_nonblocking_audit_signal")

    return FreshBehaviorScore(
        task_id=task.task_id,
        family_id=task.family_id,
        role=task.role,
        arm=arm,
        trace_valid=trace_valid,
        prompt_matches_frozen_task=prompt_matches,
        candidate_alias_in_prompt=prompt_leak,
        candidate_seen_in_initial_context=candidate_in_initial_context,
        pre_evidence_candidate_activation=pre_activation,
        activation_tool_call_index=activation_index,
        activation_tool_name=activation_name,
        first_candidate_evidence_call_index=first_evidence_index,
        candidate_seen_in_answer=answer_seen,
        answer_only_or_post_evidence_mention=answer_only_or_post,
        behavior_success=behavior_success,
        reason_codes=reasons,
    )


__all__ = [
    "FreshBehaviorTaskSpec",
    "FreshBehaviorScore",
    "FreshTaskAudit",
    "audit_fresh_behavior_task",
    "score_fresh_behavior",
]
