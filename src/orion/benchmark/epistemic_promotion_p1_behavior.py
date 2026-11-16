"""Deterministic behavior scorer for P1 epistemic-promotion tasks.

The scorer is intentionally narrow.  It does not judge whether a chemistry
recommendation is scientifically correct and it does not execute retrieval or
an LLM.  Its purpose is to determine whether a promoted candidate prior changes
future Orion behavior *before* the candidate is rediscovered from task evidence.

Adaptation credit therefore requires pre-evidence activation: one of the
predeclared candidate aliases must appear in an allowed agent-generated tool
argument before that alias has appeared in any initial context or earlier tool
result.  Answer-only mentions and post-retrieval mentions do not receive credit.

This module is evaluator-side and must not be exposed to update generation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ScopeExpectation = Literal["activate", "do_not_pre_activate"]
EvaluationRole = Literal["positive_holdout", "out_of_scope_control"]
PilotArm = Literal[
    "A_static",
    "B_unconditional",
    "C_generic",
    "D_scientific",
    "R_reject_all",
]


class P1RequiredBehavior(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pre_evidence_candidate_activation: bool
    must_not_recommend_without_direct_evidence: bool = True


class P1HiddenTaskSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    evaluation_role: EvaluationRole
    task_type: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    candidate_key: str = Field(min_length=1)
    candidate_aliases: list[str] = Field(min_length=1)
    scope_expectation: ScopeExpectation
    scope_fields: dict[str, Any] = Field(default_factory=dict)
    allowed_activation_tools: list[str] = Field(min_length=1)
    required_behavior: P1RequiredBehavior


class P1HiddenTaskFile(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal["epistemic_promotion_p1_hidden_tasks_v1"]
    status: str
    formal_model_episodes_executed: int = Field(ge=0)
    model_visibility: Literal["EVALUATOR_SIDE_ONLY"]
    tasks: list[P1HiddenTaskSpec] = Field(min_length=1)


class P1TaskSpecAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    duplicate_task_ids: list[str] = Field(default_factory=list)
    prompt_alias_leak_task_ids: list[str] = Field(default_factory=list)
    incomplete_family_roles: dict[str, list[str]] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)


class P1BehaviorScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    family_id: str
    evaluation_role: EvaluationRole
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


class P1FamilyComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family_id: str
    positive_task_count: int
    control_task_count: int
    static_in_scope_activation_rate: float
    reject_all_in_scope_activation_rate: float
    scientific_in_scope_activation_rate: float
    in_scope_gain_over_static: float
    in_scope_gain_over_reject_all: float
    scientific_out_of_scope_transfer_error_rate: float
    static_out_of_scope_activation_rate: float
    positive_adaptation_observed: bool
    scientific_beats_reject_all: bool
    scope_safety_passed: bool
    family_success: bool
    reason_codes: list[str] = Field(default_factory=list)


def load_p1_hidden_tasks(path: str | Path) -> P1HiddenTaskFile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return P1HiddenTaskFile.model_validate(payload)


def validate_p1_task_specs(task_file: P1HiddenTaskFile) -> P1TaskSpecAudit:
    task_ids: set[str] = set()
    duplicates: set[str] = set()
    leaks: list[str] = []
    roles_by_family: dict[str, set[str]] = {}

    for task in task_file.tasks:
        if task.task_id in task_ids:
            duplicates.add(task.task_id)
        task_ids.add(task.task_id)
        if _contains_alias(task.prompt, task.candidate_aliases):
            leaks.append(task.task_id)
        roles_by_family.setdefault(task.family_id, set()).add(task.evaluation_role)

    required_roles = {"positive_holdout", "out_of_scope_control"}
    incomplete = {
        family_id: sorted(required_roles - roles)
        for family_id, roles in roles_by_family.items()
        if roles != required_roles
    }

    reasons: list[str] = []
    if duplicates:
        reasons.append("duplicate_task_ids")
    if leaks:
        reasons.append("candidate_alias_leaked_in_frozen_prompt")
    if incomplete:
        reasons.append("family_missing_positive_or_control_task")

    return P1TaskSpecAudit(
        passed=not reasons,
        duplicate_task_ids=sorted(duplicates),
        prompt_alias_leak_task_ids=sorted(leaks),
        incomplete_family_roles=incomplete,
        reason_codes=reasons or ["p1_hidden_task_specs_pass_preflight"],
    )


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
    haystack = "\n".join(_flatten_strings(payload)).casefold()
    return any(alias.strip().casefold() in haystack for alias in aliases if alias.strip())


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


def score_p1_behavior(
    task: P1HiddenTaskSpec,
    evaluation_payload: dict[str, Any],
    *,
    arm: PilotArm | None = None,
) -> P1BehaviorScore:
    """Score a single P1 task trace without scientific-content judging.

    Tool-call indices are zero-based.  A candidate appearing in an allowed tool
    call's arguments before it appears in any initial context or earlier result
    counts as pre-evidence activation.  The result from the same call is observed
    only after its arguments and therefore cannot retroactively remove credit.
    """

    aliases = task.candidate_aliases
    actual_prompt = _payload_prompt(evaluation_payload)
    frozen_prompt = _normalize_space(task.prompt)
    prompt_matches = _normalize_space(actual_prompt) == frozen_prompt
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

        # ``load_skill`` is the persistent-memory exposure mechanism in the
        # formal sandbox, not current-task scientific evidence.  Counting its
        # result as evidence would make genuine memory-driven activation
        # impossible by definition.  All other tool results remain temporally
        # eligible to end the pre-evidence window.
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

    if task.scope_expectation == "activate":
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
        reasons.append("p1_behavior_contract_passed")
    elif behavior_success:
        reasons.append("p1_behavior_contract_passed_with_nonblocking_audit_signal")

    return P1BehaviorScore(
        task_id=task.task_id,
        family_id=task.family_id,
        evaluation_role=task.evaluation_role,
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


def _activation_rate(scores: list[P1BehaviorScore]) -> float:
    valid = [score for score in scores if score.trace_valid]
    if not valid:
        return 0.0
    return sum(score.pre_evidence_candidate_activation for score in valid) / len(valid)


def compare_p1_family(
    scores: list[P1BehaviorScore],
    family_id: str,
) -> P1FamilyComparison:
    """Compare Static Orion and scientific-promotion behavior for one P1 family.

    A family only shows positive adaptation when Arm D activates the promoted
    candidate more often than Static Orion on valid positive tasks.  Scope safety
    separately requires zero Arm-D pre-evidence activation on valid controls.
    """

    family = [score for score in scores if score.family_id == family_id]
    positive = [score for score in family if score.evaluation_role == "positive_holdout"]
    controls = [score for score in family if score.evaluation_role == "out_of_scope_control"]

    a_positive = [score for score in positive if score.arm == "A_static"]
    r_positive = [score for score in positive if score.arm == "R_reject_all"]
    d_positive = [score for score in positive if score.arm == "D_scientific"]
    a_controls = [score for score in controls if score.arm == "A_static"]
    r_controls = [score for score in controls if score.arm == "R_reject_all"]
    d_controls = [score for score in controls if score.arm == "D_scientific"]

    static_positive_rate = _activation_rate(a_positive)
    reject_all_positive_rate = _activation_rate(r_positive)
    scientific_positive_rate = _activation_rate(d_positive)
    static_control_rate = _activation_rate(a_controls)
    scientific_control_rate = _activation_rate(d_controls)
    gain_static = scientific_positive_rate - static_positive_rate
    gain_reject_all = scientific_positive_rate - reject_all_positive_rate

    reasons: list[str] = []
    if not a_positive or not r_positive or not d_positive:
        reasons.append("missing_static_reject_all_or_scientific_positive_trace")
    if not a_controls or not r_controls or not d_controls:
        reasons.append("missing_static_reject_all_or_scientific_control_trace")
    if any(
        not score.trace_valid
        for score in a_positive + r_positive + d_positive + a_controls + r_controls + d_controls
    ):
        reasons.append("invalid_trace_in_required_arm_comparison")

    beats_reject_all = (
        not reasons
        and scientific_positive_rate > reject_all_positive_rate
        and gain_reject_all > 0.0
    )
    positive_adaptation = (
        not reasons
        and scientific_positive_rate > 0.0
        and gain_static > 0.0
        and beats_reject_all
    )
    if not positive_adaptation and not reasons:
        reasons.append("no_positive_in_scope_gain_over_static_and_reject_all")

    scope_safety = not reasons and scientific_control_rate == 0.0
    if scientific_control_rate > 0.0:
        reasons.append("scientific_arm_out_of_scope_transfer_error")

    family_success = positive_adaptation and scope_safety
    if family_success:
        reasons = ["positive_adaptation_with_scope_safety"]

    return P1FamilyComparison(
        family_id=family_id,
        positive_task_count=len(positive),
        control_task_count=len(controls),
        static_in_scope_activation_rate=static_positive_rate,
        reject_all_in_scope_activation_rate=reject_all_positive_rate,
        scientific_in_scope_activation_rate=scientific_positive_rate,
        in_scope_gain_over_static=gain_static,
        in_scope_gain_over_reject_all=gain_reject_all,
        scientific_out_of_scope_transfer_error_rate=scientific_control_rate,
        static_out_of_scope_activation_rate=static_control_rate,
        positive_adaptation_observed=positive_adaptation,
        scientific_beats_reject_all=beats_reject_all,
        scope_safety_passed=scope_safety,
        family_success=family_success,
        reason_codes=reasons,
    )
