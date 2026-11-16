"""Follow-up v2 的结构化 planning integration。

该模块只修复 ``admitted candidate -> first scientific action`` 的行动化接口，不改动
candidate formation、seal、typed applicability 或既有 confirmatory 结果。设计上借鉴
workflow/guideline memory 将“已选择的经验”作为独立 planning input 的做法，但这里的
planning seed 必须由 ORION 已封存的 scientific candidate 确定性编译，并且只在上游
governance 已经允许 candidate 可见时才进入 planner。

核心纪律：
- 不把 planning seed 当 current-task evidence；
- 不补写 source 中缺失的科学条件；
- 不读取 benchmark signature atoms，因此不能按 scorer 定制 action；
- 只增强第一次 scientific action，后续仍由 current-task evidence 主导；
- unresolved / disqualifying candidate 由上游 withholding，本模块不会自行放行。
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.benchmark.self_evolution_fresh_v2_scope import FreshPersistentUpdateProposalV2
from orion.benchmark.self_evolution_fresh_v2_treatment_delivery import candidate_sha256_v2
from orion.task_evidence.evidence_applicability import CONDITION_FIELDS, ConditionField


_SCIENTIFIC_FIRST_ACTION_TOOLS = {
    "search_literature_multiview",
    "search_literature_chunks",
    "search_paper_chunks",
    "query_formulation_space",
}

_SCOPE_ATTR = {
    "composition": "composition",
    "salt": "salt",
    "solvent": "solvent",
    "additive": "additive",
    "temperature_c": "temperature_c",
    "electrode": "electrode_pair",
    "cell_configuration": "cell_configuration",
    "voltage_window_v": "voltage_window_v",
    "rate": "rate",
    "cycle_count": "cycle_count",
    "measurement_definition": "measurement_definition",
}


class FollowupPlanningAnchorV2(BaseModel):
    """从 sealed candidate scope 直接复制出的单个 source-visible planning anchor。"""

    model_config = ConfigDict(extra="forbid")

    field: ConditionField
    value: str = Field(min_length=1)


class FollowupPlanningSeedV2(BaseModel):
    """供 planner 使用的独立结构化输入；它不是证据对象。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "orion_self_evolution_followup_planning_seed_v2"
    ] = "orion_self_evolution_followup_planning_seed_v2"
    candidate_seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_types: list[str] = Field(min_length=1)
    guidance: str = Field(min_length=1)
    anchors: list[FollowupPlanningAnchorV2] = Field(min_length=1)
    unknown_fields: list[ConditionField] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    current_task_evidence_required: Literal[True] = True
    candidate_is_current_task_evidence: Literal[False] = False


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _render_scope_value(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, (dict, list, tuple)):
        return _canonical(value)
    return str(value)


def compile_followup_planning_seed_v2(
    proposal: FreshPersistentUpdateProposalV2,
) -> FollowupPlanningSeedV2:
    """从 sealed proposal 确定性编译 planner seed，不进行科学推断。"""

    if proposal.proposal_status != "candidate" or proposal.claimed_scope is None:
        raise ValueError("follow-up planning seed requires a candidate proposal")
    if not proposal.relevant_fields:
        raise ValueError("follow-up planning seed requires explicit relevant_fields")

    scope = proposal.claimed_scope
    anchors: list[FollowupPlanningAnchorV2] = []
    for field in proposal.relevant_fields:
        attr = _SCOPE_ATTR[field]
        value = getattr(scope, attr)
        if value is None:
            continue
        anchors.append(
            FollowupPlanningAnchorV2(
                field=field,
                value=_render_scope_value(value),
            )
        )
    unknown_fields = [
        field
        for field in CONDITION_FIELDS
        if getattr(scope, _SCOPE_ATTR[field]) is None
    ]

    if not anchors:
        raise ValueError("follow-up planning seed requires at least one explicit anchor")

    return FollowupPlanningSeedV2(
        candidate_seal_sha256=candidate_sha256_v2(proposal),
        task_types=list(scope.task_types),
        guidance=proposal.proposed_content.strip(),
        anchors=anchors,
        unknown_fields=unknown_fields,
        limitations=list(proposal.limitations_visible_from_development),
    )


def followup_planning_seed_sha256(seed: FollowupPlanningSeedV2) -> str:
    """返回结构化 planning seed 的 canonical identity。"""

    raw = _canonical(seed.model_dump(mode="json")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _seed_query(seed: FollowupPlanningSeedV2) -> str:
    """生成一个 source-faithful evidence-acquisition branch，不引入 target 证据。"""

    # guidance 是 proposer 已封存的 source-faithful 内容。独立保留现有 task-native
    # query，因此这里只加入一个 candidate-anchored branch，不替代当前任务分支。
    return seed.guidance


def _append_unique(values: list[str], value: str) -> list[str]:
    normalized = [str(item) for item in values if str(item).strip()]
    if value not in normalized:
        normalized.insert(0, value)
    return normalized


def _component_preferences(seed: FollowupPlanningSeedV2) -> dict[str, list[str]]:
    by_field = {anchor.field: anchor.value for anchor in seed.anchors}
    result: dict[str, list[str]] = {}
    if "salt" in by_field:
        try:
            salt = json.loads(by_field["salt"])
        except json.JSONDecodeError:
            salt = None
        identity = salt.get("identity") if isinstance(salt, dict) else None
        if identity:
            result["preferred_salts"] = [str(identity)]
    if "solvent" in by_field:
        result["preferred_solvents"] = [by_field["solvent"]]
    if "additive" in by_field:
        result["preferred_additives"] = [by_field["additive"]]
    return result


def apply_followup_planning_seed_v2(
    *,
    action: dict[str, Any],
    seed: FollowupPlanningSeedV2,
) -> dict[str, Any]:
    """把 admitted seed 确定性接到第一次 scientific action 的参数上。

    该函数不调用模型，不读取 scorer，也不覆盖模型原有的 task-native branch；它只把一个
    source-faithful branch 或显式 component preference 加入 planner 已选择的 action。
    """

    updated = copy.deepcopy(action)
    name = str(updated.get("name") or "")
    if name not in _SCIENTIFIC_FIRST_ACTION_TOOLS:
        return updated
    args = dict(updated.get("args") or {})
    branch = _seed_query(seed)

    if name == "search_literature_multiview":
        queries = list(args.get("queries") or [])
        args["queries"] = _append_unique([str(item) for item in queries], branch)
    elif name in {"search_literature_chunks", "search_paper_chunks"}:
        existing = str(args.get("query") or "").strip()
        args["query"] = branch if not existing else f"{branch} ; task-native verification: {existing}"
    elif name == "query_formulation_space":
        request = dict(args.get("request") or {})
        for key, seeded_values in _component_preferences(seed).items():
            existing = [str(item) for item in list(request.get(key) or []) if str(item).strip()]
            for value in reversed(seeded_values):
                if value not in existing:
                    existing.insert(0, value)
            request[key] = existing
        args["request"] = request

    updated["args"] = args
    return updated


def apply_followup_planning_seed_to_tool_calls_v2(
    *,
    tool_calls: list[dict[str, Any]],
    seed: FollowupPlanningSeedV2,
) -> tuple[list[dict[str, Any]], bool]:
    """只增强本批次中的第一个 scientific action。"""

    updated = copy.deepcopy(tool_calls)
    for index, call in enumerate(updated):
        if str(call.get("name") or "") not in _SCIENTIFIC_FIRST_ACTION_TOOLS:
            continue
        updated[index] = apply_followup_planning_seed_v2(action=call, seed=seed)
        return updated, True
    return updated, False


__all__ = [
    "FollowupPlanningAnchorV2",
    "FollowupPlanningSeedV2",
    "apply_followup_planning_seed_to_tool_calls_v2",
    "apply_followup_planning_seed_v2",
    "compile_followup_planning_seed_v2",
    "followup_planning_seed_sha256",
]
