"""把现有设计工具结果接入结构化 decision gate 的兼容运行层。



当前公开工具仍返回原有 JSON 契约。这里在 ToolNode 执行后读取同一批工具调用的

结构化参数和结果，把它们转换为 GraphState 字段。该兼容层将在工具原生返回

状态更新后删除；数值/字段策略委托给 ``core.design_decision``；显式 semantic 模式采用有出处的模型审查。

"""



from __future__ import annotations



import ast

import hashlib

import json

import os

import time

from collections.abc import Mapping

from typing import Any, Literal



from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from orion.core.task_scope import current_task_messages, current_task_state, internal_human_message

from langgraph.types import Command



from orion.core.design_decision import (

    CandidateEvaluation,

    CandidateState,

    ConstraintEvaluation,

    ConstraintSpec,

    ConstraintSupport,

    ContextFieldAuthorization,

    DecisionState,

    EvidenceGap,

    EvidenceLedgerItem,

    JointContextRequirement,

    TaskContract,

    decide_design,

)

from orion.task_evidence.condition_extraction import extract_condition_scope

from orion.tools.electrolyte_design.object_identity import (

    resolve_candidate_payload_object_identity,

)





_CAPTURED_TOOLS = {

    "query_formulation_space",

    "generate_candidate_space",

    "enrich_formulation_evidence",

    "predict_formulation_conductivity",

    "screen_formulations",

}

_VALID_EVIDENCE_KINDS = {

    "formulation",

    "property",

    "condition",

    "mechanism",

    "risk",

    "background",

}

_OBJECT_SENSITIVE_CONSTRAINT_KINDS = {

    "minimum_voltage",

    "voltage_window",

    "c_rate",

    "minimum_cycles",

    "minimum_retention",

    "minimum_conductivity",

    "maximum_viscosity",

    "temperature",

    "mechanism_keyword",

    "evidence_claim",

}

_EVALUATION_TERMINAL_RESERVE_SECONDS = float(

    os.environ.get("ORION_EVALUATION_TERMINAL_RESERVE_SECONDS", "45")

)





def _as_json_value(value: Any) -> Any:

    if hasattr(value, "model_dump"):

        return value.model_dump(mode="json")

    if isinstance(value, (list, dict)):

        return value

    if not isinstance(value, str):

        return value

    text = value.strip()

    if not text:

        return None

    try:

        return json.loads(text)

    except json.JSONDecodeError:

        try:

            return ast.literal_eval(text)

        except (SyntaxError, ValueError):

            return None





def _stable_id(prefix: str, *parts: str) -> str:

    raw = "\x1f".join(parts).encode("utf-8")

    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:16]}"





def _constraint_id(text: str) -> str:

    return _stable_id("constraint", text.strip().lower())





def _tool_args(call: Mapping[str, Any]) -> dict[str, Any]:

    args = call.get("args") or {}

    if isinstance(args, str):

        args = _as_json_value(args)

    return args if isinstance(args, dict) else {}





def _successful_terminal_payload(

    message: ToolMessage,

) -> tuple[str, dict[str, Any] | None] | None:

    """Return a successful terminal compiler payload, otherwise ``None``.



    ToolNode versions differ in how reliably they preserve tool-call ids and

    artifacts.  This check is restricted to the current terminal ToolMessage;

    it never infers terminal success from free-form model text.

    """



    if getattr(message, "status", None) == "error":

        return None

    artifact = getattr(message, "artifact", None)

    if hasattr(artifact, "model_dump"):

        artifact = artifact.model_dump(mode="json")

    if isinstance(artifact, dict) and artifact.get("status") == "error":

        return None



    content = str(getattr(message, "content", "") or "").strip()

    lowered = content.casefold()

    # Successful reports legitimately contain gap/reason codes such as

    # ``missing_evidence``.  Only an explicit error artifact or an error-shaped

    # content prefix may invalidate the terminal result.

    if not content or lowered.startswith(

        (

            "错误",

            "error",

            "traceback",

            "exception",

            "validationerror",

            "tool execution failed",

        )

    ):

        return None

    try:

        payload = json.loads(content)

    except (ValueError, TypeError):

        payload = None

    if isinstance(payload, dict) and (payload.get("error") or payload.get("status") == "error"):

        return None

    return content, artifact if isinstance(artifact, dict) else None





def terminal_result(state: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None, str | None] | None:

    """在同一任务内成对选择正文、artifact 和提交调用 ID。



    兼容后端规范化返回 ID，但仅在当前批次恰有一个报告调用时按名称关联。

    不从旧轮次、最长文本或另一条 ToolMessage 单独借用 artifact。

    """

    calls: dict[str, dict] = {}

    result = None

    for message in current_task_messages(state):

        if isinstance(message, AIMessage):

            if (not message.tool_calls and message.additional_kwargs.get("orion_final_response")

                    and str(message.content or "").strip()):

                # 最终回答就是 Agent 的原始正文；没有调用报告工具就没有工具调用 ID。

                result = (str(message.content), None, None)

            calls = {str(c.get("id") or c.get("tool_call_id") or ""): c

                     for c in message.tool_calls or []}

            continue

        if not isinstance(message, ToolMessage):

            continue

        call_id = str(message.tool_call_id or "")

        matched = calls.get(call_id)

        report_calls = [key for key, call in calls.items()

                        if call.get("name") == "synthesize_design_report"]

        if matched is not None:

            if matched.get("name") != "synthesize_design_report":

                continue

        elif message.name == "synthesize_design_report" and len(report_calls) == 1:

            pass  # 保留后端返回 ID，与既有 terminal_commit_tool_call_id 行为一致。

        elif message.name != "synthesize_design_report" or calls:

            continue

        # 无调用记录的命名 ToolMessage 保留旧版离线/直接工具兼容。

        payload = _successful_terminal_payload(message)

        if payload is not None:

            result = (*payload, call_id)

    return result





def _build_candidates(

    raw_candidates: list[Any],

) -> tuple[list[CandidateState], list[EvidenceLedgerItem]]:

    candidates: list[CandidateState] = []

    ledger: list[EvidenceLedgerItem] = []

    seen_evidence: set[str] = set()



    for raw in raw_candidates:

        if not isinstance(raw, dict):

            continue

        # Agent 可在候选生成后继续补充 property/mechanism evidence；在证据进入

        # GraphState 前重新运行一次高精度显式对象解析，避免 identity 只存在于

        # query_formulation_space 的早期 assembly 阶段。

        raw = resolve_candidate_payload_object_identity(raw)

        candidate_id = str(raw.get("candidate_id") or raw.get("scheme_id") or "").strip()

        if not candidate_id:

            continue

        scientific_object_id = str(

            raw.get("scientific_object_id") or raw.get("object_id") or ""

        ).strip() or None



        formulation_ids: list[str] = []

        for evidence in raw.get("evidence") or []:

            if not isinstance(evidence, dict):

                continue

            chunk_id = str(evidence.get("chunk_id") or "").strip()

            paper_id = str(

                evidence.get("paper_id") or evidence.get("paper_title") or ""

            ).strip()

            # chunk_id 的前缀（如 paper_id::sec::child）是同一篇论文的稳定标识；

            # 用其覆盖调用方可能截断或不一致的 paper_id，避免跨论文判定误判。

            if chunk_id and "::" in chunk_id:

                paper_id = chunk_id.split("::", 1)[0]

            excerpt = str(

                evidence.get("excerpt")

                or evidence.get("text")

                or evidence.get("content")

                or ""

            ).strip()

            if not chunk_id or not paper_id or not excerpt:

                continue

            kind = str(evidence.get("evidence_kind") or "background")

            if kind not in _VALID_EVIDENCE_KINDS:

                kind = "background"

            evidence_id = _stable_id("evidence", candidate_id, chunk_id, kind)

            evidence_object_id = str(

                evidence.get("scientific_object_id") or evidence.get("object_id") or ""

            ).strip() or None

            # 配方证据定义当前候选对象；非配方证据只有上游显式携带对象身份时

            # 才继承，避免“挂在候选列表里”本身被误当成科学归属证明。

            if kind == "formulation" and not evidence_object_id:

                evidence_object_id = scientific_object_id

            experiment_id = str(evidence.get("experiment_id") or "").strip() or None

            condition_scope = evidence.get("condition_scope")

            if hasattr(condition_scope, "model_dump"):

                condition_scope = condition_scope.model_dump(mode="json", exclude_none=True)

            if not isinstance(condition_scope, dict):

                condition_scope = None

            if condition_scope is None:

                extracted_scope = extract_condition_scope(excerpt)

                condition_scope = (

                    extracted_scope.model_dump(mode="json", exclude_none=True)

                    if extracted_scope is not None

                    else None

                )

            if evidence_id not in seen_evidence:

                ledger.append(

                    EvidenceLedgerItem(

                        evidence_id=evidence_id,

                        chunk_id=chunk_id,

                        paper_id=paper_id,

                        evidence_kind=kind,

                        excerpt=excerpt,

                        candidate_id=candidate_id,

                        scientific_object_id=evidence_object_id,

                        experiment_id=experiment_id,

                        condition_scope=condition_scope,

                    )

                )

                seen_evidence.add(evidence_id)

            if kind == "formulation":

                formulation_ids.append(evidence_id)



        unresolved = [

            EvidenceGap(

                code="candidate_unresolved_gap",

                description=str(item),

                candidate_id=candidate_id,

                search_hint=str(item),

            )

            for item in raw.get("unresolved_gaps") or []

            if str(item).strip()

        ]

        candidates.append(

            CandidateState(

                candidate_id=candidate_id,

                scientific_object_id=scientific_object_id,

                composition={

                    "salt": raw.get("salt"),

                    "salts": raw.get("salts") or [],

                    "salt_concentrations_mol_l": raw.get("salt_concentrations_mol_l") or {},

                    "concentration_mol_l": raw.get("concentration_mol_l"),

                    "solvents": raw.get("solvents") or [],

                    "solvent_ratios": raw.get("solvent_ratios") or {},

                    "solvent_ratio_basis": raw.get("solvent_ratio_basis") or "unknown",

                    "additives": raw.get("additives") or [],

                    "additive_amounts": raw.get("additive_amounts") or {},

                    # 绝对摩尔组成由 generate_candidate_space 生成；保留它，

                    # 下游预测工具才能从状态里的候选直接还原特征。

                    "component_mmol": raw.get("component_mmol") or {},

                    # Scientific-model annotations remain separate from the evidence

                    # ledger.  They may be rendered as model estimates downstream but

                    # never become ConstraintSupport without an evidence record.

                    "model_predictions": raw.get("model_predictions") or [],

                },

                formulation_evidence_ids=list(dict.fromkeys(formulation_ids)),

                unresolved_gaps=unresolved,

            )

        )

    return candidates, ledger





def _runtime_joint_context_requirement(item: dict[str, Any]) -> JointContextRequirement:

    """把新运行的 joint contract 升级为 coverage-first v2 语义。



    历史冻结 artifact 直接反序列化 ``TaskContract`` 时仍可保持 v1；只有当前

    runtime 从 tool request 新建合同才走这里。若调用方未为某个 joint constraint

    显式声明 applicability fields，则保守要求该 constraint 的支持 observation

    覆盖全部 joint ``context_fields``。调用方显式提供的 requirement-specific 子集

    保持不变，但不能通过省略某个 constraint 来旁路 coverage gate。

    """



    payload = dict(item)

    constraint_ids = [

        str(value).strip()

        for value in payload.get("constraint_ids") or []

        if str(value).strip()

    ]

    context_fields = [

        str(value).strip()

        for value in payload.get("context_fields") or []

        if str(value).strip()

    ]

    provided = {

        str(value.get("constraint_id") or "").strip(): dict(value)

        for value in payload.get("applicability_requirements") or []

        if isinstance(value, dict) and str(value.get("constraint_id") or "").strip()

    }

    payload["applicability_requirements"] = [

        provided.get(

            constraint_id,

            {

                "constraint_id": constraint_id,

                "required_fields": list(context_fields),

                "authorization_mode": "explicit_scope",

            },

        )

        for constraint_id in constraint_ids

    ]

    return JointContextRequirement.model_validate(payload)





def _build_contract(

    *,

    task_type: str,

    criteria: list[str],

    constraint_records: list[dict[str, Any]],

    joint_context_requirements: list[dict[str, Any]],

    exclusions: list[str],

) -> TaskContract:

    if constraint_records:

        constraints = []

        for item in constraint_records:

            explicit_scope = str(item.get("required_evidence_scope") or "").strip()

            kind = str(item.get("kind") or "").strip()

            # 对会把性能/机制归属到具体配方的结构化约束，默认升级为

            # ``same_object_if_resolved``。若当前没有可靠 object identity，门禁仍按

            # same-paper 工作；只有显式解析到冲突时才拒绝。调用方显式指定更严格

            # scope 时保持其选择，不覆盖。

            if not explicit_scope or explicit_scope == "candidate_same_paper":

                required_scope = (

                    "candidate_same_object_if_resolved"

                    if kind in _OBJECT_SENSITIVE_CONSTRAINT_KINDS

                    else "candidate_same_paper"

                )

            else:

                required_scope = explicit_scope

            explicit_kinds = [

                str(value).strip()

                for value in (item.get("required_evidence_kinds") or [])

                if str(value).strip()

            ]

            # Evidence-role requirements are explicit contract semantics. Do not

            # infer them from constraint kind: historical candidate assemblers can

            # legitimately encode the same observation under different evidence

            # roles, and a guessed role would create false terminal blocks.

            required_kinds = explicit_kinds

            constraints.append(

                ConstraintSpec(

                    constraint_id=str(item["constraint_id"]),

                    description=str(item["description"]),

                    required_evidence_scope=required_scope,

                    required_evidence_kinds=required_kinds,

                    can_replan=bool(item.get("can_replan", True)),

                    search_hint=str(

                        item.get("search_hint")

                        or item.get("description")

                        or ""

                    ),

                )

            )

        return TaskContract(

            task_type="inverse_design" if task_type == "inverse_design" else "design",

            hard_constraints=constraints,

            joint_context_requirements=[

                _runtime_joint_context_requirement(item)

                for item in joint_context_requirements

            ],

            exclusions=exclusions,

        )

    unique_criteria = list(dict.fromkeys(item.strip() for item in criteria if item.strip()))

    constraints = [

        ConstraintSpec(

            constraint_id=_constraint_id(item),

            description=item,

            search_hint=item,

        )

        for item in unique_criteria

    ]

    return TaskContract(

        task_type="inverse_design" if task_type == "inverse_design" else "design",

        hard_constraints=constraints,

        joint_context_requirements=[

            _runtime_joint_context_requirement(item)

            for item in joint_context_requirements

        ],

        exclusions=exclusions,

    )





def _ledger_ids_by_chunk(

    ledger: list[EvidenceLedgerItem],

    candidate_id: str,

) -> dict[str, list[str]]:

    output: dict[str, list[str]] = {}

    for item in ledger:

        if item.candidate_id != candidate_id:

            continue

        output.setdefault(item.chunk_id, []).append(item.evidence_id)

    return output





def _build_evaluations(

    raw_results: list[Any],

    ledger: list[EvidenceLedgerItem],

) -> list[CandidateEvaluation]:

    evaluations: list[CandidateEvaluation] = []

    for raw in raw_results:

        if not isinstance(raw, dict):

            continue

        candidate_id = str(raw.get("candidate_id") or "").strip()

        if not candidate_id:

            continue

        ids_by_chunk = _ledger_ids_by_chunk(ledger, candidate_id)

        constraint_results: list[ConstraintEvaluation] = []

        for result in raw.get("criteria_evaluations") or []:

            if not isinstance(result, dict):

                continue

            criterion = str(result.get("criterion") or "").strip()

            status = str(result.get("status") or "").strip()

            if not criterion or status not in {

                "satisfied",

                "failed",

                "insufficient_evidence",

                "unsupported",

            }:

                continue

            constraint_id = str(

                result.get("constraint_id") or _constraint_id(criterion)

            )

            evidence_ids: list[str] = []

            supports: list[ConstraintSupport] = []

            seen_supports: set[tuple[str, str, str, str]] = set()

            for observed in result.get("observed") or []:

                if not isinstance(observed, dict):

                    continue

                chunk_id = str(observed.get("chunk_id") or "").strip()

                mapped_ids = ids_by_chunk.get(chunk_id, [])

                evidence_ids.extend(mapped_ids)

                observation_excerpt = str(observed.get("excerpt") or "").strip()

                explicit_scope = observed.get("condition_scope")

                if hasattr(explicit_scope, "model_dump"):

                    explicit_scope = explicit_scope.model_dump(mode="json", exclude_none=True)

                explicit_scope = dict(explicit_scope) if isinstance(explicit_scope, dict) else None



                raw_authorizations = list(observed.get("context_authorizations") or [])

                context_ids = list(dict.fromkeys(

                    context_id for context in observed.get('context_evidence') or []

                    if isinstance(context, dict)

                    for context_id in ids_by_chunk.get(str(context.get('chunk_id') or ''), [])

                ))

                evidence_ids.extend(context_ids)

                for evidence_id in mapped_ids:

                    if explicit_scope is not None:

                        support_scope: dict[str, object] | None = dict(explicit_scope)

                    elif observation_excerpt:

                        observation_scope = extract_condition_scope(observation_excerpt)

                        support_scope = (

                            observation_scope.model_dump(mode="json", exclude_none=True)

                            if observation_scope is not None

                            else {}

                        )

                    else:

                        support_scope = None



                    authorization_traces: list[ContextFieldAuthorization] = []

                    for raw_authorization in raw_authorizations:

                        try:

                            trace = ContextFieldAuthorization.model_validate(raw_authorization)

                        except Exception:

                            # 外部工具给出的无效 trace 必须 fail closed：丢弃授权而不是

                            # 修补/猜测；后续 atomic-trace coverage gate 会把字段视为未授权。

                            continue

                        if trace.obligation_id != constraint_id:

                            continue

                        if support_scope is None or trace.context_key not in support_scope:

                            continue

                        authorization_traces.append(trace)



                    scope_key = json.dumps(

                        support_scope,

                        ensure_ascii=False,

                        sort_keys=True,

                        default=str,

                    )

                    authorization_key = json.dumps(

                        [item.model_dump(mode="json") for item in authorization_traces],

                        ensure_ascii=False,

                        sort_keys=True,

                    )

                    support_key = (

                        evidence_id,

                        scope_key,

                        observation_excerpt,

                        authorization_key,

                    )

                    if support_key in seen_supports:

                        continue

                    seen_supports.add(support_key)

                    supports.append(

                        ConstraintSupport(

                            evidence_id=evidence_id,

                            condition_scope=support_scope,

                            observation_excerpt=observation_excerpt,

                            context_authorizations=authorization_traces,

                            context_evidence_ids=context_ids,

                        )

                    )

            constraint_results.append(

                ConstraintEvaluation(

                    constraint_id=constraint_id,

                    status=status,

                    evidence_ids=list(dict.fromkeys(evidence_ids)),

                    supports=supports,

                    reason=str(result.get("reason") or ""),

                )

            )

        evaluations.append(

            CandidateEvaluation(

                candidate_id=candidate_id,

                constraints=constraint_results,

            )

        )

    return evaluations





def capture_design_tool_result(

    state: Mapping[str, Any],

    *,

    tool_name: str,

    tool_args: dict[str, Any],

    tool_result: Any,

) -> dict[str, Any]:

    """把单次现有工具调用转换为结构化 GraphState 更新。"""



    if tool_name not in _CAPTURED_TOOLS:

        return {}

    payload = _as_json_value(tool_result)

    # 生成类工具返回 {"status", "candidates", "space", "warnings"} 的带元信息结构，

    # 候选在 candidates 字段里；其余工具直接返回候选列表。

    if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):

        payload = payload["candidates"]

    if not isinstance(payload, list):

        return {}

    request = tool_args.get("request") or tool_args

    if not isinstance(request, dict):

        return {}



    # 原文检索结果不是候选提交；不清空或改写已有候选、评估和 ledger。

    if tool_name == "query_formulation_space" and (

        any(isinstance(row, dict) and row.get("record_type") in {"source_evidence", "retrieval_error"}

            for row in payload)

        or (not payload and request.get("result_mode", "evidence") == "evidence")

    ):

        return {}



    if tool_name in {

        "query_formulation_space",

        "generate_candidate_space",

        "enrich_formulation_evidence",

        "predict_formulation_conductivity",

    }:

        candidates, ledger = _build_candidates(payload)

        default_task_type = (

            "inverse_design" if tool_name == "generate_candidate_space" else "design"

        )

        task_type = str(

            request.get("task_type")

            or state.get("design_task_type")

            or default_task_type

        )

        return {

            "design_task_type": (

                "inverse_design" if task_type == "inverse_design" else "design"

            ),

            "design_candidates": candidates,

            "evidence_ledger": ledger,

            "candidate_evaluations": None,

            "design_decision": None,

            "design_state_revision": int(state.get("design_state_revision") or 0) + 1,

        }



    # screen_formulations 评估的是调用方显式传入的候选；若 Agent 手动构造或替换

    # 了候选，必须以请求中的候选为准，否则 state 会保留旧 candidate_id，导致

    # synthesize_design_report 渲染错误。

    if tool_name == "screen_formulations":

        from orion.core.retrieved_evidence import restore_candidate_source_text

        request = restore_candidate_source_text(request, state)

    if request.get("candidates"):

        candidates, ledger = _build_candidates(request["candidates"])

    else:

        prior_candidates = state.get("design_candidates") or []

        candidates = [

            item if isinstance(item, CandidateState) else CandidateState.model_validate(item)

            for item in prior_candidates

        ]

        prior_ledger = state.get("evidence_ledger") or []

        ledger = [

            item

            if isinstance(item, EvidenceLedgerItem)

            else EvidenceLedgerItem.model_validate(item)

            for item in prior_ledger

        ]

        if not candidates:

            candidates, ledger = _build_candidates([])



    criteria = [

        str(item)

        for item in [

            *(request.get("criteria") or []),

            *(request.get("target_conditions") or []),

        ]

        if str(item).strip()

    ]

    constraint_records = [

        item

        for item in request.get("constraint_records") or []

        if isinstance(item, dict)

        and str(item.get("constraint_id") or "").strip()

        and str(item.get("description") or "").strip()

    ]

    joint_context_explicitly_supplied = "joint_context_requirements" in request

    joint_context_requirements = [

        item

        for item in request.get("joint_context_requirements") or []

        if isinstance(item, dict)

        and str(item.get("group_id") or "").strip()

        and len(item.get("constraint_ids") or []) >= 2

        and bool(item.get("context_fields"))

    ]

    if (

        not joint_context_explicitly_supplied

        and constraint_records

        and state.get("task_contract") is not None

    ):

        prior_raw = state.get("task_contract")

        prior_contract = (

            prior_raw

            if isinstance(prior_raw, TaskContract)

            else TaskContract.model_validate(prior_raw)

        )

        current_constraint_ids = {

            str(item.get("constraint_id") or "").strip()

            for item in constraint_records

        }

        joint_context_requirements = [

            requirement.model_dump(mode="json")

            for requirement in prior_contract.joint_context_requirements

            if set(requirement.constraint_ids).issubset(current_constraint_ids)

        ]

    structured_required = state.get("configured_task_type") in {"design", "inverse_design"}

    if not criteria and not constraint_records and not structured_required:

        return {}

    if structured_required and not constraint_records:

        # 整段目标文本的旧解析可能只识别一个词，不能代替新设计任务的完整条件。

        # 不猜测化学条件、不改写工具参数；要求规划者显式声明后再次筛选。

        contract = TaskContract(task_type=str(state.get("design_task_type") or "design"), hard_constraints=[ConstraintSpec(

            constraint_id="task-conditions-not-structured",

            description="本次设计任务尚未声明结构化筛选条件。",

            can_replan=True,

            search_hint=("重新读取用户问题，将所有必需的温度、电池配置、倍率及目标指标分别写入 "

                         "constraint_records 后重新筛选；整段问题不能代替这些条件。"),

        )])

    else:

        contract = _build_contract(

            task_type=str(state.get("design_task_type") or "design"),

            criteria=criteria,

            constraint_records=constraint_records,

            joint_context_requirements=joint_context_requirements,

            exclusions=[],

        )

    return {

        "task_contract": contract,

        "design_candidates": candidates,

        "evidence_ledger": ledger,

        "candidate_evaluations": _build_evaluations(payload, ledger),

        "design_attempts": int(state.get("design_attempts") or 0) + 1,

        "design_max_attempts": int(state.get("design_max_attempts") or 2),

        "design_decision": None,

        "design_state_revision": int(state.get("design_state_revision") or 0) + 1,

    }





def capture_design_tool_batch(

    state: Mapping[str, Any],

    output: Mapping[str, Any],

) -> dict[str, Any]:

    """把 ToolNode 输出与原始 tool_call 对齐，并附加结构化状态更新。"""



    state = current_task_state(state)

    result = dict(output)

    messages = list(output.get("messages") or [])

    if not messages:

        return result

    latest_ai = next(

        (

            message

            for message in reversed(current_task_messages(state))

            if isinstance(message, AIMessage)

        ),

        None,

    )

    if latest_ai is None:

        return result

    calls = {

        str(call.get("id") or call.get("tool_call_id") or ""): call

        for call in latest_ai.tool_calls

    }

    working = dict(state)

    for message in messages:

        if not isinstance(message, ToolMessage):

            continue

        call = calls.get(str(message.tool_call_id or ""))

        if not call:

            continue

        tool_name = str(call.get("name") or message.name or "")

        update = capture_design_tool_result(

            working,

            tool_name=tool_name,

            tool_args=_tool_args(call),

            tool_result=message.content,

        )

        if update:

            working.update(update)

            result.update(update)



    terminal = terminal_result({"messages": [latest_ai, *messages]})

    if terminal is not None:

        report, artifact, call_id = terminal

        result.update({

            "last_report": report,

            "task_artifact": artifact,

            "applicability_records": artifact.get("applicability_records") if artifact else None,

            "terminal_committed": True,

            "terminal_commit_tool_call_id": call_id or None,

        })

    # 单个报告调用的后端规范化 ID 也必须回写到同一条 AI 消息，

    # 否则失败结果在下一次模型请求的协议清理中会被当作 orphan 丢掉。

    report_calls = [call for call in latest_ai.tool_calls

                    if call.get("name") == "synthesize_design_report"]

    report_results = [m for m in messages if isinstance(m, ToolMessage)

                      and m.name == "synthesize_design_report"]

    if latest_ai.id and len(report_calls) == len(report_results) == 1:

        original, returned = report_calls[0], report_results[0]

        if returned.tool_call_id and returned.tool_call_id != original.get("id"):

            aligned = [({**call, "id": returned.tool_call_id} if call is original else call)

                       for call in latest_ai.tool_calls]

            result["messages"] = [*messages, latest_ai.model_copy(update={"tool_calls": aligned})]

    return result





def _feedback_gaps(decision: DecisionState) -> list[EvidenceGap]:

    """有限反馈优先展示最接近可行的候选，不让前几个候选耗尽展示名额。



    只调整缺口的展示顺序，不修改评估、候选审批或检索预算。

    """

    groups: dict[str, list[EvidenceGap]] = {}

    global_gaps: list[EvidenceGap] = []

    for gap in decision.evidence_gaps:

        if gap.candidate_id is None:

            global_gaps.append(gap)

        else:

            groups.setdefault(gap.candidate_id, []).append(gap)

    ranked = sorted(groups.values(), key=lambda gaps: (

        any(not gap.can_replan for gap in gaps),

        sum(gap.code == "constraint_failed" for gap in gaps),

        len(gaps),

    ))

    return [*global_gaps, *(gap for gaps in ranked for gap in gaps)]





def _decision_message(decision: DecisionState) -> HumanMessage:

    """将确定性准入结果交还模型；不指定工具或结束时机。"""

    gaps = "\n".join(f"- {gap.candidate_id or 'all'} / {gap.constraint_id or gap.code}: {gap.description}"

                     for gap in _feedback_gaps(decision)[:8])

    supported = ", ".join(decision.approved_candidate_ids) or "无"

    queries = list(dict.fromkeys([*(gap.search_hint for gap in _feedback_gaps(decision)

                                   if gap.search_hint), *decision.next_queries]))

    return internal_human_message(content=(

        f"（结构化设计门禁）当前评估状态={decision.status}；获准候选={supported}。"

        "只有获准候选可作为当前约束下的推荐；其余候选不得表述为已验证。"

        "请综合用户目标和已返回原文决定下一步：可按缺口补证、核验冲突、换候选，"

        "也可明确选择不推荐的受限报告。是否继续仍受执行预算约束。\n"

        f"证据缺口：\n{gaps}\n可参考的核查方向：\n" + "\n".join(queries[:8])))





def _current_synthesis_result(

    state: Mapping[str, Any],

) -> tuple[str, dict[str, Any] | None] | None:

    """读取当前 ToolNode 批次中的成功终端报告，避免报告后再调用一次 LLM。"""



    messages = current_task_messages(state)

    latest_ai_index = next(

        (

            index

            for index in range(len(messages) - 1, -1, -1)

            if isinstance(messages[index], AIMessage)

        ),

        -1,

    )

    if latest_ai_index < 0:

        return None

    result = terminal_result({"messages": messages[latest_ai_index:]})

    return result[:2] if result else None





def _quote_fragment(excerpt: str, *, max_chars: int = 360) -> str:

    """从 ledger 原文确定性截取一段可逐字绑定的短事实，不做语义改写。"""



    text = str(excerpt or "").strip()

    if len(text) <= max_chars:

        return text

    clipped = text[:max_chars]

    boundary = max(clipped.rfind("。"), clipped.rfind("."), clipped.rfind(";"))

    if boundary >= 80:

        clipped = clipped[: boundary + 1]

    return clipped.strip()





def _called_tool_names(state: Mapping[str, Any]) -> set[str]:

    return {

        str(call.get("name") or "")

        for message in current_task_messages(state)

        if isinstance(message, AIMessage)

        for call in getattr(message, "tool_calls", []) or []

    }





def _screen_attempt_count(state: Mapping[str, Any]) -> int:

    """按消息历史统计实际发起过的 screen_formulations 次数。



    ``design_attempts`` 是结构化状态计数，但在 checkpoint/retry 或旧状态迁移时可能

    低于真实工具调用数。决策门禁取两者较大值，确保有界重规划不会因计数丢失而

    退化为依赖 LangGraph recursion_limit 才停止。

    """



    return sum(

        1

        for message in current_task_messages(state)

        if isinstance(message, AIMessage)

        for call in getattr(message, "tool_calls", []) or []

        if str(call.get("name") or "") == "screen_formulations"

    )





def _evaluation_terminal_reserve_reached(state: Mapping[str, Any]) -> bool:

    started = state.get("evaluation_started_at_monotonic")

    timeout = state.get("evaluation_timeout_seconds")

    if started is None or timeout is None:

        return False

    try:

        remaining = float(timeout) - (time.monotonic() - float(started))

    except (TypeError, ValueError):

        return False

    return remaining <= _EVALUATION_TERMINAL_RESERVE_SECONDS





def _deadline_hold(

    decision: DecisionState | None,

    state: Mapping[str, Any],

) -> DecisionState | None:

    """Convert a still-searching design to a truthful hold near eval deadline."""



    if (

        decision is None

        or decision.status != "search"

        or state.get("design_task_type") not in {"design", "inverse_design"}

        or not _evaluation_terminal_reserve_reached(state)

    ):

        return None

    return decision.model_copy(

        update={

            "status": "hold",

            "approved_candidate_ids": [],

            "next_queries": [],

            "reason_codes": list(

                dict.fromkeys(

                    [

                        *decision.reason_codes,

                        "evaluation_terminal_reserve_reached",

                    ]

                )

            ),

        }

    )





def _auto_inverse_insufficient_report_message(

    ledger: list[EvidenceLedgerItem],

    *,

    revision: int,

) -> AIMessage:

    """Safely terminalize inverse design near the evaluation deadline.



    This control path never invents a root-cause hypothesis. It forwards only

    observed evidence to the deterministic compiler and deliberately leaves the

    hypothesis ledger incomplete, so the structured gate returns

    ``insufficient_evidence`` rather than upgrading an unverified diagnosis.

    """



    evidence_chunks: list[dict[str, Any]] = []

    seen: set[tuple[str, str]] = set()

    for item in ledger:

        key = (str(item.chunk_id or ""), str(item.paper_id or ""))

        if not key[0] or key in seen:

            continue

        seen.add(key)

        evidence_chunks.append(

            {

                "chunk_id": item.chunk_id,

                "paper_id": item.paper_id,

                "content": item.excerpt,

            }

        )

        if len(evidence_chunks) >= 12:

            break

    return AIMessage(

        content="",

        tool_calls=[

            {

                "id": f"call-auto-inverse-report-{revision}",

                "name": "synthesize_design_report",

                "args": {

                    "request": {

                        "task_type": "inverse_design",

                        "question": "",

                        "direct_answer": (

                            "现有结构化证据不足以确定故障根因；本次在评测墙钟保留区内"

                            "安全收尾，不把未验证假设提升为结论。"

                        ),

                        "evidence_chunks": evidence_chunks,

                        "hypotheses": [],

                        "confounders": [],

                        "discriminating_tests": [],

                    }

                },

            }

        ],

    )





def _auto_design_report_message(

    decision: DecisionState,

    ledger: list[EvidenceLedgerItem],

    *,

    revision: int,

) -> AIMessage:

    """为确定性 design terminal 决策生成标准报告工具调用。



    claims 只复制 ledger 中的逐字片段，绝不生成新的化学陈述；这样既保留

    claim-contract 可审计性，也不会让 fallback 绕过原证据边界。

    """



    claims: list[dict[str, Any]] = []

    evidence_chunks: list[dict[str, Any]] = []

    seen_chunks: set[str] = set()

    authorized = (

        set(decision.approved_candidate_ids)

        if decision.status == "recommend"

        else None

    )

    for item in ledger:

        if authorized is not None and item.candidate_id not in authorized:

            continue

        if item.chunk_id in seen_chunks:

            continue

        quote = _quote_fragment(item.excerpt)

        if len(quote) < 12:

            continue

        seen_chunks.add(item.chunk_id)

        evidence_chunks.append(

            {

                "chunk_id": item.chunk_id,

                "content": item.excerpt,

                "paper_id": item.paper_id,

            }

        )

        claims.append(

            {

                "claim_type": "fact",

                "text": quote,

                "chunk_id": item.chunk_id,

                "evidence_quote": quote,

                "binding_requirement": "quote_bound",

            }

        )

        if len(claims) >= 3:

            break



    next_steps = list(

        dict.fromkeys(

            gap.search_hint

            for gap in decision.evidence_gaps

            if gap.search_hint

        )

    )

    return AIMessage(

        content="",

        tool_calls=[

            {

                "id": f"call-auto-design-report-{revision}",

                "name": "synthesize_design_report",

                "args": {

                    "request": {

                        "task_type": "design",

                        "question": "",

                        "evidence_chunks": evidence_chunks,

                        "claims": claims,

                        "next_steps": next_steps,

                    }

                },

            }

        ],

    )





def scientific_design_policy(

    state: Mapping[str, Any],

) -> Command[Literal["llm", "tools", "grounding"]]:

    """保留科学准入计算；非终端结果一律反馈 Agent，不生成任何工具调用。"""

    state = current_task_state(state)

    if state.get("terminal_committed") and state.get("last_report"):

        return Command(update={key: state.get(key) for key in (

            "last_report", "task_artifact", "terminal_committed", "terminal_commit_tool_call_id",

            "applicability_records")}, goto="grounding")

    terminal = terminal_result(state)

    if terminal is not None:

        body, artifact, call_id = terminal

        return Command(update={"last_report": body, "task_artifact": artifact,

            "terminal_committed": True, "terminal_commit_tool_call_id": call_id,

            "applicability_records": artifact.get("applicability_records")

                if isinstance(artifact, dict) else None}, goto="grounding")

    contract_raw = state.get("task_contract")

    evaluations_raw = state.get("candidate_evaluations")

    if not contract_raw or evaluations_raw is None:

        return Command(goto="llm")

    revision = int(state.get("design_state_revision") or 0)

    if state.get("design_decision_revision", -1) == revision:

        return Command(goto="llm")

    contract = (

        contract_raw

        if isinstance(contract_raw, TaskContract)

        else TaskContract.model_validate(contract_raw)

    )

    candidates = [

        item if isinstance(item, CandidateState) else CandidateState.model_validate(item)

        for item in state.get("design_candidates") or []

    ]

    evaluations = [

        item

        if isinstance(item, CandidateEvaluation)

        else CandidateEvaluation.model_validate(item)

        for item in evaluations_raw

    ]

    ledger = [

        item

        if isinstance(item, EvidenceLedgerItem)

        else EvidenceLedgerItem.model_validate(item)

        for item in state.get("evidence_ledger") or []

    ]

    if (state.get("design_task_type") == "design"

            and int(state.get("design_semantic_revision", -1)) == revision

            and state.get("design_semantic_results")):

        from orion.core.semantic_design_policy import semantic_decision

        decision = semantic_decision(state)

    else:

        decision = decide_design(

            contract=contract,

            candidates=candidates,

            evaluations=evaluations,

            evidence_ledger=ledger,

            attempts=max(

                int(state.get("design_attempts") or 0),

                _screen_attempt_count(state),

            ),

            max_attempts=int(state.get("design_max_attempts") or 2),

        )

    update: dict[str, Any] = {

        "design_decision": decision,

        "design_decision_revision": revision,

    }



    update["messages"] = [_decision_message(decision)]

    return Command(update=update, goto="llm")





# Backward-compatible name for historical benchmark/replay code. The policy is

# no longer materialized as a permanent LangGraph node in the main Harness.

design_decision_node = scientific_design_policy

