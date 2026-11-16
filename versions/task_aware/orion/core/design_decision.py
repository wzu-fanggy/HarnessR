"""电解液设计任务的结构化决策合同与确定性门禁。



本模块不调用 LLM、检索服务或 KG。它只消费已经产生的候选、证据台账和

硬约束评估，返回可复放的 ``search`` / ``recommend`` / ``hold`` 决策。

"""



from __future__ import annotations



import json

import re

from itertools import combinations, product

from typing import Annotated, Any, Literal



from pydantic import BaseModel, Field, model_validator





TaskType = Literal["design", "inverse_design"]

EvidenceKind = Literal[

    "formulation",

    "property",

    "condition",

    "mechanism",

    "risk",

    "background",

]

EvidenceScope = Literal[

    "candidate_same_paper",

    "candidate_same_object",

    "candidate_same_object_if_resolved",

    "component",

    "background",

]

ConstraintStatus = Literal[

    "satisfied",

    "failed",

    "insufficient_evidence",

    "unsupported",

]

DecisionStatus = Literal["search", "recommend", "hold"]

# Context fields are task-declared keys, not a closed global ontology.  The old

# ExperimentalContextField name is retained as a compatibility alias because many

# downstream functions and historical artifacts import it.  Open keys such as

# state_of_charge or calcination_duration can therefore participate in the same

# applicability/compatibility machinery without adding a new code-level enum member.

ContextFieldKey = Annotated[

    str,

    Field(min_length=1, max_length=160, pattern=r"^[A-Za-z][A-Za-z0-9_]*$"),

]

ExperimentalContextField = ContextFieldKey





class ConstraintSpec(BaseModel):

    """用户硬约束及其最低证据范围。"""



    constraint_id: str = Field(min_length=1)

    description: str = Field(min_length=1)

    required_evidence_scope: EvidenceScope = "candidate_same_paper"

    required_evidence_kinds: list[EvidenceKind] = Field(default_factory=list)

    can_replan: bool = True

    search_hint: str = ""





ContextApplicabilityRelation = Literal["exact", "interval_contains"]

ContextApplicabilityAuthorizationMode = Literal["explicit_scope", "atomic_trace"]





class ConstraintContextApplicability(BaseModel):

    """某一局部硬约束的证据在进入联合组合前必须显式覆盖的上下文字段。



    ``required_fields`` 表示该 constraint 的 observation 要被用于当前任务时，原文

    证据本身必须携带的条件字段。缺失字段是 ``unresolved coverage``，不能因为

    “没有观察到冲突”而获得授权。``field_relations`` 允许显式声明合法泛化关系；

    未声明时按 exact 处理。v1 合同不包含本结构，因此保持原有开放世界语义以便

    历史工件可复放。

    """



    constraint_id: str = Field(min_length=1)

    required_fields: list[ExperimentalContextField] = Field(min_length=1)

    authorization_mode: ContextApplicabilityAuthorizationMode = "atomic_trace"

    field_relations: dict[ExperimentalContextField, ContextApplicabilityRelation] = Field(

        default_factory=dict

    )



    @model_validator(mode="after")

    def _validate_applicability(self) -> "ConstraintContextApplicability":

        if len(self.required_fields) != len(set(self.required_fields)):

            raise ValueError("applicability required fields must be unique")

        unknown_rules = set(self.field_relations) - set(self.required_fields)

        if unknown_rules:

            raise ValueError(

                "applicability relation keys must be declared required_fields: "

                + ", ".join(sorted(unknown_rules))

            )

        return self





class JointContextRequirement(BaseModel):

    """要求一组硬约束存在可共同实例化的实验上下文见证。



    该合同不是要求所有证据来自同一个 experiment_id。v1 仅要求不同实验在声明为

    relevant 的上下文字段上不冲突。v2 可进一步通过 ``applicability_requirements``

    声明每个 constraint 必须由证据自身显式覆盖哪些关键字段，先过 coverage /

    applicability gate，再进入跨证据 compatibility gate。

    """



    group_id: str = Field(min_length=1)

    constraint_ids: list[str] = Field(min_length=2)

    context_fields: list[ExperimentalContextField] = Field(min_length=1)

    target_context: dict[str, object] = Field(default_factory=dict)

    applicability_requirements: list[ConstraintContextApplicability] = Field(

        default_factory=list

    )

    can_replan: bool = True

    search_hint: str = ""



    @model_validator(mode="after")

    def _validate_group(self) -> "JointContextRequirement":

        if len(self.constraint_ids) != len(set(self.constraint_ids)):

            raise ValueError("joint-context constraint ids must be unique")

        if len(self.context_fields) != len(set(self.context_fields)):

            raise ValueError("joint-context fields must be unique")

        unknown_targets = set(self.target_context) - set(self.context_fields)

        if unknown_targets:

            raise ValueError(

                "joint-context target keys must be declared context_fields: "

                + ", ".join(sorted(unknown_targets))

            )

        applicability_ids = [item.constraint_id for item in self.applicability_requirements]

        if len(applicability_ids) != len(set(applicability_ids)):

            raise ValueError("applicability constraint ids must be unique within a joint group")

        unknown_constraints = set(applicability_ids) - set(self.constraint_ids)

        if unknown_constraints:

            raise ValueError(

                "applicability requirements reference unknown joint constraints: "

                + ", ".join(sorted(unknown_constraints))

            )

        for item in self.applicability_requirements:

            unknown_fields = set(item.required_fields) - set(self.context_fields)

            if unknown_fields:

                raise ValueError(

                    "applicability required fields must be declared context_fields: "

                    + ", ".join(sorted(unknown_fields))

                )

        return self





class TaskContract(BaseModel):

    """由用户目标解析出的、与自然语言消息分离的任务合同。"""



    task_type: TaskType

    hard_constraints: list[ConstraintSpec] = Field(min_length=1)

    joint_context_requirements: list[JointContextRequirement] = Field(default_factory=list)

    soft_objectives: list[str] = Field(default_factory=list)

    exclusions: list[str] = Field(default_factory=list)



    @model_validator(mode="after")

    def _unique_constraint_ids(self) -> "TaskContract":

        ids = [item.constraint_id for item in self.hard_constraints]

        if len(ids) != len(set(ids)):

            raise ValueError("hard constraint ids must be unique")

        known = set(ids)

        group_ids = [item.group_id for item in self.joint_context_requirements]

        if len(group_ids) != len(set(group_ids)):

            raise ValueError("joint-context group ids must be unique")

        for group in self.joint_context_requirements:

            unknown = set(group.constraint_ids) - known

            if unknown:

                raise ValueError(

                    f"joint-context group {group.group_id} references unknown constraints: "

                    + ", ".join(sorted(unknown))

                )

        return self





class EvidenceLedgerItem(BaseModel):

    """一条可审计证据及其候选归属。



    ``candidate_id=None`` 表示背景/组分级证据；精确配方、性能或工况证据应绑定

    candidate_id。paper_id 是阻止跨论文拼接配方与性能的必要字段。

    """



    evidence_id: str = Field(min_length=1)

    chunk_id: str = Field(min_length=1)

    paper_id: str = Field(min_length=1)

    evidence_kind: EvidenceKind

    excerpt: str = Field(min_length=1)

    candidate_id: str | None = None

    # scientific_object_id/experiment_id 只接受上游显式身份；paper_id 不能替代对象或实验身份。

    scientific_object_id: str | None = None

    experiment_id: str | None = None

    condition_scope: dict[str, object] | None = None

    supported_fields: list[str] = Field(default_factory=list)

    supported_constraints: list[str] = Field(default_factory=list)





class EvidenceGap(BaseModel):

    """门禁输出的结构化失败原因，可直接反馈给 Agent 做下一轮检索。"""



    code: str = Field(min_length=1)

    description: str = Field(min_length=1)

    candidate_id: str | None = None

    constraint_id: str | None = None

    can_replan: bool = True

    search_hint: str = ""





class CandidateState(BaseModel):

    """候选配方在决策层所需的最小状态。"""



    candidate_id: str = Field(min_length=1)

    scientific_object_id: str | None = None

    composition: dict[str, object] = Field(default_factory=dict)

    formulation_evidence_ids: list[str] = Field(default_factory=list)

    unresolved_gaps: list[EvidenceGap] = Field(default_factory=list)





class ContextFieldAuthorization(BaseModel):

    """一个上下文字段获得 applicability 资格的可审计原子见证引用。"""



    obligation_id: str = Field(min_length=1)

    context_key: ExperimentalContextField

    event_id: str = Field(min_length=1)

    quantity_id: str = Field(min_length=1)

    context_identity_quote: str = Field(min_length=1)

    attachment_quote: str = Field(min_length=1)

    method: Literal["atomic_context_observation_v1"] = "atomic_context_observation_v1"





class ConstraintSupport(BaseModel):

    """单条硬约束的一项 observation-bound 支持记录。



    ``condition_scope`` 只是从 observation 中得到的候选上下文，不再自动等价于

    applicability authorization。对声明了 v2 applicability requirement 的联合合同，

    只有同时出现在 ``context_authorizations`` 中的字段才能进入后续 LEDGER-Opt。

    没有 applicability requirement 的历史 v1 合同继续保持原有回放语义。

    """



    evidence_id: str = Field(min_length=1)

    condition_scope: dict[str, object] | None = None

    observation_excerpt: str = ""

    context_authorizations: list[ContextFieldAuthorization] = Field(default_factory=list)

    # 跨段图注补全的原文记录，与性能句共同保留在最终证据见证中。

    context_evidence_ids: list[str] = Field(default_factory=list)



    @model_validator(mode="after")

    def _validate_context_authorizations(self) -> "ConstraintSupport":

        if not self.context_authorizations:

            return self

        if self.condition_scope is None:

            raise ValueError("context authorizations require observation-bound condition_scope")

        keys = [item.context_key for item in self.context_authorizations]

        if len(keys) != len(set(keys)):

            raise ValueError("context authorization keys must be unique per support")

        missing = set(keys) - set(self.condition_scope)

        if missing:

            raise ValueError(

                "authorized context keys must exist in condition_scope: "

                + ", ".join(sorted(missing))

            )

        return self





class ConstraintEvaluation(BaseModel):

    """确定性 evaluator 对单个候选硬约束的结果。"""



    constraint_id: str = Field(min_length=1)

    status: ConstraintStatus

    evidence_ids: list[str] = Field(default_factory=list)

    supports: list[ConstraintSupport] = Field(default_factory=list)

    reason: str = ""





class CandidateEvaluation(BaseModel):

    """一个候选的全部硬约束评估。"""



    candidate_id: str = Field(min_length=1)

    constraints: list[ConstraintEvaluation] = Field(default_factory=list)





class JointContextWitness(BaseModel):

    """一个 recommendation 的可审计联合实验上下文见证。"""



    witness_version: Literal[

        "experimental_context_join_v1",

        "context_qualified_evidence_v2",

    ] = "experimental_context_join_v1"

    group_id: str = Field(min_length=1)

    candidate_id: str = Field(min_length=1)

    constraint_ids: list[str] = Field(min_length=2)

    context_fields: list[ExperimentalContextField] = Field(min_length=1)

    target_context: dict[str, object] = Field(default_factory=dict)

    evidence_ids: list[str] = Field(min_length=1)

    merged_context: dict[str, object] = Field(default_factory=dict)

    applicability_coverage: dict[str, list[ExperimentalContextField]] = Field(

        default_factory=dict

    )



    @model_validator(mode="after")

    def _validate_witness(self) -> "JointContextWitness":

        if len(self.constraint_ids) != len(set(self.constraint_ids)):

            raise ValueError("joint-context witness constraint ids must be unique")

        if len(self.context_fields) != len(set(self.context_fields)):

            raise ValueError("joint-context witness fields must be unique")

        if len(self.evidence_ids) != len(set(self.evidence_ids)):

            raise ValueError("joint-context witness evidence ids must be unique")

        unknown_target = set(self.target_context) - set(self.context_fields)

        unknown_merged = set(self.merged_context) - set(self.context_fields)

        if unknown_target or unknown_merged:

            raise ValueError(

                "joint-context witness context keys must be declared context_fields"

            )

        return self





class JointContextRepairPlan(BaseModel):

    """联合实验上下文不可行时的最小证据替换方案。



    ``replace_constraint_ids`` 是在当前 target context 下，为恢复可行组合至少需要

    重新取得支持证据的一组硬约束。``anchor_context`` 由目标上下文和暂时保留的

    supporting records 合并而成，用于把下一轮检索从 generic search 变为明确的

    evidence acquisition obligation。

    """



    repair_version: Literal[

        "experimental_context_repair_v1",

        "context_qualified_repair_v2",

    ] = "experimental_context_repair_v1"

    group_id: str = Field(min_length=1)

    candidate_id: str = Field(min_length=1)

    replace_constraint_ids: list[str] = Field(min_length=1)

    retained_evidence_ids: list[str] = Field(default_factory=list)

    context_fields: list[ExperimentalContextField] = Field(min_length=1)

    anchor_context: dict[str, object] = Field(default_factory=dict)

    conflict_fields: list[ExperimentalContextField] = Field(default_factory=list)

    failure_mode: Literal["compatibility", "coverage"] = "compatibility"

    missing_context_fields: dict[str, list[ExperimentalContextField]] = Field(

        default_factory=dict

    )

    can_replan: bool = True

    search_queries: list[str] = Field(min_length=1)



    @model_validator(mode="after")

    def _validate_repair(self) -> "JointContextRepairPlan":

        if len(self.replace_constraint_ids) != len(set(self.replace_constraint_ids)):

            raise ValueError("repair constraint ids must be unique")

        if len(self.retained_evidence_ids) != len(set(self.retained_evidence_ids)):

            raise ValueError("retained evidence ids must be unique")

        if len(self.context_fields) != len(set(self.context_fields)):

            raise ValueError("repair context fields must be unique")

        if len(self.conflict_fields) != len(set(self.conflict_fields)):

            raise ValueError("repair conflict fields must be unique")

        if set(self.anchor_context) - set(self.context_fields):

            raise ValueError("repair anchor keys must be declared context_fields")

        if set(self.conflict_fields) - set(self.context_fields):

            raise ValueError("repair conflict fields must be declared context_fields")

        if set(self.missing_context_fields) - set(self.replace_constraint_ids):

            raise ValueError("missing-context keys must be replacement constraint ids")

        for fields in self.missing_context_fields.values():

            if set(fields) - set(self.context_fields):

                raise ValueError("missing-context fields must be declared context_fields")

        return self





from orion.core.semantic_evidence_review import SemanticEvidenceReview





class DecisionState(BaseModel):

    """decision gate 的不可矛盾输出。"""



    status: DecisionStatus

    approved_candidate_ids: list[str] = Field(default_factory=list)

    joint_context_witnesses: list[JointContextWitness] = Field(default_factory=list)

    joint_context_repair_plans: list[JointContextRepairPlan] = Field(default_factory=list)

    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)

    next_queries: list[str] = Field(default_factory=list)

    attempts: int = Field(ge=0)

    max_attempts: int = Field(ge=1)

    reason_codes: list[str] = Field(default_factory=list)

    # 语义判断与数值/字段评估分别保留，不把模型判断伪装成确定性见证。

    semantic_evidence_reviews: list[SemanticEvidenceReview] = Field(default_factory=list)



    @model_validator(mode="after")

    def _enforce_decision_invariants(self) -> "DecisionState":

        if self.status == "recommend" and not self.approved_candidate_ids:

            raise ValueError("recommend requires at least one approved candidate")

        if self.status != "recommend" and self.approved_candidate_ids:

            raise ValueError("only recommend may contain approved candidates")

        if self.status != "recommend" and self.joint_context_witnesses:

            raise ValueError("only recommend may contain joint-context witnesses")

        if self.status == "recommend" and self.joint_context_repair_plans:

            raise ValueError("recommend cannot contain joint-context repair plans")

        approved = set(self.approved_candidate_ids)

        witness_keys: list[tuple[str, str]] = []

        for witness in self.joint_context_witnesses:

            if witness.candidate_id not in approved:

                raise ValueError(

                    "joint-context witness must belong to an approved candidate"

                )

            witness_keys.append((witness.candidate_id, witness.group_id))

        if len(witness_keys) != len(set(witness_keys)):

            raise ValueError("joint-context witnesses must be unique per candidate/group")

        if self.status == "search" and not self.evidence_gaps:

            raise ValueError("search requires at least one evidence gap")

        return self





def _dedupe(values: list[str]) -> list[str]:

    return list(dict.fromkeys(value for value in values if value))





def _gap_for_constraint(

    *,

    code: str,

    candidate_id: str,

    constraint: ConstraintSpec,

    description: str,

) -> EvidenceGap:

    return EvidenceGap(

        code=code,

        description=description,

        candidate_id=candidate_id,

        constraint_id=constraint.constraint_id,

        can_replan=constraint.can_replan,

        search_hint=constraint.search_hint,

    )





def _candidate_gaps(

    *,

    candidate: CandidateState,

    evaluation: CandidateEvaluation | None,

    contract: TaskContract,

    ledger: dict[str, EvidenceLedgerItem],

) -> list[EvidenceGap]:

    gaps = list(candidate.unresolved_gaps)

    if evaluation is None:

        gaps.append(

            EvidenceGap(

                code="candidate_not_evaluated",

                description="候选尚未完成硬约束评估。",

                candidate_id=candidate.candidate_id,

                search_hint="evaluate candidate against all hard constraints",

            )

        )

        return gaps



    results = {item.constraint_id: item for item in evaluation.constraints}

    formulation_items = [

        ledger[evidence_id]

        for evidence_id in candidate.formulation_evidence_ids

        if evidence_id in ledger

        and ledger[evidence_id].evidence_kind == "formulation"

        and ledger[evidence_id].candidate_id in {None, candidate.candidate_id}

    ]

    formulation_papers = {item.paper_id for item in formulation_items}



    for constraint in contract.hard_constraints:

        result = results.get(constraint.constraint_id)

        if result is None:

            gaps.append(

                _gap_for_constraint(

                    code="constraint_not_evaluated",

                    candidate_id=candidate.candidate_id,

                    constraint=constraint,

                    description=f"硬约束“{constraint.description}”尚未评估。",

                )

            )

            continue



        if result.status != "satisfied":

            gaps.append(

                _gap_for_constraint(

                    code=f"constraint_{result.status}",

                    candidate_id=candidate.candidate_id,

                    constraint=constraint,

                    description=result.reason or f"硬约束“{constraint.description}”未满足。",

                )

            )

            continue



        referenced = [ledger[item] for item in result.evidence_ids if item in ledger]

        missing_ids = [item for item in result.evidence_ids if item not in ledger]

        if not result.evidence_ids or missing_ids:

            detail = (

                "没有证据引用"

                if not result.evidence_ids

                else "证据台账缺少：" + ", ".join(missing_ids)

            )

            gaps.append(

                _gap_for_constraint(

                    code="missing_evidence",

                    candidate_id=candidate.candidate_id,

                    constraint=constraint,

                    description=f"硬约束“{constraint.description}”{detail}。",

                )

            )

            continue



        candidate_mismatch = [

            item.evidence_id

            for item in referenced

            if item.candidate_id not in {None, candidate.candidate_id}

        ]

        if candidate_mismatch:

            gaps.append(

                _gap_for_constraint(

                    code="evidence_candidate_mismatch",

                    candidate_id=candidate.candidate_id,

                    constraint=constraint,

                    description=(

                        f"硬约束“{constraint.description}”引用了其它候选的证据："

                        + ", ".join(candidate_mismatch)

                    ),

                )

            )

            continue



        if constraint.required_evidence_kinds:

            present_kinds = {item.evidence_kind for item in referenced}

            missing_kinds = [

                kind

                for kind in constraint.required_evidence_kinds

                if kind not in present_kinds

            ]

            if missing_kinds:

                gaps.append(

                    _gap_for_constraint(

                        code="evidence_role_mismatch",

                        candidate_id=candidate.candidate_id,

                        constraint=constraint,

                        description=(

                            f"硬约束“{constraint.description}”缺少要求的证据角色："

                            + ", ".join(missing_kinds)

                            + "；当前引用角色："

                            + (", ".join(sorted(present_kinds)) or "none")

                        ),

                    )

                )

                continue



        if constraint.required_evidence_scope in {

            "candidate_same_paper",

            "candidate_same_object",

            "candidate_same_object_if_resolved",

        }:

            if not formulation_papers:

                gaps.append(

                    _gap_for_constraint(

                        code="missing_formulation_evidence",

                        candidate_id=candidate.candidate_id,

                        constraint=constraint,

                        description="候选缺少可确定论文归属的配方证据。",

                    )

                )

                continue

            supporting_papers = {item.paper_id for item in referenced}

            if not formulation_papers.intersection(supporting_papers):

                gaps.append(

                    _gap_for_constraint(

                        code="cross_paper_evidence",

                        candidate_id=candidate.candidate_id,

                        constraint=constraint,

                        description=(

                            f"硬约束“{constraint.description}”仅由其它论文支持，"

                            "不能迁移到当前候选。"

                        ),

                    )

                )

                continue



        if constraint.required_evidence_scope == "candidate_same_object":

            object_id = (candidate.scientific_object_id or "").strip()

            if not object_id:

                gaps.append(

                    _gap_for_constraint(

                        code="missing_scientific_object_identity",

                        candidate_id=candidate.candidate_id,

                        constraint=constraint,

                        description=(

                            "候选缺少显式 scientific_object_id；"

                            "不能用同论文关系替代同科学对象证明。"

                        ),

                    )

                )

                continue

            formulation_object_ids = {

                item.scientific_object_id

                for item in formulation_items

                if item.scientific_object_id

            }

            if object_id not in formulation_object_ids:

                gaps.append(

                    _gap_for_constraint(

                        code="missing_formulation_object_binding",

                        candidate_id=candidate.candidate_id,

                        constraint=constraint,

                        description=(

                            "候选配方证据未显式绑定到当前 scientific_object_id。"

                        ),

                    )

                )

                continue

            missing_support_object_ids = [

                item.evidence_id

                for item in referenced

                if not item.scientific_object_id

            ]

            if missing_support_object_ids:

                gaps.append(

                    _gap_for_constraint(

                        code="scientific_object_mismatch",

                        candidate_id=candidate.candidate_id,

                        constraint=constraint,

                        description=(

                            f"硬约束“{constraint.description}”包含未绑定 scientific_object_id "

                            "的支持证据：" + ", ".join(missing_support_object_ids)

                        ),

                    )

                )

                continue

            conflicting_support_ids = [

                item.evidence_id

                for item in referenced

                if item.scientific_object_id != object_id

            ]

            if conflicting_support_ids:

                gaps.append(

                    _gap_for_constraint(

                        code="scientific_object_mismatch",

                        candidate_id=candidate.candidate_id,

                        constraint=constraint,

                        description=(

                            f"硬约束“{constraint.description}”引用了其它 scientific object "

                            "的证据：" + ", ".join(conflicting_support_ids)

                        ),

                    )

                )

                continue



        if constraint.required_evidence_scope == "candidate_same_object_if_resolved":

            candidate_object_id = (candidate.scientific_object_id or "").strip()

            formulation_object_ids = {

                item.scientific_object_id

                for item in formulation_items

                if item.scientific_object_id

            }

            if candidate_object_id:

                conflicting_formulation_ids = {

                    item for item in formulation_object_ids if item != candidate_object_id

                }

                if conflicting_formulation_ids:

                    gaps.append(

                        _gap_for_constraint(

                            code="scientific_object_mismatch",

                            candidate_id=candidate.candidate_id,

                            constraint=constraint,

                            description=(

                                "候选 scientific_object_id 与配方证据对象身份冲突："

                                + ", ".join(sorted(conflicting_formulation_ids))

                            ),

                        )

                    )

                    continue

                resolved_object_id = candidate_object_id

            elif len(formulation_object_ids) == 1:

                resolved_object_id = next(iter(formulation_object_ids))

            else:

                resolved_object_id = ""



            if resolved_object_id:

                conflicting_support_ids = [

                    item.evidence_id

                    for item in referenced

                    if item.scientific_object_id

                    and item.scientific_object_id != resolved_object_id

                ]

                if conflicting_support_ids:

                    gaps.append(

                        _gap_for_constraint(

                            code="scientific_object_mismatch",

                            candidate_id=candidate.candidate_id,

                            constraint=constraint,

                            description=(

                                f"硬约束“{constraint.description}”已解析到 scientific object，"

                                "但部分支持证据属于其它对象："

                                + ", ".join(conflicting_support_ids)

                            ),

                        )

                    )

                    continue



    return gaps





def _normalize_context_value(

    value: object,

    *,

    field: ExperimentalContextField | None = None,

) -> object:

    if isinstance(value, str):

        normalized = " ".join(value.split()).casefold()

        if field == "rate":

            # 1C、1 C 与 C/1 是同一个倍率；mA、mA/g 等不同单位不做推算。

            direct = re.fullmatch(r"(\d+(?:\.\d+)?)\s*c", normalized)

            inverse = re.fullmatch(r"c\s*/\s*(\d+(?:\.\d+)?)", normalized)

            if direct:

                return round(float(direct.group(1)), 8)

            if inverse and float(inverse.group(1)) > 0:

                return round(1 / float(inverse.group(1)), 8)

        if field == "cell_configuration":

            normalized = re.sub(r"\s*(?:\|\||//)\s*", "||", normalized)

            # 先规范电池类型，再移除电极对后面的 cell；否则 full cell 会变成 full。

            cell_type = re.fullmatch(r"(pouch|coin|half|full)(?:[ -]+cells?)?", normalized)

            if cell_type:

                name = cell_type.group(1)

                return name + "-cell" if name in {"half", "full"} else name

            normalized = re.sub(r"\s+cells?$", "", normalized).strip()

            aliases = {

                "pouch": "pouch",

                "coin": "coin",

                "half-cell": "half-cell",

                "half cell": "half-cell",

                "full-cell": "full-cell",

                "full cell": "full-cell",

            }

            return aliases.get(normalized, normalized)

        if field == "electrode":

            normalized = re.sub(r"\s*(?:\|\||//|/)\s*", "/", normalized)

            normalized = re.sub(r"\bhc\b", "hard carbon", normalized)

            normalized = re.sub(r"\b(?:anode|cathode|electrode)s?\b", "", normalized)

            parts = [" ".join(part.split()) for part in normalized.split("/")]

            # 此字段检查材料身份。两种材料的列举顺序不改变电极对；单电极不能补成两种。

            return tuple(sorted(parts)) if len(parts) == 2 and all(parts) else " ".join(normalized.split())

        return normalized

    if isinstance(value, float):

        return round(value, 8)

    if isinstance(value, (list, tuple)):

        return tuple(_normalize_context_value(item, field=field) for item in value)

    if isinstance(value, dict):

        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    return value





def _evidence_context_value(

    item: EvidenceLedgerItem,

    field: ExperimentalContextField,

) -> object | None:

    if field == "scientific_object_id":

        return item.scientific_object_id

    if not item.condition_scope:

        return None

    return item.condition_scope.get(field)





def _merge_context_maps(

    *,

    base: dict[str, object],

    contexts: list[dict[str, object]],

    fields: list[ExperimentalContextField],

) -> dict[str, object] | None:

    """合并已经绑定到具体 support observation 的上下文部分赋值。"""



    merged = dict(base)

    normalized = {

        key: _normalize_context_value(value, field=key)

        for key, value in merged.items()

        if value is not None

    }

    # 单个材料是用户给出的筛选条件，不能与原文完整电极对按字符串判冲突。

    # 保留更完整的原文电极对；两个不同电极对仍然冲突，不能拼成同一次实验。

    selector = normalized.get("electrode")

    partial_electrode = isinstance(selector, str) and bool(selector)

    for context in contexts:

        for field in fields:

            value = context.get(field)

            if value is None:

                continue

            norm = _normalize_context_value(value, field=field)

            if field in normalized and normalized[field] != norm:

                previous = normalized[field]

                if field == "electrode" and partial_electrode:

                    if isinstance(norm, tuple) and previous == selector and selector in norm:

                        pass

                    elif isinstance(previous, tuple) and norm == selector and selector in previous:

                        continue

                    else:

                        return None

                else:

                    return None

            normalized[field] = norm

            merged[field] = value

    return merged





def _merge_context_items(

    *,

    base: dict[str, object],

    items: list[EvidenceLedgerItem],

    fields: list[ExperimentalContextField],

) -> dict[str, object] | None:

    contexts = [

        {

            field: value

            for field in fields

            if (value := _evidence_context_value(item, field)) is not None

        }

        for item in items

    ]

    return _merge_context_maps(base=base, contexts=contexts, fields=fields)





def _constraint_support_context(

    *,

    support: ConstraintSupport,

    item: EvidenceLedgerItem,

    fields: list[ExperimentalContextField],

    applicability: ConstraintContextApplicability | None,

) -> dict[str, object]:

    """返回某个 constraint observation 真正有权携带的上下文。



    v1 合同（``applicability is None``）保持历史 scope 回放。v2 合同实行

    positive-only admission：实验条件字段必须带有 obligation-matched atomic

    authorization trace；仅仅被抽取进 ``condition_scope`` 不再足够。

    """



    context: dict[str, object] = {}

    if "scientific_object_id" in fields and item.scientific_object_id is not None:

        context["scientific_object_id"] = item.scientific_object_id

    condition_scope = (

        item.condition_scope

        if support.condition_scope is None

        else support.condition_scope

    )

    authorized_fields: set[str] | None = None

    if applicability is not None and applicability.authorization_mode == "atomic_trace":

        authorized_fields = {

            authorization.context_key

            for authorization in support.context_authorizations

            if authorization.obligation_id == applicability.constraint_id

        }

    if condition_scope:

        for field in fields:

            if field == "scientific_object_id":

                continue

            if authorized_fields is not None and field not in authorized_fields:

                continue

            value = condition_scope.get(field)

            if value is not None:

                context[field] = value

    return context





def _constraint_witness_options(

    *,

    constraint: ConstraintSpec,

    result: ConstraintEvaluation,

    ledger: dict[str, EvidenceLedgerItem],

    requirement: JointContextRequirement,

) -> list[tuple[list[str], dict[str, object]]]:

    applicability = _applicability_by_constraint(requirement).get(constraint.constraint_id)

    # 新运行优先消费 observation-bound supports。只有 v1/legacy group 才允许

    # evidence_ids 回退到 chunk-level condition_scope；v2 applicability 不允许该旁路。

    if result.supports:

        support_entries = [

            (

                support,

                ledger[support.evidence_id],

                _constraint_support_context(

                    support=support,

                    item=ledger[support.evidence_id],

                    fields=requirement.context_fields,

                    applicability=applicability,

                ),

            )

            for support in result.supports

            if support.evidence_id in ledger

            and all(context_id in ledger

                    and ledger[context_id].paper_id == ledger[support.evidence_id].paper_id

                    and ledger[context_id].candidate_id == ledger[support.evidence_id].candidate_id

                    for context_id in support.context_evidence_ids)

        ]

        if not support_entries:

            return []

        if constraint.required_evidence_kinds:

            populations = [

                [entry for entry in support_entries if entry[1].evidence_kind == kind]

                for kind in constraint.required_evidence_kinds

            ]

            if any(not population for population in populations):

                return []

            selections = product(*populations)

        else:

            selections = ((entry,) for entry in support_entries)



        options: list[tuple[list[str], dict[str, object]]] = []

        seen: set[tuple[tuple[str, ...], str]] = set()

        for selection in selections:

            evidence_ids = list(

                dict.fromkeys(evidence_id for entry in selection

                              for evidence_id in [entry[0].evidence_id, *entry[0].context_evidence_ids])

            )

            merged = _merge_context_maps(

                base={},

                contexts=[entry[2] for entry in selection],

                fields=requirement.context_fields,

            )

            if merged is None:

                continue

            context_key = json.dumps(

                merged, ensure_ascii=False, sort_keys=True, default=str

            )

            key = (tuple(evidence_ids), context_key)

            if key in seen:

                continue

            seen.add(key)

            options.append((evidence_ids, merged))

        return options



    referenced = [ledger[item] for item in result.evidence_ids if item in ledger]

    if not referenced:

        return []

    if constraint.required_evidence_kinds:

        populations = [

            [item for item in referenced if item.evidence_kind == kind]

            for kind in constraint.required_evidence_kinds

        ]

        if any(not population for population in populations):

            return []

        selections = product(*populations)

    else:

        selections = ((item,) for item in referenced)



    options = []

    seen_legacy: set[tuple[tuple[str, ...], str]] = set()

    for selection in selections:

        unique_items = list(dict.fromkeys(item.evidence_id for item in selection))

        item_by_id = {item.evidence_id: item for item in selection}

        selected_items = [item_by_id[item_id] for item_id in unique_items]

        if applicability is None:

            merged = _merge_context_items(

                base={},

                items=selected_items,

                fields=requirement.context_fields,

            )

        else:

            # v2 没有 observation-bound authorization trace 时只保留独立验证的

            # scientific object 身份，不能从 chunk scope 推断实验条件 coverage。

            object_contexts = [

                {"scientific_object_id": item.scientific_object_id}

                for item in selected_items

                if "scientific_object_id" in requirement.context_fields

                and item.scientific_object_id is not None

            ]

            merged = _merge_context_maps(

                base={},

                contexts=object_contexts,

                fields=requirement.context_fields,

            )

        if merged is None:

            continue

        context_key = json.dumps(

            merged, ensure_ascii=False, sort_keys=True, default=str

        )

        key = (tuple(unique_items), context_key)

        if key not in seen_legacy:

            seen_legacy.add(key)

            options.append((unique_items, merged))

    return options





def _applicability_by_constraint(

    requirement: JointContextRequirement,

) -> dict[str, ConstraintContextApplicability]:

    return {

        item.constraint_id: item for item in requirement.applicability_requirements

    }





def _missing_applicability_fields(

    *,

    context: dict[str, object],

    applicability: ConstraintContextApplicability,

) -> list[ExperimentalContextField]:

    """返回 observation 对该 constraint 尚未显式覆盖的必需字段。"""



    return [

        field

        for field in applicability.required_fields

        if context.get(field) is None

    ]





def _numeric_interval_contains(evidence_value: object, target_value: object) -> bool:

    """显式 interval/subsumption 规则；不从缺失值推断任何范围。"""



    if not isinstance(evidence_value, (list, tuple)) or len(evidence_value) != 2:

        return False

    try:

        low = float(evidence_value[0])

        high = float(evidence_value[1])

    except (TypeError, ValueError):

        return False

    if low > high:

        low, high = high, low

    if isinstance(target_value, (list, tuple)) and len(target_value) == 2:

        try:

            target_low = float(target_value[0])

            target_high = float(target_value[1])

        except (TypeError, ValueError):

            return False

        if target_low > target_high:

            target_low, target_high = target_high, target_low

        return low <= target_low and target_high <= high

    try:

        target = float(target_value)

    except (TypeError, ValueError):

        return False

    return low <= target <= high





def _canonicalize_entailing_context(

    *,

    context: dict[str, object],

    applicability: ConstraintContextApplicability,

    target_context: dict[str, object],

) -> dict[str, object]:

    """把显式蕴含目标的范围证据投影到任务目标点，再参与联合相容性求解。



    exact 关系仍由后续 compatibility merge 判断。只有显式声明

    ``interval_contains`` 且证据范围确实包含目标时才允许这种投影。

    """



    normalized = dict(context)

    for field in applicability.required_fields:

        if field not in target_context or field not in normalized:

            continue

        relation = applicability.field_relations.get(field, "exact")

        if relation == "interval_contains" and _numeric_interval_contains(

            normalized[field], target_context[field]

        ):

            normalized[field] = target_context[field]

    return normalized





def _coverage_qualified_options(

    *,

    requirement: JointContextRequirement,

    constraint_id: str,

    options: list[tuple[list[str], dict[str, object]]],

) -> tuple[

    list[tuple[list[str], dict[str, object]]],

    list[ExperimentalContextField],

]:

    """先执行 requirement-specific coverage gate，再返回可进入 compatibility 的 options。"""



    applicability = _applicability_by_constraint(requirement).get(constraint_id)

    if applicability is None:

        return options, []



    qualified: list[tuple[list[str], dict[str, object]]] = []

    missing_sets: list[list[ExperimentalContextField]] = []

    for evidence_ids, context in options:

        missing = _missing_applicability_fields(

            context=context,

            applicability=applicability,

        )

        if missing:

            missing_sets.append(missing)

            continue

        qualified.append(

            (

                evidence_ids,

                _canonicalize_entailing_context(

                    context=context,

                    applicability=applicability,

                    target_context=requirement.target_context,

                ),

            )

        )

    if qualified:

        return qualified, []

    if not missing_sets:

        return [], list(applicability.required_fields)

    # 诊断选择缺失字段最少的现有 observation；tie 时保持字段声明顺序。

    best_missing = min(

        missing_sets,

        key=lambda values: (

            len(values),

            tuple(applicability.required_fields.index(field) for field in values),

        ),

    )

    return [], list(best_missing)





def _coverage_repair_query(

    *,

    constraint: ConstraintSpec,

    candidate_id: str,

    requirement: JointContextRequirement,

    required_fields: list[ExperimentalContextField],

) -> str:

    base = constraint.search_hint.strip() or f"find evidence for {constraint.description}"

    context_terms: list[str] = []

    for field in required_fields:

        if field in requirement.target_context:

            context_terms.append(f"{field}={requirement.target_context[field]}")

        else:

            context_terms.append(f"{field}=explicit")

    parts = [base, f"candidate={candidate_id}"]

    if context_terms:

        parts.append("required_context_coverage: " + ", ".join(context_terms))

    if constraint.required_evidence_kinds:

        parts.append(

            "required_evidence_kinds: " + ",".join(constraint.required_evidence_kinds)

        )

    return "; ".join(parts)





def _joint_context_base(

    *, candidate: CandidateState, requirement: JointContextRequirement

) -> dict[str, object]:

    base = dict(requirement.target_context)

    if (

        "scientific_object_id" in requirement.context_fields

        and candidate.scientific_object_id

        and "scientific_object_id" not in base

    ):

        base["scientific_object_id"] = candidate.scientific_object_id

    return base





def _best_compatible_selection(

    *,

    option_populations: list[list[tuple[list[str], dict[str, object]]]],

    base: dict[str, object],

    fields: list[ExperimentalContextField],

) -> tuple[list[str], dict[str, object]] | None:

    """确定性选择最有上下文信息、随后证据数最少的可行 support set。"""



    best: tuple[list[str], dict[str, object]] | None = None

    best_key: tuple[int, int, tuple[str, ...], str] | None = None

    for option_tuple in product(*option_populations):

        evidence_ids = sorted(

            set(item for ids, _ in option_tuple for item in ids)

        )

        merged = _merge_context_maps(

            base=base,

            contexts=[option_context for _, option_context in option_tuple],

            fields=fields,

        )

        if merged is None:

            continue

        explicit_context_coverage = sum(

            1

            for _, option_context in option_tuple

            for field in fields

            if option_context.get(field) is not None

        )

        merged_key = json.dumps(

            merged, ensure_ascii=False, sort_keys=True, default=str

        )

        key = (

            -explicit_context_coverage,

            len(evidence_ids),

            tuple(evidence_ids),

            merged_key,

        )

        if best_key is None or key < best_key:

            best_key = key

            best = (evidence_ids, merged)

    return best





def _repair_conflict_fields(

    *,

    replace_constraint_ids: list[str],

    options_by_constraint: dict[

        str, list[tuple[list[str], dict[str, object]]]

    ],

    anchor_context: dict[str, object],

    fields: list[ExperimentalContextField],

) -> list[ExperimentalContextField]:

    conflicts: list[ExperimentalContextField] = []

    for field in fields:

        anchor_value = anchor_context.get(field)

        if anchor_value is None:

            continue

        anchor_norm = _normalize_context_value(anchor_value, field=field)

        field_conflicts = False

        for constraint_id in replace_constraint_ids:

            for _, option_context in options_by_constraint[constraint_id]:

                option_value = option_context.get(field)

                if option_value is None:

                    continue

                if _normalize_context_value(option_value, field=field) != anchor_norm:

                    field_conflicts = True

                    break

            if field_conflicts:

                break

        if field_conflicts:

            conflicts.append(field)

    return conflicts





def _repair_query(

    *,

    constraint: ConstraintSpec,

    candidate_id: str,

    anchor_context: dict[str, object],

    conflict_fields: list[ExperimentalContextField],

) -> str:

    base = constraint.search_hint.strip() or f"find evidence for {constraint.description}"

    preferred_fields = [field for field in conflict_fields if field in anchor_context]

    if not preferred_fields:

        preferred_fields = [field for field in anchor_context]

    context_text = ", ".join(

        f"{field}={anchor_context[field]}" for field in preferred_fields

    )

    role_text = ",".join(constraint.required_evidence_kinds)

    parts = [base, f"candidate={candidate_id}"]

    if context_text:

        parts.append(f"required_context: {context_text}")

    if role_text:

        parts.append(f"required_evidence_kinds: {role_text}")

    return "; ".join(parts)





def _minimal_joint_context_repairs(

    *,

    candidate: CandidateState,

    requirement: JointContextRequirement,

    constraint_by_id: dict[str, ConstraintSpec],

    options_by_constraint: dict[

        str, list[tuple[list[str], dict[str, object]]]

    ],

    ledger: dict[str, EvidenceLedgerItem],

) -> list[JointContextRepairPlan]:

    """精确枚举最少需要替换多少个 constraint-support sets 才能恢复可行性。"""



    base = _joint_context_base(candidate=candidate, requirement=requirement)

    constraint_ids = list(requirement.constraint_ids)

    for replace_count in range(1, len(constraint_ids) + 1):

        plans: list[JointContextRepairPlan] = []

        for replace_tuple in combinations(constraint_ids, replace_count):

            replace_ids = list(replace_tuple)

            retained_ids = [

                item for item in constraint_ids if item not in set(replace_ids)

            ]

            retained_populations = [

                options_by_constraint[item] for item in retained_ids

            ]

            retained = _best_compatible_selection(

                option_populations=retained_populations,

                base=base,

                fields=requirement.context_fields,

            )

            if retained is None:

                continue

            retained_evidence_ids, anchor_context = retained

            conflict_fields = _repair_conflict_fields(

                replace_constraint_ids=replace_ids,

                options_by_constraint=options_by_constraint,

                anchor_context=anchor_context,

                fields=requirement.context_fields,

            )

            search_queries = [

                _repair_query(

                    constraint=constraint_by_id[constraint_id],

                    candidate_id=candidate.candidate_id,

                    anchor_context=anchor_context,

                    conflict_fields=conflict_fields,

                )

                for constraint_id in replace_ids

            ]

            plans.append(

                JointContextRepairPlan(

                    repair_version=(

                        "context_qualified_repair_v2"

                        if requirement.applicability_requirements

                        else "experimental_context_repair_v1"

                    ),

                    group_id=requirement.group_id,

                    candidate_id=candidate.candidate_id,

                    replace_constraint_ids=replace_ids,

                    retained_evidence_ids=retained_evidence_ids,

                    context_fields=list(requirement.context_fields),

                    anchor_context=anchor_context,

                    conflict_fields=conflict_fields,

                    failure_mode="compatibility",

                    can_replan=requirement.can_replan,

                    search_queries=search_queries,

                )

            )

        if plans:

            return sorted(plans, key=lambda item: tuple(item.replace_constraint_ids))

    return []





def _joint_context_check(

    *,

    candidate: CandidateState,

    evaluation: CandidateEvaluation,

    contract: TaskContract,

    ledger: dict[str, EvidenceLedgerItem],

) -> tuple[

    list[EvidenceGap],

    list[JointContextWitness],

    list[JointContextRepairPlan],

]:

    if not contract.joint_context_requirements:

        return [], [], []



    constraint_by_id = {item.constraint_id: item for item in contract.hard_constraints}

    result_by_id = {item.constraint_id: item for item in evaluation.constraints}

    gaps: list[EvidenceGap] = []

    witnesses: list[JointContextWitness] = []

    repair_plans: list[JointContextRepairPlan] = []



    for requirement in contract.joint_context_requirements:

        options_by_constraint: dict[

            str, list[tuple[list[str], dict[str, object]]]

        ] = {}

        incomplete = False

        for constraint_id in requirement.constraint_ids:

            constraint = constraint_by_id[constraint_id]

            result = result_by_id.get(constraint_id)

            if result is None or result.status != "satisfied":

                incomplete = True

                break

            options = _constraint_witness_options(

                constraint=constraint,

                result=result,

                ledger=ledger,

                requirement=requirement,

            )

            if not options:

                incomplete = True

                break

            qualified_options, missing_fields = _coverage_qualified_options(

                requirement=requirement,

                constraint_id=constraint_id,

                options=options,

            )

            if missing_fields:

                repair_plans.append(

                    JointContextRepairPlan(

                        repair_version="context_qualified_repair_v2",

                        group_id=requirement.group_id,

                        candidate_id=candidate.candidate_id,

                        replace_constraint_ids=[constraint_id],

                        retained_evidence_ids=[],

                        context_fields=list(requirement.context_fields),

                        anchor_context=dict(requirement.target_context),

                        conflict_fields=[],

                        failure_mode="coverage",

                        missing_context_fields={constraint_id: missing_fields},

                        can_replan=requirement.can_replan,

                        search_queries=[

                            _coverage_repair_query(

                                constraint=constraint,

                                candidate_id=candidate.candidate_id,

                                requirement=requirement,

                                required_fields=missing_fields,

                            )

                        ],

                    )

                )

                gaps.append(

                    EvidenceGap(

                        code="experimental_context_coverage_missing",

                        description=(

                            f"硬约束组“{requirement.group_id}”的支持证据未显式覆盖 "

                            + ", ".join(missing_fields)

                            + "，不能由开放世界兼容性推断适用性。"

                        ),

                        candidate_id=candidate.candidate_id,

                        can_replan=requirement.can_replan,

                        search_hint=repair_plans[-1].search_queries[0],

                    )

                )

                incomplete = True

                break

            options_by_constraint[constraint_id] = qualified_options

        if incomplete:

            # Local constraint gaps already explain missing/invalid evidence. The joint

            # checker adds a gap only when each local constraint has evidence but the

            # evidence cannot co-instantiate.

            continue



        option_populations = [

            options_by_constraint[item] for item in requirement.constraint_ids

        ]

        best = _best_compatible_selection(

            option_populations=option_populations,

            base=_joint_context_base(candidate=candidate, requirement=requirement),

            fields=requirement.context_fields,

        )

        if best is not None:

            evidence_ids, merged = best

            applicability_by_constraint = _applicability_by_constraint(requirement)

            witnesses.append(

                JointContextWitness(

                    witness_version=(

                        "context_qualified_evidence_v2"

                        if requirement.applicability_requirements

                        else "experimental_context_join_v1"

                    ),

                    group_id=requirement.group_id,

                    candidate_id=candidate.candidate_id,

                    constraint_ids=list(requirement.constraint_ids),

                    context_fields=list(requirement.context_fields),

                    target_context=dict(requirement.target_context),

                    evidence_ids=evidence_ids,

                    merged_context=merged,

                    applicability_coverage={

                        constraint_id: list(applicability.required_fields)

                        for constraint_id, applicability in applicability_by_constraint.items()

                    },

                )

            )

            continue



        repairs = _minimal_joint_context_repairs(

            candidate=candidate,

            requirement=requirement,

            constraint_by_id=constraint_by_id,

            options_by_constraint=options_by_constraint,

            ledger=ledger,

        )

        repair_plans.extend(repairs)

        search_hint = (

            repairs[0].search_queries[0]

            if repairs and repairs[0].search_queries

            else requirement.search_hint

            or "find evidence that jointly matches the target experimental context"

        )

        gaps.append(

            EvidenceGap(

                code="experimental_context_join_conflict",

                description=(

                    "硬约束组“"

                    + requirement.group_id

                    + "”的局部支持证据无法在声明的实验上下文字段上共同实例化；"

                    "不能把彼此冲突的真实证据拼成同一推荐。"

                ),

                candidate_id=candidate.candidate_id,

                can_replan=requirement.can_replan,

                search_hint=search_hint,

            )

        )



    return gaps, witnesses, repair_plans





def decide_design(

    *,

    contract: TaskContract,

    candidates: list[CandidateState],

    evaluations: list[CandidateEvaluation],

    evidence_ledger: list[EvidenceLedgerItem],

    attempts: int,

    max_attempts: int,

) -> DecisionState:

    """根据结构化状态决定继续搜索、允许推荐或安全暂停。



    ``attempts`` 表示已经完成的筛选/候选迭代次数。函数不修改输入，也不自行

    查询新证据。存在通过全部硬约束的候选时优先推荐；普通缺口只有在仍有

    正常迭代余量时返回 search。若恰好在正常预算边界首次得到可执行的

    joint-context minimal repair，则允许额外一次仅由 repair query 驱动的搜索；

    下一次 re-screen 若仍不可行必须 hold，不能继续透支普通检索预算。

    """



    if attempts < 0:

        raise ValueError("attempts must be non-negative")

    if max_attempts < 1:

        raise ValueError("max_attempts must be at least 1")



    ledger = {item.evidence_id: item for item in evidence_ledger}

    if len(ledger) != len(evidence_ledger):

        raise ValueError("evidence ids must be unique")

    candidate_ids = [item.candidate_id for item in candidates]

    if len(candidate_ids) != len(set(candidate_ids)):

        raise ValueError("candidate ids must be unique")

    evaluation_by_id = {item.candidate_id: item for item in evaluations}

    if len(evaluation_by_id) != len(evaluations):

        raise ValueError("candidate evaluation ids must be unique")



    if not candidates:

        gap = EvidenceGap(

            code="no_candidates",

            description="当前没有可评估的候选配方。",

            search_hint="search for an alternative fully evidenced formulation",

        )

        status: DecisionStatus = "search" if attempts < max_attempts else "hold"

        return DecisionState(

            status=status,

            evidence_gaps=[gap],

            next_queries=[gap.search_hint] if status == "search" else [],

            attempts=attempts,

            max_attempts=max_attempts,

            reason_codes=[gap.code],

        )



    approved: list[str] = []

    approved_joint_witnesses: list[JointContextWitness] = []

    all_joint_repairs: list[JointContextRepairPlan] = []

    all_gaps: list[EvidenceGap] = []

    for candidate in candidates:

        evaluation = evaluation_by_id.get(candidate.candidate_id)

        gaps = _candidate_gaps(

            candidate=candidate,

            evaluation=evaluation,

            contract=contract,

            ledger=ledger,

        )

        joint_witnesses: list[JointContextWitness] = []

        joint_repairs: list[JointContextRepairPlan] = []

        if not gaps and evaluation is not None:

            joint_gaps, joint_witnesses, joint_repairs = _joint_context_check(

                candidate=candidate,

                evaluation=evaluation,

                contract=contract,

                ledger=ledger,

            )

            gaps.extend(joint_gaps)

        if gaps:

            all_gaps.extend(gaps)

            all_joint_repairs.extend(joint_repairs)

        else:

            approved.append(candidate.candidate_id)

            approved_joint_witnesses.extend(joint_witnesses)



    if approved:

        return DecisionState(

            status="recommend",

            approved_candidate_ids=approved,

            joint_context_witnesses=approved_joint_witnesses,

            attempts=attempts,

            max_attempts=max_attempts,

            reason_codes=["all_hard_constraints_satisfied"],

        )



    repair_queries = _dedupe([

        query

        for plan in all_joint_repairs

        if plan.can_replan

        for query in plan.search_queries

    ])

    normal_search_budget = attempts < max_attempts

    repair_grace_budget = attempts == max_attempts and bool(repair_queries)

    can_replan_gap = any(gap.can_replan for gap in all_gaps)

    can_search = can_replan_gap and (normal_search_budget or repair_grace_budget)

    status = "search" if can_search else "hold"

    if normal_search_budget:

        next_queries = _dedupe(

            [

                *repair_queries,

                *[

                    gap.search_hint or gap.description

                    for gap in all_gaps

                    if gap.can_replan

                ],

            ]

        )

    elif repair_grace_budget:

        # 正常 screen 预算已经用尽时，只允许执行 deterministic repair 编译出的

        # 定向查询；不能借 repair grace 顺带继续普通候选/证据搜索。

        next_queries = repair_queries

    else:

        next_queries = []

    return DecisionState(

        status=status,

        joint_context_repair_plans=all_joint_repairs,

        evidence_gaps=all_gaps,

        next_queries=next_queries if status == "search" else [],

        attempts=attempts,

        max_attempts=max_attempts,

        reason_codes=_dedupe([gap.code for gap in all_gaps]),

    )

