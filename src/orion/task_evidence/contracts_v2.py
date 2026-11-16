"""四类电解液 Agent 任务的统一 v2 合同与 artifact。



本模块只定义可机读数据结构和纯验证逻辑，不调用 Agent、LLM、RAG 或 KG。

历史 benchmark 的 ``screening`` 标签通过 ``canonical_task_type`` 按问题语义映射到

``evidence_synthesis`` 或 ``candidate_screening``；冻结数据文件本身不需要改写。

"""



from __future__ import annotations



from typing import Annotated, Any, Literal, TypeAlias



from pydantic import BaseModel, Field, TypeAdapter, model_validator



from orion.core.claim_contract import ClaimContract

from orion.core.prediction_comparison import (

    PredictionComparisonAssessment, assess_prediction_comparison,

)

from .models import SourceSpan

from .evidence_applicability import ApplicabilityRecord

from .evidence_use import EvidenceUseDecision, compile_evidence_use_decisions

from .task_gates import (

    ComparabilityAssessment,

    ComparisonConditions,

    DiscriminatingTest,

    HypothesisRecord,

    InverseHypothesisAssessment,

)





TaskType: TypeAlias = Literal[

    "evidence_synthesis",

    "candidate_screening",

    "design",

    "inverse_design",

]

TaskTerminalStatus: TypeAlias = Literal[

    "answered_grounded",

    "out_of_scope",

    "compared",

    "model_assessed",

    "not_comparable",

    "recommended",

    "held",

    "diagnosis_supported",

    "diagnosis_hypothesis_only",

    "insufficient_evidence",

]



TASK_TYPE_SKILLS: dict[str, str] = {

    "design": "electrolyte_design",

    "inverse_design": "electrolyte_inverse_design",

    "candidate_screening": "electrolyte_screening",

    "evidence_synthesis": "literature_review",

}



_LEGACY_TASK_TYPES = {"literature_review": "evidence_synthesis"}

_COMPARISON_MARKERS = (

    "对比",

    "比较",

    "差异",

    "有何不同",

    "优劣",

    "优缺点",

    "排序",

    "筛选",

    " vs ",

    " versus ",

    "compare",

    "which is better",

    "choose between",

    "rank ",

)

_FACT_ONLY_MARKERS = (

    "只回答该论文直接报告的事实",

    "采用了哪些",

    "what conditions were used",

)



_TERMINAL_STATUSES: dict[str, set[str]] = {

    "evidence_synthesis": {

        "answered_grounded",

        "insufficient_evidence",

        "out_of_scope",

    },

    "candidate_screening": {

        "compared",

        "model_assessed",

        "not_comparable",

        "insufficient_evidence",

    },

    "design": {"recommended", "held"},

    "inverse_design": {

        "diagnosis_supported",

        "diagnosis_hypothesis_only",

        "insufficient_evidence",

    },

}





def canonical_task_type(value: Any, *, question: str = "") -> str:

    """把历史对外标签映射为四任务 canonical 名称。



    旧 ``screening`` 同时包含事实/机理问答和真正的候选比较，不能无条件映射。

    有明确比较意图时归入 ``candidate_screening``，否则保守归入

    ``evidence_synthesis``。不提供问题文本的旧程序化调用维持候选筛选语义。

    """



    normalized = str(value or "").strip()

    if normalized == "screening":

        if not question:

            return "candidate_screening"

        lowered = f" {question.lower()} "

        if any(marker in lowered for marker in _FACT_ONLY_MARKERS):

            return "evidence_synthesis"

        return (

            "candidate_screening"

            if any(marker in lowered for marker in _COMPARISON_MARKERS)

            else "evidence_synthesis"

        )

    return _LEGACY_TASK_TYPES.get(normalized, normalized)





def expected_skill_for_task_type(task_type: Any) -> str | None:

    """Return the canonical Skill for an explicitly typed task contract.



    This mapping is intentionally driven by structured ``task_type`` metadata,

    never by keywords in the user question. Untyped conversations therefore

    retain dynamic Skill selection, while evaluation and API callers that

    already possess a task contract get deterministic procedural routing.

    """



    canonical = canonical_task_type(task_type)

    return TASK_TYPE_SKILLS.get(canonical)





def task_type_for_skill(skill_name: Any) -> str | None:

    """Return the canonical task type owned by a canonical Skill.



    The reverse lookup is deliberately exact.  It is used by deterministic

    workflow guards after ``load_skill`` has succeeded; it must not infer a

    task type from free-form question text.

    """



    normalized = str(skill_name or "").strip()

    return next(

        (

            task_type

            for task_type, expected_skill in TASK_TYPE_SKILLS.items()

            if expected_skill == normalized

        ),

        None,

    )





class AgentTaskContract(BaseModel):

    """用户任务与证据/终态要求的唯一 v2 合同。"""



    contract_version: Literal["2.0"] = "2.0"

    task_type: TaskType

    required_inputs: list[str] = Field(default_factory=list)

    required_outputs: list[str] = Field(default_factory=list)

    required_evidence_kinds: list[str] = Field(default_factory=list)

    required_tools: list[str] = Field(default_factory=list)

    allowed_terminal_statuses: list[TaskTerminalStatus] = Field(

        default_factory=list

    )

    allowed_capabilities: list[str] = Field(default_factory=list)

    forbidden_claims: list[str] = Field(default_factory=list)



    @model_validator(mode="after")

    def _validate_terminal_statuses(self) -> "AgentTaskContract":

        allowed = _TERMINAL_STATUSES[self.task_type]

        invalid = [

            status

            for status in self.allowed_terminal_statuses

            if status not in allowed

        ]

        if invalid:

            raise ValueError(

                f"{self.task_type} does not allow terminal statuses: {invalid}"

            )

        return self





def default_task_contract(task_type: TaskType | str) -> AgentTaskContract:

    """返回与研究任务合同一致的最小默认合同。"""



    canonical = canonical_task_type(task_type)

    definitions: dict[str, dict[str, list[str]]] = {

        "evidence_synthesis": {

            "required_inputs": ["question"],

            "required_outputs": ["answer", "claims", "evidence_gaps"],

            "required_evidence_kinds": ["literature_chunk", "exact_quote"],

            "allowed_capabilities": [

                "literature_retrieval",

                "paper_retrieval",

                "read_only_kg_discovery",

            ],

            "forbidden_claims": [

                "unsupported_fact",

                "experiment_ready_formulation",

            ],

        },

        "candidate_screening": {

            "required_inputs": ["candidates", "comparison_objective"],

            "required_outputs": [

                "normalized_candidates",

                "comparison_axes",

                "condition_comparability",

                "comparability_assessment",

            ],

            "required_evidence_kinds": ["condition", "property"],

            "allowed_capabilities": [

                "literature_retrieval",

                "numeric_extraction",

                "criterion_evaluation",

            ],

            "forbidden_claims": [

                "ranking_without_comparability",

                "silent_candidate_generation",

            ],

        },

        "design": {

            "required_inputs": ["target", "hard_constraints"],

            "required_outputs": [

                "candidates",

                "candidate_evaluations",

                "formulation_decision",

            ],

            "required_evidence_kinds": [

                "formulation",

                "condition",

                "property",

            ],

            "allowed_capabilities": [

                "literature_retrieval",

                "read_only_kg_query",

                "candidate_evaluation",

                "bounded_replanning",

            ],

            "forbidden_claims": [

                "cross_paper_formulation_performance_join",

                "recommendation_without_decision_approval",

            ],

        },

        "inverse_design": {

            "required_inputs": ["symptom", "known_conditions"],

            "required_outputs": [

                "hypotheses",

                "confounders",

                "discriminating_tests",

                "hypothesis_assessment",

            ],

            "required_evidence_kinds": ["mechanism", "observation"],

            "allowed_capabilities": [

                "mechanism_retrieval",

                "read_only_kg_discovery",

                "hypothesis_evaluation",

            ],

            "forbidden_claims": [

                "correlation_as_proven_root_cause",

                "unvalidated_adjustment_as_resolved_failure",

            ],

        },

    }

    if canonical not in definitions:

        raise ValueError(f"unsupported task type: {task_type}")

    return AgentTaskContract(

        task_type=canonical,

        allowed_terminal_statuses=sorted(_TERMINAL_STATUSES[canonical]),

        **definitions[canonical],

    )





class ArtifactClaim(BaseModel):

    """报告中一项可追溯事实、比较、假设或建议。"""



    claim_id: str = Field(min_length=1)

    claim_type: Literal[

        "fact",

        "comparison",

        "ranking",

        "hypothesis",

        "diagnosis",

        "conditional_adjustment",

        "recommendation",

    ] = "fact"

    text: str = Field(min_length=1)

    evidence_ids: list[str] = Field(default_factory=list)





class ArtifactEvidence(BaseModel):

    """本轮运行权威 evidence ledger 中的一项证据。"""



    evidence_id: str = Field(min_length=1)

    chunk_id: str = Field(min_length=1)

    paper_id: str = ""

    quote: str = Field(min_length=1)

    evidence_kind: str = "literature_chunk"

    source_span: SourceSpan | None = None

    candidate_id: str | None = None

    scientific_object_id: str | None = None

    experiment_id: str = ""

    condition_scope: ComparisonConditions | dict[str, Any] | None = None

    condition_quote: str = ""





class ArtifactGap(BaseModel):

    """阻止更强终态的结构化证据缺口。"""



    code: str = Field(min_length=1)

    description: str = Field(min_length=1)

    candidate_id: str | None = None

    search_hint: str = ""





class CandidateEvaluationArtifact(BaseModel):

    """候选的确定性评估快照。"""



    candidate_id: str = Field(min_length=1)

    criteria: list[dict[str, Any]] = Field(default_factory=list)





class ApprovalRecord(BaseModel):

    """高风险外部动作的审批记录；只读任务通常为空。"""



    action: str = Field(min_length=1)

    status: Literal["requested", "approved", "rejected"]

    actor: str = ""

    timestamp: str = ""





class TraceEvent(BaseModel):

    """不含密钥的最小执行轨迹。"""



    event: str = Field(min_length=1)

    tool: str = ""

    status: str = ""





class _TaskArtifactBase(BaseModel):

    """四类 artifact 的共同字段。"""



    schema_version: Literal["2.0"] = "2.0"

    contract: AgentTaskContract

    task_type: TaskType

    task_status: TaskTerminalStatus

    claims: list[ArtifactClaim] = Field(default_factory=list)

    claim_contracts: list[ClaimContract] = Field(default_factory=list)

    evidence_ledger: list[ArtifactEvidence] = Field(default_factory=list)

    applicability_records: list[ApplicabilityRecord] = Field(default_factory=list)

    evidence_use_decisions: list[EvidenceUseDecision] = Field(default_factory=list)

    gaps: list[ArtifactGap] = Field(default_factory=list)

    candidate_evaluations: list[CandidateEvaluationArtifact] = Field(

        default_factory=list

    )

    formulation_decision: dict[str, Any] | None = None

    validation_plan: list[str] = Field(default_factory=list)

    approvals: list[ApprovalRecord] = Field(default_factory=list)

    trace: list[TraceEvent] = Field(default_factory=list)

    runtime_provenance: dict[str, Any] = Field(default_factory=dict)



    @model_validator(mode="after")

    def _validate_contract_and_status(self) -> "_TaskArtifactBase":

        if self.contract.task_type != self.task_type:

            raise ValueError("artifact task_type must match contract task_type")

        if self.task_status not in self.contract.allowed_terminal_statuses:

            raise ValueError(

                "artifact task_status must be allowed by the task contract"

            )

        affirmative_statuses = {

            "answered_grounded",

            "compared",

            "model_assessed",

            "recommended",

            "diagnosis_supported",

        }

        if self.task_status in affirmative_statuses:

            emitted_claim_ids = {claim.claim_id for claim in self.claims}

            violations = [

                item.assessment_id

                for item in self.applicability_records

                if item.decision in {"search", "hold", "reject_claim"}

                and item.terminal_disposition != "blocked"

                and (not item.claim_id or item.claim_id in emitted_claim_ids)

            ]

            if violations:

                raise ValueError(

                    "affirmative terminal status violates applicability action permission: "

                    + ", ".join(violations)

                )

        if self.evidence_use_decisions:

            contract_ids = {item.claim_id for item in self.claim_contracts}

            decision_by_claim = {

                item.claim_id: item for item in self.evidence_use_decisions

            }

            missing = sorted(contract_ids - set(decision_by_claim))

            if missing:

                raise ValueError(

                    "terminal ClaimContracts missing evidence-use decision: "

                    + ", ".join(missing)

                )

            blocked = sorted(

                claim_id

                for claim_id in contract_ids

                if decision_by_claim[claim_id].disposition == "block"

            )

            if blocked:

                raise ValueError(

                    "terminal ClaimContracts contain blocked evidence use: "

                    + ", ".join(blocked)

                )

            expected_use = {

                item.claim_id: item

                for item in compile_evidence_use_decisions(

                    task_type=self.task_type,

                    claims=self.claim_contracts,

                    applicability_records=self.applicability_records,

                )

            }

            stale = sorted(

                claim_id

                for claim_id in contract_ids

                if claim_id not in expected_use

                or decision_by_claim[claim_id].context_fingerprint

                != expected_use[claim_id].context_fingerprint

                or decision_by_claim[claim_id].use_kind

                != expected_use[claim_id].use_kind

                or decision_by_claim[claim_id].disposition

                != expected_use[claim_id].disposition

            )

            if stale:

                raise ValueError(

                    "stale or context-mismatched evidence-use decision: "

                    + ", ".join(stale)

                )

            required_use = {

                "recommended": "recommend",

                "diagnosis_supported": "diagnose",

            }.get(self.task_status)

            if required_use and not any(

                item.use_kind == required_use and item.disposition != "block"

                for item in self.evidence_use_decisions

            ):

                raise ValueError(

                    f"{self.task_status} requires an authorized {required_use} evidence use"

                )

        return self





class EvidenceSynthesisArtifact(_TaskArtifactBase):

    task_type: Literal["evidence_synthesis"] = "evidence_synthesis"

    task_status: Literal[

        "answered_grounded",

        "insufficient_evidence",

        "out_of_scope",

    ]

    answer: str = ""

    consensus: list[str] = Field(default_factory=list)

    disputes: list[str] = Field(default_factory=list)





class CandidateScreeningArtifact(_TaskArtifactBase):

    task_type: Literal["candidate_screening"] = "candidate_screening"

    task_status: Literal[

        "compared",

        "model_assessed",

        "not_comparable",

        "insufficient_evidence",

    ]

    normalized_candidates: list[dict[str, Any]] = Field(default_factory=list)

    comparison_axes: list[str] = Field(default_factory=list)

    condition_comparability: Literal[

        "comparable",

        "not_comparable",

        "unknown",

    ] = "unknown"

    comparability_assessment: ComparabilityAssessment | None = None

    model_comparison: PredictionComparisonAssessment | None = None



    @model_validator(mode="after")

    def _validate_comparability_gate(self) -> "CandidateScreeningArtifact":

        if self.task_status == "model_assessed":

            if self.model_comparison is None:

                raise ValueError("model_assessed requires recorded prediction comparison")

            checked = assess_prediction_comparison(

                self.model_comparison.records, self.model_comparison.request,

            )

            if checked.status != "model_assessed" or checked != self.model_comparison:

                raise ValueError("model comparison disagrees with recorded calculations")

            if not any(

                c.claim_type == "comparison" and c.evidence_binding == "computation_bound"

                and c.verification_status == "passed"

                and any(v.rule_id == "recorded_prediction_comparison_v1"

                        and v.data_used.get("assessment") == checked.model_dump(mode="json")

                        for v in c.verification_results if v.status == "passed")

                for c in self.claim_contracts

            ):

                raise ValueError("model_assessed requires a bound calculation comparison")

        if self.task_status == "compared":

            if (

                self.condition_comparability != "comparable"

                or self.comparability_assessment is None

                or self.comparability_assessment.status != "comparable"

            ):

                raise ValueError(

                    "compared status requires a comparable deterministic assessment"

                )

            if not any(

                claim.claim_type in {"comparison", "ranking"}

                for claim in self.claims

            ):

                raise ValueError(

                    "compared status requires a grounded comparison claim"

                )

        if (

            self.task_status == "not_comparable"

            and (

                self.condition_comparability != "not_comparable"

                or self.comparability_assessment is None

                or self.comparability_assessment.status != "not_comparable"

            )

        ):

            raise ValueError(

                "not_comparable status requires a deterministic mismatch assessment"

            )

        return self





class DesignTaskArtifact(_TaskArtifactBase):

    task_type: Literal["design"] = "design"

    task_status: Literal["recommended", "held"]

    candidates: list[dict[str, Any]] = Field(default_factory=list)





class InverseDesignTaskArtifact(_TaskArtifactBase):

    task_type: Literal["inverse_design"] = "inverse_design"

    task_status: Literal[

        "diagnosis_supported",

        "diagnosis_hypothesis_only",

        "insufficient_evidence",

    ]

    symptom_boundary: dict[str, Any] = Field(default_factory=dict)

    hypotheses: list[HypothesisRecord] = Field(default_factory=list)

    confounders: list[str] = Field(default_factory=list)

    discriminating_tests: list[DiscriminatingTest] = Field(default_factory=list)

    conditional_adjustments: list[str] = Field(default_factory=list)

    hypothesis_assessment: InverseHypothesisAssessment | None = None



    @model_validator(mode="after")

    def _validate_hypothesis_gate(self) -> "InverseDesignTaskArtifact":

        if self.hypothesis_assessment is None:

            if self.task_status != "insufficient_evidence":

                raise ValueError(

                    "inverse diagnosis terminal status requires hypothesis assessment"

                )

            return self

        if self.task_status != self.hypothesis_assessment.status:

            raise ValueError(

                "inverse task_status must match deterministic hypothesis assessment"

            )

        return self





TaskArtifact: TypeAlias = Annotated[

    EvidenceSynthesisArtifact

    | CandidateScreeningArtifact

    | DesignTaskArtifact

    | InverseDesignTaskArtifact,

    Field(discriminator="task_type"),

]

TASK_ARTIFACT_ADAPTER = TypeAdapter(TaskArtifact)





def validate_task_artifact_v2(value: Any) -> TaskArtifact:

    """按 ``task_type`` 判别并验证四类 artifact。"""



    return TASK_ARTIFACT_ADAPTER.validate_python(value)

