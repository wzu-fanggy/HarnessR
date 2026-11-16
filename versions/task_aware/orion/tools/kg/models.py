from __future__ import annotations



import math

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from pydantic.json_schema import SkipJsonSchema



# LLM 常给出中文或 high/medium/low，统一映射到 strong/moderate/weak。

EVIDENCE_LEVEL_MAP = {

    "strong": "strong", "moderate": "moderate", "weak": "weak",

    "high": "strong", "medium": "moderate", "low": "weak",

    "强": "strong", "中": "moderate", "中等": "moderate", "较弱": "weak", "弱": "weak",

}





def coerce_evidence_level(value) -> str:

    """宽容解析证据等级；无法识别时返回 moderate。"""

    if not isinstance(value, str) or not value.strip():

        return "moderate"

    s = value.strip()

    return EVIDENCE_LEVEL_MAP.get(s.lower(), EVIDENCE_LEVEL_MAP.get(s, "moderate"))





class KgPath(BaseModel):

    start_entity: str

    relation: str

    end_entity: str

    confidence_mean: float | None = None





class EvidenceRecord(BaseModel):

    # chunk_id 允许省略（LLM 组织证据时可能丢失），但检索/审计链路应始终带上

    chunk_id: str = ""

    parent_chunk_id: str | None = None

    paper_id: str = ""

    paper_title: str = ""

    text: str = ""

    excerpt: str = ""

    parent_text: str | None = None

    score: float = 0.0

    entities: list[str] = Field(default_factory=list)

    kg_paths: list[KgPath] = Field(default_factory=list)

    source: Literal["vector", "kg_structured", "kg_expansion"] = "vector"

    evidence_kind: Literal[

        "formulation",

        "property",

        "condition",

        "mechanism",

        "risk",

        "background",

    ] = "background"

    # 显式科学对象/实验身份由上游解析器提供；缺失时保持未知，禁止用 paper_id 猜测。

    scientific_object_id: str | None = None

    experiment_id: str | None = None

    condition_scope: dict[str, Any] | None = None



    @model_validator(mode="after")

    def _excerpt_to_text(self):

        if not self.text and self.excerpt:

            self.text = self.excerpt

        return self



    @field_validator("score", mode="before")

    @classmethod

    def _coerce_score(cls, v):

        return 0.0 if v is None else float(v)





class KgQueryResult(BaseModel):

    query_type: str

    start_entity: str

    relation: str

    end_entity: str

    confidence_mean: float | None = None

    entity_labels: list[str] = Field(default_factory=list)

    chunk_ids: list[str] = Field(default_factory=list)

    kg_paths: list[KgPath] = Field(default_factory=list)





class FormulationSpaceRequest(BaseModel):

    result_mode: Literal["evidence", "legacy_candidates"] = "evidence"

    task_type: Literal["design", "inverse_design", "screening", "screen"] = "design"

    target_conditions: list[str] = Field(default_factory=list)

    preferred_salts: list[str] | None = None

    preferred_solvents: list[str] | None = None

    preferred_additives: list[str] | None = None

    excluded_components: list[str] = Field(default_factory=list)

    evidence_chunks: list[dict[str, Any]] = Field(default_factory=list)

    max_candidates: int = Field(default=10, ge=1, le=50)





class ModelPropertyPrediction(BaseModel):

    """由显式科学模型产生的候选性质预测；与文献 evidence 分开保存。"""



    property_name: Literal["ionic_conductivity"] = "ionic_conductivity"

    status: Literal["predicted", "out_of_domain", "insufficient_composition"]

    numeric_value: float | None = None

    unit: str = "mS/cm"

    temperature_c: float | None = None

    model_id: str = "na-conductivity-rf-v1"

    source_data_doi: str = "10.5281/zenodo.18429013"

    ensemble_std_log10: float | None = None

    estimated_factor_1sigma: float | None = None

    # 新计算的不可变输入/条件归属；历史记录缺失时保持未知。

    result_binding: dict[str, Any] | None = None

    features: dict[str, float] = Field(default_factory=dict)

    reason: str = ""





class FormulationCandidate(BaseModel):

    candidate_id: str

    scientific_object_id: str | None = None

    salt: str = ""

    # 单盐字段保留兼容；多盐配方分别保存盐名与各自浓度。

    salts: list[str] = Field(default_factory=list)

    salt_concentrations_mol_l: dict[str, float] = Field(default_factory=dict)

    solvents: list[str] = Field(default_factory=list)

    concentration_mol_l: float | None = None

    solvent_ratios: dict[str, float] = Field(default_factory=dict)

    solvent_ratio_basis: Literal["mass", "volume", "molar", "unknown"] = "unknown"

    additives: list[str] = Field(default_factory=list)

    additive_amounts: dict[str, str] = Field(default_factory=dict)

    concentration: str | None = None

    electrolyte_entity: str | None = None

    rationale: str = ""

    evidence: list[EvidenceRecord] = Field(default_factory=list)

    evidence_level: Literal["strong", "moderate", "weak"] = "weak"

    inferred_fields: list[str] = Field(default_factory=list)

    unresolved_gaps: list[str] = Field(default_factory=list)

    evidence_scope: Literal["same_child", "same_paper", "unknown"] = "unknown"

    model_predictions: list[ModelPropertyPrediction] = Field(default_factory=list)



    @field_validator("salt", mode="before")

    @classmethod

    def _coerce_salt(cls, v):

        return "" if v is None else str(v)



    @field_validator("evidence_level", mode="before")

    @classmethod

    def _coerce_level(cls, v):

        return coerce_evidence_level(v) if v is not None else "weak"





class ClaimVerdict(BaseModel):

    """模型对单条断言约束在单个候选上的语义判定，须附逐字引文供程序验证。



    语义判断（这句原文是否为该候选建立了断言）由模型负责；程序只验证

    chunk_id 属于该候选的证据、引文逐字存在于该 chunk，不解析句式或词义。

    """



    candidate_id: str = Field(min_length=1)

    verdict: Literal["supported", "refuted", "not_found"] = Field(

        description=(

            "该候选证据对断言的判定：supported＝有原文明确支持；"

            "refuted＝有原文明确反驳；not_found＝检索到的证据未直接提及。"

            "背景介绍、他人体系的结果、无实验支撑的预测不算 supported。"

        ),

    )

    chunk_id: str = ""

    quote: str = ""

    basis: str = Field(

        default="",

        description="一行判定依据，说明引文为何支持或反驳断言；供人工审计。",

    )



    @model_validator(mode="after")

    def _validate_verdict_evidence(self) -> "ClaimVerdict":

        if self.verdict in {"supported", "refuted"}:

            if not self.chunk_id.strip():

                raise ValueError(

                    f"verdict={self.verdict!r} 必须给出 chunk_id，"

                    "指明引文所在证据块。"

                )

            if not self.quote.strip():

                raise ValueError(

                    f"verdict={self.verdict!r} 必须给出逐字 quote，"

                    "从该 chunk 的原文中逐字截取。"

                )

        return self





class ScreeningConstraint(BaseModel):

    """可由工具 schema 校验、可稳定复放的单条硬约束。"""



    constraint_id: str = Field(min_length=1)

    description: str = Field(min_length=1)

    kind: Literal[

        "minimum_voltage",

        "voltage_window",

        "c_rate",

        "minimum_cycles",

        "minimum_retention",

        "minimum_conductivity",

        "maximum_viscosity",

        "temperature",

        "required_component",

        "excluded_component",

        "field_presence",

        "electrode_material",

        "cell_configuration",

        "mechanism_keyword",

        "evidence_claim",

    ] = Field(

        description=(

            "约束类型。数值阈值：minimum_voltage 最低电压、voltage_window 电压窗口、"

            "c_rate 测试倍率、minimum_cycles 最少循环次数、minimum_retention 容量保持率、"

            "minimum_conductivity 离子电导率、maximum_viscosity 最大黏度、"

            "temperature 测试温度。组分/字段/机理：required_component 必须含某组分、"

            "excluded_component 必须排除某组分、field_presence 某配方字段须显式填写、"

            "electrode_material 电极材料、cell_configuration 电池形式、"

            "mechanism_keyword 机理关键词（仅表示某机理在文中被讨论，不得用于"

            "表达某结果“成立/不存在”的断言——此类要求用 evidence_claim）。断言："

            "evidence_claim＝关于该候选的"

            "定性或观测断言（如“无腐蚀”“无析钠”“提高首效”“报告保持率”），"

            "由 claim + claim_verdicts 表达，模型判语义、程序验引文。"

        ),

    )

    operator: Literal[">=", "<=", "==", "contains", "absent", "present"] | None = None

    value: float | str | list[float] | None = None

    unit: Literal[

        "V", "C", "cycles", "%",

        "mS/cm", "S/cm",

        "mPa s", "mPa·s", "cP",

        "°C", "℃",

    ] | None = Field(

        default=None,

        description=(

            "单位，必须与 kind 匹配：minimum_voltage / voltage_window → V；"

            "c_rate → C；minimum_cycles → cycles；minimum_retention → %；"

            "minimum_conductivity → mS/cm 或 S/cm；maximum_viscosity → mPa s、mPa·s 或 cP；"

            "temperature → °C 或 ℃。其余 kind（required_component、excluded_component、"

            "field_presence、electrode_material、cell_configuration、mechanism_keyword）"

            "不需要单位，保持 null。"

        ),

    )

    field: Literal[

        "salt",

        "concentration_mol_l",

        "solvents",

        "solvent_ratios",

        "additives",

        "additive_amounts",

    ] | None = None

    evidence_terms: list[str] = Field(default_factory=list)

    claim: str | None = Field(

        default=None,

        description=(

            "kind=evidence_claim 时必填：对单个候选须成立的断言，用用户原话表述，"

            "如“无集流体或壳体腐蚀被观测到”“报告相对室温容量的保持率”。"

            "不得写成关键词列表。"

        ),

    )

    claim_verdicts: list[ClaimVerdict] = Field(

        default_factory=list,

        description=(

            "kind=evidence_claim 时按候选逐个给出判定：每个候选一条 ClaimVerdict。"

            "supported/refuted 必须附 chunk_id 与逐字 quote（从该 chunk 原文截取）；"

            "证据未直接提及时如实给 not_found，由补证机制驱动下一轮检索。"

        ),

    )

    required_evidence_scope: Literal[

        "candidate_same_paper",

        "candidate_same_object",

        "candidate_same_object_if_resolved",

        "component",

        "background",

    ] = "candidate_same_paper"

    required_evidence_kinds: list[

        Literal[

            "formulation",

            "property",

            "condition",

            "mechanism",

            "risk",

            "background",

        ]

    ] = Field(default_factory=list)

    can_replan: bool = True

    search_hint: str = ""



    @model_validator(mode="after")

    def _validate_structure(self) -> "ScreeningConstraint":

        """把约束的“合不合法”校验在入口处完成，而不是让评估器事后纠正。



        这里只校验结构不变式——kind 与 unit/operator/value/field/evidence_terms

        的组合是否构成一个合法约束。它不接触候选证据，也不猜测模型意图；

        意图层面的错误（例如把“保持率”错写成 field_presence/additives）不属于

        结构错误，仍由调用方负责表达，见电解质设计技能中的约束书写规则。

        """

        numeric_kinds = {

            "minimum_voltage", "c_rate", "minimum_cycles", "minimum_retention",

            "minimum_conductivity", "maximum_viscosity", "temperature",

        }

        unit_by_kind = {

            "minimum_voltage": {"V"},

            "voltage_window": {"V"},

            "c_rate": {"C"},

            "minimum_cycles": {"cycles"},

            "minimum_retention": {"%"},

            "minimum_conductivity": {"mS/cm", "S/cm"},

            "maximum_viscosity": {"mPa s", "mPa·s", "cP"},

            "temperature": {"°C", "℃"},

        }

        evidence_term_kinds = {

            "electrode_material", "cell_configuration", "mechanism_keyword",

        }



        if self.kind in numeric_kinds and not isinstance(self.value, (int, float)):

            raise ValueError(

                f"kind={self.kind!r} 是数值约束，value 必须是数值，实际为 {self.value!r}。"

            )

        if self.kind in unit_by_kind and self.unit is not None:

            allowed = unit_by_kind[self.kind]

            if self.unit not in allowed:

                hint = ""

                if self.kind == "minimum_cycles" and self.unit.casefold() == "c":

                    hint = " C 表示倍率，不是循环次数；循环次数请用 unit=\"cycles\"，"

                    "指定倍率请改用 kind=\"c_rate\"。"

                raise ValueError(

                    f"kind={self.kind!r} 的 unit 不能是 {self.unit!r}；"

                    f"合法值：{sorted(allowed)}。{hint}"

                )

        if self.kind == "c_rate" and self.operator not in {None, "=="}:

            raise ValueError(

                "kind=c_rate 表示指定测试倍率，operator 只能是 ==，不得由更高倍率推定。"

            )

        if self.kind == "voltage_window":

            v = self.value

            if (

                not isinstance(v, list)

                or len(v) != 2

                or not all(

                    isinstance(x, (int, float)) and math.isfinite(x) for x in v

                )

                or v[0] > v[1]

                or self.operator not in {None, "=="}

            ):

                raise ValueError(

                    "kind=voltage_window 需要 value=[lower, upper] 两个有限数且 "

                    "lower <= upper，operator 只能是 ==。"

                )

        if self.kind in evidence_term_kinds and not [

            term for term in self.evidence_terms if term.strip()

        ]:

            raise ValueError(

                f"kind={self.kind!r} 需要至少一个非空 evidence_terms 才能做确定性匹配。"

            )

        if self.kind == "field_presence" and self.field is None:

            raise ValueError(

                "kind=field_presence 必须指定 field"

                "（salt/concentration_mol_l/solvents/solvent_ratios/additives/additive_amounts）。"

            )

        if self.kind == "evidence_claim":

            claim_text = str(self.claim or "").strip()

            if not claim_text:

                raise ValueError(

                    "kind=evidence_claim 必须在 claim 中写出对候选须成立的断言"

                    "（用用户原话，如“无集流体或壳体腐蚀被观测到”），不得留空。"

                )

            # 断言是关于该候选成立与否的命题，背景/组分级证据在定义上不能

            # 支持它；显式放宽到 component/background 是结构性错误，直接拒绝。

            if self.required_evidence_scope in {"component", "background"}:

                raise ValueError(

                    "kind=evidence_claim 的 required_evidence_scope 不能是 "

                    f"{self.required_evidence_scope!r}：断言必须由候选绑定的证据"

                    "（candidate_same_paper / candidate_same_object / "

                    "candidate_same_object_if_resolved）支持。"

                )

            if self.value is not None or self.unit is not None or self.field is not None:

                raise ValueError(

                    "kind=evidence_claim 不使用 value/unit/field；"

                    "断言内容写在 claim，逐候选判定写在 claim_verdicts。"

                )

        else:

            if self.claim is not None or self.claim_verdicts:

                raise ValueError(

                    f"kind={self.kind!r} 不使用 claim/claim_verdicts；"

                    "断言型要求请改用 kind=evidence_claim。"

                )

        return self





class JointContextRequirementRecord(BaseModel):

    """一组硬约束必须存在联合兼容实验上下文的显式运行时合同。"""



    group_id: str = Field(min_length=1)

    constraint_ids: list[str] = Field(min_length=2)

    context_fields: list[Literal[

        "scientific_object_id", "composition", "salt", "solvent", "additive",

        "temperature_c", "electrode", "cell_configuration", "voltage_window_v",

        "rate", "cycle_count", "measurement_definition",

    ]] = Field(min_length=1)

    target_context: dict[str, object] = Field(default_factory=dict)

    can_replan: bool = True

    search_hint: str = ""





class ScreeningRequest(BaseModel):

    candidates: list[FormulationCandidate]

    constraint_records: list[ScreeningConstraint] = Field(

        default_factory=list,

        description="提交本次用户要求的全部结构化筛选条件；不得用整段任务文本替代。",

    )

    joint_context_requirements: list[JointContextRequirementRecord] = Field(default_factory=list)

    # criteria/target_conditions 是旧调用方兼容缝。新设计任务应传 constraint_records，

    # 避免确定性 evaluator 再从自由文本猜测硬约束类型和阈值。

    # 仅为旧数据与离线复放保留；不向模型暴露会绕过结构化条件的旧入口。

    criteria: SkipJsonSchema[list[str]] = Field(default_factory=list)

    target_conditions: SkipJsonSchema[list[str]] = Field(default_factory=list)



    @model_validator(mode="after")

    def _validate_claim_verdict_targets(self) -> "ScreeningRequest":

        """断言判定的 candidate_id 必须指向本次请求的候选，防止张冠李戴。"""



        known_ids = {candidate.candidate_id for candidate in self.candidates}

        for constraint in self.constraint_records:

            if constraint.kind != "evidence_claim":

                continue

            unknown = [

                verdict.candidate_id

                for verdict in constraint.claim_verdicts

                if verdict.candidate_id not in known_ids

            ]

            if unknown:

                raise ValueError(

                    f"constraint {constraint.constraint_id!r} 的 claim_verdicts 引用了"

                    f"不存在的候选 {unknown}；只能对本次请求中的候选作判定。"

                    "修复时保留完整 constraint_records：仅调整 claim_verdicts 的候选对应关系，"

                    "或补齐本次 candidates。不得删除筛选条件来绕过参数校验。"

                )

        return self





class AgentScreeningRequest(ScreeningRequest):

    """在线工具显式要求结构化条件；旧离线输入继续由父模型处理。"""



    constraint_records: list[ScreeningConstraint] = Field(

        description="必填：本次用户要求的全部结构化筛选条件，不能只提交候选。",

    )



    @model_validator(mode="before")

    @classmethod

    def _explicit_legacy_input(cls, data: Any) -> Any:

        # 老版直接客户端已声明这些字段，仍可复放。它们不在 LLM 工具 schema 中，

        # 在线设计任务也会由运行时检查完整结构化条件，不能凭旧文本批准推荐。

        if isinstance(data, dict) and "constraint_records" not in data:

            if "criteria" in data or "target_conditions" in data:

                return {**data, "constraint_records": []}

        return data





class CriterionEvaluation(BaseModel):

    """候选配方对单条筛选条件的可审计判定。"""



    criterion: str

    constraint_id: str | None = None

    # Agent 在跨工具转述时可能省略内部分类；保留为空而非拒绝整个报告，

    # 但 screen_formulations 原始输出始终提供该字段。

    kind: str = ""

    status: Literal["satisfied", "failed", "insufficient_evidence", "unsupported"]

    expected: dict = Field(default_factory=dict)

    observed: list[dict] = Field(default_factory=list)

    reason: str = ""





class ScreeningResult(BaseModel):

    candidate_id: str

    passed: bool = False

    score: float = 0.0

    evidence_level: Literal["strong", "moderate", "weak"] = "weak"

    matched_properties: list[dict] = Field(default_factory=list)

    model_predictions: list[ModelPropertyPrediction] = Field(default_factory=list)

    criteria_evaluations: list[CriterionEvaluation] = Field(default_factory=list)

    gaps: list[str] = Field(default_factory=list)

    recommendation: str = ""





class SuspiciousKgItemRequest(BaseModel):

    item_type: Literal["entity", "relation", "evidence_chunk"]

    item_id: str | None = None

    description: str

    source_tool: str

    source_query: str | None = None

    evidence_refs: list[str] = Field(default_factory=list)

    severity: Literal["low", "medium", "high"] = "medium"





class SuspiciousKgItemResult(BaseModel):

    flag_id: str

    status: Literal["recorded"] = "recorded"

    message: str = ""





class KgFlagRecord(BaseModel):

    flag_id: str

    item_type: Literal["entity", "relation", "evidence_chunk"]

    item_id: str | None = None

    description: str

    source_tool: str

    source_query: str | None = None

    evidence_refs: list[str] = Field(default_factory=list)

    severity: Literal["low", "medium", "high"] = "medium"

    status: Literal["pending", "confirmed_valid", "confirmed_invalid", "applied", "ignored"] = "pending"

    created_at: str = ""

    reviewed_at: str | None = None

    reviewer_note: str | None = None





class ReviewKgFlagsRequest(BaseModel):

    severity: Literal["low", "medium", "high", "all"] = "all"

    limit: int = 20

    offset: int = 0





class KgCorrectionRequest(BaseModel):

    flag_id: str

    action: Literal["add_relation", "delete_relation", "merge_entity", "delete_entity", "ignore"]

    source_entity: str

    relation: str | None = None

    target_entity: str | None = None

    entity_labels: list[str] | None = None

    evidence_chunk_ids: list[str] = Field(default_factory=list)

    notes: str = ""

