from __future__ import annotations



from typing import Literal



from pydantic import AliasChoices, BaseModel, Field, field_validator



from tools.kg.models import ModelPropertyPrediction, coerce_evidence_level





EvidenceKind = Literal["formulation", "property", "condition", "mechanism", "risk"]

EvidenceLevel = Literal["strong", "moderate", "weak"]





class DesignRequest(BaseModel):

    task_type: Literal["design", "screen", "screening", "evidence", "inverse_design", "inverse"]

    objectives: list[str] = Field(default_factory=list)

    temperature_c: float | None = None

    electrode: str | None = None

    salts: list[str] = Field(default_factory=list)

    solvents: list[str] = Field(default_factory=list)

    additives: list[str] = Field(default_factory=list)

    excluded_components: list[str] = Field(default_factory=list)

    constraints: list[str] = Field(default_factory=list)

    max_candidates: int = Field(default=3, ge=1, le=10)





class EvidenceTriple(BaseModel):

    s: str

    r: str

    t: str





class EvidenceChunk(BaseModel):

    """KG/RAG 检索返回的可审计证据块契约。"""



    chunk_id: str

    content: str = ""

    paper_title: str = ""

    section_title: str = ""

    score: float | None = None

    entities: list[str] = Field(default_factory=list)

    triples: list[EvidenceTriple] = Field(default_factory=list)





class EvidenceRef(BaseModel):

    # chunk_id 允许省略（LLM 组织证据时可能丢失），但检索/审计链路应始终带上

    chunk_id: str = ""

    paper_title: str = ""

    section_title: str = ""

    score: float | None = None

    excerpt: str = ""

    evidence_kind: EvidenceKind

    # scientific_object_id / experiment_id 只承载上游显式身份；缺失时保持未知，

    # 不从 paper_title、candidate_id 或 chunk_id 反推对象/实验同一性。

    # condition_scope 可由 source text 确定性抽取，但这不构成实验身份的证明。

    scientific_object_id: str | None = None

    experiment_id: str | None = None

    condition_scope: dict[str, object] | None = None





class FormulationCandidate(BaseModel):

    candidate_id: str

    scientific_object_id: str | None = None

    salt: str | None = None

    salts: list[str] = Field(default_factory=list)

    salt_concentrations_mol_l: dict[str, float] = Field(default_factory=dict)

    concentration_mol_l: float | None = None

    solvents: list[str] = Field(default_factory=list)

    solvent_ratios: dict[str, float] = Field(default_factory=dict)

    solvent_ratio_basis: Literal["mass", "volume", "molar", "unknown"] = "unknown"

    additives: list[str] = Field(default_factory=list)

    additive_amounts: dict[str, str] = Field(default_factory=dict)

    target_rationale: list[str] = Field(default_factory=list)

    evidence: list[EvidenceRef] = Field(default_factory=list)

    evidence_level: EvidenceLevel = "moderate"

    inferred_fields: list[str] = Field(default_factory=list)

    unresolved_gaps: list[str] = Field(default_factory=list)

    evidence_scope: Literal["same_child", "same_paper", "unknown"] = "unknown"

    model_predictions: list[ModelPropertyPrediction] = Field(default_factory=list)



    @field_validator("evidence_level", mode="before")

    @classmethod

    def _coerce_level(cls, v):

        return coerce_evidence_level(v)





class ValidationResult(BaseModel):

    candidate_id: str = ""

    valid: bool = Field(default=True, validation_alias=AliasChoices("valid", "passed"))

    blocking_issues: list[str] = Field(

        default_factory=list,

        validation_alias=AliasChoices("blocking_issues", "gaps"),

    )

    warnings: list[str] = Field(default_factory=list)

    evidence_level: EvidenceLevel = "moderate"

    evidence_coverage: dict[str, bool] = Field(default_factory=dict)

    risk_flags: list[str] = Field(default_factory=list)

    evidence: list[EvidenceRef] = Field(default_factory=list)



    @field_validator("evidence_level", mode="before")

    @classmethod

    def _coerce_level(cls, v):

        return coerce_evidence_level(v)





class PropertyObservation(BaseModel):

    property_name: str

    numeric_value: float | None = None

    unit: str | None = None

    raw_value: str

    temperature_c: float | None = None

    electrode: str | None = None

    test_condition: str = ""

    evidence: EvidenceRef

    extraction_confidence: Literal["high", "medium", "low"]





class ScoreCard(BaseModel):

    candidate_id: str = ""

    total_score: float = Field(default=0.0, validation_alias=AliasChoices("total_score", "score"))

    objective_score: float = 0.0

    evidence_score: float = 0.0

    risk_penalty: float = 0.0

    uncertainty_penalty: float = 0.0

    score_breakdown: dict[str, float] = Field(default_factory=dict)

    ranking_rationale: list[str] = Field(default_factory=list)

    evidence: list[EvidenceRef] = Field(default_factory=list)





class OptimizationProposal(BaseModel):

    parent_candidate_id: str

    proposed_candidate: FormulationCandidate | None = None

    action: Literal["retrieve_more_evidence", "run_vasp", "run_experiment", "vary_component", "hold"]

    rationale: str

    required_evidence: list[str] = Field(default_factory=list)

    required_validation: list[str] = Field(default_factory=list)

