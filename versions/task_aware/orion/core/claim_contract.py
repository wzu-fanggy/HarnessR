from __future__ import annotations



import hashlib

from typing import Any, Literal



from pydantic import BaseModel, Field, model_validator





class ClaimProvenance(BaseModel):

    tool_call_id: str | None = None

    tool_name: str | None = None

    source_kind: Literal["chunk", "kg", "computation", "human"] = "chunk"





class SourceRef(BaseModel):

    chunk_id: str | None = None

    paper_id: str | None = None

    kg_entity_id: str | None = None

    raw_quote: str = ""

    span: dict[str, Any] | None = None





class VerificationResult(BaseModel):

    verifier: str

    tier: Literal["high", "medium", "low", "blocked"] | None = None

    status: Literal["pending", "passed", "failed", "inconclusive", "skipped"] = "pending"

    gap: str | None = None

    rule_id: str | None = None

    data_used: dict[str, Any] = Field(default_factory=dict)





class ClaimContract(BaseModel):

    claim_id: str = ""

    claim_text: str = ""

    claim_type: Literal[

        "fact", "comparison", "ranking",

        "recommendation", "conditional_adjustment", "diagnosis"

    ] = "fact"

    provenance: ClaimProvenance = Field(default_factory=ClaimProvenance)

    source_refs: list[SourceRef] = Field(default_factory=list)

    condition_scope: dict[str, Any] | None = None

    experiment_id: str | None = None

    # 可选的科学对象身份（例如具体 formulation/candidate record）。只有上游

    # 已显式提供或由确定性解析确认时才填写；不得用 paper_id 猜测对象同一性。

    scientific_object_id: str | None = None

    evidence_binding: Literal[

        "quote_bound", "condition_bound", "computation_bound", "human_bound"

    ] = "quote_bound"

    confidence: float | None = None

    verification_status: Literal[

        "pending", "passed", "failed", "inconclusive", "skipped"

    ] = "pending"

    verification_results: list[VerificationResult] = Field(default_factory=list)

    reflection_tags: list[str] = Field(default_factory=list)



    @model_validator(mode="before")

    @classmethod

    def _compat_synthesis_claim(cls, value: Any) -> Any:

        """兼容旧 SynthesisClaim 字段：text/chunk_id/evidence_quote/binding_requirement。"""

        if not isinstance(value, dict):

            return value

        payload = dict(value)



        # 字段映射

        if "text" in payload and "claim_text" not in payload:

            payload["claim_text"] = payload.pop("text")

        if "binding_requirement" in payload and "evidence_binding" not in payload:

            payload["evidence_binding"] = payload.pop("binding_requirement")



        # claim_type 映射

        claim_type = payload.get("claim_type")

        if claim_type == "hypothesis":

            payload["claim_type"] = "diagnosis"



        # source_refs 映射

        chunk_id = payload.pop("chunk_id", None)

        evidence_quote = payload.pop("evidence_quote", None)

        if chunk_id or evidence_quote:

            source_refs = payload.get("source_refs") or []

            if not source_refs:

                source_refs = [SourceRef(chunk_id=chunk_id, raw_quote=evidence_quote or "")]

            elif isinstance(source_refs, list) and source_refs:

                first = source_refs[0] if isinstance(source_refs[0], dict) else source_refs[0].model_dump()

                first.setdefault("chunk_id", chunk_id)

                first.setdefault("raw_quote", evidence_quote or "")

                source_refs[0] = first

            payload["source_refs"] = source_refs



        # condition_scope 映射：旧 SynthesisClaim 直接携带 condition_scope 对象

        condition_scope = payload.get("condition_scope")

        if condition_scope is not None and hasattr(condition_scope, "model_dump"):

            payload["condition_scope"] = condition_scope.model_dump()



        # claim_id 缺省生成

        if not payload.get("claim_id"):

            text = payload.get("claim_text", "")

            cid = chunk_id or ""

            payload["claim_id"] = f"claim-{hashlib.sha256(f'{cid}:{text}'.encode()).hexdigest()[:12]}"



        # provenance 缺省

        if "provenance" not in payload:

            payload["provenance"] = ClaimProvenance(tool_name="synthesize_design_report")



        return payload



    @property

    def text(self) -> str:

        return self.claim_text



    @property

    def chunk_id(self) -> str | None:

        return self.source_refs[0].chunk_id if self.source_refs else None



    @property

    def evidence_quote(self) -> str:

        return self.source_refs[0].raw_quote if self.source_refs else ""



    @property

    def binding_requirement(self) -> str:

        return self.evidence_binding

