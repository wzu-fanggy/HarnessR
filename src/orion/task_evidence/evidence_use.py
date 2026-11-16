"""Task- and condition-relative scientific evidence-use decisions.



This module makes Orion's evidence-use semantics explicit.  It does not decide

scientific truth.  Instead it records what an already represented evidence

state is allowed to do in the current task and target-condition context.



The key invariant is context locking: an EvidenceUseDecision is valid only for

its exact task/use/claim/evidence/applicability context.  Reusing the same

source under a different target condition or downstream task requires a fresh

decision rather than silently inheriting an earlier permission.

"""



from __future__ import annotations



import hashlib

import json

from collections.abc import Iterable, Mapping, Sequence

from typing import Any, Literal, TypeAlias



from pydantic import BaseModel, ConfigDict, Field



from orion.core.claim_contract import ClaimContract

from orion.task_evidence.evidence_applicability import ApplicabilityRecord, ConditionField





EvidenceUseTask: TypeAlias = Literal[

    "evidence_synthesis",

    "candidate_screening",

    "design",

    "inverse_design",

]

EvidenceUseKind: TypeAlias = Literal[

    "report",

    "compare",

    "recommend",

    "diagnose",

    "adjust",

]

EvidenceUseDisposition: TypeAlias = Literal["allow", "qualify", "block"]



_TASK_ALLOWED_USES: dict[EvidenceUseTask, frozenset[EvidenceUseKind]] = {

    "evidence_synthesis": frozenset({"report"}),

    "candidate_screening": frozenset({"report", "compare"}),

    "design": frozenset({"report", "compare", "recommend", "adjust"}),

    "inverse_design": frozenset(

        {"report", "compare", "recommend", "diagnose", "adjust"}

    ),

}



_CLAIM_USE: dict[str, EvidenceUseKind] = {

    "fact": "report",

    "comparison": "compare",

    "ranking": "compare",

    "recommendation": "recommend",

    "conditional_adjustment": "adjust",

    "diagnosis": "diagnose",

}



_STRUCTURED_RULES_BY_USE: dict[EvidenceUseKind, frozenset[str]] = {

    "recommend": frozenset({"structured_design_decision_v1"}),

    "diagnose": frozenset({"structured_inverse_hypothesis_gate_v1"}),

    "adjust": frozenset({"structured_design_decision_v1", "structured_inverse_hypothesis_gate_v1"}),

    "report": frozenset(),

    "compare": frozenset(),

}





class EvidenceUseDecision(BaseModel):

    """Auditable permission for one represented claim in one scientific context."""



    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(min_length=1)

    task_type: EvidenceUseTask

    use_kind: EvidenceUseKind

    claim_id: str = Field(min_length=1)

    claim_type: str = Field(min_length=1)

    evidence_keys: list[str] = Field(default_factory=list)

    scientific_object_id: str | None = None

    experiment_id: str | None = None

    applicability_status: Literal[

        "direct", "transferable", "insufficient", "conflicting", "not_scoped"

    ] = "not_scoped"

    disposition: EvidenceUseDisposition

    reason_codes: list[str] = Field(default_factory=list)

    relevant_condition_fields: list[ConditionField] = Field(default_factory=list)

    context_fingerprint: str = Field(min_length=64, max_length=64)





def evidence_use_kind_for_claim(claim: ClaimContract) -> EvidenceUseKind:

    """Map an accepted ClaimContract to the downstream use it requests."""



    return _CLAIM_USE.get(str(claim.claim_type), "report")





def _jsonable(value: Any) -> Any:

    if hasattr(value, "model_dump"):

        return _jsonable(value.model_dump(mode="json"))

    if isinstance(value, Mapping):

        return {str(key): _jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):

        return [_jsonable(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:

        return value

    return str(value)





def _evidence_keys(claim: ClaimContract) -> list[str]:

    values: list[str] = []

    for ref in claim.source_refs:

        key = str(ref.chunk_id or ref.kg_entity_id or "").strip()

        if key:

            values.append(key)

    for result in claim.verification_results:

        for key in sorted(result.data_used):

            if key in {"decision", "assessment", "candidate_evaluations"}:

                values.append(f"computed:{result.rule_id or result.verifier}:{key}")

    return list(dict.fromkeys(values))





def _matching_applicability_record(

    claim: ClaimContract,

    records: Sequence[ApplicabilityRecord],

    *,

    ordinal: int,

) -> ApplicabilityRecord | None:

    aliases = {str(claim.claim_id or ""), f"claim-{ordinal}"}

    for item in records:

        if item.claim_id and item.claim_id in aliases:

            return item

    global_records = [item for item in records if not item.claim_id]

    if len(global_records) == 1:

        return global_records[0]

    return None





def _source_binding_ok(claim: ClaimContract) -> bool:

    if claim.evidence_binding in {"quote_bound", "condition_bound"}:

        return bool(

            claim.source_refs

            and all(

                str(ref.chunk_id or ref.kg_entity_id or "").strip()

                and str(ref.raw_quote or "").strip()

                for ref in claim.source_refs

            )

        )

    if claim.evidence_binding == "computation_bound":

        return bool(

            claim.verification_status == "passed"

            and any(item.status == "passed" for item in claim.verification_results)

        )

    if claim.evidence_binding == "human_bound":

        return claim.verification_status in {"passed", "pending"}

    return False





def _structured_use_authorized(claim: ClaimContract, use_kind: EvidenceUseKind) -> bool:

    required_rules = _STRUCTURED_RULES_BY_USE[use_kind]

    if not required_rules:

        return True

    passed_rules = {

        str(item.rule_id or "")

        for item in claim.verification_results

        if item.status == "passed" and item.rule_id

    }

    if required_rules.intersection(passed_rules):

        return True

    if use_kind != "recommend":

        return False

    # 新路径验证的是模型审查的引文及决策一致性，不声称证明了科学语义。

    from orion.core.semantic_evidence_review import SemanticEvidenceReview, validate_review

    for verification in claim.verification_results:

        if (verification.status != "passed"

                or verification.rule_id != "model_semantic_review_with_source_verification_v1"):

            continue

        try:

            decision = verification.data_used["decision"]

            reviews = [SemanticEvidenceReview.model_validate(item)

                       for item in decision["semantic_evidence_reviews"]]

            if not reviews:

                return False

            sources = {ref.chunk_id: ref.raw_quote for ref in claim.source_refs if ref.chunk_id}

            computations = verification.data_used.get("recorded_tool_predictions") or []

            for review in reviews:

                validate_review(review, review.candidate_id, sources, computations)

            supported = {review.candidate_id for review in reviews if review.recommendation_supported}

            return (decision["status"] in {"recommend", "hold"}

                    and set(decision["approved_candidate_ids"]) == supported

                    and ((decision["status"] == "recommend") == bool(supported)))

        except (KeyError, TypeError, ValueError):

            return False

    return False





def _project_condition_scope(

    scope: Any,

    relevant_fields: Sequence[ConditionField],

) -> Any:

    """Keep only condition dimensions declared relevant to the current claim.



    If the condition payload is not a flat applicability scope (for example a

    computation-bound claim carrying nested input scopes), retain it verbatim;

    silently dropping unknown structure would make the context lock weaker.

    """



    if scope is None:

        return None

    payload = _jsonable(scope)

    if not isinstance(payload, Mapping):

        return payload

    if not relevant_fields:

        return payload

    projected = {

        field: payload.get(field)

        for field in relevant_fields

        if field in payload

    }

    return projected if projected else payload





def _applicability_context_payload(record: ApplicabilityRecord | None) -> Any:

    """Return only evidence-use-relevant applicability state.



    Runtime bookkeeping such as assessment IDs, retrieval routing, and terminal

    disposition must not revoke a permission by itself. Scientific-use context

    depends on the relevant dimensions, their source/target values, conflict or

    bridge evidence, and the reviewed transfer decision.

    """



    if record is None:

        return None

    relevant_fields = list(record.assessment.relevant_fields)

    return {

        "status": record.assessment.status,

        "relevant_fields": relevant_fields,

        "source_scope": _project_condition_scope(record.source_scope, relevant_fields),

        "target_scope": _project_condition_scope(record.target_scope, relevant_fields),

        "validated_conflict_evidence_ids": list(

            record.assessment.validated_conflict_evidence_ids

        ),

        "pending_bridge_evidence_ids": list(record.assessment.pending_bridge_evidence_ids),

        "transfer_contract_id": record.transfer_contract_id,

        "transfer_authorized_uses": list(record.transfer_authorized_uses),

        "transfer_contract_check": (

            record.transfer_contract_check.model_dump(mode="json")

            if record.transfer_contract_check is not None

            else None

        ),

    }





def evidence_use_context_fingerprint(

    *,

    task_type: EvidenceUseTask,

    claim: ClaimContract,

    applicability_record: ApplicabilityRecord | None,

) -> str:

    """Hash the scientifically relevant context for one evidence-use decision.



    The lock is relevance-scoped: changing a condition dimension outside the

    claim's declared applicability fields leaves the permission reusable, while

    changing a relevant dimension, task, evidence identity, object, experiment,

    or transfer/conflict basis invalidates it.

    """



    relevant_fields = (

        list(applicability_record.assessment.relevant_fields)

        if applicability_record is not None

        else []

    )

    payload = {

        "task_type": task_type,

        "use_kind": evidence_use_kind_for_claim(claim),

        "claim_id": claim.claim_id,

        "claim_type": claim.claim_type,

        "claim_text": claim.claim_text,

        "evidence_binding": claim.evidence_binding,

        "evidence_keys": _evidence_keys(claim),

        "scientific_object_id": claim.scientific_object_id,

        "experiment_id": claim.experiment_id,

        "condition_scope": _project_condition_scope(

            claim.condition_scope,

            relevant_fields,

        ),

        "applicability": _applicability_context_payload(applicability_record),

    }

    raw = json.dumps(

        _jsonable(payload),

        ensure_ascii=False,

        sort_keys=True,

        separators=(",", ":"),

    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()





def compile_evidence_use_decisions(

    *,

    task_type: EvidenceUseTask,

    claims: Sequence[ClaimContract],

    applicability_records: Sequence[ApplicabilityRecord] = (),

) -> list[EvidenceUseDecision]:

    """Compile accepted claim contracts into task-relative evidence-use permissions.



    The compiler is intentionally conservative:

    - task/use incompatibility blocks use;

    - missing source/computation binding blocks use;

    - insufficient/conflicting applicability blocks use;

    - transferable evidence is qualified rather than treated as direct;

    - recommendation/diagnosis/adjustment require a passed structured gate.

    """



    parsed_task: EvidenceUseTask = task_type

    decisions: list[EvidenceUseDecision] = []

    for ordinal, claim in enumerate(claims, start=1):

        use_kind = evidence_use_kind_for_claim(claim)

        record = _matching_applicability_record(

            claim,

            applicability_records,

            ordinal=ordinal,

        )

        reasons: list[str] = []

        disposition: EvidenceUseDisposition = "allow"



        if use_kind not in _TASK_ALLOWED_USES[parsed_task]:

            disposition = "block"

            reasons.append("task_use_not_allowed")

        if not _source_binding_ok(claim):

            disposition = "block"

            reasons.append("evidence_binding_not_authorized")

        if not _structured_use_authorized(claim, use_kind):

            disposition = "block"

            reasons.append("structured_gate_not_authorized")



        applicability_status = "not_scoped"

        if record is not None:

            applicability_status = record.assessment.status

            if record.assessment.status in {"insufficient", "conflicting"}:

                disposition = "block"

                reasons.append(f"applicability_{record.assessment.status}")

            elif record.assessment.status == "transferable" and disposition != "block":

                if use_kind not in set(record.transfer_authorized_uses):

                    disposition = "block"

                    reasons.append("transfer_use_not_authorized")

                else:

                    disposition = "qualify"

                    reasons.append("reviewed_transfer_only")

            elif record.assessment.status == "direct":

                reasons.append("direct_applicability")



        if disposition == "allow" and not reasons:

            reasons.append("task_and_binding_authorized")



        fingerprint = evidence_use_context_fingerprint(

            task_type=parsed_task,

            claim=claim,

            applicability_record=record,

        )

        decision_seed = f"{claim.claim_id}:{use_kind}:{fingerprint}"

        decisions.append(

            EvidenceUseDecision(

                decision_id=(

                    "use-" + hashlib.sha256(decision_seed.encode("utf-8")).hexdigest()[:16]

                ),

                task_type=parsed_task,

                use_kind=use_kind,

                claim_id=claim.claim_id,

                claim_type=str(claim.claim_type),

                evidence_keys=_evidence_keys(claim),

                scientific_object_id=claim.scientific_object_id,

                experiment_id=claim.experiment_id,

                applicability_status=applicability_status,

                disposition=disposition,

                reason_codes=list(dict.fromkeys(reasons)),

                relevant_condition_fields=(

                    list(record.assessment.relevant_fields) if record is not None else []

                ),

                context_fingerprint=fingerprint,

            )

        )

    return decisions





def evidence_use_context_matches(

    decision: EvidenceUseDecision,

    *,

    task_type: EvidenceUseTask,

    claim: ClaimContract,

    applicability_record: ApplicabilityRecord | None,

) -> bool:

    """Return whether an earlier permission is reusable in the current context."""



    if decision.task_type != task_type:

        return False

    if decision.claim_id != claim.claim_id:

        return False

    if decision.use_kind != evidence_use_kind_for_claim(claim):

        return False

    return decision.context_fingerprint == evidence_use_context_fingerprint(

        task_type=task_type,

        claim=claim,

        applicability_record=applicability_record,

    )





def unauthorized_evidence_use_ids(

    decisions: Iterable[EvidenceUseDecision],

) -> list[str]:

    """Return claim ids whose represented use is blocked."""



    return list(

        dict.fromkeys(

            item.claim_id for item in decisions if item.disposition == "block"

        )

    )

