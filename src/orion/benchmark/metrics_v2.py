"""统一 TaskArtifact v2 的确定性离线指标。



这些指标不调用模型，也不把自然语言相似度当作任务正确性。comparability accuracy

需要 fixture 显式提供 gold 状态；其余指标直接检查 artifact 的结构化门禁与证据绑定。

"""



from __future__ import annotations



from typing import Any



from pydantic import ValidationError



from orion.task_evidence.contracts_v2 import (

    CandidateScreeningArtifact,

    InverseDesignTaskArtifact,

    TaskArtifact,

    validate_task_artifact_v2,

)





def _safe_ratio(numerator: int, denominator: int, empty_value: float) -> float:

    return empty_value if denominator == 0 else numerator / denominator





def _claim_evidence_binding_rate(artifact: TaskArtifact) -> float:
    # Legacy/compiled artifacts may bind ``Claim.evidence_ids`` to the stable
    # source chunk ID while newer callers use the ledger-local evidence ID.
    # Both point to the same authoritative ledger row and must count as bound.
    known = {
        identifier
        for item in artifact.evidence_ledger
        for identifier in (item.evidence_id, item.chunk_id)
        if identifier
    }
    supported = [

        claim

        for claim in artifact.claims

        if claim.evidence_ids and set(claim.evidence_ids) <= known

    ]

    return _safe_ratio(len(supported), len(artifact.claims), 1.0)





def _candidate_scores(

    artifact: CandidateScreeningArtifact,

    expected: dict[str, Any],

) -> dict[str, float]:

    gold = expected.get("condition_comparability")

    rankings = [

        claim for claim in artifact.claims if claim.claim_type == "ranking"

    ]

    known = {item.evidence_id for item in artifact.evidence_ledger}

    unsupported = [

        claim

        for claim in rankings

        if (

            not claim.evidence_ids

            or not set(claim.evidence_ids) <= known

            or artifact.condition_comparability != "comparable"

            or artifact.task_status != "compared"

        )

    ]

    assessment = artifact.comparability_assessment

    gate_consistent = bool(

        assessment is not None

        and assessment.status == artifact.condition_comparability

        and (

            artifact.task_status != "compared"

            or assessment.status == "comparable"

        )

    )

    return {

        "comparability_accuracy": (

            0.0

            if gold is None

            else float(artifact.condition_comparability == gold)

        ),

        "comparability_gate_consistency": float(gate_consistent),

        "unsupported_ranking_rate": _safe_ratio(

            len(unsupported),

            len(rankings),

            0.0,

        ),

    }





def _inverse_scores(

    artifact: InverseDesignTaskArtifact,

    expected: dict[str, Any],

) -> dict[str, float]:

    assessment = artifact.hypothesis_assessment

    valid_ids = set(assessment.valid_hypothesis_ids if assessment else [])

    expected_ids = {

        str(item) for item in expected.get("hypothesis_ids", []) if item

    }

    if expected_ids:

        hypothesis_coverage = len(valid_ids & expected_ids) / len(expected_ids)

    else:

        hypothesis_coverage = min(len(valid_ids) / 2, 1.0)

    tests = artifact.discriminating_tests

    adequate_ids = set(assessment.adequate_test_ids if assessment else [])

    return {

        "competing_hypothesis_coverage": hypothesis_coverage,

        "discriminating_test_adequacy": _safe_ratio(

            sum(test.test_id in adequate_ids for test in tests),

            len(tests),

            0.0,

        ),

        "hypothesis_gate_consistency": float(

            assessment is not None

            and assessment.status == artifact.task_status

        ),

    }





def score_task_artifact_v2(

    artifact: TaskArtifact | dict[str, Any],

    expected: dict[str, Any] | None = None,

) -> dict[str, float]:

    """按 task_type 评分统一 artifact；非法 artifact 返回明确合同失败。"""



    try:

        parsed = validate_task_artifact_v2(artifact)

    except (ValidationError, ValueError, TypeError):

        return {"artifact_contract_validity": 0.0}



    scores = {

        "artifact_contract_validity": 1.0,

        "claim_evidence_binding_rate": _claim_evidence_binding_rate(parsed),

    }

    expected = expected or {}

    if isinstance(parsed, CandidateScreeningArtifact):

        scores.update(_candidate_scores(parsed, expected))

    elif isinstance(parsed, InverseDesignTaskArtifact):

        scores.update(_inverse_scores(parsed, expected))

    return scores

