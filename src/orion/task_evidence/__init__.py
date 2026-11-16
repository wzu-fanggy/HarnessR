"""Orion 离线任务证据与 benchmark 支持。"""



from .models import (

    EvidenceAssertion,

    ExperimentRecord,

    Observation,

    SourceSpan,

    TaskContract,

    TaskRunArtifact,

)

from .contracts_v2 import (

    AgentTaskContract,

    ArtifactClaim,

    ArtifactEvidence,

    ArtifactGap,

    CandidateScreeningArtifact,

    DesignTaskArtifact,

    EvidenceSynthesisArtifact,

    InverseDesignTaskArtifact,

    TaskArtifact,

    canonical_task_type,

    default_task_contract,

    validate_task_artifact_v2,

)

from .evidence_applicability import (

    ApplicabilityAssessment,

    ApplicabilityConditionScope,

    ApplicabilityDecision,

    ApplicabilityInput,

    ApplicabilityRecord,

    ApplicabilityStatus,

    ConditionField,

    TransferContract,

    TransferContractCheck,

    TransferEvidenceRef,

    TransferReview,

    TransferUseKind,

    assess_evidence_applicability,

    check_transfer_contract_binding,

    evaluate_applicability_input,

    preferred_applicability_decision,

)

from .evidence_use import (

    EvidenceUseDecision,

    EvidenceUseDisposition,

    EvidenceUseKind,

    EvidenceUseTask,

    compile_evidence_use_decisions,

    evidence_use_context_fingerprint,

    evidence_use_context_matches,

    evidence_use_kind_for_claim,

    unauthorized_evidence_use_ids,

)

from .task_gates import (

    CandidateComparisonRecord,

    ComparabilityAssessment,

    ComparisonConditions,

    DiscriminatingTest,

    GateIssue,

    HypothesisRecord,

    InverseHypothesisAssessment,

    evaluate_candidate_comparability,

    evaluate_inverse_hypotheses,

)



__all__ = [

    "EvidenceAssertion",

    "ExperimentRecord",

    "Observation",

    "SourceSpan",

    "TaskContract",

    "TaskRunArtifact",

    "AgentTaskContract",

    "ArtifactClaim",

    "ArtifactEvidence",

    "ArtifactGap",

    "CandidateScreeningArtifact",

    "DesignTaskArtifact",

    "EvidenceSynthesisArtifact",

    "InverseDesignTaskArtifact",

    "TaskArtifact",

    "canonical_task_type",

    "default_task_contract",

    "validate_task_artifact_v2",

    "ApplicabilityAssessment",

    "ApplicabilityConditionScope",

    "ApplicabilityDecision",

    "ApplicabilityInput",

    "ApplicabilityRecord",

    "ApplicabilityStatus",

    "ConditionField",

    "TransferContract",

    "TransferContractCheck",

    "TransferEvidenceRef",

    "TransferReview",

    "TransferUseKind",

    "assess_evidence_applicability",

    "check_transfer_contract_binding",

    "evaluate_applicability_input",

    "preferred_applicability_decision",

    "EvidenceUseDecision",

    "EvidenceUseDisposition",

    "EvidenceUseKind",

    "EvidenceUseTask",

    "compile_evidence_use_decisions",

    "evidence_use_context_fingerprint",

    "evidence_use_context_matches",

    "evidence_use_kind_for_claim",

    "unauthorized_evidence_use_ids",

    "CandidateComparisonRecord",

    "ComparabilityAssessment",

    "ComparisonConditions",

    "DiscriminatingTest",

    "GateIssue",

    "HypothesisRecord",

    "InverseHypothesisAssessment",

    "evaluate_candidate_comparability",

    "evaluate_inverse_hypotheses",

]

