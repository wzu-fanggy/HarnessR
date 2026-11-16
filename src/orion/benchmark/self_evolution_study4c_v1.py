"""Study 4C 新家族四臂实验的零模型合同与冻结前校验。

本模块把“把经验写成一个完整 Skill 就够了”的反驳直接做成强对照：

``A_static``
    不提供历史候选。
``B_unconditional``
    不经适用性门禁，始终提供与 M/G 同一完整 Skill，并把候选编译为第一次科学行动
    的 planning seed。
``M_strong_skill``
    向模型提供包含 invocation、contraindications、逐字证据和 provenance 的完整 Skill，
    但不提供 admission verdict 或可执行 permission，由模型自行决定是否调用。
``G_evidence_governed``
    科学内容与 M 完全同源；先由确定性 applicability gate 判断，再决定是否把候选
    编译为 planning seed。门禁状态不进入 Skill 文本。

导入或调用本模块不会访问 provider、RAG、KG 或 Neo4j，也不会创建正式结果目录。
现有 reviewed applicability 标注只作为单作者批准的 source curation / deterministic
gate fixture，不冒充独立化学家盲审。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.benchmark.self_evolution_followup_planning_integration_v2 import (
    FollowupPlanningAnchorV2,
    FollowupPlanningSeedV2,
    followup_planning_seed_sha256,
)
from orion.evaluation.models import EvaluationSample
from orion.task_evidence.evidence_applicability import (
    CONDITION_FIELDS,
    ApplicabilityConditionScope,
    ApplicabilityInput,
    ApplicabilityRecord,
    ConditionField,
    TransferContract,
    TransferEvidenceRef,
    TransferReview,
    evaluate_applicability_input,
)


MODULE_PATH = Path(__file__).resolve()
BENCHMARK_ROOT = MODULE_PATH.parent
ORION_ROOT = BENCHMARK_ROOT.parent
SOURCE_BANK_PATH = (
    BENCHMARK_ROOT
    / "self_evolution_study4c_v1_20260822"
    / "STUDY4C_SOURCE_BANK_DRAFT_V1.json"
)
REVIEWED_CASES_PATH = (
    BENCHMARK_ROOT
    / "evidence_applicability_v1_reviewed_2026-08-08"
    / "cases.jsonl"
)
V2_CASES_PATH = (
    BENCHMARK_ROOT
    / "evidence_applicability_v2_holdout_draft_2026-08-08"
    / "cases.jsonl"
)
V2_REVIEW_LEDGER_PATH = (
    BENCHMARK_ROOT
    / "evidence_applicability_v2_holdout_reviewed_2026-08-08"
    / "review_ledger.json"
)

Study4CArm = Literal[
    "A_static",
    "B_unconditional",
    "M_strong_skill",
    "G_evidence_governed",
]
Study4CEndpointRole = Literal[
    "useful_applicable",
    "unresolved_containment",
    "disqualifying",
    "preservation_non_disqualifying",
]

ARMS: tuple[Study4CArm, ...] = (
    "A_static",
    "B_unconditional",
    "M_strong_skill",
    "G_evidence_governed",
)
_EXPECTED_ROLE_COUNTS = {
    "useful_applicable": 2,
    "unresolved_containment": 3,
    "disqualifying": 1,
    "preservation_non_disqualifying": 2,
}
_GOVERNED_RELEASE_STATUSES = {"direct", "transferable"}
_FORBIDDEN_SKILL_KEYS = {
    "applicability_status",
    "allowed_decision",
    "admission",
    "admission_result",
    "boundary_relation",
    "gate_relation",
    "gate_verdict",
    "governed_candidate_available",
    "permission",
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical(value).encode("utf-8"))


def _relative_orion_path(path: Path) -> str:
    return path.resolve().relative_to(ORION_ROOT).as_posix()


def _render_scope_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return _canonical(value)
    return str(value)


def _iter_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_iter_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_iter_keys(item))
    return keys


class Study4CEvidenceRefV1(BaseModel):
    """Strong Skill 内逐字绑定的证据引用。"""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    raw_quote: str = Field(min_length=12)
    role: Literal["source", "bridge", "conflict"]


class Study4CStrongSkillRecordV1(BaseModel):
    """M/G 共用的科学内容；刻意不含 verdict 或 permission。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "orion_self_evolution_study4c_strong_skill_record_v1"
    ] = "orion_self_evolution_study4c_strong_skill_record_v1"
    source_asset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    candidate_seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    guidance: str = Field(min_length=1)
    action_moves: list[str] = Field(min_length=1)
    invocation_conditions: list[str] = Field(min_length=1)
    contraindications: list[str] = Field(min_length=1)
    limitations: list[str] = Field(min_length=1)
    confidence: dict[str, str]
    claimed_source_scope: ApplicabilityConditionScope
    relevant_fields: list[ConditionField] = Field(min_length=1)
    supporting_evidence: list[Study4CEvidenceRefV1] = Field(min_length=1)
    provenance: dict[str, str]
    current_task_evidence_required: Literal[True] = True
    candidate_is_current_task_evidence: Literal[False] = False
    executable_planning_seed_included: Literal[False] = False
    admission_verdict_included: Literal[False] = False
    executable_permission_included: Literal[False] = False
    independent_expert_review_claimed: Literal[False] = False

    @model_validator(mode="after")
    def _reject_hidden_gate_fields(self) -> "Study4CStrongSkillRecordV1":
        payload = self.model_dump(mode="json", exclude={"provenance"})
        leaked = _FORBIDDEN_SKILL_KEYS & set(_iter_keys(payload))
        if leaked:
            raise ValueError(f"strong skill leaked gate fields:{sorted(leaked)}")
        return self


class Study4CCaseV1(BaseModel):
    """一个 fresh source family 对应的完整未来设计任务。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "orion_self_evolution_study4c_case_v1"
    ] = "orion_self_evolution_study4c_case_v1"
    study_phase: Literal["fresh_confirmatory"] = "fresh_confirmatory"
    family_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    endpoint_role: Study4CEndpointRole
    paper_id: str = Field(min_length=1)
    source_dataset: Literal[
        "applicability_v1_reviewed",
        "applicability_v2_holdout_reviewed",
    ]
    source_case_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    candidate_aliases: list[str] = Field(min_length=1)
    expected_applicability_status: str = Field(min_length=1)
    expected_decision: str = Field(min_length=1)
    expected_governed_candidate_available: bool
    prior_full_workflow_exposure: Literal[False] = False
    annotation_identity: Literal[
        "single_author_approved_delegated_not_independent_expert_blind_review"
    ] = "single_author_approved_delegated_not_independent_expert_blind_review"

    def evaluation_sample(self) -> EvaluationSample:
        return EvaluationSample(
            question=self.prompt,
            task_type="design",
            ground_truth=(
                "完成完整电解液设计工作流。历史经验或 Skill 只能改变证据获取的优先级；"
                "最终 recommended/held 必须由本任务检索到的证据和结构化授权链决定。"
            ),
            expected_tools=[],
            tags=[
                "study4c_fresh_confirmatory",
                self.endpoint_role,
                "full_workflow",
            ],
            benchmark_case_id=self.case_id,
            contract_version="study4c-v1",
        )


class Study4CAssignmentV1(BaseModel):
    """单个 family×arm 的冻结前处理合同。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    schema_version: Literal[
        "orion_self_evolution_study4c_assignment_v1"
    ] = "orion_self_evolution_study4c_assignment_v1"
    family_id: str
    case_id: str
    arm: Study4CArm
    endpoint_role: Study4CEndpointRole
    candidate_seal_sha256: str | None = None
    scientific_content_sha256: str | None = None
    strong_skill_record: Study4CStrongSkillRecordV1 | None = None
    planning_seed: FollowupPlanningSeedV2 | None = None
    planning_seed_sha256: str | None = None
    gate_relation_observed_by_runtime: str | None = None
    candidate_available_to_planner: bool
    strong_skill_visible_to_model: bool
    external_activation_applied: bool
    model_self_invocation_is_only_activation_route: bool
    current_task_evidence_required: Literal[True] = True
    production_persistence_allowed: Literal[False] = False


class Study4CPlannedSlotV1(BaseModel):
    """重复仅控制 provider 随机性，不伪装成独立化学样本。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "orion_self_evolution_study4c_planned_slot_v1"
    ] = "orion_self_evolution_study4c_planned_slot_v1"
    slot_id: str
    family_id: str
    case_id: str
    arm: Study4CArm
    repeat_index: Literal[1, 2]
    statistical_unit: Literal["family"] = "family"
    repeat_is_independent_chemical_sample: Literal[False] = False
    assignment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def source_bank() -> dict[str, Any]:
    return json.loads(SOURCE_BANK_PATH.read_text(encoding="utf-8"))


def source_bank_sha256() -> str:
    return _sha256_bytes(SOURCE_BANK_PATH.read_bytes())


def _normalize_v2_case(
    row: dict[str, Any], *, review_ledger: dict[str, Any]
) -> dict[str, Any]:
    """把冻结的 v2 gold 字段投影为 v1 gate 接口；不新增科学判断。"""

    case_id = str(row["case_id"])
    reviewed_ids = set(review_ledger["reviewed_case_ids"])
    if case_id not in reviewed_ids:
        raise RuntimeError(f"v2 source case is absent from reviewed ledger:{case_id}")
    note = str(
        dict(review_ledger.get("case_note_overrides") or {}).get(
            case_id, review_ledger["default_note"]
        )
    )
    by_role: dict[str, list[dict[str, str]]] = {
        "source": [],
        "bridge": [],
        "conflict": [],
    }
    for span in row.get("gold_evidence_spans", []):
        role = str(span["role"])
        if role not in by_role:
            raise RuntimeError(f"unknown v2 evidence role:{case_id}:{role}")
        by_role[role].append(
            {
                "chunk_id": str(span["chunk_id"]),
                "paper_id": str(span["paper_id"]),
                "quote": str(span["quote"]),
                "role": role,
            }
        )
    decision_policy = dict(row["decision_policy"])
    return {
        "schema_version": "2.0-normalized-for-study4c",
        "_source_dataset": "applicability_v2_holdout_reviewed",
        "case_id": case_id,
        "source_evidence_file": str(row["source_evidence_file"]),
        "source_scope": dict(row["gold_source_scope"]),
        "target_scope": dict(row["gold_target_scope"]),
        "relevant_fields": list(row["gold_relevant_fields"]),
        "changed_fields": list(row["gold_changed_fields"]),
        "source_evidence": by_role["source"],
        "bridge_evidence": by_role["bridge"],
        "conflict_evidence": by_role["conflict"],
        "gold_applicability": str(row["gold_applicability"]),
        "allowed_decision": str(decision_policy["preferred_decision"]),
        "human_rationale": " ".join(
            [str(dict(row.get("review") or {}).get("note") or "").strip(), note]
        ).strip(),
        "review": {
            "status": "reviewed",
            "reviewer": str(review_ledger["reviewer"]),
            "reviewed_at": str(review_ledger["reviewed_at"]),
            "note": note,
        },
        "tags": list(row.get("tags") or []),
    }


def reviewed_cases() -> dict[str, dict[str, Any]]:
    v1_rows = [
        json.loads(line)
        for line in REVIEWED_CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in v1_rows:
        row["_source_dataset"] = "applicability_v1_reviewed"
    review_ledger = json.loads(V2_REVIEW_LEDGER_PATH.read_text(encoding="utf-8"))
    v2_rows = [
        _normalize_v2_case(json.loads(line), review_ledger=review_ledger)
        for line in V2_CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [*v1_rows, *v2_rows]
    indexed = {str(row["case_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError("reviewed applicability case_id is not unique")
    return indexed


def _family_payload(family_id: str) -> dict[str, Any]:
    families = dict(source_bank().get("families") or {})
    if family_id not in families:
        raise KeyError(f"unknown Study 4C family:{family_id}")
    return dict(families[family_id])


def study4c_cases() -> list[Study4CCaseV1]:
    cases: list[Study4CCaseV1] = []
    for family_id, raw in dict(source_bank()["families"]).items():
        future = dict(raw["future_case"])
        expected = dict(raw["expected"])
        cases.append(
            Study4CCaseV1(
                family_id=str(family_id),
                case_id=str(future["case_id"]),
                endpoint_role=str(raw["endpoint_role"]),
                paper_id=str(raw["paper_id"]),
                source_dataset=str(raw["source_dataset"]),
                source_case_id=str(raw["source_case_id"]),
                prompt=str(future["prompt"]),
                candidate_aliases=[str(item) for item in future["candidate_aliases"]],
                expected_applicability_status=str(expected["applicability_status"]),
                expected_decision=str(expected["decision"]),
                expected_governed_candidate_available=bool(
                    expected["governed_candidate_available"]
                ),
            )
        )
    return cases


def study4c_case(family_id: str) -> Study4CCaseV1:
    matches = [case for case in study4c_cases() if case.family_id == family_id]
    if len(matches) != 1:
        raise KeyError(f"unknown/non-unique Study 4C family:{family_id}")
    return matches[0]


def _reviewed_source_case(family_id: str) -> dict[str, Any]:
    case = study4c_case(family_id)
    try:
        row = reviewed_cases()[case.source_case_id]
    except KeyError as exc:
        raise RuntimeError(f"missing reviewed source case:{case.source_case_id}") from exc
    refs = [*row.get("source_evidence", []), *row.get("bridge_evidence", []), *row.get("conflict_evidence", [])]
    paper_ids = {str(ref["paper_id"]) for ref in refs}
    if paper_ids != {case.paper_id}:
        raise RuntimeError(f"source paper binding drift:{family_id}:{sorted(paper_ids)}")
    review = dict(row.get("review") or {})
    if review.get("status") != "reviewed" or review.get("reviewer") != "user":
        raise RuntimeError(f"source annotation is not user-reviewed:{family_id}")
    expected_dataset = str(_family_payload(family_id)["source_dataset"])
    if row.get("_source_dataset") != expected_dataset:
        raise RuntimeError(f"source dataset binding drift:{family_id}")
    return row


def _evidence_path(row: dict[str, Any]) -> Path:
    return BENCHMARK_ROOT / str(row["source_evidence_file"])


def _evidence_text_by_chunk(row: dict[str, Any]) -> dict[str, str]:
    bundle = json.loads(_evidence_path(row).read_text(encoding="utf-8"))
    chunks = list(bundle.get("frozen_chunks") or [])
    indexed = {
        str(chunk["chunk_id"]): str(chunk["text"])
        for chunk in chunks
        if chunk.get("chunk_id") and chunk.get("text")
    }
    if len(indexed) != len(chunks):
        raise RuntimeError(f"non-unique or incomplete frozen chunk index:{row['case_id']}")
    return indexed


def _transfer_contract_for_family(
    family_id: str,
    *,
    row: dict[str, Any],
    source_scope: ApplicabilityConditionScope,
    target_scope: ApplicabilityConditionScope,
    relevant_fields: list[ConditionField],
) -> TransferContract | None:
    if row.get("gold_applicability") != "transferable":
        return None
    source_refs = list(row.get("source_evidence") or [])
    bridge_refs = list(row.get("bridge_evidence") or [])
    if len(source_refs) != 1 or len(bridge_refs) != 1:
        raise RuntimeError(f"Study 4C transfer requires one source and one bridge:{family_id}")
    source_ref, bridge_ref = source_refs[0], bridge_refs[0]
    review = dict(row["review"])
    return TransferContract(
        transfer_id=f"STUDY4C_TRANSFER::{family_id}",
        claim_id=str(row["case_id"]),
        source_ref=TransferEvidenceRef(
            chunk_id=str(source_ref["chunk_id"]),
            paper_id=str(source_ref["paper_id"]),
            raw_quote=str(source_ref["quote"]),
        ),
        bridge_ref=TransferEvidenceRef(
            chunk_id=str(bridge_ref["chunk_id"]),
            paper_id=str(bridge_ref["paper_id"]),
            raw_quote=str(bridge_ref["quote"]),
        ),
        source_scope=source_scope,
        target_scope=target_scope,
        relevant_fields=relevant_fields,
        changed_fields=[str(item) for item in row["changed_fields"]],
        supported_dimensions=[str(item) for item in row["changed_fields"]],
        transfer_kind="temperature_range",
        limitations=[
            str(row["human_rationale"]),
            "Only a qualitative evidence-acquisition direction around 45 C is authorized; no unmeasured quantitative value may be produced.",
        ],
        authorized_uses=["report", "compare", "recommend"],
        quantitative_transfer_allowed=False,
        review=TransferReview(
            status=str(review["status"]),
            reviewer=str(review["reviewer"]),
            reviewed_at=str(review["reviewed_at"]),
            note=str(review["note"]),
        ),
    )


def evaluate_study4c_family(family_id: str) -> ApplicabilityRecord:
    """用主 Orion applicability gate 计算 family 的唯一 runtime verdict。"""

    raw = _family_payload(family_id)
    row = _reviewed_source_case(family_id)
    source_scope = ApplicabilityConditionScope.model_validate(row["source_scope"])
    target_scope = ApplicabilityConditionScope.model_validate(
        raw.get("target_scope_override") or row["target_scope"]
    )
    relevant_fields = [
        str(item)
        for item in (raw.get("relevant_fields_override") or row["relevant_fields"])
    ]
    transfer_contract = _transfer_contract_for_family(
        family_id,
        row=row,
        source_scope=source_scope,
        target_scope=target_scope,
        relevant_fields=relevant_fields,
    )
    source_ids = [str(ref["chunk_id"]) for ref in row.get("source_evidence", [])]
    bridge_ids = [str(ref["chunk_id"]) for ref in row.get("bridge_evidence", [])]
    conflict_ids = [str(ref["chunk_id"]) for ref in row.get("conflict_evidence", [])]
    frozen_texts = _evidence_text_by_chunk(row)
    evidence_text_by_chunk = {
        chunk_id: frozen_texts.get(chunk_id, "")
        for chunk_id in {*source_ids, *bridge_ids, *conflict_ids}
    }
    return evaluate_applicability_input(
        ApplicabilityInput(
            assessment_id=f"STUDY4C_ASSESSMENT::{family_id}",
            claim_id=str(row["case_id"]),
            source_scope=source_scope,
            target_scope=target_scope,
            relevant_fields=relevant_fields,
            validated_conflict_evidence_ids=conflict_ids,
            bridge_evidence_ids=bridge_ids,
            transfer_contract=transfer_contract,
        ),
        evidence_text_by_chunk=evidence_text_by_chunk,
        retrieval_allowed=True,
    )


def _scientific_content_payload(family_id: str) -> dict[str, Any]:
    raw = _family_payload(family_id)
    row = _reviewed_source_case(family_id)
    candidate = dict(raw["candidate"])
    evidence: list[dict[str, str]] = []
    for role, key in (
        ("source", "source_evidence"),
        ("bridge", "bridge_evidence"),
        ("conflict", "conflict_evidence"),
    ):
        evidence.extend(
            {
                "chunk_id": str(ref["chunk_id"]),
                "paper_id": str(ref["paper_id"]),
                "raw_quote": str(ref["quote"]),
                "role": role,
            }
            for ref in row.get(key, [])
        )
    return {
        "source_asset_id": str(candidate["source_asset_id"]),
        "version": str(candidate["version"]),
        "guidance": str(candidate["guidance"]),
        "action_moves": [str(item) for item in candidate["action_moves"]],
        "invocation_conditions": [
            str(item) for item in candidate["invocation_conditions"]
        ],
        "contraindications": [str(item) for item in candidate["contraindications"]],
        "limitations": [str(item) for item in candidate["limitations"]],
        "confidence": {
            str(key): str(value)
            for key, value in dict(candidate["confidence"]).items()
        },
        "claimed_source_scope": dict(row["source_scope"]),
        "relevant_fields": [str(item) for item in row["relevant_fields"]],
        "supporting_evidence": evidence,
        "provenance": {
            "paper_id": study4c_case(family_id).paper_id,
            "source_case_id": str(row["case_id"]),
            "source_evidence_file": _relative_orion_path(_evidence_path(row)),
            "source_evidence_sha256": _sha256_bytes(_evidence_path(row).read_bytes()),
            "annotation_identity": "single_author_approved_delegated_not_independent_expert_blind_review",
        },
    }


def compile_strong_skill_record_v1(family_id: str) -> Study4CStrongSkillRecordV1:
    content = _scientific_content_payload(family_id)
    content_sha = _sha256_json(content)
    seal_payload = {
        "schema_version": "orion_self_evolution_study4c_candidate_seal_v1",
        "family_id": family_id,
        "scientific_content": content,
    }
    return Study4CStrongSkillRecordV1(
        candidate_seal_sha256=_sha256_json(seal_payload),
        scientific_content_sha256=content_sha,
        **content,
    )


def render_strong_skill_message_v1(record: Study4CStrongSkillRecordV1) -> str:
    """渲染强 Skill 对照；不加入 applicability verdict 或执行许可。"""

    payload = record.model_dump(mode="json")
    leaked = _FORBIDDEN_SKILL_KEYS & set(_iter_keys(payload))
    if leaked:
        raise RuntimeError(f"strong skill message leaked gate fields:{sorted(leaked)}")
    return (
        "[Study 4C source-bound electrolyte experience skill]\n"
        "This reusable record is not current-task evidence and does not grant permission "
        "to recommend. Decide whether it is relevant, retrieve current-task evidence, "
        "and obey every invocation condition, contraindication, and limitation.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def compile_study4c_planning_seed_v1(family_id: str) -> FollowupPlanningSeedV2:
    """从与 M 完全相同的 scientific content 确定性编译 planning seed。"""

    record = compile_strong_skill_record_v1(family_id)
    scope = record.claimed_source_scope
    anchors: list[FollowupPlanningAnchorV2] = []
    for field in record.relevant_fields:
        value = getattr(scope, field)
        if value is None:
            continue
        anchors.append(
            FollowupPlanningAnchorV2(
                field=field,
                value=_render_scope_value(value),
            )
        )
    if not anchors:
        raise RuntimeError(f"Study 4C planning seed has no source anchor:{family_id}")
    unknown_fields = [
        field for field in CONDITION_FIELDS if getattr(scope, field) is None
    ]
    guidance = " ".join([record.guidance, *record.action_moves])
    return FollowupPlanningSeedV2(
        candidate_seal_sha256=record.candidate_seal_sha256,
        task_types=["design", "evidence_acquisition"],
        guidance=guidance,
        anchors=anchors,
        unknown_fields=unknown_fields,
        limitations=[*record.limitations, *record.contraindications],
    )


def build_study4c_assignment(
    *, family_id: str, arm: Study4CArm
) -> Study4CAssignmentV1:
    case = study4c_case(family_id)
    skill = compile_strong_skill_record_v1(family_id)
    seed = compile_study4c_planning_seed_v1(family_id)
    applicability = evaluate_study4c_family(family_id)

    visible_skill: Study4CStrongSkillRecordV1 | None = None
    visible_seed: FollowupPlanningSeedV2 | None = None
    gate_relation: str | None = None
    available = False
    skill_visible = False
    external_activation = False
    self_invocation_only = False

    if arm == "A_static":
        pass
    elif arm == "B_unconditional":
        visible_skill = skill
        visible_seed = seed
        available = True
        skill_visible = True
        external_activation = True
    elif arm == "M_strong_skill":
        visible_skill = skill
        available = True
        skill_visible = True
        self_invocation_only = True
    elif arm == "G_evidence_governed":
        gate_relation = applicability.assessment.status
        if gate_relation in _GOVERNED_RELEASE_STATUSES:
            visible_skill = skill
            visible_seed = seed
            available = True
            skill_visible = True
            external_activation = True
    else:  # pragma: no cover - Literal 外部防御
        raise ValueError(f"unsupported Study 4C arm:{arm}")

    candidate_identity_visible = arm != "A_static"
    return Study4CAssignmentV1(
        family_id=family_id,
        case_id=case.case_id,
        arm=arm,
        endpoint_role=case.endpoint_role,
        candidate_seal_sha256=(
            skill.candidate_seal_sha256 if candidate_identity_visible else None
        ),
        scientific_content_sha256=(
            skill.scientific_content_sha256 if candidate_identity_visible else None
        ),
        strong_skill_record=visible_skill,
        planning_seed=visible_seed,
        planning_seed_sha256=(
            followup_planning_seed_sha256(visible_seed) if visible_seed else None
        ),
        gate_relation_observed_by_runtime=gate_relation,
        candidate_available_to_planner=available,
        strong_skill_visible_to_model=skill_visible,
        external_activation_applied=external_activation,
        model_self_invocation_is_only_activation_route=self_invocation_only,
    )


def study4c_assignments() -> list[Study4CAssignmentV1]:
    return [
        build_study4c_assignment(family_id=case.family_id, arm=arm)
        for case in study4c_cases()
        for arm in ARMS
    ]


def _assignment_sha256(assignment: Study4CAssignmentV1) -> str:
    return _sha256_json(assignment.model_dump(mode="json"))


def study4c_planned_slots() -> list[Study4CPlannedSlotV1]:
    slots: list[Study4CPlannedSlotV1] = []
    for assignment in study4c_assignments():
        for repeat_index in (1, 2):
            slots.append(
                Study4CPlannedSlotV1(
                    slot_id=(
                        f"{assignment.family_id}::{assignment.arm}::R{repeat_index}"
                    ),
                    family_id=assignment.family_id,
                    case_id=assignment.case_id,
                    arm=assignment.arm,
                    repeat_index=repeat_index,
                    assignment_sha256=_assignment_sha256(assignment),
                )
            )
    return slots


def _validate_bound_hashes(bank: dict[str, Any]) -> None:
    bindings = dict(bank["source_bindings"])
    bound_sources = (
        (
            "reviewed_applicability_v1_cases_path",
            "reviewed_applicability_v1_cases_sha256",
            REVIEWED_CASES_PATH,
        ),
        (
            "reviewed_applicability_v2_cases_path",
            "reviewed_applicability_v2_cases_sha256",
            V2_CASES_PATH,
        ),
        (
            "reviewed_applicability_v2_ledger_path",
            "reviewed_applicability_v2_ledger_sha256",
            V2_REVIEW_LEDGER_PATH,
        ),
    )
    for path_key, hash_key, expected_path in bound_sources:
        bound_path = ORION_ROOT / str(bindings[path_key])
        if bound_path.resolve() != expected_path.resolve():
            raise RuntimeError(f"reviewed source path binding drift:{path_key}")
        if _sha256_bytes(bound_path.read_bytes()) != str(bindings[hash_key]):
            raise RuntimeError(f"reviewed source SHA-256 drift:{path_key}")
    for relative, expected_sha in dict(bindings["evidence_files_sha256"]).items():
        path = ORION_ROOT / str(relative)
        if _sha256_bytes(path.read_bytes()) != str(expected_sha):
            raise RuntimeError(f"source evidence SHA-256 drift:{relative}")


def audit_prior_self_evolution_exposure() -> dict[str, Any]:
    """扫描旧 self-evolution benchmark 文件，确认新论文未被先前实验使用。"""

    selected = set(source_bank()["freshness"]["selected_paper_ids"])
    hits: dict[str, list[str]] = {paper_id: [] for paper_id in selected}
    for path in BENCHMARK_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(BENCHMARK_ROOT).as_posix()
        lowered = relative.casefold()
        if "self_evolution" not in lowered:
            continue
        if "self_evolution_study4c" in lowered:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for paper_id in selected:
            if paper_id in text:
                hits[paper_id].append(relative)
    nonempty = {key: value for key, value in hits.items() if value}
    return {
        "passed": not nonempty,
        "selected_paper_count": len(selected),
        "unexpected_hits": nonempty,
    }


def validate_study4c_design() -> dict[str, Any]:
    """完整零模型 pre-freeze gate；任何漂移都 fail closed。"""

    bank = source_bank()
    if bank.get("schema_version") != "orion_self_evolution_study4c_source_bank_v1":
        raise RuntimeError("Study 4C source bank schema drift")
    if bank.get("status") != "FROZEN_ZERO_MODEL_NO_PROVIDER_EXECUTION":
        raise RuntimeError("Study 4C frozen source-bank status drift")
    firewall = dict(bank["execution_firewall"])
    if any(
        bool(firewall[key])
        for key in (
            "provider_execution_authorized",
            "production_persistence_authorized",
            "formal_result_directory_creation_authorized",
        )
    ):
        raise RuntimeError("Study 4C execution firewall opened before freeze")
    if not all(
        bool(firewall[key])
        for key in (
            "prompt_or_scorer_tuning_on_study4c_families_forbidden",
            "same_slot_rerun_forbidden_after_freeze",
            "selective_failure_replacement_forbidden",
        )
    ):
        raise RuntimeError("Study 4C anti-tuning firewall drift")

    _validate_bound_hashes(bank)
    cases = study4c_cases()
    assignments = study4c_assignments()
    slots = study4c_planned_slots()
    if len(cases) != 8 or len(assignments) != 32 or len(slots) != 64:
        raise RuntimeError("Study 4C requires 8 families, 32 assignments and 64 slots")
    if set(bank["design"]["arms"]) != set(ARMS):
        raise RuntimeError("Study 4C arm set drift")
    if int(bank["design"]["repeats_per_family_arm"]) != 2:
        raise RuntimeError("Study 4C repeat count drift")
    if int(bank["design"]["planned_episode_count"]) != 64:
        raise RuntimeError("Study 4C planned episode count drift")
    role_counts = {
        role: sum(case.endpoint_role == role for case in cases)
        for role in _EXPECTED_ROLE_COUNTS
    }
    if role_counts != _EXPECTED_ROLE_COUNTS:
        raise RuntimeError(f"Study 4C endpoint role drift:{role_counts}")

    paper_ids = [case.paper_id for case in cases]
    if len(set(paper_ids)) != len(paper_ids):
        raise RuntimeError("Study 4C paper reuse across families")
    freshness = dict(bank["freshness"])
    if set(paper_ids) != set(freshness["selected_paper_ids"]):
        raise RuntimeError("Study 4C selected paper binding drift")
    if set(paper_ids) & set(freshness["excluded_study4d_paper_ids"]):
        raise RuntimeError("Study 4C overlaps Study 4D paper identities")
    exposure = audit_prior_self_evolution_exposure()
    if not exposure["passed"]:
        raise RuntimeError(
            f"Study 4C family exposed in prior self-evolution namespace:{exposure['unexpected_hits']}"
        )

    for case in cases:
        prompt_folded = case.prompt.casefold()
        leaked_aliases = [
            alias
            for alias in case.candidate_aliases
            if alias.strip() and alias.strip().casefold() in prompt_folded
        ]
        if leaked_aliases:
            raise RuntimeError(
                f"Study 4C future prompt leaked treatment candidate:{case.family_id}:{leaked_aliases}"
            )
        row = _reviewed_source_case(case.family_id)
        evidence_path = _evidence_path(row)
        if not evidence_path.is_file():
            raise RuntimeError(f"missing source evidence file:{case.family_id}")
        evidence_text_by_chunk = _evidence_text_by_chunk(row)
        for ref in [
            *row.get("source_evidence", []),
            *row.get("bridge_evidence", []),
            *row.get("conflict_evidence", []),
        ]:
            chunk_text = evidence_text_by_chunk.get(str(ref["chunk_id"]), "")
            if chunk_text.count(str(ref["quote"])) != 1:
                raise RuntimeError(
                    f"source quote is not uniquely bound:{case.family_id}:{ref['chunk_id']}"
                )

        applicability = evaluate_study4c_family(case.family_id)
        if applicability.assessment.status != case.expected_applicability_status:
            raise RuntimeError(f"Study 4C applicability drift:{case.family_id}")
        if applicability.decision != case.expected_decision:
            raise RuntimeError(f"Study 4C decision drift:{case.family_id}")
        expected_available = (
            applicability.assessment.status in _GOVERNED_RELEASE_STATUSES
        )
        if expected_available != case.expected_governed_candidate_available:
            raise RuntimeError(f"Study 4C expected governed availability drift:{case.family_id}")

        family_assignments = [
            item for item in assignments if item.family_id == case.family_id
        ]
        if {item.arm for item in family_assignments} != set(ARMS):
            raise RuntimeError(f"Study 4C arm balance drift:{case.family_id}")
        by_arm = {item.arm: item for item in family_assignments}
        baseline = by_arm["A_static"]
        unconditional = by_arm["B_unconditional"]
        memory = by_arm["M_strong_skill"]
        governed = by_arm["G_evidence_governed"]
        if baseline.candidate_available_to_planner or baseline.strong_skill_record:
            raise RuntimeError(f"Study 4C A contamination:{case.family_id}")
        if not unconditional.planning_seed or not unconditional.strong_skill_record:
            raise RuntimeError(f"Study 4C B treatment drift:{case.family_id}")
        if not memory.strong_skill_record or memory.planning_seed is not None:
            raise RuntimeError(f"Study 4C M is not strong-skill-only:{case.family_id}")
        if not memory.model_self_invocation_is_only_activation_route:
            raise RuntimeError(f"Study 4C M self-invocation contract drift:{case.family_id}")
        rendered_skill = render_strong_skill_message_v1(memory.strong_skill_record)
        if "current-task evidence" not in rendered_skill:
            raise RuntimeError(f"Study 4C M evidence disclaimer missing:{case.family_id}")
        if governed.candidate_available_to_planner != expected_available:
            raise RuntimeError(f"Study 4C G availability drift:{case.family_id}")
        if governed.gate_relation_observed_by_runtime != applicability.assessment.status:
            raise RuntimeError(f"Study 4C G gate relation drift:{case.family_id}")
        if expected_available:
            if not governed.strong_skill_record or not governed.planning_seed:
                raise RuntimeError(f"Study 4C G released incomplete treatment:{case.family_id}")
            if (
                governed.strong_skill_record.model_dump(mode="json")
                != memory.strong_skill_record.model_dump(mode="json")
            ):
                raise RuntimeError(f"Study 4C M/G science payload drift:{case.family_id}")
            if (
                unconditional.strong_skill_record.model_dump(mode="json")
                != memory.strong_skill_record.model_dump(mode="json")
            ):
                raise RuntimeError(f"Study 4C B/M science payload drift:{case.family_id}")
            if governed.planning_seed_sha256 != unconditional.planning_seed_sha256:
                raise RuntimeError(f"Study 4C B/G planning compiler drift:{case.family_id}")
        elif governed.strong_skill_record or governed.planning_seed:
            raise RuntimeError(f"Study 4C G failed to withhold candidate:{case.family_id}")
        hashes = {
            unconditional.scientific_content_sha256,
            memory.scientific_content_sha256,
            governed.scientific_content_sha256,
        }
        if len(hashes) != 1 or None in hashes:
            raise RuntimeError(f"Study 4C arm science identity drift:{case.family_id}")

    return {
        "status": "FROZEN_ZERO_MODEL_PASS",
        "source_bank_sha256": source_bank_sha256(),
        "reviewed_v1_cases_sha256": _sha256_bytes(REVIEWED_CASES_PATH.read_bytes()),
        "reviewed_v2_cases_sha256": _sha256_bytes(V2_CASES_PATH.read_bytes()),
        "reviewed_v2_ledger_sha256": _sha256_bytes(
            V2_REVIEW_LEDGER_PATH.read_bytes()
        ),
        "family_count": len(cases),
        "arm_count": len(ARMS),
        "assignment_count": len(assignments),
        "repeat_count": 2,
        "slot_count": len(slots),
        "endpoint_role_counts": role_counts,
        "prior_self_evolution_exposure_audit": exposure,
        "strong_skill_contains_verdict_or_permission": False,
        "future_prompt_candidate_alias_leakage": False,
        "m_g_scientific_content_identity_enforced": True,
        "deterministic_primary_endpoints": True,
        "independent_expert_blind_review": False,
        "provider_execution_authorized": False,
        "production_persistence_authorized": False,
        "kg_write_authorized": False,
    }


__all__ = [
    "ARMS",
    "SOURCE_BANK_PATH",
    "Study4CArm",
    "Study4CAssignmentV1",
    "Study4CCaseV1",
    "Study4CPlannedSlotV1",
    "Study4CStrongSkillRecordV1",
    "audit_prior_self_evolution_exposure",
    "build_study4c_assignment",
    "compile_strong_skill_record_v1",
    "compile_study4c_planning_seed_v1",
    "evaluate_study4c_family",
    "render_strong_skill_message_v1",
    "source_bank",
    "source_bank_sha256",
    "study4c_assignments",
    "study4c_case",
    "study4c_cases",
    "study4c_planned_slots",
    "validate_study4c_design",
]
