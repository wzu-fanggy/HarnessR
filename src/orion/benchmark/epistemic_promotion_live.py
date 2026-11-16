"""Frozen live-execution adapter for the epistemic-promotion pilot.

This module is deliberately inert at import time.  It performs no model call,
network request, lock acquisition, or production asset mutation unless an
explicit function is invoked by the formal CLI/runner.

The adapter closes the execution degrees of freedom that the earlier callback-
only preregistration left unspecified:

* deterministic raw passage resolution from source-hashed local ``output.md``;
* strict proposer prompt rendering and JSON parsing;
* strong Arm-B and Arm-C live admission acquisition;
* deterministic Arm-D construction from the immutable common proposal;
* isolated candidate-skill mounting in a temporary evolution store;
* a reduced, read-only Orion transfer probe with phase-specific paper guards;
* fail-closed runtime/RAG/model fingerprint validation before one-shot locks.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from orion.benchmark.epistemic_promotion_baselines import (
    BaselineAdmissionDecision,
    GenericAuditedAdmissionRecord,
    PairedGenericQualityJudgment,
    ScientificMemoryAdmissionRecord,
    assess_generic_audited_baseline,
    assess_scientific_memory_baseline,
)
from orion.benchmark.epistemic_promotion_candidate_proposal import PersistentUpdateProposal
from orion.evolution.assets.skill_asset import SkillAsset
from orion.evolution.assets.storage import FileSystemAssetStorage
from orion.evolution.scientific_promotion import (
    PromotionEvidenceRef,
    PromotionVerifierResult,
    ScientificPromotionCandidate,
    ScientificPromotionDecision,
    assess_scientific_promotion,
)
from orion.task_evidence.evidence_applicability import ApplicabilityConditionScope


ROOT = Path(__file__).resolve().parent.parent
PILOT_DIR = ROOT / "benchmark" / "epistemic_promotion_pilot_draft_20260815"
LIVE_CONTRACT_PATH = PILOT_DIR / "LIVE_EXECUTION_CONTRACT_DRAFT.json"
PASSAGE_MANIFEST_PATH = PILOT_DIR / "SOURCE_PASSAGE_MANIFEST_DRAFT.json"
ARM_C_TASKS_PATH = PILOT_DIR / "ARM_C_GENERIC_AUDIT_TASKS_DRAFT.json"
ARM_D_RULES_PATH = PILOT_DIR / "ARM_D_ADMISSION_RULES_DRAFT.json"
PROPOSER_PROMPT_PATH = PILOT_DIR / "UPDATE_GENERATION_PROMPT_DRAFT.md"
GENERIC_JUDGE_PROMPT_PATH = PILOT_DIR / "GENERIC_AUDIT_PROMPT_DRAFT.md"
P1_TASKS_PATH = PILOT_DIR / "P1_HIDDEN_EVALUATION_TASKS_DRAFT.json"
P3_TASK_PATH = PILOT_DIR / "P3_HIDDEN_CONTRADICTION_TASK_DRAFT.json"
P4_CONTROL_PATH = PILOT_DIR / "P4_TRANSIENT_FAILURE_CONTROL_DRAFT.json"
BASELINE_SKILL_PATH = ROOT / "skills" / "electrolyte_design" / "SKILL.md"


class LiveExecutionError(RuntimeError):
    """Fail-closed error raised before or during formal live execution."""


class ResolvedPassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    paper_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: str
    phase: str
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeBindingAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks: dict[str, bool]
    observed: dict[str, Any]
    reason_codes: list[str]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )


def _hash_endpoint(value: str | None) -> str:
    normalized = (value or "").strip().rstrip("/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _extract_prompt_section(path: Path, start_heading: str, end_marker: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.find(start_heading)
    if start < 0:
        raise LiveExecutionError(f"prompt heading missing: {start_heading}")
    start += len(start_heading)
    end = text.find(end_marker, start)
    if end < 0:
        raise LiveExecutionError(f"prompt end marker missing: {end_marker}")
    section = text[start:end].strip()
    if not section:
        raise LiveExecutionError(f"empty prompt section: {path}")
    return section


def resolve_source_passages(
    family_id: str,
    *,
    phase: str,
    manifest_path: str | Path = PASSAGE_MANIFEST_PATH,
) -> list[ResolvedPassage]:
    """Resolve canonical raw passages with source-hash and anchor/span checks."""

    manifest = _load_json(manifest_path)
    family = (manifest.get("families") or {}).get(family_id)
    if family is None:
        raise LiveExecutionError(f"unknown passage family: {family_id}")
    key = {
        "development": "development_passages",
        "d_admission": "d_admission_passages",
    }.get(phase)
    if key is None:
        raise ValueError("phase must be 'development' or 'd_admission'")

    resolved: list[ResolvedPassage] = []
    for spec in family.get(key) or []:
        source_path = Path(spec["source_path"])
        if not source_path.exists():
            raise LiveExecutionError(f"source file missing: {source_path}")
        raw_bytes = source_path.read_bytes()
        observed_source_sha = _sha256_bytes(raw_bytes)
        if observed_source_sha != spec["source_sha256"]:
            raise LiveExecutionError(
                f"source hash drift for {spec['passage_id']}: "
                f"expected={spec['source_sha256']} observed={observed_source_sha}"
            )
        # Canonicalize line endings to LF before applying frozen character
        # spans.  The source SHA still binds the original bytes; this text
        # normalization makes span resolution deterministic across Windows and
        # POSIX and matches Python's universal-newline reading semantics used
        # when the preregistered spans were derived.
        raw_text = raw_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        resolver_type = spec["resolver_type"]
        if resolver_type == "unique_anchor_window":
            anchor = str(spec["anchor"])
            count = raw_text.count(anchor)
            if count != 1:
                raise LiveExecutionError(
                    f"anchor occurrence count must be 1 for {spec['passage_id']}: {count}"
                )
            start = raw_text.index(anchor)
            end = min(len(raw_text), start + int(spec["max_chars_after"]))
            passage_text = raw_text[start:end]
        elif resolver_type == "char_span":
            start = int(spec["char_start"])
            end = int(spec["char_end"])
            if start < 0 or end <= start or end > len(raw_text):
                raise LiveExecutionError(f"invalid char span for {spec['passage_id']}")
            passage_text = raw_text[start:end]
        else:
            raise LiveExecutionError(f"unknown passage resolver: {resolver_type}")

        if not passage_text.strip():
            raise LiveExecutionError(f"empty passage: {spec['passage_id']}")
        text_sha = hashlib.sha256(passage_text.encode("utf-8")).hexdigest()
        expected_prefix = spec.get("expected_raw_text_sha256_prefix")
        if expected_prefix and not text_sha.startswith(str(expected_prefix)):
            raise LiveExecutionError(
                f"raw passage hash drift for {spec['passage_id']}: "
                f"expected_prefix={expected_prefix} observed={text_sha}"
            )
        resolved.append(
            ResolvedPassage(
                evidence_id=spec["passage_id"],
                paper_id=spec["paper_id"],
                source_sha256=observed_source_sha,
                source_path=str(source_path),
                phase=spec["phase"],
                text=passage_text,
                text_sha256=text_sha,
            )
        )
    return resolved


def resolve_all_preregistered_passages(
    manifest_path: str | Path = PASSAGE_MANIFEST_PATH,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    manifest = _load_json(manifest_path)
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family_id in (manifest.get("families") or {}):
        out[family_id] = {}
        for phase in ("development", "d_admission"):
            out[family_id][phase] = [
                item.model_dump(mode="json")
                for item in resolve_source_passages(
                    family_id,
                    phase=phase,
                    manifest_path=manifest_path,
                )
            ]
    return out


def _p4_proposer_payload() -> dict[str, Any]:
    control = _load_json(P4_CONTROL_PATH)
    return {
        "development_task": control["development_task"],
        "fault_injection": control["fault_injection"],
        "failed_development_outcome": control["failed_development_outcome"],
        "recovery_replay_not_visible_to_proposer": True,
    }


def proposer_user_payload(
    family_id: str,
    development_passages: list[ResolvedPassage],
) -> dict[str, Any]:
    source_run_id = f"formal_dev::{family_id}"
    allowlist_by_paper: dict[str, dict[str, Any]] = {}
    for item in development_passages:
        entry = allowlist_by_paper.setdefault(
            item.paper_id,
            {
                "paper_id": item.paper_id,
                "source_sha256": item.source_sha256,
                "evidence_ids": [],
            },
        )
        entry["evidence_ids"].append(item.evidence_id)

    if family_id == "P4_B_TRANSIENT_RETRIEVAL_TIMEOUT":
        evidence_bundle: list[dict[str, Any]] = []
        outcome: Any = _p4_proposer_payload()
        validation = "infrastructure_invalid"
    else:
        evidence_bundle = [
            {
                "source_run_id": source_run_id,
                "evidence_id": item.evidence_id,
                "paper_id": item.paper_id,
                "source_sha256": item.source_sha256,
                "raw_text_sha256": item.text_sha256,
                "text": item.text,
            }
            for item in development_passages
        ]
        outcome = (
            "The development material is a source-hash-validated primary-study observation. "
            "Treat the supplied raw passages as the only scientific evidence available in this "
            "proposal episode. A locally successful observation may still be too condition-local "
            "to become persistent behavior; candidate generation must preserve visible scope."
        )
        validation = "validated_primary_source_observation"

    return {
        "development_case_id": family_id,
        "source_run_id": source_run_id,
        "development_allowlist": list(allowlist_by_paper.values()),
        "development_evidence_bundle": evidence_bundle,
        "development_trajectory_or_outcome": outcome,
        "local_validation_status": validation,
    }


def render_common_proposer_messages(
    family_id: str,
    development_passages: list[ResolvedPassage],
) -> tuple[str, str]:
    system = _extract_prompt_section(
        PROPOSER_PROMPT_PATH,
        "## SYSTEM",
        "---\n\n## USER TEMPLATE",
    )
    payload = proposer_user_payload(family_id, development_passages)
    user = (
        "Instantiate the frozen USER TEMPLATE using this canonical development payload. "
        "Return exactly one JSON object conforming to PersistentUpdateProposal and no other text.\n\n"
        + canonical_json(payload, indent=2)
    )
    return system, user


def validate_proposal_evidence_refs(
    proposal: PersistentUpdateProposal,
    *,
    family_id: str,
    development_passages: list[ResolvedPassage],
) -> None:
    if proposal.proposal_status == "no_update":
        return
    allowed_ids = {item.evidence_id for item in development_passages}
    source_run_id = f"formal_dev::{family_id}"
    if not allowed_ids:
        raise LiveExecutionError("scientific candidate cannot be bound to an empty evidence set")
    for ref in proposal.source_evidence_refs:
        if ref.source_run_id != source_run_id:
            raise LiveExecutionError(
                f"proposal source_run_id outside current development episode: {ref.source_run_id}"
            )
        if ref.evidence_id not in allowed_ids:
            raise LiveExecutionError(
                f"proposal evidence ref outside development passage set: {ref.evidence_id}"
            )


def parse_common_proposer_response(
    text: str,
    *,
    family_id: str,
    development_passages: list[ResolvedPassage],
) -> PersistentUpdateProposal:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise LiveExecutionError("common proposer returned non-JSON output; no repair/retry allowed") from exc
    proposal = PersistentUpdateProposal.model_validate(payload)
    validate_proposal_evidence_refs(
        proposal,
        family_id=family_id,
        development_passages=development_passages,
    )
    return proposal


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content or "")


def invoke_common_proposer_once(
    family_id: str,
    development_passages: list[ResolvedPassage],
    *,
    raw_sink: Callable[[str, str], None] | None = None,
) -> tuple[PersistentUpdateProposal, str]:
    """Perform the one prespecified proposer model request. Caller must lock first."""

    from orion.llms import main_model_nonstream

    system, user = render_common_proposer_messages(family_id, development_passages)
    response = main_model_nonstream.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    raw = _message_text(response)
    if raw_sink is not None:
        raw_sink("common_proposer", raw)
    proposal = parse_common_proposer_response(
        raw,
        family_id=family_id,
        development_passages=development_passages,
    )
    return proposal, raw


def render_candidate_skill(
    proposal: PersistentUpdateProposal,
    *,
    contract_path: str | Path = LIVE_CONTRACT_PATH,
    baseline_skill_path: str | Path = BASELINE_SKILL_PATH,
) -> str:
    if proposal.proposal_status != "candidate":
        raise ValueError("only candidate proposals can be mounted")
    contract = _load_json(contract_path)
    baseline_path = Path(baseline_skill_path)
    baseline = baseline_path.read_text(encoding="utf-8")
    expected = contract["candidate_skill_sandbox"]["baseline_skill_sha256"]
    observed = sha256_path(baseline_path)
    if observed != expected:
        raise LiveExecutionError(
            f"baseline skill drift before candidate mount: expected={expected} observed={observed}"
        )
    template = contract["candidate_skill_sandbox"]["exact_append_template"]
    append = template.format(
        proposed_content=proposal.proposed_content.strip(),
        claimed_scope_canonical_json=canonical_json(
            proposal.claimed_scope.model_dump(mode="json"), indent=2
        ),
        limitations_canonical_json=canonical_json(
            proposal.limitations_visible_from_development, indent=2
        ),
    )
    return baseline.rstrip() + append


def mount_candidate_skill(
    store_root: str | Path,
    proposal: PersistentUpdateProposal,
) -> str:
    content = render_candidate_skill(proposal)
    storage = FileSystemAssetStorage(Path(store_root))
    asset = SkillAsset(asset_id="electrolyte_design", content=content)
    storage.save(asset)
    loaded = storage.load("electrolyte_design", "skill", SkillAsset)
    if loaded is None or str(loaded.active_content) != content:
        raise LiveExecutionError("candidate skill sandbox round-trip mismatch")
    return content


@contextlib.contextmanager
def temporary_environment(updates: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_arm_b_decision(
    candidate_sha256: str,
    proposal: PersistentUpdateProposal,
    *,
    family_id: str,
    development_passages: list[ResolvedPassage],
) -> BaselineAdmissionDecision:
    validate_proposal_evidence_refs(
        proposal,
        family_id=family_id,
        development_passages=development_passages,
    )
    record = ScientificMemoryAdmissionRecord(
        candidate_id=candidate_sha256,
        development_trajectory_successful=proposal.proposal_status == "candidate",
        local_outcome_validated=(
            family_id != "P4_B_TRANSIENT_RETRIEVAL_TIMEOUT" and bool(development_passages)
        ),
        provenance_bound=bool(proposal.source_evidence_refs),
        reusable_memory_serialized=True,
        failure_notes_or_guardrails_preserved=True,
    )
    return assess_scientific_memory_baseline(record)


def _fixed_generic_task_prompt(
    *,
    system_contract: dict[str, Any],
    user_task: str,
    generic_task_contract: list[str],
    evidence: list[dict[str, Any]],
    candidate: PersistentUpdateProposal | None,
) -> tuple[str, str]:
    system = (
        "You are executing a fixed generic quality-audit task. Do not make any scientific "
        "promotion/admission decision. Use only the supplied evidence. Return JSON only with "
        "exactly these required keys: task_completion, evidence_refs_used, explicit_boundaries, "
        "next_actions, unsupported_claims. unsupported_claims must be an empty list; if support "
        "is insufficient, state hold/insufficient and request missing evidence instead of inventing facts."
    )
    payload: dict[str, Any] = {
        "USER_TASK": user_task,
        "FIXED_GENERIC_TASK_CONTRACT": generic_task_contract,
        "FIXED_OUTPUT_CONTRACT": system_contract,
        "EVIDENCE": evidence,
    }
    if candidate is not None:
        payload["PERSISTENT_UPDATE_CANDIDATE"] = {
            "proposed_content": candidate.proposed_content,
            "claimed_scope": candidate.claimed_scope.model_dump(mode="json"),
            "limitations_visible_from_development": candidate.limitations_visible_from_development,
            "instruction": "Optional advice, not evidence; do not override the supplied evidence or task contract.",
        }
    return system, canonical_json(payload, indent=2)


def _invoke_deepseek_text_task_once(
    system: str,
    user: str,
    *,
    label: str,
    raw_sink: Callable[[str, str], None] | None = None,
) -> tuple[dict[str, Any], str]:
    from orion.llms import main_model_nonstream

    response = main_model_nonstream.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    raw = _message_text(response)
    if raw_sink is not None:
        raw_sink(label, raw)
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise LiveExecutionError("Arm-C trajectory returned non-JSON output; no repair/retry allowed") from exc
    if not isinstance(parsed, dict):
        raise LiveExecutionError("Arm-C trajectory JSON must be an object")
    return parsed, raw


def verify_generic_fixed_contract(
    payload: dict[str, Any],
    *,
    allowed_evidence_ids: set[str],
    require_hold_semantics: bool = False,
    forbidden_summary_text: str | None = None,
) -> bool:
    required = {
        "task_completion",
        "evidence_refs_used",
        "explicit_boundaries",
        "next_actions",
        "unsupported_claims",
    }
    if set(payload) != required:
        return False
    if not isinstance(payload["task_completion"], str) or not payload["task_completion"].strip():
        return False
    refs = payload["evidence_refs_used"]
    if not isinstance(refs, list) or any(str(item) not in allowed_evidence_ids for item in refs):
        return False
    if not isinstance(payload["explicit_boundaries"], list):
        return False
    if not isinstance(payload["next_actions"], list) or not payload["next_actions"]:
        return False
    if payload["unsupported_claims"] != []:
        return False
    if require_hold_semantics:
        lowered = payload["task_completion"].casefold()
        if "hold" not in lowered and "insufficient" not in lowered:
            return False
    if forbidden_summary_text:
        normalized_candidate = re.sub(r"\s+", " ", forbidden_summary_text.strip()).casefold()
        normalized_summary = re.sub(r"\s+", " ", payload["task_completion"]).casefold()
        if normalized_candidate and normalized_candidate in normalized_summary:
            return False
    return True


def _arm_c_development_evidence(passages: list[ResolvedPassage]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": item.evidence_id,
            "paper_id": item.paper_id,
            "source_sha256": item.source_sha256,
            "text": item.text,
        }
        for item in passages
    ]


def _render_generic_judge_messages(
    *,
    family_id: str,
    user_task: str,
    generic_task_contract: list[str],
    baseline_trajectory: dict[str, Any],
    candidate_trajectory: dict[str, Any],
) -> tuple[str, str]:
    system = _extract_prompt_section(
        GENERIC_JUDGE_PROMPT_PATH,
        "## SYSTEM",
        "---\n\n## USER TEMPLATE",
    )
    payload = {
        "task_id": f"ARM_C_DEV::{family_id}",
        "same_user_task": user_task,
        "same_tool_and_task_contract": generic_task_contract,
        "baseline_trajectory": baseline_trajectory,
        "candidate_trajectory": candidate_trajectory,
        "note": "No hidden scientific source, scientific transfer label, or Orion Arm-D admission record is present.",
    }
    return system, canonical_json(payload, indent=2)


def _invoke_generic_judge_once(
    system: str,
    user: str,
    *,
    raw_sink: Callable[[str, str], None] | None = None,
) -> tuple[PairedGenericQualityJudgment, str]:
    from langchain_openai import ChatOpenAI
    from orion import env_utils  # noqa: F401 - loads project .env

    base_url = os.getenv("RAGAS_BASE_URL", "").rstrip("/")
    api_key = os.getenv("RAGAS_API_KEY", "")
    model = os.getenv("RAGAS_MODEL", "gpt-5.6-terra")
    if not base_url or not api_key:
        raise LiveExecutionError("RAGAS_BASE_URL/RAGAS_API_KEY missing for frozen Arm-C judge")
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        timeout=120,
        max_retries=0,
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    raw = _message_text(response)
    if raw_sink is not None:
        raw_sink("arm_c_generic_judge", raw)
    try:
        payload = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise LiveExecutionError("Arm-C judge returned non-JSON output; no repair/retry allowed") from exc
    return PairedGenericQualityJudgment.model_validate(payload), raw


def run_arm_c_live_admission_once(
    candidate_sha256: str,
    proposal: PersistentUpdateProposal,
    *,
    family_id: str,
    development_passages: list[ResolvedPassage],
    raw_sink: Callable[[str, str], None] | None = None,
) -> tuple[BaselineAdmissionDecision, dict[str, Any]]:
    """Execute the prespecified strong generic audit. Caller must hold C lock."""

    tasks = _load_json(ARM_C_TASKS_PATH)
    family_task = tasks["family_development_tasks"][family_id]
    output_contract = tasks["fixed_output_contract"]
    dev_evidence = _arm_c_development_evidence(development_passages)
    dev_ids = {item["evidence_id"] for item in dev_evidence}

    base_system, base_user = _fixed_generic_task_prompt(
        system_contract=output_contract,
        user_task=family_task["user_task"],
        generic_task_contract=family_task["generic_task_contract"],
        evidence=dev_evidence,
        candidate=None,
    )
    cand_system, cand_user = _fixed_generic_task_prompt(
        system_contract=output_contract,
        user_task=family_task["user_task"],
        generic_task_contract=family_task["generic_task_contract"],
        evidence=dev_evidence,
        candidate=proposal,
    )
    baseline_traj, baseline_raw = _invoke_deepseek_text_task_once(
        base_system,
        base_user,
        label="arm_c_development_baseline",
        raw_sink=raw_sink,
    )
    candidate_traj, candidate_raw = _invoke_deepseek_text_task_once(
        cand_system,
        cand_user,
        label="arm_c_development_candidate",
        raw_sink=raw_sink,
    )
    baseline_contract_ok = verify_generic_fixed_contract(
        baseline_traj,
        allowed_evidence_ids=dev_ids,
    )
    candidate_contract_ok = verify_generic_fixed_contract(
        candidate_traj,
        allowed_evidence_ids=dev_ids,
    )

    holdout = tasks["isolated_generic_holdout"]
    hold_system, hold_user = _fixed_generic_task_prompt(
        system_contract=output_contract,
        user_task=holdout["user_task"],
        generic_task_contract=holdout["generic_task_contract"],
        evidence=holdout["evidence"],
        candidate=proposal,
    )
    holdout_traj, holdout_raw = _invoke_deepseek_text_task_once(
        hold_system,
        hold_user,
        label="arm_c_generic_holdout",
        raw_sink=raw_sink,
    )
    holdout_ok = verify_generic_fixed_contract(
        holdout_traj,
        allowed_evidence_ids={item["evidence_id"] for item in holdout["evidence"]},
        require_hold_semantics=True,
    )

    retention = tasks["capability_retention_task"]
    ret_system, ret_user = _fixed_generic_task_prompt(
        system_contract=output_contract,
        user_task=retention["user_task"],
        generic_task_contract=retention["generic_task_contract"],
        evidence=retention["evidence"],
        candidate=proposal,
    )
    retention_traj, retention_raw = _invoke_deepseek_text_task_once(
        ret_system,
        ret_user,
        label="arm_c_capability_retention",
        raw_sink=raw_sink,
    )
    retention_ok = verify_generic_fixed_contract(
        retention_traj,
        allowed_evidence_ids={item["evidence_id"] for item in retention["evidence"]},
        forbidden_summary_text=proposal.proposed_content,
    )

    judge_system, judge_user = _render_generic_judge_messages(
        family_id=family_id,
        user_task=family_task["user_task"],
        generic_task_contract=family_task["generic_task_contract"],
        baseline_trajectory=baseline_traj,
        candidate_trajectory=candidate_traj,
    )
    judgment, judge_raw = _invoke_generic_judge_once(
        judge_system,
        judge_user,
        raw_sink=raw_sink,
    )

    record = GenericAuditedAdmissionRecord(
        candidate_id=candidate_sha256,
        development_trajectory_successful=baseline_contract_ok and candidate_contract_ok,
        paired_quality_judgment=judgment,
        fixed_contract_verifier_passed=candidate_contract_ok,
        isolated_generic_holdout_passed=holdout_ok,
        capability_retention_passed=retention_ok,
        rollback_regression_check_passed=(
            not judgment.candidate_introduces_generic_task_regression
            and holdout_ok
            and retention_ok
        ),
    )
    decision = assess_generic_audited_baseline(record)
    audit = {
        "record": record.model_dump(mode="json"),
        "baseline_trajectory": baseline_traj,
        "candidate_trajectory": candidate_traj,
        "generic_holdout_trajectory": holdout_traj,
        "capability_retention_trajectory": retention_traj,
        "raw": {
            "baseline_trajectory": baseline_raw,
            "candidate_trajectory": candidate_raw,
            "generic_holdout": holdout_raw,
            "capability_retention": retention_raw,
            "generic_judge": judge_raw,
        },
    }
    return decision, audit


def _proposal_search_text(proposal: PersistentUpdateProposal) -> str:
    return canonical_json(proposal.model_dump(mode="json"), indent=None).casefold()


def _contains_any(haystack: str, terms: list[str]) -> bool:
    folded = str(haystack).casefold()
    return any(str(term).casefold() in folded for term in terms)


def _contains_unnegated_any(haystack: str, terms: list[str], *, prefix_window: int = 64) -> bool:
    folded = str(haystack).casefold()
    negation_cues = (
        "do not",
        "don't",
        "must not",
        "cannot",
        "can't",
        "not ",
        "no evidence",
        "avoid ",
        "without assuming",
        "without claiming",
    )
    for term in terms:
        needle = str(term).casefold()
        start = 0
        while True:
            index = folded.find(needle, start)
            if index < 0:
                break
            prefix = folded[max(0, index - prefix_window):index]
            if not any(cue in prefix for cue in negation_cues):
                return True
            start = index + len(needle)
    return False


def _family_d_checks(
    family_id: str,
    proposal: PersistentUpdateProposal,
    d_passages: list[ResolvedPassage],
) -> tuple[bool, bool, list[str]]:
    all_rules = _load_json(ARM_D_RULES_PATH)
    rules = all_rules["families"][family_id]
    haystack = _proposal_search_text(proposal)
    reasons: list[str] = []
    support_ok = True

    required_task_types = set(all_rules.get("required_claimed_task_types") or [])
    if set(proposal.claimed_scope.task_types) != required_task_types:
        support_ok = False
        reasons.append("claimed_task_types_outside_frozen_design_scope")

    required_fields = set(rules.get("required_relevant_fields") or [])
    declared_fields = {str(item) for item in proposal.relevant_fields}
    missing_relevant = sorted(required_fields - declared_fields)
    if missing_relevant:
        support_ok = False
        reasons.append("required_relevant_fields_missing:" + "|".join(missing_relevant))

    for field, terms in (rules.get("claimed_scope_condition_checks") or {}).items():
        value = getattr(proposal.claimed_scope.conditions, field, None)
        value_text = canonical_json(value) if not isinstance(value, str) else value
        if value is None or not _contains_any(value_text, [str(item) for item in terms]):
            support_ok = False
            reasons.append(
                "claimed_scope_condition_missing_or_outside_envelope:"
                + field
                + ":"
                + "|".join(str(item) for item in terms)
            )
    for check in rules.get("candidate_support_checks") or []:
        terms = [str(item) for item in check.get("terms") or []]
        if check["check"] == "contains_any" and not _contains_any(haystack, terms):
            support_ok = False
            reasons.append("candidate_missing_required_scope_signal:" + "|".join(terms))
        elif check["check"] == "forbid_any" and _contains_unnegated_any(haystack, terms):
            support_ok = False
            reasons.append("candidate_contains_forbidden_broadening_signal:" + "|".join(terms))

    d_by_id = {item.evidence_id: item.text.casefold() for item in d_passages}
    boundary_ok = True
    for check in rules.get("d_boundary_checks") or []:
        passage_id = check["passage_id"]
        text = d_by_id.get(passage_id)
        if text is None:
            boundary_ok = False
            reasons.append(f"d_boundary_passage_missing:{passage_id}")
            continue
        if not _contains_any(text, [str(item) for item in check["must_contain_any"]]):
            boundary_ok = False
            reasons.append(f"d_boundary_expected_signal_missing:{passage_id}")

    object_specific = True
    object_rule = rules.get("object_specificity_check")
    if object_rule and object_rule.get("unsafe_if_missing_all"):
        object_specific = _contains_any(
            haystack,
            [str(item) for item in object_rule.get("safe_if_contains_any") or []],
        )
        if not object_specific:
            reasons.append("generic_hard_carbon_scope_not_object_specific")

    applicability_pass = support_ok and boundary_ok
    contradiction_pass = boundary_ok
    if family_id == "P3_DIGLYME_HC" and not object_specific:
        applicability_pass = False
        contradiction_pass = False
    if family_id == "P2_A_FEC_DOSE_TRADEOFF" and not support_ok:
        contradiction_pass = False
    return applicability_pass, contradiction_pass, reasons


def build_arm_d_candidate(
    candidate_sha256: str,
    proposal: PersistentUpdateProposal,
    *,
    family_id: str,
    development_passages: list[ResolvedPassage],
    d_passages: list[ResolvedPassage],
    isolated_holdout_passed: bool | None = None,
    capability_retention_passed: bool | None = None,
) -> ScientificPromotionCandidate:
    if proposal.proposal_status != "candidate":
        raise ValueError("Arm D candidate requires proposal_status=candidate")
    validate_proposal_evidence_refs(
        proposal,
        family_id=family_id,
        development_passages=development_passages,
    )
    rules = _load_json(ARM_D_RULES_PATH)
    family_rules = rules["families"][family_id]
    source_scope = ApplicabilityConditionScope.model_validate(
        family_rules.get("development_source_scope") or {}
    )
    dev_by_id = {item.evidence_id: item for item in development_passages}
    sources: list[PromotionEvidenceRef] = []
    for ref in proposal.source_evidence_refs:
        if ref.evidence_id not in dev_by_id:
            raise LiveExecutionError(f"D provenance ref missing: {ref.evidence_id}")
        sources.append(
            PromotionEvidenceRef(
                source_run_id=ref.source_run_id,
                evidence_id=ref.evidence_id,
                claim_id=ref.claim_id,
                source_scope=source_scope,
            )
        )

    applicability_pass, contradiction_pass, check_reasons = _family_d_checks(
        family_id,
        proposal,
        d_passages,
    )
    provenance_pass = bool(sources)
    evidence_use_pass = all(
        ref.source_run_id == f"formal_dev::{family_id}" for ref in proposal.source_evidence_refs
    )
    verifier_results = [
        PromotionVerifierResult(
            verifier="provenance",
            status="passed" if provenance_pass else "failed",
            reason_codes=["development_source_hash_and_ref_binding_passed"]
            if provenance_pass
            else ["development_provenance_missing"],
        ),
        PromotionVerifierResult(
            verifier="evidence_use",
            status="passed" if evidence_use_pass else "failed",
            reason_codes=["candidate_uses_development_refs_only"]
            if evidence_use_pass
            else ["candidate_evidence_ref_outside_development_episode"],
        ),
        PromotionVerifierResult(
            verifier="applicability",
            status="passed" if applicability_pass else "failed",
            reason_codes=["predeclared_family_scope_checks_passed"]
            if applicability_pass
            else check_reasons or ["predeclared_family_scope_check_failed"],
        ),
        PromotionVerifierResult(
            verifier="contradiction",
            status="passed" if contradiction_pass else "failed",
            reason_codes=["no_disqualifying_predeclared_contradiction_for_immutable_scope"]
            if contradiction_pass
            else check_reasons or ["predeclared_contradiction_disqualifies_immutable_scope"],
        ),
    ]
    return ScientificPromotionCandidate(
        promotion_id=candidate_sha256,
        update_kind=proposal.update_kind,
        target_asset_id=proposal.target_asset_id,
        parent_version="baseline-skill-sha256:" + sha256_path(BASELINE_SKILL_PATH),
        proposed_content=proposal.proposed_content,
        sources=sources,
        relevant_fields=proposal.relevant_fields,
        claimed_scope=proposal.claimed_scope,
        admitted_scope=proposal.claimed_scope.model_copy(deep=True),
        required_verifiers=list(rules["required_verifiers"]),
        verifier_results=verifier_results,
        isolated_holdout_passed=isolated_holdout_passed,
        capability_retention_passed=capability_retention_passed,
        human_approved=False,
    )


def build_arm_d_precheck_decision(
    candidate_sha256: str,
    proposal: PersistentUpdateProposal,
    *,
    family_id: str,
    development_passages: list[ResolvedPassage],
    d_passages: list[ResolvedPassage],
) -> tuple[ScientificPromotionDecision, ScientificPromotionCandidate]:
    candidate = build_arm_d_candidate(
        candidate_sha256,
        proposal,
        family_id=family_id,
        development_passages=development_passages,
        d_passages=d_passages,
        isolated_holdout_passed=None,
        capability_retention_passed=None,
    )
    return assess_scientific_promotion(candidate), candidate


def build_arm_d_final_decision(
    precheck_candidate: ScientificPromotionCandidate,
    *,
    isolated_holdout_passed: bool | None,
    capability_retention_passed: bool | None,
) -> ScientificPromotionDecision:
    candidate = precheck_candidate.model_copy(
        update={
            "isolated_holdout_passed": isolated_holdout_passed,
            "capability_retention_passed": capability_retention_passed,
            "human_approved": False,
        },
        deep=True,
    )
    return assess_scientific_promotion(candidate)


def _tool_by_name(tools: tuple[Any, ...] | list[Any], name: str) -> Any:
    for item in tools:
        if str(getattr(item, "name", "")) == name:
            return item
    raise LiveExecutionError(f"required Orion tool missing: {name}")


def build_promotion_probe_tools(
    *,
    paper_allowlist: list[str],
) -> tuple[Any, ...]:
    """Build same-name guarded tools for the controlled future-transfer probe."""

    import orion.main as main

    base = tuple(main._BASE_TOOLS)
    original_chunks = _tool_by_name(base, "search_literature_chunks")
    original_multiview = _tool_by_name(base, "search_literature_multiview")
    original_paper = _tool_by_name(base, "search_paper_chunks")
    original_screen = _tool_by_name(base, "screen_formulations")
    original_report = _tool_by_name(base, "synthesize_design_report")
    original_load_skill = _tool_by_name(base, "load_skill")
    allowed = tuple(paper_allowlist)

    def search_literature_chunks(
        query: str,
        top_k: int = 8,
        mode: str = "hybrid",
        sections: list[str] | None = None,
        paper_ids: list[str] | None = None,
        entities: list[str] | None = None,
        exclude_low_value: bool = True,
        expand_hops: int = 0,
        include_parent: bool = True,
    ) -> Any:
        if not allowed:
            return []
        return original_chunks.invoke(
            {
                "query": query,
                "top_k": top_k,
                "mode": mode,
                "sections": sections,
                "paper_ids": list(allowed),
                "entities": entities,
                "exclude_low_value": exclude_low_value,
                "expand_hops": 0,
                "include_parent": include_parent,
            }
        )

    def search_literature_multiview(
        queries: list[str],
        top_k: int = 10,
        per_query_top_k: int = 10,
        weights: list[float] | None = None,
        mode: str = "hybrid",
        sections: list[str] | None = None,
        paper_ids: list[str] | None = None,
        entities: list[str] | None = None,
        max_per_paper: int = 2,
        rrf_k: int = 60,
    ) -> Any:
        if not allowed:
            return []
        return original_multiview.invoke(
            {
                "queries": queries,
                "top_k": top_k,
                "per_query_top_k": per_query_top_k,
                "weights": weights,
                "mode": mode,
                "sections": sections,
                "paper_ids": list(allowed),
                "entities": entities,
                "max_per_paper": max_per_paper,
                "rrf_k": rrf_k,
            }
        )

    def search_paper_chunks(
        paper_id: str,
        query: str,
        top_k: int = 8,
        sections: list[str] | None = None,
    ) -> Any:
        if paper_id not in allowed:
            return {
                "error": "paper_not_in_formal_evaluator_allowlist",
                "paper_id": paper_id,
                "allowed_paper_ids": list(allowed),
                "results": [],
            }
        return original_paper.invoke(
            {
                "paper_id": paper_id,
                "query": query,
                "top_k": top_k,
                "sections": sections,
            }
        )

    def query_formulation_space(request: Any) -> Any:
        # This probe intentionally records model-generated candidate preferences
        # in the tool arguments but never opens full-corpus/KG fallback.
        return {
            "candidates": [],
            "source": "formal_transfer_probe_no_global_kg",
            "request_received": request,
            "decision": "hold_for_paper_scoped_evidence",
        }

    def screen_formulations(request: Any) -> Any:
        # The scientific-transfer probes are behavioral tests, not a production
        # optimization benchmark.  Preserve the production tool name/schema so
        # model planning remains comparable, but never let this stage open a
        # second data path or manufacture task evidence.
        return {
            "status": "hold",
            "source": "formal_transfer_probe_deterministic_screen",
            "request_received": request,
            "reason": "paper_scoped_evidence_must_be_checked_explicitly_before_recommendation",
        }

    def synthesize_design_report(request: Any, state: dict[str, Any]) -> Any:
        return {
            "decision": "hold",
            "source": "formal_transfer_probe_deterministic_report",
            "request_received": request,
            "state_received": state,
            "reason": "formal_probe_does_not_authorize_a_production_recommendation",
        }

    guarded_chunks = StructuredTool.from_function(
        func=search_literature_chunks,
        name="search_literature_chunks",
        description=original_chunks.description,
        args_schema=original_chunks.args_schema,
    )
    guarded_multiview = StructuredTool.from_function(
        func=search_literature_multiview,
        name="search_literature_multiview",
        description=original_multiview.description,
        args_schema=original_multiview.args_schema,
    )
    guarded_paper = StructuredTool.from_function(
        func=search_paper_chunks,
        name="search_paper_chunks",
        description=original_paper.description,
        args_schema=original_paper.args_schema,
    )
    guarded_query = StructuredTool.from_function(
        func=query_formulation_space,
        name="query_formulation_space",
        description=_tool_by_name(base, "query_formulation_space").description,
        args_schema=_tool_by_name(base, "query_formulation_space").args_schema,
    )
    guarded_screen = StructuredTool.from_function(
        func=screen_formulations,
        name="screen_formulations",
        description=original_screen.description,
        args_schema=original_screen.args_schema,
    )
    guarded_report = StructuredTool.from_function(
        func=synthesize_design_report,
        name="synthesize_design_report",
        description=original_report.description,
        args_schema=original_report.args_schema,
    )
    return (
        original_load_skill,
        guarded_chunks,
        guarded_multiview,
        guarded_paper,
        guarded_query,
        guarded_screen,
        guarded_report,
    )


def promotion_probe_agent_factory(*, paper_allowlist: list[str]):
    """Return a factory that builds Orion with only the frozen probe tool profile."""

    def factory():
        import orion.main as main

        selected = build_promotion_probe_tools(paper_allowlist=paper_allowlist)
        original = main._BASE_TOOLS
        main._BASE_TOOLS = selected
        try:
            return main.build_streaming_agent()
        finally:
            main._BASE_TOOLS = original

    return factory


def run_future_transfer_probe_once(
    *,
    task_id: str,
    task_type: str,
    prompt: str,
    paper_allowlist: list[str],
    proposal: PersistentUpdateProposal | None,
) -> dict[str, Any]:
    """Run one controlled Orion future-task episode. Caller must hold arm lock."""

    from types import SimpleNamespace

    from orion.evaluation.agent_runner import run_agent
    from orion.evaluation.models import EvaluationSample

    sample = EvaluationSample(
        question=prompt,
        task_type=task_type,
        ground_truth="",
        expected_tools=[],
    )
    env = {
        "ORION_GROUNDING_CHECK": "0",
        "ORION_LLM_CACHE": "0",
        "ORION_ABLATION_NO_KG": "0",
        "ORION_ABLATION_NO_FORMULATION": "0",
        "ORION_EVAL_AGENT_TIMEOUT": "900",
        "ORION_EVAL_LLM_STEP_TIMEOUT": "90",
        "ORION_EVAL_LLM_REQUEST_MAX_RETRIES": "0",
        "ORION_LLM_REQUEST_MAX_RETRIES": "0",
        "ORION_LLM_STREAM": "0",
    }
    # Import the production agent while cwd is still the stable repository root.
    # ``main.SYSTEM_PROMPT`` captures cwd at import time; importing it only after
    # changing into a random temporary directory would add arm-specific prompt
    # noise unrelated to the scientific treatment.
    import orion.main as main  # noqa: F401

    with tempfile.TemporaryDirectory(prefix=f"orion-promotion-{task_id}-") as tmp:
        tmp_path = Path(tmp)
        if proposal is not None:
            store = tmp_path / "evolution_store"
            mount_candidate_skill(store, proposal)
            env["ORION_EVOLUTION_STORE"] = str(store)
        else:
            env["ORION_EVOLUTION_STORE"] = str(tmp_path / "empty_evolution_store")

        original_cwd = Path.cwd()
        with temporary_environment(env):
            try:
                os.chdir(tmp_path)
                raw = run_agent(
                    sample,
                    config=SimpleNamespace(auto_approve=True),
                    agent_factory=promotion_probe_agent_factory(
                        paper_allowlist=paper_allowlist
                    ),
                )
            finally:
                os.chdir(original_cwd)

    tool_calls = []
    for call in raw.get("tool_calls") or []:
        if hasattr(call, "model_dump"):
            tool_calls.append(call.model_dump(mode="json"))
        elif isinstance(call, dict):
            tool_calls.append(call)
        else:
            tool_calls.append(
                {
                    "name": str(getattr(call, "name", "")),
                    "args": getattr(call, "args", {}),
                    "result": getattr(call, "result", None),
                }
            )
    return {
        "task_id": task_id,
        "sample": sample.model_dump(mode="json"),
        "answer": raw.get("answer", ""),
        # Formal P1/P3 scorers treat ``contexts`` as evidence present before
        # the first agent action.  ``run_agent`` returns contexts extracted
        # from the entire completed trajectory, so those later retrievals
        # must not be mislabeled as initial context.  The controlled probe
        # starts with no external context; temporality is reconstructed from
        # ordered tool-call results below.
        "contexts": [],
        "retrieved_contexts": raw.get("contexts", []),
        "tool_calls": tool_calls,
        "failure_reasons": raw.get("failure_reasons", []),
        "ok": raw.get("ok", True),
        "task_artifact": raw.get("task_artifact"),
    }


def validate_live_runtime_binding(
    *,
    contract_path: str | Path = LIVE_CONTRACT_PATH,
    verify_interpreter: bool = True,
    verify_rag: bool = True,
    verify_model_endpoints: bool = True,
) -> RuntimeBindingAudit:
    """Perform read-only fail-closed checks before any formal one-shot lock."""

    contract = _load_json(contract_path)
    checks: dict[str, bool] = {}
    observed: dict[str, Any] = {}
    reasons: list[str] = []

    expected_skill = contract["candidate_skill_sandbox"]["baseline_skill_sha256"]
    observed_skill = sha256_path(BASELINE_SKILL_PATH)
    checks["baseline_skill_hash"] = observed_skill == expected_skill
    observed["baseline_skill_sha256"] = observed_skill

    if verify_interpreter:
        import importlib.metadata as importlib_metadata
        import platform

        expected_suffix = contract["interpreter"]["project_relative_path"].replace("/", "\\").casefold()
        observed_exe = str(Path(sys.executable).resolve())
        checks["project_interpreter"] = observed_exe.casefold().endswith(expected_suffix)
        observed["sys_executable"] = observed_exe

        env_fingerprint = _load_json(
            ROOT / contract["runtime_environment_fingerprint_path"]
        )
        expected_python = env_fingerprint["python"]
        checks["python_implementation"] = (
            platform.python_implementation() == expected_python["implementation"]
        )
        checks["python_version"] = sys.version == expected_python["version"]
        observed["python_implementation"] = platform.python_implementation()
        observed["python_version"] = sys.version
        package_versions: dict[str, str | None] = {}
        package_versions_match = True
        for package_name, expected_version in env_fingerprint["packages"].items():
            try:
                actual_version = importlib_metadata.version(package_name)
            except importlib_metadata.PackageNotFoundError:
                actual_version = None
            package_versions[package_name] = actual_version
            if actual_version != expected_version:
                package_versions_match = False
        checks["runtime_package_versions"] = package_versions_match
        observed["runtime_package_versions"] = package_versions

    resolved_contract = _load_json(
        ROOT / contract["model_bindings"]["resolved_contract_path"]
    )
    if verify_model_endpoints:
        from orion import env_utils  # noqa: F401 - loads .env

        proposer = resolved_contract["proposer_and_orion_task_agent"]
        judge = resolved_contract["generic_arm_c_quality_judge"]
        observed_prop = _hash_endpoint(os.getenv(proposer["endpoint_source_env"]))
        observed_judge = _hash_endpoint(os.getenv(judge["endpoint_source_env"]))
        checks["proposer_endpoint_fingerprint"] = (
            observed_prop == proposer["endpoint_fingerprint_sha256"]
        )
        checks["judge_endpoint_fingerprint"] = (
            observed_judge == judge["endpoint_fingerprint_sha256"]
        )
        checks["judge_model"] = os.getenv("RAGAS_MODEL", "gpt-5.6-terra") == judge["model"]
        observed["proposer_endpoint_fingerprint_sha256"] = observed_prop
        observed["judge_endpoint_fingerprint_sha256"] = observed_judge
        observed["judge_model"] = os.getenv("RAGAS_MODEL", "gpt-5.6-terra")

    if verify_rag:
        rag = contract["runtime_rag_binding"]
        session = requests.Session()
        session.trust_env = False
        health = session.get(rag["health_url"], timeout=5)
        checks["rag_health"] = health.status_code == 200
        meta_response = session.get(rag["metadata_url"], timeout=10)
        checks["rag_metadata_http"] = meta_response.status_code == 200
        meta = meta_response.json() if meta_response.status_code == 200 else {}
        observed["rag_metadata"] = meta
        checks["rag_database"] = meta.get("database") == rag["database"]
        checks["rag_snapshot_fingerprint"] = (
            meta.get("snapshot_fingerprint") == rag["snapshot_fingerprint"]
        )
        checks["rag_runtime_code_fingerprint"] = (
            (meta.get("runtime_code") or {}).get("fingerprint")
            == rag["runtime_code_fingerprint"]
        )
        expected_index = rag["vector_index"]
        item = meta.get("vector_index") or {}
        index_ok = (
            isinstance(item, dict)
            and item.get("name") == expected_index["name"]
            and item.get("state") == expected_index["state"]
            and item.get("dimensions") == expected_index["dimensions"]
            and str(item.get("similarity_function") or "").upper()
            == expected_index["similarity_function"]
            and item.get("entity_type") == expected_index["entity_type"]
            and item.get("labels_or_types") == expected_index["labels_or_types"]
            and item.get("properties") == expected_index["properties"]
        )
        checks["rag_vector_index"] = index_ok

    for name, passed in checks.items():
        if not passed:
            reasons.append(name + "_mismatch")
    return RuntimeBindingAudit(
        passed=not reasons,
        checks=checks,
        observed=observed,
        reason_codes=reasons or ["live_runtime_binding_passed"],
    )
