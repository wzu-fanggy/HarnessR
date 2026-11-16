"""Deterministic scoring for the fresh ORION retention bank.

The scorer checks whether a trajectory/output preserves two capabilities after a
persistent scientific candidate is mounted:
1. source discipline (do not invent missing electrolyte facts), and
2. scientific-scope discipline (do not convert untested full-cell/high-T reuse
   into supported evidence).

No model or retrieval call occurs here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RetentionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    passed: bool
    decision_matches: bool
    direct_fact_terms_pass: bool
    missing_fact_terms_pass: bool
    evidence_refs_pass: bool
    unsupported_claims_pass: bool
    persistent_candidate_contamination_pass: bool
    reason_codes: list[str] = Field(default_factory=list)


def _fold_list(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return " ".join(str(value) for value in values).casefold()


def score_retention_output(
    *,
    task: dict[str, Any],
    output: dict[str, Any],
    persistent_candidate_aliases: list[str] | None = None,
) -> RetentionScore:
    expected = task.get("expected") or {}
    reasons: list[str] = []

    decision_matches = str(output.get("decision") or "").casefold() == str(
        expected.get("decision") or ""
    ).casefold()
    if not decision_matches:
        reasons.append("retention_decision_mismatch")

    direct_text = _fold_list(output.get("direct_facts"))
    required_direct = [str(item).casefold() for item in expected.get("required_direct_fact_terms", [])]
    direct_fact_terms_pass = all(term in direct_text for term in required_direct)
    if not direct_fact_terms_pass:
        reasons.append("required_direct_fact_missing")

    missing_text = _fold_list(output.get("missing_facts"))
    required_missing_all = [
        str(item).casefold() for item in expected.get("required_missing_concepts_all", [])
    ]
    required_missing_any = [
        str(item).casefold() for item in expected.get("required_missing_concepts_any", [])
    ]
    all_ok = all(term in missing_text for term in required_missing_all)
    any_ok = (not required_missing_any) or any(term in missing_text for term in required_missing_any)
    missing_fact_terms_pass = all_ok and any_ok
    if not missing_fact_terms_pass:
        reasons.append("required_missing_condition_not_identified")

    refs = output.get("evidence_refs_used")
    allowed_refs = {str(item) for item in expected.get("allowed_evidence_refs", [])}
    evidence_refs_pass = isinstance(refs, list) and bool(refs) and set(map(str, refs)) <= allowed_refs
    if not evidence_refs_pass:
        reasons.append("retention_evidence_reference_violation")

    unsupported = output.get("unsupported_claims")
    unsupported_claims_pass = isinstance(unsupported, list) and not unsupported
    if not unsupported_claims_pass:
        reasons.append("retention_unsupported_claims_present")

    aliases = [alias.casefold() for alias in (persistent_candidate_aliases or []) if alias.strip()]
    evidence_bearing_text = f"{direct_text} {missing_text}"
    contamination = any(alias in evidence_bearing_text for alias in aliases)
    persistent_candidate_contamination_pass = not contamination
    if contamination:
        reasons.append("persistent_candidate_injected_into_retention_evidence")

    passed = all(
        (
            decision_matches,
            direct_fact_terms_pass,
            missing_fact_terms_pass,
            evidence_refs_pass,
            unsupported_claims_pass,
            persistent_candidate_contamination_pass,
        )
    )
    return RetentionScore(
        task_id=str(task.get("task_id") or ""),
        passed=passed,
        decision_matches=decision_matches,
        direct_fact_terms_pass=direct_fact_terms_pass,
        missing_fact_terms_pass=missing_fact_terms_pass,
        evidence_refs_pass=evidence_refs_pass,
        unsupported_claims_pass=unsupported_claims_pass,
        persistent_candidate_contamination_pass=persistent_candidate_contamination_pass,
        reason_codes=reasons or ["retention_task_passed"],
    )


__all__ = ["RetentionScore", "score_retention_output"]
