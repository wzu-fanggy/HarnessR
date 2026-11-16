"""Deterministic preflight for fresh ORION self-evolution study families.

This module is development-only. It performs no model calls, writes no persistent
scientific asset, and does not authorize formal execution. Its purpose is to
reject scientifically or causally non-discriminative family designs before they
consume semantic-model budget.

The checks encode lessons from the historical epistemic-promotion runs:
- a positive family cannot demonstrate adaptation if Static ORION is already at
  the behavioral ceiling;
- evaluator prompts/initial context cannot reveal the candidate before the
  scored action;
- a safe answer must be representable without post-output schema repair;
- development/evaluator source roles must remain separated;
- at least one predeclared arm contrast must be behaviorally discriminative;
- candidate/scorer contracts must be sealable before semantic output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


SCHEMA_VERSION = "orion_self_evolution_fresh_family_v1"
VALID_FAMILY_CLASSES = frozenset({"F1", "F2", "F3", "F4", "F5"})
VALID_CONTRASTS = frozenset({"A_vs_D", "B_vs_D", "C_vs_D", "D_vs_R"})


@dataclass(frozen=True)
class PreflightCheck:
    check_id: str
    passed: bool
    reason_code: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


def _as_str_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.strip()} if value.strip() else set()
    if not isinstance(value, Iterable) or isinstance(value, (bytes, bytearray, dict)):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _rate(value: Any, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _source_collisions(source_roles: dict[str, Any]) -> dict[str, list[str]]:
    """Return forbidden exact source-id collisions across hidden roles.

    Admission evidence is allowed to be shared across B/C/D because that is the
    matched-evidence fairness requirement. The collision check is across
    scientific *roles*: development, future positive, future control, retention.
    """

    role_sets = {
        role: _as_str_set(source_roles.get(role))
        for role in (
            "development",
            "future_positive",
            "future_control",
            "retention",
        )
    }
    collisions: dict[str, list[str]] = {}
    roles = list(role_sets)
    for i, left in enumerate(roles):
        for right in roles[i + 1 :]:
            overlap = sorted(role_sets[left] & role_sets[right])
            if overlap:
                collisions[f"{left}__{right}"] = overlap
    return collisions


def evaluate_family_preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one proposed fresh-study family manifest.

    The function is intentionally fail-closed: missing declarations that are
    required to establish causal interpretability are treated as failures.
    """

    checks: list[PreflightCheck] = []

    schema_ok = manifest.get("schema_version") == SCHEMA_VERSION
    checks.append(
        PreflightCheck(
            "schema_version",
            schema_ok,
            "ok" if schema_ok else "invalid_schema_version",
            f"expected {SCHEMA_VERSION}",
        )
    )

    family_id = str(manifest.get("family_id") or "").strip()
    family_id_ok = bool(family_id)
    checks.append(
        PreflightCheck(
            "family_id",
            family_id_ok,
            "ok" if family_id_ok else "missing_family_id",
            family_id or "family_id is empty",
        )
    )

    family_class = str(manifest.get("family_class") or "").strip()
    family_class_ok = family_class in VALID_FAMILY_CLASSES
    checks.append(
        PreflightCheck(
            "family_class",
            family_class_ok,
            "ok" if family_class_ok else "invalid_family_class",
            family_class or "family_class is empty",
        )
    )

    historical = manifest.get("historical_exposure") or {}
    burned = bool(historical.get("exact_behavioral_comparison_previously_observed"))
    main_result_eligible = bool(manifest.get("fresh_main_result_eligible"))
    freshness_ok = main_result_eligible and not burned
    checks.append(
        PreflightCheck(
            "freshness",
            freshness_ok,
            "ok" if freshness_ok else "historical_or_nonfresh_main_result",
            "fresh main-result family required; exact previously observed behavioral comparison is ineligible",
        )
    )

    positive = manifest.get("positive_endpoint") or {}
    requires_positive = family_class in {"F1", "F3"}
    static_rate = _rate(positive.get("static_rate"))
    ceiling = _rate(positive.get("ceiling"), default=1.0)
    min_headroom = _rate(positive.get("min_required_headroom"), default=0.1)

    if requires_positive:
        rate_declared = static_rate is not None and ceiling is not None and min_headroom is not None
        valid_bounds = bool(
            rate_declared
            and 0.0 <= static_rate <= ceiling
            and 0.0 <= min_headroom <= ceiling
        )
        headroom = (ceiling - static_rate) if valid_bounds else None
        headroom_ok = bool(valid_bounds and headroom is not None and headroom >= min_headroom)
        checks.append(
            PreflightCheck(
                "static_headroom",
                headroom_ok,
                "ok" if headroom_ok else "insufficient_or_undeclared_static_headroom",
                (
                    f"static={static_rate}, ceiling={ceiling}, headroom={headroom}, required={min_headroom}"
                    if rate_declared
                    else "positive family must declare static_rate, ceiling, and min_required_headroom"
                ),
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "static_headroom",
                True,
                "not_required_for_family_class",
                f"{family_class} is not a primary positive-transfer family",
            )
        )

    behavior_observable = bool(positive.get("behavior_observable_before_current_task_evidence"))
    if requires_positive:
        checks.append(
            PreflightCheck(
                "behavioral_observability",
                behavior_observable,
                "ok" if behavior_observable else "positive_behavior_not_observable_pre_evidence",
                "primary adaptation must change a scored action before current-task evidence reveals the target",
            )
        )
    else:
        negative = manifest.get("negative_endpoint") or {}
        negative_observable = bool(
            negative.get("behavior_observable_before_current_task_evidence")
            or family_class == "F5"
        )
        checks.append(
            PreflightCheck(
                "behavioral_observability",
                negative_observable,
                "ok" if negative_observable else "negative_behavior_not_observable_pre_evidence",
                "negative-transfer families need a pre-evidence scored action; F5 scores update creation itself",
            )
        )

    leakage = manifest.get("leakage") or {}
    prompt_leak = bool(leakage.get("candidate_alias_in_future_prompt"))
    context_leak = bool(leakage.get("candidate_alias_in_initial_context"))
    leakage_ok = not prompt_leak and not context_leak
    checks.append(
        PreflightCheck(
            "evaluator_leakage",
            leakage_ok,
            "ok" if leakage_ok else "candidate_leaked_before_scored_action",
            f"prompt_leak={prompt_leak}, initial_context_leak={context_leak}",
        )
    )

    schema = manifest.get("scientific_schema") or {}
    expressible = bool(schema.get("safe_answer_and_scope_expressible"))
    checks.append(
        PreflightCheck(
            "scientific_schema",
            expressible,
            "ok" if expressible else "safe_scientific_answer_not_expressible",
            "fresh sidecars are allowed only when explicitly versioned before semantic output",
        )
    )

    source_roles = manifest.get("source_roles") or {}
    collisions = _source_collisions(source_roles) if isinstance(source_roles, dict) else {"invalid": []}
    source_separation_ok = not collisions
    checks.append(
        PreflightCheck(
            "source_role_separation",
            source_separation_ok,
            "ok" if source_separation_ok else "source_role_collision",
            f"collisions={collisions}",
        )
    )

    contrasts = _as_str_set(manifest.get("arm_discriminability"))
    unknown_contrasts = sorted(contrasts - VALID_CONTRASTS)
    discriminability_ok = bool(contrasts & VALID_CONTRASTS) and not unknown_contrasts
    checks.append(
        PreflightCheck(
            "arm_discriminability",
            discriminability_ok,
            "ok" if discriminability_ok else "no_valid_arm_discriminability",
            f"contrasts={sorted(contrasts)}, unknown={unknown_contrasts}",
        )
    )

    reject_all = manifest.get("reject_all") or {}
    if requires_positive:
        reject_all_ok = bool(reject_all.get("strict_utility_improvement_possible"))
        checks.append(
            PreflightCheck(
                "reject_all_utility",
                reject_all_ok,
                "ok" if reject_all_ok else "reject_all_strict_utility_impossible",
                "positive family must permit D to beat reject-all on utility",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "reject_all_utility",
                True,
                "not_required_for_family_class",
                f"{family_class} is not a primary positive-transfer family",
            )
        )

    lifecycle = manifest.get("lifecycle") or {}
    candidate_sealable = bool(lifecycle.get("candidate_sealable_before_boundary_and_evaluator"))
    scorer_prefrozen = bool(lifecycle.get("scorer_frozen_before_semantic_output"))
    task_prefrozen = bool(lifecycle.get("future_task_spec_prefrozen"))
    lifecycle_ok = candidate_sealable and scorer_prefrozen and task_prefrozen
    checks.append(
        PreflightCheck(
            "lifecycle_prefreeze",
            lifecycle_ok,
            "ok" if lifecycle_ok else "candidate_or_scorer_not_prefrozen",
            f"candidate_sealable={candidate_sealable}, scorer_prefrozen={scorer_prefrozen}, future_task_prefrozen={task_prefrozen}",
        )
    )

    matched = manifest.get("matched_evidence") or {}
    requires_c_vs_d = "C_vs_D" in contrasts
    if requires_c_vs_d:
        matched_ok = bool(matched.get("same_boundary_evidence_for_B_C_D"))
        checks.append(
            PreflightCheck(
                "matched_evidence_fairness",
                matched_ok,
                "ok" if matched_ok else "c_vs_d_boundary_evidence_not_matched",
                "C-vs-D scientific-scope claim requires B/C/D access to the same boundary evidence",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "matched_evidence_fairness",
                True,
                "not_required_without_C_vs_D_claim",
                "no C_vs_D contrast declared",
            )
        )

    passed = all(check.passed for check in checks)
    return {
        "schema_version": "orion_self_evolution_fresh_preflight_report_v1",
        "family_id": family_id,
        "family_class": family_class,
        "passed": passed,
        "eligible_for_prefreeze_family_shortlist": passed,
        "checks": [check.to_dict() for check in checks],
        "failure_reason_codes": [
            check.reason_code for check in checks if not check.passed
        ],
        "development_only_not_formal_authorization": True,
    }
