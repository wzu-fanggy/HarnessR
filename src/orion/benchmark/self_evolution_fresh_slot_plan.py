"""Mechanical slot-plan validation for the fresh self-evolution study.

No model calls occur here. The validator recomputes confirmatory future-evaluator
slot counts from frozen task bundles and the slot/statistics contract so the
formal snapshot cannot silently change denominators after results are observed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orion.benchmark.self_evolution_fresh_behavior import FreshBehaviorTaskSpec
from orion.benchmark.self_evolution_fresh_productive import FreshProductiveActionContract


class FreshSlotPlanAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "orion_self_evolution_fresh_slot_plan_audit_v1"
    passed: bool
    primary_future_evaluator_slots: int
    secondary_ablation_slots: int
    total_future_evaluator_slots: int
    family_task_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_slot_statistics_contract(
    contract: dict[str, Any],
    *,
    root: str | Path = ".",
) -> FreshSlotPlanAudit:
    root_path = Path(root)
    reasons: list[str] = []

    replicate_count = int(contract["replicate_policy"]["replicates_per_arm_task_variant"])
    if replicate_count <= 0:
        reasons.append("replicate_count_must_be_positive")

    positive_arms = contract["arm_sets"]["positive_transfer"]
    control_arms = contract["arm_sets"]["out_of_scope_control"]
    if len(positive_arms) != len(set(positive_arms)):
        reasons.append("duplicate_positive_arm")
    if len(control_arms) != len(set(control_arms)):
        reasons.append("duplicate_control_arm")

    primary_slots = 0
    family_counts: dict[str, dict[str, int]] = {}
    seen_task_ids: set[str] = set()

    for bundle in contract["task_bundles"]:
        task_payload = _load_json(root_path / bundle["path"])
        productive_payload = _load_json(root_path / bundle["productive_contract_path"])
        tasks = [FreshBehaviorTaskSpec.model_validate(item) for item in task_payload["tasks"]]
        productive = [
            FreshProductiveActionContract.model_validate(item)
            for item in productive_payload["contracts"]
        ]

        task_ids = {task.task_id for task in tasks}
        productive_ids = {item.task_id for item in productive}
        if task_ids != productive_ids:
            reasons.append(f"productive_contract_task_mismatch:{bundle['family_id']}")
        duplicate_across = seen_task_ids & task_ids
        if duplicate_across:
            reasons.append(f"duplicate_task_ids_across_bundles:{bundle['family_id']}")
        seen_task_ids |= task_ids

        positive_count = sum(task.role == "positive_transfer" for task in tasks)
        control_count = sum(task.role == "out_of_scope_control" for task in tasks)
        if positive_count != bundle["expected_positive_variants"]:
            reasons.append(f"unexpected_positive_variant_count:{bundle['family_id']}")
        if control_count != bundle["expected_control_variants"]:
            reasons.append(f"unexpected_control_variant_count:{bundle['family_id']}")

        family_slots = replicate_count * (
            positive_count * len(positive_arms) + control_count * len(control_arms)
        )
        primary_slots += family_slots
        family_counts[bundle["family_id"]] = {
            "positive_variants": positive_count,
            "control_variants": control_count,
            "primary_slots": family_slots,
        }

    secondary_slots = 0
    secondary = contract["arm_sets"].get("secondary_ablations", {})
    bundle_by_short_id = {
        "A1": family_counts.get("FRESH_A1_NADFP_WIDE_TEMP"),
        "A4": family_counts.get("FRESH_A4_DTD_HIGH_TEMP"),
        "A5": family_counts.get("FRESH_A5_ETHER_HIGH_VOLTAGE_BOUNDARY"),
        "A6": family_counts.get("FRESH_A6_GLYME_LOW_VOLTAGE_PRESERVATION"),
    }
    for short_id, ablation_arms in secondary.items():
        counts = bundle_by_short_id.get(short_id)
        if counts is None:
            reasons.append(f"secondary_ablation_unknown_family:{short_id}")
            continue
        variant_count = counts["positive_variants"] + counts["control_variants"]
        secondary_slots += replicate_count * variant_count * len(ablation_arms)

    declared = contract["slot_counts"]
    if primary_slots != declared["primary_future_evaluator_slots_without_secondary_ablations"]:
        reasons.append("declared_primary_slot_count_mismatch")
    if secondary_slots != declared["secondary_ablation_future_evaluator_slots"]:
        reasons.append("declared_secondary_slot_count_mismatch")
    total_slots = primary_slots + secondary_slots
    if total_slots != declared["total_future_evaluator_slots_if_all_secondary_ablations_run"]:
        reasons.append("declared_total_slot_count_mismatch")

    return FreshSlotPlanAudit(
        passed=not reasons,
        primary_future_evaluator_slots=primary_slots,
        secondary_ablation_slots=secondary_slots,
        total_future_evaluator_slots=total_slots,
        family_task_counts=family_counts,
        reason_codes=reasons or ["fresh_slot_statistics_contract_passed"],
    )


__all__ = ["FreshSlotPlanAudit", "audit_slot_statistics_contract"]
