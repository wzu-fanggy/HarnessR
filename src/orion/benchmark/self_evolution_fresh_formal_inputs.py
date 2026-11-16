"""Deterministic validation and in-memory assembly for the fresh formal study inputs.

This module performs no model, RAG, KG, network or persistent-production calls.
It exists so sidecar artifacts (notably A8) cannot become part of a formal
snapshot merely by existing on disk: they must be explicitly wired into the
frozen family/source/contract/comparator graph and pass the same mechanical
audits as the base artifacts.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orion.benchmark.self_evolution_fresh_admission import FreshScientificFamilyContract
from orion.benchmark.self_evolution_fresh_generic_c import audit_generic_c_spec
from orion.benchmark.self_evolution_fresh_preflight import evaluate_family_preflight
from orion.benchmark.self_evolution_fresh_slot_plan import audit_slot_statistics_contract


EXPECTED_FORMAL_FAMILIES = {
    "FRESH_A1_NADFP_WIDE_TEMP",
    "FRESH_A4_DTD_HIGH_TEMP",
    "FRESH_A5_ETHER_HIGH_VOLTAGE_BOUNDARY",
    "FRESH_A6_GLYME_LOW_VOLTAGE_PRESERVATION",
    "FRESH_A8_PFPN_FULL_CELL_PRESERVATION",
    "FRESH_C1_INFRASTRUCTURE_NO_UPDATE",
}
EXCLUDED_A7 = "FRESH_A7_FEC_FULL_CELL_PRESERVATION"


class FreshFormalInputAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "orion_self_evolution_fresh_formal_input_audit_v1"
    passed: bool
    family_allowlist: list[str]
    source_families: list[str]
    scientific_contract_families: list[str]
    generic_c_families: list[str]
    slot_primary_count: int
    slot_total_with_ablations: int
    retention_candidate_slots: int
    formal_roots_zero: bool
    reason_codes: list[str] = Field(default_factory=list)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(root: Path, rel: str, reasons: list[str]) -> Path:
    path = root / rel
    if not path.is_file():
        reasons.append(f"missing_formal_input:{rel}")
    return path


def assemble_formal_source_bundle(*, root: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    base = _load(root / manifest["source_authorities"]["base"])
    merged = copy.deepcopy(base)
    families = merged.setdefault("families", {})
    # Historical/development A7 remains on disk but is not present in the formal in-memory bundle.
    families.pop(EXCLUDED_A7, None)
    for family_id, rel in manifest["source_authorities"].get("sidecars", {}).items():
        sidecar = _load(root / rel)
        if sidecar.get("family_id") != family_id:
            raise ValueError(f"source sidecar family mismatch: {family_id}")
        sidecar_families = sidecar.get("families") or {}
        if set(sidecar_families) != {family_id}:
            raise ValueError(f"source sidecar family payload mismatch: {family_id}")
        families[family_id] = copy.deepcopy(sidecar_families[family_id])
    return merged


def assemble_formal_scientific_contracts(*, root: str | Path, manifest: dict[str, Any]) -> dict[str, FreshScientificFamilyContract]:
    root = Path(root)
    base = _load(root / manifest["scientific_contract_authorities"]["base"])
    raw_families = copy.deepcopy(base.get("families") or {})
    raw_families.pop(EXCLUDED_A7, None)
    for family_id, rel in manifest["scientific_contract_authorities"].get("sidecars", {}).items():
        sidecar = _load(root / rel)
        # A family-specific scientific-contract sidecar is itself the contract
        # object; family identity is supplied by the formal manifest key.
        raw_families[family_id] = sidecar
    return {
        family_id: FreshScientificFamilyContract.model_validate(payload)
        for family_id, payload in raw_families.items()
        if family_id in EXPECTED_FORMAL_FAMILIES
    }


def assemble_formal_generic_c_spec(*, root: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    spec = copy.deepcopy(_load(root / manifest["generic_c_authorities"]["base"]))
    tasks = spec.setdefault("family_tasks", {})
    tasks.pop(EXCLUDED_A7, None)
    for family_id, rel in manifest["generic_c_authorities"].get("family_sidecars", {}).items():
        sidecar = _load(root / rel)
        sidecar_tasks = sidecar.get("family_tasks") or {}
        if set(sidecar_tasks) != {family_id}:
            raise ValueError(f"generic-C sidecar family mismatch: {family_id}")
        tasks[family_id] = sidecar_tasks[family_id]
    # C1 is no-update and intentionally has no candidate trajectory/judge episode.
    allowed_generic = EXPECTED_FORMAL_FAMILIES - {"FRESH_C1_INFRASTRUCTURE_NO_UPDATE"}
    spec["family_tasks"] = {k: v for k, v in tasks.items() if k in allowed_generic}
    return spec


def audit_fresh_formal_inputs(
    *,
    root: str | Path = ".",
    manifest_path: str | Path = "benchmark/self_evolution_fresh_v1/FRESH_FORMAL_INPUTS_V1_2026-08-18.json",
) -> FreshFormalInputAudit:
    root = Path(root)
    manifest_file = root / Path(manifest_path)
    manifest = _load(manifest_file)
    reasons: list[str] = []

    allowlist = set(manifest.get("family_allowlist") or [])
    if allowlist != EXPECTED_FORMAL_FAMILIES:
        reasons.append("formal_family_allowlist_mismatch")
    deny = manifest.get("family_denylist") or {}
    if EXCLUDED_A7 not in deny:
        reasons.append("a7_not_explicitly_denied")
    if EXCLUDED_A7 in allowlist:
        reasons.append("a7_illegally_present_in_formal_allowlist")

    # Every declared path must exist before snapshot creation.
    path_fields = [
        manifest["blueprint"],
        manifest["source_authorities"]["base"],
        manifest["scientific_contract_authorities"]["base"],
        manifest["generic_c_authorities"]["base"],
        manifest["generic_c_authorities"]["judge_prompt"],
        manifest["common_proposer"]["prompt"],
        manifest["common_proposer"]["proposal_schema_module"],
        manifest["arm_semantics"],
        manifest["slot_statistics"],
        manifest["retention_bank"],
        manifest["mechanism_ablations"],
        manifest["behavior_scorer_module"],
        manifest["productive_scorer_module"],
        manifest["retention_scorer_module"],
        manifest["parallel_admission_module"],
        manifest["scientific_admission_module"],
        manifest["scope_module"],
    ]
    for mapping_key in ("family_manifests", "task_bundles", "productive_contracts", "eligibility_receipts"):
        values = manifest.get(mapping_key) or {}
        path_fields.extend(str(v) for v in values.values())
    path_fields.extend(str(v) for v in manifest["source_authorities"].get("sidecars", {}).values())
    path_fields.extend(str(v) for v in manifest["scientific_contract_authorities"].get("sidecars", {}).values())
    path_fields.extend(str(v) for v in manifest["generic_c_authorities"].get("family_sidecars", {}).values())
    for rel in path_fields:
        _require_file(root, rel, reasons)

    source_bundle = assemble_formal_source_bundle(root=root, manifest=manifest)
    source_families = set(source_bundle.get("families") or {})
    if not EXPECTED_FORMAL_FAMILIES <= source_families:
        reasons.append("formal_source_family_missing")
    if EXCLUDED_A7 in source_families:
        reasons.append("a7_present_in_formal_source_bundle")

    contracts = assemble_formal_scientific_contracts(root=root, manifest=manifest)
    contract_families = set(contracts)
    if contract_families != EXPECTED_FORMAL_FAMILIES:
        reasons.append("formal_scientific_contract_family_mismatch")

    generic_spec = assemble_formal_generic_c_spec(root=root, manifest=manifest)
    generic_families = set(generic_spec.get("family_tasks") or {})
    expected_generic = EXPECTED_FORMAL_FAMILIES - {"FRESH_C1_INFRASTRUCTURE_NO_UPDATE"}
    if generic_families != expected_generic:
        reasons.append("formal_generic_c_family_mismatch")
    generic_audit = audit_generic_c_spec(
        spec=generic_spec,
        source_bundle=source_bundle,
        root=root,
    )
    if not generic_audit.passed:
        reasons.extend(f"generic_c:{code}" for code in generic_audit.reason_codes)

    # All scientific family manifests must pass the existing family preflight.
    for family_id in sorted(EXPECTED_FORMAL_FAMILIES):
        manifest_rel = manifest["family_manifests"].get(family_id)
        if not manifest_rel:
            reasons.append(f"missing_family_manifest_binding:{family_id}")
            continue
        report = evaluate_family_preflight(_load(root / manifest_rel))
        if not report["passed"]:
            reasons.extend(f"family_preflight:{family_id}:{code}" for code in report["failure_reason_codes"])

    a8_receipt = _load(root / manifest["eligibility_receipts"]["FRESH_A8_PFPN_FULL_CELL_PRESERVATION"])
    if a8_receipt.get("headroom_passed") is not True:
        reasons.append("a8_static_headroom_not_passed")
    if (a8_receipt.get("confirmatory_consequence") or {}).get("A8_behavior_family_eligible") is not True:
        reasons.append("a8_not_confirmatory_eligible")

    slot_contract = _load(root / manifest["slot_statistics"])
    slot_audit = audit_slot_statistics_contract(slot_contract, root=root)
    if not slot_audit.passed:
        reasons.extend(f"slot_plan:{code}" for code in slot_audit.reason_codes)
    if slot_audit.primary_future_evaluator_slots != 168:
        reasons.append("formal_primary_slot_count_not_168")
    if slot_audit.total_future_evaluator_slots != 180:
        reasons.append("formal_total_slot_count_not_180")

    retention = _load(root / manifest["retention_bank"])
    retention_slots = int((retention.get("slot_projection") or {}).get("candidate_mounted_retention_slots", -1))
    if retention_slots != 45:
        reasons.append("formal_retention_candidate_slot_count_not_45")
    if len(retention.get("eligible_task_ids") or []) != 3:
        reasons.append("formal_retention_eligible_anchor_count_not_3")

    # No formal outputs may exist before snapshot/authorization.
    formal_roots = manifest["formal_roots"]
    formal_files: list[Path] = []
    for rel in (formal_roots["locks"], formal_roots["results"]):
        tree = root / rel
        if tree.exists():
            formal_files.extend(path for path in tree.rglob("*") if path.is_file())
    roots_zero = not formal_files
    if not roots_zero:
        reasons.append("formal_roots_not_zero_before_freeze")

    formal_execution = manifest.get("formal_execution") or {}
    if formal_execution.get("authorized") is not False:
        reasons.append("formal_inputs_must_remain_unauthorized")
    if formal_execution.get("production_persistence_authorized") is not False:
        reasons.append("production_persistence_must_remain_unauthorized")

    return FreshFormalInputAudit(
        passed=not reasons,
        family_allowlist=sorted(allowlist),
        source_families=sorted(source_families & EXPECTED_FORMAL_FAMILIES),
        scientific_contract_families=sorted(contract_families),
        generic_c_families=sorted(generic_families),
        slot_primary_count=slot_audit.primary_future_evaluator_slots,
        slot_total_with_ablations=slot_audit.total_future_evaluator_slots,
        retention_candidate_slots=retention_slots,
        formal_roots_zero=roots_zero,
        reason_codes=reasons or ["fresh_formal_inputs_passed"],
    )


__all__ = [
    "EXPECTED_FORMAL_FAMILIES",
    "EXCLUDED_A7",
    "FreshFormalInputAudit",
    "assemble_formal_generic_c_spec",
    "assemble_formal_scientific_contracts",
    "assemble_formal_source_bundle",
    "audit_fresh_formal_inputs",
]
