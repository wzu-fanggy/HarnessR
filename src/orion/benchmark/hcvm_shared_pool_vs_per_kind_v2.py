"""HCVM V2: shared heterogeneous pager vs strong per-kind lazy-manager baselines.

The goal is not to claim generic resource pooling as new.  It tests a concrete Harness consequence of
managing Tool/Skill/Plugin/Process/Session with independent lazy loaders.

Arms
----
HCVM_SHARED
    One governor with one global multi-resource budget.  Idle capacity can be reused across kinds and
    unsafe cross-kind compositions are rejected by the same admission rule.
PER_KIND_STATIC_ORACLE
    Strong safe baseline.  Each kind has its own governor and fixed quota.  An oracle sees the entire
    feasible workload and chooses the subset/quota assignment that maximizes admitted phases under the
    same total prompt/RAM/slot budget.  No runtime borrowing is allowed.
PER_KIND_FULL_BUDGET
    Strong utilization baseline.  Every kind receives the full global budget independently.  This
    avoids static partition stranding but has no global admission invariant; cross-kind compositions
    can be individually admitted while their aggregate exceeds the physical budget.

The capabilities use the same V9R2 real-backed Plugin/Process/Session and Tool/Skill adapters.  Routing
is removed from this test: phases contain already-resolved logical capability handles so only resource
management is measured.
"""

from __future__ import annotations

import itertools
import json
import tempfile
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from orion.benchmark.evolvable_plugin_mixed_capability_runtime_v9r1_dev import (
    DESIGN_SKILL_ID,
    DRAFT_SKILL_ID,
    EVIDENCE_TOOL_ID,
    PROCESS_ID,
    SCIENTIFIC_ID,
    SESSION_ID,
    build_mixed_inventory,
)
from orion.benchmark.evolvable_plugin_mixed_capability_runtime_v9r2_dev import (
    PROCESS_RUNTIME_OVERHEAD_MB,
)
from orion.capvm import (
    CapabilityCatalog,
    CapabilityKind,
    CapabilityResidencyGovernor,
    EvolvableLangGraphPlugin,
    ResidencyBudgetExceeded,
    ResidencyLevel,
    ResourceVector,
    VirtualCapability,
)
from orion.capvm.adapters import EvolvablePluginCapabilityAdapter
from orion.capvm.model import MeasuredResidencyBudgetExceeded


GLOBAL_BUDGET = ResourceVector(
    prompt_tokens=80.0,
    host_ram_mb=20.0,
    process_slots=1.0,
    connection_slots=1.0,
)
RAM_PROBE_PLUGIN_ID = "ram-probe-plugin"

KINDS = (
    CapabilityKind.TOOL,
    CapabilityKind.SKILL,
    CapabilityKind.PLUGIN,
    CapabilityKind.PROCESS,
    CapabilityKind.SESSION,
)


def _required_level(capability: VirtualCapability) -> ResidencyLevel:
    return ResidencyLevel[str(capability.metadata.get("routing_required_level") or "BOUND")]


def _rv(value: ResourceVector) -> dict[str, float]:
    return {
        "prompt_tokens": value.prompt_tokens,
        "host_ram_mb": value.host_ram_mb,
        "process_slots": value.process_slots,
        "connection_slots": value.connection_slots,
    }


def _sum_vectors(values: Iterable[ResourceVector]) -> ResourceVector:
    total = ResourceVector()
    for value in values:
        total = total + value
    return total


def _max_vectors(values: Iterable[ResourceVector]) -> ResourceVector:
    peak = ResourceVector()
    for value in values:
        peak = ResourceVector(
            prompt_tokens=max(peak.prompt_tokens, value.prompt_tokens),
            host_ram_mb=max(peak.host_ram_mb, value.host_ram_mb),
            process_slots=max(peak.process_slots, value.process_slots),
            connection_slots=max(peak.connection_slots, value.connection_slots),
        )
    return peak


def _phases() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    families = (
        ("plugin", (SCIENTIFIC_ID,)),
        ("tool", (EVIDENCE_TOOL_ID,)),
        ("skill_design", (DESIGN_SKILL_ID,)),
        ("skill_draft", (DRAFT_SKILL_ID,)),
        ("process", (PROCESS_ID,)),
        ("session", (SESSION_ID,)),
        ("plugin_tool", (SCIENTIFIC_ID, EVIDENCE_TOOL_ID)),
        ("process_session", (PROCESS_ID, SESSION_ID)),
    )
    for family, capability_ids in families:
        for repeat in range(4):
            rows.append(
                {
                    "phase_id": f"{family}-{repeat:02d}",
                    "family": family,
                    "capability_ids": capability_ids,
                }
            )
    return tuple(rows)


def _unsafe_phases() -> tuple[dict[str, Any], ...]:
    return (
        {
            "phase_id": "unsafe_prompt_cross_kind",
            "capability_ids": (SCIENTIFIC_ID, EVIDENCE_TOOL_ID, DESIGN_SKILL_ID),
            "expected_violated_dimension": "prompt_tokens",
        },
        {
            "phase_id": "unsafe_ram_cross_kind",
            "capability_ids": (
                SCIENTIFIC_ID,
                RAM_PROBE_PLUGIN_ID,
                PROCESS_ID,
                SESSION_ID,
            ),
            "expected_violated_dimension": "host_ram_mb",
        },
    )


def _add_ram_probe_plugin(
    catalog: CapabilityCatalog,
    *,
    state_root: Path,
) -> CapabilityCatalog:
    plugin = EvolvableLangGraphPlugin(
        plugin_id=RAM_PROBE_PLUGIN_ID,
        version="hcvm-v2-probe",
        state_root=state_root,
        resident_mb=2.0,
    )
    capability = VirtualCapability(
        capability_id=RAM_PROBE_PLUGIN_ID,
        kind=CapabilityKind.PLUGIN,
        summary="second real-backed plugin used only for cross-kind RAM safety probe",
        adapter=EvolvablePluginCapabilityAdapter(
            plugin=plugin,
            interface_text="ram_probe()",
            reasoning_text="ram probe",
        ),
        maximum_level=ResidencyLevel.BOUND,
        metadata={"routing_required_level": "BOUND"},
    )
    return CapabilityCatalog([*catalog.all(), capability])


def _subset_catalog(catalog: CapabilityCatalog, kind: CapabilityKind) -> CapabilityCatalog:
    return CapabilityCatalog(
        [capability for capability in catalog.all() if capability.kind == kind]
    )


def _kind_for_id(catalog: CapabilityCatalog, capability_id: str) -> CapabilityKind:
    return catalog.get(capability_id).kind


def _phase_requests(catalog: CapabilityCatalog, capability_ids: Iterable[str]):
    return tuple(
        (capability_id, _required_level(catalog.get(capability_id)))
        for capability_id in capability_ids
    )


def _phase_declared_cost(catalog: CapabilityCatalog, capability_ids: Iterable[str]) -> ResourceVector:
    return _sum_vectors(
        catalog.get(capability_id).adapter.declared_cost(
            _required_level(catalog.get(capability_id))
        )
        for capability_id in capability_ids
    )


def _workload_kind_requirements(
    catalog: CapabilityCatalog,
    phases: Iterable[dict[str, Any]],
) -> dict[CapabilityKind, ResourceVector]:
    per_kind_vectors: dict[CapabilityKind, list[ResourceVector]] = {kind: [] for kind in KINDS}
    for phase in phases:
        grouped: dict[CapabilityKind, list[str]] = {kind: [] for kind in KINDS}
        for capability_id in phase["capability_ids"]:
            grouped[_kind_for_id(catalog, capability_id)].append(capability_id)
        for kind, capability_ids in grouped.items():
            if capability_ids:
                per_kind_vectors[kind].append(_phase_declared_cost(catalog, capability_ids))
    return {
        kind: _max_vectors(values) if values else ResourceVector()
        for kind, values in per_kind_vectors.items()
    }


def _oracle_static_partition(
    catalog: CapabilityCatalog,
    phases: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    requirements = _workload_kind_requirements(catalog, phases)
    best: dict[str, Any] | None = None
    for mask in range(1 << len(KINDS)):
        supported = {
            kind for index, kind in enumerate(KINDS) if mask & (1 << index)
        }
        quota_total = _sum_vectors(requirements[kind] for kind in supported)
        if not quota_total.fits(GLOBAL_BUDGET):
            continue
        success_count = sum(
            all(_kind_for_id(catalog, capability_id) in supported for capability_id in phase["capability_ids"])
            for phase in phases
        )
        candidate = {
            "supported": supported,
            "success_count": success_count,
            "quota_total": quota_total,
        }
        if best is None:
            best = candidate
            continue
        best_key = (
            int(best["success_count"]),
            len(best["supported"]),
            -best["quota_total"].prompt_tokens,
            -best["quota_total"].host_ram_mb,
        )
        candidate_key = (
            int(candidate["success_count"]),
            len(candidate["supported"]),
            -candidate["quota_total"].prompt_tokens,
            -candidate["quota_total"].host_ram_mb,
        )
        if candidate_key > best_key:
            best = candidate
    if best is None:
        raise AssertionError("no feasible static partition, even empty subset")
    quotas = {
        kind: requirements[kind] if kind in best["supported"] else ResourceVector()
        for kind in KINDS
    }
    return {
        "supported_kinds": tuple(sorted(kind.value for kind in best["supported"])),
        "success_count": int(best["success_count"]),
        "kind_requirements": {kind.value: _rv(requirements[kind]) for kind in KINDS},
        "kind_quotas": {kind.value: _rv(quotas[kind]) for kind in KINDS},
        "quota_total": _rv(best["quota_total"]),
        "unallocated_budget": _rv(GLOBAL_BUDGET - best["quota_total"]),
    }


def _execute_global_shared(
    catalog: CapabilityCatalog,
    phases: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    governor = CapabilityResidencyGovernor(catalog=catalog, budget=GLOBAL_BUDGET)
    rows: list[dict[str, Any]] = []
    peak_measured = ResourceVector()
    try:
        for phase in phases:
            try:
                with governor.acquire_many(
                    _phase_requests(catalog, phase["capability_ids"]),
                    request_id=f"shared:{phase['phase_id']}",
                    reconcile_before_active=True,
                ):
                    measured = governor.total_measured_cost()
                    peak_measured = _max_vectors((peak_measured, measured))
                    violations = measured.violated_dimensions(GLOBAL_BUDGET)
                    rows.append(
                        {
                            "phase_id": phase["phase_id"],
                            "success": not violations,
                            "violations": list(violations),
                            "measured": _rv(measured),
                        }
                    )
            except (ResidencyBudgetExceeded, MeasuredResidencyBudgetExceeded) as exc:
                rows.append(
                    {
                        "phase_id": phase["phase_id"],
                        "success": False,
                        "error": str(exc),
                    }
                )
        unsafe_rows: list[dict[str, Any]] = []
        for phase in _unsafe_phases():
            try:
                with governor.acquire_many(
                    _phase_requests(catalog, phase["capability_ids"]),
                    request_id=f"shared:{phase['phase_id']}",
                    reconcile_before_active=True,
                ):
                    measured = governor.total_measured_cost()
                    unsafe_rows.append(
                        {
                            "phase_id": phase["phase_id"],
                            "rejected": False,
                            "aggregate_measured": _rv(measured),
                            "violations": list(measured.violated_dimensions(GLOBAL_BUDGET)),
                        }
                    )
            except (ResidencyBudgetExceeded, MeasuredResidencyBudgetExceeded) as exc:
                unsafe_rows.append(
                    {
                        "phase_id": phase["phase_id"],
                        "rejected": True,
                        "error": str(exc),
                        "expected_violated_dimension": phase["expected_violated_dimension"],
                    }
                )
        return {
            "success_count": sum(bool(row["success"]) for row in rows),
            "phase_count": len(rows),
            "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
            "peak_measured": _rv(peak_measured),
            "unsafe_rows": unsafe_rows,
            "unsafe_rejection_rate": sum(bool(row["rejected"]) for row in unsafe_rows) / len(unsafe_rows),
            "event_counts": dict(
                sorted(
                    {
                        event_type: sum(event.event_type == event_type for event in governor.events)
                        for event_type in {event.event_type for event in governor.events}
                    }.items()
                )
            ),
            "rows": rows,
        }
    finally:
        governor.close()


def _build_kind_governors(
    catalog: CapabilityCatalog,
    kind_budgets: dict[CapabilityKind, ResourceVector],
) -> dict[CapabilityKind, CapabilityResidencyGovernor]:
    governors: dict[CapabilityKind, CapabilityResidencyGovernor] = {}
    for kind in KINDS:
        governors[kind] = CapabilityResidencyGovernor(
            catalog=_subset_catalog(catalog, kind),
            budget=kind_budgets[kind],
        )
    return governors


def _close_governors(governors: dict[CapabilityKind, CapabilityResidencyGovernor]) -> None:
    for governor in governors.values():
        governor.close()


def _reset_governors_to_pinned_floor(
    governors: dict[CapabilityKind, CapabilityResidencyGovernor],
) -> None:
    for governor in governors.values():
        for state in governor.states():
            capability = governor.catalog.get(state.capability_id)
            if state.current_level > capability.pinned_min_level:
                governor.demote(state.capability_id, capability.pinned_min_level)


def _execute_partitioned_phase(
    catalog: CapabilityCatalog,
    governors: dict[CapabilityKind, CapabilityResidencyGovernor],
    capability_ids: Iterable[str],
    *,
    request_id: str,
) -> tuple[bool, ResourceVector, tuple[str, ...], str]:
    grouped: dict[CapabilityKind, list[str]] = {kind: [] for kind in KINDS}
    for capability_id in capability_ids:
        grouped[_kind_for_id(catalog, capability_id)].append(capability_id)
    try:
        with ExitStack() as stack:
            for kind, ids in grouped.items():
                if not ids:
                    continue
                subcatalog = governors[kind].catalog
                requests = tuple(
                    (capability_id, _required_level(subcatalog.get(capability_id)))
                    for capability_id in ids
                )
                stack.enter_context(
                    governors[kind].acquire_many(
                        requests,
                        request_id=f"{request_id}:{kind.value}",
                        reconcile_before_active=True,
                    )
                )
            aggregate = _sum_vectors(
                governor.total_measured_cost() for governor in governors.values()
            )
            violations = aggregate.violated_dimensions(GLOBAL_BUDGET)
            return True, aggregate, violations, ""
    except (ResidencyBudgetExceeded, MeasuredResidencyBudgetExceeded, ValueError) as exc:
        aggregate = _sum_vectors(
            governor.total_measured_cost() for governor in governors.values()
        )
        return False, aggregate, aggregate.violated_dimensions(GLOBAL_BUDGET), str(exc)


def _execute_static_oracle(
    catalog: CapabilityCatalog,
    phases: tuple[dict[str, Any], ...],
    partition: dict[str, Any],
) -> dict[str, Any]:
    supported = {CapabilityKind(value) for value in partition["supported_kinds"]}
    budgets = {
        kind: ResourceVector(**partition["kind_quotas"][kind.value])
        for kind in KINDS
    }
    governors = _build_kind_governors(catalog, budgets)
    rows: list[dict[str, Any]] = []
    try:
        for phase in phases:
            required_kinds = {
                _kind_for_id(catalog, capability_id)
                for capability_id in phase["capability_ids"]
            }
            if not required_kinds.issubset(supported):
                rows.append(
                    {
                        "phase_id": phase["phase_id"],
                        "success": False,
                        "reason": "kind_has_no_safe_static_quota",
                        "required_kinds": sorted(kind.value for kind in required_kinds),
                    }
                )
                continue
            success, aggregate, violations, error = _execute_partitioned_phase(
                catalog,
                governors,
                phase["capability_ids"],
                request_id=f"static:{phase['phase_id']}",
            )
            rows.append(
                {
                    "phase_id": phase["phase_id"],
                    "success": bool(success and not violations),
                    "aggregate_measured": _rv(aggregate),
                    "violations": list(violations),
                    "error": error,
                }
            )
        return {
            "success_count": sum(bool(row["success"]) for row in rows),
            "phase_count": len(rows),
            "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
            "rows": rows,
        }
    finally:
        _close_governors(governors)


def _execute_full_budget_independent(
    catalog: CapabilityCatalog,
    phases: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    budgets = {kind: GLOBAL_BUDGET for kind in KINDS}
    governors = _build_kind_governors(catalog, budgets)
    rows: list[dict[str, Any]] = []
    try:
        for phase in phases:
            _reset_governors_to_pinned_floor(governors)
            success, aggregate, violations, error = _execute_partitioned_phase(
                catalog,
                governors,
                phase["capability_ids"],
                request_id=f"independent:{phase['phase_id']}",
            )
            rows.append(
                {
                    "phase_id": phase["phase_id"],
                    "local_admission_success": success,
                    "aggregate_measured": _rv(aggregate),
                    "global_violations": list(violations),
                    "error": error,
                }
            )
        unsafe_rows: list[dict[str, Any]] = []
        for phase in _unsafe_phases():
            _reset_governors_to_pinned_floor(governors)
            success, aggregate, violations, error = _execute_partitioned_phase(
                catalog,
                governors,
                phase["capability_ids"],
                request_id=f"independent:{phase['phase_id']}",
            )
            unsafe_rows.append(
                {
                    "phase_id": phase["phase_id"],
                    "all_local_managers_admitted": success,
                    "aggregate_measured": _rv(aggregate),
                    "global_violations": list(violations),
                    "global_safety_violation": bool(success and violations),
                    "expected_violated_dimension": phase["expected_violated_dimension"],
                    "error": error,
                }
            )
        return {
            "local_success_count": sum(bool(row["local_admission_success"]) for row in rows),
            "phase_count": len(rows),
            "local_success_rate": sum(bool(row["local_admission_success"]) for row in rows) / len(rows),
            "globally_safe_phase_count": sum(
                bool(row["local_admission_success"]) and not row["global_violations"]
                for row in rows
            ),
            "unsafe_rows": unsafe_rows,
            "unsafe_global_violation_rate": sum(
                bool(row["global_safety_violation"]) for row in unsafe_rows
            ) / len(unsafe_rows),
            "rows": rows,
        }
    finally:
        _close_governors(governors)


def run_benchmark() -> dict[str, Any]:
    phases = _phases()

    # Fresh real-backed inventories per arm avoid adapter/process/session state leakage.
    with tempfile.TemporaryDirectory(prefix="orion-hcvm-shared-pool-") as tmp:
        root = Path(tmp)
        shared_base, _, _ = build_mixed_inventory(
            state_root=root / "shared-state",
            process_runtime_overhead_mb=PROCESS_RUNTIME_OVERHEAD_MB,
        )
        static_base, _, _ = build_mixed_inventory(
            state_root=root / "static-state",
            process_runtime_overhead_mb=PROCESS_RUNTIME_OVERHEAD_MB,
        )
        full_base, _, _ = build_mixed_inventory(
            state_root=root / "full-state",
            process_runtime_overhead_mb=PROCESS_RUNTIME_OVERHEAD_MB,
        )
        shared_catalog = _add_ram_probe_plugin(
            shared_base, state_root=root / "shared-ram-probe"
        )
        static_catalog = _add_ram_probe_plugin(
            static_base, state_root=root / "static-ram-probe"
        )
        full_catalog = _add_ram_probe_plugin(
            full_base, state_root=root / "full-ram-probe"
        )

        partition = _oracle_static_partition(static_catalog, phases)
        shared = _execute_global_shared(shared_catalog, phases)
        static = _execute_static_oracle(static_catalog, phases, partition)
        independent = _execute_full_budget_independent(full_catalog, phases)

    return {
        "schema_version": "hcvm-shared-pool-vs-per-kind-v2.0",
        "status": "DEVELOPMENT_ONLY_RESOURCE_POOLING_KILL_TEST",
        "global_budget": _rv(GLOBAL_BUDGET),
        "phase_count": len(phases),
        "unsafe_probe_count": len(_unsafe_phases()),
        "oracle_static_partition": partition,
        "arms": {
            "HCVM_SHARED": shared,
            "PER_KIND_STATIC_ORACLE": static,
            "PER_KIND_FULL_BUDGET": independent,
        },
        "closure_gate": {
            "shared_completes_all_feasible_phases": shared["success_count"] == len(phases),
            "shared_rejects_all_unsafe_compositions": shared["unsafe_rejection_rate"] == 1.0,
            "oracle_static_has_stranding_loss": static["success_count"] < len(phases),
            "independent_full_budget_has_global_safety_loss": (
                independent["unsafe_global_violation_rate"] > 0.0
            ),
            "shared_avoids_static_stranding_and_independent_overcommit": (
                shared["success_count"] == len(phases)
                and shared["unsafe_rejection_rate"] == 1.0
                and static["success_count"] < len(phases)
                and independent["unsafe_global_violation_rate"] > 0.0
            ),
        },
        "interpretation_boundary": (
            "This demonstrates the systems benefit of one shared Harness capability budget over "
            "safe fixed per-kind partitions or unsafe independent full-budget managers. Generic "
            "resource pooling is not claimed as new; the architecture question is whether heterogeneous "
            "agent-Harness capabilities should participate in one VM-style residency protocol."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2))
