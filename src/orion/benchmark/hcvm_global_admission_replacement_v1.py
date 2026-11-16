"""HCVM: isolate cross-kind replacement from generic global admission.

Motivation
----------
MARS, Agent libOS and other agent runtimes already establish that global resource
visibility/admission is not new.  This benchmark therefore gives the baseline a
single global residency-capacity gate and asks a narrower question: what happens
when subsystem-local lazy managers cannot evict another subsystem's idle backing?

Arms
----
HCVM_SHARED
    Existing shared governor: global admission + cross-kind replacement.
GLOBAL_ADMISSION_KEEP_CACHE
    Per-kind lazy managers share one external global admission check, so unsafe
    aggregate compositions are rejected.  Successful idle residents remain cached,
    but the coordinator has no cross-kind replacement operation.  If stale cache
    consumes the capacity needed by a new kind, the request is safely rejected.
GLOBAL_ADMISSION_RELEASE_ALL_IDLE
    Same global admission rule, but all inactive subsystem caches are demoted to
    their pinned floor before every phase.  This is a strong safety/completion
    baseline: it can avoid stale-cache blockage without implementing cross-kind
    victim selection, at the cost of repeated refault/materialization.

No routing is measured.  The frozen phase sequence and real-backed mixed inventory
are reused from hcvm_shared_pool_vs_per_kind_v2.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import orion.benchmark.hcvm_shared_pool_vs_per_kind_v2 as v2
from orion.benchmark.evolvable_plugin_mixed_capability_runtime_v9r1_dev import (
    build_mixed_inventory,
)
from orion.benchmark.evolvable_plugin_mixed_capability_runtime_v9r2_dev import (
    PROCESS_RUNTIME_OVERHEAD_MB,
)
from orion.capvm import (
    CapabilityCatalog,
    CapabilityKind,
    CapabilityResidencyGovernor,
    ResidencyBudgetExceeded,
    ResidencyLevel,
    ResourceVector,
)
from orion.capvm.model import MeasuredResidencyBudgetExceeded


def _aggregate_declared(governors) -> ResourceVector:
    return v2._sum_vectors(governor.total_declared_cost() for governor in governors.values())


def _aggregate_measured(governors) -> ResourceVector:
    return v2._sum_vectors(governor.total_measured_cost() for governor in governors.values())


def _projected_declared(
    catalog: CapabilityCatalog,
    governors,
    capability_ids: tuple[str, ...],
) -> ResourceVector:
    """Project global resident cost without allowing cross-kind eviction.

    The frozen mixed inventory currently has no dependency edges.  Fail loudly if
    that changes so this baseline is not silently made weaker than HCVM.
    """

    projected = _aggregate_declared(governors)
    seen: set[str] = set()
    for capability_id in capability_ids:
        if capability_id in seen:
            continue
        seen.add(capability_id)
        capability = catalog.get(capability_id)
        if capability.dependencies:
            raise AssertionError(
                "GLOBAL_ADMISSION baseline must be updated for capability dependencies"
            )
        kind = capability.kind
        state = governors[kind].state(capability_id)
        target = v2._required_level(capability)
        if state.current_level >= target:
            continue
        projected = (
            projected
            - capability.adapter.declared_cost(state.current_level)
            + capability.adapter.declared_cost(target)
        )
    return projected


def _event_counts(governors) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for governor in governors.values():
        counts.update(event.event_type for event in governor.events)
    return dict(sorted(counts.items()))


def _reset_shared_to_pinned_floor(governor: CapabilityResidencyGovernor) -> None:
    for state in governor.states():
        capability = governor.catalog.get(state.capability_id)
        if state.current_level > capability.pinned_min_level:
            governor.demote(state.capability_id, capability.pinned_min_level)


def _execute_hcvm_shared(
    catalog: CapabilityCatalog,
    phases: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Shared HCVM arm with workload events separated from unsafe probes."""

    governor = CapabilityResidencyGovernor(catalog=catalog, budget=v2.GLOBAL_BUDGET)
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    try:
        for phase in phases:
            try:
                with governor.acquire_many(
                    v2._phase_requests(catalog, phase["capability_ids"]),
                    request_id=f"shared:{phase['phase_id']}",
                    reconcile_before_active=True,
                ):
                    measured = governor.total_measured_cost()
                    violations = measured.violated_dimensions(v2.GLOBAL_BUDGET)
                    rows.append(
                        {
                            "phase_id": phase["phase_id"],
                            "success": not violations,
                            "violations": list(violations),
                            "measured": v2._rv(measured),
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

        # Freeze feasible-workload event counts before the separate safety probes.
        workload_event_counts = dict(
            sorted(Counter(event.event_type for event in governor.events).items())
        )

        unsafe_rows: list[dict[str, Any]] = []
        for phase in v2._unsafe_phases():
            _reset_shared_to_pinned_floor(governor)
            try:
                with governor.acquire_many(
                    v2._phase_requests(catalog, phase["capability_ids"]),
                    request_id=f"shared-probe:{phase['phase_id']}",
                    reconcile_before_active=True,
                ):
                    measured = governor.total_measured_cost()
                    unsafe_rows.append(
                        {
                            "phase_id": phase["phase_id"],
                            "rejected": False,
                            "aggregate_measured": v2._rv(measured),
                            "violations": list(
                                measured.violated_dimensions(v2.GLOBAL_BUDGET)
                            ),
                        }
                    )
            except (ResidencyBudgetExceeded, MeasuredResidencyBudgetExceeded) as exc:
                unsafe_rows.append(
                    {
                        "phase_id": phase["phase_id"],
                        "rejected": True,
                        "error": str(exc),
                        "expected_violated_dimension": phase[
                            "expected_violated_dimension"
                        ],
                    }
                )

        elapsed = perf_counter() - started
        return {
            "success_count": sum(bool(row["success"]) for row in rows),
            "phase_count": len(rows),
            "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
            "unsafe_rejection_rate": sum(bool(row["rejected"]) for row in unsafe_rows)
            / len(unsafe_rows),
            "event_counts": workload_event_counts,
            "elapsed_s": elapsed,
            "unsafe_rows": unsafe_rows,
            "rows": rows,
        }
    finally:
        governor.close()


def _execute_global_admission_per_kind(
    catalog: CapabilityCatalog,
    phases: tuple[dict[str, Any], ...],
    *,
    release_all_idle_before_phase: bool,
) -> dict[str, Any]:
    budgets = {kind: v2.GLOBAL_BUDGET for kind in v2.KINDS}
    governors = v2._build_kind_governors(catalog, budgets)
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    try:
        for phase in phases:
            if release_all_idle_before_phase:
                v2._reset_governors_to_pinned_floor(governors)

            ids = tuple(str(item) for item in phase["capability_ids"])
            projected = _projected_declared(catalog, governors, ids)
            projected_violations = projected.violated_dimensions(v2.GLOBAL_BUDGET)
            if projected_violations:
                rows.append(
                    {
                        "phase_id": phase["phase_id"],
                        "success": False,
                        "reason": "global_residency_capacity_preflight_reject",
                        "projected_declared": v2._rv(projected),
                        "violations": list(projected_violations),
                        "resident_before": {
                            kind.value: [
                                state.capability_id
                                for state in governors[kind].states()
                                if state.current_level > ResidencyLevel.CATALOG
                            ]
                            for kind in v2.KINDS
                        },
                    }
                )
                continue

            success, aggregate, violations, error = v2._execute_partitioned_phase(
                catalog,
                governors,
                ids,
                request_id=(
                    "release-all" if release_all_idle_before_phase else "keep-cache"
                )
                + f":{phase['phase_id']}",
            )
            rows.append(
                {
                    "phase_id": phase["phase_id"],
                    "success": bool(success and not violations),
                    "reason": "" if success and not violations else "post_page_in_global_violation",
                    "projected_declared": v2._rv(projected),
                    "aggregate_measured": v2._rv(aggregate),
                    "violations": list(violations),
                    "error": error,
                }
            )

        # Freeze feasible-workload state before safety probes reset caches.
        workload_event_counts = _event_counts(governors)
        workload_final_declared = _aggregate_declared(governors)
        workload_final_measured = _aggregate_measured(governors)

        unsafe_rows: list[dict[str, Any]] = []
        for phase in v2._unsafe_phases():
            # Unsafe probes compare the requested composition itself, not stale
            # cache left by the preceding feasible workload.  Reset every arm
            # to the same pinned floor before each probe.
            v2._reset_governors_to_pinned_floor(governors)
            ids = tuple(str(item) for item in phase["capability_ids"])
            projected = _projected_declared(catalog, governors, ids)
            violations = projected.violated_dimensions(v2.GLOBAL_BUDGET)
            unsafe_rows.append(
                {
                    "phase_id": phase["phase_id"],
                    "rejected": bool(violations),
                    "projected_declared": v2._rv(projected),
                    "violations": list(violations),
                    "expected_violated_dimension": phase["expected_violated_dimension"],
                }
            )

        elapsed = perf_counter() - started
        return {
            "success_count": sum(bool(row["success"]) for row in rows),
            "phase_count": len(rows),
            "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
            "unsafe_rejection_rate": sum(bool(row["rejected"]) for row in unsafe_rows)
            / len(unsafe_rows),
            "event_counts": workload_event_counts,
            "final_declared_residency": v2._rv(workload_final_declared),
            "final_measured_residency": v2._rv(workload_final_measured),
            "elapsed_s": elapsed,
            "unsafe_rows": unsafe_rows,
            "rows": rows,
        }
    finally:
        v2._close_governors(governors)


def run_benchmark() -> dict[str, Any]:
    phases = v2._phases()
    with tempfile.TemporaryDirectory(prefix="orion-hcvm-global-admission-") as tmp:
        root = Path(tmp)
        shared_base, _, _ = build_mixed_inventory(
            state_root=root / "shared-state",
            process_runtime_overhead_mb=PROCESS_RUNTIME_OVERHEAD_MB,
        )
        keep_base, _, _ = build_mixed_inventory(
            state_root=root / "keep-state",
            process_runtime_overhead_mb=PROCESS_RUNTIME_OVERHEAD_MB,
        )
        release_base, _, _ = build_mixed_inventory(
            state_root=root / "release-state",
            process_runtime_overhead_mb=PROCESS_RUNTIME_OVERHEAD_MB,
        )
        shared_catalog = v2._add_ram_probe_plugin(
            shared_base, state_root=root / "shared-probe"
        )
        keep_catalog = v2._add_ram_probe_plugin(
            keep_base, state_root=root / "keep-probe"
        )
        release_catalog = v2._add_ram_probe_plugin(
            release_base, state_root=root / "release-probe"
        )

        shared = _execute_hcvm_shared(shared_catalog, phases)
        keep = _execute_global_admission_per_kind(
            keep_catalog,
            phases,
            release_all_idle_before_phase=False,
        )
        release = _execute_global_admission_per_kind(
            release_catalog,
            phases,
            release_all_idle_before_phase=True,
        )

    return {
        "schema_version": "hcvm-global-admission-replacement-v1.0",
        "status": "DEVELOPMENT_STRONG_GLOBAL_ADMISSION_BASELINE",
        "global_residency_capacity": v2._rv(v2.GLOBAL_BUDGET),
        "phase_count": len(phases),
        "phase_order": [str(phase["phase_id"]) for phase in phases],
        "arms": {
            "HCVM_SHARED": {
                "success_count": shared["success_count"],
                "success_rate": shared["success_rate"],
                "unsafe_rejection_rate": shared["unsafe_rejection_rate"],
                "event_counts": shared["event_counts"],
                "elapsed_s": shared["elapsed_s"],
            },
            "GLOBAL_ADMISSION_KEEP_CACHE_NO_CROSS_KIND_REPLACEMENT": keep,
            "GLOBAL_ADMISSION_RELEASE_ALL_IDLE": release,
        },
        "closure_gate": {
            "all_arms_globally_safe_on_unsafe_probes": (
                shared["unsafe_rejection_rate"] == 1.0
                and keep["unsafe_rejection_rate"] == 1.0
                and release["unsafe_rejection_rate"] == 1.0
            ),
            "hcvm_completes_all_feasible_phases": shared["success_count"] == len(phases),
            "keep_cache_without_cross_kind_replacement_has_capacity_blockage": (
                keep["success_count"] < len(phases)
            ),
            "release_all_idle_recovers_completion": release["success_count"] == len(phases),
            "release_all_idle_has_more_refaults_than_hcvm": (
                release["event_counts"].get("refault", 0)
                > shared["event_counts"].get("refault", 0)
            ),
        },
        "interpretation_guard": (
            "Global admission/resource visibility is deliberately granted to both baselines. "
            "The experiment isolates residency replacement: keeping subsystem caches without "
            "cross-kind replacement can safely block later feasible phases, whereas eagerly "
            "releasing all idle backing can recover completion but increases refault/materialization. "
            "HCVM's claimed mechanism is the shared cross-kind residency/replacement contract, not "
            "generic global admission or multi-resource budgeting."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2, sort_keys=True))
