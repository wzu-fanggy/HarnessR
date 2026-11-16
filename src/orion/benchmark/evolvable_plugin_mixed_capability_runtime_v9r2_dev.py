"""V9R2：在 V9R1 mixed-real bank 上开发 semantic abstention + measured reconciliation。

这是 R1 暴露两个失败后的 development repair，不是独立 confirmation。数据、任务与 inventory
沿用 V9R1；V8B 仍完全不读。R2 只修改 runtime policy：

1. sparse development embedding 的 query-signal confidence floor；
2. page-in 后 measured-cost reconciliation；
3. process payload 与解释器/runtime baseline 分开声明，并以真实物理 working-set envelope
   设置 hard budget，而不是把 3 MB payload 误当成完整 child RSS。
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from orion.benchmark.evolvable_plugin_mixed_capability_runtime_v9r1_dev import (
    PROCESS_ID,
    SCIENTIFIC_ID,
    SESSION_ID,
    _composition_tasks,
    _encoder,
    _family_summary,
    _request,
    _round_robin_single_tasks,
    build_mixed_inventory,
)
from orion.capvm import (
    CapabilityControlPlane,
    CapabilityResidencyGovernor,
    CascadedDiscoveryBackend,
    EmbeddingDiscoveryBackend,
    EvolvableCapabilityRouter,
    ExactTriggerDiscoveryBackend,
    MetadataDiscoveryBackend,
    ResourceVector,
    RoutedCapabilityRuntime,
)


def _max_vector(left: ResourceVector, right: ResourceVector) -> ResourceVector:
    return ResourceVector(
        prompt_tokens=max(left.prompt_tokens, right.prompt_tokens),
        host_ram_mb=max(left.host_ram_mb, right.host_ram_mb),
        process_slots=max(left.process_slots, right.process_slots),
        connection_slots=max(left.connection_slots, right.connection_slots),
    )


PROCESS_RUNTIME_OVERHEAD_MB = 16.0
PHYSICAL_HOST_BUDGET_MB = 24.0


def run_benchmark() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="orion-v9r2-mixed-") as tmp:
        root = Path(tmp)
        catalog, plugin, assets = build_mixed_inventory(
            state_root=root / "scientific-state",
            process_runtime_overhead_mb=PROCESS_RUNTIME_OVERHEAD_MB,
        )
        discovery = CascadedDiscoveryBackend(
            (
                ExactTriggerDiscoveryBackend(),
                MetadataDiscoveryBackend(),
                EmbeddingDiscoveryBackend(
                    encoder=_encoder,
                    minimum_similarity=0.68,
                    # R1 no-match failure had only one weak sparse concept signal ("engine").
                    # This is an encoder-specific development confidence floor, not a universal
                    # threshold for normalized dense embeddings.
                    minimum_query_vector_norm=1.5,
                ),
            )
        )
        router = EvolvableCapabilityRouter(
            catalog=catalog,
            discovery=discovery,
            control=CapabilityControlPlane(),
        )
        # R1 exposed that a 3 MB worker payload is not the same quantity as full child RSS.
        # On this runtime the interpreter baseline is materially larger than the payload, so R2
        # declares an explicit process runtime envelope and verifies it against live telemetry.
        budget = ResourceVector(
            prompt_tokens=256.0,
            host_ram_mb=PHYSICAL_HOST_BUDGET_MB,
            process_slots=1.0,
            connection_slots=1.0,
        )
        governor = CapabilityResidencyGovernor(catalog=catalog, budget=budget)
        runtime = RoutedCapabilityRuntime(
            router=router,
            governor=governor,
            reconcile_measured=True,
        )

        single_rows: list[dict[str, Any]] = []
        composition_rows: list[dict[str, Any]] = []
        peak_measured = ResourceVector()
        post_reconcile_violation_count = 0
        try:
            for task in _round_robin_single_tasks():
                with runtime.acquire(
                    _request(task.text, authorities=task.authorities),
                    request_id=f"r2-{task.task_id}",
                ) as lease:
                    measured = governor.total_measured_cost()
                    peak_measured = _max_vector(peak_measured, measured)
                    violations = governor.measured_budget_violations()
                    post_reconcile_violation_count += int(bool(violations))

                    selected = lease.materialized_capability_id
                    recovery = lease.recovery_action
                    success = (
                        selected == task.expected_capability_id
                        if task.expected_capability_id is not None
                        else selected is None and recovery == task.expected_recovery
                    )
                    execution_ok = True
                    detail: dict[str, Any] = {}
                    if selected == SCIENTIFIC_ID and task.scientific_case_id is not None:
                        observed = plugin.invoke({"case_id": task.scientific_case_id})
                        execution_ok = observed["status"] == task.expected_scientific_status
                        detail = {
                            "scientific_status": observed["status"],
                            "policy_mode": observed["policy_mode"],
                        }
                    elif selected == PROCESS_ID:
                        execution_ok = bool(assets["process_adapter"].is_running)
                        detail = {"process_running": assets["process_adapter"].is_running}
                    elif selected == SESSION_ID:
                        execution_ok = bool(assets["session_adapter"].is_connected)
                        detail = {"session_connected": assets["session_adapter"].is_connected}
                    single_rows.append(
                        {
                            "task_id": task.task_id,
                            "family": task.family,
                            "selected": selected,
                            "expected": task.expected_capability_id,
                            "recovery": recovery.value,
                            "success": bool(success and execution_ok),
                            "model_visible_count": lease.plan.candidate_aperture,
                            "measured_working_set": asdict(measured),
                            "post_reconcile_violations": list(violations),
                            "admission_failures": [asdict(item) for item in lease.admission_failures],
                            **detail,
                        }
                    )

            for task in _composition_tasks():
                with runtime.acquire_composition(
                    _request(task.text),
                    max_capabilities=2,
                    request_id=f"r2-{task.task_id}",
                ) as lease:
                    measured = governor.total_measured_cost()
                    peak_measured = _max_vector(peak_measured, measured)
                    violations = governor.measured_budget_violations()
                    post_reconcile_violation_count += int(bool(violations))
                    selected = frozenset(item.capability_id for item in lease.capabilities)
                    execution_ok = True
                    detail: dict[str, Any] = {}
                    if SCIENTIFIC_ID in selected and task.scientific_case_id is not None:
                        observed = plugin.invoke({"case_id": task.scientific_case_id})
                        execution_ok = observed["status"] == task.expected_scientific_status
                        detail["scientific_status"] = observed["status"]
                    if PROCESS_ID in selected:
                        execution_ok = execution_ok and bool(assets["process_adapter"].is_running)
                        detail["process_running"] = assets["process_adapter"].is_running
                    if SESSION_ID in selected:
                        execution_ok = execution_ok and bool(assets["session_adapter"].is_connected)
                        detail["session_connected"] = assets["session_adapter"].is_connected
                    composition_rows.append(
                        {
                            "task_id": task.task_id,
                            "family": task.family,
                            "selected": sorted(selected),
                            "expected": sorted(task.expected_capability_ids),
                            "success": selected == task.expected_capability_ids and execution_ok,
                            "model_visible_count": lease.plan.routing_plan.candidate_aperture,
                            "measured_working_set": asdict(measured),
                            "post_reconcile_violations": list(violations),
                            **detail,
                        }
                    )

            # Independent lifecycle probe: do not rely on artificial memory pressure to make
            # the scientific plugin disappear. Explicitly page it out, then fault it back in and
            # verify that the promoted learned generation/policy survives the residency transition.
            scientific_state_before_pageout = governor.state(SCIENTIFIC_ID)
            if scientific_state_before_pageout.current_level > catalog.get(SCIENTIFIC_ID).pinned_min_level:
                governor.demote(SCIENTIFIC_ID, catalog.get(SCIENTIFIC_ID).pinned_min_level)
            with runtime.acquire(
                _request("judge evidence applicability under conflicting experimental conditions"),
                request_id="v9r2-final-scientific-refault",
            ) as final_lease:
                final_measured = governor.total_measured_cost()
                peak_measured = _max_vector(peak_measured, final_measured)
                final_violations = governor.measured_budget_violations()
                post_reconcile_violation_count += int(bool(final_violations))
                final_observed = plugin.invoke({"case_id": "temperature_conflict"})

            event_counts: dict[str, int] = {}
            for event in governor.events:
                event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            measured_evictions = sum(
                event.event_type == "demote" and event.detail == "measured_budget_eviction"
                for event in governor.events
            )
            single_success = sum(row["success"] for row in single_rows)
            composition_success = sum(row["success"] for row in composition_rows)
            return {
                "schema_version": "orion-evolvable-plugin-v9r2-mixed-capability-runtime-dev-v1",
                "status": "DEVELOPMENT_REPAIR_ZERO_PROVIDER_NOT_CONFIRMATORY",
                "inventory_size": len(catalog.ids()),
                "single_task_count": len(single_rows),
                "single_success_count": single_success,
                "single_success_rate": single_success / len(single_rows),
                "single_family_summary": _family_summary(single_rows),
                "composition_task_count": len(composition_rows),
                "composition_success_count": composition_success,
                "composition_success_rate": composition_success / len(composition_rows),
                "composition_family_summary": _family_summary(composition_rows),
                "budget": asdict(budget),
                "resource_model": {
                    "process_payload_mb": 3.0,
                    "process_declared_runtime_overhead_mb": PROCESS_RUNTIME_OVERHEAD_MB,
                    "session_payload_mb": 1.0,
                    "physical_host_budget_mb": PHYSICAL_HOST_BUDGET_MB,
                    "measured_telemetry_authoritative": True,
                },
                "peak_measured": asdict(peak_measured),
                "post_reconcile_violation_count": post_reconcile_violation_count,
                "measured_budget_eviction_count": measured_evictions,
                "event_counts": dict(sorted(event_counts.items())),
                "scientific_learned_state": {
                    "explicit_pageout_before_final_probe": True,
                    "strict_update": assets["strict_update"],
                    "final_selected": final_lease.materialized_capability_id,
                    "final_generation": int(plugin.learned_state.generation),
                    "final_policy": str(final_observed["policy_mode"]),
                    "final_status": str(final_observed["status"]),
                    "refaults": governor.state(SCIENTIFIC_ID).refaults,
                },
                "single_rows": single_rows,
                "composition_rows": composition_rows,
                "provider_calls": 0,
                "kg_calls": 0,
                "rag_calls": 0,
                "v8b_confirmatory_queries_read": 0,
                "protected_six_case_pool_consumed": 0,
                "interpretation_boundary": (
                    "V9R2 is a development repair on the same mixed-real bank that exposed the R1 "
                    "failures. Its semantic confidence floor and physical process-runtime envelope "
                    "were introduced after observing R1 and therefore cannot be treated as independent "
                    "confirmation. Live measured telemetry remains authoritative over declared cost."
                ),
            }
        finally:
            governor.close()


def main() -> int:
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
