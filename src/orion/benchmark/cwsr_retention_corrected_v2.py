"""Corrected retention comparison: common admission, actual type-local reclamation.

The dependent objects are really prepared. The endpoint is dependency readiness,
not completion of a scientific business function. Historical v1 results are not
modified. This script writes JSON to stdout only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import tempfile

from orion.benchmark import cwsr_provider_token_replacement_replication_v1 as calibration
from orion.benchmark import hcvm_shared_pool_vs_per_kind_v2 as inventory
from orion.benchmark.hcvm_global_admission_replacement_order_sensitivity_v1 import _orders
from orion.benchmark.evolvable_plugin_mixed_capability_runtime_v9r1_dev import build_mixed_inventory
from orion.benchmark.evolvable_plugin_mixed_capability_runtime_v9r2_dev import PROCESS_RUNTIME_OVERHEAD_MB
from orion.capvm import (
    CapabilityResidencyGovernor, ResidencyLevel, ResidencyBudgetExceeded,
    MeasuredResidencyBudgetExceeded, ResourceVector,
)
from orion.capvm.policies.lru_cost import TypeLocalLRUEvictionPolicy

STRATEGIES = ("type_local", "release_all", "cross_type", "pressure_flush")


class PressureFlushGovernor(CapabilityResidencyGovernor):
    """Retain between requests; clear all idle dependencies only on capacity pressure."""
    def _plan_complete_request(self, required, protected):
        plan = super()._plan_complete_request(required, protected)
        if not plan:
            return plan
        chosen = {item for item, _ in plan}
        for state in self.states():
            cap = self.catalog.get(state.capability_id)
            if (state.capability_id not in protected and state.capability_id not in chosen
                    and not state.active_count and state.current_level > cap.pinned_min_level):
                plan.append((state.capability_id, cap.pinned_min_level))
        return plan


def _budget(tokens=54, host=20, processes=1, connections=1):
    return ResourceVector(prompt_tokens=tokens, host_ram_mb=host,
                          process_slots=processes, connection_slots=connections)


def run_case(strategy, phases, budget):
    with tempfile.TemporaryDirectory(prefix="cwsr-corrected-v2-") as tmp:
        catalog, _, _ = build_mixed_inventory(
            state_root=Path(tmp), process_runtime_overhead_mb=PROCESS_RUNTIME_OVERHEAD_MB,
        )
        policy = TypeLocalLRUEvictionPolicy() if strategy == "type_local" else None
        cls = PressureFlushGovernor if strategy == "pressure_flush" else CapabilityResidencyGovernor
        g = cls(catalog=catalog, budget=budget, policy=policy)
        flags, failures, peak = [], [], ResourceVector()
        try:
            for index, phase in enumerate(phases):
                try:
                    with g.acquire_many(inventory._phase_requests(catalog, phase["capability_ids"]),
                                        request_id=str(phase["phase_id"]), reconcile_before_active=True):
                        measured = g.total_measured_cost()
                        if not measured.fits(budget):
                            raise AssertionError("a prepared request exceeds the measured budget")
                        peak = ResourceVector(**{
                            name: max(getattr(peak, name), getattr(measured, name))
                            for name in asdict(peak)
                        })
                    flags.append("1")
                    if strategy == "release_all":
                        for state in g.states():
                            cap = catalog.get(state.capability_id)
                            if state.current_level > cap.pinned_min_level and not state.active_count:
                                g.demote(state.capability_id, cap.pinned_min_level)
                except (ResidencyBudgetExceeded, MeasuredResidencyBudgetExceeded) as exc:
                    flags.append("0")
                    failures.append([index, type(exc).__name__, list(exc.violated_dimensions)])
            repeated, initial = Counter(), Counter()
            for state in g.states():
                kind = catalog.get(state.capability_id).kind.value
                if state.promotions:
                    initial[kind] += 1
                    repeated[kind] += max(0, state.promotions - 1)
            return {
                "ready": flags.count("1"), "requested": len(flags),
                "ready_flags": "".join(flags),
                "repreparations": sum(repeated.values()),
                "repreparations_by_kind": dict(sorted(repeated.items())),
                "initial_preparations": sum(initial.values()),
                "failures": failures, "peak_measured": asdict(peak),
            }
        finally:
            g.close()


def run_suite(suite):
    old, budget_old = calibration._patch_token_estimator()
    try:
        result = {
            "schema_version": "cwsr-retention-corrected-v2",
            "suite": suite,
            "endpoint": "all requested dependencies prepared and measured-resource check passed",
            "counter": "successful preparations per dependency after its first preparation",
            "source_sha256": {
                rel: hashlib.sha256((Path(__file__).resolve().parents[1] / rel).read_bytes()).hexdigest()
                for rel in ("capvm/governor.py", "capvm/policies/lru_cost.py",
                            "benchmark/cwsr_retention_corrected_v2.py")
            },
            "strategies": list(STRATEGIES),
        }
        phases = inventory._phases()
        if suite == "main":
            result["budget"] = asdict(_budget())
            result["repetitions"] = 3
            result["cases"] = {
                strategy: [run_case(strategy, phases, _budget()) for _ in range(3)]
                for strategy in STRATEGIES
            }
        elif suite == "orders":
            rows = []
            for name, ordered in _orders():
                row = {"order": name}
                for strategy in STRATEGIES:
                    case = run_case(strategy, ordered, _budget())
                    row[strategy] = [case["ready"], case["repreparations"]]
                rows.append(row)
            result["columns_per_strategy"] = ["ready", "repreparations"]
            result["cases"] = rows
            result["summary"] = {
                strategy: {
                    "ready_range": [min(r[strategy][0] for r in rows), max(r[strategy][0] for r in rows)],
                    "repreparations_range": [min(r[strategy][1] for r in rows), max(r[strategy][1] for r in rows)],
                    "repreparations_median": statistics.median(r[strategy][1] for r in rows),
                } for strategy in STRATEGIES
            }
        elif suite in {"capacity", "tokens", "relaxed"}:
            if suite == "capacity":
                conditions = list(itertools.product((54, 80), (20, 24, 32), (1, 2), (1, 2)))
                repeats = 3
            elif suite == "tokens":
                conditions = [(value, 20, 1, 1) for value in range(31, 60)]
                repeats = 1
            else:
                conditions, repeats = [(100000, 20, 1, 1)], 3
            result["columns_per_strategy"] = ["ready", "repreparations"]
            result["budget_columns"] = ["tokens", "host_MiB", "processes", "connections"]
            result["cases"] = []
            for condition in conditions:
                row = {"budget": list(condition)}
                for strategy in STRATEGIES:
                    cases = [run_case(strategy, phases, _budget(*condition)) for _ in range(repeats)]
                    row[strategy] = [[c["ready"], c["repreparations"]] for c in cases]
                result["cases"].append(row)
        else:
            raise ValueError(suite)
        return result
    finally:
        calibration._restore(old, budget_old)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=("main", "orders", "capacity", "tokens", "relaxed"))
    args = parser.parse_args()
    print(json.dumps(run_suite(args.suite), ensure_ascii=False, separators=(",", ":")))
