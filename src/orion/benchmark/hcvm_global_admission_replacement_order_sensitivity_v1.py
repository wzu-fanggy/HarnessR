"""Order-sensitivity check for the strong global-admission replacement baseline.

The main replacement benchmark uses one frozen grouped phase order.  This audit
checks that its qualitative conclusion is not an artifact of that single order.
It evaluates original, reverse, repeat-interleaved, and deterministic shuffled
orders while keeping the same 32 feasible phases, backing implementations, and
residency-capacity budget.
"""

from __future__ import annotations

import json
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any

import orion.benchmark.hcvm_global_admission_replacement_v1 as rep
import orion.benchmark.hcvm_shared_pool_vs_per_kind_v2 as v2
from orion.benchmark.evolvable_plugin_mixed_capability_runtime_v9r1_dev import (
    build_mixed_inventory,
)
from orion.benchmark.evolvable_plugin_mixed_capability_runtime_v9r2_dev import (
    PROCESS_RUNTIME_OVERHEAD_MB,
)


RANDOM_SEEDS = tuple(range(24))


def _interleaved(phases: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    # Frozen workload has four repeats for each of eight families.  Interleave
    # families by repeat index instead of grouping all repeats of one family.
    by_repeat: list[dict[str, Any]] = []
    for repeat in range(4):
        suffix = f"-{repeat:02d}"
        by_repeat.extend(phase for phase in phases if str(phase["phase_id"]).endswith(suffix))
    if len(by_repeat) != len(phases):
        raise AssertionError("unexpected frozen phase shape")
    return tuple(by_repeat)


def _orders() -> tuple[tuple[str, tuple[dict[str, Any], ...]], ...]:
    base = v2._phases()
    rows: list[tuple[str, tuple[dict[str, Any], ...]]] = [
        ("grouped_original", base),
        ("reverse", tuple(reversed(base))),
        ("repeat_interleaved", _interleaved(base)),
    ]
    for seed in RANDOM_SEEDS:
        items = list(base)
        random.Random(seed).shuffle(items)
        rows.append((f"shuffle_{seed:02d}", tuple(items)))
    return tuple(rows)


def _fresh_catalogs(root: Path):
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
    return (
        v2._add_ram_probe_plugin(shared_base, state_root=root / "shared-probe"),
        v2._add_ram_probe_plugin(keep_base, state_root=root / "keep-probe"),
        v2._add_ram_probe_plugin(release_base, state_root=root / "release-probe"),
    )


def run_benchmark() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for order_index, (order_name, phases) in enumerate(_orders()):
        with tempfile.TemporaryDirectory(prefix=f"orion-hcvm-order-{order_index:02d}-") as tmp:
            shared_catalog, keep_catalog, release_catalog = _fresh_catalogs(Path(tmp))
            shared = rep._execute_hcvm_shared(shared_catalog, phases)
            keep = rep._execute_global_admission_per_kind(
                keep_catalog,
                phases,
                release_all_idle_before_phase=False,
            )
            release = rep._execute_global_admission_per_kind(
                release_catalog,
                phases,
                release_all_idle_before_phase=True,
            )
        results.append(
            {
                "order": order_name,
                "hcvm_success": shared["success_count"],
                "hcvm_refaults": shared["event_counts"].get("refault", 0),
                "hcvm_unsafe_rejection_rate": shared["unsafe_rejection_rate"],
                "keep_cache_success": keep["success_count"],
                "keep_cache_unsafe_rejection_rate": keep["unsafe_rejection_rate"],
                "release_all_success": release["success_count"],
                "release_all_refaults": release["event_counts"].get("refault", 0),
                "release_all_unsafe_rejection_rate": release["unsafe_rejection_rate"],
                "release_minus_hcvm_refaults": (
                    release["event_counts"].get("refault", 0)
                    - shared["event_counts"].get("refault", 0)
                ),
            }
        )

    keep_successes = [int(row["keep_cache_success"]) for row in results]
    refault_deltas = [int(row["release_minus_hcvm_refaults"]) for row in results]
    return {
        "schema_version": "hcvm-global-admission-replacement-order-sensitivity-v1.0",
        "status": "DEVELOPMENT_ORDER_SENSITIVITY",
        "order_count": len(results),
        "phase_count_per_order": len(v2._phases()),
        "summary": {
            "hcvm_all_orders_complete": all(row["hcvm_success"] == 32 for row in results),
            "release_all_all_orders_complete": all(
                row["release_all_success"] == 32 for row in results
            ),
            "all_arms_safe_on_all_orders": all(
                row["hcvm_unsafe_rejection_rate"] == 1.0
                and row["keep_cache_unsafe_rejection_rate"] == 1.0
                and row["release_all_unsafe_rejection_rate"] == 1.0
                for row in results
            ),
            "keep_cache_blocked_in_all_orders": all(value < 32 for value in keep_successes),
            "keep_cache_success_min": min(keep_successes),
            "keep_cache_success_median": statistics.median(keep_successes),
            "keep_cache_success_max": max(keep_successes),
            "release_all_more_refaults_in_all_orders": all(delta > 0 for delta in refault_deltas),
            "release_minus_hcvm_refaults_min": min(refault_deltas),
            "release_minus_hcvm_refaults_median": statistics.median(refault_deltas),
            "release_minus_hcvm_refaults_max": max(refault_deltas),
        },
        "orders": results,
        "interpretation_guard": (
            "This sensitivity audit does not establish a universal optimal replacement policy. "
            "It tests whether the qualitative trade-off in the frozen 32-phase workload depends "
            "on one hand-picked phase order. Global admission is granted to all compared arms."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2, sort_keys=True))
