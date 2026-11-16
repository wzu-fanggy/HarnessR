"""Runtime-object-only variant of the corrected retention comparison.

Companion to the frozen ``cwsr_retention_corrected_v2.py`` (imported, not
modified).  This module re-runs the same four strategies with the
capability-information token budget non-binding, matching the revised
manuscript scope in which on-demand capability text is not managed by the
runtime resource manager. By default, text-only entries remain in the frozen
request sequence with non-binding capacity. The optional
``--resource-requests-only`` mode excludes those entries from every suite and
reports only the 20 requests that require managed runtime resources.

Re-preparations are additionally broken down by capability kind so that
runtime-object re-initializations (plugin runtime state / process / session)
can be counted separately from text re-provisions (tool / skill), and both
counters are reported.

Suites
------
main
    Budget (100000, 20, 1, 1), four strategies, three repetitions, full
    per-kind breakdown.
orders
    The same 27 frozen phase orders as the order-sensitivity audit, four
    strategies, relaxed token budget; reports per-order and median
    re-preparations under both counters.
capacity
    Host RAM x process slots x connection slots sweep (12 combinations) with
    the token budget non-binding, one repetition per condition.

No model calls are made.  Pure simulation over the same frozen inventory
builders as the frozen script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from orion.benchmark import cwsr_provider_token_replacement_replication_v1 as calibration
from orion.benchmark import cwsr_retention_corrected_v2 as frozen
from orion.benchmark.hcvm_global_admission_replacement_order_sensitivity_v1 import _orders

TEXT_KINDS = frozenset({"tool", "skill"})
RUNTIME_KINDS = frozenset({"plugin", "model", "process", "session"})
RELAXED_TOKENS = 100000.0
TEXT_ONLY_FAMILIES = frozenset({"tool", "skill_design", "skill_draft"})


def _runtime_only(by_kind: dict[str, int]) -> int:
    return int(sum(count for kind, count in by_kind.items() if kind in RUNTIME_KINDS))


def _text_only(by_kind: dict[str, int]) -> int:
    return int(sum(count for kind, count in by_kind.items() if kind in TEXT_KINDS))


def _budget(tokens: float, host: float, processes: int, connections: int):
    return frozen._budget(tokens=tokens, host=host, processes=processes, connections=connections)


def _summarize_case(case: dict) -> dict:
    by_kind = case["repreparations_by_kind"]
    return {
        "ready": case["ready"],
        "requested": case["requested"],
        "repreparations_total": case["repreparations"],
        "repreparations_runtime_only": _runtime_only(by_kind),
        "repreparations_text_only": _text_only(by_kind),
        "repreparations_by_kind": by_kind,
    }


def _source_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
        for rel in (
            "capvm/governor.py",
            "capvm/policies/lru_cost.py",
            "benchmark/cwsr_retention_corrected_v2.py",
            "benchmark/hcvm_shared_pool_vs_per_kind_v2.py",
            "benchmark/hcvm_global_admission_replacement_order_sensitivity_v1.py",
            "benchmark/cwsr_provider_token_replacement_replication_v1.py",
            "benchmark/cwsr_runtime_only_retention_v1.py",
        )
    }


def _select_phases(phases, resource_requests_only: bool):
    if not resource_requests_only:
        return phases
    return tuple(phase for phase in phases if phase["family"] not in TEXT_ONLY_FAMILIES)


def run_suite(suite: str, resource_requests_only: bool = False) -> dict:
    old, budget_old = calibration._patch_token_estimator()
    try:
        result = {
            "schema_version": "cwsr-runtime-only-retention-v1",
            "suite": suite,
            "relaxed_prompt_token_budget": RELAXED_TOKENS,
            "counter_definitions": {
                "repreparations_runtime_only": (
                    "re-preparations of plugin runtime state / model / process / session "
                    "capabilities (the objects the revised system manages)"
                ),
                "repreparations_text_only": (
                    "re-provisions of tool / skill text entries; reported for audit only, "
                    "not counted as re-initialization in the revised manuscript"
                ),
            },
            "source_sha256": _source_sha256(),
            "strategies": list(frozen.STRATEGIES),
            "request_selection": "managed_resource_requests_only" if resource_requests_only else "full_frozen_sequence",
        }
        phases = _select_phases(frozen.inventory._phases(), resource_requests_only)
        result["request_count"] = len(phases)

        if suite == "main":
            result["budget"] = {
                "prompt_tokens": RELAXED_TOKENS,
                "host_ram_mb": 20.0,
                "process_slots": 1,
                "connection_slots": 1,
            }
            result["repetitions"] = 3
            result["cases"] = {
                strategy: [
                    _summarize_case(
                        frozen.run_case(strategy, phases, _budget(RELAXED_TOKENS, 20, 1, 1))
                    )
                    for _ in range(3)
                ]
                for strategy in frozen.STRATEGIES
            }
        elif suite == "orders":
            rows = []
            for name, ordered in _orders():
                ordered = _select_phases(ordered, resource_requests_only)
                row = {"order": name}
                for strategy in frozen.STRATEGIES:
                    case = frozen.run_case(strategy, ordered, _budget(RELAXED_TOKENS, 20, 1, 1))
                    summary = _summarize_case(case)
                    row[strategy] = [
                        summary["ready"],
                        summary["repreparations_runtime_only"],
                        summary["repreparations_total"],
                    ]
                rows.append(row)
            result["columns_per_strategy"] = [
                "ready",
                "repreparations_runtime_only",
                "repreparations_total",
            ]
            result["cases"] = rows
            result["summary"] = {
                strategy: {
                    "ready_range": [
                        min(r[strategy][0] for r in rows),
                        max(r[strategy][0] for r in rows),
                    ],
                    "repreparations_runtime_only_range": [
                        min(r[strategy][1] for r in rows),
                        max(r[strategy][1] for r in rows),
                    ],
                    "repreparations_runtime_only_median": statistics.median(
                        r[strategy][1] for r in rows
                    ),
                    "repreparations_total_median": statistics.median(
                        r[strategy][2] for r in rows
                    ),
                }
                for strategy in frozen.STRATEGIES
            }
        elif suite == "capacity":
            conditions = [
                (host, processes, connections)
                for host in (20.0, 24.0, 32.0)
                for processes in (1, 2)
                for connections in (1, 2)
            ]
            result["budget_columns"] = ["host_MiB", "processes", "connections"]
            result["columns_per_strategy"] = [
                "ready",
                "repreparations_runtime_only",
                "repreparations_total",
            ]
            result["cases"] = []
            for host, processes, connections in conditions:
                row = {"budget": [host, processes, connections]}
                for strategy in frozen.STRATEGIES:
                    case = frozen.run_case(
                        strategy, phases, _budget(RELAXED_TOKENS, host, processes, connections)
                    )
                    summary = _summarize_case(case)
                    row[strategy] = [
                        summary["ready"],
                        summary["repreparations_runtime_only"],
                        summary["repreparations_total"],
                    ]
                result["cases"].append(row)
        else:
            raise ValueError(suite)
        return result
    finally:
        calibration._restore(old, budget_old)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", required=True, choices=("main", "orders", "capacity")
    )
    parser.add_argument("--resource-requests-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_suite(args.suite, args.resource_requests_only), ensure_ascii=False, separators=(",", ":")))
