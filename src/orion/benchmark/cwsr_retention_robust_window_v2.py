"""带单请求余量预检的 CWSR 机制实验修订版。

20 MiB 旧实验保留为历史工件。本入口在实测 Process WorkingSet 与声明成本
造成的竞争区间内选用 21.5 MiB，并在比较策略前检查全部单请求可执行且有余量。
预检失败会写出原始结果和失败状态，不会伪装为策略效果。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from orion.benchmark import cwsr_provider_token_replacement_replication_v1 as calibration
from orion.benchmark import cwsr_retention_corrected_v2 as frozen
from orion.benchmark import cwsr_runtime_only_retention_v1 as runtime_only
from orion.benchmark.hcvm_global_admission_replacement_order_sensitivity_v1 import _orders


HOST_BUDGET_MIB = 21.5
PREFLIGHT_MIN_MARGIN_MIB = 0.75
SOURCE_FILES = (
    "benchmark/cwsr_retention_robust_window_v2.py",
    "benchmark/cwsr_runtime_only_retention_v1.py",
    "benchmark/cwsr_retention_corrected_v2.py",
    "benchmark/cwsr_provider_token_replacement_replication_v1.py",
    "benchmark/hcvm_shared_pool_vs_per_kind_v2.py",
    "benchmark/hcvm_global_admission_replacement_order_sensitivity_v1.py",
    "benchmark/evolvable_plugin_mixed_capability_runtime_v9r1_dev.py",
    "benchmark/evolvable_plugin_mixed_capability_runtime_v9r2_dev.py",
    "capvm/adapters/process.py",
    "capvm/telemetry.py",
    "capvm/governor.py",
    "capvm/policies/lru_cost.py",
)


def _budget(host_mib: float):
    return runtime_only._budget(runtime_only.RELAXED_TOKENS, host_mib, 1, 1)


def _runtime_repreparations(case: dict) -> int:
    return runtime_only._runtime_only(case["repreparations_by_kind"])


def _run(suite: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    whoami = subprocess.run(
        ["whoami"], capture_output=True, text=True, check=False
    ).stdout.strip()
    result: dict = {
        "schema_version": "cwsr-retention-robust-window-v2",
        "suite": suite,
        "design_note": (
            "21.5 MiB is a post-review revised budget. The historical 20 MiB results "
            "are retained separately; this run does not retroactively replace them."
        ),
        "environment": {
            "whoami": whoami,
            "python_executable": sys.executable,
            "python_base_executable": getattr(sys, "_base_executable", ""),
            "python_version": sys.version,
            "platform": platform.platform(),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "source_sha256": {
            rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
            for rel in SOURCE_FILES
        },
        "budget": asdict(_budget(HOST_BUDGET_MIB)),
        "preflight_min_margin_mib": PREFLIGHT_MIN_MARGIN_MIB,
        "request_selection": "managed_resource_requests_only",
    }
    old, budget_old = calibration._patch_token_estimator()
    try:
        phases = runtime_only._select_phases(frozen.inventory._phases(), True)
        budget = _budget(HOST_BUDGET_MIB)
        result["request_ids"] = [phase["phase_id"] for phase in phases]
        singleton = {
            phase["phase_id"]: frozen.run_case("release_all", (phase,), budget)
            for phase in phases
        }
        result["preflight_singleton"] = singleton
        peak = max(case["peak_measured"]["host_ram_mb"] for case in singleton.values())
        margin = HOST_BUDGET_MIB - peak
        result["preflight_peak_host_ram_mib"] = peak
        result["preflight_margin_mib"] = margin
        if not all(case["ready"] == 1 for case in singleton.values()):
            result["status"] = "PREFLIGHT_FAILED_REQUEST"
            return result
        if margin < PREFLIGHT_MIN_MARGIN_MIB:
            result["status"] = "PREFLIGHT_FAILED_MARGIN"
            return result

        if suite in {"main", "all"}:
            result["main"] = {
                strategy: [frozen.run_case(strategy, phases, budget) for _ in range(3)]
                for strategy in frozen.STRATEGIES
            }
            result["main_summary"] = {
                strategy: [
                    [case["ready"], _runtime_repreparations(case)]
                    for case in result["main"][strategy]
                ]
                for strategy in frozen.STRATEGIES
            }
        if suite in {"orders", "all"}:
            result["orders"] = [
                {
                    "order": name,
                    "cases": {
                        strategy: frozen.run_case(
                            strategy,
                            runtime_only._select_phases(ordered, True),
                            budget,
                        )
                        for strategy in frozen.STRATEGIES
                    },
                }
                for name, ordered in _orders()
            ]
            result["orders_summary"] = {
                strategy: {
                    "ready_range": [
                        min(row["cases"][strategy]["ready"] for row in result["orders"]),
                        max(row["cases"][strategy]["ready"] for row in result["orders"]),
                    ],
                    "runtime_repreparations_median": statistics.median(
                        _runtime_repreparations(row["cases"][strategy])
                        for row in result["orders"]
                    ),
                }
                for strategy in frozen.STRATEGIES
            }
        if suite in {"capacity", "all"}:
            result["capacity"] = [
                {
                    "budget": asdict(runtime_only._budget(runtime_only.RELAXED_TOKENS, host, slots, connections)),
                    "cases": {
                        strategy: frozen.run_case(
                            strategy,
                            phases,
                            runtime_only._budget(runtime_only.RELAXED_TOKENS, host, slots, connections),
                        )
                        for strategy in frozen.STRATEGIES
                    },
                }
                for host in (HOST_BUDGET_MIB, 24.0, 32.0)
                for slots in (1, 2)
                for connections in (1, 2)
            ]
        result["status"] = "PASS"
        return result
    finally:
        calibration._restore(old, budget_old)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("main", "orders", "capacity", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = _run(args.suite)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": payload["status"], "preflight_margin_mib": payload["preflight_margin_mib"]}))
    if payload["status"] != "PASS":
        raise SystemExit(2)
