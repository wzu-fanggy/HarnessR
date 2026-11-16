"""Audit complete immutable warm-toolnode A/B experiments; do not rerun tools."""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


TASKS = ("U1", "U2", "P1", "P2", "C1", "C2", "C3", "D1")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(directories: list[Path]) -> dict:
    by_task = defaultdict(list)
    evidence = []
    seen = set()
    for directory in directories:
        plan_path, summary_path = directory / "plan.json", directory / "summary.json"
        plan, summary = read_json(plan_path), read_json(summary_path)
        assert summary["status"] == "PASS", summary_path
        assert summary["scientific_outputs_identical"]
        assert summary["source_files_unchanged"]
        assert summary["metadata_snapshot_fingerprint_unchanged"]
        assert summary["no_evictions"]
        assert not any(
            n for counters in summary["root_resource_counter_deltas"].values()
            for n in counters.values()
        )
        for task in plan["task_order"]:
            assert task in TASKS and task not in seen, f"duplicate or invalid task {task}"
            seen.add(task)
            source = Path(plan["source_files"][task])
            assert source.is_file()
            assert sha256(source.read_bytes()).hexdigest() == plan["source_sha256"][task]
            refs = set()
            for pair in range(plan["pairs"]):
                measured = {}
                for arm in ("native", "runtime"):
                    filename = directory / f"{task}-p{pair}-{arm}.json"
                    row = read_json(filename)
                    assert row["task"] == task and row["pair"] == pair and row["arm"] == arm
                    assert row["rows"]
                    refs.add(row["outputs_sha256"])
                    measured[arm] = row
                native, runtime = measured["native"], measured["runtime"]
                assert [b["tool_names"] for b in native["rows"]] == [
                    b["tool_names"] for b in runtime["rows"]
                ]
                assert [b["output_sha256"] for b in native["rows"]] == [
                    b["output_sha256"] for b in runtime["rows"]
                ]
                actual_percent = 100 * (runtime["elapsed_s"] / native["elapsed_s"] - 1)
                original = next(p for p in summary["paired"] if p["task"] == task and p["pair"] == pair)
                assert abs(actual_percent - original["percent"]) < 0.001
                by_task[task].append({
                    "pair": pair,
                    "native_s": native["elapsed_s"],
                    "runtime_s": runtime["elapsed_s"],
                    "runtime_minus_native_s": runtime["elapsed_s"] - native["elapsed_s"],
                    "relative_percent": round(actual_percent, 3),
                    "runtime_bridge_overhead_s": runtime["overhead_s"],
                    "runtime_tool_s": runtime["tool_s"],
                    "native_tool_s": native["tool_s"],
                    "tool_batch_count": len(runtime["rows"]),
                })
            assert len(refs) == 1, f"science outputs drift across repeated {task} runs"
            evidence.append({
                "task": task, "source": str(source),
                "source_sha256": plan["source_sha256"][task],
                "result_directory": str(directory),
                "all_science_outputs_sha256": next(iter(refs)),
            })
    assert set(by_task) == set(TASKS), f"missing: {set(TASKS)-set(by_task)}"
    assert all(len(by_task[task]) == 2 for task in TASKS)

    pairs = [item for task in TASKS for item in by_task[task]]
    task_medians = {
        task: round(median(x["relative_percent"] for x in by_task[task]), 3)
        for task in TASKS
    }
    native_total = sum(x["native_s"] for x in pairs)
    managed_total = sum(x["runtime_s"] for x in pairs)
    overhead_total = sum(x["runtime_bridge_overhead_s"] for x in pairs)
    return {
        "schema": "cwsr-v17-warm-toolnode-ab-audit-v1",
        "status": "PASS",
        "predeclared_task_set": list(TASKS),
        "cohort": "r1-native traces, 2 AB/BA warm-cache pairs/task",
        "compared_scope": "Production ToolNode stage, no LLM, shared resident BGE/Neo4j",
        "independent_task_families": 8,
        "total_pairs": len(pairs),
        "total_science_equal": True,
        "same_frozen_calls_and_batch_boundaries": True,
        "zero_reloads_or_reclaims": True,
        "task_median_relative_percent": task_medians,
        "median_of_task_medians_relative_percent": round(median(task_medians.values()), 3),
        "pair_median_relative_percent": round(median(x["relative_percent"] for x in pairs), 3),
        "pair_median_delta_s": round(median(x["runtime_minus_native_s"] for x in pairs), 6),
        "relative_percent_min_max": [
            round(min(x["relative_percent"] for x in pairs), 3),
            round(max(x["relative_percent"] for x in pairs), 3),
        ],
        "pairs_over_3_percent": sum(x["relative_percent"] > 3 for x in pairs),
        "pairs_runtime_faster": sum(x["relative_percent"] < 0 for x in pairs),
        "native_total_s": round(native_total, 3),
        "runtime_total_s": round(managed_total, 3),
        "pooled_wall_relative_percent": round(100 * (managed_total / native_total - 1), 3),
        "runtime_bridge_overhead_total_s": round(overhead_total, 3),
        "runtime_bridge_overhead_median_s": round(median(x["runtime_bridge_overhead_s"] for x in pairs), 3),
        "per_task_pairs": dict(by_task),
        "source_evidence": evidence,
        "scientific_scope_caveat": (
            "This is an observational warm-cache fixed workload. It measures the "
            "tool stage, not LLM end-to-end non-inferiority or extra speedup. "
            "Two pairs per task cannot establish a statistical equivalence interval."
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit artifact exists; do not overwrite")
    report = audit([path.resolve() for path in args.input])
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_task_pairs", "source_evidence")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
