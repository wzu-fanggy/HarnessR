from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT.parents[1] / 'results/fixed_calls/raw'

PRIMARY_BUDGET = 750
PROFILED_BGE_HOST_MIB = 558.13671875
PROFILED_PROCESS_HOST_MIB = 383.4609375

PRIMARY_FILES = {
    1: {
        "type_local": "cwsr_cross_kind_profiled_type_local_750.json",
        "cwsr": "cwsr_cross_kind_profiled_cwsr_750.json",
        "release_all": "cwsr_cross_kind_profiled_release_all_750_clean.json",
    },
    2: {
        "type_local": "cwsr_cross_kind_profiled_type_local_750_r2_clean.json",
        "cwsr": "cwsr_cross_kind_profiled_cwsr_750_r2_clean.json",
        "release_all": "cwsr_cross_kind_profiled_release_all_750_r2_clean.json",
    },
    3: {
        "type_local": "cwsr_cross_kind_profiled_type_local_750_r3_clean.json",
        "cwsr": "cwsr_cross_kind_profiled_cwsr_750_r3_clean.json",
        "release_all": "cwsr_cross_kind_profiled_release_all_750_r3_clean.json",
    },
}

SENSITIVITY_FILES = {
    500: {
        "type_local": "cwsr_cross_kind_profiled_type_local_500.json",
        "cwsr": "cwsr_cross_kind_profiled_cwsr_500.json",
        "release_all": "cwsr_cross_kind_profiled_release_all_500.json",
    },
    654: {
        "type_local": "cwsr_cross_kind_profiled_type_local_654_clean.json",
        "cwsr": "cwsr_cross_kind_profiled_cwsr_654_clean.json",
        "release_all": "cwsr_cross_kind_profiled_release_all_654_clean.json",
    },
    846: {
        "type_local": "cwsr_cross_kind_profiled_type_local_846_clean.json",
        "cwsr": "cwsr_cross_kind_profiled_cwsr_846_clean.json",
        "release_all": "cwsr_cross_kind_profiled_release_all_846_clean.json",
    },
    1050: {
        "type_local": "cwsr_cross_kind_profiled_type_local_1050_clean.json",
        "cwsr": "cwsr_cross_kind_profiled_cwsr_1050_clean.json",
        "release_all": "cwsr_cross_kind_profiled_release_all_1050_clean.json",
    },
}


def load(name: str) -> dict[str, Any]:
    return json.loads((SCRATCH / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def run() -> dict[str, Any]:
    primary = []
    for repeat, mapping in PRIMARY_FILES.items():
        row: dict[str, Any] = {"repeat": repeat}
        for strategy, filename in mapping.items():
            row[strategy] = load(filename)
        row["cwsr_release_outputs_equal"] = (
            row["cwsr"]["result_signature"]
            == row["release_all"]["result_signature"]
        )
        primary.append(row)

    sensitivity = []
    for budget, mapping in SENSITIVITY_FILES.items():
        row = {"host_budget_mib": budget}
        for strategy, filename in mapping.items():
            row[strategy] = load(filename)
        row["cwsr_release_outputs_equal"] = (
            row["cwsr"]["result_signature"]
            == row["release_all"]["result_signature"]
        )
        sensitivity.append(row)

    checks = {
        "primary_type_local_stops_before_process_start_all_repeats": all(
            not r["type_local"]["complete"]
            and r["type_local"]["completed"] == 2
            and r["type_local"]["process_promotions"] == 0
            and r["type_local"]["failure"]["error_type"] == "ResidencyBudgetExceeded"
            for r in primary
        ),
        "primary_cwsr_complete_all_repeats": all(
            r["cwsr"]["complete"] and r["cwsr"]["completed"] == 6
            for r in primary
        ),
        "primary_release_all_complete_all_repeats": all(
            r["release_all"]["complete"]
            and r["release_all"]["completed"] == 6
            for r in primary
        ),
        "primary_cwsr_pre_admission_cross_kind_eviction_all_repeats": all(
            [e["capability_id"] for e in r["cwsr"]["budget_evictions"]]
            == ["model:bge-m3-real", "process:na-conductivity-real"]
            and all(
                e["detail"] == "budget_eviction"
                for e in r["cwsr"]["budget_evictions"]
            )
            for r in primary
        ),
        "primary_cwsr_fewer_preparations_all_repeats": all(
            r["cwsr"]["model_load_count"] == 2
            and r["cwsr"]["process_promotions"] == 1
            and r["release_all"]["model_load_count"] == 4
            and r["release_all"]["process_promotions"] == 2
            for r in primary
        ),
        "primary_scientific_outputs_equal_all_repeats": all(
            r["cwsr_release_outputs_equal"] for r in primary
        ),
        "low_budget_no_strategy_can_admit_single_bge": all(
            not sensitivity[0][s]["complete"]
            and sensitivity[0][s]["completed"] == 0
            and sensitivity[0][s]["model_load_count"] == 0
            for s in ("type_local", "cwsr", "release_all")
        ),
        "intermediate_budget_cross_kind_difference": all(
            not r["type_local"]["complete"]
            and r["type_local"]["completed"] == 2
            and r["type_local"]["process_promotions"] == 0
            and r["cwsr"]["complete"]
            and r["release_all"]["complete"]
            and r["cwsr_release_outputs_equal"]
            for r in sensitivity
            if r["host_budget_mib"] in (654, 846)
        ),
        "high_budget_convergence_without_cwsr_eviction": (
            all(
                sensitivity[-1][s]["complete"]
                and sensitivity[-1][s]["completed"] == 6
                for s in ("type_local", "cwsr", "release_all")
            )
            and sensitivity[-1]["type_local"]["model_load_count"] == 1
            and sensitivity[-1]["type_local"]["process_promotions"] == 1
            and sensitivity[-1]["cwsr"]["model_load_count"] == 1
            and sensitivity[-1]["cwsr"]["process_promotions"] == 1
            and sensitivity[-1]["cwsr"]["budget_evictions"] == []
        ),
    }

    source_files = []
    for mapping in list(PRIMARY_FILES.values()) + list(SENSITIVITY_FILES.values()):
        for filename in mapping.values():
            path = SCRATCH / filename
            source_files.append(
                {
                    "path": str(path.relative_to(ROOT.parents[1])),
                    "sha256": sha256(path),
                }
            )

    return {
        "schema_version": "cwsr-cross-kind-real-pressure-formal-v2.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "design": {
            "objects": [
                "real BGE-M3 Model",
                "real sodium-electrolyte conductivity Process",
            ],
            "operation_sequence": [
                "BGE",
                "BGE",
                "Process",
                "Process",
                "BGE",
                "BGE",
            ],
            "profiled_cost_source": (
                "independent measurements completed before this pressure comparison"
            ),
            "profiled_bge_host_mib": PROFILED_BGE_HOST_MIB,
            "profiled_process_host_mib": PROFILED_PROCESS_HOST_MIB,
            "admission_rule": (
                "profiled costs are used before materialization; measured telemetry "
                "is checked after materialization"
            ),
            "primary_host_budget_mib": PRIMARY_BUDGET,
            "primary_repeats": 3,
            "sensitivity_budgets_mib": [500, 654, 846, 1050],
            "timing_role": "diagnostic only; not a primary endpoint",
            "scope": (
                "prospective controlled physical-resource validation; not evidence "
                "that this exact mixed sequence occurred in archived agent traces"
            ),
            "neo4j_included": False,
        },
        "primary_summary": {
            "type_local_completed": [
                r["type_local"]["completed"] for r in primary
            ],
            "cwsr_completed": [r["cwsr"]["completed"] for r in primary],
            "release_all_completed": [
                r["release_all"]["completed"] for r in primary
            ],
            "cwsr_model_load_count": [
                r["cwsr"]["model_load_count"] for r in primary
            ],
            "release_all_model_load_count": [
                r["release_all"]["model_load_count"] for r in primary
            ],
            "cwsr_process_establishment_count": [
                r["cwsr"]["process_promotions"] for r in primary
            ],
            "release_all_process_establishment_count": [
                r["release_all"]["process_promotions"] for r in primary
            ],
            "cwsr_elapsed_s_diagnostic": summary(
                [float(r["cwsr"]["elapsed_s"]) for r in primary]
            ),
            "release_all_elapsed_s_diagnostic": summary(
                [float(r["release_all"]["elapsed_s"]) for r in primary]
            ),
        },
        "sensitivity_summary": [
            {
                "host_budget_mib": r["host_budget_mib"],
                "type_local_completed": r["type_local"]["completed"],
                "cwsr_completed": r["cwsr"]["completed"],
                "release_all_completed": r["release_all"]["completed"],
                "cwsr_budget_evictions": r["cwsr"]["budget_evictions"],
                "outputs_equal_cwsr_release_all": r["cwsr_release_outputs_equal"],
            }
            for r in sensitivity
        ],
        "source_files": source_files,
    }


if __name__ == "__main__":
    result = run()
    out = ROOT / "benchmark" / "cwsr_cross_kind_real_pressure_formal_v2_result.json"
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": result["checks"],
                "primary_summary": result["primary_summary"],
                "sensitivity_summary": result["sensitivity_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
