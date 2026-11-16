"""Token-calibrated replication of the CWSR replacement benchmarks.

The original controlled benchmark used a deterministic character-based capability-
text cost. This replication replaces that text dimension with provider-grounded
DeepSeek token counts measured by ``cwsr_deepseek_token_calibration_v1.py`` while
leaving RAM/process/session measurements and the frozen phase sequences unchanged.

The text budget is converted deterministically from the frozen experiment by
preserving the original budget-to-pressure-composition ratio:
    round(80 * 60 / 89) = 54 provider tokens.
Here 89 is the old estimated cost and 60 the measured DeepSeek incremental input-
token cost of the same plugin+tool+design pressure composition.

The script prints JSON to stdout and does not write result files itself.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable

import orion.benchmark.hcvm_global_admission_replacement_v1 as rep
import orion.benchmark.hcvm_global_admission_replacement_order_sensitivity_v1 as order
import orion.benchmark.hcvm_shared_pool_vs_per_kind_v2 as v2
import orion.capvm.adapters.base as base_adapter
import orion.capvm.adapters.plugin as plugin_adapter
import orion.capvm.adapters.skill as skill_adapter
import orion.capvm.adapters.tool as tool_adapter
import orion.capvm.governor as governor_module
from orion.capvm.model import ResourceVector


# Exact block-level provider counts from the paired DeepSeek V4 calibration.
TOKEN_BY_TEXT: dict[str, int] = {
    "ORION evidence composition capability": 6,
    "Qualify locally grounded evidence under a declared joint scientific context.": 12,
    "search_experimental_evidence(query) -> grounded evidence records": 13,
    "electrolyte design workflow skill": 7,
    "Define objective and constraints; retrieve evidence gaps; propose candidates; evaluate trade-offs; record uncertainty and exit conditions.": 22,
    "scientific report drafting skill": 6,
    "Draft structured scientific report content without performing file writes.": 11,
    "ram_probe()": 4,
    "ram probe": 3,
}

ORIGINAL_ESTIMATE_PRESSURE_COST = 89
PROVIDER_PRESSURE_COST = 60
ORIGINAL_TEXT_BUDGET = 80
TOKEN_BUDGET = round(
    ORIGINAL_TEXT_BUDGET * PROVIDER_PRESSURE_COST / ORIGINAL_ESTIMATE_PRESSURE_COST
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _provider_token_count(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    try:
        return TOKEN_BY_TEXT[normalized]
    except KeyError as exc:
        raise KeyError(
            "unmapped model-facing capability representation in token-calibrated replication: "
            f"sha256={_sha256(normalized)} chars={len(normalized)}"
        ) from exc


def _patch_token_estimator() -> tuple[dict[str, Callable], ResourceVector]:
    old_functions: dict[str, Callable] = {
        "base": base_adapter.estimate_prompt_tokens,
        "plugin": plugin_adapter.estimate_prompt_tokens,
        "skill": skill_adapter.estimate_prompt_tokens,
        "tool": tool_adapter.estimate_prompt_tokens,
        "governor": governor_module.estimate_prompt_tokens,
    }
    base_adapter.estimate_prompt_tokens = _provider_token_count
    plugin_adapter.estimate_prompt_tokens = _provider_token_count
    skill_adapter.estimate_prompt_tokens = _provider_token_count
    tool_adapter.estimate_prompt_tokens = _provider_token_count
    governor_module.estimate_prompt_tokens = _provider_token_count

    old_budget = v2.GLOBAL_BUDGET
    v2.GLOBAL_BUDGET = ResourceVector(
        prompt_tokens=float(TOKEN_BUDGET),
        host_ram_mb=old_budget.host_ram_mb,
        gpu_vram_mb=old_budget.gpu_vram_mb,
        process_slots=old_budget.process_slots,
        connection_slots=old_budget.connection_slots,
    )
    return old_functions, old_budget


def _restore(old_functions: dict[str, Callable], old_budget: ResourceVector) -> None:
    base_adapter.estimate_prompt_tokens = old_functions["base"]
    plugin_adapter.estimate_prompt_tokens = old_functions["plugin"]
    skill_adapter.estimate_prompt_tokens = old_functions["skill"]
    tool_adapter.estimate_prompt_tokens = old_functions["tool"]
    governor_module.estimate_prompt_tokens = old_functions["governor"]
    v2.GLOBAL_BUDGET = old_budget


def main() -> None:
    old_functions, old_budget = _patch_token_estimator()
    try:
        main_results = [rep.run_benchmark() for _ in range(3)]
        main_result = main_results[0]
        order_result = order.run_benchmark()
    finally:
        _restore(old_functions, old_budget)

    arms = main_result["arms"]
    output = {
        "schema_version": "cwsr-provider-token-replacement-replication-v1",
        "status": "PASS",
        "calibration": {
            "source": "paired DeepSeek V4 provider usage.prompt_tokens calibration",
            "flash_and_pro_counts_identical": True,
            "original_text_budget_estimate_units": ORIGINAL_TEXT_BUDGET,
            "original_pressure_composition_estimate_units": ORIGINAL_ESTIMATE_PRESSURE_COST,
            "provider_pressure_composition_tokens": PROVIDER_PRESSURE_COST,
            "budget_conversion_rule": "round(80 * 60 / 89)",
            "provider_token_budget": TOKEN_BUDGET,
            "token_by_text_sha256": {
                _sha256(text): count for text, count in TOKEN_BY_TEXT.items()
            },
        },
        "global_residency_capacity": main_result["global_residency_capacity"],
        "main_32_phase": {
            "hcvm_success": arms["HCVM_SHARED"]["success_count"],
            "hcvm_refaults": arms["HCVM_SHARED"]["event_counts"].get("refault", 0),
            "keep_cache_success": arms[
                "GLOBAL_ADMISSION_KEEP_CACHE_NO_CROSS_KIND_REPLACEMENT"
            ]["success_count"],
            "release_all_success": arms["GLOBAL_ADMISSION_RELEASE_ALL_IDLE"][
                "success_count"
            ],
            "release_all_refaults": arms["GLOBAL_ADMISSION_RELEASE_ALL_IDLE"][
                "event_counts"
            ].get("refault", 0),
            "all_arms_unsafe_rejection_rate": {
                "hcvm": arms["HCVM_SHARED"]["unsafe_rejection_rate"],
                "keep_cache": arms[
                    "GLOBAL_ADMISSION_KEEP_CACHE_NO_CROSS_KIND_REPLACEMENT"
                ]["unsafe_rejection_rate"],
                "release_all": arms["GLOBAL_ADMISSION_RELEASE_ALL_IDLE"][
                    "unsafe_rejection_rate"
                ],
            },
            "closure_gate": main_result["closure_gate"],
        },
        "repeatability": {
            "repetitions": len(main_results),
            "hcvm_success_counts": [
                item["arms"]["HCVM_SHARED"]["success_count"] for item in main_results
            ],
            "hcvm_refault_counts": [
                item["arms"]["HCVM_SHARED"]["event_counts"].get("refault", 0)
                for item in main_results
            ],
            "keep_cache_success_counts": [
                item["arms"]["GLOBAL_ADMISSION_KEEP_CACHE_NO_CROSS_KIND_REPLACEMENT"][
                    "success_count"
                ]
                for item in main_results
            ],
            "release_all_success_counts": [
                item["arms"]["GLOBAL_ADMISSION_RELEASE_ALL_IDLE"]["success_count"]
                for item in main_results
            ],
            "release_all_refault_counts": [
                item["arms"]["GLOBAL_ADMISSION_RELEASE_ALL_IDLE"]["event_counts"].get(
                    "refault", 0
                )
                for item in main_results
            ],
            "identical_headline_results": len(
                {
                    (
                        item["arms"]["HCVM_SHARED"]["success_count"],
                        item["arms"]["HCVM_SHARED"]["event_counts"].get("refault", 0),
                        item["arms"]["GLOBAL_ADMISSION_KEEP_CACHE_NO_CROSS_KIND_REPLACEMENT"][
                            "success_count"
                        ],
                        item["arms"]["GLOBAL_ADMISSION_RELEASE_ALL_IDLE"]["success_count"],
                        item["arms"]["GLOBAL_ADMISSION_RELEASE_ALL_IDLE"]["event_counts"].get(
                            "refault", 0
                        ),
                    )
                    for item in main_results
                }
            )
            == 1,
        },
        "order_sensitivity_27": order_result["summary"],
        "representative_orders": [
            row
            for row in order_result["orders"]
            if row["order"] in {"grouped_original", "reverse", "repeat_interleaved", "shuffle_06"}
        ],
        "interpretation_guard": (
            "This is a unit-calibrated replication of the frozen mechanism experiment. "
            "It replaces only the model-facing capability-representation cost and budget with DeepSeek provider "
            "token units; workload phases, RAM/process/session measurements, replacement "
            "mechanisms and comparison arms remain unchanged."
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
