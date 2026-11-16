"""Runtime-policy replay over frozen sodium-ion electrolyte workflow traces.

This experiment reuses the exact ordered scientific-tool calls already frozen in the 64 Study4C
runs. It changes only the post-call residency policy while keeping the same semantic Harness graph,
capability bindings, proxy resource budget, and archived tool outputs. It therefore measures runtime
feasibility and rematerialization pressure on a real historical scientific access sequence; it is not
a fresh chemistry-quality experiment.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

from orion.benchmark.graph_native_capability_composition_v1 import build_capability_catalog
from orion.benchmark.graph_native_runtime_trace_replay_v3 import _tool_sequence, _trace_files
from orion.benchmark.graph_native_scientific_composition_v0 import (
    ORION_ROOT,
    build_composed_harness_graph,
)
from orion.capvm.governor import CapabilityResidencyGovernor
from orion.capvm.model import (
    CapabilityRuntimeState,
    ResidencyBudgetExceeded,
    ResidencyLevel,
    ResourceVector,
)
from orion.capvm.catalog import CapabilityCatalog
from orion.core.semantic_harness_compiler import (
    compile_capability_catalog_view,
    compile_semantic_skill_agent_loop,
)
from orion.evolution.change_domain import add_capability_contract_projection, add_skill_workflow_projection


SKILL_NAME = "electrolyte_design"
SKILL_PATH = ORION_ROOT / "skills" / SKILL_NAME / "SKILL.md"
BUDGET = ResourceVector(prompt_tokens=160.0, host_ram_mb=4.0)
STRATEGIES = ("retain_all", "release_all", "cwsr")
REPEAT_COUNT = 3


class NoEvictionPolicy:
    """Keep all inactive residents; fail when the next request no longer fits."""

    def choose(
        self,
        *,
        states: Mapping[str, CapabilityRuntimeState],
        catalog: CapabilityCatalog,
        violated_dimensions: tuple[str, ...],
        protected_ids: set[str],
        cost_basis: str = "declared",
    ) -> tuple[str, ResidencyLevel] | None:
        del states, catalog, violated_dimensions, protected_ids, cost_basis
        return None


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_profile():
    graph, _diagnostics = build_composed_harness_graph()
    template_catalog = build_capability_catalog()
    add_capability_contract_projection(graph, template_catalog.all())
    add_skill_workflow_projection(graph, skill_path=SKILL_PATH)
    _app, profile = compile_semantic_skill_agent_loop(
        graph,
        skill_name=SKILL_NAME,
        enabled_optional_step_ids=None,
        executor=lambda _binding, _state: None,
    )
    return graph, profile


def _new_governor(graph, strategy: str) -> CapabilityResidencyGovernor:
    catalog = compile_capability_catalog_view(graph, build_capability_catalog())
    if strategy == "retain_all":
        return CapabilityResidencyGovernor(
            catalog=catalog,
            budget=BUDGET,
            policy=NoEvictionPolicy(),
        )
    return CapabilityResidencyGovernor(catalog=catalog, budget=BUDGET)


def _release_all_inactive(governor: CapabilityResidencyGovernor) -> int:
    released = 0
    for state in governor.states():
        if state.current_level <= ResidencyLevel.CATALOG:
            continue
        governor.demote(state.capability_id, ResidencyLevel.CATALOG)
        released += 1
    return released


def _run_strategy(graph, profile, scientific: tuple[str, ...], strategy: str, slot_id: str) -> dict[str, object]:
    governor = _new_governor(graph, strategy)
    completed = 0
    failed_call_index: int | None = None
    failed_tool: str | None = None
    failure_dimensions: tuple[str, ...] = ()
    explicit_release_count = 0
    peak_resident_count = 0
    peak_prompt_tokens = 0.0
    peak_host_ram_mb = 0.0

    try:
        for call_index, tool_name in enumerate(scientific):
            binding = profile.binding_for_tool(tool_name)
            try:
                with governor.acquire(
                    binding.capability_id,
                    ResidencyLevel.BOUND,
                    request_id=f"workflow-policy:{strategy}:{slot_id}:{call_index}:{tool_name}",
                ):
                    for capability_id, level_name in binding.residency_requirements:
                        if governor.state(capability_id).current_level < ResidencyLevel[level_name]:
                            raise AssertionError(
                                f"unsatisfied compiled residency: {tool_name}:{capability_id}:{level_name}"
                            )
                    current = governor.total_declared_cost()
                    resident_count = sum(
                        state.current_level > ResidencyLevel.CATALOG
                        for state in governor.states()
                    )
                    peak_resident_count = max(peak_resident_count, resident_count)
                    peak_prompt_tokens = max(peak_prompt_tokens, current.prompt_tokens)
                    peak_host_ram_mb = max(peak_host_ram_mb, current.host_ram_mb)
                completed += 1
                if strategy == "release_all":
                    explicit_release_count += _release_all_inactive(governor)
            except ResidencyBudgetExceeded as exc:
                failed_call_index = call_index
                failed_tool = tool_name
                failure_dimensions = tuple(exc.violated_dimensions)
                break

        events = Counter(event.event_type for event in governor.events)
        budget_evictions = sum(
            1
            for event in governor.events
            if event.event_type == "demote" and event.detail == "budget_eviction"
        )
        explicit_demotions = sum(
            1
            for event in governor.events
            if event.event_type == "demote" and event.detail == "explicit"
        )
        return {
            "strategy": strategy,
            "scientific_call_count": len(scientific),
            "completed_call_count": completed,
            "complete_trace": completed == len(scientific),
            "failed_call_index": failed_call_index,
            "failed_tool": failed_tool,
            "failure_dimensions": list(failure_dimensions),
            "peak_resident_count": peak_resident_count,
            "peak_prompt_tokens": peak_prompt_tokens,
            "peak_host_ram_mb": peak_host_ram_mb,
            "promotion_count": int(events.get("promote", 0)),
            "compulsory_fault_count": int(events.get("compulsory_fault", 0)),
            "refault_count": int(events.get("refault", 0)),
            "budget_eviction_count": budget_evictions,
            "explicit_release_count": explicit_release_count,
            "explicit_demotions_recorded": explicit_demotions,
        }
    finally:
        governor.close()


def _single_run() -> dict[str, object]:
    graph, profile = _build_profile()
    allowed = set(profile.allowed_tool_names)
    rows: list[dict[str, object]] = []

    for path in _trace_files():
        record = json.loads(path.read_text(encoding="utf-8"))
        raw = _tool_sequence(record)
        scientific = tuple(tool for tool in raw if tool in allowed)
        slot_id = str(record.get("slot_id") or path.stem)
        strategy_rows = {
            strategy: _run_strategy(graph, profile, scientific, strategy, slot_id)
            for strategy in STRATEGIES
        }
        rows.append(
            {
                "file": path.name,
                "slot_id": slot_id,
                "scientific_tool_call_count": len(scientific),
                "strategies": strategy_rows,
            }
        )

    active = [row for row in rows if int(row["scientific_tool_call_count"]) > 0]
    aggregate: dict[str, dict[str, object]] = {}
    for strategy in STRATEGIES:
        srows = [row["strategies"][strategy] for row in active]
        aggregate[strategy] = {
            "active_trace_count": len(active),
            "complete_trace_count": sum(bool(row["complete_trace"]) for row in srows),
            "total_scientific_call_count": sum(int(row["scientific_call_count"]) for row in srows),
            "completed_scientific_call_count": sum(int(row["completed_call_count"]) for row in srows),
            "promotion_count": sum(int(row["promotion_count"]) for row in srows),
            "compulsory_fault_count": sum(int(row["compulsory_fault_count"]) for row in srows),
            "refault_count": sum(int(row["refault_count"]) for row in srows),
            "budget_eviction_count": sum(int(row["budget_eviction_count"]) for row in srows),
            "explicit_release_count": sum(int(row["explicit_release_count"]) for row in srows),
            "blocked_trace_count": sum(not bool(row["complete_trace"]) for row in srows),
            "failure_dimensions": dict(
                Counter(
                    dimension
                    for row in srows
                    for dimension in row["failure_dimensions"]
                )
            ),
            "peak_resident_count": max((int(row["peak_resident_count"]) for row in srows), default=0),
            "peak_prompt_tokens": max((float(row["peak_prompt_tokens"]) for row in srows), default=0.0),
            "peak_host_ram_mb": max((float(row["peak_host_ram_mb"]) for row in srows), default=0.0),
        }

    cwsr = aggregate["cwsr"]
    release = aggregate["release_all"]
    retain = aggregate["retain_all"]
    return {
        "schema_version": "cwsr-electrolyte-workflow-policy-replay-v1",
        "experiment_role": (
            "Runtime-policy comparison on frozen sodium-ion electrolyte access sequences; archived tool "
            "outputs are held fixed, so this is execution-feasibility/rematerialization evidence rather "
            "than a chemistry-quality comparison."
        ),
        "trace_count": len(rows),
        "active_trace_count": len(active),
        "budget": {
            "prompt_tokens": BUDGET.prompt_tokens,
            "host_ram_mb": BUDGET.host_ram_mb,
        },
        "aggregate": aggregate,
        "headline_checks": {
            "cwsr_completes_all_active_traces": int(cwsr["complete_trace_count"]) == len(active),
            "release_all_completes_all_active_traces": int(release["complete_trace_count"]) == len(active),
            "retain_all_blocks_at_least_one_active_trace": int(retain["blocked_trace_count"]) > 0,
            "cwsr_uses_fewer_promotions_than_release_all": int(cwsr["promotion_count"]) < int(release["promotion_count"]),
            "all_scientific_calls_are_the_same_frozen_sequence": sum(
                int(row["scientific_tool_call_count"]) for row in active
            ) == int(cwsr["total_scientific_call_count"]),
        },
        "rows": rows,
    }


def run() -> dict[str, object]:
    repeated = [_single_run() for _ in range(REPEAT_COUNT)]
    digests = [_sha256_json(item) for item in repeated]
    first = repeated[0]
    return {
        **first,
        "repeat_count": REPEAT_COUNT,
        "repeat_result_sha256": digests,
        "repetitions_identical": len(set(digests)) == 1,
        "interpretation_guard": (
            "The retain-all and release-all arms alter only runtime residency policy. The experiment "
            "does not rerun scientific tools or rescore archived chemistry outputs; it tests whether "
            "the same real historical scientific access sequences remain executable under the same "
            "proxy resource budget and how much runtime rematerialization each policy requires."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
