"""Development-only replay of frozen Study4C runtime traces through the R6 composition stack.

This is the first probe that joins all of the following using already-existing run data:

* Pi-style Skill workflow frontmatter;
* canonical Agent ToolIdentity;
* explicit capability providers/dependencies;
* the multiplex Harness graph;
* the existing capvm residency governor;
* the ordered tool-call traces from 64 frozen Study4C formal runs.

Optional capability activation is therefore not hand-authored for this experiment.  A recorded tool
call activates the provider of that exact ToolIdentity and its dependency closure.  The governor is
allowed to retain inactive residents as a cache and evict them only under its existing hard-budget
policy; logical requested working set and cached resident set are reported separately.

The replay also detects a useful contract-consistency condition: a runtime tool may have a known
capability provider but be absent from the Skill's machine-readable workflow.  Such a call is marked
``mapped_but_undeclared`` rather than silently treated as a declared Skill step.
"""

from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from orion.benchmark.graph_native_capability_composition_v1 import build_capability_catalog
from orion.benchmark.graph_native_scientific_composition_v0 import build_composed_harness_graph
from orion.capvm.governor import CapabilityResidencyGovernor
from orion.capvm.model import ResidencyLevel, ResourceVector
from orion.evolution.change_domain import (
    HarnessArtifact,
    active_skill_tool_nodes,
    add_capability_contract_projection,
    add_skill_workflow_projection,
    project_tool_capability_working_set,
)


ORION_ROOT = Path(__file__).resolve().parents[1]
TRACE_GLOB = ORION_ROOT / "benchmark" / "self_evolution_study4c_v1_20260822" / "formal_run_v1" / "raw" / "*.json"
SKILL_PATH = ORION_ROOT / "skills" / "electrolyte_design" / "SKILL.md"
BUDGET = ResourceVector(prompt_tokens=160.0, host_ram_mb=4.0)


def _cost_dict(value: ResourceVector) -> dict[str, float]:
    return {
        "prompt_tokens": value.prompt_tokens,
        "host_ram_mb": value.host_ram_mb,
        "process_slots": value.process_slots,
        "connection_slots": value.connection_slots,
    }


def _resident_ids(governor: CapabilityResidencyGovernor) -> tuple[str, ...]:
    return tuple(
        sorted(
            state.capability_id
            for state in governor.states()
            if state.current_level > ResidencyLevel.CATALOG
        )
    )


def _active_skill_names(skill) -> set[str]:
    return {step.tool_name for step in skill.steps}


def _provider_for_tool(graph, tool_name: str) -> tuple[str, ...]:
    tool_node = f"tool_identity:{tool_name}"
    return tuple(
        sorted(
            edge.source.removeprefix("capability:")
            for edge in graph.edges
            if edge.target == tool_node
            and edge.relation == "implemented_by_tool"
            and edge.source.startswith("capability:")
        )
    )


def _event_delta(governor: CapabilityResidencyGovernor, start: int) -> Counter[str]:
    return Counter(event.event_type for event in governor.events[start:])


def _budget_evictions(governor: CapabilityResidencyGovernor, start: int) -> int:
    return sum(
        1
        for event in governor.events[start:]
        if event.event_type == "demote" and event.detail == "budget_eviction"
    )


def _trace_files() -> tuple[Path, ...]:
    return tuple(Path(item) for item in sorted(glob.glob(str(TRACE_GLOB))))


def _tool_sequence(record: dict[str, Any]) -> tuple[str, ...]:
    sequence: list[str] = []
    for item in record.get("tool_calls") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            sequence.append(name)
    return tuple(sequence)


def run_runtime_trace_replay() -> dict[str, object]:
    graph, graph_diagnostics = build_composed_harness_graph()
    template_catalog = build_capability_catalog()
    template_capabilities = tuple(template_catalog.all())
    capability_projection = add_capability_contract_projection(graph, template_capabilities)
    skill = add_skill_workflow_projection(graph, skill_path=SKILL_PATH)
    skill_declared_names = _active_skill_names(skill)
    required_skill_nodes = active_skill_tool_nodes(skill)
    all_skill_nodes = tuple(step.tool_node_id for step in skill.steps)

    provider_map = {
        artifact.label: _provider_for_tool(graph, artifact.label)
        for artifact in graph.artifacts.values()
        if artifact.kind == "agent_tool_identity"
    }
    mapped_tool_names = {name for name, providers in provider_map.items() if providers}

    # Static profiles are computed once to contextualize the trace-driven replay.
    static_task_node = "task:study4c-trace-static-profile"
    graph.add_artifact(
        HarnessArtifact(
            artifact_id=static_task_node,
            kind="task",
            label="Study4C trace replay static profile",
        )
    )
    required_profile = project_tool_capability_working_set(
        graph,
        task_id="study4c-trace-static-profile",
        task_node_id=static_task_node,
        tool_node_ids=required_skill_nodes,
    )
    all_skill_profile = project_tool_capability_working_set(
        graph,
        task_id="study4c-trace-static-profile",
        task_node_id=static_task_node,
        tool_node_ids=all_skill_nodes,
    )

    rows: list[dict[str, object]] = []
    global_call_classes: Counter[str] = Counter()
    global_mapped_tool_calls: Counter[str] = Counter()
    global_event_counts: Counter[str] = Counter()
    global_budget_evictions = 0
    global_peak_requested = 0
    global_peak_resident = 0
    global_peak_prompt = 0.0
    global_peak_ram = 0.0

    for path in _trace_files():
        record = json.loads(path.read_text(encoding="utf-8"))
        task_node = f"task:trace:{str(record.get('slot_id') or path.stem).replace(':', '_').replace(' ', '_')}"
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=task_node,
                kind="runtime_trace_task",
                label=str(record.get("case_id") or path.stem),
                metadata={
                    "family_id": str(record.get("family_id") or ""),
                    "arm": str(record.get("arm") or ""),
                },
            )
        )

        catalog = build_capability_catalog()
        governor = CapabilityResidencyGovernor(catalog=catalog, budget=BUDGET)
        sequence = _tool_sequence(record)
        episode_classes: Counter[str] = Counter()
        episode_tool_calls: Counter[str] = Counter()
        episode_events: Counter[str] = Counter()
        episode_budget_evictions = 0
        episode_peak_requested = 0
        episode_peak_resident = 0
        episode_peak_prompt = 0.0
        episode_peak_ram = 0.0
        episode_unique_capabilities: set[str] = set()
        mapped_sequence: list[dict[str, object]] = []

        try:
            for call_index, tool_name in enumerate(sequence):
                providers = provider_map.get(tool_name, ())
                if not providers:
                    call_class = "unmodeled_auxiliary"
                    episode_classes[call_class] += 1
                    global_call_classes[call_class] += 1
                    continue

                declared = tool_name in skill_declared_names
                call_class = "skill_declared" if declared else "mapped_but_undeclared"
                episode_classes[call_class] += 1
                global_call_classes[call_class] += 1
                episode_tool_calls[tool_name] += 1
                global_mapped_tool_calls[tool_name] += 1

                working_set = project_tool_capability_working_set(
                    graph,
                    task_id=str(record.get("slot_id") or path.stem),
                    task_node_id=task_node,
                    tool_node_ids=(f"tool_identity:{tool_name}",),
                )
                requested_ids = working_set.materialization_ids
                episode_unique_capabilities.update(requested_ids)
                episode_peak_requested = max(episode_peak_requested, len(requested_ids))
                global_peak_requested = max(global_peak_requested, len(requested_ids))

                event_start = len(governor.events)
                requests = tuple((item, ResidencyLevel.BOUND) for item in requested_ids)
                with governor.acquire_many(
                    requests,
                    request_id=f"trace::{path.stem}::{call_index}::{tool_name}",
                ):
                    resident_ids = _resident_ids(governor)
                    resident_cost = governor.total_declared_cost()
                    episode_peak_resident = max(episode_peak_resident, len(resident_ids))
                    global_peak_resident = max(global_peak_resident, len(resident_ids))
                    episode_peak_prompt = max(episode_peak_prompt, resident_cost.prompt_tokens)
                    episode_peak_ram = max(episode_peak_ram, resident_cost.host_ram_mb)
                    global_peak_prompt = max(global_peak_prompt, resident_cost.prompt_tokens)
                    global_peak_ram = max(global_peak_ram, resident_cost.host_ram_mb)
                    mapped_sequence.append(
                        {
                            "index": call_index,
                            "tool": tool_name,
                            "declared_in_skill": declared,
                            "selected_capabilities": list(working_set.selected_capability_ids),
                            "dependency_closure": list(requested_ids),
                            "requested_working_set_size": len(requested_ids),
                            "resident_count_during_call": len(resident_ids),
                            "resident_ids_during_call": list(resident_ids),
                            "declared_cost_during_call": _cost_dict(resident_cost),
                        }
                    )
                delta = _event_delta(governor, event_start)
                evictions = _budget_evictions(governor, event_start)
                episode_events.update(delta)
                global_event_counts.update(delta)
                episode_budget_evictions += evictions
                global_budget_evictions += evictions
        finally:
            final_resident = _resident_ids(governor)
            final_cost = governor.total_declared_cost()
            governor.close()

        rows.append(
            {
                "file": path.name,
                "slot_id": str(record.get("slot_id") or ""),
                "family_id": str(record.get("family_id") or ""),
                "case_id": str(record.get("case_id") or ""),
                "arm": str(record.get("arm") or ""),
                "repeat_index": record.get("repeat_index"),
                "original_ok": bool(record.get("ok")),
                "raw_tool_call_count": len(sequence),
                "call_classes": dict(episode_classes),
                "mapped_tool_call_count": sum(episode_tool_calls.values()),
                "mapped_tool_call_counts": dict(episode_tool_calls),
                "mapped_but_undeclared_tools": sorted(
                    name for name in episode_tool_calls if name not in skill_declared_names
                ),
                "unique_mapped_tool_count": len(episode_tool_calls),
                "unique_capability_count_used": len(episode_unique_capabilities),
                "peak_requested_working_set_size": episode_peak_requested,
                "peak_cached_resident_count": episode_peak_resident,
                "peak_declared_prompt_tokens": episode_peak_prompt,
                "peak_declared_host_ram_mb": episode_peak_ram,
                "budget_eviction_count": episode_budget_evictions,
                "residency_events": dict(episode_events),
                "final_cached_resident_ids": list(final_resident),
                "final_cached_resident_cost": _cost_dict(final_cost),
                "mapped_sequence": mapped_sequence,
            }
        )

    mapped_rows = [row for row in rows if int(row["mapped_tool_call_count"]) > 0]
    family_tool_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family = str(row["family_id"])
        family_tool_sets[family].update(dict(row["mapped_tool_call_counts"]).keys())

    return {
        "schema_version": "graph-native-runtime-trace-replay-v3.0",
        "status": "development_only_replay_no_provider_calls",
        "trace_source": str(TRACE_GLOB.relative_to(ORION_ROOT)),
        "trace_count": len(rows),
        "graph": {
            "artifact_count_before_trace_tasks": graph_diagnostics.artifact_count,
            "edge_count_before_trace_tasks": graph_diagnostics.edge_count,
            "multi_role_artifact_count": graph_diagnostics.multi_role_artifact_count,
            "tool_backed_capability_count": sum(
                1 for links in capability_projection.tool_provider_links.values() if links
            ),
        },
        "skill_contract": {
            "name": skill.skill_name,
            "declared_tool_count": len(skill.steps),
            "declared_tools": [step.tool_name for step in skill.steps],
            "required_profile_capabilities": list(required_profile.materialization_ids),
            "required_profile_size": len(required_profile.materialization_ids),
            "all_skill_profile_capabilities": list(all_skill_profile.materialization_ids),
            "all_skill_profile_size": len(all_skill_profile.materialization_ids),
        },
        "budget": _cost_dict(BUDGET),
        "aggregate": {
            "episodes_with_mapped_calls": len(mapped_rows),
            "call_classes": dict(global_call_classes),
            "mapped_tool_call_counts": dict(global_mapped_tool_calls),
            "mapped_but_undeclared_tool_names": sorted(
                name for name in global_mapped_tool_calls if name not in skill_declared_names
            ),
            "peak_requested_working_set_size": global_peak_requested,
            "peak_cached_resident_count": global_peak_resident,
            "peak_declared_prompt_tokens": global_peak_prompt,
            "peak_declared_host_ram_mb": global_peak_ram,
            "budget_eviction_count": global_budget_evictions,
            "residency_events": dict(global_event_counts),
            "mean_unique_capabilities_used_per_episode": (
                sum(int(row["unique_capability_count_used"]) for row in mapped_rows) / len(mapped_rows)
                if mapped_rows
                else 0.0
            ),
            "mean_peak_requested_working_set_size": (
                sum(int(row["peak_requested_working_set_size"]) for row in mapped_rows) / len(mapped_rows)
                if mapped_rows
                else 0.0
            ),
            "mean_peak_cached_resident_count": (
                sum(int(row["peak_cached_resident_count"]) for row in mapped_rows) / len(mapped_rows)
                if mapped_rows
                else 0.0
            ),
        },
        "family_unique_mapped_tools": {
            family: sorted(names) for family, names in sorted(family_tool_sets.items())
        },
        "episodes": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run_runtime_trace_replay(), ensure_ascii=False, indent=2))
