"""Warm-cache production ToolNode A/B on frozen real-agent electrolyte traces.

Both arms run exactly the same tool names, arguments, and ToolNode batches
against an isolated real RAG/Neo4j service with BGE-M3 and Driver preloaded.
No LLM, no database writes, no resource eviction, no alternative science tool.
Report an empirical no-reclamation overhead, not full-agent acceleration.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from orion.benchmark.cwsr_fixed_pressure_replay_v1 import (
    bind_isolated_service, canonical, load_trace, tool_executor,
    validate_tool_outputs, workspace,
)
from orion.benchmark.cwsr_complete_case_resource_v1 import dedicated_service
from orion.capvm import (CapabilityCatalog, CapabilityKind,
                         CapabilityResidencyGovernor, ResidencyLevel,
                         ResourceVector, VirtualCapability)
from orion.capvm.adapters import RootServiceResourceAdapter
from orion.core.default_resource_runtime import (
    ROOT_BGE_RESOURCE_ID, ROOT_NEO4J_RESOURCE_ID,
    _MODEL_SESSION_TOOLS, _SESSION_ONLY_TOOLS,
)
from orion.core.langgraph_residency import (
    RuntimeResourceRequirement, ToolCallResidencyBridge, ToolResidencyBinding,
)

TASKS = ("U1", "U2", "P1", "P2", "C1", "C2", "C3", "D1")
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "reports/cwsr_original_eight_native_runtime_20261006_v11"
MODEL_REMOTE = "bge-m3-model"
DB_REMOTE = "neo4j-session"


def available_host_mib() -> float:
    """Windows-native preflight, without installing another Python dependency."""
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    state = MemoryStatus()
    state.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise OSError("Cannot sample available physical RAM")
    return state.ullAvailPhys / 1024**2


def paired_order(pair_index: int) -> tuple[str, str]:
    if pair_index < 0:
        raise ValueError("pair_index must be nonnegative")
    return ("native", "runtime") if pair_index % 2 == 0 else ("runtime", "native")


def build_bridge(client):
    caps = (
        VirtualCapability(
            capability_id=ROOT_BGE_RESOURCE_ID, kind=CapabilityKind.MODEL,
            summary="resident BGE-M3",
            maximum_level=ResidencyLevel.BOUND,
            adapter=RootServiceResourceAdapter(
                client=client, remote_resource_id=MODEL_REMOTE,
                declared_cost=ResourceVector(host_ram_mb=550, gpu_vram_mb=2300),
            ),
        ),
        VirtualCapability(
            capability_id=ROOT_NEO4J_RESOURCE_ID, kind=CapabilityKind.SESSION,
            summary="resident Neo4j Driver",
            maximum_level=ResidencyLevel.BOUND,
            adapter=RootServiceResourceAdapter(
                client=client, remote_resource_id=DB_REMOTE,
                declared_cost=ResourceVector(connection_slots=1),
            ),
        ),
    )
    governor = CapabilityResidencyGovernor(
        catalog=CapabilityCatalog(caps),
        budget=ResourceVector(host_ram_mb=4096, gpu_vram_mb=4096,
                              connection_slots=4),
    )
    bindings = {
        name: ToolResidencyBinding(resources=tuple(
            RuntimeResourceRequirement(resource_id) for resource_id in (
                (ROOT_BGE_RESOURCE_ID, ROOT_NEO4J_RESOURCE_ID)
                if name in _MODEL_SESSION_TOOLS else (ROOT_NEO4J_RESOURCE_ID,)
            )
        ))
        for name in (_MODEL_SESSION_TOOLS | _SESSION_ONLY_TOOLS)
    }
    bridge = ToolCallResidencyBridge(
        governor=governor, bindings=bindings,
        reconcile_measured=True, observation_recorder=None,
    )
    return governor, bridge


def run_trace(trace, node, bridge, arm):
    from langchain_core.messages import AIMessage, ToolMessage

    rows = []
    started = perf_counter()
    for index, batch in enumerate(trace["batches"]):
        state = {"messages": [AIMessage(content="", tool_calls=batch["calls"])]}
        before = perf_counter()
        lease = (nullcontext() if arm == "native"
                 else bridge.acquire_for_state(state, request_id=f"warm-ab-{index}"))
        with lease:
            entered = perf_counter()
            result = node.invoke(state)
            tool_s = perf_counter() - entered
        total_s = perf_counter() - before
        messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        if [m.tool_call_id for m in messages] != [c["id"] for c in batch["calls"]]:
            raise RuntimeError("ToolMessage order/IDs changed")
        validate_tool_outputs(messages)
        outputs = [(m.tool_call_id, m.name, m.status, m.content) for m in messages]
        rows.append({
            "batch": index, "source_message_index": batch["source_message_index"],
            "tool_names": [c["name"] for c in batch["calls"]],
            "tool_s": round(tool_s, 6),
            "total_s": round(total_s, 6),
            "overhead_s": round(total_s - tool_s, 6),
            "output_sha256": canonical(outputs),
        })
    return {
        "arm": arm, "elapsed_s": round(perf_counter() - started, 6),
        "tool_s": round(sum(r["tool_s"] for r in rows), 6),
        "overhead_s": round(sum(r["overhead_s"] for r in rows), 6),
        "outputs_sha256": canonical([r["output_sha256"] for r in rows]),
        "rows": rows,
    }


def verify_preflight(client):
    snapshot = client.get("/v1/metadata")
    index = snapshot.get("vector_index") or {}
    if index.get("state") != "ONLINE":
        raise RuntimeError("Neo4j vector index is not ONLINE")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", choices=TASKS, required=True,
                        help="Use complete r1-native source trace; may repeat for multiple tasks")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-port", type=int, default=8814)
    args = parser.parse_args()
    if args.pairs < 1 or len(set(args.task)) != len(args.task):
        parser.error("--pairs must be positive and tasks must be unique")
    # Fixed task names and repository path: do not permit arbitrary user paths.
    input_paths = [SOURCE_ROOT / f"{task}-r1-native/result.json" for task in args.task]
    sources = [(name, load_trace(path)) for name, path in zip(args.task, input_paths)]
    output = workspace(args.output)
    if output.exists():
        raise FileExistsError("Results are immutable; choose a fresh output directory")
    output.mkdir(parents=True)
    frozen = {name: source["sha256"] for name, source in sources}
    plan = {
        "schema": "cwsr-v17-warm-toolnode-ab-v1", "task_order": args.task,
        "pairs": args.pairs, "arms": [list(paired_order(i)) for i in range(args.pairs)],
        "source_files": {name: source["path"] for name, source in sources},
        "source_sha256": frozen,
        "scope": "Warm resident BGE-M3+Neo4j; real production ToolNode and same trace; no LLM",
        "selection": "Entire frozen r1-native trace of every preselected task, preserving original parallel batches",
        "reclamation": "Disabled by sufficient declared budget; report if any demotion occurs",
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    if available_host_mib() < 1400:
        raise RuntimeError("Insufficient free host memory for safe BGE-M3 test (1400 MiB)")
    from tools.config import main_tools
    selected_names = {call["name"] for _, source in sources
                      for batch in source["batches"] for call in batch["calls"]}
    allowed = _MODEL_SESSION_TOOLS | _SESSION_ONLY_TOOLS
    if selected_names - allowed:
        raise RuntimeError("Unmapped tool(s): " + repr(sorted(selected_names - allowed)))
    tools = [tool for tool in main_tools if tool.name in selected_names]
    if {t.name for t in tools} != selected_names:
        raise RuntimeError("Missing production tool registrations")
    node = tool_executor(tools)

    result_rows = []
    with dedicated_service(output, args.service_port) as url:
        client = bind_isolated_service(url)
        initial = verify_preflight(client)
        # Initialize both arms equally before the timed region: root state,
        # BGE weights and driver connection are established exactly once.
        client.acquire_runtime_resource(MODEL_REMOTE)
        client.acquire_runtime_resource(DB_REMOTE)
        governor, bridge = build_bridge(client)
        try:
            from langchain_core.messages import AIMessage
            warm_state = {"messages": [AIMessage(content="", tool_calls=[{
                "name": next(iter(_MODEL_SESSION_TOOLS)),
                "id": "warm-pool", "args": {}, "type": "tool_call",
            }])]}
            with bridge.acquire_for_state(warm_state, request_id="outside-measured-warm-up"):
                pass
            before_resources = client.runtime_resources()
            for task, trace in sources:
                # Common scientific workload preheat eliminates one-arm-only
                # database/OS query warm-up differences.
                reference = run_trace(trace, node, bridge, "native")["outputs_sha256"]
                for pair in range(args.pairs):
                    measurements = {}
                    for arm in paired_order(pair):
                        observation = run_trace(trace, node, bridge, arm)
                        observation.update(task=task, pair=pair)
                        measurements[arm] = observation
                        result_rows.append(observation)
                        (output / f"{task}-p{pair}-{arm}.json").write_text(
                            json.dumps(observation, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        print(json.dumps({
                            "task": task, "pair": pair, "arm": arm,
                            "elapsed_s": observation["elapsed_s"],
                            "overhead_s": observation["overhead_s"],
                            "output_matches_reference": observation["outputs_sha256"] == reference,
                        }), flush=True)
                    if any(measurements[arm]["outputs_sha256"] != reference for arm in paired_order(pair)):
                        raise RuntimeError(f"Scientific output mismatch in {task} pair {pair}")
            after_resources = client.runtime_resources()
            deltas = {
                key: {field: int(after_resources[key].get(field, 0)) -
                             int(before_resources[key].get(field, 0))
                      for field in fields}
                for key, fields in ((MODEL_REMOTE, ("load_count", "release_count")),
                                    (DB_REMOTE, ("open_count", "close_count")))
            }
            evictions = [e for e in governor.events if e.event_type in {"demote", "evict"}]
            final = verify_preflight(client)
            index_unchanged = initial["snapshot_fingerprint"] == final["snapshot_fingerprint"]
            input_unchanged = all(
                sha256(Path(source["path"]).read_bytes()).hexdigest() == frozen[task]
                for task, source in sources
            )
            pairs = []
            for task in args.task:
                for pair in range(args.pairs):
                    native = next(r for r in result_rows if r["task"] == task and r["pair"] == pair and r["arm"] == "native")
                    runtime = next(r for r in result_rows if r["task"] == task and r["pair"] == pair and r["arm"] == "runtime")
                    pairs.append({"task": task, "pair": pair,
                                  "native_s": native["elapsed_s"],
                                  "runtime_s": runtime["elapsed_s"],
                                  "runtime_minus_native_s": round(runtime["elapsed_s"] - native["elapsed_s"], 6),
                                  "percent": round(100 * (runtime["elapsed_s"] / native["elapsed_s"] - 1), 3)})
            parity = all(r["outputs_sha256"] == next(
                x["outputs_sha256"] for x in result_rows if x["task"] == r["task"]
            ) for r in result_rows)
            safe = parity and index_unchanged and input_unchanged and not evictions and all(
                value == 0 for counters in deltas.values() for value in counters.values())
            summary = {
                "schema": plan["schema"], "status": "PASS" if safe else "FAIL",
                "scientific_outputs_identical": parity,
                "source_files_unchanged": input_unchanged,
                "metadata_snapshot_fingerprint_unchanged": index_unchanged,
                "no_evictions": not evictions,
                "root_resource_counter_deltas": deltas,
                "paired": pairs, "paired_delta_median_s": median(p["runtime_minus_native_s"] for p in pairs),
                "paired_percent_median": median(p["percent"] for p in pairs),
                "over_3_percent_pairs": sum(p["percent"] > 3 for p in pairs),
            }
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "paired"}, ensure_ascii=False), flush=True)
            if not safe:
                raise RuntimeError("Scientific/capacity correctness check failed")
        finally:
            governor.close()


if __name__ == "__main__":
    main()
