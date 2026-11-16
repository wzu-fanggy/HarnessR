"""固定真实调用与批次，比较已有缓存、超限全部释放与选择性回收。

默认只冻结方案。--execute 使用独立只读 RAG 服务，不调用 Agent/LLM，
不改写原始记录；所有方式采用相同的批次拆分安排。
"""
from __future__ import annotations
from release_paths import release_path

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
if ROOT.name != "orion":
    ROOT = Path(release_path('${ORION_SOURCE}'))
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

ARMS = ("native_cached", "pressure_flush", "runtime")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


def workspace(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.parent):
        raise ValueError("Path outside project workspace")
    return path


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def source_fingerprints():
    """冻结重放实际使用的工具、运行时、服务代码，不读取配置密钥。"""
    files = [Path(__file__), ROOT / "benchmark/cwsr_complete_case_resource_v1.py",
             ROOT / "benchmark/cwsr_native_runtime_pair_v1.py", ROOT / "benchmark/cwsr_na_conductivity_ml_worker_v1.py"]
    for folder in (ROOT / "core", ROOT / "capvm", ROOT / "tools", ROOT.parent / "kg_rag"):
        files.extend(folder.rglob("*.py"))
    return {str(path.relative_to(ROOT.parent)).replace("\\", "/"): sha(path) for path in sorted(set(files))}


def load_trace(path):
    from orion.core.default_resource_runtime import _MODEL_SESSION_TOOLS, _SESSION_ONLY_TOOLS
    managed = _MODEL_SESSION_TOOLS | _SESSION_ONLY_TOOLS | {"predict_formulation_conductivity"}
    path = workspace(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("ok") is not True:
        raise ValueError("Only completed autonomous source runs are accepted")
    batches = []
    for index, message in enumerate(raw.get("messages") or []):
        data = message.get("data", message)
        calls = data.get("tool_calls") or []
        selected = [{"name": call["name"], "args": call.get("args") or {},
                     "id": call["id"], "type": "tool_call"}
                    for call in calls if call.get("name") in managed]
        if selected:
            batches.append({"source_message_index": index, "calls": selected})
    executed = [{"name": call["name"], "args": call.get("args") or {}}
                for call in raw.get("tool_calls") or [] if call.get("name") in managed]
    requested = [{"name": call["name"], "args": call["args"]}
                 for batch in batches for call in batch["calls"]]
    if not requested or requested != executed:
        raise ValueError("Saved requests and executed tools do not match")
    ids = [call["id"] for batch in batches for call in batch["calls"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source tool-call IDs")
    return {"path": str(path), "sha256": sha(path), "arm": raw.get("arm"),
            "batches": batches, "managed_call_count": len(requested)}


def resource_requirements(name):
    from orion.core.default_resource_runtime import (
        _MODEL_SESSION_TOOLS, _SESSION_ONLY_TOOLS, ROOT_BGE_RESOURCE_ID,
        ROOT_NEO4J_RESOURCE_ID, CONDUCTIVITY_PROCESS_RESOURCE_ID)
    if name in _MODEL_SESSION_TOOLS:
        return (ROOT_BGE_RESOURCE_ID, ROOT_NEO4J_RESOURCE_ID)
    if name in _SESSION_ONLY_TOOLS:
        return (ROOT_NEO4J_RESOURCE_ID,)
    if name == "predict_formulation_conductivity":
        return (CONDUCTIVITY_PROCESS_RESOURCE_ID,)
    raise ValueError("Unregistered resource tool: " + name)


def required_cost(calls, costs):
    ids = {item for call in calls for item in resource_requirements(call["name"])}
    return sum(costs[item] for item in ids)


def execution_parts(trace, calibration):
    """固定相同拆分，防止策略比较同时改变执行安排。"""
    from orion.core.default_resource_runtime import ROOT_BGE_RESOURCE_ID, ROOT_NEO4J_RESOURCE_ID, CONDUCTIVITY_PROCESS_RESOURCE_ID
    costs = {ROOT_BGE_RESOURCE_ID: calibration["bge_declared_mib"],
             ROOT_NEO4J_RESOURCE_ID: 0, CONDUCTIVITY_PROCESS_RESOURCE_ID: calibration["rf_declared_mib"]}
    limit = calibration["managed_host_ram_mib"]
    parts = []
    for index, batch in enumerate(trace["batches"]):
        calls = batch["calls"]
        if any(required_cost([call], costs) > limit for call in calls):
            raise ValueError("A source call cannot fit individually")
        subsets = [[call] for call in calls] if required_cost(calls, costs) > limit else [calls]
        parts.extend({"source_batch_index": index, "split_overfull_batch": len(subsets) > 1,
                      "calls": subset} for subset in subsets)
    return parts


def make_runtime(client, model, calibration):
    from orion.capvm import CapabilityCatalog, CapabilityKind, CapabilityResidencyGovernor, ResidencyLevel, ResourceVector, VirtualCapability, physical_resource_budget
    from orion.capvm.adapters import RootServiceResourceAdapter
    from orion.core.default_resource_runtime import ROOT_BGE_RESOURCE_ID, ROOT_NEO4J_RESOURCE_ID, CONDUCTIVITY_PROCESS_RESOURCE_ID
    from orion.core.langgraph_residency import ToolCallResidencyBridge, ToolResidencyBinding, RuntimeResourceRequirement
    from tools.electrolyte_design.conductivity_process import SodiumConductivityMlProcessAdapter
    class TrackedRF(SodiumConductivityMlProcessAdapter):
        starts = 0
        stops = 0

        def promote(self, current, target):
            was_running = self.is_running
            super().promote(current, target)
            self.starts += int(not was_running and self.is_running)

        def demote(self, current, target):
            was_running = self.is_running
            super().demote(current, target)
            self.stops += int(was_running and not self.is_running)
    rf = TrackedRF(model_path=model, declared_host_ram_mb=calibration["rf_declared_mib"])
    catalog = CapabilityCatalog((
        VirtualCapability(capability_id=ROOT_BGE_RESOURCE_ID, kind=CapabilityKind.MODEL, summary="BGE-M3", maximum_level=ResidencyLevel.BOUND,
                          adapter=RootServiceResourceAdapter(client=client, remote_resource_id="bge-m3-model", declared_cost=ResourceVector(host_ram_mb=calibration["bge_declared_mib"], gpu_vram_mb=2300))),
        VirtualCapability(capability_id=ROOT_NEO4J_RESOURCE_ID, kind=CapabilityKind.SESSION, summary="Neo4j Driver", maximum_level=ResidencyLevel.BOUND,
                          adapter=RootServiceResourceAdapter(client=client, remote_resource_id="neo4j-session", declared_cost=ResourceVector(connection_slots=1))),
        VirtualCapability(capability_id=CONDUCTIVITY_PROCESS_RESOURCE_ID, kind=CapabilityKind.PROCESS, summary="conductivity worker", adapter=rf, maximum_level=ResidencyLevel.BOUND)))
    governor = CapabilityResidencyGovernor(catalog=catalog, budget=physical_resource_budget(host_ram_mb=calibration["managed_host_ram_mib"], gpu_vram_mb=6144, process_slots=4, connection_slots=8))
    from orion.core.default_resource_runtime import _MODEL_SESSION_TOOLS, _SESSION_ONLY_TOOLS
    names = _MODEL_SESSION_TOOLS | _SESSION_ONLY_TOOLS | {"predict_formulation_conductivity"}
    bindings = {name: ToolResidencyBinding(resources=tuple(RuntimeResourceRequirement(item, ResidencyLevel.BOUND) for item in resource_requirements(name))) for name in names}
    return governor, ToolCallResidencyBridge(governor=governor, bindings=bindings, reconcile_measured=True), rf


def flush_on_pressure(governor, requirements):
    """新需求连同当前缓存超限时释放全部空闲对象，当前调用尚未开始。"""
    from orion.capvm import ResidencyLevel, ResourceVector
    total = ResourceVector()
    measured_projection = governor.total_measured_cost()
    required = dict(requirements)
    for resource in governor.catalog.all():
        level = max(governor.state(resource.capability_id).current_level,
                    required.get(resource.capability_id, ResidencyLevel.CATALOG))
        total += resource.adapter.declared_cost(level)
        if governor.state(resource.capability_id).current_level < required.get(resource.capability_id, ResidencyLevel.CATALOG):
            measured_projection += resource.adapter.declared_cost(required[resource.capability_id])
    if total.fits(governor.budget) and measured_projection.fits(governor.budget):
        return []
    released = []
    for resource in governor.catalog.all():
        state = governor.state(resource.capability_id)
        if state.active_count:
            raise RuntimeError("Cannot flush an active resource")
        if state.current_level > ResidencyLevel.CATALOG:
            governor.demote(resource.capability_id)
            released.append(resource.capability_id)
    return released


def release_remote(client):
    for resource_id, snapshot in client.runtime_resources().items():
        if resource_id in {"bge-m3-model", "neo4j-session"} and snapshot.get("resident"):
            client.release_runtime_resource(resource_id)


def counter_delta(before, after):
    keys = {"bge-m3-model": ("load_count", "release_count"), "neo4j-session": ("open_count", "close_count")}
    return {name: {key: after.get(name, {}).get(key, 0) - before.get(name, {}).get(key, 0)
                   for key in values} for name, values in keys.items()}


def tool_executor(tools):
    """提供当前 LangGraph 版本要求的 Runtime；图中只有工具节点，没有模型节点。"""
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.prebuilt import ToolNode
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=False))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile()


def validate_tool_outputs(messages):
    """工具以正常 ToolMessage 返回服务错误时，仍视为执行失败。"""
    for message in messages:
        if getattr(message, "status", "success") == "error":
            raise RuntimeError("Tool returned an error message")
        content = message.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except ValueError:
                continue
        values = content if isinstance(content, list) else [content]
        if any(isinstance(value, dict) and (value.get("error") or value.get("content_type") == "error") for value in values):
            raise RuntimeError("Production tool returned a service-error fallback")


def bind_isolated_service(url):
    """统一现有两种包名及 RAG 独立客户端，禁止流向常驻8799服务。"""
    import importlib
    os.environ["ORION_RAG_SERVICE_URL"] = url
    from tools.rag.service_client import RootKgServiceClient
    client = RootKgServiceClient(base_url=url)
    for name in ("tools.rag.service_client", "orion.tools.rag.service_client"):
        importlib.import_module(name)._root_kg_service_client = client
    for name in ("tools.rag.kg_rag_tools", "orion.tools.rag.kg_rag_tools"):
        module = importlib.import_module(name)
        module._rag_service_client = module._RagServiceClient(base_url=url)
        if module._get_rag_service_client().base_url != url:
            raise RuntimeError("RAG service isolation failed")
    return client


def run_arm(trace, parts, arm, client, model, calibration):
    from langchain_core.messages import AIMessage, ToolMessage
    from tools.config import main_tools
    from tools.electrolyte_design import conductivity_tool
    from tools.electrolyte_design.conductivity_process import ConductivityProcessPredictor
    from orion.capvm import ResidencyLevel
    from orion.benchmark.cwsr_complete_case_resource_v1 import memory
    release_remote(client)
    import tools.rag.kg_rag_tools as rag_tools
    import orion.tools.rag.service_client as kg_client
    if rag_tools._get_rag_service_client().base_url != client.base_url or kg_client.get_root_kg_service_client().base_url != client.base_url:
        raise RuntimeError("Production tool service URL differs from measured service")
    governor, bridge, rf = make_runtime(client, model, calibration)
    previous_model = conductivity_tool._MODEL
    class LazyPredictor(ConductivityProcessPredictor):
        def predict_features(self, features):
            if not self.adapter.is_running:
                self.adapter.promote(ResidencyLevel.CATALOG, ResidencyLevel.BOUND)
            return super().predict_features(features)
    if arm == "native_cached":
        conductivity_tool._MODEL = LazyPredictor(rf)
    selected_names = {call["name"] for part in parts for call in part["calls"]}
    node = tool_executor([tool for tool in main_tools if tool.name in selected_names])
    before = client.runtime_resources()
    rows = []
    last_result = None
    started = monotonic()
    error = None
    try:
        for index, part in enumerate(parts):
            state = {"messages": [AIMessage(content="", tool_calls=part["calls"])]}
            prepare_started = monotonic()
            flushed = flush_on_pressure(governor, bridge.requirements(state)) if arm == "pressure_flush" else []
            lease = nullcontext() if arm == "native_cached" else bridge.acquire_for_state(state, request_id=f"replay-{index}")
            with lease:
                prepare_s = monotonic() - prepare_started
                query_started = monotonic()
                output = node.invoke(state)
                tool_s = monotonic() - query_started
                last_result = monotonic() - started
                messages = [message for message in output["messages"] if isinstance(message, ToolMessage)]
                if [message.tool_call_id for message in messages] != [call["id"] for call in part["calls"]]:
                    raise RuntimeError("Tool outputs do not match source request IDs and order")
                validate_tool_outputs(messages)
                values = [{"tool_call_id": message.tool_call_id, "name": message.name, "content": message.content} for message in messages]
            snapshot = client.runtime_resources()
            rf_memory = memory(rf.pid)
            if rf_memory.get("rss_mib") is None:
                raise RuntimeError("RF memory measurement unavailable")
            resource_ram = float(snapshot.get("bge-m3-model", {}).get("host_ram_mib") or 0) + rf_memory["rss_mib"]
            rows.append({**part, "prepare_and_reclaim_s": prepare_s, "production_tool_s": tool_s,
                         "elapsed_to_result_s": last_result, "pressure_flushed": flushed,
                         "outputs": values, "output_signature": canonical(values), "resources": snapshot,
                         "rf_memory": rf_memory, "sampled_managed_ram_mib": resource_ram,
                         "sample_exceeds_managed_ram_budget": resource_ram > calibration["managed_host_ram_mib"]})
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        after = client.runtime_resources()
        rf_counts = {"starts": rf.starts, "stops_before_final_cleanup": rf.stops}
        events = [{"event_type": event.event_type, "capability_id": event.capability_id, "detail": event.detail} for event in governor.events]
        try:
            rf.close()
            governor.close()
        finally:
            conductivity_tool._MODEL = previous_model
            release_remote(client)
    return {"arm": arm, "ok": error is None and len(rows) == len(parts), "error": error,
            "elapsed_to_last_result_s": last_result if error is None else None,
            "rows": rows, "root_counter_deltas": counter_delta(before, after), "governor_events": events,
            "rf_counts": rf_counts,
            "outputs_signature": canonical([row["outputs"] for row in rows]),
            "native_cached_budget_is_observed_not_enforced": arm == "native_cached"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--service-port", type=int, default=8814)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = workspace(args.output)
    if output.exists():
        raise FileExistsError("Use a fresh output directory; archived runs are immutable")
    calibration_path, model = workspace(args.calibration), workspace(args.model)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    traces = [load_trace(path) for path in args.source]
    plan = {"schema": "cwsr-fixed-pressure-replay-v1", "traces": traces,
            "calibration": calibration, "calibration_path": str(calibration_path), "calibration_sha256": sha(calibration_path),
            "model": str(model), "model_sha256": sha(model), "script_sha256": sha(__file__),
            "source_fingerprints": source_fingerprints(),
            "orders": [list(ARMS), list(reversed(ARMS))],
            "scope": "Production resource-bearing tools only; no LLM or complete report regeneration",
            "batch_rule": "Preserve original parallel batches when they fit; split overfull batches identically in every arm",
            "time_rule": "Cold managed resources per arm; first prepare through last tool result; intermediate reclamation and common observations included, final cleanup excluded",
            "output_rule": "Exact ordered tool-call IDs, names and complete ToolMessage content; no volatile field exclusions"}
    plan["parts"] = [execution_parts(trace, calibration) for trace in traces]
    output.mkdir(parents=True)
    dump(output / "plan.json", plan)
    if not args.execute:
        print(json.dumps({"status": "PLAN_ONLY", "sources": len(traces), "calls": [trace["managed_call_count"] for trace in traces]}))
        return
    from orion.benchmark.cwsr_complete_case_resource_v1 import dedicated_service
    results = []
    with dedicated_service(output, args.service_port) as url:
        client = bind_isolated_service(url)
        # 先只读健康检查；工具只走服务现有的检索与资源端点。
        dump(output / "service-preflight.json", client.get("/health"))
        for source_index, (trace, parts) in enumerate(zip(traces, plan["parts"])):
            for order_index, order in enumerate(plan["orders"]):
                for arm in order:
                    result = run_arm(trace, parts, arm, client, model, calibration)
                    result.update(source_index=source_index, order_index=order_index)
                    results.append(result)
                    dump(output / f"source-{source_index}-order-{order_index}-{arm}.json", result)
                    print(json.dumps({"source": source_index, "order": order_index, "arm": arm,
                                      "ok": result["ok"], "seconds": result["elapsed_to_last_result_s"],
                                      "root_counts": result["root_counter_deltas"]}), flush=True)
    checks = []
    for index, trace in enumerate(traces):
        group = [result for result in results if result["source_index"] == index]
        checks.append({"source_index": index, "all_completed": all(result["ok"] for result in group),
                       "all_outputs_identical": len({result["outputs_signature"] for result in group}) == 1,
                       "source_unchanged": sha(trace["path"]) == trace["sha256"]})
    frozen = sha(model) == plan["model_sha256"] and sha(calibration_path) == plan["calibration_sha256"] and source_fingerprints() == plan["source_fingerprints"]
    dump(output / "summary.json", {"checks": checks, "inputs_and_script_unchanged": frozen,
         "results": [{key: result[key] for key in ("source_index", "order_index", "arm", "ok", "elapsed_to_last_result_s", "root_counter_deltas", "error")} for result in results]})
    if not frozen or not all(all(value for key, value in check.items() if key != "source_index") for check in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
