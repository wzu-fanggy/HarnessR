"""完整电导率评估任务的资源对照。默认只做独立资源测量，不调用 LLM。

对照将两臂的 RF 均放在同一种缓存进程中。cached_process 臂持续保留，
runtime 臂使用正常 ToolNode 与资源运行时。问题、工具及科学判断保持一致。
该对照补充原生部署对照，不将管理预算误称为操作系统内存限制。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from orion.benchmark import cwsr_native_runtime_pair_v1 as base

SELF = Path(__file__).resolve()
_original_snapshot = base.source_snapshot
_original_environment = base.worker_environment


def snapshot():
    values = _original_snapshot()
    values[str(SELF.relative_to(ROOT.parent)).replace("\\", "/")] = base.sha(SELF)
    original_script = ROOT / "benchmark/cwsr_native_runtime_pair_v1.py"
    values[str(original_script.relative_to(ROOT.parent)).replace("\\", "/")] = base.sha(original_script)
    return values


def environment(arm, output, service_url, settings):
    values = _original_environment(arm, output, service_url, settings)
    measured = settings["calibration"]
    values.update({
        "ORION_BGE_DECLARED_RAM_MB": str(measured["bge_declared_mib"]),
        "ORION_CONDUCTIVITY_DECLARED_RAM_MB": str(measured["rf_declared_mib"]),
        "ORION_RESOURCE_PROCESS_SLOTS": "4", "ORION_RESOURCE_CONNECTION_SLOTS": "8",
    })
    return values


def configure_base():
    # 复用已验证的服务生命周期与冻结核查，子进程进入本文件的新 worker。
    base.source_snapshot = snapshot
    base.worker_environment = environment
    base.__file__ = str(SELF)


@contextmanager
def dedicated_service(output, port):
    """独立只读 KG 服务，只结束已核实属于本次启动器的进程。"""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT.parent), PYTHONIOENCODING="utf-8")
    worker_pid = None
    with (output / "calibration-service.stdout.log").open("w", encoding="utf-8") as stdout, \
         (output / "calibration-service.stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen([str(base.ROOT_PYTHON), "-m", "uvicorn", "kg_rag.service:app",
                                    "--host", "127.0.0.1", "--port", str(port)],
                                   cwd=ROOT.parent, env=env, stdout=stdout, stderr=stderr)
        try:
            started = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Calibration service exited")
                try:
                    base.get_json(url + "/health", timeout=1)
                    break
                except Exception:
                    if time.monotonic() - started > 60:
                        raise TimeoutError("Calibration service startup timed out")
                    time.sleep(0.25)
            worker_pid = int(base.get_json(url + "/v1/resource-residency")["pid"])
            if not base.owned_worker(worker_pid, process.pid):
                raise RuntimeError("Calibration PID ownership check failed")
            yield url
        finally:
            if process.poll() is None:
                if worker_pid and worker_pid != process.pid and base.owned_worker(worker_pid, process.pid):
                    try:
                        os.kill(worker_pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)


def memory(pid):
    if pid is None:
        return {"rss_mib": 0.0, "private_commit_mib": 0.0}
    # 使用 Windows 原生计数器，不为此次实验安装额外依赖。
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile", "private")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x0400 | 0x0010, False, int(pid))
    if not handle:
        return {"rss_mib": None, "private_commit_mib": None, "error": "OpenProcessFailed"}
    try:
        value = Counters()
        value.cb = ctypes.sizeof(value)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(value), value.cb):
            return {"rss_mib": None, "private_commit_mib": None, "error": "GetProcessMemoryInfoFailed"}
        return {"rss_mib": value.working_set / 2**20, "private_commit_mib": value.private / 2**20}
    finally:
        kernel.CloseHandle(handle)


def summarize_memory(samples):
    """合计本任务三个所属进程的读数，不将读数缺失计作零占用。"""
    from statistics import median
    summary = {"scope": "Agent + dedicated RAG service + RF worker; excludes Neo4j and whole-machine RAM",
               "rss_is_sum_of_working_sets_not_unique_physical_memory": True,
               "private_commit_is_not_resident_ram": True, "samples": len(samples)}
    for field in ("rss_mib", "private_commit_mib"):
        complete = []
        for row in samples:
            values = [row[name].get(field) for name in ("agent", "root_service", "rf_worker")]
            if all(value is not None for value in values):
                complete.append(sum(values))
        summary[field] = {"valid_samples": len(complete), "peak": max(complete) if complete else None,
                          "median": median(complete) if complete else None}
    phases = {}
    for row in samples:
        resources = (row.get("root_resources") or {}).get("resources", {})
        bge = bool((resources.get("bge-m3-model") or {}).get("resident"))
        rf = row["rf_worker"].get("rss_mib")
        if rf is None or "resource_error" in row:
            continue
        label = "both" if bge and rf > 0 else "bge_only" if bge else "rf_only" if rf > 0 else "neither"
        phases[label] = phases.get(label, 0) + 1
    summary["observed_residency_phase_samples"] = phases
    return summary


def measure_calibration(output, model, port):
    from tools.rag.service_client import RootKgServiceClient
    from tools.electrolyte_design.conductivity_process import SodiumConductivityMlProcessAdapter
    from orion.capvm import ResidencyLevel
    from tools.electrolyte_design.conductivity_tool import _explicit_features
    inputs = json.loads(base.CONDUCTIVITY_CASE.read_text(encoding="utf-8"))
    rows = []
    with dedicated_service(output, port) as url:
        client = RootKgServiceClient(base_url=url)
        for repeat in range(3):
            before = client.get("/v1/resource-residency")
            bge = client.acquire_runtime_resource("bge-m3-model")
            loaded = client.get("/v1/resource-residency")
            rf = SodiumConductivityMlProcessAdapter(model_path=model)
            try:
                rf.promote(ResidencyLevel.CATALOG, ResidencyLevel.BOUND)
                predictions = []
                for candidate in inputs["candidates"]:
                    features = _explicit_features(candidate, temperature_c=0, override=None)
                    predictions.append(rf.predict_features(vars(features)))
                simultaneous = client.get("/v1/resource-residency")
                rf_reading = memory(rf.pid)
                released = client.release_runtime_resource("bge-m3-model")
                after_release = client.get("/v1/resource-residency")
                rows.append({"repeat": repeat + 1, "before": before, "bge_resource": bge,
                             "loaded": loaded, "rf_memory": rf_reading,
                             "simultaneous": simultaneous, "released_resource": released,
                             "after_release": after_release, "predictions": predictions})
            finally:
                rf.close()
    bge_peak = max(row["bge_resource"]["host_ram_mib"] for row in rows)
    rf_peak = max(row["rf_memory"]["rss_mib"] for row in rows)
    if not (bge_peak > 0 and rf_peak > 0):
        raise RuntimeError("Invalid resource measurements; do not proceed to paid Agent runs")
    # 在最大单项与两项合计之间取值，测量先于任务结果；不是 OS 强制上限。
    bge_declared = math.ceil(bge_peak * 1.15 + 16)
    rf_declared = math.ceil(rf_peak * 1.15 + 16)
    budget = math.ceil(max(bge_declared, rf_declared) + 0.30 * min(bge_declared, rf_declared))
    result = {"schema": "cwsr_complete_case_calibration_v1", "rows": rows,
              "bge_declared_mib": bge_declared, "rf_declared_mib": rf_declared,
              "managed_host_ram_mib": budget,
              "rule": "ceil(max(declared) + 0.30*min(declared)); declared=ceil(1.15*independent_peak+16)",
              "scope": "BGE load-attributed service RSS plus complete RF worker RSS; excludes service base, Agent and Neo4j",
              "not_os_memory_limit": True, "synthetic_buffers": False}
    base.dump(output / "calibration.json", result)
    return result


def worker(arm, output, url, plan):
    """正常主 Agent，自主决定调用顺序；新增部分只统一部署与记录资源。"""
    base.verify_plan(plan)
    os.environ.update(environment(arm, output, url, plan["settings"]))
    with tempfile.TemporaryDirectory(prefix="agent-state-", dir=output, ignore_cleanup_errors=True) as tmp:
        os.environ["ORION_EVOLUTION_STORE"] = str(Path(tmp) / "empty-evolution")
        previous = Path.cwd()
        os.chdir(tmp)
        adapter = None
        stop = threading.Event()
        samples = []
        observer = None
        try:
            from langchain_core.messages import AIMessage, ToolMessage
            from orion.evaluation.models import EvaluationSample, ToolCallRecord
            import orion.evaluation.agent_runner as runner
            from orion.benchmark.self_evolution_fresh_v2_development_runtime import _V2IsolatedContextManager
            from tools.electrolyte_design import conductivity_tool
            from tools.electrolyte_design.conductivity_process import (
                ConductivityProcessPredictor, SodiumConductivityMlProcessAdapter)
            from orion.capvm import ResidencyLevel
            from orion.core.default_resource_runtime import CONDUCTIVITY_PROCESS_RESOURCE_ID
            main = importlib.import_module("orion.main")
            runtime = main._default_resource_runtime
            if (runtime is not None) != (arm == "runtime"):
                raise RuntimeError("Runtime switch did not take effect")
            if runtime:
                adapter = runtime.governor.catalog.get(CONDUCTIVITY_PROCESS_RESOURCE_ID).adapter
                if plan["settings"].get("split_overfull_batches"):
                    from orion.core.resource_batch_execution import ResourceBudgetedToolBridge
                    previous_bridge = runtime.bridge
                    runtime.bridge = ResourceBudgetedToolBridge(
                        governor=previous_bridge.governor, bindings=previous_bridge.bindings,
                        strict=previous_bridge.strict, reconcile_measured=previous_bridge.reconcile_measured,
                        observation_recorder=previous_bridge.observation_recorder)
                    main.set_tool_residency_bridge(runtime.bridge)
            else:
                adapter = SodiumConductivityMlProcessAdapter(model_path=Path(plan["settings"]["conductivity_model_path"]))
                class LazyCachedPredictor(ConductivityProcessPredictor):
                    def predict_features(self, features):
                        if not self.adapter.is_running:
                            self.adapter.promote(ResidencyLevel.CATALOG, ResidencyLevel.BOUND)
                        return super().predict_features(features)
                conductivity_tool._MODEL = LazyCachedPredictor(adapter)
            main._context_manager = _V2IsolatedContextManager(memory_service=main._memory_service,
                                                            model_name="cwsr-complete-case")
            exposed = {tool.name: tool for tool in main._BASE_TOOLS}
            main._BASE_TOOLS = tuple(exposed[name] for name in plan["tasks"]["T1"]["allowed_tools"])
            main._configure_runtime_tools()
            messages = []
            def extract(items):
                messages.extend(message.model_dump(mode="json") for message in items)
                results = {message.tool_call_id: message for message in items if isinstance(message, ToolMessage)}
                return [ToolCallRecord(name=call["name"], args=call.get("args") or {},
                                       result=results[call["id"]].content if call["id"] in results else None)
                        for message in items if isinstance(message, AIMessage)
                        for call in message.tool_calls or []]
            runner.extract_tool_calls = extract
            events = []
            def sink(event):
                events.append(event)
                with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            service_pid = base.get_json(url + "/v1/resource-residency")["pid"]
            def observe():
                while not stop.is_set():
                    row = {"time": time.monotonic(), "agent": memory(os.getpid()),
                           "root_service": memory(service_pid), "rf_worker": memory(adapter.pid)}
                    try:
                        row["root_resources"] = base.get_json(url + "/v1/runtime/resources", timeout=2)
                    except Exception as exc:
                        row["resource_error"] = type(exc).__name__
                    samples.append(row)
                    stop.wait(0.5)
            observer = threading.Thread(target=observe, daemon=True)
            observer.start()
            sample = EvaluationSample(question=plan["tasks"]["T1"]["question"], task_type="design",
                                      ground_truth="", expected_tools=[], benchmark_case_id="CWSR_T1_COMMON_PROCESS")
            raw = runner.run_agent(sample, config=SimpleNamespace(auto_approve=True),
                                   agent_factory=lambda: main._build_graph(main.llm_async, rebuild_memory=False),
                                   event_sink=sink)
            raw["tool_calls"] = [record.model_dump(mode="json") for record in raw.get("tool_calls", [])]
            raw.update(arm=arm, task="T1", messages=messages, sample=sample.model_dump(mode="json"),
                       runtime_snapshot_before_close=list(runtime.snapshot()) if runtime else [],
                       root_resources_before_close=base.get_json(url + "/v1/runtime/resources"),
                       root_telemetry_before_close=base.get_json(url + "/v1/resource-residency"),
                       timing=base.summarize_timing(events, raw["run_duration_seconds"]),
                       common_prediction_process=True, runtime_enabled_observed=runtime is not None)
            base.dump(output / "result.json", raw)
            (output / "report.md").write_text(raw.get("answer") or "", encoding="utf-8")
        finally:
            stop.set()
            if observer:
                observer.join(timeout=5)
            base.dump(output / "all-owned-resource-samples.json", samples)
            base.dump(output / "memory-summary.json", summarize_memory(samples))
            if adapter:
                adapter.close()
            if "main" in locals() and main._default_resource_runtime:
                main._default_resource_runtime.close()
            os.chdir(previous)
    base.verify_plan(plan)
    base.dump(output / "freeze_check.json", {"source_unchanged": True, "input_unchanged": True})


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-artifact")
    parser.add_argument("--service-port", type=int, default=8811)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--split-overfull-batches", action="store_true")
    parser.add_argument("--calibration-input")
    parser.add_argument("--runtime-first", action="store_true")
    parser.add_argument("--worker-task", help=argparse.SUPPRESS)
    parser.add_argument("--worker-arm", choices=base.ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--service-url", help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure_base()
    output = base.workspace_path(args.output)
    if args.worker_task:
        worker(args.worker_arm, output, args.service_url,
               json.loads((output.parent / "manifest.json").read_text(encoding="utf-8")))
        return
    output.mkdir(parents=True, exist_ok=True)
    model = base.workspace_path(args.model_artifact)
    calibration_path = output / "calibration.json"
    if args.calibration_input and not calibration_path.exists():
        calibration_source = base.workspace_path(args.calibration_input)
        calibration_path.write_bytes(calibration_source.read_bytes())
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() \
        else measure_calibration(output, model, args.service_port)
    plan = base.make_plan(["T1"], args.pairs, 600, "deepseek-v4-flash", str(model))
    if args.runtime_first:
        for pair in plan["schedule"]:
            pair["arm_order"] = pair["arm_order"][::-1]
    plan["settings"]["calibration"] = calibration
    plan["settings"]["split_overfull_batches"] = args.split_overfull_batches
    plan["settings"]["calibration_sha256"] = base.sha(calibration_path)
    plan["settings"]["runtime_managed_budget"] = {
        "host_ram_mib": calibration["managed_host_ram_mib"], "gpu_vram_mib": 6144,
        "process_slots": 4, "connection_slots": 8}
    plan["comparison"].update(native="同一 RF 子进程首次使用后持续缓存；不添加逐次释放",
                              common_process_deployment=True, original_native_reference="cwsr_conductivity_native_runtime_canary_20261006_v10",
                              primary_outcome="实测进程 RSS 和私有提交内存；任务完成时间作为并列观测",
                              model_and_task_paths_unchanged=True)
    base.dump(output / "manifest.json", plan)
    print(json.dumps({"stage": "calibrated", "budget_mib": calibration["managed_host_ram_mib"],
                      "declared_bge_mib": calibration["bge_declared_mib"],
                      "declared_rf_mib": calibration["rf_declared_mib"]}), flush=True)
    if not args.execute:
        return
    observations = []
    for pair in plan["schedule"]:
        for arm in pair["arm_order"]:
            arm_output = output / f"{pair['pair_id']}-{arm}"
            if arm_output.exists():
                raise RuntimeError("Do not overwrite or silently reuse prior arms")
            if base.sha(calibration_path) != plan["settings"]["calibration_sha256"]:
                raise RuntimeError("Independent calibration changed")
            observation = base.run_arm("T1", arm, arm_output, plan, args.service_port)
            observation["memory"] = json.loads((arm_output / "memory-summary.json").read_text(encoding="utf-8"))
            observation["display_arm"] = "cached_process" if arm == "native" else "runtime"
            observations.append(observation)
            base.dump(output / "progress.json", observations)
            print(json.dumps({"stage": "arm_finished", "arm": arm, "ok": observations[-1]["ok"]}), flush=True)
            if not observation["ok"]:
                base.dump(output / "incomplete-summary.json", {"observations": observations,
                          "stopped_after_failure": True, "remaining_arms_not_run": True})
                raise RuntimeError("Full task failed; retained artifacts require inspection before additional paid runs")
    base.dump(output / "summary.json", {"observations": observations, "plan_sha256": base.sha(output / "manifest.json"),
                                       "interpretation": "Live plans may differ; completion alone is not runtime advantage"})


if __name__ == "__main__":
    main_cli()
