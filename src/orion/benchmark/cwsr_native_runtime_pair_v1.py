"""原八题的原有智能体与工具资源运行时对照。默认只生成方案，不访问服务或模型。

两组均使用原来的工具缓存、同一问题、工具集及科学判断流程。
每次执行使用独立 Agent 进程与独立根 RAG 服务进程，避免共享缓存污染。
"""
from __future__ import annotations
from release_paths import release_path

import argparse
from collections import Counter
import hashlib
import importlib
from functools import wraps
import json
import os
import signal
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace
import urllib.request

# 安装到 benchmark 后由文件位置确定项目根目录。
ROOT = Path(__file__).resolve().parents[1]
if ROOT.name != "orion":
    ROOT = Path(release_path('${ORION_SOURCE}'))
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from orion.benchmark import cwsr_eight_task_followup_v1 as original
from orion.benchmark import cwsr_eight_task_autonomous_frozen_v1 as frozen

ARMS = ("native", "runtime")
CONDUCTIVITY_CASE = ROOT / "benchmark/cwsr_conductivity_case_input_v1.json"
TASKS = (*original.ORDER, "T1")
ROOT_PYTHON = Path(sys.executable)
AGENT_PYTHON = Path(sys.executable)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def diagnostic_terminal_tool(tool, output):
    """保留报告编译异常及其输入，不改变返回、判断或重试行为。"""
    wrapped = tool.model_copy()
    original_func = tool.func
    @wraps(original_func)
    def invoke(*args, **kwargs):
        try:
            return original_func(*args, **kwargs)
        except Exception:
            path = output / "terminal-tool-error.json"
            if not path.exists():
                from langchain_core.load import dumps
                dump(path, {"traceback": traceback.format_exc(),
                            "serialized_args": dumps({"args": args, "kwargs": kwargs})})
            raise
    wrapped.func = invoke
    return wrapped


def workspace_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT.parent):
        raise ValueError("Output must stay within ${PROJECT_SOURCE}")
    return path


def source_snapshot():
    hashes = {"orion/" + k: v for k, v in frozen.snapshot().items()}
    for path in (ROOT.parent / "kg_rag").rglob("*.py"):
        hashes[str(path.relative_to(ROOT.parent)).replace("\\", "/")] = sha(path)
    for path in (ROOT / "capvm").rglob("*.py"):
        hashes[str(path.relative_to(ROOT.parent)).replace("\\", "/")] = sha(path)
    for path in (Path(__file__), ROOT / "benchmark/self_evolution_fresh_v2_development_runtime.py"):
        hashes[str(path.relative_to(ROOT.parent)).replace("\\", "/")] = sha(path)
    hashes[str(CONDUCTIVITY_CASE.relative_to(ROOT.parent)).replace("\\", "/")] = sha(CONDUCTIVITY_CASE)
    return dict(sorted(hashes.items()))


def selected_tasks():
    tasks = original.selected()
    tasks["T1"] = (CONDUCTIVITY_CASE, json.loads(CONDUCTIVITY_CASE.read_text(encoding="utf-8")))
    return tasks


def allowed_tools(task):
    profile = json.loads(original.PROFILE.read_text(encoding="utf-8"))
    allowed = set(profile["allowed_tool_names"])
    if task == "T1":
        allowed.add("predict_formulation_conductivity")
    if allowed & set(profile["blocked_side_effect_or_out_of_scope_tool_names"]):
        raise RuntimeError("Unsafe tool profile")
    return sorted(allowed)


def worker_environment(arm, output, service_url, settings):
    if arm not in ARMS:
        raise ValueError(arm)
    # 两臂只有开关及记录路径不同，不替换缓存函数、不添加逐次释放。
    values = {
        "ORION_AGENT_PROVIDER": "deepseek", "ORION_AGENT_MODEL": settings["model"],
        "ORION_AGENT_TEMPERATURE": "0.0", "ORION_AGENT_TOP_P": "1.0",
        "ORION_AGENT_THINKING": "enabled", "ORION_DESIGN_REASONING_MODE": "semantic",
        "ORION_GROUNDING_CHECK": "0", "ORION_LLM_CACHE": "0", "ORION_LLM_STREAM": "0",
        "ORION_REFLECTION_INJECT": "0", "ORION_ABLATION_NO_KG": "0",
        "ORION_ABLATION_NO_FORMULATION": "0",
        "ORION_RESOURCE_POOL_ENABLED": "1" if arm == "runtime" else "0",
        "ORION_RESOURCE_EVENTS_PATH": str(Path(output) / "runtime-events.jsonl"),
        "ORION_RAG_SERVICE_URL": service_url,
        "ORION_EVAL_AGENT_TIMEOUT": str(settings["timeout_seconds"]),
        "ORION_EVAL_LLM_STEP_TIMEOUT": "90", "ORION_EVAL_RECURSION_LIMIT": "100",
        "ORION_EVAL_LLM_REQUEST_MAX_RETRIES": "2", "ORION_LLM_REQUEST_MAX_RETRIES": "2",
        "ORION_RESOURCE_HOST_RAM_MB": str(settings["runtime_managed_budget"]["host_ram_mib"]),
        "ORION_RESOURCE_GPU_VRAM_MB": str(settings["runtime_managed_budget"]["gpu_vram_mib"]),
        "ORION_RESOURCE_PROCESS_SLOTS": "1", "ORION_RESOURCE_CONNECTION_SLOTS": "1",
    }
    if settings.get("conductivity_model_path"):
        values["ORION_NA_CONDUCTIVITY_MODEL_PATH"] = settings["conductivity_model_path"]
    return values


def make_plan(tasks, pairs, timeout, model, conductivity_model_path=None):
    if pairs < 1 or timeout <= 0 or not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Tasks must be unique; pairs and timeout must be positive")
    selected = selected_tasks()
    if set(tasks) - set(selected):
        raise ValueError("Unknown original task")
    schedule = []
    for repeat in range(1, pairs + 1):
        for index, task in enumerate(tasks):
            arms = ARMS if (repeat + index) % 2 else ARMS[::-1]
            schedule.append({"pair_id": f"{task}-r{repeat}", "task": task,
                             "repeat": repeat, "arm_order": list(arms)})
    settings = {"model": model, "timeout_seconds": timeout,
                "runtime_managed_budget": {"host_ram_mib": 1024, "gpu_vram_mib": 4096,
                                           "process_slots": 1, "connection_slots": 1}}
    if "T1" in tasks:
        if not conductivity_model_path:
            raise ValueError("T1 requires an explicitly prepared conductivity model")
        artifact = workspace_path(conductivity_model_path)
        settings.update({"conductivity_model_path": str(artifact), "conductivity_model_sha256": sha(artifact)})
    return {
        "schema_version": "cwsr_native_runtime_pair_v2", "settings": settings,
        "tasks": {task: {"question": selected[task][1]["sample"]["question"],
                         "raw_path": str(selected[task][0]), "raw_sha256": sha(selected[task][0]),
                         "task_role": "supporting_conductivity_case" if task == "T1" else "original_eight",
                         "allowed_tools": allowed_tools(task)}
                  for task in tasks},
        "source_sha256": source_snapshot(), "tool_profile_sha256": sha(original.PROFILE),
        "schedule": schedule, "planned_live_runs": len(schedule) * 2,
        "comparison": {
            "native": "原有 Agent 直接调用工具，保持工具原本的模型及数据库连接缓存",
            "runtime": "同一 Agent 通过正常 ToolNode 工具资源运行时调用同一工具",
            "independent_processes_per_arm": True, "fresh_root_service_per_arm": True,
            "os_file_cache_flushed": False, "external_provider_cache_controlled": False,
            "forced_tool_sequence": False, "llm_fixed_seed": False,
            "development_questions": True, "chemical_quality_requires_source_audit": True,
            "native_has_no_runtime_admission_budget": True,
            "managed_budget_is_not_os_memory_limit": True,
            "profile_exposes_conductivity_prediction": "T1" in tasks,
            "preserve_timeouts_and_unsuccessful_runs": True,
        },
    }


def verify_plan(plan):
    if source_snapshot() != plan["source_sha256"]:
        raise RuntimeError("Source changed after plan freeze")
    current = selected_tasks()
    for task, expected in plan["tasks"].items():
        path, row = current[task]
        if sha(path) != expected["raw_sha256"] or row["sample"]["question"] != expected["question"]:
            raise RuntimeError("Original input changed: " + task)
    if plan.get("settings", {}).get("conductivity_model_path"):
        if sha(plan["settings"]["conductivity_model_path"]) != plan["settings"]["conductivity_model_sha256"]:
            raise RuntimeError("Conductivity model artifact changed")


def interval_union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Invalid interval")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def summarize_timing(events, wall):
    intervals = {"model": [], "tool": []}
    starts, incomplete = {}, []
    kinds = {"on_chat_model": "model", "on_llm": "model", "on_tool": "tool"}
    for event in events:
        label = event.get("event", "")
        prefix, _, suffix = label.rpartition("_")
        if prefix not in kinds or suffix not in {"start", "end", "error"}:
            continue
        key = (prefix, event["run_id"])
        at = float(event["elapsed_seconds"])
        if at < 0 or at > wall + 0.01:
            raise ValueError("Event outside task timing interval")
        if suffix == "start":
            starts[key] = at
        elif key in starts:
            intervals[kinds[prefix]].append((starts.pop(key), min(at, wall)))
        else:
            incomplete.append({"event": label, "run_id": event["run_id"]})
    for (prefix, run_id), at in starts.items():
        intervals[kinds[prefix]].append((at, wall))
        incomplete.append({"event": prefix + "_start_without_end", "run_id": run_id})
    model = interval_union(intervals["model"])
    tool = interval_union(intervals["tool"])
    combined = interval_union(intervals["model"] + intervals["tool"])
    return {"task_wall_seconds": wall, "model_request_union_seconds": model,
            "tool_function_union_seconds": tool,
            "model_tool_overlap_seconds": max(0.0, model + tool - combined),
            "outside_model_and_tool_seconds": max(0.0, wall - combined),
            "incomplete_events": incomplete,
            "model_request_includes_network_and_provider_wait": True,
            "outside_intervals_is_not_a_resource_preparation_measurement": True}


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def parent_pid(pid):
    """只读取得进程父 ID，识别 Windows venv 启动器后的真实解释器。"""
    if os.name != "nt":
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    import ctypes
    from ctypes import wintypes
    class BasicInfo(ctypes.Structure):
        _fields_ = [("exit_status", wintypes.LONG), ("peb", ctypes.c_void_p),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.LONG),
                    ("pid", ctypes.c_size_t), ("parent", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), "Cannot inspect owned service process")
    try:
        info = BasicInfo()
        query = ctypes.WinDLL("ntdll").NtQueryInformationProcess
        query.argtypes = [wintypes.HANDLE, wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG, ctypes.c_void_p]
        query.restype = wintypes.LONG
        if query(handle, 0, ctypes.byref(info), ctypes.sizeof(info), None) != 0:
            raise OSError("Cannot query service process parent")
        return int(info.parent)
    finally:
        kernel.CloseHandle(handle)


def owned_worker(worker_pid, launcher_pid):
    seen = set()
    while worker_pid > 0 and worker_pid not in seen and len(seen) < 32:
        if worker_pid == launcher_pid:
            return True
        seen.add(worker_pid)
        worker_pid = parent_pid(worker_pid)
    return False


def worker(task, arm, output, service_url, plan):
    verify_plan(plan)
    os.environ.update(worker_environment(arm, output, service_url, plan["settings"]))
    with tempfile.TemporaryDirectory(prefix="agent-state-", dir=output, ignore_cleanup_errors=True) as tmp:
        os.environ["ORION_EVOLUTION_STORE"] = str(Path(tmp) / "empty-evolution")
        previous_cwd = Path.cwd()
        os.chdir(tmp)
        try:
            from langchain_core.messages import AIMessage, ToolMessage
            from orion.evaluation.models import EvaluationSample, ToolCallRecord
            import orion.evaluation.agent_runner as runner
            from orion.benchmark.self_evolution_fresh_v2_development_runtime import _V2IsolatedContextManager
            main = importlib.import_module("orion.main")
            enabled = main._default_resource_runtime is not None
            if enabled != (arm == "runtime"):
                raise RuntimeError("Runtime switch did not take effect before import")
            main._context_manager = _V2IsolatedContextManager(memory_service=main._memory_service,
                                                            model_name="cwsr-native-runtime")
            profile = json.loads(original.PROFILE.read_text(encoding="utf-8"))
            allowed = set(plan["tasks"][task]["allowed_tools"])
            exposed = {tool.name: tool for tool in main._BASE_TOOLS}
            if allowed & set(profile["blocked_side_effect_or_out_of_scope_tool_names"]) or allowed - set(exposed):
                raise RuntimeError("Tool profile unavailable or unsafe")
            main._BASE_TOOLS = tuple(exposed[name] for name in sorted(allowed))
            main._BASE_TOOLS = tuple(diagnostic_terminal_tool(tool, output)
                                    if tool.name == "synthesize_design_report" else tool
                                    for tool in main._BASE_TOOLS)
            main._configure_runtime_tools()
            messages = []

            def extract_by_id(items):
                messages.extend(message.model_dump(mode="json") for message in items)
                results = {message.tool_call_id: message for message in items if isinstance(message, ToolMessage)}
                return [ToolCallRecord(name=call["name"], args=call.get("args") or {},
                                       result=results[call["id"]].content if call["id"] in results else None)
                        for message in items if isinstance(message, AIMessage)
                        for call in message.tool_calls or []]

            runner.extract_tool_calls = extract_by_id
            sample = EvaluationSample(question=plan["tasks"][task]["question"], task_type="design",
                                      ground_truth="", expected_tools=[],
                                      benchmark_case_id=selected_tasks()[task][1]["case_id"])
            events = []
            def sink(event):
                events.append(event)
                with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            try:
                raw = runner.run_agent(sample, config=SimpleNamespace(auto_approve=True),
                                       agent_factory=lambda: main._build_graph(main.llm_async, rebuild_memory=False),
                                       event_sink=sink)
                raw["tool_calls"] = [record.model_dump(mode="json") for record in raw.get("tool_calls", [])]
                raw.update({"arm": arm, "task": task, "messages": messages,
                            "sample": sample.model_dump(mode="json")})
                # 观测端点失败也保留已经完成的任务与模型失败记录。
                dump(output / "result.json", raw)
                runtime = main._default_resource_runtime
                raw["runtime_snapshot_before_close"] = list(runtime.snapshot()) if runtime else []
                raw["root_resources_before_close"] = get_json(service_url + "/v1/runtime/resources")
                raw["root_telemetry_before_close"] = get_json(service_url + "/v1/resource-residency")
                raw["runtime_enabled_observed"] = enabled
                raw["timing"] = summarize_timing(events, raw["run_duration_seconds"])
                raw["arm"] = arm
                raw["task"] = task
                raw["messages"] = messages
                raw["sample"] = sample.model_dump(mode="json")
                raw["run_environment"] = worker_environment(arm, output, service_url, plan["settings"])
                dump(output / "result.json", raw)
                (output / "report.md").write_text(raw.get("answer") or "", encoding="utf-8")
            finally:
                if main._default_resource_runtime is not None:
                    main._default_resource_runtime.close()
        finally:
            os.chdir(previous_cwd)
    verify_plan(plan)
    dump(output / "freeze_check.json", {"source_unchanged": True, "input_unchanged": True})


def run_arm(task, arm, output, plan, port):
    verify_plan(plan)
    output.mkdir()
    # 检查端口占用，不对现有 8799 服务执行释放或重启操作。
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    url = f"http://127.0.0.1:{port}"
    stopped = threading.Event()
    service_env = os.environ.copy()
    service_env["PYTHONPATH"] = str(ROOT.parent)
    service_env["PYTHONIOENCODING"] = "utf-8"
    samples = []
    service = None
    service_worker_pid = None
    poller = None
    with (output / "service.stdout.log").open("w", encoding="utf-8") as stdout,\
         (output / "service.stderr.log").open("w", encoding="utf-8") as stderr:
        try:
            started = time.monotonic()
            service = subprocess.Popen([str(ROOT_PYTHON), "-m", "uvicorn", "kg_rag.service:app",
                                        "--host", "127.0.0.1", "--port", str(port)],
                                       cwd=ROOT.parent, env=service_env, stdout=stdout, stderr=stderr)
            while True:
                if service.poll() is not None:
                    raise RuntimeError("Dedicated root RAG service exited; inspect service logs")
                try:
                    get_json(url + "/health", timeout=1)
                    break
                except Exception:
                    if time.monotonic() - started > 60:
                        raise TimeoutError("Dedicated service startup timed out")
                    stopped.wait(0.25)
            telemetry = get_json(url + "/v1/resource-residency")
            if not owned_worker(int(telemetry["pid"]), service.pid):
                raise RuntimeError("Service PID does not belong to this experiment")
            service_worker_pid = int(telemetry["pid"])
            metadata = get_json(url + "/v1/metadata", timeout=30)
            initial = get_json(url + "/v1/runtime/resources", timeout=60)
            if any(row["resident"] for row in initial["resources"].values()):
                raise RuntimeError("Dedicated service is not the expected cold worker")
            dump(output / "service-preflight.json", {"metadata": metadata, "initial_resources": initial,
                                                     "telemetry": telemetry,
                                                     "service_setup_seconds": time.monotonic() - started,
                                                     "service_setup_excluded_from_task_timer": True})
            def observe():
                while not stopped.is_set():
                    try:
                        samples.append({"sampled_at_monotonic": time.monotonic(),
                                        "telemetry": get_json(url + "/v1/resource-residency", timeout=2)})
                    except Exception as exc:
                        samples.append({"error_type": type(exc).__name__})
                    stopped.wait(0.5)
            poller = threading.Thread(target=observe, daemon=True)
            poller.start()
            with (output / "agent.stdout.log").open("w", encoding="utf-8") as agent_stdout,\
                 (output / "agent.stderr.log").open("w", encoding="utf-8") as agent_stderr:
                process = subprocess.run([str(AGENT_PYTHON), str(Path(__file__).resolve()), "--execute",
                                          "--worker-task", task, "--worker-arm", arm,
                                          "--output", str(output), "--service-url", url],
                                         cwd=ROOT, stdout=agent_stdout, stderr=agent_stderr,
                                         timeout=plan["settings"]["timeout_seconds"] + 180)
            if process.returncode:
                raise RuntimeError("Agent worker failed; inspect retained logs")
        finally:
            stopped.set()
            if poller is not None:
                poller.join(timeout=5)
            dump(output / "service-rss-samples.json", samples)
            if service is not None and service.poll() is None:
                if service_worker_pid is not None and service_worker_pid != service.pid:
                    # 只终止已经核实归属于本轮启动器的真实服务进程。
                    if owned_worker(service_worker_pid, service.pid):
                        try:
                            os.kill(service_worker_pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                service.terminate()
                try:
                    service.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    service.kill()
                    service.wait(timeout=15)
    verify_plan(plan)
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    readings = [row["telemetry"]["process_rss_mib"] for row in samples if "telemetry" in row]
    readings = [value for value in readings if value is not None]
    artifact = result.get("task_artifact") or {}
    return {"arm": arm, "ok": result["ok"], "task_status": artifact.get("task_status"),
            "timing": result["timing"], "tool_counts": dict(Counter(call["name"] for call in result["tool_calls"])),
            "root_resources": result["root_resources_before_close"],
            "root_service_sampled_peak_rss_mib": max(readings) if readings else None,
            "rss_scope": "root RAG service only; excludes Agent, Neo4j server and whole-machine RAM",
            "quality_verified": False, "result_path": str(output / "result.json")}


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(original.ORDER))
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=480)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--conductivity-model")
    parser.add_argument("--service-port", type=int, default=8809)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker-task", choices=TASKS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--service-url", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = workspace_path(args.output)
    if args.worker_task:
        if not args.execute or not args.worker_arm or not args.service_url:
            raise ValueError("Worker invocation incomplete")
        plan = json.loads((output.parent / "manifest.json").read_text(encoding="utf-8"))
        worker(args.worker_task, args.worker_arm, output, args.service_url, plan)
        return
    if output.exists():
        raise FileExistsError("Historical output must not be overwritten")
    if args.execute and Path(sys.executable).resolve() != AGENT_PYTHON.resolve():
        raise ValueError("Execution requires the Orion project Python")
    plan = make_plan(args.tasks, args.pairs, args.timeout, args.model, args.conductivity_model)
    output.mkdir(parents=True)
    dump(output / "manifest.json", plan)
    print(json.dumps({"planned_live_runs": plan["planned_live_runs"], "execute": args.execute,
                      "manifest": str(output / "manifest.json")}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    completed = []
    for pair in plan["schedule"]:
        observations = {}
        for arm in pair["arm_order"]:
            print(f"START {pair['pair_id']} {arm}", flush=True)
            observations[arm] = run_arm(pair["task"], arm, output / f"{pair['pair_id']}-{arm}", plan, args.service_port)
            dump(output / "collection-progress.json", {"complete_pairs": completed, "current_pair": pair,
                                                       "current_observations": observations})
            # 运行错误不补成功、不替换任务，保留并停止后续付费请求。
            if not observations[arm]["ok"]:
                raise RuntimeError("Task did not finish normally; audit saved result before more paid runs")
        native = json.loads(Path(observations["native"]["result_path"]).read_text(encoding="utf-8"))
        runtime = json.loads(Path(observations["runtime"]["result_path"]).read_text(encoding="utf-8"))
        signatures = lambda row: [(call["name"], call["args"]) for call in row["tool_calls"]]
        completed.append({**pair, "arms": observations,
                          "tool_call_plan_equal": signatures(native) == signatures(runtime),
                          "observed_native_minus_runtime_seconds": native["run_duration_seconds"] - runtime["run_duration_seconds"],
                          "causal_speedup_verified": False, "chemical_quality_verified": False})
        dump(output / "summary.json", {"complete_pairs": completed, "collection_complete": False})
    verify_plan(plan)
    dump(output / "summary.json", {"complete_pairs": completed, "collection_complete": True,
                                  "source_and_inputs_unchanged": True})


if __name__ == "__main__":
    main_cli()
