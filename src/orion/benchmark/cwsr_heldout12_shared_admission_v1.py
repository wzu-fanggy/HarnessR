"""仅用于冻结留出实验的共同预算检查及 Agent worker 包装，不修改生产工具。"""
from __future__ import annotations

import functools
import importlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

from orion.benchmark import cwsr_native_runtime_pair_v1 as base
from orion.benchmark import cwsr_complete_case_resource_v1 as complete


class SharedAdmissionGate:
    """同一预计占用预算；原生组检查但不释放，运行时组使用已有回收策略。

    检查点位于 ToolNode 实际工具函数入口：运行时组先经其资源租借及回收，
    原生组则直接检查缓存状态及并发请求的预计驻留资源。
    此门禁仅判断主机内存预算，不自行调用任何资源的释放操作。
    """

    def __init__(self, arm: str, service_url: str, settings: dict):
        self.arm = arm
        self.url = service_url
        self.budget = float(settings["runtime_managed_budget"]["host_ram_mib"])
        self.bge_cost = float(settings["calibration"]["bge_declared_mib"])
        self.rf_cost = float(settings["calibration"]["rf_declared_mib"])
        self.lock = threading.Lock()
        # 基线只有一个缓存预测进程，避免多个 LLM 并行请求竞态启动同一实例。
        self.prediction_lock = threading.Lock()
        self.reserved_bge = False
        self.reserved_rf = False
        self.adapter = None
        self.records: list[dict] = []

    @staticmethod
    def dependencies(tool_name: str) -> tuple[bool, bool]:
        from orion.core.default_resource_runtime import _MODEL_SESSION_TOOLS, CONDUCTIVITY_TOOL_NAME
        return tool_name in _MODEL_SESSION_TOOLS, tool_name == CONDUCTIVITY_TOOL_NAME

    def inspect(self, tool_name: str) -> None:
        needs_bge, needs_rf = self.dependencies(tool_name)
        if not (needs_bge or needs_rf):
            return
        root = base.get_json(self.url + "/v1/runtime/resources", timeout=12)
        bge_status = root.get("resources", {}).get("bge-m3-model", {})
        resident_bge = bool(bge_status.get("resident"))
        resident_rf = bool(self.adapter is not None and self.adapter.is_running)
        with self.lock:
            expected_bge = resident_bge or needs_bge or (self.arm == "native" and self.reserved_bge)
            expected_rf = resident_rf or needs_rf or (self.arm == "native" and self.reserved_rf)
            projected = self.bge_cost * expected_bge + self.rf_cost * expected_rf
            admitted = projected <= self.budget + 1e-9
            record = {
                "tool": tool_name, "arm": self.arm,
                "need_bge": needs_bge, "need_rf": needs_rf,
                "observed_bge_resident": resident_bge, "observed_rf_running": resident_rf,
                "planned_bge_resident": bool(expected_bge), "planned_rf_resident": bool(expected_rf),
                "projected_mib": projected, "budget_mib": self.budget,
                "admitted": admitted, "time_monotonic": time.monotonic(),
            }
            self.records.append(record)
            if not admitted:
                raise RuntimeError(
                    f"COMMON_CAPACITY_DENIED:{tool_name}:projected={projected:.1f}"
                    f":budget={self.budget:.1f}; no resource was released"
                )
            if self.arm == "native":
                # 记住本批尚未完成的首次加载，避免并行 ToolNode 竞态漏掉容量。
                self.reserved_bge = expected_bge
                self.reserved_rf = expected_rf

    def wrap(self, tool):
        needs_bge, needs_rf = self.dependencies(tool.name)
        if not (needs_bge or needs_rf):
            return tool
        wrapped = tool.model_copy()
        tool_name = tool.name
        if tool.func is not None:
            original_func = tool.func

            @functools.wraps(original_func)
            def checked_func(*args, **kwargs):
                from orion.core.default_resource_runtime import CONDUCTIVITY_TOOL_NAME
                if self.arm == "native" and tool_name == CONDUCTIVITY_TOOL_NAME:
                    with self.prediction_lock:
                        self.inspect(tool_name)
                        return original_func(*args, **kwargs)
                self.inspect(tool_name)
                return original_func(*args, **kwargs)

            wrapped.func = checked_func
        if getattr(tool, "coroutine", None) is not None:
            original_coroutine = tool.coroutine

            @functools.wraps(original_coroutine)
            async def checked_async(*args, **kwargs):
                self.inspect(tool_name)
                return await original_coroutine(*args, **kwargs)

            wrapped.coroutine = checked_async
        return wrapped


def correct_heldout_sample(sample):
    """只纠正旧 T1 runner 的任务分类，保留同一科学问题与模型输入。"""
    if sample.benchmark_case_id == "CWSR_T1_COMMON_PROCESS":
        if sample.task_type != "design":
            raise ValueError("预期需要修正的旧 T1 任务类型已变化")
        sample.source_task_type = "benchmark_T1_default_design"
        sample.task_type = "candidate_screening"
        sample.benchmark_case_id = "CWSR_HELDOUT12_CONDUCTIVITY_SCREENING"
    return sample


def run_worker(arm: str, output: Path, url: str, plan: dict) -> None:
    """复用既有完整 Agent worker，纠正任务类型并插入对称预算观察。"""
    base.verify_plan(plan)
    os.environ.update(complete.environment(arm, output, url, plan["settings"]))
    gate = SharedAdmissionGate(arm, url, plan["settings"])

    # 与既有 worker 一样，在导入 Agent 前创建隔离的记忆目录。
    with tempfile.TemporaryDirectory(prefix="heldout-worker-root-", dir=output, ignore_cleanup_errors=True) as tmp:
        original_cwd = Path.cwd()
        os.environ["ORION_EVOLUTION_STORE"] = str(Path(tmp) / "isolated-evolution")
        os.chdir(tmp)
        try:
            main = importlib.import_module("orion.main")
        finally:
            os.chdir(original_cwd)
        original_configure = main._configure_runtime_tools
        runner = importlib.import_module("orion.evaluation.agent_runner")
        original_run_agent = runner.run_agent

        @functools.wraps(original_run_agent)
        def routed_run_agent(sample, *args, **kwargs):
            return original_run_agent(correct_heldout_sample(sample), *args, **kwargs)

        runner.run_agent = routed_run_agent

        def monitored_configure():
            from orion.core.default_resource_runtime import CONDUCTIVITY_PROCESS_RESOURCE_ID
            from tools.electrolyte_design import conductivity_tool
            if arm == "runtime":
                gate.adapter = main._default_resource_runtime.governor.catalog.get(
                    CONDUCTIVITY_PROCESS_RESOURCE_ID
                ).adapter
            else:
                gate.adapter = conductivity_tool._MODEL.adapter
            main._BASE_TOOLS = tuple(gate.wrap(tool) for tool in main._BASE_TOOLS)
            return original_configure()

        main._configure_runtime_tools = monitored_configure
        try:
            complete.worker(arm, output, url, plan)
        finally:
            main._configure_runtime_tools = original_configure
            runner.run_agent = original_run_agent
            base.dump(output / "shared-admission-checks.json", {
                "arm": arm, "budget_mib": gate.budget,
                "bge_declared_mib": gate.bge_cost,
                "rf_declared_mib": gate.rf_cost, "checks": gate.records,
                "refusal_count": sum(not record["admitted"] for record in gate.records),
                "reclaimer_in_this_checker": False,
            })
