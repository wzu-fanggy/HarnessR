"""Production wiring for statically declared tool runtime resources.

This module does not choose tools.  It only maps an already selected LangGraph tool
call to resource instances owned by the residency governor.
"""

from __future__ import annotations

import atexit
import math
import os
from dataclasses import dataclass
from pathlib import Path

from orion.capvm import (
    CapabilityCatalog,
    CapabilityKind,
    CapabilityResidencyGovernor,
    ResidencyLevel,
    VirtualCapability,
    physical_resource_budget,
)
from orion.core.langgraph_residency import (
    RuntimeResourceRequirement,
    ToolCallResidencyBridge,
    ToolResidencyBinding,
)
from orion.capvm.adapters import (
    LocalPluginRuntimeStateAdapter,
    RootServiceResourceAdapter,
)
from orion.capvm.model import ResourceVector
from orion.capvm.resource_observation import ResourceObservationRecorder
from orion.capvm.shared_runtime import SharedResourceRuntime
from tools.rag.service_client import get_root_kg_service_client
from tools.electrolyte_design.conductivity_process import (
    SodiumConductivityMlProcessAdapter,
)
from tools.electrolyte_design.conductivity_tool import (
    CONDUCTIVITY_PROCESS_RESOURCE_ID,
    ensure_serialized_public_model,
)
from tools.electrolyte_design.temperature_fit_process import (
    TemperatureFitProcessAdapter,
)
from tools.electrolyte_design.temperature_fit_tool import (
    TRANSPORT_FIT_PROCESS_RESOURCE_ID,
)


from tools.electrolyte_design.robust_ranking_process import (
    RobustRankingProcessAdapter, DEFAULT_DECLARED_HOST_RAM_MB,
)
from tools.electrolyte_design.robust_ranking_tool import ROBUST_RANKING_PROCESS_RESOURCE_ID


CONDUCTIVITY_TOOL_NAME = "predict_formulation_conductivity"
TEMPERATURE_FIT_TOOL_NAME = "fit_conductivity_temperature_law"

ROOT_BGE_RESOURCE_ID = "root-rag:bge-m3-model"
ROOT_NEO4J_RESOURCE_ID = "root-rag:neo4j-session"
LOCAL_PLUGIN_RUNTIME_RESOURCE_ID = "local-capability:plugin-runtime-state"

_MODEL_SESSION_TOOLS = frozenset(
    {
        "search_literature_chunks",
        "search_paper_chunks",
        "search_literature_multiview",
        "verify_parameter_claim",
    }
)
_SESSION_ONLY_TOOLS = frozenset(
    {
        "resolve_literature_papers",
        "explore_entity_neighborhood",
        "aggregate_evidence_by_entities",
        "kg_query",
        "query_formulation_space",
        "enrich_formulation_evidence",
    }
)


def _nonnegative_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    value = float(raw) if raw else float(default)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


@dataclass
class DefaultToolResourceRuntime:
    """Own the production governor and its ToolNode bridge."""

    governor: CapabilityResidencyGovernor
    bridge: ToolCallResidencyBridge
    recorder: ResourceObservationRecorder
    plugin_adapter: LocalPluginRuntimeStateAdapter | None = None
    profile_store: object | None = None
    _closed: bool = False

    def snapshot(self) -> tuple[dict[str, object], ...]:
        """Return current read-only resource lifecycle state."""

        return self.governor.resource_observations()

    def sync_plugin_runtime(self) -> None:
        """Align externally attached local plugins with the governor residency state."""

        adapter = self.plugin_adapter
        if self._closed or adapter is None:
            return
        state = self.governor.state(LOCAL_PLUGIN_RUNTIME_RESOURCE_ID)
        if adapter.has_plugins():
            # approve()/hot-swap mounts a plugin before this hook runs. Reflect that
            # physical state in the shared governor without changing user approval.
            if state.current_level == ResidencyLevel.CATALOG and adapter.any_mounted():
                self.governor.ensure_many(
                    ((LOCAL_PLUGIN_RUNTIME_RESOURCE_ID, ResidencyLevel.BOUND),),
                    request_id="local-plugin-attach",
                )
            if state.current_level > ResidencyLevel.CATALOG:
                self.governor.reconcile_measured_budget()
        elif state.current_level > ResidencyLevel.CATALOG:
            self.governor.demote(LOCAL_PLUGIN_RUNTIME_RESOURCE_ID)

    def close(self) -> None:
        if self._closed:
            return
        shared = self.bridge.shared_runtime
        if shared is not None:
            pending = shared.snapshot()
            if pending["active"] or pending["waiting"]:
                raise RuntimeError("runtime has pending work; await aclose()")
        try:
            self.recorder.record_all(
                "runtime_close_snapshot",
                governor=self.governor,
                request_id="runtime-close",
            )
        finally:
            try:
                if shared is not None:
                    shared.close()
                else:
                    self.governor.close()
            finally:
                if self.profile_store is not None:
                    self.profile_store.close()
            self._closed = True
            from tools.local_capabilities import set_local_capability_runtime_sync_hook

            set_local_capability_runtime_sync_hook(None)

    async def aclose(self) -> None:
        """共享准入模式先排空活动调用；普通模式沿用同步关闭。"""
        if self._closed:
            return
        if self.bridge.shared_runtime is None:
            self.close()
            return
        await self.bridge.shared_runtime.aclose()
        if self.profile_store is not None:
            self.profile_store.close()
        self._closed = True
        from tools.local_capabilities import set_local_capability_runtime_sync_hook
        set_local_capability_runtime_sync_hook(None)


def build_default_tool_resource_runtime() -> DefaultToolResourceRuntime | None:
    """Build the normal Agent runtime-resource pool without materializing resources."""

    if os.environ.get("ORION_RESOURCE_POOL_ENABLED", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return None

    coordination_enabled = os.environ.get("ORION_RESOURCE_COORDINATION", "0") == "1"
    task_intent_enabled = os.environ.get("ORION_RESOURCE_TASK_INTENT", "0") == "1"
    if task_intent_enabled and not coordination_enabled:
        raise ValueError('任务意图接线需要启用资源协调')
    if coordination_enabled and (os.environ.get("ORION_RESOURCE_ADMISSION_QUEUE", "0") != "1"
            or not os.environ.get("ORION_RESOURCE_PROFILE_PATH", "").strip()):
        raise ValueError("resource coordination requires shared admission and a profile path")

    root_service_client = get_root_kg_service_client()
    bge_model = RootServiceResourceAdapter(
        client=root_service_client,
        remote_resource_id="bge-m3-model",
        declared_cost=ResourceVector(
            host_ram_mb=_nonnegative_env(
                "ORION_BGE_DECLARED_RAM_MB",
                558.13671875,
            ),
            gpu_vram_mb=_nonnegative_env("ORION_BGE_DECLARED_GPU_MB", 2300.0),
        ),
    )
    neo4j_session = RootServiceResourceAdapter(
        client=root_service_client,
        remote_resource_id="neo4j-session",
        declared_cost=ResourceVector(connection_slots=1.0),
    )

    def _plugin_registry():
        from tools.local_capabilities import get_local_capability_manager

        return get_local_capability_manager().host.registry

    plugin_runtime = LocalPluginRuntimeStateAdapter(_plugin_registry)

    explicit_model_path=os.environ.get("ORION_NA_CONDUCTIVITY_MODEL_PATH", "").strip()
    conductivity_process = SodiumConductivityMlProcessAdapter(
        model_path=Path(explicit_model_path).expanduser() if explicit_model_path else None,
        model_path_factory=None if explicit_model_path else ensure_serialized_public_model,
        declared_host_ram_mb=_nonnegative_env(
            "ORION_CONDUCTIVITY_DECLARED_RAM_MB",
            384.0,
        ),
    )
    temperature_fit_process = TemperatureFitProcessAdapter(
        declared_host_ram_mb=_nonnegative_env(
            "ORION_TRANSPORT_FIT_DECLARED_RAM_MB",
            192.0,
        ),
    )
    from tools.electrolyte_design.robust_ranking import resolve_bundle_dir
    ranking_process = RobustRankingProcessAdapter(
        bundle_dir=resolve_bundle_dir(),
        declared_host_ram_mb=_nonnegative_env(
            "ORION_ROBUST_RANKING_DECLARED_RAM_MB", DEFAULT_DECLARED_HOST_RAM_MB),
    )
    catalog = CapabilityCatalog(
        (
            VirtualCapability(
                capability_id=ROOT_BGE_RESOURCE_ID,
                kind=CapabilityKind.MODEL,
                summary="root-service BGE-M3 embedding model",
                adapter=bge_model,
                maximum_level=ResidencyLevel.BOUND,
            ),
            VirtualCapability(
                capability_id=ROOT_NEO4J_RESOURCE_ID,
                kind=CapabilityKind.SESSION,
                summary="root-service Neo4j connection/session resource",
                adapter=neo4j_session,
                maximum_level=ResidencyLevel.BOUND,
            ),
            VirtualCapability(
                capability_id=LOCAL_PLUGIN_RUNTIME_RESOURCE_ID,
                kind=CapabilityKind.PLUGIN,
                summary="approved local-capability plugin runtime state",
                adapter=plugin_runtime,
                maximum_level=ResidencyLevel.BOUND,
            ),
            VirtualCapability(
                capability_id=CONDUCTIVITY_PROCESS_RESOURCE_ID,
                kind=CapabilityKind.PROCESS,
                summary="sodium-electrolyte conductivity prediction worker",
                adapter=conductivity_process,
                maximum_level=ResidencyLevel.BOUND,
            ),
            VirtualCapability(
                capability_id=ROBUST_RANKING_PROCESS_RESOURCE_ID,
                kind=CapabilityKind.PROCESS,
                summary="稳健排序集成模型工作进程",
                adapter=ranking_process,
                maximum_level=ResidencyLevel.BOUND,
            ),
            VirtualCapability(
                capability_id=TRANSPORT_FIT_PROCESS_RESOURCE_ID,
                kind=CapabilityKind.PROCESS,
                summary="electrolyte conductivity temperature-law fitting worker",
                adapter=temperature_fit_process,
                maximum_level=ResidencyLevel.BOUND,
            ),
        )
    )
    budget = physical_resource_budget(
        host_ram_mb=_nonnegative_env("ORION_RESOURCE_HOST_RAM_MB", 1024.0),
        gpu_vram_mb=_nonnegative_env("ORION_RESOURCE_GPU_VRAM_MB", 4096.0),
        process_slots=_nonnegative_env("ORION_RESOURCE_PROCESS_SLOTS", 1.0),
        connection_slots=_nonnegative_env("ORION_RESOURCE_CONNECTION_SLOTS", 1.0),
    )
    # 共享准入默认启用整机检查；历史非共享运行须显式开启。
    from orion.capvm.host_memory import HostMemoryGuard, PreparationReservation
    guard_enabled = os.environ.get("ORION_RESOURCE_HOST_GUARD",
        os.environ.get("ORION_RESOURCE_ADMISSION_QUEUE", "0")) == "1"
    guard = HostMemoryGuard(
        physical_floor_mb=_nonnegative_env("ORION_RESOURCE_PHYSICAL_FLOOR_MB", 2048.0),
        commit_floor_mb=_nonnegative_env("ORION_RESOURCE_COMMIT_FLOOR_MB", 4096.0),
        preparation_multiplier=_nonnegative_env("ORION_RESOURCE_PREPARE_MULTIPLIER", 2.0),
        reservations={ROOT_BGE_RESOURCE_ID: PreparationReservation(
            # 2026-11-04 单线程冷加载采样：服务总私有内存 6658MiB、工作集 3145MiB。
            # 向上取整并保留余量；不是增量峰值的精确标定，也不改变驻留声明。
            physical_mb=_nonnegative_env("ORION_BGE_PREPARE_PHYSICAL_MB", 4096.0),
            commit_mb=_nonnegative_env("ORION_BGE_PREPARE_COMMIT_MB", 8192.0),
        )},
    ) if guard_enabled else None
    reserve_policy=None
    adaptive_reserve=os.environ.get('ORION_RESOURCE_ADAPTIVE_RESERVE','0')=='1'
    if adaptive_reserve:
        if guard is None or os.environ.get('ORION_RESOURCE_AUTO_MEMORY','0')!='1' or os.environ.get('ORION_RESOURCE_ADMISSION_QUEUE','0')!='1':
            raise ValueError('adaptive reserve requires host guard, automatic memory and shared admission')
        from orion.capvm.host_memory import windows_host_memory
        from orion.capvm.adaptive_reserve import AdaptiveReserve
        _,capacity=windows_host_memory(include_capacity=True)
        reserve_policy=AdaptiveReserve(capacity)
        guard.reserve_policy=reserve_policy
    auto_memory = None
    if os.environ.get("ORION_RESOURCE_AUTO_MEMORY", "0") == "1":
        if guard is None or os.environ.get("ORION_RESOURCE_ADMISSION_QUEUE", "0") != "1":
            raise ValueError("automatic memory requires shared admission and host guard")
        from orion.capvm.adaptive_memory import AdaptiveMemoryBudget
        raw_limit = os.environ.get("ORION_RESOURCE_HOST_RAM_MB", "").strip()
        auto_memory = AdaptiveMemoryBudget(
            physical_floor_mb=guard.physical_floor_mb, commit_floor_mb=guard.commit_floor_mb,
            user_limit_mb=_nonnegative_env("ORION_RESOURCE_HOST_RAM_MB", 1024.0) if raw_limit else None,
            reserve_policy=reserve_policy)
    governor = CapabilityResidencyGovernor(catalog=catalog, budget=budget,
        preparation_guard=guard)
    profile_store = None
    profile_path = os.environ.get("ORION_RESOURCE_PROFILE_PATH", "").strip()
    if profile_path:
        from orion.capvm.profile_writer import BufferedProfileStore
        from orion.capvm.profile_identity import ProcessIdentityResolver
        from orion.capvm.process_metrics import process_memory_snapshot
        from orion.capvm.resource_profiles import LifecycleProfiler
        profile_store = BufferedProfileStore(Path(profile_path).expanduser())
        resolver = ProcessIdentityResolver()
        local_ids = {CONDUCTIVITY_PROCESS_RESOURCE_ID, TRANSPORT_FIT_PROCESS_RESOURCE_ID,
                     ROBUST_RANKING_PROCESS_RESOURCE_ID}
        profiler = LifecycleProfiler(profile_store, {}, resource_ids=local_ids,
            identity_provider=resolver.resolve,
            process_sampler=lambda cap: process_memory_snapshot(cap.adapter.pid))
        governor.lifecycle_profiler = profiler
        # 只恢复可解析的本地资源身份，不启动模型工厂、不把远程服务归给单个工具。
        for resource_id in sorted(local_ids):
            profiler.begin(catalog.get(resource_id))
    raw_event_path = os.environ.get("ORION_RESOURCE_EVENTS_PATH", "").strip()
    recorder = ResourceObservationRecorder(
        Path(raw_event_path).expanduser() if raw_event_path else None
    )
    bindings: dict[str, ToolResidencyBinding] = {}
    for tool_name in _MODEL_SESSION_TOOLS:
        bindings[tool_name] = ToolResidencyBinding(
            resources=(
                RuntimeResourceRequirement(ROOT_BGE_RESOURCE_ID, ResidencyLevel.BOUND),
                RuntimeResourceRequirement(ROOT_NEO4J_RESOURCE_ID, ResidencyLevel.BOUND),
            )
        )
    for tool_name in _SESSION_ONLY_TOOLS:
        bindings[tool_name] = ToolResidencyBinding(
            resources=(
                RuntimeResourceRequirement(ROOT_NEO4J_RESOURCE_ID, ResidencyLevel.BOUND),
            )
        )
    bindings["invoke_local_capability"] = ToolResidencyBinding(
        resources=(
            RuntimeResourceRequirement(
                LOCAL_PLUGIN_RUNTIME_RESOURCE_ID,
                ResidencyLevel.BOUND,
            ),
        )
    )
    bindings[CONDUCTIVITY_TOOL_NAME] = ToolResidencyBinding(
        resources=(
            RuntimeResourceRequirement(
                CONDUCTIVITY_PROCESS_RESOURCE_ID,
                ResidencyLevel.BOUND,
            ),
        )
    )
    bindings[TEMPERATURE_FIT_TOOL_NAME] = ToolResidencyBinding(
        resources=(
            RuntimeResourceRequirement(
                TRANSPORT_FIT_PROCESS_RESOURCE_ID,
                ResidencyLevel.BOUND,
            ),
        )
    )

    bindings["rank_formulations_under_uncertainty"] = ToolResidencyBinding(
        resources=(RuntimeResourceRequirement(ROBUST_RANKING_PROCESS_RESOURCE_ID),),
    )
    # 子代理包装器保留独立工具名，但使用同一份模型/连接依赖声明。
    bindings["async_search_kg_rag_hybrid"] = bindings["search_literature_chunks"]

    bridge = ToolCallResidencyBridge(
        governor=governor,
        bindings=bindings,
        strict=False,
        reconcile_measured=True,
        # Detailed lease-boundary snapshots require extra root-service HTTP calls.
        # They are only on the timed path when event collection was requested.
        observation_recorder=recorder if raw_event_path else None,
        # 显式启用共享事件循环的排队准入，不改变历史实验的默认拒绝语义。
        shared_runtime=(SharedResourceRuntime(governor, memory_budget=auto_memory,
            host_pressure_reclaim_ids=(CONDUCTIVITY_PROCESS_RESOURCE_ID,TRANSPORT_FIT_PROCESS_RESOURCE_ID,ROBUST_RANKING_PROCESS_RESOURCE_ID) if adaptive_reserve else ())
            if os.environ.get("ORION_RESOURCE_ADMISSION_QUEUE", "0") == "1" else None),
    )
    if coordination_enabled:
        from orion.capvm.demand_coordinator import DemandCoordinator
        # 默认使用实测成本；任务意图与其API额度均须另外显式启用。
        requirement_registry=None
        if os.environ.get("ORION_RESOURCE_REQUIREMENT_COVERAGE", "0") == "1":
            from orion.core.scientific_requirement_facts import scientific_requirement_registry
            requirement_registry=scientific_requirement_registry()
        task_intents = None
        if task_intent_enabled:
            from orion.core.task_board_intent import TaskBoardIntentSource
            from orion.capvm.task_intent import TaskIntentController
            maximum = int(os.environ.get('ORION_RESOURCE_INTENT_JEV_MAX_CALLS', '0'))
            if maximum < 0:
                raise ValueError('任务意图API额度不能为负')
            assessor = None
            if maximum:
                from orion.capvm.jev_task_intent import JevTaskIntentAssessor
                assessor = JevTaskIntentAssessor(os.environ.get('TYPESAFE_API_KEY', ''),
                                                max_calls=maximum, ttl_s=300)
            mapped = {name: tuple(r for r,_ in governor.expand_requirements(binding.requirements()))
                      for name,binding in bindings.items()}
            descriptions = {cap.capability_id: {'description': cap.summary,
                'tools': sorted(name for name,resources in mapped.items() if cap.capability_id in resources)}
                for cap in governor.catalog.all()}
            task_intents = TaskIntentController(TaskBoardIntentSource(), mapped, descriptions,
                assessor=assessor, selection_rule=os.environ.get('ORION_RESOURCE_INTENT_RULE','dominance'))
        DemandCoordinator(bridge.shared_runtime, bindings, governor.lifecycle_profiler,
                          active=True, bundle_mode=True, requirement_registry=requirement_registry,
                          task_intents=task_intents,
                          intent_prefetch=os.environ.get('ORION_RESOURCE_INTENT_PREFETCH') == '1')

    runtime = DefaultToolResourceRuntime(
        governor=governor,
        bridge=bridge,
        recorder=recorder,
        plugin_adapter=plugin_runtime,
        profile_store=profile_store,
    )
    from tools.local_capabilities import set_local_capability_runtime_sync_hook

    set_local_capability_runtime_sync_hook(runtime.sync_plugin_runtime)
    atexit.register(runtime.close)
    return runtime
