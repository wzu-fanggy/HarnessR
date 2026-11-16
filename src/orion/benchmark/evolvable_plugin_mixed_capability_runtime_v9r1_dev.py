"""V9R1：异构真实 backing capability + ORION scientific plugin 的 routing development bank。

该 benchmark 仍是 development-only：不调用 provider/KG/RAG，也不读取 V8B 82-query
confirmatory set。与 V9 synthetic factorial 不同，本轮把真实 ORION scientific LangGraph
plugin、Tool/Skill progressive disclosure、真实 child process 与真实 socket session 放入同一
catalog，并把 single routing、authority fallback、semantic recovery、multi-capability joint
admission 和 learned-state refault 放进同一个运行时。
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from orion.benchmark.evolvable_plugin_scientific_integration_v5 import (
    _build_plugin,
    _propose_policy,
    build_scientific_cases,
)
from orion.capvm import (
    CapabilityCatalog,
    CapabilityControlPlane,
    CapabilityKind,
    CapabilityResidencyGovernor,
    CascadedDiscoveryBackend,
    EmbeddingDiscoveryBackend,
    EvolvableCapabilityRouter,
    ExactTriggerDiscoveryBackend,
    MetadataDiscoveryBackend,
    RecoveryAction,
    ResidencyLevel,
    ResourceVector,
    RoutedCapabilityRuntime,
    RoutingRequest,
    VirtualCapability,
)
from orion.capvm.adapters import (
    ProcessCapabilityAdapter,
    SessionCapabilityAdapter,
    SkillCapabilityAdapter,
    ToolCapabilityAdapter,
)


SCIENTIFIC_ID = "orion-evidence-composition"
EVIDENCE_TOOL_ID = "literature-evidence-tool"
DESIGN_SKILL_ID = "electrolyte-design-skill"
PROCESS_ID = "molecular-simulation-worker"
SESSION_ID = "mcp-scientific-session"
EXPORT_TOOL_ID = "filesystem-report-export"
DRAFT_SKILL_ID = "report-drafting-skill"
DISTRACTOR_COUNT = 41
SHORTLIST_K = 6


def _routing_metadata(
    *,
    terms: Sequence[str],
    required_level: ResidencyLevel,
    triggers: Sequence[str] = (),
    authorities: Sequence[str] = (),
) -> dict[str, object]:
    return {
        "version": "v9r1-dev",
        "routing_terms": tuple(terms),
        "routing_triggers": tuple(triggers),
        "routing_required_level": required_level.name,
        "required_authorities": tuple(authorities),
    }


def _tool(
    capability_id: str,
    summary: str,
    *,
    terms: Sequence[str],
    interface_text: str,
    authorities: Sequence[str] = (),
) -> VirtualCapability:
    return VirtualCapability(
        capability_id=capability_id,
        kind=CapabilityKind.TOOL,
        summary=summary,
        adapter=ToolCapabilityAdapter(
            interface_text=interface_text,
            reasoning_text=f"Use {capability_id} only when its declared contract is relevant.",
        ),
        metadata=_routing_metadata(
            terms=terms,
            required_level=ResidencyLevel.INTERFACE,
            authorities=authorities,
        ),
    )


def _skill(
    capability_id: str,
    summary: str,
    *,
    terms: Sequence[str],
    index_text: str,
    instruction_text: str,
) -> VirtualCapability:
    return VirtualCapability(
        capability_id=capability_id,
        kind=CapabilityKind.SKILL,
        summary=summary,
        adapter=SkillCapabilityAdapter(
            index_text=index_text,
            instruction_text=instruction_text,
        ),
        metadata=_routing_metadata(
            terms=terms,
            required_level=ResidencyLevel.REASONING,
        ),
    )


def _concept_vector(text: str) -> tuple[float, ...]:
    """Development-only concept encoder，用于验证 embedding-stage contract。

    它不是性能 claim，也不是生产 embedding。正式系统可以把同一接口替换成 bge-m3 或
    provider/native retrieval；这里避免引入模型/网络调用，同时保留真正的向量 cosine path。
    """

    normalized = text.casefold()
    groups = (
        # evidence/literature
        ("evidence", "literature", "paper", "papers", "report", "reports", "observed", "experimental"),
        # applicability/context validity
        ("applicability", "applicable", "applies", "valid", "validity", "context", "condition", "conditions", "conflict", "judge", "qualify"),
        # electrolyte design
        ("design", "formulation", "recipe", "plan", "electrolyte", "formulate"),
        # local process/simulation
        ("simulation", "simulate", "molecular", "compute", "worker", "process", "engine"),
        # live session/service
        ("mcp", "session", "connection", "remote", "service", "socket", "live"),
        # report/export/draft
        ("report", "export", "draft", "write", "document"),
        # generic distractor axis
        ("archive", "calendar", "format", "conversion", "meeting"),
    )
    return tuple(float(sum(term in normalized for term in group)) for group in groups)


def _encoder(texts: Sequence[str]) -> Sequence[Sequence[float]]:
    return tuple(_concept_vector(text) for text in texts)


@dataclass(frozen=True)
class SingleTask:
    task_id: str
    family: str
    text: str
    expected_capability_id: str | None
    expected_recovery: RecoveryAction = RecoveryAction.NONE
    authorities: frozenset[str] = frozenset()
    scientific_case_id: str | None = None
    expected_scientific_status: str | None = None


@dataclass(frozen=True)
class CompositionTask:
    task_id: str
    family: str
    text: str
    expected_capability_ids: frozenset[str]
    scientific_case_id: str | None = None
    expected_scientific_status: str | None = None


def _round_robin_single_tasks() -> tuple[SingleTask, ...]:
    families: tuple[tuple[str, tuple[str, ...], str | None, RecoveryAction, str | None, str | None], ...] = (
        (
            "scientific_semantic",
            (
                "judge whether an observed voltage result still applies when operating conditions conflict",
                "qualify prior experimental observations under a changed cell context",
                "decide contextual validity of a measured battery result under conflicting conditions",
                "is the previous observation applicable after the experimental setting changes",
            ),
            SCIENTIFIC_ID,
            RecoveryAction.NONE,
            "temperature_conflict",
            "search",
        ),
        (
            "evidence_tool",
            (
                "retrieve experimental literature evidence for sodium electrolyte behavior",
                "find battery evidence records from prior literature",
                "search experimental evidence for this electrolyte question",
                "locate literature evidence about sodium battery electrolyte performance",
            ),
            EVIDENCE_TOOL_ID,
            RecoveryAction.NONE,
            None,
            None,
        ),
        (
            "design_skill",
            (
                "plan an electrolyte formulation workflow for a sodium battery",
                "prepare a sodium electrolyte design protocol",
                "formulate a stepwise electrolyte design workflow",
                "organize the procedure for designing a battery electrolyte recipe",
            ),
            DESIGN_SKILL_ID,
            RecoveryAction.NONE,
            None,
            None,
        ),
        (
            "authority_fallback",
            (
                "write and export a scientific report",
                "prepare a report document for export",
                "draft the scientific report even if file writing is unavailable",
                "write the report content without requiring filesystem side effects",
            ),
            DRAFT_SKILL_ID,
            RecoveryAction.NONE,
            None,
            None,
        ),
        (
            "process_runtime",
            (
                "run the local molecular simulation compute worker",
                "start a molecular simulation process for electrolyte analysis",
                "use the local compute engine for a molecular simulation",
                "launch the simulation worker process for this electrolyte task",
            ),
            PROCESS_ID,
            RecoveryAction.NONE,
            None,
            None,
        ),
        (
            "session_runtime",
            (
                "open the live MCP scientific service session",
                "connect to a remote scientific tool service session",
                "establish the MCP connection for the current workflow",
                "use a live socket-backed scientific service session",
            ),
            SESSION_ID,
            RecoveryAction.NONE,
            None,
            None,
        ),
        (
            "no_match_recovery",
            (
                "optimize a quantum transition-state saddle point with a DFT optimizer",
                "perform crystallographic Rietveld refinement on diffraction peaks",
                "generate a retrosynthetic route for an organic drug target",
                "solve a protein docking pose with a structural biology engine",
            ),
            None,
            RecoveryAction.PROPOSE_CAPABILITY,
            None,
            None,
        ),
    )

    tasks: list[SingleTask] = []
    for variant in range(4):
        for family, texts, expected, recovery, case_id, expected_status in families:
            tasks.append(
                SingleTask(
                    task_id=f"{family}-{variant:02d}",
                    family=family,
                    text=texts[variant],
                    expected_capability_id=expected,
                    expected_recovery=recovery,
                    scientific_case_id=case_id,
                    expected_scientific_status=expected_status,
                )
            )
    return tuple(tasks)


def _composition_tasks() -> tuple[CompositionTask, ...]:
    scientific = (
        "retrieve experimental evidence and judge its applicability under conflicting conditions",
        "find literature evidence then qualify whether it applies to the changed experimental context",
        "search battery evidence and evaluate contextual validity before using the observation",
        "retrieve prior evidence and decide applicability for the current cell conditions",
    )
    process_session = (
        "run the molecular simulation worker and keep a live MCP service session",
        "start the local simulation process together with the remote scientific session",
        "use the compute worker while maintaining the MCP connection",
        "launch molecular simulation and a live socket-backed service session together",
    )
    tasks: list[CompositionTask] = []
    for index, text in enumerate(scientific):
        tasks.append(
            CompositionTask(
                task_id=f"scientific-composition-{index:02d}",
                family="scientific_composition",
                text=text,
                expected_capability_ids=frozenset({EVIDENCE_TOOL_ID, SCIENTIFIC_ID}),
                scientific_case_id="temperature_conflict",
                expected_scientific_status="search",
            )
        )
    for index, text in enumerate(process_session):
        tasks.append(
            CompositionTask(
                task_id=f"resource-composition-{index:02d}",
                family="resource_composition",
                text=text,
                expected_capability_ids=frozenset({PROCESS_ID, SESSION_ID}),
            )
        )
    return tuple(tasks)


def build_mixed_inventory(
    *,
    state_root: Path,
    process_runtime_overhead_mb: float = 0.0,
) -> tuple[CapabilityCatalog, Any, dict[str, Any]]:
    scientific_cases = build_scientific_cases()
    original_scientific_capability, plugin = _build_plugin(
        state_root=state_root,
        cases=scientific_cases,
    )

    # First create a validated learned state exactly through the V5 mechanism, then page it out.
    bootstrap = CapabilityResidencyGovernor(
        catalog=CapabilityCatalog((original_scientific_capability,)),
        budget=ResourceVector(prompt_tokens=128.0, host_ram_mb=3.0),
    )
    try:
        bootstrap.access(SCIENTIFIC_ID, ResidencyLevel.BOUND, request_id="v9r1-bootstrap")
        strict_update = _propose_policy(
            plugin,
            policy_mode="strict-joint",
            validation_cases=scientific_cases[:4],
            require_strict_improvement=True,
        )
        generation = int(plugin.learned_state.generation)
        bootstrap.demote(SCIENTIFIC_ID, ResidencyLevel.CATALOG)
    finally:
        bootstrap.close()

    scientific_capability = VirtualCapability(
        capability_id=SCIENTIFIC_ID,
        kind=CapabilityKind.PLUGIN,
        summary="judge scientific evidence applicability under joint experimental context",
        adapter=original_scientific_capability.adapter,
        maximum_level=ResidencyLevel.BOUND,
        metadata=_routing_metadata(
            terms=(
                "scientific",
                "evidence",
                "applicability",
                "context",
                "conditions",
                "validity",
                "conflict",
                "observation",
            ),
            required_level=ResidencyLevel.BOUND,
        )
        | {"learned_generation": generation},
    )

    evidence_tool = _tool(
        EVIDENCE_TOOL_ID,
        "retrieve experimental literature evidence without loading a full scientific plugin",
        terms=("retrieve", "experimental", "literature", "evidence", "battery", "electrolyte"),
        interface_text="search_experimental_evidence(query) -> grounded evidence records",
    )
    design_skill = _skill(
        DESIGN_SKILL_ID,
        "progressively disclosed workflow for sodium electrolyte formulation design",
        terms=("electrolyte", "formulation", "design", "workflow", "recipe", "plan"),
        index_text="electrolyte design workflow skill",
        instruction_text=(
            "Define objective and constraints; retrieve evidence gaps; propose candidates; evaluate "
            "trade-offs; record uncertainty and exit conditions."
        ),
    )
    process_adapter = ProcessCapabilityAdapter(
        resident_mb=3,
        runtime_overhead_mb=process_runtime_overhead_mb,
        startup_delay_s=0.0,
    )
    process_capability = VirtualCapability(
        capability_id=PROCESS_ID,
        kind=CapabilityKind.PROCESS,
        summary="real local child process for molecular simulation compute work",
        adapter=process_adapter,
        maximum_level=ResidencyLevel.BOUND,
        metadata=_routing_metadata(
            terms=("molecular", "simulation", "compute", "worker", "process", "local"),
            required_level=ResidencyLevel.BOUND,
        ),
    )
    session_adapter = SessionCapabilityAdapter(resident_mb=1.0)
    session_capability = VirtualCapability(
        capability_id=SESSION_ID,
        kind=CapabilityKind.SESSION,
        summary="real socket-backed MCP-like scientific service session",
        adapter=session_adapter,
        maximum_level=ResidencyLevel.BOUND,
        metadata=_routing_metadata(
            terms=("mcp", "scientific", "service", "session", "connection", "socket", "remote"),
            required_level=ResidencyLevel.BOUND,
        ),
    )
    export_tool = _tool(
        EXPORT_TOOL_ID,
        "write and export a scientific report file to the filesystem",
        terms=("write", "export", "scientific", "report", "file", "document"),
        interface_text="write_report_file(path, content) -> artifact",
        authorities=("filesystem.write",),
    )
    draft_skill = _skill(
        DRAFT_SKILL_ID,
        "draft scientific report content without filesystem authority",
        terms=("write", "draft", "scientific", "report", "document", "content"),
        index_text="scientific report drafting skill",
        instruction_text="Draft structured scientific report content without performing file writes.",
    )

    capabilities: list[VirtualCapability] = [
        scientific_capability,
        evidence_tool,
        design_skill,
        process_capability,
        session_capability,
        export_tool,
        draft_skill,
    ]
    for index in range(DISTRACTOR_COUNT):
        if index % 2:
            capabilities.append(
                _tool(
                    f"archive-tool-{index:02d}",
                    "generic archive format conversion helper",
                    terms=("archive", "format", "conversion", f"archive{index}"),
                    interface_text="convert_archive_format(input) -> output",
                )
            )
        else:
            capabilities.append(
                _skill(
                    f"calendar-skill-{index:02d}",
                    "generic calendar meeting organization skill",
                    terms=("calendar", "meeting", "schedule", f"calendar{index}"),
                    index_text="calendar helper",
                    instruction_text="Organize calendar events and meeting notes.",
                )
            )

    return (
        CapabilityCatalog(capabilities),
        plugin,
        {
            "strict_update": strict_update,
            "process_adapter": process_adapter,
            "session_adapter": session_adapter,
        },
    )


def _request(text: str, *, authorities: frozenset[str] = frozenset()) -> RoutingRequest:
    return RoutingRequest(
        task_text=text,
        authorities=authorities,
        max_candidates=SHORTLIST_K,
        minimum_matched_terms=2,
        minimum_query_coverage=0.12,
    )


def _family_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in sorted({row["family"] for row in rows}):
        subset = [row for row in rows if row["family"] == family]
        result[family] = {
            "count": len(subset),
            "success": sum(row["success"] for row in subset),
            "success_rate": sum(row["success"] for row in subset) / len(subset),
        }
    return result


def run_benchmark() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="orion-v9r1-mixed-") as tmp:
        root = Path(tmp)
        catalog, plugin, assets = build_mixed_inventory(state_root=root / "scientific-state")
        discovery = CascadedDiscoveryBackend(
            (
                ExactTriggerDiscoveryBackend(),
                MetadataDiscoveryBackend(),
                EmbeddingDiscoveryBackend(encoder=_encoder, minimum_similarity=0.68),
            )
        )
        router = EvolvableCapabilityRouter(
            catalog=catalog,
            discovery=discovery,
            control=CapabilityControlPlane(),
        )
        budget = ResourceVector(
            prompt_tokens=256.0,
            host_ram_mb=4.5,
            process_slots=1.0,
            connection_slots=1.0,
        )
        governor = CapabilityResidencyGovernor(catalog=catalog, budget=budget)
        # V9R1 intentionally preserves the pre-reconciliation runtime semantics that exposed
        # declared-fit != measured-fit. V9R2 enables measured reconciliation.
        runtime = RoutedCapabilityRuntime(
            router=router,
            governor=governor,
            reconcile_measured=False,
        )

        single_rows: list[dict[str, Any]] = []
        composition_rows: list[dict[str, Any]] = []
        try:
            for task in _round_robin_single_tasks():
                with runtime.acquire(
                    _request(task.text, authorities=task.authorities),
                    request_id=task.task_id,
                ) as lease:
                    selected = lease.materialized_capability_id
                    recovery = lease.recovery_action
                    success = (
                        selected == task.expected_capability_id
                        if task.expected_capability_id is not None
                        else selected is None and recovery == task.expected_recovery
                    )
                    execution_ok = True
                    execution_detail: dict[str, Any] = {}
                    if selected == SCIENTIFIC_ID and task.scientific_case_id is not None:
                        observed = plugin.invoke({"case_id": task.scientific_case_id})
                        execution_ok = observed["status"] == task.expected_scientific_status
                        execution_detail = {
                            "scientific_status": observed["status"],
                            "policy_mode": observed["policy_mode"],
                        }
                    elif selected == PROCESS_ID:
                        process_adapter = assets["process_adapter"]
                        execution_ok = bool(process_adapter.is_running)
                        execution_detail = {"process_running": process_adapter.is_running}
                    elif selected == SESSION_ID:
                        session_adapter = assets["session_adapter"]
                        execution_ok = bool(session_adapter.is_connected)
                        execution_detail = {"session_connected": session_adapter.is_connected}
                    success = bool(success and execution_ok)
                    single_rows.append(
                        {
                            "task_id": task.task_id,
                            "family": task.family,
                            "selected": selected,
                            "expected": task.expected_capability_id,
                            "recovery": recovery.value,
                            "success": success,
                            "model_visible_count": lease.plan.candidate_aperture,
                            "admission_failures": [asdict(item) for item in lease.admission_failures],
                            **execution_detail,
                        }
                    )

            for task in _composition_tasks():
                with runtime.acquire_composition(
                    _request(task.text),
                    max_capabilities=2,
                    request_id=task.task_id,
                ) as lease:
                    selected = frozenset(item.capability_id for item in lease.capabilities)
                    execution_ok = True
                    execution_detail: dict[str, Any] = {}
                    if SCIENTIFIC_ID in selected and task.scientific_case_id is not None:
                        observed = plugin.invoke({"case_id": task.scientific_case_id})
                        execution_ok = observed["status"] == task.expected_scientific_status
                        execution_detail["scientific_status"] = observed["status"]
                    if PROCESS_ID in selected:
                        execution_ok = execution_ok and bool(assets["process_adapter"].is_running)
                        execution_detail["process_running"] = assets["process_adapter"].is_running
                    if SESSION_ID in selected:
                        execution_ok = execution_ok and bool(assets["session_adapter"].is_connected)
                        execution_detail["session_connected"] = assets["session_adapter"].is_connected
                    measured = governor.total_measured_cost()
                    composition_rows.append(
                        {
                            "task_id": task.task_id,
                            "family": task.family,
                            "selected": sorted(selected),
                            "expected": sorted(task.expected_capability_ids),
                            "success": selected == task.expected_capability_ids and execution_ok,
                            "model_visible_count": lease.plan.routing_plan.candidate_aperture,
                            "measured_working_set": asdict(measured),
                            **execution_detail,
                        }
                    )

            # Force a final scientific refault after process/session pressure and verify learned state.
            with runtime.acquire(
                _request("judge evidence applicability under conflicting experimental conditions"),
                request_id="v9r1-final-scientific-refault",
            ) as final_lease:
                final_observed = plugin.invoke({"case_id": "temperature_conflict"})
                final_generation = int(plugin.learned_state.generation)
                final_policy = str(final_observed["policy_mode"])
                final_status = str(final_observed["status"])
                final_selected = final_lease.materialized_capability_id

            event_counts: dict[str, int] = {}
            for event in governor.events:
                event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            single_success = sum(row["success"] for row in single_rows)
            composition_success = sum(row["success"] for row in composition_rows)
            return {
                "schema_version": "orion-evolvable-plugin-v9r1-mixed-capability-runtime-dev-v1",
                "status": "DEVELOPMENT_ONLY_ZERO_PROVIDER_MIXED_REAL_BACKING",
                "inventory_size": len(catalog.ids()),
                "capability_kinds_present": sorted({item.kind.value for item in catalog.all()}),
                "single_task_count": len(single_rows),
                "single_success_count": single_success,
                "single_success_rate": single_success / len(single_rows),
                "single_family_summary": _family_summary(single_rows),
                "composition_task_count": len(composition_rows),
                "composition_success_count": composition_success,
                "composition_success_rate": composition_success / len(composition_rows),
                "composition_family_summary": _family_summary(composition_rows),
                "average_single_model_visible": (
                    sum(row["model_visible_count"] for row in single_rows) / len(single_rows)
                ),
                "average_composition_model_visible": (
                    sum(row["model_visible_count"] for row in composition_rows)
                    / len(composition_rows)
                ),
                "budget": asdict(budget),
                "event_counts": dict(sorted(event_counts.items())),
                "scientific_learned_state": {
                    "strict_update": assets["strict_update"],
                    "final_selected": final_selected,
                    "final_generation": final_generation,
                    "final_policy": final_policy,
                    "final_status": final_status,
                    "refaults": governor.state(SCIENTIFIC_ID).refaults,
                },
                "process_runtime": {
                    "startup_latency_s": assets["process_adapter"].last_startup_latency_s,
                    "shutdown_latency_s": assets["process_adapter"].last_shutdown_latency_s,
                },
                "session_runtime": {
                    "startup_latency_s": assets["session_adapter"].last_startup_latency_s,
                    "shutdown_latency_s": assets["session_adapter"].last_shutdown_latency_s,
                },
                "single_rows": single_rows,
                "composition_rows": composition_rows,
                "provider_calls": 0,
                "kg_calls": 0,
                "rag_calls": 0,
                "v8b_confirmatory_queries_read": 0,
                "protected_six_case_pool_consumed": 0,
                "interpretation_boundary": (
                    "V9R1 is a development systems-integration bank. It uses real ORION scientific "
                    "decision code plus real process/socket backing resources, but task routing texts "
                    "and the development concept encoder are constructed for mechanism development. "
                    "It is not a naturalistic or prospective confirmation of routing utility."
                ),
            }
        finally:
            governor.close()


def main() -> int:
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
