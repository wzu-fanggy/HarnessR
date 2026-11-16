"""Evolvable Plugin V9：Capability Discovery / Routing 的零 provider development benchmark。

该 benchmark 使用全新的合成 capability inventory 与任务，不读取 V8B 82-query confirmatory
结果，不调用 provider/KG/RAG。目的仅是开发并验证 routing plane 的结构性质，不作为外部效度
或论文 confirmatory evidence。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from orion.capvm.catalog import CapabilityCatalog
from orion.capvm.deployment import (
    CapabilityDeploymentQualifier,
    DeploymentInputProfile,
    DeploymentScopeContract,
)
from orion.capvm.governor import CapabilityResidencyGovernor
from orion.capvm.model import CapabilityKind, ResidencyLevel, ResourceVector, VirtualCapability
from orion.capvm.routing import (
    AdmissibilityDecision,
    CapabilityControlPlane,
    DeploymentQualificationRule,
    EvolvableCapabilityRouter,
    MetadataDiscoveryBackend,
    RecoveryAction,
    RoutedCapabilityRuntime,
    RoutingRequest,
)


INVENTORY_SIZE = 256
SHORTLIST_K = 5
TASKS_PER_FAMILY = 8


class BenchmarkAdapter:
    """只用于 V9 development 的确定性多资源 adapter。"""

    def __init__(self, *, interface_tokens: float, bound_ram_mb: float) -> None:
        self.interface_tokens = interface_tokens
        self.bound_ram_mb = bound_ram_mb
        self.promotions = 0
        self.demotions = 0

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level <= ResidencyLevel.CATALOG:
            return ResourceVector()
        if level < ResidencyLevel.BOUND:
            return ResourceVector(prompt_tokens=self.interface_tokens)
        return ResourceVector(
            prompt_tokens=self.interface_tokens,
            host_ram_mb=self.bound_ram_mb,
        )

    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:
        return self.declared_cost(level)

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        del current, target
        self.promotions += 1

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        del current, target
        self.demotions += 1

    def close(self) -> None:
        return None


def _capability(
    capability_id: str,
    summary: str,
    *,
    terms: tuple[str, ...],
    interface_tokens: float = 32.0,
    bound_ram_mb: float = 2.0,
    required_authorities: tuple[str, ...] = (),
    learned_generation: int | None = None,
    deployment_scope_id: str | None = None,
) -> VirtualCapability:
    metadata: dict[str, object] = {
        "routing_terms": terms,
        "required_authorities": required_authorities,
        "routing_required_level": "BOUND",
        "version": "v9-dev",
    }
    if learned_generation is not None:
        metadata["learned_generation"] = learned_generation
    if deployment_scope_id is not None:
        metadata["deployment_scope_id"] = deployment_scope_id
    return VirtualCapability(
        capability_id=capability_id,
        kind=CapabilityKind.PLUGIN,
        summary=summary,
        adapter=BenchmarkAdapter(
            interface_tokens=interface_tokens,
            bound_ram_mb=bound_ram_mb,
        ),
        metadata=metadata,
    )


def build_inventory() -> tuple[CapabilityCatalog, CapabilityDeploymentQualifier]:
    capabilities = [
        _capability(
            "kg-structured-query",
            "query structured sodium battery knowledge graph evidence",
            terms=("knowledge", "graph", "structured", "query", "sodium", "battery"),
        ),
        _capability(
            "report-file-writer",
            "write export scientific report file to filesystem",
            terms=("write", "export", "scientific", "report", "file"),
            required_authorities=("filesystem.write",),
        ),
        _capability(
            "report-drafter",
            "draft scientific report content without filesystem side effects",
            terms=("draft", "scientific", "report", "write"),
        ),
        _capability(
            "strict-applicability",
            "strict provenance aware scientific evidence applicability decision",
            terms=("strict", "provenance", "scientific", "evidence", "applicability"),
            learned_generation=3,
            deployment_scope_id="represented-input",
        ),
        _capability(
            "legacy-applicability",
            "scientific evidence applicability conservative fallback policy",
            terms=("scientific", "evidence", "applicability", "fallback"),
        ),
        _capability(
            "heavy-molecular-simulation",
            "accurate molecular simulation for electrolyte behavior",
            terms=("accurate", "molecular", "simulation", "electrolyte", "behavior"),
            bound_ram_mb=24.0,
        ),
        _capability(
            "light-molecular-simulation",
            "lightweight surrogate molecular simulation for electrolyte behavior",
            terms=("lightweight", "surrogate", "molecular", "simulation", "electrolyte", "behavior"),
            bound_ram_mb=4.0,
        ),
    ]

    distractor_count = INVENTORY_SIZE - len(capabilities)
    for index in range(distractor_count):
        capabilities.append(
            _capability(
                f"archive-helper-{index:03d}",
                "generic archive formatting conversion helper",
                terms=("archive", "format", "conversion", f"helper{index}"),
                interface_tokens=24.0,
                bound_ram_mb=1.0,
            )
        )

    qualifier = CapabilityDeploymentQualifier()
    qualifier.register(
        DeploymentScopeContract(
            plugin_id="strict-applicability",
            generation=3,
            scope_id="represented-input",
            required_features=frozenset({"represented_input"}),
        )
    )
    return CapabilityCatalog(capabilities), qualifier


@dataclass(frozen=True)
class DevTask:
    task_id: str
    family: str
    text: str
    expected_capability_id: str | None
    expected_recovery: RecoveryAction = RecoveryAction.NONE
    authorities: frozenset[str] = frozenset()
    deployment_profile: DeploymentInputProfile | None = None


def build_tasks() -> tuple[DevTask, ...]:
    families = (
        (
            "plain_match",
            "structured sodium battery knowledge graph query",
            "kg-structured-query",
            RecoveryAction.NONE,
            frozenset(),
            None,
        ),
        (
            "authority_fallback",
            "write export scientific report file",
            "report-drafter",
            RecoveryAction.NONE,
            frozenset(),
            None,
        ),
        (
            "deployment_fallback",
            "strict provenance scientific evidence applicability",
            "legacy-applicability",
            RecoveryAction.NONE,
            frozenset(),
            DeploymentInputProfile(features=frozenset({"raw_input"})),
        ),
        (
            "resource_fallback",
            "accurate molecular simulation electrolyte behavior",
            "light-molecular-simulation",
            RecoveryAction.FALLBACK,
            frozenset(),
            None,
        ),
        (
            "no_match_recovery",
            "quantum transition state reaction saddle point search",
            None,
            RecoveryAction.PROPOSE_CAPABILITY,
            frozenset(),
            None,
        ),
    )
    tasks: list[DevTask] = []
    suffixes = (
        "for this task",
        "for the current request",
        "with a concise result",
        "under the current runtime",
        "for a scientific agent",
        "without changing the host graph",
        "using the available capability catalog",
        "for the present workflow",
    )
    for family, base, expected, recovery, authorities, profile in families:
        for index, suffix in enumerate(suffixes):
            tasks.append(
                DevTask(
                    task_id=f"{family}-{index:02d}",
                    family=family,
                    text=f"{base} {suffix}",
                    expected_capability_id=expected,
                    expected_recovery=recovery,
                    authorities=authorities,
                    deployment_profile=profile,
                )
            )
    return tuple(tasks)


def _request(task: DevTask, *, max_candidates: int = SHORTLIST_K) -> RoutingRequest:
    return RoutingRequest(
        task_text=task.text,
        authorities=task.authorities,
        deployment_profile=task.deployment_profile,
        max_candidates=max_candidates,
        minimum_matched_terms=2,
        minimum_query_coverage=0.15,
    )


def _interface_tokens(catalog: CapabilityCatalog, capability_ids: tuple[str, ...]) -> float:
    return sum(
        catalog.get(capability_id).adapter.declared_cost(ResidencyLevel.INTERFACE).prompt_tokens
        for capability_id in capability_ids
    )


def _success(
    task: DevTask,
    *,
    selected: str | None,
    recovery: RecoveryAction,
) -> bool:
    if task.expected_capability_id is not None:
        return selected == task.expected_capability_id
    return selected is None and recovery == task.expected_recovery


class DeploymentOnlyControl:
    """V8-style baseline：只检查显式 deployment scope，不承担 authority 等控制职责。"""

    def __init__(self, qualifier: CapabilityDeploymentQualifier) -> None:
        self.rule = DeploymentQualificationRule(qualifier=qualifier)

    def evaluate(self, *, request: RoutingRequest, descriptor) -> AdmissibilityDecision:
        reasons = self.rule.evaluate(request=request, descriptor=descriptor)
        return AdmissibilityDecision(
            capability_id=descriptor.capability_id,
            admissible=not reasons,
            reason_codes=reasons,
        )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "task_count": count,
        "success_count": sum(row["success"] for row in rows),
        "task_success_rate": sum(row["success"] for row in rows) / count,
        "wrong_choice_count": sum(
            row["selected_capability_id"] is not None and not row["success"]
            for row in rows
        ),
        "recovery_count": sum(row["selected_capability_id"] is None for row in rows),
        "average_model_visible_capabilities": (
            sum(row["model_visible_count"] for row in rows) / count
        ),
        "max_model_visible_capabilities": max(row["model_visible_count"] for row in rows),
        "average_interface_tokens_exposed": (
            sum(row["interface_tokens_exposed"] for row in rows) / count
        ),
        "resource_fallback_count": sum(row.get("resource_fallback", False) for row in rows),
        "rows": rows,
    }


def _run_all_visible(
    catalog: CapabilityCatalog,
    tasks: tuple[DevTask, ...],
) -> dict[str, Any]:
    discovery = MetadataDiscoveryBackend()
    descriptors = tuple(
        # 只读取 metadata；all-visible 的“eager”成本在下面按完整 inventory 计入。
        EvolvableCapabilityRouter(catalog=catalog).descriptors()
    )
    all_ids = catalog.ids()
    full_tokens = _interface_tokens(catalog, all_ids)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        candidates = discovery.discover(
            _request(task, max_candidates=INVENTORY_SIZE),
            descriptors,
        )
        selected = candidates[0].capability_id if candidates else None
        recovery = RecoveryAction.PROPOSE_CAPABILITY if selected is None else RecoveryAction.NONE
        rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "selected_capability_id": selected,
                "recovery_action": recovery.value,
                "success": _success(task, selected=selected, recovery=recovery),
                "model_visible_count": INVENTORY_SIZE,
                "interface_tokens_exposed": full_tokens,
                "resource_fallback": False,
            }
        )
    return _summarize(rows)


def _run_router_arm(
    *,
    catalog: CapabilityCatalog,
    qualifier: CapabilityDeploymentQualifier,
    tasks: tuple[DevTask, ...],
    arm: str,
) -> dict[str, Any]:
    if arm == "METADATA_TOP1":
        control = DeploymentOnlyControl(CapabilityDeploymentQualifier())
    elif arm == "BINARY_DEPLOYMENT_ONLY":
        control = DeploymentOnlyControl(qualifier)
    elif arm in {"SHORTLIST_CONTROL", "FULL_ROUTED_RUNTIME"}:
        control = CapabilityControlPlane(
            rules=(DeploymentQualificationRule(qualifier=qualifier),)
        )
    else:
        raise ValueError(arm)

    router = EvolvableCapabilityRouter(catalog=catalog, control=control)
    governor = None
    runtime = None
    if arm == "FULL_ROUTED_RUNTIME":
        governor = CapabilityResidencyGovernor(
            catalog=catalog,
            budget=ResourceVector(
                prompt_tokens=256.0,
                host_ram_mb=8.0,
                process_slots=1.0,
                connection_slots=1.0,
            ),
        )
        runtime = RoutedCapabilityRuntime(router=router, governor=governor)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        request = _request(task)
        if runtime is None:
            plan = router.plan(request)
            selected = plan.selected_capability_id
            recovery = plan.recovery_action
            visible_ids = plan.model_visible_capability_ids
            resource_fallback = False
        else:
            with runtime.acquire(request, request_id=task.task_id) as lease:
                plan = lease.plan
                selected = lease.materialized_capability_id
                recovery = lease.recovery_action
                visible_ids = plan.model_visible_capability_ids
                resource_fallback = bool(lease.admission_failures)

        rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "selected_capability_id": selected,
                "recovery_action": recovery.value,
                "success": _success(task, selected=selected, recovery=recovery),
                "model_visible_count": len(visible_ids),
                "interface_tokens_exposed": _interface_tokens(catalog, visible_ids),
                "resource_fallback": resource_fallback,
            }
        )

    if governor is not None:
        governor.close()
    return _summarize(rows)


def run_benchmark() -> dict[str, Any]:
    catalog, qualifier = build_inventory()
    tasks = build_tasks()

    arms = {
        "ALL_VISIBLE_UNGOVERNED": _run_all_visible(catalog, tasks),
        "METADATA_TOP1": _run_router_arm(
            catalog=catalog,
            qualifier=qualifier,
            tasks=tasks,
            arm="METADATA_TOP1",
        ),
        "BINARY_DEPLOYMENT_ONLY": _run_router_arm(
            catalog=catalog,
            qualifier=qualifier,
            tasks=tasks,
            arm="BINARY_DEPLOYMENT_ONLY",
        ),
        "SHORTLIST_CONTROL": _run_router_arm(
            catalog=catalog,
            qualifier=qualifier,
            tasks=tasks,
            arm="SHORTLIST_CONTROL",
        ),
        "FULL_ROUTED_RUNTIME": _run_router_arm(
            catalog=catalog,
            qualifier=qualifier,
            tasks=tasks,
            arm="FULL_ROUTED_RUNTIME",
        ),
    }

    return {
        "schema_version": "orion-evolvable-plugin-v9-routing-development-v1",
        "status": "NEW_DEVELOPMENT_ZERO_PROVIDER",
        "inventory_size": len(catalog.ids()),
        "shortlist_k": SHORTLIST_K,
        "task_count": len(tasks),
        "task_family_count": len({task.family for task in tasks}),
        "tasks_per_family": TASKS_PER_FAMILY,
        "arms": arms,
        "provider_calls": 0,
        "kg_calls": 0,
        "rag_calls": 0,
        "v8b_confirmatory_queries_read": 0,
        "protected_six_case_pool_consumed": 0,
        "interpretation_boundary": (
            "V9 development mechanism benchmark only. It validates routing-plane separation, "
            "progressive disclosure, hard authority/deployment control and residency fallback on a "
            "new synthetic inventory; it is not prospective scientific utility evidence."
        ),
    }


def main() -> int:
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
