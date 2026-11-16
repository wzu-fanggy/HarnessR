"""Evolvable Plugin V5：把真实 ORION 证据组合决策装入 stateful LangGraph plugin。

V5 复用 `experimental_context_join_conformance` 中已经存在的 ORION scientific decision
logic。插件 learned state 只决定 composition policy（legacy-local vs strict-joint）；真正的
recommend/search 决策仍由 ORION `decide_design` 路径产生。

目标是验证：
1. capability-local evolution 能改善真实科学决策 conformance；
2. candidate update 先验证再 promote，坏更新可 rollback；
3. scientific improvement 能跨 eviction/remount 保留。

无 LLM/KG/RAG/网络调用。
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, TypedDict

from langgraph.graph import StateGraph

from orion.benchmark.experimental_context_join_conformance import _item, _run
from orion.capvm import (
    CapabilityCatalog,
    CapabilityKind,
    CapabilityLearnedState,
    CapabilityResidencyGovernor,
    ResidencyLevel,
    ResourceVector,
    StatefulLangGraphCapabilityPlugin,
    VirtualCapability,
)
from orion.capvm.adapters import EvolvablePluginCapabilityAdapter


_POLICY_KEY = "evidence-composition-policy"
_LEGACY = "legacy-local"
_STRICT = "strict-joint"


@dataclass(frozen=True)
class ScientificPolicyCase:
    case_id: str
    expected_status: str
    run: Callable[[bool], dict[str, object]]


class ScientificExecutionState(TypedDict, total=False):
    case_id: str
    policy_mode: str
    status: str
    reason_codes: list[str]
    approved_candidate_ids: list[str]
    witness_count: int


def _compatible_scope() -> dict[str, object]:
    return {
        "temperature_c": -20.0,
        "electrode": "hard carbon",
        "cell_configuration": "coin cell",
        "rate": "0.5C",
    }


def build_scientific_cases() -> tuple[ScientificPolicyCase, ...]:
    compatible = _compatible_scope()
    return (
        ScientificPolicyCase(
            case_id="compatible_operating_point",
            expected_status="recommend",
            run=lambda joint: _run(voltage_scope=compatible, joint=joint),
        ),
        ScientificPolicyCase(
            case_id="temperature_conflict",
            expected_status="search",
            run=lambda joint: _run(
                voltage_scope={
                    "temperature_c": 25.0,
                    "electrode": "hard carbon",
                    "cell_configuration": "coin cell",
                },
                joint=joint,
            ),
        ),
        ScientificPolicyCase(
            case_id="electrode_conflict",
            expected_status="search",
            run=lambda joint: _run(
                voltage_scope={
                    "temperature_c": -20.0,
                    "electrode": "NVP",
                    "cell_configuration": "coin cell",
                },
                joint=joint,
            ),
        ),
        ScientificPolicyCase(
            case_id="irrelevant_rate_difference",
            expected_status="recommend",
            run=lambda joint: _run(
                voltage_scope={
                    "temperature_c": -20.0,
                    "electrode": "hard carbon",
                    "cell_configuration": "coin cell",
                    "rate": "0.1C",
                },
                joint=joint,
            ),
        ),
        ScientificPolicyCase(
            case_id="cell_configuration_conflict_holdout",
            expected_status="search",
            run=lambda joint: _run(
                voltage_scope={
                    "temperature_c": -20.0,
                    "electrode": "hard carbon",
                    "cell_configuration": "pouch cell",
                },
                joint=joint,
            ),
        ),
        ScientificPolicyCase(
            case_id="alternative_support_holdout",
            expected_status="recommend",
            run=lambda joint: _run(
                voltage_scope={
                    "temperature_c": 25.0,
                    "electrode": "hard carbon",
                    "cell_configuration": "coin cell",
                },
                joint=joint,
                voltage_ids=["voltage", "voltage-compatible"],
                extra=[
                    _item(
                        "voltage-compatible",
                        kind="property",
                        experiment_id="exp-voltage-alt-v5",
                        scope=compatible,
                    )
                ],
            ),
        ),
    )


def _graph_factory(cases: tuple[ScientificPolicyCase, ...]):
    by_id = {case.case_id: case for case in cases}

    def factory(store):
        def scientific_decision(state: ScientificExecutionState) -> ScientificExecutionState:
            case_id = str(state.get("case_id") or "")
            try:
                case = by_id[case_id]
            except KeyError as exc:
                raise KeyError(f"unknown scientific case: {case_id}") from exc
            learned = store.load()
            policy_mode = learned.rules.get(_POLICY_KEY, _LEGACY)
            joint = policy_mode == _STRICT
            observed = case.run(joint)
            return {
                "case_id": case_id,
                "policy_mode": policy_mode,
                "status": str(observed["status"]),
                "reason_codes": list(observed["reason_codes"]),
                "approved_candidate_ids": list(observed["approved_candidate_ids"]),
                "witness_count": int(observed["witness_count"]),
            }

        graph = StateGraph(ScientificExecutionState)
        graph.add_node("scientific_decision", scientific_decision)
        graph.set_entry_point("scientific_decision")
        graph.set_finish_point("scientific_decision")
        return graph.compile()

    return factory


def _build_plugin(*, state_root: Path, cases: tuple[ScientificPolicyCase, ...]):
    plugin = StatefulLangGraphCapabilityPlugin(
        plugin_id="orion-evidence-composition",
        version="v5",
        state_root=state_root,
        graph_factory=_graph_factory(cases),
        resident_mb=2.0,
    )
    adapter = EvolvablePluginCapabilityAdapter(
        plugin=plugin,
        interface_text="ORION evidence composition capability",
        reasoning_text="Qualify locally grounded evidence under a declared joint scientific context.",
    )
    capability = VirtualCapability(
        capability_id=plugin.plugin_id,
        kind=CapabilityKind.PLUGIN,
        summary="ORION scientific evidence-composition capability",
        adapter=adapter,
        maximum_level=ResidencyLevel.BOUND,
    )
    return capability, plugin


def _score(plugin: StatefulLangGraphCapabilityPlugin, cases) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    correct = 0
    for case in cases:
        observed = plugin.invoke({"case_id": case.case_id})
        passed = observed["status"] == case.expected_status
        correct += int(passed)
        rows.append(
            {
                "case_id": case.case_id,
                "expected_status": case.expected_status,
                "observed_status": observed["status"],
                "policy_mode": observed["policy_mode"],
                "passed": passed,
            }
        )
    return {
        "correct": correct,
        "case_count": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "rows": rows,
    }


def _propose_policy(
    plugin: StatefulLangGraphCapabilityPlugin,
    *,
    policy_mode: str,
    validation_cases: tuple[ScientificPolicyCase, ...],
    require_strict_improvement: bool,
) -> dict[str, object]:
    if not plugin.is_mounted:
        raise RuntimeError("plugin must be mounted")
    previous = plugin.learned_state
    baseline = _score(plugin, validation_cases)
    candidate_rules = dict(previous.rules)
    candidate_rules[_POLICY_KEY] = policy_mode
    candidate = CapabilityLearnedState(
        plugin_id=plugin.plugin_id,
        generation=previous.generation + 1,
        rules=candidate_rules,
    )
    plugin.state_store.save(candidate)
    rolled_back = False
    try:
        candidate_score = _score(plugin, validation_cases)
        baseline_accuracy = float(baseline["accuracy"])
        candidate_accuracy = float(candidate_score["accuracy"])
        if require_strict_improvement:
            promoted = candidate_accuracy > baseline_accuracy
        else:
            promoted = candidate_accuracy >= baseline_accuracy
        if not promoted:
            plugin.state_store.save(previous)
            rolled_back = True
    except Exception:
        plugin.state_store.save(previous)
        raise
    return {
        "policy_mode": policy_mode,
        "promoted": promoted,
        "rolled_back": rolled_back,
        "generation_before": previous.generation,
        "generation_after": plugin.learned_state.generation,
        "baseline_accuracy": baseline["accuracy"],
        "candidate_accuracy": candidate_score["accuracy"],
    }


def run_benchmark() -> dict[str, object]:
    cases = build_scientific_cases()
    dev_cases = cases[:4]
    holdout_cases = cases[4:]
    with tempfile.TemporaryDirectory(prefix="orion-plugin-v5-") as tmp:
        root = Path(tmp)
        capability, plugin = _build_plugin(state_root=root, cases=cases)
        governor = CapabilityResidencyGovernor(
            catalog=CapabilityCatalog([capability]),
            budget=ResourceVector(prompt_tokens=200, host_ram_mb=2.5),
        )
        try:
            governor.access(plugin.plugin_id, ResidencyLevel.BOUND, request_id="v5-mount")
            before_dev = _score(plugin, dev_cases)
            before_holdout = _score(plugin, holdout_cases)

            strict_update = _propose_policy(
                plugin,
                policy_mode=_STRICT,
                validation_cases=dev_cases,
                require_strict_improvement=True,
            )
            after_dev = _score(plugin, dev_cases)
            after_holdout = _score(plugin, holdout_cases)

            # 一个会把 strict policy 退回 legacy 的候选必须被插件自己的 dev validation 拒绝。
            bad_update = _propose_policy(
                plugin,
                policy_mode=_LEGACY,
                validation_cases=dev_cases,
                require_strict_improvement=False,
            )
            after_bad_candidate = _score(plugin, dev_cases)

            governor.demote(plugin.plugin_id, ResidencyLevel.CATALOG)
            resident_mb_after_eviction = governor.total_measured_cost().host_ram_mb
            governor.access(plugin.plugin_id, ResidencyLevel.BOUND, request_id="v5-refault")
            after_refault_holdout = _score(plugin, holdout_cases)
            refaults = governor.state(plugin.plugin_id).refaults
            generation = plugin.learned_state.generation

            # fresh runtime object over the same capability-owned state namespace.
            governor.close()
            fresh_capability, fresh_plugin = _build_plugin(state_root=root, cases=cases)
            fresh_governor = CapabilityResidencyGovernor(
                catalog=CapabilityCatalog([fresh_capability]),
                budget=ResourceVector(prompt_tokens=200, host_ram_mb=2.5),
            )
            try:
                fresh_governor.access(
                    fresh_plugin.plugin_id,
                    ResidencyLevel.BOUND,
                    request_id="v5-fresh-runtime",
                )
                fresh_holdout = _score(fresh_plugin, holdout_cases)
                fresh_generation = fresh_plugin.learned_state.generation
            finally:
                fresh_governor.close()

            return {
                "schema_version": "orion-evolvable-plugin-scientific-integration-v5",
                "status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
                "scientific_logic": "existing ORION experimental-context composition / decide_design",
                "dev_case_ids": [case.case_id for case in dev_cases],
                "holdout_case_ids": [case.case_id for case in holdout_cases],
                "before": {
                    "dev": before_dev,
                    "holdout": before_holdout,
                },
                "strict_policy_update": strict_update,
                "after_strict": {
                    "dev": after_dev,
                    "holdout": after_holdout,
                },
                "bad_candidate_update": bad_update,
                "after_bad_candidate_dev": after_bad_candidate,
                "lifecycle": {
                    "resident_mb_after_eviction": resident_mb_after_eviction,
                    "refaults": refaults,
                    "generation_after_refault": generation,
                    "holdout_after_refault": after_refault_holdout,
                    "fresh_runtime_generation": fresh_generation,
                    "holdout_after_fresh_runtime": fresh_holdout,
                },
                "interpretation_boundary": (
                    "V5 integrates existing ORION scientific decision code into the plugin lifecycle. "
                    "Its cases are controlled conformance cases, not a naturalistic prevalence study or "
                    "provider-backed end-to-end scientific benchmark."
                ),
            }
        finally:
            # close() is idempotent; it may already have been called before fresh-runtime reconstruction.
            governor.close()


def main() -> int:
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
