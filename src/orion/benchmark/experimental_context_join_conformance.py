"""Zero-model conformance for joint experimental-context evidence closure.

This suite tests one implementation property: locally satisfied hard constraints
may authorize a recommendation only when every declared joint-context group has
at least one compatible evidence witness. It is not a scientific-accuracy or
natural-prevalence benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from orion.core.design_decision import (
    CandidateEvaluation,
    CandidateState,
    ConstraintEvaluation,
    ConstraintSpec,
    EvidenceLedgerItem,
    JointContextRequirement,
    TaskContract,
    decide_design,
)


@dataclass(frozen=True)
class Case:
    case_id: str
    description: str
    expected_status: str
    expected_reason: str | None
    run: Callable[[], dict[str, Any]]


def _contract(*, joint: bool = True) -> TaskContract:
    return TaskContract(
        task_type="design",
        hard_constraints=[
            ConstraintSpec(
                constraint_id="retention",
                description="capacity retention >= 90%",
                required_evidence_kinds=["property"],
            ),
            ConstraintSpec(
                constraint_id="voltage",
                description="stable operation >= 4.2 V",
                required_evidence_kinds=["property"],
            ),
        ],
        joint_context_requirements=(
            [
                JointContextRequirement(
                    group_id="operating-point",
                    constraint_ids=["retention", "voltage"],
                    context_fields=[
                        "temperature_c",
                        "electrode",
                        "cell_configuration",
                    ],
                    target_context={"temperature_c": -20.0},
                    search_hint="find evidence compatible with one target operating point",
                )
            ]
            if joint
            else []
        ),
    )


def _candidate() -> CandidateState:
    return CandidateState(
        candidate_id="C1",
        composition={"salt": "NaPF6"},
        formulation_evidence_ids=["form"],
    )


def _evaluation(*, voltage_ids: list[str] | None = None) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id="C1",
        constraints=[
            ConstraintEvaluation(
                constraint_id="retention",
                status="satisfied",
                evidence_ids=["retention"],
            ),
            ConstraintEvaluation(
                constraint_id="voltage",
                status="satisfied",
                evidence_ids=voltage_ids or ["voltage"],
            ),
        ],
    )


def _item(
    evidence_id: str,
    *,
    kind: str,
    scope: dict[str, object] | None = None,
    experiment_id: str | None = None,
) -> EvidenceLedgerItem:
    return EvidenceLedgerItem(
        evidence_id=evidence_id,
        chunk_id=f"chunk::{evidence_id}",
        paper_id="paper-A",
        evidence_kind=kind,
        excerpt=f"evidence {evidence_id}",
        candidate_id="C1",
        experiment_id=experiment_id,
        condition_scope=scope,
    )


def _run(
    *,
    voltage_scope: dict[str, object],
    joint: bool = True,
    voltage_ids: list[str] | None = None,
    extra: list[EvidenceLedgerItem] | None = None,
) -> dict[str, Any]:
    ledger = [
        _item("form", kind="formulation"),
        _item(
            "retention",
            kind="property",
            experiment_id="exp-retention",
            scope={
                "temperature_c": -20.0,
                "electrode": "hard carbon",
                "cell_configuration": "coin cell",
                "rate": "1C",
            },
        ),
        _item(
            "voltage",
            kind="property",
            experiment_id="exp-voltage",
            scope=voltage_scope,
        ),
        *(extra or []),
    ]
    decision = decide_design(
        contract=_contract(joint=joint),
        candidates=[_candidate()],
        evaluations=[_evaluation(voltage_ids=voltage_ids)],
        evidence_ledger=ledger,
        attempts=1,
        max_attempts=2,
    )
    return {
        "status": decision.status,
        "reason_codes": decision.reason_codes,
        "approved_candidate_ids": decision.approved_candidate_ids,
        "witness_count": len(decision.joint_context_witnesses),
        "witness_evidence_ids": (
            decision.joint_context_witnesses[0].evidence_ids
            if decision.joint_context_witnesses
            else []
        ),
    }


def build_cases() -> list[Case]:
    compatible = {
        "temperature_c": -20.0,
        "electrode": "hard carbon",
        "cell_configuration": "coin cell",
        "rate": "0.5C",
    }
    return [
        Case(
            "different_experiments_compatible",
            "Different experiment IDs may jointly witness a recommendation when declared context fields agree.",
            "recommend",
            None,
            lambda: _run(voltage_scope=compatible),
        ),
        Case(
            "local_pass_temperature_join_empty",
            "Locally satisfied constraints cannot compose when target temperature assignments conflict.",
            "search",
            "experimental_context_join_conflict",
            lambda: _run(
                voltage_scope={
                    "temperature_c": 25.0,
                    "electrode": "hard carbon",
                    "cell_configuration": "coin cell",
                }
            ),
        ),
        Case(
            "local_pass_electrode_join_empty",
            "Locally satisfied constraints cannot compose across incompatible electrode contexts.",
            "search",
            "experimental_context_join_conflict",
            lambda: _run(
                voltage_scope={
                    "temperature_c": -20.0,
                    "electrode": "NVP",
                    "cell_configuration": "coin cell",
                }
            ),
        ),
        Case(
            "irrelevant_rate_conflict_nonblocking",
            "A difference outside the declared joint-context fields does not empty the join.",
            "recommend",
            None,
            lambda: _run(
                voltage_scope={
                    "temperature_c": -20.0,
                    "electrode": "hard carbon",
                    "cell_configuration": "coin cell",
                    "rate": "0.1C",
                }
            ),
        ),
        Case(
            "legacy_local_semantics_preserved",
            "Without an explicit joint-context obligation, legacy local-constraint behavior is preserved.",
            "recommend",
            None,
            lambda: _run(
                joint=False,
                voltage_scope={
                    "temperature_c": 25.0,
                    "electrode": "NVP",
                    "cell_configuration": "pouch cell",
                },
            ),
        ),
        Case(
            "alternative_support_recovers_nonempty_join",
            "When one cited support conflicts but another support is compatible, the existential witness preserves recommendation.",
            "recommend",
            None,
            lambda: _run(
                voltage_scope={
                    "temperature_c": 25.0,
                    "electrode": "hard carbon",
                    "cell_configuration": "coin cell",
                },
                voltage_ids=["voltage", "voltage-compatible"],
                extra=[
                    _item(
                        "voltage-compatible",
                        kind="property",
                        experiment_id="exp-voltage-alt",
                        scope=compatible,
                    )
                ],
            ),
        ),
    ]


def run_suite() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in build_cases():
        observed = case.run()
        reason_ok = (
            case.expected_reason is None
            or case.expected_reason in observed["reason_codes"]
        )
        passed = observed["status"] == case.expected_status and reason_ok
        rows.append(
            {
                "case_id": case.case_id,
                "description": case.description,
                "expected": {
                    "status": case.expected_status,
                    "reason": case.expected_reason,
                },
                "observed": observed,
                "passed": passed,
            }
        )
    return {
        "schema_version": "experimental-context-join-conformance-v1",
        "scope": "zero-model implementation conformance; not scientific accuracy or natural prevalence",
        "case_count": len(rows),
        "passed": sum(int(row["passed"]) for row in rows),
        "failed": sum(int(not row["passed"]) for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run_suite(), ensure_ascii=False, indent=2))
