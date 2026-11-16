"""Regression-evidence bridge for version-specific capability qualification.

The generic deployment qualifier intentionally does not understand scientific
semantics.  This module provides a thin evidence-to-contract bridge:

1. a previously qualified executable establishes a validated behavior envelope;
2. a replacement executable is evaluated on the same frozen regression suite;
3. only a replacement that preserves the protected envelope and avoids forbidden
   incidents may receive a new version-specific DeploymentScopeContract.

This is deliberately domain-agnostic.  Chemistry, materials, or other scientific
workloads define the obligation identifiers and incident codes outside CapVM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .deployment import (
    CapabilityDeploymentQualifier,
    DeploymentInputProfile,
    DeploymentQualificationDecision,
    DeploymentScopeContract,
)


@dataclass(frozen=True)
class RegressionQualificationSnapshot:
    """Frozen regression evidence for one capability executable version."""

    capability_id: str
    executable_version: str
    learned_generation: int
    suite_id: str
    suite_sha256: str
    passed_obligations: frozenset[str] = frozenset()
    incident_codes: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        capability_id = str(self.capability_id).strip()
        version = str(self.executable_version).strip()
        suite_id = str(self.suite_id).strip()
        suite_sha256 = str(self.suite_sha256).strip().lower()
        if not capability_id:
            raise ValueError("capability_id is required")
        if not version:
            raise ValueError("executable_version is required")
        if self.learned_generation < 0:
            raise ValueError("learned_generation must be non-negative")
        if not suite_id:
            raise ValueError("suite_id is required")
        if len(suite_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in suite_sha256):
            raise ValueError("suite_sha256 must be a lowercase 64-character SHA-256 hex string")
        obligations = frozenset(
            str(item).strip() for item in self.passed_obligations if str(item).strip()
        )
        incidents = frozenset(
            str(item).strip() for item in self.incident_codes if str(item).strip()
        )
        object.__setattr__(self, "capability_id", capability_id)
        object.__setattr__(self, "executable_version", version)
        object.__setattr__(self, "suite_id", suite_id)
        object.__setattr__(self, "suite_sha256", suite_sha256)
        object.__setattr__(self, "passed_obligations", obligations)
        object.__setattr__(self, "incident_codes", incidents)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class RegressionCompatibilityPolicy:
    """Frozen non-regression contract used before issuing a replacement Q."""

    policy_id: str
    scope_id: str
    protected_obligations: frozenset[str]
    forbidden_incident_codes: frozenset[str] = frozenset()
    require_same_generation: bool = True
    required_context: Mapping[str, str] = field(default_factory=dict)
    required_features: frozenset[str] = frozenset()
    forbidden_features: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        policy_id = str(self.policy_id).strip()
        scope_id = str(self.scope_id).strip()
        if not policy_id:
            raise ValueError("policy_id is required")
        if not scope_id:
            raise ValueError("scope_id is required")
        obligations = frozenset(
            str(item).strip() for item in self.protected_obligations if str(item).strip()
        )
        if not obligations:
            raise ValueError("protected_obligations must be non-empty")
        incidents = frozenset(
            str(item).strip() for item in self.forbidden_incident_codes if str(item).strip()
        )
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "scope_id", scope_id)
        object.__setattr__(self, "protected_obligations", obligations)
        object.__setattr__(self, "forbidden_incident_codes", incidents)
        object.__setattr__(self, "required_context", dict(self.required_context))
        object.__setattr__(self, "required_features", frozenset(self.required_features))
        object.__setattr__(self, "forbidden_features", frozenset(self.forbidden_features))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def preserve_baseline_envelope(
        cls,
        *,
        policy_id: str,
        scope_id: str,
        baseline: RegressionQualificationSnapshot,
        forbidden_incident_codes: frozenset[str] = frozenset(),
        require_same_generation: bool = True,
        required_context: Mapping[str, str] | None = None,
        required_features: frozenset[str] = frozenset(),
        forbidden_features: frozenset[str] = frozenset(),
        metadata: Mapping[str, Any] | None = None,
    ) -> "RegressionCompatibilityPolicy":
        """Protect every behavior obligation validated by the qualified baseline."""

        return cls(
            policy_id=policy_id,
            scope_id=scope_id,
            protected_obligations=baseline.passed_obligations,
            forbidden_incident_codes=forbidden_incident_codes,
            require_same_generation=require_same_generation,
            required_context=dict(required_context or {}),
            required_features=required_features,
            forbidden_features=forbidden_features,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class RegressionQualificationAssessment:
    """Auditable result of comparing one replacement against a baseline envelope."""

    capability_id: str
    baseline_version: str
    candidate_version: str
    learned_generation: int
    suite_id: str
    policy_id: str
    qualified_for_contract: bool
    reason_codes: tuple[str, ...] = ()
    missing_protected_obligations: tuple[str, ...] = ()
    forbidden_incidents_present: tuple[str, ...] = ()


class RegressionQualificationGate:
    """Turn frozen regression evidence into version-specific deployment authority."""

    def __init__(
        self,
        *,
        baseline: RegressionQualificationSnapshot,
        policy: RegressionCompatibilityPolicy,
    ) -> None:
        if not policy.protected_obligations.issubset(baseline.passed_obligations):
            raise ValueError("policy protects obligations not validated by the baseline")
        self.baseline = baseline
        self.policy = policy

    def assess(
        self,
        candidate: RegressionQualificationSnapshot,
    ) -> RegressionQualificationAssessment:
        reasons: list[str] = []
        if candidate.capability_id != self.baseline.capability_id:
            reasons.append("capability_identity_mismatch")
        if candidate.suite_id != self.baseline.suite_id:
            reasons.append("regression_suite_id_mismatch")
        if candidate.suite_sha256 != self.baseline.suite_sha256:
            reasons.append("regression_suite_hash_mismatch")
        if (
            self.policy.require_same_generation
            and candidate.learned_generation != self.baseline.learned_generation
        ):
            reasons.append("learned_generation_changed")

        missing = tuple(
            sorted(self.policy.protected_obligations - candidate.passed_obligations)
        )
        if missing:
            reasons.append("protected_behavior_regressed")

        forbidden = tuple(
            sorted(self.policy.forbidden_incident_codes & candidate.incident_codes)
        )
        if forbidden:
            reasons.append("forbidden_regression_incident")

        return RegressionQualificationAssessment(
            capability_id=candidate.capability_id,
            baseline_version=self.baseline.executable_version,
            candidate_version=candidate.executable_version,
            learned_generation=candidate.learned_generation,
            suite_id=candidate.suite_id,
            policy_id=self.policy.policy_id,
            qualified_for_contract=not reasons,
            reason_codes=tuple(reasons),
            missing_protected_obligations=missing,
            forbidden_incidents_present=forbidden,
        )

    def issue_contract_if_qualified(
        self,
        *,
        candidate: RegressionQualificationSnapshot,
        qualifier: CapabilityDeploymentQualifier,
        profile: DeploymentInputProfile,
    ) -> tuple[RegressionQualificationAssessment, DeploymentQualificationDecision]:
        """Register candidate Q only after a successful regression assessment.

        Failed assessments intentionally register nothing.  If only the baseline
        version has a contract, the existing DeploymentQualifier will therefore
        continue to report STALE_QUALIFICATION for the replacement version.
        """

        assessment = self.assess(candidate)
        if assessment.qualified_for_contract:
            qualifier.register(
                DeploymentScopeContract(
                    plugin_id=candidate.capability_id,
                    plugin_version=candidate.executable_version,
                    generation=candidate.learned_generation,
                    scope_id=self.policy.scope_id,
                    required_context=self.policy.required_context,
                    required_features=self.policy.required_features,
                    forbidden_features=self.policy.forbidden_features,
                    metadata={
                        **dict(self.policy.metadata),
                        "regression_policy_id": self.policy.policy_id,
                        "regression_suite_id": candidate.suite_id,
                        "regression_suite_sha256": candidate.suite_sha256,
                        "qualification_baseline_version": self.baseline.executable_version,
                        "protected_obligation_count": len(self.policy.protected_obligations),
                    },
                )
            )

        decision = qualifier.evaluate(
            plugin_id=candidate.capability_id,
            plugin_version=candidate.executable_version,
            generation=candidate.learned_generation,
            profile=profile,
            scope_id=self.policy.scope_id,
        )
        return assessment, decision
