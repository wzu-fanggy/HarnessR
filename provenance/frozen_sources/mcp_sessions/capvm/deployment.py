"""Evolvable capability 的部署资格与作用域控制。

本模块不理解具体科学语义。它只把部署资格绑定到：

- capability/plugin identity；
- learned-state generation；
- 显式 scope contract；
- 当前 runtime/input profile。

核心不变量：validated generation != globally deployable generation。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol


class DeploymentQualificationStatus(str, Enum):
    """部署资格决策状态。"""

    QUALIFIED = "qualified"
    QUARANTINE = "quarantine"
    STALE_QUALIFICATION = "stale_qualification"
    UNQUALIFIED = "unqualified"


@dataclass(frozen=True)
class DeploymentInputProfile:
    """当前 Host/input regime 的通用、显式特征描述。"""

    context: Mapping[str, str] = field(default_factory=dict)
    features: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        normalized_context = {
            str(key).strip(): str(value).strip()
            for key, value in self.context.items()
        }
        if any(not key or not value for key, value in normalized_context.items()):
            raise ValueError("deployment profile context keys/values must be non-empty")
        normalized_features = frozenset(
            str(item).strip() for item in self.features if str(item).strip()
        )
        object.__setattr__(self, "context", normalized_context)
        object.__setattr__(self, "features", normalized_features)


@dataclass(frozen=True)
class DeploymentScopeContract:
    """一个 capability generation 的可部署作用域。"""

    plugin_id: str
    generation: int
    scope_id: str
    plugin_version: str | None = None
    required_context: Mapping[str, str] = field(default_factory=dict)
    required_features: frozenset[str] = frozenset()
    forbidden_features: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plugin_id.strip():
            raise ValueError("plugin_id is required")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if not self.scope_id.strip():
            raise ValueError("scope_id is required")
        plugin_version = (
            str(self.plugin_version).strip() if self.plugin_version is not None else None
        )
        if plugin_version == "":
            plugin_version = None
        required_context = {
            str(key).strip(): str(value).strip()
            for key, value in self.required_context.items()
        }
        if any(not key or not value for key, value in required_context.items()):
            raise ValueError("required_context keys/values must be non-empty")
        required = frozenset(
            str(item).strip() for item in self.required_features if str(item).strip()
        )
        forbidden = frozenset(
            str(item).strip() for item in self.forbidden_features if str(item).strip()
        )
        overlap = required & forbidden
        if overlap:
            raise ValueError(
                "feature cannot be both required and forbidden: "
                + ",".join(sorted(overlap))
            )
        object.__setattr__(self, "plugin_version", plugin_version)
        object.__setattr__(self, "required_context", required_context)
        object.__setattr__(self, "required_features", required)
        object.__setattr__(self, "forbidden_features", forbidden)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "generation": self.generation,
            "scope_id": self.scope_id,
            "plugin_version": self.plugin_version,
            "required_context": dict(sorted(self.required_context.items())),
            "required_features": sorted(self.required_features),
            "forbidden_features": sorted(self.forbidden_features),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeploymentScopeContract":
        return cls(
            plugin_id=str(payload["plugin_id"]),
            generation=int(payload["generation"]),
            scope_id=str(payload["scope_id"]),
            plugin_version=(
                str(payload["plugin_version"])
                if payload.get("plugin_version") is not None
                else None
            ),
            required_context=dict(payload.get("required_context") or {}),
            required_features=frozenset(payload.get("required_features") or ()),
            forbidden_features=frozenset(payload.get("forbidden_features") or ()),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class DeploymentQualificationDecision:
    """一次可审计部署资格判定。"""

    plugin_id: str
    generation: int
    scope_id: str | None
    status: DeploymentQualificationStatus
    plugin_version: str | None = None
    reason_codes: tuple[str, ...] = ()
    missing_context: tuple[str, ...] = ()
    mismatched_context: tuple[str, ...] = ()
    missing_features: tuple[str, ...] = ()
    forbidden_features_present: tuple[str, ...] = ()

    @property
    def qualified(self) -> bool:
        return self.status == DeploymentQualificationStatus.QUALIFIED


class CapabilityDeploymentQualifier:
    """保存并评估 generation-specific deployment scope contracts。"""

    def __init__(self) -> None:
        self._contracts: dict[tuple[str, str | None, int, str], DeploymentScopeContract] = {}

    def register(self, contract: DeploymentScopeContract) -> None:
        key = (
            contract.plugin_id,
            contract.plugin_version,
            contract.generation,
            contract.scope_id,
        )
        existing = self._contracts.get(key)
        if existing is not None and existing != contract:
            raise ValueError(f"deployment scope already registered with different content: {key}")
        self._contracts[key] = contract

    def contracts_for(self, plugin_id: str) -> tuple[DeploymentScopeContract, ...]:
        values = [
            contract
            for (pid, _version, _generation, _scope), contract in self._contracts.items()
            if pid == plugin_id
        ]
        return tuple(
            sorted(
                values,
                key=lambda item: (
                    item.generation,
                    item.plugin_version or "",
                    item.scope_id,
                ),
            )
        )

    def export_contracts(self, plugin_id: str) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.contracts_for(plugin_id)]

    def import_contracts(self, payloads: list[Mapping[str, Any]]) -> None:
        for payload in payloads:
            self.register(DeploymentScopeContract.from_dict(payload))

    def evaluate(
        self,
        *,
        plugin_id: str,
        generation: int,
        profile: DeploymentInputProfile,
        scope_id: str | None = None,
        plugin_version: str | None = None,
    ) -> DeploymentQualificationDecision:
        all_contracts = self.contracts_for(plugin_id)
        current_generation = [
            contract for contract in all_contracts if contract.generation == generation
        ]
        if scope_id is not None:
            current_generation = [
                contract for contract in current_generation if contract.scope_id == scope_id
            ]

        if not current_generation:
            if all_contracts:
                known_generations = {contract.generation for contract in all_contracts}
                if generation not in known_generations:
                    return DeploymentQualificationDecision(
                        plugin_id=plugin_id,
                        generation=generation,
                        scope_id=scope_id,
                        status=DeploymentQualificationStatus.STALE_QUALIFICATION,
                        plugin_version=plugin_version,
                        reason_codes=("generation_not_qualified",),
                    )
            return DeploymentQualificationDecision(
                plugin_id=plugin_id,
                generation=generation,
                scope_id=scope_id,
                status=DeploymentQualificationStatus.UNQUALIFIED,
                plugin_version=plugin_version,
                reason_codes=("no_matching_scope_contract",),
            )

        version_specific = [
            contract for contract in current_generation if contract.plugin_version is not None
        ]
        version_agnostic = [
            contract for contract in current_generation if contract.plugin_version is None
        ]
        if plugin_version is None:
            eligible = version_agnostic
            if not eligible and version_specific:
                return DeploymentQualificationDecision(
                    plugin_id=plugin_id,
                    generation=generation,
                    scope_id=scope_id,
                    status=DeploymentQualificationStatus.STALE_QUALIFICATION,
                    plugin_version=None,
                    reason_codes=("plugin_version_missing",),
                )
        else:
            eligible = version_agnostic + [
                contract
                for contract in version_specific
                if contract.plugin_version == plugin_version
            ]
            if not eligible and version_specific:
                return DeploymentQualificationDecision(
                    plugin_id=plugin_id,
                    generation=generation,
                    scope_id=scope_id,
                    status=DeploymentQualificationStatus.STALE_QUALIFICATION,
                    plugin_version=plugin_version,
                    reason_codes=("plugin_version_not_qualified",),
                )

        # 任一同 generation / executable-version 的 scope 完全满足即可部署。
        failures: list[DeploymentQualificationDecision] = []
        for contract in eligible:
            decision = self._evaluate_contract(contract, profile)
            if decision.qualified:
                return decision
            failures.append(decision)

        # scope_id 明确时只有一个预期 contract；未明确时返回信息量最大的失败。
        failures.sort(
            key=lambda item: (
                len(item.reason_codes),
                item.scope_id or "",
            )
        )
        return failures[0]

    @staticmethod
    def _evaluate_contract(
        contract: DeploymentScopeContract,
        profile: DeploymentInputProfile,
    ) -> DeploymentQualificationDecision:
        missing_context: list[str] = []
        mismatched_context: list[str] = []
        for key, expected in contract.required_context.items():
            observed = profile.context.get(key)
            if observed is None:
                missing_context.append(key)
            elif observed != expected:
                mismatched_context.append(f"{key}:{observed}!={expected}")

        missing_features = sorted(contract.required_features - profile.features)
        forbidden_present = sorted(contract.forbidden_features & profile.features)
        reasons: list[str] = []
        if missing_context:
            reasons.append("required_context_missing")
        if mismatched_context:
            reasons.append("required_context_mismatch")
        if missing_features:
            reasons.append("required_feature_missing")
        if forbidden_present:
            reasons.append("forbidden_feature_present")

        status = (
            DeploymentQualificationStatus.QUALIFIED
            if not reasons
            else DeploymentQualificationStatus.QUARANTINE
        )
        return DeploymentQualificationDecision(
            plugin_id=contract.plugin_id,
            generation=contract.generation,
            scope_id=contract.scope_id,
            status=status,
            plugin_version=contract.plugin_version,
            reason_codes=tuple(reasons),
            missing_context=tuple(sorted(missing_context)),
            mismatched_context=tuple(sorted(mismatched_context)),
            missing_features=tuple(missing_features),
            forbidden_features_present=tuple(forbidden_present),
        )


class LearnedStatePlugin(Protocol):
    """部署门只需要的最小 stateful-plugin contract。"""

    plugin_id: str

    @property
    def learned_state(self): ...

    def invoke(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ScopedInvocationResult:
    """部署门后的调用结果；未 qualified 时绝不调用插件。"""

    qualification: DeploymentQualificationDecision
    invoked: bool
    output: Mapping[str, Any] | None = None


class DeploymentQualifiedInvoker:
    """在 capability invocation 前强制执行 generation-specific deployment gate。"""

    def __init__(
        self,
        *,
        plugin: LearnedStatePlugin,
        qualifier: CapabilityDeploymentQualifier,
    ) -> None:
        self.plugin = plugin
        self.qualifier = qualifier

    def invoke(
        self,
        payload: Mapping[str, Any],
        *,
        profile: DeploymentInputProfile,
        scope_id: str | None = None,
    ) -> ScopedInvocationResult:
        generation = int(self.plugin.learned_state.generation)
        plugin_version = str(getattr(self.plugin, "version", "")).strip() or None
        decision = self.qualifier.evaluate(
            plugin_id=self.plugin.plugin_id,
            generation=generation,
            profile=profile,
            scope_id=scope_id,
            plugin_version=plugin_version,
        )
        if not decision.qualified:
            return ScopedInvocationResult(
                qualification=decision,
                invoked=False,
                output=None,
            )
        output = self.plugin.invoke(payload)
        return ScopedInvocationResult(
            qualification=decision,
            invoked=True,
            output=output,
        )
