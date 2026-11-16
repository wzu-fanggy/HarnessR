"""First-class lifecycle coordinates for long-lived capabilities.

Tool, Skill, plugin, process, and session describe implementation/execution
forms. A long-lived Capability is identified instead by lifecycle coordinates
that survive or change independently across those forms:

    I / V / G / Q / R / A

This module does not introduce a new runtime state store. It materializes and
cross-checks state already owned by the executable, deployment qualifier, and
residency governor so callers can reason about one capability as a persistent
logical object rather than as a single tool/plugin instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .deployment import (
    DeploymentQualificationDecision,
    DeploymentQualificationStatus,
)
from .model import CapabilityRuntimeState, ResidencyLevel


class LifecycleExecutable(Protocol):
    """Minimal executable surface needed to read I/V/G."""

    plugin_id: str
    version: str

    @property
    def learned_state(self): ...


@dataclass(frozen=True)
class CapabilityLifecycleCoordinates:
    """Machine-checkable I/V/G/Q/R/A view of one logical capability.

    ``qualification`` is optional because a capability may exist before any
    deployment contract has been evaluated. When present, its identity,
    generation, and executable version must refer to these same coordinates;
    stale/unqualified *status* is allowed and is itself meaningful Q state.
    """

    identity: str
    executable_version: str
    learned_generation: int
    qualification: DeploymentQualificationDecision | None
    residency: ResidencyLevel
    active_leases: int

    def __post_init__(self) -> None:
        identity = self.identity.strip()
        version = self.executable_version.strip()
        if not identity:
            raise ValueError("capability identity is required")
        if not version:
            raise ValueError("executable version is required")
        if self.learned_generation < 0:
            raise ValueError("learned generation must be non-negative")
        if self.active_leases < 0:
            raise ValueError("active lease count must be non-negative")

        qualification = self.qualification
        if qualification is not None:
            if qualification.plugin_id != identity:
                raise ValueError("qualification identity does not match capability identity")
            if qualification.generation != self.learned_generation:
                raise ValueError("qualification generation does not match learned generation")
            qualified_version = (
                str(qualification.plugin_version).strip()
                if qualification.plugin_version is not None
                else None
            )
            if qualified_version is not None and qualified_version != version:
                raise ValueError("qualification executable version does not match capability version")

        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "executable_version", version)

    @property
    def I(self) -> str:
        return self.identity

    @property
    def V(self) -> str:
        return self.executable_version

    @property
    def G(self) -> int:
        return self.learned_generation

    @property
    def Q(self) -> DeploymentQualificationStatus | None:
        return self.qualification.status if self.qualification is not None else None

    @property
    def R(self) -> ResidencyLevel:
        return self.residency

    @property
    def A(self) -> bool:
        return self.active_leases > 0

    def to_dict(self) -> dict[str, object]:
        qualification = self.qualification
        return {
            "I": self.I,
            "V": self.V,
            "G": self.G,
            "Q": self.Q.value if self.Q is not None else None,
            "Q_scope": qualification.scope_id if qualification is not None else None,
            "R": self.R.name,
            "A": self.A,
            "active_leases": self.active_leases,
        }

    @classmethod
    def from_runtime(
        cls,
        *,
        executable: LifecycleExecutable,
        runtime_state: CapabilityRuntimeState,
        qualification: DeploymentQualificationDecision | None = None,
    ) -> "CapabilityLifecycleCoordinates":
        """Assemble coordinates from existing executable/governor/qualification state."""

        identity = str(executable.plugin_id).strip()
        if runtime_state.capability_id != identity:
            raise ValueError("runtime-state identity does not match executable identity")
        generation = int(executable.learned_state.generation)
        return cls(
            identity=identity,
            executable_version=str(executable.version),
            learned_generation=generation,
            qualification=qualification,
            residency=runtime_state.current_level,
            active_leases=int(runtime_state.active_count),
        )
