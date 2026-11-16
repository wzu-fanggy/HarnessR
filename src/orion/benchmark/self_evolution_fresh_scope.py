"""Fresh-study scientific scope representation and boundary relation semantics.

The historical production applicability schema stores ``temperature_c`` as one
scalar. That is intentionally left unchanged for audit compatibility, but it is
not expressive enough for the fresh self-evolution study, whose source evidence
may state intervals, discrete sets, or one-sided scientific bounds.

This module is a benchmark-only sidecar. It does not mutate production scope
objects or historical v1-v5 artifacts.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.task_evidence.evidence_applicability import ConditionField


NumericDomainKind = Literal["scalar", "interval", "set", "lower_bound", "upper_bound"]
BoundaryRelation = Literal[
    "unresolved",
    "resolved_non_disqualifying",
    "disqualifying_match",
]


class NumericScientificDomain(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: NumericDomainKind
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    values: list[float] = Field(default_factory=list)
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_shape(self) -> "NumericScientificDomain":
        if self.kind == "scalar":
            if self.value is None or self.lower is not None or self.upper is not None or self.values:
                raise ValueError("scalar requires value only")
        elif self.kind == "interval":
            if self.lower is None or self.upper is None or not self.lower < self.upper:
                raise ValueError("interval requires ordered lower < upper")
            if self.value is not None or self.values:
                raise ValueError("interval must not contain scalar/set values")
        elif self.kind == "set":
            if not self.values:
                raise ValueError("set requires at least one value")
            if len(self.values) != len(set(self.values)):
                raise ValueError("set values must be unique")
            if self.value is not None or self.lower is not None or self.upper is not None:
                raise ValueError("set must contain values only")
        elif self.kind == "lower_bound":
            if self.lower is None or self.value is not None or self.upper is not None or self.values:
                raise ValueError("lower_bound requires lower only")
        elif self.kind == "upper_bound":
            if self.upper is None or self.value is not None or self.lower is not None or self.values:
                raise ValueError("upper_bound requires upper only")
        return self


class FreshScientificScope(BaseModel):
    """Benchmark-only faithful scope for one reusable scientific rule."""

    model_config = ConfigDict(extra="forbid")

    task_types: list[str] = Field(min_length=1)
    scientific_object_id: str | None = None
    composition: str | None = None
    salt: str | None = None
    solvent: str | None = None
    additive: str | None = None
    temperature_c: NumericScientificDomain | None = None
    electrode: str | None = None
    cell_configuration: str | None = None
    voltage_window_v: NumericScientificDomain | None = None
    rate: str | None = None
    cycle_count: int | None = Field(default=None, ge=0)
    measurement_definition: str | None = None

    @model_validator(mode="after")
    def _units_are_correct(self) -> "FreshScientificScope":
        if self.temperature_c is not None and self.temperature_c.unit != "C":
            raise ValueError("temperature_c domain unit must be C")
        if self.voltage_window_v is not None and self.voltage_window_v.unit != "V":
            raise ValueError("voltage_window_v domain unit must be V")
        if len(self.task_types) != len(set(self.task_types)):
            raise ValueError("task_types must be unique")
        return self


class CategoricalScientificBoundary(BaseModel):
    """Fresh-only categorical adverse/non-transfer domain.

    ``canonical_aliases`` maps a frozen canonical value to the exact natural-
    language aliases allowed to resolve a sealed candidate scope string. The
    boundary itself is expressed as one or more adverse canonical values.
    Missing or unresolvable candidate scope remains ``unresolved`` rather than
    being treated as safely outside the boundary.
    """

    model_config = ConfigDict(extra="forbid")

    field: Literal["cell_configuration"]
    canonical_aliases: dict[str, list[str]] = Field(min_length=2)
    adverse_values: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_categories(self) -> "CategoricalScientificBoundary":
        canonical = set(self.canonical_aliases)
        if len(canonical) != len(self.canonical_aliases):
            raise ValueError("categorical canonical values must be unique")
        if not set(self.adverse_values).issubset(canonical):
            raise ValueError("adverse_values must be declared canonical values")
        for key, aliases in self.canonical_aliases.items():
            cleaned = [alias.strip().casefold() for alias in aliases if alias.strip()]
            if not cleaned:
                raise ValueError(f"categorical value {key} requires aliases")
            if len(cleaned) != len(set(cleaned)):
                raise ValueError(f"categorical aliases for {key} must be unique")
        return self


def resolve_categorical_scope_value(
    candidate_value: str | None,
    boundary: CategoricalScientificBoundary,
) -> str | None:
    """Resolve a natural candidate scope string to exactly one canonical value."""

    if candidate_value is None or not candidate_value.strip():
        return None
    folded = candidate_value.casefold()
    matches: list[str] = []
    for canonical, aliases in boundary.canonical_aliases.items():
        if any(alias.strip().casefold() in folded for alias in aliases if alias.strip()):
            matches.append(canonical)
    if len(matches) != 1:
        return None
    return matches[0]


def classify_categorical_boundary_relation(
    candidate_value: str | None,
    boundary: CategoricalScientificBoundary,
) -> BoundaryRelation:
    """Apply the same open-world three-state rule to a categorical condition."""

    resolved = resolve_categorical_scope_value(candidate_value, boundary)
    if resolved is None:
        return "unresolved"
    if resolved in set(boundary.adverse_values):
        return "disqualifying_match"
    return "resolved_non_disqualifying"


class FreshPersistentUpdateProposal(BaseModel):
    """Common development-only proposer output for the new fresh study."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["orion_self_evolution_fresh_proposal_v1"] = (
        "orion_self_evolution_fresh_proposal_v1"
    )
    proposal_status: Literal["candidate", "no_update"]
    proposed_content: str = ""
    claimed_scope: FreshScientificScope | None = None
    relevant_fields: list[ConditionField] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)
    development_rationale: str = ""
    limitations_visible_from_development: list[str] = Field(default_factory=list)
    no_update_reason: str = ""

    @model_validator(mode="after")
    def _candidate_or_no_update(self) -> "FreshPersistentUpdateProposal":
        if len(self.relevant_fields) != len(set(self.relevant_fields)):
            raise ValueError("relevant_fields must be unique")
        if self.proposal_status == "candidate":
            if self.claimed_scope is None:
                raise ValueError("candidate requires claimed_scope")
            if not self.proposed_content.strip():
                raise ValueError("candidate requires proposed_content")
            if not self.source_evidence_ids:
                raise ValueError("candidate requires source_evidence_ids")
            if self.no_update_reason.strip():
                raise ValueError("candidate must not contain no_update_reason")
        else:
            if self.claimed_scope is not None:
                raise ValueError("no_update must not claim scientific scope")
            if self.proposed_content.strip() or self.relevant_fields or self.source_evidence_ids:
                raise ValueError("no_update must not contain persistent scientific content")
            if not self.no_update_reason.strip():
                raise ValueError("no_update requires reason")
        return self


def _intervals(domain: NumericScientificDomain) -> list[tuple[float, float, bool, bool]]:
    """Convert a numeric domain to interval primitives; set elements stay discrete."""

    if domain.kind == "scalar":
        assert domain.value is not None
        return [(domain.value, domain.value, True, True)]
    if domain.kind == "set":
        return [(value, value, True, True) for value in domain.values]
    if domain.kind == "interval":
        assert domain.lower is not None and domain.upper is not None
        return [(domain.lower, domain.upper, domain.lower_inclusive, domain.upper_inclusive)]
    if domain.kind == "lower_bound":
        assert domain.lower is not None
        return [(domain.lower, math.inf, domain.lower_inclusive, False)]
    assert domain.upper is not None
    return [(-math.inf, domain.upper, False, domain.upper_inclusive)]


def _overlap(
    left: tuple[float, float, bool, bool],
    right: tuple[float, float, bool, bool],
) -> bool:
    l_lo, l_hi, l_lo_inc, l_hi_inc = left
    r_lo, r_hi, r_lo_inc, r_hi_inc = right
    low = max(l_lo, r_lo)
    high = min(l_hi, r_hi)
    if low < high:
        return True
    if low > high:
        return False
    left_contains = (
        (low > l_lo or (low == l_lo and l_lo_inc))
        and (low < l_hi or (low == l_hi and l_hi_inc))
    )
    right_contains = (
        (low > r_lo or (low == r_lo and r_lo_inc))
        and (low < r_hi or (low == r_hi and r_hi_inc))
    )
    return left_contains and right_contains


def classify_numeric_boundary_relation(
    candidate: NumericScientificDomain | None,
    adverse_boundary: NumericScientificDomain,
) -> BoundaryRelation:
    """Classify whether an adverse/non-transfer numeric boundary hits candidate scope.

    Missing candidate scope remains unresolved. Explicit disjoint scope is a
    resolved non-disqualifying boundary. Any overlap is a disqualifying match;
    no after-the-fact narrowing is performed here.
    """

    if candidate is None:
        return "unresolved"
    if candidate.unit != adverse_boundary.unit:
        raise ValueError("candidate and boundary units must match")
    for c_interval in _intervals(candidate):
        for b_interval in _intervals(adverse_boundary):
            if _overlap(c_interval, b_interval):
                return "disqualifying_match"
    return "resolved_non_disqualifying"


__all__ = [
    "BoundaryRelation",
    "CategoricalScientificBoundary",
    "FreshPersistentUpdateProposal",
    "FreshScientificScope",
    "NumericScientificDomain",
    "classify_categorical_boundary_relation",
    "classify_numeric_boundary_relation",
    "resolve_categorical_scope_value",
]
