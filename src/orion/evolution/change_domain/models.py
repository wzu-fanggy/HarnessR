"""Change Domain V0 的类型化多层 Harness 图数据模型。

V0 的目标是把不同来源的耦合关系放进同一个可计算视图，而不是把所有数据
物理迁移到同一个 Neo4j 数据库。所有对象均为内存模型；生产 ``kg-v1`` 保持只读。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class CouplingLayer(str, Enum):
    """V0 明确区分的 Harness 耦合来源。"""

    CONTROL = "control"
    STATE = "state"
    SERVICE = "service"
    BEHAVIOR = "behavior"
    REGRESSION = "regression"
    SCIENTIFIC = "scientific"


@dataclass(frozen=True)
class HarnessArtifact:
    """一个可寻址的 Harness / task / scientific / test 图对象。

    ``artifact_id`` 是跨 extractor 的稳定 identity；``kind`` 保留首个投影视图的
    主类型，``roles`` 记录同一对象在不同耦合层中的全部角色。例如同一个函数可以
    同时是 ``stateful_function`` 和 ``scientific_field_consumer``，而不应复制成两个
    逻辑 artifact。
    """

    artifact_id: str
    kind: str
    label: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id is required")
        if not self.kind.strip():
            raise ValueError("artifact kind is required")
        normalized_roles = tuple(sorted({self.kind, *[role for role in self.roles if role]}))
        object.__setattr__(self, "roles", normalized_roles)

    def has_role(self, role: str) -> bool:
        return role in self.roles


@dataclass(frozen=True)
class CouplingEdge:
    """两个图对象之间的一条有来源的耦合边。

    ``directed`` 表示原始关系是否有方向。Change Domain 推断默认把演化耦合看作
    双向关联，但保留方向信息供后续 impact-analysis 版本使用。
    """

    source: str
    target: str
    layer: CouplingLayer
    weight: float = 1.0
    directed: bool = False
    relation: str = "coupled_to"
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("edge endpoints are required")
        if self.source == self.target:
            raise ValueError("self edges are not allowed in V0")
        if self.weight <= 0:
            raise ValueError("edge weight must be positive")


@dataclass
class MultiplexHarnessGraph:
    """由同一组 artifact 和多种 coupling layer 组成的只读计算视图。"""

    artifacts: dict[str, HarnessArtifact] = field(default_factory=dict)
    edges: list[CouplingEdge] = field(default_factory=list)

    def add_artifact(self, artifact: HarnessArtifact) -> None:
        existing = self.artifacts.get(artifact.artifact_id)
        if existing is None:
            self.artifacts[artifact.artifact_id] = artifact
            return
        if existing == artifact:
            return

        if existing.label and artifact.label and existing.label != artifact.label:
            raise ValueError(
                f"artifact id has conflicting labels: {artifact.artifact_id}: "
                f"{existing.label!r} != {artifact.label!r}"
            )
        metadata = dict(existing.metadata)
        for key, value in artifact.metadata.items():
            if key in metadata and metadata[key] not in {"", value} and value != "":
                raise ValueError(
                    f"artifact id has conflicting metadata: {artifact.artifact_id}:{key}: "
                    f"{metadata[key]!r} != {value!r}"
                )
            if value != "" or key not in metadata:
                metadata[key] = value
        self.artifacts[artifact.artifact_id] = HarnessArtifact(
            artifact_id=existing.artifact_id,
            kind=existing.kind,
            label=existing.label or artifact.label,
            metadata=metadata,
            roles=tuple(sorted(set(existing.roles) | set(artifact.roles))),
        )

    def add_edge(self, edge: CouplingEdge) -> None:
        missing = [node_id for node_id in (edge.source, edge.target) if node_id not in self.artifacts]
        if missing:
            raise ValueError("edge references unknown artifacts: " + ", ".join(missing))
        if edge not in self.edges:
            self.edges.append(edge)

    def layers(self) -> set[CouplingLayer]:
        return {edge.layer for edge in self.edges}

    def edges_for_layers(self, layers: set[CouplingLayer] | None = None) -> list[CouplingEdge]:
        if layers is None:
            return list(self.edges)
        return [edge for edge in self.edges if edge.layer in layers]


@dataclass(frozen=True)
class ChangeRequest:
    """一次待恢复 evolution boundary 的查询条件。"""

    request_id: str
    seed_artifact_ids: tuple[str, ...]
    layer_weights: Mapping[CouplingLayer, float] = field(default_factory=dict)
    restart_probability: float = 0.35
    relative_threshold: float = 0.18
    closure_relative_strength: float | None = 0.35
    max_steps: int = 80
    tolerance: float = 1e-10
    enabled_layers: frozenset[CouplingLayer] | None = None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id is required")
        if not self.seed_artifact_ids:
            raise ValueError("at least one seed artifact is required")
        if len(set(self.seed_artifact_ids)) != len(self.seed_artifact_ids):
            raise ValueError("seed artifacts must be unique")
        if not 0 < self.restart_probability < 1:
            raise ValueError("restart_probability must be in (0, 1)")
        if not 0 < self.relative_threshold <= 1:
            raise ValueError("relative_threshold must be in (0, 1]")
        if (
            self.closure_relative_strength is not None
            and not 0 < self.closure_relative_strength <= 1
        ):
            raise ValueError("closure_relative_strength must be in (0, 1] or None")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        for layer, weight in self.layer_weights.items():
            if not isinstance(layer, CouplingLayer):
                raise TypeError("layer_weights keys must be CouplingLayer values")
            if weight < 0:
                raise ValueError("layer weights must be non-negative")


@dataclass(frozen=True)
class ChangeDomainResult:
    """一次 query-conditioned Change Domain 推断结果。"""

    request_id: str
    member_ids: tuple[str, ...]
    scores: Mapping[str, float]
    explanation_paths: Mapping[str, tuple[str, ...]]
    iterations: int
    enabled_layers: tuple[CouplingLayer, ...]

    def contains(self, artifact_id: str) -> bool:
        return artifact_id in self.member_ids
