"""Multiplex Change-Domain Recovery (MCDR) 的只读 V0 推断器。

算法采用 query-conditioned personalized diffusion。其用途是验证“不同 Harness
耦合信号能否恢复有用的 evolution boundary”，并不主张扩散算法本身具有新颖性。
"""

from __future__ import annotations

from collections import deque

from .models import (
    ChangeDomainResult,
    ChangeRequest,
    CouplingLayer,
    MultiplexHarnessGraph,
)


def _effective_layers(
    graph: MultiplexHarnessGraph,
    request: ChangeRequest,
) -> set[CouplingLayer]:
    available = graph.layers()
    if request.enabled_layers is None:
        return available
    return available & set(request.enabled_layers)


def _layer_weight(request: ChangeRequest, layer: CouplingLayer) -> float:
    return float(request.layer_weights.get(layer, 1.0))


def _weighted_adjacency(
    graph: MultiplexHarnessGraph,
    request: ChangeRequest,
    layers: set[CouplingLayer],
) -> dict[str, dict[str, float]]:
    adjacency: dict[str, dict[str, float]] = {
        node_id: {} for node_id in graph.artifacts
    }
    for edge in graph.edges_for_layers(layers):
        layer_weight = _layer_weight(request, edge.layer)
        if layer_weight <= 0:
            continue
        weight = float(edge.weight) * layer_weight
        adjacency[edge.source][edge.target] = (
            adjacency[edge.source].get(edge.target, 0.0) + weight
        )
        # V0 恢复的是共同演化边界，因此即使原关系带方向，也先作为对称耦合使用。
        adjacency[edge.target][edge.source] = (
            adjacency[edge.target].get(edge.source, 0.0) + weight
        )
    return adjacency


_DIRECTIONAL_SCIENTIFIC_RELATIONS = {
    "requires_context_field",
    "requires_evidence_kind",
    "queries_context_field",
    "declares_scientific_responsibility",
    "references_scientific_field",
    "kg_evidence_supports_change",
    "extracts_context_observation",
    "instantiates_context_field",
    "supported_by_exact_chunk",
    "recorded_as_runtime_evidence",
    "supports_runtime_claim",
    "authorized_runtime_use",
    "requires_revalidation",
    "evaluates_source_applicability",
    "blocked_source_context",
    "direct_source_context",
}


def _closure_adjacency(
    graph: MultiplexHarnessGraph,
    request: ChangeRequest,
    layers: set[CouplingLayer],
) -> dict[str, dict[str, float]]:
    """为强闭包构造 typed meta-path 邻接。

    普通 control/state/service/behavior/regression 耦合仍按无向关系处理；只有明确的
    scientific responsibility 边保留其 impact 方向，防止通过多字段消费者反向跳到
    不相关科学字段形成 hub expansion。
    """

    adjacency: dict[str, dict[str, float]] = {
        node_id: {} for node_id in graph.artifacts
    }
    for edge in graph.edges_for_layers(layers):
        layer_weight = _layer_weight(request, edge.layer)
        if layer_weight <= 0:
            continue
        weight = float(edge.weight) * layer_weight
        adjacency[edge.source][edge.target] = (
            adjacency[edge.source].get(edge.target, 0.0) + weight
        )
        preserve_direction = (
            edge.layer is CouplingLayer.SCIENTIFIC
            and edge.directed
            and edge.relation in _DIRECTIONAL_SCIENTIFIC_RELATIONS
        )
        if not preserve_direction:
            adjacency[edge.target][edge.source] = (
                adjacency[edge.target].get(edge.source, 0.0) + weight
            )
    return adjacency


def _normalize_rows(
    adjacency: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    transitions: dict[str, dict[str, float]] = {}
    for source, neighbors in adjacency.items():
        total = sum(neighbors.values())
        if total <= 0:
            transitions[source] = {}
            continue
        transitions[source] = {
            target: weight / total for target, weight in neighbors.items()
        }
    return transitions


def _strong_bottleneck_closure(
    *,
    adjacency: dict[str, dict[str, float]],
    seeds: set[str],
    relative_strength: float | None,
) -> set[str]:
    """沿强耦合边恢复不会随路径长度指数衰减的局部闭包。

    阈值相对于当前 query 图中最强有效边定义。V0 先用这个简单、可解释的
    bottleneck 规则检验“强多跳链是否构成共同演化域”；若 development gate 通过，
    再研究更稳健的局部 conductance / size regularization。
    """

    if relative_strength is None:
        return set(seeds)
    max_weight = max(
        (weight for neighbors in adjacency.values() for weight in neighbors.values()),
        default=0.0,
    )
    if max_weight <= 0:
        return set(seeds)
    cutoff = max_weight * relative_strength
    closure = set(seeds)
    queue: deque[str] = deque(sorted(seeds))
    while queue:
        current = queue.popleft()
        for neighbor, weight in sorted(adjacency.get(current, {}).items()):
            if weight < cutoff or neighbor in closure:
                continue
            closure.add(neighbor)
            queue.append(neighbor)
    return closure


def _explanation_paths(
    *,
    adjacency: dict[str, dict[str, float]],
    seeds: set[str],
    members: set[str],
) -> dict[str, tuple[str, ...]]:
    """为每个成员返回一条最短 seed->member 结构路径。"""

    paths: dict[str, tuple[str, ...]] = {seed: (seed,) for seed in seeds if seed in members}
    queue: deque[str] = deque(paths)
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency.get(current, {})):
            if neighbor not in members or neighbor in paths:
                continue
            paths[neighbor] = paths[current] + (neighbor,)
            queue.append(neighbor)
    return paths


def recover_change_domain(
    graph: MultiplexHarnessGraph,
    request: ChangeRequest,
) -> ChangeDomainResult:
    """从多层耦合图恢复一次 query-conditioned Change Domain。

    选择规则使用相对阈值：保留所有 seed，并保留分数至少达到最高非 seed 分数
    ``relative_threshold`` 比例的节点。该规则用于 V0 的可证伪比较，后续若通过
    development gate 才值得研究更复杂的 local-conductance / size-regularized selector。
    """

    missing = [seed for seed in request.seed_artifact_ids if seed not in graph.artifacts]
    if missing:
        raise ValueError("unknown seed artifacts: " + ", ".join(missing))

    layers = _effective_layers(graph, request)
    adjacency = _weighted_adjacency(graph, request, layers)
    closure_adjacency = _closure_adjacency(graph, request, layers)
    transitions = _normalize_rows(adjacency)
    node_ids = tuple(sorted(graph.artifacts))
    seeds = set(request.seed_artifact_ids)
    seed_mass = 1.0 / len(seeds)
    restart = {
        node_id: (seed_mass if node_id in seeds else 0.0) for node_id in node_ids
    }
    scores = dict(restart)

    iterations = 0
    rho = 1.0 - request.restart_probability
    for step in range(1, request.max_steps + 1):
        next_scores = {
            node_id: request.restart_probability * restart[node_id]
            for node_id in node_ids
        }
        dangling_mass = 0.0
        for source in node_ids:
            outgoing = transitions[source]
            if not outgoing:
                dangling_mass += rho * scores[source]
                continue
            for target, probability in outgoing.items():
                next_scores[target] += rho * scores[source] * probability
        # 将孤立/无出边质量送回查询 seed，避免无意义丢失概率质量。
        if dangling_mass:
            for seed in seeds:
                next_scores[seed] += dangling_mass * seed_mass
        delta = sum(abs(next_scores[node] - scores[node]) for node in node_ids)
        scores = next_scores
        iterations = step
        if delta <= request.tolerance:
            break

    non_seed_scores = [score for node, score in scores.items() if node not in seeds]
    reference = max(non_seed_scores, default=0.0)
    cutoff = reference * request.relative_threshold
    members = set(seeds)
    if reference > 0:
        members.update(
            node_id
            for node_id, score in scores.items()
            if node_id not in seeds and score >= cutoff
        )
    members.update(
        _strong_bottleneck_closure(
            adjacency=closure_adjacency,
            seeds=seeds,
            relative_strength=request.closure_relative_strength,
        )
    )

    paths = _explanation_paths(adjacency=adjacency, seeds=seeds, members=members)
    # 任何没有结构解释路径的非 seed 成员都不应进入 V0 domain。
    members = {node_id for node_id in members if node_id in paths}
    ordered_members = tuple(
        sorted(members, key=lambda node_id: (-scores[node_id], node_id))
    )
    return ChangeDomainResult(
        request_id=request.request_id,
        member_ids=ordered_members,
        scores={node_id: scores[node_id] for node_id in ordered_members},
        explanation_paths={node_id: paths[node_id] for node_id in ordered_members},
        iterations=iterations,
        enabled_layers=tuple(sorted(layers, key=lambda layer: layer.value)),
    )
