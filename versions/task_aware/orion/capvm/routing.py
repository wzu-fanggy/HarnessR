"""Evolvable Plugin Architecture 的 capability discovery / routing plane。



V9 的核心目的不是再造一个特定模型 router，而是把以下责任显式分离：



- Discovery：从常驻 catalog metadata 中找出小候选集；

- Control：判断候选是否具备 authority / Host / deployment 等硬约束；

- Semantic selection：在 admissible shortlist 中决定 SHOULD 使用哪个；

- Residency：最终 materialization 仍由 CapabilityResidencyGovernor 决定；

- Recovery：没有可用能力时返回结构化 fallback，而不是强行调用错误 capability。



重要不变量：语义可见性与执行权限不是一回事；routing plan 本身不得物化 backing

resource，也不得授予 authority。

"""



from __future__ import annotations



import math

import re

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

from contextlib import contextmanager

from dataclasses import dataclass, field

from enum import Enum

from typing import Protocol



from .catalog import CapabilityCatalog

from .deployment import (

    CapabilityDeploymentQualifier,

    DeploymentInputProfile,

)

from .governor import CapabilityResidencyGovernor

from .model import (

    CapabilityKind,

    MeasuredResidencyBudgetExceeded,

    ResidencyBudgetExceeded,

    ResidencyLevel,

    VirtualCapability,

)





_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._:+/-][a-z0-9]+)*|[\u4e00-\u9fff]", re.IGNORECASE)





def _tokens(text: str) -> tuple[str, ...]:

    return tuple(_TOKEN_RE.findall(text.casefold()))





def _string_tuple(value: object) -> tuple[str, ...]:

    if value is None:

        return ()

    if isinstance(value, str):

        values = (value,)

    elif isinstance(value, Iterable):

        values = tuple(str(item) for item in value)

    else:

        values = (str(value),)

    normalized = tuple(item.strip() for item in values if item.strip())

    return tuple(dict.fromkeys(normalized))





def _parse_residency(value: object, *, default: ResidencyLevel) -> ResidencyLevel:

    if value is None:

        return default

    if isinstance(value, ResidencyLevel):

        return value

    if isinstance(value, int):

        return ResidencyLevel(value)

    normalized = str(value).strip().upper()

    try:

        return ResidencyLevel[normalized]

    except KeyError as exc:

        raise ValueError(f"unknown routing residency level: {value}") from exc





@dataclass(frozen=True)

class CapabilityDescriptor:

    """常驻 catalog 层的轻量 routing 描述符。



    描述符只引用 metadata，不持有或加载 graph/process/session 等 backing state。

    """



    capability_id: str

    kind: CapabilityKind

    summary: str

    version: str = "0"

    triggers: tuple[str, ...] = ()

    semantic_terms: tuple[str, ...] = ()

    aliases: tuple[str, ...] = ()

    examples: tuple[str, ...] = ()

    intent_contract: tuple[str, ...] = ()

    negative_boundaries: tuple[str, ...] = ()

    input_contract: tuple[str, ...] = ()

    output_contract: tuple[str, ...] = ()

    required_authorities: frozenset[str] = frozenset()

    required_host_features: frozenset[str] = frozenset()

    forbidden_host_features: frozenset[str] = frozenset()

    required_level: ResidencyLevel = ResidencyLevel.BOUND

    learned_generation: int | None = None

    deployment_scope_id: str | None = None

    metadata: Mapping[str, object] = field(default_factory=dict)



    @classmethod

    def from_capability(cls, capability: VirtualCapability) -> "CapabilityDescriptor":

        metadata = dict(capability.metadata)

        generation_value = metadata.get("learned_generation")

        generation = int(generation_value) if generation_value is not None else None

        scope_value = metadata.get("deployment_scope_id")

        scope_id = str(scope_value).strip() if scope_value is not None else None

        if scope_id == "":

            scope_id = None



        required_level = _parse_residency(

            metadata.get("routing_required_level"),

            default=min(ResidencyLevel.BOUND, capability.maximum_level),

        )

        if required_level > capability.maximum_level:

            raise ValueError(

                f"routing required level exceeds capability maximum: {capability.capability_id}"

            )



        return cls(

            capability_id=capability.capability_id,

            kind=capability.kind,

            summary=capability.summary,

            version=str(metadata.get("version", "0")),

            triggers=_string_tuple(metadata.get("routing_triggers") or metadata.get("triggers")),

            semantic_terms=_string_tuple(metadata.get("routing_terms")),

            aliases=_string_tuple(metadata.get("routing_aliases") or metadata.get("aliases")),

            examples=_string_tuple(metadata.get("routing_examples") or metadata.get("examples")),

            intent_contract=_string_tuple(metadata.get("routing_intent_contract")),

            negative_boundaries=_string_tuple(metadata.get("routing_negative_boundaries")),

            input_contract=_string_tuple(metadata.get("input_contract")),

            output_contract=_string_tuple(metadata.get("output_contract")),

            required_authorities=frozenset(

                _string_tuple(metadata.get("required_authorities"))

            ),

            required_host_features=frozenset(

                _string_tuple(metadata.get("required_host_features"))

            ),

            forbidden_host_features=frozenset(

                _string_tuple(metadata.get("forbidden_host_features"))

            ),

            required_level=required_level,

            learned_generation=generation,

            deployment_scope_id=scope_id,

            metadata=metadata,

        )



    def searchable_text(self) -> str:

        fields = (

            self.capability_id,

            self.kind.value,

            self.summary,

            *self.triggers,

            *self.semantic_terms,

            *self.aliases,

            *self.examples,

            *self.input_contract,

            *self.output_contract,

        )

        return " ".join(item for item in fields if item)



    def semantic_routing_text(self) -> str:

        """供 SHOULD selector 使用的 task-role 视图，而不是全文内容摘要。



        ``intent_contract`` 显式描述 capability 何时应该主导任务；它与 capability 内部

        知识主题分开，避免两个都谈“电解液/证据”的能力仅因内容相似而混淆。

        negative boundaries 以带标签的形式保留给更强 selector，但不混入 cheap lexical

        retrieval 的 searchable_text。

        """



        sections = [

            f"capability: {self.capability_id}",

            f"summary: {self.summary}",

        ]

        if self.intent_contract:

            sections.append("task intent: " + "; ".join(self.intent_contract))

        if self.input_contract:

            sections.append("required input role: " + "; ".join(self.input_contract))

        if self.output_contract:

            sections.append("expected output role: " + "; ".join(self.output_contract))

        if self.examples:

            sections.append("examples: " + "; ".join(self.examples))

        if self.negative_boundaries:

            sections.append("do not choose when: " + "; ".join(self.negative_boundaries))

        return "\n".join(sections)





@dataclass(frozen=True)

class RoutingRequest:

    """一次 capability routing 请求。"""



    task_text: str

    context: Mapping[str, str] = field(default_factory=dict)

    host_features: frozenset[str] = frozenset()

    authorities: frozenset[str] = frozenset()

    allowed_kinds: frozenset[CapabilityKind] | None = None

    max_candidates: int = 5

    minimum_discovery_score: float = 0.0

    minimum_matched_terms: int = 1

    minimum_query_coverage: float = 0.0

    deployment_profile: DeploymentInputProfile | None = None

    allow_capability_proposal: bool = True



    def __post_init__(self) -> None:

        if not self.task_text.strip():

            raise ValueError("task_text is required")

        if self.max_candidates <= 0:

            raise ValueError("max_candidates must be positive")

        if self.minimum_discovery_score < 0:

            raise ValueError("minimum_discovery_score must be non-negative")

        if self.minimum_matched_terms <= 0:

            raise ValueError("minimum_matched_terms must be positive")

        if not 0.0 <= self.minimum_query_coverage <= 1.0:

            raise ValueError("minimum_query_coverage must be between 0 and 1")

        object.__setattr__(

            self,

            "context",

            {str(key).strip(): str(value).strip() for key, value in self.context.items()},

        )

        object.__setattr__(

            self,

            "host_features",

            frozenset(str(item).strip() for item in self.host_features if str(item).strip()),

        )

        object.__setattr__(

            self,

            "authorities",

            frozenset(str(item).strip() for item in self.authorities if str(item).strip()),

        )





@dataclass(frozen=True)

class DiscoveryCandidate:

    """仅由 catalog metadata 产生的候选，不代表可执行。"""



    descriptor: CapabilityDescriptor

    score: float

    matched_terms: tuple[str, ...]

    query_token_count: int

    query_coverage: float

    discovery_rank: int



    @property

    def capability_id(self) -> str:

        return self.descriptor.capability_id





class CapabilityDiscoveryBackend(Protocol):

    """可替换的 metadata / embedding / learned discovery backend。"""



    def discover(

        self,

        request: RoutingRequest,

        descriptors: Sequence[CapabilityDescriptor],

    ) -> tuple[DiscoveryCandidate, ...]: ...





class ExactTriggerDiscoveryBackend:

    """第一阶段低成本 trigger matcher。



    只读取 descriptor 中显式声明的 trigger；命中时直接形成小候选集。该阶段对应

    structured-skill 系统常见的 cheap trigger path，不调用 embedding/LLM，也不物化

    capability backing state。

    """



    def discover(

        self,

        request: RoutingRequest,

        descriptors: Sequence[CapabilityDescriptor],

    ) -> tuple[DiscoveryCandidate, ...]:

        query_text = request.task_text.casefold()

        query_tokens = set(_tokens(request.task_text))

        ranked: list[tuple[float, str, tuple[str, ...], CapabilityDescriptor]] = []

        for descriptor in descriptors:

            matched_triggers = tuple(

                trigger

                for trigger in descriptor.triggers

                if trigger.casefold() in query_text

            )

            if not matched_triggers:

                continue

            matched_tokens = tuple(

                sorted(

                    set().union(

                        *(set(_tokens(trigger)) & query_tokens for trigger in matched_triggers)

                    )

                )

            )

            score = 100.0 + max(len(_tokens(trigger)) for trigger in matched_triggers)

            ranked.append((score, descriptor.capability_id, matched_tokens, descriptor))



        ranked.sort(key=lambda item: (-item[0], item[1]))

        return tuple(

            DiscoveryCandidate(

                descriptor=item[3],

                score=float(item[0]),

                matched_terms=item[2],

                query_token_count=len(query_tokens),

                query_coverage=(

                    len(item[2]) / len(query_tokens) if query_tokens else 0.0

                ),

                discovery_rank=index + 1,

            )

            for index, item in enumerate(ranked[: request.max_candidates])

        )





class MetadataDiscoveryBackend:

    """V9 development 的确定性 metadata retriever。



    这不是 novelty claim；正式系统可替换为 embedding、BM25、learned retriever 或

    provider-native tool search，而不改变后续 control/residency 语义。

    """



    def discover(

        self,

        request: RoutingRequest,

        descriptors: Sequence[CapabilityDescriptor],

    ) -> tuple[DiscoveryCandidate, ...]:

        query_tokens = set(_tokens(request.task_text))

        query_text = request.task_text.casefold().strip()

        ranked: list[tuple[float, str, tuple[str, ...], CapabilityDescriptor]] = []



        for descriptor in descriptors:

            field_tokens = {

                "id": set(_tokens(descriptor.capability_id)),

                "summary": set(_tokens(descriptor.summary)),

                "terms": set(_tokens(" ".join(descriptor.semantic_terms))),

                "aliases": set(_tokens(" ".join(descriptor.aliases))),

                "examples": set(_tokens(" ".join(descriptor.examples))),

                "contracts": set(

                    _tokens(" ".join((*descriptor.input_contract, *descriptor.output_contract)))

                ),

            }

            matched = set().union(*(tokens & query_tokens for tokens in field_tokens.values()))

            score = 0.0

            score += 5.0 * len(field_tokens["terms"] & query_tokens)

            score += 4.0 * len(field_tokens["aliases"] & query_tokens)

            score += 3.0 * len(field_tokens["id"] & query_tokens)

            score += 2.0 * len(field_tokens["summary"] & query_tokens)

            score += 1.5 * len(field_tokens["examples"] & query_tokens)

            score += 1.0 * len(field_tokens["contracts"] & query_tokens)



            searchable = descriptor.searchable_text().casefold()

            if query_text and query_text in searchable:

                score += 6.0



            query_token_count = len(query_tokens)

            query_coverage = (

                len(matched) / query_token_count if query_token_count else 0.0

            )

            if score <= 0 or score < request.minimum_discovery_score:

                continue

            if len(matched) < request.minimum_matched_terms:

                continue

            if query_coverage < request.minimum_query_coverage:

                continue

            ranked.append(

                (score, descriptor.capability_id, tuple(sorted(matched)), descriptor)

            )



        ranked.sort(key=lambda item: (-item[0], item[1]))

        limited = ranked[: request.max_candidates]

        return tuple(

            DiscoveryCandidate(

                descriptor=item[3],

                score=float(item[0]),

                matched_terms=item[2],

                query_token_count=len(query_tokens),

                query_coverage=(

                    len(item[2]) / len(query_tokens) if query_tokens else 0.0

                ),

                discovery_rank=index + 1,

            )

            for index, item in enumerate(limited)

        )





class EmbeddingDiscoveryBackend:

    """可插拔的向量语义检索阶段。



    ``encoder`` 只需要把一批文本编码为等长数值向量；因此 runtime 不绑定某个具体

    embedding provider。生产系统可以接 bge-m3、远程 embedding、learned retriever，

    测试也可以注入确定性 encoder。该阶段只读取 descriptor 文本，不物化 capability。

    """



    def __init__(

        self,

        *,

        encoder: Callable[[Sequence[str]], Sequence[Sequence[float]]],

        minimum_similarity: float = 0.25,

        minimum_query_vector_norm: float = 0.0,

    ) -> None:

        if not -1.0 <= minimum_similarity <= 1.0:

            raise ValueError("minimum_similarity must be between -1 and 1")

        if minimum_query_vector_norm < 0.0:

            raise ValueError("minimum_query_vector_norm must be non-negative")

        self.encoder = encoder

        self.minimum_similarity = float(minimum_similarity)

        self.minimum_query_vector_norm = float(minimum_query_vector_norm)



    @staticmethod

    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:

        if len(left) != len(right):

            raise ValueError("embedding dimensions must match")

        if not left:

            raise ValueError("embedding vectors must not be empty")

        dot = sum(float(a) * float(b) for a, b in zip(left, right))

        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))

        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))

        if left_norm == 0.0 or right_norm == 0.0:

            return 0.0

        return dot / (left_norm * right_norm)



    def discover(

        self,

        request: RoutingRequest,

        descriptors: Sequence[CapabilityDescriptor],

    ) -> tuple[DiscoveryCandidate, ...]:

        if not descriptors:

            return ()

        texts = [request.task_text, *(item.searchable_text() for item in descriptors)]

        encoded = tuple(tuple(float(value) for value in vector) for vector in self.encoder(texts))

        if len(encoded) != len(texts):

            raise ValueError("embedding encoder returned an unexpected vector count")

        query_vector = encoded[0]

        query_norm = math.sqrt(sum(value * value for value in query_vector))

        if query_norm < self.minimum_query_vector_norm:

            return ()

        query_tokens = set(_tokens(request.task_text))

        ranked: list[tuple[float, str, tuple[str, ...], CapabilityDescriptor]] = []

        for descriptor, vector in zip(descriptors, encoded[1:]):

            similarity = self._cosine(query_vector, vector)

            if similarity < self.minimum_similarity:

                continue

            descriptor_tokens = set(_tokens(descriptor.searchable_text()))

            matched = tuple(sorted(query_tokens & descriptor_tokens))

            ranked.append((similarity, descriptor.capability_id, matched, descriptor))



        ranked.sort(key=lambda item: (-item[0], item[1]))

        return tuple(

            DiscoveryCandidate(

                descriptor=item[3],

                score=float(item[0]),

                matched_terms=item[2],

                query_token_count=len(query_tokens),

                query_coverage=(len(item[2]) / len(query_tokens) if query_tokens else 0.0),

                discovery_rank=index + 1,

            )

            for index, item in enumerate(ranked[: request.max_candidates])

        )





class ReciprocalRankFusionDiscoveryBackend:

    """并行收集多路 discovery evidence，再用 RRF 形成候选并集。



    与 ``CascadedDiscoveryBackend`` 的 first-hit semantics 不同，本实现不会让 cheap

    trigger/metadata hit 获得终局 routing authority。每个 backend 都只是候选证据源；

    不同 raw score 不直接相加，而用成熟信息检索中的 reciprocal-rank fusion 统一排名。

    最终 SHOULD-use 顺序仍由独立 ``SemanticCapabilitySelector`` 决定。

    """



    def __init__(

        self,

        backends: Sequence[CapabilityDiscoveryBackend],

        *,

        weights: Sequence[float] | None = None,

        rrf_k: float = 60.0,

    ) -> None:

        self.backends = tuple(backends)

        if not self.backends:

            raise ValueError("RRF discovery requires at least one backend")

        if rrf_k <= 0:

            raise ValueError("rrf_k must be positive")

        if weights is None:

            parsed_weights = (1.0,) * len(self.backends)

        else:

            parsed_weights = tuple(float(item) for item in weights)

            if len(parsed_weights) != len(self.backends):

                raise ValueError("RRF weights must match backend count")

            if any(item <= 0 for item in parsed_weights):

                raise ValueError("RRF weights must be positive")

        self.weights = parsed_weights

        self.rrf_k = float(rrf_k)



    def discover(

        self,

        request: RoutingRequest,

        descriptors: Sequence[CapabilityDescriptor],

    ) -> tuple[DiscoveryCandidate, ...]:

        fused: dict[str, dict[str, object]] = {}

        for backend, weight in zip(self.backends, self.weights):

            candidates = backend.discover(request, descriptors)

            for candidate in candidates:

                slot = fused.setdefault(

                    candidate.capability_id,

                    {

                        "descriptor": candidate.descriptor,

                        "score": 0.0,

                        "matched_terms": set(),

                        "query_token_count": candidate.query_token_count,

                        "query_coverage": 0.0,

                    },

                )

                slot["score"] = float(slot["score"]) + (

                    weight / (self.rrf_k + candidate.discovery_rank)

                )

                matched = slot["matched_terms"]

                assert isinstance(matched, set)

                matched.update(candidate.matched_terms)

                slot["query_coverage"] = max(

                    float(slot["query_coverage"]),

                    candidate.query_coverage,

                )



        ranked = sorted(

            fused.items(),

            key=lambda item: (-float(item[1]["score"]), item[0]),

        )[: request.max_candidates]

        return tuple(

            DiscoveryCandidate(

                descriptor=item[1]["descriptor"],  # type: ignore[arg-type]

                score=float(item[1]["score"]),

                matched_terms=tuple(sorted(item[1]["matched_terms"])),  # type: ignore[arg-type]

                query_token_count=int(item[1]["query_token_count"]),

                query_coverage=float(item[1]["query_coverage"]),

                discovery_rank=index + 1,

            )

            for index, item in enumerate(ranked)

        )





class CascadedDiscoveryBackend:

    """按成本顺序执行 discovery backend，首个非空阶段即停止。



    典型组合为 exact trigger -> lexical/BM25 -> embedding -> provider-native/LLM matcher。

    V9 核心只定义 cascade contract；后续更强语义阶段可插件化替换，不改变 Control Plane

    或 residency lifecycle。

    """



    def __init__(self, backends: Sequence[CapabilityDiscoveryBackend]) -> None:

        self.backends = tuple(backends)

        if not self.backends:

            raise ValueError("cascaded discovery requires at least one backend")



    def discover(

        self,

        request: RoutingRequest,

        descriptors: Sequence[CapabilityDescriptor],

    ) -> tuple[DiscoveryCandidate, ...]:

        for backend in self.backends:

            candidates = backend.discover(request, descriptors)

            if candidates:

                return candidates

        return ()





@dataclass(frozen=True)

class AdmissibilityDecision:

    """Control Plane 的硬约束结果；不负责语义 relevance。"""



    capability_id: str

    admissible: bool

    reason_codes: tuple[str, ...] = ()

    missing_authorities: tuple[str, ...] = ()

    missing_host_features: tuple[str, ...] = ()

    forbidden_host_features_present: tuple[str, ...] = ()





class CapabilityAdmissibilityRule(Protocol):

    """额外硬约束规则；返回空 reason_codes 表示不阻断。"""



    def evaluate(

        self,

        *,

        request: RoutingRequest,

        descriptor: CapabilityDescriptor,

    ) -> tuple[str, ...]: ...





class DeploymentQualificationRule:

    """把 V7/V8 generation-specific qualification 作为一个可选 control rule。



    它只约束已有 qualification contract 的 capability；没有 contract 的 capability

    由其它 policy 决定，避免把 V8 的 hard gate 再次升级成 universal router。

    """



    def __init__(

        self,

        *,

        qualifier: CapabilityDeploymentQualifier,

        generation_getter: Callable[[CapabilityDescriptor], int | None] | None = None,

        scope_getter: Callable[[CapabilityDescriptor], str | None] | None = None,

    ) -> None:

        self.qualifier = qualifier

        self.generation_getter = generation_getter or (lambda item: item.learned_generation)

        self.scope_getter = scope_getter or (lambda item: item.deployment_scope_id)



    def evaluate(

        self,

        *,

        request: RoutingRequest,

        descriptor: CapabilityDescriptor,

    ) -> tuple[str, ...]:

        contracts = self.qualifier.contracts_for(descriptor.capability_id)

        if not contracts:

            return ()

        generation = self.generation_getter(descriptor)

        if generation is None:

            return ("deployment_generation_missing",)

        if request.deployment_profile is None:

            return ("deployment_profile_missing",)

        decision = self.qualifier.evaluate(

            plugin_id=descriptor.capability_id,

            generation=int(generation),

            profile=request.deployment_profile,

            scope_id=self.scope_getter(descriptor),

            plugin_version=descriptor.version,

        )

        if decision.qualified:

            return ()

        reasons = [f"deployment_status:{decision.status.value}"]

        reasons.extend(f"deployment:{item}" for item in decision.reason_codes)

        return tuple(reasons)





class CapabilityControlPlane:

    """执行 semantic-independent hard constraints。"""



    def __init__(self, rules: Iterable[CapabilityAdmissibilityRule] = ()) -> None:

        self.rules = tuple(rules)



    def evaluate(

        self,

        *,

        request: RoutingRequest,

        descriptor: CapabilityDescriptor,

    ) -> AdmissibilityDecision:

        reasons: list[str] = []

        missing_authorities = sorted(descriptor.required_authorities - request.authorities)

        missing_host_features = sorted(

            descriptor.required_host_features - request.host_features

        )

        forbidden_present = sorted(

            descriptor.forbidden_host_features & request.host_features

        )



        if request.allowed_kinds is not None and descriptor.kind not in request.allowed_kinds:

            reasons.append("capability_kind_not_allowed")

        if missing_authorities:

            reasons.append("required_authority_missing")

        if missing_host_features:

            reasons.append("required_host_feature_missing")

        if forbidden_present:

            reasons.append("forbidden_host_feature_present")



        for rule in self.rules:

            reasons.extend(rule.evaluate(request=request, descriptor=descriptor))



        reason_codes = tuple(dict.fromkeys(reasons))

        return AdmissibilityDecision(

            capability_id=descriptor.capability_id,

            admissible=not reason_codes,

            reason_codes=reason_codes,

            missing_authorities=tuple(missing_authorities),

            missing_host_features=tuple(missing_host_features),

            forbidden_host_features_present=tuple(forbidden_present),

        )





@dataclass(frozen=True)

class RoutedCandidate:

    """Discovery score 与 Control Plane 结果的组合视图。"""



    discovery: DiscoveryCandidate

    admissibility: AdmissibilityDecision



    @property

    def capability_id(self) -> str:

        return self.discovery.capability_id



    @property

    def descriptor(self) -> CapabilityDescriptor:

        return self.discovery.descriptor





class SemanticCapabilitySelector(Protocol):

    """只在 admissible shortlist 中决定 SHOULD-use ordering。"""



    def order(

        self,

        request: RoutingRequest,

        candidates: Sequence[RoutedCandidate],

    ) -> tuple[str, ...]: ...





class EmbeddingSemanticSelector:

    """在 Control Plane 通过后的 shortlist 中用 embedding 决定 SHOULD-use 顺序。



    该层与 embedding discovery 的职责不同：discovery 负责召回候选，selector 负责在

    admissible aperture 内解释任务主意图。trigger/metadata hit 因此只能帮助召回，不能

    直接压过更强的语义证据。encoder 仍是可插拔接口，可接本地 bge-m3 或其它模型。

    """



    def __init__(

        self,

        *,

        encoder: Callable[[Sequence[str]], Sequence[Sequence[float]]],

        minimum_similarity: float = -1.0,

        minimum_top_margin: float = 0.0,

    ) -> None:

        if not -1.0 <= minimum_similarity <= 1.0:

            raise ValueError("minimum_similarity must be between -1 and 1")

        if minimum_top_margin < 0.0:

            raise ValueError("minimum_top_margin must be non-negative")

        self.encoder = encoder

        self.minimum_similarity = float(minimum_similarity)

        self.minimum_top_margin = float(minimum_top_margin)



    def order(

        self,

        request: RoutingRequest,

        candidates: Sequence[RoutedCandidate],

    ) -> tuple[str, ...]:

        if not candidates:

            return ()

        texts = [

            request.task_text,

            *(item.descriptor.searchable_text() for item in candidates),

        ]

        encoded = tuple(

            tuple(float(value) for value in vector)

            for vector in self.encoder(texts)

        )

        if len(encoded) != len(texts):

            raise ValueError("semantic selector encoder returned an unexpected vector count")

        query_vector = encoded[0]

        scored: list[tuple[float, int, str]] = []

        for index, (candidate, vector) in enumerate(zip(candidates, encoded[1:])):

            similarity = EmbeddingDiscoveryBackend._cosine(query_vector, vector)

            if similarity >= self.minimum_similarity:

                scored.append((similarity, index, candidate.capability_id))

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))

        if not scored:

            return ()

        if (

            self.minimum_top_margin > 0.0

            and len(scored) > 1

            and scored[0][0] - scored[1][0] < self.minimum_top_margin

        ):

            return ()

        return tuple(item[2] for item in scored)





class DiscoveryRankSelector:

    """默认 development selector：保持 discovery rank，不引入 provider 依赖。"""



    def order(

        self,

        request: RoutingRequest,

        candidates: Sequence[RoutedCandidate],

    ) -> tuple[str, ...]:

        del request

        return tuple(item.capability_id for item in candidates)





class RecoveryAction(str, Enum):

    NONE = "none"

    REQUEST_CONTEXT = "request_context"

    FALLBACK = "fallback"

    PROPOSE_CAPABILITY = "propose_capability"

    ABSTAIN = "abstain"





class SemanticCapabilityComposer(Protocol):

    """从 admissible shortlist 中选择需要同时存在的 capability working set。"""



    def select(

        self,

        request: RoutingRequest,

        candidates: Sequence[RoutedCandidate],

        *,

        max_capabilities: int,

    ) -> tuple[str, ...]: ...





class GreedyQueryCoverageComposer:

    """零 provider development composer：按新增 query-term coverage 贪心选能力。



    该实现不是论文 novelty；它只提供一个可审计的默认 composition policy。正式系统可

    替换为 LLM planner、learned composer 或任务图规划器，而 governor 的 joint admission

    语义保持不变。

    """



    def select(

        self,

        request: RoutingRequest,

        candidates: Sequence[RoutedCandidate],

        *,

        max_capabilities: int,

    ) -> tuple[str, ...]:

        del request

        if max_capabilities <= 0:

            raise ValueError("max_capabilities must be positive")

        remaining = list(candidates)

        selected: list[str] = []

        covered: set[str] = set()

        while remaining and len(selected) < max_capabilities:

            ranked = sorted(

                remaining,

                key=lambda item: (

                    -len(set(item.discovery.matched_terms) - covered),

                    -item.discovery.score,

                    item.discovery.discovery_rank,

                    item.capability_id,

                ),

            )

            best = ranked[0]

            gain = set(best.discovery.matched_terms) - covered

            if not gain:

                break

            selected.append(best.capability_id)

            covered.update(best.discovery.matched_terms)

            remaining = [item for item in remaining if item.capability_id != best.capability_id]

        return tuple(selected)





@dataclass(frozen=True)

class RoutingPlan:

    """不物化 backing state 的完整 routing trace。"""



    logical_capability_count: int

    discovered: tuple[RoutedCandidate, ...]

    admissible: tuple[RoutedCandidate, ...]

    model_visible_capability_ids: tuple[str, ...]

    execution_order: tuple[str, ...]

    selected_capability_id: str | None

    recovery_action: RecoveryAction

    reason_codes: tuple[str, ...] = ()



    @property

    def candidate_aperture(self) -> int:

        return len(self.model_visible_capability_ids)





@dataclass(frozen=True)

class CapabilityCompositionPlan:

    """一个需要同时驻留的 capability composition，不等同于 fallback 顺序。"""



    routing_plan: RoutingPlan

    selected_capability_ids: tuple[str, ...]

    covered_query_terms: tuple[str, ...]

    recovery_action: RecoveryAction = RecoveryAction.NONE



    @property

    def composed(self) -> bool:

        return bool(self.selected_capability_ids)





class EvolvableCapabilityRouter:

    """metadata discovery -> hard control -> semantic selection 的组合 router。"""



    def __init__(

        self,

        *,

        catalog: CapabilityCatalog,

        discovery: CapabilityDiscoveryBackend | None = None,

        control: CapabilityControlPlane | None = None,

        selector: SemanticCapabilitySelector | None = None,

        composer: SemanticCapabilityComposer | None = None,

    ) -> None:

        self.catalog = catalog

        self.discovery = discovery or CascadedDiscoveryBackend(

            (ExactTriggerDiscoveryBackend(), MetadataDiscoveryBackend())

        )

        self.control = control or CapabilityControlPlane()

        self.selector = selector or DiscoveryRankSelector()

        self.composer = composer or GreedyQueryCoverageComposer()



    def descriptors(self) -> tuple[CapabilityDescriptor, ...]:

        return tuple(CapabilityDescriptor.from_capability(item) for item in self.catalog.all())



    def plan(self, request: RoutingRequest) -> RoutingPlan:

        descriptors = self.descriptors()

        discoveries = self.discovery.discover(request, descriptors)

        routed = tuple(

            RoutedCandidate(

                discovery=item,

                admissibility=self.control.evaluate(

                    request=request,

                    descriptor=item.descriptor,

                ),

            )

            for item in discoveries

        )

        admissible = tuple(item for item in routed if item.admissibility.admissible)



        if not routed:

            action = (

                RecoveryAction.PROPOSE_CAPABILITY

                if request.allow_capability_proposal

                else RecoveryAction.ABSTAIN

            )

            return RoutingPlan(

                logical_capability_count=len(descriptors),

                discovered=(),

                admissible=(),

                model_visible_capability_ids=(),

                execution_order=(),

                selected_capability_id=None,

                recovery_action=action,

                reason_codes=("no_semantic_candidate",),

            )



        if not admissible:

            all_reasons = tuple(

                dict.fromkeys(

                    reason

                    for item in routed

                    for reason in item.admissibility.reason_codes

                )

            )

            requestable = any(

                reason in {"deployment_profile_missing", "required_host_feature_missing"}

                for reason in all_reasons

            )

            action = RecoveryAction.REQUEST_CONTEXT if requestable else RecoveryAction.ABSTAIN

            return RoutingPlan(

                logical_capability_count=len(descriptors),

                discovered=routed,

                admissible=(),

                model_visible_capability_ids=(),

                execution_order=(),

                selected_capability_id=None,

                recovery_action=action,

                reason_codes=all_reasons or ("no_admissible_candidate",),

            )



        visible_ids = tuple(item.capability_id for item in admissible)

        ordered = self.selector.order(request, admissible)

        allowed_ids = set(visible_ids)

        if len(ordered) != len(set(ordered)):

            raise ValueError("semantic selector returned duplicate capability ids")

        unknown = [item for item in ordered if item not in allowed_ids]

        if unknown:

            raise ValueError(

                "semantic selector returned non-admissible capability ids: "

                + ",".join(unknown)

            )

        if not ordered:

            return RoutingPlan(

                logical_capability_count=len(descriptors),

                discovered=routed,

                admissible=admissible,

                model_visible_capability_ids=visible_ids,

                execution_order=(),

                selected_capability_id=None,

                recovery_action=RecoveryAction.ABSTAIN,

                reason_codes=("semantic_selector_abstained",),

            )



        return RoutingPlan(

            logical_capability_count=len(descriptors),

            discovered=routed,

            admissible=admissible,

            model_visible_capability_ids=visible_ids,

            execution_order=tuple(ordered),

            selected_capability_id=ordered[0],

            recovery_action=RecoveryAction.NONE,

        )



    def compose(

        self,

        request: RoutingRequest,

        *,

        max_capabilities: int = 3,

    ) -> CapabilityCompositionPlan:

        """选择需要共同驻留的 capability set；不触发任何 materialization。"""



        if max_capabilities <= 0:

            raise ValueError("max_capabilities must be positive")

        routing_plan = self.plan(request)

        if not routing_plan.admissible:

            return CapabilityCompositionPlan(

                routing_plan=routing_plan,

                selected_capability_ids=(),

                covered_query_terms=(),

                recovery_action=routing_plan.recovery_action,

            )

        selected = self.composer.select(

            request,

            routing_plan.admissible,

            max_capabilities=max_capabilities,

        )

        admissible_ids = {item.capability_id for item in routing_plan.admissible}

        if len(selected) != len(set(selected)):

            raise ValueError("capability composer returned duplicate capability ids")

        unknown = [item for item in selected if item not in admissible_ids]

        if unknown:

            raise ValueError(

                "capability composer returned non-admissible capability ids: "

                + ",".join(unknown)

            )

        if len(selected) > max_capabilities:

            raise ValueError("capability composer exceeded max_capabilities")

        candidate_by_id = {

            item.capability_id: item for item in routing_plan.admissible

        }

        covered = tuple(

            sorted(

                {

                    term

                    for capability_id in selected

                    for term in candidate_by_id[capability_id].discovery.matched_terms

                }

            )

        )

        return CapabilityCompositionPlan(

            routing_plan=routing_plan,

            selected_capability_ids=tuple(selected),

            covered_query_terms=covered,

            recovery_action=(

                RecoveryAction.NONE if selected else RecoveryAction.ABSTAIN

            ),

        )





@dataclass(frozen=True)

class RuntimeAdmissionFailure:

    capability_id: str

    reason: str

    violated_dimensions: tuple[str, ...] = ()





@dataclass(frozen=True)

class RoutedCapabilityLease:

    """一次 route + residency admission 的最终结果。"""



    plan: RoutingPlan

    capability: VirtualCapability | None

    materialized_capability_id: str | None

    admission_failures: tuple[RuntimeAdmissionFailure, ...] = ()

    recovery_action: RecoveryAction = RecoveryAction.NONE



    @property

    def acquired(self) -> bool:

        return self.capability is not None





@dataclass(frozen=True)

class RoutedCapabilityCompositionLease:

    """joint route + residency admission 的同时驻留 lease。"""



    plan: CapabilityCompositionPlan

    capabilities: tuple[VirtualCapability, ...]



    @property

    def acquired(self) -> bool:

        return bool(self.capabilities)





class RoutedCapabilityRuntime:

    """把 routing plan 接到已有 residency governor。



    semantic selection 只给出优先级；最终资源 admission 仍由 governor 执行。若最高优先

    候选无法在 hard budget 下物化，可尝试 selector 已明确排序的后备候选。

    """



    def __init__(

        self,

        *,

        router: EvolvableCapabilityRouter,

        governor: CapabilityResidencyGovernor,

        reconcile_measured: bool = True,

    ) -> None:

        if router.catalog is not governor.catalog:

            raise ValueError("router and governor must share the same capability catalog")

        self.router = router

        self.governor = governor

        self.reconcile_measured = bool(reconcile_measured)



    @contextmanager

    def acquire(

        self,

        request: RoutingRequest,

        *,

        request_id: str = "",

    ) -> Iterator[RoutedCapabilityLease]:

        plan = self.router.plan(request)

        if not plan.execution_order:

            yield RoutedCapabilityLease(

                plan=plan,

                capability=None,

                materialized_capability_id=None,

                recovery_action=plan.recovery_action,

            )

            return



        descriptor_by_id = {

            item.capability_id: item.descriptor for item in plan.admissible

        }

        failures: list[RuntimeAdmissionFailure] = []

        for capability_id in plan.execution_order:

            descriptor = descriptor_by_id[capability_id]

            try:

                with self.governor.acquire_many(

                    ((capability_id, descriptor.required_level),),

                    request_id=request_id,

                    reconcile_before_active=self.reconcile_measured,

                ) as capabilities:

                    capability = capabilities[0]

                    yield RoutedCapabilityLease(

                        plan=plan,

                        capability=capability,

                        materialized_capability_id=capability_id,

                        admission_failures=tuple(failures),

                        recovery_action=(

                            RecoveryAction.FALLBACK if failures else RecoveryAction.NONE

                        ),

                    )

                    return

            except MeasuredResidencyBudgetExceeded as exc:

                failures.append(

                    RuntimeAdmissionFailure(

                        capability_id=capability_id,

                        reason="measured_residency_budget_exceeded",

                        violated_dimensions=exc.violated_dimensions,

                    )

                )

                # PREPARE failed before ACTIVE commit. Roll the candidate back to its

                # pinned floor, then try the next semantically ordered candidate.

                state = self.governor.state(capability_id)

                capability = self.governor.catalog.get(capability_id)

                if state.current_level > capability.pinned_min_level:

                    self.governor.demote(capability_id, capability.pinned_min_level)

                continue

            except ResidencyBudgetExceeded as exc:

                failures.append(

                    RuntimeAdmissionFailure(

                        capability_id=capability_id,

                        reason="residency_budget_exceeded",

                        violated_dimensions=exc.violated_dimensions,

                    )

                )

                state = self.governor.state(capability_id)

                capability = self.governor.catalog.get(capability_id)

                if state.current_level > capability.pinned_min_level:

                    self.governor.demote(capability_id, capability.pinned_min_level)

                continue



        yield RoutedCapabilityLease(

            plan=plan,

            capability=None,

            materialized_capability_id=None,

            admission_failures=tuple(failures),

            recovery_action=RecoveryAction.ABSTAIN,

        )



    @contextmanager

    def acquire_composition(

        self,

        request: RoutingRequest,

        *,

        max_capabilities: int = 3,

        request_id: str = "",

    ) -> Iterator[RoutedCapabilityCompositionLease]:

        """同时 materialize 并 pin 住一个动态选出的 capability composition。



        joint working-set admission 由 governor.acquire_many() 统一处理，因此不会为了装入

        composition 后半部分而淘汰前半部分。若整个 working set 超过 hard budget，沿用

        governor 的显式 ResidencyBudgetExceeded，而不是悄悄退化成不完整组合。

        """



        plan = self.router.compose(request, max_capabilities=max_capabilities)

        if not plan.selected_capability_ids:

            yield RoutedCapabilityCompositionLease(plan=plan, capabilities=())

            return

        descriptor_by_id = {

            item.capability_id: item.descriptor

            for item in plan.routing_plan.admissible

        }

        requests = tuple(

            (capability_id, descriptor_by_id[capability_id].required_level)

            for capability_id in plan.selected_capability_ids

        )

        with self.governor.acquire_many(

            requests,

            request_id=request_id,

            reconcile_before_active=self.reconcile_measured,

        ) as capabilities:

            yield RoutedCapabilityCompositionLease(

                plan=plan,

                capabilities=capabilities,

            )

