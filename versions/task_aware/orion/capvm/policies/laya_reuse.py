"""Laya 辅助的资源回收排序；所有准入与回收边界仍由 governor 决定。"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ..catalog import CapabilityCatalog
from ..model import CapabilityRuntimeState, ResidencyLevel
from .lru_cost import LRUCostEvictionPolicy


class DecisionModel(Protocol):
    def predict(self, state: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]: ...


class LayaDecisionModel:
    """显式启用时才加载一个固定 checkpoint；模型文件由调用方配置。"""

    def __init__(
        self,
        model_id: str,
        *,
        subfolder: str | None = None,
        device: str | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id is required")
        self.model_id = model_id
        self.subfolder = subfolder
        self.device = device
        self._model: Any | None = None

    def preload(self) -> None:
        if self._model is not None:
            return
        import laya

        kwargs: dict[str, Any] = {}
        if self.subfolder:
            kwargs["subfolder"] = self.subfolder
        if self.device is not None:
            kwargs["device"] = self.device
        self._model = laya.load(self.model_id, **kwargs)

    def predict(self, state: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        self.preload()
        assert self._model is not None
        return self._model.predict(dict(state), dict(questions))


class LayaReuseEvictionPolicy(LRUCostEvictionPolicy):
    """仅重排合法的 idle 回收候选；默认 shadow 模式保留原 LRU 决策。"""

    def __init__(
        self,
        model: DecisionModel,
        *,
        active: bool = False,
        history_size: int = 8,
        max_candidates: int = 8,
        min_probability_gap: float = 0.2,
    ) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if not 0 <= min_probability_gap <= 1:
            raise ValueError("min_probability_gap must be in [0, 1]")
        self.model = model
        self.active = active
        self.min_probability_gap = min_probability_gap
        self.max_candidates = max_candidates
        self.history: deque[dict[str, object]] = deque(maxlen=history_size)
        self.last_probabilities: dict[str, float] = {}
        self.last_error: str | None = None

    def observe_tool_batch(
        self, tool_names: Sequence[str], capability_ids: Sequence[str],
        *, catalog: CapabilityCatalog | None = None,
    ) -> None:
        """只记录调用名和静态依赖 ID，不保存参数或用户文本。"""

        seen: set[str] = set()

        def visit(item: str) -> None:
            if item in seen:
                return
            seen.add(item)
            if catalog is not None:
                for dependency in catalog.get(item).dependencies:
                    visit(dependency.capability_id)

        for item in capability_ids:
            visit(item)
        self.history.append({
            "tools": list(tool_names),
            "capabilities": sorted(seen),
        })

    def choose(
        self,
        *,
        states: Mapping[str, CapabilityRuntimeState],
        catalog: CapabilityCatalog,
        violated_dimensions: tuple[str, ...],
        protected_ids: set[str],
        cost_basis: str = "declared",
    ) -> tuple[str, ResidencyLevel] | None:
        baseline = super().choose(
            states=states, catalog=catalog,
            violated_dimensions=violated_dimensions,
            protected_ids=protected_ids, cost_basis=cost_basis,
        )
        self.last_probabilities = {}
        self.last_error = None
        if baseline is None:
            return None

        eligible: list[tuple[int, str, ResidencyLevel]] = []
        for item, runtime in states.items():
            capability = catalog.get(item)
            target = capability.pinned_min_level
            if item in protected_ids or runtime.active_count or runtime.current_level <= target:
                continue
            cost = (capability.adapter.measured_cost if cost_basis == "measured"
                    else capability.adapter.declared_cost)
            released = cost(runtime.current_level) - cost(target)
            if any(released.value(dimension) > 0 for dimension in violated_dimensions):
                eligible.append((runtime.last_used_tick, item, target))
        if len(eligible) < 2 or len(eligible) > self.max_candidates:
            return baseline
        eligible.sort()

        questions = {
            f"reuse_{index}": {
                "type": "noul",
                "instructions": (
                    f"Will the agent need runtime resource {item} again within the next "
                    "two tool-call steps? Answer from the recent tool sequence only."
                ),
            }
            for index, (_tick, item, _target) in enumerate(eligible)
        }
        state: dict[str, object] = {
            "recent_tool_batches": list(self.history),
            "candidate_resources": [item for _tick, item, _target in eligible],
            "candidate_details": [
                {"id": item, "kind": catalog.get(item).kind.value,
                 "summary": catalog.get(item).summary[:120], "last_used_tick": tick}
                for tick, item, _target in eligible
            ],
            "resource_pressure": list(violated_dimensions),
        }
        try:
            result = self.model.predict(state, questions)
            answers = result["answers"]
            if not isinstance(answers, Mapping):
                raise ValueError("Laya answers must be a mapping")
            probabilities: dict[str, float] = {}
            for index, (_tick, item, _target) in enumerate(eligible):
                answer = answers[f"reuse_{index}"]
                if not isinstance(answer, Mapping):
                    raise ValueError("Laya noul answer must be a mapping")
                probability = float(answer["noul"])
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Laya noul probability must be in [0, 1]")
                probabilities[item] = probability
            self.last_probabilities = probabilities
        except Exception as exc:
            self.last_error = type(exc).__name__
            return baseline

        if not self.active:
            return baseline
        selected = min(eligible, key=lambda row: (probabilities[row[1]], row[0], row[1]))
        if probabilities[baseline[0]] - probabilities[selected[1]] < self.min_probability_gap:
            return baseline
        return selected[1], selected[2]
