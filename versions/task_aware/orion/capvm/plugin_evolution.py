"""可演化 capability plugin 的验证式更新控制器。

V1 关注一个最小原则：插件经验可以更新，但更新必须在插件自己的验证集上通过后
才允许保留。实现故意采用“先写候选 -> 真实调用验证 -> 失败则回滚”的方式，
从而验证持久 learned state 本身也受生命周期治理，而不是无条件追加。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .evolvable_plugin import CapabilityLearnedState, EvolvableLangGraphPlugin


@dataclass(frozen=True)
class PluginValidationCase:
    """一个插件局部的确定性验证案例。"""

    signal: str
    expected_action: str


@dataclass(frozen=True)
class PluginEvolutionResult:
    """一次候选 learned-state 更新的结果。"""

    plugin_id: str
    signal: str
    candidate_action: str
    promoted: bool
    rolled_back: bool
    baseline_correct: int
    candidate_correct: int
    case_count: int
    generation_before: int
    generation_after: int

    @property
    def baseline_accuracy(self) -> float:
        if self.case_count == 0:
            return 0.0
        return self.baseline_correct / self.case_count

    @property
    def candidate_accuracy(self) -> float:
        if self.case_count == 0:
            return 0.0
        return self.candidate_correct / self.case_count


class ValidatedPluginEvolution:
    """对 plugin-private learned state 执行候选更新、验证和回滚。"""

    def __init__(self, plugin: EvolvableLangGraphPlugin) -> None:
        self.plugin = plugin

    def propose_rule(
        self,
        *,
        signal: str,
        action: str,
        validation_cases: Iterable[PluginValidationCase],
        require_non_degradation: bool = True,
        require_target_correct: bool = True,
    ) -> PluginEvolutionResult:
        if not self.plugin.is_mounted:
            raise RuntimeError("plugin must be mounted before evolution")
        signal_key = signal.strip()
        action_value = action.strip()
        if not signal_key or not action_value:
            raise ValueError("signal and action are required")
        cases = tuple(validation_cases)
        if not cases:
            raise ValueError("at least one validation case is required")

        previous = self.plugin.learned_state
        baseline_correct = self._score(cases)

        candidate_rules = dict(previous.rules)
        candidate_rules[signal_key] = action_value
        candidate = CapabilityLearnedState(
            plugin_id=self.plugin.state_store.plugin_id,
            generation=previous.generation + 1,
            rules=candidate_rules,
        )

        # 刻意先将候选写入持久层，再通过真实 plugin invocation 验证；失败时恢复
        # previous snapshot。这样测试覆盖的是“持久状态回滚”，而非只验证内存副本。
        self.plugin.state_store.save(candidate)
        rolled_back = False
        try:
            candidate_correct = self._score(cases)
            target_correct = True
            if require_target_correct:
                matching = [case for case in cases if case.signal == signal_key]
                target_correct = bool(matching) and all(
                    self.plugin.invoke(case.signal)["action"] == case.expected_action
                    for case in matching
                )
            non_degrading = (
                candidate_correct >= baseline_correct
                if require_non_degradation
                else True
            )
            promoted = bool(non_degrading and target_correct)
            if not promoted:
                self.plugin.state_store.save(previous)
                rolled_back = True
        except Exception:
            self.plugin.state_store.save(previous)
            raise

        final_state = self.plugin.learned_state
        return PluginEvolutionResult(
            plugin_id=self.plugin.plugin_id,
            signal=signal_key,
            candidate_action=action_value,
            promoted=promoted,
            rolled_back=rolled_back,
            baseline_correct=baseline_correct,
            candidate_correct=candidate_correct,
            case_count=len(cases),
            generation_before=previous.generation,
            generation_after=final_state.generation,
        )

    def _score(self, cases: tuple[PluginValidationCase, ...]) -> int:
        correct = 0
        for case in cases:
            observed = self.plugin.invoke(case.signal)["action"]
            if observed == case.expected_action:
                correct += 1
        return correct
