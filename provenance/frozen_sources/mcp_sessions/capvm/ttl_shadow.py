"""真实 idle TTL 到期点的只读预测探针；不执行保活或卸载。"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Protocol


class ReusePredictor(Protocol):
    def predict(self, state: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class IdleResource:
    """资源管理器在 TTL 到期时提供的实例快照；单位必须是实测值。"""

    instance_id: str
    resource_key: str
    resource_type: str
    idle_since_s: float
    ttl_s: float
    active_count: int
    resident: bool
    reload_cost_s: float | None = None
    host_ram_mib: float | None = None
    gpu_vram_mib: float | None = None


@dataclass(frozen=True)
class DecisionContext:
    """当前决策点可见的任务和资源信息；不包含未来工具调用。"""

    task_type: str
    task_stage: str
    task_intent: str
    queued_requests: int
    host_ram_pressure: float | None = None
    gpu_vram_pressure: float | None = None


class TTLShadowProbe:
    """在宿主计时器触发时预测，并在后续两步工具调用后补标签。"""

    def __init__(
        self,
        predictor: ReusePredictor | None = None,
        *,
        clock: Callable[[], float] = monotonic,
        history_size: int = 8,
        horizon_calls: int = 2,
    ) -> None:
        if history_size <= 0 or horizon_calls <= 0:
            raise ValueError("history_size and horizon_calls must be positive")
        self.predictor = predictor
        self.clock = clock
        self.horizon_calls = horizon_calls
        self.history: deque[dict[str, object]] = deque(maxlen=history_size)
        self._pending: list[dict[str, object]] = []
        self.records: list[dict[str, object]] = []
        self._next_id = 0

    def observe_tool_step(
        self,
        tool_names: Sequence[str],
        dependency_ids: Sequence[str],
        *,
        argument_summaries: Sequence[str] | None = None,
        succeeded: Sequence[bool | None] | None = None,
    ) -> None:
        """仅记录已发生的调用；一次并行 ToolNode batch 算一步。"""

        names = tuple(str(name).strip() for name in tool_names)
        dependencies = tuple(sorted(set(str(item).strip() for item in dependency_ids)))
        if not names or any(not name for name in names) or any(not item for item in dependencies):
            raise ValueError("tool step requires valid tool and dependency identities")
        if argument_summaries is not None and len(argument_summaries) != len(names):
            raise ValueError("argument summaries must align with tool names")
        if succeeded is not None and len(succeeded) != len(names):
            raise ValueError("outcomes must align with tool names")
        summaries = [str(item)[:200] for item in argument_summaries] if argument_summaries else []
        outcomes = list(succeeded) if succeeded is not None else []
        for row in tuple(self._pending):
            row["observed_future_calls"] = int(row["observed_future_calls"]) + 1
            if row["resource_key"] in dependencies:
                row["reused_within_horizon"] = True
            if row["observed_future_calls"] >= self.horizon_calls:
                row["status"] = "complete"
                self._pending.remove(row)
        self.history.append({
            "tools": list(names), "dependencies": list(dependencies),
            "argument_summaries": summaries, "succeeded": outcomes,
        })

    def on_ttl_due(self, resource: IdleResource, context: DecisionContext) -> dict[str, object]:
        """必须由真实 TTL 调度器调用；任何结果均不改变资源池。"""

        now = self.clock()
        self._validate(resource, context, now)
        if any(row["instance_id"] == resource.instance_id for row in self._pending):
            raise ValueError("an unresolved TTL decision already exists for this instance")
        idle_s = now - resource.idle_since_s
        state: dict[str, object] = {
            "trigger": "idle_ttl_due",
            "task": {
                "type": context.task_type,
                "stage": context.task_stage,
                "intent": context.task_intent[:500],
            },
            "recent_tool_batches": list(self.history),
            "candidate": {
                "instance_id": resource.instance_id,
                "resource_key": resource.resource_key,
                "resource_type": resource.resource_type,
                "idle_seconds": round(idle_s, 6),
                "ttl_seconds": resource.ttl_s,
                "active_count": resource.active_count,
                "resident": resource.resident,
                "reload_cost_seconds": resource.reload_cost_s,
                "host_ram_mib": resource.host_ram_mib,
                "gpu_vram_mib": resource.gpu_vram_mib,
            },
            "pool": {
                "queued_requests": context.queued_requests,
                "host_ram_pressure": context.host_ram_pressure,
                "gpu_vram_pressure": context.gpu_vram_pressure,
            },
        }
        question = {
            "reuse_soon": {
                "type": "noul",
                "instructions": (
                    "At this idle TTL expiry, will a tool call need this resource dependency "
                    f"within the next {self.horizon_calls} tool-call steps?"
                ),
            }
        }
        probability: float | None = None
        error: str | None = None
        started = self.clock()
        if self.predictor is not None:
            try:
                answer = self.predictor.predict(state, question)["answers"]["reuse_soon"]
                probability = float(answer["noul"])
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("invalid Laya probability")
            except Exception as exc:
                error = type(exc).__name__
                probability = None
        latency_s = self.clock() - started
        self._next_id += 1
        row: dict[str, object] = {
            "event_id": self._next_id,
            "instance_id": resource.instance_id,
            "resource_key": resource.resource_key,
            "resource_type": resource.resource_type,
            "trigger": "idle_ttl_due",
            "idle_seconds": round(idle_s, 6),
            "ttl_seconds": resource.ttl_s,
            "reload_cost_seconds": resource.reload_cost_s,
            "host_ram_mib": resource.host_ram_mib,
            "gpu_vram_mib": resource.gpu_vram_mib,
            "host_ram_pressure": context.host_ram_pressure,
            "gpu_vram_pressure": context.gpu_vram_pressure,
            "queued_requests": context.queued_requests,
            "task_type": context.task_type,
            "task_stage": context.task_stage,
            "recent_tool_batches": [
                {"tools": batch["tools"], "dependencies": batch["dependencies"],
                 "succeeded": batch["succeeded"]}
                for batch in self.history
            ],
            "laya_probability": probability,
            "prediction_error": error,
            "prediction_latency_seconds": round(latency_s, 6),
            "baseline_action": "evict_at_ttl",
            "actual_action": "unobserved",
            "observed_future_calls": 0,
            "reused_within_horizon": False,
            "status": "pending",
        }
        # 不持久化用户目标原文；预测时临时使用的 task_intent 留在局部 state。
        self.records.append(row)
        self._pending.append(row)
        return row

    def write_jsonl(self, path: Path) -> None:
        """持久化审计记录；pending 事件不得当作负例。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in self.records),
            encoding="utf-8",
        )

    @staticmethod
    def _validate(resource: IdleResource, context: DecisionContext, now: float) -> None:
        if not resource.instance_id or not resource.resource_key or not resource.resource_type:
            raise ValueError("resource identity is required")
        if not resource.resident or resource.active_count != 0:
            raise ValueError("TTL candidate must be resident and idle")
        if not math.isfinite(now) or not math.isfinite(resource.idle_since_s):
            raise ValueError("finite monotonic times are required")
        if not math.isfinite(resource.ttl_s) or resource.ttl_s <= 0:
            raise ValueError("positive finite TTL is required")
        if now < resource.idle_since_s + resource.ttl_s:
            raise ValueError("TTL has not expired")
        if context.queued_requests < 0:
            raise ValueError("queued_requests cannot be negative")
        if not context.task_type.strip() or not context.task_stage.strip() or not context.task_intent.strip():
            raise ValueError("task type, stage and intent are required at TTL decision time")
        for value in (resource.reload_cost_s, resource.host_ram_mib, resource.gpu_vram_mib):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("resource measurements must be finite and nonnegative")
        for value in (context.host_ram_pressure, context.gpu_vram_pressure):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("pressure must be a fraction in [0, 1]")
