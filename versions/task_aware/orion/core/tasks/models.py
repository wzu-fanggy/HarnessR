"""任务系统的数据模型。

对应 Learn Claude Code s12 的任务板，但比教学版多一个 ``escalated`` 状态：
配方筛选里"证据不足 / 适用域外 / 约束冲突"必须停下来问人，机器人无权自行
放宽用户给的约束。``blocked`` 不单独存，由 ``blocked_by`` 实时计算得出。

字段设计参考 Claude Code 的 TaskRecord（9 字段），这里保留对科学筛选有用的
那几个，去掉 ``activeForm``（前端展示用）和 ``blocks``（可反查，不必存）。
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """任务生命周期。ESCLATED 是"需要人判断"，不是失败。"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ESCALATED = "escalated"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_task_id() -> str:
    """任务 ID：时间戳 + 随机串。

    任务板改用 ``.highwatermark`` 递增 ID（见 store），这个函数只作为兜底。
    """
    return f"task_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:4]}"


INTENT_HISTORY_LIMIT = 16


def _validate_execution_intent(intent: Any) -> None:
    """当前声明与历史快照使用相同的字段校验。"""
    if not isinstance(intent, dict):
        raise ValueError('execution_intent必须是对象')
    if not intent:
        return
    if set(intent) != {'source', 'mode', 'note', 'planned_tools'} or intent['source'] != 'agent_plan':
        raise ValueError('执行意图只能是Agent计划软提示')
    if not isinstance(intent['note'], str) or len(intent['note']) > 4000:
        raise ValueError('执行意图文字无效或过长')
    tools = intent['planned_tools']
    if (not isinstance(tools, list) or len(tools) > 32
        or any(not isinstance(t, str) or not t or len(t) > 128 for t in tools)):
        raise ValueError('执行意图工具列表无效')
    if intent['mode'] not in {'explicit_tools', 'semantic_note'}:
        raise ValueError('执行意图模式无效')
    if intent['mode'] == 'semantic_note' and (not intent['note'].strip() or tools):
        raise ValueError('语义意图需要文字且不能夹带工具列表')


@dataclass
class Task:
    """一条可恢复的任务卡。

    预算字段（budget/used）刻意保持成字典而不是固定结构：配方筛选的预算维度
    可能随实验设计变化（工具调用数、token、墙钟时间、候选评估数），写死反而
    不好扩展。
    """

    id: str
    subject: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    owner: str | None = None
    blocked_by: list[str] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 2
    budget: dict[str, Any] = field(default_factory=dict)
    used: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    # 仅声明本卡尚待执行的工作；不是资源命令，也不作为科学完成证明。
    execution_intent: dict[str, Any] = field(default_factory=dict)
    # 每次失败都留一条结构化记录（kind / error / strategy / at），不是只留最后一次的
    # 自由文本。第 5 章消融要看失败分布：域外和约束冲突不该和工具超时混在一起统计。
    failures: list[dict[str, Any]] = field(default_factory=list)
    escalation: dict[str, Any] | None = None
    result: str | None = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    # 追加字段以兼容旧位置参数；版本0不伪造历史，历史仅保留最近16次变化。
    execution_intent_revision: int = 0
    execution_intent_history: list[dict[str, Any]] = field(default_factory=list)

    # ---------- 校验 ----------

    def validate(self) -> None:
        """完成门槛：检查全绿 + 至少一条证据，否则不允许标 completed。"""
        _validate_execution_intent(self.execution_intent)
        revision, history = self.execution_intent_revision, self.execution_intent_history
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError('执行意图版本无效')
        if not isinstance(history, list) or len(history) > INTENT_HISTORY_LIMIT:
            raise ValueError('执行意图历史无效或过长')
        previous = -1
        for entry in history:
            if not isinstance(entry, dict) or set(entry) != {'revision', 'operation', 'update_kind', 'at', 'intent'}:
                raise ValueError('执行意图历史字段无效')
            number = entry['revision']
            if isinstance(number, bool) or not isinstance(number, int) or number <= previous:
                raise ValueError('执行意图历史版本必须递增')
            if entry['operation'] not in {'create', 'claim', 'set_execution_intent', 'legacy_snapshot'}:
                raise ValueError('执行意图历史操作无效')
            if entry['update_kind'] not in {'note', 'replace_plan', 'invalidate_plan', 'legacy_snapshot'}:
                raise ValueError('执行意图历史变更类型无效')
            if not isinstance(entry['at'], str) or not entry['at']:
                raise ValueError('执行意图历史时间缺失')
            _validate_execution_intent(entry['intent'])
            previous = number
        if history:
            if history[-1]['revision'] != revision or history[-1]['intent'] != self.execution_intent:
                raise ValueError('执行意图与最新历史版本不一致')
        elif revision:
            raise ValueError('执行意图非零版本缺少历史')
        if self.status is not TaskStatus.COMPLETED:
            return
        failed = [c for c in self.checks if not c.get("passed")]
        if failed:
            names = ", ".join(str(c.get("name")) for c in failed)
            raise ValueError(f"任务 {self.id} 有未通过的检查：{names}")
        if not self.evidence_refs:
            raise ValueError(f"任务 {self.id} 缺少 evidence_refs，不能标记完成")

    # ---------- 序列化 ----------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        data = dict(d)
        data["status"] = TaskStatus(data.get("status", "pending"))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)

    # ---------- 展示 ----------

    def one_line(self) -> str:
        bits = [f"{self.id}", f"[{self.status.value}]", self.subject]
        if self.owner:
            bits.append(f"@{self.owner}")
        if self.blocked_by:
            bits.append(f"blockedBy={','.join(self.blocked_by)}")
        if self.escalation:
            bits.append(f"待决：{self.escalation.get('question', '')[:60]}")
        return " ".join(bits)
