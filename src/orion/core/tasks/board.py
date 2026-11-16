"""任务板的业务逻辑：创建、依赖检查、认领、完成、升级。



状态机（四态）::



    pending ──claim──▶ in_progress ──complete──▶ completed

                            │

                            └──escalate──▶ escalated（带一个具体待决问题）



``blocked`` 不是状态，是 ``pending`` 任务的一种临时处境，由 ``can_start``

实时算出：``blocked_by`` 里有任何一个没 completed 就开不了工。



三条纪律（比教学版 s12 多出来的部分）：



1. ``complete`` 必须过 ``Task.validate()``：检查全绿 + 至少一条 evidence_refs；

2. ``record_failure`` 累计 attempts，超过 max_attempts 自动转 escalated，

   不允许"失败就标完成"或者静默跳过；

3. ``escalate`` 必须带一个**能被人直接回答的问题**，不是一句"出错了"。

"""



from __future__ import annotations



from typing import Iterable

from copy import deepcopy



from .failures import FailureKind, classify, render_question, summarize_failures

from .models import INTENT_HISTORY_LIMIT, Task, TaskStatus, now_iso

from .store import TaskStore





def _set_intent(task: Task, note: str | None, planned_tools: list[str] | None,

                *, operation: str, invalidate_plan: bool = False) -> None:

    """分开处理备注、计划替换和明确作废；只修改尚未保存的对象。"""

    if note is not None and not isinstance(note, str):

        raise ValueError('工作备注必须是文字')

    if not isinstance(invalidate_plan, bool):

        raise ValueError('invalidate_plan必须为布尔值')

    if planned_tools is not None and not isinstance(planned_tools, list):

        raise ValueError('planned_tools必须是工具名称列表')

    if invalidate_plan and planned_tools is not None:

        raise ValueError('不能同时替换和作废工具计划')

    task.validate()

    old = deepcopy(task.execution_intent)

    if invalidate_plan:

        if old.get('mode') != 'explicit_tools':

            raise ValueError('当前没有可作废的明确工具计划')

        intent = {'source': 'agent_plan', 'mode': 'semantic_note',

            'note': note if note and note.strip() else '原工具计划已明确作废；剩余计划尚未确定。',

            'planned_tools': []}

        kind = 'invalidate_plan'

    elif planned_tools is not None:

        intent = {'source': 'agent_plan', 'mode': 'explicit_tools',

            'note': note if note is not None else old.get('note', ''),

            'planned_tools': list(planned_tools)}

        kind = 'replace_plan'

    elif note is not None:

        intent = {**old, 'note': note} if old else {

            'source': 'agent_plan', 'mode': 'semantic_note', 'note': note, 'planned_tools': []}

        kind = 'note'

    else:

        task.validate()

        return

    if intent == old:

        task.validate()

        return

    history = deepcopy(task.execution_intent_history)

    if old and not history:

        # 仅保存当前读到的旧记录；不是重建此前已丢失的操作历史。

        history.append({'revision': task.execution_intent_revision,

            'operation': 'legacy_snapshot', 'update_kind': 'legacy_snapshot',

            'at': task.updated_at, 'intent': old})

    task.execution_intent_revision += 1

    history.append({'revision': task.execution_intent_revision, 'operation': operation,

        'update_kind': kind, 'at': now_iso(), 'intent': deepcopy(intent)})

    task.execution_intent_history = history[-INTENT_HISTORY_LIMIT:]

    task.execution_intent = intent

    task.validate()





class TaskBoard:

    def __init__(self, store: TaskStore | None = None) -> None:

        self.store = store or TaskStore()



    # ---------- 创建 ----------



    def create(

        self,

        subject: str,

        description: str = "",

        blocked_by: Iterable[str] | None = None,

        budget: dict | None = None,

        max_attempts: int = 2,

        *,

        execution_note: str | None = None,

        planned_tools: list[str] | None = None,

    ) -> Task:

        deps = [d for d in (blocked_by or []) if d]

        with self.store.lock():

            task = Task(

                id=self.store.next_id(),

                subject=subject,

                description=description,

                blocked_by=deps,

                budget=dict(budget or {}),

                max_attempts=max_attempts,

            )

            if execution_note is not None or planned_tools is not None:

                _set_intent(task, execution_note, planned_tools, operation='create')

            self.store.save(task)

        return task



    # ---------- 查询 ----------



    def set_execution_intent(self, task_id: str, note: str | None = None,

                             planned_tools: list[str] | None = None, *,

                             invalidate_plan: bool = False) -> Task:

        """备注单独更新；完整清单显式替换，失效须明确声明。"""

        with self.store.lock():

            task = self.store.load(task_id)

            if task.status is TaskStatus.COMPLETED:

                raise ValueError('已完成任务不能更新执行意图')

            _set_intent(task, note, planned_tools, operation='set_execution_intent',

                        invalidate_plan=invalidate_plan)

            task.updated_at = now_iso()

            self.store.save(task)

        return task



    def get(self, task_id: str) -> Task:

        return self.store.load(task_id)



    def list(self, status: TaskStatus | str | None = None) -> list[Task]:

        tasks = self.store.list_all()

        if status is None:

            return tasks

        want = TaskStatus(status) if isinstance(status, str) else status

        return [t for t in tasks if t.status is want]



    def can_start(self, task_id: str) -> tuple[bool, str]:

        """依赖检查：blocked_by 必须全部 completed；不存在的依赖视为未满足。"""

        task = self.store.load(task_id)

        missing, unfinished = [], []

        for dep_id in task.blocked_by:

            try:

                dep = self.store.load(dep_id)

            except FileNotFoundError:

                missing.append(dep_id)

                continue

            if dep.status is not TaskStatus.COMPLETED:

                unfinished.append(f"{dep_id}({dep.status.value})")

        if missing:

            return False, f"依赖不存在：{', '.join(missing)}"

        if unfinished:

            return False, f"上游未完成：{', '.join(unfinished)}"

        return True, "ready"



    # ---------- 认领 ----------



    def claim(self, task_id: str, owner: str = "agent", *,

              execution_note: str | None = None,

              planned_tools: list[str] | None = None,

              invalidate_plan: bool = False) -> Task:

        """pending → in_progress。



        多 agent 并发下这里需要 per-task 锁 + 读完重验；当前是单 agent，

        靠目录锁已足够。若将来要并行，在锁内重新 load 一次再判状态即可。

        """

        with self.store.lock():

            task = self.store.load(task_id)

            if task.status is TaskStatus.COMPLETED:

                raise ValueError(f"{task_id} 已完成，不能认领")

            if task.status is TaskStatus.ESCALATED:

                raise ValueError(f"{task_id} 已升级待人处理，不能认领")

            if task.status is TaskStatus.IN_PROGRESS and task.owner != owner:

                raise ValueError(f"{task_id} 已被 {task.owner} 认领")

            ok, reason = self.can_start(task_id)

            if not ok:

                raise ValueError(f"{task_id} 暂不可开工：{reason}")

            task.status = TaskStatus.IN_PROGRESS

            task.owner = owner

            _set_intent(task, execution_note, planned_tools, operation='claim',

                        invalidate_plan=invalidate_plan)

            task.updated_at = now_iso()

            self.store.save(task)

        return task



    def release(self, task_id: str, reason: str = "") -> Task:

        """把被认领但没做完的任务放回 pending（对应 CC 的 unassign）。"""

        with self.store.lock():

            task = self.store.load(task_id)

            if task.status is not TaskStatus.IN_PROGRESS:

                return task

            task.status = TaskStatus.PENDING

            task.owner = None

            if reason:

                task.result = f"released: {reason}"

            task.updated_at = now_iso()

            self.store.save(task)

        return task



    # ---------- 完成 ----------



    def complete(

        self,

        task_id: str,

        result: str = "",

        evidence_refs: Iterable[str] | None = None,

        checks: list[dict] | None = None,

    ) -> tuple[Task, list[Task]]:

        """in_progress → completed，并返回刚被解锁的下游任务。"""

        with self.store.lock():

            task = self.store.load(task_id)

            if task.status is TaskStatus.COMPLETED:

                return task, []

            if evidence_refs:

                task.evidence_refs = sorted({*task.evidence_refs, *evidence_refs})

            if checks:

                task.checks = [*task.checks, *checks]

            task.status = TaskStatus.COMPLETED

            task.result = result or task.result

            task.updated_at = now_iso()

            task.validate()

            self.store.save(task)

            unblocked = self._newly_unblocked(exclude=task_id)

        return task, unblocked



    def _newly_unblocked(self, exclude: str) -> list[Task]:

        """只找"依赖里包含刚完成那张卡"的 pending 任务，避免把本来就可开工的算进来。"""

        out = []

        for t in self.store.list_all():

            if t.id == exclude or t.status is not TaskStatus.PENDING:

                continue

            if exclude not in t.blocked_by:

                continue

            ok, _ = self.can_start(t.id)

            if ok:

                out.append(t)

        return out



    # ---------- 失败与升级 ----------



    def record_failure(

        self,

        task_id: str,

        error: str,

        kind: FailureKind | str | None = None,

        hint: str = "",

    ) -> Task:

        """记一次失败，按类别决定是重试还是直接转人工。



        不可重试的类别（域外 / 约束冲突）立即升级：重试不会让域外的配方变回域内，

        也不会让互相冲突的约束变得可满足，那只是烧预算。

        可重试的类别（查不到 / 证据不足 / 工具异常）必须留下"下次怎么改"，

        否则模型会把同一个查询原样再跑一遍。

        """

        policy = classify(error, kind)

        with self.store.lock():

            task = self.store.load(task_id)

            task.attempts += 1

            task.updated_at = now_iso()

            task.failures.append(

                {

                    "attempt": task.attempts,

                    "kind": policy.kind.value,

                    "label": policy.label,

                    "error": error,

                    "retry_strategy": policy.retry_strategy,

                    "at": task.updated_at,

                }

            )

            if not policy.retryable:

                task.status = TaskStatus.ESCALATED

                task.escalation = {

                    "reason": policy.terminal_reason,

                    "kind": policy.kind.value,

                    "attempts": task.attempts,

                    "question": render_question(policy, task.subject, hint or error[:80]),

                    "at": now_iso(),

                }

                task.result = f"attempt {task.attempts} 不可重试（{policy.label}）：{error}"

            elif task.attempts >= task.max_attempts:

                task.status = TaskStatus.ESCALATED

                task.escalation = {

                    "reason": policy.terminal_reason,

                    "kind": policy.kind.value,

                    "attempts": task.attempts,

                    "question": render_question(policy, task.subject, error[:80]),

                    "at": now_iso(),

                }

                task.result = f"attempt {task.attempts} 失败且已达上限（{policy.label}）：{error}"

            else:

                task.status = TaskStatus.PENDING

                task.owner = None

                task.result = (

                    f"attempt {task.attempts} 失败（{policy.label}）：{error}\n"

                    f"下次必须改：{policy.retry_strategy}"

                )

            self.store.save(task)

        return task



    def failure_stats(self, task_id: str | None = None) -> dict[str, int]:

        """失败分类统计。全看板或单卡，用于第 5 章的失败分布分析。"""

        if task_id is not None:

            return summarize_failures(self.store.load(task_id).failures)

        return summarize_failures([f for t in self.store.list_all() for f in t.failures])



    def escalate(self, task_id: str, reason: str, question: str) -> Task:

        """主动升级：必须给一个具体到能直接回答的问题。"""

        if not question.strip():

            raise ValueError("升级必须带一个具体待决问题，不能只写原因")

        with self.store.lock():

            task = self.store.load(task_id)

            task.status = TaskStatus.ESCALATED

            task.escalation = {"reason": reason, "question": question, "at": now_iso()}

            task.updated_at = now_iso()

            self.store.save(task)

        return task



    def resolve_escalation(self, task_id: str, decision: str) -> Task:

        """人给出决定后，把任务放回 pending，并把决定写进描述供下轮读。"""

        with self.store.lock():

            task = self.store.load(task_id)

            if task.status is not TaskStatus.ESCALATED:

                raise ValueError(f"{task_id} 不处于 escalated 状态")

            task.status = TaskStatus.PENDING

            task.owner = None

            task.attempts = 0

            task.description = f"{task.description}\n[人工决定] {decision}".strip()

            task.escalation = None

            task.updated_at = now_iso()

            self.store.save(task)

        return task



    # ---------- 上下文注入 ----------



    def summary(self, limit: int | None = None) -> str:

        """给模型看的一行式看板，跨会话恢复时直接塞进上下文。"""

        tasks = self.store.list_all()

        if not tasks:

            return "任务板为空。"

        by_status: dict[str, list[Task]] = {}

        for t in tasks:

            by_status.setdefault(t.status.value, []).append(t)

        order = ["escalated", "in_progress", "pending", "completed"]

        lines = [f"任务板：共 {len(tasks)} 张"]

        for st in order:

            group = by_status.get(st) or []

            if not group:

                continue

            lines.append(f"\n## {st}（{len(group)}）")

            for t in group[: limit or len(group)]:

                extra = ""

                if st == "pending":

                    ok, reason = self.can_start(t.id)

                    extra = "  ← 可开工" if ok else f"  ← {reason}"

                lines.append(f"- {t.one_line()}{extra}")

                if st in ("pending", "in_progress"):

                    # 恢复时可直接看到正常任务内容，不需为其他组件重复登记计划。

                    if t.description.strip():

                        lines.append(f"  目标与验收：{t.description[:500].replace(chr(10), ' ')}")

                    note = t.execution_intent.get('note', '')

                    if (t.execution_intent.get('mode') == 'semantic_note' and any(

                            e['update_kind'] == 'invalidate_plan' for e in t.execution_intent_history)):

                        lines.append('  原工具计划已作废；剩余计划尚未确认。')

                    if note.strip():

                        lines.append(f"  进展与剩余工作：{note[:500].replace(chr(10), ' ')}")

                # 重试前必须让模型看到上一轮错在哪、这次该改什么，

                # 否则恢复会话后它会原样再跑一遍同样的查询。

                if st in ("pending", "in_progress") and t.failures:

                    last = t.failures[-1]

                    lines.append(

                        f"  ↳ 上次失败：{last.get('label')}（{str(last.get('error'))[:60]}）"

                    )

                    if last.get("retry_strategy"):

                        lines.append(f"    下次必须改：{last.get('retry_strategy')}")

        return "\n".join(lines)

