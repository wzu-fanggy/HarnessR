"""任务系统的 LangChain 工具封装。

模型只能通过这组工具动任务板，不能直接改 JSON 文件——和 Claude Code 把
TaskCreate/TaskUpdate/TaskList 做成工具、而不是让模型写文件，是同一个考虑：
写文件能绕开依赖检查和完成门槛。
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from .board import TaskBoard
from .models import Task, TaskStatus
from orion.core.task_board_intent import notify_board_task

_BOARD: TaskBoard | None = None


def get_board() -> TaskBoard:
    global _BOARD
    if _BOARD is None:
        _BOARD = TaskBoard()
    return _BOARD


def set_board(board: TaskBoard | None) -> None:
    """测试注入用。"""
    global _BOARD
    _BOARD = board


def _fmt(task: Task, extra: str = "") -> str:
    return f"{task.one_line()}{extra}"


@tool
def board_create(
    subject: str,
    description: str = "",
    blocked_by: list[str] | None = None,
    execution_note: str | None = None,
    planned_tools: list[str] | None = None,
) -> str:
    """在任务板上新建一张任务卡。

    用于需要跟踪进展、依赖或跨上下文继续的工作；简单任务可以直接完成，无需建卡。
    被依赖的任务先建，再把它的 id 填进下游任务的 blocked_by；依赖未完成不能认领。
    在description记录目标和验收条件，可同时附上当前进展或剩余工作备注。
    计划尚不明确时允许省略；不要求预判全部未来操作，之后按实际进展修订。

    :param subject: 一句话任务标题，动宾结构，例如"核对数据并整理结果"
    :param description: 这张卡具体要做什么、验收标准是什么
    :param blocked_by: 上游任务 id 列表；没有依赖就留空
    :param execution_note: 可选的当前进展、剩余工作或阻塞条件，供继续任务和交接使用
    :param planned_tools: 可选的当前完整剩余工具计划；不确定时省略，计划变化时修订
    """
    try:
        task = get_board().create(subject, description, blocked_by or [],
            execution_note=execution_note, planned_tools=planned_tools)
        notify_board_task(task.id)
    except Exception as exc:  # noqa: BLE001 - 工具层要把异常变成模型能读的文字
        return f"创建失败：{exc}"
    dep = f"（依赖 {', '.join(task.blocked_by)}）" if task.blocked_by else ""
    return f"已创建 {task.id}：{task.subject}{dep}"


@tool
def board_list(status: str | None = None) -> str:
    """列出任务板。跨会话恢复时先调这个，看清做到哪了。

    :param status: 可选过滤：pending / in_progress / completed / escalated
    """
    try:
        board = get_board()
        tasks = board.list(status) if status else board.list()
    except Exception as exc:  # noqa: BLE001
        return f"读取失败：{exc}"
    if not tasks:
        return "没有符合条件的任务。"

    lines = []
    for t in tasks:
        line = f"- {_fmt(t)}"
        if t.status is TaskStatus.PENDING:
            ok, reason = board.can_start(t.id)
            line += "  ← 可开工" if ok else f"  ← {reason}"
        if t.attempts:
            line += f"  [已试 {t.attempts}/{t.max_attempts}]"
        lines.append(line)
    return "\n".join(lines)


@tool
def board_set_execution_intent(task_id: str, note: str = "",
                               planned_tools: list[str] | None = None) -> str:
    """更新任务的进展和剩余工作，供继续任务、恢复上下文和交接使用。

    记录已完成什么、尚待完成什么、遇到什么阻塞；不确定的后续步骤允许保留未知。
    可在普通建卡或认领操作中一起填写备注，无需为重复记录同一内容单独调用。
    planned_tools仅在当前完整剩余工具计划已确定时填写；空列表表示当前计划暂无工具。
    备注和工具计划均可修订，不代替结果、检查项或完成证据。

    :param task_id: 当前任务卡ID
    :param note: 当前进展、剩余工作或阻塞条件
    :param planned_tools: 可选的完整剩余工具名称列表
    :return: 更新结果与保留的任务状态
    """
    try:
        task = get_board().set_execution_intent(task_id, note, planned_tools)
        notify_board_task(task.id)
    except Exception as exc:
        return f"更新执行意图失败：{exc}"
    return f"已更新 {task.id} 的执行意图（{task.execution_intent['mode']}）；任务仍为 {task.status.value}"


@tool
def board_get(task_id: str) -> str:
    """读取一张任务卡的完整内容（描述、检查、证据、升级问题）。

    :param task_id: 任务 id，例如 task_0003
    """
    try:
        task = get_board().get(task_id)
    except Exception as exc:  # noqa: BLE001
        return f"读取失败：{exc}"
    return json.dumps(task.to_dict(), ensure_ascii=False, indent=2)


@tool
def board_claim(task_id: str, owner: str = "agent", execution_note: str | None = None,
                planned_tools: list[str] | None = None) -> str:
    """认领任务：pending → in_progress。上游没做完会被拒绝。

    :param task_id: 要开工的任务 id
    :param owner: 认领者名字，单 agent 场景保持默认即可
    :param execution_note: 可在认领时更新当前进展或剩余工作，供继续任务和交接使用
    :param planned_tools: 可选完整剩余工具清单；省略时保留已有意图
    """
    try:
        task = get_board().claim(task_id, owner, execution_note=execution_note, planned_tools=planned_tools)
        notify_board_task(task.id)
    except Exception as exc:  # noqa: BLE001
        return f"认领失败：{exc}"
    return f"已认领 {task.id}：{task.subject}"


@tool
def board_complete(
    task_id: str,
    result: str = "",
    evidence_refs: list[str] | None = None,
    checks: list[dict] | None = None,
) -> str:
    """完成任务的唯一出口。必须带证据；有检查项时必须全绿，否则会被拒绝。

    :param task_id: 任务 id
    :param result: 一句话结果摘要（不要在这里写长报告）
    :param evidence_refs: 支撑结论的编号，如测量记录 id、chunk id、论文 id
    :param checks: 检查结果列表，每项形如 {"name": "约束满足", "passed": true}
    """
    try:
        task, unblocked = get_board().complete(
            task_id, result=result,
            evidence_refs=evidence_refs or [],
            checks=checks or [],
        )
    except Exception as exc:  # noqa: BLE001
        return f"完成失败：{exc}"
    msg = f"已完成 {task.id}：{task.subject}"
    if unblocked:
        msg += "\n解锁：" + "、".join(f"{t.id}（{t.subject}）" for t in unblocked)
    return msg


@tool
def board_escalate(task_id: str, reason: str, question: str) -> str:
    """卡住时交给人的唯一出口。必须提出一个能被直接回答的问题。

    什么时候该升级：证据不足以判断、体系落在适用域外、用户约束彼此冲突、
    需要放宽条件才能继续。不要自己放宽约束后当作完成任务。

    :param task_id: 任务 id
    :param reason: 为什么卡住，简短
    :param question: 具体待决问题，例如"该体系只有 2 个数据点，是否放宽到 -15℃？"
    """
    try:
        task = get_board().escalate(task_id, reason, question)
    except Exception as exc:  # noqa: BLE001
        return f"升级失败：{exc}"
    return f"{task.id} 已转人工：{question}"


@tool
def board_fail(task_id: str, error: str, kind: str = "") -> str:
    """记录一次失败。按失败类别决定是重试还是直接转人工。

    域外和约束冲突**不会重试**——多试几次不会让域外的配方变回域内，也不会让
    互相冲突的约束变得可满足，那只是烧预算。这两类会直接转人工并附上具体问题。

    :param task_id: 任务 id
    :param error: 失败的具体表现，例如"放宽到 ±5 ℃ 仍无 NaPF6|G2 的 −20 ℃ 记录"
    :param kind: 失败类别。可以直接把上游工具返回的 status 原样贴进来
                 （out_of_domain / insufficient_composition / not_found / timeout 等，
                 系统会映射）；也可以传枚举值 not_found（查不到）/
                 out_of_domain（适用域外）/ insufficient_input（组成或条件缺失）/
                 constraint_conflict（约束冲突）/ evidence_weak（证据不足或不可分辨）/
                 tool_error（工具超时或崩溃）。不填则按 error 文本推断，
                 推不出来会被要求补填——不允许无分类地反复重试。
    """
    if not error.strip():
        return (
            "请写清失败的具体表现：查了哪个体系哪个条件、哪两条约束冲突、"
            "或哪个工具报的什么错。空白的失败原因无法判断该不该重试。"
        )
    try:
        task = get_board().record_failure(task_id, error, kind.strip() or None)
    except ValueError as exc:  # 类别非法 / 成功状态误报，属于调用方式问题
        return f"没有记录失败：{exc}"
    except Exception as exc:  # noqa: BLE001
        return f"记录失败状态时出错：{exc}"
    if task.status is TaskStatus.ESCALATED:
        esc = task.escalation or {}
        if str(esc.get("reason", "")).endswith("_exhausted") or esc.get("reason") == "unclassified":
            prefix = f"{task.id} 已试 {task.attempts} 次，转人工"
        else:
            prefix = f"{task.id} 属于不可重试的失败，直接转人工"
        return f"{prefix}：{esc.get('question', '')}"
    last = task.failures[-1]
    return (
        f"{task.id} 记一次失败（{task.attempts}/{task.max_attempts}，{last['label']}）。\n"
        f"下次必须改：{last['retry_strategy']}"
    )


TASK_TOOLS = [
    board_create,
    board_set_execution_intent,
    board_list,
    board_get,
    board_claim,
    board_complete,
    board_escalate,
    board_fail,
]
