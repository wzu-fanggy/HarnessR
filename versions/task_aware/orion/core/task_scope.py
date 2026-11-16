"""当前用户任务的只读消息视图；不裁剪或重写 checkpoint 历史。"""

from __future__ import annotations

from typing import Any, Mapping

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Overwrite


# 旧 checkpoint 没有角色元数据，只对已知的内部提醒保留兼容识别。
_LEGACY_CONTROL_PREFIXES = (
    "（系统提醒）", "（结构化设计门禁）", "（文献语义审查）", "(System control)",
    "你的上一次回复被截断了，请继续输出剩余内容",
)


class TaskBoundaryError(ValueError):
    """有显式边界却找不到原始用户消息时，禁止退回整段历史。"""


def internal_human_message(content: str, **kwargs: Any) -> HumanMessage:
    metadata = {**kwargs.pop("additional_kwargs", {}), "orion_internal": True}
    return HumanMessage(content=content, additional_kwargs=metadata, **kwargs)


def is_user_input(message: Any) -> bool:
    if not isinstance(message, HumanMessage):
        return False
    if (message.additional_kwargs or {}).get("orion_internal"):
        return False
    if (message.additional_kwargs or {}).get("orion_task_input"):
        return True
    return not str(message.content or "").startswith(_LEGACY_CONTROL_PREFIXES)


def current_task_messages(state: Mapping[str, Any] | None) -> list[Any]:
    """优先使用消息 ID；旧状态按最后一条真实用户输入恢复边界。

    无 HumanMessage 的历史兼容直接工具/离线单批次调用。有显式但缺失的
    ID 则报错，不把旧证据误当作当前证据。返回列表从不写回 messages。
    """
    messages = list((state or {}).get("messages") or [])
    anchor = (state or {}).get("task_start_message_id")
    if anchor:
        for index, message in enumerate(messages):
            if getattr(message, "id", None) == anchor:
                if not isinstance(message, HumanMessage):
                    raise TaskBoundaryError("当前任务边界没有指向用户消息")
                # 标注局部视图，避免旧版前缀启发式误认用户原话；不写回历史。
                user = message.model_copy(update={"additional_kwargs": {
                    **message.additional_kwargs, "orion_task_input": True,
                }})
                return [user, *messages[index + 1:]]
        raise TaskBoundaryError("当前任务的用户消息 ID 不在历史中；不能使用整段历史继续执行")
    for index in range(len(messages) - 1, -1, -1):
        if is_user_input(messages[index]):
            return messages[index:]
    if any(isinstance(m, HumanMessage) and not m.additional_kwargs.get("orion_internal")
           for m in messages):
        raise TaskBoundaryError("旧状态只包含内部提醒，无法确定原始用户任务")
    return messages


def current_task_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """只改变局部读取视图，节点仍只返回增量更新。"""
    messages = current_task_messages(state)
    result = {**state, "messages": messages}
    # 老 checkpoint 的标量终态没有所属消息 ID。多轮时只认可本轮工具记录，
    # 不能让一个来源不明的 last_report 绕过消息范围检查。
    prefix = list(state.get("messages") or [])[:len(state.get("messages") or []) - len(messages)]
    if not state.get("task_start_message_id") and any(is_user_input(m) for m in prefix):
        commit_id = state.get("terminal_commit_tool_call_id")
        owned = any(
            isinstance(m, AIMessage) and not m.tool_calls
            and m.additional_kwargs.get("orion_final_response")
            and str(m.content) == str(state.get("last_report") or "")
            and state.get("task_artifact") is None and not commit_id
            for m in messages
        ) or any(
            isinstance(m, ToolMessage) and m.status != "error"
            and str(m.content).strip() == str(state.get("last_report") or "").strip()
            and (state.get("task_artifact") is None or m.artifact == state.get("task_artifact"))
            and (
                (commit_id and m.tool_call_id == commit_id)
                or (not commit_id and m.name == "synthesize_design_report"
                    and str(m.content) == state.get("last_report"))
            ) for m in messages
        )
        if not owned:
            result.update(last_report=None, task_artifact=None, terminal_committed=False,
                          terminal_commit_tool_call_id=None, applicability_records=None,
                          claim_contract_summary=None)
    return result


def current_user_message(state: Mapping[str, Any]) -> HumanMessage | None:
    messages = current_task_messages(state)
    return next((m for m in messages if is_user_input(m)), None)


def task_entry_updates(state: Mapping[str, Any]) -> dict[str, Any]:
    """仅由新输入入口调用；interrupt/resume 不经过此入口。"""
    # 入口收到的是新用户输入，即使用户文字恰好以旧提醒前缀开头也要接受。
    user = next((m for m in reversed(state.get("messages") or [])
                 if isinstance(m, HumanMessage)
                 and not (m.additional_kwargs or {}).get("orion_internal")), None)
    if user is None:
        raise TaskBoundaryError("新任务没有真实用户输入")
    return {
        "task_start_message_id": user.id,
        "todo_list": Overwrite([]),
        "design_original_question": str(user.content),
        "nudge_count": 0,
        "design_search_rounds": 0,
        "workflow_evidence_rounds": 0,
        "llm_transport_retry_count": 0,
        "claim_contract_summary": None,
        "applicability_records": None,
    }
