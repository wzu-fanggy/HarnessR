"""把 DeepSeek 思考模式的 ``reasoning_content`` 回传给出站请求。

背景
----
``langchain_deepseek`` 入站已把 ``reasoning_content`` 放进
``AIMessage.additional_kwargs``（``chat_models.py`` 的 ``_create_chat_result`` /
流式分支）。但出站由 ``langchain_openai`` 的 ``_convert_message_to_dict`` 负责，
它只还原 ``tool_calls`` / ``function_call``，**不会**把
``additional_kwargs["reasoning_content"]`` 写回请求体。

DeepSeek 在思考模式下要求：上一轮 assistant 的 ``reasoning_content`` 必须随该
消息一起发回，否则第二轮请求直接返回::

    400 - The `reasoning_content` in the thinking mode must be passed back to the API.

多轮工具循环必然产生第二轮，于是"开思考"的运行会随机在第一次工具返回后崩掉。

本模块只做一件事：构造请求体时，把仍带 ``reasoning_content`` 的 assistant 消息
原样补回。不改变任何判断、不改工具、不按任务或论文分支；对未携带
``reasoning_content`` 的消息（即思考关闭的运行）是逐字节无操作。
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_deepseek import ChatDeepSeek


def _reasoning_content(message: BaseMessage) -> str | None:
    """取出该消息携带的推理内容；没有或类型不符时返回 None。"""

    if not isinstance(message, AIMessage):
        return None
    kwargs = getattr(message, "additional_kwargs", None) or {}
    value = kwargs.get("reasoning_content")
    return value if isinstance(value, str) and value else None


def restore_reasoning_content(
    source_messages: list[BaseMessage],
    payload: dict,
) -> dict:
    """把 ``source_messages`` 的推理内容补回 ``payload["messages"]``。

    请求体与前缀消息一一对应（``BaseChatOpenAI._get_request_payload`` 逐个转换），
    因此按位置配对即可。只在两端都是 assistant 且源消息确实带推理内容时写入。
    """

    body_messages = payload.get("messages")
    if not isinstance(body_messages, list):
        return payload
    for source, body in zip(source_messages, body_messages):
        if not isinstance(body, dict) or body.get("role") != "assistant":
            continue
        reasoning = _reasoning_content(source)
        if reasoning is not None:
            body["reasoning_content"] = reasoning
    return payload


class ReasoningPreservingChatDeepSeek(ChatDeepSeek):
    """``ChatDeepSeek`` 的子类，出站时保留思考模式的推理内容。"""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        try:
            source_messages = self._convert_input(input_).to_messages()
        except Exception:
            # 输入形态异常时保持上游行为，不因本修复引入新的失败面。
            return payload
        return restore_reasoning_content(source_messages, payload)
