from __future__ import annotations



from typing import Any



from langchain_core.messages import AnyMessage





class ShortTermMemory:

    """基于 LangGraph checkpointer 的短期会话记忆包装。"""



    def __init__(self, checkpointer: Any):

        self.checkpointer = checkpointer



    async def get_messages(self, session_id: str) -> list[AnyMessage]:

        if self.checkpointer is None:

            return []

        try:

            config = {"configurable": {"thread_id": session_id}}

            state = await self.checkpointer.aget(config)

            if state is None:

                return []

            # LangGraph checkpoint state 里 messages 字段保存历史消息

            values = state.get("channel_values") or state.get("values") or {}

            messages = values.get("messages", [])

            return [m for m in messages if isinstance(m, AnyMessage)]

        except Exception:

            return []



    async def save_messages(

        self, session_id: str, messages: list[AnyMessage]

    ) -> None:

        """LangGraph 运行时会自动保存 checkpoint，这里通常不需要手动调用。"""

        pass



    async def delete_thread(self, session_id: str) -> None:

        if self.checkpointer is None:

            return

        try:

            config = {"configurable": {"thread_id": session_id}}

            if hasattr(self.checkpointer, "adelete"):

                await self.checkpointer.adelete(config)

        except Exception:

            pass

