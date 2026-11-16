from __future__ import annotations



from typing import Any



from langchain_core.messages import AnyMessage



from .facts import _default_facts

from .long_term import LongTermMemory

from .short_term import ShortTermMemory

from .wiki_gateway import WikiGateway





class MemoryService:

    def __init__(

        self,

        short_term: ShortTermMemory | None = None,

        long_term: LongTermMemory | None = None,

        wiki: WikiGateway | None = None,

    ):

        self.short_term = short_term or ShortTermMemory(checkpointer=None)

        self.long_term = long_term or LongTermMemory()

        self.wiki = wiki or WikiGateway()



    async def get_messages(self, session_id: str) -> list[AnyMessage]:

        return await self.short_term.get_messages(session_id)



    async def remember_preference(

        self, session_id: str, key: str, content: dict[str, Any]

    ) -> None:

        await self.long_term.remember(

            namespace=("session", session_id, "preference"),

            key=key,

            content=content,

        )



    async def recall_preferences(

        self, session_id: str, query: str, top_k: int = 5

    ) -> list[dict[str, Any]]:

        return await self.long_term.recall(

            namespace=("session", session_id, "preference"),

            query=query,

            top_k=top_k,

        )



    async def search_wiki(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:

        return await self.wiki.search_wiki(query, top_k)



    async def write_wiki(

        self, title: str, content: str, source_run_id: str

    ) -> str:

        return await self.wiki.write_wiki(title, content, source_run_id)



    async def search_facts(self, query: str, k: int = 5) -> list[dict[str, Any]]:

        """从长期记忆事实库中按关键词召回相关事实。"""

        import asyncio



        return await asyncio.to_thread(_default_facts.search, query, k=k)

