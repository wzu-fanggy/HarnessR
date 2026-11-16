from __future__ import annotations



import os

from collections.abc import Sequence



from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from langchain_core.tools import BaseTool



from orion.core.memory.reflection_store import ReflectionStore

from orion.core.memory.service import MemoryService

from .budget import get_context_limit

from .compact import compact_messages, estimate_token_count





class ContextCapacityExceeded(RuntimeError):

    """Required context cannot fit within the configured model-input budget."""





class ContextManager:

    """在用户回合入口组装上下文，不代替 Agent 选择科研工具。



    The assembler follows a fixed priority flow:



    1. Load system constraints.

    2. Attach caller-provided capability context.

    3. Recall long-term memory / preferences.

    4. Fit everything into the token budget.

    """



    def __init__(

        self,

        memory_service: MemoryService,

        model_name: str = "deepseek-v4-pro",

    ):

        self.memory = memory_service

        self.model_name = model_name

        self.reflection_store = ReflectionStore()



    async def assemble(

        self,

        session_id: str,

        user_message: str,

        system_prompt: str,

        tools: list[BaseTool] | None = None,

        loop_state_file: str | None = None,

        task_type: str = "design",

        capability_context_blocks: Sequence[tuple[str, str]] = (),

        optional_capability_context_blocks: Sequence[tuple[str, str]] = (),

    ) -> list[AnyMessage]:

        messages: list[AnyMessage] = [SystemMessage(content=system_prompt)]



        capability_message = self._capability_context_message(capability_context_blocks)

        if capability_message is not None:

            messages.append(capability_message)

        # 一个能力的多段说明作为一个可选单元；调用方按 LRU 顺序提供这些单元。

        required_ids = {item for item, _ in capability_context_blocks}

        optional: dict[str, list[tuple[str, str]]] = {}

        for item, text in optional_capability_context_blocks:

            if item not in required_ids:

                optional.setdefault(item, []).append((item, text))

        for blocks in optional.values():

            message = self._capability_context_message(blocks, required=False)

            if message is not None:

                messages.append(message)



        # 文献/KG 检索由 Agent 显式选择，经工具审批、预算与来源捕获路径执行。

        # 入口不检索或注入无工具轨迹的证据摘要。



        memory = await self._recall_memory(session_id, user_message)

        if memory:

            messages.append(SystemMessage(content=f"长期记忆：\n{memory}"))



        if os.environ.get("ORION_REFLECTION_INJECT") == "1":

            reflections = self._load_reflections(user_message, task_type)

            if reflections:

                messages.append(SystemMessage(content=f"历史教训（动态约束）：\n{reflections}"))



        # NOTE: 不再从 memory 加载历史。LangGraph 的 state.messages 已经保存并管理

        # 完整对话历史；context_prepare 只负责注入系统约束与长期记忆。

        # 从 memory 加载最近 N 条再 extend 会导致 HumanMessage / SystemMessage 重复累积，

        # 最终触发 LLM API "tool message without tool_calls" 错误。

        # history = await self._compact_history(session_id)

        # if history:

        #     messages.extend(history)



        # 追加当前用户消息；若 state.messages 末尾已经是同内容 HumanMessage（LangGraph

        # 正常流程），则跳过，避免重复。

        if user_message is not None:

            if not (

                messages

                and isinstance(messages[-1], HumanMessage)

                and messages[-1].content == user_message

            ):

                messages.append(HumanMessage(content=user_message))



        return self._fit_budget(

            messages,

            enforce_required_fit=bool(capability_context_blocks or optional_capability_context_blocks),

        )



    @staticmethod

    def _capability_context_message(

        blocks: Sequence[tuple[str, str]],

        *,

        required: bool = True,

    ) -> SystemMessage | None:

        """Build the capability-specific stage context required by the current call.



        The capability runtime owns residency decisions. The context manager only

        consumes the resulting blocks and protects them from history compaction.

        """



        normalized: list[str] = []

        for capability_id, content in blocks:

            text = str(content).strip()

            if not text:

                continue

            normalized.append(f"[{capability_id}]\n{text}")

        if not normalized:

            return None

        return SystemMessage(

            content="当前能力上下文：\n" + "\n\n".join(normalized),

            additional_kwargs={

                "context_role": "capability_required" if required else "capability_optional",

                "protected_context": required,

            },

        )



    def _load_reflections(self, goal: str, task_type: str = "design") -> str:

        """当启用 ORION_REFLECTION_INJECT 时，把匹配的 active reflection 注入系统提示。"""

        try:

            entries = self.reflection_store.query(

                task_types=[task_type],

                domain_tags=self._extract_domain_tags(goal),

                min_confidence=0.3,

                limit=5,

            )

            if not entries:

                return ""

            lines = []

            for i, e in enumerate(entries, 1):

                lines.append(f"{i}. [{e.pattern_code}] {e.corrective_constraint}")

            return "\n".join(lines)

        except Exception:

            return ""



    def _extract_domain_tags(self, goal: str) -> list[str]:

        tags = []

        lowered = goal.lower()

        if "hard carbon" in lowered or "hc" in lowered:

            tags.append("hard-carbon")

        if "low temp" in lowered or "-20" in lowered:

            tags.append("low-temp")

        if "sodium" in lowered or "na" in lowered:

            tags.append("sodium-ion")

        return tags



    async def _retrieve_evidence(self, goal: str) -> str:

        """历史显式调用接口；assemble 不调用，科研检索应通过主图工具执行。"""

        if len(goal.strip()) < 3:

            return ""

        try:

            import asyncio

            from tools.rag.kg_rag_tools import search_literature_multiview



            def _search():

                return search_literature_multiview.invoke({"queries": [goal], "top_k": 5})



            results = await asyncio.to_thread(_search)

            if not results:

                return ""

            lines = []

            for i, r in enumerate(results[:5], 1):

                content = str(r.get("content", ""))[:300].replace("\n", " ")

                title = r.get("paper_title", "")

                lines.append(f"[{i}] {content} (from {title})")

            return "\n".join(lines)

        except Exception as exc:

            return f"证据召回失败：{exc}"



    async def _recall_memory(self, session_id: str, goal: str) -> str:

        parts = []



        prefs = await self.memory.recall_preferences(session_id, goal, top_k=3)

        if prefs:

            parts.append("偏好：\n" + "\n".join(f"- {p.get('value', p)}" for p in prefs))



        try:

            facts = await self.memory.search_wiki(goal, top_k=3)

            if facts:

                parts.append("Wiki：\n" + "\n".join(f"- {f.get('title', '')}: {str(f.get('content', ''))[:200]}" for f in facts))

        except Exception:

            pass



        return "\n\n".join(parts)



    async def _compact_history(self, session_id: str) -> list[AnyMessage]:

        msgs = await self.memory.get_messages(session_id)

        recent = msgs[-4:] if len(msgs) > 4 else msgs

        return recent



    def _fit_budget(

        self,

        messages: list[AnyMessage],

        *,

        enforce_required_fit: bool = False,

    ) -> list[AnyMessage]:

        messages = list(messages)

        token_count = estimate_token_count(messages)

        limit = int(get_context_limit() * 0.75)

        # 先让出闲置能力信息占用的输入容量，再压缩普通历史；必需信息始终保留。

        for message in tuple(messages):

            if token_count <= limit:

                break

            metadata = getattr(message, "additional_kwargs", {}) or {}

            if metadata.get("context_role") == "capability_optional":

                messages.remove(message)

                token_count = estimate_token_count(messages)

        if token_count <= limit:

            return messages

        compacted, compacted_count = compact_messages(

            messages,

            token_count=token_count,

            context_limit=limit,

            keep_last_turns=4,

            preserve_evidence=True,

        )

        if enforce_required_fit and compacted_count > limit:

            raise ContextCapacityExceeded(

                "required capability context and ordinary context exceed the model-input budget "

                f"after compaction ({compacted_count} > {limit} estimated tokens)"

            )

        return compacted

