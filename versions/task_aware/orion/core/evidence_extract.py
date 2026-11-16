"""证据 contexts 抽取（中立模块）。



从工具结果消息中抽取"可作为证据"的 contexts，供两处共用：

- 评估侧（orion.evaluation.agent_runner）：喂给 RAGAS judge；

- 运行侧（orion.core.grounding）：答案接地校验回路的证据来源。



两侧必须用同一套抽取逻辑，保证"评估看到的证据"与"生产校验看到的证据"一致。

本模块不依赖 ragas 等评估库；消息范围与生产侧共用当前任务边界。

"""



from __future__ import annotations



import ast

import json

import os

import re

from typing import Any



# 产生检索证据的工具：只有这些工具的结果应作为证据 contexts。

# 设计/评分类工具（screen/validate/optimize）输出的是派生判断而非证据，不计入。

EVIDENCE_TOOL_PREFIXES = (

    "search_literature",

    "search_paper_chunks",

    "kg_",

    "query_formulation_space",

    "find_formulations",

    "extract_numeric_properties",

)



# 设计/评分类工具:其输出本身不是证据,但其中引用的 chunk_id 指向证据——

# 这些 chunk 必须在截断排序中优先保留,否则答案里来自筛选结果的配方断言

# 会因证据被 cap 裁掉而被 NLI 判为不忠实(v42 MISS 归因 ~10-15%)。

DESIGN_TOOL_PREFIXES = (

    "screen_formulations",

    "optimize_formulations",

    "validate_formulations",

)



# 单条 context 长度上限与总条数上限：控制 judge/校验输入规模。

# 截断前先按 chunk_id 去重、再做"被引用优先"排序，避免按消息顺序截断

# 把 agent 实际引用过的 chunk 裁掉导致 faithfulness/context_recall 假性归零

# （v3 实测 cap=25 直接截断时 faithfulness 从 0.231 跌回 0.065）。

PER_CONTEXT_LIMIT = 1200

# 答案中显式引用的 chunk 需要保留更多原文，否则 judge 看不到支持细节，

# 导致 claim_supported_ratio / performance_claim_support 假性偏低。

PER_CONTEXT_LIMIT_CITED = int(os.environ.get("EVIDENCE_CITED_LIMIT", "3000"))

MAX_CONTEXTS = 25





def _render_structured_row(item: dict) -> str:

    """把无 content 字段的结构化行(kg_query/配方空间)渲染为裁判可读的 key: value 行。"""

    parts = []

    for k, v in item.items():

        if isinstance(v, (dict, list)):

            continue

        text = str(v).strip()

        if text:

            parts.append(f"{k}: {text}")

    return "; ".join(parts)





def _cited_rank(ctx: str, answer: str) -> int:

    """答案引用了该 context 所属论文时返回 1，否则 0。



    ctx 形如 "[0002_Cai 等 - 2024 - A dual-functional ...] 正文"；答案中的

    引用通常是 "Cai 等（2024）"/"Cai et al. 2024" 形式，故用（作者, 年份）

    双 token 匹配。无法解析出 作者/年份 的标题（如 arXiv 风格）返回 0。

    """

    if not answer or not ctx.startswith("["):

        return 0

    title = ctx[1:].split("]", 1)[0]

    m = re.search(r"([^_\[\]]+?) 等 - (\d{4})", title)

    if not m:

        return 0

    author, year = m.group(1).strip(), m.group(2)

    return 1 if (author and author.lower() in answer.lower() and year in answer) else 0





def parse_chunk_payload(content: str) -> list | None:

    """把工具返回内容解析为 chunk 列表；非列表 JSON 时返回 None。"""

    text = content.strip()

    if not text.startswith("["):

        return None

    for loader in (json.loads, ast.literal_eval):

        try:

            data = loader(text)

        except Exception:

            continue

        if isinstance(data, list):

            return data

    return None





def _find_evidence_chunks(obj: Any) -> list[dict]:

    """递归查找 dict/list 中的 evidence_chunks 字段，返回其中的 chunk 列表。



    用于捕获 synthesize_design_report 等工具在 tool_call args 里显式传入的

    evidence_chunks，这些 chunk 是 agent 最终答案实际依赖的核心证据。

    """

    found: list[dict] = []

    if isinstance(obj, dict):

        for k, v in obj.items():

            if k == "evidence_chunks" and isinstance(v, list):

                for item in v:

                    if isinstance(item, dict):

                        found.append(item)

            else:

                found.extend(_find_evidence_chunks(v))

    elif isinstance(obj, list):

        for item in obj:

            found.extend(_find_evidence_chunks(item))

    return found





def _entry_from_chunk_item(

    item: dict, per_context_limit: int, answer: str = ""

) -> tuple[str, str] | None:

    """把单个 chunk dict 渲染为 (chunk_id, ctx) 条目。



    若该 chunk 被答案显式引用，则使用更宽松的长度上限，避免 judge 因截断

    而判定该证据不支持答案中的数值/性能声明。

    """

    text = str(item.get("content") or "").strip()

    if not text:

        return None

    chunk_id = str(item.get("chunk_id") or "").strip()

    title = str(item.get("paper_title") or "").strip()

    is_cited = bool(answer and chunk_id and f"[{chunk_id}]" in answer)

    limit = PER_CONTEXT_LIMIT_CITED if is_cited else per_context_limit

    if chunk_id and title:

        ctx = f"[{title}] [{chunk_id}] {text}"

    elif chunk_id:

        ctx = f"[{chunk_id}] {text}"

    elif title:

        ctx = f"[{title}] {text}"

    else:

        ctx = text

    return chunk_id, ctx[:limit]





def extract_contexts(

    messages: list[Any],

    per_context_limit: int = PER_CONTEXT_LIMIT,

    max_contexts: int = MAX_CONTEXTS,

    answer: str = "",

) -> list[str]:

    """从工具结果消息中抽取证据 contexts。



    仅保留检索类工具结果;kg_rag 检索返回的 chunk 列表按条拆分为独立

    context(取 content 字段,附 paper_title 便于溯源);无 content 字段的

    结构化结果(如 kg_query 行)逐行渲染为 key: value 文本。多轮检索重复

    返回的 chunk 按 chunk_id 去重;截断排序时,答案引用过的论文对应 chunk

    与设计工具(screen/optimize/validate)引用过的 chunk 优先保留。

    """

    from orion.core.task_scope import current_task_messages



    messages = current_task_messages({"messages": messages})

    entries: list[tuple[str, str]] = []  # (chunk_id, ctx);无 chunk_id 用 ""

    seen_chunk_ids: set[str] = set()

    design_texts: list[str] = []

    for msg in messages:

        content = getattr(msg, "content", None)

        name = getattr(msg, "name", None) or ""

        if not content or not name:

            continue

        if any(name.startswith(prefix) for prefix in DESIGN_TOOL_PREFIXES):

            design_texts.append(str(content))

            continue

        if not any(name.startswith(prefix) for prefix in EVIDENCE_TOOL_PREFIXES):

            continue

        payload = parse_chunk_payload(str(content))

        if payload is None:

            # 非 chunk 列表(纯文本、解析失败)→ 原始文本回退

            entries.append(("", str(content)[:per_context_limit]))

            continue

        saw_content = False

        structured_rows: list[dict] = []

        for item in payload:

            if not isinstance(item, dict):

                continue

            text = str(item.get("content") or "").strip()

            if not text:

                structured_rows.append(item)

                continue

            saw_content = True

            entry = _entry_from_chunk_item(item, per_context_limit, answer=answer)

            if entry is None:

                continue

            chunk_id, ctx = entry

            if chunk_id:

                if chunk_id in seen_chunk_ids:

                    continue

                seen_chunk_ids.add(chunk_id)

            entries.append((chunk_id, ctx))

        if not saw_content:

            if structured_rows:

                # 纯结构化结果(如 kg_query 行)→ 逐行 key: value 渲染

                for item in structured_rows:

                    row = _render_structured_row(item)

                    if row:

                        entries.append(("", row[:per_context_limit]))

            else:

                entries.append(("", str(content)[:per_context_limit]))



    # v67: agent 在 synthesize_design_report 等工具的 args 中显式指定的 evidence_chunks

    # 是最终答案实际依赖的核心证据。若只从检索工具结果里抽，这些 chunk 可能在

    # MAX_CONTEXTS 截断中被丢弃，导致 claim_support 假性偏低。

    for msg in messages:

        for tc in getattr(msg, "tool_calls", []) or []:

            if not isinstance(tc, dict):

                continue

            args = tc.get("args") or {}

            for item in _find_evidence_chunks(args):

                if not isinstance(item, dict):

                    continue

                entry = _entry_from_chunk_item(item, per_context_limit, answer=answer)

                if entry is None:

                    continue

                chunk_id, ctx = entry

                if chunk_id:

                    if chunk_id in seen_chunk_ids:

                        continue

                    seen_chunk_ids.add(chunk_id)

                entries.append((chunk_id, ctx))



    design_blob = "\n".join(design_texts)



    def _keep_rank(entry: tuple[str, str]) -> int:

        chunk_id, ctx = entry

        cited = _cited_rank(ctx, answer) if answer else 0

        design_ref = 1 if chunk_id and chunk_id in design_blob else 0

        return -(cited * 2 + design_ref)



    # 稳定排序:被引用 chunk 最先、设计工具引用次之,同级保持原有相对顺序

    entries.sort(key=_keep_rank)

    return [ctx for _, ctx in entries[:max_contexts]]

