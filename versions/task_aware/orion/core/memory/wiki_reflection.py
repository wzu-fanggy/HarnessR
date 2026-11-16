from __future__ import annotations
from release_paths import release_path



import os

import re

import threading

from datetime import datetime

from pathlib import Path

from typing import Annotated, Any



from langchain_core.tools import tool

from langgraph.prebuilt import InjectedState



from orion.core.claim_contract import ClaimContract

from orion.core.memory.reflection_miner import mine_reflections

from orion.core.memory.reflection_store import ReflectionStore



from .facts import _default_facts as _default_ltm





_DEEPAGENTS_ROOT = Path(os.environ.get("DEEPAGENTS_ROOT", release_path('${ORION_SOURCE}')))

WIKI_ROOT = _DEEPAGENTS_ROOT / "wiki"

REFLECTIONS_DIR = WIKI_ROOT / "reflections"



_index_lock = threading.Lock()





def _reflections_dir() -> Path:

    return WIKI_ROOT / "reflections"





def _sanitize_filename(s: str) -> str:

    return re.sub(r"[^\w\-_.]", "_", s)[:80]





def _extract_reflection_title(text: str) -> str:

    m = re.search(r"### 反思笔记[：:]\s*(.+)", text)

    if m:

        return m.group(1).strip()

    for line in text.splitlines():

        line = line.strip()

        if line.startswith("#"):

            title = line.lstrip("#").strip()

            if title:

                return title

    return "reflection"





def _ensure_dirs():

    _reflections_dir().mkdir(parents=True, exist_ok=True)

    (WIKI_ROOT / "entities").mkdir(parents=True, exist_ok=True)

    (WIKI_ROOT / "concepts").mkdir(parents=True, exist_ok=True)





def _append_to_index(entry: str) -> None:

    index_path = WIKI_ROOT / "index.md"

    with _index_lock:

        existing = (

            index_path.read_text(encoding="utf-8")

            if index_path.exists()

            else "# LLM Wiki 索引\n\n"

        )

        if not existing.endswith("\n"):

            existing += "\n"

        index_path.write_text(existing + entry, encoding="utf-8")





def rebuild_index() -> None:

    """Rebuild the Wiki index from existing reflection files."""

    _ensure_dirs()

    index_path = WIKI_ROOT / "index.md"

    entries = []

    for path in sorted(_reflections_dir().glob("*.md")):

        text = path.read_text(encoding="utf-8")

        title = _extract_reflection_title(text)

        entries.append(f"- [{path.stem}](reflections/{path.name}) - {title[:60]}...\n")

    with _index_lock:

        index_path.write_text("# LLM Wiki 索引\n\n" + "".join(entries), encoding="utf-8")

    print(f"[wiki_reflection] rebuilt index with {len(entries)} entries", flush=True)





def _load_llm():

    import sys

    if str(_DEEPAGENTS_ROOT) not in sys.path:

        sys.path.insert(0, str(_DEEPAGENTS_ROOT))

    from llms import main_model

    return main_model





def _extract_facts_from_reflection(reflection: str) -> list[dict[str, str]]:

    """从反思文本中简单抽取候选事实（entity relation value）。"""

    facts = []

    for line in reflection.splitlines():

        line = line.strip()

        if not line.startswith("-"):

            continue

        # 简单规则："- FEC improves SEI" -> entity=FEC, relation=improves, value=SEI

        parts = line.lstrip("-").strip().split(None, 2)

        if len(parts) >= 3:

            facts.append({"entity": parts[0], "relation": parts[1], "value": parts[2]})

    return facts





def reflect_and_update(

    session_id: str,

    run_id: str,

    messages: list[dict[str, Any]] | None = None,

    tool_calls: list[dict[str, Any]] | None = None,

    claims: list[dict[str, Any] | ClaimContract] | None = None,

    task_type: str = "design",

) -> None:

    """

    根据一次运行生成反思笔记，写入 Wiki 索引，并把 episode + facts 存入长期记忆。



    当传入 claims 时，会优先通过结构化 miner 抽取失败模式并写入 ReflectionStore；

    保留 messages/tool_calls 参数以兼容旧调用点。

    """

    _ensure_dirs()



    # ---- 结构化 reflection ----

    if claims:

        claim_objects: list[ClaimContract] = []

        for c in claims:

            if isinstance(c, ClaimContract):

                claim_objects.append(c)

            elif isinstance(c, dict):

                claim_objects.append(ClaimContract.model_validate(c))

        entries = mine_reflections(claim_objects, task_type)

        if entries:

            store = ReflectionStore()

            for entry in entries:

                store.add_or_update(entry)

            print(f"[wiki_reflection] persisted {len(entries)} structured reflection(s)", flush=True)



    user_msgs = [m for m in (messages or []) if m.get("role") == "user"]

    assistant_msgs = [m for m in (messages or []) if m.get("role") == "assistant"]

    question = user_msgs[-1].get("content", "") if user_msgs else ""

    answer = assistant_msgs[-1].get("content", "") if assistant_msgs else ""



    search_results = []

    for tc in (tool_calls or []):

        if tc.get("tool_name") in ("search_kg_rag_hybrid", "search_literature_chunks", "search_literature_multiview"):

            output = tc.get("output")

            if isinstance(output, list):

                search_results.extend(output)

            elif isinstance(output, str):

                try:

                    import json

                    parsed = json.loads(output)

                    if isinstance(parsed, list):

                        search_results.extend(parsed)

                except Exception:

                    pass



    snippets = "\n\n".join(

        f"[{i+1}] {r.get('paper_title', '')}: {str(r.get('content', ''))[:300]}"

        for i, r in enumerate(search_results[:5])

    )



    prompt = f"""你是一位科学文献助手维护员。请根据以下用户问题、检索证据和回答，生成一段结构化的反思笔记，用于更新知识库 Wiki。



## 用户问题

{question}



## 检索证据

{snippets}



## 助手回答摘要

{answer[:800]}



请用 Markdown 输出：

1. 核心结论（3 条以内）

2. 涉及的关键实体/概念（建议创建 Wiki 页的名称）

3. 知识缺口或需要进一步验证的地方

4. 来源之间是否存在矛盾

5. 针对该问题未来可改进的检索关键词

"""



    try:

        llm = _load_llm()

        reflection = llm.invoke(prompt).content

    except Exception as exc:

        print(f"[wiki_reflection] llm reflection failed: {exc}", flush=True)

        return



    date_str = datetime.now().strftime("%Y-%m-%d")

    short_run_id = run_id[:8]

    filename = f"{date_str}-{short_run_id}-{_sanitize_filename(session_id)}.md"

    reflection_path = _reflections_dir() / filename



    reflection_path.write_text(

        f"# Reflection {run_id}\n\n**Session:** {session_id}\n\n**Time:** {datetime.now().isoformat()}\n\n{reflection}\n",

        encoding="utf-8",

    )



    entry = f"- [{date_str}-{short_run_id}-{session_id}](reflections/{filename}) - {question[:60]}...\n"

    _append_to_index(entry)



    # 写入长期记忆

    summary = f"{question[:120]} -> {answer[:200]}"

    entities = re.findall(r"[A-Z][a-zA-Z0-9]+", reflection)[:10]

    try:

        _default_ltm.write_episode(session_id, run_id, summary, entities=list(set(entities)))

        for fact in _extract_facts_from_reflection(reflection):

            _default_ltm.write_fact(

                session_id=session_id,

                run_id=run_id,

                entity=fact["entity"],

                relation=fact["relation"],

                value=fact["value"],

                source="wiki_reflection",

            )

    except Exception as exc:

        print(f"[wiki_reflection] memory write failed: {exc}", flush=True)



    print(f"[wiki_reflection] wrote {reflection_path}", flush=True)





@tool

def reflect_now(

    session_id: str = "",

    run_id: str = "",

    state: Annotated[dict, InjectedState] | None = None,

) -> str:

    """触发一次 Wiki 反思和长期记忆更新。



    未提供 session_id/run_id 时，从 agent state 的 configurable.thread_id

    自动推断，确保在 skill 工作流末尾可无参数调用。

    """

    try:

        messages: list[dict[str, Any]] = []

        tool_calls: list[dict[str, Any]] = []

        if state is not None:

            config = state.get("configurable") or {}

            if not session_id:

                session_id = config.get("thread_id", "default")

            if not run_id:

                # 优先使用最后一条 HumanMessage 的 id 作为 run_id，回退到 thread_id

                from langchain_core.messages import HumanMessage

                raw_messages = state.get("messages", [])

                for m in reversed(raw_messages):

                    if isinstance(m, HumanMessage):

                        run_id = getattr(m, "id", "") or ""

                        break

                if not run_id:

                    run_id = session_id

            # 将 LangGraph state 中的消息与工具结果转成 reflect_and_update 需要的格式

            for m in state.get("messages", []):

                msg_dict: dict[str, Any] = {"role": getattr(m, "type", "")}

                if hasattr(m, "content"):

                    msg_dict["content"] = m.content

                messages.append(msg_dict)

                if getattr(m, "type", "") == "tool":

                    tool_calls.append({

                        "tool_name": getattr(m, "name", ""),

                        "output": getattr(m, "content", "") if not hasattr(m, "artifact") else m.artifact,

                    })

        if not session_id:

            session_id = "default"

        if not run_id:

            run_id = session_id

        reflect_and_update(

            session_id=session_id,

            run_id=run_id,

            messages=messages or None,

            tool_calls=tool_calls or None,

        )

        return f"已完成反思：session={session_id}, run={run_id}"

    except Exception as e:

        return f"反思失败：{e}"

