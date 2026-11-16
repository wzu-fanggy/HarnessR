from __future__ import annotations



import asyncio

import json

import os

import time

from datetime import datetime, timezone

from hashlib import sha1

from pathlib import Path

from typing import TYPE_CHECKING, Any, Callable

from uuid import uuid4



from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from orion.core.task_scope import current_task_messages, current_task_state

from orion.core.design_runtime import terminal_result



from orion.evaluation.models import EvaluationSample, ToolCallRecord



if TYPE_CHECKING:

    from orion.evaluation.config import EvalConfig



# 证据 contexts 抽取逻辑已上移到中立模块 orion.core.evidence_extract

# （生产侧 grounding 校验与评估侧共用同一套口径）。此处 re-export 保持兼容。

from orion.core.evidence_extract import (

    EVIDENCE_TOOL_PREFIXES as _EVIDENCE_TOOL_PREFIXES,

    MAX_CONTEXTS as _MAX_CONTEXTS,

    PER_CONTEXT_LIMIT as _PER_CONTEXT_LIMIT,

    _cited_rank,

    extract_contexts,

    parse_chunk_payload as _parse_chunk_payload,

)





def extract_tool_calls(messages: list[Any]) -> list[ToolCallRecord]:

    """按 tool_call_id 而不是消息相邻顺序关联科研工具返回值。



    同一 AIMessage 可包含多项并行调用，ToolMessage 到达顺序不保证与

    请求顺序一致。缺失或未知 ID 的返回值不归入其他工具，避免生成

    错误的化学结果对应关系；ToolCallRecord 的历史输出结构保持不变。

    """

    messages = current_task_messages({"messages": messages})

    from collections import defaultdict, deque



    records: list[ToolCallRecord] = []

    pending: dict[str, deque[ToolCallRecord]] = defaultdict(deque)

    for msg in messages:

        if isinstance(msg, AIMessage):

            for tc in getattr(msg, "tool_calls", []) or []:

                record = ToolCallRecord(

                    name=tc.get("name", ""),

                    args=tc.get("args", {}),

                )

                records.append(record)

                call_id = str(tc.get("id") or "").strip()

                if call_id:

                    pending[call_id].append(record)

        elif isinstance(msg, ToolMessage):

            call_id = str(getattr(msg, "tool_call_id", "") or "").strip()

            candidates = pending.get(call_id)

            if not candidates:

                continue

            record = candidates[0]

            result_name = str(getattr(msg, "name", "") or "").strip()

            if result_name and result_name != record.name:

                # 即使 ID 被复用或消息损坏，也不得把另一工具的结果塞入。

                continue

            candidates.popleft()

            record.result = msg.content

    return records





def extract_task_artifact(

    messages: list[Any],

    state_artifact: Any = None,

) -> dict[str, Any] | None:

    """提取终端报告的结构化 artifact，不从 Markdown 反向猜测。"""



    messages = current_task_messages({"messages": messages})

    artifact = state_artifact

    if hasattr(artifact, "model_dump"):

        artifact = artifact.model_dump(mode="json")

    if isinstance(artifact, dict):

        return artifact

    result = terminal_result({"messages": messages})

    return result[1] if result else None





def _event_sink_from_env(

    sample: EvaluationSample,

) -> Callable[[dict[str, Any]], None] | None:

    """为 custom/in-process runner 补上与子进程相同的 trace 记录。"""



    trace_dir = os.environ.get("ORION_EVAL_TRACE_DIR", "")

    if not trace_dir:

        return None

    root = Path(trace_dir)

    root.mkdir(parents=True, exist_ok=True)

    sample_id = sha1(sample.question.encode("utf-8")).hexdigest()[:12]

    trace_path = root / f"{sample_id}.jsonl"



    def _sink(event: dict[str, Any]) -> None:

        payload = {

            "timestamp_utc": datetime.now(timezone.utc).isoformat(),

            "sample_id": sample_id,

            "task_type": sample.task_type,

            **event,

        }

        with trace_path.open("a", encoding="utf-8") as handle:

            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")



    return _sink





def _tool_name(tc: dict) -> str:

    """兼容 LangChain 不同版本的 tool_call name 字段位置。"""

    return tc.get("name") or tc.get("function", {}).get("name", "")





# 不应被视为最终答案的中间/确认消息集合

_INTERMEDIATE_ANSWER_MARKERS = {

    "报告已生成，正在进行 grounding 校验",

    "报告已生成，正在进行 grounding 校验。",

    "report generated, proceeding to grounding check",

    "报告已完成，以上为总结。",

}





def _is_intermediate_answer(content: str) -> bool:

    """判断内容是否是 agent 在调用 synthesize_design_report 后输出的过渡确认句。"""

    if not content:

        return True

    stripped = content.strip()

    if len(stripped) < 10:

        return True

    return stripped in _INTERMEDIATE_ANSWER_MARKERS





def extract_answer(messages: list[Any], last_report: str | None = None) -> str:

    """提取最终答案。



    优先顺序：

    1. state 中保存的 last_report（review 节点从 synthesize_design_report

       工具结果提取并写入 state，最可靠）；

    2. `synthesize_design_report` 工具返回的完整 markdown 报告；

    3. 最后一条非空且非中间确认消息的 AI 消息（grounding 节点会把最终答案写回 messages）。



    v58 修复：当 ORION_GROUNDING_CHECK=0 时 grounding 节点会直接 END，

    若不在此处读取 last_report 或最后一条 AI message，评估会误把中间

    进度消息当成最终答案，导致 answer_relevancy/design_feasibility 全部崩盘。

    v60 修复：降低 last_report 长度阈值以兼容 screening 简洁回答；过滤掉

    "报告已生成，正在进行 grounding 校验" 等过渡消息。

    """

    messages = current_task_messages({"messages": messages})

    target = "synthesize_design_report"



    # 1) 优先使用 review 节点保存的终端报告。

    # screening 回答可能只有 50-200 字，阈值过高会导致 fallback 到中间消息。

    if last_report and len(str(last_report).strip()) >= 50:

        return str(last_report)

    #    优先于相邻对查找，因为 agent 可能在同一条 AI 消息中同时发起

    #    write_todos 与 synthesize_design_report；相邻对 heuristic 会

    #    误取 write_todos 的结果（同为 >200 字），导致 answer 变成 todo_list。

    #    (v55 实测 sample 4 inverse_design)

    result = terminal_result({"messages": messages})

    if result and len(result[0]) > 200:

        return result[0]



    # 4) 兜底：取最后一条非空 AI 消息，但跳过已知的过渡确认句。

    #    v58 之前用“最长 AI 消息”，但 agent 可能在早期就生成很长的中间

    #    复述；最终答案（经 grounding 写回）通常位于最后一条。

    for msg in reversed(messages):

        if isinstance(msg, AIMessage):

            content = str(getattr(msg, "content", None) or "")

            if content and not _is_intermediate_answer(content):

                return content

    return ""





def run_agent(

    sample: EvaluationSample,

    config: EvalConfig,

    agent_factory=None,

    event_sink: Callable[[dict[str, Any]], None] | None = None,

) -> dict[str, Any]:

    """运行一次 Agent 并抽取评估所需字段。"""

    previous_auto_approve = os.environ.get("ORION_AUTO_APPROVE")

    os.environ["ORION_AUTO_APPROVE"] = "1" if config.auto_approve else "0"

    started_at = time.monotonic()

    if event_sink is None:

        event_sink = _event_sink_from_env(sample)

    captured_events: list[dict[str, Any]] = []

    downstream_event_sink = event_sink

    if downstream_event_sink is not None:

        def _capture_and_forward(event: dict[str, Any]) -> None:

            captured_events.append(dict(event))

            downstream_event_sink(event)



        event_sink = _capture_and_forward



    try:

        if agent_factory is None:

            from orion.main import build_streaming_agent



            # run_agent 本身是异步执行路径；使用 async LLM 节点但通过

            # configurable.llm_stream=False 关闭 token streaming。这样既保留

            # AsyncSqliteSaver，也能让 llm_async 的单步 timeout 生效。

            agent_factory = build_streaming_agent



        thread_id = f"eval-{uuid4().hex[:8]}"

        timeout_seconds = float(os.environ.get("ORION_EVAL_AGENT_TIMEOUT", "900"))

        default_llm_step_timeout = min(90.0, max(15.0, timeout_seconds / 6.0))

        llm_step_timeout = float(

            os.environ.get("ORION_EVAL_LLM_STEP_TIMEOUT", str(default_llm_step_timeout))

        )

        llm_request_max_retries = int(

            os.environ.get(

                "ORION_EVAL_LLM_REQUEST_MAX_RETRIES",

                os.environ.get("ORION_LLM_REQUEST_MAX_RETRIES", "2"),

            )

        )

        runnable_config = {

            "configurable": {

                "thread_id": thread_id,

                # 评测必须使用异步非流式 LLM 节点，才能让单步 timeout 真正生效，

                # 避免某一次长响应吃掉整个样本预算并让 checkpoint 结果丢失。

                "llm_stream": False,

                "llm_step_timeout": llm_step_timeout,

                "llm_request_max_retries": llm_request_max_retries,

                # grounding 用这两个值把自身预算钳制到样本剩余墙钟时间内。

                "evaluation_started_at_monotonic": started_at,

                "evaluation_timeout_seconds": timeout_seconds,

                "task_type": sample.task_type,

            },

            # 配方设计会经过多轮检索、筛选、补证与 review；LangGraph 默认 25

            # 会把正常工作流误判为循环。评测仅提高图步数上限，不改变领域决策逻辑。

            "recursion_limit": int(os.environ.get("ORION_EVAL_RECURSION_LIMIT", "80")),

        }



        async def _invoke() -> dict[str, Any]:

            # agent 必须在事件循环内构建：orion.main._build_graph 检测到运行中

            # 的 loop 才会选 AsyncSqliteSaver；在同步上下文先构建再 ainvoke 会

            # 拿到 SqliteSaver，直接报 "does not support async methods"

            # （v5 样本5 修复 sync-invoke 后再次踩中）。

            agent = agent_factory()

            try:

                # 必须异步调用：task_dispatch 等子代理工具为 async-only，同步

                # invoke 会在 ToolNode 抛 "StructuredTool does not support

                # sync invocation" 中断整个样本（v5 样本5 实测踩中）。

                inputs = {"messages": [HumanMessage(content=sample.question)]}



                async def _checkpoint_after_timeout() -> dict[str, Any]:

                    """样本墙钟超时后保留最近 checkpoint，而不是退化成空 agent_error。"""

                    try:

                        state = await asyncio.wait_for(

                            agent.aget_state(runnable_config), timeout=5.0

                        )

                        values = dict(state.values)

                    except Exception:

                        values = {"messages": []}

                    values["evaluation_timed_out"] = True

                    values["evaluation_timeout_seconds"] = timeout_seconds

                    return values



                async def _checkpoint_after_error(exc: BaseException) -> dict[str, Any]:

                    """Preserve the latest durable trace instead of returning an empty failure."""



                    try:

                        state = await asyncio.wait_for(

                            agent.aget_state(runnable_config), timeout=5.0

                        )

                        values = dict(state.values)

                    except Exception:

                        values = {"messages": []}

                    values["evaluation_runtime_error"] = {

                        "error_type": type(exc).__name__,

                        "error_message": str(exc).replace("\r", " ").replace("\n", " ")[:1000],

                    }

                    failed_request_retries = int(

                        getattr(exc, "orion_llm_transport_retry_count", 0) or 0

                    )

                    failed_request_max_retries = getattr(

                        exc, "orion_llm_request_max_retries", None

                    )

                    values["llm_transport_retry_count"] = int(

                        values.get("llm_transport_retry_count") or 0

                    ) + failed_request_retries

                    if failed_request_max_retries is not None:

                        values["llm_request_max_retries"] = int(

                            failed_request_max_retries

                        )

                    values["evaluation_runtime_error"].update(

                        {

                            "llm_transport_retry_count": failed_request_retries,

                            "llm_request_max_retries": (

                                int(failed_request_max_retries)

                                if failed_request_max_retries is not None

                                else None

                            ),

                        }

                    )

                    return values



                if event_sink is None:

                    try:

                        return await asyncio.wait_for(

                            agent.ainvoke(inputs, config=runnable_config), timeout=timeout_seconds

                        )

                    except asyncio.TimeoutError:

                        return await _checkpoint_after_timeout()

                    except Exception as exc:

                        return await _checkpoint_after_error(exc)



                # 复用 LangGraph 原生事件流；最终状态仍由 checkpoint 读取，

                # 因而不会把流式片段误当成最终答案。

                async def _stream() -> None:

                    async for event in agent.astream_events(inputs, runnable_config, version="v2"):

                        kind = event.get("event", "")

                        if kind in {

                            "on_chain_start",

                            "on_chain_end",

                            "on_chain_error",

                            "on_chat_model_start",

                            "on_chat_model_end",

                            "on_chat_model_error",

                            "on_llm_start",

                            "on_llm_end",

                            "on_llm_error",

                            "on_tool_start",

                            "on_tool_end",

                            "on_tool_error",

                        }:

                            # 单调时钟使 trace 可直接配对 start/end 并计算耗时；

                            # 同时保留 LangGraph run_id/parent_ids，以区分同名嵌套链与重试。

                            event_sink(

                                {

                                    "event": kind,

                                    "name": event.get("name", ""),

                                    "run_id": event.get("run_id", ""),

                                    "parent_ids": event.get("parent_ids", []),

                                    "elapsed_seconds": round(time.monotonic() - started_at, 3),

                                }

                            )



                try:

                    await asyncio.wait_for(_stream(), timeout=timeout_seconds)

                except asyncio.TimeoutError:

                    return await _checkpoint_after_timeout()

                except Exception as exc:

                    return await _checkpoint_after_error(exc)

                state = await agent.aget_state(runnable_config)

                return dict(state.values)

            finally:

                # AsyncSqliteSaver 的 aiosqlite worker 是非守护线程，不显式关闭

                # 进程退出会挂死（见 orion.main.aclose_agent docstring）。

                try:

                    from orion.main import aclose_agent



                    await aclose_agent(agent)

                except Exception:

                    pass



        final_state = asyncio.run(_invoke())



        final_state = current_task_state(final_state)

        messages = current_task_messages(final_state)

        answer = extract_answer(messages, last_report=final_state.get("last_report"))



        contexts = extract_contexts(messages, answer=answer)

        tool_calls = extract_tool_calls(messages)

        tool_messages = [str(m.content) for m in messages if isinstance(m, ToolMessage)]

        task_artifact = extract_task_artifact(

            messages,

            state_artifact=final_state.get("task_artifact"),

        )

        if task_artifact is not None and not task_artifact.get("trace"):

            artifact_trace = [

                {

                    "event": str(event.get("event") or ""),

                    "tool": str(event.get("name") or ""),

                    "status": (

                        "started"

                        if str(event.get("event") or "").endswith("_start")

                        else "completed"

                    ),

                }

                for event in captured_events

                if event.get("event") in {"on_tool_start", "on_tool_end"}

            ]

            if artifact_trace:

                task_artifact = {**task_artifact, "trace": artifact_trace}



        runtime_error = final_state.get("evaluation_runtime_error")

        timed_out = bool(final_state.get("evaluation_timed_out"))

        failure_reasons: list[str] = []

        if timed_out:

            failure_reasons.append("agent_timeout")

        if isinstance(runtime_error, dict):

            failure_reasons.append(

                f"agent_error:{runtime_error.get('error_type') or 'RuntimeError'}"

            )



        return {

            "ok": not timed_out and not runtime_error,

            "answer": answer,

            "contexts": contexts,

            "tool_calls": tool_calls,

            "tool_messages": tool_messages,

            "task_artifact": task_artifact,

            "event_trace_count": len(captured_events),

            "failure_reasons": failure_reasons,

            "error_type": (

                runtime_error.get("error_type")

                if isinstance(runtime_error, dict)

                else None

            ),

            "error_message": (

                runtime_error.get("error_message")

                if isinstance(runtime_error, dict)

                else None

            ),

            "grounding_status": final_state.get("grounding_status"),

            "grounding_error": final_state.get("grounding_error"),

            "grounding_duration_seconds": final_state.get("grounding_duration_seconds"),

            "grounding_fallback_used": bool(final_state.get("grounding_fallback_used")),

            "grounding_primary_error": final_state.get("grounding_primary_error"),

            "claim_contract_summary": final_state.get("claim_contract_summary"),

            # Persist the task identity that the graph actually carried.  The

            # sample's declared task_type is the requested route; this field is

            # the independent runtime observation used by evaluation audits to

            # detect task-type drift.

            "configured_task_type": final_state.get("configured_task_type"),

            "run_duration_seconds": round(time.monotonic() - started_at, 3),

            "evaluation_timed_out": bool(final_state.get("evaluation_timed_out")),

            "evaluation_timeout_seconds": final_state.get("evaluation_timeout_seconds"),

            "llm_request_max_retries": final_state.get("llm_request_max_retries"),

            "llm_transport_retry_count": int(

                final_state.get("llm_transport_retry_count") or 0

            ),

        }

    finally:

        if previous_auto_approve is None:

            os.environ.pop("ORION_AUTO_APPROVE", None)

        else:

            os.environ["ORION_AUTO_APPROVE"] = previous_auto_approve

