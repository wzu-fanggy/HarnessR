"""Scientific design policy attached to generic Harness review seams.



This module keeps electrolyte/design-specific convergence and terminalization

rules out of the generic ``review`` node. The Harness owns stable review seams;

a scientific policy may inspect state and tool requests and either return a

LangGraph ``Command`` or decline to intervene.



The policy deliberately receives small callbacks for runtime utilities instead

of importing ``main``. This avoids a circular dependency while keeping the

main graph ignorant of scientific decision semantics.

"""



from __future__ import annotations



from collections.abc import Callable, Mapping, Sequence

from dataclasses import dataclass

from typing import Any



from langchain_core.messages import AIMessage, AnyMessage, RemoveMessage

from langgraph.types import Command



from orion.core.task_scope import internal_human_message

from orion.core.design_decision import DecisionState





@dataclass(frozen=True)

class ScientificReviewContext:

    tool_name: Callable[[dict], str]

    active_task_type: Callable[[Mapping[str, Any], list[AnyMessage]], str]

    successful_tool_names: Callable[[list[AnyMessage]], set[str]]

    all_retrieval_only: Callable[[list[dict]], bool]

    prior_equivalent_successful_batch_count: Callable[[list[AnyMessage], list[dict]], int]

    evaluation_remaining_seconds: Callable[[Mapping[str, Any]], float | None]

    control_formulation_call_from_evidence: Callable[..., AIMessage | None]

    control_screen_call_from_formulation_result: Callable[..., AIMessage | None]

    tool_call_count: Callable[[list[AnyMessage], str], int]

    incomplete_error_type: type[Exception]

    nudge_max: int

    design_max_search_rounds: int

    evaluation_terminal_reserve_seconds: float

    design_state_changing_tools: frozenset[str]

    design_retrieval_tools: frozenset[str]

    design_max_pre_screen_rounds: int = 12





def _decision_from_state(state: Mapping[str, Any]) -> DecisionState | None:

    raw = state.get("design_decision")

    if isinstance(raw, DecisionState):

        return raw

    if raw:

        return DecisionState.model_validate(raw)

    return None





def _hold_from_search(decision: DecisionState, reason: str) -> DecisionState:

    return decision.model_copy(

        update={

            "status": "hold",

            "approved_candidate_ids": [],

            "next_queries": [],

            "reason_codes": [*decision.reason_codes, reason],

        }

    )





def handle_no_tool_output(

    state: Mapping[str, Any],

    messages: list[AnyMessage],

    ctx: ScientificReviewContext,

) -> Command | None:

    """Handle a no-tool answer while a design decision still requires evidence."""



    decision = _decision_from_state(state)

    if decision is None or decision.status != "search":

        return None



    nudge_count = int(state.get("nudge_count") or 0)

    if nudge_count < ctx.nudge_max:

        reminder = internal_human_message(

            content=(

                "（系统提醒）结构化设计门禁仍为 search，当前证据不足以推荐。"

                "请按 evidence_gaps 继续检索或更换候选，并重新调用 "

                "screen_formulations；不得直接结束或生成最终报告。"

            )

        )

        return Command(

            update={"messages": [reminder], "nudge_count": nudge_count + 1},

            goto="llm",

        )



    hold = _hold_from_search(decision, "agent_replan_limit_reached")

    reminder = internal_human_message(

        content=(

            "（系统提醒）有界重规划次数已用尽。结构化设计门禁已转为 hold。"

            "请立即调用 `synthesize_design_report` 工具输出证据不足的暂停结论、"

            "未解决缺口和后续验证计划；不得推荐任何候选，也不得继续检索。"

        )

    )

    return Command(

        update={

            "messages": [reminder],

            "design_decision": hold,

            "nudge_count": 0,

        },

        goto="llm",

    )





def handle_tool_calls_early(

    state: Mapping[str, Any],

    messages: list[AnyMessage],

    last_msg: AIMessage,

    ctx: ScientificReviewContext,

) -> Command | None:

    """Bound evidence wandering before the first formulation-space call."""



    task_type = ctx.active_task_type(state, messages)

    successful_tools = ctx.successful_tool_names(messages)

    if (

        task_type not in {"design", "inverse_design"}

        or "query_formulation_space" in successful_tools

        or not ctx.all_retrieval_only(list(last_msg.tool_calls))

    ):

        return None



    completed_rounds = int(state.get("design_search_rounds") or 0)

    # 形成候选前也需要分别寻找组成、方法和性能，不使用筛选后的补证预算。

    max_search_rounds = ctx.design_max_pre_screen_rounds

    repeated_batch = (

        ctx.prior_equivalent_successful_batch_count(

            messages[:-1], list(last_msg.tool_calls)

        )

        > 0

    )

    remaining = ctx.evaluation_remaining_seconds(state)

    reserve_reached = (

        remaining is not None

        and remaining <= ctx.evaluation_terminal_reserve_seconds

    )

    if not (

        repeated_batch

        or completed_rounds + 1 > max_search_rounds

        or reserve_reached

    ):

        return None



    nudge_count = int(state.get("nudge_count") or 0)

    if nudge_count >= ctx.nudge_max:

        raise ctx.incomplete_error_type(

            "pre-formulation evidence budget exhausted without an agent-selected candidate"

        )

    reminder = internal_human_message(content=(

        "（系统提醒）形成候选前的检索已达到预算或重复了相同查询。"

        "请根据已返回原文，自行选择与用户条件相关的候选并调用 query_formulation_space。"

        "保留真实的配方出处，不添加未经检索的组分或浓度。"

        "若没有足够组成证据，请说明缺口；系统不会替你填写候选参数。"

    ))

    return Command(

        update={

            "messages": [RemoveMessage(id=last_msg.id), reminder],

            "design_search_rounds": completed_rounds,

            "nudge_count": nudge_count + 1,

        },

        goto="llm",

    )





def handle_tool_calls_late(

    state: Mapping[str, Any],

    messages: list[AnyMessage],

    last_msg: AIMessage,

    ctx: ScientificReviewContext,

) -> Command | None:

    """Apply design-specific pre-screen, search/hold, and terminal guards."""



    decision = _decision_from_state(state)

    task_type = ctx.active_task_type(state, messages)

    successful_tools = ctx.successful_tool_names(messages)

    pending_report = any(

        ctx.tool_name(call) == "synthesize_design_report"

        for call in last_msg.tool_calls

    )



    # After formulation exists, bound retrieval/enrichment before the first

    # deterministic screen. This policy owns scientific convergence, not the

    # generic review node.

    if (

        decision is None

        and task_type in {"design", "inverse_design"}

        and "query_formulation_space" in successful_tools

        and "screen_formulations" not in successful_tools

        and not any(

            ctx.tool_name(call) in {"screen_formulations", "synthesize_design_report"}

            for call in last_msg.tool_calls

        )

        and (

            ctx.all_retrieval_only(list(last_msg.tool_calls))

            or any(

                ctx.tool_name(call)

                in {

                    "query_formulation_space",

                    "enrich_formulation_evidence",

                    "predict_formulation_conductivity",

                }

                for call in last_msg.tool_calls

            )

        )

    ):

        completed_rounds = int(state.get("design_search_rounds") or 0)

        # 初次整理配方、测试方法和性能证据需要不同检索。与已筛选后的

        # 缺口补证分开计数，避免一次宽检索后即替 Agent 决定筛选参数。

        max_search_rounds = ctx.design_max_pre_screen_rounds

        repeated_formulation = any(

            ctx.tool_name(call) == "query_formulation_space"

            for call in last_msg.tool_calls

        ) and ctx.prior_equivalent_successful_batch_count(messages[:-1], list(last_msg.tool_calls)) >= 1

        repeated_enrichment = any(

            ctx.tool_name(call) == "enrich_formulation_evidence"

            for call in last_msg.tool_calls

        ) and ctx.prior_equivalent_successful_batch_count(messages[:-1], list(last_msg.tool_calls)) >= 1

        if (

            repeated_formulation

            or repeated_enrichment

            or completed_rounds + 1 > max_search_rounds

        ):

            nudge_count = int(state.get("nudge_count") or 0)

            if nudge_count >= ctx.nudge_max:

                raise ctx.incomplete_error_type(

                    "pre-screen retrieval budget exhausted but the observed "

                    "formulation request did not contain deterministic screening "

                    "conditions"

                )

            reminder = internal_human_message(

                content=(

                    "(System control) The bounded pre-screen retrieval budget is "

                    "exhausted. Stop retrieval and call screen_formulations now "

                    "with the observed candidates, structured constraint_records derived "

                    "from the original question, and any "

                    "joint-context requirement already derived for this task."

                )

            )

            return Command(

                update={

                    "messages": [RemoveMessage(id=last_msg.id), reminder],

                    "design_search_rounds": completed_rounds,

                    "nudge_count": nudge_count + 1,

                },

                goto="llm",

            )



    if decision is not None and decision.status == "search" and pending_report:

        nudge_count = int(state.get("nudge_count") or 0)

        if nudge_count >= ctx.nudge_max:

            hold = _hold_from_search(decision, "agent_replan_limit_reached")

            reminder = internal_human_message(

                content=(

                    "（系统提醒）有界重规划次数已用尽，门禁已转为 hold。"

                    "请基于 hold 状态重新生成不含候选推荐的报告。"

                )

            )

            update = {

                "messages": [RemoveMessage(id=last_msg.id), reminder],

                "design_decision": hold,

                "nudge_count": 0,

            }

        else:

            reminder = internal_human_message(

                content=(

                    "（系统提醒）结构化设计门禁仍为 search，本次不执行 "

                    "synthesize_design_report。请先根据 evidence_gaps 继续检索或"

                    "更换候选，并重新调用 screen_formulations；重新筛选必须复用"

                    "同一份 constraint_records 与已声明的 joint_context_requirements。"

                )

            )

            update = {

                "messages": [RemoveMessage(id=last_msg.id), reminder],

                "nudge_count": nudge_count + 1,

            }

        return Command(update=update, goto="llm")



    if (

        decision is not None

        and decision.status == "search"

        and ctx.all_retrieval_only(list(last_msg.tool_calls))

    ):

        search_rounds = int(state.get("design_search_rounds") or 0) + 1

        max_search_rounds = int(

            state.get("design_max_search_rounds") or ctx.design_max_search_rounds

        )

        if search_rounds > max_search_rounds:

            attempts = max(

                int(state.get("design_attempts") or 0),

                ctx.tool_call_count(messages, "screen_formulations"),

            )

            max_attempts = int(state.get("design_max_attempts") or 2)

            if attempts < max_attempts:

                reminder = internal_human_message(

                    content=(

                        "（系统提醒）补证检索轮次已用尽。请立即停止检索，"

                        "用当前候选和原有 constraint_records / joint_context_requirements "

                        "重新调用 screen_formulations 评估；若继续检索将无法收敛。"

                    )

                )

                return Command(

                    update={

                        "messages": [RemoveMessage(id=last_msg.id), reminder],

                        "design_search_rounds": 0,

                        "nudge_count": 0,

                    },

                    goto="llm",

                )

            hold = _hold_from_search(decision, "agent_search_round_limit_reached")

            reminder = internal_human_message(

                content=(

                    "（系统提醒）补证检索轮次和重规划次数均已用尽，"

                    "门禁已强制转为 hold。请立即调用 synthesize_design_report "

                    "输出 hold 结论、证据缺口和后续验证计划，不得继续检索。"

                )

            )

            return Command(

                update={

                    "messages": [RemoveMessage(id=last_msg.id), reminder],

                    "design_decision": hold,

                    "design_search_rounds": 0,

                    "nudge_count": 0,

                },

                goto="llm",

            )



    nonterminal_design_tools = (

        ctx.design_retrieval_tools

        | (ctx.design_state_changing_tools - {"synthesize_design_report"})

    )

    if (

        decision is not None

        and decision.status == "hold"

        and any(

            ctx.tool_name(call) in nonterminal_design_tools

            for call in last_msg.tool_calls

        )

    ):

        reminder = internal_human_message(

            content=(

                "（系统提醒）门禁已是 hold，不允许继续检索、生成候选或重新筛选。"

                "请立即调用 synthesize_design_report 输出 hold 报告。"

            )

        )

        return Command(

            update={

                "messages": [RemoveMessage(id=last_msg.id), reminder],

                "design_search_rounds": 0,

                "nudge_count": 0,

            },

            goto="llm",

        )



    return None





def approved_call_state_updates(

    state: Mapping[str, Any],

    messages: list[AnyMessage],

    approved_calls: Sequence[dict],

    ctx: ScientificReviewContext,

) -> dict[str, Any]:

    """Return design-policy counters after generic approval has filtered calls."""



    decision = _decision_from_state(state)

    task_type = ctx.active_task_type(state, messages)

    rounds = int(state.get("design_search_rounds") or 0)

    design_flow_active = decision is not None or task_type in {"design", "inverse_design"}

    approved = list(approved_calls)

    if design_flow_active and ctx.all_retrieval_only(approved):

        rounds += 1

    if any(

        ctx.tool_name(call) in ctx.design_state_changing_tools for call in approved

    ):

        rounds = 0

    return {

        "design_search_rounds": rounds,

        "design_max_search_rounds": int(

            state.get("design_max_search_rounds") or ctx.design_max_search_rounds

        ),

    }

