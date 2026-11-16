import operator

from typing import Annotated, NotRequired, TypedDict



from langchain_core.messages import AnyMessage

from langgraph.graph import add_messages



from orion.core.design_decision import (

    CandidateEvaluation,

    CandidateState,

    DecisionState,

    EvidenceLedgerItem,

    TaskContract,

)

from orion.tools.todo_tools import TodoItem

from orion.task_evidence.evidence_applicability import ApplicabilityRecord





class GraphState(TypedDict):

    messages: Annotated[list[AnyMessage], add_messages]

    approved_tools: NotRequired[dict[str, str]]

    todo_list: NotRequired[Annotated[list[TodoItem], operator.add]]

    session_id: NotRequired[str]

    # 当前真实用户输入的消息 ID；内部提醒与审批恢复不改变该边界。

    task_start_message_id: NotRequired[str | None]

    # 仅保存本回合明确创建、认领或声明意图的任务卡；恢复使用checkpoint中的绑定。

    runtime_board_task_ids: NotRequired[list[str]]

    # 协议恢复与重复操作反馈计数（防止死循环），不表示科研结论是否充分。

    nudge_count: NotRequired[int]

    # Explicit request-level transport resilience. These fields count retries

    # of an individual model request only; tools and whole episodes are never

    # replayed by this mechanism.

    llm_request_max_retries: NotRequired[int]

    llm_transport_retry_count: NotRequired[int]

    # 答案接地校验的修订计数（每个用户回合在 context_prepare 重置），

    # 见 orion.core.grounding

    grounding_attempts: NotRequired[int]

    # grounding 外部裁判的运行状态。生产侧允许 fail-open，但评估侧必须

    # 区分 completed/failed，避免把裁判连接失败的样本计为实验成功。

    grounding_status: NotRequired[str]

    grounding_error: NotRequired[str | None]

    # grounding 节点自身的墙钟耗时（秒）。用于区分 Agent 总耗时与外部裁判耗时，

    # 并为冻结 benchmark 的交互/离线预算策略提供可审计证据。

    grounding_duration_seconds: NotRequired[float | None]

    # 评测样本的墙钟边界。context_prepare 从 RunnableConfig 固化到 GraphState，

    # grounding 优先读取这里，避免同步节点执行时自定义 configurable 丢失。

    evaluation_started_at_monotonic: NotRequired[float | None]

    evaluation_timeout_seconds: NotRequired[float | None]

    # 首选 grounding 模型失败后是否切换到 fallback，以及首选失败的安全摘要。

    grounding_fallback_used: NotRequired[bool]

    grounding_primary_error: NotRequired[str | None]

    # Claim-Verifiable 路径的结构化校验摘要。contract-first grounding 将语义

    # verified/total 写入此处，evaluation 以它作为结构化任务的 authoritative CSR。

    claim_contract_summary: NotRequired[dict | None]

    # 主 Agent 直接形成的最终回答，或旧终端工具提交的正文。

    last_report: NotRequired[str]

    # 与 last_report 同源的可机读 TaskArtifact。评测、Workbench 和离线 replay

    # 必须读取该字段，而不是从 Markdown 反推结构化终态。

    task_artifact: NotRequired[dict | None]

    # claim-evidence-target 的条件适用性状态。终端编译器写入该字段，供评测、

    # Workbench 与 replay 直接检查状态→动作，而不是从自然语言报告反推。

    applicability_records: NotRequired[list[ApplicabilityRecord] | None]

    # Agent 最终回答或旧工具终端结果已提交；不表示经过独立语义验证。

    terminal_committed: NotRequired[bool]

    terminal_commit_tool_call_id: NotRequired[str | None]

    # Canonical task type supplied by an external task contract. Unlike

    # ``design_task_type``, this applies to all four task families and must not

    # be overwritten by question-keyword classification at terminalization.

    configured_task_type: NotRequired[str | None]

    # 电解液设计的结构化决策状态。Phase 2 由 ToolNode 兼容收集层填充，

    # 后续工具原生输出状态后可移除消息适配。

    design_task_type: NotRequired[str | None]

    design_original_question: NotRequired[str | None]

    design_semantic_results: NotRequired[list[dict] | None]

    design_semantic_sources: NotRequired[dict[str, str] | None]

    design_semantic_revision: NotRequired[int]

    task_contract: NotRequired[TaskContract | None]

    evidence_ledger: NotRequired[list[EvidenceLedgerItem] | None]

    design_candidates: NotRequired[list[CandidateState] | None]

    candidate_evaluations: NotRequired[list[CandidateEvaluation] | None]

    design_decision: NotRequired[DecisionState | None]

    design_attempts: NotRequired[int]

    design_max_attempts: NotRequired[int]

    design_state_revision: NotRequired[int]

    design_decision_revision: NotRequired[int]

    # 在 design_decision == search 时，记录连续纯检索批次的次数，防止 Agent

    # 在证据缺口前无限调用检索工具而不回到 screen_formulations / synthesize。

    design_search_rounds: NotRequired[int]

    design_max_search_rounds: NotRequired[int]

    # Literature screening/synthesis tasks can keep emitting retrieval calls and

    # never reach the no-tool workflow gate.  Bound those evidence-only batches

    # independently from the design decision search controller.

    workflow_evidence_rounds: NotRequired[int]

    workflow_max_evidence_rounds: NotRequired[int]

    # 上一次放行的工具名（排序去重）。用于判断 agent 是否在"原地重复同一工具"：

    # 重复调用不视为实质进展，不清零 nudge_count，从而让 review 的 _NUDGE_MAX

    # 上限能真正触发 fail-closed —— 否则"提醒 → 同一工具失败 → 放行即清零 →

    # 再提醒"会形成无限重试（实测一次 run 内 synthesize_design_report 被调 24 次）。

    last_approved_tools: NotRequired[list[str] | None]

