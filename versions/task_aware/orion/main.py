import asyncio
import inspect
import os
import signal
import sqlite3
import sys
import json
import time
import types
from pathlib import Path
from typing import Any, Literal, Mapping

# Ensure the repository can be imported both as a package and when main.py is
# executed directly. Register ``orion`` as a namespace alias pointing at
# the worktree root so that ``from orion.core.xxx`` imports resolve.
_DEEPAGENTS_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEEPAGENTS_ROOT.parent
for _path in (_DEEPAGENTS_ROOT, _PROJECT_ROOT):
    s = str(_path)
    if s not in sys.path:
        sys.path.insert(0, s)
if "orion" not in sys.modules:
    _deepagents_pkg = types.ModuleType("orion")
    _deepagents_pkg.__path__ = [str(_DEEPAGENTS_ROOT)]
    sys.modules["orion"] = _deepagents_pkg

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
import httpx
import yaml
from openai import APIConnectionError

from uuid import uuid4

from orion.core.memory.service import MemoryService
from orion.core.memory.short_term import ShortTermMemory
from orion.core.context.manager import ContextManager
from orion.core.design_decision import (
    CandidateEvaluation,
    CandidateState,
    ConstraintEvaluation,
    ConstraintSpec,
    DecisionState,
    EvidenceGap,
    EvidenceLedgerItem,
    TaskContract,
)
from orion.core.design_runtime import (
    capture_design_tool_batch,
    terminal_result,
    scientific_design_policy as scientific_design_post_tool_policy,
)
from orion.core.harness_policy import apply_post_tool_policy
from orion.core.langgraph_residency import ToolCallResidencyBridge
from orion.core.scientific_review_policy import (
    ScientificReviewContext,
    approved_call_state_updates,
    handle_no_tool_output,
    handle_tool_calls_early,
    handle_tool_calls_late,
)
from orion.core.grounding import make_grounding_node

# Historical replay scripts monkeypatch ``main.design_decision_node``. Keep that
# name as a compatibility handle, while the main graph treats it as a post-tool
# domain policy rather than a permanent LangGraph node.
design_decision_node = scientific_design_post_tool_policy
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig, interrupt, Command
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.styles import Style
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError
except ModuleNotFoundError:
    PromptSession = None
    HTML = None
    Style = None
    NoConsoleScreenBufferError = Exception

from orion.core.task_scope import (current_task_messages, current_task_state,
                                  internal_human_message, task_entry_updates)
from orion.core.state import GraphState
from orion.core.evidence_extract import parse_chunk_payload
from orion.task_evidence.contracts_v2 import (
    expected_skill_for_task_type,
    task_type_for_skill,
)
from llms import main_model, main_model_nonstream, get_grounding_model
from tools.config import main_tools
from orion.core.default_resource_runtime import build_default_tool_resource_runtime
from tools.local_capabilities import (
    attach_local_capability,
    detach_local_capability,
    get_local_capability_manager,
    local_capability_snapshot,
    scan_local_capabilities,
)
from tools.todo_tools import TodoItem
from tools.skill_loader import (
    format_skill_index_for_system,
    get_skill_workflow,
    load_skill as _load_skill_file,
)
from orion.core.context.compact import compact_messages, estimate_token_count
from orion.evolution.assets.prompt_asset import load_prompt_asset
from orion.evolution.assets.skill_asset import load_skill_asset
from orion.evolution.assets.storage import FileSystemAssetStorage
from orion.core.memory.wiki_reflection import reflect_and_update
from orion.core.hooks import (hookify, register_hook, collect_hooks,
                              _merge_updates, _state_with_update)
from orion.core.grounding import deliver_terminal_result
from orion.workbench.api.app.audit import AuditLogger
_DEEPAGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_DEEPAGENTS_DIR, "model", "config.json"), "r", encoding="utf-8") as f:
    CONTEXT_LIMIT = json.load(f).get("context_windows_limit")

serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        TodoItem,
        ConstraintSpec,
        TaskContract,
        EvidenceLedgerItem,
        EvidenceGap,
        CandidateState,
        ConstraintEvaluation,
        CandidateEvaluation,
        DecisionState,
    ]
)

# Windows 后台进程默认 stdout 可能是 GBK，遇到 emoji 会抛 UnicodeEncodeError。
# 强制改成 UTF-8 并用替换字符兜底，避免打印调试信息时中断整个对话。
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

tools = main_tools
# 保留 tools.config 导出的原始工具集。评测消融必须从这份实际绑定给 LLM 的
# 工具清单裁剪，不能只设置环境变量后继续复用完整工具集。
_BASE_TOOLS = tuple(main_tools)

# Post-selection runtime bridge. The production LangGraph topology and tool selection
# remain unchanged: static bindings only materialize resources for tools already chosen
# by the Agent. No TTL or learned policy is enabled here.
_default_resource_runtime = build_default_tool_resource_runtime()
_tool_residency_bridge: ToolCallResidencyBridge | None = (
    _default_resource_runtime.bridge if _default_resource_runtime is not None else None
)


def set_tool_residency_bridge(bridge: ToolCallResidencyBridge | None) -> None:
    """Install or remove runtime residency management at the stable ToolNode seam."""

    global _tool_residency_bridge
    _tool_residency_bridge = bridge


_evolution_store_default = Path(__file__).parent / "evolution" / "store"


def _evolution_storage() -> FileSystemAssetStorage:
    root = Path(os.environ.get("ORION_EVOLUTION_STORE", _evolution_store_default))
    return FileSystemAssetStorage(root_dir=root)


def _tool_name(tc: dict) -> str:
    """兼容 LangChain 不同版本的 tool_call name 字段位置。"""
    return tc.get("name") or tc.get("function", {}).get("name", "")


def _find_terminal_report(messages: list) -> str:
    """兼容入口：正文与 artifact 共享当前任务中的同一条终端结果。"""
    result = terminal_result({"messages": messages})
    return result[0] if result else ""


def _find_terminal_artifact(messages: list) -> dict[str, Any] | None:
    """兼容入口：不能从另一条报告消息借用 artifact。"""
    result = terminal_result({"messages": messages})
    return result[1] if result else None


def load_skill_from_store(skill_name: str) -> str | None:
    storage = _evolution_storage()
    asset = load_skill_asset(f"{skill_name}.skill", storage)
    if asset:
        return asset.active_content
    return None


def load_prompt_from_store(prompt_id: str) -> str | None:
    """从 evolution store 加载指定 prompt；不存在时返回 ``None``。"""
    storage = _evolution_storage()
    asset = load_prompt_asset(prompt_id, storage)
    if asset:
        return asset.active_content
    return None


def get_system_prompt() -> str:
    """返回运行时系统提示词；优先使用 evolution store 中的 ``prompts/system/main``，否则回退到默认构造。"""
    store_prompt = load_prompt_from_store("system/main")
    if store_prompt is not None:
        return store_prompt
    from orion.core.semantic_design_policy import semantic_mode_enabled
    if semantic_mode_enabled():
        return SYSTEM_PROMPT + "\n语义理解由主 Agent 结合原始问题与本轮原文完成；需要独立上下文或复核视角时显式委派子代理。结构化筛选评估已编码约束，不替代完整科学理解。已有语义审查记录仍须保留真实引文和实验关联；报告不能绕过现有证据准入。"
    return SYSTEM_PROMPT


def load_skill(name: str) -> str:
    """加载指定 skill 的完整内容；优先从 evolution store 读取，否则回退到 skills/ 目录。"""
    store_content = load_skill_from_store(name)
    if store_content is not None:
        return store_content
    return _load_skill_file(name)


_memory_service = MemoryService()
_context_manager = ContextManager(memory_service=_memory_service)

_skill_index = format_skill_index_for_system()
WORK_DIR = os.getcwd()
SYSTEM_PROMPT = f"""你是一个可使用多种工具完成任务的智能助手。

核心工作方式：
1. 理解用户目标，结合已有候选、证据和缺口选择行动。
2. 需要领域方法时调用 load_skill 加载 Skill，按实际任务选择能力；Skill 不是固定工具序列。
3. 使用检索工具查找事实时，在回答中用 [n] 标注来源，n 为检索结果中的序号。
4. 如果工具需要用户才能确定的参数而用户没有给出，必须先询问用户，严禁自行假设。
5. 数值比较需要可核查的数值、单位和测试条件；缺少结构化数值时可使用 extract_numeric_properties，不为形式完整重复调用。

{_skill_index}

当前所在目录：{WORK_DIR}

检索与证据纪律（严禁违反）：
- 科学事实必须有真实来源；先检查本轮用户资料和已返回证据，按缺口选择文献/KG检索。用户提供的数据、文献实测和模型预测须区分，不能伪造来源。
- 指定论文且身份不明确时，可先解析 paper_id 再读取原文；已有可靠 paper_id 和相关原文时复用证据。不得因作者名未命中就断言论文未收录。
- 对指定论文的结论必须绑定目标论文；需要跨论文比较时明确区分来源，不能将不同候选或实验拼接成验证结果。
- 工具结果返回后自行理解证据并选择下一步。复杂、篇幅较大或需要独立视角的核验可显式委派子代理；返回结果须带来源和限制，最终由主 Agent 综合。

Skill 与交付纪律：
- 当 skill 指引要求特定引用格式（如 chunk_id）时，严格按 skill 格式输出，不要自行改为 [n] 编号。
- 你负责判断证据、选择行动并直接撰写最终回答；不要求调用报告工具或采用固定章节。按用户问题交代结论、真实来源、测试条件与尚未解决的限制，不把未执行的工作写成已完成。
- 检索、计算、筛选和规划均按实际缺口选择；不预设适用于所有科研任务的顺序。工具中的确定性检查不能替代原文语义理解；需要独立核验时显式委派子代理。
- 证据不足时可以明确说明尚不能推荐，保留已有配方线索和具体缺口后结束；不能把推测、域外计算或不同配方/实验的性能拼接成已验证结果。
- `synthesize_design_report` 是历史结构化模板兼容工具，普通回答不需要调用。显式使用时仍须满足其原有证据合同；失败反馈不代表整个科研任务必须失败。已有成功的旧协议终端提交继续按原样成对交付。
"""
PROMPT_STYLE = Style.from_dict({
    "prompt": "bold ansicyan",
    "hint": "ansibrightblack",
}) if Style is not None else None


llm_with_tools = main_model.bind_tools(tools)
llm_with_tools_nonstream = main_model_nonstream.bind_tools(tools)


def _llm_stream_enabled(config: RunnableConfig | None) -> bool:
    """是否使用流式调用。可视化场景（Web UI/CLI）默认流式；E2E 测试等无
    可视化场景通过 config.configurable.llm_stream=False 或环境变量
    ORION_LLM_STREAM=0 切换为非流式，避免流式停滞问题干扰测试。

    v68: deepseek-v4-flash 的流式 tool_call chunk 存在空 name/空 id 的畸形
    输出，会导致后续请求 400；因此当环境变量未显式设置且主模型为
    deepseek-v4-flash 时，默认关闭流式（可通过显式设置 ORION_LLM_STREAM=1
    强制开启）。"""
    if config and isinstance(config.get("configurable"), dict):
        value = config["configurable"].get("llm_stream")
        if value is not None:
            return bool(value)

    env_value = os.environ.get("ORION_LLM_STREAM")
    if env_value is not None:
        return env_value.lower() not in ("0", "false", "no")

    # 未显式设置时，对已知流式 tool_call 有问题的模型默认走非流式
    if getattr(main_model, "model_name", "") == "deepseek-v4-flash":
        return False

    return True


_LLM_REQUEST_MAX_RETRIES_DEFAULT = 2
_LLM_REQUEST_MAX_RETRIES_CAP = 5


def _llm_request_max_retries(config: RunnableConfig | None) -> int:
    """Resolve the explicit request-level retry limit.

    A retry repeats only the current model HTTP request. It never replays a
    completed tool call or restarts a LangGraph episode. Formal evaluations can
    freeze the value through ``configurable.llm_request_max_retries``; normal
    runtime uses ``ORION_LLM_REQUEST_MAX_RETRIES`` (default: 2).
    """

    configurable = config.get("configurable", {}) if config else {}
    raw_value = (
        configurable.get("llm_request_max_retries")
        if isinstance(configurable, dict)
        and configurable.get("llm_request_max_retries") is not None
        else os.environ.get(
            "ORION_LLM_REQUEST_MAX_RETRIES",
            str(_LLM_REQUEST_MAX_RETRIES_DEFAULT),
        )
    )
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("llm_request_max_retries must be an integer") from exc
    if value < 0 or value > _LLM_REQUEST_MAX_RETRIES_CAP:
        raise ValueError(
            "llm_request_max_retries must be between 0 and "
            f"{_LLM_REQUEST_MAX_RETRIES_CAP}"
        )
    return value


def _is_retryable_llm_transport_error(exc: BaseException) -> bool:
    """Recognize connection, timeout, and incomplete-response failures only."""

    current: BaseException | None = exc
    seen: set[int] = set()
    retryable_names = {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionResetError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutError",
        "WriteError",
        "WriteTimeout",
    }
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (APIConnectionError, httpx.TransportError, asyncio.TimeoutError),
        ):
            return True
        if type(current).__name__ in retryable_names:
            return True
        current = current.__cause__ or current.__context__
    return False


# Thread-local runtime bookkeeping for values that should not pollute GraphState.
_thread_runtime: dict[str, dict] = {}


def _get_thread_id(config: RunnableConfig | None) -> str:
    if config is None:
        return "default"
    configurable = config.get("configurable")
    if isinstance(configurable, dict):
        return configurable.get("thread_id", "default")
    return "default"


def _evaluation_timing_from_config(
    config: RunnableConfig | None,
) -> dict[str, float | None]:
    """把评测墙钟边界固化进 GraphState，供同步 grounding 节点可靠读取。"""

    configurable = config.get("configurable") if config else None
    if not isinstance(configurable, dict):
        return {
            "evaluation_started_at_monotonic": None,
            "evaluation_timeout_seconds": None,
        }
    started = configurable.get("evaluation_started_at_monotonic")
    timeout = configurable.get("evaluation_timeout_seconds")
    return {
        "evaluation_started_at_monotonic": (
            float(started) if started is not None else None
        ),
        "evaluation_timeout_seconds": (
            float(timeout) if timeout is not None else None
        ),
    }


def _task_board_message() -> SystemMessage | None:
    """注入通用任务约定，并在有未完成工作时附上恢复摘要。

    任务状态落在 ``.tasks/`` 而不是对话历史里，正是为了跨上下文窗口恢复：
    压缩会把"做到第几步"抹掉，但文件不会。无卡时也保留领域无关的使用约定，
    不要求简单工作建卡，不为其他组件强制登记计划。
    """
    if not any(getattr(t, 'name', '') == 'board_create' for t in tools):
        return None
    try:
        from orion.core.tasks import get_board, TaskStatus

        board = get_board()
        active = [t for t in board.list() if t.status is not TaskStatus.COMPLETED]
        return SystemMessage(
            id="orion:task-board-context",
            additional_kwargs={"orion_context": "task_board"},
            content=(
                "## 任务管理\n"
                "需要跟踪多个阶段、依赖或跨上下文继续时，用任务卡管理工作；简单任务可直接完成。"
                "board_create 记录目标、验收条件和依赖，board_claim 认领。"
                "按实际进展更新已完成事项、剩余工作和阻塞；计划不明确时允许保留未知。"
                "备注可随建卡或认领一起提交，避免重复登记；仅在计划变化时另行更新。"
                "board_complete 提交结果、evidence_refs 与 checks；卡住用 board_escalate 记录具体待决问题。\n\n"
                + (board.summary() if active else "当前没有未完成任务卡。")
            )
        )
    except Exception:
        # 任务板不可用（目录权限、旧版本状态）不能阻断主流程
        return None


def _typed_task_contract_message(
    config: RunnableConfig | None,
) -> SystemMessage | None:
    """Render authoritative routing metadata supplied by an external contract.

    No question text is classified here. When ``task_type`` is absent Orion
    keeps its normal dynamic Skill selection behavior.
    """

    configurable = config.get("configurable", {}) if config else {}
    if not isinstance(configurable, dict):
        return None
    task_type = configurable.get("task_type")
    expected_skill = expected_skill_for_task_type(task_type) if task_type else None
    if expected_skill is None:
        return None
    return SystemMessage(
        content=(
            "## Structured task contract (authoritative)\n"
            f"- task_type: `{task_type}`\n"
            f"- applicable Skill: `{expected_skill}`\n"
            f"When domain guidance is needed, use `load_skill(name=\"{expected_skill}\")`; "
            "reuse guidance and evidence already available in this task. This contract does not prescribe "
            "a scientific tool sequence. Preserve the supplied task_type. "
            "Auxiliary Skills may be loaded as needed without changing this task contract. "
            "The load_skill boundary will reject a mismatched canonical task Skill and ask you to correct it."
        )
    )


def _configured_task_type(config: RunnableConfig | None) -> str:
    configurable = config.get("configurable", {}) if config else {}
    if not isinstance(configurable, dict):
        return "design"
    return str(configurable.get("task_type") or "design")


def _explicit_task_type_from_config(config: RunnableConfig | None) -> str | None:
    """Return only an externally supplied canonical task contract type."""

    configurable = config.get("configurable", {}) if config else {}
    if not isinstance(configurable, dict):
        return None
    task_type = configurable.get("task_type")
    return str(task_type) if expected_skill_for_task_type(task_type) else None


def _inject_task_contract(state: GraphState, config: RunnableConfig | None = None):
    """hook：回合开始注入结构化任务契约（仅当配置了 task_type）。"""
    return _typed_task_contract_message(config)


def _inject_task_board(state: GraphState, config: RunnableConfig | None = None):
    """hook：回合开始注入未完成的任务板。"""
    return _task_board_message()


# 回合开始的"上下文注入"是横切行为，按博客哲学抽成可注册 hook：
# 想再加一种注入，只需 register_hook("inject_context", ...)，不改本节点。
# 模块级 import 只执行一次，故不会重复注册。
register_hook("inject_context", _inject_task_contract)
register_hook("inject_context", _inject_task_board)


def _refresh_task_board_context(state: GraphState, config: RunnableConfig | None = None):
    """逐轮刷新模型输入视图，移除旧任务摘要，保持工具调用与返回的原顺序。"""
    def is_board_message(message):
        return isinstance(message, SystemMessage) and (
            message.id == "orion:task-board-context"
            or message.additional_kwargs.get("orion_context") == "task_board"
            or isinstance(message.content, str) and message.content.startswith((
                "## 任务管理\n", "## 任务板（持久化，跨会话有效）\n")))
    messages = [message for message in state["messages"] if not is_board_message(message)]
    current = _task_board_message()
    if current is None and len(messages) == len(state["messages"]):
        return None
    if current is not None:
        messages.insert(1 if messages and isinstance(messages[0], SystemMessage) else 0, current)
    return {"messages": messages}


def _compact_before_llm(
    state: GraphState, config: RunnableConfig | None = None
) -> dict | None:
    """before_llm hook：每轮 LLM 调用前检查上下文是否接近上限，超限则压缩历史。

    对齐博客 s08 的层级——压缩检查放在"每轮 LLM 调用前"而非回合入口：上下文
    是每轮都在增长（工具循环里每读一个文件就涨一段），只有逐轮检查才能在任意
    一轮调用前保证不超限。用 estimate_token_count 按当前 messages 估算 token
    （而非读上一轮调用的 total_tokens），与博客 L4 的判断方式一致。

    返回语义：None 放行；dict 作为 state 更新，整体替换 messages 为压缩后的
    "prefix + summary + suffix" 列表（hook 层是 dict merge 替换语义，正好对应
    压缩要"重写消息列表"的需求；被折叠的中间消息这一轮即从模型输入中移除）。
    """
    token_count = estimate_token_count(state["messages"])
    if token_count < CONTEXT_LIMIT * 0.8:
        return None
    try:
        new_messages, _new_count = compact_messages(
            state["messages"], token_count, CONTEXT_LIMIT
        )
        return {"messages": new_messages}
    except Exception as exc:
        print(f"[context_manager] compaction failed: {exc}", flush=True)
        return None


def _check_execution_deadline(state: GraphState, config: RunnableConfig | None = None):
    """执行预算不改变科学评估，也不触发领域工具调用。"""
    remaining = _evaluation_remaining_seconds(state)
    if remaining is not None and remaining <= 0:
        raise WorkflowIncompleteError("执行时间预算已耗尽；没有发起新的模型请求")


register_hook("before_llm", _check_execution_deadline)
register_hook("before_llm", _refresh_task_board_context)
register_hook("before_llm", _compact_before_llm)


async def context_prepare_async(state: GraphState, config: RunnableConfig | None = None):
    boundary = task_entry_updates(state)
    _thread_runtime.pop(_get_thread_id(config), None)
    session_id = state.get("session_id", "default")
    user_message = state["messages"][-1].content
    msgs = await _context_manager.assemble(
        session_id=session_id,
        user_message=user_message,
        system_prompt=get_system_prompt(),
        tools=tools,
        task_type=_configured_task_type(config),
    )
    # 回合开始的上下文注入（task 契约、任务板等）已抽成 inject_context hook。
    for extra in collect_hooks("inject_context", state, config):
        msgs.insert(1, extra)
    # Avoid duplicating the HumanMessage that is already in state["messages"];
    # the add_messages reducer would otherwise append a second copy on retries.
    if msgs and isinstance(msgs[-1], HumanMessage):
        msgs = msgs[:-1]
    # 新用户回合开始：重置接地校验修订计数，并清空上回合可能残留的终端报告。
    return {
        "messages": msgs,
        "grounding_attempts": 0,
        "grounding_status": "pending",
        "grounding_error": None,
        "grounding_duration_seconds": None,
        **_evaluation_timing_from_config(config),
        "grounding_fallback_used": False,
        "grounding_primary_error": None,
        "last_report": None,
        "task_artifact": None,
        "terminal_committed": False,
        "terminal_commit_tool_call_id": None,
        "configured_task_type": _explicit_task_type_from_config(config),
        "design_task_type": None,
        "design_original_question": str(state["messages"][-1].content),
        "design_semantic_results": None,
        "design_semantic_sources": None,
        "design_semantic_revision": -1,
        "task_contract": None,
        "evidence_ledger": None,
        "design_candidates": None,
        "candidate_evaluations": None,
        "design_decision": None,
        "design_attempts": 0,
        "design_max_attempts": 2,
        "design_state_revision": 0,
        "design_decision_revision": -1,
        # 新回合重置：工具轨迹从零开始，"换工具"才能清零 nudge。
        "last_approved_tools": None,
        **boundary,
    }


def context_prepare_sync(state: GraphState, config: RunnableConfig | None = None):
    boundary = task_entry_updates(state)
    _thread_runtime.pop(_get_thread_id(config), None)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    async def _prepare():
        session_id = state.get("session_id", "default")
        user_message = state["messages"][-1].content
        return await _context_manager.assemble(
            session_id=session_id,
            user_message=user_message,
            system_prompt=get_system_prompt(),
            tools=tools,
            task_type=_configured_task_type(config),
        )

    if loop is None:
        msgs = asyncio.run(_prepare())
    else:
        # 在已有事件循环的上下文（如某些测试）中，用线程池执行避免嵌套 loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            msgs = pool.submit(asyncio.run, _prepare()).result()
    # Avoid duplicating the HumanMessage that is already in state["messages"].
    if msgs and isinstance(msgs[-1], HumanMessage):
        msgs = msgs[:-1]
    # 回合开始的上下文注入（task 契约、任务板等）已抽成 inject_context hook。
    for extra in collect_hooks("inject_context", state, config):
        msgs.insert(1, extra)
    # 新用户回合开始：重置接地校验修订计数，并清空上回合可能残留的终端报告。
    return {
        "messages": msgs,
        "grounding_attempts": 0,
        "grounding_status": "pending",
        "grounding_error": None,
        "grounding_duration_seconds": None,
        **_evaluation_timing_from_config(config),
        "grounding_fallback_used": False,
        "grounding_primary_error": None,
        "last_report": None,
        "task_artifact": None,
        "terminal_committed": False,
        "terminal_commit_tool_call_id": None,
        "configured_task_type": _explicit_task_type_from_config(config),
        "design_task_type": None,
        "design_original_question": str(state["messages"][-1].content),
        "design_semantic_results": None,
        "design_semantic_sources": None,
        "design_semantic_revision": -1,
        "task_contract": None,
        "evidence_ledger": None,
        "design_candidates": None,
        "candidate_evaluations": None,
        "design_decision": None,
        "design_attempts": 0,
        "design_max_attempts": 2,
        "design_state_revision": 0,
        "design_decision_revision": -1,
        # 新回合重置：工具轨迹从零开始，"换工具"才能清零 nudge。
        "last_approved_tools": None,
        **boundary,
    }


def _to_json_safe(value: Any) -> Any:
    """将对象递归转为可 JSON 序列化的结构，未知类型退化为字符串。"""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return _to_json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return _to_json_safe(value.dict())
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return str(value)


def _messages_to_json(messages: list[AnyMessage]) -> list[dict]:
    out = []
    for m in messages:
        d = {"role": getattr(m, "type", "unknown"), "content": str(getattr(m, "content", ""))}
        if hasattr(m, "tool_calls") and m.tool_calls:
            d["tool_calls"] = _to_json_safe(m.tool_calls)
        out.append(d)
    return out


def _extract_tool_calls_for_reflection(messages: list[AnyMessage]) -> list[dict]:
    """把消息历史中的 ToolMessage 提取为反思可用的 (tool_name, output) 列表。"""
    tool_calls = []
    for m in messages:
        if isinstance(m, ToolMessage):
            name = getattr(m, "name", None) or ""
            content = getattr(m, "content", "")
            try:
                output = json.loads(content) if isinstance(content, str) else content
            except Exception:
                output = content
            tool_calls.append({"tool_name": name, "output": _to_json_safe(output)})
    return tool_calls


async def _auto_reflect(session_id: str, state: dict) -> None:
    """一次运行结束后自动触发 Wiki 反思与长期记忆写入。"""
    if os.environ.get("ORION_AUTO_REFLECT", "1") == "0":
        return
    messages = current_task_messages(state)
    if not messages:
        return
    run_id = f"run-{uuid4().hex[:12]}"
    try:
        await asyncio.to_thread(
            reflect_and_update,
            session_id=session_id,
            run_id=run_id,
            messages=_messages_to_json(messages),
            tool_calls=_extract_tool_calls_for_reflection(messages),
        )
    except Exception as exc:
        print(f"[auto_reflect] 反思失败：{exc}", flush=True)


_audit_logger = AuditLogger()


def _is_tool_failure(content: Any) -> tuple[bool, str]:
    """判断 ToolMessage 内容是否表示失败，返回 (是否失败, 错误信息)。"""
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("error"):
                return True, str(item["error"])
        return False, ""
    if isinstance(content, str):
        # 显式错误前缀（LangGraph ToolNode 异常、中文报错等）
        if content.startswith(("Error", "错误", "Traceback", "Exception", "NotImplementedError")):
            return True, content
        # JSON 显式错误对象
        try:
            parsed = json.loads(content)
            if isinstance(parsed, (dict, list)):
                failed, detail = _is_tool_failure(parsed)
                if failed:
                    return True, detail
        except Exception:
            pass
        # 通用失败关键词（不区分大小写）
        lowered = content.lower()
        if any(marker in lowered for marker in (
            "tool execution failed",
            "tool execution error",
            "failed to",
            "call failed",
            "not implemented",
            "module not found",
            "connection error",
            "timed out",
            "timeout",
            "报告合成失败",
        )):
            return True, content
    elif isinstance(content, dict) and content.get("error"):
        return True, str(content.get("error", ""))
    return False, ""


def _log_tool_failures(session_id: str, state: dict) -> None:
    """把本轮运行中的工具失败写入 audit.tool_failures，供 Coder Agent 自进化使用。"""
    if os.environ.get("ORION_AUTO_LOG_FAILURES", "1") == "0":
        return
    messages = current_task_messages(state)
    if not messages:
        return
    # 收集 AI tool_call 的参数，按 tool_call_id 索引
    tool_call_inputs: dict[str, dict] = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", []) or []:
                tcid = tc.get("id") or tc.get("tool_call_id")
                if tcid:
                    tool_call_inputs[tcid] = tc.get("args") or {}

    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        content = getattr(m, "content", "")
        failed, error = _is_tool_failure(content)
        if not failed:
            continue
        tcid = getattr(m, "tool_call_id", None)
        tool_name = getattr(m, "name", "unknown")
        input_data = tool_call_inputs.get(tcid, {})
        try:
            output_data = json.loads(content) if isinstance(content, str) else content
        except Exception:
            output_data = content
        try:
            _audit_logger.log_failure(
                run_id=session_id,
                session_id=session_id,
                tool_name=tool_name,
                input_data=input_data,
                output_data=output_data,
                error_message=error,
                failure_kind="error",
                tool_call_id=tcid,
            )
            print(f"[auto_log] 已记录工具失败：{tool_name}", flush=True)
        except Exception as exc:
            print(f"[auto_log] 记录失败：{exc}", flush=True)


_NUDGE_MAX = 2
_EVALUATION_TERMINAL_RESERVE_SECONDS = float(
    os.environ.get("ORION_EVALUATION_TERMINAL_RESERVE_SECONDS", "45")
)


class WorkflowIncompleteError(RuntimeError):
    """The agent exhausted replanning without completing a required workflow."""

# design_decision == "search" 时，允许连续纯检索的批次上限。
# 超过后必须回到 screen_formulations（若还有重规划次数）或强制 hold 并报告。
_DESIGN_MAX_SEARCH_ROUNDS = int(os.environ.get("ORION_DESIGN_MAX_SEARCH_ROUNDS", "3"))

# Literature workflows can otherwise evade the no-tool workflow gate by
# continuously issuing new (or identical) retrieval calls.  This counter is
# deliberately separate from the design decision search budget: it bounds
# evidence gathering before the terminal compiler, without treating retrieval
# as scientific success.
_WORKFLOW_MAX_EVIDENCE_ROUNDS = int(
    os.environ.get("ORION_WORKFLOW_MAX_EVIDENCE_ROUNDS", "16")
)

# 视为“纯补证检索”的工具集合；调用这些工具不会递增 design_attempts，
# 因此需要单独的 design_search_rounds 来防止无限循环。
_DESIGN_RETRIEVAL_TOOLS = frozenset({
    "search_literature_multiview",
    "search_literature_chunks",
    "search_kg_rag_hybrid",
    "search_paper_chunks",
    "explore_entity_neighborhood",
    "aggregate_evidence_by_entities",
    "kg_query",
    "resolve_literature_papers",
    "extract_numeric_properties",
    "verify_parameter_claim",
    "semantic_scholar_search",
    "memory_search",
})

# 会改变设计结构化状态、应当重置 search round 计数的工具。
_DESIGN_STATE_CHANGING_TOOLS = frozenset({
    "query_formulation_space",
    "generate_candidate_space",
    "enrich_formulation_evidence",
    "predict_formulation_conductivity",
    "screen_formulations",
    "optimize_formulations",
    "synthesize_design_report",
})


def _is_retrieval_only_tool(tool_name: str | None) -> bool:
    return tool_name in _DESIGN_RETRIEVAL_TOOLS


def _all_retrieval_only(tool_calls: list[dict]) -> bool:
    return bool(tool_calls) and all(
        _is_retrieval_only_tool(_tool_name(call)) for call in tool_calls
    )


_WORKFLOW_EVIDENCE_HOUSEKEEPING_TOOLS = frozenset({"read_todos", "write_todos"})


def _all_workflow_evidence_calls(tool_calls: list[dict]) -> bool:
    """Return whether a batch only gathers evidence or updates task bookkeeping."""

    names = [_tool_name(call) for call in tool_calls]
    allowed = _DESIGN_RETRIEVAL_TOOLS | _WORKFLOW_EVIDENCE_HOUSEKEEPING_TOOLS
    return bool(names) and any(name in _DESIGN_RETRIEVAL_TOOLS for name in names) and all(
        name in allowed for name in names
    )


def _tool_batch_fingerprint(tool_calls: list[dict]) -> str:
    """Canonicalize tool names and arguments while excluding volatile call ids."""

    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        args: Any = call.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, ValueError):
                pass
        normalized.append({"name": _tool_name(call), "args": args})
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)


def _prior_equivalent_successful_batch_count(
    messages: list[AnyMessage],
    tool_calls: list[dict],
) -> int:
    """Count earlier identical evidence batches whose tool results succeeded."""

    target = _tool_batch_fingerprint(tool_calls)
    successful_call_ids = {
        str(getattr(message, "tool_call_id", "") or "")
        for message in messages
        if isinstance(message, ToolMessage)
        and getattr(message, "tool_call_id", None)
        and getattr(message, "status", None) != "error"
        and not _is_tool_failure(getattr(message, "content", ""))[0]
    }
    count = 0
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        calls = list(getattr(message, "tool_calls", []) or [])
        if not _all_workflow_evidence_calls(calls):
            continue
        call_ids = {
            str(call.get("id") or call.get("tool_call_id") or "") for call in calls
        }
        if (
            call_ids
            and call_ids <= successful_call_ids
            and _tool_batch_fingerprint(calls) == target
        ):
            count += 1
    return count


def _control_formulation_call_from_evidence(
    messages: list[AnyMessage],
    *,
    task_type: str,
) -> AIMessage | None:
    """Advance a stuck design workflow to its first deterministic formulation query.

    This is a workflow-progress control, not a scientific decision.  It reuses
    only the original user question and already observed authoritative evidence.
    The formulation tool may still return no candidates; that outcome is valid
    and will be handled by the frozen screen/hold gates.
    """

    messages = current_task_messages({"messages": messages})
    if task_type not in {"design", "inverse_design"}:
        return None
    question = next(
        (
            str(getattr(message, "content", "") or "").strip()
            for message in messages
            if isinstance(message, HumanMessage)
            and str(getattr(message, "content", "") or "").strip()
            and not str(getattr(message, "content", "") or "").lstrip().startswith(
                ("（系统", "(System")
            )
        ),
        "",
    )
    evidence_chunks = _authoritative_evidence_chunks_from_messages(messages)
    if not question and not evidence_chunks:
        return None
    target_conditions = [question] if question else []
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "query_formulation_space",
                "args": {
                    "request": {
                        "task_type": task_type,
                        "target_conditions": target_conditions,
                        "evidence_chunks": evidence_chunks[:12],
                        "max_candidates": 3,
                    }
                },
                "id": f"workflow-formulation-{uuid4().hex}",
            }
        ],
    )


def _control_screen_call_from_formulation_result(
    messages: list[AnyMessage],
    *,
    task_type: str,
) -> AIMessage | None:
    """Compile a deterministic screen call from an observed formulation result.

    Before the first screen there is no ``DecisionState`` yet.  A model that
    repeatedly asks for more retrieval can therefore evade the post-decision
    search budget and run until LangGraph's recursion limit.  The formulation
    tool has already returned the exact candidates and target conditions needed
    by the read-only deterministic screener, so after the bounded retrieval
    allowance the control layer can advance the frozen workflow without making
    a semantic decision itself.
    """

    messages = current_task_messages({"messages": messages})
    results_by_call_id = {
        str(getattr(message, "tool_call_id", "") or ""): message
        for message in messages
        if isinstance(message, ToolMessage)
        and getattr(message, "tool_call_id", None)
        and not _is_tool_failure(getattr(message, "content", ""))[0]
    }
    screen_context: tuple[list[str], list[dict[str, Any]]] | None = None
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for call in reversed(getattr(message, "tool_calls", []) or []):
            if _tool_name(call) != "query_formulation_space":
                continue
            args = call.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (TypeError, ValueError):
                    args = {}
            request = args.get("request") if isinstance(args, dict) else {}
            request = request if isinstance(request, dict) else {}
            target_conditions = [
                str(item)
                for item in request.get("target_conditions") or []
                if str(item).strip()
            ]
            constraint_records = [
                item
                for item in request.get("constraint_records") or []
                if isinstance(item, dict)
            ]
            if target_conditions or constraint_records:
                screen_context = (target_conditions, constraint_records)
                break
        if screen_context is not None:
            break

    if screen_context is None:
        # A model may formulate from retrieved evidence without copying the
        # user's target into ``target_conditions``.  That omission must not
        # turn a required deterministic screen into a workflow abort.  Reuse
        # the original request only as a screening condition; it creates no
        # scientific claim and makes the eventual decision fail safely.
        question = next(
            (
                str(getattr(message, "content", "") or "").strip()
                for message in messages
                if isinstance(message, HumanMessage)
                and str(getattr(message, "content", "") or "").strip()
            ),
            "",
        )
        if not question:
            return None
        screen_context = ([question], [])
    target_conditions, constraint_records = screen_context

    # Prefer the latest enriched candidate population.  Falling back to the
    # original formulation result preserves the earlier pre-screen behavior.
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for call in reversed(getattr(message, "tool_calls", []) or []):
            if _tool_name(call) not in {
                "predict_formulation_conductivity",
                "enrich_formulation_evidence",
                "query_formulation_space",
            }:
                continue
            call_id = str(call.get("id") or call.get("tool_call_id") or "")
            tool_message = results_by_call_id.get(call_id)
            if tool_message is None:
                continue
            raw_candidates: Any = getattr(tool_message, "content", "")
            if isinstance(raw_candidates, str):
                try:
                    raw_candidates = json.loads(raw_candidates)
                except (TypeError, ValueError):
                    continue
            if isinstance(raw_candidates, dict):
                raw_candidates = raw_candidates.get("candidates")
            if not isinstance(raw_candidates, list):
                continue
            candidates = [item for item in raw_candidates if isinstance(item, dict)]
            if not candidates:
                # ``screen_formulations`` accepts an explicit empty-evidence
                # candidate and deterministically returns a non-recommendation.
                # This closes an inverse-design episode as ``hold`` instead of
                # relying on the model to invent a candidate or aborting the
                # workflow after a premature terminal compiler call.
                candidates = [{
                    "candidate_id": "deterministic-no-candidate",
                    "rationale": "No formulation candidate was returned by the bounded formulation step.",
                    "evidence": [],
                    "evidence_level": "weak",
                    "unresolved_gaps": ["No eligible formulation candidate was produced."],
                    "evidence_scope": "unknown",
                }]
            screen_request: dict[str, Any] = {
                "candidates": candidates,
                "target_conditions": target_conditions,
            }
            if constraint_records:
                screen_request["constraint_records"] = constraint_records
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call-control-screen-{call_id}",
                        "name": "screen_formulations",
                        "args": {"request": screen_request},
                    }
                ],
            )
    return None


def _unfinished_todos(todo_list: list) -> list[tuple[str, str]]:
    """提取未完成待办项的 (id, content)。

    兼容 TodoItem 与 dict 两种形态（todo_list 经 checkpoint serde 往返后会
    退化为 dict，与 todo_tools.read_todos 的双形态处理一致）。
    """
    out = []
    for t in todo_list or []:
        if isinstance(t, TodoItem):
            status, tid, content = t.status, t.id, t.content
        elif isinstance(t, dict):
            status, tid, content = t.get("status"), t.get("id", "?"), t.get("content", "")
        else:
            continue
        if status != "completed":
            out.append((tid, content))
    return out


def _active_skill_name(messages: list[AnyMessage]) -> str | None:
    """返回最近一次成功完成的 load_skill 工具调用中的 Skill 名。"""

    messages = current_task_messages({"messages": messages})
    successful_call_ids = {
        getattr(message, "tool_call_id", None)
        for message in messages
        if isinstance(message, ToolMessage)
        and getattr(message, "tool_call_id", None)
        and not _is_tool_failure(getattr(message, "content", ""))[0]
    }
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        for tc in getattr(msg, "tool_calls", []) or []:
            if _tool_name(tc) != "load_skill":
                continue
            call_id = tc.get("id") or tc.get("tool_call_id")
            if not call_id or call_id not in successful_call_ids:
                continue
            args = tc.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if isinstance(args, dict):
                return args.get("name")
    return None


def _called_tool_names(messages: list[AnyMessage]) -> set[str]:
    """返回消息历史中所有被发起过的工具名集合（含失败调用）。"""
    messages = current_task_messages({"messages": messages})
    return {
        _tool_name(tc)
        for msg in messages
        if isinstance(msg, AIMessage)
        for tc in getattr(msg, "tool_calls", []) or []
    }


def _successful_tool_names(messages: list[AnyMessage]) -> set[str]:
    """Return tools with a completed, non-error ToolMessage.

    Merely emitting a tool call is not workflow completion. This distinction
    is especially important for the terminal compiler: a failed synthesis call
    must never let ``review`` release a partial answer to grounding.
    """

    messages = current_task_messages({"messages": messages})
    calls_by_id: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", []) or []:
            call_id = call.get("id") or call.get("tool_call_id")
            tool_name = _tool_name(call)
            if call_id and tool_name:
                calls_by_id[call_id] = tool_name

    successful: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        call_id = getattr(message, "tool_call_id", None)
        tool_name = calls_by_id.get(call_id or "") or getattr(message, "name", None)
        if (tool_name and getattr(message, "status", None) != "error"
                and not _is_tool_failure(getattr(message, "content", ""))[0]):
            successful.add(tool_name)
    return successful


def _tool_call_count(messages: list[AnyMessage], tool_name: str) -> int:
    """按消息历史统计工具实际发起次数，作为结构化计数器的保守下界。"""
    messages = current_task_messages({"messages": messages})
    return sum(
        1
        for msg in messages
        if isinstance(msg, AIMessage)
        for tc in getattr(msg, "tool_calls", []) or []
        if _tool_name(tc) == tool_name
    )


def _missing_required_workflow_steps(messages: list[AnyMessage]) -> list[dict]:
    """基于当前激活 Skill 的 workflow 声明，返回尚未调用的 required 步骤。"""
    messages = current_task_messages({"messages": messages})
    completed = _successful_tool_names(messages)
    skill_name = _active_skill_name(messages)
    if not skill_name:
        # Agent 尚未激活任何 Skill 时，没有可执行的工作流步骤可检查。
        # 系统提示已要求匹配 skill 触发词时先调用 load_skill；review gate
        # 不在此处强制 load_skill，避免非 skill 对话被错误拦截。
        return []
    steps = get_skill_workflow(skill_name)
    if not steps:
        return []
    missing = []
    for step in steps:
        tool = step.get("tool")
        if step.get("required") and tool and tool not in completed:
            missing.append(step)
    return missing


def _evaluation_remaining_seconds(state: GraphState) -> float | None:
    """Return remaining formal-evaluation wall clock, if one was configured."""

    started = state.get("evaluation_started_at_monotonic")
    timeout = state.get("evaluation_timeout_seconds")
    if started is None or timeout is None:
        return None
    try:
        return max(0.0, float(timeout) - (time.monotonic() - float(started)))
    except (TypeError, ValueError):
        return None


def _authoritative_evidence_chunks_from_messages(
    messages: list[AnyMessage],
    *,
    limit: int = 16,
) -> list[dict[str, Any]]:
    """Recover exact retrieval chunks for deterministic workflow completion."""

    messages = current_task_messages({"messages": messages})
    allowed = {
        "search_literature_multiview",
        "search_literature_chunks",
        "search_paper_chunks",
    }
    chunks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        if str(getattr(message, "name", "") or "") not in allowed:
            continue
        failed, _ = _is_tool_failure(getattr(message, "content", ""))
        if failed:
            continue
        payload = parse_chunk_payload(str(getattr(message, "content", "") or ""))
        for item in payload or []:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id") or "").strip()
            content = str(item.get("content") or "").strip()
            if not chunk_id or not content or chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "paper_id": str(item.get("paper_id") or ""),
                    "paper_title": str(item.get("paper_title") or ""),
                    "section_title": str(item.get("section_title") or ""),
                    "source": str(item.get("source") or ""),
                }
            )
            if len(chunks) >= limit:
                return chunks
    return chunks


def _force_safe_terminal_synthesis(
    state: GraphState,
    messages: list[AnyMessage],
    missing_steps: list[dict],
) -> Command[Literal["tools"]] | None:
    """Complete deterministic tail steps when the model omitted terminalization.

    This is not an episode retry and it does not call the model. It is allowed
    for the terminal compiler, plus candidate-screening numeric extraction when
    exact retrieved chunks already exist. Design may use it only after the
    structured decision gate is terminal. All other exhaustion paths fail closed.
    """

    missing_tools = [str(step.get("tool") or "") for step in missing_steps]
    if "synthesize_design_report" in _called_tool_names(messages):
        return None

    task_type = str(
        state.get("configured_task_type")
        or task_type_for_skill(_active_skill_name(messages))
    )
    evidence_chunks: list[dict[str, Any]] = []
    forced_tools: list[tuple[str, dict[str, Any]]] = []
    append_terminal_compiler = True
    if task_type == "evidence_synthesis" and missing_tools == [
        "synthesize_design_report"
    ]:
        pass
    elif task_type == "candidate_screening" and missing_tools == [
        "synthesize_design_report"
    ]:
        pass
    elif task_type == "candidate_screening" and missing_tools == [
        "extract_numeric_properties",
        "synthesize_design_report",
    ]:
        evidence_chunks = _authoritative_evidence_chunks_from_messages(messages)
        if not evidence_chunks:
            return None
        forced_tools.append(
            (
                "extract_numeric_properties",
                {"evidence_chunks": evidence_chunks, "property_names": []},
            )
        )
    elif task_type in {"design", "inverse_design"} and missing_tools == [
        "synthesize_design_report"
    ]:
        decision_raw = state.get("design_decision")
        decision = (
            decision_raw
            if isinstance(decision_raw, DecisionState)
            else DecisionState.model_validate(decision_raw)
            if decision_raw
            else None
        )
        if decision is None or decision.status not in {"recommend", "hold"}:
            return None
    elif task_type == "inverse_design" and missing_tools == [
        "screen_formulations",
        "synthesize_design_report",
    ]:
        forced_screen = _control_screen_call_from_formulation_result(
            messages,
            task_type=task_type,
        )
        if forced_screen is None:
            return None
        forced_tools.append(
            (
                "screen_formulations",
                dict(forced_screen.tool_calls[0].get("args") or {}),
            )
        )
        append_terminal_compiler = False
    else:
        return None

    if append_terminal_compiler:
        forced_tools.append(
            (
                "synthesize_design_report",
                {
                    "request": {
                        "task_type": task_type,
                        "claims": [],
                        "next_steps": [
                            "Preserve unresolved evidence gaps and do not overstate the conclusion."
                        ],
                    }
                },
            )
        )

    approved_tools = dict(state.get("approved_tools", {}))
    approval_needed = [
        name
        for name, _ in forced_tools
        if approved_tools.get(name) != "1"
    ]
    if approval_needed and os.environ.get("ORION_AUTO_APPROVE") != "1":
        result = interrupt(
            "Required deterministic workflow completion needs approval:\n"
            + "\n".join(approval_needed)
            + "\n"
            "Enter 0 to allow once, 1 to allow for this session, or provide a rejection reason."
        )
        if result not in ("0", "1"):
            raise WorkflowIncompleteError(
                "required deterministic workflow completion was rejected after exhaustion"
            )
        if result == "1":
            for name in approval_needed:
                approved_tools[name] = "1"

    from langchain_core.messages import RemoveMessage

    last_message = messages[-1]
    updates: list[AnyMessage] = []
    last_id = getattr(last_message, "id", None)
    if last_id:
        updates.append(RemoveMessage(id=last_id))
    updates.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"workflow-terminal-{name}-{uuid4().hex}",
                }
                for name, args in forced_tools
            ],
        )
    )
    print(
        "[review] workflow budget exhausted; force deterministic "
        f"{task_type} tail: {', '.join(name for name, _ in forced_tools)}",
        flush=True,
    )
    return Command(
        update={
            "messages": updates,
            "approved_tools": approved_tools,
            "nudge_count": 0,
            "workflow_evidence_rounds": 0,
        },
        goto="tools",
    )


def _force_required_workflow_prerequisites(
    state: GraphState,
    messages: list[AnyMessage],
    missing_steps: list[dict],
) -> Command[Literal["tools"]] | None:
    """Run deterministic prerequisites before a terminal compiler call.

    Required steps are successful-tool obligations, not prompt suggestions. A
    model may place ``synthesize_design_report`` in the same batch as, or ahead
    of, a missing prerequisite. Replace that pending terminal batch with the
    prerequisite so ordering is observable and the compiler cannot commit a
    formally incomplete workflow.
    """

    task_type = str(
        state.get("configured_task_type")
        or task_type_for_skill(_active_skill_name(messages))
        or ""
    )
    missing_tools = [
        str(step.get("tool") or "")
        for step in missing_steps
        if str(step.get("tool") or "") != "synthesize_design_report"
    ]
    forced_name: str
    forced_args: dict[str, Any]
    if task_type == "inverse_design" and missing_tools == ["screen_formulations"]:
        # inverse_design 在终态合成前缺失 screen_formulations（R4
        # c2f_r3_inverse_06 变体：模型绕开筛选直接请求终态合成）。与
        # _force_safe_terminal_synthesis 使用同一确定性 fallback，按已观察的
        # formulation 结果构造保守 screen 调用（无候选时由 fallback 提供
        # deterministic-no-candidate 弱证据候选）。这只是确定性约束，
        # 不产生任何科学断言；无法构造时保持 fail-closed。
        forced_screen = _control_screen_call_from_formulation_result(
            messages[:-1],
            task_type=task_type,
        )
        if forced_screen is None:
            return None
        forced_name = "screen_formulations"
        forced_args = dict(forced_screen.tool_calls[0].get("args") or {})
    elif task_type == "candidate_screening" and missing_tools == [
        "extract_numeric_properties"
    ]:
        evidence_chunks = _authoritative_evidence_chunks_from_messages(messages[:-1])
        if not evidence_chunks:
            return None
        forced_name = "extract_numeric_properties"
        forced_args = {
            "evidence_chunks": evidence_chunks,
            "property_names": [],
        }
    elif task_type == "design" and "query_formulation_space" in missing_tools:
        # design 任务在终态合成前缺失 query_formulation_space（模型常以
        # generate_candidate_space 采样新组合，替代"从 KG 检索文献候选"这一
        # required 第一步）。复用与"卡住的 design workflow"相同的确定性推进
        # 逻辑，把已检索到的权威证据喂给 formulation query，使工作流可观测地
        # 走完 formulate → screen → report，而不是 fail-closed 抛异常。
        control = _control_formulation_call_from_evidence(
            messages[:-1], task_type=task_type
        )
        if control is None:
            return None
        forced_name = "query_formulation_space"
        forced_args = dict(control.tool_calls[0].get("args") or {})
    else:
        return None

    approved_tools = dict(state.get("approved_tools", {}))
    if (
        approved_tools.get(forced_name) != "1"
        and os.environ.get("ORION_AUTO_APPROVE") != "1"
    ):
        result = interrupt(
            "Required workflow prerequisite needs approval:\n"
            f"{forced_name}\n"
            "Enter 0 to allow once, 1 to allow for this session, or provide a rejection reason."
        )
        if result not in ("0", "1"):
            raise WorkflowIncompleteError(
                f"required workflow prerequisite {forced_name} was rejected "
                "before terminal synthesis"
            )
        if result == "1":
            approved_tools[forced_name] = "1"

    from langchain_core.messages import RemoveMessage

    replacement: list[AnyMessage] = []
    last_id = getattr(messages[-1], "id", None)
    if last_id:
        replacement.append(RemoveMessage(id=last_id))
    replacement.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": forced_name,
                    "args": forced_args,
                    "id": f"workflow-prerequisite-{forced_name}-{uuid4().hex}",
                }
            ],
        )
    )
    return Command(
        update={
            "messages": replacement,
            "approved_tools": approved_tools,
            "nudge_count": 0,
        },
        goto="tools",
    )


def _repetitive_text(messages: list[AnyMessage]) -> str | None:
    """检测最近无 tool_calls 的 AIMessage 是否出现重复内容，防止陷入自语循环。

    返回需要替换成的强制提醒；未检测到重复返回 None。
    """
    recent_texts = [
        str(getattr(msg, "content", "") or "").strip()
        for msg in messages[-6:]
        if isinstance(msg, AIMessage)
        and not (getattr(msg, "tool_calls", None))
        and len(str(getattr(msg, "content", "") or "").strip()) > 60
    ]
    if not recent_texts:
        return None
    from collections import Counter

    counts = Counter(recent_texts)
    most_common, count = counts.most_common(1)[0]
    if count >= 3:
        return (
            "（系统提醒）检测到你在重复同一段自由文本而没有调用工具。"
            "请停止重复，结合已有结果选择有进展的行动或说明限制；"
            "不要把当前无法推荐误写成已验证。"
        )
    # 也检测同一句子在单条消息内被重复多次（常见于 LLM 自语循环）。
    last_text = recent_texts[-1]
    first_sentence = last_text.split("。")[0].strip()
    if first_sentence and len(first_sentence) > 20:
        sentence_hits = last_text.count(first_sentence)
        if sentence_hits >= 3:
            return (
                "（系统提醒）检测到当前回复在同一条消息内重复同一句话，"
                "请检查已有结果并决定如何继续或结束，"
                "避免重复执行。"
            )
    return None


def _unanswered_tool_calls(messages: list[AnyMessage]) -> list[dict]:
    """返回缺少协议所需 ToolMessage 的工具调用。"""
    answered = {
        getattr(message, "tool_call_id", None)
        for message in messages
        if isinstance(message, ToolMessage) and getattr(message, "tool_call_id", None)
    }
    pending: list[dict] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", []) or []:
            call_id = call.get("id") or call.get("tool_call_id")
            if call_id and call_id not in answered:
                pending.append(call)
    return pending


def _latest_tool_result_failed(
    messages: list[AnyMessage],
    tool_name: str,
) -> bool:
    """检查指定工具最近一次已返回结果是否失败，不推断任何领域工作流。"""

    messages = current_task_messages({"messages": messages})
    matching_call_ids: set[str] = set()
    latest: bool | None = None
    for message in messages:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", []) or []:
                if _tool_name(call) == tool_name:
                    call_id = call.get("id") or call.get("tool_call_id")
                    if call_id:
                        matching_call_ids.add(call_id)
        elif (
            isinstance(message, ToolMessage)
            and (getattr(message, "tool_call_id", None) in matching_call_ids
                 or getattr(message, "name", None) == tool_name)
        ):
            latest, _ = _is_tool_failure(getattr(message, "content", ""))
            latest = latest or getattr(message, "status", None) == "error"
    return latest is True


def _active_task_type_for_review(
    state: Mapping[str, Any],
    messages: list[AnyMessage],
) -> str:
    return str(
        state.get("configured_task_type")
        or state.get("design_task_type")
        or task_type_for_skill(_active_skill_name(messages))
        or ""
    )


_SCIENTIFIC_REVIEW_CONTEXT = ScientificReviewContext(
    tool_name=_tool_name,
    active_task_type=_active_task_type_for_review,
    successful_tool_names=_successful_tool_names,
    all_retrieval_only=_all_retrieval_only,
    prior_equivalent_successful_batch_count=_prior_equivalent_successful_batch_count,
    evaluation_remaining_seconds=_evaluation_remaining_seconds,
    control_formulation_call_from_evidence=_control_formulation_call_from_evidence,
    control_screen_call_from_formulation_result=_control_screen_call_from_formulation_result,
    tool_call_count=_tool_call_count,
    incomplete_error_type=WorkflowIncompleteError,
    nudge_max=_NUDGE_MAX,
    design_max_search_rounds=_DESIGN_MAX_SEARCH_ROUNDS,
    evaluation_terminal_reserve_seconds=_EVALUATION_TERMINAL_RESERVE_SECONDS,
    design_state_changing_tools=_DESIGN_STATE_CHANGING_TOOLS,
    design_retrieval_tools=_DESIGN_RETRIEVAL_TOOLS,
)


def review(state: GraphState) -> Command[Literal["tools", "llm", "grounding", "__end__"]]:
    """停止出口：协议与交付完整性检查，不规划科研步骤、不执行语义模型。"""
    state = current_task_state(state)
    messages = list(state["messages"])
    last = messages[-1]
    reason = _incomplete_response_reason(last) or _unknown_final_citations(state, last)
    if reason:
        count = int(state.get("nudge_count") or 0)
        if count >= _NUDGE_MAX:
            raise WorkflowIncompleteError("模型响应仍不完整；本轮未完成：" + reason)
        incomplete = last.model_copy(update={"tool_calls": [], "invalid_tool_calls": [],
            "additional_kwargs": {**{key: value for key, value in last.additional_kwargs.items()
                                      if key not in {"tool_calls", "function_call"}},
                "orion_incomplete_tool_calls": [*last.tool_calls, *last.invalid_tool_calls],
                "orion_incomplete_provider_calls": {key: last.additional_kwargs[key]
                    for key in ("tool_calls", "function_call") if key in last.additional_kwargs}}})
        return Command(update={"messages": [incomplete,
            internal_human_message(content="（系统提醒）"+reason)],
            "nudge_count": count+1}, goto="llm")
    if isinstance(last, AIMessage) and last.tool_calls:
        return pre_tool_gate(state)  # 旧快照/直接调用的兼容入口。
    pending = _unanswered_tool_calls(messages)
    if pending:
        return Command(update={"messages": [ToolMessage(
            content="Error: 工具调用未执行；请根据当前任务重新决定行动。",
            name=_tool_name(call), tool_call_id=call.get("id") or call.get("tool_call_id"),
            status="error") for call in pending]}, goto="llm")
    terminal = terminal_result(state)
    if terminal is not None:
        body, artifact, call_id = terminal
        return Command(update={"last_report": body, "task_artifact": artifact,
            "terminal_committed": True, "terminal_commit_tool_call_id": call_id},
            goto="grounding")
    # 正常结束由 Agent 的最终回答表达；加载 Skill 或指定任务类型不强制报告工具。
    # 截断、空响应和无效工具参数属于模型协议恢复，不是科研上的 hold。
    if not isinstance(last, AIMessage):
        raise WorkflowIncompleteError("模型尚未返回最终回答")
    final = last.model_copy(update={"additional_kwargs": {
        **last.additional_kwargs, "orion_final_response": True,
    }})
    return Command(update={"messages": [final], "last_report": str(last.content),
        "task_artifact": None, "terminal_committed": True,
        "terminal_commit_tool_call_id": None}, goto="grounding")


def _incomplete_response_reason(message: AnyMessage) -> str | None:
    """只识别响应协议问题，不判断研究结论或工具序列是否充分。"""
    if not isinstance(message, AIMessage):
        return None
    metadata = message.response_metadata or {}
    finish = metadata.get("finish_reason") or metadata.get("stop_reason")
    if finish in {"length", "max_tokens"}:
        return "上次模型响应因输出长度限制被截断，尚未完成。请缩短输出，重新给出完整回答或完整工具调用；不要重复已执行的操作。"
    if message.invalid_tool_calls:
        return "上次工具调用参数无法解析。请修正参数，重新选择行动；不得把未执行的调用视为成功。"
    if finish == "tool_calls" and not message.tool_calls:
        return "模型声明了工具调用，但没有返回可执行参数。请重新给出完整调用或明确回答。"
    if not message.tool_calls and not str(message.content or "").strip():
        return "上次模型响应为空，尚未形成最终回答。请结合已有结果继续，或明确说明限制。"
    return None


def _unknown_final_citations(state: Mapping[str, Any], message: AnyMessage) -> str | None:
    """只检查明确的 chunk ID 是否来自本轮资料，不做科学语义裁决。"""
    if not isinstance(message, AIMessage) or message.tool_calls:
        return None
    import re
    from orion.core.retrieved_evidence import _retrieved_sources
    from orion.core.task_scope import current_user_message
    cited = re.findall(r"\[([^\[\]\n]*::[^\[\]\n]*)\]", str(message.content or ""))
    sources = _retrieved_sources(state)
    user = current_user_message(state)
    supplied = str(user.content) if user else ""
    missing = sorted({key for key in cited if key not in sources and key not in supplied})
    if missing:
        return "这些引用 ID 不在本轮工具结果或用户资料中：" + ", ".join(missing[:5]) + "。请修正引用或说明缺少来源，不得伪造文献证据。"
    return None


def pre_tool_gate(state: GraphState) -> Command[Literal["tools", "llm", "grounding", "__end__"]]:
    """工具前生命周期：预算、重复执行和审批；保持 Agent 选择的工具及参数。"""
    state = current_task_state(state)
    messages = list(state["messages"])
    last = messages[-1]
    terminal = terminal_result({"messages": messages[:-1]})
    if terminal is not None:
        body, artifact, call_id = terminal
        from langchain_core.messages import RemoveMessage
        return Command(update={"messages": [RemoveMessage(id=last.id)],
            "last_report": body, "task_artifact": artifact, "terminal_committed": True,
            "terminal_commit_tool_call_id": call_id}, goto="grounding")
    remaining = _evaluation_remaining_seconds(state)
    if remaining is not None and remaining <= 0:
        raise WorkflowIncompleteError("执行时间预算已耗尽；没有执行新的工具")
    repeated = _prior_equivalent_successful_batch_count(messages[:-1], list(last.tool_calls)) > 0
    identical_attempts = sum(
        isinstance(message, AIMessage) and bool(message.tool_calls)
        and _tool_batch_fingerprint(message.tool_calls) == _tool_batch_fingerprint(last.tool_calls)
        for message in messages[:-1])
    rounds = int(state.get("workflow_evidence_rounds") or 0)
    evidence_only = _all_workflow_evidence_calls(last.tool_calls)
    exhausted = evidence_only and rounds >= int(state.get("workflow_max_evidence_rounds")
                                                or _WORKFLOW_MAX_EVIDENCE_ROUNDS)
    if repeated or exhausted or identical_attempts >= _NUDGE_MAX + 1:
        count = int(state.get("nudge_count") or 0)
        if count >= _NUDGE_MAX:
            raise WorkflowIncompleteError("重复操作或检索预算已耗尽；未替 Agent 生成工具调用")
        from langchain_core.messages import RemoveMessage
        return Command(update={"messages": [RemoveMessage(id=last.id), internal_human_message(
            content="（系统提醒）该操作已成功执行或当前检索预算已用尽。请检查已有结果，"
                    "选择有进展的行动，或明确说明限制并收尾。系统不会替你选择科研工具。")],
            "nudge_count": count+1}, goto="llm")
    approved_tools = dict(state.get("approved_tools") or {})
    saved = dict((last.additional_kwargs or {}).get("orion_tool_approvals") or {})
    approved, rejected = [], []
    for call in last.tool_calls:
        name = _tool_name(call)
        fingerprint = json.dumps({"name": name, "args": call.get("args", {})},
                                 ensure_ascii=False, sort_keys=True)
        if (saved.get(call.get("id")) == fingerprint or approved_tools.get(name) == "1"
                or os.environ.get("ORION_AUTO_APPROVE") == "1"):
            answer = "0"
        else:
            answer = interrupt(f"工具请求审批：\n{name}({call.get('args', {})})\n"
                               "输入 0 本次允许，输入 1 会话级允许，或输入拒绝原因")
        if answer in ("0", "1"):
            approved.append(call)
            saved[call.get("id")] = fingerprint
            if answer == "1":
                approved_tools[name] = "1"
        else:
            rejected.append(ToolMessage(content=f"Error: 用户拒绝工具 {name}：{answer}",
                name=name, tool_call_id=call.get("id"), status="error"))
    if not approved:
        return Command(update={"messages": rejected, "approved_tools": approved_tools}, goto="llm")
    metadata = {**last.additional_kwargs, "orion_tool_approvals": saved}
    if rejected:
        metadata["orion_rejected_tools"] = [m.content for m in rejected]
    approved_message = last.model_copy(update={"tool_calls": approved, "additional_kwargs": metadata})
    names = sorted({_tool_name(c) for c in approved})
    nudge = int(state.get("nudge_count") or 0) if names == state.get("last_approved_tools") else 0
    return Command(update={"messages": [approved_message], "approved_tools": approved_tools,
        "workflow_evidence_rounds": rounds+1 if evidence_only else 0,
        "nudge_count": nudge, "last_approved_tools": names}, goto="tools")


def route_after_llm(state: GraphState) -> str:
    """旧图兼容路由；新图由节点出口 Hook 直接返回 Command。"""
    last = state["messages"][-1]
    return "pre_tool_gate" if isinstance(last, AIMessage) and last.tool_calls else "review"


#======================================上下文压缩

def _sanitize_messages_legacy(messages: list[AnyMessage]) -> list[AnyMessage]:
    """保证 message 序列对 LLM 后端合法。

    1. 删除没有后续匹配 ToolMessage 的 tool_call；
    2. 删除没有对应 tool_call 的 orphan ToolMessage。

    该函数用于在每次调用模型前兜底，防止 `compact` 等节点保留证据
    ToolMessage 但丢弃父 AIMessage 后触发 "assistant message with
    tool_calls must be followed by tool messages" 错误。
    """
    tool_call_ids = {
        getattr(m, "tool_call_id", None)
        for m in messages
        if isinstance(m, ToolMessage)
    }
    seen_tool_call_ids: set[str] = set()
    sanitized: list[AnyMessage] = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            valid_calls: list[dict] = []
            for tc in tool_calls:
                tcid = tc.get("id") or tc.get("tool_call_id")
                if tcid and tcid in tool_call_ids:
                    valid_calls.append(tc)
                    seen_tool_call_ids.add(tcid)
            if valid_calls:
                if len(valid_calls) == len(tool_calls):
                    sanitized.append(msg)
                else:
                    sanitized.append(msg.model_copy(update={"tool_calls": valid_calls}))
            else:
                sanitized.append(msg.model_copy(update={"tool_calls": []}))
        elif isinstance(msg, ToolMessage):
            tcid = getattr(msg, "tool_call_id", None)
            if tcid in seen_tool_call_ids:
                sanitized.append(msg)
            # orphan ToolMessage 被丢弃
        else:
            sanitized.append(msg)
    return sanitized


#======================================
def _sanitize_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """保留严格相邻的 AI tool-call / ToolMessage 组，丢弃不完整历史。"""

    def _clean_ai_message(message: AIMessage, valid_calls: list[dict]) -> AIMessage:
        """同步清除 provider 保留的 malformed raw tool calls。"""
        valid_ids = {
            call.get("id") or call.get("tool_call_id")
            for call in valid_calls
        }
        kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
        raw_calls = kwargs.get("tool_calls")
        if isinstance(raw_calls, list):
            kwargs["tool_calls"] = [
                call for call in raw_calls
                if isinstance(call, dict) and (call.get("id") or call.get("tool_call_id")) in valid_ids
            ]
        return message.model_copy(
            update={
                "tool_calls": valid_calls,
                "invalid_tool_calls": [],
                "additional_kwargs": kwargs,
            }
        )

    sanitized: list[AnyMessage] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = getattr(message, "tool_calls", None) or [] if isinstance(message, AIMessage) else []
        if not tool_calls:
            if not isinstance(message, ToolMessage):
                raw_calls = getattr(message, "additional_kwargs", {}).get("tool_calls")
                malformed_calls = getattr(message, "invalid_tool_calls", None) or []
                # 某些 provider 会只留下无法解析的 raw/invalid tool call，
                # 此时 ``tool_calls`` 为空，仍不能把原消息再次发回后端。
                if raw_calls or malformed_calls:
                    sanitized.append(_clean_ai_message(message, []))
                else:
                    sanitized.append(message)
            index += 1
            continue

        following: list[ToolMessage] = []
        cursor = index + 1
        while cursor < len(messages) and isinstance(messages[cursor], ToolMessage):
            following.append(messages[cursor])
            cursor += 1
        answered_ids = {getattr(item, "tool_call_id", None) for item in following}
        valid_calls = [
            call for call in tool_calls
            if (call.get("id") or call.get("tool_call_id")) in answered_ids
        ]
        valid_ids = {call.get("id") or call.get("tool_call_id") for call in valid_calls}
        sanitized.append(_clean_ai_message(message, valid_calls))
        sanitized.extend(
            item for item in following
            if getattr(item, "tool_call_id", None) in valid_ids
        )
        index = cursor
    return sanitized


def _invoke_sync_model_request(
    model: Any,
    messages: list[AnyMessage],
    *,
    max_retries: int,
) -> tuple[Any, int]:
    """Invoke one synchronous model request with bounded transport retries."""

    for attempt in range(max_retries + 1):
        try:
            return model.invoke(messages), attempt
        except Exception as exc:
            if attempt < max_retries and _is_retryable_llm_transport_error(exc):
                print(
                    "[LLM retry] request transport failed; "
                    f"retry {attempt + 1}/{max_retries}",
                    flush=True,
                )
                continue
            _annotate_llm_request_failure(
                exc,
                transport_retries=attempt,
                max_retries=max_retries,
            )
            raise
    raise AssertionError("unreachable request retry loop")


def _annotate_llm_request_failure(
    exc: BaseException,
    *,
    transport_retries: int,
    max_retries: int,
) -> None:
    """Keep bounded request-attempt metadata on a terminal exception.

    A failed LangGraph node does not commit its state update.  Without carrying
    these two values on the exception, the evaluation checkpoint truthfully
    preserves the failure but incorrectly reports zero transport retries.
    Exception attributes survive LangGraph's in-process propagation and let the
    runner merge the failed request with retries already committed by earlier
    successful model steps.
    """

    for name, value in (
        ("orion_llm_transport_retry_count", int(transport_retries)),
        ("orion_llm_request_max_retries", int(max_retries)),
    ):
        try:
            setattr(exc, name, value)
        except Exception:
            # Some third-party exception implementations may forbid custom
            # attributes.  Preserve the original failure in that rare case.
            pass


def _allow_next_budgeted_request(budget) -> bool:
    """预算暂停发生在请求前；继续仅授权一次新请求，不重发已保存响应。"""
    if budget is None:
        return True
    result = budget.check_before_request()
    if not result.exceeded:
        return True
    answer = interrupt(
        f"运行已暂停：{result.reason}。输入 /continue 授权一次新请求，或 /stop 结束。"
    )
    return str(answer).strip() == "/continue"


def llm(state: GraphState, config: RunnableConfig | None = None):
    thread_id = _get_thread_id(config)
    runtime = _thread_runtime.setdefault(thread_id, {})

    budget = None
    if config and isinstance(config.get("configurable"), dict):
        budget = config["configurable"].get("budget")

    if not _allow_next_budgeted_request(budget):
        return {"messages": [AIMessage(content="已按要求停止；当前科研任务未因此被标记完成。") ]}

    msgs = _sanitize_messages(state["messages"])

    previous_tool_count = runtime.get("_printed_tool_messages", 0)
    current_tool_messages = [msg for msg in state["messages"][previous_tool_count:] if isinstance(msg, ToolMessage)]
    for msg in current_tool_messages:
        try:
            msg.pretty_print()
        except UnicodeEncodeError:
            pass

    token_count = 0
    max_retries = _llm_request_max_retries(config)
    transport_retries = 0
    try:
        res, transport_retries = _invoke_sync_model_request(
            llm_with_tools,
            msgs,
            max_retries=max_retries,
        )
        token_count = (res.usage_metadata or {}).get("total_tokens", 0)
        print(token_count)
    except Exception as e:
        err_msg = str(e)
        if "finish_reason" in err_msg and "length" in err_msg:
            print("[截断托底] 检测到输出截断，正在续写...")
            continuation_prompt = internal_human_message(content="你的上一次回复被截断了，请继续输出剩余内容，不要重复已经输出的部分。")
            res = llm_with_tools.invoke(msgs + [continuation_prompt])
            token_count = (res.usage_metadata or {}).get("total_tokens", 0)
        else:
            raise

    if budget is not None:
        # 每个响应的 usage 是本次请求用量，不是会话累计值；不在响应写入前暂停。
        budget.record_response(token_count if res.usage_metadata and "total_tokens" in res.usage_metadata else None)

    try:
        res.pretty_print()
    except UnicodeEncodeError:
        pass
    runtime["_printed_tool_messages"] = len(state["messages"])
    runtime["token_count"] = token_count
    return {
        "messages": [res],
        "llm_request_max_retries": max_retries,
        "llm_transport_retry_count": int(
            state.get("llm_transport_retry_count") or 0
        )
        + transport_retries,
    }


# from langgraph.checkpoint.sqlite import SqliteSaver
#
# # 连接到本地 SQLite 数据库文件
# with SqliteSaver.from_conn_string("my_checkpoints.db") as checkpointer:
#     # 编译时传入 checkpointer
#     graph = workflow.compile(checkpointer=checkpointer)
#
#     # 之后的所有运行状态都会自动持久化到该文件中
#     config = {"configurable": {"thread_id": "session_1"}}
#     graph.invoke({"input": "Hello"}, config)


async def llm_async(state: GraphState, config: RunnableConfig | None = None):
    """异步 LLM 节点。默认流式输出（供 Web UI/CLI 逐 token 显示）；
    ORION_LLM_STREAM=0 或 config.configurable.llm_stream=False 时走非流式
    （E2E 测试等无可视化场景，避免流式停滞干扰，见 _llm_stream_enabled）。"""
    thread_id = _get_thread_id(config)
    runtime = _thread_runtime.setdefault(thread_id, {})

    budget = None
    if config and isinstance(config.get("configurable"), dict):
        budget = config["configurable"].get("budget")

    if not _allow_next_budgeted_request(budget):
        return {"messages": [AIMessage(content="已按要求停止；当前科研任务未因此被标记完成。") ]}

    msgs = _sanitize_messages(state["messages"])

    previous_tool_count = runtime.get("_printed_tool_messages", 0)
    current_tool_messages = [msg for msg in state["messages"][previous_tool_count:] if isinstance(msg, ToolMessage)]
    for msg in current_tool_messages:
        try:
            msg.pretty_print()
        except UnicodeEncodeError:
            pass

    token_count = 0
    res = None
    use_stream = _llm_stream_enabled(config)
    # 单步 LLM 调用硬上限，避免某一次调用耗尽整个样本超时。
    # 流式场景默认用于 Web UI，不在这里强制超时；评测 deepseek-v4-flash 默认非流式。
    configurable = config.get("configurable", {}) if config else {}
    configured_step_timeout = (
        configurable.get("llm_step_timeout")
        if isinstance(configurable, dict)
        else None
    )
    llm_step_timeout = float(
        configured_step_timeout
        if configured_step_timeout is not None
        else os.environ.get("ORION_LLM_STEP_TIMEOUT", "180")
    )
    max_retries = _llm_request_max_retries(config)
    transport_retries = 0
    for attempt in range(max_retries + 1):
        try:
            if use_stream:
                chunks = []
                async for chunk in llm_with_tools.astream(msgs):
                    chunks.append(chunk)
                res = chunks[0] if len(chunks) == 1 else sum(chunks[1:], chunks[0])
            else:
                res = await asyncio.wait_for(
                    llm_with_tools_nonstream.ainvoke(msgs), timeout=llm_step_timeout
                )
            token_count = res.usage_metadata.get("total_tokens", 0) if res.usage_metadata else 0
            transport_retries = attempt
            print(token_count)
            break
        except asyncio.TimeoutError:
            print(f"[LLM 单步超时] {llm_step_timeout}s 内无响应")
            if attempt < max_retries:
                print("[LLM 重试] 单步超时，正在重试一次...")
                continue
            timeout_error = asyncio.TimeoutError(
                f"LLM request timed out after {max_retries + 1} attempt(s)"
            )
            _annotate_llm_request_failure(
                timeout_error,
                transport_retries=attempt,
                max_retries=max_retries,
            )
            raise timeout_error
        except Exception as e:
            err_msg = str(e)
            if "finish_reason" in err_msg and "length" in err_msg:
                print("[截断托底] 检测到输出截断，正在续写...")
                continuation_prompt = internal_human_message(content="你的上一次回复被截断了，请继续输出剩余内容，不要重复已经输出的部分。")
                if use_stream:
                    chunks = []
                    async for chunk in llm_with_tools.astream(msgs + [continuation_prompt]):
                        chunks.append(chunk)
                    res = chunks[0] if len(chunks) == 1 else sum(chunks[1:], chunks[0])
                else:
                    res = await asyncio.wait_for(
                        llm_with_tools_nonstream.ainvoke(msgs + [continuation_prompt]),
                        timeout=llm_step_timeout,
                    )
                token_count = res.usage_metadata.get("total_tokens", 0) if res.usage_metadata else 0
                break
            # 流式停滞/连接瞬断（request_timeout 触发后也归入此类）重试一次
            if attempt < max_retries and _is_retryable_llm_transport_error(e):
                print("[LLM 重试] 流式响应停滞或连接中断，正在重试一次...")
                continue
            _annotate_llm_request_failure(
                e,
                transport_retries=attempt,
                max_retries=max_retries,
            )
            raise

    if budget is not None:
        # 每个响应的 usage 是本次请求用量，不是会话累计值；不在响应写入前暂停。
        budget.record_response(token_count if res.usage_metadata and "total_tokens" in res.usage_metadata else None)

    try:
        res.pretty_print()
    except UnicodeEncodeError:
        pass
    runtime["_printed_tool_messages"] = len(state["messages"])
    runtime["token_count"] = token_count
    return {
        "messages": [res],
        "llm_request_max_retries": max_retries,
        "llm_transport_retry_count": int(
            state.get("llm_transport_retry_count") or 0
        )
        + transport_retries,
    }


async def aclose_agent(agent) -> None:
    """关闭 agent checkpointer 持有的 aiosqlite 连接。

    aiosqlite 工作线程是非守护线程；CPython 解释器退出时先 join 非守护
    线程、再执行 atexit 钩子——钩子永远等不到执行机会，进程因此挂死。
    所以必须在事件循环结束前（agent 使用完毕后）由调用方显式关闭连接。
    """
    checkpointer = getattr(agent, "checkpointer", None)
    conn = getattr(checkpointer, "conn", None)
    close = getattr(conn, "close", None)
    if close is None:
        return
    try:
        if inspect.iscoroutinefunction(close):
            await close()
        else:
            close()
    except Exception:
        pass


def _runtime_tools_for_ablation() -> list[Any]:
    """按评测消融环境变量返回实际暴露给 Agent 的工具集。

    该函数仅在构建评测 Agent 时读取环境变量；正常交互未设置变量时原样返回。
    禁用结构化 KG 不会移除文献向量检索，禁用配方工具也保留最终报告工具，
    使每个消融只改变一个研究变量。
    """
    active = list(_BASE_TOOLS)
    blocked: set[str] = set()
    if os.environ.get("ORION_ABLATION_NO_KG") == "1":
        blocked.update({"kg_query", "explore_entity_neighborhood", "aggregate_evidence_by_entities"})
    if os.environ.get("ORION_ABLATION_NO_FORMULATION") == "1":
        blocked.update({"query_formulation_space", "screen_formulations", "optimize_formulations"})
    return [item for item in active if getattr(item, "name", "") not in blocked]


def _configure_runtime_tools() -> None:
    """让 LLM、ToolNode 与上下文装配使用同一份消融后的工具集。"""
    global tools, llm_with_tools, llm_with_tools_nonstream
    tools = _runtime_tools_for_ablation()
    llm_with_tools = main_model.bind_tools(tools)
    llm_with_tools_nonstream = main_model_nonstream.bind_tools(tools)


def _build_graph(llm_node, *, rebuild_memory: bool = True):
    # 显式文献辅助推荐试验：检索→读原文→论文内补证→回答→独立语义审查→反馈/补检索→修订。
    # 不进入旧字段覆盖门禁；科学适用性由本轮原文上的独立模型审查处理，默认工作流仍供历史回放。
    if os.environ.get("ORION_DESIGN_REASONING_MODE") == "literature":
        if os.environ.get("ORION_AUTO_APPROVE") != "1":
            raise ValueError("literature 试验入口需显式开启测试自动审批")
        from orion.core.literature_design_workflow import build_literature_design_graph
        model = main_model_nonstream.bind(max_tokens=32768, reasoning_effort="low",
                                          response_format={"type": "json_object"})
        review_model = get_grounding_model()
        return build_literature_design_graph(model, tools, review_model=review_model)
    if (_tool_residency_bridge is not None
            and _tool_residency_bridge.shared_runtime is not None):
        from orion.core.resource_tool_node import LeaseSafeToolNode
        raw_tools_node = LeaseSafeToolNode(tools)
    else:
        raw_tools_node = ToolNode(tools)

    def finish(command, state, config):
        if command.goto != "grounding":
            return command
        merged = _state_with_update(state, command.update or {})
        delivered = deliver_terminal_result(merged, config)
        return Command(update=_merge_updates(command.update, delivered.update), goto="__end__")

    def before_tools(state, config):
        command = pre_tool_gate(state)
        if command.goto == "tools":
            return command.update
        return finish(command, state, config)

    def after_llm(state, config):
        last = current_task_messages(state)[-1]
        if _incomplete_response_reason(last):
            return finish(review(state), state, config)
        if isinstance(last, AIMessage) and last.tool_calls:
            return Command(goto="tools")
        return finish(review(state), state, config)

    def after_tools(state, config):
        # 科研工具结果交回模型。通用循环不自动运行领域决策或安排下一步科研行动。
        # 旧报告工具的显式终端提交继续沿用原成对交付协议。
        command = Command(goto="grounding" if terminal_result(state) is not None else "llm")
        latest_call = next((m for m in reversed(current_task_messages(state))
                            if isinstance(m, AIMessage) and m.tool_calls), None)
        rejected = latest_call.additional_kwargs.get("orion_rejected_tools") if latest_call else None
        if rejected:
            command = Command(goto=command.goto, update=_merge_updates(command.update,
                {"messages": [internal_human_message(content="（系统提醒）本批次部分工具未获批准：\n"
                                                         + "\n".join(rejected))]}))
        return finish(command, state, config)

    if inspect.iscoroutinefunction(llm_node):
        async def tools_node(state: GraphState, config: RunnableConfig | None = None):
            bridge = _tool_residency_bridge
            output = (await raw_tools_node.ainvoke(state, config) if bridge is None else
                      await bridge.ainvoke(raw_tools_node, state, config,
                                           request_id="main-agent-toolnode"))
            return capture_design_tool_batch(state, output)
    else:
        def tools_node(state: GraphState, config: RunnableConfig | None = None):
            bridge = _tool_residency_bridge
            output = (raw_tools_node.invoke(state, config) if bridge is None else
                      bridge.invoke(raw_tools_node, state, config,
                                    request_id="main-agent-toolnode"))
            return capture_design_tool_batch(state, output)

    def topology(*, legacy=False):
        graph = StateGraph(GraphState)
        context_prepare = (context_prepare_async if inspect.iscoroutinefunction(llm_node)
                           else context_prepare_sync)
        graph.add_node("context_prepare", context_prepare)
        graph.add_node("llm", hookify("llm", after=(after_llm,))(llm_node),
                       destinations=("tools", "llm", "__end__"))
        graph.add_node("tools", hookify("tools", before=(before_tools,), after=(after_tools,),
                                        persist_before=True)(tools_node), destinations=("llm", "__end__"))
        graph.add_edge("__start__", "context_prepare")
        graph.add_edge("context_prepare", "llm")
        if legacy:
            # 仅旧挂起 checkpoint 使用旧节点名；默认图始终只有三个节点。
            graph.add_node("pre_tool_gate", pre_tool_gate)
            graph.add_node("review", review)
            graph.add_node("grounding", deliver_terminal_result)
        return graph

    os.makedirs("short_memory", exist_ok=True)
    try:
        asyncio.get_running_loop()
        has_async_loop = True
    except RuntimeError:
        has_async_loop = False
    if has_async_loop:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        conn = aiosqlite.connect("short_memory/sessions.db")
        checkpointer = AsyncSqliteSaver(conn=conn, serde=serde)
        # aiosqlite 工作线程是非守护线程，进程退出前必须由调用方
        # 通过 aclose_agent() 显式关闭连接（atexit 来不及，见函数 docstring）。
    else:
        conn = sqlite3.connect("short_memory/sessions.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn, serde=serde)

    if rebuild_memory:
        global _memory_service, _context_manager
        _memory_service = MemoryService(short_term=ShortTermMemory(checkpointer))
        _context_manager = ContextManager(memory_service=_memory_service)

    from orion.core.graph_resume import ResumeCompatibleGraph
    primary = topology().compile(checkpointer=checkpointer)
    graph = ResumeCompatibleGraph(primary,
        lambda: topology(legacy=True).compile(checkpointer=checkpointer))
    from orion.core.resource_task_graph import bind_resource_task_graph
    return bind_resource_task_graph(graph, _tool_residency_bridge, role="main")


def build_agent():
    _configure_runtime_tools()
    return _build_graph(llm)


def build_streaming_agent():
    _configure_runtime_tools()
    return _build_graph(llm_async)


def write_graph_png(agent) -> None:
    """Draw the compiled graph to graph.png, with a workaround for LangGraph
    <1.3 sorting edges that contain None labels.
    """
    import builtins

    orig_sorted = builtins.sorted

    def _safe_sorted(iterable, **kwargs):
        if "key" not in kwargs and iterable:
            first = next(iter(iterable))
            if isinstance(first, tuple):
                kwargs["key"] = lambda x: tuple(
                    "" if item is None else str(item) for item in x
                )
        return orig_sorted(iterable, **kwargs)

    builtins.sorted = _safe_sorted
    try:
        png_data = agent.get_graph().draw_mermaid_png()
        with open("./graph.png", "wb") as f:
            f.write(png_data)
    finally:
        builtins.sorted = orig_sorted


def exit_handler(sig, frame):
    print("\n已退出")
    sys.exit(0)


def read_user_input(prompt: str = "你> ") -> str | None:
    try:
        if PromptSession is None:
            return input(prompt)
        try:
            session = PromptSession()
        except NoConsoleScreenBufferError:
            # 非 Windows 原生控制台环境（如 PyCharm 运行窗口）降级为内置 input
            return input(prompt)
        if HTML is not None:
            prompt_text = HTML('<prompt>你</prompt><hint> › </hint>') if prompt == "你> " else HTML('<prompt>审批</prompt><hint> › </hint>')
        else:
            prompt_text = prompt
        return session.prompt(
            prompt_text,
            placeholder="在这里输入，回车发送" if HTML is not None else None,
            bottom_toolbar="Enter 发送 · Ctrl+C 退出 · Ctrl+D 结束" if HTML is not None else None,
            style=PROMPT_STYLE,
        )
    except EOFError:
        return input(prompt)


def _print_runtime_resource_snapshot() -> None:
    """Print managed runtime resources without materializing inactive entries."""

    runtime = _default_resource_runtime
    if runtime is None:
        print("运行资源池未启用。")
        return
    rows = runtime.snapshot()
    if not rows:
        print("运行资源目录为空。")
        return
    print("运行资源：")
    for row in rows:
        idle = row.get("idle_seconds")
        idle_text = "-" if idle is None else f"{float(idle):.1f}s"
        measured = row.get("measured_cost") or {}
        host_ram = float(measured.get("host_ram_mb") or 0.0)
        print(
            f"- {row['resource_id']}: resident={row['resident']}, "
            f"active={row['active_count']}, idle={idle_text}, "
            f"pid={row.get('instance_pid')}, rss={host_ram:.1f} MiB, "
            f"prepare={row.get('last_prepare_seconds')}"
        )


def _print_local_capability_events() -> None:
    """在 CLI turn 边界扫描并提示本地 capability 目录变化。"""

    try:
        events = scan_local_capabilities()
    except Exception as exc:
        print(f"[capability] 本地能力扫描失败：{exc}", flush=True)
        return
    for event in events:
        capability_id = event.capability_id or event.package_dir.name
        if event.event_type.value == "discovered":
            print(
                f"[capability] 发现新能力 {capability_id}。"
                f"使用 /cap attach {capability_id} 审核并连接。",
                flush=True,
            )
        elif event.event_type.value == "updated":
            print(
                f"[capability] 能力 {capability_id} 的本地文件已变化。"
                f"使用 /cap attach {capability_id} 审核并热更新。",
                flush=True,
            )
        elif event.event_type.value == "removed":
            print(f"[capability] 能力 {capability_id} 已从目录移除并断开。", flush=True)
        else:
            print(
                f"[capability] {capability_id} 无法载入 descriptor：{event.message}",
                flush=True,
            )


def _print_local_capability_list() -> None:
    """打印本地 capability descriptor/连接状态，不执行未批准代码。"""

    rows = local_capability_snapshot(scan=True)
    if not rows:
        manager = get_local_capability_manager()
        print(f"本地能力目录为空：{manager.root}")
        return
    print("本地能力：")
    for row in rows:
        active = row.get("active_version")
        version_text = (
            f"active={active}, package={row['version']}"
            if active and active != row["version"]
            else f"version={row['version']}"
        )
        generation = row.get("learned_generation")
        generation_text = "" if generation is None else f", generation={generation}"
        print(
            f"- {row['id']}: {row['status']} ({version_text}{generation_text})\n"
            f"  {row['summary']}"
        )


async def _handle_capability_cli_command(user_text: str) -> bool:
    """处理 /cap list|attach|detach；返回是否已消费该条 CLI 输入。"""

    if not (user_text == "/cap" or user_text.startswith("/cap ")):
        return False
    parts = user_text.split()
    action = parts[1].lower() if len(parts) >= 2 else "list"
    if action in {"list", "ls"}:
        _print_local_capability_list()
        return True
    if action not in {"attach", "detach"} or len(parts) != 3:
        print("用法：/cap list | /cap attach <capability_id> | /cap detach <capability_id>")
        return True

    capability_id = parts[2]
    if action == "detach":
        try:
            detach_local_capability(capability_id)
        except Exception as exc:
            print(f"[capability] 断开失败：{exc}")
        else:
            print(f"[capability] 已断开 {capability_id}；主 Agent 图未重编译。")
        return True

    # attach/update 前先只读扫描 descriptor，再显示权限并要求一次显式确认。
    scan_local_capabilities()
    manager = get_local_capability_manager()
    try:
        record = manager.get(capability_id)
    except KeyError as exc:
        print(f"[capability] {exc}")
        return True
    print(
        f"准备连接 capability：{record.manifest.capability_id}\n"
        f"版本：{record.manifest.version}\n"
        f"说明：{record.manifest.summary}\n"
        f"权限：{', '.join(record.manifest.permissions) if record.manifest.permissions else '未声明额外权限'}\n"
        f"来源：{record.package_dir}"
    )
    approval = await asyncio.to_thread(
        read_user_input,
        "确认执行这个本地 capability 的代码？输入 y 确认，其他内容取消：",
    )
    if approval is None or approval.strip().lower() not in {"y", "yes"}:
        print("[capability] 已取消。")
        return True
    try:
        swap = attach_local_capability(capability_id)
    except Exception as exc:
        print(f"[capability] 连接/更新失败，旧能力（若存在）保持原状：{exc}")
        return True
    if swap is None:
        print(f"[capability] 已热挂载 {capability_id}；当前 Agent 可立即通过 capability gateway 使用。")
    else:
        print(
            f"[capability] 已热更新 {capability_id}: "
            f"{swap.previous_version} -> {swap.active_version}; "
            f"learned generation {swap.generation_before} -> {swap.generation_after}。"
        )
    return True


async def run_cli() -> None:
    signal.signal(signal.SIGINT, exit_handler)
    agent = build_streaming_agent()
    write_graph_png(agent)
    config = {"configurable": {"thread_id": "1"}}

    print("Orion CLI 已启动。输入 /exit 退出。")
    try:
        while True:
            # 把目录变化限制在 turn 边界处理，避免 watcher 线程在 capability 正执行时竞态卸载。
            _print_local_capability_events()
            user_text = await asyncio.to_thread(read_user_input)
            if user_text is None:
                print("检测到非交互输入，已退出。")
                return

            user_text = user_text.strip()
            if not user_text:
                continue
            if user_text in {"/exit", "exit", "quit"}:
                print("已退出")
                return
            if user_text in {"/resources", "/resource", "/res"}:
                _print_runtime_resource_snapshot()
                continue
            if await _handle_capability_cli_command(user_text):
                continue

            session_id = config.get("configurable", {}).get("thread_id", "cli")
            res: dict | None = None
            try:
                res = await agent.ainvoke(input={"messages": [HumanMessage(content=user_text)]}, config=config)
                while res.get("__interrupt__"):
                    approval = await asyncio.to_thread(read_user_input, f"审批请求：{res['__interrupt__']}\n审批意见：")
                    if approval is None:
                        print("检测到非交互输入，审批已取消。")
                        return
                    approval = approval.strip()
                    if approval in {"/exit", "exit", "quit"}:
                        print("已退出")
                        return
                    res = await agent.ainvoke(Command(resume=approval), config=config)
                _log_tool_failures(session_id, res)
            except Exception:
                if isinstance(res, dict):
                    _log_tool_failures(session_id, res)
                raise
            await _auto_reflect(session_id=session_id, state=res)
    finally:
        # 必须在事件循环结束前关闭 aiosqlite 连接，否则进程无法退出。
        await aclose_agent(agent)


if __name__ == "__main__":
    asyncio.run(run_cli())
