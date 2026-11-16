"""任务系统（Learn Claude Code s12 的领域化版本）。

对外只需要两样东西：

- ``TaskBoard``：业务 API（create / claim / complete / escalate / record_failure）
- ``TASK_TOOLS``：给模型用的 LangChain 工具列表

存储落在 ``.tasks/``，可用环境变量 ``ORION_TASKS_DIR`` 改位置。
"""

from .board import TaskBoard
from .failures import TOOL_STATUS_MAP, FailureKind, FailurePolicy, classify, infer_kind
from .models import Task, TaskStatus, now_iso
from .store import TaskStore
from .tools import TASK_TOOLS, get_board, set_board

__all__ = [
    "Task",
    "TaskStatus",
    "TaskStore",
    "TaskBoard",
    "FailureKind",
    "FailurePolicy",
    "TOOL_STATUS_MAP",
    "classify",
    "infer_kind",
    "TASK_TOOLS",
    "get_board",
    "set_board",
    "now_iso",
]
