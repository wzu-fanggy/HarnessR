from typing import Annotated, Literal



from langchain_core.tools import tool

from langgraph.prebuilt import InjectedState

from pydantic import BaseModel





class TodoItem(BaseModel):

    id: str

    content: str

    status:  Literal["pending", "in_progress", "completed"] = "pending"





@tool

def write_todos(todos: list[TodoItem]):

    """

    替换当前任务列表。传入完整且有序的任务列表（包括已完成的）。

    每个任务必须有 id、content 和 status（pending/in_progress/completed 之一）。

    同一时间有且只有一个任务处于 in_progress 状态。

    所有任务都完成时，全部设为 completed。

    :param todos: list[TodoItem]

    """

    for t in todos:

        if t.status not in ["pending", "in_progress", "completed"]:

            return {"todo_list": [], "error": f"任务 {t.id} 状态非法"}

    return {"todo_list": todos}





@tool

def read_todos(state: Annotated[dict, InjectedState]) -> str:

    """

    返回当前任务列表。应由 Agent 在需要了解计划进度时调用。

    """

    todos = state.get("todo_list", [])

    if not todos:

        return "当前没有任务。"

    lines = []

    for t in todos:

        if isinstance(t, TodoItem):

            lines.append(f"[{t.status}] {t.id}: {t.content}")

        elif isinstance(t, dict):

            lines.append(f"[{t.get('status')}] {t.get('id')}: {t.get('content')}")

        else:

            lines.append(str(t))

    return "\n".join(lines)

