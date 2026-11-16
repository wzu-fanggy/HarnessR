import json



from langchain_core.runnables import RunnableConfig

from langchain_core.tools import tool



from orion.task_evidence.contracts_v2 import expected_skill_for_task_type, task_type_for_skill

from tools.skill_loader import load_skill as _load_skill_file





def _skill_route_error(name: str, config: RunnableConfig | None) -> dict | None:

    """显式任务合同约束主 Skill 路由，辅助 Skill 可按需组合。"""



    configurable = config.get("configurable", {}) if config else {}

    if not isinstance(configurable, dict):

        return None

    task_type = configurable.get("task_type")

    expected = expected_skill_for_task_type(task_type) if task_type else None

    if expected is None or name == expected or task_type_for_skill(name) is None:

        return None

    return {

        "error": "skill_route_mismatch",

        "task_type": str(task_type),

        "requested_skill": name,

        "expected_skill": expected,

        "recoverable": True,

        "instruction": f"Call load_skill again with name={expected!r}.",

    }





@tool

def load_skill(name: str, config: RunnableConfig) -> str:

    """加载指定 skill 的完整内容。name 为 skill 名称，如 "literature_review"。

    优先从 evolution store 读取 evolved 版本，不存在时回退到 skills/ 目录文件。

    返回该 Skill 的完整领域方法；Agent 依据任务与证据按需选择能力，不强制工具序列。"""

    route_error = _skill_route_error(name, config)

    if route_error is not None:

        return json.dumps(route_error, ensure_ascii=False, sort_keys=True)



    from orion.main import load_skill_from_store



    store_content = load_skill_from_store(name)

    if store_content is not None:

        return store_content

    return _load_skill_file(name)

