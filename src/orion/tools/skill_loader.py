"""

Skill 加载器：扫描 skills/ 目录下的 .md 文件，

解析 frontmatter 元信息，提供索引和按需加载能力。

"""



import os

import re

from pathlib import Path



import yaml



SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"





def _parse_frontmatter(text: str) -> tuple[dict, str]:

    """解析 markdown frontmatter，返回 (元信息字典, 正文)。"""

    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)

    if not m:

        return {}, text

    try:

        meta = yaml.safe_load(m.group(1).strip()) or {}

    except Exception:

        meta = {}

    return meta, m.group(2).strip()





def get_skill_index() -> list[dict]:

    """扫描所有 skill 文件，返回索引列表 [{name, description, file, path}]。



    只识别名为 ``SKILL.md`` 的主文件，忽略 ``assets/``、``references/``

    等目录下的模板与参考资料，避免污染系统提示词。

    """

    index = []

    if not SKILLS_DIR.exists():

        return index

    for f in sorted(SKILLS_DIR.rglob("SKILL.md")):

        text = f.read_text(encoding="utf-8")

        meta, _ = _parse_frontmatter(text)

        # 相对 skills 目录的路径，如 "kg-retrieval/SKILL.md"

        rel_path = f.relative_to(SKILLS_DIR).as_posix()

        index.append({

            "name": meta.get("name", f.stem),

            "description": meta.get("description", ""),

            "triggers": meta.get("triggers", []),

            "file": f.name,

            "path": rel_path,

        })

    return index





def _find_skill_file(name: str) -> Path | None:

    """按 skill 名或 frontmatter name 查找文件。"""

    if not SKILLS_DIR.exists():

        return None

    for f in SKILLS_DIR.rglob("*.md"):

        if f.stem == name:

            return f

    for f in SKILLS_DIR.rglob("*.md"):

        text = f.read_text(encoding="utf-8")

        meta, _ = _parse_frontmatter(text)

        if meta.get("name") == name:

            return f

    return None





def load_skill(name: str) -> str:

    """按 skill 名称或路径加载完整内容。"""

    f = _find_skill_file(name)

    if f is None:

        return f"未找到 skill: {name}"

    return f.read_text(encoding="utf-8")





def get_skill_workflow(name: str) -> list[dict] | None:

    """返回 skill 声明的 workflow.steps；无 workflow 或 skill 不存在时返回 None。"""

    f = _find_skill_file(name)

    if f is None:

        return None

    text = f.read_text(encoding="utf-8")

    meta, _ = _parse_frontmatter(text)

    workflow = meta.get("workflow")

    if not workflow:

        return None

    return workflow.get("steps") or []





def format_skill_index_for_system() -> str:

    """生成注入系统提示词的 skill 索引文本。"""

    index = get_skill_index()

    if not index:

        return ""

    lines = ["## 可用 Skills", "按需调用 load_skill 工具加载完整步骤。", ""]

    for s in index:

        triggers = ", ".join(s.get("triggers", []))

        lines.append(f"- **{s['name']}**: {s['description']}  (`{s['path']}`)  triggers: [{triggers}]")

    return "\n".join(lines)

