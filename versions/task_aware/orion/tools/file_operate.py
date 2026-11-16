import os

from pathlib import Path



from langchain_core.tools import tool



WORKDIR = Path(__file__).resolve().parent.parent



def safe_path(p: str) -> Path:

    path = (WORKDIR / p).resolve()

    if not path.is_relative_to(WORKDIR):

        raise ValueError(f"Path escapes workspace: {p}")

    return path

@tool

def write_file(path,content):

    """

    Write content to file.

    :param path:文件路径

    :return:提示信息

    """

    try:

        fp = safe_path(path)

        fp.parent.mkdir(parents=True, exist_ok=True)

        fp.write_text(content,encoding="utf-8")

        return f"Wrote {len(content)} bytes to {path}"

    except Exception as e:

        return f"Error: {e}"

@tool

def read_file(path: str, limit: int = None) -> str:

    """

    Read file contents.

    :param path: 文件路径

    :param limit: 读取行数限制

    :return:文件内容/提示信息

    """

    try:

        text = safe_path(path).read_text(encoding="utf-8")

        lines = text.splitlines()

        if limit and limit < len(lines):

            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]

        return "\n".join(lines)[:50000]

    except Exception as e:

        return f"Error: {e}"