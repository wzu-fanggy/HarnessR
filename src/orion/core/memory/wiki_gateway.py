from __future__ import annotations
from release_paths import release_path

import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


_DEEPAGENTS_ROOT = Path(os.environ.get("DEEPAGENTS_ROOT", release_path('${ORION_SOURCE}')))
WIKI_ROOT = _DEEPAGENTS_ROOT / "wiki"

_index_lock = threading.Lock()


def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^\w\-_.]", "_", s)[:80]


def _ensure_dirs():
    (WIKI_ROOT / "entities").mkdir(parents=True, exist_ok=True)
    (WIKI_ROOT / "concepts").mkdir(parents=True, exist_ok=True)
    (WIKI_ROOT / "reflections").mkdir(parents=True, exist_ok=True)


def _wiki_files() -> list[Path]:
    if not WIKI_ROOT.exists():
        return []
    files = []
    for subdir in ("entities", "concepts", "reflections"):
        files.extend((WIKI_ROOT / subdir).glob("*.md"))
    return files


def _score_page(text: str, query: str) -> float:
    query_terms = [t.lower() for t in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", query)]
    if not query_terms:
        return 0.0
    text_lower = text.lower()
    title = ""
    for line in text.splitlines()[:5]:
        if line.strip().startswith("#"):
            title = line.lstrip("#").strip().lower()
            break
    score = 0.0
    for term in query_terms:
        if term in title:
            score += 3.0
        score += text_lower.count(term) * 0.5
    return score


class WikiGateway:
    """基于本地 Markdown 文件的轻量级 Wiki 网关。

    实体页写入 ``wiki/entities/``，概念页写入 ``wiki/concepts/``，
    反思笔记由 ``wiki_reflection`` 写入 ``wiki/reflections/``。
    搜索时按关键词对标题和内容做简单打分召回。
    """

    async def search_wiki(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        scored = []
        for path in _wiki_files():
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            score = _score_page(text, query)
            if score <= 0:
                continue
            title = ""
            for line in text.splitlines()[:10]:
                if line.strip().startswith("#"):
                    title = line.lstrip("#").strip()
                    break
            if not title:
                title = path.stem
            scored.append(
                {
                    "title": title,
                    "path": str(path.relative_to(_DEEPAGENTS_ROOT)),
                    "content": text[:2000],
                    "score": score,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    async def write_wiki(
        self,
        title: str,
        content: str,
        source_run_id: str,
        category: str = "entities",
    ) -> str:
        """写入一篇 Wiki 页面并返回页面路径。"""
        _ensure_dirs()
        if category not in ("entities", "concepts", "reflections"):
            category = "entities"

        safe_title = _sanitize_filename(title)
        filename = f"{safe_title}.md"
        page_path = WIKI_ROOT / category / filename

        header = f"# {title}\n\n"
        meta = (
            f"**Source run:** {source_run_id}\n\n"
            f"**Created at:** {datetime.now().isoformat()}\n\n"
        )
        full_content = header + meta + content
        page_path.write_text(full_content, encoding="utf-8")
        return str(page_path.relative_to(_DEEPAGENTS_ROOT))
