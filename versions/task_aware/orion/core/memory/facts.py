from __future__ import annotations



import json

import os

import sqlite3

from datetime import datetime, timezone

from pathlib import Path

from typing import Any



from langchain_core.tools import tool





_DEEPAGENTS_DIR = Path(__file__).resolve().parents[2]

DB_PATH = Path(os.environ.get("LTM_DB_PATH", _DEEPAGENTS_DIR / "data" / "long_term_memory.db"))





class FactsStore:

    """基于 SQLite 的结构化事实与 episode 存储，供 memory_search / memory_write 工具使用。"""



    def __init__(self, db_path: str | Path | None = None):

        self.db_path = Path(db_path) if db_path else DB_PATH

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_schema()



    def _conn(self):

        conn = sqlite3.connect(self.db_path, check_same_thread=False)

        conn.row_factory = sqlite3.Row

        return conn



    def _init_schema(self):

        with self._conn() as conn:

            conn.execute("""

                CREATE TABLE IF NOT EXISTS episodes (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    session_id TEXT NOT NULL,

                    run_id TEXT NOT NULL,

                    summary TEXT NOT NULL,

                    entities TEXT,

                    created_at TEXT NOT NULL

                )

            """)

            conn.execute("""

                CREATE TABLE IF NOT EXISTS facts (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    session_id TEXT NOT NULL,

                    run_id TEXT NOT NULL,

                    entity TEXT NOT NULL,

                    relation TEXT NOT NULL,

                    value TEXT NOT NULL,

                    confidence TEXT,

                    source TEXT,

                    created_at TEXT NOT NULL

                )

            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_search ON episodes(summary, entities)")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_search ON facts(entity, relation, value)")



    def write_episode(

        self,

        session_id: str,

        run_id: str,

        summary: str,

        entities: list[str] | None = None,

    ) -> int:

        with self._conn() as conn:

            cur = conn.execute(

                "INSERT INTO episodes (session_id, run_id, summary, entities, created_at) VALUES (?, ?, ?, ?, ?)",

                (

                    session_id,

                    run_id,

                    summary,

                    json.dumps(entities or [], ensure_ascii=False),

                    datetime.now(timezone.utc).isoformat(),

                ),

            )

            return cur.lastrowid



    def write_fact(

        self,

        session_id: str,

        run_id: str,

        entity: str,

        relation: str,

        value: str,

        confidence: str = "medium",

        source: str = "",

    ) -> int:

        with self._conn() as conn:

            cur = conn.execute(

                "INSERT INTO facts (session_id, run_id, entity, relation, value, confidence, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",

                (

                    session_id,

                    run_id,

                    entity,

                    relation,

                    value,

                    confidence,

                    source,

                    datetime.now(timezone.utc).isoformat(),

                ),

            )

            return cur.lastrowid



    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:

        keywords = [w.strip() for w in query.split() if w.strip()]

        if not keywords:

            return []

        like_clauses = " OR ".join(["summary LIKE ? OR entities LIKE ?"] * len(keywords))

        params = []

        for kw in keywords:

            params.extend([f"%{kw}%", f"%{kw}%"])



        with self._conn() as conn:

            rows = conn.execute(

                f"SELECT * FROM episodes WHERE {like_clauses} ORDER BY created_at DESC LIMIT ?",

                (*params, k),

            ).fetchall()

            results = [dict(r) for r in rows]

            remaining = max(0, k - len(results))



            if remaining > 0:

                fact_clauses = " OR ".join(["entity LIKE ? OR relation LIKE ? OR value LIKE ?"] * len(keywords))

                fact_params = []

                for kw in keywords:

                    fact_params.extend([f"%{kw}%"] * 3)

                fact_rows = conn.execute(

                    f"SELECT * FROM facts WHERE {fact_clauses} ORDER BY created_at DESC LIMIT ?",

                    (*fact_params, remaining),

                ).fetchall()

                results.extend([dict(r) for r in fact_rows])



        return results[:k]





_default_facts = FactsStore()





@tool

def memory_search(query: str, k: int = 5) -> list[dict[str, Any]]:

    """按关键词搜索长期记忆中的 episodes 和 facts。"""

    try:

        return _default_facts.search(query, k=k)

    except Exception as e:

        return [{"error": f"记忆搜索失败：{e}"}]





@tool

def memory_write(

    entity: str,

    relation: str,

    value: str,

    confidence: str = "medium",

    source: str = "",

    session_id: str = "",

    run_id: str = "",

) -> str:

    """把学习到的事实写入长期记忆。"""

    try:

        _default_facts.write_fact(

            session_id=session_id,

            run_id=run_id,

            entity=entity,

            relation=relation,

            value=value,

            confidence=confidence,

            source=source,

        )

        return f"已记录事实：{entity} {relation} {value}"

    except Exception as e:

        return f"记忆写入失败：{e}"

