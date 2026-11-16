from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from orion.core.reflection_entry import ReflectionEntry


class ReflectionStore:
    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = os.environ.get("ORION_REFLECTION_STORE_PATH")
        if db_path is None:
            db_path = Path(__file__).resolve().parents[2] / "data" / "wiki.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reflection_entries (
                    reflection_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    domain_tags TEXT,
                    task_types TEXT,
                    status TEXT NOT NULL,
                    observed_count INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    superseded_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reflection_query
                ON reflection_entries(status, task_types, domain_tags)
            """)

    def add_or_update(self, entry: ReflectionEntry) -> ReflectionEntry:
        """新增或更新 reflection；已存在时累加 observed_count 与 confidence。"""
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT payload FROM reflection_entries WHERE reflection_id = ?",
                (entry.reflection_id,),
            ).fetchone()
            if existing:
                old = ReflectionEntry.model_validate_json(existing[0])
                entry = old.model_copy(update={
                    "observed_count": old.observed_count + 1,
                    "confidence": min(1.0, old.confidence + 0.1),
                    "supporting_claim_ids": list(set(old.supporting_claim_ids + entry.supporting_claim_ids)),
                    "supporting_run_ids": list(set(old.supporting_run_ids + entry.supporting_run_ids)),
                })
            conn.execute(
                """INSERT OR REPLACE INTO reflection_entries
                   (reflection_id, payload, domain_tags, task_types, status, observed_count, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.reflection_id,
                    entry.model_dump_json(),
                    " ".join(entry.domain_tags),
                    " ".join(entry.task_types),
                    entry.status,
                    entry.observed_count,
                    entry.confidence,
                ),
            )
        return entry

    def query(
        self,
        task_types: list[str] | None = None,
        domain_tags: list[str] | None = None,
        status: str = "active",
        min_confidence: float = 0.3,
        limit: int = 5,
    ) -> list[ReflectionEntry]:
        """按任务类型、领域标签、状态和置信度查询 active reflection。"""
        with sqlite3.connect(self.db_path) as conn:
            sql = "SELECT payload FROM reflection_entries WHERE status = ? AND confidence >= ?"
            params = [status, min_confidence]
            if task_types:
                sql += " AND (" + " OR ".join(["task_types LIKE ?"] * len(task_types)) + ")"
                params.extend(f"%{t}%" for t in task_types)
            if domain_tags:
                sql += " AND (" + " OR ".join(["domain_tags LIKE ?"] * len(domain_tags)) + ")"
                params.extend(f"%{t}%" for t in domain_tags)
            sql += " ORDER BY observed_count DESC, confidence DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [ReflectionEntry.model_validate_json(r[0]) for r in rows]

    def supersede(self, reflection_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload FROM reflection_entries WHERE reflection_id = ?",
                (reflection_id,),
            ).fetchone()
            if row is None:
                return
            entry = ReflectionEntry.model_validate_json(row[0])
            entry = entry.model_copy(update={"status": "superseded", "superseded_at": now})
            conn.execute(
                """INSERT OR REPLACE INTO reflection_entries
                   (reflection_id, payload, domain_tags, task_types, status, observed_count, confidence, superseded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.reflection_id,
                    entry.model_dump_json(),
                    " ".join(entry.domain_tags),
                    " ".join(entry.task_types),
                    entry.status,
                    entry.observed_count,
                    entry.confidence,
                    entry.superseded_at,
                ),
            )
