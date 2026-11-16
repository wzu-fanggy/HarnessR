import asyncio

import json

import os

import sqlite3

import sys

from datetime import datetime, timezone

from pathlib import Path



DB_PATH = Path(

    os.environ.get(

        "AUDIT_DB_PATH",

        Path(__file__).resolve().parents[3] / "data" / "audit.db",

    )

)



# Serialize all writes to the SQLite database so concurrent WebSocket

# connections do not contend for the same write lock.

_write_lock = asyncio.Lock()





def _open_db(db_path: Path):

    conn = sqlite3.connect(db_path, check_same_thread=False)

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA busy_timeout = 5000")

    return conn





def _ensure_columns(

    conn: sqlite3.Connection, table: str, columns: dict[str, str]

) -> None:

    """为已有表补充缺失列，保证升级过程可重复执行。"""

    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}

    for name, definition in columns.items():

        if name not in existing:

            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")





def _init_schema(conn: sqlite3.Connection):

    conn.execute("""

        CREATE TABLE IF NOT EXISTS runs (

            run_id TEXT PRIMARY KEY,

            session_id TEXT NOT NULL,

            started_at TEXT NOT NULL,

            finished_at TEXT,

            status TEXT DEFAULT 'running'

        )

    """)

    conn.execute("""

        CREATE TABLE IF NOT EXISTS tool_calls (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            run_id TEXT NOT NULL,

            tool_name TEXT NOT NULL,

            input TEXT,

            output TEXT,

            decision TEXT,

            created_at TEXT NOT NULL,

            tool_call_id TEXT,

            status TEXT NOT NULL DEFAULT 'completed',

            error TEXT,

            started_at TEXT,

            finished_at TEXT

        )

    """)

    _ensure_columns(conn, "tool_calls", {

        "tool_call_id": "TEXT",

        "status": "TEXT NOT NULL DEFAULT 'completed'",

        "error": "TEXT",

        "started_at": "TEXT",

        "finished_at": "TEXT",

        "claim_id": "TEXT",

    })

    conn.execute(

        "CREATE INDEX IF NOT EXISTS idx_tool_calls_run_id ON tool_calls(run_id)"

    )

    conn.execute(

        "CREATE INDEX IF NOT EXISTS idx_tool_calls_run_created ON tool_calls(run_id, created_at, id)"

    )

    conn.execute(

        "CREATE INDEX IF NOT EXISTS idx_tool_calls_run_call_id ON tool_calls(run_id, tool_call_id)"

    )

    conn.execute("""

        CREATE TABLE IF NOT EXISTS messages (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            run_id TEXT NOT NULL,

            role TEXT NOT NULL,

            content TEXT,

            created_at TEXT NOT NULL

        )

    """)

    conn.execute(

        "CREATE INDEX IF NOT EXISTS idx_messages_run_id ON messages(run_id)"

    )

    conn.execute(

        "CREATE INDEX IF NOT EXISTS idx_runs_status_finished_run "

        "ON runs(status, finished_at, run_id)"

    )

    conn.execute("""

        CREATE TABLE IF NOT EXISTS tool_failures (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            run_id TEXT NOT NULL,

            session_id TEXT NOT NULL,

            tool_name TEXT NOT NULL,

            tool_call_id TEXT,

            input_json TEXT,

            output_json TEXT,

            error_message TEXT,

            failure_kind TEXT,

            created_at TEXT NOT NULL,

            processed_at TEXT,

            proposal_id TEXT

        )

    """)

    _ensure_columns(conn, "tool_failures", {

        "claim_id": "TEXT",

    })

    conn.execute("""

        CREATE INDEX IF NOT EXISTS idx_tool_failures_unprocessed

        ON tool_failures(processed_at, tool_name, failure_kind)

    """)

    conn.execute("""

        CREATE TABLE IF NOT EXISTS patch_proposals (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            proposal_id TEXT UNIQUE NOT NULL,

            source_failure_ids TEXT,

            target_files TEXT,

            patch_diff TEXT,

            test_command TEXT,

            test_stdout TEXT,

            test_stderr TEXT,

            test_returncode INTEGER,

            status TEXT DEFAULT 'pending',

            reviewed_by TEXT,

            reviewed_at TEXT,

            applied_at TEXT,

            created_at TEXT NOT NULL

        )

    """)

    _ensure_columns(conn, "patch_proposals", {

        "patch_content_json": "TEXT",

    })

    conn.execute("""

        CREATE INDEX IF NOT EXISTS idx_patch_proposals_status

        ON patch_proposals(status)

    """)





def init_db():

    """Initialize the default audit database (kept for backwards compatibility)."""

    _init_db_for_path(DB_PATH)





def _init_db_for_path(db_path: Path):

    db_path.parent.mkdir(parents=True, exist_ok=True)

    with _open_db(db_path) as conn:

        _init_schema(conn)





class AuditLogger:

    def __init__(self, db_path: Path | str | None = None):

        self._db_path = Path(db_path) if db_path else DB_PATH

        self._initialized = False



    def _ensure_db(self):

        if self._initialized:

            return

        _init_db_for_path(self._db_path)

        self._initialized = True



    def _conn(self):

        return _open_db(self._db_path)



    # ------------------------------------------------------------------

    # Synchronous primitives (safe to call from sync code).

    # ------------------------------------------------------------------

    def start_run(self, run_id: str, session_id: str):

        self._ensure_db()

        with self._conn() as conn:

            conn.execute(

                "INSERT INTO runs (run_id, session_id, started_at, status) VALUES (?, ?, ?, ?)",

                (run_id, session_id, datetime.now(timezone.utc).isoformat(), "running"),

            )



    def log_tool_call(

        self,

        run_id: str,

        tool_name: str,

        input_data,

        output_data,

        decision: str = "auto",

        tool_call_id: str | None = None,

        status: str = "completed",

        error: str | None = None,

        started_at: str | None = None,

        finished_at: str | None = None,

        claim_id: str | None = None,

    ):

        self._ensure_db()

        created_at = datetime.now(timezone.utc).isoformat()

        started_at = started_at or created_at

        finished_at = finished_at or created_at

        with self._conn() as conn:

            conn.execute(

                """

                INSERT INTO tool_calls (

                    run_id, tool_name, input, output, decision, created_at,

                    tool_call_id, status, error, started_at, finished_at, claim_id

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

                """,

                (

                    run_id,

                    tool_name,

                    json.dumps(input_data, ensure_ascii=False),

                    json.dumps(output_data, ensure_ascii=False),

                    decision,

                    created_at,

                    tool_call_id,

                    status,

                    error,

                    started_at,

                    finished_at,

                    claim_id,

                ),

            )

            # 工具调用失败时同步记录到 tool_failures，便于触发补丁流程。

            failure_session_id = None

            if status != "completed" or error is not None:

                run_row = conn.execute(

                    "SELECT session_id FROM runs WHERE run_id = ?", (run_id,)

                ).fetchone()

                failure_session_id = (

                    run_row["session_id"]

                    if run_row is not None and run_row["session_id"]

                    else run_id

                )

        if failure_session_id is not None:

            self.log_failure(

                run_id=run_id,

                session_id=failure_session_id,

                tool_name=tool_name,

                input_data=input_data,

                output_data=output_data,

                error_message=error or status or "unknown",

                failure_kind=status if status != "completed" else "error",

                tool_call_id=tool_call_id,

                claim_id=claim_id,

            )



    def log_message(self, run_id: str, role: str, content: str | None):

        self._ensure_db()

        with self._conn() as conn:

            conn.execute(

                "INSERT INTO messages (run_id, role, content, created_at) VALUES (?, ?, ?, ?)",

                (run_id, role, content, datetime.now(timezone.utc).isoformat()),

            )



    def finish_run(

        self,

        run_id: str,

        status: str = "completed",

        *,

        finished_at: str | None = None,

    ):

        """仅将 running audit run 终结一次；同状态重放不改变 finished_at。"""

        self._ensure_db()

        with self._conn() as conn:

            row = conn.execute(

                "SELECT status FROM runs WHERE run_id = ?", (run_id,)

            ).fetchone()

            if row is None:

                raise ValueError(f"未找到需完成的审计运行：{run_id}")

            if row["status"] != "running":

                if row["status"] == status:

                    return

                raise ValueError(f"审计运行终态冲突：{run_id} 已为 {row['status']}")

            conn.execute(

                "UPDATE runs SET finished_at = ?, status = ? WHERE run_id = ?",

                (finished_at or datetime.now(timezone.utc).isoformat(), status, run_id),

            )



    def log_failure(

        self,

        run_id: str,

        session_id: str,

        tool_name: str,

        input_data,

        output_data,

        error_message: str | None = None,

        failure_kind: str = "error",

        tool_call_id: str | None = None,

        claim_id: str | None = None,

    ) -> int:

        """记录工具调用失败，供后续补丁提案使用。"""

        self._ensure_db()

        created_at = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:

            cur = conn.execute(

                """

                INSERT INTO tool_failures (

                    run_id, session_id, tool_name, tool_call_id, input_json, output_json,

                    error_message, failure_kind, created_at, claim_id

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

                """,

                (

                    run_id, session_id, tool_name, tool_call_id,

                    json.dumps(input_data, ensure_ascii=False),

                    json.dumps(output_data, ensure_ascii=False),

                    error_message, failure_kind, created_at, claim_id,

                ),

            )

            return cur.lastrowid



    def create_proposal(

        self,

        proposal_id: str,

        source_failure_ids: list[int],

        target_files: list[str],

        patch_diff: str,

        test_command: str,

        test_stdout: str,

        test_stderr: str,

        test_returncode: int,

        patch_content: dict[str, str] | None = None,

    ) -> None:

        """基于工具失败创建一个待审核的补丁提案。"""

        self._ensure_db()

        created_at = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:

            conn.execute(

                """

                INSERT INTO patch_proposals (

                    proposal_id, source_failure_ids, target_files, patch_diff,

                    patch_content_json, test_command, test_stdout, test_stderr,

                    test_returncode, created_at

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

                """,

                (

                    proposal_id,

                    json.dumps(source_failure_ids),

                    json.dumps(target_files),

                    patch_diff,

                    json.dumps(patch_content or {}, ensure_ascii=False),

                    test_command,

                    test_stdout,

                    test_stderr,

                    test_returncode,

                    created_at,

                ),

            )



    def list_pending_proposals(self) -> list[dict]:

        """列出所有待审核的补丁提案。"""

        self._ensure_db()

        with self._conn() as conn:

            rows = conn.execute(

                "SELECT * FROM patch_proposals WHERE status = 'pending' ORDER BY created_at"

            ).fetchall()

        return [dict(r) for r in rows]



    def approve_proposal(self, proposal_id: str, reviewed_by: str = "admin") -> None:

        """将补丁提案标记为已批准。"""

        self._ensure_db()

        reviewed_at = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:

            conn.execute(

                "UPDATE patch_proposals SET status = 'approved', reviewed_by = ?, reviewed_at = ? WHERE proposal_id = ?",

                (reviewed_by, reviewed_at, proposal_id),

            )



    def get_run(self, run_id: str):

        """Return a run with its tool calls and messages, or ``None`` if not found.



        ``input`` and ``output`` of each tool call are deserialized from JSON.

        """

        self._ensure_db()

        with self._conn() as conn:

            row = conn.execute(

                "SELECT run_id, session_id, started_at, finished_at, status FROM runs WHERE run_id = ?",

                (run_id,),

            ).fetchone()

            if row is None:

                return None



            run = dict(row)

            tool_rows = conn.execute(

                """

                SELECT id, tool_name, input, output, decision, created_at,

                       tool_call_id, status, error,

                       COALESCE(started_at, created_at) AS started_at,

                       COALESCE(finished_at, created_at) AS finished_at

                FROM tool_calls

                WHERE run_id = ?

                ORDER BY created_at, id

                """,

                (run_id,),

            ).fetchall()



            tool_calls = []

            for tool_row in tool_rows:

                tool_call = dict(tool_row)

                if tool_call["input"] is not None:

                    tool_call["input"] = json.loads(tool_call["input"])

                if tool_call["output"] is not None:

                    tool_call["output"] = json.loads(tool_call["output"])

                tool_calls.append(tool_call)



            run["tool_calls"] = tool_calls



            message_rows = conn.execute(

                "SELECT id, role, content, created_at FROM messages WHERE run_id = ? ORDER BY created_at",

                (run_id,),

            ).fetchall()

            run["messages"] = [dict(m) for m in message_rows]



            return run



    def list_completed_runs_after(

        self,

        cursor: tuple[str, str],

        limit: int,

    ) -> list[dict]:

        self._ensure_db()

        finished_at, run_id = cursor

        with self._conn() as conn:

            rows = conn.execute(

                """

                SELECT run_id

                FROM runs

                WHERE status = 'completed'

                  AND finished_at IS NOT NULL

                  AND (finished_at > ? OR (finished_at = ? AND run_id > ?))

                ORDER BY finished_at ASC, run_id ASC

                LIMIT ?

                """,

                (finished_at, finished_at, run_id, limit),

            ).fetchall()

        return [self.get_run(row["run_id"]) for row in rows]



    # ------------------------------------------------------------------

    # Async wrappers that serialize writes through ``_write_lock``.

    # ------------------------------------------------------------------

    async def astart_run(self, run_id: str, session_id: str):

        async with _write_lock:

            self.start_run(run_id, session_id)



    async def alog_tool_call(

        self,

        run_id: str,

        tool_name: str,

        input_data,

        output_data,

        decision: str = "auto",

        tool_call_id: str | None = None,

        status: str = "completed",

        error: str | None = None,

        started_at: str | None = None,

        finished_at: str | None = None,

    ):

        async with _write_lock:

            self.log_tool_call(

                run_id, tool_name, input_data, output_data, decision,

                tool_call_id, status, error, started_at, finished_at,

            )



    async def alog_message(self, run_id: str, role: str, content: str | None):

        async with _write_lock:

            self.log_message(run_id, role, content)



    async def afinish_run(self, run_id: str, status: str = "completed"):

        async with _write_lock:

            self.finish_run(run_id, status)



    async def alog_failure(

        self,

        run_id: str,

        session_id: str,

        tool_name: str,

        input_data,

        output_data,

        error_message: str | None = None,

        failure_kind: str = "error",

        tool_call_id: str | None = None,

    ) -> int:

        async with _write_lock:

            return self.log_failure(

                run_id, session_id, tool_name, input_data, output_data,

                error_message, failure_kind, tool_call_id,

            )



    async def acreate_proposal(

        self,

        proposal_id: str,

        source_failure_ids: list[int],

        target_files: list[str],

        patch_diff: str,

        test_command: str,

        test_stdout: str,

        test_stderr: str,

        test_returncode: int,

        patch_content: dict[str, str] | None = None,

    ) -> None:

        async with _write_lock:

            self.create_proposal(

                proposal_id, source_failure_ids, target_files, patch_diff,

                test_command, test_stdout, test_stderr, test_returncode,

                patch_content,

            )



    async def aapprove_proposal(self, proposal_id: str, reviewed_by: str = "admin") -> None:

        async with _write_lock:

            self.approve_proposal(proposal_id, reviewed_by)



    def error(self, msg: str):

        """兼容 chat.py 中 logger.error 的调用，输出到 stderr。"""

        print(msg, file=sys.stderr)





logger = AuditLogger()

