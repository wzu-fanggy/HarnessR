"""任务卡的文件存储。

一个任务一个 JSON 文件，落在 ``.tasks/{id}.json``，和 Learn Claude Code s12
的做法一致。为什么不用 SQLite：任务卡是给人也要能直接看的审计对象，纯文本
diff 得出来；而且任务数量在几十到几百量级，文件足够。

并发：单机单 agent 用目录级锁文件（``.lock``）足够；多 agent 并发认领时需要
Claude Code 那种 per-task 双重锁 + 读完重验（防 TOCTOU），这里留了接口没有
实现，见 ``claim`` 的注释。
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .models import Task

DEFAULT_TASKS_DIR = ".tasks"
LOCK_STALE_SECONDS = 60.0


class TaskStore:
    def __init__(self, root: str | Path | None = None) -> None:
        env_root = os.environ.get("ORION_TASKS_DIR")
        self.root = Path(root or env_root or DEFAULT_TASKS_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- 路径 ----------

    def _path(self, task_id: str) -> Path:
        return self.root / f"{task_id}.json"

    @property
    def _highwatermark(self) -> Path:
        return self.root / ".highwatermark"

    # ---------- 锁 ----------

    @contextmanager
    def lock(self, timeout: float = 5.0) -> Iterator[None]:
        """目录级互斥锁：O_CREAT|O_EXCL 建锁文件，陈旧锁自动回收。"""
        lock_path = self.root / ".lock"
        start = time.time()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                age = time.time() - lock_path.stat().st_mtime if lock_path.exists() else 0
                if age > LOCK_STALE_SECONDS:
                    lock_path.unlink(missing_ok=True)
                    continue
                if time.time() - start > timeout:
                    raise TimeoutError(f"获取任务板锁超时：{lock_path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            lock_path.unlink(missing_ok=True)

    # ---------- 写入 ----------

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def next_id(self) -> str:
        """递增 ID + 高水位，任务被删也不重用（对应 CC 的 .highwatermark）。"""
        n = 0
        if self._highwatermark.exists():
            try:
                n = int(self._highwatermark.read_text(encoding="utf-8").strip() or 0)
            except ValueError:
                n = 0
        n += 1
        self._atomic_write(self._highwatermark, str(n))
        return f"task_{n:04d}"

    def save(self, task: Task) -> None:
        self._atomic_write(
            self._path(task.id),
            json.dumps(task.to_dict(), ensure_ascii=False, indent=2),
        )

    # ---------- 读取 ----------

    def load(self, task_id: str) -> Task:
        path = self._path(task_id)
        if not path.exists():
            raise FileNotFoundError(f"任务不存在：{task_id}")
        return Task.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_all(self) -> list[Task]:
        tasks: list[Task] = []
        for p in sorted(self.root.glob("task_*.json")):
            try:
                tasks.append(Task.from_dict(json.loads(p.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, TypeError, ValueError):
                # 坏文件不阻断整个看板：跳过，交给人工看
                continue
        return tasks

    def delete(self, task_id: str) -> None:
        self._path(task_id).unlink(missing_ok=True)
