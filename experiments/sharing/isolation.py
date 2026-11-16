"""实验会话隔离：工具接口与科学计算不变，仅隔离任务板和文件根目录。"""
from contextvars import ContextVar
from pathlib import Path
from threading import RLock

agent_scope = ContextVar('core_experiment_agent', default=None)


class WorkspaceRoot:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def current(self):
        owner = agent_scope.get()
        return self.root / owner if owner else self.root

    def __fspath__(self):
        return str(self.current())

    def __truediv__(self, path):
        return self.current() / path

    def __getattr__(self, name):
        return getattr(self.current(), name)


class OwnedBoard:
    """共用原任务存储，工具只可见各自任务；内部协调器另读完整原板。"""
    def __init__(self, board):
        self.board = board
        self.owners = {}
        self.lock = RLock()

    def _owner(self):
        owner = agent_scope.get()
        if owner is None:
            raise RuntimeError('工具任务操作缺少agent作用域')
        return owner

    def _check(self, task_id):
        if self.owners.get(task_id) != self._owner():
            raise PermissionError('任务不属于当前智能体')

    def create(self, *args, **kwargs):
        owner = self._owner()
        dependencies = kwargs.get('blocked_by') or (args[2] if len(args) > 2 else [])
        with self.lock:
            for task_id in dependencies:
                self._check(task_id)
            task = self.board.create(*args, **kwargs)
            self.owners[task.id] = owner
            return task

    def list(self, status=None):
        owner = self._owner()
        with self.lock:
            return [task for task in self.board.list(status) if self.owners.get(task.id) == owner]

    def get(self, task_id):
        with self.lock:
            self._check(task_id)
            return self.board.get(task_id)

    def __getattr__(self, name):
        method = getattr(self.board, name)
        if not callable(method):
            return method

        def owned(task_id, *args, **kwargs):
            with self.lock:
                self._check(task_id)
                return method(task_id, *args, **kwargs)
        return owned
