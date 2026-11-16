"""任务板到运行时的只读快照与显式绑定；不扫描其他会话的任务。"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import re
from copy import deepcopy

_binding = ContextVar('runtime_board_intent_binding', default=None)


def intent_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def board_task_ids(value):
    """配置和恢复都只接受明确任务卡ID，不从描述或会话号猜关联。"""
    if not isinstance(value, (list, tuple)):
        raise ValueError('任务卡绑定必须是ID列表')
    if any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', v) for v in value):
        raise ValueError('任务卡ID无效')
    return list(dict.fromkeys(value))


def notify_board_task(task_id):
    """工具实际创建、认领或更新意图后绑定；默认无监听器、无副作用。"""
    callback = _binding.get()
    if callback is not None:
        callback(task_id)


class TaskBoardIntentSource:
    def __init__(self, get_board=None):
        self.get_board = get_board

    @contextmanager
    def bind(self, callback):
        token = _binding.set(callback)
        try:
            yield
        finally:
            _binding.reset(token)

    def snapshot(self, ids):
        ids = board_task_ids(ids)
        if not ids:
            return {'complete': False, 'reason': 'unbound_task_board', 'cards': []}
        if len(ids) > 32:
            return {'complete': False, 'reason': 'binding_limit', 'cards': []}
        if self.get_board is None:
            from .tasks.tools import get_board
            board = get_board()
        else:
            board = self.get_board()
        rows = []
        # 获取同一时刻的任务与直接依赖，避免异步模型收到混合更新。
        with board.store.lock():
            for task_id in sorted(ids):
                task = board.get(task_id)
                task.validate()
                dependencies = [{'id': dep, 'status': board.get(dep).status.value}
                                for dep in board_task_ids(task.blocked_by)]
                # 最多附一份此前明确清单，仅作历史来源；不提升为当前需求。
                previous_plan = next((entry for entry in reversed(task.execution_intent_history[:-1])
                    if entry['intent'].get('mode') == 'explicit_tools'), None)
                historical = ({'historical_only': True, 'revision': previous_plan['revision'],
                    'operation': previous_plan['operation'],
                    'intent': deepcopy(previous_plan['intent'])} if previous_plan else None)
                rows.append({'id': task.id, 'created_at': task.created_at,
                    'subject': task.subject, 'description': task.description,
                    'status': task.status.value,
                    'dependencies': dependencies, 'execution_intent': task.execution_intent,
                    'execution_intent_revision': task.execution_intent_revision,
                    'historical_plan': historical})
        result = {'complete': True, 'source': str(board.store.root.resolve()), 'cards': rows}
        result['version'] = intent_digest(result)
        return result
