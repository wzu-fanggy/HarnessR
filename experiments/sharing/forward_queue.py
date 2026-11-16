"""实验入口适配：已提交请求的需求跨阶段保留；不准备资源或调用模型。"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import sys
from threading import RLock

from orion.capvm.task_intent import TaskIntentController


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def announce_received_request(queue, value, config, *, source):
    """登记已收到的原始消息；身份算法与ResourceTaskGraph一致。"""
    from langchain_core.messages import HumanMessage, convert_to_messages
    messages = convert_to_messages(value['messages'])
    user = next((m for m in reversed(messages) if isinstance(m, HumanMessage)
                 and not m.additional_kwargs.get('orion_internal')), None)
    if user is None or not user.id:
        raise ValueError('入队前须由harness为真实用户消息分配稳定ID')
    options = config.get('configurable', {})
    identity = [options.get('thread_id'), options.get('checkpoint_ns'), user.id]
    logical_id = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    queue.submit(logical_id, logical_id, str(user.content),
                 source={**source, 'message_id': user.id, 'thread_id': options.get('thread_id')})
    return logical_id


@dataclass
class SubmittedRequest:
    request_id: str
    logical_task_id: str
    goal: str
    source: dict
    revision: int = 0
    status: str = 'pending'


class SubmittedRequests:
    """只接收harness已经收到的请求，不从计划或后续轨迹生成新任务。"""
    def __init__(self):
        self.records = {}
        self.lock = RLock()
        self.revision = 0

    def submit(self, request_id, logical_task_id, goal, *, source):
        if not request_id or not logical_task_id or not isinstance(goal, str) or not goal.strip():
            raise ValueError('请求缺少身份或真实目标')
        if not isinstance(source, dict) or not source:
            raise ValueError('必须记录已提交请求的来源')
        with self.lock:
            if request_id in self.records or any(r.logical_task_id == logical_task_id for r in self.records.values()):
                raise ValueError('请求身份不能重复使用')
            self.records[request_id] = SubmittedRequest(request_id, logical_task_id, goal, deepcopy(source))
            self.revision += 1

    def find(self, logical_task_id):
        with self.lock:
            return next((r for r in self.records.values() if r.logical_task_id == logical_task_id), None)

    def activate(self, request_id, goal):
        with self.lock:
            row = self.records[request_id]
            if row.status != 'pending' or row.goal != goal:
                raise ValueError('请求已取消、已开始或目标版本不匹配')
            row.status = 'in_progress'
            self.revision += 1

    def finish(self, request_id, outcome):
        with self.lock:
            row = self.records[request_id]
            if row.status != 'in_progress':
                raise ValueError('不能结束非活动请求')
            row.status = outcome
            self.revision += 1

    def revise(self, request_id, goal):
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError('目标不能为空')
        with self.lock:
            row = self.records[request_id]
            if row.status != 'pending':
                raise ValueError('只能修改尚未开始的请求；活动任务变更应作为新回合处理')
            row.goal = goal
            row.revision += 1
            self.revision += 1

    def cancel(self, request_id):
        with self.lock:
            row = self.records[request_id]
            if row.status != 'pending':
                raise ValueError('这里只取消待执行请求；活动任务走原取消流程')
            row.status = 'cancelled'
            self.revision += 1


class QueueAwareIntentController(TaskIntentController):
    """复用原解释/选择/失效逻辑，仅扩展已到达请求的可见范围。"""
    def __init__(self, *args, submitted_requests, pending_window=1, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(pending_window, bool) or not isinstance(pending_window, int) or pending_window < 1:
            raise ValueError('待执行需求窗口必须为正整数')
        self.submitted_requests = submitted_requests
        self.pending_window = pending_window

    @contextmanager
    def scope(self, token, ids, *, goal=None):
        queue = self.submitted_requests
        with self._lock, queue.lock:
            row = queue.find(token[0])
            request_id = row.request_id if row else None
            if request_id:
                queue.activate(request_id, goal)
            manager = super().scope(token, ids, goal=goal)
            try:
                manager.__enter__()
            except BaseException:
                if request_id:
                    queue.finish(request_id, 'failed')
                raise
        outcome = 'completed'
        error_info = (None, None, None)
        try:
            yield
        except BaseException as error:
            error_info = sys.exc_info()
            outcome = 'cancelled' if type(error).__name__ == 'CancelledError' else 'failed'
            raise
        finally:
            # 原控制器撤销旧合并建议；下一请求的原始目标独立保留。
            with self._lock, queue.lock:
                try:
                    manager.__exit__(*error_info)
                finally:
                    if request_id:
                        queue.finish(request_id, outcome)

    def snapshot(self):
        queue = self.submitted_requests
        with self._lock, queue.lock:
            base = super().snapshot()
            if not base.get('complete') and base.get('reason') != 'no_active_task_scope':
                return base
            pending = [r for r in queue.records.values() if r.status == 'pending'][:self.pending_window]
            if not base.get('complete'):
                if not pending:
                    return base
                base = {'complete': True, 'scope_versions': [], 'cards': [],
                        'bindings': self.tool_resources, 'resource_descriptions': self.resource_descriptions,
                        'selection_rule': self.selection_rule}
            else:
                base = deepcopy(base)
            for row in pending:
                key = digest(['submitted_request', row.request_id])
                base['cards'].append({'id': row.request_id, 'key': key,
                    'source_kind': 'harness_request_queue', 'subject': '已提交待执行任务',
                    'description': row.goal, 'status': 'pending', 'dependencies': [],
                    'execution_intent': {}, 'observed_scope_batches': [],
                    'request_revision': row.revision})
            base['request_envelopes'] = [{'request_id': r.request_id,
                'logical_task_id': r.logical_task_id, 'revision': r.revision,
                'status': r.status, 'source': deepcopy(r.source)} for r in queue.records.values()
                if r.status == 'in_progress' or r in pending]
            base['queue_revision'] = queue.revision
            base.pop('version', None)
            base['version'] = digest(base)
            return base
