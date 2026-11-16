"""任务板意图只形成回收软偏好；硬预算与活动租约仍由原governor执行。"""
import asyncio
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from threading import RLock
from time import monotonic


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def minimax_regrets(options, costs, bounds):
    """盒状评分区间内的最大相对损失，共同回收项抵消；不是实际耗时保证。"""
    groups = {option['id']:set(option['resource_ids']) for option in options}
    return {key:max(0., max(
        sum(costs[r]*bounds[r][1] for r in group-other)
        - sum(costs[r]*bounds[r][0] for r in other-group)
        for other in groups.values())) for key,group in groups.items()}


class TaskIntentController:
    def __init__(self, source, tool_resources, resource_descriptions, *, assessor=None,
                 ttl_s=30, clock=monotonic, selection_rule='dominance'):
        if not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError('意图建议有效期无效')
        self.source = source
        if selection_rule not in ('dominance', 'minimax_regret'):
            raise ValueError('未知意图选择规则')
        self.selection_rule = selection_rule
        self.tool_resources = {k: tuple(v) for k,v in tool_resources.items()}
        self.resource_descriptions = deepcopy(resource_descriptions)
        self.assessor, self.ttl_s, self.clock = assessor, ttl_s, clock
        self.events = deque(maxlen=256)
        self._scopes, self._hint, self._lock = {}, None, RLock()
        self._observations = {}
        self._goals = {}
        self._prefetch_task = None
        self._prefetch_key = None
        self._prefetch_tasks = set()

    def _cancel_prefetch(self):
        task = self._prefetch_task
        self._prefetch_task = self._prefetch_key = None
        if task is not None and not task.done():
            # 任务工具可能在线程内回调attach，不直接跨线程操作asyncio任务。
            task.get_loop().call_soon_threadsafe(task.cancel)

    async def drain_prefetch(self):
        pending = tuple(self._prefetch_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def start_prefetch(self, resources):
        """只预取语义解释，不预选工具、不准备资源、不安装可执行建议。"""
        try:
            resources = tuple(sorted(resources))
            if (self.assessor is None or not 2 <= len(resources) <= 8
                    or self.assessor.calls >= self.assessor.max_calls):
                return
            snapshot = self.snapshot()
            requirements = self._requirements(snapshot, resources)
            if requirements is None or not requirements[1]:
                return
            key = (snapshot['version'], resources)
            if self._prefetch_key == key:
                return
            descriptions = {r:{k:v for k,v in self.resource_descriptions[r].items()
                               if k!='literal_terms'} for r in resources}
            semantic = requirements[1]
            if self.assessor.cached_assessment(semantic, descriptions) is not None:
                return
            self._cancel_prefetch()
            self._prefetch_key = key
            task = asyncio.create_task(self._prefetch(semantic, descriptions, key))
            self._prefetch_task = task
            self._prefetch_tasks.add(task)
            task.add_done_callback(self._prefetch_tasks.discard)
        except Exception as error:
            self.events.append({'event':'intent_prefetch_skipped', 'error':type(error).__name__})

    async def _prefetch(self, semantic, descriptions, key):
        started = self.clock()
        self.events.append({'event':'intent_prefetch_started', 'version':key[0],
                            'resources':list(key[1])})
        try:
            result = await self.assessor.assess(semantic, descriptions)
            fresh = self.snapshot()
            status = ('ready' if result is not None and fresh.get('version') == key[0]
                      else 'stale_or_unavailable')
        except asyncio.CancelledError:
            status = 'cancelled'
        except Exception as error:
            status = type(error).__name__
        self.events.append({'event':'intent_prefetch_finished', 'version':key[0],
                            'status':status, 'elapsed_s':self.clock()-started})

    def observe(self, token, batch):
        """仅收录仍活动的作用域内已返回的工具批次，不推断要求已经满足。"""
        with self._lock:
            if token not in self._scopes:
                return
            self._observations[token].append(deepcopy(batch))
            self._hint = None
            self._cancel_prefetch()

    def ids(self, token):
        with self._lock:
            return list(self._scopes.get(token, ()))

    def attach(self, token, task_id):
        with self._lock:
            if token not in self._scopes:
                return
            ids = self._scopes[token]
            if task_id not in ids:
                ids.append(task_id)
                self._cancel_prefetch()
                self.events.append({'event': 'board_task_bound', 'token': token, 'task_id': task_id})

    @contextmanager
    def scope(self, token, ids, *, goal=None):
        if (not isinstance(ids, (list, tuple))
            or any(not isinstance(v, str) or not v for v in ids)):
            raise ValueError('任务卡绑定无效')
        with self._lock:
            self._scopes[token] = list(dict.fromkeys(ids))
            self._observations[token] = deque(maxlen=4)
            self._goals[token] = goal
        try:
            with self.source.bind(lambda task_id: self.attach(token, task_id)):
                yield
        finally:
            with self._lock:
                self._scopes.pop(token, None)
                self._observations.pop(token, None)
                self._goals.pop(token, None)
            self._hint = None
            self._cancel_prefetch()
            self.events.append({'event': 'intent_scope_closed', 'token': token})

    def snapshot(self):
        with self._lock:
            scopes = [(token, list(ids), deepcopy(list(self._observations[token])), self._goals[token])
                      for token, ids in self._scopes.items()]
        if not scopes:
            return {'complete': False, 'reason': 'no_active_task_scope'}
        cards, versions = {}, []
        try:
            for token, ids, observations, goal in sorted(scopes):
                if not ids and isinstance(goal, str) and goal.strip():
                    # 这是实际运行请求的状态，不伪称LLM创建了任务卡或执行计划。
                    key = _digest(['runtime_task_scope', token])
                    cards[key] = {'id':key, 'key':key, 'source_kind':'runtime_task_scope',
                        'subject':'当前用户任务', 'description':goal, 'status':'in_progress',
                        'dependencies':[], 'execution_intent':{}, 'observed_scope_batches':observations}
                    versions.append([token, _digest(goal)])
                    continue
                snapshot = self.source.snapshot(ids)
                if not snapshot['complete']:
                    return {'complete': False, 'reason': snapshot['reason']}
                versions.append([token, snapshot['version']])
                for card in snapshot['cards']:
                    key = _digest([snapshot['source'], card['id'], card['created_at']])
                    row = cards.setdefault(key, {**card, 'key': key, 'source_kind':'task_board',
                                                'observed_scope_batches': []})
                    row['observed_scope_batches'].extend(observations)
            state = {'complete': True, 'scope_versions': versions,
                     'cards': list(cards.values()), 'bindings': self.tool_resources,
                     'resource_descriptions': self.resource_descriptions,
                     'selection_rule': self.selection_rule}
            state['version'] = _digest(state)
            return state
        except Exception as error:
            return {'complete': False, 'reason': 'task_board_unavailable', 'error': type(error).__name__}

    def _requirements(self, snapshot, resources):
        resolved, semantic = {}, []
        if not snapshot.get('complete'):
            return None
        for card in snapshot['cards']:
            if card['status'] == 'completed':
                continue
            if card['status'] == 'escalated':
                return None
            intent = card.get('execution_intent') or {}
            if intent.get('mode') == 'explicit_tools':
                if any(t not in self.tool_resources for t in intent['planned_tools']):
                    return None
                needed = {r for t in intent['planned_tools'] for r in self.tool_resources[t]}
                resolved[card['key']] = {r: [float(r in needed)]*2 for r in resources}
            elif intent.get('mode') in (None, 'semantic_note'):
                note = intent.get('note', '')
                description = card.get('description', '')
                if not all(isinstance(text, str) for text in (card['subject'], description, note)):
                    return None
                if not any(text.strip() for text in (card['subject'], description, note)):
                    return None
                # 普通任务记录也可解释；词语命中不能证明其他能力已不再需要。
                semantic.append({'key': card['key'], 'task_id': card['id'],
                    'source_kind': card.get('source_kind', 'task_board'),
                    'subject': card['subject'], 'description': description,
                    'status': card['status'],
                    'dependencies': card['dependencies'], 'intent': note,
                    'observed_scope_batches': card.get('observed_scope_batches', [])})
            else:
                return None
        return resolved, semantic

    async def prepare(self, state, options, costs):
        """可选语义分析失败时撤销建议；取消仍交给调用者处理。"""
        try:
            await self._prepare(state, options, costs)
        except asyncio.CancelledError:
            self._hint = None
            raise
        except Exception as error:
            self._hint = None
            self.events.append({'event': 'intent_fallback', 'reason': 'assessment_error',
                                'error': type(error).__name__})

    async def _prepare(self, state, options, costs):
        if len(options) < 2:
            self.events.append({'event': 'intent_not_needed',
                'reason': 'fewer_than_two_enumerated_feasible_plans', 'option_count': len(options)})
            return
        resources = sorted({r for o in options for r in o['resource_ids']})
        snapshot = self.snapshot()
        requirements = self._requirements(snapshot, resources)
        if requirements is None:
            self.events.append({'event': 'intent_fallback', 'reason': snapshot.get('reason', 'incomplete_intent')})
            return
        _, semantic = requirements
        if not semantic:
            self.events.append({'event': 'intent_resolved', 'source': 'declared_tool_bindings',
                                'version': snapshot['version'], 'api_calls': 0})
            return
        if self._valid_hint(snapshot, resources) is not None:
            return
        if self.assessor is None:
            self.events.append({'event': 'intent_fallback', 'reason': 'no_explicit_assessor'})
            return
        prefetch = self._prefetch_task
        key = (snapshot['version'], tuple(resources))
        if prefetch is not None:
            if self._prefetch_key != key:
                self._cancel_prefetch()
            elif not prefetch.done():
                started = self.clock()
                await asyncio.shield(prefetch)
                self.events.append({'event':'intent_prefetch_joined',
                                    'residual_wait_s':self.clock()-started})
                if self.snapshot().get('version') != snapshot['version']:
                    return
        # 历史字面别名不作需求证据；有效缓存先于网络开销门控读取。
        descriptions = {r:{k:v for k,v in self.resource_descriptions[r].items()
                           if k!='literal_terms'} for r in resources}
        lookup=getattr(self.assessor,'cached_assessment',None)
        result=lookup(semantic,descriptions) if lookup else None
        cached=result is not None
        if not cached:
            from .demand_coordinator import prediction_value_ceiling
            ceiling = prediction_value_ceiling(options, costs)
            overhead = self.assessor.estimated_overhead_s
            if (ceiling is None or not math.isfinite(overhead)
                or ceiling['avoidable_cost_ceiling_s'] <= overhead):
                self.events.append({'event': 'intent_fallback', 'reason': 'cost_gate', 'ceiling': ceiling})
                return
            result = await self.assessor.assess(semantic, descriptions)
        fresh = self.snapshot()
        if result is None or not fresh.get('complete') or fresh['version'] != snapshot['version']:
            self._hint = None
            self.events.append({'event': 'intent_fallback', 'reason': 'missing_or_stale_assessment'})
            return
        valid_for = min(self.ttl_s, result['valid_for_s'])
        self._hint = (snapshot['version'], tuple(resources), deepcopy(result['bounds']), self.clock()+valid_for)
        self.events.append({'event': 'intent_assessed', 'version': snapshot['version'],
                            'source':'semantic_cache' if cached else 'assessment'})

    def _valid_hint(self, snapshot, resources):
        hint = self._hint
        if (hint is None or hint[0] != snapshot.get('version') or self.clock() >= hint[3]
            or not set(resources) <= set(hint[1])):
            return None
        return hint[2]

    def decision_state(self):
        snapshot = self.snapshot()
        hint = self._hint
        active_hint = hint if hint and self.clock() < hint[3] and hint[0] == snapshot.get('version') else None
        return {'snapshot': snapshot, 'hint': active_hint}

    def choose(self, options, costs):
        resources = sorted({r for o in options for r in o['resource_ids']})
        if len(options) < 2 or any(r not in costs or not math.isfinite(costs[r]) or costs[r] <= 0 for r in resources):
            return None
        snapshot = self.snapshot()
        requirements = self._requirements(snapshot, resources)
        if requirements is None:
            return None
        resolved, semantic = requirements
        hint = self._valid_hint(snapshot, resources)
        if semantic:
            if hint is None:
                return None
            for card in semantic:
                if card['key'] not in hint or not set(resources) <= set(hint[card['key']]):
                    return None
                resolved[card['key']] = hint[card['key']]
        # 这是意图关联的启发式分数区间，不是已校准的未来调用概率。
        bounds = {r: [max((b[r][0] for b in resolved.values()), default=0),
                      min(1., sum(b[r][1] for b in resolved.values()))] for r in resources}
        scores = {o['id']: [sum(costs[r]*bounds[r][i] for r in set(o['resource_ids']))
                            for i in (0,1)] for o in options}
        winner = next((key for key,value in scores.items()
                       if all(value[1] < other[0]-1e-9 for k,other in scores.items() if k != key)), None)
        baseline = min(options, key=lambda o: (sum(costs[r] for r in set(o['resource_ids'])), o['id']))['id']
        regrets = {}
        if winner is None and self.selection_rule == 'minimax_regret':
            regrets = minimax_regrets(options, costs, bounds)
            best = min(options, key=lambda o:(regrets[o['id']],
                sum(costs[r] for r in set(o['resource_ids'])),o['id']))['id']
            # 仅当相对损失上界严格优于成本基线才改变选择；平分仍回退。
            if regrets[best] < regrets[baseline]-1e-9:
                winner = best
        if winner is not None:
            self.events.append({'event': 'intent_plan_selected', 'version': snapshot['version'],
                'source': 'semantic_intent' if semantic else 'declared_tool_bindings',
                'choice': winner, 'cost_baseline': baseline, 'changed': winner != baseline,
                'scores': scores, 'selection_rule':self.selection_rule, 'regrets':regrets})
        return winner


class TaskIntentDecider:
    def __init__(self, controller, fallback, costs):
        self.controller, self.fallback, self.costs = controller, fallback, costs

    def decision_state(self):
        previous = getattr(self.fallback, 'decision_state', None)
        return {'task_intent': self.controller.decision_state(),
                'fallback': previous() if previous else None, 'costs': dict(self.costs)}

    async def __call__(self, state, options):
        choice = self.controller.choose(options, self.costs)
        return choice if choice is not None else await self.fallback(state, options)
