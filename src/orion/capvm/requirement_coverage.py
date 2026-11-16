"""工具能力要求的版本化覆盖台账；语义判断仅作为驻留建议，不证明任务完成。"""

from copy import deepcopy

import hashlib

import json





def digest(value):

    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,

                                    allow_nan=False).encode()).hexdigest()





class RequirementCoverageLedger:

    """要求槽来自能力目录；目标是否需要、证据是否覆盖由独立判断器评估。"""

    def __init__(self, goal, registry, *, max_facts=64):

        if isinstance(max_facts,bool) or not isinstance(max_facts,int) or not 1<=max_facts<=256:

            raise ValueError("invalid fact bound")

        self.goal = goal

        self.registry = dict(registry)

        self.max_facts = max_facts

        self.facts = {}

        self.selected = {}

        self.generation = 0

        self.closed = False

        self.complete_history = True



    def select(self, calls):

        try:

            if self.closed:raise RuntimeError('task ledger closed')

            if len(calls)>64:raise ValueError('too many selected calls')

            self._select(calls)

        except Exception:

            self.complete_history=False;self.selected={}

            raise



    def _select(self, calls):

        selected = {}

        for call in calls:

            if call['name'] not in self.registry:

                continue

            key = call['id']

            if key in selected or key in self.facts:

                raise ValueError('call ID must be unique in task')

            scope = self.registry[call['name']].request_scope(call['args'])

            selected[key] = {'call_id': key, 'tool': call['name'], 'scope': scope,

                             'arguments_sha256': digest(call['args'])}

        self.selected = selected



    def record(self, result):

        try:

            if self.closed:raise RuntimeError('task ledger closed')

            self._record(result)

        except Exception:

            self.complete_history=False;self.selected={}

            raise



    def _record(self, result):

        messages = result.get('messages', ())

        received = {}

        for message in messages:

            key = getattr(message, 'tool_call_id', None)

            if key not in self.selected:

                continue

            if key in received:

                raise ValueError('duplicate tool result')

            expected = self.selected[key]

            raw = getattr(message, 'content', '')

            value = json.loads(raw) if isinstance(raw, str) else raw

            adapter = self.registry[expected['tool']]

            valid = (getattr(message, 'name', None) == expected['tool']

                     and getattr(message, 'status', 'success') == 'success'

                     and adapter.matches(expected['scope'], value))

            received[key] = {**deepcopy(expected), 'verified': bool(valid),

                             'result_sha256': digest(value)}

            if valid:

                received[key]['outcome'] = adapter.outcome(value)

        for key, expected in self.selected.items():

            fact = received.get(key, {**deepcopy(expected), 'verified': False})

            self.facts[key] = fact

        if len(self.facts) > self.max_facts:

            # 不截断后假装历史完整；失去历史完整性时全部语义覆盖建议撤销。

            self.complete_history = False

            self.facts = dict(list(self.facts.items())[-self.max_facts:])

        self.selected = {}



    def snapshot(self):

        if self.closed:

            raise RuntimeError('task coverage ledger closed')

        return deepcopy({'goal': self.goal, 'generation': self.generation,

                         'confirmed_facts': list(self.facts.values()),

                         'conditional_current_calls': list(self.selected.values()),

                         'complete_history': self.complete_history})



    def confirms(self, snapshot):

        """异步判断只能在当时选定调用全部真实成功、历史未被替换后采纳。"""

        if (self.closed or snapshot['generation'] != self.generation

                or snapshot['goal'] != self.goal or not self.complete_history

                or not snapshot['complete_history']):

            return False

        previous = {f['call_id']: f for f in snapshot['confirmed_facts']}

        pending = {f['call_id']: f for f in snapshot['conditional_current_calls']}

        if set(self.facts) != set(previous) | set(pending):

            return False

        if any(self.facts[k] != v for k, v in previous.items()):

            return False

        return all(self.facts[k].get('verified') and all(self.facts[k][field] == v[field]

                   for field in ('tool', 'scope', 'arguments_sha256')) for k, v in pending.items())



    def revise(self, goal):

        self.generation += 1

        self.goal = goal

        self.facts.clear()

        self.selected.clear()

        self.complete_history = True



    def close(self):

        self.closed = True

        self.facts.clear()

        self.selected.clear()





def remaining_judgments(snapshot, bindings, answers, prefix):

    """分别判断需求与覆盖，以Fréchet界组合，避免假设两次语义判断独立。"""

    by_resource = {}

    for index, tool in enumerate(sorted(bindings)):

        required = answers[f'{prefix}_{index}_required']['probabilities']

        covered = answers[f'{prefix}_{index}_covered']['probabilities']

        # unknown对应不确定质量，不能直接补零；这些不是校准置信区间。

        required_low, required_high = required['yes'], required['yes'] + required['unknown']

        covered_low, covered_high = covered['yes'], covered['yes'] + covered['unknown']

        if not snapshot['complete_history'] or not any(

                f.get('verified') or f in snapshot['conditional_current_calls']

                for f in snapshot['confirmed_facts'] + snapshot['conditional_current_calls']

                if f['tool'] == tool):

            covered_low, covered_high = 0., 0.

        low = max(0., required_low - covered_high)

        high = min(required_high, 1. - covered_low)

        for resource in bindings[tool]:

            by_resource.setdefault(resource, []).append((low, high))

    judgments = {}

    for resource, bounds in by_resource.items():

        low, high = max(b[0] for b in bounds), min(1., sum(b[1] for b in bounds))

        judgments[resource] = {'probabilities': {'needed': low, 'not_needed': 1.-high,

                                               'unknown': max(0., high-low)}}

    return {'conditional_on_current_success': True, 'judgments': judgments}

