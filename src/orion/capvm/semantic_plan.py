"""完整回收集合的有界选择器；异步取建议，同步执行前重新验证。



仅管理声明驻留预算；整机建立预留仍由原 governor 检查。默认只观测。

"""

import asyncio

from dataclasses import asdict

import hashlib

from itertools import combinations

import json

import math

from time import monotonic



from .model import ResidencyLevel, ResourceVector





class SemanticPlanAdvisor:

    """模型仅返回代码生成的方案 ID；不允许修改资源层级或预算。"""

    def __init__(self, decide, *, active=False, max_calls=2, timeout_s=1.0,

                 ttl_s=5.0, max_candidates=8, max_plans=16, clock=monotonic,

                 use_unique_plan=False):

        if any(isinstance(v, bool) or not isinstance(v, int) for v in (max_candidates, max_plans)) or (max_calls is not None and (isinstance(max_calls,bool) or not isinstance(max_calls,int) or max_calls < 0)) or not 1 <= max_candidates <= 12 or not 2 <= max_plans <= 64:

            raise ValueError('invalid decision limits')

        if any(not math.isfinite(v) or v <= 0 for v in (timeout_s, ttl_s)):

            raise ValueError('timeouts must be positive and finite')

        self.decide, self.active = decide, active

        self.use_unique_plan = use_unique_plan

        self.max_calls, self.timeout_s, self.ttl_s = max_calls, timeout_s, ttl_s

        self.max_candidates, self.clock = max_candidates, clock

        self.max_plans = max_plans

        self.calls = 0

        self.events = []

        self.context = {}

        self.cached = None



    def set_task_context(self, context):

        """调用者提供含任务版本的当时信息；深拷贝避免并发修改。"""

        self.context = json.loads(json.dumps(context, ensure_ascii=False))

        self.cached = None



    def snapshot(self, governor, requests):

        required = dict(governor.expand_requirements(requests))

        protected = set(required)

        states = {s.capability_id:s for s in governor.states()}

        pins = dict(governor.expand_requirements(

            (i,governor.catalog.get(i).pinned_min_level) for i in governor.pinned_ids))

        protected.update(pins)

        for i,s in states.items():

            if s.active_count:

                protected.update(dict(governor.expand_requirements([(i,s.current_level)])))

        rows = []

        for i,s in sorted(states.items()):

            if s.current_level == ResidencyLevel.CATALOG and i not in required:

                continue

            cap = governor.catalog.get(i)

            target = cap.pinned_min_level

            projected = max(s.current_level,required.get(i,ResidencyLevel.CATALOG))

            cost = cap.adapter.declared_cost(projected)

            released = cost - cap.adapter.declared_cost(target)

            rows.append({'id':i,'level':int(s.current_level),'target':int(target),

                'active':s.active_count,'tick':s.last_used_tick,'summary':cap.summary,

                'protected':i in protected,'cost':asdict(cost),'released':asdict(released)})

        state = {'resources':rows,'required':{i:int(v) for i,v in sorted(required.items())},

                 'budget':asdict(governor.budget),'task_context':self.context}

        dependency_state = getattr(self.decide, "decision_state", None)

        if dependency_state is not None:

            state["decision_state"] = dependency_state()

        fingerprint=hashlib.sha256(json.dumps(state,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

        return state,fingerprint



    def options(self, state):

        """枚举包含关系最小的可行回收集合；规模过大则回退，不截断成伪最优。"""

        total = ResourceVector()

        for row in state['resources']: total += ResourceVector(**row['cost'])

        budget=ResourceVector(**state['budget'])

        violations=total.violated_dimensions(budget)

        if not violations: return []

        eligible=[r for r in state['resources'] if not r['protected'] and not r['active']

                  and r['level'] > r['target'] and any(r['released'][d]>0 for d in violations)]

        if len(eligible)>self.max_candidates: return []

        options=[]

        for n in range(1,len(eligible)+1):

            for group in combinations(eligible,n):

                ids={r['id'] for r in group}

                if any(set(o['resource_ids']) <= ids for o in options): continue

                released=ResourceVector()

                for row in group: released+=ResourceVector(**row['released'])

                if (total-released).fits(budget):

                    options.append({'id':f'plan_{len(options)}','resource_ids':sorted(ids),

                        'actions':[(r['id'],r['target']) for r in group],

                        'released':asdict(released)})

                    if len(options) > self.max_plans: return []

        return options



    async def prepare(self, governor, requests):

        """不得在租约内部调用；预算预占且不自动重试。"""

        self.cached=None

        state,fingerprint=self.snapshot(governor,requests)

        options=self.options(state)

        if self.use_unique_plan and len(options)==1:

            # 唯一最小可行集合无需询问模型；执行时仍核验指纹、租约和有效期。

            chosen=options[0]

            self.cached=(fingerprint,self.clock(),chosen)

            self.events.append({'snapshot':fingerprint,'option_count':1,

                'status':'ready','source':'unique_minimal_plan',

                'choice':chosen['id'],'actions':chosen['actions']})

            del self.events[:-256]

            return

        if len(options)<2 or (self.max_calls is not None and self.calls>=self.max_calls): return

        self.calls+=1

        started=self.clock()

        event={'snapshot':fingerprint,'option_count':len(options),'call':self.calls}

        try:

            choice=await asyncio.wait_for(self.decide(state,options),self.timeout_s)

            chosen=next((o for o in options if o['id']==choice),None)

            if chosen is None: raise ValueError('unknown plan ID')

            _,current=self.snapshot(governor,requests)

            if current!=fingerprint:

                event['status']='stale'

            else:

                self.cached=(fingerprint,self.clock(),chosen)

                event.update(status='ready',choice=choice,actions=chosen['actions'])

        except asyncio.CancelledError:

            event['status']='cancelled'

            raise

        except Exception as error:

            event.update(status='fallback',error=type(error).__name__)

        finally:

            event['elapsed_s']=self.clock()-started

            self.events.append(event)

            del self.events[:-256]



    def select(self, governor, required, baseline):

        """执行前重建快照和可行集合；任何不匹配均保留原策略。"""

        if self.cached is None: return baseline

        fingerprint,created,chosen=self.cached

        self.cached=None

        state,current=self.snapshot(governor,required.items())

        if current!=fingerprint or self.clock()-created>self.ttl_s: return baseline

        if chosen not in self.options(state): return baseline

        self.events.append({'status':'applied' if self.active else 'shadow',

            'baseline':[(i,int(v)) for i,v in baseline],'suggested':chosen['actions']})

        del self.events[:-256]

        if not self.active: return baseline

        return [(i,ResidencyLevel(v)) for i,v in chosen['actions']]

