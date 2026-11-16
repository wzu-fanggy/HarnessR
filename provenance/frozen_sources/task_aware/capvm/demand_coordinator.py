"""显式任务作用域绑定共享运行时；任务状态不放入全局可变目标。"""
from collections import deque
import math
import json
from contextlib import asynccontextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict
from .task_demand import TaskDemandLedger,DemandPlanDecider
from .semantic_plan import SemanticPlanAdvisor
from .forecast_feedback import ForecastFeedback

def prediction_value_ceiling(options, costs):
    """单次首次重载模型下，相对唯一成本基线可避免的最大成本。

    对基线 B 和候选 P，改善不超过 cost(B - P)；共有回收项会抵消。
    取所有候选的最大值。历史成本是估计值，因此不是实际耗时上界。
    """
    if len(options)<2:return None
    victims={r for option in options for r in option['resource_ids']}
    if any(r not in costs or isinstance(costs[r],bool) or not isinstance(costs[r],(int,float))
           or not math.isfinite(costs[r]) or costs[r]<=0 for r in victims):return None
    ranked=sorted((sum(costs[r] for r in set(o['resource_ids'])),o['id'],set(o['resource_ids'])) for o in options)
    # 成本平分时执行层可能采用原策略，不能假定某个基线集合。
    if math.isclose(ranked[0][0],ranked[1][0]):return None
    baseline=ranked[0][2]
    return {'baseline_plan':ranked[0][1],
            'avoidable_cost_ceiling_s':max(sum(costs[r] for r in baseline-set(o['resource_ids'])) for o in options),
            'common_resources':sorted(set.intersection(*(set(o['resource_ids']) for o in options)))}

class CostFallbackDecider(DemandPlanDecider):
    """语义证据失效时使用完整集合的实测成本；未知需求从不补零。"""
    async def __call__(self, state, options):
        choice=await super().__call__(state,options)
        if choice is not None:return choice
        victims={r for o in options for r in o['resource_ids']}
        if any(r not in self.costs or not math.isfinite(self.costs[r]) or self.costs[r]<=0 for r in victims):return None
        ranked=sorted((sum(self.costs[r] for r in o['resource_ids']),o['id']) for o in options)
        if len(ranked)<2 or math.isclose(ranked[0][0],ranked[1][0]):return None
        self.events.append({'source':'observed_cost_fallback','choice':ranked[0][1]})
        del self.events[:-128]
        return ranked[0][1]

class DemandCoordinator:
    def __init__(self, runtime, bindings, profiler, *, predictor_factory=None, active=False, bundle_mode=False, requirement_registry=None, task_intents=None, intent_prefetch=False):
        if task_intents is not None and predictor_factory is not None:
            raise ValueError('任务意图与旧需求预测器不能同时启用')
        if getattr(runtime,'demand_coordinator',None) is not None or runtime.governor.plan_advisor is not None:
            raise ValueError('runtime already has a coordinator or advisor')
        self.runtime,self.profiler=runtime,profiler
        self.task_intents = task_intents
        if intent_prefetch and task_intents is None:
            raise ValueError('意图预取需要任务语义控制器')
        self.intent_prefetch = bool(intent_prefetch)
        self.feedback=ForecastFeedback()
        self.requirement_registry=dict(requirement_registry or {})
        self.requirement_ledgers={}
        self.bundle_mode=bundle_mode
        self.context=ContextVar('resource_demand_task',default=None)
        self.tasks=set();self.goals={};self.observed_context={};self.selected_calls={};self.events=deque(maxlen=256)
        self.resource_tools={}
        self.tool_resources={}
        for tool,binding in bindings.items():
            expanded=runtime.governor.expand_requirements(binding.requirements())
            self.tool_resources[tool]=tuple(sorted({r for r,_ in expanded}))
            for resource in self.tool_resources[tool]:
                self.resource_tools.setdefault(resource,[]).append(tool)
        if bundle_mode:
            from .tool_bundle_demand import ToolBundleLedger,BundlePlanDecider
            self.ledger=ToolBundleLedger(self.tool_resources,feedback=self.feedback)
            decider=BundlePlanDecider(self.ledger,profiler.costs)
        else:
            self.ledger=TaskDemandLedger(feedback=self.feedback)
            decider=CostFallbackDecider(self.ledger,profiler.costs)
        if task_intents is not None:
            from .task_intent import TaskIntentDecider
            decider = TaskIntentDecider(task_intents, decider, profiler.costs)
        self.predictor=predictor_factory(self.ledger) if predictor_factory else None
        self.advisor=SemanticPlanAdvisor(decider,active=active,max_calls=None,
            use_unique_plan=bundle_mode)
        # 这里的计数是本地决策次数；模型额度仍独立由 predictor 严格控制。
        runtime.governor.plan_advisor=self.advisor
        runtime.governor.lifecycle_profiler=profiler
        runtime.demand_coordinator=self

    @asynccontextmanager
    async def task(self, task_id, *, goal, metadata=None):
        if task_id in self.tasks:raise ValueError('task ID already active')
        token=self.ledger.begin(task_id);self.tasks.add(task_id)
        self.goals[token]=goal
        if self.requirement_registry:
            from .requirement_coverage import RequirementCoverageLedger
            self.requirement_ledgers[token]=RequirementCoverageLedger(goal,self.requirement_registry)
        self.observed_context[token]=deque(maxlen=4)
        ctx=self.context.set(token)
        self.events.append({'event':'task_started','token':token,**(metadata or {})})
        outcome='returned'
        try:
            scope = (self.task_intents.scope(token, (metadata or {}).get('board_task_ids', []), goal=goal)
                     if self.task_intents is not None else nullcontext())
            with scope:
                self._start_intent_prefetch()
                yield token
        except BaseException as error:
            outcome=type(error).__name__
            raise
        finally:
            try:
                if self.task_intents is not None:
                    await self.task_intents.drain_prefetch()
            finally:
                self.events.append({'event':'task_released','token':token,'outcome':outcome})
                coverage=self.requirement_ledgers.pop(token,None)
                if coverage is not None:coverage.close()
                self.ledger.finish(token);self.observed_context.pop(token,None);self.selected_calls.pop(token,None);self.goals.pop(token,None);self.tasks.remove(task_id);self.context.reset(ctx)

    async def prepare_prediction(self, requests, token):
        if token is not None and self.task_intents is not None:
            state, _ = self.advisor.snapshot(self.runtime.governor, requests)
            await self.task_intents.prepare(state, self.advisor.options(state), self.profiler.costs)
        if token is None or self.predictor is None or not self.ledger.needs_prediction(token):return
        state,_=self.advisor.snapshot(self.runtime.governor,requests)
        options=self.advisor.options(state)
        if len(options)<2:return
        victims={r for o in options for r in o['resource_ids']}
        if not victims <= self.profiler.costs.keys():return
        if self.bundle_mode:
            value=prediction_value_ceiling(options,self.profiler.costs)
            if value is None:
                self.events.append({'event':'prediction_skipped','token':token,
                    'reason':'baseline_or_cost_unresolved'})
                return
            ceiling=value['avoidable_cost_ceiling_s']
            overhead=getattr(self.predictor,'estimated_overhead_s',float('inf'))
            if isinstance(overhead,bool) or not isinstance(overhead,(int,float)) or not math.isfinite(overhead) or overhead<0 or ceiling<=overhead:
                self.events.append({'event':'prediction_skipped','token':token,
                    'reason':'avoidable_cost_not_above_overhead',**value})
                return
            await self.predictor.refresh(token,goal=self.goals[token],tool_resources=self.tool_resources,
                current_resources=tuple(dict(self.runtime.governor.expand_requirements(requests))),
                observed_context=self.prediction_context(token))
        else:
            await self.predictor.refresh(token,goal=self.goals[token],resource_tools=self.resource_tools)

    def _start_intent_prefetch(self):
        """容量已占满且存在多个已知成本的闲置资源时，机会性解释当前任务。"""
        if not self.intent_prefetch or self.runtime.snapshot()['waiting']:
            return
        try:
            governor = self.runtime.governor
            total = asdict(governor.total_declared_cost())
            if not any(limit is not None and limit > 0 and total[name] >= limit
                       for name,limit in asdict(governor.budget).items()):
                return
            protected = governor._active_protected_ids()
            protected.update(dict(governor.expand_requirements(
                (r,governor.catalog.get(r).pinned_min_level) for r in governor.pinned_ids)))
            resources = [s.capability_id for s in governor.states()
                if s.capability_id not in protected and not s.active_count
                and s.current_level > governor.catalog.get(s.capability_id).pinned_min_level
                and s.capability_id in self.profiler.costs
                and self.profiler.costs[s.capability_id] > 0]
            self.task_intents.start_prefetch(resources)
        except Exception as error:
            self.events.append({'event':'intent_prefetch_skipped','error':type(error).__name__})

    def select_tool_calls(self, token, calls):
        """记录已由Agent选定的当前批次；不读取下一条计划。"""
        if token not in self.observed_context:return
        coverage=self.requirement_ledgers.get(token)
        if coverage is not None:coverage.select(calls or ())
        selected=[]
        for call in list(calls or ())[:16]:
            if not isinstance(call,dict):continue
            args=json.dumps(call.get('args',{}),ensure_ascii=False,default=str)
            selected.append({'tool':call.get('name'), 'arguments_excerpt':args[:1024],
                'arguments_truncated':len(args)>1024})
        self.selected_calls[token]=selected

    def prediction_context(self, token):
        return {'completed_batches':list(self.observed_context.get(token,())),
            'selected_current_calls':self.selected_calls.get(token,[])}

    def record_tool_results(self, token, tool_names, result):
        """只记录本任务已返回的结果，保留有界片段；不读取计划中的后续调用。"""
        coverage=self.requirement_ledgers.get(token)
        if coverage is not None:coverage.record(result)
        history=self.observed_context.get(token)
        if history is None:return
        messages=result.get('messages',()) if isinstance(result,dict) else ()
        summaries=[]
        for message in list(messages)[-4:]:
            content=getattr(message,'content','')
            text=content if isinstance(content,str) else json.dumps(content,ensure_ascii=False,default=str)
            summaries.append({'tool':getattr(message,'name',None),
                'status':getattr(message,'status','unknown'),
                'result_excerpt':text[:1024],'truncated':len(text)>1024})
        history.append({'tools':list(tool_names)[:16],
            'selected_calls':self.selected_calls.pop(token,[]),'results':summaries})
        if self.task_intents is not None:
            self.task_intents.observe(token, history[-1])
            self._start_intent_prefetch()

    def token(self):return self.context.get()

    def completed(self, token, resource_ids):
        if token is None:return
        # 无论科研工具是否成功，实际调用已发生；撤销已消费预测，不做成功率学习冒充。
        accepted=self.ledger.observe(token,resource_ids)
        self.events.append({'event':'consumed','token':token,'resources':list(resource_ids),'accepted':accepted})
