"""工具组合映射共享资源；未知相关性下使用区间启发式，不假装概率已校准。"""
import math
from .task_demand import TaskDemandLedger


class ToolBundleLedger(TaskDemandLedger):
    def __init__(self, tool_resources, **kwargs):
        super().__init__(**kwargs)
        self.tool_resources = {t: tuple(sorted(set(rs))) for t, rs in tool_resources.items()}
        self.resources = set(r for rs in self.tool_resources.values() for r in rs)

    def publish(self, *args, **kwargs):
        raise TypeError('bundle ledger requires publish_bundles, not independent resource scores')

    def publish_bundles(self, token, scenarios, *, unknown_weight=0, ttl_s,
                        source=None, horizon_batches=2, skip_current=True):
        """各场景是互斥的未来窗口工具集合；空集合表示没有目录内资源需求。

        other/未知权重保留，不能重归一化成已知需求。当前已确定调用不计入未来窗口。
        """
        task = self._tasks.get(token[0])
        if task is None or task['token'] != token:
            return False
        scenarios = list(scenarios)
        if not scenarios or len(scenarios) > 16 or not isinstance(skip_current, bool):
            raise ValueError('invalid bundle forecast')
        def weight(value):
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('invalid scenario weight')
            return value
        unknown_weight = weight(unknown_weight)
        marginal = dict.fromkeys(self.resources, 0.)
        clean = []
        for scenario in scenarios:
            tools = scenario['tools']
            if not isinstance(tools, (list,tuple)) or any(t not in self.tool_resources for t in tools):
                raise ValueError('unregistered tool in forecast')
            mass = weight(scenario['weight'])
            resources = {r for t in tools for r in self.tool_resources[t]}
            for r in resources:
                marginal[r] += mass
            clean.append({'tools': sorted(set(tools)), 'resources': sorted(resources), 'weight': mass})
        if not math.isclose(sum(s['weight'] for s in clean)+unknown_weight, 1., abs_tol=1e-8):
            raise ValueError('scenario weights including unknown must sum to one')
        marginal = {r:min(1.,p) for r,p in marginal.items()}
        accepted = super().publish(token, marginal, ttl_s=ttl_s, source=source,
                                   horizon_batches=horizon_batches)
        if accepted:
            task['forecast']['unknown_weight'] = unknown_weight
            task.update(bundle_scenarios=clean, unknown_weight=unknown_weight,
                        skip_observations=int(skip_current), consumed=set(), decision_valid=True)
        return accepted

    def needs_prediction(self, token):
        task = self._tasks.get(token[0])
        return bool(task and task['token']==token and
                    (super().needs_prediction(token) or not task.get('decision_valid',False)))

    def observe(self, token, resource_ids):
        resource_ids = tuple(resource_ids)
        task = self._tasks.get(token[0])
        if task is None or task['token'] != token:
            return False
        if task.get('skip_observations', 0):
            task['skip_observations'] -= 1
            task['observations'] += 1
            self.events.append({'status':'excluded_current_batch','token':token})
            return True
        task.setdefault('consumed', set()).update(resource_ids)
        # 已看到一批未来调用后，原窗口分布不等于剩余窗口的条件分布。
        # 保留原预测收集完整反馈，但停止据它决策；不能将“用过”推成“不再用”。
        task['decision_valid'] = False
        return super().observe(token, resource_ids)

    def _score_forecast(self, forecast):
        if forecast['source'] is not None and self.feedback is not None:
            intervals = {r: [p, min(1.,p+forecast['unknown_weight'])]
                         for r,p in forecast['weights'].items()}
            self.feedback.record_intervals(forecast['source'], intervals, forecast['seen'])

    def snapshot(self):
        """汇总本次回收后首次重载的启发式损失，每个资源最多计一次。

        max(lower) / min(1,sum(upper)) 不假定并发任务独立；未经校准的输入
        使其仅为评分区间，不是统计置信区间或真实耗时保证。
        """
        rows = []
        complete = bool(self._tasks)
        for task_id, task in self._tasks.items():
            forecast = task['forecast']
            usable = (forecast is not None and task.get('decision_valid',False) and task['expires'] > self.clock()
                and (forecast['source'] is None or self.feedback is None
                     or self.feedback.allowed(forecast['source'])))
            if not usable:
                complete = False
                continue
            unknown = task['unknown_weight']
            intervals = {}
            for r in self.resources:
                low = task['weights'].get(r, 0.)
                high = min(1., low+unknown) if r not in task['consumed'] else 0.
                intervals[r] = [low, high]
            rows.append({'task':task_id, 'version':task['token'][1],
                         'intervals':intervals, 'scenarios':task['bundle_scenarios']})
        intervals = {r: [max(row['intervals'][r][0] for row in rows),
                        min(1., sum(row['intervals'][r][1] for row in rows))]
                     for r in self.resources} if rows else {}
        return {'complete':complete, 'intervals':intervals, 'contributors':rows}


class BundlePlanDecider:
    def __init__(self, ledger, costs):
        self.ledger, self.costs = ledger, costs
        self.events = []

    def decision_state(self):
        return {'demand':self.ledger.snapshot(), 'costs':dict(self.costs)}

    async def __call__(self, state, options):
        victims = {r for o in options for r in o['resource_ids']}
        if len(options)<2 or any(r not in self.costs or isinstance(self.costs[r],bool)
                or not math.isfinite(self.costs[r]) or self.costs[r]<=0 for r in victims):
            return None
        demand = self.ledger.snapshot()
        intervals = demand['intervals']
        scores = {}
        if demand['complete'] and victims <= intervals.keys():
            scores = {o['id']: [sum(self.costs[r]*intervals[r][i]
                        for r in set(o['resource_ids'])) for i in (0,1)] for o in options}
            winner = next((key for key, bound in scores.items()
                if all(bound[1] < other[0]-1e-9 for k,other in scores.items() if k!=key)), None)
            if winner is not None:
                self._record({'source':'bundle_interval','choice':winner,'scores':scores})
                return winner
        # 区间重叠、缺失任务预测或门控未通过时，使用原实测成本基线。
        ranked = sorted((sum(self.costs[r] for r in set(o['resource_ids'])),o['id']) for o in options)
        if math.isclose(ranked[0][0],ranked[1][0]):
            return None
        self._record({'source':'observed_cost_fallback','choice':ranked[0][1], 'scores':scores})
        return ranked[0][1]

    def _record(self, event):
        self.events.append(event)
        del self.events[:-128]
