"""请求级需求登记与共享资源汇总；分数是预测权重，不是校准概率。"""

from time import monotonic

from collections import deque

import math



class TaskDemandLedger:

    """同一事件循环使用。版本令牌防止迟到预测覆盖任务的新阶段。"""

    def __init__(self, *, clock=monotonic, max_tasks=128, feedback=None):

        if isinstance(max_tasks,bool) or not isinstance(max_tasks,int) or max_tasks<1:

            raise ValueError('invalid task limit')

        self.clock,self.max_tasks=clock,max_tasks

        self.feedback=feedback

        self.events=deque(maxlen=256)

        self._generation=0

        self._tasks={}



    def begin(self, task_id):

        if not isinstance(task_id,str) or not task_id.strip(): raise ValueError('task ID required')

        if task_id not in self._tasks and len(self._tasks)>=self.max_tasks:

            raise ValueError('active task limit reached')

        old=self._tasks.get(task_id)

        if old is not None:self._censor(old,'phase_changed')

        self._generation+=1

        token=(task_id,self._generation)

        self._tasks[task_id]={'token':token,'expires':0,'weights':{},'observations':0,'forecast':None}

        return token



    def revision(self, token):

        """阶段或已观察调用变化时，正在返回的旧预测不得覆盖新状态。"""

        task=self._tasks.get(token[0])

        return (token,task["observations"]) if task and task["token"]==token else None



    def needs_prediction(self, token):

        task=self._tasks.get(token[0])

        return bool(task and task["token"]==token and (task["forecast"] is None or task["expires"]<=self.clock()))



    def publish(self, token, weights, *, ttl_s, source=None, horizon_batches=2):

        task=self._tasks.get(token[0])

        if task is None or task['token']!=token: return False

        if not math.isfinite(ttl_s) or ttl_s<=0: raise ValueError('invalid lifetime')

        if isinstance(horizon_batches,bool) or not isinstance(horizon_batches,int) or horizon_batches<1:raise ValueError("invalid horizon")

        clean=dict(weights)

        if any(not isinstance(k,str) or not k or isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for k,v in clean.items()):

            raise ValueError('invalid resource weights')

        self._censor(task,"replaced")

        task.update(weights=clean,expires=self.clock()+ttl_s,forecast={"weights":dict(clean),"source":source,"seen":set(),"batches":0,"horizon":horizon_batches})

        return True



    def observe(self, token, resource_ids):

        """消费本任务的一次近程需求；其他任务对同一资源的需求继续有效。"""

        task=self._tasks.get(token[0])

        if task is None or task['token']!=token:return False

        forecast=task['forecast']

        resources=set(resource_ids)

        if forecast is not None:

            if task['expires']<=self.clock():

                self._censor(task,'expired')

                task['weights'].clear()

            else:

                forecast['seen'].update(resources);forecast['batches']+=1

                if forecast['batches']>=forecast['horizon']:

                    self._score_forecast(forecast)

                    self.events.append({'status':'observed','token':token,'source':forecast['source'],

                                        'weights':forecast['weights'],'observed':sorted(forecast['seen'])})

                    task['forecast']=None;task['weights'].clear()

        for resource in resources:task['weights'].pop(resource,None)

        task['observations']+=1

        return True



    def _score_forecast(self, forecast):

        if forecast['source'] is not None and self.feedback is not None:

            self.feedback.record(forecast['source'],forecast['weights'],forecast['seen'])



    def _censor(self, task, reason):

        forecast=task.get('forecast')

        if forecast is not None:

            self.events.append({'status':'censored','reason':reason,'token':task['token'],

                                'source':forecast['source'],'observed_batches':forecast['batches']})

            task['forecast']=None



    def finish(self, token):

        task=self._tasks.get(token[0])

        if task is None or task['token']!=token:return False

        self._censor(task,'task_finished')

        del self._tasks[token[0]]

        return True



    def snapshot(self):

        now=self.clock(); totals={};contributors={}

        for task_id,task in self._tasks.items():

            if task['expires']<=now:continue

            forecast=task['forecast']

            if forecast is not None and forecast['source'] is not None and self.feedback is not None and not self.feedback.allowed(forecast['source']):continue

            for resource,weight in task['weights'].items():

                totals[resource]=totals.get(resource,0)+weight

                contributors.setdefault(resource,[]).append({'task':task_id,'version':task['token'][1],'weight':weight})

        return {'weights':totals,'contributors':contributors}



class DemandPlanDecider:

    """汇总需求乘历史建立成本作为启发式损失；不把它宣称为期望耗时。"""

    def __init__(self, ledger, observed_reload_seconds):

        self.ledger=ledger

        self.costs=observed_reload_seconds

        self.events=[]



    def decision_state(self):

        return {"demand":self.ledger.snapshot(),"costs":dict(self.costs)}



    async def __call__(self, state, options):

        demand=self.ledger.snapshot();weights=demand['weights']

        victims={r for option in options for r in option['resource_ids']}

        if not victims or not victims <= weights.keys() or not any(weights.get(r,0)>0 for r in victims):return None

        if any(r not in self.costs or isinstance(self.costs[r],bool) or not math.isfinite(self.costs[r]) or self.costs[r]<=0 for r in victims):return None

        scores={o['id']:sum(weights.get(r,0)*self.costs[r] for r in o['resource_ids']) for o in options}

        ranked=sorted(options,key=lambda o:(scores[o['id']],o['id']))

        # 无明确区别就交还原策略，不按方案名称制造优势。

        if len(ranked)<2 or math.isclose(scores[ranked[0]['id']],scores[ranked[1]['id']]):return None

        chosen=ranked[0]['id']

        self.events.append({'demand':demand,'scores':scores,'choice':chosen})

        del self.events[:-128]

        return chosen

