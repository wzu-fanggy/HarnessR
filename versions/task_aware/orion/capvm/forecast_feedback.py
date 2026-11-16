"""完整预测窗口的经验误差门控；非概率校准器、非最坏情况性能保证。"""

from collections import OrderedDict,deque

import math



class ForecastFeedback:

    """先影子观察再启用；失准后继续影子评分，因此可以恢复。"""

    def __init__(self, *, min_windows=8, window=32, max_sources=32, threshold=.25):

        if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (min_windows,window,max_sources)) or min_windows>window:

            raise ValueError('invalid feedback window')

        if not math.isfinite(threshold) or not 0<threshold<1:raise ValueError('invalid threshold')

        self.min_windows,self.window,self.max_sources,self.threshold=min_windows,window,max_sources,threshold

        self.history=OrderedDict();self.events=deque(maxlen=256)



    def record(self, source, weights, observed):

        if not source or not weights:raise ValueError('source and forecast required')

        if any(isinstance(p,bool) or not isinstance(p,(float,int)) or not math.isfinite(p) or not 0<=p<=1 for p in weights.values()):raise ValueError('invalid forecast')

        loss=sum((p-float(r in observed))**2 for r,p in weights.items())/len(weights)

        return self._record_loss(source, loss)



    def record_intervals(self, source, intervals, observed):

        """区间内最坏端点平方误差；未知区间不能靠宽范围获得虚假好评。"""

        if not source or not intervals:raise ValueError('source and intervals required')

        losses=[]

        for resource,bounds in intervals.items():

            if len(bounds)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in bounds) or not 0<=bounds[0]<=bounds[1]<=1:

                raise ValueError('invalid forecast interval')

            target=float(resource in observed)

            losses.append(max((bound-target)**2 for bound in bounds))

        return self._record_loss(source,sum(losses)/len(losses))



    def _record_loss(self, source, loss):

        bucket=self.history.setdefault(source,deque(maxlen=self.window))

        self.history.move_to_end(source);bucket.append(loss)

        while len(self.history)>self.max_sources:self.history.popitem(last=False)

        result=self.state(source);self.events.append({'source':source,'loss':loss,**result})

        return result



    def state(self, source):

        values=self.history.get(source,())

        mean=sum(values)/len(values) if values else None

        return {'windows':len(values),'mean_squared_error':mean,

                'allowed':len(values)>=self.min_windows and mean<self.threshold}



    def allowed(self, source):return self.state(source)['allowed']

