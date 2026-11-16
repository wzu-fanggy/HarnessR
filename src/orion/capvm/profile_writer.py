"""有界后台画像写入；运行时读内存镜像，磁盘延迟不进入每次建立。"""
from collections import deque
import json
import math
from time import time
from pathlib import Path
from queue import Queue,Empty,Full
from threading import Event,Thread
from .resource_profiles import ResourceProfileStore

class BufferedProfileStore:
    def __init__(self, path, *, queue_limit=128, flush_timeout_s=5):
        if isinstance(queue_limit,bool) or not isinstance(queue_limit,int) or queue_limit<1:raise ValueError('invalid queue size')
        if not math.isfinite(flush_timeout_s) or flush_timeout_s<=0:raise ValueError('invalid flush timeout')
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.live=ResourceProfileStore(':memory:')
        disk=ResourceProfileStore(self.path)
        try:disk.db.backup(self.live.db)
        finally:disk.close()
        self.errors=deque(maxlen=64);self.queue=Queue(maxsize=queue_limit)
        self.stop=Event();self.closed=False;self.flush_timeout_s=flush_timeout_s;self.flush_complete=None
        self.worker=Thread(target=self._run,name='orion-profile-writer',daemon=True);self.worker.start()

    def _run(self):
        disk=None
        try:
            disk=ResourceProfileStore(self.path)
            while not self.stop.is_set() or not self.queue.empty():
                try:method,args=self.queue.get(timeout=.05)
                except Empty:continue
                try:getattr(disk,method)(*args)
                except Exception as e:self.errors.append({'stage':'write','method':method,'error':type(e).__name__})
                finally:self.queue.task_done()
        except Exception as e:self.errors.append({'stage':'writer','error':type(e).__name__})
        finally:
            if disk is not None:disk.close()

    def _write(self, method, args):
        if self.closed:raise RuntimeError('profile store closed')
        # 深拷贝确保后台写入的就是当时快照，不读取随后被修改的字典。
        args=json.loads(json.dumps(args,allow_nan=False))
        getattr(self.live,method)(*args)
        try:self.queue.put_nowait((method,args))
        except Full:self.errors.append({'stage':'queue','method':method,'error':'QueueFull'})

    def record(self,resource,fingerprint,seconds,success,ready_rss_mb=None):
        self._write('record',(resource,fingerprint,seconds,success,ready_rss_mb,time()))
    def record_identity(self,*args):self._write('record_identity',args)
    def record_stage(self,resource,fingerprint,phase,measurement):
        self._write('record_stage',(resource,fingerprint,phase,measurement,time()))
    def summary(self,*args,**kwargs):return self.live.summary(*args,**kwargs)
    def inspect(self,*args):return self.live.inspect(*args)

    def close(self):
        if self.closed:return self.flush_complete
        self.closed=True;self.stop.set();self.worker.join(self.flush_timeout_s)
        self.flush_complete=not self.worker.is_alive() and self.queue.empty() and not self.errors
        if self.worker.is_alive():self.errors.append({'stage':'close','error':'FlushTimeout'})
        self.live.close()
        return self.flush_complete

