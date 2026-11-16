"""按资源版本持久化生命周期观测；不把端点占用伪装成建立峰值。"""

from collections import deque

import hashlib

import json

import math

import sqlite3

import statistics

from time import time



def profile_fingerprint(*, artifacts, environment):

    """调用者提供模型/适配器内容哈希与设备/精度/线程配置，不使用易变路径代替版本。"""

    if not artifacts or not environment:raise ValueError('artifact and environment identity required')

    return hashlib.sha256(json.dumps({'artifacts':artifacts,'environment':environment},sort_keys=True,allow_nan=False).encode()).hexdigest()



class ResourceProfileStore:

    def __init__(self, path, *, clock=time):

        self.clock=clock

        self.db=sqlite3.connect(str(path),timeout=.2)

        self.db.execute('CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY, resource TEXT, fingerprint TEXT, at REAL, seconds REAL, success INTEGER, ready_rss_mb REAL)')

        self.db.execute('CREATE INDEX IF NOT EXISTS profile_key ON samples(resource,fingerprint,id)')

        self.db.execute('CREATE TABLE IF NOT EXISTS identities (resource TEXT, fingerprint TEXT, payload TEXT, PRIMARY KEY(resource,fingerprint))')

        self.db.execute('CREATE TABLE IF NOT EXISTS measurements (id INTEGER PRIMARY KEY, resource TEXT, fingerprint TEXT, at REAL, phase TEXT, payload TEXT)')

        self.db.commit()



    def record(self, resource, fingerprint, seconds, success, ready_rss_mb=None, observed_at=None):

        observed_at=self.clock() if observed_at is None else observed_at

        if not math.isfinite(observed_at):raise ValueError('invalid observation timestamp')

        if not resource or not fingerprint or not math.isfinite(seconds) or seconds<0:raise ValueError('invalid observation')

        if ready_rss_mb is not None and (not math.isfinite(ready_rss_mb) or ready_rss_mb<0):raise ValueError('invalid measurement')

        with self.db:

            self.db.execute('INSERT INTO samples(resource,fingerprint,at,seconds,success,ready_rss_mb) VALUES(?,?,?,?,?,?)',(resource,fingerprint,observed_at,seconds,int(success),ready_rss_mb))

            self.db.execute('DELETE FROM samples WHERE resource=? AND fingerprint=? AND id NOT IN (SELECT id FROM samples WHERE resource=? AND fingerprint=? ORDER BY id DESC LIMIT 64)',(resource,fingerprint,resource,fingerprint))



    def summary(self, resource, fingerprint, *, max_age_s=604800):

        if not math.isfinite(max_age_s) or max_age_s<=0:raise ValueError('invalid age')

        rows=self.db.execute('SELECT seconds,success,ready_rss_mb FROM samples WHERE resource=? AND fingerprint=? AND at>=? ORDER BY id',(resource,fingerprint,self.clock()-max_age_s)).fetchall()

        success=[r for r in rows if r[1]];times=[r[0] for r in success if r[0]>0];memory=[r[2] for r in success if r[2] is not None]

        return {'samples':len(rows),'successes':len(success),'failures':len(rows)-len(success),

                'prepare_seconds':statistics.median(times) if times else None,

                'max_observed_ready_rss_mb':max(memory) if memory else None,

                'memory_basis':'READY endpoint; not load or execution peak'}



    def record_identity(self, resource, fingerprint, identity):

        payload=json.dumps(identity,ensure_ascii=False,allow_nan=False)

        with self.db:self.db.execute('INSERT OR REPLACE INTO identities VALUES(?,?,?)',(resource,fingerprint,payload))



    def record_stage(self, resource, fingerprint, phase, measurement, observed_at=None):

        observed_at=self.clock() if observed_at is None else observed_at

        if not math.isfinite(observed_at):raise ValueError('invalid observation timestamp')

        if phase not in ('ready','execution_end','release_start'):raise ValueError('invalid measurement phase')

        payload=json.dumps(measurement,ensure_ascii=False,allow_nan=False)

        with self.db:

            self.db.execute('INSERT INTO measurements(resource,fingerprint,at,phase,payload) VALUES(?,?,?,?,?)',(resource,fingerprint,observed_at,phase,payload))

            self.db.execute('DELETE FROM measurements WHERE resource=? AND fingerprint=? AND id NOT IN (SELECT id FROM measurements WHERE resource=? AND fingerprint=? ORDER BY id DESC LIMIT 128)',(resource,fingerprint,resource,fingerprint))



    def inspect(self, resource, fingerprint):

        identity=self.db.execute('SELECT payload FROM identities WHERE resource=? AND fingerprint=?',(resource,fingerprint)).fetchone()

        measurements=self.db.execute('SELECT at,phase,payload FROM measurements WHERE resource=? AND fingerprint=? ORDER BY id',(resource,fingerprint)).fetchall()

        return {'identity':json.loads(identity[0]) if identity else None,'summary':self.summary(resource,fingerprint),

            'measurements':[{'at':at,'phase':phase,**json.loads(payload)} for at,phase,payload in measurements]}



    def close(self):self.db.close()



class LifecycleProfiler:

    """只登记明确归属的资源；数据库/采样失败可见，且不改变工具成功状态。"""

    def __init__(self, store, fingerprints, *, memory_sampler=None, identity_provider=None, process_sampler=None, resource_ids=None):

        self.store,self.fingerprints=store,dict(fingerprints)

        self.memory_sampler=memory_sampler

        self.identity_provider=identity_provider

        self.process_sampler=process_sampler

        self.resource_ids=set(resource_ids) if resource_ids is not None else set(fingerprints)

        self.errors=deque(maxlen=64)

        self.costs={}

        self.refresh()



    def refresh(self):

        fresh={}

        try:

            for resource,fingerprint in self.fingerprints.items():

                seconds=self.store.summary(resource,fingerprint)['prepare_seconds']

                if seconds is not None:fresh[resource]=seconds

        except Exception as e:

            self.errors.append({'stage':'restore','error':type(e).__name__})

        self.costs.clear();self.costs.update(fresh)



    def begin(self, capability):

        resource=capability.capability_id

        if resource not in self.resource_ids or self.identity_provider is None:return

        try:

            fingerprint,identity=self.identity_provider(capability)

            previous=self.fingerprints.get(resource)

            self.fingerprints[resource]=fingerprint

            self.store.record_identity(resource,fingerprint,identity)

            if previous!=fingerprint:self.costs.pop(resource,None)

            seconds=self.store.summary(resource,fingerprint)['prepare_seconds']

            if seconds is not None:self.costs[resource]=seconds

        except Exception as e:

            self.fingerprints.pop(resource,None);self.costs.pop(resource,None)

            self.errors.append({'resource':resource,'stage':'identity','error':type(e).__name__})



    def sample(self, capability, phase):

        resource=capability.capability_id

        if resource not in self.fingerprints or self.process_sampler is None:return None

        try:

            payload=self.process_sampler(capability)

            payload={'status':'measured',**payload}

        except Exception as e:

            payload={'status':'unavailable','error':type(e).__name__}

            self.errors.append({'resource':resource,'stage':phase,'error':type(e).__name__})

        try:self.store.record_stage(resource,self.fingerprints[resource],phase,payload)

        except Exception as e:self.errors.append({'resource':resource,'stage':'store_measurement','error':type(e).__name__})

        return payload



    def record(self, capability, seconds, success):

        resource=capability.capability_id

        if success and resource in self.resource_ids and resource not in self.fingerprints:

            # 某些模型路径由真实建立过程解析；仅在READY后补录实际身份。

            self.begin(capability)

        if resource not in self.fingerprints:return

        memory=None

        measurement=self.sample(capability,'ready') if success else None

        if measurement and measurement['status']=='measured':memory=measurement.get('rss_mb')

        if success and self.memory_sampler is not None:

            try:memory=self.memory_sampler(capability)

            except Exception as e:self.errors.append({'resource':resource,'stage':'sample','error':type(e).__name__})

        try:

            self.store.record(resource,self.fingerprints[resource],seconds,success,memory)

            summary=self.store.summary(resource,self.fingerprints[resource])

            if summary['prepare_seconds'] is not None:self.costs[resource]=summary['prepare_seconds']

        except Exception as e:self.errors.append({'resource':resource,'stage':'store','error':type(e).__name__})

