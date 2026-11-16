"""一次真实化学接续验证；复用归档结果，共享运行时贯穿两个阶段。"""
from release_paths import release_path, resolve_json

import asyncio

from collections import Counter

from contextlib import redirect_stdout, redirect_stderr

from contextvars import ContextVar

from copy import deepcopy

from dataclasses import asdict

import hashlib

import json

import math

import os

from pathlib import Path

import sys

from threading import RLock

from time import perf_counter

import traceback



ROOT = Path(__file__).resolve().parent

ARCHIVE = ROOT.parent/'sharing'

REPO = Path(release_path('${ORION_SOURCE}'))

OUT = ROOT.parents[1]/'outputs/integrated'

P,R,F = IDS = ('process:na-conductivity-rf','process:robust-ranking-worker','process:transport-fit-kernel')

SCIENCE = ('predict_formulation_conductivity','rank_formulations_under_uncertainty','fit_conductivity_temperature_law')

INPUTS = json.loads((ROOT.parents[1]/'data/integrated_inputs.json').read_text(encoding='utf-8'))
INPUTS = resolve_json(INPUTS)

PHASE = ContextVar('core_validation_phase',default='setup')

sys.path[:0] = [str(ARCHIVE),str(REPO.parent),str(REPO)]

sys.dont_write_bytecode = True



def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()

def compact(v): return json.dumps(v,ensure_ascii=False,separators=(',',':'))



SETTINGS = {

 'PYTHONDONTWRITEBYTECODE':'1','ORION_ENV_FILE':os.environ.get('ORION_ENV_FILE',str(REPO/'.env')),

 'ORION_RESOURCE_POOL_ENABLED':'1','ORION_RESOURCE_ADMISSION_QUEUE':'1',

 'ORION_RESOURCE_COORDINATION':'1','ORION_RESOURCE_TASK_INTENT':'1',

 'ORION_RESOURCE_INTENT_JEV_MAX_CALLS':'0','ORION_RESOURCE_INTENT_PREFETCH':'0',

 'ORION_RESOURCE_REQUIREMENT_COVERAGE':'0',

 'ORION_RESOURCE_INTENT_RULE':'minimax_regret','ORION_RESOURCE_HOST_GUARD':'1',

 'ORION_RESOURCE_HOST_RAM_MB':'3072','ORION_RESOURCE_PROCESS_SLOTS':'2',

 'ORION_RESOURCE_AUTO_MEMORY':'0','ORION_RESOURCE_ADAPTIVE_RESERVE':'0',

 'ORION_RESOURCE_PROFILE_PATH':str(OUT/'profiles.db'),

 'ORION_NA_CONDUCTIVITY_MODEL_PATH':INPUTS['model']['path'],

 'ORION_ROBUST_RANKING_BUNDLE_DIR':release_path('${PROJECT_SOURCE}/reports/cwsr_inputs/na_training_bundle_v1'),

 'ORION_TASKS_DIR':str(OUT/'tasks'),'LTM_DB_PATH':str(OUT/'facts.db'),

 'LTM_VECTORS_DB_PATH':str(OUT/'vectors.db'),

 'ORION_REFLECTION_STORE_PATH':str(OUT/'reflections.db'),

 'ORION_CAPABILITY_STATE_ROOT':str(OUT/'capability-state'),'DEEPAGENTS_ROOT':str(OUT),

 'ORION_AUTO_REFLECT':'0','ORION_REFLECTION_INJECT':'0','ORION_LLM_CACHE':'0',

 'ORION_LLM_STREAM':'0','ORION_AUTO_APPROVE':'1','ORION_LLM_REQUEST_MAX_RETRIES':'0',

 'ORION_LLM_STEP_TIMEOUT':'90','LANGCHAIN_TRACING_V2':'false','LANGSMITH_TRACING':'false',

 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}



CANDIDATES = [deepcopy(r['candidate']) for r in INPUTS['candidates']]

REFS = [{'candidate_id':r['reference_id'],'salt':'NaPF6','solvents':['EC','EMC','PC'],

         'additives':[],'component_mmol':r['component_mmol']} for r in INPUTS['references']]

ALL = {c['candidate_id']:c for c in CANDIDATES+REFS}

SWEEPS = {}

for ref in INPUTS['references']:

    for sweep in ref['source_sweeps']:

        label = ref['reference_id']+'-'+sweep['source_sweep_id']

        SWEEPS[label] = {'formulation_id':label,'measurements':[

            {'temperature_c':m['temperature_c'],'conductivity_ms_cm':m['conductivity_ms_cm']}

            for m in sweep['measurements']], 'provenance':{

                'kind':'published_measurements','source_zip_sha256':INPUTS['source']['sha256'],

                'member':ref['source_member'],'row_zero_based':[m['row_zero_based'] for m in sweep['measurements']],

                'source_sweep_id':sweep['source_sweep_id'],

                'independence_of_sweeps':'not established'}}



BASE_INSTRUCTIONS = (

 '这是一次有固定研究阶段的钠离子电池电解液候选复核。仅使用本请求给定的配方、温度、阈值和真实工具返回。'

 '已有预测直接引用，不重复计算，不生成新候选，不做额外检索，不调用工具来预热。'

 '模型预测、集成成员达标比例和文献测量严格区分。参照配方可能在训练数据中，不称为独立留出验证。'

 '工具返回后直接综合回答即可，无须另外生成文件。若计算或资源准入失败，如实说明，不无限重试。'

 '不虚构实验测量，不把拟合优度解释为化学机理或实验验证。'

)

OLD_POINTS = [{'candidate_id':r['candidate']['candidate_id'], 'composition_id':r['composition_id'],

               'predictions':[{'temperature_c':x['temperature_c'],'mS_cm':x['prediction_mS_cm'],

                               'record_id':x['record_id']} for x in r['existing_predictions']]}

              for r in INPUTS['candidates']]

PROMPT_A = BASE_INSTRUCTIONS + '\n你负责阶段一的候选复核。候选来自既有三批96组成初筛，不重跑初筛。\n候选：'+compact(CANDIDATES)+(

 '\n已完成的冻结点模型预测：'+compact(OLD_POINTS)+

 '\n请补齐这三候选在-10°C的点预测，结合原有-20、0、20°C比较四温度电导率。'

 '再使用既有多温度排序工具，对相同三候选、temperature_c=[-20,-10,0,20]、'

 'min_conductivity_ms_cm=2.0和ensemble_budget=32进行一次成员一致性比较。'

 '2.0 mS/cm是本项任务给定比较目标，不是通用电解液标准。'

 '点RF与组自助集成是不同方法。不要将两者差异解释为资源层改变了科学预测。完整component_mmol可直接放在候选对象中；不要自行另算或添加模型特征。'

 '按集成联合达标比例优先、最差温度中位电导率次优的顺序提出优先项；科学指标并列保留并列，'

 '不要把工具按候选编号作的显示排序解释为物理优势。给出结果表、还需解释的温度问题和后续测量建议。')

PROMPT_B1 = BASE_INSTRUCTIONS + '\n你负责阶段一的测量参照核对。参照配方是事前按组成距离确定的两种公开测量配方。\n配方：'+compact(REFS)+(

 '\n请对每个参照组成在-20、-10、0、20°C各作一次冻结点RF预测。两个参照可在每个温度合批。'

 'predict_formulation_conductivity的temperature_c是一个数值，不是温度数组。四个温度可并列提交四个工具调用，每个调用包含两个参照。完整component_mmol原样保留，不另算模型特征。同一组成温度只预测一次，然后与该温度三条原始扫描分别核对，不平均、不挑扫描。'

 '本阶段只作这些点预测与测量核对；温度曲线解释属于接下来的阶段。\n测量：')+compact([

    {'reference_id':r['reference_id'],'sweeps':[

        {'sweep_id':s['source_sweep_id'],'measurements':[m for m in s['measurements'] if m['temperature_c'] in (-20,-10,0,20)]}

        for s in r['source_sweeps']]} for r in INPUTS['references']])



def follow_prompt(handoff=None):

    return BASE_INSTRUCTIONS + (

      '\n你负责阶段二的温度解释及追加核查。阶段一的候选点预测、四温度集成比较和测量参照核对已经完成，'

      '本阶段不重新做候选排序或已完成的点预测。'

      '先对下列两种参照配方的全部六条原始温度扫描分别调用温度关系拟合，比较Arrhenius和VTF的AICc及留一误差。'

      '不得删除、平均或替换扫描，也不把六条扫描称为独立实验重复。'

      '阅读拟合结果，解释这些原始测量的温度趋势与差异，然后对交接表中所有科学并列优先候选作25°C点RF追加核查。'

      '追加预测应在收到温度拟合结果后进行，不能把尚未返回的拟合当作已分析。'

      '拟合是参照组成的测量拟合，不是候选配方的实测。拟合统计不代表建立了输运机理。'

      '如果拟合不理想也保留并解释，不因此换参照或只选好曲线。完成后给出配方优先与测量温区建议。'

      '\n六条完整原始拟合输入：')+compact(list(SWEEPS.values()))+'\n阶段一交接：'+compact(handoff or {

         'status':'阶段一尚未完成，交接将以实际工具输出更新',

         'candidate_pool':CANDIDATES,'future_check_temperature_c':25,

         'tie_rule':'保持联合达标比例和最差温度电导率两项科学指标的全部并列'})



def static_check():

    assert sha(INPUTS['model']['path']) == INPUTS['model']['sha256']

    assert sha(INPUTS['source']['path']) == INPUTS['source']['sha256']

    assert len(SWEEPS)==6 and sum(len(x['measurements']) for x in SWEEPS.values())==60

    for item in INPUTS['candidates']:

        for row in item['existing_predictions']:

            record=json.loads(Path(row['record_path']).read_text(encoding='utf-8'))

            assert record['model_reference']['artifact_sha256']==INPUTS['model']['sha256']

    # Private manuscript hashes are not runtime inputs to the release.



def prepare():

    static_check()

    OUT.mkdir(exist_ok=True)

    if (OUT/'started.json').exists(): raise RuntimeError('本次执行已开始，不允许从头重复运行')

    for name,value in [('A_request.txt',PROMPT_A),('B1_request.txt',PROMPT_B1),('B2_pending_request.txt',follow_prompt())]:

        (OUT/name).write_text(value,encoding='utf-8')

    sources=[Path(__file__),ROOT.parents[1]/'data/integrated_inputs.json',ARCHIVE/'forward_queue.py',ARCHIVE/'isolation.py',

        REPO/'main.py',REPO/'core/default_resource_runtime.py',REPO/'core/resource_task_graph.py',

        REPO/'core/langgraph_residency.py',REPO/'core/resource_tool_node.py',REPO/'capvm/shared_runtime.py',

        REPO/'capvm/task_intent.py',REPO/'capvm/jev_task_intent.py',REPO/'capvm/semantic_plan.py',

        REPO/'capvm/demand_coordinator.py',REPO/'capvm/governor.py',Path(INPUTS['model']['path'])]

    sources+=list((REPO/'tools/electrolyte_design').glob('*.py'))

    sources+=list(Path(SETTINGS['ORION_ROBUST_RANKING_BUNDLE_DIR']).glob('*.csv'))

    protocol={'status':'frozen_before_API_or_scientific_calls','inputs_sha256':sha(ROOT.parents[1]/'data/integrated_inputs.json'),

      'source_hashes':{str(p):sha(p) for p in sources},'settings':SETTINGS,

      'prompts':{'A':PROMPT_A,'B1':PROMPT_B1,'B2_pending':follow_prompt()},

      'execution_kind':'one prospectively staged two-agent scientific integration episode',

      'max_primary_requests':{'A':4,'B_total':8},'max_jev_calls':2,'max_seconds':1800,

      'tools':SCIENCE,'no_replay_of_old_science':True,'technical_retry_of':'execution','minimum_recompute_points_if_authorized':11,'aggregate_primary_request_limit_including_attempt':24,'no_access_order_tuning':True,

      'reference_point_temperatures_c':[-20,-10,0,20],

      'priority_rule':['maximum joint_meets_constraint_probability','maximum worst_temperature_median_ms_cm','preserve all ties'],

      'jev_trigger':'first genuine multi-option resource decision; explicit consultation, then normal validation; one refresh only if invalidated',

      'economic_gate':'consultation is explicit for this mechanism validation; no claim of net acceleration',

      'working_file_hashes':{}}

    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')

    print(compact({'status':'prepared','folder':str(OUT),'model_or_science_calls':0,'raw_curves':6,'raw_measurements':60}))



async def execute():

    static_check()

    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))

    assert all(sha(p)==h for p,h in protocol['source_hashes'].items()),'冻结后源码改变'

    if (OUT/'started.json').exists(): raise RuntimeError('禁止重复启动这一轮')

    os.environ.update(SETTINGS)

    import env_utils

    os.environ.update(SETTINGS)

    os.environ.pop('ORION_RESOURCE_EVENTS_PATH',None)

    os.environ.pop('ORION_DESIGN_REASONING_MODE',None)

    os.chdir(OUT)

    secrets=[v for k,v in os.environ.items() if any(x in k for x in ('KEY','TOKEN','PASSWORD')) and len(v)>12]

    def safe(v):

        text=json.dumps(v,ensure_ascii=False,default=str)

        for secret in secrets: text=text.replace(secret,'[REDACTED]')

        return json.loads(text)

    def dump(name,v): (OUT/name).write_text(json.dumps(safe(v),ensure_ascii=False,indent=2),encoding='utf-8')

    from orion.capvm.host_memory import windows_host_memory

    initial=asdict(windows_host_memory())

    if initial['physical_available_mb']<4800 or initial['commit_available_mb']<8000:

        dump('pre_execution_block.json',{'reason':'initial_host_headroom','host':initial,'calls':0})

        print(compact({'status':'host_headroom_blocked','host':initial})); return

    dump('started.json',{'host':initial,'source_protocol_sha256':sha(OUT/'protocol.json')})

    from isolation import agent_scope,WorkspaceRoot,OwnedBoard

    from forward_queue import SubmittedRequests,QueueAwareIntentController,announce_received_request

    from langchain_core.messages import AIMessage,HumanMessage,ToolMessage

    from langchain_core.callbacks import BaseCallbackHandler

    from orion.core.tasks import TaskBoard,TaskStore,set_board

    from orion.core.budget import BudgetTracker

    from orion.core.task_board_intent import TaskBoardIntentSource

    from orion.capvm.jev_task_intent import JevTaskIntentAssessor

    import tools.file_operate as file_ops

    file_ops.WORKDIR=WorkspaceRoot(OUT)

    raw_board=TaskBoard(TaskStore(OUT/'tasks'));owned_board=OwnedBoard(raw_board);set_board(owned_board)

    for owner in ('A','B'): (OUT/owner).mkdir(exist_ok=True)

    started=perf_counter();events=[];lock=RLock();runtime=None;controller=None

    primary=Counter();science=[];cached={};point_cache={};generations=Counter();outcomes={};graphs=[]

    def emit(event,**values):

        with lock:

            row=safe({'sequence':len(events),'elapsed_s':perf_counter()-started,

                      'phase':PHASE.get(),'agent':agent_scope.get(),'event':event,**values})

            events.append(row)

            with (OUT/'events.jsonl').open('a',encoding='utf-8') as f: f.write(compact(row)+'\n')

        return row

    def snapshots(closed=False):

        if runtime is None:return []

        return [{'resource_id':r,'pid':runtime.governor.catalog.get(r).adapter.pid,

                 'generation':generations[r], 'active_count':None if closed else runtime.governor.state(r).active_count,

                 'level':None if closed else runtime.governor.state(r).current_level.name,

                 'last_used_tick':None if closed else runtime.governor.state(r).last_used_tick,

                 'model_sha256':INPUTS['model']['sha256'] if r==P else None} for r in IDS]

    class Recorder(BaseCallbackHandler):

        raise_error=True

        def on_chat_model_start(self,serialized,messages,**kwargs):

            owner=agent_scope.get()

            primary[owner]+=1

            if primary[owner]>(4 if owner=='A' else 8): raise RuntimeError('主模型请求额度已用尽')

            emit('llm_start',request=primary[owner],messages=[m.model_dump(mode='json') for m in messages[0]])

        def on_llm_end(self,response,**kwargs): emit('llm_end',usage=response.llm_output)

        def on_llm_error(self,error,**kwargs): emit('llm_error',error_type=type(error).__name__)

    queue=SubmittedRequests();first_decision_version=None;priority=[]

    class ObservedController(QueueAwareIntentController):

        async def prepare(self,state,options,costs):

            nonlocal first_decision_version

            if len(options)>=2:

                snap=self.snapshot();resource_ids=sorted({r for o in options for r in o['resource_ids']})

                rows={r['id']:r for r in state['resources']}

                lru=min(options,key=lambda o:tuple(sorted((rows[r]['tick'],r) for r in o['resource_ids'])))['id']

                cost=min(options,key=lambda o:(sum(costs[r] for r in o['resource_ids']),o['id']))['id'] if all(r in costs for r in resource_ids) else None

                no_task=await runtime.bridge.shared_runtime.demand_coordinator.advisor.decide.fallback(state,options)

                emit('actual_decision_before',task_snapshot=snap,state=state,options=options,

                     costs=dict(costs),lru_reference=lru,cost_reference=cost,no_task_reference=no_task,

                     resources=snapshots())

                req=self._requirements(snap,resource_ids)

                first=first_decision_version is None

                refresh=first_decision_version is not None and snap.get('version')!=first_decision_version and self.assessor.calls==1

                if req is not None and req[1] and self.assessor.calls<2 and (first or refresh):

                    first_decision_version=snap.get('version')

                    desc={r:{k:v for k,v in self.resource_descriptions[r].items() if k!='literal_terms'} for r in resource_ids}

                    built=self.assessor._request(req[1],desc)

                    if built is None:raise RuntimeError('Jev请求不能构建，保留现场，不伪称完成判断')

                    emit('jev_send_preview',destination='https://api.typesafe.ai/v1/systemone',body=built[0],

                         task_version=snap['version'])

                    answer=await self.assessor.assess(req[1],desc)

                    emit('jev_response',answer=answer,events=list(self.assessor.events),

                         same_task_version=self.snapshot().get('version')==snap.get('version'))

            await super().prepare(state,options,costs)

            if len(options)>=2:emit('decision_hint_ready',intent_events=list(self.events),state=self.decision_state())

    try:

        import orion.main as main

        import model_config

        runtime=main._default_resource_runtime

        shared=runtime.bridge.shared_runtime;coordinator=shared.demand_coordinator

        old=coordinator.task_intents

        assessor=JevTaskIntentAssessor(os.environ['TYPESAFE_API_KEY'],max_calls=2,timeout_s=20,ttl_s=60)

        controller=ObservedController(TaskBoardIntentSource(lambda:raw_board),old.tool_resources,

            old.resource_descriptions,assessor=assessor,submitted_requests=queue,

            selection_rule='minimax_regret',ttl_s=60)

        coordinator.task_intents=controller;coordinator.advisor.decide.controller=controller

        await old.aclose()

        main._BASE_TOOLS=tuple(t for t in main._BASE_TOOLS if t.name in SCIENCE)

        assert {t.name for t in main._BASE_TOOLS}==set(SCIENCE)

        for rid in IDS:

            adapter=runtime.governor.catalog.get(rid).adapter

            for action in ('promote','demote'):

                fn=getattr(adapter,action)

                def observed(current,target,*,fn=fn,rid=rid,a=adapter,action=action):

                    before=a.pid;t=perf_counter()

                    try:return fn(current,target)

                    finally:

                        if before is None and a.pid is not None:generations[rid]+=1

                        emit('resource_transition',resource_id=rid,action=action,before_pid=before,

                             after_pid=a.pid,generation=generations[rid],duration_s=perf_counter()-t)

                setattr(adapter,action,observed)

        original_acquire=shared.acquire

        from contextlib import asynccontextmanager

        @asynccontextmanager

        async def observed_acquire(requests,**kwargs):

            emit('lease_requested',request_id=kwargs.get('request_id'),requests=requests,resources=snapshots())

            async with original_acquire(requests,**kwargs) as lease:

                emit('lease_enter',request_id=kwargs.get('request_id'),resources=snapshots())

                try:yield lease

                finally:emit('lease_body_end',request_id=kwargs.get('request_id'),resources=snapshots())

            emit('lease_returned',request_id=kwargs.get('request_id'),resources=snapshots())

        shared.acquire=observed_acquire

        original_bridge=runtime.bridge.ainvoke

        def validate(call):

            name=call['name'];request=call['args']['request'];phase=PHASE.get()

            if name==SCIENCE[0]:

                temp=float(request['temperature_c']);ids=[c['candidate_id'] for c in request['candidates']]

                allowed=({c['candidate_id'] for c in CANDIDATES} if phase=='A' else

                         {r['candidate_id'] for r in REFS} if phase=='B1' else set(priority))

                allowed_temps=([-10.] if phase=='A' else [-20.,-10.,0.,20.] if phase=='B1' else [25.])

                if not set(ids)<=allowed or temp not in allowed_temps:raise ValueError('本轮不重复已有预测或增加方案外输入')

                for c in request['candidates']:

                    amounts=c.get('component_mmol') or c.get('composition_inputs',{}).get('component_mmol') or request.get('composition_inputs',{}).get(c['candidate_id'],{}).get('component_mmol')

                    if not amounts or set(amounts)!=set(ALL[c['candidate_id']]['component_mmol']):raise ValueError('请直接使用给定完整component_mmol组成')

                    if any(not math.isclose(float(amounts[k]),float(v),rel_tol=1e-10,abs_tol=1e-9) for k,v in ALL[c['candidate_id']]['component_mmol'].items()):raise ValueError('禁止改变事前固定组成')

                return (name,temp,tuple(sorted(ids)))

            if name==SCIENCE[1]:

                if phase!='A' or {c['candidate_id'] for c in request['candidates']}!={c['candidate_id'] for c in CANDIDATES}:raise ValueError('排序只用于阶段一给定候选')

                if list(request['temperature_c'])!=[-20,-10,0,20] or float(request['min_conductivity_ms_cm'])!=2 or request.get('ensemble_budget',32)!=32:raise ValueError('保持固定四温度、2.0目标和32成员')

                for c in request['candidates']:

                    amounts=c.get('component_mmol') or c.get('composition_inputs',{}).get('component_mmol') or request.get('composition_inputs',{}).get(c['candidate_id'],{}).get('component_mmol')

                    if amounts is None or any(not math.isclose(float(amounts.get(k,float('nan'))),float(v),rel_tol=1e-10,abs_tol=1e-9) for k,v in ALL[c['candidate_id']]['component_mmol'].items()):raise ValueError('排序请使用给定完整component_mmol组成')

                return (name,'fixed-four-temperature-comparison')

            if name==SCIENCE[2]:

                label=request.get('formulation_id')

                if phase!='B2' or label not in SWEEPS:raise ValueError('分别保留六条扫描的给定formulation_id')

                expected=SWEEPS[label]['measurements']

                actual=sorted(request['measurements'],key=lambda x:x['temperature_c'])

                if len(actual)!=len(expected) or any(any(not math.isclose(float(a[k]),float(e[k]),rel_tol=1e-12,abs_tol=1e-12) for k in ('temperature_c','conductivity_ms_cm')) for a,e in zip(actual,expected)):raise ValueError('请完整保留该扫描10个原始测量点，不平均删点')

                return (name,label)

            raise ValueError('未授权工具')

        async def observed_bridge(node,state,config=None,**kwargs):

            message=next(m for m in reversed(state['messages']) if isinstance(m,AIMessage) and m.tool_calls)

            calls=message.tool_calls;new=[];reused=[];keys={}

            for call in calls:

                try:key=validate(call)

                except (ValueError,KeyError,TypeError) as e:

                    reused.append(ToolMessage(content=compact({'status':'invalid_request','error':str(e),'recoverable':True}),

                                              tool_call_id=call['id'],name=call['name']))

                    emit('input_rejected_before_science',name=call['name'],reason=str(e));continue

                keys[call['id']]=key

                if key in cached:

                    reused.append(ToolMessage(content=cached[key],tool_call_id=call['id'],name=call['name']))

                    emit('same_episode_result_reused',name=call['name'],key=key);continue

                if call['name']==SCIENCE[0]:

                    point_keys=[(i,key[1]) for i in key[2]]

                    if any(k in point_cache for k in point_keys):

                        if all(k in point_cache for k in point_keys):

                            reused.append(ToolMessage(content=compact([point_cache[k] for k in point_keys]),

                                tool_call_id=call['id'],name=call['name']))

                            emit('same_episode_points_reused',point_keys=point_keys)

                        else:

                            reused.append(ToolMessage(content=compact({'status':'invalid_request','error':

                                '该批次含已预测点，请只提交尚未得到结果的组成；已完成结果保留在先前工具返回中。'}),

                                tool_call_id=call['id'],name=call['name']))

                        continue

                if any(keys.get(c['id'])==key for c in new):

                    reused.append(ToolMessage(content=compact({'status':'invalid_request','error':'同批重复计算请求，请使用前一个调用结果。'}),

                                              tool_call_id=call['id'],name=call['name']))

                    continue

                new.append(call)

            if not new:return {'messages':reused}

            revised=dict(state);revised['messages']=list(state['messages'][:-1])+[message.model_copy(update={'tool_calls':new})]

            emit('science_batch_start',calls=new,resources=snapshots())

            response=await original_bridge(node,revised,config,**kwargs)

            returned={m.tool_call_id:m for m in response['messages'] if isinstance(m,ToolMessage)}

            for call in new:

                result=returned[call['id']]

                try:payload=json.loads(result.content)

                except (ValueError,TypeError):payload=result.content

                entry={'phase':PHASE.get(),'agent':agent_scope.get(),'call':call,

                       'payload':payload,'resources':snapshots(),'tool_status':result.status}

                science.append(entry)

                if result.status!='error' and not (isinstance(payload,dict) and payload.get('status') in ('invalid_request','error','admission_denied')):

                    if call['name']!=SCIENCE[1] or (isinstance(payload,dict) and len(payload.get('candidates',[]))==3):

                        cached[keys[call['id']]]=result.content

                    if call['name']==SCIENCE[0] and isinstance(payload,list):

                        for row in payload:

                            if row.get('model_predictions') and row['model_predictions'][-1].get('status')=='predicted':

                                point_cache[(row['candidate_id'],float(call['args']['request']['temperature_c']))]=row

                dump('science_results.json',science)

            emit('science_batch_end',call_ids=[c['id'] for c in new],resources=snapshots(),

                 advisor_events=list(coordinator.advisor.events),intent_events=list(controller.events))

            return {**response,'messages':list(response['messages'])+reused}

        runtime.bridge.ainvoke=observed_bridge

        graphs=[main.build_streaming_agent(),main.build_streaming_agent()]

        assert all(g.coordinator is coordinator for g in graphs)

        dump('execution_environment.json',{'model':model_config.AGENT_MODEL,'temperature':model_config.AGENT_TEMPERATURE,

             'host_after_import':asdict(windows_host_memory()),'tool_subset':[t.name for t in main._BASE_TOOLS],

             'source_fingerprints_match':True,'python':sys.executable})

        budgets={'A':BudgetTracker(max_iterations=4,max_wall_seconds=1800),

                 'B':BudgetTracker(max_iterations=8,max_wall_seconds=1800)}

        configs={label:{'configurable':{'thread_id':'integrated-core-'+('A' if label=='A' else 'B'),

                 'llm_stream':False,'llm_request_max_retries':0,'llm_step_timeout':90,'budget':budgets['A' if label=='A' else 'B']},

                 'metadata':{'agent_id':'A' if label=='A' else 'B'},'recursion_limit':70,'callbacks':[Recorder()]}

                 for label in ('A','B1','B2')}

        values={label:{'messages':[HumanMessage(content=prompt,id='core-request-'+label)],

                       'session_id':'core-'+('A' if label=='A' else 'B')}

                for label,prompt in [('A',PROMPT_A),('B1',PROMPT_B1),('B2',follow_prompt())]}

        request_ids={label:announce_received_request(queue,value,configs[label],source={'kind':'prospectively_submitted_chemical_request','phase':label})

                     for label,value in values.items()}

        async def invoke(label,graph):

            owner='A' if label=='A' else 'B';token=agent_scope.set(owner);phase=PHASE.set(label)

            emit('phase_start',request_id=request_ids[label],resources=snapshots(),task_state=controller.snapshot())

            try:

                result=await graph.ainvoke(values[label],configs[label])

                status='paused' if result.get('__interrupt__') else 'completed'

                item={'status':status,'messages':[m.model_dump(mode='json') for m in result.get('messages',[])],

                      'final_answer':next((m.content for m in reversed(result.get('messages',[])) if isinstance(m,AIMessage) and not m.tool_calls),None)}

                outcomes[label]=item;dump(label+'_result.json',item)

                if status!='completed':raise RuntimeError('阶段因额度/审批暂停，不重复运行')

                emit('phase_end',label=label,status=status,resources=snapshots())

                return item

            finally:PHASE.reset(phase);agent_scope.reset(token)

        async def episode():

            async with asyncio.TaskGroup() as group:

                group.create_task(invoke('A',graphs[0]))

                group.create_task(invoke('B1',graphs[1]))

            ranked=[x['payload'] for x in science if x['phase']=='A' and x['call']['name']==SCIENCE[1]

                    and isinstance(x['payload'],dict) and x['payload'].get('candidates')]

            assert len(ranked)==1,'阶段一须有且仅有一次有效排序'

            scored=ranked[0]['candidates'];best=max((r['joint_meets_constraint_probability'],r['worst_temperature_median_ms_cm']) for r in scored)

            priority.extend(r['candidate_id'] for r in scored if math.isclose(r['joint_meets_constraint_probability'],best[0],abs_tol=1e-12)

                            and math.isclose(r['worst_temperature_median_ms_cm'],best[1],abs_tol=1e-12))

            assert 1<=len(priority)<=3

            handoff={'priority_candidates':[ALL[i] for i in priority], 'ranking_statistics':scored,

                     'point_predictions_already_available':OLD_POINTS,

                     'scientific_tie_rule':'equal joint member frequency and worst-temperature median; preserve all ties',

                     'source':'actual stage-A tool responses in this episode; no resource outcome used for selection'}

            dump('handoff.json',handoff)

            new_prompt=follow_prompt(handoff)

            queue.revise(request_ids['B2'],new_prompt)

            values['B2']['messages']=[HumanMessage(content=new_prompt,id='core-request-B2')]

            (OUT/'B2_actual_request.txt').write_text(new_prompt,encoding='utf-8')

            assert queue.records[request_ids['A']].status==queue.records[request_ids['B1']].status=='completed'

            snap=controller.snapshot();assert len(snap['cards'])==1 and snap['cards'][0]['description']==new_prompt

            emit('stage_handoff',priority=priority,task_state=snap,resources=snapshots())

            await invoke('B2',graphs[1])

        await asyncio.wait_for(episode(),timeout=max(1,1800-(perf_counter()-started)))

        required_points={(c['candidate_id'],-10.) for c in CANDIDATES}

        required_points|={(r['candidate_id'],float(t)) for r in REFS for t in (-20,-10,0,20)}

        required_points|={(i,25.) for i in priority}

        fitted={x['payload'].get('formulation_id') for x in science if x['call']['name']==SCIENCE[2]

                and isinstance(x['payload'],dict) and x['payload'].get('status')=='fitted'}

        outcomes['scientific_coverage']={'point_predictions':len(point_cache),'required_point_predictions':len(required_points),

            'missing_points':sorted(required_points-set(point_cache)),'fitted_sweeps':sorted(fitted),

            'missing_sweeps':sorted(set(SWEEPS)-fitted),'raw_measurements':60,'old_points_reused':9}

        assert required_points<=set(point_cache) and set(SWEEPS)<=fitted,'化学分析覆盖不完整，不能按完成报告'

        outcomes['status']='completed'

        prior=json.loads((ROOT/'execution/result.json').read_text(encoding='utf-8'))

        outcomes['primary_requests_including_startup_attempt']=sum(primary.values())+sum(prior['primary_requests'].values())

        assert outcomes['primary_requests_including_startup_attempt']<=24

    except BaseException as error:

        outcomes.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())

        emit('execution_failed',error_type=type(error).__name__,error=str(error))

    finally:

        outcomes.update(primary_requests=dict(primary),priority_candidates=priority,elapsed_s=perf_counter()-started,

                        science_calls=len(science),resources_before_close=snapshots(),

                        source_hashes_unchanged=all(sha(p)==h for p,h in protocol['source_hashes'].items()),

                        manuscripts_unchanged=all(sha(p)==h for p,h in protocol['working_file_hashes'].items()))

        if runtime is not None:

            outcomes.update(admission=runtime.bridge.shared_runtime.snapshot(),

               advisor_events=list(coordinator.advisor.events),coordinator_events=list(coordinator.events),

               host_guard_events=list(runtime.governor.preparation_guard.events))

        if controller is not None:

            outcomes.update(jev_calls=controller.assessor.calls,jev_events=list(controller.assessor.events),

               intent_events=list(controller.events),final_task_state=controller.snapshot(),

               queue={k:asdict(v) for k,v in queue.records.items()})

        for graph in graphs:await main.aclose_agent(graph)

        if runtime is not None:await runtime.aclose()

        outcomes.update(resources_after_close=snapshots(True),host_after=asdict(windows_host_memory()))

        dump('result.json',outcomes)

        dump('progress.json',{'status':outcomes['status'],'primary_requests':dict(primary),

             'jev_calls':outcomes.get('jev_calls',0),'science_calls':len(science)})

    print(compact({'status':outcomes['status'],'primary_requests':dict(primary),

                  'jev_calls':outcomes.get('jev_calls',0),'science_calls':len(science)}))



if __name__=='__main__':

    if len(sys.argv)!=2 or sys.argv[1] not in ('--prepare','--execute'):raise SystemExit('使用 --prepare 或 --execute')

    if sys.argv[1]=='--prepare':prepare()

    else:

        with (OUT/'console.txt').open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):

            asyncio.run(execute())

