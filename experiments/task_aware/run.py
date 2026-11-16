"""单个隔离进程运行真实主图，任务卡来自真实主LLM；无卡时仅使用真实运行请求，不预造计划。"""
from release_paths import release_path

import asyncio

from collections import Counter

from contextlib import redirect_stdout,redirect_stderr

from copy import deepcopy

import hashlib

import json

import os

from pathlib import Path

import sys

from time import perf_counter

import traceback



base=Path(__file__).resolve().parents[2]/'outputs/task_aware'

plan=__import__('release_paths').resolve_json(json.loads((base/'protocol.json').read_text(encoding='utf-8')))

spec=next(r for r in plan['order'] if r['id']==sys.argv[1])

out=base/spec['id'];out.mkdir(parents=True,exist_ok=False)

repo=Path(__file__).resolve().parents[2]/'versions/task_aware/orion'

sys.path[:0]=[str(repo.parent),str(repo),str(Path(__file__).resolve().parents[2]/'src')]

os.environ['PYTHONDONTWRITEBYTECODE']='1'

os.environ.setdefault('ORION_ENV_FILE',str(repo/'.env'))

from env_utils import DEEPSEEK_API_KEY

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'

for key in ('ORION_RESOURCE_POOL_ENABLED','ORION_RESOURCE_ADMISSION_QUEUE',

            'ORION_RESOURCE_COORDINATION','ORION_RESOURCE_TASK_INTENT','ORION_AUTO_APPROVE'):

    os.environ[key]='1'

for key in ('ORION_LLM_CACHE','ORION_AUTO_REFLECT','ORION_REFLECTION_INJECT','ORION_LLM_STREAM',

            'ORION_RESOURCE_AUTO_MEMORY','ORION_RESOURCE_ADAPTIVE_RESERVE'):

    os.environ[key]='0'

os.environ['ORION_RESOURCE_HOST_RAM_MB']='3072'

os.environ['ORION_RESOURCE_PROCESS_SLOTS']=str(spec['process_slots'])

os.environ['ORION_RESOURCE_INTENT_PREFETCH']='1' if spec['arm']=='async' else '0'

os.environ['ORION_RESOURCE_INTENT_RULE']=plan['selection_rule']

os.environ['ORION_RESOURCE_INTENT_JEV_MAX_CALLS']='0'

os.environ['ORION_RESOURCE_PROFILE_PATH']=str(out/'profiles.db')

os.environ['ORION_NA_CONDUCTIVITY_MODEL_PATH']=release_path('${PROJECT_SOURCE}/reports/cwsr_inputs/cwsr_conductivity_rf_formal_v1.joblib')

os.environ['ORION_LLM_REQUEST_MAX_RETRIES']='0'

os.environ['ORION_LLM_STEP_TIMEOUT']='45'

for key,path in {'ORION_TASKS_DIR':'tasks','LTM_DB_PATH':'facts.db','LTM_VECTORS_DB_PATH':'vectors.db',

    'ORION_REFLECTION_STORE_PATH':'reflections.db','ORION_CAPABILITY_STATE_ROOT':'capability-state'}.items():

    os.environ[key]=str(out/path)

os.environ['DEEPAGENTS_ROOT']=str(out)

os.environ.pop('ORION_RESOURCE_EVENTS_PATH',None)

os.environ.pop('ORION_DESIGN_REASONING_MODE',None)

os.chdir(out)

secrets=[v for k,v in os.environ.items() if any(t in k for t in ('KEY','TOKEN','PASSWORD')) and len(v)>12]

def safe(value):

    text=json.dumps(value,ensure_ascii=False,default=str)

    for secret in secrets:text=text.replace(secret,'[REDACTED]')

    return json.loads(text)

def dump(name,value):

    (out/name).write_text(json.dumps(safe(value),ensure_ascii=False,indent=2),encoding='utf-8')

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()



from langchain_core.callbacks import BaseCallbackHandler

from langchain_core.messages import AIMessage,HumanMessage

from pydantic import ConfigDict,create_model

from orion.capvm.jev_task_intent import JevTaskIntentAssessor

from orion.core.tasks import get_board

from orion.core.budget import BudgetTracker

from tools.electrolyte_design import prediction_records

from orion.capvm import ResidencyLevel



events=[];created=perf_counter();arrival=None;runtime=None

class Recorder(BaseCallbackHandler):

    def add(self,event,**data):

        row=safe({'event':event,'elapsed_s':perf_counter()-created,**data});events.append(row)

        with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')

    def on_chat_model_start(self,serialized,messages,**kwargs):

        visible=[[{'type':m.type,'content':m.content} for m in group] for group in messages]

        self.add('llm_start',run_id=str(kwargs.get('run_id')),visible_input_hash=digest(visible),

            messages=[m.model_dump(mode='json') for m in messages[0]])

    def on_llm_end(self,response,**kwargs):

        self.add('llm_end',run_id=str(kwargs.get('run_id')),llm_output=response.llm_output)

    def on_llm_error(self,error,**kwargs):self.add('llm_error',kind=type(error).__name__)

    def on_tool_start(self,serialized,input_str,**kwargs):

        self.add('tool_start',name=serialized.get('name'),input=input_str,run_id=str(kwargs.get('run_id')))

    def on_tool_end(self,output,**kwargs):

        self.add('tool_end',run_id=str(kwargs.get('run_id')),output=output.model_dump(mode='json') if hasattr(output,'model_dump') else output,

            task_cards=[c.to_dict() for c in get_board().list()],

            pids={r:runtime.governor.catalog.get(r).adapter.pid for r in

                ('process:na-conductivity-rf','process:robust-ranking-worker','process:transport-fit-kernel')} if runtime else {})

    def on_tool_error(self,error,**kwargs):self.add('tool_error',kind=type(error).__name__)

tracer=Recorder()



prior=[]

for item in plan['latency_calibration']:

    path=Path(item['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']

    prior.append(item['elapsed_s'])

class ObservedLatencyJev(JevTaskIntentAssessor):

    @property

    def estimated_overhead_s(self):

        current=[e['elapsed_s'] for e in self.events if 'elapsed_s' in e]

        return max((prior+current)[-8:])

    async def assess(self,cards,resources):

        if self.calls>=self.max_calls:return None

        journal=base/'network-calls.jsonl'

        used=len(journal.read_text(encoding='utf-8').splitlines()) if journal.exists() else 0

        if used>=plan['max_live_jev_calls_total']:return None

        with journal.open('a',encoding='utf-8') as f:f.write(json.dumps({'attempt':used+1,'run':spec['id']})+'\n')

        tracer.add('jev_start',cards=cards,resources=resources,estimated_overhead_s=self.estimated_overhead_s)

        result=await super().assess(cards,resources)

        tracer.add('jev_end',usable=result is not None,events=list(self.events))

        return result





def semantic_tool(tool):

    return tool





async def run():

    global runtime,arrival

    assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in plan['source_hashes'].items())

    import orion.main as main

    assert get_board().list()==[], '禁止预建任务卡或预填意图'

    main.SYSTEM_PROMPT=main.SYSTEM_PROMPT.replace(str(out),'[本次隔离工作区]')

    main._BASE_TOOLS=tuple(semantic_tool(t) for t in main._BASE_TOOLS if t.name in plan['allowed_tools'])

    graph=main.build_streaming_agent()

    runtime=main._default_resource_runtime;shared=runtime.bridge.shared_runtime;c=shared.demand_coordinator

    controller=c.task_intents

    if spec['arm']=='lru':

        c.advisor.active=False

    elif spec['arm']=='cost':

        c.advisor.decide=c.advisor.decide.fallback

        for value in controller.resource_descriptions.values():value.pop('literal_terms',None)

    elif spec['arm'] in ('sync','async'):

        for value in controller.resource_descriptions.values():value.pop('literal_terms',None)

        controller.assessor=ObservedLatencyJev(os.environ['TYPESAFE_API_KEY'],max_calls=plan['max_live_jev_calls_per_run'],timeout_s=10,ttl_s=300)

    adapters={r:runtime.governor.catalog.get(r).adapter for r in

        ('process:na-conductivity-rf','process:robust-ranking-worker','process:transport-fit-kernel')}

    lifecycle=[];starts={r:0 for r in adapters}

    for rid,adapter in adapters.items():

        promote=adapter.promote;demote=adapter.demote

        def observed_promote(current,target,*,rid=rid,adapter=adapter,method=promote):

            previous=adapter.pid;t=perf_counter();method(current,target)

            if adapter.pid and adapter.pid!=previous:starts[rid]+=1

            row={'resource':rid,'action':'promote','before_pid':previous,'after_pid':adapter.pid,'seconds':perf_counter()-t}

            lifecycle.append(row);tracer.add('resource_transition',**row)

        def observed_demote(current,target,*,rid=rid,adapter=adapter,method=demote):

            previous=adapter.pid;t=perf_counter();method(current,target)

            row={'resource':rid,'action':'demote','before_pid':previous,'after_pid':adapter.pid,'seconds':perf_counter()-t}

            lifecycle.append(row);tracer.add('resource_transition',**row)

        adapter.promote=observed_promote;adapter.demote=observed_demote

    original_path=prediction_records.safe_path;record_root=(out/'prediction-records').resolve();record_root.mkdir()

    prediction_records.RECORD_DIR=str(record_root)

    def scoped_path(value):

        path=Path(value).resolve()

        return path if path.is_relative_to(record_root) else original_path(value)

    prediction_records.safe_path=scoped_path

    result={'spec':spec,'status':'failed','kind':plan['scope'],'actual_tools':[t.name for t in main._BASE_TOOLS],

        'tool_schema_hash':digest([t.args_schema.model_json_schema() for t in main._BASE_TOOLS]),

        'semantic_interface_descriptions':{t.name:t.description for t in main._BASE_TOOLS if t.name.startswith('board_')},

        'no_prefilled_cards':True,'source_hashes':plan['source_hashes']}

    budget=None;config=None

    try:

        for rid in plan['warm_inventory']:

            async with shared.acquire([(rid,ResidencyLevel.BOUND)],reconcile_measured=True):pass

        result['warmup_and_import_s']=perf_counter()-created

        result['initial_pids']={r:a.pid for r,a in adapters.items()}

        result['initial_costs']=dict(c.profiler.costs)

        budget=BudgetTracker(max_iterations=plan['max_primary_requests_per_run'],max_wall_seconds=plan['max_run_seconds'])

        config={'configurable':{'thread_id':spec['id'],'llm_stream':False,'llm_request_max_retries':0,

            'llm_step_timeout':45,'budget':budget},'recursion_limit':48,'callbacks':[tracer]}

        arrival=perf_counter();tracer.add('task_arrived')

        state=await asyncio.wait_for(graph.ainvoke({'messages':[HumanMessage(plan['prompts'][spec['case']])],

            'session_id':spec['id']},config),plan['max_run_seconds'])

        result['task_wall_s']=perf_counter()-arrival

        result['status']='budget_paused' if state.get('__interrupt__') else 'completed'

    except BaseException as error:

        result['error']={'kind':type(error).__name__,'detail':str(error),'traceback':traceback.format_exc()}

        if arrival is not None:result['task_wall_s']=perf_counter()-arrival

    finally:

        snapshot=await graph.aget_state(config) if config else None

        state=snapshot.values if snapshot else {}

        messages=state.get('messages',[])

        result.update(primary_requests=budget.iterations if budget else 0,primary_tokens=budget.tokens_used if budget else 0,

            task_cards=[card.to_dict() for card in get_board().list()],

            messages=[m.model_dump(mode='json') for m in messages],

            actual_tool_calls=[tc for m in messages if isinstance(m,AIMessage) for tc in m.tool_calls],

            checkpoint_bound_ids=state.get('runtime_board_task_ids',[]),

            checkpoint_next=list(snapshot.next) if snapshot else [],

            final_answer=next((m.content for m in reversed(messages) if isinstance(m,AIMessage) and not m.tool_calls),None),

            starts=dict(starts),lifecycle=list(lifecycle),intent_events=list(controller.events),

            advisor_events=list(c.advisor.events),coordinator_events=list(c.events),

            jev_calls=controller.assessor.calls if controller.assessor else 0,

            jev_events=list(controller.assessor.events) if controller.assessor else [],

            runtime_before_close=shared.snapshot())

        dump('result.json',result)

        await main.aclose_agent(graph);await runtime.aclose()

        result.update(wall_including_import_setup_close_s=perf_counter()-created,

            lifecycle=list(lifecycle),runtime_after_close=shared.snapshot(),adapters_closed=all(a.pid is None for a in adapters.values()),

            source_hashes_unchanged=all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in plan['source_hashes'].items()))

        dump('result.json',result)

        print(json.dumps({'run':spec['id'],'status':result['status'],'primary_requests':result['primary_requests'],

            'jev_calls':result['jev_calls'],'task_wall_s':result.get('task_wall_s'),

            'starts':starts,'tool_counts':dict(Counter(tc['name'] for tc in result['actual_tool_calls'])),

            'error':result.get('error',{}).get('detail')},ensure_ascii=False),file=sys.__stdout__,flush=True)





with (out/'console.txt').open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):

    try:asyncio.run(run())

    finally:

        log.flush();text=(out/'console.txt').read_text(encoding='utf-8')

        for secret in secrets:text=text.replace(secret,'[REDACTED]')

        (out/'console.txt').write_text(text,encoding='utf-8')

