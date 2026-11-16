"""归档真实科学调用的配对接续；每分支至多一次真实Jev判断，不运行主模型。"""
from release_paths import release_path
import asyncio
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

BASE = Path(__file__).resolve().parent
REPO = Path(release_path('${ORION_SOURCE}'))
OUT = BASE.parents[1] / 'outputs' / sys.argv[1]
OUT.mkdir(exist_ok=False)
sys.path[:0] = [str(REPO.parent), str(REPO)]
os.environ.update({
    'PYTHONDONTWRITEBYTECODE':'1', 'ORION_ENV_FILE':os.environ.get('ORION_ENV_FILE',str(REPO/'.env')),
    'ORION_RESOURCE_POOL_ENABLED':'1', 'ORION_RESOURCE_ADMISSION_QUEUE':'1',
    'ORION_RESOURCE_COORDINATION':'1', 'ORION_RESOURCE_TASK_INTENT':'1',
    'ORION_RESOURCE_INTENT_JEV_MAX_CALLS':'0',
    'ORION_NA_CONDUCTIVITY_MODEL_PATH':release_path('${PROJECT_SOURCE}/reports/cwsr_inputs/cwsr_conductivity_rf_formal_v1.joblib'),
    'ORION_ROBUST_RANKING_BUNDLE_DIR':release_path('${PROJECT_SOURCE}/reports/cwsr_inputs/na_training_bundle_v1'),
    'ORION_RESOURCE_HOST_RAM_MB':'3072', 'ORION_RESOURCE_PROCESS_SLOTS':'2',
    'ORION_RESOURCE_AUTO_MEMORY':'0', 'ORION_RESOURCE_ADAPTIVE_RESERVE':'0',
    'ORION_RESOURCE_INTENT_PREFETCH':'0', 'ORION_RESOURCE_INTENT_RULE':'minimax_regret',
    'ORION_RESOURCE_HOST_GUARD':'1', 'ORION_AUTO_REFLECT':'0',
    'ORION_REFLECTION_INJECT':'0', 'ORION_LLM_CACHE':'0',
    'LANGCHAIN_TRACING_V2':'false', 'LANGSMITH_TRACING':'false',
    'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
})
os.environ.pop('ORION_RESOURCE_EVENTS_PATH', None)
os.chdir(OUT)
import env_utils
SECRETS = [v for k,v in os.environ.items() if any(s in k for s in ('KEY','TOKEN','PASSWORD')) and len(v)>12]
def safe(v):
    value=json.dumps(v,ensure_ascii=False,default=str)
    for secret in SECRETS: value=value.replace(secret,'[REDACTED]')
    return json.loads(value)
def dump(path,v):
    Path(path).write_text(json.dumps(safe(v),ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
from orion.core.default_resource_runtime import build_default_tool_resource_runtime
from orion.core.resource_tool_node import LeaseSafeToolNode
from orion.core.resource_task_graph import ResourceTaskGraph
from orion.capvm.jev_task_intent import JevTaskIntentAssessor
from orion.capvm.host_memory import windows_host_memory
from tools.electrolyte_design.conductivity_tool import predict_formulation_conductivity
from tools.electrolyte_design.temperature_fit_tool import fit_conductivity_temperature_law
from tools.electrolyte_design.robust_ranking_tool import rank_formulations_under_uncertainty
import tools.file_operate as file_ops
from forward_queue import SubmittedRequests, QueueAwareIntentController, announce_received_request

IDS=('process:na-conductivity-rf','process:robust-ranking-worker','process:transport-fit-kernel')
class NoBoard:
    def bind(self,callback): return nullcontext()
    def snapshot(self,ids): raise AssertionError('本验证不制造任务卡')
def batches(source):
    messages=json.loads(Path(source).read_text(encoding='utf-8'))['messages']
    out=[]
    for m in messages:
        if not m.get('tool_calls'): continue
        calls=m['tool_calls']
        result={r['tool_call_id']:r for r in messages if r['type']=='tool' and r['tool_call_id'] in {c['id'] for c in calls}}
        assert len(result)==len(calls)
        out.append({'calls':calls,'reference_results':result})
    return out
def payload(message):
    v=message.content
    if isinstance(v,str): return json.loads(v)
    return v
def numeric_leaves(v,path=''):
    # 数值科学输出逐项比对；排除运行身份、耗时和存储路径等元数据。
    skip={'execution_id','record_id','record_path','timestamp','created_at','elapsed_s','duration_s','wall_s','pid'}
    if isinstance(v,dict):
        return {p:x for k,item in v.items() if k not in skip for p,x in numeric_leaves(item,path+'/'+k).items()}
    if isinstance(v,list):
        return {p:x for i,item in enumerate(v) for p,x in numeric_leaves(item,path+'/'+str(i)).items()}
    if isinstance(v,(int,float)) and not isinstance(v,bool): return {path:v}
    return {}

PROTOCOL=__import__('release_paths').resolve_json(json.loads((BASE/'前瞻任务感知修订方案.json').read_text(encoding='utf-8')))
assert sha(PROTOCOL['previous_real_activity']['source'])==PROTOCOL['previous_real_activity']['sha256']
for r in PROTOCOL['incoming_requests']: assert sha(r['source'])==r['sha256']
frozen_files=[Path(__file__),BASE/'forward_queue.py',BASE/'前瞻任务感知修订方案.json',
 REPO/'core/default_resource_runtime.py',REPO/'core/resource_task_graph.py',
 REPO/'core/langgraph_residency.py',REPO/'capvm/task_intent.py',REPO/'capvm/jev_task_intent.py',
 REPO/'capvm/demand_coordinator.py',REPO/'capvm/semantic_plan.py',
 Path(os.environ['ORION_NA_CONDUCTIVITY_MODEL_PATH'])]
frozen_files+=list(Path(os.environ['ORION_ROBUST_RANKING_BUNDLE_DIR']).glob('*.csv'))
HASHES={str(p):sha(p) for p in frozen_files}
dump(OUT/'protocol.json',{'original_plan':PROTOCOL,'source_hashes':HASHES,
 'execution_kind':'paired archived real scientific call continuation; no new autonomous agent planning',
 'maximum_actual_jev_calls':2,'main_model_calls':0,'no_response_tuning':True,
 'intent_trigger':'explicit actual multi-option decision; bypass economic request gate in isolated experiment only',
 'jev_timeout_s':20,'intent_ttl_s':60,'selection_rule':'minimax_regret',
 'scientific_call_batches':{r['id']:batches(r['source']) for r in PROTOCOL['incoming_requests']},
 'settings':{k:v for k,v in os.environ.items() if k.startswith('ORION_RESOURCE_')},
 'host_before':asdict(windows_host_memory())})

async def branch(spec):
    folder=OUT/spec['id'];folder.mkdir()
    for key,name in [('ORION_RESOURCE_PROFILE_PATH','profiles.db'),('ORION_TASKS_DIR','tasks'),
      ('LTM_DB_PATH','facts.db'),('LTM_VECTORS_DB_PATH','vectors.db'),
      ('ORION_REFLECTION_STORE_PATH','reflections.db'),('ORION_CAPABILITY_STATE_ROOT','capability-state')]:
        os.environ[key]=str(folder/name)
    os.environ['DEEPAGENTS_ROOT']=str(folder)
    file_ops.WORKDIR=folder
    runtime=build_default_tool_resource_runtime()
    governor=runtime.governor;coordinator=runtime.bridge.shared_runtime.demand_coordinator
    queue=SubmittedRequests();events=[];start=perf_counter();results=[]
    def emit(event,**values):
        row=safe({'event':event,'elapsed_s':perf_counter()-start,**values})
        events.append(row)
        with (folder/'events.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    def snapshot(closed=False):
        return [{'resource_id':r,'pid':governor.catalog.get(r).adapter.pid,
          'level':None if closed else governor.state(r).current_level.name,
          'active_count':None if closed else governor.state(r).active_count,
          'last_used_tick':None if closed else governor.state(r).last_used_tick} for r in IDS]
    class ObservedController(QueueAwareIntentController):
        async def prepare(self,state,options,costs):
            if len(options)>=2:
                resources=sorted({r for o in options for r in o['resource_ids']})
                before=self.snapshot();resolved=self._requirements(before,resources)
                rows={r['id']:r for r in state['resources']}
                lru=min(options,key=lambda o:tuple(sorted((rows[r]['tick'],r) for r in o['resource_ids'])))['id']
                cost=min(options,key=lambda o:(sum(costs[r] for r in o['resource_ids']),o['id']))['id']
                emit('actual_decision_before',task_snapshot=before,state=state,options=options,
                     measured_costs=dict(costs),lru_shadow=lru,cost_shadow=cost,resources=snapshot())
                if resolved is not None and resolved[1] and self.assessor.calls<1:
                    descriptions={r:{k:v for k,v in self.resource_descriptions[r].items() if k!='literal_terms'} for r in resources}
                    emit('jev_request_binding',cards=resolved[1],descriptions=descriptions,version=before['version'])
                    answer=await self.assessor.assess(resolved[1],descriptions)
                    emit('jev_real_response',answer=answer,raw_events=list(self.assessor.events),
                         same_version=self.snapshot().get('version')==before['version'])
            await super().prepare(state,options,costs)
            if len(options)>=2: emit('actual_decision_prepared',intent_events=list(self.events))
    old=coordinator.task_intents
    assessor=JevTaskIntentAssessor(os.environ['TYPESAFE_API_KEY'],max_calls=1,timeout_s=20,ttl_s=60)
    controller=ObservedController(NoBoard(),old.tool_resources,old.resource_descriptions,
        assessor=assessor,submitted_requests=queue,selection_rule='minimax_regret',ttl_s=60)
    coordinator.task_intents=controller
    coordinator.advisor.decide.controller=controller
    await old.aclose()
    for rid in IDS:
        adapter=governor.catalog.get(rid).adapter
        for action in ('promote','demote'):
            original=getattr(adapter,action)
            def observed(current,target,*,fn=original,rid=rid,adapter=adapter,action=action):
                pid=adapter.pid;t=perf_counter()
                try: return fn(current,target)
                finally: emit('resource_transition',resource_id=rid,action=action,before_pid=pid,
                              after_pid=adapter.pid,duration_s=perf_counter()-t)
            setattr(adapter,action,observed)
    node=LeaseSafeToolNode([predict_formulation_conductivity,fit_conductivity_temperature_law,
                           rank_formulations_under_uncertainty],handle_tool_errors=False)
    previous=json.loads(Path(PROTOCOL['previous_real_activity']['source']).read_text(encoding='utf-8'))
    prior_goal=next(m['content'] for m in previous['messages'] if m['type']=='human')
    prior_input={'messages':[HumanMessage(content=prior_goal,id='archived-previous')]}
    next_input={'messages':[HumanMessage(content=spec['goal'],id='archived-incoming-'+spec['id'])]}
    prior_config={'configurable':{'thread_id':'continuation-'+spec['id']+'-previous'}}
    next_config={'configurable':{'thread_id':'continuation-'+spec['id']+'-incoming'}}
    for value,config,source in [(prior_input,prior_config,PROTOCOL['previous_real_activity']),
                               (next_input,next_config,spec)]:
        announce_received_request(queue,value,config,source={'kind':'received_archived_request',
            'file':source['source'],'sha256':source['sha256']})
    scientific_batches=batches(spec['source'])
    async def invoke_batch(batch,config,label):
        calls=deepcopy(batch['calls']);emit('scientific_batch_start',label=label,calls=calls,resources=snapshot())
        response=await runtime.bridge.ainvoke(node,{'messages':[AIMessage(content='',tool_calls=calls)]},
                                              config,request_id=spec['id']+'-'+label)
        serialized=[m.model_dump(mode='json') for m in response['messages']]
        dump(folder/(label+'-raw-response.json'),serialized)
        for m in response['messages']:
            if m.status=='error': raise RuntimeError('Scientific tool returned an error: '+m.content)
            value=payload(m)
            if isinstance(value,dict) and value.get('status') in ('invalid_request','error','admission_denied'):
                raise RuntimeError('Scientific tool did not execute successfully')
        item={'label':label,'calls':calls,'messages':serialized}
        if batch.get('reference_results'):
            comparisons=[]
            for m in response['messages']:
                oldvalue=json.loads(batch['reference_results'][m.tool_call_id]['content'])
                current=numeric_leaves(payload(m));prior=numeric_leaves(oldvalue)
                differences={k:[prior.get(k),current.get(k)] for k in set(prior)|set(current)
                    if k not in current or k not in prior or abs(current[k]-prior[k])>1e-9*max(1,abs(prior[k]))}
                comparisons.append({'tool_call_id':m.tool_call_id,'numeric_values':len(current),
                                    'differences':differences})
            item['scientific_numeric_comparison']=comparisons
        results.append(item);dump(folder/'scientific_results.json',results)
        emit('scientific_batch_end',label=label,resources=snapshot(),
             comparisons=item.get('scientific_numeric_comparison'))
        return response
    async def restore(state,config: RunnableConfig):
        for i,row in enumerate(PROTOCOL['previous_real_activity']['restore_calls']):
            await invoke_batch({'calls':[row['call']]},config,'restore-'+str(i))
        emit('prior_finishing_next_pending',queue={k:asdict(v) for k,v in queue.records.items()},
             task_snapshot=controller.snapshot(),resources=snapshot())
        return {'messages':[AIMessage(content='Archived prior scientific calls completed.')]}
    async def follow(state,config: RunnableConfig):
        emit('incoming_active',task_snapshot=controller.snapshot(),resources=snapshot())
        for i,batch in enumerate(scientific_batches):
            if i>0:
                # 只用本次真实拟合结果确认原任务条件，不能从归档结果替代计算。
                fits=[json.loads(m['content']) for r in results if r['label']=='incoming-0' for m in r['messages']]
                metrics=[]
                for fit in fits:
                    selected=fit['preferred_by_aicc']
                    model=next(m for m in fit['models'] if m['model']==selected)
                    metrics.append(model['loo_rmse_log10'])
                assert len(metrics)==2 and all(x<.1 for x in metrics), 'Actual fit condition not met; do not force subsequent calls'
                emit('actual_fit_condition_passed',loo_rmse_log10=metrics,threshold=.1)
            await invoke_batch(batch,config,'incoming-'+str(i))
        return {'messages':[AIMessage(content='Archived continuation scientific calls completed; no new report generated.')]}
    def graph(work):
        builder=StateGraph(MessagesState);builder.add_node('work',work)
        builder.add_edge(START,'work');builder.add_edge('work',END)
        return ResourceTaskGraph(builder.compile(),coordinator,'archived-scientific-continuation')
    outcome={'id':spec['id'],'status':'started'}
    try:
        await graph(restore).ainvoke(prior_input,prior_config)
        emit('between_tasks',task_snapshot=controller.snapshot(),resources=snapshot(),
             queue={k:asdict(v) for k,v in queue.records.items()})
        assert len(controller.snapshot()['cards'])==1
        assert controller.snapshot()['cards'][0]['description']==spec['goal']
        assert controller.snapshot()['cards'][0]['status']=='pending'
        await graph(follow).ainvoke(next_input,next_config)
        outcome['status']='completed'
    except Exception as error:
        outcome.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        outcome.update(jev_calls=assessor.calls,jev_events=list(assessor.events),
          intent_events=list(controller.events),advisor_events=list(coordinator.advisor.events),
          coordinator_events=list(coordinator.events),resources_before_close=snapshot(),
          host_guard_events=list(governor.preparation_guard.events),
          final_task_snapshot=controller.snapshot(),queue={k:asdict(v) for k,v in queue.records.items()})
        await runtime.aclose()
        outcome.update(resources_after_close=snapshot(True),jev_client_closed=not assessor.has_open_client,
                       scientific_batches=len(results))
        dump(folder/'result.json',outcome)
    return outcome

async def run():
    outcomes=[]
    for spec in PROTOCOL['incoming_requests']:
        print('START',spec['id'],flush=True)
        with (OUT/(spec['id']+'-runtime.log')).open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
            result=await branch(spec)
        outcomes.append(result)
        print('END',spec['id'],result['status'],'Jev',result['jev_calls'],flush=True)
        if result['status']!='completed':
            print('Preserved failure:',result.get('error'),flush=True)
            break
    dump(OUT/'result.json',{'branches':outcomes,'jev_calls':sum(r['jev_calls'] for r in outcomes),
        'source_hashes_unchanged':all(sha(p)==v for p,v in HASHES.items()),
        'host_after':asdict(windows_host_memory()),'main_model_calls':0})

if __name__=='__main__': asyncio.run(run())
