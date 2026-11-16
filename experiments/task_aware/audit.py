"""核对冻结主图矩阵：原始输入、时间分解、实际进程动作和全部运行。"""
from release_paths import release_path
import ast
from collections import Counter, defaultdict
import csv
import ctypes
from ctypes import wintypes
import hashlib
import html
import json
import math
from pathlib import Path
import sqlite3
import statistics
import subprocess

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'async-comparison-v1'
REPO=Path(release_path('${ORION_SOURCE}'))
RF='process:na-conductivity-rf'
RANK='process:robust-ranking-worker'
FIT='process:transport-fit-kernel'
FT='fit_conductivity_temperature_law'
PT='predict_formulation_conductivity'
RT='rank_formulations_under_uncertainty'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def near(a,b,tol=1e-7):assert math.isclose(a,b,rel_tol=tol,abs_tol=tol),(a,b)
def clean(v):
    if isinstance(v,dict):
        return {k:clean(x) for k,x in v.items() if k not in {
            'result_binding','execution_id','tool_run_id','record_id','record_path',
            'thread_id','formulation_id','provenance'}}
    if isinstance(v,list):return [clean(x) for x in v]
    return v

plan=read(BASE/'protocol.json')
assert all((BASE/s['id']/'result.json').exists() for s in plan['order']), '矩阵尚未结束'
assert all(sha(REPO/p)==h==sha(BASE/'frozen-sources'/p) for p,h in plan['source_hashes'].items())
results=[];rows=[];checks=[];scientific=defaultdict(set);inputs=defaultdict(set);schemas=set()
by_label={c['label']:c for c in plan['cases']}
data_signatures={canonical(c['measurements']):c['label'] for c in plan['cases']}
for spec in plan['order']:
    folder=BASE/spec['id'];r=read(folder/'result.json');results.append(r)
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    status=r['status']=='completed' and not r.get('error')
    assert r['spec']==spec and r['source_hashes_unchanged'] and r['no_prefilled_cards']
    assert r['adapters_closed'] and r['runtime_after_close']['closed']
    assert not r['runtime_after_close']['active'] and not r['runtime_after_close']['waiting']
    start={e['run_id']:e for e in events if e['event']=='llm_start'}
    end={e['run_id']:e for e in events if e['event']=='llm_end'}
    assert len(start)==r['primary_requests']
    spans=sorted((v['elapsed_s'],end[k]['elapsed_s']) for k,v in start.items() if k in end)
    memory_injections=sum(isinstance(m.get('content'),str) and m['content'].startswith('长期记忆：\n')
        for event in start.values() for m in event['messages'])
    assert memory_injections==0, '出现长期记忆上下文，需要更新证据说明'
    assert all(e<=ns for (_,e),(ns,_) in zip(spans,spans[1:]))
    llm=sum(e-s for s,e in spans)
    inputs[spec['case']].add(next(e['visible_input_hash'] for e in events if e['event']=='llm_start'))
    schemas.add(r['tool_schema_hash'])
    call_by_id={c['id']:c for c in r['actual_tool_calls']}
    observed=[];fit_labels=[];point_labels=[];ranking_count=0
    for m in r['messages']:
        if m['type']!='tool':continue
        name=m['name'];call=call_by_id[m['tool_call_id']]
        if name not in (FT,PT,RT):continue
        content=json.loads(m['content']) if isinstance(m['content'],str) else m['content']
        request=call['args']['request']
        if name==FT:
            sig=canonical(request['measurements']);assert sig in data_signatures, '测量被改变'
            label=data_signatures[sig];fit_labels.append(label)
            assert content['status']=='fitted' and content['measurement_count']==10
            best=next(x for x in content['models'] if x['model']==content['preferred_by_aicc'])
            assert best['loo_rmse_log10']<.1
            observed.append((name,label,clean(content)))
        elif name==PT:
            assert request['temperature_c']==20
            for candidate in request['candidates']:
                label=candidate['candidate_id'];assert label in by_label
                expected=by_label[label]['prediction_request']
                assert candidate==expected['candidates'][0]
                assert request['composition_inputs'][label]==expected['composition_inputs'][label]
                point_labels.append(label)
            assert isinstance(content,list) and all(
                p['status']=='predicted' for candidate in content for p in candidate['model_predictions'])
            observed.extend((name,c['candidate_id'],clean(c)) for c in content)
        elif name==RT:
            assert spec['case']=='ranking'
            assert sorted(request['temperature_c'])==[0,20,40]
            assert request['min_conductivity_ms_cm']==3.0
            assert {c['candidate_id'] for c in request['candidates']}==set(by_label)
            for c in request['candidates']:
                label=c['candidate_id'];expected=by_label[label]['prediction_request']
                assert all(c[k]==v for k,v in expected['candidates'][0].items())
                assert not c.get('additives')
                composition=(request.get('composition_inputs') or {}).get(label,c.get('composition_inputs'))
                assert composition==expected['composition_inputs'][label]
            ranking_count+=1
            observed.append((name,'all',clean(content)))
    if status:
        assert sorted(fit_labels)==sorted(by_label)
        assert (sorted(point_labels)==sorted(by_label) and not ranking_count) if spec['case']=='point' else (ranking_count==1 and not point_labels)
        scientific[spec['case']].add(canonical(sorted(observed,key=lambda x:(x[0],x[1]))))
    arrival=next(e['elapsed_s'] for e in events if e['event']=='task_arrived')
    life=[e for e in events if e['event']=='resource_transition' and e['elapsed_s']>=arrival]
    load=sum(e['seconds'] for e in life if e['action']=='promote' and e['before_pid']!=e['after_pid'])
    tool_start={e['run_id']:e['elapsed_s'] for e in events if e['event']=='tool_start'}
    tool_end={e['run_id']:e['elapsed_s'] for e in events if e['event']=='tool_end'}
    # 并行工具在边界内不相加；合并区间计量工具实际占用的墙钟。
    merged=[]
    for s,e in sorted((v,tool_end[k]) for k,v in tool_start.items() if k in tool_end):
        if merged and s<=merged[-1][1]:merged[-1][1]=max(e,merged[-1][1])
        else:merged.append([s,e])
    tool_wall=sum(e-s for s,e in merged)
    jev_starts=[e['elapsed_s'] for e in events if e['event']=='jev_start']
    jev_ends=[e['elapsed_s'] for e in events if e['event']=='jev_end']
    assert len(jev_starts)==len(jev_ends)==r['jev_calls']
    jev_spans=list(zip(jev_starts,jev_ends))
    # 与主模型使用同一perf_counter，避免SDK的monotonic时钟精度制造虚假的毫秒级重叠。
    network=sum(e-s for s,e in jev_spans)
    overlap=sum(max(0,min(je,le)-max(js,ls)) for js,je in jev_spans for ls,le in spans)
    prefetch=any(e.get('event')=='intent_prefetch_started' for e in r['intent_events'])
    if prefetch and r['jev_calls']:
        assert any(e.get('event')=='intent_assessed' and e.get('source')=='semantic_cache'
                   for e in r['intent_events'])
        assert all(je<spans[0][0] for _,je in jev_spans), '本次预取完成位置有变化，需单独分析'
        admission_wait=sum(e['residual_wait_s'] for e in r['intent_events'] if e.get('event')=='intent_prefetch_joined')
    else:
        admission_wait=network
    dbs=list(folder.rglob('sessions.db'));assert len(dbs)==1
    with sqlite3.connect(f'file:{dbs[0].as_posix()}?mode=ro',uri=True) as db:
        count=db.execute('select count(*) from checkpoints where thread_id=?',(spec['id'],)).fetchone()[0]
    assert count>0
    applied=[e for e in r['advisor_events'] if e.get('status')=='applied']
    changed=[e for e in r['intent_events'] if e.get('event')=='intent_plan_selected' and e.get('changed')]
    row={'run':spec['id'],'case':spec['case'],'arm':spec['arm'],'repeat':spec['repeat'],
         'process_slots':spec['process_slots'],'completed':bool(status),'task_wall_s':r.get('task_wall_s'),
         'llm_wait_s':llm,'non_llm_wall_s':r.get('task_wall_s',0)-llm,'tool_wall_s':tool_wall,
         'primary_calls':r['primary_requests'],'primary_tokens':r['primary_tokens'],
         'jev_calls':r['jev_calls'],'jev_network_s':network,'jev_llm_overlap_s':overlap,
         'jev_admission_wait_s':admission_wait,
         'jev_outside_llm_s':max(0,network-overlap),'post_arrival_load_s':load,
         'rf_starts_total':r['starts'][RF],'ranking_starts_total':r['starts'][RANK],
         'fit_starts_total':r['starts'][FIT],'task_cards':len(r['task_cards']),
         'checkpoint_count':count,'intent_changes':len(changed),
         'memory_context_injections':memory_injections,
         'rf_reloads':max(0,r['starts'][RF]-1),'ranking_reloads':max(0,r['starts'][RANK]-1),
         'input_hash':next(e['visible_input_hash'] for e in events if e['event']=='llm_start'),
         'source_hashes_unchanged':r['source_hashes_unchanged']}
    rows.append(row)
    checks.append({'run':spec['id'],'actual_tools':dict(Counter(c['name'] for c in r['actual_tool_calls'])),
                   'applied':applied,'intent_selections':changed,'jev_answers':r['jev_events'],
                   'checkpoints':count,'status':r['status'],'error':r.get('error')})
assert len(schemas)==1 and all(len(s)==1 for s in inputs.values()), '首轮输入或接口不同'
assert all(len(s)==1 for s in scientific.values()), '各组科学数值不同，需要逐项核对'
journal=BASE/'network-calls.jsonl'
calls=len(journal.read_text(encoding='utf-8').splitlines()) if journal.exists() else 0
assert calls==sum(r['jev_calls'] for r in rows)<=plan['max_live_jev_calls_total']
assert all(r['jev_calls']==0 for r in rows if r['arm'] in ('cost','lru') or r['process_slots']==3)
groups=[]
for case in plan['prompts']:
    for arm in plan['arms']:
        subset=[r for r in rows if r['case']==case and r['arm']==arm and r['process_slots']==2]
        assert len(subset)==3
        group={'case':case,'arm':arm,'n':3,'completed':sum(r['completed'] for r in subset)}
        for key in ('task_wall_s','llm_wait_s','non_llm_wall_s','tool_wall_s','jev_network_s',
                    'jev_outside_llm_s','jev_admission_wait_s','post_arrival_load_s','rf_reloads','ranking_reloads'):
            group[key+'_median']=statistics.median(r[key] for r in subset)
            group[key+'_range']=[min(r[key] for r in subset),max(r[key] for r in subset)]
        groups.append(group)
comparisons=[]
for case in plan['prompts']:
    for baseline in ('lru','cost','sync'):
        pairs=[]
        for rep in range(3):
            a=next(r for r in rows if r['case']==case and r['arm']=='async' and r['repeat']==rep and r['process_slots']==2)
            b=next(r for r in rows if r['case']==case and r['arm']==baseline and r['repeat']==rep)
            pairs.append({'repeat':rep,'wall_s_saved':b['task_wall_s']-a['task_wall_s'],
                          'non_llm_s_saved':b['non_llm_wall_s']-a['non_llm_wall_s']})
        comparisons.append({'case':case,'baseline':baseline,'pairs':pairs,
                            'wall_wins':sum(p['wall_s_saved']>0 for p in pairs),
                            'non_llm_wins':sum(p['non_llm_s_saved']>0 for p in pairs)})
# 采用既有Windows进程核查方法，核对退出和PID被其他进程复用。
tree=ast.parse((ROOT/'audit_full_chain.py').read_text(encoding='utf-8'))
function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='process_audit')
exec(compile(ast.Module(body=[function],type_ignores=[]),'<process_audit>','exec'),globals())
processes=process_audit(results)
summary={'verification_status':'ANALYZED','complete_runs':len(rows),'completed':sum(r['completed'] for r in rows),
         'jev_calls':calls,'primary_calls':sum(r['primary_calls'] for r in rows),
         'primary_tokens':sum(r['primary_tokens'] for r in rows),
         'groups':groups,'comparisons':comparisons,'process_audit':processes,
         'source_hashes_unchanged':True,'initial_inputs_equal_within_case':True,
         'schemas_equal':True,'scientific_outputs_equal_within_case':True,
         'scope':'两种开发案例、每组3轮；非独立科学留出、不作显著性推断、不证明长程科研。'}
(BASE/'audit.json').write_text(json.dumps({'summary':summary,'rows':rows,'run_checks':checks},ensure_ascii=False,indent=2),encoding='utf-8')
with (BASE/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summary,ensure_ascii=False,indent=2))
