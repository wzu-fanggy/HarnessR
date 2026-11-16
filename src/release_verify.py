"""Verify saved primary evidence without invoking models, APIs, or database services."""
from pathlib import Path
import csv,hashlib,json,math,statistics,sys
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def verify():
 checks={}
 d=read('results/controlled/cwsr_retention_robust_window_v2_full_2026-09-29.json')
 assert len(d['request_ids'])==20 and len(d['orders'])==27
 # Interface-description text is explicitly not a managed tool resource.
 med={k:statistics.median(sum(v for kind,v in x['cases'][k]['repreparations_by_kind'].items() if kind!='tool') for x in d['orders']) for k in ['release_all','pressure_flush','cross_type']}
 assert med=={'release_all':21,'pressure_flush':10,'cross_type':7}
 checks['controlled']={'requests':20,'orders':27,'repeated_creation_medians':med}
 d=read('results/resource_lifecycle/cwsr_cross_kind_real_pressure_formal_v2_result.json')
 assert all(d['checks'].values())
 assert d['primary_summary']['cwsr_completed']==[6,6,6]
 assert d['primary_summary']['cwsr_model_load_count']==[2,2,2]
 checks['fixed_calls']={'primary_trials':3,'sensitivity_budgets':len(d['sensitivity_summary']),'all_checks':True}
 d=read('results/task_aware/audit.json');assert d['summary']['complete_runs']==26 and d['summary']['jev_calls']==12
 checks['task_aware']={'runs':26,'jev_calls':12}
 d=read('results/mcp_sessions/audit.json');assert d['formal_sequences']==36
 assert d['byte_matching_file_reads']==192 and d['confirmed_server_process_exits']==120
 checks['mcp_sessions']={'sequences':36,'matching_file_reads':192,'confirmed_exits':120}
 d=read('results/resource_lifecycle/hcvm_neo4j_driver_integration_v2_result.json')
 assert d['rows_exactly_invariant'] and d['row_count_before']==d['row_count_after']==7
 checks['neo4j']={'rows_before':7,'rows_after':7,'unchanged':True}
 d=read('results/integrated/result.json');assert d['status']=='completed' and d['jev_calls']==1
 c=d['scientific_coverage'];assert c['raw_measurements']==60 and len(c['fitted_sweeps'])==6
 assert not c['missing_points'] and not c['missing_sweeps']
 checks['integrated']={'status':d['status'],'fitted_sweeps':6,'measurements':60,'jev_calls':1}
 d=read('results/sharing/continuations/result.json');assert d['jev_calls']==2
 assert all(b['status']=='completed' for b in d['branches'])
 checks['continuations']={'completed_branches':len(d['branches']),'jev_calls':2}
 d=read('results/heldout_quality/rf_heldout12_rmse.json');assert d['pooled']['n']==48 and d['pooled']['n_formulations']==12
 checks['heldout']={k:d['pooled'][k] for k in ['n','n_formulations','mae_ms_cm','rmse_ms_cm']}
 # Derive the completion table from run records, including the failed database
 # attempt and its independent retry, rather than assuming all attempts succeeded.
 counts={}
 for category in ['screening','heldout']:
  for folder in (ROOT/'results'/category).iterdir():
   if not folder.is_dir() or not (folder/'manifest.json').exists():continue
   manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
   budget=manifest['settings']['runtime_managed_budget']['host_ram_mib']
   for case in folder.iterdir():
    if not case.is_dir() or not (case/'result.json').exists():continue
    value=json.loads((case/'result.json').read_text(encoding='utf8'))
    arm='runtime' if 'runtime' in case.name else 'native'
    key=f'{budget}:{arm}';pair=counts.setdefault(key,[0,0])
    pair[0]+=int(value['ok']);pair[1]+=1
 expected={'477:native':[0,2],'477:runtime':[2,2],'750:native':[2,2],'750:runtime':[2,2],
  '878:native':[0,3],'878:runtime':[3,3],'1233:native':[3,3],'1233:runtime':[3,3],
  '479:native':[0,1],'479:runtime':[1,1],'754:native':[1,1],'754:runtime':[1,2]}
 assert counts==expected,counts
 checks['completion_table']=counts
 d=read('results/supplementary/warm_toolnode/summary.json');assert d['pair_count']==16 and d['all_provenance_and_output_checks_passed']
 checks['timing']={'pairs':16,'task_count':d['task_count']}
 d=read('results/supplementary/cwsr_real_rag_uniform_all8_v2_result.json');assert len(d['rows'])==8
 assert all(x['scientific_outputs_identical'] for x in d['rows'])
 checks['retrieval_retention']={'sequences':8,'unchanged_outputs':True}
 model=ROOT/'data/cwsr_conductivity_rf_formal_v1.joblib'
 assert hashlib.sha256(model.read_bytes()).hexdigest()=='cb60829a64e6ac4f1590b49a496b650f06a73563c0b002d3b1579e1de642e182'
 checks['frozen_model_hash']=True
 assert hashlib.sha256((ROOT/'data/heldout/rf_model.joblib').read_bytes()).hexdigest()=='27ce6c1ab1dceb6bca3a55201fc4980b8a4dad55a17f11c79893fc26b86ec400'
 checks['heldout_model_hash']=True
 out={'status':'PASS','checks':checks,'external_calls':0}
 (ROOT/'checks').mkdir(exist_ok=True)
 (ROOT/'checks/evidence-check.json').write_text(json.dumps(out,indent=2),encoding='utf8')
 return out
def science():
 # Load the author's fitted forest and check actual archived predictions.
 # Only load model files obtained from this verified author bundle.
 import joblib
 model=joblib.load(ROOT/'data/cwsr_conductivity_rf_formal_v1.joblib')
 rows=read('results/integrated/science_results.json')
 count=0;max_error=0.0
 for row in rows:
  if row.get('call',{}).get('name')!='predict_formulation_conductivity':continue
  for candidate in row.get('payload',[]):
   for p in candidate.get('model_predictions',[]):
    if p.get('status')!='predicted':continue
    v=candidate['component_mmol']
    actual=model.predict_mmol(temperature_c=p['temperature_c'],napf6_mmol=v['NaPF6'],ec_mmol=v['EC'],emc_mmol=v['EMC'],pc_mmol=v['PC']).to_dict()
    error=abs(actual['conductivity_ms_cm']-p['numeric_value'])
    assert error<1e-10,(candidate['candidate_id'],error)
    count+=1;max_error=max(max_error,error)
 assert count>0
 out={'status':'PASS','archived_predictions_recomputed':count,'max_absolute_error_ms_cm':max_error,'external_calls':0}
 (ROOT/'checks').mkdir(exist_ok=True)
 (ROOT/'checks/science-check.json').write_text(json.dumps(out,indent=2),encoding='utf8')
 return out
