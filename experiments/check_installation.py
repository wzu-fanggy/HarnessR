"""Check agent imports and real local compute workers without external services."""
from pathlib import Path
import json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = r'''
from pathlib import Path
import importlib, json, os, site, socket, sys
root=Path(sys.argv[1]);version=sys.argv[2]
source=root/'versions/task_aware' if version=='historical' else root/'src'
sys.path[:0]=[str(source),str(source/'orion'),str(root/'src')]
def deny(*args,**kwargs):raise RuntimeError('Network disabled during installation check')
socket.socket.connect=deny
socket.create_connection=deny
out=root/'outputs/installation'/version;out.mkdir(parents=True,exist_ok=True)
os.environ.update(ORION_AGENT_API_KEY='offline-validation-only',DEEPSEEK_API_KEY='offline-validation-only',
 OPENROUTER_API_KEY='offline-validation-only',OPENROUTER_BASE_URL='https://openrouter.ai/api/v1',
 MLFZ_API_KEY='offline-validation-only',MLFZ_BASE_URL='https://api.deepseek.com',ORION_LLM_CACHE='0',
 ORION_AUTO_REFLECT='0',ORION_RESOURCE_POOL_ENABLED='1',DEEPAGENTS_ROOT=str(out),
 LTM_DB_PATH=str(out/'facts.db'),ORION_WIKI_DB_PATH=str(out/'wiki.db'),
 ORION_TASKS_DIR=str(out/'tasks'),ORION_CAPABILITY_STATE_ROOT=str(out/'capability-state'),
 ORION_RESOURCE_PROFILE_PATH=str(out/'profiles.db'),ORION_REFLECTION_STORE_PATH=str(out/'reflections.db'))
os.chdir(out)
import orion.main
import kg_rag.service
import kg_rag.service_kg_query
tools=importlib.import_module('tools.electrolyte_design')
for name in tools.__all__:getattr(tools,name)
from tools.electrolyte_design import conductivity_tool
from tools.electrolyte_design.conductivity_process import SodiumConductivityMlProcessAdapter
from tools.electrolyte_design.robust_ranking import resolve_bundle_dir
from tools.electrolyte_design.robust_ranking_process import RobustRankingProcessAdapter
from capvm import ResidencyLevel
model_path=conductivity_tool.ensure_serialized_public_model()
assert model_path.resolve()==(root/'data/cwsr_conductivity_rf_formal_v1.joblib').resolve()
assert resolve_bundle_dir().resolve()==(root/'data/na_training_bundle_v1').resolve()
import joblib
model=joblib.load(model_path)
rows=json.loads((root/'results/integrated/science_results.json').read_text(encoding='utf8'))
candidate=next(c for r in rows if r.get('call',{}).get('name')=='predict_formulation_conductivity' for c in r.get('payload',[]) if c.get('model_predictions'))
point=next(p for p in candidate['model_predictions'] if p.get('status')=='predicted')
v=candidate['component_mmol']
expected=model.predict_mmol(temperature_c=point['temperature_c'],napf6_mmol=v['NaPF6'],ec_mmol=v['EC'],emc_mmol=v['EMC'],pc_mmol=v['PC']).to_dict()
adapter=SodiumConductivityMlProcessAdapter(model_path=model_path)
try:
 adapter.promote(ResidencyLevel.CATALOG,ResidencyLevel.BOUND)
 result=adapter.predict_features(expected['features'])
 assert result['status']=='RESULT',result
 actual=result['prediction']['conductivity_ms_cm']
 assert abs(actual-point['numeric_value'])<1e-10,(actual,point['numeric_value'])
finally:adapter.close()
assert not adapter.is_running
ranking=RobustRankingProcessAdapter()
try:
 ranking.promote(ResidencyLevel.CATALOG,ResidencyLevel.BOUND)
 assert ranking.ready_metadata['training_formulations']>0
finally:ranking.close()
assert not ranking.is_running
bad=[]
for name,module in list(sys.modules.items()):
 value=getattr(module,'__file__',None)
 if value and 'kg_create' in Path(value).parts and not Path(value).resolve().is_relative_to(root):bad.append(name)
assert not bad,bad
assert not site.ENABLE_USER_SITE
print(json.dumps({'status':'PASS','version':version,'agent_module':Path(orion.main.__file__).relative_to(root).as_posix(),
 'tool_count':len(orion.main._BASE_TOOLS),'prediction_worker_recomputed':1,'prediction_error':abs(actual-point['numeric_value']),
 'ranking_worker_started_and_closed':True,'default_data_paths_in_package':True,'original_project_imports':bad,'external_calls':0}))
'''

def main():
    clean={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('API_KEY','TOKEN','PASSWORD','SECRET','ORION_','DEEPAGENTS_ROOT','LTM_'))}
    clean.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LANGSMITH_TRACING='false',LANGCHAIN_TRACING_V2='false',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',PYTHONNOUSERSITE='1')
    checks=[]
    for version in ['current','historical']:
        p=subprocess.run([sys.executable,'-B','-c',BOOTSTRAP,str(ROOT),version],env=clean,cwd=ROOT.parent,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8',timeout=180)
        if p.returncode:
            print(p.stderr,file=sys.stderr);print(p.stdout,file=sys.stderr);raise SystemExit(p.returncode)
        lines=p.stdout.strip().splitlines()
        checks.append(json.loads(lines[-1]))
    result={'status':'PASS','python':sys.version.split()[0],'isolated_virtual_environment':sys.prefix!=sys.base_prefix,'versions':checks,'paid_api_calls':0}
    (ROOT/'checks/installation-check.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
