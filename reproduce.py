"""Offline-first submission entry point. No paid requests without explicit opt-in."""
from pathlib import Path
import argparse,json,os,runpy,sys,tempfile
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/orion')]
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',choices=['verify','science','tests','controlled','fixed-summary','integrated','task-aware','mcp-preflight'])
 p.add_argument('--allow-api-calls',action='store_true')
 p.add_argument('--suite',choices=['main','orders','capacity','all'],default='main')
 p.add_argument('--run-id',help='One run id from experiments/task_aware/protocol.json')
 args=p.parse_args()
 os.chdir(ROOT)
 (ROOT/'outputs').mkdir(exist_ok=True)
 if args.command in {'verify','science'}:
  from release_verify import verify,science
  print(json.dumps(verify() if args.command=='verify' else science(),ensure_ascii=False,indent=2));return
 if args.command=='tests':
  import pytest
  names=['capvm/test_governor.py','capvm/test_shared_resource_runtime.py','capvm/test_semantic_plan.py',
   'capvm/test_task_intent_cache.py','tools/electrolyte_design/test_conductivity_ml.py',
   'tools/electrolyte_design/test_temperature_fit.py','tools/electrolyte_design/test_temperature_fit_process.py',
   'capvm/test_temperature_fit_runtime_resource.py']
  raise SystemExit(pytest.main(['-q','--tb=short','--junitxml=checks/tests.xml',*[str(ROOT/'src/orion/tests'/x) for x in names]]))
 if args.command=='controlled':
  sys.argv=['controlled','--suite',args.suite,'--output',str(ROOT/'outputs'/('controlled-'+args.suite+'.json'))]
  runpy.run_module('orion.benchmark.cwsr_retention_robust_window_v2',run_name='__main__');return
 if args.command=='fixed-summary':
  from orion.benchmark.cwsr_cross_kind_real_pressure_formal_v2 import run
  result=run();out=ROOT/'outputs/fixed-summary.json';out.write_text(json.dumps(result,indent=2),encoding='utf8')
  print(json.dumps({'status':result['status'],'checks':result['checks']},indent=2));return
 if args.command=='mcp-preflight':
  sys.path.insert(0,str(ROOT/'experiments/mcp_sessions'))
  from run_case import preflight
  directory=Path(tempfile.mkdtemp(prefix='mcp-preflight-',dir=ROOT/'outputs'))
  print(json.dumps(preflight(directory),ensure_ascii=False,default=str,indent=2));return
 if not args.allow_api_calls:p.error('This command can incur API charges. Configure your own environment and pass --allow-api-calls explicitly.')
 if not os.environ.get('ORION_ENV_FILE'):p.error('Set ORION_ENV_FILE to your private configuration file first.')
 from dotenv import load_dotenv
 load_dotenv(os.environ['ORION_ENV_FILE'],override=False)
 for name in ['TYPESAFE_API_KEY','DEEPSEEK_API_KEY']:
  if not os.environ.get(name):p.error('Missing required credential: '+name)
 if args.command=='integrated':
  sys.path.insert(0,str(ROOT/'experiments/sharing'))
  runpy.run_path(str(ROOT/'experiments/integrated/execute.py'),run_name='__main__')
 elif args.command=='task-aware':
  if not args.run_id:p.error('--run-id is required; use the frozen protocol run identifiers.')
  import shutil
  out=ROOT/'outputs/task_aware';out.mkdir(parents=True,exist_ok=True)
  if not (out/'protocol.json').exists():shutil.copyfile(ROOT/'experiments/task_aware/protocol.json',out/'protocol.json')
  sys.argv=['task-aware',args.run_id]
  runpy.run_path(str(ROOT/'experiments/task_aware/run.py'),run_name='__main__')

if __name__=='__main__':main()
