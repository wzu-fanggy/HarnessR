"""Rebuild only the two random forests used by this paper, without APIs."""
from pathlib import Path
import argparse,csv,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/orion')]
import joblib
from tools.electrolyte_design.conductivity_ml import (
 SodiumConductivityRandomForest,load_upstream_na_frames,prepare_training_table,ConductivityFeatures)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',choices=['full','heldout','both'],default='both')
 a=p.parse_args()
 table=prepare_training_table(load_upstream_na_frames(ROOT/'data/na_conductivity_training_source_v1.zip'))
 assert len(table)==647 and table.formulation_group.nunique()==72
 split=json.loads((ROOT/'data/heldout/split.json').read_text(encoding='utf8'))
 out=ROOT/'outputs/retrained';out.mkdir(parents=True,exist_ok=True);report={}
 for name in ['full','heldout'] if a.model=='both' else [a.model]:
  train=table
  if name=='heldout':
   ids=set(split['training_formulations']['ids'])
   train=table.loc[table.formulation_group.isin(ids)].reset_index(drop=True)
   assert train.formulation_group.nunique()==60
  model=SodiumConductivityRandomForest(n_estimators=192,min_samples_leaf=2,random_state=20260906).fit(train)
  frozen=joblib.load(ROOT/('data/cwsr_conductivity_rf_formal_v1.joblib' if name=='full' else 'data/heldout/rf_model.joblib'))
  import numpy as np
  x=train[['inverse temperature','x_NaSalt','x_EC','x_PC']].to_numpy()
  error=float(np.max(np.abs(model.model.predict(x)-frozen.model.predict(x))))
  assert error<1e-12,error
  joblib.dump(model,out/(name+'.joblib'),compress=0)
  report[name]={'rows':len(train),'formulations':int(train.formulation_group.nunique()),'max_log_prediction_difference_from_frozen':error}
 (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
