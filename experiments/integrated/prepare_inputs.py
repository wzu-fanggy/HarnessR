"""Read archived records and prepare a proposal. No inference, fitting, agent or API calls."""
from release_paths import release_path
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import csv
import hashlib
import io
import json
import math

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
RUNS = BASE / 'harnessr_core_20261108'
SOURCE = Path(release_path('${PROJECT_SOURCE}\\reports\\cwsr_inputs\\na_conductivity_training_source_v1.zip'))
MODEL = SOURCE.parent / 'cwsr_conductivity_rf_formal_v1.joblib'
ARTICLE = Path(release_path('${USER_HOME}\\Desktop\\文章'))
COMPONENTS = ('NaPF6', 'EC', 'EMC', 'PC')

def load(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

protected = {str(p): sha(p) for p in ARTICLE.glob('*工作稿.*') if p.is_file()}
resume = load(RUNS / 'r0_resume/result.json')['A']
previous_b = load(RUNS / 'r0/B/result.json')
candidate_request = next(c['args']['request'] for c in resume['calls']
                         if c['name'] == 'predict_formulation_conductivity' and c['args']['request']['temperature_c'] == 20)
candidates = candidate_request['candidates']
assert len(candidates) == 3
records = []
for path in (RUNS / 'r0_resume/A/reports/transport_predictions').glob('*.json'):
    record = load(path)
    record['_archive_record_id'] = path.stem
    record['_archive_path'] = str(path)
    records.append(record)
selected = []
for c in candidates:
    amounts = c['component_mmol']
    total = sum(amounts.values())
    fractions = {k: amounts[k] / total for k in COMPONENTS}
    identity = next(r['composition_id'] for r in records if r['candidate_id'] == c['candidate_id'])
    saved = [r for r in records if r['composition_id'] == identity]
    assert all(r['model_reference']['artifact_sha256'] == sha(MODEL) for r in saved)
    conditions = sorted(set(float(r['conditions']['temperature_c']) for r in saved))
    assert conditions == [-20.0, 0.0, 20.0]
    selected.append({'candidate': c, 'composition_id': identity, 'mole_fractions': fractions,
                     'existing_temperatures_c': conditions,
                     'existing_predictions': [
                         {'temperature_c': r['conditions']['temperature_c'], 'prediction_mS_cm': r['prediction']['numeric_value'], 'record_id': r['_archive_record_id'], 'record_path': r['_archive_path']}
                         for r in sorted(saved, key=lambda r:r['conditions']['temperature_c'])],
                     'new_requested_temperatures_c': [-10.0],
                     'conditional_followup_temperature_c': 25.0})

centroid = {k: sum(c['mole_fractions'][k] for c in selected)/len(selected) for k in COMPONENTS}
groups = defaultdict(list)
with ZipFile(SOURCE) as archive:
    for member in archive.namelist():
        if member.startswith('data/Na/raw/') and member.endswith('.csv'):
            rows = csv.DictReader(io.StringIO(archive.read(member).decode('utf-8-sig')))
            for row_number, row in enumerate(rows):
                amounts = tuple(float(row[k+'_mmol']) for k in COMPONENTS)
                temp = round(1000 / float(row['inverse temperature']) - 273.15, 8)
                conductivity = float(row['conductivity']) * 1000
                groups[(member,amounts)].append({'row_zero_based': row_number, 'temperature_c':temp, 'conductivity_ms_cm':conductivity})

previous_amounts = []
for c in previous_b['calls']:
    if c['name'] == 'predict_formulation_conductivity':
        req = c['args']['request']
        name = req['candidates'][0]['candidate_id']
        previous_amounts.append(tuple(req['composition_inputs'][name]['component_mmol'][k] for k in COMPONENTS))
eligible = []
for (member,amounts),rows in groups.items():
    # Preserve every source-order sweep. Do not average repeated values or label
    # the sweeps as independent experimental replicates without source metadata.
    sweeps=[]
    current=[]
    temperatures=set()
    for row in rows:
        if row['temperature_c'] in temperatures:
            sweeps.append(current)
            current=[]
            temperatures=set()
        current.append(row)
        temperatures.add(row['temperature_c'])
    if current:
        sweeps.append(current)
    if not sweeps:
        continue
    chosen=sweeps[0]
    if len(chosen) != 10 or len({r['temperature_c'] for r in chosen}) != 10:
        continue
    if not {-20.0,-10.0,0.0,20.0}.issubset({r['temperature_c'] for r in chosen}):
        continue
    if any(not math.isfinite(r['conductivity_ms_cm']) or r['conductivity_ms_cm'] <= 0 for r in chosen):
        continue
    total = sum(amounts)
    fractions = {k:v/total for k,v in zip(COMPONENTS,amounts)}
    distance = sum(abs(fractions[k]-centroid[k]) for k in COMPONENTS)
    duplicate_previous = any(all(abs(x-y)<1e-8 for x,y in zip(amounts,a)) for a in previous_amounts)
    eligible.append({'source_member':member,'component_mmol':dict(zip(COMPONENTS,amounts)),
                     'mole_fractions':fractions,'distance_to_shortlist_centroid':distance,
                     'previous_B_formulation':duplicate_previous,
                     'source_temperature_sweep_count':len(sweeps),
                     'source_sweeps':[
                         {'source_sweep_id':f'sweep-{i+1}',
                          'measurements':sorted(sweep,key=lambda r:r['temperature_c']),
                          'fit_input_valid':len(sweep)>=5 and len({r['temperature_c'] for r in sweep})==len(sweep) and all(math.isfinite(r['conductivity_ms_cm']) and r['conductivity_ms_cm']>0 for r in sweep)}
                         for i,sweep in enumerate(sweeps)],
                     'measurement_handling':'Every raw sweep retained separately. Invalid inputs are flagged, not silently removed. Sweep identity does not establish independent experimental replication.'})
eligible.sort(key=lambda r:(r['distance_to_shortlist_centroid'],r['source_member'],r['source_sweeps'][0]['measurements'][0]['row_zero_based']))
references=[]
identities=set()
for item in eligible:
    identity = tuple(round(item['mole_fractions'][k],10) for k in COMPONENTS)
    if identity in identities:
        continue
    identities.add(identity)
    references.append({'reference_id':f'Ref-{len(references)+1}', **item})
    if len(references)==2:
        break
assert len(references)==2
assert not any(r['previous_B_formulation'] for r in references), 'Nearest references were already fitted; reuse instead of planning another fit.'

output={
    'status':'planning only, not an executed experiment',
    'scientific_question':'Follow up the archived low-temperature shortlist with intermediate-temperature checks, model-member threshold agreement, and temperature interpretation against measured reference formulations.',
    'candidate_selection_rule':'Reuse the three candidates already selected and checked at 0 and 20 C by the archived screening agent. Do not select candidates using new model outputs.',
    'reference_selection_rule':'Two distinct formulations nearest to the shortlist centroid by L1 distance in all-component mole fractions with a complete positive first source sweep. Selection is fixed before inference or fitting. Preserve and analyze all source-order sweeps for the selected formulations separately; do not pool or relabel them as independent experimental repeats.',
    'complete_temperature_series_available':len(eligible),
    'candidates':selected,'references':references,
    'screening_temperature_points_c':[-20,-10,0,20],
    'task_specific_conductivity_target_mS_cm':2.0,
    'target_provenance':'The archived screening request used 2.0 mS/cm as its demonstration criterion at -20, 0 and 20 C. This proposal adds the -10 C intermediate check used elsewhere in the manuscript. It is not a universal electrolyte requirement.',
    'model':{'path':str(MODEL),'sha256':sha(MODEL),'retrain':False},
    'ranking_backend':{'method':'existing group-bootstrap RF ensemble', 'members':32,'trees_per_member':96,'seed':20260924,'initialization':'the existing worker fits its ensemble from the frozen bundle at startup; keep the existing implementation and reserved split unchanged','same_model_as_point_predictor':False},
    'design_classification':'prospectively staged integration validation with real chemical inputs and numerical tools; not an unconstrained autonomous scheduling experiment',
    'stage_protocol':'Candidate comparison followed by reference temperature interpretation is a predefined research sequence. Reference fitting is not claimed to be mathematically dependent on candidate ranking. No access order or timestamps may be tuned after observing a policy outcome.',
    'same_snapshot_controls':'At a genuine decision, preserve candidates, ownership, capacities and measured costs, and record the existing fallback and cost-only decisions without semantic relevance. This is a local decision comparison, not a second executed scientific run.',
    'task_information_effect_rule':'Actual advice and subsequent reuse establish an integrated path. An incremental effect of task information requires a different choice from the same-state no-task-information decision. A difference from LRU alone is insufficient if cost-only selects the same plan.',
    'tie_rule':'Keep candidates tied under the predefined scientific ranking. Evaluate every tied priority candidate at 25 C, at most the three already fixed candidates. Do not break scientific ties using a resource outcome.',
    'sharing_identity_rule':'Use resource identity, process PID, creation generation or ordered creation/exit events, model hash and task ownership together. A reused cached number or matching PID alone is insufficient.',
    'source':{'path':str(SOURCE),'sha256':sha(SOURCE),'new_measurements':False},
    'protected_working_file_hashes':protected,
    'scientific_calls_executed':0,'API_calls':0,'experiments_executed':0,
    'working_files_unchanged':all(sha(Path(p))==h for p,h in protected.items())
}
(ROOT/'proposed_inputs.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'candidate_ids':[c['candidate']['candidate_id'] for c in selected],
                  'candidate_existing_temperatures':[c['existing_temperatures_c'] for c in selected],
                  'references':[{'id':r['reference_id'],'member':r['source_member'],'first_row_zero_based':r['source_sweeps'][0]['measurements'][0]['row_zero_based'],'source_sweeps':len(r['source_sweeps']),'mole_fractions':{k:round(v,6) for k,v in r['mole_fractions'].items()},'already_fitted_in_previous_B':r['previous_B_formulation']} for r in references],
                  'complete_reference_series_available':len(eligible),'new_calls':0},ensure_ascii=False))
