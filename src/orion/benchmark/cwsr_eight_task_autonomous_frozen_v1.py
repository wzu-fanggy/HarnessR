"""冻结实现后运行原八题；输入只含原问题，不提供文献或配方线索。"""

from __future__ import annotations



import argparse

import hashlib

import json

import os

from pathlib import Path

import subprocess

import sys

import urllib.request



ROOT = Path(__file__).resolve().parents[1]

sys.path[:0] = [str(ROOT), str(ROOT.parent)]

from orion.benchmark import cwsr_eight_task_followup_v1 as base

from orion.core.scientific_review_policy import ScientificReviewContext





def snapshot():

    files = {ROOT/name for name in ('main.py', 'llms.py', 'model_config.py', 'env_utils.py')}

    for folder in ('core', 'tools', 'task_evidence', 'skills', 'evaluation'):

        files.update(p for p in (ROOT/folder).rglob('*') if p.suffix in {'.py', '.md'})

    files.update([Path(__file__).resolve(), Path(base.__file__).resolve(), base.PROFILE])

    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()

            for p in sorted(files)}





def verify(manifest):

    if snapshot() != manifest['source_sha256']:

        raise RuntimeError('Frozen source fingerprint changed; stop collection')

    tasks = base.selected()

    for short, expected in manifest['inputs'].items():

        path, row = tasks[short]

        if row['sample']['question'] != expected['question'] or hashlib.sha256(path.read_bytes()).hexdigest() != expected['original_raw_sha256']:

            raise RuntimeError('Original input changed: ' + short)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--output', required=True)

    parser.add_argument('--execute', action='store_true')

    parser.add_argument('--worker', choices=base.ORDER)

    parser.add_argument('--timeout', type=int, default=480)

    args = parser.parse_args()

    sys.stdout.reconfigure(encoding='utf-8')

    output = Path(args.output).resolve()

    if not output.is_relative_to(ROOT.parent):

        raise ValueError('Output must remain in workspace')

    if not args.execute:

        raise ValueError('Explicit execution required')

    if args.worker:

        manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))

        verify(manifest)

        base.snapshot = snapshot

        base.run_one(args.worker, output, args.timeout)

        verify(manifest)

        base.dump(output/f'{args.worker}.freeze_check.json', {'source_unchanged': True, 'input_unchanged': True})

        return

    if output.exists():

        raise FileExistsError('Historical outputs must not be overwritten')

    with urllib.request.urlopen('http://127.0.0.1:8799/v1/metadata', timeout=30) as response:

        metadata = json.loads(response.read())

    tasks = base.selected()

    manifest = {

        'schema_version': 'original_eight_autonomous_frozen_v1',

        'source_sha256': snapshot(), 'preflight': metadata,

        'inputs': {k: {'question': row['sample']['question'], 'original_raw_path': str(path),

                       'original_raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

                   for k, (path, row) in tasks.items()},

        'tool_profile_sha256': hashlib.sha256(base.PROFILE.read_bytes()).hexdigest(),

        'conditions': {'previously_seen_development_questions': True, 'fresh_process_per_task': True,

                       'paper_title_injection': False, 'candidate_injection': False,

                       'expected_decision_injection': False, 'model': 'deepseek-v4-flash',

                       'temperature': 0, 'top_p': 1, 'thinking': 'enabled',

                       'timeout_seconds': args.timeout,

                       'design_reasoning_mode': 'semantic', 'resource_pool_enabled': True,

                       'runtime_resource_observation': True,

                       'pre_screen_search_batches': ScientificReviewContext.__dataclass_fields__['design_max_pre_screen_rounds'].default,

                       'post_screen_search_batches': int(os.environ.get('ORION_DESIGN_MAX_SEARCH_ROUNDS', '3')),

                       'workflow_evidence_batches': int(os.environ.get('ORION_WORKFLOW_MAX_EVIDENCE_ROUNDS', '16')),

                       'grounding_llm_check': False, 'independent_literature_audit_required': True},

    }

    output.mkdir(parents=True)

    base.dump(output/'manifest.json', manifest)

    results = []

    for short in base.ORDER:

        verify(manifest)

        print('START ' + short, flush=True)

        with (output/f'{short}.stdout.log').open('w', encoding='utf-8') as out, (output/f'{short}.stderr.log').open('w', encoding='utf-8') as err:

            try:

                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--execute',

                                          '--worker', short, '--output', str(output), '--timeout', str(args.timeout)],

                                         cwd=ROOT, stdout=out, stderr=err, timeout=args.timeout+120)

                results.append({'task': short, 'process_exit': process.returncode})

            except subprocess.TimeoutExpired:

                results.append({'task': short, 'process_timeout': True})

        verify(manifest)

        base.dump(output/'collection_progress.json', results)

        print('END ' + json.dumps(results[-1]), flush=True)

    base.dump(output/'collection_complete.json', {'source_unchanged': True, 'input_unchanged': True, 'results': results})

    print('COLLECTION FINISHED', flush=True)





if __name__ == '__main__':

    main()

