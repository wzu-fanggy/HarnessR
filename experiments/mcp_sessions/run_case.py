"""真实会话可行性与冻结序列比较，失败记录保留在独立运行目录。"""
from __future__ import annotations
from release_paths import release_path

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from mcp_adapter import MCPSessionAdapter, ROOT
from capvm import (CapabilityCatalog, CapabilityKind, CapabilityResidencyGovernor,
                   ResidencyLevel, ResourceVector, VirtualCapability)
from capvm.model import ResidencyBudgetExceeded


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def response_text(response):
    return '\n'.join(item['text'] for item in response['content'] if item['type'] == 'text')


def verify_response(response, workspace, tool, relative):
    assert not response.get('isError'), response
    text = response_text(response)
    if tool == 'list_directory':
        expected = {'[FILE] ' + p.name for p in workspace.iterdir()}
        assert set(text.splitlines()) == expected, (text, expected)
    else:
        # 避免Python文本模式将Windows CRLF折叠成LF；按实际文件字节解码核对。
        expected = (workspace / relative).read_bytes().decode('utf-8')
        assert text == expected, 'Returned text differs from frozen source'
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def make_adapters(directory):
    return {label: MCPSessionAdapter(ROOT / 'materials' / label, directory / label) for label in ['A', 'B']}


def make_governor(adapters, quota):
    return CapabilityResidencyGovernor(
        catalog=CapabilityCatalog(VirtualCapability(
            capability_id=label, kind=CapabilityKind.SESSION, summary='Task-scoped real MCP session',
            adapter=adapter, maximum_level=ResidencyLevel.BOUND)
            for label, adapter in adapters.items()),
        budget=ResourceVector(connection_slots=quota))


def preflight(directory):
    adapters = make_adapters(directory)
    governor = make_governor(adapters, 1)
    checks = {}
    try:
        with governor.acquire_many([('A', ResidencyLevel.BOUND)]):
            a = adapters['A']
            first_gen = a.generation
            evidence = {'path': str(a.workspace / 'evidence.json')}
            first = a.call('read_text_file', evidence)
            verify_response(first, a.workspace, 'read_text_file', 'evidence.json')
            listed = a.call('list_directory', {'path': str(a.workspace)})
            verify_response(listed, a.workspace, 'list_directory', '.')
            checks['batch_sharing_single_creation'] = a.generation == first_gen == 1
            # 确认真正受服务端约束，授权隔离不是对象名称上的划分。
            outside = a.call('read_text_file', {'path': str(adapters['B'].workspace / 'report.md')})
            checks['server_enforces_task_scope'] = bool(outside.get('isError'))
            before = len(a.events)
            try:
                with governor.acquire_many([('B', ResidencyLevel.BOUND)]):
                    raise AssertionError('Conflicting active request was admitted')
            except ResidencyBudgetExceeded:
                checks['active_lease_protected'] = a.live and len(a.events) == before and not adapters['B'].live
        with governor.acquire_many([('A', ResidencyLevel.BOUND)]):
            checks['reuse_after_batch'] = adapters['A'].generation == 1
        governor.demote('A', ResidencyLevel.CATALOG)
        checks['real_close_confirmed'] = not adapters['A'].live and adapters['A'].events[-1]['process_exit_confirmed']
        try:
            adapters['A'].call('read_text_file', evidence)
        except RuntimeError:
            checks['closed_generation_unusable'] = True
        with governor.acquire_many([('A', ResidencyLevel.BOUND)]):
            second = adapters['A'].call('read_text_file', evidence)
            checks['recreation_preserves_output'] = adapters['A'].generation == 2 and response_text(first) == response_text(second)
        governor.demote('A', ResidencyLevel.CATALOG)
        before_counts = {label: adapter.generation for label, adapter in adapters.items()}
        try:
            with governor.acquire_many([('A', ResidencyLevel.BOUND), ('B', ResidencyLevel.BOUND)]):
                raise AssertionError('Impossible simultaneous request was admitted')
        except ResidencyBudgetExceeded:
            checks['impossible_batch_rejected_without_creation'] = before_counts == {
                label: adapter.generation for label, adapter in adapters.items()}
        with governor.acquire_many([('B', ResidencyLevel.BOUND)]):
            verify_response(adapters['B'].call('read_text_file', {'path': str(adapters['B'].workspace / 'report.md')}),
                            adapters['B'].workspace, 'read_text_file', 'report.md')
            checks['second_workspace_available'] = True
    finally:
        governor.close()
        save(directory / 'adapter_events.json', {k: a.events for k, a in adapters.items()})
        save(directory / 'runtime_events.json', [asdict(e) for e in governor.events])
    checks['all_services_closed'] = all(not a.live for a in adapters.values())
    assert checks and all(checks.values()), checks
    return {'phase': 'preflight', 'checks': checks, 'success': True}


def sequence(directory, strategy, quota, manifest):
    directory.mkdir(parents=True, exist_ok=False)
    adapters = make_adapters(directory)
    governor = make_governor(adapters, quota) if strategy == 'runtime' else None
    rows = []
    stopped = None
    started = time.perf_counter()
    try:
        for index, batch in enumerate(manifest['batches'], 1):
            label = batch['workspace']
            adapter = adapters[label]
            if strategy == 'retain_all' and not adapter.live and sum(a.live for a in adapters.values()) >= quota:
                stopped = {'batch': index, 'reason': 'retained_sessions_fill_quota'}
                break
            lease = (governor.acquire_many([(label, ResidencyLevel.BOUND)],
                     request_id=f'batch-{index}', reconcile_before_active=True) if governor else nullcontext())
            with lease:
                if not governor and not adapter.live:
                    adapter.promote(ResidencyLevel.CATALOG, ResidencyLevel.BOUND)
                assert sum(a.live for a in adapters.values()) <= quota
                calls = []
                for tool, relative in batch['calls']:
                    arguments = {'path': str(adapter.workspace if relative == '.' else adapter.workspace / relative)}
                    response = adapter.call(tool, arguments)
                    sha = verify_response(response, adapter.workspace, tool, relative)
                    calls.append({'tool': tool, 'relative_path': relative, 'response_sha256': sha,
                                  'generation': adapter.generation})
                rows.append({'batch': index, 'workspace': label, 'calls': calls,
                             'ready_session_count': sum(a.live for a in adapters.values())})
            if strategy == 'release_after_batch':
                adapter.close()
        # 单独列出任务内关闭与最终清理，避免把退出实验误写成压力回收。
        close_before_cleanup = sum(e['type'] == 'close' for a in adapters.values() for e in a.events)
        elapsed = time.perf_counter() - started
    finally:
        if governor:
            governor.close()
        else:
            for adapter in adapters.values():
                adapter.close()
        save(directory / 'adapter_events.json', {k: a.events for k, a in adapters.items()})
        if governor:
            save(directory / 'runtime_events.json', [asdict(e) for e in governor.events])
    opens = sum(a.generation for a in adapters.values())
    closes = [e for a in adapters.values() for e in a.events if e['type'] == 'close']
    result = {
        'strategy': strategy, 'quota': quota, 'completed_batches': len(rows),
        'completed_calls': sum(len(row['calls']) for row in rows), 'creations': opens,
        'repeated_creations': sum(max(0, a.generation - 1) for a in adapters.values()),
        'closures_during_sequence': close_before_cleanup, 'closures_including_cleanup': len(closes),
        'all_created_services_confirmed_closed': len(closes) == opens and all(e['process_exit_confirmed'] for e in closes),
        'all_successful_results_match_frozen_files': True,
        'ready_boundaries_within_quota': all(row['ready_session_count'] <= quota for row in rows),
        'elapsed_s_descriptive_only': elapsed, 'stopped': stopped, 'batches': rows,
    }
    save(directory / 'result.json', result)
    return result


def check_frozen(manifest):
    for source in manifest['sources']:
        for filename, facts in source['files'].items():
            path = ROOT / 'materials' / source['workspace'] / filename
            assert hashlib.sha256(path.read_bytes()).hexdigest() == facts['sha256']
    for relative, expected in manifest['runtime_python_sha256'].items():
        path = ROOT.parents[1] / 'provenance/frozen_sources/mcp_sessions' / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Runtime source changed: {relative}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['preflight', 'pilot', 'formal'])
    parser.add_argument('--blocks', type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    check_frozen(manifest)
    if args.phase == 'formal':
        protocol = json.loads((ROOT / 'formal_protocol.json').read_text(encoding='utf-8'))
        assert args.blocks == protocol['blocks'], 'Formal block count differs from frozen protocol'
        assert hashlib.sha256((ROOT / 'manifest.json').read_bytes()).hexdigest() == protocol['manifest_sha256']
        for filename, expected in protocol['implementation_sha256'].items():
            assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
    directory = ROOT / 'runs' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.phase)
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / 'invocation.json', {'phase': args.phase, 'blocks': args.blocks, 'argv': sys.argv,
         'manifest_sha256': hashlib.sha256((ROOT / 'manifest.json').read_bytes()).hexdigest()})
    try:
        if args.phase == 'preflight':
            result = preflight(directory)
        else:
            results = []
            strategies = manifest['strategies']
            # 三个区组循环轮换策略先后；不将序列内八次调用作为八个独立重复。
            for block in range(args.blocks):
                order = strategies[block % 3:] + strategies[:block % 3]
                quotas = manifest['quotas'] if block % 2 == 0 else manifest['quotas'][::-1]
                for quota in quotas:
                    for strategy in order:
                        sub = directory / f'block{block+1}_q{quota}_{strategy}'
                        result = sequence(sub, strategy, quota, manifest)
                        result['block'] = block + 1
                        results.append(result)
            result = {'phase': args.phase, 'blocks': args.blocks, 'results': results}
        check_frozen(manifest)
        save(directory / 'summary.json', result)
        print(str(directory))
        print(json.dumps(result if args.phase == 'preflight' else [
            {k: r[k] for k in ['block', 'strategy', 'quota', 'completed_batches', 'creations', 'repeated_creations']}
            for r in result['results']], ensure_ascii=True))
    except BaseException:
        (directory / 'FAILED.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
