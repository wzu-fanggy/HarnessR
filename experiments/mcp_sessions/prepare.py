"""冻结已有任务材料及实验来源，不执行智能体或更改原始记录。"""
from release_paths import release_path
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
ORION = Path(release_path('${ORION_SOURCE}'))
SOURCE = ORION / 'reports/cwsr_original_eight_native_runtime_20261006_v11'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = ROOT / 'manifest.json'
    if manifest_path.exists():
        raise RuntimeError('材料已冻结，禁止覆盖；新材料需要新版本目录')
    sources = []
    for label, case in [('A', 'P2-r1-runtime'), ('B', 'U2-r1-runtime')]:
        src = SOURCE / case
        data = json.loads((src / 'result.json').read_text(encoding='utf-8'))
        assert data['ok'] and data['contexts']
        dst = ROOT / 'materials' / label
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / 'report.md', dst / 'report.md')
        # 只序列化已有contexts字段，保留原值，不生成或改写证据。
        (dst / 'evidence.json').write_text(
            json.dumps(data['contexts'], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        sources.append({
            'workspace': label, 'task_run': case,
            'source_result': str(src / 'result.json'),
            'source_result_sha256': sha(src / 'result.json'),
            'source_report': str(src / 'report.md'),
            'source_report_sha256': sha(src / 'report.md'),
            'evidence_transform': 'JSON serialization of the unchanged contexts field',
            'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                      for p in sorted(dst.iterdir())},
        })
    batches = [
        {'workspace': 'A', 'calls': [['list_directory', '.'], ['read_text_file', 'evidence.json']]},
        {'workspace': 'A', 'calls': [['read_text_file', 'report.md']]},
        {'workspace': 'B', 'calls': [['list_directory', '.'], ['read_text_file', 'evidence.json']]},
        {'workspace': 'B', 'calls': [['read_text_file', 'report.md']]},
        {'workspace': 'A', 'calls': [['read_text_file', 'evidence.json']]},
        {'workspace': 'A', 'calls': [['read_text_file', 'report.md']]},
    ]
    package = ROOT / 'server/node_modules/@modelcontextprotocol/server-filesystem/package.json'
    manifest = {
        'schema': 'real-mcp-session-case-v1', 'stage': 'frozen-before-preflight',
        'python_packages': {x.metadata['Name']: x.version for x in metadata.distributions()},
        'node_version': subprocess.check_output([__import__('shutil').which('node') or 'node', '--version'], text=True).strip(),
        'server_package': json.loads(package.read_text(encoding='utf-8'))['version'],
        'package_lock_sha256': sha(ROOT / 'server/package-lock.json'),
        'runtime_python_sha256': {str(p.relative_to(ORION)): sha(p)
                                  for p in sorted((ORION / 'capvm').rglob('*.py'))},
        'sources': sources, 'batches': batches,
        'scope_contract': 'Prospective task-isolated authorized roots; not a historical MCP trace.',
        'quotas': [1, 2], 'quota_unit': 'managed stdio MCP session instances',
        'strategies': ['retain_all', 'release_after_batch', 'runtime'],
        'primary_outcomes': ['completed_batches', 'correct_tool_results', 'creations',
                             'repeated_creations', 'confirmed_closures', 'quota_compliance'],
        'performance_claim': False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('FROZEN: two existing task workspaces; six batches; eight read-only tool calls')


if __name__ == '__main__':
    main()
