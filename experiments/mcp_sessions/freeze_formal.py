"""可行性与预实验完成后固定正式功能验证计划，不修改既有材料。"""
from pathlib import Path
from datetime import datetime
import hashlib
import json

root = Path(__file__).resolve().parent
target = root / 'formal_protocol.json'
assert not target.exists(), '不得覆盖已冻结的正式方案'
protocol = {
    'frozen_at_local': datetime.now().isoformat(),
    'scope': 'Functional resource-lifecycle validation with controlled task authorization',
    'blocks': 6,
    'block_rationale': 'Balance all three strategy positions twice and both quota orders three times; no timing inference.',
    'unit': 'One full fresh six-batch sequence; eight tool calls are not independent replicates.',
    'groups_per_block': 6,
    'strategy_order': 'Rotate retain_all, release_after_batch, runtime by block modulo three.',
    'quota_order': 'Alternate [1,2] and [2,1] by block.',
    'scope_selection': 'A=P2 and B=U2 are task-isolated authorized workspaces by prospective experimental contract, not unavoidable MCP semantics.',
    'preflight_summary': str(root / 'runs/20261030_221250_204118_preflight/summary.json'),
    'pilot_summary': str(root / 'runs/20261030_221315_541331_pilot/summary.json'),
    'manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
    'implementation_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ['run_case.py', 'mcp_adapter.py', 'pid_probe.cjs', 'prepare.py']},
    'primary': ['completion', 'output consistency', 'creation counts', 'confirmed process exits', 'session quota compliance'],
    'success_requirements': ['All non-error results match frozen files.',
                             'Runtime and release-after-batch complete each sequence.',
                             'Every opened server process has a confirmed close, including cleanup.',
                             'No ready boundary exceeds the managed-session quota.',
                             'Preflight verifies active-lease protection, scope denial and impossible-batch rejection.'],
    'no_claims': ['timing superiority', 'all resource families covered', 'new LRU policy',
                  'restoration of arbitrary application session state', 'MCP use in historical recommendation runs'],
    'preflight_repair': 'First check compared Python universal-newline text with unchanged CRLF server output. Oracle now decodes original bytes; failed attempt retained.',
}
target.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('FORMAL FROZEN: six order-balanced blocks, 36 complete sequence attempts; functional/count evidence only')
