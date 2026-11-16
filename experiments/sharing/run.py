"""两个真实主图共用资源池；只观测能力和同状态影子选择，不做性能排名。"""
from release_paths import release_path
import asyncio
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from threading import RLock
from time import perf_counter
import traceback

from isolation import agent_scope, OwnedBoard, WorkspaceRoot

BASE = Path(__file__).resolve().parent
REPO = Path(release_path('${ORION_SOURCE}'))
OLD = Path(release_path('${TASK_INTENT_ARCHIVE}'))
MODE = sys.argv[1]
IS_OFFLINE = MODE.startswith('offline') or MODE.endswith('_offline')
IS_RESUME = MODE.startswith('r0_resume')
OUT = BASE.parents[1] / 'outputs' / MODE
OUT.mkdir(exist_ok=False)
previous = None
if IS_RESUME:
    import sqlite3
    previous = json.loads((BASE / 'r0/result.json').read_text(encoding='utf-8'))
    assert previous['A']['status'] == 'budget_paused'
    assert previous['B']['status'] == 'completed'
    assert previous['A']['primary_requests'] < 28
    assert not previous['task_owners'], '本次恢复只适用于首段没有任务卡的真实记录'
    (OUT / 'short_memory').mkdir()
    with sqlite3.connect(f'file:{(BASE / "r0/short_memory/sessions.db").as_posix()}?mode=ro', uri=True) as source:
        with sqlite3.connect(OUT / 'short_memory/sessions.db') as destination:
            source.backup(destination)
    shutil.copytree(BASE / 'r0/A', OUT / 'A')
sys.path[:0] = [str(REPO.parent), str(REPO)]

settings = {
    'PYTHONDONTWRITEBYTECODE': '1', 'ORION_ENV_FILE': os.environ.get('ORION_ENV_FILE',str(REPO / '.env')),
    'ORION_RESOURCE_POOL_ENABLED': '1', 'ORION_RESOURCE_ADMISSION_QUEUE': '1',
    'ORION_RESOURCE_COORDINATION': '1', 'ORION_RESOURCE_TASK_INTENT': '1',
    'ORION_RESOURCE_INTENT_JEV_MAX_CALLS': '0',
    'ORION_RESOURCE_PROFILE_PATH': str(OUT / 'profiles.db'),
    'ORION_NA_CONDUCTIVITY_MODEL_PATH': release_path('${PROJECT_SOURCE}/reports/cwsr_inputs/cwsr_conductivity_rf_formal_v1.joblib'),
    'ORION_RESOURCE_HOST_RAM_MB': '3072', 'ORION_RESOURCE_PROCESS_SLOTS': '2',
    'ORION_RESOURCE_AUTO_MEMORY': '0', 'ORION_RESOURCE_ADAPTIVE_RESERVE': '0',
    'ORION_RESOURCE_INTENT_PREFETCH': '0', 'ORION_RESOURCE_INTENT_RULE': 'minimax_regret',
    'ORION_RESOURCE_HOST_GUARD': '1',
    'ORION_TASKS_DIR': str(OUT / 'tasks'), 'LTM_DB_PATH': str(OUT / 'facts.db'),
    'LTM_VECTORS_DB_PATH': str(OUT / 'vectors.db'),
    'ORION_REFLECTION_STORE_PATH': str(OUT / 'reflections.db'),
    'ORION_CAPABILITY_STATE_ROOT': str(OUT / 'capability-state'),
    'DEEPAGENTS_ROOT': str(OUT), 'ORION_AUTO_REFLECT': '0',
    'ORION_REFLECTION_INJECT': '0', 'ORION_LLM_CACHE': '0', 'ORION_LLM_STREAM': '0',
    'ORION_AUTO_APPROVE': '1', 'ORION_LLM_REQUEST_MAX_RETRIES': '0',
    'ORION_LLM_STEP_TIMEOUT': '60',
    'LANGCHAIN_TRACING_V2': 'false', 'LANGSMITH_TRACING': 'false',
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
}
os.environ.update(settings)
os.environ.pop('ORION_RESOURCE_EVENTS_PATH', None)
os.environ.pop('ORION_DESIGN_REASONING_MODE', None)
os.chdir(OUT)
import env_utils

secret_values = [v for k, v in os.environ.items()
                 if any(x in k for x in ('KEY', 'TOKEN', 'PASSWORD')) and len(v) > 12]


def safe(value):
    text = json.dumps(value, ensure_ascii=False, default=str)
    for value in secret_values:
        text = text.replace(value, '[REDACTED]')
    return json.loads(text)


def dump(name, value):
    (OUT / name).write_text(json.dumps(safe(value), ensure_ascii=False, indent=2), encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


original_a = json.loads((OLD / 'feedback-workflow-v1/feedback-native-pilot-3/result-resumed.json').read_text(encoding='utf-8'))
prompt_a = next(m['content'] for m in original_a['messages'] if m['type'] == 'human')
prompt_a = prompt_a.replace('至少展示一次有依据的范围调整。', '是否调整搜索范围由实际结果决定，并说明依据。')
old_protocol = __import__('release_paths').resolve_json(json.loads((OLD / 'async-comparison-v1/protocol.json').read_text(encoding='utf-8')))
prompts = {'A': prompt_a, 'B': old_protocol['prompts']['point']}
for owner in prompts:
    folder = OUT / owner
    folder.mkdir(exist_ok=True)
    for source in (REPO / 'skills').rglob('*.md'):
        target = folder / 'skills' / source.relative_to(REPO / 'skills')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (folder / 'request.txt').write_text(prompts[owner], encoding='utf-8')

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from orion.core.tasks import TaskBoard, TaskStore, set_board
from orion.core.task_board_intent import TaskBoardIntentSource
from orion.core.budget import BudgetTracker
from orion.capvm.jev_task_intent import JevTaskIntentAssessor
from orion.capvm.host_memory import windows_host_memory
import tools.file_operate as file_ops

file_ops.WORKDIR = WorkspaceRoot(OUT)
raw_board = TaskBoard(TaskStore(OUT / 'tasks'))
owned_board = OwnedBoard(raw_board)
set_board(owned_board)
started = perf_counter()
events = []
event_lock = RLock()
run_owners = {}
runtime = None
resources = ('process:na-conductivity-rf', 'process:transport-fit-kernel', 'process:robust-ranking-worker')


def emit(event, **values):
    with event_lock:
        item = safe({'sequence': len(events), 'event': event,
                     'elapsed_s': perf_counter() - started, 'agent': agent_scope.get(), **values})
        events.append(item)
        with (OUT / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    return item


def resource_snapshot(closed=False):
    if runtime is None:
        return []
    return [{'resource_id': r, 'pid': runtime.governor.catalog.get(r).adapter.pid,
             'active_count': None if closed else runtime.governor.state(r).active_count,
             'level': 'CLOSED' if closed else runtime.governor.state(r).current_level.name} for r in resources]


class Recorder(BaseCallbackHandler):
    def owner(self, kwargs):
        run_id = str(kwargs.get('run_id'))
        owner = (kwargs.get('metadata') or {}).get('agent_id') or agent_scope.get() or run_owners.get(run_id)
        if owner:
            run_owners[run_id] = owner
        return owner

    def on_chat_model_start(self, serialized, messages, **kwargs):
        emit('llm_start', owner=self.owner(kwargs), run_id=str(kwargs.get('run_id')),
             messages=[m.model_dump(mode='json') for m in messages[0]])

    def on_llm_end(self, response, **kwargs):
        emit('llm_end', owner=self.owner(kwargs), run_id=str(kwargs.get('run_id')),
             usage=response.llm_output)

    def on_llm_error(self, error, **kwargs):
        emit('llm_error', owner=self.owner(kwargs), error_type=type(error).__name__)

    def on_tool_start(self, serialized, input_str, **kwargs):
        emit('tool_start', owner=self.owner(kwargs), name=serialized.get('name'),
             input=input_str, run_id=str(kwargs.get('run_id')), resources=resource_snapshot())

    def on_tool_end(self, output, **kwargs):
        emit('tool_end', owner=self.owner(kwargs), run_id=str(kwargs.get('run_id')),
             output=output.model_dump(mode='json') if hasattr(output, 'model_dump') else output,
             resources=resource_snapshot(), task_state=controller.snapshot())

    def on_tool_error(self, error, **kwargs):
        emit('tool_error', owner=self.owner(kwargs), run_id=str(kwargs.get('run_id')),
             error_type=type(error).__name__)


async def run():
    global runtime, controller
    import orion.main as main
    import model_config
    runtime = main._default_resource_runtime
    shared = runtime.bridge.shared_runtime
    coordinator = shared.demand_coordinator
    controller = coordinator.task_intents
    # 工具只能操作自己的任务板；共同协调器按真实任务绑定读取完整原始板。
    controller.source = TaskBoardIntentSource(get_board=lambda: raw_board)

    class LoggedJev(JevTaskIntentAssessor):
        async def assess(self, cards, descriptions):
            emit('jev_input', cards=cards, descriptions=descriptions)
            result = await super().assess(cards, descriptions)
            emit('jev_output', result=result, events=list(self.events))
            return result

    if not IS_OFFLINE:
        controller.assessor = LoggedJev(os.environ['TYPESAFE_API_KEY'],
            max_calls=3-(previous['jev_calls'] if previous else 0), timeout_s=2, ttl_s=30)
    original_gate = main.pre_tool_gate
    denied = {'apply_kg_correction', 'flag_suspicious_kg_item', 'request_evolve', 'run_script',
              'task_dispatch', 'task_collect'}

    def scientific_gate(state):
        message = next((m for m in reversed(state['messages']) if isinstance(m, AIMessage)), None)
        if message and any(c['name'] in denied for c in message.tool_calls):
            emit('outside_scope_tool', names=[c['name'] for c in message.tool_calls])
            raise RuntimeError('本次数值实验不执行图谱写入、自修改、脚本或子代理任务')
        return original_gate(state)
    main.pre_tool_gate = scientific_gate

    for rid in resources:
        adapter = runtime.governor.catalog.get(rid).adapter
        for action in ('promote', 'demote'):
            original = getattr(adapter, action)

            def observed(current, target, *, method=original, resource=rid, adapter=adapter, action=action):
                previous = adapter.pid
                t = perf_counter()
                try:
                    return method(current, target)
                finally:
                    emit('resource_transition', resource_id=resource, action=action,
                         before_pid=previous, after_pid=adapter.pid, duration_s=perf_counter()-t)
            setattr(adapter, action, observed)

    original_prepare = coordinator.advisor.prepare

    async def observed_prepare(governor, requests):
        state, fingerprint = coordinator.advisor.snapshot(governor, requests)
        options = coordinator.advisor.options(state)
        costs = dict(coordinator.profiler.costs)
        ranked = [(sum(costs[r] for r in set(o['resource_ids'])), o['id']) for o in options
                  if all(r in costs for r in set(o['resource_ids']))]
        baseline = min(ranked)[1] if ranked and len(ranked) == len(options) else None
        if options:
            emit('decision_before', resource_state=state, options=options,
                 task_state=controller.snapshot(), costs=costs, cost_only_shadow=baseline)
        result = await original_prepare(governor, requests)
        if options:
            emit('decision_after', advisor_events=list(coordinator.advisor.events),
                 intent_events=list(controller.events), resources=resource_snapshot())
        return result
    coordinator.advisor.prepare = observed_prepare

    selected = [REPO / 'main.py', REPO / 'core/default_resource_runtime.py',
                REPO / 'core/langgraph_residency.py', REPO / 'core/resource_task_graph.py',
                REPO / 'capvm/shared_runtime.py', REPO / 'capvm/task_intent.py',
                REPO / 'capvm/jev_task_intent.py', REPO / 'capvm/demand_coordinator.py',
                REPO / 'capvm/governor.py', Path(settings['ORION_NA_CONDUCTIVITY_MODEL_PATH']),
                Path(__file__), BASE / 'isolation.py']
    selected += list((REPO / 'tools/electrolyte_design').glob('*.py'))
    fingerprints = {str(p): sha(p) for p in selected}
    dump('protocol.json', {'status': 'frozen_before_task_calls', 'mode': MODE,
         'research_questions': ['跨agent不同工具共用同一实例', '真实任务状态进入资源决策'],
         'prompts': prompts, 'settings': settings, 'fingerprints': fingerprints,
         'model': model_config.AGENT_MODEL, 'temperature': model_config.AGENT_TEMPERATURE,
         'tool_schemas_hash': digest([convert_to_openai_tool(t) for t in main._BASE_TOOLS]),
         'actual_tool_count': len(main._BASE_TOOLS), 'shadow_scope': '同状态的成本方案，不执行反事实任务',
         'maximum_requests_per_agent': 28, 'maximum_seconds_per_agent': 900,
         'maximum_tokens_per_agent': None if IS_RESUME else 400000,
         'continuation_of': 'r0' if IS_RESUME else None,
         'resume_reason': '累计token上限暂停，保持原消息与科学记录，总请求仍不超过28' if IS_RESUME else None,
         'maximum_jev_calls': 3, 'initial_host_memory': asdict(windows_host_memory()),
         'runner_isolation': '共享原任务存储，工具按agent限制可见和修改，文件按agent根目录路由；不改科学接口'})
    graphs = []
    results = {}
    try:
        graphs = [main.build_streaming_agent() for _ in range(1 if IS_RESUME else 2)]
        if IS_RESUME:
            original = previous['A']['messages']
            snapshot = await graphs[0].aget_state({'configurable': {'thread_id': 'r0-A'}})
            restored = [m.model_dump(mode='json') for m in snapshot.values.get('messages', [])]
            assert restored == original, '恢复的消息与原记录不一致'
            emit('checkpoint_restored', original_message_count=len(original),
                 restored_message_count=len(restored), exact_match=True,
                 prior_primary_requests=previous['A']['primary_requests'], checkpoint_next=list(snapshot.next))
        assert all(row['pid'] is None for row in resource_snapshot())
        dump('progress.json', {'status': 'graphs_ready', 'model_requests': 0, 'resources': resource_snapshot()})
        if IS_OFFLINE:
            results['status'] = 'passed_without_model_or_science_calls'
            return

        async def invoke(owner, graph):
            token = agent_scope.set(owner)
            budget = BudgetTracker(max_iterations=28, max_tokens=None if IS_RESUME else 400000, max_wall_seconds=900)
            if IS_RESUME:
                budget.iterations = previous['A']['primary_requests']
                budget.tokens_used = previous['A']['primary_tokens']
            session_prefix = 'r0' if IS_RESUME else MODE
            config = {'configurable': {'thread_id': f'{session_prefix}-{owner}', 'llm_stream': False,
                      'llm_request_max_retries': 0, 'llm_step_timeout': 60, 'budget': budget},
                      'metadata': {'agent_id': owner}, 'recursion_limit': 90, 'callbacks': [Recorder()]}
            emit('task_arrived', prompt_sha256=digest(prompts[owner]))
            item = {'status': 'failed', 'agent': owner}
            try:
                from langgraph.types import Command
                request = (Command(resume='/continue') if IS_RESUME else
                    {'messages': [HumanMessage(content=prompts[owner], id=f'{MODE}-{owner}-request')],
                     'session_id': f'{MODE}-{owner}'})
                state = await asyncio.wait_for(graph.ainvoke(request, config), timeout=900)
                messages = state.get('messages', [])
                item.update(status='budget_paused' if state.get('__interrupt__') else 'completed',
                    messages=[m.model_dump(mode='json') for m in messages],
                    calls=[c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls],
                    final_answer=next((m.content for m in reversed(messages)
                                       if isinstance(m, AIMessage) and not m.tool_calls), None))
            except BaseException as error:
                item.update(error_type=type(error).__name__, traceback=traceback.format_exc())
                snapshot = await graph.aget_state(config)
                item['messages'] = [m.model_dump(mode='json') for m in snapshot.values.get('messages', [])]
                item['checkpoint_next'] = list(snapshot.next)
            finally:
                item.update(primary_requests=budget.iterations, primary_tokens=budget.tokens_used,
                            task_cards=[t.to_dict() for t in owned_board.list()])
                dump(f'{owner}/result.json', item)
                emit('task_finished', status=item['status'])
                results[owner] = item
                dump('progress.json', {'status': 'running', 'finished_agents':
                     {k:v['status'] for k,v in results.items() if isinstance(v,dict)},
                     'event_count': len(events), 'resources': resource_snapshot()})
                agent_scope.reset(token)
            return item

        if IS_RESUME:
            await invoke('A', graphs[0])
            results['continuation_of'] = 'r0'
            results['prior_primary_requests'] = previous['A']['primary_requests']
            assert results['A']['primary_requests'] <= 28
            results['status'] = results['A']['status']
        else:
            await asyncio.gather(invoke('A', graphs[0]), invoke('B', graphs[1]))
            results['status'] = 'completed' if all(results[o]['status'] == 'completed' for o in prompts) else 'incomplete'
    finally:
        results.update(task_owners=owned_board.owners, resources_before_close=resource_snapshot(),
                       coordinator_events=list(coordinator.events), intent_events=list(controller.events),
                       advisor_events=list(coordinator.advisor.events),
                       jev_calls=controller.assessor.calls if controller.assessor else 0,
                       jev_events=list(controller.assessor.events) if controller.assessor else [],
                       admission=shared.snapshot(), source_hashes_unchanged=all(sha(p)==h for p,h in fingerprints.items()))
        for graph in graphs:
            await main.aclose_agent(graph)
        await runtime.aclose()
        results['resources_after_close'] = resource_snapshot(closed=True)
        results['elapsed_including_setup_s'] = perf_counter() - started
        dump('result.json', results)
        dump('progress.json', {'status': results.get('status', 'failed'), 'event_count': len(events),
                              'jev_calls': results['jev_calls']})


with (OUT / 'console.txt').open('w', encoding='utf-8') as console:
    with redirect_stdout(console), redirect_stderr(console):
        try:
            asyncio.run(run())
        except BaseException as error:
            dump('fatal.json', {'error_type': type(error).__name__, 'traceback': traceback.format_exc()})
            raise
print(json.dumps({'output': str(OUT), 'status_file': str(OUT / 'progress.json')}, ensure_ascii=False))
