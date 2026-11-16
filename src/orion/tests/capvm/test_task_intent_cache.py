"""内容缓存按实际任务内容失效；词语命中不得抹去未知需求。"""
import json
import httpx
import pytest
from orion.capvm.jev_task_intent import JevTaskIntentAssessor
from orion.capvm.task_intent import TaskIntentController
from orion.core.tasks import TaskBoard,TaskStore
from orion.core.task_board_intent import TaskBoardIntentSource


def model_transport(seen):
    def handle(request):
        body=json.loads(request.content);seen.append(body)
        return httpx.Response(200,json={'answers':{q:{'type':'choice','choice':'relevant',
            'probabilities':{'relevant':.98,'not_relevant':.01,'unknown':.01}}
            for q in body['questions']}})
    return httpx.MockTransport(handle)


def card(key='first',intent='给出一个估计'):
    return {'key':key,'task_id':key,'subject':'输运分析','status':'in_progress','dependencies':[],'intent':intent}


async def test_equivalent_new_card_uses_cache_but_rebinds_result():
    seen=[];a=JevTaskIntentAssessor('test-only',max_calls=1,transport=model_transport(seen))
    resources={'rf':{'description':'点预测'}}
    await a.assess([card()],resources)
    second=await a.assess([card('different-card')],resources)
    assert a.calls==1 and len(seen)==1
    assert set(second['bounds'])=={'different-card'}
    assert 'key' not in seen[0]['state']['cards'][0] and 'task_id' not in seen[0]['state']['cards'][0]


@pytest.mark.parametrize('change',['intent','description','status','subject','dependencies','resource'])
async def test_changed_semantic_input_does_not_hit_cache(change):
    seen=[];a=JevTaskIntentAssessor('test-only',max_calls=1,transport=model_transport(seen))
    resources={'rf':{'description':'点预测'}};first=card()
    await a.assess([first],resources)
    changed=dict(first)
    if change=='resource':resources={'rf':{'description':'不同能力'}}
    elif change=='dependencies':changed[change]=[{'id':'upstream','status':'pending'}]
    else:changed[change]='different'
    assert a.cached_assessment([changed],resources) is None


async def test_expiry_not_extended_and_new_task_can_refresh_expired_content():
    tick=[0.];seen=[]
    a=JevTaskIntentAssessor('test-only',max_calls=2,ttl_s=30,clock=lambda:tick[0],transport=model_transport(seen))
    resources={'rf':{'description':'点预测'}}
    await a.assess([card()],resources);tick[0]=29
    assert a.cached_assessment([card('new')],resources)['valid_for_s']==1
    tick[0]=31
    assert a.cached_assessment([card('new')],resources) is None
    assert await a.assess([card('new')],resources) is not None
    assert a.calls==2


async def test_cached_result_is_used_before_network_cost_gate(tmp_path):
    class CachedOnly:
        estimated_overhead_s=1000.
        def cached_assessment(self,cards,resources):
            return {'bounds':{cards[0]['key']:{r:([1.,1.] if r=='cheap' else [0.,0.]) for r in resources}},'valid_for_s':20.}
        async def assess(self,*args):raise AssertionError('缓存命中不应访问网络')
    board=TaskBoard(TaskStore(tmp_path/'tasks'));card=board.create('分析',execution_note='核对已有数据')
    controller=TaskIntentController(TaskBoardIntentSource(lambda:board),{},
        {'cheap':{'description':'原始证据'},'costly':{'description':'数值分析'}},assessor=CachedOnly())
    options=[{'id':'drop-cheap','resource_ids':['cheap']},{'id':'drop-costly','resource_ids':['costly']}]
    with controller.scope('work',[card.id]):
        await controller.prepare({},options,{'cheap':3.,'costly':8.})
        assert controller.choose(options,{'cheap':3.,'costly':8.})=='drop-costly'
        assert any(e.get('source')=='semantic_cache' for e in controller.events)


@pytest.mark.parametrize('note',[
    '继续单点估计，再比较候选受到扰动后的表现',
    '单点估计已经完成，后面整理现有记录',
    '如果证据不足再做单点估计，否则只整理已有结果',
    '继续稳健排序，不做单点',
])
async def test_keyword_mentions_never_close_an_incomplete_requirement_set(tmp_path, note):
    board=TaskBoard(TaskStore(tmp_path/'tasks'));card=board.create('分析',execution_note=note)
    descriptions={'rf':{'literal_terms':['单点']},'rank':{'literal_terms':['稳健排序']}}
    controller=TaskIntentController(TaskBoardIntentSource(lambda:board),{},descriptions)
    options=[{'id':'drop-rf','resource_ids':['rf']},{'id':'drop-rank','resource_ids':['rank']}]
    with controller.scope('work',[card.id]):
        await controller.prepare({},options,{'rf':1.,'rank':5.})
        assert controller.choose(options,{'rf':1.,'rank':5.}) is None
        assert any(e.get('reason')=='no_explicit_assessor' for e in controller.events)
