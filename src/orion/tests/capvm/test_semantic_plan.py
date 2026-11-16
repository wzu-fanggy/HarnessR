import asyncio
import pytest
from orion.capvm import CapabilityCatalog, CapabilityKind, CapabilityResidencyGovernor, ResidencyLevel, ResourceVector, VirtualCapability
from orion.capvm.adapters.base import BaseCapabilityAdapter
from orion.capvm.semantic_plan import SemanticPlanAdvisor
from orion.capvm.shared_runtime import SharedResourceRuntime
B=ResidencyLevel.BOUND
class Adapter(BaseCapabilityAdapter):
    def __init__(self,n): self.n=n
    def declared_cost(self,level): return ResourceVector(process_slots=self.n if level>=B else 0)
    def promote(self,current,target): pass
    def demote(self,current,target): pass

def setup(advisor):
    g=CapabilityResidencyGovernor(catalog=CapabilityCatalog([VirtualCapability(i,CapabilityKind.PROCESS,i,Adapter(n)) for i,n in [('a',1),('b',1),('c',2),('d',2)]]),budget=ResourceVector(process_slots=4),plan_advisor=advisor)
    g.ensure_many([('a',B),('b',B),('c',B)])
    return g
async def choose_pair(state,options):
    return next(o['id'] for o in options if o['resource_ids']==['a','b'])

@pytest.mark.parametrize('active',[False,True])
async def test_complete_plan_shadow_or_apply(active):
    advisor=SemanticPlanAdvisor(choose_pair,active=active)
    g=setup(advisor)
    state,_=advisor.snapshot(g,[('d',B)])
    assert {tuple(o['resource_ids']) for o in advisor.options(state)}=={('a','b'),('c',)}
    async with SharedResourceRuntime(g).acquire([('d',B)],reconcile_measured=False):
        assert g.state('d').active_count==1
    assert advisor.events[-1]['status']==('applied' if active else 'shadow')
    if active: assert g.state('c').current_level==B
    assert advisor.calls==1

@pytest.mark.parametrize('mode',['invalid','error','timeout'])
async def test_fallback(mode):
    async def decide(state,options):
        if mode=='error': raise RuntimeError()
        if mode=='timeout': await asyncio.sleep(1)
        return 'invalid'
    a=SemanticPlanAdvisor(decide,active=True,timeout_s=.01)
    g=setup(a)
    await a.prepare(g,[('d',B)])
    assert a.cached is None and a.events[-1]['status']=='fallback'
    g.ensure_many([('d',B)])
    assert g.state('d').current_level==B

async def test_stale_context_and_call_budget():
    a=None
    async def decide(state,options):
        a.set_task_context({'version':2})
        return options[0]['id']
    a=SemanticPlanAdvisor(decide,max_calls=1)
    g=setup(a)
    await a.prepare(g,[('d',B)])
    assert a.events[-1]['status']=='stale' and a.cached is None
    await a.prepare(g,[('d',B)])
    assert a.calls==1

async def test_expiry_and_active_protection():
    now=[0.]
    a=SemanticPlanAdvisor(choose_pair,active=True,clock=lambda:now[0])
    g=setup(a)
    await a.prepare(g,[('d',B)])
    now[0]=6
    sentinel=[('c',ResidencyLevel.CATALOG)]
    assert a.select(g,{'d':B},sentinel)==sentinel
    with g.acquire_many([('a',B)],reconcile_before_active=False):
        state,_=a.snapshot(g,[('d',B)])
        assert all('a' not in o['resource_ids'] for o in a.options(state))

async def test_cancel_during_decision_leaves_queue_clean():
    started=asyncio.Event()
    async def decide(state,options):
        started.set()
        await asyncio.Event().wait()
    a=SemanticPlanAdvisor(decide)
    runtime=SharedResourceRuntime(setup(a))
    async def run():
        async with runtime.acquire([('d',B)]): pass
    t=asyncio.create_task(run()); await started.wait(); t.cancel()
    with pytest.raises(asyncio.CancelledError): await t
    assert not runtime.snapshot()['waiting'] and not runtime.snapshot()['active']
    assert a.calls==1 and a.events[-1]['status']=='cancelled'
    await runtime.aclose()

async def test_admission_deadline_covers_model_wait():
    async def decide(state,options): await asyncio.sleep(1)
    a=SemanticPlanAdvisor(decide)
    runtime=SharedResourceRuntime(setup(a))
    with pytest.raises(TimeoutError):
        async with runtime.acquire([('d',B)],timeout=.01): pass
    assert not runtime.snapshot()['waiting']
    assert runtime.governor.state('d').current_level==ResidencyLevel.CATALOG





class RamAdapter(Adapter):

    def declared_cost(self,level):

        return ResourceVector(host_ram_mb=self.n if level>=B else 0)





def unique_setup(advisor):

    # 使用默认科学进程的声明大小；不启动实际进程。

    g=CapabilityResidencyGovernor(catalog=CapabilityCatalog([

        VirtualCapability(i,CapabilityKind.PROCESS,i,RamAdapter(n))

        for i,n in [('fit',192),('ranking',800),('predictor',384)]

    ]),budget=ResourceVector(host_ram_mb=1024),plan_advisor=advisor)

    g.ensure_many([('fit',B)])

    g.ensure_many([('ranking',B)])

    return g





@pytest.mark.parametrize('enabled,active,retained',[(False,True,False),(True,False,False),(True,True,True)])

async def test_unique_minimal_plan_avoids_extra_eviction(enabled,active,retained):

    async def forbidden(*args):

        pytest.fail('唯一方案不得调用模型或成本决策函数')

    a=SemanticPlanAdvisor(forbidden,active=active,max_calls=0,use_unique_plan=enabled)

    g=unique_setup(a)

    state,_=a.snapshot(g,[('predictor',B)])

    assert [o['resource_ids'] for o in a.options(state)]==[['ranking']]

    runtime=SharedResourceRuntime(g)

    async with runtime.acquire([('predictor',B)],reconcile_measured=False):

        assert (g.state('fit').current_level==B)==retained

        assert g.state('ranking').current_level<B

        assert g.state('predictor').active_count==1

    assert a.calls==0

    await runtime.aclose()





@pytest.mark.parametrize('change',['expiry','context','active'])

async def test_unique_minimal_plan_revalidates(change):

    now=[0.]

    a=SemanticPlanAdvisor(None,active=True,use_unique_plan=True,clock=lambda:now[0])

    g=unique_setup(a)

    await a.prepare(g,[('predictor',B)])

    baseline=[('fit',ResidencyLevel.CATALOG),('ranking',ResidencyLevel.CATALOG)]

    if change=='expiry':now[0]=6

    if change=='context':a.set_task_context({'version':2})

    if change=='active':

        with g.acquire_many([('ranking',B)],reconcile_before_active=False):

            assert a.select(g,{'predictor':B},baseline)==baseline

    else:assert a.select(g,{'predictor':B},baseline)==baseline

