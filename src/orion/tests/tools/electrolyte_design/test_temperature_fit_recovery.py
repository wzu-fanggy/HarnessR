"""输入拒绝应保留工具配对并允许修正；基础设施异常不得吞掉。"""
import json
import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph,MessagesState,START,END
from tools.electrolyte_design import temperature_fit_tool as tool_module
from tools.electrolyte_design.temperature_fit import TemperatureFitInputError, fit_temperature_laws
from tools.electrolyte_design.temperature_fit_process import TemperatureFitProcessAdapter
from tools.electrolyte_design.temperature_fit_tool import TRANSPORT_FIT_PROCESS_RESOURCE_ID as RID
from orion.capvm import (CapabilityCatalog,CapabilityKind,CapabilityResidencyGovernor,
                        ResidencyLevel,ResourceVector,VirtualCapability)
from orion.capvm.shared_runtime import SharedResourceRuntime
from orion.core.langgraph_residency import ToolCallResidencyBridge,ToolResidencyBinding,RuntimeResourceRequirement
from orion.core.resource_tool_node import LeaseSafeToolNode


def request():
    return {'measurements':[{'temperature_c':t,'conductivity_ms_cm':10**(4-900/(t+273.15))}
                            for t in (-20,-5,10,25,40,55,70)]}


def state(value,call='fit'):
    return {'messages':[AIMessage(content='',tool_calls=[{'name':'fit_conductivity_temperature_law',
        'args':{'request':value},'id':call,'type':'tool_call'}])]}


async def invoke(node,value,bridge=None):
    async def execute(state,config):
        return await bridge.ainvoke(node,state,config) if bridge else await node.ainvoke(state,config)
    graph=StateGraph(MessagesState)
    graph.add_node('tools',execute);graph.add_edge(START,'tools');graph.add_edge('tools',END)
    result=await graph.compile().ainvoke(value)
    return {'messages':result['messages'][1:]}


async def test_real_worker_input_error_then_correction_reuses_process(tmp_path):
    adapter=TemperatureFitProcessAdapter()
    g=CapabilityResidencyGovernor(catalog=CapabilityCatalog([VirtualCapability(RID,CapabilityKind.PROCESS,
        '温度拟合验收',adapter,maximum_level=ResidencyLevel.BOUND)]),budget=ResourceVector(host_ram_mb=2048,process_slots=1))
    shared=SharedResourceRuntime(g)
    bridge=ToolCallResidencyBridge(governor=g,shared_runtime=shared,strict=True,
        bindings={'fit_conductivity_temperature_law':ToolResidencyBinding(resources=(RuntimeResourceRequirement(RID),))})
    node=LeaseSafeToolNode([tool_module.fit_conductivity_temperature_law])
    try:
        invalid=request();invalid['measurements'].append(invalid['measurements'][0])
        rejected=await invoke(node,state(invalid,'bad'),bridge)
        message=rejected['messages'][0]
        assert message.status=='error' and message.tool_call_id=='bad'
        assert 'distinct' in message.content
        pid=adapter.pid
        assert pid and g.state(RID).active_count==0
        corrected=await invoke(node,state(request(),'good'),bridge)
        assert corrected['messages'][0].status=='success'
        result=json.loads(corrected['messages'][0].content)
        assert result['status']=='fitted' and adapter.pid==pid
        assert g.state(RID).active_count==0
        (tmp_path/'recovery-evidence.json').write_text(json.dumps({'worker_pid':pid,
            'rejected':message.model_dump(mode='json'),'corrected':result,
            'shared_runtime_events':shared.snapshot()['events']},ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        await shared.aclose()
    assert not adapter.is_running and not shared.snapshot()['active']


async def test_unleased_input_error_is_returned_as_error_message():
    node=LeaseSafeToolNode([tool_module.fit_conductivity_temperature_law])
    result=await invoke(node,state({'measurements':[]}))
    assert result['messages'][0].status=='error'


async def test_unexpected_compute_failure_still_propagates(monkeypatch):
    def fail(_):raise RuntimeError('infrastructure-failure')
    monkeypatch.setattr(tool_module,'fit_temperature_laws',fail)
    node=LeaseSafeToolNode([tool_module.fit_conductivity_temperature_law])
    with pytest.raises(RuntimeError,match='infrastructure-failure'):
        await invoke(node,state(request()))


@pytest.mark.parametrize('value',[{'measurements':[{}]*5}, {'measurements':[None]*5},
    {'measurements':[{'temperature_c':0,'conductivity_ms_cm':-1}]*5}])
def test_explicit_input_error_type(value):
    with pytest.raises(TemperatureFitInputError):fit_temperature_laws(value)
