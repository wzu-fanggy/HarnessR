from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from orion.capvm import (
    CapabilityCatalog,
    CapabilityKind,
    CapabilityResidencyGovernor,
    ResidencyLevel,
    ResourceVector,
    VirtualCapability,
)
from orion.capvm.resource_observation import ResourceObservationRecorder
from orion.core.langgraph_residency import (
    RuntimeResourceRequirement,
    ToolCallResidencyBridge,
    ToolResidencyBinding,
)
from tools.electrolyte_design.temperature_fit_process import TemperatureFitProcessAdapter
from tools.electrolyte_design.temperature_fit_tool import (
    TRANSPORT_FIT_PROCESS_RESOURCE_ID,
    fit_conductivity_temperature_law,
)


def _state(call_id: str, shift: float = 0.0) -> dict:
    measurements = []
    for temperature_c in (-20.0, -5.0, 10.0, 25.0, 40.0, 55.0):
        temperature_k = temperature_c + 273.15
        log10_conductivity = 3.8 - 900.0 / temperature_k + shift
        measurements.append(
            {
                "temperature_c": temperature_c,
                "conductivity_ms_cm": 10.0 ** log10_conductivity,
            }
        )
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fit_conductivity_temperature_law",
                        "args": {
                            "request": {
                                "formulation_id": call_id,
                                "measurements": measurements,
                            }
                        },
                        "id": call_id,
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }


def test_temperature_fit_tool_reuses_governed_process_between_calls(tmp_path) -> None:
    adapter = TemperatureFitProcessAdapter(declared_host_ram_mb=256.0)
    governor = CapabilityResidencyGovernor(
        catalog=CapabilityCatalog(
            (
                VirtualCapability(
                    capability_id=TRANSPORT_FIT_PROCESS_RESOURCE_ID,
                    kind=CapabilityKind.PROCESS,
                    summary="test transport fitting process",
                    adapter=adapter,
                    maximum_level=ResidencyLevel.BOUND,
                ),
            )
        ),
        budget=ResourceVector(host_ram_mb=2048.0, process_slots=1.0),
    )
    event_path = tmp_path / "resource-events.jsonl"
    bridge = ToolCallResidencyBridge(
        governor=governor,
        bindings={
            "fit_conductivity_temperature_law": ToolResidencyBinding(
                resources=(
                    RuntimeResourceRequirement(
                        TRANSPORT_FIT_PROCESS_RESOURCE_ID,
                        ResidencyLevel.BOUND,
                    ),
                )
            )
        },
        strict=True,
        reconcile_measured=True,
        observation_recorder=ResourceObservationRecorder(event_path),
    )
    node = ToolNode([fit_conductivity_temperature_law])
    graph = StateGraph(MessagesState)
    graph.add_node(
        "tools",
        lambda state: bridge.invoke(node, state, request_id="fit-runtime-test"),
    )
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    app = graph.compile()

    try:
        first = app.invoke(_state("fit-call-1"))
        first_pid = adapter.pid
        assert first_pid is not None
        assert "fitted" in str(first["messages"][-1].content)
        assert (
            governor.state(TRANSPORT_FIT_PROCESS_RESOURCE_ID).current_level
            == ResidencyLevel.BOUND
        )
        assert governor.state(TRANSPORT_FIT_PROCESS_RESOURCE_ID).active_count == 0

        second = app.invoke(_state("fit-call-2", shift=0.02))
        assert adapter.pid == first_pid
        assert "fitted" in str(second["messages"][-1].content)

        rows = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        assert [row["event"] for row in rows] == [
            "lease_acquired",
            "lease_released",
            "lease_acquired",
            "lease_released",
        ]
        assert rows[0]["instance_pid"] == first_pid
        assert rows[2]["instance_pid"] == first_pid
        assert rows[2]["reuse_after_idle_count"] == 1
        assert rows[2]["promotions"] == 1
    finally:
        governor.close()

    assert adapter.pid is None
